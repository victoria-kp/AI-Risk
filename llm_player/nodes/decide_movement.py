"""Node 4: Decide post-combat troop movement (freemove).

Receives the board summary from Node 1. Sends a prompt to the Qwen
model, which may call tools (0-3) via <tool_call> tags for analysis.

After tool results are injected, the model outputs a JSON decision:
    {"movement": {"src": "Territory", "target": "Territory", "count": N}}
    or {"movement": null} to skip.

Validates: both territories owned, connected, 0 < count < src.forces.
Falls back to moving strongest inland troop to weakest connected border.
"""

import json
import re
from typing import Dict, Optional, Tuple

from risk_env.tool_interface import run_tool_loop


MOVEMENT_PROMPT = """You are playing Risk. Here is the current board state:

{board_summary}

You may make one free troop movement: move troops from one of your territories
to another connected friendly territory.

You may optionally call tools before deciding. Available tools:
- battle_sim(attacking=N, defending=N) — simulate battle odds
- threat_analyzer() — analyze threats to your territories
- position_evaluator() — evaluate your overall board position

To call a tool, write: <tool_call>tool_name(args)</tool_call>

After any tool analysis, output your final decision as JSON:
```json
{{"movement": {{"src": "YourTerritory", "target": "YourTerritory", "count": N}}}}
```

Or to skip movement:
```json
{{"movement": null}}
```

Rules:
- Both src and target must be territories you own
- src and target must be adjacent
- count must be between 1 and (src troops - 1) — you must leave at least 1 troop behind
"""


def decide_movement(state: dict) -> dict:
    """Decide post-combat troop movement.

    Args:
        state: RiskTurnState with game, player, model, board_summary.

    Returns:
        Dict with movement_decision and movement_raw.
    """
    game = state["game"]
    player = state["player"]
    model = state["model"]
    board_summary = state["board_summary"]

    # Build prompt
    prompt = MOVEMENT_PROMPT.format(board_summary=board_summary)

    # Get model output
    raw_output = model.generate(prompt, max_tokens=10000, temperature=0.7, caller="movement") or ""

    # Process any tool calls
    processed_output, tool_log = run_tool_loop(
        raw_output, game=game, player=player
    )

    # If tools were called, re-prompt for final decision
    if tool_log:
        followup = (
            f"{prompt}\n\nYou called tools and got these results:\n"
            f"{processed_output}\n\n"
            f"Now output your final movement decision as JSON."
        )
        raw_output = model.generate(followup, max_tokens=10000, temperature=0.3, caller="movement_followup") or ""

    # Parse and validate
    fallback = False
    decision = _parse_movement(raw_output)
    if decision is not None:
        decision = _validate_movement(decision, player, game)
        if decision is None:
            fallback = True
    elif _parsed_as_skip(raw_output):
        decision = None
    else:
        decision = _fallback_movement(player)
        fallback = True

    return {
        "movement_decision": decision,
        "movement_raw": raw_output,
        "movement_fallback": fallback,
    }


def _parsed_as_skip(text: str) -> bool:
    """Check if the model output explicitly contains movement: null."""
    fence_match = re.search(r'```json\s*\n?(.*?)\n?\s*```', text, re.DOTALL)
    json_str = fence_match.group(1).strip() if fence_match else text.strip()

    try:
        data = json.loads(json_str)
        return isinstance(data, dict) and "movement" in data and data["movement"] is None
    except json.JSONDecodeError:
        return False


def _parse_movement(text: str) -> Optional[Dict]:
    """Extract movement JSON from model output.

    Returns {"src": str, "target": str, "count": int} or None.
    None means either skip or parse failure (caller distinguishes).
    """
    # Try to find ```json ... ``` block first
    fence_match = re.search(r'```json\s*\n?(.*?)\n?\s*```', text, re.DOTALL)
    json_str = fence_match.group(1).strip() if fence_match else text.strip()

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        obj_match = re.search(r'\{[^{}]*"movement"[^{}]*\{[^{}]*\}[^{}]*\}', text)
        if obj_match:
            try:
                data = json.loads(obj_match.group(0))
            except json.JSONDecodeError:
                return None
        else:
            return None

    if not isinstance(data, dict) or "movement" not in data:
        return None

    movement = data["movement"]

    # null means skip
    if movement is None:
        return None

    if not isinstance(movement, dict):
        return None

    if "src" not in movement or "target" not in movement or "count" not in movement:
        return None

    if not isinstance(movement["src"], str) or not isinstance(movement["target"], str):
        return None

    try:
        count = int(movement["count"])
    except (ValueError, TypeError):
        return None

    return {"src": movement["src"], "target": movement["target"], "count": count}


def _validate_movement(movement: Dict, player, game) -> Optional[Dict]:
    """Validate that the movement is legal.

    Returns the movement dict if valid, None otherwise.
    """
    src_name = movement["src"]
    target_name = movement["target"]
    count = movement["count"]

    owned_names = {t.name for t in player.territories}

    # Both must be owned
    if src_name not in owned_names or target_name not in owned_names:
        return None

    src_territory = game.world.territories.get(src_name)
    target_territory = game.world.territories.get(target_name)

    if src_territory is None or target_territory is None:
        return None

    # Must be adjacent
    if target_territory not in src_territory.connect:
        return None

    # Count must be valid
    if count <= 0 or count >= src_territory.forces:
        return None

    return movement


def _fallback_movement(player) -> Optional[Dict]:
    """Skip movement."""
    return None
