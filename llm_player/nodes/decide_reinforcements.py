"""Node 2: Decide where to place reinforcement troops.

Receives the board summary from Node 1 and the available troop count.
Sends a prompt to the Qwen model, which may call tools (0-3) via
<tool_call> tags processed by tool_interface.run_tool_loop().
After tool results are injected, the model outputs a JSON decision:
    {"reinforcements": {"TerritoryName": count, ...}}

Validates: sum equals available, all territories owned by player.
Falls back to spreading troops across border territories if parsing fails.
"""

import json
import re
from typing import Dict, Optional

from risk_env.tool_interface import run_tool_loop


REINFORCEMENT_PROMPT = """You are playing Risk. Here is the current board state:

{board_summary}

You have {available} reinforcement troops to place on your territories.

You may optionally call tools before deciding. Available tools:
- battle_sim(attacking=N, defending=N) — simulate battle odds
- threat_analyzer() — analyze threats to your territories
- position_evaluator() — evaluate your overall board position

To call a tool, write: <tool_call>tool_name(args)</tool_call>

After any tool analysis, output your final decision as JSON:
```json
{{"reinforcements": {{"TerritoryName": count, ...}}}}
```

Rules:
- Troop counts must sum to exactly {available}
- You can only place troops on territories you own
- Place troops strategically: prioritize border territories under threat
"""


def decide_reinforcements(state: dict) -> dict:
    """Decide where to place reinforcement troops.

    Args:
        state: RiskTurnState with game, player, model, board_summary,
               reinforcements_available.

    Returns:
        Dict with reinforcement_decision and reinforcement_raw.
    """
    game = state["game"]
    player = state["player"]
    model = state["model"]
    board_summary = state["board_summary"]
    available = state["reinforcements_available"]

    # Build prompt
    prompt = REINFORCEMENT_PROMPT.format(
        board_summary=board_summary,
        available=available,
    )

    # Get model output
    raw_output = model.generate(prompt, max_tokens=10000, temperature=0.7, caller="reinforcements") or ""

    # Process any tool calls
    processed_output, tool_log = run_tool_loop(
        raw_output, game=game, player=player
    )

    # If tools were called, re-prompt for final decision
    if tool_log:
        followup = (
            f"{prompt}\n\nYou called tools and got these results:\n"
            f"{processed_output}\n\n"
            f"Now output your final reinforcement decision as JSON."
        )
        raw_output = model.generate(followup, max_tokens=10000, temperature=0.3, caller="reinforcements_followup") or ""

    # Parse and validate
    fallback = False
    decision = _parse_reinforcements(raw_output)
    if decision is not None:
        decision = _validate_reinforcements(decision, player, available)

    # Fallback if parsing or validation failed
    if decision is None:
        decision = _fallback_reinforcements(player, available)
        fallback = True

    return {
        "reinforcement_decision": decision,
        "reinforcement_raw": raw_output,
        "reinforcement_fallback": fallback,
    }


def _parse_reinforcements(text: str) -> Optional[Dict[str, int]]:
    """Extract reinforcement JSON from model output.

    Handles both raw JSON and ```json fenced blocks.
    Returns {"TerritoryName": count, ...} or None if parsing fails.
    """
    # Try to find ```json ... ``` block first
    fence_match = re.search(r'```json\s*\n?(.*?)\n?\s*```', text, re.DOTALL)
    json_str = fence_match.group(1).strip() if fence_match else text.strip()

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        # Try to find any JSON object in the text
        obj_match = re.search(r'\{[^{}]*"reinforcements"[^{}]*\{[^{}]*\}[^{}]*\}', text)
        if obj_match:
            try:
                data = json.loads(obj_match.group(0))
            except json.JSONDecodeError:
                return None
        else:
            return None

    if not isinstance(data, dict) or "reinforcements" not in data:
        return None

    reinforcements = data["reinforcements"]
    if not isinstance(reinforcements, dict):
        return None

    # Convert values to int
    try:
        return {k: int(v) for k, v in reinforcements.items()}
    except (ValueError, TypeError):
        return None


def _validate_reinforcements(placements: Dict[str, int],
                             player, available: int) -> Optional[Dict[str, int]]:
    """Validate that placements are legal.

    Returns the placements dict if valid, None otherwise.
    """
    # Check sum
    if sum(placements.values()) != available:
        return None

    # Check all values are positive
    if any(v <= 0 for v in placements.values()):
        return None

    # Check all territories are owned by player
    owned_names = {t.name for t in player.territories}
    if not all(name in owned_names for name in placements):
        return None

    return placements


def _fallback_reinforcements(player, available: int) -> Dict[str, int]:
    """Spread troops evenly across border territories.

    If no border territories exist, spread across all territories.
    """
    border = [t for t in player.territories if t.border]
    targets = border if border else list(player.territories)

    if not targets:
        return {}

    # Sort by name for deterministic output
    targets = sorted(targets, key=lambda t: t.name)

    placements = {}
    per_territory = available // len(targets)
    remainder = available % len(targets)

    for i, t in enumerate(targets):
        count = per_territory + (1 if i < remainder else 0)
        if count > 0:
            placements[t.name] = count

    return placements
