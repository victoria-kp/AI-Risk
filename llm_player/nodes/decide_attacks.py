"""Node 3: Decide which attacks to execute.

Receives the board summary from Node 1. Sends a prompt to the Qwen
model, which may call tools (0-3) via <tool_call> tags — e.g.,
battle_sim to check odds before committing to an attack.

After tool results are injected, the model outputs a JSON decision:
    {"attacks": [{"src": "Territory", "target": "Territory"}, ...]}
    or {"attacks": []} to skip attacking.

Validates: src owned with >1 troops, target is adjacent enemy.
Falls back to empty attack list (no attacks) if parsing fails.
"""

import json
import re
from typing import Dict, List, Optional

from risk_env.tool_interface import run_tool_loop


ATTACK_PROMPT = """You are playing Risk. Here is the current board state:

{board_summary}

Decide which attacks to execute this turn. You may attack 0 or more times.

You may optionally call tools before deciding. Available tools:
- battle_sim(attacking=N, defending=N) — simulate battle odds
- threat_analyzer() — analyze threats to your territories
- position_evaluator() — evaluate your overall board position

To call a tool, write: <tool_call>tool_name(args)</tool_call>

After any tool analysis, output your final decision as JSON:
```json
{{"attacks": [{{"src": "YourTerritory", "target": "EnemyTerritory"}}, ...]}}
```

Or to skip attacking:
```json
{{"attacks": []}}
```

Rules:
- src must be a territory you own with more than 1 troop
- target must be adjacent to src and owned by an enemy
- You can list multiple attacks; they execute in order
"""


def decide_attacks(state: dict) -> dict:
    """Decide which attacks to execute.

    Args:
        state: RiskTurnState with game, player, model, board_summary.

    Returns:
        Dict with attack_decisions and attack_raw.
    """
    game = state["game"]
    player = state["player"]
    model = state["model"]
    board_summary = state["board_summary"]

    # Build prompt
    prompt = ATTACK_PROMPT.format(board_summary=board_summary)

    # Get model output
    raw_output = model.generate(prompt, max_tokens=10000, temperature=0.7, caller="attacks") or ""

    # Process any tool calls
    processed_output, tool_log = run_tool_loop(
        raw_output, game=game, player=player
    )

    # If tools were called, re-prompt for final decision
    if tool_log:
        followup = (
            f"{prompt}\n\nYou called tools and got these results:\n"
            f"{processed_output}\n\n"
            f"Now output your final attack decision as JSON."
        )
        raw_output = model.generate(followup, max_tokens=10000, temperature=0.3, caller="attacks_followup") or ""

    # Parse and validate
    fallback = False
    attacks = _parse_attacks(raw_output)
    if attacks is not None:
        attacks = _validate_attacks(attacks, player, game)

    # Fallback if parsing or validation failed
    if attacks is None:
        attacks = _fallback_attacks()
        fallback = True

    return {
        "attack_decisions": attacks,
        "attack_raw": raw_output,
        "attack_fallback": fallback,
    }


def _parse_attacks(text: str) -> Optional[List[Dict[str, str]]]:
    """Extract attack JSON from model output.

    Handles both raw JSON and ```json fenced blocks.
    Returns [{"src": ..., "target": ...}, ...] or None if parsing fails.
    """
    # Try to find ```json ... ``` block first
    fence_match = re.search(r'```json\s*\n?(.*?)\n?\s*```', text, re.DOTALL)
    json_str = fence_match.group(1).strip() if fence_match else text.strip()

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        # Try to find any JSON object with "attacks" key
        obj_match = re.search(r'\{[^{}]*"attacks"\s*:\s*\[.*?\]\s*\}', text, re.DOTALL)
        if obj_match:
            try:
                data = json.loads(obj_match.group(0))
            except json.JSONDecodeError:
                return None
        else:
            return None

    if not isinstance(data, dict) or "attacks" not in data:
        return None

    attacks = data["attacks"]
    if not isinstance(attacks, list):
        return None

    # Validate each attack has src and target strings
    for attack in attacks:
        if not isinstance(attack, dict):
            return None
        if "src" not in attack or "target" not in attack:
            return None
        if not isinstance(attack["src"], str) or not isinstance(attack["target"], str):
            return None

    return [{"src": a["src"], "target": a["target"]} for a in attacks]


def _validate_attacks(attacks: List[Dict[str, str]],
                      player, game) -> Optional[List[Dict[str, str]]]:
    """Validate that each attack is legal.

    Returns the list with only valid attacks, or None if all are invalid.
    An empty list (no attacks) is always valid.
    """
    if not attacks:
        return attacks

    owned_names = {t.name for t in player.territories}
    valid = []

    for attack in attacks:
        src_name = attack["src"]
        target_name = attack["target"]

        # src must be owned
        if src_name not in owned_names:
            continue

        # src must have >1 troops
        src_territory = game.world.territories.get(src_name)
        if src_territory is None or src_territory.forces <= 1:
            continue

        # target must exist
        target_territory = game.world.territories.get(target_name)
        if target_territory is None:
            continue

        # target must be enemy
        if target_territory.owner == player:
            continue

        # target must be adjacent to src
        if target_territory not in src_territory.connect:
            continue

        valid.append(attack)

    return valid if valid else []


def _fallback_attacks() -> List[Dict[str, str]]:
    """Return empty attack list (skip attacking)."""
    return []
