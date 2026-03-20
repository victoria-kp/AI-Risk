"""Node 0: Decide initial territory placement.

Called repeatedly during the initial placement phase of pyrisk.
Two phases:
  1. Claiming: pick one unclaimed territory from the empty list.
  2. Reinforcing: pick one owned territory to add a troop to.

No tools are used (no combat at this stage). The model receives the
current board state and outputs:
    {"territory": "TerritoryName"}

Falls back to a heuristic (prioritize small continents) if parsing fails.
"""

import json
import re
from typing import List, Optional


# Continents sorted by size (smallest first) for fallback heuristic
CONTINENT_PRIORITY = [
    "Australia",       # 4 territories, +2 bonus
    "South America",   # 4 territories, +2 bonus
    "Africa",          # 6 territories, +3 bonus
    "North America",   # 9 territories, +5 bonus
    "Europe",          # 7 territories, +5 bonus
    "Asia",            # 12 territories, +7 bonus
]


CLAIMING_PROMPT = """You are playing Risk. The game is in the initial placement phase.
You must claim one unclaimed territory.

{board_summary}

Unclaimed territories: {empty_names}

You have {remaining} troops left to place.

Pick one territory to claim. Output your choice as JSON:
```json
{{"territory": "TerritoryName"}}
```
"""


REINFORCING_PROMPT = """You are playing Risk. The game is in the initial placement phase.
All territories have been claimed. You must place one troop on a territory you own.

{board_summary}

You have {remaining} troops left to place.

Pick one of your territories to reinforce. Output your choice as JSON:
```json
{{"territory": "TerritoryName"}}
```
"""


def decide_placement(state: dict) -> dict:
    """Decide which territory to claim or reinforce.

    Args:
        state: dict with game, player, model, empty (list or None), remaining (int).

    Returns:
        Dict with placement_decision (territory name string).
    """
    game = state["game"]
    player = state["player"]
    model = state["model"]
    empty = state["empty"]
    remaining = state["remaining"]

    board_summary = _placement_summary(game, player)

    if empty:
        empty_names = [t.name for t in empty]
        prompt = CLAIMING_PROMPT.format(
            board_summary=board_summary,
            empty_names=", ".join(sorted(empty_names)),
            remaining=remaining,
        )
    else:
        prompt = REINFORCING_PROMPT.format(
            board_summary=board_summary,
            remaining=remaining,
        )

    raw_output = model.generate(prompt, max_tokens=10000, temperature=0.7, caller="placement") or ""

    fallback = False
    decision = _parse_placement(raw_output)
    if decision is not None:
        decision = _validate_placement(decision, player, game, empty)

    if decision is None:
        decision = _fallback_placement(player, game, empty)
        fallback = True

    return {
        "placement_decision": decision,
        "placement_raw": raw_output,
        "placement_fallback": fallback,
    }


def _parse_placement(text: str) -> Optional[str]:
    """Extract territory name from model output.

    Returns territory name string or None if parsing fails.
    """
    fence_match = re.search(r'```json\s*\n?(.*?)\n?\s*```', text, re.DOTALL)
    json_str = fence_match.group(1).strip() if fence_match else text.strip()

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        obj_match = re.search(r'\{[^{}]*"territory"[^{}]*\}', text)
        if obj_match:
            try:
                data = json.loads(obj_match.group(0))
            except json.JSONDecodeError:
                return None
        else:
            return None

    if not isinstance(data, dict) or "territory" not in data:
        return None

    territory = data["territory"]
    if not isinstance(territory, str):
        return None

    return territory


def _validate_placement(territory_name: str, player, game, empty) -> Optional[str]:
    """Validate the territory choice.

    Returns territory name if valid, None otherwise.
    """
    territory = game.world.territories.get(territory_name)
    if territory is None:
        return None

    if empty:
        # Claiming phase: must be in empty list
        if territory not in empty:
            return None
    else:
        # Reinforcing phase: must be owned by player
        if territory.owner != player:
            return None

    return territory_name


def _placement_summary(game, player) -> str:
    """Build a board summary safe for the placement phase (handles unowned territories)."""
    lines = []

    # Your territories
    owned = sorted(player.territories, key=lambda t: t.name)
    if owned:
        lines.append("Your territories:")
        for t in owned:
            lines.append(f"  {t.name} ({t.area.name}): {t.forces} troops")
    else:
        lines.append("You do not own any territories yet.")

    # Continent progress
    lines.append("")
    lines.append("Continents:")
    for area in sorted(game.world.areas.values(), key=lambda a: a.name):
        total = len(area.territories)
        yours = sum(1 for t in area.territories if t.owner == player)
        lines.append(f"  {area.name}: you own {yours}/{total} (bonus: +{area.value})")

    return "\n".join(lines)


def _fallback_placement(player, game, empty) -> str:
    """Fallback heuristic: prioritize small continents.

    Returns a territory name.
    """
    if empty:
        for continent in CONTINENT_PRIORITY:
            for t in empty:
                if t.area.name == continent:
                    return t.name
        return empty[0].name
    else:
        owned = list(player.territories)
        border = [t for t in owned if t.border]
        targets = border if border else owned
        for continent in CONTINENT_PRIORITY:
            for t in targets:
                if t.area.name == continent:
                    return t.name
        return targets[0].name
