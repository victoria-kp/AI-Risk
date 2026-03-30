"""Prompt builders and parsers for the hybrid player.

The hybrid player uses BetterAI for placement and movement, and the LLM
for reinforcements and attacks. This module builds the prompts (with rules,
adjacency graph, situation, and phase-specific options) and parses the
model's JSON responses.

No tool calling — all information is in the prompt. The model outputs
brief reasoning + JSON.
"""

import json
import re
from typing import Dict, List, Optional, Tuple


# ── Risk board constants ────────────────────────────────────────────

ADJACENCY_GRAPH = """TERRITORY CONNECTIONS:
  Afghanistan: China, India, Middle East, Ukraine, Ural
  Alaska: Alberta, Kamchatka, Northwest Territories
  Alberta: Alaska, Northwest Territories, Ontario, Western United States
  Argentina: Brazil, Peru
  Brazil: Argentina, North Africa, Peru, Venezuela
  China: Afghanistan, India, Mongolia, Siberia, South East Asia, Ural
  Congo: East Africa, North Africa, South Africa
  East Africa: Congo, Egypt, Madagascar, Middle East, North Africa, South Africa
  Eastern Australia: New Guinea, Western Australia
  Eastern United States: Mexico, Ontario, Quebec, Western United States
  Egypt: East Africa, North Africa, Middle East, Southern Europe
  Great Britain: Iceland, Northern Europe, Scandinavia, Western Europe
  Greenland: Iceland, Northwest Territories, Ontario, Quebec
  Iceland: Great Britain, Greenland, Scandinavia
  India: Afghanistan, China, Middle East, South East Asia
  Indonesia: New Guinea, South East Asia, Western Australia
  Irkutsk: Kamchatka, Mongolia, Siberia, Yakutsk
  Japan: Kamchatka, Mongolia
  Kamchatka: Alaska, Irkutsk, Japan, Mongolia, Yakutsk
  Madagascar: East Africa, South Africa
  Mexico: Eastern United States, Venezuela, Western United States
  Middle East: Afghanistan, East Africa, Egypt, India, Southern Europe, Ukraine
  Mongolia: China, Irkutsk, Japan, Kamchatka, Siberia
  New Guinea: Eastern Australia, Indonesia, Western Australia
  North Africa: Brazil, Congo, East Africa, Egypt, Southern Europe, Western Europe
  Northern Europe: Great Britain, Scandinavia, Southern Europe, Ukraine, Western Europe
  Northwest Territories: Alaska, Alberta, Greenland, Ontario
  Ontario: Alberta, Eastern United States, Greenland, Northwest Territories, Quebec, Western United States
  Peru: Argentina, Brazil, Venezuela
  Quebec: Eastern United States, Greenland, Ontario
  Scandinavia: Great Britain, Iceland, Northern Europe, Ukraine
  Siberia: China, Irkutsk, Mongolia, Ural, Yakutsk
  South Africa: Congo, East Africa, Madagascar
  South East Asia: China, India, Indonesia
  Southern Europe: Egypt, Middle East, North Africa, Northern Europe, Ukraine, Western Europe
  Ukraine: Afghanistan, Middle East, Northern Europe, Scandinavia, Southern Europe, Ural
  Ural: Afghanistan, China, Siberia, Ukraine
  Venezuela: Brazil, Mexico, Peru
  Western Australia: Eastern Australia, Indonesia, New Guinea
  Western Europe: Great Britain, North Africa, Northern Europe, Southern Europe
  Western United States: Alberta, Eastern United States, Mexico, Ontario
  Yakutsk: Irkutsk, Kamchatka, Siberia"""

RULES_HEADER = """RISK RULES:
- Control all territories in a continent for bonus troops each turn
- Attacks: more troops = higher chance of winning. Must leave 1 on source.
- Troops per turn: (territories / 3) + continent bonuses

CONTINENT BONUSES:
  Australia (4 territories): +2    South America (4 territories): +2
  Africa (6 territories): +3       Europe (7 territories): +5
  North America (9 territories): +5 Asia (12 territories): +7"""

# Map territory -> continent for continent progress computation
TERRITORY_TO_CONTINENT = {
    "Alaska": "North America", "Alberta": "North America",
    "Eastern United States": "North America", "Greenland": "North America",
    "Mexico": "North America", "Northwest Territories": "North America",
    "Ontario": "North America", "Quebec": "North America",
    "Western United States": "North America",
    "Argentina": "South America", "Brazil": "South America",
    "Peru": "South America", "Venezuela": "South America",
    "Congo": "Africa", "East Africa": "Africa", "Egypt": "Africa",
    "Madagascar": "Africa", "North Africa": "Africa", "South Africa": "Africa",
    "Great Britain": "Europe", "Iceland": "Europe",
    "Northern Europe": "Europe", "Scandinavia": "Europe",
    "Southern Europe": "Europe", "Ukraine": "Europe",
    "Western Europe": "Europe",
    "Afghanistan": "Asia", "China": "Asia", "India": "Asia",
    "Irkutsk": "Asia", "Japan": "Asia", "Kamchatka": "Asia",
    "Middle East": "Asia", "Mongolia": "Asia", "Siberia": "Asia",
    "South East Asia": "Asia", "Ural": "Asia", "Yakutsk": "Asia",
    "Eastern Australia": "Australia", "Indonesia": "Australia",
    "New Guinea": "Australia", "Western Australia": "Australia",
}

CONTINENT_SIZES = {
    "Australia": 4, "South America": 4, "Africa": 6,
    "Europe": 7, "North America": 9, "Asia": 12,
}

CONTINENT_BONUSES = {
    "Australia": 2, "South America": 2, "Africa": 3,
    "Europe": 5, "North America": 5, "Asia": 7,
}


# ── Situation builder ───────────────────────────────────────────────

def build_situation(snapshot: dict) -> str:
    """Build the SITUATION block from a board snapshot."""
    owned = set(snapshot.get("owned_territories", []))
    territory_map = snapshot.get("territory_map", {})
    players = snapshot.get("players", {})
    player_name = snapshot.get("player_name", "LLM")

    total_troops = sum(
        territory_map[t]["forces"] for t in owned if t in territory_map
    )

    # Continent progress
    continent_owned = {}
    for t in owned:
        cont = TERRITORY_TO_CONTINENT.get(t)
        if cont:
            continent_owned[cont] = continent_owned.get(cont, 0) + 1

    complete = []
    progress = []
    for cont in ["Australia", "South America", "Africa",
                  "Europe", "North America", "Asia"]:
        have = continent_owned.get(cont, 0)
        total = CONTINENT_SIZES[cont]
        bonus = CONTINENT_BONUSES[cont]
        if have == total:
            complete.append(f"{cont} (+{bonus})")
        elif have > 0:
            # Find missing territories
            missing = [t for t, c in TERRITORY_TO_CONTINENT.items()
                       if c == cont and t not in owned]
            progress.append(f"{cont}: {have}/{total} (need: {', '.join(missing)})")

    lines = ["SITUATION:"]
    lines.append(f"- You own {len(owned)} territories, {total_troops} total troops")
    if complete:
        lines.append(f"- Continents: {', '.join(complete)}")
    for p in progress:
        lines.append(f"- {p}")

    # Opponents
    for pname, pinfo in players.items():
        if pname == player_name:
            continue
        lines.append(
            f"- {pname}: {pinfo.get('territories', 0)} territories "
            f"({pinfo.get('forces', 0)} troops)"
        )

    return "\n".join(lines)


# ── Reinforcement prompt ────────────────────────────────────────────

def build_reinforce_prompt(snapshot: dict, available: int) -> str:
    """Build the full reinforcement prompt.

    Args:
        snapshot: board_snapshot dict with territory_map, owned_territories, etc.
        available: number of troops to place.

    Returns:
        Complete prompt string.
    """
    owned = set(snapshot.get("owned_territories", []))
    territory_map = snapshot.get("territory_map", {})
    player_name = snapshot.get("player_name", "LLM")

    situation = build_situation(snapshot)

    # Build territory list with border info
    territory_lines = []
    for t_name in sorted(owned):
        info = territory_map.get(t_name, {})
        forces = info.get("forces", 0)

        # Find enemy neighbors
        enemy_neighbors = []
        for adj_name in info.get("adjacent", []):
            adj_info = territory_map.get(adj_name, {})
            adj_owner = adj_info.get("owner", "")
            if adj_owner and adj_owner != player_name:
                adj_forces = adj_info.get("forces", 0)
                enemy_neighbors.append(f"{adj_name}[{adj_owner},{adj_forces}]")

        if enemy_neighbors:
            border_str = f" (borders: {', '.join(enemy_neighbors)})"
        else:
            border_str = ""

        territory_lines.append(f"  {t_name}: {forces} troops{border_str}")

    territories_block = "\n".join(territory_lines)

    prompt = f"""{RULES_HEADER}

{ADJACENCY_GRAPH}

{situation}

YOU HAVE {available} TROOPS TO PLACE on your territories:
{territories_block}

Decide where to place your {available} troops. Brief reasoning (1-2 sentences), then JSON:
```json
{{"reinforcements": {{"TerritoryName": count, ...}}}}
```"""

    return prompt


def parse_reinforcements(text: str, available: int,
                         owned: set) -> Optional[Dict[str, int]]:
    """Parse reinforcement JSON from model output.

    Returns dict of {territory: count} or None if parsing fails.
    Validates: all territories owned, counts are positive ints, sum == available.
    """
    # Try fenced JSON first
    match = re.search(r'```json\s*\n?(.*?)\n?\s*```', text, re.DOTALL)
    if match:
        json_str = match.group(1).strip()
    else:
        # Try bare JSON
        brace = re.search(r'\{[^{}]*"reinforcements"[^{}]*\{[^{}]*\}[^{}]*\}',
                          text, re.DOTALL)
        if brace:
            json_str = brace.group(0)
        else:
            return None

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        return None

    if not isinstance(data, dict) or "reinforcements" not in data:
        return None

    reinf = data["reinforcements"]
    if not isinstance(reinf, dict):
        return None

    # Validate
    result = {}
    for territory, count in reinf.items():
        if not isinstance(territory, str):
            return None
        if not isinstance(count, (int, float)) or count < 0:
            return None
        count = int(count)
        if count > 0:
            if territory not in owned:
                continue  # skip invalid, keep the rest
            result[territory] = count

    # Check sum
    if sum(result.values()) != available:
        # Try to salvage: if sum < available, it's still usable (leftover handled by fallback)
        if sum(result.values()) > available or sum(result.values()) == 0:
            return None

    return result if result else None


# ── Attack menu & prompt ────────────────────────────────────────────

def build_attack_menu(snapshot: dict) -> List[dict]:
    """Build numbered list of valid attack options from board state.

    Returns list of dicts with keys: idx, src, target, src_forces, tgt_forces,
    tgt_owner. Sorted by src_forces descending. Capped at 15.
    """
    owned = set(snapshot.get("owned_territories", []))
    territory_map = snapshot.get("territory_map", {})
    player_name = snapshot.get("player_name", "LLM")

    options = []
    for src_name in owned:
        src_info = territory_map.get(src_name, {})
        src_forces = src_info.get("forces", 0)
        if src_forces <= 1:
            continue

        for adj_name in src_info.get("adjacent", []):
            adj_info = territory_map.get(adj_name, {})
            adj_owner = adj_info.get("owner", "")
            if adj_owner == player_name or not adj_owner:
                continue
            tgt_forces = adj_info.get("forces", 0)
            if tgt_forces <= 0:
                continue

            options.append({
                "src": src_name,
                "target": adj_name,
                "src_forces": src_forces,
                "tgt_forces": tgt_forces,
                "tgt_owner": adj_owner,
            })

    # Sort by src_forces descending (strongest attacks first)
    options.sort(key=lambda x: x["src_forces"], reverse=True)

    # Cap at 15
    options = options[:15]

    # Add indices
    for i, opt in enumerate(options):
        opt["idx"] = i + 1

    return options


def build_attack_prompt(snapshot: dict, menu: List[dict]) -> str:
    """Build the full attack prompt with numbered options.

    Args:
        snapshot: board_snapshot dict.
        menu: list from build_attack_menu().

    Returns:
        Complete prompt string.
    """
    situation = build_situation(snapshot)

    if not menu:
        menu_str = "  (no valid attacks available)"
    else:
        menu_lines = []
        for opt in menu:
            menu_lines.append(
                f"  [{opt['idx']}] {opt['src']} ({opt['src_forces']} troops) -> "
                f"{opt['target']} ({opt['tgt_forces']} troops, {opt['tgt_owner']})"
            )
        menu_str = "\n".join(menu_lines)

    prompt = f"""{RULES_HEADER}

{ADJACENCY_GRAPH}

{situation}

AVAILABLE ATTACKS:
{menu_str}

Pick which attacks to execute (or empty list to skip). Brief reasoning (1-2 sentences), then JSON:
```json
{{"attacks": [1, 2]}}
```"""

    return prompt


def parse_attack_indices(text: str, menu_size: int) -> Optional[List[int]]:
    """Parse attack indices from model output.

    Returns list of valid indices (1-based) or None if parsing fails.
    """
    # Try fenced JSON first
    match = re.search(r'```json\s*\n?(.*?)\n?\s*```', text, re.DOTALL)
    if match:
        json_str = match.group(1).strip()
    else:
        # Try bare JSON
        brace = re.search(r'\{[^{}]*"attacks"[^{}]*\}', text, re.DOTALL)
        if brace:
            json_str = brace.group(0)
        else:
            return None

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        return None

    if not isinstance(data, dict) or "attacks" not in data:
        return None

    attacks = data["attacks"]
    if not isinstance(attacks, list):
        return None

    # Validate indices
    valid = []
    for idx in attacks:
        if isinstance(idx, (int, float)):
            idx = int(idx)
            if 1 <= idx <= menu_size:
                valid.append(idx)

    # Deduplicate preserving order
    seen = set()
    deduped = []
    for idx in valid:
        if idx not in seen:
            seen.add(idx)
            deduped.append(idx)

    return deduped


# ── SFT data helpers ────────────────────────────────────────────────

def map_attack_decisions_to_indices(decisions: list,
                                     menu: List[dict]) -> List[int]:
    """Map heuristic AI attack decisions to menu indices.

    Args:
        decisions: list of {"src": str, "target": str, ...} from heuristic AI.
        menu: list from build_attack_menu().

    Returns:
        List of matching menu indices (1-based).
    """
    # Build lookup
    menu_lookup = {}
    for opt in menu:
        key = (opt["src"], opt["target"])
        menu_lookup[key] = opt["idx"]

    indices = []
    for d in decisions:
        key = (d.get("src", ""), d.get("target", ""))
        if key in menu_lookup:
            idx = menu_lookup[key]
            if idx not in indices:
                indices.append(idx)

    return indices
