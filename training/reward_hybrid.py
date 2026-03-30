"""Reward functions for the hybrid player (reinforce + attack only).

Two separate reward functions, one per phase. Each returns a float in [0, 1].
No tool calling — rewards are based purely on JSON quality and strategic merit.
"""

import json
import re
from typing import Dict, List, Optional


# ── Reinforcement reward ────────────────────────────────────────────

def compute_reinforce_reward(completion: str,
                              available: int,
                              board_snapshot: dict) -> float:
    """Score a reinforcement completion.

    Components (sum to 1.0):
      0.15  JSON validity — valid JSON with "reinforcements" dict
      0.10  Correctness — troops sum to available, all territories owned
      0.30  Concentration — penalize spreading across 4+ territories
      0.25  Border placement — fraction of troops on border territories
      0.20  Continent completion — reward reinforcing near completable continents

    Args:
        completion: model output text.
        available: number of troops that should be placed.
        board_snapshot: dict with owned_territories, border_territories, etc.

    Returns:
        Float reward in [0, 1].
    """
    owned = set(board_snapshot.get("owned_territories", []))
    borders = set(board_snapshot.get("border_territories", []))
    score = 0.0

    # Component 1: JSON validity (0.15)
    reinf = _parse_reinforcements(completion)
    if reinf is None:
        if "reinforcements" in completion:
            score += 0.05
        return min(1.0, score)
    score += 0.15

    # Component 2: Correctness (0.10)
    total_placed = 0
    valid_placed = 0
    for territory, count in reinf.items():
        if not isinstance(count, (int, float)) or count < 0:
            continue
        count = int(count)
        total_placed += count
        if territory in owned:
            valid_placed += count

    if total_placed > 0:
        validity_frac = valid_placed / total_placed
        if total_placed == available:
            sum_score = 1.0
        elif total_placed < available:
            sum_score = total_placed / available
        else:
            sum_score = max(0.0, 1.0 - (total_placed - available) / available)
        score += 0.10 * (0.5 * validity_frac + 0.5 * sum_score)

    # Component 3: Concentration (0.30)
    n_territories = len([t for t, c in reinf.items()
                         if isinstance(c, (int, float)) and int(c) > 0])
    if n_territories == 0:
        concentration = 0.0
    elif n_territories == 1:
        concentration = 1.0
    elif n_territories == 2:
        concentration = 0.7
    elif n_territories == 3:
        concentration = 0.3
    else:
        concentration = 0.0  # spreading across 4+ is terrible
    score += 0.30 * concentration

    # Component 4: Border placement (0.25)
    if valid_placed > 0:
        border_placed = sum(
            int(reinf.get(t, 0)) for t in borders if t in reinf
        )
        border_frac = border_placed / valid_placed
        score += 0.25 * border_frac

    # Component 5: Continent completion (0.20)
    continent_score = _continent_reinforce_score(reinf, board_snapshot)
    score += 0.20 * continent_score

    return min(1.0, max(0.0, score))


def _continent_reinforce_score(reinf: dict, snapshot: dict) -> float:
    """Score how well reinforcements target near-complete continents."""
    territory_map = snapshot.get("territory_map", {})
    player_name = snapshot.get("player_name", "LLM")

    # Build continent -> (owned, total, missing_territories) map
    continents = {}
    for tname, tinfo in territory_map.items():
        cont = tinfo.get("continent", "Unknown")
        if cont not in continents:
            continents[cont] = {"owned": 0, "total": 0, "border_to_missing": set()}
        continents[cont]["total"] += 1
        if tinfo.get("owner") == player_name:
            continents[cont]["owned"] += 1
            # Check if this territory borders a missing territory in same continent
            for adj_name in tinfo.get("adjacent", []):
                adj_info = territory_map.get(adj_name, {})
                if (adj_info.get("continent") == cont and
                        adj_info.get("owner") != player_name):
                    continents[cont]["border_to_missing"].add(tname)

    # Find continents that are close to completion (>=50% owned, missing <=3)
    target_territories = set()
    for cont, info in continents.items():
        missing = info["total"] - info["owned"]
        if 0 < missing <= 3 and info["owned"] / info["total"] >= 0.5:
            target_territories.update(info["border_to_missing"])

    if not target_territories:
        return 0.5  # neutral if no continent is close

    # What fraction of troops went to continent-completing territories?
    total = 0
    on_target = 0
    for t, count in reinf.items():
        if not isinstance(count, (int, float)) or int(count) <= 0:
            continue
        c = int(count)
        total += c
        if t in target_territories:
            on_target += c

    if total == 0:
        return 0.0

    return on_target / total


# ── Attack reward ───────────────────────────────────────────────────

def compute_attack_reward(completion: str,
                           attack_menu: list,
                           board_snapshot: dict) -> float:
    """Score an attack completion.

    Components (sum to 1.0):
      0.15  JSON validity — valid JSON with "attacks" list of ints
      0.10  Index validity — all indices in range [1, menu_size]
      0.30  Attack quality — troop ratios with steeper curve
      0.15  Activity bonus — reward attacking when good options exist
      0.30  Continent targeting — reward attacks that help complete continents

    Args:
        completion: model output text.
        attack_menu: list of dicts from build_attack_menu().
        board_snapshot: dict with territory info.

    Returns:
        Float reward in [0, 1].
    """
    menu_size = len(attack_menu)
    score = 0.0

    # Component 1: JSON validity (0.15)
    indices = _parse_attack_indices(completion)
    if indices is None:
        if "attacks" in completion:
            score += 0.05
        return min(1.0, score)
    score += 0.15

    # Component 2: Index validity (0.10)
    if len(indices) == 0:
        score += 0.10
    else:
        valid = [i for i in indices if 1 <= i <= menu_size]
        score += 0.10 * (len(valid) / len(indices))
        indices = valid

    # Component 3: Attack quality (0.30) — steeper curve
    if indices:
        ratios = []
        for idx in indices:
            opt = attack_menu[idx - 1]
            src_f = opt.get("src_forces", 1)
            tgt_f = opt.get("tgt_forces", 1)
            ratio = src_f / max(tgt_f, 1)
            ratios.append(ratio)

        avg_ratio = sum(ratios) / len(ratios)
        if avg_ratio >= 4.0:
            quality = 1.0
        elif avg_ratio >= 3.0:
            quality = 0.8
        elif avg_ratio >= 2.0:
            quality = 0.5
        elif avg_ratio >= 1.5:
            quality = 0.3
        else:
            quality = 0.0  # attacking at bad odds is punished
        score += 0.30 * quality

    # Component 4: Activity bonus (0.15)
    if menu_size > 0:
        best_ratio = max(
            opt["src_forces"] / max(opt["tgt_forces"], 1)
            for opt in attack_menu
        )
        if best_ratio >= 3.0:
            # Great options exist: strongly reward attacking, punish passivity
            score += 0.15 if indices else 0.0
        elif best_ratio >= 2.0:
            score += 0.12 if indices else 0.03
        else:
            # Only bad options: skipping is fine
            score += 0.08 if not indices else 0.10
    else:
        score += 0.15

    # Component 5: Continent targeting (0.30)
    continent_score = _continent_attack_score(indices, attack_menu, board_snapshot)
    score += 0.30 * continent_score

    return min(1.0, max(0.0, score))


def _continent_attack_score(indices: list, attack_menu: list,
                             snapshot: dict) -> float:
    """Score how well attacks target continent completion or denial."""
    if not indices and not attack_menu:
        return 0.5  # no attacks possible, neutral

    territory_map = snapshot.get("territory_map", {})
    player_name = snapshot.get("player_name", "LLM")

    # Build continent -> (owned, total) for player
    continents = {}
    for tname, tinfo in territory_map.items():
        cont = tinfo.get("continent", "Unknown")
        if cont not in continents:
            continents[cont] = {"owned": 0, "total": 0}
        continents[cont]["total"] += 1
        if tinfo.get("owner") == player_name:
            continents[cont]["owned"] += 1

    # Find near-complete continents (missing <=3, >=50% owned)
    target_continents = {}
    for cont, info in continents.items():
        missing = info["total"] - info["owned"]
        if 0 < missing <= 3 and info["owned"] / info["total"] >= 0.5:
            target_continents[cont] = info

    # Find opponent near-complete continents to deny
    opp_continents = {}
    for cont, info in continents.items():
        cont_territories = [t for t, ti in territory_map.items()
                           if ti.get("continent") == cont]
        for opp_name in set(ti.get("owner") for ti in territory_map.values()
                           if ti.get("owner") and ti.get("owner") != player_name):
            opp_owned = sum(1 for t in cont_territories
                          if territory_map[t].get("owner") == opp_name)
            missing = len(cont_territories) - opp_owned
            if 0 < missing <= 2 and opp_owned / len(cont_territories) >= 0.6:
                opp_continents[cont] = opp_name

    # Map each attack menu option's target to its continent
    menu_continent_targets = {}
    for opt in attack_menu:
        target = opt.get("target", "")
        tinfo = territory_map.get(target, {})
        cont = tinfo.get("continent", "Unknown")
        menu_continent_targets[opt.get("idx", 0)] = cont

    if not indices:
        # Skipping: penalize if good continent-completing attacks existed
        if target_continents or opp_continents:
            # Check if any menu option targets these continents
            completable = any(
                menu_continent_targets.get(opt.get("idx")) in target_continents
                or menu_continent_targets.get(opt.get("idx")) in opp_continents
                for opt in attack_menu
                if opt["src_forces"] / max(opt["tgt_forces"], 1) >= 2.0
            )
            return 0.0 if completable else 0.5
        return 0.5

    # Score chosen attacks
    continent_hits = 0
    for idx in indices:
        cont = menu_continent_targets.get(idx, "")
        if cont in target_continents or cont in opp_continents:
            continent_hits += 1

    if not target_continents and not opp_continents:
        return 0.5  # no continent goals, neutral

    return continent_hits / len(indices)


# ── Dispatch ────────────────────────────────────────────────────────

def compute_reward(completion: str, phase: str,
                   board_snapshot: dict, **kwargs) -> float:
    """Route to the appropriate reward function based on phase.

    Args:
        completion: model output text.
        phase: "reinforcements" or "attacks".
        board_snapshot: dict with board state.
        **kwargs: additional args (available, attack_menu).

    Returns:
        Float reward in [0, 1].
    """
    if phase == "reinforcements":
        available = kwargs.get("available", 3)
        return compute_reinforce_reward(completion, available, board_snapshot)
    elif phase == "attacks":
        attack_menu = kwargs.get("attack_menu", [])
        return compute_attack_reward(completion, attack_menu, board_snapshot)
    else:
        return 0.0


# ── Internal parsers ────────────────────────────────────────────────

def _parse_reinforcements(text: str) -> Optional[dict]:
    """Extract reinforcements dict from text."""
    match = re.search(r'```json\s*\n?(.*?)\n?\s*```', text, re.DOTALL)
    if match:
        json_str = match.group(1).strip()
    else:
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

    if isinstance(data, dict) and "reinforcements" in data:
        r = data["reinforcements"]
        if isinstance(r, dict):
            return r
    return None


def _parse_attack_indices(text: str) -> Optional[list]:
    """Extract attack indices list from text."""
    match = re.search(r'```json\s*\n?(.*?)\n?\s*```', text, re.DOTALL)
    if match:
        json_str = match.group(1).strip()
    else:
        brace = re.search(r'\{[^{}]*"attacks"[^{}]*\}', text, re.DOTALL)
        if brace:
            json_str = brace.group(0)
        else:
            return None

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        return None

    if isinstance(data, dict) and "attacks" in data:
        attacks = data["attacks"]
        if isinstance(attacks, list):
            return [int(x) for x in attacks if isinstance(x, (int, float))]
    return None
