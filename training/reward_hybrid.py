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
      0.25  JSON validity — valid JSON with "reinforcements" dict
      0.25  Correctness — troops sum to available, all territories owned
      0.30  Concentration — penalize spreading across 4+ territories
      0.20  Border placement — fraction of troops on border territories

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

    # Component 1: JSON validity (0.25)
    reinf = _parse_reinforcements(completion)
    if reinf is None:
        # Partial credit for attempted JSON
        if "reinforcements" in completion:
            score += 0.10
        return min(1.0, score)
    score += 0.25

    # Component 2: Correctness (0.25)
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
        # Fraction of troops on valid territories
        validity_frac = valid_placed / total_placed
        # Sum correctness: closer to available is better
        if total_placed == available:
            sum_score = 1.0
        elif total_placed < available:
            sum_score = total_placed / available
        else:
            sum_score = max(0.0, 1.0 - (total_placed - available) / available)
        score += 0.25 * (0.5 * validity_frac + 0.5 * sum_score)
    # else: 0 troops placed = 0 correctness

    # Component 3: Concentration (0.30)
    n_territories = len([t for t, c in reinf.items()
                         if isinstance(c, (int, float)) and int(c) > 0])
    if n_territories == 0:
        concentration = 0.0
    elif n_territories == 1:
        concentration = 1.0
    elif n_territories == 2:
        concentration = 0.8
    elif n_territories == 3:
        concentration = 0.5
    else:
        concentration = 0.2  # spreading across 4+ is bad
    score += 0.30 * concentration

    # Component 4: Border placement (0.20)
    if valid_placed > 0:
        border_placed = sum(
            int(reinf.get(t, 0)) for t in borders if t in reinf
        )
        border_frac = border_placed / valid_placed
        score += 0.20 * border_frac
    # else: 0 border score

    return min(1.0, max(0.0, score))


# ── Attack reward ───────────────────────────────────────────────────

def compute_attack_reward(completion: str,
                           attack_menu: list,
                           board_snapshot: dict) -> float:
    """Score an attack completion.

    Components (sum to 1.0):
      0.25  JSON validity — valid JSON with "attacks" list of ints
      0.25  Index validity — all indices in range [1, menu_size]
      0.30  Attack quality — average troop ratio of chosen attacks
      0.20  Activity bonus — reward attacking when good options exist

    Args:
        completion: model output text.
        attack_menu: list of dicts from build_attack_menu().
        board_snapshot: dict with territory info.

    Returns:
        Float reward in [0, 1].
    """
    menu_size = len(attack_menu)
    score = 0.0

    # Component 1: JSON validity (0.25)
    indices = _parse_attack_indices(completion)
    if indices is None:
        # Partial credit
        if "attacks" in completion:
            score += 0.10
        return min(1.0, score)
    score += 0.25

    # Component 2: Index validity (0.25)
    if len(indices) == 0:
        # Empty list is valid JSON
        score += 0.25
    else:
        valid = [i for i in indices if 1 <= i <= menu_size]
        score += 0.25 * (len(valid) / len(indices))
        indices = valid  # use only valid indices from here

    # Component 3: Attack quality (0.30)
    if indices:
        ratios = []
        for idx in indices:
            opt = attack_menu[idx - 1]
            src_f = opt.get("src_forces", 1)
            tgt_f = opt.get("tgt_forces", 1)
            ratio = src_f / max(tgt_f, 1)
            ratios.append(ratio)

        # Score based on average ratio: ratio 3+ = full credit, 1 = half, <1 = low
        avg_ratio = sum(ratios) / len(ratios)
        if avg_ratio >= 3.0:
            quality = 1.0
        elif avg_ratio >= 2.0:
            quality = 0.8
        elif avg_ratio >= 1.5:
            quality = 0.6
        elif avg_ratio >= 1.0:
            quality = 0.4
        else:
            quality = 0.2
        score += 0.30 * quality

    # Component 4: Activity bonus (0.20)
    if menu_size > 0:
        best_ratio = max(
            opt["src_forces"] / max(opt["tgt_forces"], 1)
            for opt in attack_menu
        )
        if best_ratio >= 2.0:
            # Good options exist: reward attacking, penalize passivity
            score += 0.20 if indices else 0.0
        elif best_ratio >= 1.5:
            # Decent options: slight reward either way
            score += 0.15 if indices else 0.10
        else:
            # Only bad options: skipping is fine
            score += 0.10 if not indices else 0.15
    else:
        # No menu = no attacks possible, always fine
        score += 0.20

    return min(1.0, max(0.0, score))


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
