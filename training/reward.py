"""Reward function for GRPO training.

Scores model completions on four weighted components:

1. Format + Decision Quality (weight 0.65):
   - Valid JSON output (+0.2), correct territory names (+0.2),
     sum constraints (+0.1), strategic quality heuristics (+0.5)
   - Round 4: smooth scoring — partial credit for partially correct
     territory names instead of all-or-nothing cliffs.

2. Tool Use Appropriateness (weight 0.15):
   - Did it call relevant tools for the task type?
   - threat_analyzer for reinforcement, battle_sim for attacks, etc.

3. Efficiency (weight 0.10):
   - 1-2 tool calls ideal for decisions
   - Penalize 0 calls (missed info) or 3+ (wasteful)

4. Partial Credit (weight 0.10):
   - Credit for attempted tool/JSON syntax.
   - Creates reward variance among completions that fail on main
     components, enabling GRPO to learn from relative advantages.

Evolution: Round 1 (off-policy bootstrap) → Round 2 (strategic quality
focus) → Round 3 (anti-passivity) → Round 4 (smooth gradients for GRPO).

Handles three phase types: reinforcements, attacks, movement.
Returns float reward in [0, 1].
"""

import json
import re
from typing import Dict, List, Optional


# ── Weights ────────────────────────────────────────────────────────────

# ── Round 1 weights (off-policy bootstrapping) ────────────────────────
# W_QUALITY = 0.35
# W_TOOL_APPROPRIATENESS = 0.15
# W_EFFICIENCY = 0.10
# W_PARTIAL = 0.30
# WIN_BONUS = 0.10

# ── Round 2 weights (strategic quality focus) ─────────────────────────
# W_QUALITY = 0.60
# W_TOOL_APPROPRIATENESS = 0.15
# W_EFFICIENCY = 0.10
# W_PARTIAL = 0.05
# WIN_BONUS = 0.10
# Base weights sum to 0.90, leaving room for WIN_BONUS

# ── Round 4 weights (smooth gradients for GRPO) ──────────────────────
# Changes from Round 2:
# - Removed WIN_BONUS: all G completions share the same outcome for a
#   given prompt, so the bonus cancels out in GRPO advantage computation.
# - Increased W_QUALITY (0.60→0.65): quality is the main signal source.
# - Increased W_PARTIAL (0.05→0.10): creates more reward variance among
#   completions that fail on main components.
W_QUALITY = 0.65
W_TOOL_APPROPRIATENESS = 0.15
W_EFFICIENCY = 0.10
W_PARTIAL = 0.10


# ── Preferred tools per phase ─────────────────────────────────────────

PREFERRED_TOOLS = {
    "reinforcements": {"threat_analyzer", "position_evaluator"},
    "attacks": {"battle_sim", "threat_analyzer"},
    "movement": {"threat_analyzer", "position_evaluator"},
}

ALL_TOOLS = {"battle_sim", "threat_analyzer", "position_evaluator"}


# ── Main entry point ──────────────────────────────────────────────────

def compute_reward(completion: str, phase: str,
                   board_snapshot: dict, tool_log: list,
                   outcome: str = None) -> float:
    """Score a single model completion. Returns float in [0, 1].

    Args:
        completion: model's text output (after tool loop processing).
        phase: one of "reinforcements", "attacks", "movement".
        board_snapshot: game state dict from turns.jsonl with keys:
            player_name, owned_territories, border_territories,
            territory_map, reinforcements, players, turn.
        tool_log: list of {"tool_name", "kwargs", "result"} dicts
            from run_tool_loop().
        outcome: "win" or "loss" (optional). Winning games get +0.10.

    Returns:
        Float reward clamped to [0, 1].
    """
    quality = _score_format_and_quality(completion, phase, board_snapshot)
    tool_score = _score_tool_appropriateness(tool_log, phase)
    efficiency = _score_efficiency(tool_log)
    partial = _score_partial_credit(completion, phase, board_snapshot)

    reward = (W_QUALITY * quality
              + W_TOOL_APPROPRIATENESS * tool_score
              + W_EFFICIENCY * efficiency
              + W_PARTIAL * partial)

    # Round 2: win bonus added +0.10 for winning games.
    # Round 4: removed — all G completions for the same prompt share the
    # same outcome, so the bonus cancels out in GRPO advantage computation.
    # if outcome == "win":
    #     reward += WIN_BONUS

    return max(0.0, min(1.0, reward))


# ── Component 1: Format + Decision Quality ────────────────────────────

def _score_format_and_quality(completion: str, phase: str,
                              snapshot: dict) -> float:
    """Score format validity and strategic quality of the decision."""
    if phase == "reinforcements":
        return _score_reinforcements(completion, snapshot)
    elif phase == "attacks":
        return _score_attacks(completion, snapshot)
    elif phase == "movement":
        return _score_movement(completion, snapshot)
    # Unknown phase — minimal score
    return 0.3


def _score_reinforcements(completion: str, snapshot: dict) -> float:
    """Score a reinforcement decision.

    Breakdown (sums to 1.0):
      +0.20  valid JSON with "reinforcements" key
      +0.20  all territory names are owned by the player
      +0.10  troop sum matches available reinforcements
      +0.50  strategic quality — fraction of troops placed on borders
    """
    score = 0.0
    owned = set(snapshot["owned_territories"])
    borders = set(snapshot["border_territories"])
    available = snapshot["reinforcements"]

    parsed = _extract_json(completion, "reinforcements")
    if parsed is None:
        return 0.0

    reinf = parsed.get("reinforcements")
    if not isinstance(reinf, dict) or not reinf:
        return 0.0

    score += 0.20  # valid JSON

    try:
        reinf = {k: int(v) for k, v in reinf.items()}
    except (ValueError, TypeError):
        return score

    # Territory validity
    valid_count = sum(1 for name in reinf if name in owned)
    score += 0.20 * (valid_count / len(reinf))

    # Sum check
    total = sum(reinf.values())
    if total == available:
        score += 0.10
    elif available > 0 and abs(total - available) <= 1:
        score += 0.05

    # Strategic quality: troops on border territories
    # Round 2 (cliff — required all_owned, one wrong name → 0 strategic score):
    # if all_owned and total > 0:
    #     border_troops = sum(v for k, v in reinf.items() if k in borders)
    #     score += 0.50 * (border_troops / total)
    #
    # Round 4 (smooth — proportional credit for partially correct names):
    # Gives gradient signal even with some wrong names, e.g. 4/5 valid names
    # with good border placement → 0.50 * 0.80 * border_frac instead of 0.
    if total > 0:
        valid_reinf = {k: v for k, v in reinf.items() if k in owned}
        if valid_reinf:
            valid_total = sum(valid_reinf.values())
            border_troops = sum(v for k, v in valid_reinf.items() if k in borders)
            validity_frac = len(valid_reinf) / len(reinf)
            border_frac = border_troops / valid_total if valid_total > 0 else 0
            score += 0.50 * validity_frac * border_frac

    return score


def _score_attacks(completion: str, snapshot: dict) -> float:
    """Score an attack decision.

    Breakdown (sums to 1.0):
      +0.20  valid JSON with "attacks" key
      +0.20  src/target names are valid (owned, enemy, adjacent)
      +0.30  attacks use favorable odds (attacker > defender)
      +0.30  reasonable attack activity (attacks when possible)
    """
    score = 0.0
    owned = set(snapshot["owned_territories"])
    territory_map = snapshot["territory_map"]
    player_name = snapshot["player_name"]

    parsed = _extract_json(completion, "attacks")
    if parsed is None:
        return 0.0

    attacks = parsed.get("attacks")
    if not isinstance(attacks, list):
        return 0.0

    score += 0.20  # valid JSON

    # Empty attacks list — check if skipping is reasonable
    if not attacks:
        has_options = _can_attack(owned, territory_map, player_name)
        if has_options:
            # Round 2: score += 0.30  # valid but passive
            score += 0.05  # Round 3: penalize skipping when attacks available
        else:
            score += 0.80  # correctly chose not to attack
        return score

    # Validate each attack
    valid_attacks = []
    for atk in attacks:
        if not isinstance(atk, dict):
            continue
        src = atk.get("src", "")
        target = atk.get("target", "")
        if not isinstance(src, str) or not isinstance(target, str):
            continue
        if src not in owned:
            continue
        src_info = territory_map.get(src, {})
        if src_info.get("forces", 0) <= 1:
            continue
        target_info = territory_map.get(target, {})
        if target_info.get("owner") == player_name:
            continue
        if target not in src_info.get("adjacent", []):
            continue
        valid_attacks.append(atk)

    # +0.20 validity fraction
    validity_frac = len(valid_attacks) / len(attacks)
    score += 0.20 * validity_frac

    if not valid_attacks:
        return score

    # +0.30 favorable odds
    favorable = 0
    for atk in valid_attacks:
        src_forces = territory_map.get(atk["src"], {}).get("forces", 0)
        tgt_forces = territory_map.get(atk["target"], {}).get("forces", 0)
        if src_forces > tgt_forces:
            favorable += 1
    score += 0.30 * (favorable / len(valid_attacks))

    # +0.30 has valid attacks
    score += 0.30

    return score


def _score_movement(completion: str, snapshot: dict) -> float:
    """Score a movement decision.

    Breakdown (sums to 1.0):
      +0.20  valid JSON with "movement" key
      +0.20  territory names valid (both owned, adjacent, valid count)
      +0.60  strategic quality (moving troops toward borders)
    """
    score = 0.0
    owned = set(snapshot["owned_territories"])
    borders = set(snapshot["border_territories"])
    territory_map = snapshot["territory_map"]

    parsed = _extract_json(completion, "movement")
    if parsed is None:
        return 0.0

    score += 0.20  # valid JSON

    movement = parsed.get("movement")

    # null = skip movement
    if movement is None:
        # Round 2: score += 0.50  # always gave free points for skipping
        # Round 3: penalize skipping when useful moves exist
        if _can_move_useful(owned, borders, territory_map):
            score += 0.10  # penalize: should be moving troops toward borders
        else:
            score += 0.40  # no useful moves — skipping is reasonable
        return score

    if not isinstance(movement, dict):
        return score

    src = movement.get("src", "")
    target = movement.get("target", "")
    count = movement.get("count", 0)

    if not isinstance(src, str) or not isinstance(target, str):
        return score

    try:
        count = int(count)
    except (ValueError, TypeError):
        return score

    # Validate ownership, adjacency, count
    # Round 2 (cliff — early return for any validation failure):
    # if src not in owned or target not in owned:
    #     return score
    # ...
    # if count <= 0 or count >= src_forces:
    #     return score
    # score += 0.20  # all valid
    #
    # Round 4 (smooth — partial credit for partially correct moves):
    # Accumulates validity score across 4 checks instead of early-returning.
    # Creates gradient between "almost valid" and "completely wrong."
    validity = 0.0
    total_checks = 4
    if src in owned:
        validity += 1
    if target in owned:
        validity += 1
    src_info = territory_map.get(src, {})
    if target in src_info.get("adjacent", []):
        validity += 1
    src_forces = src_info.get("forces", 0)
    if 0 < count < src_forces:
        validity += 1

    score += 0.20 * (validity / total_checks)

    # Strategic quality only if fully valid (all 4 checks pass)
    if validity < total_checks:
        return score

    # Strategic quality: inland -> border is best
    src_is_border = src in borders
    tgt_is_border = target in borders

    if not src_is_border and tgt_is_border:
        score += 0.60  # inland -> border (excellent)
    elif src_is_border and tgt_is_border:
        score += 0.40  # border -> border (redistributing)
    elif not src_is_border and not tgt_is_border:
        score += 0.15  # inland -> inland (low value)
    else:
        score += 0.05  # border -> inland (bad)

    return score


# ── Component 2: Tool Use Appropriateness ──────────────────────────────

def _score_tool_appropriateness(tool_log: list, phase: str) -> float:
    """Score whether the model used relevant tools for this phase.

    Returns 0-1:
      1.0  called at least one preferred tool
      0.7  called a valid tool but not the most relevant
      0.3  called no tools
      0.1  all tool calls had errors
    """
    if not tool_log:
        return 0.3

    preferred = PREFERRED_TOOLS.get(phase, ALL_TOOLS)
    tools_called = {entry["tool_name"] for entry in tool_log}
    has_errors = any("Error" in entry.get("result", "") for entry in tool_log)

    if tools_called & preferred:
        return 1.0 if not has_errors else 0.8
    elif tools_called & ALL_TOOLS:
        return 0.7 if not has_errors else 0.5
    else:
        return 0.1


# ── Component 3: Efficiency ────────────────────────────────────────────

def _score_efficiency(tool_log: list) -> float:
    """Score efficiency based on number of tool calls.

    0 calls → 0.3   (missed information)
    1 call  → 1.0   (ideal)
    2 calls → 0.9   (good)
    3 calls → 0.5   (wasteful)
    """
    n = len(tool_log)
    if n == 0:
        return 0.3
    elif n == 1:
        return 1.0
    elif n == 2:
        return 0.9
    else:
        return 0.5


# ── Component 4: Partial Credit ────────────────────────────────────────

def _score_partial_credit(completion: str, phase: str,
                          snapshot: dict) -> float:
    """Give fine-grained credit for partial attempts.

    This creates reward variance among completions that would otherwise
    all score 0 on the main components, enabling GRPO to learn.

    Scores 0-1 based on:
      +0.25  attempted tool call syntax (<tool_call> appears)
      +0.15  closing tag present (</tool_call>)
      +0.20  attempted JSON output (``` json or { found)
      +0.20  mentions territory names from the board
      +0.20  conciseness (shorter completions score higher)
    """
    score = 0.0
    if not completion:
        return 0.0

    # Attempted tool call syntax
    if "<tool_call>" in completion:
        score += 0.50
        if "</tool_call>" in completion:
            score += 0.20

    # Attempted JSON output
    if "```json" in completion or '{"' + phase[:-1] in completion:
        score += 0.20
    elif "{" in completion and "}" in completion:
        score += 0.10

    # Mentions real territory names from the board
    territories = set(snapshot.get("owned_territories", []))
    if territories:
        mentioned = sum(1 for t in territories if t in completion)
        frac = min(1.0, mentioned / max(1, len(territories) // 3))
        score += 0.20 * frac

    # Conciseness: reward shorter completions (less rambling)
    length = len(completion)
    if length < 300:
        score += 0.20
    elif length < 600:
        score += 0.15
    elif length < 1000:
        score += 0.10
    else:
        score += 0.05

    return min(1.0, score)


# ── Helpers ────────────────────────────────────────────────────────────

def _extract_json(text: str, key: str) -> Optional[dict]:
    """Extract a JSON object containing the given key from model output.

    Tries fenced ```json blocks first, then bare JSON, then regex fallback.
    """
    if not text:
        return None

    # Try ```json ... ``` block
    fence_match = re.search(r'```json\s*\n?(.*?)\n?\s*```', text, re.DOTALL)
    if fence_match:
        try:
            data = json.loads(fence_match.group(1).strip())
            if isinstance(data, dict) and key in data:
                return data
        except json.JSONDecodeError:
            pass

    # Try bare JSON
    try:
        data = json.loads(text.strip())
        if isinstance(data, dict) and key in data:
            return data
    except json.JSONDecodeError:
        pass

    # Regex fallback: find a JSON object containing the key
    # Handles nested braces one level deep (covers all decision formats)
    pattern = (r'\{[^{}]*"' + re.escape(key)
               + r'"[^{}]*(?:\{[^{}]*\}[^{}]*|\[[^\[\]]*\][^{}]*)*\}')
    match = re.search(pattern, text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            if isinstance(data, dict) and key in data:
                return data
        except json.JSONDecodeError:
            pass

    return None


def _can_move_useful(owned: set, borders: set, territory_map: dict) -> bool:
    """Check if player has inland troops that could move toward a border."""
    for name in owned:
        if name in borders:
            continue
        info = territory_map.get(name, {})
        if info.get("forces", 0) <= 1:
            continue
        for adj in info.get("adjacent", []):
            if adj in borders and adj in owned:
                return True
    return False


def _can_attack(owned: set, territory_map: dict, player_name: str) -> bool:
    """Check if the player has any territory that could attack."""
    for name in owned:
        info = territory_map.get(name, {})
        if info.get("forces", 0) <= 1:
            continue
        for adj_name in info.get("adjacent", []):
            adj_info = territory_map.get(adj_name, {})
            if adj_info.get("owner") != player_name:
                return True
    return False
