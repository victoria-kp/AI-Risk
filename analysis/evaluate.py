"""Evaluate model quality from turns.jsonl decision data.

Computes decision-level metrics from one or more turns.jsonl files and
prints a side-by-side comparison table. No game-playing required — works
entirely from the logged data.

Metrics computed (per phase and overall):
  - Fallback rate: % of decisions where parsing failed
  - Valid JSON rate: % of responses containing parseable phase-specific JSON
  - Valid territory rate: % of territory names that exist on the board
  - Border reinforcement %: fraction of reinforcement troops on borders
  - Favorable attack ratio: % of attacks with attacker > defender
  - Movement quality: % of movements that go toward borders
  - Avg response length: verbosity of model output
  - Win rate: % of games won (game-level)

Usage:
    # Compare Gemini vs Round 1 Qwen
    python analysis/evaluate.py \\
        data/benchmark_results/turns.jsonl \\
        data/on_policy_results/turns.jsonl \\
        --labels "Gemini" "Round 1 Qwen"

    # Single dataset
    python analysis/evaluate.py data/benchmark_results/turns.jsonl
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


# ── JSON extraction (mirrors training/reward.py) ────────────────────────

def _extract_json(text, key):
    """Extract a JSON object containing the given key from model output."""
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

    # Regex fallback
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


# ── Per-entry metric computation ─────────────────────────────────────────

def _analyze_reinforcements(entry):
    """Analyze a reinforcement decision. Returns dict of metrics."""
    snap = entry["board_snapshot"]
    owned = set(snap["owned_territories"])
    borders = set(snap["border_territories"])
    available = snap["reinforcements"]

    metrics = {
        "valid_json": False,
        "valid_territories": 0.0,
        "correct_sum": False,
        "border_pct": None,
    }

    # Try to extract JSON from response
    parsed = _extract_json(entry["response"], "reinforcements")
    if parsed is None:
        return metrics

    reinf = parsed.get("reinforcements")
    if not isinstance(reinf, dict) or not reinf:
        return metrics

    metrics["valid_json"] = True

    try:
        reinf = {k: int(v) for k, v in reinf.items()}
    except (ValueError, TypeError):
        return metrics

    # Territory validity
    valid_count = sum(1 for name in reinf if name in owned)
    metrics["valid_territories"] = valid_count / len(reinf) if reinf else 0.0

    # Sum check
    total = sum(reinf.values())
    metrics["correct_sum"] = (total == available)

    # Border reinforcement %
    if total > 0 and valid_count == len(reinf):
        border_troops = sum(v for k, v in reinf.items() if k in borders)
        metrics["border_pct"] = border_troops / total

    return metrics


def _analyze_attacks(entry):
    """Analyze an attack decision. Returns dict of metrics."""
    snap = entry["board_snapshot"]
    owned = set(snap["owned_territories"])
    territory_map = snap["territory_map"]
    player_name = snap["player_name"]

    metrics = {
        "valid_json": False,
        "valid_attacks": 0.0,
        "favorable_ratio": None,
        "attack_count": 0,
        "skipped": False,
    }

    parsed = _extract_json(entry["response"], "attacks")
    if parsed is None:
        return metrics

    attacks = parsed.get("attacks")
    if not isinstance(attacks, list):
        return metrics

    metrics["valid_json"] = True

    if not attacks:
        metrics["skipped"] = True
        return metrics

    metrics["attack_count"] = len(attacks)

    # Validate each attack
    valid = 0
    favorable = 0
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
        valid += 1
        if src_info.get("forces", 0) > target_info.get("forces", 0):
            favorable += 1

    metrics["valid_attacks"] = valid / len(attacks) if attacks else 0.0
    if valid > 0:
        metrics["favorable_ratio"] = favorable / valid

    return metrics


def _analyze_movement(entry):
    """Analyze a movement decision. Returns dict of metrics."""
    snap = entry["board_snapshot"]
    owned = set(snap["owned_territories"])
    borders = set(snap["border_territories"])
    territory_map = snap["territory_map"]

    metrics = {
        "valid_json": False,
        "skipped": False,
        "valid_movement": False,
        "toward_border": None,
    }

    parsed = _extract_json(entry["response"], "movement")
    if parsed is None:
        return metrics

    metrics["valid_json"] = True

    movement = parsed.get("movement")
    if movement is None:
        metrics["skipped"] = True
        return metrics

    if not isinstance(movement, dict):
        return metrics

    src = movement.get("src", "")
    target = movement.get("target", "")
    count = movement.get("count", 0)

    if not isinstance(src, str) or not isinstance(target, str):
        return metrics

    try:
        count = int(count)
    except (ValueError, TypeError):
        return metrics

    # Validate
    if src not in owned or target not in owned:
        return metrics
    src_info = territory_map.get(src, {})
    if target not in src_info.get("adjacent", []):
        return metrics
    if count <= 0 or count >= src_info.get("forces", 0):
        return metrics

    metrics["valid_movement"] = True

    # Direction quality
    src_is_border = src in borders
    tgt_is_border = target in borders
    if not src_is_border and tgt_is_border:
        metrics["toward_border"] = True  # inland -> border (best)
    elif src_is_border and not tgt_is_border:
        metrics["toward_border"] = False  # border -> inland (worst)
    else:
        metrics["toward_border"] = None  # neutral

    return metrics


# ── Aggregate metrics ────────────────────────────────────────────────────

def compute_metrics(entries):
    """Compute all metrics from a list of turn entries.

    Returns dict with overall and per-phase metrics.
    """
    # Filter out placement
    entries = [e for e in entries if e["phase"] != "placement"]
    if not entries:
        return {"error": "No non-placement entries"}

    results = {
        "total_decisions": len(entries),
        "phases": defaultdict(int),
    }

    # Game-level: win rate
    games = {}
    for e in entries:
        gid = e.get("game_id", 0)
        if gid not in games:
            games[gid] = {
                "outcome": e["outcome"],
                "matchup": e.get("matchup", "unknown"),
            }
    total_games = len(games)
    wins = sum(1 for g in games.values() if g["outcome"] == "win")
    results["win_rate"] = wins / total_games if total_games else 0
    results["total_games"] = total_games
    results["wins"] = wins

    # Per-matchup win rate
    matchup_games = defaultdict(lambda: {"wins": 0, "total": 0})
    for g in games.values():
        m = g["matchup"]
        matchup_games[m]["total"] += 1
        if g["outcome"] == "win":
            matchup_games[m]["wins"] += 1
    results["matchup_win_rates"] = {
        m: v["wins"] / v["total"] if v["total"] else 0
        for m, v in sorted(matchup_games.items())
    }

    # Overall fallback rate
    fallback_count = sum(1 for e in entries if e.get("fallback", False))
    results["fallback_rate"] = fallback_count / len(entries)

    # Average response length
    response_lengths = [len(e.get("response") or "") for e in entries]
    results["avg_response_length"] = sum(response_lengths) / len(response_lengths)

    # Tool call rate (from response text)
    tool_call_count = sum(
        1 for e in entries if "<tool_call>" in (e.get("response") or "")
    )
    results["tool_call_rate"] = tool_call_count / len(entries)

    # Per-phase metrics
    phase_entries = defaultdict(list)
    for e in entries:
        phase_entries[e["phase"]].append(e)

    # Reinforcements
    reinf_entries = phase_entries.get("reinforcements", [])
    if reinf_entries:
        results["phases"]["reinforcements"] = len(reinf_entries)
        analyses = [_analyze_reinforcements(e) for e in reinf_entries]

        valid_json = sum(1 for a in analyses if a["valid_json"])
        results["reinf_valid_json_rate"] = valid_json / len(analyses)

        # Territory validity (among those with valid JSON)
        with_json = [a for a in analyses if a["valid_json"]]
        if with_json:
            results["reinf_valid_territory_rate"] = (
                sum(a["valid_territories"] for a in with_json) / len(with_json)
            )
        else:
            results["reinf_valid_territory_rate"] = 0.0

        # Correct sum
        results["reinf_correct_sum_rate"] = (
            sum(1 for a in analyses if a["correct_sum"]) / len(analyses)
        )

        # Border reinforcement %
        border_vals = [a["border_pct"] for a in analyses
                       if a["border_pct"] is not None]
        results["reinf_border_pct"] = (
            sum(border_vals) / len(border_vals) if border_vals else None
        )

        # Fallback rate for this phase
        results["reinf_fallback_rate"] = (
            sum(1 for e in reinf_entries if e.get("fallback")) / len(reinf_entries)
        )

    # Attacks
    atk_entries = phase_entries.get("attacks", [])
    if atk_entries:
        results["phases"]["attacks"] = len(atk_entries)
        analyses = [_analyze_attacks(e) for e in atk_entries]

        valid_json = sum(1 for a in analyses if a["valid_json"])
        results["atk_valid_json_rate"] = valid_json / len(analyses)

        # Valid attack structure
        with_attacks = [a for a in analyses
                        if a["valid_json"] and not a["skipped"] and a["attack_count"] > 0]
        if with_attacks:
            results["atk_valid_structure_rate"] = (
                sum(a["valid_attacks"] for a in with_attacks) / len(with_attacks)
            )
        else:
            results["atk_valid_structure_rate"] = None

        # Favorable ratio
        favorable_vals = [a["favorable_ratio"] for a in analyses
                          if a["favorable_ratio"] is not None]
        results["atk_favorable_ratio"] = (
            sum(favorable_vals) / len(favorable_vals) if favorable_vals else None
        )

        # Skip rate
        results["atk_skip_rate"] = (
            sum(1 for a in analyses if a["skipped"]) / len(analyses)
        )

        # Avg attacks per turn (when attacking)
        attacking = [a for a in analyses if not a["skipped"] and a["attack_count"] > 0]
        results["atk_avg_count"] = (
            sum(a["attack_count"] for a in attacking) / len(attacking)
            if attacking else 0
        )

        results["atk_fallback_rate"] = (
            sum(1 for e in atk_entries if e.get("fallback")) / len(atk_entries)
        )

    # Movement
    mov_entries = phase_entries.get("movement", [])
    if mov_entries:
        results["phases"]["movement"] = len(mov_entries)
        analyses = [_analyze_movement(e) for e in mov_entries]

        valid_json = sum(1 for a in analyses if a["valid_json"])
        results["mov_valid_json_rate"] = valid_json / len(analyses)

        # Valid movement
        with_movement = [a for a in analyses
                         if a["valid_json"] and not a["skipped"]]
        if with_movement:
            results["mov_valid_rate"] = (
                sum(1 for a in with_movement if a["valid_movement"]) / len(with_movement)
            )
        else:
            results["mov_valid_rate"] = None

        # Toward border rate
        border_dirs = [a["toward_border"] for a in analyses
                       if a["toward_border"] is not None]
        results["mov_toward_border_rate"] = (
            sum(1 for d in border_dirs if d) / len(border_dirs)
            if border_dirs else None
        )

        # Skip rate
        results["mov_skip_rate"] = (
            sum(1 for a in analyses if a["skipped"]) / len(analyses)
        )

        results["mov_fallback_rate"] = (
            sum(1 for e in mov_entries if e.get("fallback")) / len(mov_entries)
        )

    return results


# ── Display ──────────────────────────────────────────────────────────────

def _fmt(val, fmt=".1%"):
    """Format a value for display."""
    if val is None:
        return "N/A"
    if fmt == ".1%":
        return f"{val:.1%}"
    elif fmt == ".0f":
        return f"{val:.0f}"
    elif fmt == ".1f":
        return f"{val:.1f}"
    return str(val)


def print_comparison(all_results, labels):
    """Print a side-by-side comparison table."""
    col_width = max(16, max(len(l) for l in labels) + 2)
    label_row = "".join(l.rjust(col_width) for l in labels)

    def row(name, key, fmt=".1%"):
        vals = "".join(
            _fmt(r.get(key), fmt).rjust(col_width)
            for r in all_results
        )
        print(f"  {name:<30}{vals}")

    print("=" * (32 + col_width * len(labels)))
    print(f"  {'METRIC':<30}{label_row}")
    print("=" * (32 + col_width * len(labels)))

    # Game-level
    print("\n  GAME-LEVEL")
    print("  " + "-" * 28)
    row("Total games", "total_games", ".0f")
    row("Total decisions", "total_decisions", ".0f")
    row("Win rate", "win_rate")

    # Per-matchup win rates
    all_matchups = set()
    for r in all_results:
        all_matchups.update(r.get("matchup_win_rates", {}).keys())
    for m in sorted(all_matchups):
        vals = "".join(
            _fmt(r.get("matchup_win_rates", {}).get(m), ".1%").rjust(col_width)
            for r in all_results
        )
        print(f"    {m:<28}{vals}")

    # Overall
    print(f"\n  OVERALL")
    print("  " + "-" * 28)
    row("Fallback rate", "fallback_rate")
    row("Tool call rate (in response)", "tool_call_rate")
    row("Avg response length (chars)", "avg_response_length", ".0f")

    # Reinforcements
    print(f"\n  REINFORCEMENTS")
    print("  " + "-" * 28)
    counts = "".join(
        str(r.get("phases", {}).get("reinforcements", 0)).rjust(col_width)
        for r in all_results
    )
    print(f"  {'Decisions':<30}{counts}")
    row("Fallback rate", "reinf_fallback_rate")
    row("Valid JSON rate", "reinf_valid_json_rate")
    row("Valid territory rate", "reinf_valid_territory_rate")
    row("Correct troop sum", "reinf_correct_sum_rate")
    row("Border reinforcement %", "reinf_border_pct")

    # Attacks
    print(f"\n  ATTACKS")
    print("  " + "-" * 28)
    counts = "".join(
        str(r.get("phases", {}).get("attacks", 0)).rjust(col_width)
        for r in all_results
    )
    print(f"  {'Decisions':<30}{counts}")
    row("Fallback rate", "atk_fallback_rate")
    row("Valid JSON rate", "atk_valid_json_rate")
    row("Valid attack structure", "atk_valid_structure_rate")
    row("Favorable odds ratio", "atk_favorable_ratio")
    row("Skip (no attack) rate", "atk_skip_rate")
    row("Avg attacks per turn", "atk_avg_count", ".1f")

    # Movement
    print(f"\n  MOVEMENT")
    print("  " + "-" * 28)
    counts = "".join(
        str(r.get("phases", {}).get("movement", 0)).rjust(col_width)
        for r in all_results
    )
    print(f"  {'Decisions':<30}{counts}")
    row("Fallback rate", "mov_fallback_rate")
    row("Valid JSON rate", "mov_valid_json_rate")
    row("Valid movement rate", "mov_valid_rate")
    row("Toward-border rate", "mov_toward_border_rate")
    row("Skip (no move) rate", "mov_skip_rate")

    print("=" * (32 + col_width * len(labels)))


# ── Main ──────────────────────────────────────────────────────────────────

def load_entries(path):
    """Load all entries from a turns.jsonl file."""
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate model decision quality from turns.jsonl files"
    )
    parser.add_argument(
        "data", nargs="+",
        help="Path(s) to turns.jsonl files to evaluate",
    )
    parser.add_argument(
        "--labels", nargs="+", default=None,
        help="Labels for each dataset (default: filenames)",
    )
    args = parser.parse_args()

    if args.labels and len(args.labels) != len(args.data):
        print(f"Error: {len(args.labels)} labels for {len(args.data)} datasets")
        sys.exit(1)

    labels = args.labels or [
        os.path.basename(os.path.dirname(p)) for p in args.data
    ]

    all_results = []
    for path, label in zip(args.data, labels):
        print(f"Loading {label}: {path}")
        entries = load_entries(path)
        print(f"  {len(entries)} total entries")
        metrics = compute_metrics(entries)
        all_results.append(metrics)

    print()
    print_comparison(all_results, labels)


if __name__ == "__main__":
    main()
