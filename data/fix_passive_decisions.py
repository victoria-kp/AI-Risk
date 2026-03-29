"""Fix passive attack and movement decisions in SFT data.

The original Gemini benchmark data has {"attacks": null} for ALL 1,275
attack-phase examples — the model never saw an actual attack during SFT.
This script generates real attack and movement decisions from the board
snapshots.

Takes the trimmed SFT data and:
1. Replaces null/empty attacks with real attacks (best force-ratio targets)
2. Replaces null movement with real inland→border moves
3. Updates tool call prefixes (battle_sim for real attacks)
4. Updates bridge text to match the new decisions

Original data is NOT modified. Output goes to a new file.

Usage:
    python data/fix_passive_decisions.py
    python data/fix_passive_decisions.py --input data/sft_trimmed/turns.jsonl \
                                          --output data/sft_fixed/turns.jsonl
"""

import argparse
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'pyrisk_vendor'))

from tools.battle_sim import simulate_battle
from risk_env.tool_interface import _format_result


INPUT_PATH = "data/sft_trimmed/turns.jsonl"
OUTPUT_PATH = "data/sft_fixed/turns.jsonl"


# ── Attack generation ────────────────────────────────────────────────

def find_viable_attacks(snapshot):
    """Find all viable attacks sorted by force ratio (best first).

    Returns list of {"src", "target", "src_forces", "tgt_forces", "ratio"}.
    """
    owned = set(snapshot.get("owned_territories", []))
    territory_map = snapshot.get("territory_map", {})
    player_name = snapshot.get("player_name", "")

    candidates = []
    for name in owned:
        info = territory_map.get(name, {})
        src_forces = info.get("forces", 0)
        if src_forces <= 1:
            continue
        for adj_name in info.get("adjacent", []):
            adj_info = territory_map.get(adj_name, {})
            if adj_info.get("owner") == player_name:
                continue
            tgt_forces = adj_info.get("forces", 0)
            if tgt_forces <= 0:
                continue
            ratio = src_forces / tgt_forces
            candidates.append({
                "src": name,
                "target": adj_name,
                "src_forces": src_forces,
                "tgt_forces": tgt_forces,
                "ratio": ratio,
            })

    # Sort by force ratio (best odds first)
    candidates.sort(key=lambda x: x["ratio"], reverse=True)
    return candidates


def generate_attack_decision(snapshot, max_attacks=3):
    """Generate a realistic attack decision from board state.

    Picks 1-3 attacks with favorable odds (ratio > 1.0).
    Returns (attacks_list, battle_sim_prefix) or (None, None) if no viable attacks.
    """
    candidates = find_viable_attacks(snapshot)
    if not candidates:
        return None, None

    # Pick attacks with ratio > 1.0, up to max_attacks
    # Use different src territories to avoid attacking from the same place twice
    attacks = []
    used_srcs = set()
    for c in candidates:
        if c["ratio"] < 1.0:
            break
        if c["src"] in used_srcs:
            continue
        attacks.append({
            "src": c["src"],
            "target": c["target"],
            "count": c["src_forces"] - 1,  # attack with all available
        })
        used_srcs.add(c["src"])
        if len(attacks) >= max_attacks:
            break

    # If no attacks with ratio > 1.0, take the best one anyway
    # (teaches model to attack even with marginal odds)
    if not attacks and candidates:
        best = candidates[0]
        if best["ratio"] >= 0.7:  # at least 70% of defender's forces
            attacks.append({
                "src": best["src"],
                "target": best["target"],
                "count": best["src_forces"] - 1,
            })

    if not attacks:
        return None, None

    # Generate battle_sim tool prefix for the first attack
    first = attacks[0]
    src_forces = None
    tgt_forces = None
    territory_map = snapshot.get("territory_map", {})
    for c in candidates:
        if c["src"] == first["src"] and c["target"] == first["target"]:
            src_forces = c["src_forces"]
            tgt_forces = c["tgt_forces"]
            break

    prefix = None
    if src_forces and tgt_forces:
        try:
            sim_result = simulate_battle(src_forces, tgt_forces,
                                         num_simulations=1000)
            call = f"<tool_call>battle_sim(attacking={src_forces}, defending={tgt_forces})</tool_call>"
            result_text = _format_result(sim_result)
            result = f"<tool_result>\n{result_text}\n</tool_result>"
            prefix = f"{call}\n{result}"
        except Exception:
            pass

    return attacks, prefix


# ── Movement generation ──────────────────────────────────────────────

def generate_movement_decision(snapshot):
    """Generate a realistic movement from inland to border.

    Returns movement dict or None.
    """
    owned = set(snapshot.get("owned_territories", []))
    borders = set(snapshot.get("border_territories", []))
    territory_map = snapshot.get("territory_map", {})

    # Find inland territories with troops that could move to a border
    candidates = []
    for name in owned:
        if name in borders:
            continue
        info = territory_map.get(name, {})
        forces = info.get("forces", 0)
        if forces <= 1:
            continue
        for adj in info.get("adjacent", []):
            if adj in owned and adj in borders:
                candidates.append({
                    "src": name,
                    "target": adj,
                    "count": forces - 1,
                    "forces": forces,
                })

    if not candidates:
        return None

    # Pick the one with the most troops to move
    candidates.sort(key=lambda x: x["forces"], reverse=True)
    best = candidates[0]
    return {
        "src": best["src"],
        "target": best["target"],
        "count": best["count"],
    }


# ── Response rebuilding ──────────────────────────────────────────────

ATTACK_BRIDGES = [
    "The simulation shows favorable odds. Proceeding with the attack.",
    "Based on the battle odds, here are my attacks.",
    "The analysis supports attacking vulnerable targets.",
    "Attacking where I have force advantage.",
    "Battle odds look good — launching attacks on vulnerable positions.",
]

MOVEMENT_BRIDGES = [
    "Based on the position evaluation, redistributing troops to strengthen borders.",
    "Moving troops to reinforce vulnerable positions.",
    "Repositioning based on the strategic assessment.",
    "Consolidating forces on border territories.",
]


def rebuild_attack_response(entry, attacks, tool_prefix, rng):
    """Replace null/empty attacks with real attacks in the response."""
    response = entry["response"]

    # Build new JSON block
    attacks_json = json.dumps({"attacks": attacks}, indent=2)
    json_block = f"```json\n{attacks_json}\n```"

    # Build bridge
    bridge = rng.choice(ATTACK_BRIDGES)

    # Check if response has tool_call already
    has_tool = "<tool_call>" in response

    if has_tool:
        # Response already has tool call prefix — replace everything after
        # </tool_result> with bridge + new JSON
        tr_end = response.find("</tool_result>")
        if tr_end != -1:
            # Keep everything up to and including </tool_result>
            # But replace the tool call if it was threat_analyzer (for empty attacks)
            if "threat_analyzer" in response[:tr_end] and tool_prefix:
                new_response = f"{tool_prefix}\n\n{bridge}\n\n{json_block}"
            else:
                prefix_part = response[:tr_end + len("</tool_result>")]
                new_response = f"{prefix_part}\n\n{bridge}\n\n{json_block}"
        else:
            # No tool_result end — just replace from tool_call
            if tool_prefix:
                new_response = f"{tool_prefix}\n\n{bridge}\n\n{json_block}"
            else:
                new_response = f"{bridge}\n\n{json_block}"
    else:
        # No tool call in response — add one
        if tool_prefix:
            new_response = f"{tool_prefix}\n\n{bridge}\n\n{json_block}"
        else:
            new_response = f"{bridge}\n\n{json_block}"

    return new_response


def rebuild_movement_response(entry, movement, rng):
    """Replace null movement with real movement in the response."""
    response = entry["response"]

    # Build new JSON block
    movement_json = json.dumps({"movement": movement}, indent=2)
    json_block = f"```json\n{movement_json}\n```"

    bridge = rng.choice(MOVEMENT_BRIDGES)

    # Keep tool call prefix if present
    tr_end = response.find("</tool_result>")
    if tr_end != -1:
        prefix_part = response[:tr_end + len("</tool_result>")]
        new_response = f"{prefix_part}\n\n{bridge}\n\n{json_block}"
    else:
        new_response = f"{bridge}\n\n{json_block}"

    return new_response


# ── Detection helpers ────────────────────────────────────────────────

def has_empty_attacks(response):
    """Check if response contains null or empty attacks."""
    match = re.search(r'```json\s*\n?(.*?)\n?\s*```', response, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(1).strip())
            if isinstance(data, dict):
                attacks = data.get("attacks")
                if attacks is None or attacks == []:
                    return True
        except (json.JSONDecodeError, KeyError):
            pass
    return False


def has_null_movement(response):
    """Check if response contains null movement."""
    match = re.search(r'```json\s*\n?(.*?)\n?\s*```', response, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(1).strip())
            if isinstance(data, dict):
                movement = data.get("movement")
                if movement is None:
                    return True
        except (json.JSONDecodeError, KeyError):
            pass
    return False


# ── Main ─────────────────────────────────────────────────────────────

def fix(input_path, output_path, seed=42):
    rng = random.Random(seed)

    entries = []
    with open(input_path) as f:
        for line in f:
            entries.append(json.loads(line))

    stats = {
        "total": len(entries),
        "attacks_fixed": 0,
        "attacks_no_viable": 0,
        "attacks_already_real": 0,
        "movement_fixed": 0,
        "movement_no_viable": 0,
        "movement_already_real": 0,
    }

    for entry in entries:
        phase = entry["phase"]
        response = entry["response"]
        snapshot = entry.get("board_snapshot", {})

        if phase == "attacks":
            if has_empty_attacks(response):
                attacks, tool_prefix = generate_attack_decision(snapshot)
                if attacks:
                    entry["response"] = rebuild_attack_response(
                        entry, attacks, tool_prefix, rng
                    )
                    stats["attacks_fixed"] += 1
                else:
                    stats["attacks_no_viable"] += 1
            else:
                stats["attacks_already_real"] += 1

        elif phase == "movement":
            if has_null_movement(response):
                movement = generate_movement_decision(snapshot)
                if movement:
                    entry["response"] = rebuild_movement_response(
                        entry, movement, rng
                    )
                    stats["movement_fixed"] += 1
                else:
                    stats["movement_no_viable"] += 1
            else:
                stats["movement_already_real"] += 1

    # Write output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")

    # Print stats
    print(f"=== Fix Passive Decisions ===")
    print(f"Input:              {input_path} ({stats['total']} entries)")
    print(f"")
    print(f"Attacks phase:")
    print(f"  Fixed (null→real): {stats['attacks_fixed']}")
    print(f"  No viable targets: {stats['attacks_no_viable']}")
    print(f"  Already real:      {stats['attacks_already_real']}")
    print(f"")
    print(f"Movement phase:")
    print(f"  Fixed (null→real): {stats['movement_fixed']}")
    print(f"  No viable moves:   {stats['movement_no_viable']}")
    print(f"  Already real:      {stats['movement_already_real']}")
    print(f"")
    print(f"Output:             {output_path}")

    # Write summary
    summary_path = output_path.replace(".jsonl", "_summary.json")
    with open(summary_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Summary:            {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fix passive attack/movement decisions in SFT data"
    )
    parser.add_argument("--input", type=str, default=INPUT_PATH,
                        help=f"Input turns.jsonl (default: {INPUT_PATH})")
    parser.add_argument("--output", type=str, default=OUTPUT_PATH,
                        help=f"Output path (default: {OUTPUT_PATH})")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    args = parser.parse_args()

    fix(args.input, args.output, args.seed)
