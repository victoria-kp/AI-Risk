"""Augment SFT training data with realistic tool calls.

Reads benchmark turns.jsonl, filters bad examples, and prepends
proper <tool_call>/<tool_result> blocks to responses using the
actual tool implementations and board_snapshot data.

This teaches the model:
1. When to call tools (phase-appropriate tool selection)
2. Correct <tool_call> syntax
3. How to interpret <tool_result> output
4. How to follow up with a JSON decision

Original data is NOT modified. Output goes to a new file.

Usage:
    python data/augment_sft_data.py
    python data/augment_sft_data.py --input data/benchmark_results/turns.jsonl \
                                     --output data/sft_augmented/turns.jsonl
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
from tools.threat_analyzer import analyze_threats_from_snapshot
from tools.position_evaluator import evaluate_position_from_snapshot
from risk_env.tool_interface import _format_result


# ── Defaults ──────────────────────────────────────────────────────────

INPUT_PATH = "data/benchmark_results/turns.jsonl"
OUTPUT_PATH = "data/sft_augmented/turns.jsonl"
AUGMENT_RATIO = 0.70  # augment 70% of examples, leave 30% without tools


# ── Filtering ─────────────────────────────────────────────────────────

def is_bad_example(entry):
    """Return True if this example should be excluded from SFT data."""
    if entry["phase"] == "placement":
        return True
    if entry.get("fallback", False):
        return True
    response = entry.get("response")
    if not response:
        return True
    # Has <tool_call> but no JSON fence — teaches pure failure
    if "<tool_call>" in response and "```json" not in response:
        return True
    # Has hallucinated <tool_result> — teaches model to fake results
    if "<tool_result>" in response:
        return True
    # Degenerate long response (spam/repetition)
    if len(response) > 5000:
        return True
    return False


def strip_tool_artifacts(response):
    """Remove any existing <tool_call>/<tool_result> tags from response."""
    # Remove <tool_call>...</tool_call> blocks
    response = re.sub(r'<tool_call>.*?</tool_call>', '', response, flags=re.DOTALL)
    # Remove </tool_call> fragments (malformed)
    response = re.sub(r'</tool_call>', '', response)
    # Remove <tool_result>...</tool_result> blocks
    response = re.sub(r'<tool_result>.*?</tool_result>', '', response, flags=re.DOTALL)
    # Remove </tool_result> fragments
    response = re.sub(r'</tool_result>', '', response)
    # Clean up excess whitespace from removals
    response = re.sub(r'\n{3,}', '\n\n', response)
    return response.strip()


# ── Tool call generation ──────────────────────────────────────────────

def generate_tool_prefix_reinforcements(snapshot):
    """Generate a threat_analyzer() call + result for reinforcement phase."""
    try:
        threats = analyze_threats_from_snapshot(snapshot)
    except Exception:
        return None

    if not threats:
        return None

    call = "<tool_call>threat_analyzer()</tool_call>"
    result_text = _format_result(threats[:5])  # top 5 threats
    result = f"<tool_result>\n{result_text}\n</tool_result>"
    return f"{call}\n{result}\n\n"


def generate_tool_prefix_attacks(response, snapshot):
    """Generate battle_sim() call(s) + result for attack phase.

    If the decision includes attacks, simulate the first attack.
    If empty attacks, use threat_analyzer instead.
    """
    # Try to parse the JSON decision to find attacks
    attacks = _extract_attacks(response)

    if attacks:
        # Simulate the first attack
        src = attacks[0].get("src", "")
        target = attacks[0].get("target", "")
        territory_map = snapshot.get("territory_map", {})
        src_forces = territory_map.get(src, {}).get("forces", 0)
        tgt_forces = territory_map.get(target, {}).get("forces", 0)

        if src_forces > 1 and tgt_forces > 0:
            try:
                sim_result = simulate_battle(src_forces, tgt_forces,
                                             num_simulations=1000)
            except Exception:
                return None

            call = f"<tool_call>battle_sim(attacking={src_forces}, defending={tgt_forces})</tool_call>"
            result_text = _format_result(sim_result)
            result = f"<tool_result>\n{result_text}\n</tool_result>"
            return f"{call}\n{result}\n\n"

    # No attacks or can't parse — use threat_analyzer
    try:
        threats = analyze_threats_from_snapshot(snapshot)
    except Exception:
        return None

    if not threats:
        return None

    call = "<tool_call>threat_analyzer()</tool_call>"
    result_text = _format_result(threats[:5])
    result = f"<tool_result>\n{result_text}\n</tool_result>"
    return f"{call}\n{result}\n\n"


def generate_tool_prefix_movement(snapshot):
    """Generate a position_evaluator() call + result for movement phase."""
    try:
        position = evaluate_position_from_snapshot(snapshot)
    except Exception:
        return None

    if not position:
        return None

    call = "<tool_call>position_evaluator()</tool_call>"
    result_text = _format_result(position)
    result = f"<tool_result>\n{result_text}\n</tool_result>"
    return f"{call}\n{result}\n\n"


def _extract_attacks(response):
    """Try to extract attack list from response JSON."""
    # Try ```json block
    match = re.search(r'```json\s*\n?(.*?)\n?\s*```', response, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(1).strip())
            if isinstance(data, dict) and "attacks" in data:
                attacks = data["attacks"]
                if isinstance(attacks, list) and attacks:
                    return attacks
        except (json.JSONDecodeError, KeyError):
            pass
    return None


# ── Main ──────────────────────────────────────────────────────────────

def augment(input_path, output_path, augment_ratio=AUGMENT_RATIO, seed=42):
    random.seed(seed)

    # Read and filter
    all_entries = []
    with open(input_path) as f:
        for line in f:
            all_entries.append(json.loads(line))

    total_raw = len(all_entries)

    # Filter
    clean_entries = [e for e in all_entries if not is_bad_example(e)]
    filtered_count = total_raw - len(clean_entries)

    # Strip existing tool artifacts from responses
    for entry in clean_entries:
        entry["response"] = strip_tool_artifacts(entry["response"])

    # Augment
    augmented_count = 0
    skipped_augment = 0

    for entry in clean_entries:
        # Decide whether to augment this example
        if random.random() > augment_ratio:
            continue

        snapshot = entry.get("board_snapshot", {})
        phase = entry["phase"]

        if phase == "reinforcements":
            prefix = generate_tool_prefix_reinforcements(snapshot)
        elif phase == "attacks":
            prefix = generate_tool_prefix_attacks(entry["response"], snapshot)
        elif phase == "movement":
            prefix = generate_tool_prefix_movement(snapshot)
        else:
            prefix = None

        if prefix:
            entry["response"] = prefix + entry["response"]
            augmented_count += 1
        else:
            skipped_augment += 1

    # Write output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        for entry in clean_entries:
            f.write(json.dumps(entry) + "\n")

    # Stats
    phase_counts = {}
    for entry in clean_entries:
        p = entry["phase"]
        phase_counts[p] = phase_counts.get(p, 0) + 1

    print(f"=== SFT Data Augmentation ===")
    print(f"Input:          {input_path} ({total_raw} entries)")
    print(f"Filtered out:   {filtered_count} bad examples")
    print(f"Clean examples: {len(clean_entries)}")
    print(f"Augmented:      {augmented_count} with tool calls ({augment_ratio:.0%} target)")
    print(f"Skipped augment:{skipped_augment} (tool generation failed)")
    print(f"Phases:         {phase_counts}")
    print(f"Output:         {output_path}")
    print()

    # Write summary
    summary_path = output_path.replace(".jsonl", "_summary.json")
    summary = {
        "input": input_path,
        "output": output_path,
        "total_raw": total_raw,
        "filtered_out": filtered_count,
        "clean_examples": len(clean_entries),
        "augmented_with_tools": augmented_count,
        "augment_ratio": augment_ratio,
        "phase_counts": phase_counts,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary:        {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Augment SFT training data with realistic tool calls"
    )
    parser.add_argument("--input", type=str, default=INPUT_PATH,
                        help=f"Input turns.jsonl (default: {INPUT_PATH})")
    parser.add_argument("--output", type=str, default=OUTPUT_PATH,
                        help=f"Output path (default: {OUTPUT_PATH})")
    parser.add_argument("--augment-ratio", type=float, default=AUGMENT_RATIO,
                        help=f"Fraction to augment (default: {AUGMENT_RATIO})")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    args = parser.parse_args()

    augment(args.input, args.output, args.augment_ratio, args.seed)
