"""Trim augmented SFT data to be concise.

Takes the augmented turns.jsonl and trims verbose reasoning between
<tool_result> and ```json to 1-2 sentences. This teaches the model
to go: tool call → tool result → brief reasoning → JSON decision.

The original augmented data has avg 1,185 chars of reasoning between
tool result and JSON. This script trims it to ~50-150 chars.

Original data is NOT modified. Output goes to a new file.

Usage:
    python data/trim_sft_data.py
    python data/trim_sft_data.py --input data/sft_augmented/turns.jsonl \
                                   --output data/sft_trimmed/turns.jsonl
"""

import argparse
import json
import os
import re
import sys


INPUT_PATH = "data/sft_augmented/turns.jsonl"
OUTPUT_PATH = "data/sft_trimmed/turns.jsonl"

# Short reasoning templates per phase
REINFORCEMENT_BRIDGES = [
    "Based on the threat analysis, I'll reinforce the most vulnerable border territories.",
    "The threats show my borders are exposed. Reinforcing key positions.",
    "Prioritizing reinforcement on threatened border territories.",
    "Reinforcing borders based on threat assessment.",
]

ATTACK_BRIDGES = [
    "The simulation shows favorable odds. Proceeding with the attack.",
    "Based on the battle odds, here are my attacks.",
    "The analysis supports attacking vulnerable targets.",
    "Attacking where I have force advantage.",
]

ATTACK_EMPTY_BRIDGES = [
    "No favorable attack opportunities this turn.",
    "The threats are too strong to attack safely. Skipping attacks.",
    "Holding position — no good attack targets.",
]

MOVEMENT_BRIDGES = [
    "Based on the position evaluation, redistributing troops to strengthen borders.",
    "Moving troops to reinforce vulnerable positions.",
    "Repositioning based on the strategic assessment.",
    "Consolidating forces on border territories.",
]

MOVEMENT_NULL_BRIDGES = [
    "Current positions are adequate. No movement needed.",
    "Troops are well-positioned. Skipping movement.",
]


def get_bridge(phase, response):
    """Pick a short bridge sentence based on phase and content."""
    import random

    if phase == "reinforcements":
        return random.choice(REINFORCEMENT_BRIDGES)
    elif phase == "attacks":
        # Check if empty attacks
        match = re.search(r'```json\s*\n?(.*?)\n?\s*```', response, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(1).strip())
                if isinstance(data, dict) and data.get("attacks") == []:
                    return random.choice(ATTACK_EMPTY_BRIDGES)
            except (json.JSONDecodeError, KeyError):
                pass
        return random.choice(ATTACK_BRIDGES)
    elif phase == "movement":
        match = re.search(r'```json\s*\n?(.*?)\n?\s*```', response, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(1).strip())
                if isinstance(data, dict) and data.get("movement") is None:
                    return random.choice(MOVEMENT_NULL_BRIDGES)
            except (json.JSONDecodeError, KeyError):
                pass
        return random.choice(MOVEMENT_BRIDGES)
    return ""


def trim_response(response, phase):
    """Trim verbose reasoning between </tool_result> and ```json."""
    # Find the tool_result end and json start
    tr_end_tag = "</tool_result>"
    json_fence = "```json"

    tr_end_idx = response.find(tr_end_tag)
    json_idx = response.find(json_fence)

    # No tool result or no json — return as-is
    if tr_end_idx == -1 or json_idx == -1:
        # Even without tool calls, trim verbose text before ```json
        if json_idx > 200:
            bridge = get_bridge(phase, response)
            return f"{bridge}\n\n{response[json_idx:]}"
        return response

    # Already concise (< 200 chars between)
    middle_start = tr_end_idx + len(tr_end_tag)
    middle = response[middle_start:json_idx].strip()
    if len(middle) < 200:
        return response

    # Replace verbose middle with short bridge
    bridge = get_bridge(phase, response)
    prefix = response[:middle_start]
    suffix = response[json_idx:]

    return f"{prefix}\n\n{bridge}\n\n{suffix}"


def trim(input_path, output_path, seed=42):
    import random
    random.seed(seed)

    entries = []
    with open(input_path) as f:
        for line in f:
            entries.append(json.loads(line))

    total = len(entries)
    trimmed_count = 0

    for entry in entries:
        original_len = len(entry["response"])
        entry["response"] = trim_response(entry["response"], entry["phase"])
        if len(entry["response"]) < original_len:
            trimmed_count += 1

    # Write output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")

    # Stats
    lengths_before = []
    lengths_after = []
    with open(input_path) as f:
        for line in f:
            e = json.loads(line)
            lengths_before.append(len(e["response"]))
    with open(output_path) as f:
        for line in f:
            e = json.loads(line)
            lengths_after.append(len(e["response"]))

    print(f"=== SFT Data Trimming ===")
    print(f"Input:           {input_path} ({total} entries)")
    print(f"Trimmed:         {trimmed_count} responses")
    print(f"Avg response len: {sum(lengths_before)//len(lengths_before)} → {sum(lengths_after)//len(lengths_after)} chars")
    print(f"Output:          {output_path}")

    # Write summary
    summary_path = output_path.replace(".jsonl", "_summary.json")
    summary = {
        "input": input_path,
        "output": output_path,
        "total_entries": total,
        "trimmed_count": trimmed_count,
        "avg_len_before": sum(lengths_before) // len(lengths_before),
        "avg_len_after": sum(lengths_after) // len(lengths_after),
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary:         {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Trim augmented SFT data to concise responses"
    )
    parser.add_argument("--input", type=str, default=INPUT_PATH,
                        help=f"Input turns.jsonl (default: {INPUT_PATH})")
    parser.add_argument("--output", type=str, default=OUTPUT_PATH,
                        help=f"Output path (default: {OUTPUT_PATH})")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    args = parser.parse_args()

    trim(args.input, args.output, args.seed)
