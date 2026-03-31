"""Score turn logs with reward_hybrid and print average rewards.

Usage:
    python analysis/score_turns.py results/gemini_benchmark/turns.jsonl
    python analysis/score_turns.py data/hybrid_data/turns.jsonl --max 500
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'pyrisk_vendor'))

from training.reward_hybrid import compute_reward as compute_reward_hybrid


def main():
    parser = argparse.ArgumentParser(description="Score turns with reward_hybrid")
    parser.add_argument("path", type=str, help="Path to turns.jsonl")
    parser.add_argument("--max", type=int, default=None, help="Max entries to score")
    args = parser.parse_args()

    rewards = {"reinforcements": [], "attacks": [], "all": []}

    with open(args.path) as f:
        for i, line in enumerate(f):
            if args.max and i >= args.max:
                break

            entry = json.loads(line)
            phase = entry.get("phase", "")
            if phase not in ("reinforcements", "attacks"):
                continue

            response = entry.get("response", "")
            # Support both benchmark format (snapshot) and training format (board_snapshot)
            snapshot = entry.get("board_snapshot") or entry.get("snapshot", {})
            available = entry.get("available", 3)
            menu = entry.get("attack_menu", [])

            r = compute_reward_hybrid(
                response, phase, snapshot,
                available=available, attack_menu=menu,
            )
            rewards[phase].append(r)
            rewards["all"].append(r)

    print(f"File: {args.path}")
    print(f"Entries scored: {len(rewards['all'])}")
    for key in ["reinforcements", "attacks", "all"]:
        vals = rewards[key]
        if vals:
            avg = sum(vals) / len(vals)
            print(f"  {key:>16}: {avg:.4f} avg  (n={len(vals)}, "
                  f"min={min(vals):.3f}, max={max(vals):.3f})")


if __name__ == "__main__":
    main()
