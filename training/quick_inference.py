"""Quick inference check: run the trained model on a few sample prompts.

Usage:
    python training/quick_inference.py --model ./risk_sft_output
    python training/quick_inference.py --model ./risk_sft_output --n 10
"""

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

DATA = "data/hybrid_data/turns.jsonl"


def load_samples(path, n=5, seed=42):
    """Load n random prompts from training data."""
    random.seed(seed)
    entries = []
    with open(path) as f:
        for line in f:
            entries.append(json.loads(line))
    samples = random.sample(entries, min(n, len(entries)))
    return samples


def main():
    parser = argparse.ArgumentParser(description="Quick inference on sample prompts")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to PEFT adapter (e.g. ./risk_sft_output)")
    parser.add_argument("--data", type=str, default=DATA,
                        help=f"Path to turns.jsonl (default: {DATA})")
    parser.add_argument("--n", type=int, default=5,
                        help="Number of samples (default: 5)")
    parser.add_argument("--max-tokens", type=int, default=256,
                        help="Max tokens to generate (default: 256)")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature (default: 0.7)")
    args = parser.parse_args()

    os.environ["RISK_MODEL_PATH"] = args.model
    from llm_player.model import ModelBackend

    print(f"Loading model from {args.model}...")
    backend = ModelBackend()
    print(f"Backend: {backend.backend_type}\n")

    samples = load_samples(args.data, n=args.n)

    for i, sample in enumerate(samples):
        phase = sample["phase"]
        prompt = sample["prompt"]
        expected = sample["response"]

        print(f"{'='*60}")
        print(f"Sample {i+1}/{len(samples)} — {phase}")
        print(f"{'='*60}")

        # Show just the last part of the prompt (situation + decision)
        lines = prompt.split("\n")
        situation_start = None
        for j, line in enumerate(lines):
            if line.startswith("SITUATION:"):
                situation_start = j
                break
        if situation_start is not None:
            print("\n".join(lines[situation_start:]))
        else:
            print(prompt[-500:])

        print(f"\n--- Model output ---")
        output = backend.generate(prompt, max_tokens=args.max_tokens,
                                  temperature=args.temperature)
        print(output)

        print(f"\n--- Expected (heuristic) ---")
        print(expected)
        print()


if __name__ == "__main__":
    main()
