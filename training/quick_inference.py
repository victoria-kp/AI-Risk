"""Quick inference check on SFT/GRPO checkpoint.

Loads the PEFT adapter and generates completions on a sample of prompts
from the training data. Useful for spot-checking before running GRPO.

Usage:
    python training/quick_inference.py --model-path ./risk_sft_output_v2
    python training/quick_inference.py --model-path ./risk_sft_output_v2 --num-prompts 10
"""

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

DATA = "data/sft_augmented/turns.jsonl"


def load_prompts(data_path, num_prompts, seed=42):
    """Load a random sample of prompts, balanced across phases."""
    random.seed(seed)
    by_phase = {}
    with open(data_path) as f:
        for line in f:
            entry = json.loads(line)
            phase = entry["phase"]
            if phase not in by_phase:
                by_phase[phase] = []
            by_phase[phase].append(entry)

    # Sample evenly across phases
    per_phase = max(1, num_prompts // len(by_phase))
    sampled = []
    for phase, entries in by_phase.items():
        sampled.extend(random.sample(entries, min(per_phase, len(entries))))

    random.shuffle(sampled)
    return sampled[:num_prompts]


def main():
    parser = argparse.ArgumentParser(description="Quick inference check")
    parser.add_argument("--model-path", type=str, default="./risk_sft_output_v2",
                        help="Path to PEFT adapter")
    parser.add_argument("--data", type=str, default=DATA,
                        help="Prompts source file")
    parser.add_argument("--num-prompts", type=int, default=9,
                        help="Number of prompts to test (default: 9)")
    parser.add_argument("--max-tokens", type=int, default=512,
                        help="Max tokens per generation (default: 512)")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature (default: 0.7)")
    args = parser.parse_args()

    # Load model
    print(f"Loading model from {args.model_path}...")
    from llm_player.model import ModelBackend
    backend = ModelBackend(backend="peft", model_path=args.model_path)
    print(f"Backend: {backend.backend_type}\n")

    # Load prompts
    prompts = load_prompts(args.data, args.num_prompts)
    print(f"Testing {len(prompts)} prompts ({args.temperature} temp, {args.max_tokens} max tokens)\n")

    # Output directory
    output_dir = "data/inference_checks"
    os.makedirs(output_dir, exist_ok=True)

    # Track stats
    stats = {"tool_call": 0, "json_fence": 0, "total": 0}
    results = []

    for i, entry in enumerate(prompts):
        phase = entry["phase"]
        prompt_preview = entry["prompt"][:100].replace("\n", " ") + "..."

        print(f"{'='*60}")
        print(f"[{i+1}/{len(prompts)}] Phase: {phase}")
        print(f"Prompt: {prompt_preview}")
        print(f"-"*60)

        completion = backend.generate(
            entry["prompt"],
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            caller="inference_check",
        )

        print(completion)

        # Track stats
        stats["total"] += 1
        if "<tool_call>" in completion:
            stats["tool_call"] += 1
        if "```json" in completion:
            stats["json_fence"] += 1

        results.append({
            "phase": phase,
            "prompt": entry["prompt"],
            "completion": completion,
            "has_tool_call": "<tool_call>" in completion,
            "has_json_fence": "```json" in completion,
        })

        print()

    # Summary
    print(f"{'='*60}")
    print(f"SUMMARY ({stats['total']} completions)")
    print(f"  Tool calls:  {stats['tool_call']}/{stats['total']} ({stats['tool_call']/stats['total']*100:.0f}%)")
    print(f"  JSON fenced: {stats['json_fence']}/{stats['total']} ({stats['json_fence']/stats['total']*100:.0f}%)")

    # Save full results
    model_name = os.path.basename(args.model_path.rstrip("/"))
    output_path = os.path.join(output_dir, f"{model_name}_inference.jsonl")
    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\nFull results saved to: {output_path}")


if __name__ == "__main__":
    main()
