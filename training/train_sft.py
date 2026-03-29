"""SFT (Supervised Fine-Tuning) script for Qwen on Risk game data.

Teaches the model the correct output format (valid JSON, tool call syntax,
territory names) by training on Gemini game data. This is the first stage
of a two-stage pipeline:

  1. SFT  — learn the format (this script)
  2. GRPO — improve strategy (train_grpo.py --resume-from <sft_output>)

Training data: data/benchmark_results/turns.jsonl
Each line has {prompt, response, phase, outcome, fallback}.
The model learns to produce the response given the prompt.

Usage:
    # GPU training (Colab)
    python training/train_sft.py --max-steps 100

    # CPU debug (local)
    python training/train_sft.py --cpu --max-steps 2 --max-examples 10

    # Then run GRPO on top
    python training/train_grpo.py --resume-from ./risk_sft_output --max-steps 200
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'pyrisk_vendor'))

from training.train_grpo import load_model, MODEL_GPU, MODEL_CPU


# ── Defaults ──────────────────────────────────────────────────────────

DATA = "data/benchmark_results/turns.jsonl"
OUTPUT_DIR = "./risk_sft_output"


# ── Dataset loading ───────────────────────────────────────────────────

def load_dataset_sft(paths, max_examples=None, wins_only=False):
    """Load turns.jsonl files for SFT training.

    Formats each example as a chat conversation (user prompt + assistant
    response) so SFTTrainer applies the model's chat template.

    Filters:
      - Skips placement phase (no tools involved)
      - Skips fallback entries (bad examples where parsing failed)
      - Optional: only winning game decisions (--wins-only)

    Args:
        paths: single path or list of paths to turns.jsonl files.
        max_examples: optional cap on total examples.
        wins_only: if True, only include decisions from winning games.
    """
    from datasets import Dataset as HFDataset

    if isinstance(paths, str):
        paths = [paths]

    examples = []
    for path in paths:
        with open(path) as f:
            for line in f:
                entry = json.loads(line)
                if entry["phase"] == "placement":
                    continue
                if entry.get("fallback", False):
                    continue
                if wins_only and entry.get("outcome") != "win":
                    continue
                examples.append({
                    "messages": [
                        {"role": "user", "content": entry["prompt"]},
                        {"role": "assistant", "content": entry["response"]},
                    ],
                })

    if max_examples:
        examples = examples[:max_examples]

    return HFDataset.from_list(examples)


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="SFT training for Risk — teaches output format"
    )
    parser.add_argument("--model", type=str, default=None,
                        help="Model name or path (auto-selected if omitted)")
    parser.add_argument("--data", type=str, nargs="+", default=[DATA],
                        help=f"Path(s) to turns.jsonl (default: {DATA})")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR,
                        help=f"Output directory (default: {OUTPUT_DIR})")
    parser.add_argument("--max-steps", type=int, default=100,
                        help="Training steps (default: 100)")
    parser.add_argument("--max-examples", type=int, default=None,
                        help="Limit dataset size (for debugging)")
    parser.add_argument("--learning-rate", type=float, default=2e-5,
                        help="Learning rate (default: 2e-5)")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Per-device batch size (default: 1)")
    parser.add_argument("--grad-accum", type=int, default=4,
                        help="Gradient accumulation steps (default: 4)")
    parser.add_argument("--max-seq-length", type=int, default=3584,
                        help="Max sequence length (default: 3584)")
    parser.add_argument("--logging-steps", type=int, default=1,
                        help="Log every N steps (default: 1)")
    parser.add_argument("--save-steps", type=int, default=50,
                        help="Save checkpoint every N steps (default: 50)")
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Path to PEFT adapter to continue training")
    parser.add_argument("--wins-only", action="store_true",
                        help="Only train on decisions from winning games")
    parser.add_argument("--cpu", action="store_true",
                        help="CPU debug mode (smaller model, no quantization)")
    args = parser.parse_args()

    # Pick default model based on mode
    if args.model is None:
        args.model = MODEL_CPU if args.cpu else MODEL_GPU

    mode = "CPU (debug)" if args.cpu else "GPU"
    print(f"=== SFT Training for Risk [{mode}] ===")
    print(f"Model:        {args.model}")
    if args.resume_from:
        print(f"Resume from:  {args.resume_from}")
    print(f"Data:         {', '.join(args.data)}")
    print(f"Output:       {args.output_dir}")
    print(f"Max steps:    {args.max_steps}")
    print(f"LR:           {args.learning_rate}")
    if args.wins_only:
        print(f"Filter:       wins only")
    if args.max_examples:
        print(f"Max examples: {args.max_examples}")
    print()

    # Load model (reuses GRPO's load_model with same LoRA + quantization)
    print("Loading model...")
    model, tokenizer = load_model(args.model, cpu=args.cpu,
                                  resume_from=args.resume_from)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset_sft(args.data, max_examples=args.max_examples,
                               wins_only=args.wins_only)
    print(f"Dataset: {len(dataset)} examples")
    print()

    # Configure SFT trainer
    from trl import SFTConfig, SFTTrainer

    config_kwargs = dict(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        max_seq_length=args.max_seq_length,
    )
    if args.cpu:
        config_kwargs.update(bf16=False, fp16=False, no_cuda=True)

    config = SFTConfig(**config_kwargs)

    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    # Train
    print("Starting SFT training...")
    trainer.train()

    # Save
    print(f"\nSaving model to {args.output_dir}...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Done!")
    print(f"\nNext step: GRPO training on top of SFT")
    print(f"  python training/train_grpo.py --resume-from {args.output_dir} --max-steps 200")


if __name__ == "__main__":
    main()
