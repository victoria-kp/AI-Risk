"""GRPO training script for Qwen 2.5 3B.

Run on Google Colab with T4 GPU, or locally on CPU for debugging.

Training data: data/benchmark_results/turns.jsonl
Each line has {prompt, phase, board_snapshot, outcome}.

For each prompt, GRPO generates G completions (default 4).
Each completion is:
  1. Processed by run_tool_loop() for <tool_call> tags
  2. Scored by reward.compute_reward() using board_snapshot
No ground truth needed — GRPO uses relative advantages
between the G completions to update the policy.

The model learns both tool-calling (when to use
battle_sim, threat_analyzer, position_evaluator)
and decision-making (reinforcements, attacks, movement).

Note on tool execution during training:
  - battle_sim works fully (only needs attacking/defending counts)
  - threat_analyzer and position_evaluator return errors (need game/player
    objects not available during training). The model still gets credit
    for calling the right tool — it just won't see real results. During
    actual gameplay via LangGraph, all tools work with real game state.

Usage:
    # CPU debug (local, catch bugs before using GPU hours)
    # Uses Qwen 0.5B via plain transformers + PEFT — slow but finds bugs
    python training/train_grpo.py --cpu --max-steps 2 --num-generations 2 --max-examples 10

    # GPU training (Colab T4)
    # Uses Qwen 2.5 3B 4-bit via bitsandbytes + QLoRA
    python training/train_grpo.py --max-steps 200

    # Full training run
    python training/train_grpo.py --max-steps 200 --save-steps 50
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'pyrisk_vendor'))

from risk_env.tool_interface import run_tool_loop
from training.reward import compute_reward

# GPU-only imports (trl, datasets, bitsandbytes) are deferred to the
# functions that need them so that load_dataset() and reward_function()
# remain importable on CPU-only machines (used by analysis/evaluate.py).


# ── Defaults ──────────────────────────────────────────────────────────

MODEL_GPU = "Qwen/Qwen2.5-3B-Instruct"
MODEL_CPU = "Qwen/Qwen2.5-0.5B-Instruct"
DATA = "data/benchmark_results/turns.jsonl"
OUTPUT_DIR = "./risk_grpo_output"


# ── Dataset loading ───────────────────────────────────────────────────

def load_dataset(path, max_examples=None):
    """Load turns.jsonl into a HuggingFace Dataset.

    Filters out placement phase (no tools involved).
    Serializes board_snapshot as JSON string for HF Dataset compatibility.
    """
    from datasets import Dataset as HFDataset

    examples = []
    with open(path) as f:
        for line in f:
            entry = json.loads(line)
            if entry["phase"] == "placement":
                continue
            examples.append({
                "prompt": entry["prompt"],
                "phase": entry["phase"],
                "board_snapshot": json.dumps(entry["board_snapshot"]),
                "outcome": entry["outcome"],
            })

    if max_examples:
        examples = examples[:max_examples]

    return HFDataset.from_list(examples)


# ── Reward function ───────────────────────────────────────────────────

# Global log file handle, set by main() before training starts.
_completions_log_file = None
_completions_step = 0


def reward_function(completions, phase, board_snapshot, outcome, **kwargs):
    """GRPO reward: process tool calls, then score each completion.

    Called by GRPOTrainer for each batch of generated completions.
    Extra dataset columns (phase, board_snapshot, outcome) are passed
    automatically as keyword arguments.

    Args:
        completions: list of model-generated text strings.
        phase: list of phase strings per completion.
        board_snapshot: list of JSON-encoded board state strings.
        outcome: list of "win"/"loss" strings.

    Returns:
        List of float rewards in [0, 1].
    """
    global _completions_step
    _completions_step += 1

    rewards = []
    for completion, p, snap_json, out in zip(
        completions, phase, board_snapshot, outcome
    ):
        snap = json.loads(snap_json)
        # Process any <tool_call> tags in the completion
        processed, tool_log = run_tool_loop(
            completion, game=None, player=None
        )
        r = compute_reward(processed, p, snap, tool_log, outcome=out)
        rewards.append(r)

        # Log completion to file
        if _completions_log_file is not None:
            tools_called = [t["tool_name"] for t in tool_log]
            record = {
                "step": _completions_step,
                "phase": p,
                "outcome": out,
                "reward": round(r, 4),
                "tools_called": tools_called,
                "completion": completion[:2000],
            }
            _completions_log_file.write(json.dumps(record) + "\n")
            _completions_log_file.flush()

    return rewards


# ── Model loading ─────────────────────────────────────────────────────

def load_model(model_name=MODEL_GPU, max_seq_length=4096, cpu=False):
    """Load model for training.

    GPU mode (default): bitsandbytes 4-bit quantization + QLoRA.
    CPU mode (--cpu):   Plain transformers + PEFT LoRA with a smaller
                        model for debugging the full pipeline locally.
    """
    if cpu:
        return _load_model_cpu(model_name, max_seq_length)
    return _load_model_gpu(model_name, max_seq_length)


def _load_model_gpu(model_name, max_seq_length):
    """Load with bitsandbytes 4-bit quantization + PEFT LoRA (requires GPU)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb_config, trust_remote_code=True
    )
    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    return model, tokenizer


def _load_model_cpu(model_name, max_seq_length):
    """Load with plain transformers + PEFT for CPU debugging.

    Uses a smaller model (Qwen 0.5B by default) in float32.
    Slow but catches: tokenization errors, tool call parsing bugs,
    reward function crashes, shape mismatches, NaN losses.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    return model, tokenizer


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="GRPO training for Risk strategy with tool-use"
    )
    parser.add_argument("--model", type=str, default=None,
                        help="Model name or path (auto-selected if omitted)")
    parser.add_argument("--data", type=str, default=DATA,
                        help=f"Path to turns.jsonl (default: {DATA})")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR,
                        help=f"Output directory (default: {OUTPUT_DIR})")
    parser.add_argument("--max-steps", type=int, default=200,
                        help="Training steps (default: 200)")
    parser.add_argument("--num-generations", type=int, default=4,
                        help="Completions per prompt for GRPO (default: 4)")
    parser.add_argument("--max-examples", type=int, default=None,
                        help="Limit dataset size (for CPU debugging)")
    parser.add_argument("--learning-rate", type=float, default=5e-6,
                        help="Learning rate (default: 5e-6)")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Per-device batch size (default: 1)")
    parser.add_argument("--grad-accum", type=int, default=4,
                        help="Gradient accumulation steps (default: 4)")
    parser.add_argument("--max-completion-length", type=int, default=1024,
                        help="Max tokens per completion (default: 1024)")
    parser.add_argument("--logging-steps", type=int, default=5,
                        help="Log every N steps (default: 5)")
    parser.add_argument("--save-steps", type=int, default=50,
                        help="Save checkpoint every N steps (default: 50)")
    parser.add_argument("--cpu", action="store_true",
                        help="CPU debug mode: uses smaller model via "
                             "plain transformers + PEFT (no Unsloth)")
    args = parser.parse_args()

    # Pick default model based on mode
    if args.model is None:
        args.model = MODEL_CPU if args.cpu else MODEL_GPU

    mode = "CPU (debug)" if args.cpu else "GPU"
    print(f"=== GRPO Training for Risk [{mode}] ===")
    print(f"Model:        {args.model}")
    print(f"Data:         {args.data}")
    print(f"Output:       {args.output_dir}")
    print(f"Max steps:    {args.max_steps}")
    print(f"Generations:  {args.num_generations}")
    if args.max_examples:
        print(f"Max examples: {args.max_examples}")
    print()

    # Load model
    print("Loading model...")
    model, tokenizer = load_model(args.model, cpu=args.cpu)

    # Ensure pad token is set (required by GRPOTrainer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset(args.data, max_examples=args.max_examples)
    print(f"Dataset: {len(dataset)} examples "
          f"(phases: {set(dataset['phase'])})")
    print()

    # Configure trainer (imports deferred — only needed on GPU)
    from trl import GRPOConfig, GRPOTrainer

    config_kwargs = dict(
        output_dir=args.output_dir,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
    )
    if args.cpu:
        config_kwargs.update(bf16=False, fp16=False, no_cuda=True)

    config = GRPOConfig(**config_kwargs)

    trainer = GRPOTrainer(
        model=model,
        args=config,
        reward_funcs=reward_function,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    # Open completions log
    global _completions_log_file
    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, "completions.jsonl")
    _completions_log_file = open(log_path, "a")
    print(f"Logging completions to {log_path}")

    # Train
    print("Starting training...")
    trainer.train()

    # Close completions log
    _completions_log_file.close()
    _completions_log_file = None

    # Save final model
    print(f"\nSaving model to {args.output_dir}...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Done!")


if __name__ == "__main__":
    main()
