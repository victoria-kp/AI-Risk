"""GRPO training script for Qwen on Risk game data.

Scores reinforce + attack completions using reward_hybrid.
Shorter completions (256 tokens). No tool calling.

Usage:
    python training/train_grpo.py --resume-from ./risk_sft_output --max-steps 30

    # CPU debug
    python training/train_grpo.py --cpu --max-steps 2 --num-generations 2 --max-examples 10
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'pyrisk_vendor'))

from training.reward_hybrid import compute_reward as compute_reward_hybrid

# GPU-only imports (trl, datasets, bitsandbytes) are deferred to the
# functions that need them so that load_model() remains importable
# on CPU-only machines (used by train_sft.py).


# ── Defaults ──────────────────────────────────────────────────────────

MODEL_GPU = "Qwen/Qwen2.5-7B-Instruct"
MODEL_CPU = "Qwen/Qwen2.5-0.5B-Instruct"
DATA = "data/hybrid_data/turns.jsonl"
OUTPUT_DIR = "./risk_grpo_output"


# ── Dataset loading ───────────────────────────────────────────────────

def load_dataset(paths, max_examples=None):
    """Load hybrid turns.jsonl (reinforce + attack only).

    Includes attack_menu as JSON string for attack reward computation.
    """
    from datasets import Dataset as HFDataset

    if isinstance(paths, str):
        paths = [paths]

    examples = []
    for path in paths:
        with open(path) as f:
            for line in f:
                entry = json.loads(line)
                phase = entry["phase"]
                if phase not in ("reinforcements", "attacks"):
                    continue

                ex = {
                    "prompt": [{"role": "user", "content": entry["prompt"]}],
                    "phase": phase,
                    "board_snapshot": json.dumps(entry["board_snapshot"]),
                }
                if phase == "reinforcements":
                    ex["available"] = entry.get("available", 3)
                    ex["attack_menu"] = "[]"
                elif phase == "attacks":
                    ex["available"] = 0
                    ex["attack_menu"] = json.dumps(
                        entry.get("attack_menu", []))

                examples.append(ex)

    if max_examples:
        examples = examples[:max_examples]

    return HFDataset.from_list(examples)


# ── Reward function ───────────────────────────────────────────────────

_completions_log_file = None
_completions_step = 0


def reward_function(completions, phase, board_snapshot,
                    available, attack_menu, **kwargs):
    """GRPO reward: score JSON output for reinforce/attack completions.

    Args:
        completions: list of generated texts.
        phase: list of phase strings.
        board_snapshot: list of JSON board state strings.
        available: list of int (troops to place, 0 for attacks).
        attack_menu: list of JSON attack menu strings.
    """
    global _completions_step
    _completions_step += 1

    rewards = []
    for completion, p, snap_json, avail, menu_json in zip(
        completions, phase, board_snapshot, available, attack_menu
    ):
        if isinstance(completion, list):
            completion = completion[-1]["content"] if completion else ""
        elif isinstance(completion, dict):
            completion = completion.get("content", "")

        snap = json.loads(snap_json)
        menu = json.loads(menu_json)

        r = compute_reward_hybrid(
            completion, p, snap,
            available=avail, attack_menu=menu,
        )
        rewards.append(r)

        if _completions_log_file is not None:
            record = {
                "step": _completions_step,
                "phase": p,
                "reward": round(r, 4),
                "completion": completion[:2000],
            }
            _completions_log_file.write(json.dumps(record) + "\n")
            _completions_log_file.flush()

    return rewards


# ── Model loading ─────────────────────────────────────────────────────

def load_model(model_name=MODEL_GPU, max_seq_length=4096, cpu=False,
               resume_from=None):
    """Load model for training.

    GPU mode (default): bitsandbytes 4-bit quantization + QLoRA.
    CPU mode (--cpu):   Plain transformers + PEFT LoRA with a smaller
                        model for debugging the full pipeline locally.

    Args:
        resume_from: path to a PEFT adapter directory from a previous
            training run. Loads the base model + existing LoRA weights
            instead of creating a fresh LoRA. The model_name arg is
            ignored when resume_from is set (base model is read from
            the adapter config).
    """
    if cpu:
        return _load_model_cpu(model_name, max_seq_length, resume_from)
    return _load_model_gpu(model_name, max_seq_length, resume_from)


def _load_model_gpu(model_name, max_seq_length, resume_from=None):
    """Load with bitsandbytes 4-bit quantization + PEFT LoRA (requires GPU)."""
    import json as _json
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, PeftModel

    if resume_from:
        config_path = os.path.join(resume_from, "adapter_config.json")
        with open(config_path) as f:
            adapter_cfg = _json.load(f)
        model_name = adapter_cfg["base_model_name_or_path"]
        print(f"Resuming from {resume_from} (base: {model_name})")

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

    if resume_from:
        model = PeftModel.from_pretrained(model, resume_from, is_trainable=True)
    else:
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

    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    # Gradient checkpointing corrupts autoregressive generation in train
    # mode, but we need it for the training forward pass (OOM without it).
    # Wrap generate() to temporarily switch to eval mode.
    _orig_generate = model.generate

    def _safe_generate(*args, **kwargs):
        model.eval()
        try:
            return _orig_generate(*args, **kwargs)
        finally:
            model.train()

    model.generate = _safe_generate

    return model, tokenizer


def _load_model_cpu(model_name, max_seq_length, resume_from=None):
    """Load with plain transformers + PEFT for CPU debugging.

    Uses a smaller model (Qwen 0.5B by default) in float32.
    Slow but catches: tokenization errors, reward function crashes,
    shape mismatches, NaN losses.
    """
    import json as _json
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, PeftModel

    if resume_from:
        config_path = os.path.join(resume_from, "adapter_config.json")
        with open(config_path) as f:
            adapter_cfg = _json.load(f)
        model_name = adapter_cfg["base_model_name_or_path"]
        print(f"Resuming from {resume_from} (base: {model_name})")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True
    )

    if resume_from:
        model = PeftModel.from_pretrained(model, resume_from, is_trainable=True)
    else:
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
        description="GRPO training for Risk strategy"
    )
    parser.add_argument("--model", type=str, default=None,
                        help="Model name or path (auto-selected if omitted)")
    parser.add_argument("--data", type=str, nargs="+", default=[DATA],
                        help=f"Path(s) to turns.jsonl (default: {DATA})")
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
    parser.add_argument("--max-completion-length", type=int, default=256,
                        help="Max tokens per completion (default: 256)")
    parser.add_argument("--max-prompt-length", type=int, default=3072,
                        help="Max tokens per prompt (default: 3072)")
    parser.add_argument("--logging-steps", type=int, default=1,
                        help="Log every N steps (default: 1)")
    parser.add_argument("--save-steps", type=int, default=50,
                        help="Save checkpoint every N steps (default: 50)")
    parser.add_argument("--temperature", type=float, default=0.9,
                        help="Sampling temperature for generation (default: 0.9)")
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Path to PEFT adapter from previous round.")
    parser.add_argument("--cpu", action="store_true",
                        help="CPU debug mode: smaller model, no quantization")
    args = parser.parse_args()

    if args.model is None:
        args.model = MODEL_CPU if args.cpu else MODEL_GPU

    hw_str = "CPU (debug)" if args.cpu else "GPU"
    print(f"=== GRPO Training for Risk [{hw_str}] ===")
    print(f"Model:        {args.model}")
    if args.resume_from:
        print(f"Resume from:  {args.resume_from}")
    print(f"Data:         {', '.join(args.data)}")
    print(f"Output:       {args.output_dir}")
    print(f"Max steps:    {args.max_steps}")
    print(f"Generations:  {args.num_generations}")
    print(f"LR:           {args.learning_rate}")
    if args.max_examples:
        print(f"Max examples: {args.max_examples}")
    print()

    # Load model
    print("Loading model...")
    model, tokenizer = load_model(args.model, cpu=args.cpu,
                                  resume_from=args.resume_from)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset(args.data, max_examples=args.max_examples)
    print(f"Dataset: {len(dataset)} examples "
          f"(phases: {set(dataset['phase'])})")
    print()

    # Configure trainer
    from trl import GRPOConfig, GRPOTrainer

    config_kwargs = dict(
        output_dir=args.output_dir,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        temperature=args.temperature,
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
