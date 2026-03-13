"""GRPO training script for Qwen 2.5 3B. Run on Google Colab with T4 GPU.

Loads the 4-bit quantized model via Unsloth, applies QLoRA, and trains with TRL's
GRPOTrainer using the custom reward function. Generates 4 completions per prompt
and optimizes via group-relative advantages.
"""
