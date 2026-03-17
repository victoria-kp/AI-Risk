"""GRPO training script for Qwen 2.5 3B. Run on Google Colab with T4 GPU.

Loads the 4-bit quantized model via Unsloth, applies QLoRA, and trains
with TRL's GRPOTrainer. For each prompt, generates 4 completions.
Each completion is processed by tool_interface.run_tool_loop() to handle
any <tool_call> tags, then scored by reward.compute_reward().

Training data includes both decision prompts (reinforcement, attack,
movement) and strategy questions — the model learns tool-calling and
strategic reasoning from the same reward signal.

The reward drives GRPO's group-relative advantages: completions that
produce valid, strategically sound decisions with appropriate tool use
get higher rewards than those that don't.
"""
