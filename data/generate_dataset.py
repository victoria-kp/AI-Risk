"""Generate training dataset from pyrisk self-play.

Runs 200 games with StupidAI/BetterAI mixtures, snapshots board states
at each turn, and creates four types of training prompts per snapshot:

1. Reinforcement prompt — "You have N troops to place. Call tools, output JSON."
2. Attack prompt — "Decide your attacks. Call tools, output JSON."
3. Movement prompt — "Decide troop movements. Call tools, output JSON."
4. Strategy questions — from question_generator.py, free-text reasoning.

Each example includes: prompt, prompt_type, game_phase, board_state,
game_snapshot (serializable territory/player data), and metadata.

Outputs: data/train.jsonl and data/test.jsonl
"""
