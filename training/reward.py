"""Reward function for GRPO training.

Scores model completions on three weighted components:

1. Format + Decision Quality (weight 0.60):
   - Valid JSON output, correct territory names, sum constraints
   - Strategic quality heuristics (reinforcing borders, favorable attack odds)

2. Tool Use Appropriateness (weight 0.25):
   - Did it call relevant tools for the task type?
   - threat_analyzer for reinforcement, battle_sim for attacks, etc.

3. Efficiency (weight 0.15):
   - 1-2 tool calls ideal for decisions
   - Penalize 0 calls (missed info) or 3+ (wasteful)

Handles all four prompt types: reinforcement, attack, movement, strategy_question.
Returns float reward in [0, 1].
"""
