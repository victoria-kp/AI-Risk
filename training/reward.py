"""Reward function for GRPO training.

Scores model completions on three weighted components: answer quality vs ground truth
(judged by Gemini, weight 0.6), tool use appropriateness—did it call the right tool
for the question type (weight 0.25), and efficiency—penalizing excessive or zero tool
calls (weight 0.15).
"""
