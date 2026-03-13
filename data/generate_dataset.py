"""Generate training dataset from pyrisk self-play.

Runs hundreds of pyrisk games with random AIs, snapshots game states at each turn,
samples diverse states across early/mid/late game, pairs them with generated questions,
and computes ground-truth quality scores via simulation rollouts. Outputs train.jsonl.
"""
