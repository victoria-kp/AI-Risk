"""pyrisk AI wrapper that uses the LangGraph pipeline.

Subclass of pyrisk's AI class. The full LangGraph pipeline runs once
during reinforce() — computing reinforcement, attack, and movement
decisions from the current board state. attack() and freemove() read
cached results from that single pipeline invocation.

Methods:
- start(): initialize ModelBackend
- initial_placement(): simple heuristic (prioritize small continents)
- reinforce(available): run pipeline, return {territory: count} dict
- attack(): yield (src, target, None, None) tuples from cached decisions
- freemove(): return (src, target, count) or None from cached decision
"""
