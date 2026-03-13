"""LangGraph game coaching pipeline.

Defines a 5-node StateGraph: parse game log, identify critical moments, run the
RL-trained advisor on each moment, find recurring patterns across analyses, and
generate a final markdown coaching report.
"""
