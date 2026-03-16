"""LangGraph game-playing pipeline.

Orchestrates one turn of Risk: analyze the board with tools, decide where
to reinforce, evaluate and execute attacks, then move troops. The RL-trained
model (or Gemini Flash fallback) makes the strategic decisions; LangGraph
structures the multi-step reasoning each turn requires.
"""
