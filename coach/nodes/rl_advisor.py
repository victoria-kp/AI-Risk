"""Call the RL-trained model on each critical moment.

Loads the trained LoRA adapter, serializes each critical moment's game state,
generates analysis with tool calls, and collects strategic recommendations.
This is where the RL layer integrates into the LangGraph pipeline.
"""
