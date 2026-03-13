"""Convert a pyrisk game state into a text prompt for the LLM.

Takes a pyrisk game object and a player, and produces a multi-line string describing
the board from that player's perspective: territory ownership, troop counts, adjacencies,
continent progress, and opponent summaries.
"""
