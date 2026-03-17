"""Node 1: Serialize the board state for downstream decision nodes.

Pure computation — no model, no tools. Calls only
state_serializer.serialize_game_state() to produce a text summary
of the board from the current player's perspective.

The decision nodes (2, 3, 4) receive this summary and decide on
their own whether to call tools (threat_analyzer, position_evaluator,
battle_sim) for additional analysis.
"""
