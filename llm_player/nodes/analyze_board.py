"""Node 1: Serialize the board state for downstream decision nodes.

Pure computation — no model, no tools. Calls only
state_serializer.serialize_game_state() to produce a text summary
of the board from the current player's perspective.

The decision nodes (2, 3, 4) receive this summary and decide on
their own whether to call tools (threat_analyzer, position_evaluator,
battle_sim) for additional analysis.
"""

from risk_env.state_serializer import serialize_game_state


def analyze_board(state: dict) -> dict:
    """Serialize the board and store the summary in state.

    Args:
        state: RiskTurnState dict with 'game' and 'player' keys.

    Returns:
        Dict with 'board_summary' key containing the serialized text.
    """
    board_summary = serialize_game_state(state["game"], state["player"])
    return {"board_summary": board_summary}
