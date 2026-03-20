"""LangGraph game-playing pipeline.

Orchestrates one turn of Risk with a linear 4-node graph:
    analyze_board -> decide_reinforcements -> decide_attacks -> decide_movement -> END

State (RiskTurnState) flows through all nodes carrying:
- game/player objects (injected, not modified by nodes)
- model backend reference (shared across decision nodes)
- board_summary (set by Node 1, read by Nodes 2-4)
- decision outputs and raw model text (set by each decision node)

Node 1 is pure computation (state_serializer only).
Nodes 2-4 invoke the Qwen model, which decides which tools to call.
"""

from typing import Any, Dict, List, Optional

from typing_extensions import TypedDict

from langgraph.graph import StateGraph, END

from llm_player.nodes.analyze_board import analyze_board
from llm_player.nodes.decide_reinforcements import decide_reinforcements
from llm_player.nodes.decide_attacks import decide_attacks
from llm_player.nodes.decide_movement import decide_movement


class RiskTurnState(TypedDict, total=False):
    """State flowing through the LangGraph pipeline for one Risk turn."""

    # Injected before pipeline runs
    game: Any
    player: Any
    model: Any
    reinforcements_available: int

    # Set by analyze_board (Node 1)
    board_summary: str

    # Set by decide_reinforcements (Node 2)
    reinforcement_decision: Dict[str, int]
    reinforcement_raw: str

    # Set by decide_attacks (Node 3)
    attack_decisions: List[Dict[str, str]]
    attack_raw: str

    # Set by decide_movement (Node 4)
    movement_decision: Optional[Dict]
    movement_raw: str


def build_turn_graph():
    """Build and compile the 4-node turn pipeline.

    Returns a compiled LangGraph that can be invoked with:
        result = graph.invoke(initial_state)
    """
    graph = StateGraph(RiskTurnState)

    graph.add_node("analyze_board", analyze_board)
    graph.add_node("decide_reinforcements", decide_reinforcements)
    graph.add_node("decide_attacks", decide_attacks)
    graph.add_node("decide_movement", decide_movement)

    graph.set_entry_point("analyze_board")
    graph.add_edge("analyze_board", "decide_reinforcements")
    graph.add_edge("decide_reinforcements", "decide_attacks")
    graph.add_edge("decide_attacks", "decide_movement")
    graph.add_edge("decide_movement", END)

    return graph.compile()


def run_turn(game, player, model, reinforcements_available: int) -> dict:
    """Run one full turn through the pipeline.

    Args:
        game: pyrisk Game object.
        player: pyrisk Player object.
        model: ModelBackend instance.
        reinforcements_available: Number of troops to place.

    Returns:
        Final RiskTurnState dict with all decisions.
    """
    graph = build_turn_graph()
    initial_state = {
        "game": game,
        "player": player,
        "model": model,
        "reinforcements_available": reinforcements_available,
    }
    return graph.invoke(initial_state)
