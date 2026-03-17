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
