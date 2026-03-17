"""Node 2: Decide where to place reinforcement troops.

Receives the board summary from Node 1 and the available troop count.
Sends a prompt to the Qwen model, which may call tools (0-3) via
<tool_call> tags processed by tool_interface.run_tool_loop().
After tool results are injected, the model outputs a JSON decision:
    {"reinforcements": {"TerritoryName": count, ...}}

Validates: sum equals available, all territories owned by player.
Falls back to spreading troops across border territories if parsing fails.
"""
