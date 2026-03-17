"""Node 4: Decide post-combat troop movement (freemove).

Receives the board summary from Node 1. Sends a prompt to the Qwen
model, which may call tools (0-3) via <tool_call> tags for analysis.

After tool results are injected, the model outputs a JSON decision:
    {"movement": {"src": "Territory", "target": "Territory", "count": N}}
    or {"movement": null} to skip.

Validates: both territories owned, 0 < count < src.forces.
Falls back to moving strongest inland troop to weakest connected border.
"""
