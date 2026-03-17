"""Node 3: Decide which attacks to execute.

Receives the board summary from Node 1. Sends a prompt to the Qwen
model, which may call tools (0-3) via <tool_call> tags — e.g.,
battle_sim to check odds before committing to an attack.

After tool results are injected, the model outputs a JSON decision:
    {"attacks": [{"src": "Territory", "target": "Territory"}, ...]}
    or {"attacks": []} to skip attacking.

Validates: src owned with >1 troops, target is adjacent enemy.
Falls back to empty attack list (no attacks) if parsing fails.
"""
