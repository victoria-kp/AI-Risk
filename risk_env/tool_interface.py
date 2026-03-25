"""Parse LLM tool calls, dispatch to tools, and inject results back into context.

Used during RL training so the Qwen model can practice calling tools.
The LangGraph player pipeline calls tools directly and does not need this.

Flow:
1. Model generates text containing <tool_call>tool_name(args)</tool_call>
2. parse_tool_call() extracts tool name and kwargs
3. dispatch_tool() calls the function and returns formatted <tool_result>
4. run_tool_loop() handles multi-turn: generate → parse → dispatch → inject → generate
"""

import re
import json
from typing import Optional, Tuple, Dict, Any, List

from tools.battle_sim import simulate_battle
from tools.threat_analyzer import analyze_threats, analyze_threats_from_snapshot
from tools.position_evaluator import evaluate_position, evaluate_position_from_snapshot

TOOL_CALL_PATTERN = re.compile(
    r'<tool_call>\s*(\w+)\s*\((.*?)\)\s*</tool_call>', re.DOTALL
)

TOOL_REGISTRY = {
    "battle_sim": simulate_battle,
    "threat_analyzer": analyze_threats,
    "position_evaluator": evaluate_position,
}


def parse_tool_call(text: str) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Extract the first tool call from model output.

    Parses patterns like:
        <tool_call>battle_sim(attacking=8, defending=3)</tool_call>
        <tool_call>threat_analyzer()</tool_call>

    Returns:
        (tool_name, kwargs_dict) or None if no tool call found.
    """
    match = TOOL_CALL_PATTERN.search(text)
    if not match:
        return None
    tool_name = match.group(1).strip()
    args_str = match.group(2).strip()
    kwargs = _parse_kwargs(args_str)
    return tool_name, kwargs


def _parse_kwargs(args_str: str) -> Dict[str, Any]:
    """Parse 'key=value, key=value' into a dict.

    Handles int, float, and string values.
    Empty string returns empty dict.
    """
    if not args_str:
        return {}

    kwargs = {}
    # Split on commas that are not inside quotes
    for part in re.split(r',\s*', args_str):
        part = part.strip()
        if not part or '=' not in part:
            continue
        key, value = part.split('=', 1)
        key = key.strip()
        value = value.strip()
        kwargs[key] = _parse_value(value)
    return kwargs


def _parse_value(value: str) -> Any:
    """Convert a string value to int, float, or str."""
    # Remove surrounding quotes
    if (value.startswith('"') and value.endswith('"')) or \
       (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    # Try int
    try:
        return int(value)
    except ValueError:
        pass
    # Try float
    try:
        return float(value)
    except ValueError:
        pass
    # Return as string
    return value


def dispatch_tool(tool_name: str, kwargs: Dict[str, Any],
                  game=None, player=None,
                  board_snapshot=None) -> str:
    """Call a tool and return a formatted <tool_result> string.

    Args:
        tool_name: name of the tool to call
        kwargs: parsed arguments from the model
        game: pyrisk Game object (injected, not from model)
        player: pyrisk Player object (injected, not from model)
        board_snapshot: dict from turns.jsonl (used during training
                        when game/player are unavailable)

    Returns:
        A <tool_result>...</tool_result> string to inject into context.
    """
    if tool_name not in TOOL_REGISTRY:
        return (f"<tool_result>Error: Unknown tool '{tool_name}'. "
                f"Available tools: {', '.join(TOOL_REGISTRY.keys())}"
                f"</tool_result>")

    try:
        # During training: use snapshot-based versions for tools that
        # need game/player objects. battle_sim works either way.
        if game is None and board_snapshot is not None:
            if tool_name == "threat_analyzer":
                result = analyze_threats_from_snapshot(board_snapshot)
            elif tool_name == "position_evaluator":
                result = evaluate_position_from_snapshot(board_snapshot)
            else:
                fn = TOOL_REGISTRY[tool_name]
                result = fn(**kwargs, game=game, player=player)
        else:
            fn = TOOL_REGISTRY[tool_name]
            result = fn(**kwargs, game=game, player=player)
        formatted = _format_result(result)
        return f"<tool_result>\n{formatted}\n</tool_result>"
    except Exception as e:
        return f"<tool_result>Error calling {tool_name}: {str(e)}</tool_result>"


def _format_result(result) -> str:
    """Convert tool output to readable text for the model.

    Handles dicts, lists of dicts, and other types.
    """
    if isinstance(result, dict):
        return _format_dict(result)
    if isinstance(result, list):
        return _format_list(result)
    return str(result)


def _format_dict(d: Dict, indent: int = 0) -> str:
    """Format a dict as key: value lines."""
    lines = []
    prefix = "  " * indent
    for key, value in d.items():
        if isinstance(value, list):
            lines.append(f"{prefix}{key}:")
            for item in value:
                if isinstance(item, dict):
                    lines.append(f"{prefix}  -")
                    lines.append(_format_dict(item, indent + 2))
                else:
                    lines.append(f"{prefix}  - {item}")
        elif isinstance(value, dict):
            lines.append(f"{prefix}{key}:")
            lines.append(_format_dict(value, indent + 1))
        else:
            lines.append(f"{prefix}{key}: {value}")
    return "\n".join(lines)


def _format_list(items: List) -> str:
    """Format a list of dicts as numbered entries."""
    lines = []
    for i, item in enumerate(items):
        if isinstance(item, dict):
            lines.append(f"[{i + 1}]")
            lines.append(_format_dict(item, indent=1))
        else:
            lines.append(f"[{i + 1}] {item}")
    return "\n".join(lines)


def run_tool_loop(text: str, game=None, player=None,
                  max_calls: int = 3,
                  board_snapshot=None) -> Tuple[str, List[Dict]]:
    """Parse and execute all tool calls in the model's output.

    Processes tool calls sequentially (up to max_calls). For each
    tool call found, dispatches it and replaces the <tool_call> block
    with the <tool_result> block in the text.

    Args:
        text: the model's generated text
        game: pyrisk Game object
        player: pyrisk Player object
        max_calls: maximum number of tool calls to process
        board_snapshot: dict from turns.jsonl (used during training
                        when game/player are unavailable)

    Returns:
        (updated_text, tool_log) where:
        - updated_text has <tool_call> blocks replaced with <tool_result>
        - tool_log is a list of {tool_name, kwargs, result} dicts
    """
    tool_log = []

    if text is None:
        return "", tool_log

    for _ in range(max_calls):
        parsed = parse_tool_call(text)
        if parsed is None:
            break

        tool_name, kwargs = parsed
        result_str = dispatch_tool(tool_name, kwargs,
                                   game=game, player=player,
                                   board_snapshot=board_snapshot)
        tool_log.append({
            "tool_name": tool_name,
            "kwargs": kwargs,
            "result": result_str,
        })

        # Replace the tool_call with the tool_result
        match = TOOL_CALL_PATTERN.search(text)
        text = text[:match.start()] + result_str + text[match.end():]

    return text, tool_log
