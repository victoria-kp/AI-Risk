"""Parse LLM tool calls, dispatch to tools, and inject results back into context.

Uses regex to extract <tool_call>tool_name(args)</tool_call> patterns from model
output, routes them to the correct tool function, and formats the result as a
<tool_result> block appended to the LLM's context.
"""
