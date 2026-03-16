"""Ask the RL model where to place reinforcements.

Formats board analysis and reinforcement count into a prompt, runs the
RL-trained model (or Gemini Flash fallback), and parses the JSON output
into a {territory_name: troop_count} dict.
"""
