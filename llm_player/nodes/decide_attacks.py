"""Ask the RL model which attacks to execute.

For promising attack opportunities, calls battle_sim to get odds. Sends
combined analysis to the RL model. Parses output into a list of
{src, target, troops, move} action dicts.
"""
