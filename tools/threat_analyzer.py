"""Analyze border vulnerabilities for a player.

For each territory the player owns, sums enemy troops on adjacent territories and
computes a threat score (enemy_adjacent / your_troops). Returns territories sorted
by vulnerability, identifying the most dangerous neighbor for each.
"""
