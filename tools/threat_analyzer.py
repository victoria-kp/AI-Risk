"""Analyze border vulnerabilities for a player.

Uses pyrisk's built-in adjacency graph (territory.connect) to compute
threat scores for each border territory. Returns territories sorted by
vulnerability so the LLM can reason about defensive priorities.
"""

from typing import Dict, List


def analyze_threats(game=None, player=None, **kwargs) -> List[Dict]:
    """Score each owned border territory by threat level.

    For each territory the player owns that borders an enemy:
    - Sums enemy troops on adjacent territories
    - Identifies the most dangerous neighbor (highest troops)
    - Computes threat_score = enemy_adjacent / your_troops

    Args:
        game: a pyrisk Game object (unused, kept for tool interface)
        player: a pyrisk Player object

    Returns:
        List of dicts sorted by threat_score descending:
        [{
            "territory": str,
            "your_troops": int,
            "enemy_troops_adjacent": int,
            "threat_score": float,
            "most_dangerous_neighbor": str,
            "most_dangerous_neighbor_troops": int,
            "most_dangerous_neighbor_owner": str,
        }, ...]
    """
    threats = []
    for t in player.territories:
        if not t.border:
            continue

        enemies = list(t.adjacent(friendly=False))
        if not enemies:
            continue

        enemy_troops = sum(adj.forces for adj in enemies)
        most_dangerous = max(enemies, key=lambda adj: adj.forces)
        threat_score = enemy_troops / t.forces if t.forces > 0 else float('inf')

        threats.append({
            "territory": t.name,
            "your_troops": t.forces,
            "enemy_troops_adjacent": enemy_troops,
            "threat_score": round(threat_score, 2),
            "most_dangerous_neighbor": most_dangerous.name,
            "most_dangerous_neighbor_troops": most_dangerous.forces,
            "most_dangerous_neighbor_owner": most_dangerous.owner.name,
        })

    threats.sort(key=lambda x: x["threat_score"], reverse=True)
    return threats
