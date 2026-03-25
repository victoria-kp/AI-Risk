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


def analyze_threats_from_snapshot(snapshot: dict) -> List[Dict]:
    """Snapshot-based version for use during training (no live game needed).

    Uses board_snapshot dict with keys: player_name, owned_territories,
    border_territories, territory_map.

    Returns the same format as analyze_threats().
    """
    player_name = snapshot["player_name"]
    borders = set(snapshot["border_territories"])
    territory_map = snapshot["territory_map"]

    threats = []
    for t_name in snapshot["owned_territories"]:
        if t_name not in borders:
            continue

        t_info = territory_map.get(t_name, {})
        your_troops = t_info.get("forces", 0)

        # Find enemy neighbors
        enemies = []
        for adj_name in t_info.get("adjacent", []):
            adj_info = territory_map.get(adj_name, {})
            if adj_info.get("owner") != player_name:
                enemies.append((adj_name, adj_info))

        if not enemies:
            continue

        enemy_troops = sum(info.get("forces", 0) for _, info in enemies)
        most_dangerous_name, most_dangerous_info = max(
            enemies, key=lambda x: x[1].get("forces", 0)
        )
        threat_score = (enemy_troops / your_troops
                        if your_troops > 0 else float('inf'))

        threats.append({
            "territory": t_name,
            "your_troops": your_troops,
            "enemy_troops_adjacent": enemy_troops,
            "threat_score": round(threat_score, 2),
            "most_dangerous_neighbor": most_dangerous_name,
            "most_dangerous_neighbor_troops": most_dangerous_info.get("forces", 0),
            "most_dangerous_neighbor_owner": most_dangerous_info.get("owner", "?"),
        })

    threats.sort(key=lambda x: x["threat_score"], reverse=True)
    return threats
