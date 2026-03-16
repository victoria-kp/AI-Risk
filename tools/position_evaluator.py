"""Evaluate every player's strategic position on the board.

Computes continent progress, projected reinforcements, troop totals, and
identifies high-value expansion targets and defensive priorities for the
calling player. Also reports the same continent/reinforcement data for
opponents so the LLM can reason about relative strength and blocking.
"""

from typing import Dict, List


def evaluate_position(game=None, player=None, **kwargs) -> Dict:
    """Return a full positional evaluation for the board.

    Args:
        game: a pyrisk Game object
        player: a pyrisk Player object (the one asking)

    Returns:
        {
            "your_position": {
                "territories": int,
                "total_troops": int,
                "reinforcements_next_turn": int,
                "continent_progress": [{
                    "continent": str,
                    "owned": int,
                    "total": int,
                    "bonus": int,
                    "missing": [str, ...],
                }, ...],
            },
            "opponents": [{
                "name": str,
                "territories": int,
                "total_troops": int,
                "reinforcements_next_turn": int,
                "continent_progress": [{
                    "continent": str,
                    "owned": int,
                    "total": int,
                    "bonus": int,
                    "missing": [str, ...],
                }, ...],
            }, ...],
            "expansion_targets": [{
                "territory": str,
                "owner": str,
                "troops": int,
                "completes_continent": str,
                "continent_bonus": int,
            }, ...],
            "defensive_priorities": [{
                "territory": str,
                "your_troops": int,
                "would_complete_for": str,
                "continent": str,
                "continent_bonus": int,
            }, ...],
        }
    """
    your_position = _player_position(game.world, player)

    opponents = []
    for name, p in game.players.items():
        if p == player or not p.alive:
            continue
        opponents.append({
            "name": p.name,
            **_player_position(game.world, p),
        })
    # Strongest opponent first
    opponents.sort(key=lambda o: o["reinforcements_next_turn"], reverse=True)

    expansion_targets = _expansion_targets(game.world, player)
    defensive_priorities = _defensive_priorities(game.world, player, game)

    return {
        "your_position": your_position,
        "opponents": opponents,
        "expansion_targets": expansion_targets,
        "defensive_priorities": defensive_priorities,
    }


def _player_position(world, player) -> Dict:
    """Compute territory count, troops, reinforcements, and continent progress."""
    return {
        "territories": player.territory_count,
        "total_troops": player.forces,
        "reinforcements_next_turn": player.reinforcements,
        "continent_progress": _continent_progress(world, player),
    }


def _continent_progress(world, player) -> List[Dict]:
    """Return continent completion status for a player."""
    progress = []
    for area in sorted(world.areas.values(), key=lambda a: a.name):
        total = len(area.territories)
        owned = sum(1 for t in area.territories if t.owner == player)
        missing = [t.name for t in sorted(area.territories, key=lambda t: t.name)
                   if t.owner != player]
        progress.append({
            "continent": area.name,
            "owned": owned,
            "total": total,
            "bonus": area.value,
            "missing": missing,
        })
    return progress


def _expansion_targets(world, player) -> List[Dict]:
    """Find enemy territories whose capture would complete a continent for you."""
    targets = []
    for area in world.areas.values():
        missing = [t for t in area.territories if t.owner != player]
        if len(missing) == 1:
            t = missing[0]
            targets.append({
                "territory": t.name,
                "owner": t.owner.name,
                "troops": t.forces,
                "completes_continent": area.name,
                "continent_bonus": area.value,
            })
    # Highest bonus first
    targets.sort(key=lambda x: x["continent_bonus"], reverse=True)
    return targets


def _defensive_priorities(world, player, game) -> List[Dict]:
    """Find your territories that, if lost, would complete a continent for an opponent."""
    priorities = []
    for area in world.areas.values():
        your_territories = [t for t in area.territories if t.owner == player]
        if len(your_territories) != 1:
            continue
        # You own exactly 1 territory in this continent — check if one opponent owns the rest
        t = your_territories[0]
        other_territories = [ot for ot in area.territories if ot != t]
        if not other_territories:
            continue
        owners = set(ot.owner for ot in other_territories)
        if len(owners) == 1:
            opponent = owners.pop()
            if opponent != player and opponent.alive:
                priorities.append({
                    "territory": t.name,
                    "your_troops": t.forces,
                    "would_complete_for": opponent.name,
                    "continent": area.name,
                    "continent_bonus": area.value,
                })
    # Highest bonus first
    priorities.sort(key=lambda x: x["continent_bonus"], reverse=True)
    return priorities
