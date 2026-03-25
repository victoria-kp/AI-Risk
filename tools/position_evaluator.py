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


# ── Snapshot-based version for training ────────────────────────────────

CONTINENT_BONUSES = {
    "Africa": 3, "Asia": 7, "Australia": 2,
    "Europe": 5, "North America": 5, "South America": 2,
}


def evaluate_position_from_snapshot(snapshot: dict) -> Dict:
    """Snapshot-based version for use during training (no live game needed).

    Uses board_snapshot dict with keys: player_name, territory_map, players.
    Returns the same format as evaluate_position().
    """
    player_name = snapshot["player_name"]
    territory_map = snapshot["territory_map"]
    players_info = snapshot.get("players", {})

    # Build continent -> [territory_name] mapping
    continents = {}
    for t_name, t_info in territory_map.items():
        cont = t_info.get("continent", "")
        if cont not in continents:
            continents[cont] = []
        continents[cont].append(t_name)

    # Your position
    your_position = _player_position_snap(
        player_name, territory_map, continents, players_info
    )

    # Opponents
    opponents = []
    for p_name, p_info in players_info.items():
        if p_name == player_name or not p_info.get("alive", True):
            continue
        opp_pos = _player_position_snap(
            p_name, territory_map, continents, players_info
        )
        opponents.append({"name": p_name, **opp_pos})
    opponents.sort(
        key=lambda o: o["reinforcements_next_turn"], reverse=True
    )

    expansion_targets = _expansion_targets_snap(
        player_name, territory_map, continents
    )
    defensive_priorities = _defensive_priorities_snap(
        player_name, territory_map, continents, players_info
    )

    return {
        "your_position": your_position,
        "opponents": opponents,
        "expansion_targets": expansion_targets,
        "defensive_priorities": defensive_priorities,
    }


def _player_position_snap(player_name, territory_map, continents,
                          players_info) -> Dict:
    """Compute position for a player from snapshot data."""
    owned = [t for t, info in territory_map.items()
             if info.get("owner") == player_name]
    total_troops = sum(territory_map[t].get("forces", 0) for t in owned)

    # Reinforcements: base (territories // 3, min 3) + continent bonuses
    base_reinf = max(3, len(owned) // 3)
    continent_bonus = 0
    continent_progress = []
    for cont_name, cont_territories in sorted(continents.items()):
        total = len(cont_territories)
        owned_count = sum(1 for t in cont_territories
                          if territory_map[t].get("owner") == player_name)
        bonus = CONTINENT_BONUSES.get(cont_name, 0)
        if owned_count == total:
            continent_bonus += bonus
        missing = [t for t in sorted(cont_territories)
                   if territory_map[t].get("owner") != player_name]
        continent_progress.append({
            "continent": cont_name,
            "owned": owned_count,
            "total": total,
            "bonus": bonus,
            "missing": missing,
        })

    return {
        "territories": len(owned),
        "total_troops": total_troops,
        "reinforcements_next_turn": base_reinf + continent_bonus,
        "continent_progress": continent_progress,
    }


def _expansion_targets_snap(player_name, territory_map, continents) -> List[Dict]:
    """Find territories whose capture would complete a continent."""
    targets = []
    for cont_name, cont_territories in continents.items():
        missing = [t for t in cont_territories
                   if territory_map[t].get("owner") != player_name]
        if len(missing) == 1:
            t_name = missing[0]
            t_info = territory_map[t_name]
            targets.append({
                "territory": t_name,
                "owner": t_info.get("owner", "?"),
                "troops": t_info.get("forces", 0),
                "completes_continent": cont_name,
                "continent_bonus": CONTINENT_BONUSES.get(cont_name, 0),
            })
    targets.sort(key=lambda x: x["continent_bonus"], reverse=True)
    return targets


def _defensive_priorities_snap(player_name, territory_map, continents,
                               players_info) -> List[Dict]:
    """Find your territories that, if lost, would complete a continent for an opponent."""
    priorities = []
    for cont_name, cont_territories in continents.items():
        yours = [t for t in cont_territories
                 if territory_map[t].get("owner") == player_name]
        if len(yours) != 1:
            continue
        t_name = yours[0]
        others = [t for t in cont_territories if t != t_name]
        if not others:
            continue
        owners = set(territory_map[t].get("owner") for t in others)
        if len(owners) == 1:
            opp_name = owners.pop()
            if opp_name != player_name:
                opp_info = players_info.get(opp_name, {})
                if opp_info.get("alive", True):
                    priorities.append({
                        "territory": t_name,
                        "your_troops": territory_map[t_name].get("forces", 0),
                        "would_complete_for": opp_name,
                        "continent": cont_name,
                        "continent_bonus": CONTINENT_BONUSES.get(cont_name, 0),
                    })
    priorities.sort(key=lambda x: x["continent_bonus"], reverse=True)
    return priorities
