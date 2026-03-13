"""Convert a pyrisk game state into a text prompt for the LLM.

Takes a pyrisk game object and a player, and produces a multi-line string describing
the board from that player's perspective: territory ownership, troop counts, adjacencies,
continent progress, and opponent summaries.
"""

from typing import Dict, List


def serialize_game_state(game, player) -> str:
    """Produce a full text description of the game from player's POV.

    Args:
        game: a pyrisk Game object (has .world, .players)
        player: a pyrisk Player object

    Returns:
        A multi-line string ready to be inserted into an LLM prompt.
    """
    lines = ["=== RISK GAME STATE ==="]

    # Player overview
    territory_count = player.territory_count
    total_troops = player.forces
    reinforcements = player.reinforcements
    territory_bonus = max(territory_count // 3, 3)
    continent_bonus = sum(a.value for a in player.areas)

    lines.append(
        f"Your color: {player.name} | Territories: {territory_count} | "
        f"Total troops: {total_troops}"
    )
    lines.append(
        f"Reinforcements next turn: {reinforcements} "
        f"(base {territory_bonus} + continent bonus {continent_bonus})"
    )
    lines.append("")

    # Continent progress
    lines.append("CONTINENT STATUS:")
    for cp in _continent_progress(game.world, player):
        if cp["owned"] == cp["total"]:
            lines.append(
                f"- {cp['name']}: COMPLETE ({cp['total']}/{cp['total']}, "
                f"+{cp['bonus']} bonus)"
            )
        else:
            missing_str = ", ".join(cp["missing"])
            lines.append(
                f"- {cp['name']}: You own {cp['owned']}/{cp['total']} "
                f"(missing: {missing_str}) [+{cp['bonus']} bonus if complete]"
            )
    lines.append("")

    # Player's territories
    lines.append("YOUR TERRITORIES:")
    owned = sorted(player.territories, key=lambda t: t.area.name)
    for t in owned:
        lines.append(_format_territory(t, player))
    lines.append("")

    # Opponent summaries
    lines.append("OPPONENT SUMMARY:")
    for name, p in game.players.items():
        if p == player or not p.alive:
            continue
        opp_territories = p.territory_count
        opp_troops = p.forces
        opp_continents = [a.name for a in p.areas]
        continent_str = ", ".join(opp_continents) if opp_continents else "none"
        lines.append(
            f"- {p.name}: {opp_territories} territories, ~{opp_troops} troops, "
            f"controls {continent_str}"
        )
    lines.append("")

    # Full board — every territory grouped by continent
    lines.append("FULL BOARD:")
    for area in sorted(game.world.areas.values(), key=lambda a: a.name):
        lines.append(f"  {area.name} (+{area.value} bonus):")
        for t in sorted(area.territories, key=lambda t: t.name):
            owner_name = "YOU" if t.owner == player else t.owner.name
            lines.append(f"    {t.name}: {owner_name}, {t.forces} troops")

    return "\n".join(lines)


def _format_territory(territory, player) -> str:
    """Format one territory line with neighbors and their owners."""
    neighbors = []
    for adj in sorted(territory.connect, key=lambda t: t.name):
        owner_name = adj.owner.name if adj.owner else "unowned"
        if adj.owner == player:
            neighbors.append(f"{adj.name}(YOU,{adj.forces})")
        else:
            neighbors.append(f"{adj.name}({owner_name},{adj.forces})")
    neighbor_str = ", ".join(neighbors)
    return f"- {territory.name} ({territory.forces} troops) borders: {neighbor_str}"


def _continent_progress(world, player) -> List[Dict]:
    """Return list of {name, owned, total, bonus, missing} for each continent."""
    progress = []
    for area in sorted(world.areas.values(), key=lambda a: a.name):
        total = len(area.territories)
        owned = sum(1 for t in area.territories if t.owner == player)
        missing = [t.name for t in sorted(area.territories, key=lambda t: t.name)
                   if t.owner != player]
        progress.append({
            "name": area.name,
            "owned": owned,
            "total": total,
            "bonus": area.value,
            "missing": missing,
        })
    return progress
