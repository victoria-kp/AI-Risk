"""Generate strategic questions from a game state.

Programmatically creates tactical and strategic questions based on the current
board configuration. Seven question types cover reinforcement, attack, continent
strategy, focus decisions, troop movement, and bridge conquests.
"""

import random
from typing import List


def generate_questions(game, player, n=3) -> List[str]:
    """Produce n strategic questions relevant to the current state.

    Args:
        game: a pyrisk Game object
        player: a pyrisk Player object
        n: number of questions to return

    Returns:
        List of question strings.
    """
    questions = []
    questions.extend(_reinforcement_questions(game, player))
    questions.extend(_attack_questions(game, player))
    questions.extend(_continent_questions(game, player))
    questions.extend(_push_or_split_questions(game, player))
    questions.extend(_target_player_questions(game, player))
    questions.extend(_troop_movement_questions(game, player))
    questions.extend(_bridge_questions(game, player))
    random.shuffle(questions)
    return questions[:n]


def _reinforcement_questions(game, player) -> List[str]:
    """Type 1: Should I add troops on [owned territory]?"""
    questions = []
    border_territories = [t for t in player.territories if t.border]
    for t in border_territories:
        enemy_pressure = sum(
            adj.forces for adj in t.connect if adj.owner != player
        )
        if enemy_pressure > t.forces:
            questions.append(
                f"Should I add troops on {t.name}? "
                f"I have {t.forces} troops there but face "
                f"{enemy_pressure} enemy troops on adjacent territories."
            )
    return questions


def _attack_questions(game, player) -> List[str]:
    """Type 2: Should I attack [enemy territory] from [my territory]?"""
    questions = []
    for t in player.territories:
        for adj in t.connect:
            if adj.owner and adj.owner != player and t.forces > 1:
                questions.append(
                    f"Should I try to attack {adj.name} "
                    f"({adj.owner.name}, {adj.forces} troops) "
                    f"from {t.name} ({t.forces} troops)?"
                )
    return questions


def _continent_questions(game, player) -> List[str]:
    """Type 3: Should I try to conquer [continent]?"""
    questions = []
    for area in game.world.areas.values():
        owned = sum(1 for t in area.territories if t.owner == player)
        total = len(area.territories)
        if 0 < owned < total:
            missing = [t for t in area.territories if t.owner != player]
            missing_troops = sum(t.forces for t in missing)
            questions.append(
                f"Should I try to conquer {area.name}? "
                f"I own {owned}/{total} territories there "
                f"({total - owned} remaining with ~{missing_troops} enemy troops). "
                f"Completing it gives +{area.value} reinforcements per turn."
            )
    return questions


def _push_or_split_questions(game, player) -> List[str]:
    """Type 4: Should I push in [territory] or attack somewhere else?"""
    questions = []
    # Find territories where the player has a strong position (high troops)
    strong_borders = [
        t for t in player.territories
        if t.border and t.forces >= 3
    ]
    for t in strong_borders:
        enemies = [adj for adj in t.connect if adj.owner and adj.owner != player]
        if enemies:
            other_fronts = [
                other for other in player.territories
                if other != t and other.border and other.forces >= 2
            ]
            if other_fronts:
                other = random.choice(other_fronts)
                questions.append(
                    f"Should I push in {t.name} ({t.forces} troops) "
                    f"or try to also attack from {other.name} "
                    f"({other.forces} troops)?"
                )
    return questions


def _target_player_questions(game, player) -> List[str]:
    """Type 5: Should I focus on attacking [some player]?"""
    questions = []
    opponents = [
        p for p in game.players.values()
        if p != player and p.alive
    ]
    for opp in opponents:
        # Check if this opponent borders the player
        borders_us = any(
            adj.owner == opp
            for t in player.territories
            for adj in t.connect
        )
        if borders_us:
            opp_continents = [a.name for a in opp.areas]
            if opp_continents:
                continent_str = ", ".join(opp_continents)
                questions.append(
                    f"Should I focus on attacking {opp.name}? "
                    f"They control {continent_str} "
                    f"and have {opp.territory_count} territories "
                    f"with ~{opp.forces} troops."
                )
            else:
                questions.append(
                    f"Should I focus on attacking {opp.name}? "
                    f"They have {opp.territory_count} territories "
                    f"with ~{opp.forces} troops."
                )
    return questions


def _troop_movement_questions(game, player) -> List[str]:
    """Type 6: Should I move my troops from [country] to [country]?"""
    questions = []
    # Find inland territories with troops that could be moved to the border
    inland = [
        t for t in player.territories
        if not t.border and t.forces > 1
    ]
    border = [t for t in player.territories if t.border]
    for src in inland:
        # Find a border territory connected (directly or through friendly chain)
        connected_borders = [
            adj for adj in src.connect
            if adj.owner == player and adj.border
        ]
        for dest in connected_borders:
            questions.append(
                f"Should I move my troops from {src.name} "
                f"({src.forces} troops, inland) "
                f"to {dest.name} ({dest.forces} troops, on the border)?"
            )
    # Also consider moving between border territories
    for t in border:
        if t.forces >= 3:
            friendly_borders = [
                adj for adj in t.connect
                if adj.owner == player and adj.border and adj.forces < t.forces
            ]
            for dest in friendly_borders:
                questions.append(
                    f"Should I move my troops from {t.name} "
                    f"({t.forces} troops) to {dest.name} "
                    f"({dest.forces} troops)?"
                )
    return questions


def _bridge_questions(game, player) -> List[str]:
    """Type 7: Should I conquer [territory] to connect [my territory] to
    [my other territory]?"""
    questions = []
    # Find enemy territories that sit between two player territories
    for t in game.world.territories.values():
        if t.owner == player:
            continue
        # Get player-owned neighbors of this enemy territory
        friendly_neighbors = [
            adj for adj in t.connect if adj.owner == player
        ]
        if len(friendly_neighbors) >= 2:
            # This enemy territory bridges two of our territories
            t1 = friendly_neighbors[0]
            t2 = friendly_neighbors[1]
            # Check they aren't already directly connected through friendly territory
            t1_friendly_connections = {
                adj.name for adj in t1.connect if adj.owner == player
            }
            if t2.name not in t1_friendly_connections:
                questions.append(
                    f"Should I try to conquer {t.name} "
                    f"({t.owner.name}, {t.forces} troops) "
                    f"to connect {t1.name} to {t2.name} "
                    f"and allow troop movement between them?"
                )
    return questions
