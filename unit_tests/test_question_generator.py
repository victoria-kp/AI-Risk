"""Unit tests for risk_env/question_generator.py"""

import sys
import os
import unittest
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'pyrisk_vendor'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from game import Game
from world import CONNECT, MAP, KEY, AREAS
from ai.stupid import StupidAI
from risk_env.question_generator import (
    generate_questions,
    _reinforcement_questions,
    _attack_questions,
    _continent_questions,
    _push_or_split_questions,
    _target_player_questions,
    _troop_movement_questions,
    _bridge_questions,
)


def make_game(n_players=3, seed=42, turns=6):
    """Helper: create a pyrisk game and simulate some turns."""
    random.seed(seed)
    g = Game(curses=False, connect=CONNECT, cmap=MAP, ckey=KEY, areas=AREAS)
    names = ['RED', 'BLUE', 'GREEN', 'YELLOW', 'PURPLE'][:n_players]
    for name in names:
        g.add_player(name, StupidAI)
    g.turn_order = list(g.players)
    for i, name in enumerate(g.turn_order):
        g.players[name].color = i + 1
        g.players[name].ord = ord('x')
        g.players[name].ai.start()
    g.initial_placement()
    for _ in range(turns):
        live = [p for p in g.players.values() if p.alive]
        if len(live) < 2:
            break
        if g.player.alive:
            try:
                choices = g.player.ai.reinforce(g.player.reinforcements)
            except IndexError:
                g.turn += 1
                continue
            for tt, ff in choices.items():
                t = g.world.territory(tt)
                if t and t.owner == g.player:
                    t.forces += int(ff)
            for src, target, attack, move in g.player.ai.attack():
                st = g.world.territory(src)
                tt = g.world.territory(target)
                if (st and tt and st.owner == g.player
                        and tt.owner != g.player and tt in st.connect):
                    g.combat(st, tt, attack, move)
        g.turn += 1
    return g


class TestReinforcementQuestions(unittest.TestCase):
    """Tests for _reinforcement_questions()."""

    def setUp(self):
        self.game = make_game()
        self.player = self.game.players['RED']

    def test_returns_list(self):
        qs = _reinforcement_questions(self.game, self.player)
        self.assertIsInstance(qs, list)

    def test_all_strings(self):
        qs = _reinforcement_questions(self.game, self.player)
        for q in qs:
            self.assertIsInstance(q, str)

    def test_references_owned_territories(self):
        """Each question should mention a territory the player owns."""
        owned_names = {t.name for t in self.player.territories}
        qs = _reinforcement_questions(self.game, self.player)
        for q in qs:
            self.assertTrue(
                any(name in q for name in owned_names),
                f'Question does not reference owned territory: {q}'
            )

    def test_only_fires_when_outpressured(self):
        """Questions should only appear for territories where enemy pressure > own troops."""
        qs = _reinforcement_questions(self.game, self.player)
        mentioned = set()
        for t in self.player.territories:
            if t.border:
                enemy_pressure = sum(
                    adj.forces for adj in t.connect if adj.owner != self.player
                )
                if enemy_pressure > t.forces:
                    mentioned.add(t.name)
        for q in qs:
            self.assertTrue(
                any(name in q for name in mentioned),
                f'Question for non-pressured territory: {q}'
            )

    def test_contains_troop_counts(self):
        """Each question should mention troop numbers."""
        qs = _reinforcement_questions(self.game, self.player)
        for q in qs:
            self.assertIn('troops', q)

    def test_no_questions_for_interior_territories(self):
        """Interior territories (no enemy neighbors) should not generate questions."""
        qs = _reinforcement_questions(self.game, self.player)
        interior_names = {t.name for t in self.player.territories if not t.border}
        for q in qs:
            for name in interior_names:
                # Check the territory isn't the subject (it could appear as context)
                self.assertFalse(
                    q.startswith(f'Should I add troops on {name}'),
                    f'Interior territory {name} generated a question'
                )


class TestAttackQuestions(unittest.TestCase):
    """Tests for _attack_questions()."""

    def setUp(self):
        self.game = make_game(turns=30)
        self.player = self.game.players['RED']

    def test_returns_list(self):
        qs = _attack_questions(self.game, self.player)
        self.assertIsInstance(qs, list)

    def test_contains_from_keyword(self):
        qs = _attack_questions(self.game, self.player)
        for q in qs:
            self.assertIn('from', q)

    def test_contains_attack_keyword(self):
        qs = _attack_questions(self.game, self.player)
        for q in qs:
            self.assertIn('attack', q.lower())

    def test_source_is_owned(self):
        """The 'from' territory should be owned by the player."""
        owned_names = {t.name for t in self.player.territories}
        qs = _attack_questions(self.game, self.player)
        for q in qs:
            # Format: "... from SourceName (N troops)?"
            from_idx = q.index('from ')
            rest = q[from_idx + 5:]
            source_name = rest.split(' (')[0]
            self.assertIn(source_name, owned_names, f'Source not owned: {q}')

    def test_target_is_enemy(self):
        """The target territory should NOT be owned by the player."""
        owned_names = {t.name for t in self.player.territories}
        qs = _attack_questions(self.game, self.player)
        for q in qs:
            # Format: "Should I try to attack TargetName (OWNER, ..."
            target_part = q.split('attack ')[1]
            target_name = target_part.split(' (')[0]
            self.assertNotIn(target_name, owned_names, f'Target is owned: {q}')

    def test_no_questions_with_one_troop(self):
        """Territories with only 1 troop should not generate attack questions."""
        # Force all RED territories to 1 troop
        for t in self.player.territories:
            t.forces = 1
        qs = _attack_questions(self.game, self.player)
        self.assertEqual(len(qs), 0)

    def test_questions_appear_with_troops(self):
        """A territory with many troops adjacent to an enemy should generate questions."""
        # Use early game and force a border territory to have lots of troops
        g = make_game(seed=42, turns=6)
        player = g.players['RED']
        for t in player.territories:
            if t.border:
                t.forces = 10
                break
        qs = _attack_questions(g, player)
        self.assertGreater(len(qs), 0)


class TestContinentQuestions(unittest.TestCase):
    """Tests for _continent_questions()."""

    def setUp(self):
        self.game = make_game()
        self.player = self.game.players['RED']

    def test_returns_list(self):
        qs = _continent_questions(self.game, self.player)
        self.assertIsInstance(qs, list)

    def test_no_question_for_zero_owned(self):
        """Continents where the player owns 0 territories should not appear."""
        qs = _continent_questions(self.game, self.player)
        for area in self.game.world.areas.values():
            owned = sum(1 for t in area.territories if t.owner == self.player)
            if owned == 0:
                for q in qs:
                    self.assertNotIn(
                        f'conquer {area.name}', q,
                        f'Question for 0-owned continent {area.name}'
                    )

    def test_no_question_for_complete_continent(self):
        """Continents fully owned should not generate questions."""
        # Give RED all of Australia
        for t in self.game.world.areas['Australia'].territories:
            t.owner = self.player
        qs = _continent_questions(self.game, self.player)
        for q in qs:
            self.assertNotIn(
                'conquer Australia', q,
                'Question generated for fully owned continent'
            )

    def test_mentions_bonus(self):
        """Each continent question should mention the reinforcement bonus."""
        qs = _continent_questions(self.game, self.player)
        for q in qs:
            self.assertIn('reinforcements per turn', q)

    def test_mentions_owned_count(self):
        """Each question should include the X/Y ownership fraction."""
        qs = _continent_questions(self.game, self.player)
        for q in qs:
            self.assertIn('/', q)


class TestPushOrSplitQuestions(unittest.TestCase):
    """Tests for _push_or_split_questions()."""

    def setUp(self):
        self.game = make_game(turns=30)
        self.player = self.game.players['RED']

    def test_returns_list(self):
        qs = _push_or_split_questions(self.game, self.player)
        self.assertIsInstance(qs, list)

    def test_mentions_push(self):
        qs = _push_or_split_questions(self.game, self.player)
        for q in qs:
            self.assertIn('push in', q.lower())

    def test_mentions_alternative(self):
        qs = _push_or_split_questions(self.game, self.player)
        for q in qs:
            self.assertIn('also attack', q.lower())

    def test_fires_with_strong_borders(self):
        """Should generate questions when player has multiple strong borders."""
        # Force two border territories to have lots of troops
        borders = [t for t in self.player.territories if t.border]
        if len(borders) >= 2:
            borders[0].forces = 5
            borders[1].forces = 3
            qs = _push_or_split_questions(self.game, self.player)
            self.assertGreater(len(qs), 0)

    def test_no_questions_with_weak_borders(self):
        """Should not fire when all borders have <3 troops."""
        for t in self.player.territories:
            t.forces = 1
        qs = _push_or_split_questions(self.game, self.player)
        self.assertEqual(len(qs), 0)


class TestTargetPlayerQuestions(unittest.TestCase):
    """Tests for _target_player_questions()."""

    def setUp(self):
        self.game = make_game()
        self.player = self.game.players['RED']

    def test_returns_list(self):
        qs = _target_player_questions(self.game, self.player)
        self.assertIsInstance(qs, list)

    def test_no_self_targeting(self):
        """Should never suggest attacking yourself."""
        qs = _target_player_questions(self.game, self.player)
        for q in qs:
            self.assertNotIn(
                f'attacking {self.player.name}', q,
                f'Self-targeting question: {q}'
            )

    def test_mentions_opponent_name(self):
        """Each question should name a specific opponent."""
        qs = _target_player_questions(self.game, self.player)
        opponent_names = {
            p.name for p in self.game.players.values()
            if p != self.player and p.alive
        }
        for q in qs:
            self.assertTrue(
                any(name in q for name in opponent_names),
                f'No opponent named in: {q}'
            )

    def test_mentions_troop_count(self):
        qs = _target_player_questions(self.game, self.player)
        for q in qs:
            self.assertIn('troops', q)

    def test_dead_players_excluded(self):
        """Dead players should not generate questions."""
        # Kill GREEN
        for t in list(self.game.players['GREEN'].territories):
            t.owner = self.player
        qs = _target_player_questions(self.game, self.player)
        for q in qs:
            self.assertNotIn('GREEN', q)

    def test_mentions_continents_if_held(self):
        """If opponent controls a continent, question should mention it."""
        # Give BLUE all of Australia
        blue = self.game.players['BLUE']
        for t in self.game.world.areas['Australia'].territories:
            t.owner = blue
        qs = _target_player_questions(self.game, self.player)
        blue_qs = [q for q in qs if 'BLUE' in q]
        if blue_qs:
            self.assertTrue(
                any('Australia' in q for q in blue_qs),
                'BLUE controls Australia but it is not mentioned'
            )


class TestTroopMovementQuestions(unittest.TestCase):
    """Tests for _troop_movement_questions()."""

    def setUp(self):
        self.game = make_game()
        self.player = self.game.players['RED']

    def test_returns_list(self):
        qs = _troop_movement_questions(self.game, self.player)
        self.assertIsInstance(qs, list)

    def test_contains_move_keyword(self):
        # Force scenario to get questions
        for t in self.player.territories:
            if not t.border:
                t.forces = 5
                break
        qs = _troop_movement_questions(self.game, self.player)
        for q in qs:
            self.assertIn('move', q.lower())

    def test_inland_to_border(self):
        """Inland territory with >1 troops adjacent to border should generate a question."""
        # Find inland territory and pump it up
        for t in self.player.territories:
            if not t.border:
                t.forces = 8
                connected_borders = [
                    adj for adj in t.connect
                    if adj.owner == self.player and adj.border
                ]
                if connected_borders:
                    qs = _troop_movement_questions(self.game, self.player)
                    inland_qs = [q for q in qs if 'inland' in q]
                    self.assertGreater(len(inland_qs), 0)
                    return
        self.skipTest('No inland territory with friendly border neighbor found')

    def test_no_questions_with_one_troop(self):
        """No movement questions when all territories have 1 troop."""
        for t in self.player.territories:
            t.forces = 1
        qs = _troop_movement_questions(self.game, self.player)
        self.assertEqual(len(qs), 0)

    def test_border_to_border_imbalance(self):
        """Strong border territory connected to weak border should fire."""
        borders = [t for t in self.player.territories if t.border]
        if len(borders) >= 2:
            # Find two connected border territories
            for b1 in borders:
                for adj in b1.connect:
                    if adj.owner == self.player and adj.border and adj != b1:
                        b1.forces = 6
                        adj.forces = 1
                        qs = _troop_movement_questions(self.game, self.player)
                        border_qs = [q for q in qs if b1.name in q and adj.name in q]
                        self.assertGreater(len(border_qs), 0)
                        return
        self.skipTest('No connected border pair found')


class TestBridgeQuestions(unittest.TestCase):
    """Tests for _bridge_questions()."""

    def setUp(self):
        self.game = make_game()
        self.player = self.game.players['RED']

    def test_returns_list(self):
        qs = _bridge_questions(self.game, self.player)
        self.assertIsInstance(qs, list)

    def test_contains_connect_keyword(self):
        qs = _bridge_questions(self.game, self.player)
        for q in qs:
            self.assertIn('connect', q.lower())

    def test_mentions_troop_movement(self):
        qs = _bridge_questions(self.game, self.player)
        for q in qs:
            self.assertIn('troop movement', q.lower())

    def test_target_is_enemy(self):
        """The territory to conquer should not be owned by the player."""
        owned_names = {t.name for t in self.player.territories}
        qs = _bridge_questions(self.game, self.player)
        for q in qs:
            target_part = q.split('conquer ')[1]
            target_name = target_part.split(' (')[0]
            self.assertNotIn(target_name, owned_names, f'Bridge target is owned: {q}')

    def test_forced_bridge_scenario(self):
        """Set up a known bridge: enemy territory between two player territories
        that are not directly connected."""
        # Find an enemy territory adjacent to two of our territories
        for t in self.game.world.territories.values():
            if t.owner == self.player:
                continue
            friendly = [adj for adj in t.connect if adj.owner == self.player]
            if len(friendly) >= 2:
                # Ensure the two friendly territories aren't directly connected
                t1, t2 = friendly[0], friendly[1]
                if t2 not in {adj for adj in t1.connect if adj.owner == self.player}:
                    qs = _bridge_questions(self.game, self.player)
                    bridge_qs = [q for q in qs if t.name in q]
                    self.assertGreater(
                        len(bridge_qs), 0,
                        f'{t.name} bridges {t1.name} and {t2.name} but no question'
                    )
                    return
        self.skipTest('No bridge scenario found in this game state')


class TestGenerateQuestions(unittest.TestCase):
    """Tests for the main generate_questions() wrapper."""

    def setUp(self):
        self.game = make_game(turns=15)
        self.player = self.game.players['RED']

    def test_returns_list(self):
        qs = generate_questions(self.game, self.player)
        self.assertIsInstance(qs, list)

    def test_all_strings(self):
        qs = generate_questions(self.game, self.player)
        for q in qs:
            self.assertIsInstance(q, str)

    def test_default_n_is_3(self):
        random.seed(1)
        qs = generate_questions(self.game, self.player)
        self.assertLessEqual(len(qs), 3)

    def test_respects_n_parameter(self):
        random.seed(1)
        for n in [1, 2, 5]:
            qs = generate_questions(self.game, self.player, n=n)
            self.assertLessEqual(len(qs), n)

    def test_n_larger_than_available(self):
        """If n exceeds total questions, return all available."""
        random.seed(1)
        qs = generate_questions(self.game, self.player, n=1000)
        self.assertGreater(len(qs), 0)
        self.assertLessEqual(len(qs), 1000)

    def test_no_duplicate_questions(self):
        random.seed(1)
        qs = generate_questions(self.game, self.player, n=20)
        self.assertEqual(len(qs), len(set(qs)))

    def test_works_for_all_players(self):
        """Should produce questions for any alive player."""
        for name, p in self.game.players.items():
            if p.alive:
                random.seed(1)
                qs = generate_questions(self.game, p, n=3)
                self.assertIsInstance(qs, list)

    def test_two_player_game(self):
        g = make_game(n_players=2, turns=10)
        random.seed(1)
        qs = generate_questions(g, g.players['RED'], n=5)
        self.assertIsInstance(qs, list)
        self.assertGreater(len(qs), 0)

    def test_five_player_game(self):
        g = make_game(n_players=5, turns=10)
        random.seed(1)
        qs = generate_questions(g, g.players['RED'], n=5)
        self.assertIsInstance(qs, list)
        self.assertGreater(len(qs), 0)

    def test_shuffled(self):
        """Two calls with different seeds should (likely) produce different order."""
        random.seed(1)
        qs1 = generate_questions(self.game, self.player, n=10)
        random.seed(999)
        qs2 = generate_questions(self.game, self.player, n=10)
        # Both should return valid question lists of the same length
        self.assertEqual(len(qs1), len(qs2))
        # With different seeds, order (or selection due to internal randomness)
        # should likely differ
        self.assertNotEqual(qs1, qs2)


if __name__ == '__main__':
    unittest.main()
