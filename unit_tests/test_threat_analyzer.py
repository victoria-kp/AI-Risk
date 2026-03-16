"""Unit tests for tools/threat_analyzer.py"""

import sys
import os
import unittest
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'pyrisk_vendor'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from game import Game
from world import CONNECT, MAP, KEY, AREAS
from ai.stupid import StupidAI
from tools.threat_analyzer import analyze_threats


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
        if g.player.alive:
            choices = g.player.ai.reinforce(g.player.reinforcements)
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


class TestOutputStructure(unittest.TestCase):
    """Tests for the return value format."""

    def setUp(self):
        self.game = make_game()
        self.player = self.game.players['RED']

    def test_returns_list(self):
        result = analyze_threats(game=self.game, player=self.player)
        self.assertIsInstance(result, list)

    def test_non_empty(self):
        result = analyze_threats(game=self.game, player=self.player)
        self.assertGreater(len(result), 0)

    def test_entries_are_dicts(self):
        result = analyze_threats(game=self.game, player=self.player)
        for entry in result:
            self.assertIsInstance(entry, dict)

    def test_has_all_keys(self):
        result = analyze_threats(game=self.game, player=self.player)
        expected = {
            'territory', 'your_troops', 'enemy_troops_adjacent',
            'threat_score', 'most_dangerous_neighbor',
            'most_dangerous_neighbor_troops', 'most_dangerous_neighbor_owner',
        }
        for entry in result:
            self.assertEqual(set(entry.keys()), expected)

    def test_territory_is_string(self):
        result = analyze_threats(game=self.game, player=self.player)
        for entry in result:
            self.assertIsInstance(entry['territory'], str)

    def test_your_troops_is_int(self):
        result = analyze_threats(game=self.game, player=self.player)
        for entry in result:
            self.assertIsInstance(entry['your_troops'], int)

    def test_enemy_troops_adjacent_is_int(self):
        result = analyze_threats(game=self.game, player=self.player)
        for entry in result:
            self.assertIsInstance(entry['enemy_troops_adjacent'], int)

    def test_threat_score_is_float(self):
        result = analyze_threats(game=self.game, player=self.player)
        for entry in result:
            self.assertIsInstance(entry['threat_score'], float)

    def test_most_dangerous_neighbor_is_string(self):
        result = analyze_threats(game=self.game, player=self.player)
        for entry in result:
            self.assertIsInstance(entry['most_dangerous_neighbor'], str)

    def test_most_dangerous_neighbor_troops_is_int(self):
        result = analyze_threats(game=self.game, player=self.player)
        for entry in result:
            self.assertIsInstance(entry['most_dangerous_neighbor_troops'], int)

    def test_most_dangerous_neighbor_owner_is_string(self):
        result = analyze_threats(game=self.game, player=self.player)
        for entry in result:
            self.assertIsInstance(entry['most_dangerous_neighbor_owner'], str)


class TestSorting(unittest.TestCase):
    """Results should be sorted by threat_score descending."""

    def setUp(self):
        self.game = make_game()
        self.player = self.game.players['RED']

    def test_sorted_descending(self):
        result = analyze_threats(game=self.game, player=self.player)
        scores = [r['threat_score'] for r in result]
        for i in range(len(scores) - 1):
            self.assertGreaterEqual(scores[i], scores[i + 1])

    def test_highest_threat_first(self):
        result = analyze_threats(game=self.game, player=self.player)
        max_score = max(r['threat_score'] for r in result)
        self.assertEqual(result[0]['threat_score'], max_score)


class TestThreatScoreCalculation(unittest.TestCase):
    """Verify threat_score = enemy_troops_adjacent / your_troops."""

    def setUp(self):
        self.game = make_game()
        self.player = self.game.players['RED']

    def test_score_matches_formula(self):
        result = analyze_threats(game=self.game, player=self.player)
        for r in result:
            expected = round(r['enemy_troops_adjacent'] / r['your_troops'], 2)
            self.assertEqual(r['threat_score'], expected,
                             f'{r["territory"]}: expected {expected}, got {r["threat_score"]}')

    def test_score_positive(self):
        result = analyze_threats(game=self.game, player=self.player)
        for r in result:
            self.assertGreater(r['threat_score'], 0)

    def test_manual_score(self):
        """Force specific troops and verify exact score."""
        game = make_game()
        player = game.players['RED']
        # Find a border territory
        for t in player.territories:
            enemies = [adj for adj in t.connect if adj.owner != player]
            if enemies:
                t.forces = 4
                total = sum(adj.forces for adj in enemies)
                result = analyze_threats(game=game, player=player)
                entry = [r for r in result if r['territory'] == t.name][0]
                self.assertEqual(entry['threat_score'], round(total / 4, 2))
                break


class TestMostDangerousNeighbor(unittest.TestCase):
    """Verify most_dangerous_neighbor has highest troops among enemy adjacents."""

    def setUp(self):
        self.game = make_game()
        self.player = self.game.players['RED']

    def test_is_actual_max(self):
        result = analyze_threats(game=self.game, player=self.player)
        for r in result:
            t = self.game.world.territory(r['territory'])
            enemies = [adj for adj in t.connect if adj.owner != self.player]
            actual_max = max(enemies, key=lambda adj: adj.forces)
            self.assertEqual(r['most_dangerous_neighbor'], actual_max.name)
            self.assertEqual(r['most_dangerous_neighbor_troops'], actual_max.forces)

    def test_owner_matches(self):
        result = analyze_threats(game=self.game, player=self.player)
        for r in result:
            t = self.game.world.territory(r['most_dangerous_neighbor'])
            self.assertEqual(r['most_dangerous_neighbor_owner'], t.owner.name)

    def test_owner_is_enemy(self):
        result = analyze_threats(game=self.game, player=self.player)
        for r in result:
            self.assertNotEqual(r['most_dangerous_neighbor_owner'], self.player.name)

    def test_manual_most_dangerous(self):
        """Force a specific neighbor to have the most troops."""
        game = make_game()
        player = game.players['RED']
        for t in player.territories:
            enemies = [adj for adj in t.connect if adj.owner != player]
            if len(enemies) >= 2:
                enemies[0].forces = 20
                enemies[1].forces = 1
                result = analyze_threats(game=game, player=player)
                entry = [r for r in result if r['territory'] == t.name][0]
                self.assertEqual(entry['most_dangerous_neighbor'], enemies[0].name)
                self.assertEqual(entry['most_dangerous_neighbor_troops'], 20)
                break


class TestBorderOnly(unittest.TestCase):
    """Only border territories with enemy neighbors should appear."""

    def setUp(self):
        self.game = make_game()
        self.player = self.game.players['RED']

    def test_all_results_are_border_territories(self):
        result = analyze_threats(game=self.game, player=self.player)
        result_names = {r['territory'] for r in result}
        for name in result_names:
            t = self.game.world.territory(name)
            self.assertEqual(t.owner, self.player)
            enemies = [adj for adj in t.connect if adj.owner != self.player]
            self.assertGreater(len(enemies), 0)

    def test_interior_territories_excluded(self):
        result = analyze_threats(game=self.game, player=self.player)
        result_names = {r['territory'] for r in result}
        for t in self.player.territories:
            enemies = [adj for adj in t.connect if adj.owner != self.player]
            if not enemies:
                self.assertNotIn(t.name, result_names)

    def test_all_border_territories_included(self):
        result = analyze_threats(game=self.game, player=self.player)
        result_names = {r['territory'] for r in result}
        for t in self.player.territories:
            enemies = [adj for adj in t.connect if adj.owner != self.player]
            if enemies:
                self.assertIn(t.name, result_names)

    def test_count_matches_border_count(self):
        result = analyze_threats(game=self.game, player=self.player)
        expected_count = sum(
            1 for t in self.player.territories
            if any(adj.owner != self.player for adj in t.connect)
        )
        self.assertEqual(len(result), expected_count)


class TestEnemyTroopsAdjacent(unittest.TestCase):
    """Verify enemy_troops_adjacent sums correctly."""

    def test_sum_matches_manual(self):
        game = make_game()
        player = game.players['RED']
        result = analyze_threats(game=game, player=player)
        for r in result:
            t = game.world.territory(r['territory'])
            enemies = [adj for adj in t.connect if adj.owner != player]
            expected_sum = sum(adj.forces for adj in enemies)
            self.assertEqual(r['enemy_troops_adjacent'], expected_sum)

    def test_manual_troop_sum(self):
        """Force known troop counts and verify sum."""
        game = make_game()
        player = game.players['RED']
        for t in player.territories:
            enemies = [adj for adj in t.connect if adj.owner != player]
            if len(enemies) >= 2:
                enemies[0].forces = 7
                enemies[1].forces = 3
                for e in enemies[2:]:
                    e.forces = 1
                expected = 7 + 3 + len(enemies[2:])
                result = analyze_threats(game=game, player=player)
                entry = [r for r in result if r['territory'] == t.name][0]
                self.assertEqual(entry['enemy_troops_adjacent'], expected)
                break


class TestMultiplePlayers(unittest.TestCase):
    """Test with different player counts."""

    def test_2_players(self):
        game = make_game(n_players=2)
        player = list(game.players.values())[0]
        result = analyze_threats(game=game, player=player)
        self.assertGreater(len(result), 0)

    def test_3_players(self):
        game = make_game(n_players=3)
        player = list(game.players.values())[0]
        result = analyze_threats(game=game, player=player)
        self.assertGreater(len(result), 0)

    def test_4_players(self):
        game = make_game(n_players=4)
        player = list(game.players.values())[0]
        result = analyze_threats(game=game, player=player)
        self.assertGreater(len(result), 0)

    def test_5_players(self):
        game = make_game(n_players=5)
        player = list(game.players.values())[0]
        result = analyze_threats(game=game, player=player)
        self.assertGreater(len(result), 0)

    def test_different_players_get_different_results(self):
        game = make_game(n_players=3)
        players = list(game.players.values())
        r0 = analyze_threats(game=game, player=players[0])
        r1 = analyze_threats(game=game, player=players[1])
        names0 = {r['territory'] for r in r0}
        names1 = {r['territory'] for r in r1}
        # Different players should own different territories
        self.assertNotEqual(names0, names1)


class TestKwargsIgnored(unittest.TestCase):
    """The function accepts **kwargs for compatibility with the tool interface."""

    def test_extra_kwargs_ignored(self):
        game = make_game()
        player = game.players['RED']
        result = analyze_threats(game=game, player=player, extra_arg='test')
        self.assertIsInstance(result, list)


if __name__ == '__main__':
    unittest.main()
