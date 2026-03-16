"""Unit tests for tools/position_evaluator.py"""

import sys
import os
import unittest
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'pyrisk_vendor'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from game import Game
from world import CONNECT, MAP, KEY, AREAS
from ai.stupid import StupidAI
from tools.position_evaluator import evaluate_position


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

    def test_returns_dict(self):
        result = evaluate_position(game=self.game, player=self.player)
        self.assertIsInstance(result, dict)

    def test_top_level_keys(self):
        result = evaluate_position(game=self.game, player=self.player)
        expected = {'your_position', 'opponents', 'expansion_targets', 'defensive_priorities'}
        self.assertEqual(set(result.keys()), expected)

    def test_your_position_keys(self):
        result = evaluate_position(game=self.game, player=self.player)
        yp = result['your_position']
        expected = {'territories', 'total_troops', 'reinforcements_next_turn', 'continent_progress'}
        self.assertEqual(set(yp.keys()), expected)

    def test_your_position_types(self):
        result = evaluate_position(game=self.game, player=self.player)
        yp = result['your_position']
        self.assertIsInstance(yp['territories'], int)
        self.assertIsInstance(yp['total_troops'], int)
        self.assertIsInstance(yp['reinforcements_next_turn'], int)
        self.assertIsInstance(yp['continent_progress'], list)

    def test_continent_progress_keys(self):
        result = evaluate_position(game=self.game, player=self.player)
        for cp in result['your_position']['continent_progress']:
            expected = {'continent', 'owned', 'total', 'bonus', 'missing'}
            self.assertEqual(set(cp.keys()), expected)

    def test_continent_progress_types(self):
        result = evaluate_position(game=self.game, player=self.player)
        for cp in result['your_position']['continent_progress']:
            self.assertIsInstance(cp['continent'], str)
            self.assertIsInstance(cp['owned'], int)
            self.assertIsInstance(cp['total'], int)
            self.assertIsInstance(cp['bonus'], int)
            self.assertIsInstance(cp['missing'], list)

    def test_opponents_is_list(self):
        result = evaluate_position(game=self.game, player=self.player)
        self.assertIsInstance(result['opponents'], list)

    def test_opponent_has_name_and_position(self):
        result = evaluate_position(game=self.game, player=self.player)
        for opp in result['opponents']:
            self.assertIn('name', opp)
            self.assertIn('territories', opp)
            self.assertIn('total_troops', opp)
            self.assertIn('reinforcements_next_turn', opp)
            self.assertIn('continent_progress', opp)

    def test_expansion_target_keys(self):
        result = evaluate_position(game=self.game, player=self.player)
        for et in result['expansion_targets']:
            expected = {'territory', 'owner', 'troops', 'completes_continent', 'continent_bonus'}
            self.assertEqual(set(et.keys()), expected)

    def test_defensive_priority_keys(self):
        result = evaluate_position(game=self.game, player=self.player)
        for dp in result['defensive_priorities']:
            expected = {'territory', 'your_troops', 'would_complete_for', 'continent', 'continent_bonus'}
            self.assertEqual(set(dp.keys()), expected)


class TestYourPosition(unittest.TestCase):
    """Verify your_position matches pyrisk player data."""

    def setUp(self):
        self.game = make_game()
        self.player = self.game.players['RED']

    def test_territory_count(self):
        result = evaluate_position(game=self.game, player=self.player)
        self.assertEqual(result['your_position']['territories'], self.player.territory_count)

    def test_total_troops(self):
        result = evaluate_position(game=self.game, player=self.player)
        self.assertEqual(result['your_position']['total_troops'], self.player.forces)

    def test_reinforcements(self):
        result = evaluate_position(game=self.game, player=self.player)
        self.assertEqual(
            result['your_position']['reinforcements_next_turn'],
            self.player.reinforcements,
        )

    def test_six_continents(self):
        result = evaluate_position(game=self.game, player=self.player)
        self.assertEqual(len(result['your_position']['continent_progress']), 6)


class TestContinentProgress(unittest.TestCase):
    """Verify continent progress accuracy."""

    def setUp(self):
        self.game = make_game()
        self.player = self.game.players['RED']

    def test_owned_plus_missing_equals_total(self):
        result = evaluate_position(game=self.game, player=self.player)
        for cp in result['your_position']['continent_progress']:
            self.assertEqual(cp['owned'] + len(cp['missing']), cp['total'])

    def test_missing_territories_not_owned_by_player(self):
        result = evaluate_position(game=self.game, player=self.player)
        for cp in result['your_position']['continent_progress']:
            for tname in cp['missing']:
                t = self.game.world.territory(tname)
                self.assertNotEqual(t.owner, self.player)

    def test_owned_count_matches_actual(self):
        result = evaluate_position(game=self.game, player=self.player)
        for cp in result['your_position']['continent_progress']:
            area = self.game.world.area(cp['continent'])
            actual = sum(1 for t in area.territories if t.owner == self.player)
            self.assertEqual(cp['owned'], actual)

    def test_bonus_matches_area_value(self):
        result = evaluate_position(game=self.game, player=self.player)
        for cp in result['your_position']['continent_progress']:
            area = self.game.world.area(cp['continent'])
            self.assertEqual(cp['bonus'], area.value)


class TestOpponents(unittest.TestCase):
    """Tests for opponent data."""

    def setUp(self):
        self.game = make_game()
        self.player = self.game.players['RED']

    def test_excludes_self(self):
        result = evaluate_position(game=self.game, player=self.player)
        names = {opp['name'] for opp in result['opponents']}
        self.assertNotIn(self.player.name, names)

    def test_correct_opponent_count(self):
        result = evaluate_position(game=self.game, player=self.player)
        alive = sum(1 for p in self.game.players.values()
                    if p != self.player and p.alive)
        self.assertEqual(len(result['opponents']), alive)

    def test_sorted_by_reinforcements_descending(self):
        result = evaluate_position(game=self.game, player=self.player)
        reinfs = [opp['reinforcements_next_turn'] for opp in result['opponents']]
        for i in range(len(reinfs) - 1):
            self.assertGreaterEqual(reinfs[i], reinfs[i + 1])

    def test_opponent_data_matches_pyrisk(self):
        result = evaluate_position(game=self.game, player=self.player)
        for opp in result['opponents']:
            p = self.game.players[opp['name']]
            self.assertEqual(opp['territories'], p.territory_count)
            self.assertEqual(opp['total_troops'], p.forces)
            self.assertEqual(opp['reinforcements_next_turn'], p.reinforcements)

    def test_opponent_has_continent_progress(self):
        result = evaluate_position(game=self.game, player=self.player)
        for opp in result['opponents']:
            self.assertEqual(len(opp['continent_progress']), 6)
            for cp in opp['continent_progress']:
                self.assertEqual(cp['owned'] + len(cp['missing']), cp['total'])

    def test_opponent_continent_progress_matches_actual(self):
        result = evaluate_position(game=self.game, player=self.player)
        for opp in result['opponents']:
            p = self.game.players[opp['name']]
            for cp in opp['continent_progress']:
                area = self.game.world.area(cp['continent'])
                actual = sum(1 for t in area.territories if t.owner == p)
                self.assertEqual(cp['owned'], actual)


class TestExpansionTargets(unittest.TestCase):
    """Tests for expansion_targets."""

    def setUp(self):
        self.game = make_game()
        self.player = self.game.players['RED']

    def test_target_is_last_missing(self):
        """Each target should be the only territory the player doesn't own in a continent."""
        result = evaluate_position(game=self.game, player=self.player)
        for et in result['expansion_targets']:
            t = self.game.world.territory(et['territory'])
            area = t.area
            others = [ot for ot in area.territories if ot != t]
            for ot in others:
                self.assertEqual(ot.owner, self.player)

    def test_target_not_owned_by_player(self):
        result = evaluate_position(game=self.game, player=self.player)
        for et in result['expansion_targets']:
            t = self.game.world.territory(et['territory'])
            self.assertNotEqual(t.owner, self.player)

    def test_owner_matches(self):
        result = evaluate_position(game=self.game, player=self.player)
        for et in result['expansion_targets']:
            t = self.game.world.territory(et['territory'])
            self.assertEqual(et['owner'], t.owner.name)

    def test_troops_match(self):
        result = evaluate_position(game=self.game, player=self.player)
        for et in result['expansion_targets']:
            t = self.game.world.territory(et['territory'])
            self.assertEqual(et['troops'], t.forces)

    def test_sorted_by_bonus_descending(self):
        result = evaluate_position(game=self.game, player=self.player)
        bonuses = [et['continent_bonus'] for et in result['expansion_targets']]
        for i in range(len(bonuses) - 1):
            self.assertGreaterEqual(bonuses[i], bonuses[i + 1])

    def test_continent_bonus_matches_area(self):
        result = evaluate_position(game=self.game, player=self.player)
        for et in result['expansion_targets']:
            t = self.game.world.territory(et['territory'])
            self.assertEqual(et['continent_bonus'], t.area.value)


class TestDefensivePriorities(unittest.TestCase):
    """Tests for defensive_priorities."""

    def setUp(self):
        self.game = make_game()
        self.player = self.game.players['RED']

    def test_territory_owned_by_player(self):
        result = evaluate_position(game=self.game, player=self.player)
        for dp in result['defensive_priorities']:
            t = self.game.world.territory(dp['territory'])
            self.assertEqual(t.owner, self.player)

    def test_opponent_owns_rest_of_continent(self):
        result = evaluate_position(game=self.game, player=self.player)
        for dp in result['defensive_priorities']:
            t = self.game.world.territory(dp['territory'])
            area = t.area
            opponent = self.game.players[dp['would_complete_for']]
            others = [ot for ot in area.territories if ot != t]
            for ot in others:
                self.assertEqual(ot.owner, opponent)

    def test_troops_match(self):
        result = evaluate_position(game=self.game, player=self.player)
        for dp in result['defensive_priorities']:
            t = self.game.world.territory(dp['territory'])
            self.assertEqual(dp['your_troops'], t.forces)

    def test_sorted_by_bonus_descending(self):
        result = evaluate_position(game=self.game, player=self.player)
        bonuses = [dp['continent_bonus'] for dp in result['defensive_priorities']]
        for i in range(len(bonuses) - 1):
            self.assertGreaterEqual(bonuses[i], bonuses[i + 1])

    def test_would_complete_for_is_not_self(self):
        result = evaluate_position(game=self.game, player=self.player)
        for dp in result['defensive_priorities']:
            self.assertNotEqual(dp['would_complete_for'], self.player.name)

    def test_continent_bonus_matches_area(self):
        result = evaluate_position(game=self.game, player=self.player)
        for dp in result['defensive_priorities']:
            t = self.game.world.territory(dp['territory'])
            self.assertEqual(dp['continent_bonus'], t.area.value)


class TestMultiplePlayers(unittest.TestCase):
    """Test with different player counts."""

    def test_2_players(self):
        game = make_game(n_players=2)
        player = list(game.players.values())[0]
        result = evaluate_position(game=game, player=player)
        self.assertEqual(len(result['opponents']), 1)

    def test_3_players(self):
        game = make_game(n_players=3)
        player = list(game.players.values())[0]
        result = evaluate_position(game=game, player=player)
        alive = sum(1 for p in game.players.values() if p != player and p.alive)
        self.assertEqual(len(result['opponents']), alive)

    def test_4_players(self):
        game = make_game(n_players=4)
        player = list(game.players.values())[0]
        result = evaluate_position(game=game, player=player)
        alive = sum(1 for p in game.players.values() if p != player and p.alive)
        self.assertEqual(len(result['opponents']), alive)

    def test_5_players(self):
        game = make_game(n_players=5)
        player = list(game.players.values())[0]
        result = evaluate_position(game=game, player=player)
        alive = sum(1 for p in game.players.values() if p != player and p.alive)
        self.assertEqual(len(result['opponents']), alive)

    def test_different_players_see_different_positions(self):
        game = make_game(n_players=3)
        players = list(game.players.values())
        r0 = evaluate_position(game=game, player=players[0])
        r1 = evaluate_position(game=game, player=players[1])
        # Different players should see different opponent lists
        opp_names_0 = {opp['name'] for opp in r0['opponents']}
        opp_names_1 = {opp['name'] for opp in r1['opponents']}
        self.assertNotEqual(opp_names_0, opp_names_1)


class TestKwargsIgnored(unittest.TestCase):
    """The function accepts **kwargs for compatibility with the tool interface."""

    def test_extra_kwargs_ignored(self):
        game = make_game()
        player = game.players['RED']
        result = evaluate_position(game=game, player=player, extra='test')
        self.assertIn('your_position', result)


if __name__ == '__main__':
    unittest.main()
