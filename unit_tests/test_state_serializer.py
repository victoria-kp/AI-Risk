"""Unit tests for risk_env/state_serializer.py"""

import sys
import os
import unittest
import random

# Add paths for pyrisk and project modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'pyrisk_vendor'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from game import Game
from world import CONNECT, MAP, KEY, AREAS
from ai.stupid import StupidAI
from risk_env.state_serializer import (
    serialize_game_state,
    _format_territory,
    _continent_progress,
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


class TestContinentProgress(unittest.TestCase):
    """Tests for _continent_progress()."""

    def setUp(self):
        self.game = make_game()
        self.player = self.game.players['RED']

    def test_returns_all_six_continents(self):
        progress = _continent_progress(self.game.world, self.player)
        names = {cp['name'] for cp in progress}
        self.assertEqual(names, {
            'Africa', 'Asia', 'Australia', 'Europe',
            'North America', 'South America',
        })

    def test_owned_plus_missing_equals_total(self):
        """owned + len(missing) should always equal total."""
        progress = _continent_progress(self.game.world, self.player)
        for cp in progress:
            self.assertEqual(cp['owned'] + len(cp['missing']), cp['total'])

    def test_bonus_values_match_pyrisk(self):
        """Bonus values should match the AREAS definition."""
        expected_bonuses = {name: val for name, (val, _) in AREAS.items()}
        progress = _continent_progress(self.game.world, self.player)
        for cp in progress:
            self.assertEqual(cp['bonus'], expected_bonuses[cp['name']])

    def test_complete_continent_has_no_missing(self):
        """Force a player to own all of Australia, check missing is empty."""
        for t in self.game.world.areas['Australia'].territories:
            t.owner = self.player
        progress = _continent_progress(self.game.world, self.player)
        australia = next(cp for cp in progress if cp['name'] == 'Australia')
        self.assertEqual(australia['missing'], [])
        self.assertEqual(australia['owned'], australia['total'])

    def test_missing_territories_not_owned_by_player(self):
        """Every territory in 'missing' should not be owned by the player."""
        progress = _continent_progress(self.game.world, self.player)
        for cp in progress:
            for name in cp['missing']:
                t = self.game.world.territories[name]
                self.assertNotEqual(t.owner, self.player)


class TestFormatTerritory(unittest.TestCase):
    """Tests for _format_territory()."""

    def setUp(self):
        self.game = make_game()
        self.player = self.game.players['RED']

    def test_starts_with_dash_and_territory_name(self):
        t = next(iter(self.player.territories))
        result = _format_territory(t, self.player)
        self.assertTrue(result.startswith(f'- {t.name} ('))

    def test_contains_troop_count(self):
        t = next(iter(self.player.territories))
        result = _format_territory(t, self.player)
        self.assertIn(f'{t.forces} troops', result)

    def test_contains_borders_keyword(self):
        t = next(iter(self.player.territories))
        result = _format_territory(t, self.player)
        self.assertIn('borders:', result)

    def test_friendly_neighbors_labeled_you(self):
        """If a neighbor is owned by the same player, it should say YOU."""
        # Find a territory with at least one friendly neighbor
        for t in self.player.territories:
            friendly = [adj for adj in t.connect if adj.owner == self.player]
            if friendly:
                result = _format_territory(t, self.player)
                self.assertIn('YOU', result)
                return
        self.skipTest('No territory with friendly neighbor found')

    def test_enemy_neighbors_labeled_with_owner_name(self):
        """Enemy neighbors should show their owner's name, not YOU."""
        for t in self.player.territories:
            enemies = [adj for adj in t.connect if adj.owner and adj.owner != self.player]
            if enemies:
                result = _format_territory(t, self.player)
                for adj in enemies:
                    self.assertIn(adj.owner.name, result)
                return
        self.skipTest('No territory with enemy neighbor found')

    def test_all_neighbors_appear(self):
        """Every connected territory should appear in the output."""
        t = next(iter(self.player.territories))
        result = _format_territory(t, self.player)
        for adj in t.connect:
            self.assertIn(adj.name, result)


class TestSerializeGameState(unittest.TestCase):
    """Tests for serialize_game_state()."""

    def setUp(self):
        self.game = make_game()
        self.player = self.game.players['RED']
        self.output = serialize_game_state(self.game, self.player)

    def test_starts_with_header(self):
        self.assertTrue(self.output.startswith('=== RISK GAME STATE ==='))

    def test_contains_player_name(self):
        self.assertIn(f'Your color: {self.player.name}', self.output)

    def test_contains_territory_count(self):
        count = self.player.territory_count
        self.assertIn(f'Territories: {count}', self.output)

    def test_contains_total_troops(self):
        troops = self.player.forces
        self.assertIn(f'Total troops: {troops}', self.output)

    def test_contains_reinforcements(self):
        reinforcements = self.player.reinforcements
        self.assertIn(f'Reinforcements next turn: {reinforcements}', self.output)

    def test_contains_all_sections(self):
        for section in ['CONTINENT STATUS:', 'YOUR TERRITORIES:',
                        'OPPONENT SUMMARY:', 'FULL BOARD:']:
            self.assertIn(section, self.output)

    def test_all_owned_territories_listed(self):
        """Every territory the player owns should appear in YOUR TERRITORIES."""
        your_section_start = self.output.index('YOUR TERRITORIES:')
        your_section_end = self.output.index('OPPONENT SUMMARY:')
        your_section = self.output[your_section_start:your_section_end]
        for t in self.player.territories:
            self.assertIn(t.name, your_section)

    def test_opponents_listed(self):
        """All alive opponents should appear in OPPONENT SUMMARY."""
        for name, p in self.game.players.items():
            if p != self.player and p.alive:
                self.assertIn(p.name, self.output)

    def test_dead_opponents_not_listed(self):
        """Dead players should not appear in OPPONENT SUMMARY."""
        # Kill GREEN by giving all their territories to RED
        for t in list(self.game.players['GREEN'].territories):
            t.owner = self.player
        output = serialize_game_state(self.game, self.player)
        opp_start = output.index('OPPONENT SUMMARY:')
        opp_end = output.index('FULL BOARD:')
        opp_section = output[opp_start:opp_end]
        self.assertNotIn('GREEN', opp_section)

    def test_full_board_lists_all_42_territories(self):
        """FULL BOARD should contain every territory on the map."""
        full_board_start = self.output.index('FULL BOARD:')
        full_board = self.output[full_board_start:]
        for name in self.game.world.territories:
            self.assertIn(name, full_board)

    def test_full_board_groups_by_continent(self):
        """Each continent name should appear as a header in FULL BOARD."""
        full_board_start = self.output.index('FULL BOARD:')
        full_board = self.output[full_board_start:]
        for area_name in self.game.world.areas:
            self.assertIn(area_name, full_board)

    def test_full_board_shows_you_for_owned(self):
        """Player's own territories should say YOU in FULL BOARD."""
        full_board_start = self.output.index('FULL BOARD:')
        full_board = self.output[full_board_start:]
        owned_name = next(iter(self.player.territories)).name
        # Find the line for this territory in FULL BOARD
        for line in full_board.split('\n'):
            if owned_name in line and line.strip().startswith(owned_name):
                self.assertIn('YOU', line)
                return
        self.fail(f'{owned_name} not found in FULL BOARD')

    def test_reinforcement_math_correct(self):
        """The base + continent bonus in the output should sum to total."""
        territory_bonus = max(self.player.territory_count // 3, 3)
        continent_bonus = sum(a.value for a in self.player.areas)
        expected = territory_bonus + continent_bonus
        self.assertIn(
            f'Reinforcements next turn: {expected} '
            f'(base {territory_bonus} + continent bonus {continent_bonus})',
            self.output,
        )

    def test_different_player_pov(self):
        """Serializing from BLUE's POV should list BLUE's territories."""
        blue = self.game.players['BLUE']
        output_blue = serialize_game_state(self.game, blue)
        self.assertIn(f'Your color: BLUE', output_blue)
        # RED should appear as opponent
        self.assertIn('RED', output_blue)


class TestEdgeCases(unittest.TestCase):
    """Edge cases and special scenarios."""

    def test_two_player_game(self):
        """Serializer should work with 2 players."""
        g = make_game(n_players=2)
        output = serialize_game_state(g, g.players['RED'])
        self.assertIn('BLUE', output)
        self.assertNotIn('GREEN', output)

    def test_five_player_game(self):
        """Serializer should work with 5 players."""
        g = make_game(n_players=5)
        output = serialize_game_state(g, g.players['RED'])
        for name in ['BLUE', 'GREEN', 'YELLOW', 'PURPLE']:
            self.assertIn(name, output)

    def test_complete_continent_shows_complete(self):
        """When a player owns an entire continent, output should say COMPLETE."""
        g = make_game()
        player = g.players['RED']
        for t in g.world.areas['Australia'].territories:
            t.owner = player
            t.forces = 2
        output = serialize_game_state(g, player)
        self.assertIn('Australia: COMPLETE', output)

    def test_output_is_string(self):
        g = make_game()
        output = serialize_game_state(g, g.players['RED'])
        self.assertIsInstance(output, str)

    def test_no_empty_lines_at_end(self):
        """Output should not end with trailing whitespace/newlines."""
        g = make_game()
        output = serialize_game_state(g, g.players['RED'])
        self.assertEqual(output, output.rstrip())


if __name__ == '__main__':
    unittest.main()
