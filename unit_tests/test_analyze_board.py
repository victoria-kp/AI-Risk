"""Unit tests for llm_player/nodes/analyze_board.py"""

import os
import sys
import unittest
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'pyrisk_vendor'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from game import Game
from world import CONNECT, MAP, KEY, AREAS
from ai.stupid import StupidAI
from llm_player.nodes.analyze_board import analyze_board


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


# ── analyze_board() return value ─────────────────────────────────────


class TestAnalyzeBoardReturn(unittest.TestCase):
    """Test that analyze_board returns correct structure."""

    def setUp(self):
        self.game = make_game()
        self.player = self.game.players['RED']
        self.state = {"game": self.game, "player": self.player}

    def test_returns_dict(self):
        result = analyze_board(self.state)
        self.assertIsInstance(result, dict)

    def test_has_board_summary_key(self):
        result = analyze_board(self.state)
        self.assertIn("board_summary", result)

    def test_board_summary_is_string(self):
        result = analyze_board(self.state)
        self.assertIsInstance(result["board_summary"], str)

    def test_only_returns_board_summary(self):
        """Should not modify other state keys — only return board_summary."""
        result = analyze_board(self.state)
        self.assertEqual(list(result.keys()), ["board_summary"])


# ── Board summary content ────────────────────────────────────────────


class TestBoardSummaryContent(unittest.TestCase):
    """Test that board_summary contains expected sections."""

    def setUp(self):
        self.game = make_game()
        self.player = self.game.players['RED']
        self.state = {"game": self.game, "player": self.player}
        self.summary = analyze_board(self.state)["board_summary"]

    def test_starts_with_header(self):
        self.assertTrue(self.summary.startswith("=== RISK GAME STATE ==="))

    def test_contains_player_name(self):
        self.assertIn(f"Your color: {self.player.name}", self.summary)

    def test_contains_all_sections(self):
        for section in ['CONTINENT STATUS:', 'YOUR TERRITORIES:',
                        'OPPONENT SUMMARY:', 'FULL BOARD:']:
            self.assertIn(section, self.summary)

    def test_contains_territory_count(self):
        count = self.player.territory_count
        self.assertIn(f"Territories: {count}", self.summary)

    def test_contains_troop_count(self):
        troops = self.player.forces
        self.assertIn(f"Total troops: {troops}", self.summary)

    def test_lists_all_owned_territories(self):
        your_start = self.summary.index('YOUR TERRITORIES:')
        your_end = self.summary.index('OPPONENT SUMMARY:')
        your_section = self.summary[your_start:your_end]
        for t in self.player.territories:
            self.assertIn(t.name, your_section)

    def test_lists_opponents(self):
        for name, p in self.game.players.items():
            if p != self.player and p.alive:
                self.assertIn(p.name, self.summary)


# ── Different players / game states ──────────────────────────────────


class TestAnalyzeBoardDifferentPlayers(unittest.TestCase):
    """Test analyze_board with different player perspectives."""

    def setUp(self):
        self.game = make_game()

    def test_red_perspective(self):
        state = {"game": self.game, "player": self.game.players['RED']}
        summary = analyze_board(state)["board_summary"]
        self.assertIn("Your color: RED", summary)

    def test_blue_perspective(self):
        state = {"game": self.game, "player": self.game.players['BLUE']}
        summary = analyze_board(state)["board_summary"]
        self.assertIn("Your color: BLUE", summary)

    def test_different_perspectives_differ(self):
        state_red = {"game": self.game, "player": self.game.players['RED']}
        state_blue = {"game": self.game, "player": self.game.players['BLUE']}
        summary_red = analyze_board(state_red)["board_summary"]
        summary_blue = analyze_board(state_blue)["board_summary"]
        self.assertNotEqual(summary_red, summary_blue)


# ── Matches direct serializer call ───────────────────────────────────


class TestAnalyzeBoardMatchesSerializer(unittest.TestCase):
    """Verify analyze_board output matches direct serialize_game_state call."""

    def test_output_matches_serializer(self):
        from risk_env.state_serializer import serialize_game_state
        game = make_game()
        player = game.players['RED']
        state = {"game": game, "player": player}

        result = analyze_board(state)
        direct = serialize_game_state(game, player)
        self.assertEqual(result["board_summary"], direct)


# ── Edge cases ────────────────────────────────────────────────────────


class TestAnalyzeBoardEdgeCases(unittest.TestCase):
    """Edge cases for analyze_board."""

    def test_two_player_game(self):
        game = make_game(n_players=2)
        state = {"game": game, "player": game.players['RED']}
        result = analyze_board(state)
        self.assertIn("BLUE", result["board_summary"])
        self.assertNotIn("GREEN", result["board_summary"])

    def test_five_player_game(self):
        game = make_game(n_players=5)
        state = {"game": game, "player": game.players['RED']}
        result = analyze_board(state)
        for name in ['BLUE', 'GREEN', 'YELLOW', 'PURPLE']:
            self.assertIn(name, result["board_summary"])

    def test_early_game(self):
        """Works with 0 turns simulated (just after initial placement)."""
        game = make_game(turns=0)
        state = {"game": game, "player": game.players['RED']}
        result = analyze_board(state)
        self.assertIn("=== RISK GAME STATE ===", result["board_summary"])

    def test_late_game(self):
        """Works after many turns."""
        game = make_game(turns=20)
        state = {"game": game, "player": game.players['RED']}
        result = analyze_board(state)
        self.assertIn("=== RISK GAME STATE ===", result["board_summary"])

    def test_does_not_modify_input_state(self):
        """analyze_board should not mutate the input state dict."""
        game = make_game()
        state = {"game": game, "player": game.players['RED']}
        original_keys = set(state.keys())
        analyze_board(state)
        self.assertEqual(set(state.keys()), original_keys)


if __name__ == '__main__':
    unittest.main()
