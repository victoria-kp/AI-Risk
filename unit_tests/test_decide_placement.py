"""Unit tests for llm_player/nodes/decide_placement.py"""

import os
import sys
import unittest
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'pyrisk_vendor'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from game import Game
from world import CONNECT, MAP, KEY, AREAS
from ai.stupid import StupidAI
from llm_player.model import MockModelBackend
from llm_player.nodes.decide_placement import (
    decide_placement,
    _parse_placement,
    _validate_placement,
    _fallback_placement,
    _placement_summary,
    CONTINENT_PRIORITY,
)


def make_game_before_placement(n_players=3, seed=42):
    """Helper: create a pyrisk game before initial placement."""
    random.seed(seed)
    g = Game(curses=False, connect=CONNECT, cmap=MAP, ckey=KEY, areas=AREAS)
    names = ['RED', 'BLUE', 'GREEN'][:n_players]
    for name in names:
        g.add_player(name, StupidAI)
    g.turn_order = list(g.players)
    for i, name in enumerate(g.turn_order):
        g.players[name].color = i + 1
        g.players[name].ord = ord('x')
        g.players[name].ai.start()
    return g


def make_game_after_placement(n_players=3, seed=42):
    """Helper: create a pyrisk game after initial placement."""
    g = make_game_before_placement(n_players, seed)
    g.initial_placement()
    return g


# ── _parse_placement tests ──────────────────────────────────────────


class TestParsePlacement(unittest.TestCase):
    """Test JSON parsing from model output."""

    def test_parse_fenced_json(self):
        text = '```json\n{"territory": "Alaska"}\n```'
        self.assertEqual(_parse_placement(text), "Alaska")

    def test_parse_raw_json(self):
        text = '{"territory": "Brazil"}'
        self.assertEqual(_parse_placement(text), "Brazil")

    def test_parse_with_surrounding_text(self):
        text = 'I choose:\n```json\n{"territory": "Peru"}\n```\nDone.'
        self.assertEqual(_parse_placement(text), "Peru")

    def test_parse_returns_none_for_garbage(self):
        self.assertIsNone(_parse_placement("no json here"))

    def test_parse_returns_none_for_wrong_key(self):
        text = '```json\n{"attacks": []}\n```'
        self.assertIsNone(_parse_placement(text))

    def test_parse_returns_none_for_non_string_territory(self):
        text = '{"territory": 42}'
        self.assertIsNone(_parse_placement(text))

    def test_parse_returns_none_for_null_territory(self):
        text = '{"territory": null}'
        self.assertIsNone(_parse_placement(text))

    def test_parse_returns_none_for_list_territory(self):
        text = '{"territory": ["Alaska", "Brazil"]}'
        self.assertIsNone(_parse_placement(text))

    def test_parse_returns_none_for_empty_object(self):
        text = '{}'
        self.assertIsNone(_parse_placement(text))

    def test_parse_extracts_from_embedded_json(self):
        text = 'My analysis says:\n{"territory": "Egypt"}\nThat is my choice.'
        self.assertEqual(_parse_placement(text), "Egypt")


# ── _validate_placement tests ───────────────────────────────────────


class TestValidatePlacementClaiming(unittest.TestCase):
    """Test validation during claiming phase (empty list provided)."""

    def setUp(self):
        self.game = make_game_before_placement()
        self.player = self.game.players['RED']
        self.empty = list(self.game.world.territories.values())

    def test_valid_territory_in_empty(self):
        name = self.empty[0].name
        self.assertEqual(_validate_placement(name, self.player, self.game, self.empty), name)

    def test_nonexistent_territory(self):
        self.assertIsNone(_validate_placement("Atlantis", self.player, self.game, self.empty))

    def test_territory_not_in_empty(self):
        # Remove a territory from empty list, then try to claim it
        removed = self.empty.pop(0)
        self.assertIsNone(_validate_placement(removed.name, self.player, self.game, self.empty))

    def test_all_territories_are_valid_claims(self):
        for t in self.empty:
            result = _validate_placement(t.name, self.player, self.game, self.empty)
            self.assertEqual(result, t.name)


class TestValidatePlacementReinforcing(unittest.TestCase):
    """Test validation during reinforcing phase (empty is None)."""

    def setUp(self):
        self.game = make_game_after_placement()
        self.player = self.game.players['RED']

    def test_valid_owned_territory(self):
        owned = list(self.player.territories)[0]
        self.assertEqual(
            _validate_placement(owned.name, self.player, self.game, None),
            owned.name,
        )

    def test_enemy_territory_returns_none(self):
        enemy = [t for t in self.game.world.territories.values()
                 if t.owner != self.player][0]
        self.assertIsNone(_validate_placement(enemy.name, self.player, self.game, None))

    def test_nonexistent_territory_returns_none(self):
        self.assertIsNone(_validate_placement("Atlantis", self.player, self.game, None))

    def test_all_owned_territories_are_valid(self):
        for t in self.player.territories:
            result = _validate_placement(t.name, self.player, self.game, None)
            self.assertEqual(result, t.name)


# ── _fallback_placement tests ───────────────────────────────────────


class TestFallbackPlacementClaiming(unittest.TestCase):
    """Test fallback during claiming phase."""

    def setUp(self):
        self.game = make_game_before_placement()
        self.player = self.game.players['RED']
        self.empty = list(self.game.world.territories.values())

    def test_returns_string(self):
        result = _fallback_placement(self.player, self.game, self.empty)
        self.assertIsInstance(result, str)

    def test_returns_territory_in_empty(self):
        result = _fallback_placement(self.player, self.game, self.empty)
        empty_names = {t.name for t in self.empty}
        self.assertIn(result, empty_names)

    def test_prefers_small_continents(self):
        result = _fallback_placement(self.player, self.game, self.empty)
        t = self.game.world.territories[result]
        # Should pick from Australia or South America (first two in priority)
        self.assertIn(t.area.name, CONTINENT_PRIORITY[:2])

    def test_returns_valid_when_only_one_empty(self):
        single = [self.empty[0]]
        result = _fallback_placement(self.player, self.game, single)
        self.assertEqual(result, single[0].name)


class TestFallbackPlacementReinforcing(unittest.TestCase):
    """Test fallback during reinforcing phase."""

    def setUp(self):
        self.game = make_game_after_placement()
        self.player = self.game.players['RED']

    def test_returns_string(self):
        result = _fallback_placement(self.player, self.game, None)
        self.assertIsInstance(result, str)

    def test_returns_owned_territory(self):
        result = _fallback_placement(self.player, self.game, None)
        owned_names = {t.name for t in self.player.territories}
        self.assertIn(result, owned_names)

    def test_prefers_border_territories(self):
        result = _fallback_placement(self.player, self.game, None)
        t = self.game.world.territories[result]
        border = [tt for tt in self.player.territories if tt.border]
        if border:
            self.assertTrue(t.border)


# ── _placement_summary tests ────────────────────────────────────────


class TestPlacementSummary(unittest.TestCase):
    """Test placement board summary."""

    def test_before_placement_says_no_territories(self):
        game = make_game_before_placement()
        player = game.players['RED']
        summary = _placement_summary(game, player)
        self.assertIn("You do not own any territories yet", summary)

    def test_after_placement_lists_territories(self):
        game = make_game_after_placement()
        player = game.players['RED']
        summary = _placement_summary(game, player)
        self.assertIn("Your territories:", summary)
        # Should contain at least one owned territory name
        owned = list(player.territories)
        found = any(t.name in summary for t in owned)
        self.assertTrue(found)

    def test_contains_continent_info(self):
        game = make_game_after_placement()
        player = game.players['RED']
        summary = _placement_summary(game, player)
        self.assertIn("Continents:", summary)
        self.assertIn("Australia", summary)
        self.assertIn("bonus", summary)

    def test_returns_string(self):
        game = make_game_before_placement()
        player = game.players['RED']
        summary = _placement_summary(game, player)
        self.assertIsInstance(summary, str)

    def test_no_crash_with_unowned_territories(self):
        """Summary should not crash when territories have no owner."""
        game = make_game_before_placement()
        player = game.players['RED']
        # All territories are unowned — should not raise
        _placement_summary(game, player)


# ── decide_placement integration tests ──────────────────────────────


class TestDecidePlacementClaiming(unittest.TestCase):
    """Test full decide_placement during claiming phase."""

    def setUp(self):
        self.game = make_game_before_placement()
        self.player = self.game.players['RED']
        self.empty = list(self.game.world.territories.values())

    def _make_state(self, model):
        return {
            "game": self.game,
            "player": self.player,
            "model": model,
            "empty": self.empty,
            "remaining": 29,
        }

    def test_returns_dict(self):
        model = MockModelBackend(responses=['{"territory": "Alaska"}'])
        result = decide_placement(self._make_state(model))
        self.assertIsInstance(result, dict)

    def test_has_required_keys(self):
        model = MockModelBackend(responses=['{"territory": "Alaska"}'])
        result = decide_placement(self._make_state(model))
        self.assertIn("placement_decision", result)
        self.assertIn("placement_raw", result)

    def test_valid_territory_is_used(self):
        name = self.empty[0].name
        model = MockModelBackend(responses=[f'{{"territory": "{name}"}}'])
        result = decide_placement(self._make_state(model))
        self.assertEqual(result["placement_decision"], name)

    def test_invalid_territory_triggers_fallback(self):
        model = MockModelBackend(responses=['{"territory": "Atlantis"}'])
        result = decide_placement(self._make_state(model))
        # Fallback should return a valid territory from the empty list
        empty_names = {t.name for t in self.empty}
        self.assertIn(result["placement_decision"], empty_names)

    def test_garbage_triggers_fallback(self):
        model = MockModelBackend(responses=["I pick the moon"])
        result = decide_placement(self._make_state(model))
        empty_names = {t.name for t in self.empty}
        self.assertIn(result["placement_decision"], empty_names)

    def test_raw_output_is_string(self):
        model = MockModelBackend(responses=["anything"])
        result = decide_placement(self._make_state(model))
        self.assertIsInstance(result["placement_raw"], str)

    def test_model_is_called_once(self):
        model = MockModelBackend(responses=['{"territory": "Alaska"}'])
        decide_placement(self._make_state(model))
        self.assertEqual(len(model.call_log), 1)


class TestDecidePlacementReinforcing(unittest.TestCase):
    """Test full decide_placement during reinforcing phase."""

    def setUp(self):
        self.game = make_game_after_placement()
        self.player = self.game.players['RED']

    def _make_state(self, model):
        return {
            "game": self.game,
            "player": self.player,
            "model": model,
            "empty": None,
            "remaining": 10,
        }

    def test_valid_owned_territory_is_used(self):
        owned = list(self.player.territories)[0]
        model = MockModelBackend(responses=[f'{{"territory": "{owned.name}"}}'])
        result = decide_placement(self._make_state(model))
        self.assertEqual(result["placement_decision"], owned.name)

    def test_enemy_territory_triggers_fallback(self):
        enemy = [t for t in self.game.world.territories.values()
                 if t.owner != self.player][0]
        model = MockModelBackend(responses=[f'{{"territory": "{enemy.name}"}}'])
        result = decide_placement(self._make_state(model))
        # Should fallback to an owned territory
        owned_names = {t.name for t in self.player.territories}
        self.assertIn(result["placement_decision"], owned_names)

    def test_garbage_triggers_fallback(self):
        model = MockModelBackend(responses=["no idea"])
        result = decide_placement(self._make_state(model))
        owned_names = {t.name for t in self.player.territories}
        self.assertIn(result["placement_decision"], owned_names)


# ── Prompt construction tests ───────────────────────────────────────


class TestPlacementPromptConstruction(unittest.TestCase):
    """Test that prompts contain expected info."""

    def test_claiming_prompt_contains_empty_territories(self):
        game = make_game_before_placement()
        player = game.players['RED']
        empty = list(game.world.territories.values())
        model = MockModelBackend(responses=["garbage"])
        state = {
            "game": game, "player": player, "model": model,
            "empty": empty, "remaining": 29,
        }
        decide_placement(state)
        prompt = model.call_log[0]["prompt"]
        self.assertIn("Unclaimed territories:", prompt)
        self.assertIn("Alaska", prompt)

    def test_claiming_prompt_contains_remaining(self):
        game = make_game_before_placement()
        player = game.players['RED']
        empty = list(game.world.territories.values())
        model = MockModelBackend(responses=["garbage"])
        state = {
            "game": game, "player": player, "model": model,
            "empty": empty, "remaining": 29,
        }
        decide_placement(state)
        prompt = model.call_log[0]["prompt"]
        self.assertIn("29", prompt)

    def test_reinforcing_prompt_says_all_claimed(self):
        game = make_game_after_placement()
        player = game.players['RED']
        model = MockModelBackend(responses=["garbage"])
        state = {
            "game": game, "player": player, "model": model,
            "empty": None, "remaining": 10,
        }
        decide_placement(state)
        prompt = model.call_log[0]["prompt"]
        self.assertIn("All territories have been claimed", prompt)

    def test_reinforcing_prompt_contains_remaining(self):
        game = make_game_after_placement()
        player = game.players['RED']
        model = MockModelBackend(responses=["garbage"])
        state = {
            "game": game, "player": player, "model": model,
            "empty": None, "remaining": 15,
        }
        decide_placement(state)
        prompt = model.call_log[0]["prompt"]
        self.assertIn("15", prompt)

    def test_prompt_contains_board_summary(self):
        game = make_game_after_placement()
        player = game.players['RED']
        model = MockModelBackend(responses=["garbage"])
        state = {
            "game": game, "player": player, "model": model,
            "empty": None, "remaining": 10,
        }
        decide_placement(state)
        prompt = model.call_log[0]["prompt"]
        self.assertIn("Continents:", prompt)

    def test_no_tool_instructions_in_prompt(self):
        """Placement should not mention tools."""
        game = make_game_before_placement()
        player = game.players['RED']
        empty = list(game.world.territories.values())
        model = MockModelBackend(responses=["garbage"])
        state = {
            "game": game, "player": player, "model": model,
            "empty": empty, "remaining": 29,
        }
        decide_placement(state)
        prompt = model.call_log[0]["prompt"]
        self.assertNotIn("tool_call", prompt)
        self.assertNotIn("battle_sim", prompt)


if __name__ == '__main__':
    unittest.main()
