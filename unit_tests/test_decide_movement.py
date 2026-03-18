"""Unit tests for llm_player/nodes/decide_movement.py"""

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
from llm_player.nodes.decide_movement import (
    decide_movement,
    _parse_movement,
    _validate_movement,
    _fallback_movement,
    _parsed_as_skip,
)


def make_game(n_players=3, seed=42, turns=6):
    """Helper: create a pyrisk game and simulate some turns."""
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
    g.initial_placement()
    for _ in range(turns):
        if g.player.alive:
            choices = g.player.ai.reinforce(g.player.reinforcements)
            for tt, ff in choices.items():
                t = g.world.territory(tt)
                if t and t.owner == g.player:
                    t.forces += int(ff)
        g.turn += 1
    return g


def find_valid_movement(game, player):
    """Find a valid (src, target, count) for the player."""
    for t in player.territories:
        if t.forces > 1:
            for adj in t.connect:
                if adj.owner == player:
                    return t.name, adj.name, t.forces - 1
    return None, None, None


# ── _parse_movement tests ────────────────────────────────────────────


class TestParseMovement(unittest.TestCase):
    """Test JSON parsing from model output."""

    def test_parse_fenced_json(self):
        text = '```json\n{"movement": {"src": "Alaska", "target": "Kamchatka", "count": 3}}\n```'
        result = _parse_movement(text)
        self.assertEqual(result, {"src": "Alaska", "target": "Kamchatka", "count": 3})

    def test_parse_raw_json(self):
        text = '{"movement": {"src": "Brazil", "target": "Peru", "count": 2}}'
        result = _parse_movement(text)
        self.assertEqual(result, {"src": "Brazil", "target": "Peru", "count": 2})

    def test_parse_null_movement(self):
        text = '```json\n{"movement": null}\n```'
        result = _parse_movement(text)
        self.assertIsNone(result)

    def test_parse_with_surrounding_text(self):
        text = 'I will move:\n```json\n{"movement": {"src": "A", "target": "B", "count": 1}}\n```\nDone.'
        result = _parse_movement(text)
        self.assertEqual(result, {"src": "A", "target": "B", "count": 1})

    def test_parse_returns_none_for_garbage(self):
        result = _parse_movement("no json here")
        self.assertIsNone(result)

    def test_parse_returns_none_for_wrong_key(self):
        text = '```json\n{"attacks": []}\n```'
        result = _parse_movement(text)
        self.assertIsNone(result)

    def test_parse_returns_none_for_missing_src(self):
        text = '```json\n{"movement": {"target": "A", "count": 1}}\n```'
        result = _parse_movement(text)
        self.assertIsNone(result)

    def test_parse_returns_none_for_missing_count(self):
        text = '```json\n{"movement": {"src": "A", "target": "B"}}\n```'
        result = _parse_movement(text)
        self.assertIsNone(result)

    def test_parse_converts_count_to_int(self):
        text = '```json\n{"movement": {"src": "A", "target": "B", "count": 3.0}}\n```'
        result = _parse_movement(text)
        self.assertEqual(result["count"], 3)

    def test_parse_returns_none_for_non_dict_movement(self):
        text = '```json\n{"movement": "Alaska"}\n```'
        result = _parse_movement(text)
        self.assertIsNone(result)


# ── _parsed_as_skip tests ────────────────────────────────────────────


class TestParsedAsSkip(unittest.TestCase):
    """Test detection of explicit skip."""

    def test_null_movement_is_skip(self):
        text = '```json\n{"movement": null}\n```'
        self.assertTrue(_parsed_as_skip(text))

    def test_valid_movement_is_not_skip(self):
        text = '```json\n{"movement": {"src": "A", "target": "B", "count": 1}}\n```'
        self.assertFalse(_parsed_as_skip(text))

    def test_garbage_is_not_skip(self):
        self.assertFalse(_parsed_as_skip("no json"))

    def test_wrong_key_is_not_skip(self):
        text = '```json\n{"attacks": null}\n```'
        self.assertFalse(_parsed_as_skip(text))


# ── _validate_movement tests ─────────────────────────────────────────


class TestValidateMovement(unittest.TestCase):
    """Test validation of parsed movement."""

    def setUp(self):
        self.game = make_game()
        self.player = self.game.players['RED']
        self.src, self.target, self.count = find_valid_movement(self.game, self.player)

    def test_valid_movement(self):
        if not self.src:
            self.skipTest("No valid movement found")
        movement = {"src": self.src, "target": self.target, "count": self.count}
        result = _validate_movement(movement, self.player, self.game)
        self.assertIsNotNone(result)

    def test_unowned_src_returns_none(self):
        enemy = [t.name for t in self.game.world.territories.values()
                 if t.owner != self.player][0]
        movement = {"src": enemy, "target": self.target or "X", "count": 1}
        result = _validate_movement(movement, self.player, self.game)
        self.assertIsNone(result)

    def test_unowned_target_returns_none(self):
        if not self.src:
            self.skipTest("No valid movement found")
        enemy = [t.name for t in self.game.world.territories.values()
                 if t.owner != self.player][0]
        movement = {"src": self.src, "target": enemy, "count": 1}
        result = _validate_movement(movement, self.player, self.game)
        self.assertIsNone(result)

    def test_non_adjacent_returns_none(self):
        """Two owned territories that are not adjacent."""
        owned = list(self.player.territories)
        for t1 in owned:
            for t2 in owned:
                if t1 != t2 and t2 not in t1.connect and t1.forces > 1:
                    movement = {"src": t1.name, "target": t2.name, "count": 1}
                    result = _validate_movement(movement, self.player, self.game)
                    self.assertIsNone(result)
                    return
        self.skipTest("No non-adjacent owned pair found")

    def test_zero_count_returns_none(self):
        if not self.src:
            self.skipTest("No valid movement found")
        movement = {"src": self.src, "target": self.target, "count": 0}
        result = _validate_movement(movement, self.player, self.game)
        self.assertIsNone(result)

    def test_negative_count_returns_none(self):
        if not self.src:
            self.skipTest("No valid movement found")
        movement = {"src": self.src, "target": self.target, "count": -1}
        result = _validate_movement(movement, self.player, self.game)
        self.assertIsNone(result)

    def test_count_equal_to_forces_returns_none(self):
        """Can't move all troops — must leave at least 1."""
        if not self.src:
            self.skipTest("No valid movement found")
        src_t = self.game.world.territories[self.src]
        movement = {"src": self.src, "target": self.target, "count": src_t.forces}
        result = _validate_movement(movement, self.player, self.game)
        self.assertIsNone(result)

    def test_nonexistent_territory_returns_none(self):
        movement = {"src": "Atlantis", "target": "Nowhere", "count": 1}
        result = _validate_movement(movement, self.player, self.game)
        self.assertIsNone(result)


# ── _fallback_movement tests ─────────────────────────────────────────


class TestFallbackMovement(unittest.TestCase):
    """Test fallback movement logic."""

    def setUp(self):
        self.game = make_game()
        self.player = self.game.players['RED']

    def test_returns_none(self):
        result = _fallback_movement(self.player)
        self.assertIsNone(result)


# ── decide_movement integration tests ────────────────────────────────


class TestDecideMovementReturn(unittest.TestCase):
    """Test the full decide_movement function."""

    def setUp(self):
        self.game = make_game()
        self.player = self.game.players['RED']
        self.src, self.target, self.count = find_valid_movement(self.game, self.player)

    def _make_state(self, model):
        return {
            "game": self.game,
            "player": self.player,
            "model": model,
            "board_summary": "Test board summary",
        }

    def test_returns_dict(self):
        model = MockModelBackend(responses=['{"movement": null}'])
        result = decide_movement(self._make_state(model))
        self.assertIsInstance(result, dict)

    def test_has_required_keys(self):
        model = MockModelBackend(responses=['{"movement": null}'])
        result = decide_movement(self._make_state(model))
        self.assertIn("movement_decision", result)
        self.assertIn("movement_raw", result)

    def test_valid_movement_is_used(self):
        if not self.src:
            self.skipTest("No valid movement found")
        model = MockModelBackend(responses=[
            f'{{"movement": {{"src": "{self.src}", "target": "{self.target}", "count": {self.count}}}}}'
        ])
        result = decide_movement(self._make_state(model))
        decision = result["movement_decision"]
        self.assertIsNotNone(decision)
        self.assertEqual(decision["src"], self.src)

    def test_null_movement_accepted(self):
        model = MockModelBackend(responses=['{"movement": null}'])
        result = decide_movement(self._make_state(model))
        self.assertIsNone(result["movement_decision"])

    def test_invalid_output_triggers_fallback(self):
        model = MockModelBackend(responses=["I don't want to move"])
        result = decide_movement(self._make_state(model))
        # Fallback returns dict or None — either is acceptable
        decision = result["movement_decision"]
        if decision is not None:
            self.assertIn("src", decision)
            self.assertIn("target", decision)
            self.assertIn("count", decision)

    def test_invalid_movement_returns_none(self):
        """Invalid movement (bad territory) should fail validation."""
        model = MockModelBackend(responses=[
            '{"movement": {"src": "Atlantis", "target": "Nowhere", "count": 1}}'
        ])
        result = decide_movement(self._make_state(model))
        # Validation fails, returns None (no fallback for invalid movement dict)
        self.assertIsNone(result["movement_decision"])

    def test_raw_output_is_string(self):
        model = MockModelBackend(responses=["anything"])
        result = decide_movement(self._make_state(model))
        self.assertIsInstance(result["movement_raw"], str)


# ── Prompt construction tests ────────────────────────────────────────


class TestMovementPromptConstruction(unittest.TestCase):
    """Test that the prompt contains expected info."""

    def setUp(self):
        self.game = make_game()
        self.player = self.game.players['RED']

    def test_prompt_contains_board_summary(self):
        model = MockModelBackend(responses=["garbage"])
        state = {
            "game": self.game,
            "player": self.player,
            "model": model,
            "board_summary": "UNIQUE_MOVEMENT_MARKER",
        }
        decide_movement(state)
        prompt = model.call_log[0]["prompt"]
        self.assertIn("UNIQUE_MOVEMENT_MARKER", prompt)

    def test_prompt_contains_tool_instructions(self):
        model = MockModelBackend(responses=["garbage"])
        state = {
            "game": self.game,
            "player": self.player,
            "model": model,
            "board_summary": "summary",
        }
        decide_movement(state)
        prompt = model.call_log[0]["prompt"]
        self.assertIn("battle_sim", prompt)
        self.assertIn("tool_call", prompt)

    def test_prompt_mentions_movement(self):
        model = MockModelBackend(responses=["garbage"])
        state = {
            "game": self.game,
            "player": self.player,
            "model": model,
            "board_summary": "summary",
        }
        decide_movement(state)
        prompt = model.call_log[0]["prompt"]
        self.assertIn("movement", prompt.lower())


# ── Tool loop integration tests ──────────────────────────────────────


class TestMovementToolLoopIntegration(unittest.TestCase):
    """Test that tool calls in model output are processed."""

    def setUp(self):
        self.game = make_game()
        self.player = self.game.players['RED']

    def test_tool_call_triggers_second_model_call(self):
        responses = [
            '<tool_call>battle_sim(attacking=5, defending=3)</tool_call>',
            '{"movement": null}',
        ]
        model = MockModelBackend(responses=responses)
        state = {
            "game": self.game,
            "player": self.player,
            "model": model,
            "board_summary": "summary",
        }
        decide_movement(state)
        self.assertEqual(len(model.call_log), 2)

    def test_no_tool_call_means_single_model_call(self):
        model = MockModelBackend(responses=['{"movement": null}'])
        state = {
            "game": self.game,
            "player": self.player,
            "model": model,
            "board_summary": "summary",
        }
        decide_movement(state)
        self.assertEqual(len(model.call_log), 1)


if __name__ == '__main__':
    unittest.main()
