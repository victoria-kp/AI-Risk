"""Unit tests for llm_player/nodes/decide_reinforcements.py"""

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
from llm_player.nodes.decide_reinforcements import (
    decide_reinforcements,
    _parse_reinforcements,
    _validate_reinforcements,
    _fallback_reinforcements,
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


def get_owned_names(player):
    """Get sorted list of territory names owned by a player."""
    return sorted(t.name for t in player.territories)


# ── _parse_reinforcements tests ──────────────────────────────────────


class TestParseReinforcements(unittest.TestCase):
    """Test JSON parsing from model output."""

    def test_parse_fenced_json(self):
        text = '```json\n{"reinforcements": {"Alaska": 3, "Kamchatka": 2}}\n```'
        result = _parse_reinforcements(text)
        self.assertEqual(result, {"Alaska": 3, "Kamchatka": 2})

    def test_parse_raw_json(self):
        text = '{"reinforcements": {"Brazil": 5}}'
        result = _parse_reinforcements(text)
        self.assertEqual(result, {"Brazil": 5})

    def test_parse_json_with_surrounding_text(self):
        text = 'I think we should reinforce:\n```json\n{"reinforcements": {"Egypt": 2}}\n```\nGood luck!'
        result = _parse_reinforcements(text)
        self.assertEqual(result, {"Egypt": 2})

    def test_parse_returns_none_for_garbage(self):
        result = _parse_reinforcements("no json here at all")
        self.assertIsNone(result)

    def test_parse_returns_none_for_missing_key(self):
        text = '```json\n{"attacks": []}\n```'
        result = _parse_reinforcements(text)
        self.assertIsNone(result)

    def test_parse_returns_none_for_non_dict_value(self):
        text = '```json\n{"reinforcements": "Alaska"}\n```'
        result = _parse_reinforcements(text)
        self.assertIsNone(result)

    def test_parse_converts_to_int(self):
        text = '```json\n{"reinforcements": {"Alaska": 3.0}}\n```'
        result = _parse_reinforcements(text)
        self.assertEqual(result, {"Alaska": 3})

    def test_parse_empty_reinforcements(self):
        text = '```json\n{"reinforcements": {}}\n```'
        result = _parse_reinforcements(text)
        self.assertEqual(result, {})

    def test_parse_multiple_territories(self):
        text = '```json\n{"reinforcements": {"Alaska": 1, "Kamchatka": 2, "Brazil": 3}}\n```'
        result = _parse_reinforcements(text)
        self.assertEqual(result, {"Alaska": 1, "Kamchatka": 2, "Brazil": 3})


# ── _validate_reinforcements tests ───────────────────────────────────


class TestValidateReinforcements(unittest.TestCase):
    """Test validation of parsed reinforcements."""

    def setUp(self):
        self.game = make_game()
        self.player = self.game.players['RED']
        self.owned = get_owned_names(self.player)

    def test_valid_placement(self):
        """Valid placement: one owned territory, correct sum."""
        placements = {self.owned[0]: 5}
        result = _validate_reinforcements(placements, self.player, 5)
        self.assertEqual(result, placements)

    def test_valid_split_placement(self):
        """Valid split across two owned territories."""
        placements = {self.owned[0]: 3, self.owned[1]: 2}
        result = _validate_reinforcements(placements, self.player, 5)
        self.assertEqual(result, placements)

    def test_wrong_sum_returns_none(self):
        placements = {self.owned[0]: 3}
        result = _validate_reinforcements(placements, self.player, 5)
        self.assertIsNone(result)

    def test_enemy_territory_returns_none(self):
        """Placing on territory not owned by player should fail."""
        enemy_names = [t.name for t in self.game.world.territories.values()
                       if t.owner != self.player]
        placements = {enemy_names[0]: 5}
        result = _validate_reinforcements(placements, self.player, 5)
        self.assertIsNone(result)

    def test_negative_count_returns_none(self):
        placements = {self.owned[0]: -1, self.owned[1]: 6}
        result = _validate_reinforcements(placements, self.player, 5)
        self.assertIsNone(result)

    def test_zero_count_returns_none(self):
        placements = {self.owned[0]: 0, self.owned[1]: 5}
        result = _validate_reinforcements(placements, self.player, 5)
        self.assertIsNone(result)

    def test_nonexistent_territory_returns_none(self):
        placements = {"Atlantis": 5}
        result = _validate_reinforcements(placements, self.player, 5)
        self.assertIsNone(result)


# ── _fallback_reinforcements tests ───────────────────────────────────


class TestFallbackReinforcements(unittest.TestCase):
    """Test fallback distribution of reinforcements."""

    def setUp(self):
        self.game = make_game()
        self.player = self.game.players['RED']

    def test_returns_dict(self):
        result = _fallback_reinforcements(self.player, 5)
        self.assertIsInstance(result, dict)

    def test_sum_equals_available(self):
        result = _fallback_reinforcements(self.player, 5)
        self.assertEqual(sum(result.values()), 5)

    def test_all_territories_owned(self):
        owned_names = {t.name for t in self.player.territories}
        result = _fallback_reinforcements(self.player, 5)
        for name in result:
            self.assertIn(name, owned_names)

    def test_prefers_border_territories(self):
        """All territories in fallback should be border territories if any exist."""
        border_names = {t.name for t in self.player.territories if t.border}
        result = _fallback_reinforcements(self.player, 5)
        if border_names:
            for name in result:
                self.assertIn(name, border_names)

    def test_all_values_positive(self):
        result = _fallback_reinforcements(self.player, 5)
        for v in result.values():
            self.assertGreater(v, 0)

    def test_deterministic(self):
        """Same input should give same output."""
        r1 = _fallback_reinforcements(self.player, 5)
        r2 = _fallback_reinforcements(self.player, 5)
        self.assertEqual(r1, r2)

    def test_large_reinforcement(self):
        result = _fallback_reinforcements(self.player, 20)
        self.assertEqual(sum(result.values()), 20)

    def test_single_troop(self):
        result = _fallback_reinforcements(self.player, 1)
        self.assertEqual(sum(result.values()), 1)
        self.assertEqual(len(result), 1)


# ── decide_reinforcements integration tests ──────────────────────────


class TestDecideReinforcementsReturn(unittest.TestCase):
    """Test the full decide_reinforcements function."""

    def setUp(self):
        self.game = make_game()
        self.player = self.game.players['RED']
        self.owned = get_owned_names(self.player)

    def _make_state(self, model):
        return {
            "game": self.game,
            "player": self.player,
            "model": model,
            "board_summary": "Test board summary",
            "reinforcements_available": 5,
        }

    def test_returns_dict(self):
        model = MockModelBackend(responses=[
            f'{{"reinforcements": {{"{self.owned[0]}": 5}}}}'
        ])
        result = decide_reinforcements(self._make_state(model))
        self.assertIsInstance(result, dict)

    def test_has_required_keys(self):
        model = MockModelBackend(responses=[
            f'{{"reinforcements": {{"{self.owned[0]}": 5}}}}'
        ])
        result = decide_reinforcements(self._make_state(model))
        self.assertIn("reinforcement_decision", result)
        self.assertIn("reinforcement_raw", result)

    def test_valid_model_output_is_used(self):
        model = MockModelBackend(responses=[
            f'```json\n{{"reinforcements": {{"{self.owned[0]}": 3, "{self.owned[1]}": 2}}}}\n```'
        ])
        result = decide_reinforcements(self._make_state(model))
        decision = result["reinforcement_decision"]
        self.assertEqual(decision[self.owned[0]], 3)
        self.assertEqual(decision[self.owned[1]], 2)

    def test_decision_sum_equals_available(self):
        model = MockModelBackend(responses=[
            f'{{"reinforcements": {{"{self.owned[0]}": 5}}}}'
        ])
        result = decide_reinforcements(self._make_state(model))
        self.assertEqual(sum(result["reinforcement_decision"].values()), 5)

    def test_invalid_output_triggers_fallback(self):
        model = MockModelBackend(responses=["I have no idea what to do"])
        result = decide_reinforcements(self._make_state(model))
        decision = result["reinforcement_decision"]
        # Fallback should still produce valid placements
        self.assertEqual(sum(decision.values()), 5)
        owned_names = {t.name for t in self.player.territories}
        for name in decision:
            self.assertIn(name, owned_names)

    def test_wrong_sum_triggers_fallback(self):
        model = MockModelBackend(responses=[
            f'{{"reinforcements": {{"{self.owned[0]}": 99}}}}'
        ])
        result = decide_reinforcements(self._make_state(model))
        self.assertEqual(sum(result["reinforcement_decision"].values()), 5)

    def test_enemy_territory_triggers_fallback(self):
        enemy = [t.name for t in self.game.world.territories.values()
                 if t.owner != self.player][0]
        model = MockModelBackend(responses=[
            f'{{"reinforcements": {{"{enemy}": 5}}}}'
        ])
        result = decide_reinforcements(self._make_state(model))
        owned_names = {t.name for t in self.player.territories}
        for name in result["reinforcement_decision"]:
            self.assertIn(name, owned_names)

    def test_raw_output_is_string(self):
        model = MockModelBackend(responses=["anything"])
        result = decide_reinforcements(self._make_state(model))
        self.assertIsInstance(result["reinforcement_raw"], str)


# ── Prompt construction tests ────────────────────────────────────────


class TestPromptConstruction(unittest.TestCase):
    """Test that the prompt sent to the model contains expected info."""

    def setUp(self):
        self.game = make_game()
        self.player = self.game.players['RED']

    def test_prompt_contains_board_summary(self):
        model = MockModelBackend(responses=["garbage"])
        state = {
            "game": self.game,
            "player": self.player,
            "model": model,
            "board_summary": "UNIQUE_BOARD_MARKER",
            "reinforcements_available": 5,
        }
        decide_reinforcements(state)
        prompt = model.call_log[0]["prompt"]
        self.assertIn("UNIQUE_BOARD_MARKER", prompt)

    def test_prompt_contains_troop_count(self):
        model = MockModelBackend(responses=["garbage"])
        state = {
            "game": self.game,
            "player": self.player,
            "model": model,
            "board_summary": "summary",
            "reinforcements_available": 7,
        }
        decide_reinforcements(state)
        prompt = model.call_log[0]["prompt"]
        self.assertIn("7", prompt)

    def test_prompt_contains_tool_instructions(self):
        model = MockModelBackend(responses=["garbage"])
        state = {
            "game": self.game,
            "player": self.player,
            "model": model,
            "board_summary": "summary",
            "reinforcements_available": 5,
        }
        decide_reinforcements(state)
        prompt = model.call_log[0]["prompt"]
        self.assertIn("battle_sim", prompt)
        self.assertIn("threat_analyzer", prompt)
        self.assertIn("tool_call", prompt)


# ── Tool loop integration tests ──────────────────────────────────────


class TestToolLoopIntegration(unittest.TestCase):
    """Test that tool calls in model output are processed."""

    def setUp(self):
        self.game = make_game()
        self.player = self.game.players['RED']
        self.owned = get_owned_names(self.player)

    def test_tool_call_triggers_second_model_call(self):
        """If model outputs a tool call, it should be called twice."""
        responses = [
            '<tool_call>battle_sim(attacking=5, defending=3)</tool_call>',
            f'{{"reinforcements": {{"{self.owned[0]}": 5}}}}',
        ]
        model = MockModelBackend(responses=responses)
        state = {
            "game": self.game,
            "player": self.player,
            "model": model,
            "board_summary": "summary",
            "reinforcements_available": 5,
        }
        decide_reinforcements(state)
        self.assertEqual(len(model.call_log), 2)

    def test_no_tool_call_means_single_model_call(self):
        model = MockModelBackend(responses=[
            f'{{"reinforcements": {{"{self.owned[0]}": 5}}}}'
        ])
        state = {
            "game": self.game,
            "player": self.player,
            "model": model,
            "board_summary": "summary",
            "reinforcements_available": 5,
        }
        decide_reinforcements(state)
        self.assertEqual(len(model.call_log), 1)


if __name__ == '__main__':
    unittest.main()
