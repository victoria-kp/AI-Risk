"""Unit tests for llm_player/nodes/decide_attacks.py"""

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
from llm_player.nodes.decide_attacks import (
    decide_attacks,
    _parse_attacks,
    _validate_attacks,
    _fallback_attacks,
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


def find_valid_attack(game, player):
    """Find a valid (src, target) pair for the player."""
    for t in player.territories:
        if t.forces > 1:
            for adj in t.connect:
                if adj.owner != player:
                    return t.name, adj.name
    return None, None


# ── _parse_attacks tests ─────────────────────────────────────────────


class TestParseAttacks(unittest.TestCase):
    """Test JSON parsing from model output."""

    def test_parse_fenced_json(self):
        text = '```json\n{"attacks": [{"src": "Alaska", "target": "Kamchatka"}]}\n```'
        result = _parse_attacks(text)
        self.assertEqual(result, [{"src": "Alaska", "target": "Kamchatka"}])

    def test_parse_raw_json(self):
        text = '{"attacks": [{"src": "Brazil", "target": "Peru"}]}'
        result = _parse_attacks(text)
        self.assertEqual(result, [{"src": "Brazil", "target": "Peru"}])

    def test_parse_empty_attacks(self):
        text = '```json\n{"attacks": []}\n```'
        result = _parse_attacks(text)
        self.assertEqual(result, [])

    def test_parse_multiple_attacks(self):
        text = '```json\n{"attacks": [{"src": "A", "target": "B"}, {"src": "C", "target": "D"}]}\n```'
        result = _parse_attacks(text)
        self.assertEqual(len(result), 2)

    def test_parse_with_surrounding_text(self):
        text = 'I will attack!\n```json\n{"attacks": [{"src": "X", "target": "Y"}]}\n```\nDone.'
        result = _parse_attacks(text)
        self.assertEqual(result, [{"src": "X", "target": "Y"}])

    def test_parse_returns_none_for_garbage(self):
        result = _parse_attacks("no json here")
        self.assertIsNone(result)

    def test_parse_returns_none_for_wrong_key(self):
        text = '```json\n{"reinforcements": {}}\n```'
        result = _parse_attacks(text)
        self.assertIsNone(result)

    def test_parse_returns_none_for_non_list(self):
        text = '```json\n{"attacks": "Alaska"}\n```'
        result = _parse_attacks(text)
        self.assertIsNone(result)

    def test_parse_returns_none_for_missing_src(self):
        text = '```json\n{"attacks": [{"target": "Alaska"}]}\n```'
        result = _parse_attacks(text)
        self.assertIsNone(result)

    def test_parse_returns_none_for_missing_target(self):
        text = '```json\n{"attacks": [{"src": "Alaska"}]}\n```'
        result = _parse_attacks(text)
        self.assertIsNone(result)

    def test_parse_returns_none_for_non_string_src(self):
        text = '```json\n{"attacks": [{"src": 123, "target": "Alaska"}]}\n```'
        result = _parse_attacks(text)
        self.assertIsNone(result)

    def test_parse_strips_extra_keys(self):
        text = '```json\n{"attacks": [{"src": "A", "target": "B", "troops": 5}]}\n```'
        result = _parse_attacks(text)
        self.assertEqual(result, [{"src": "A", "target": "B"}])


# ── _validate_attacks tests ──────────────────────────────────────────


class TestValidateAttacks(unittest.TestCase):
    """Test validation of parsed attacks."""

    def setUp(self):
        self.game = make_game()
        self.player = self.game.players['RED']
        self.src, self.target = find_valid_attack(self.game, self.player)

    def test_valid_attack(self):
        if not self.src:
            self.skipTest("No valid attack found")
        attacks = [{"src": self.src, "target": self.target}]
        result = _validate_attacks(attacks, self.player, self.game)
        self.assertEqual(len(result), 1)

    def test_empty_list_is_valid(self):
        result = _validate_attacks([], self.player, self.game)
        self.assertEqual(result, [])

    def test_unowned_src_filtered(self):
        enemy = [t.name for t in self.game.world.territories.values()
                 if t.owner != self.player][0]
        attacks = [{"src": enemy, "target": "whatever"}]
        result = _validate_attacks(attacks, self.player, self.game)
        self.assertEqual(result, [])

    def test_src_with_one_troop_filtered(self):
        """Territories with exactly 1 troop can't attack."""
        for t in self.player.territories:
            if t.forces == 1:
                enemy_adj = [a for a in t.connect if a.owner != self.player]
                if enemy_adj:
                    attacks = [{"src": t.name, "target": enemy_adj[0].name}]
                    result = _validate_attacks(attacks, self.player, self.game)
                    self.assertEqual(result, [])
                    return
        self.skipTest("No territory with 1 troop and enemy neighbor")

    def test_friendly_target_filtered(self):
        """Can't attack your own territory."""
        owned = list(self.player.territories)
        if len(owned) >= 2:
            # Find two owned territories that are adjacent
            for t in owned:
                for adj in t.connect:
                    if adj.owner == self.player and t.forces > 1:
                        attacks = [{"src": t.name, "target": adj.name}]
                        result = _validate_attacks(attacks, self.player, self.game)
                        self.assertEqual(result, [])
                        return
        self.skipTest("No adjacent owned pair found")

    def test_non_adjacent_target_filtered(self):
        """Can't attack a non-adjacent territory."""
        if not self.src:
            self.skipTest("No valid attack found")
        src_t = self.game.world.territories[self.src]
        # Find an enemy not adjacent to src
        for t in self.game.world.territories.values():
            if t.owner != self.player and t not in src_t.connect:
                attacks = [{"src": self.src, "target": t.name}]
                result = _validate_attacks(attacks, self.player, self.game)
                self.assertEqual(result, [])
                return
        self.skipTest("No non-adjacent enemy found")

    def test_nonexistent_src_filtered(self):
        attacks = [{"src": "Atlantis", "target": "Nowhere"}]
        result = _validate_attacks(attacks, self.player, self.game)
        self.assertEqual(result, [])

    def test_mixed_valid_and_invalid(self):
        """Valid attacks kept, invalid ones filtered."""
        if not self.src:
            self.skipTest("No valid attack found")
        attacks = [
            {"src": self.src, "target": self.target},
            {"src": "Atlantis", "target": "Nowhere"},
        ]
        result = _validate_attacks(attacks, self.player, self.game)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["src"], self.src)


# ── _fallback_attacks tests ──────────────────────────────────────────


class TestFallbackAttacks(unittest.TestCase):
    """Test fallback behavior."""

    def test_returns_empty_list(self):
        result = _fallback_attacks()
        self.assertEqual(result, [])

    def test_returns_list_type(self):
        result = _fallback_attacks()
        self.assertIsInstance(result, list)


# ── decide_attacks integration tests ─────────────────────────────────


class TestDecideAttacksReturn(unittest.TestCase):
    """Test the full decide_attacks function."""

    def setUp(self):
        self.game = make_game()
        self.player = self.game.players['RED']
        self.src, self.target = find_valid_attack(self.game, self.player)

    def _make_state(self, model):
        return {
            "game": self.game,
            "player": self.player,
            "model": model,
            "board_summary": "Test board summary",
        }

    def test_returns_dict(self):
        model = MockModelBackend(responses=['{"attacks": []}'])
        result = decide_attacks(self._make_state(model))
        self.assertIsInstance(result, dict)

    def test_has_required_keys(self):
        model = MockModelBackend(responses=['{"attacks": []}'])
        result = decide_attacks(self._make_state(model))
        self.assertIn("attack_decisions", result)
        self.assertIn("attack_raw", result)

    def test_valid_attack_is_used(self):
        if not self.src:
            self.skipTest("No valid attack found")
        model = MockModelBackend(responses=[
            f'{{"attacks": [{{"src": "{self.src}", "target": "{self.target}"}}]}}'
        ])
        result = decide_attacks(self._make_state(model))
        self.assertEqual(len(result["attack_decisions"]), 1)
        self.assertEqual(result["attack_decisions"][0]["src"], self.src)

    def test_empty_attacks_accepted(self):
        model = MockModelBackend(responses=['{"attacks": []}'])
        result = decide_attacks(self._make_state(model))
        self.assertEqual(result["attack_decisions"], [])

    def test_invalid_output_triggers_fallback(self):
        model = MockModelBackend(responses=["I refuse to attack"])
        result = decide_attacks(self._make_state(model))
        self.assertEqual(result["attack_decisions"], [])

    def test_invalid_attack_filtered(self):
        model = MockModelBackend(responses=[
            '{"attacks": [{"src": "Atlantis", "target": "Nowhere"}]}'
        ])
        result = decide_attacks(self._make_state(model))
        self.assertEqual(result["attack_decisions"], [])

    def test_raw_output_is_string(self):
        model = MockModelBackend(responses=["anything"])
        result = decide_attacks(self._make_state(model))
        self.assertIsInstance(result["attack_raw"], str)


# ── Prompt construction tests ────────────────────────────────────────


class TestAttackPromptConstruction(unittest.TestCase):
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
            "board_summary": "UNIQUE_ATTACK_MARKER",
        }
        decide_attacks(state)
        prompt = model.call_log[0]["prompt"]
        self.assertIn("UNIQUE_ATTACK_MARKER", prompt)

    def test_prompt_contains_tool_instructions(self):
        model = MockModelBackend(responses=["garbage"])
        state = {
            "game": self.game,
            "player": self.player,
            "model": model,
            "board_summary": "summary",
        }
        decide_attacks(state)
        prompt = model.call_log[0]["prompt"]
        self.assertIn("battle_sim", prompt)
        self.assertIn("tool_call", prompt)

    def test_prompt_mentions_attacks(self):
        model = MockModelBackend(responses=["garbage"])
        state = {
            "game": self.game,
            "player": self.player,
            "model": model,
            "board_summary": "summary",
        }
        decide_attacks(state)
        prompt = model.call_log[0]["prompt"]
        self.assertIn("attack", prompt.lower())


# ── Tool loop integration tests ──────────────────────────────────────


class TestAttackToolLoopIntegration(unittest.TestCase):
    """Test that tool calls in model output are processed."""

    def setUp(self):
        self.game = make_game()
        self.player = self.game.players['RED']

    def test_tool_call_triggers_second_model_call(self):
        responses = [
            '<tool_call>battle_sim(attacking=5, defending=3)</tool_call>',
            '{"attacks": []}',
        ]
        model = MockModelBackend(responses=responses)
        state = {
            "game": self.game,
            "player": self.player,
            "model": model,
            "board_summary": "summary",
        }
        decide_attacks(state)
        self.assertEqual(len(model.call_log), 2)

    def test_no_tool_call_means_single_model_call(self):
        model = MockModelBackend(responses=['{"attacks": []}'])
        state = {
            "game": self.game,
            "player": self.player,
            "model": model,
            "board_summary": "summary",
        }
        decide_attacks(state)
        self.assertEqual(len(model.call_log), 1)


if __name__ == '__main__':
    unittest.main()
