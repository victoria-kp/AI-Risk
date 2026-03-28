"""Unit tests for training/reward.py

Covers all components of the reward function:
  - _score_reinforcements
  - _score_attacks (including list-valued target edge case)
  - _score_movement (including list-valued src/target edge case)
  - _score_tool_appropriateness
  - _score_efficiency
  - _score_partial_credit
  - compute_reward (integration)
  - _extract_json helper
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from training.reward import (
    compute_reward,
    _score_format_and_quality,
    _score_reinforcements,
    _score_attacks,
    _score_movement,
    _score_tool_appropriateness,
    _score_efficiency,
    _score_partial_credit,
    _extract_json,
    _can_attack,
    W_QUALITY, W_TOOL_APPROPRIATENESS, W_EFFICIENCY, W_PARTIAL, WIN_BONUS,
)


# ── Shared test fixtures ──────────────────────────────────────────────

def _base_snapshot():
    """Minimal board snapshot for testing."""
    return {
        "player_name": "RED",
        "owned_territories": ["Alaska", "Kamchatka", "Japan", "India"],
        "border_territories": ["Alaska", "Kamchatka"],
        "territory_map": {
            "Alaska": {
                "owner": "RED", "forces": 5,
                "adjacent": ["Kamchatka", "Northwest Territory"],
            },
            "Kamchatka": {
                "owner": "RED", "forces": 8,
                "adjacent": ["Alaska", "Japan", "Mongolia"],
            },
            "Japan": {
                "owner": "RED", "forces": 3,
                "adjacent": ["Kamchatka", "Mongolia"],
            },
            "India": {
                "owner": "RED", "forces": 2,
                "adjacent": ["Japan", "Middle East"],
            },
            "Northwest Territory": {
                "owner": "BLUE", "forces": 4,
                "adjacent": ["Alaska", "Greenland"],
            },
            "Mongolia": {
                "owner": "BLUE", "forces": 2,
                "adjacent": ["Kamchatka", "Japan"],
            },
            "Middle East": {
                "owner": "BLUE", "forces": 3,
                "adjacent": ["India"],
            },
        },
        "reinforcements": 5,
        "players": {"RED": {}, "BLUE": {}},
        "turn": 3,
    }


def _tool_log(tools):
    """Build a quick tool log from tool name list."""
    return [{"tool_name": t, "kwargs": {}, "result": "ok"} for t in tools]


# ── Test _extract_json ─────────────────────────────────────────────────

class TestExtractJson(unittest.TestCase):

    def test_fenced_json(self):
        text = 'Here is my plan:\n```json\n{"attacks": []}\n```'
        result = _extract_json(text, "attacks")
        self.assertEqual(result, {"attacks": []})

    def test_bare_json(self):
        text = '{"reinforcements": {"Alaska": 3}}'
        result = _extract_json(text, "reinforcements")
        self.assertIn("reinforcements", result)

    def test_missing_key(self):
        text = '{"foo": "bar"}'
        self.assertIsNone(_extract_json(text, "attacks"))

    def test_empty_text(self):
        self.assertIsNone(_extract_json("", "attacks"))
        self.assertIsNone(_extract_json(None, "attacks"))

    def test_malformed_json(self):
        self.assertIsNone(_extract_json("{bad json", "attacks"))

    def test_regex_fallback(self):
        text = 'My decision: {"movement": {"src": "A", "target": "B", "count": 2}} end'
        result = _extract_json(text, "movement")
        self.assertIsNotNone(result)
        self.assertIn("movement", result)


# ── Test _score_reinforcements ─────────────────────────────────────────

class TestScoreReinforcements(unittest.TestCase):

    def setUp(self):
        self.snap = _base_snapshot()

    def test_valid_border_placement(self):
        """All troops on border territories should get near-perfect score."""
        completion = '```json\n{"reinforcements": {"Alaska": 2, "Kamchatka": 3}}\n```'
        score = _score_reinforcements(completion, self.snap)
        # 0.20 (json) + 0.20 (all owned) + 0.10 (sum=5) + 0.50 (all border)
        self.assertAlmostEqual(score, 1.0, places=2)

    def test_valid_non_border_placement(self):
        """Troops on owned but inland territories: lower strategic score."""
        completion = '```json\n{"reinforcements": {"Japan": 3, "India": 2}}\n```'
        score = _score_reinforcements(completion, self.snap)
        # 0.20 + 0.20 + 0.10 + 0.0 (none on border)
        self.assertAlmostEqual(score, 0.50, places=2)

    def test_wrong_sum(self):
        """Wrong troop sum loses the +0.10 sum bonus but keeps other points."""
        completion = '```json\n{"reinforcements": {"Alaska": 10}}\n```'
        score = _score_reinforcements(completion, self.snap)
        # 0.20 (json) + 0.20 (owned) + 0.00 (bad sum) + 0.50 (all on border)
        self.assertAlmostEqual(score, 0.90, places=2)
        # Verify it is less than a correct-sum version
        correct = '```json\n{"reinforcements": {"Alaska": 5}}\n```'
        score_correct = _score_reinforcements(correct, self.snap)
        self.assertGreater(score_correct, score)

    def test_invalid_territory(self):
        """Territory not owned should reduce validity score."""
        completion = '```json\n{"reinforcements": {"Mongolia": 5}}\n```'
        score = _score_reinforcements(completion, self.snap)
        # Mongolia is not owned by RED
        self.assertLess(score, 0.5)

    def test_garbage_input(self):
        score = _score_reinforcements("not json at all", self.snap)
        self.assertEqual(score, 0.0)

    def test_empty_reinforcements(self):
        completion = '```json\n{"reinforcements": {}}\n```'
        score = _score_reinforcements(completion, self.snap)
        self.assertEqual(score, 0.0)


# ── Test _score_attacks ────────────────────────────────────────────────

class TestScoreAttacks(unittest.TestCase):

    def setUp(self):
        self.snap = _base_snapshot()

    def test_valid_favorable_attack(self):
        """Kamchatka (8) attacks Mongolia (2) — favorable odds."""
        completion = '```json\n{"attacks": [{"src": "Kamchatka", "target": "Mongolia"}]}\n```'
        score = _score_attacks(completion, self.snap)
        # 0.20 (json) + 0.20 (valid) + 0.30 (favorable) + 0.30 (has attacks)
        self.assertAlmostEqual(score, 1.0, places=2)

    def test_empty_attacks_when_can_attack(self):
        """Empty attacks list when attacks are possible — passive but valid."""
        completion = '```json\n{"attacks": []}\n```'
        score = _score_attacks(completion, self.snap)
        # 0.20 (json) + 0.05 (Round 3: penalize skipping when attacks available)
        self.assertAlmostEqual(score, 0.25, places=2)

    def test_skip_attack_no_options(self):
        """Empty attacks when no attack options exist — correct decision."""
        snap = _base_snapshot()
        # Set all forces to 1 so no territory can attack
        for t in snap["territory_map"]:
            if snap["territory_map"][t]["owner"] == "RED":
                snap["territory_map"][t]["forces"] = 1
        completion = '```json\n{"attacks": []}\n```'
        score = _score_attacks(completion, snap)
        # 0.20 + 0.80
        self.assertAlmostEqual(score, 1.0, places=2)

    def test_invalid_src_territory(self):
        """Attack from territory not owned."""
        completion = '```json\n{"attacks": [{"src": "Mongolia", "target": "Kamchatka"}]}\n```'
        score = _score_attacks(completion, self.snap)
        # Only 0.20 for valid JSON, 0.0 for validity (0 valid attacks)
        self.assertAlmostEqual(score, 0.20, places=2)

    def test_garbage_input(self):
        score = _score_attacks("random gibberish", self.snap)
        self.assertEqual(score, 0.0)

    # ── Edge case: target is a list instead of a string ──────────────

    def test_target_is_list_does_not_crash(self):
        """When target is a list like ['India', 'Japan'], the function
        must not crash — it should skip that attack gracefully."""
        completion = '```json\n{"attacks": [{"src": "Kamchatka", "target": ["India", "Japan"]}]}\n```'
        # Must not raise
        score = _score_attacks(completion, self.snap)
        self.assertIsInstance(score, float)
        # The attack with list target should be skipped as invalid,
        # so validity fraction is 0/1 = 0
        # Score = 0.20 (json) + 0.20 * 0 = 0.20
        self.assertAlmostEqual(score, 0.20, places=2)

    def test_src_is_list_does_not_crash(self):
        """When src is a list, function must not crash."""
        completion = '```json\n{"attacks": [{"src": ["Alaska", "Kamchatka"], "target": "Mongolia"}]}\n```'
        score = _score_attacks(completion, self.snap)
        self.assertIsInstance(score, float)
        self.assertAlmostEqual(score, 0.20, places=2)

    def test_mixed_valid_and_list_attacks(self):
        """One valid attack and one with list target — only valid one scored."""
        completion = ('```json\n{"attacks": ['
                      '{"src": "Kamchatka", "target": "Mongolia"}, '
                      '{"src": "Alaska", "target": ["India", "Japan"]}'
                      ']}\n```')
        score = _score_attacks(completion, self.snap)
        self.assertIsInstance(score, float)
        # 1 out of 2 attacks is valid
        # 0.20 (json) + 0.20 * 0.5 + 0.30 * 1.0 (favorable) + 0.30 (has valid)
        self.assertGreater(score, 0.20)


# ── Test _score_movement ───────────────────────────────────────────────

class TestScoreMovement(unittest.TestCase):

    def setUp(self):
        self.snap = _base_snapshot()

    def test_inland_to_border_movement(self):
        """Japan (inland) -> Kamchatka (border) — excellent move."""
        completion = '```json\n{"movement": {"src": "Japan", "target": "Kamchatka", "count": 2}}\n```'
        score = _score_movement(completion, self.snap)
        # 0.20 (json) + 0.20 (valid) + 0.60 (inland->border)
        self.assertAlmostEqual(score, 1.0, places=2)

    def test_null_movement(self):
        """Null movement (skip) — reasonable default."""
        completion = '```json\n{"movement": null}\n```'
        score = _score_movement(completion, self.snap)
        # 0.20 + 0.10 (Round 3: penalize skipping when useful moves exist)
        self.assertAlmostEqual(score, 0.30, places=2)

    def test_border_to_inland(self):
        """Kamchatka (border) -> Japan (inland) — bad move."""
        completion = '```json\n{"movement": {"src": "Kamchatka", "target": "Japan", "count": 2}}\n```'
        score = _score_movement(completion, self.snap)
        # 0.20 + 0.20 + 0.05
        self.assertAlmostEqual(score, 0.45, places=2)

    def test_not_adjacent(self):
        """Alaska -> Japan — not adjacent, should fail validity."""
        completion = '```json\n{"movement": {"src": "Alaska", "target": "Japan", "count": 1}}\n```'
        score = _score_movement(completion, self.snap)
        # Only 0.20 for JSON
        self.assertAlmostEqual(score, 0.20, places=2)

    def test_count_too_high(self):
        """Moving all forces (count >= src_forces) is invalid."""
        completion = '```json\n{"movement": {"src": "Japan", "target": "Kamchatka", "count": 3}}\n```'
        score = _score_movement(completion, self.snap)
        # Japan has 3 forces; count=3 means count >= forces -> invalid
        self.assertAlmostEqual(score, 0.20, places=2)

    def test_garbage_input(self):
        score = _score_movement("whatever", self.snap)
        self.assertEqual(score, 0.0)

    # ── Edge case: src or target is a list instead of a string ────────

    def test_target_is_list_does_not_crash(self):
        """When movement target is a list, function must not crash."""
        completion = '```json\n{"movement": {"src": "Japan", "target": ["Kamchatka", "India"], "count": 1}}\n```'
        score = _score_movement(completion, self.snap)
        self.assertIsInstance(score, float)
        # Should return 0.20 (only JSON valid, type check catches list)
        self.assertAlmostEqual(score, 0.20, places=2)

    def test_src_is_list_does_not_crash(self):
        """When movement src is a list, function must not crash."""
        completion = '```json\n{"movement": {"src": ["Japan", "India"], "target": "Kamchatka", "count": 1}}\n```'
        score = _score_movement(completion, self.snap)
        self.assertIsInstance(score, float)
        self.assertAlmostEqual(score, 0.20, places=2)

    def test_both_src_and_target_are_lists(self):
        """When both src and target are lists, function must not crash."""
        completion = '```json\n{"movement": {"src": ["Japan"], "target": ["Kamchatka"], "count": 1}}\n```'
        score = _score_movement(completion, self.snap)
        self.assertIsInstance(score, float)
        self.assertAlmostEqual(score, 0.20, places=2)


# ── Test _score_tool_appropriateness ───────────────────────────────────

class TestToolAppropriateness(unittest.TestCase):

    def test_preferred_tool_for_attacks(self):
        log = _tool_log(["battle_sim"])
        self.assertAlmostEqual(_score_tool_appropriateness(log, "attacks"), 1.0)

    def test_non_preferred_but_valid(self):
        log = _tool_log(["position_evaluator"])
        score = _score_tool_appropriateness(log, "attacks")
        self.assertAlmostEqual(score, 0.7)

    def test_no_tools(self):
        self.assertAlmostEqual(_score_tool_appropriateness([], "attacks"), 0.3)

    def test_all_errors(self):
        log = [{"tool_name": "unknown_thing", "kwargs": {}, "result": "Error: not found"}]
        self.assertAlmostEqual(_score_tool_appropriateness(log, "attacks"), 0.1)

    def test_preferred_with_error(self):
        log = [{"tool_name": "battle_sim", "kwargs": {}, "result": "Error: timeout"}]
        self.assertAlmostEqual(_score_tool_appropriateness(log, "attacks"), 0.8)


# ── Test _score_efficiency ─────────────────────────────────────────────

class TestEfficiency(unittest.TestCase):

    def test_zero_calls(self):
        self.assertAlmostEqual(_score_efficiency([]), 0.3)

    def test_one_call(self):
        self.assertAlmostEqual(_score_efficiency(_tool_log(["a"])), 1.0)

    def test_two_calls(self):
        self.assertAlmostEqual(_score_efficiency(_tool_log(["a", "b"])), 0.9)

    def test_three_plus_calls(self):
        self.assertAlmostEqual(_score_efficiency(_tool_log(["a", "b", "c"])), 0.5)


# ── Test _score_partial_credit ─────────────────────────────────────────

class TestPartialCredit(unittest.TestCase):

    def setUp(self):
        self.snap = _base_snapshot()

    def test_empty_completion(self):
        self.assertEqual(_score_partial_credit("", "attacks", self.snap), 0.0)

    def test_tool_call_attempt(self):
        text = "<tool_call>battle_sim</tool_call>"
        score = _score_partial_credit(text, "attacks", self.snap)
        self.assertGreater(score, 0.5)

    def test_mentions_territories(self):
        text = "I will reinforce Alaska and Kamchatka"
        score = _score_partial_credit(text, "reinforcements", self.snap)
        self.assertGreater(score, 0.0)

    def test_conciseness_bonus(self):
        short = "plan"
        long = "x" * 2000
        s_short = _score_partial_credit(short, "attacks", self.snap)
        s_long = _score_partial_credit(long, "attacks", self.snap)
        self.assertGreater(s_short, s_long)


# ── Test compute_reward (integration) ─────────────────────────────────

class TestComputeReward(unittest.TestCase):

    def setUp(self):
        self.snap = _base_snapshot()

    def test_perfect_reinforcement(self):
        completion = '```json\n{"reinforcements": {"Alaska": 2, "Kamchatka": 3}}\n```'
        tool_log = _tool_log(["threat_analyzer"])
        reward = compute_reward(completion, "reinforcements", self.snap, tool_log)
        self.assertGreater(reward, 0.7)
        self.assertLessEqual(reward, 1.0)

    def test_win_bonus(self):
        completion = '```json\n{"reinforcements": {"Alaska": 2, "Kamchatka": 3}}\n```'
        tool_log = _tool_log(["threat_analyzer"])
        r_loss = compute_reward(completion, "reinforcements", self.snap, tool_log, outcome="loss")
        r_win = compute_reward(completion, "reinforcements", self.snap, tool_log, outcome="win")
        self.assertAlmostEqual(r_win - r_loss, WIN_BONUS, places=2)

    def test_reward_bounded(self):
        """Reward must always be in [0, 1]."""
        for phase in ("reinforcements", "attacks", "movement"):
            for text in ("", "garbage", '{"attacks": []}'):
                r = compute_reward(text, phase, self.snap, [])
                self.assertGreaterEqual(r, 0.0)
                self.assertLessEqual(r, 1.0)

    def test_unknown_phase(self):
        """Unknown phase should not crash, returns low-but-nonzero reward."""
        r = compute_reward("test", "unknown_phase", self.snap, [])
        self.assertGreaterEqual(r, 0.0)
        self.assertLessEqual(r, 1.0)


# ── Test _can_attack helper ────────────────────────────────────────────

class TestCanAttack(unittest.TestCase):

    def test_can_attack_true(self):
        snap = _base_snapshot()
        owned = set(snap["owned_territories"])
        self.assertTrue(_can_attack(owned, snap["territory_map"], "RED"))

    def test_can_attack_false_all_one_force(self):
        snap = _base_snapshot()
        for t in snap["territory_map"]:
            if snap["territory_map"][t]["owner"] == "RED":
                snap["territory_map"][t]["forces"] = 1
        owned = set(snap["owned_territories"])
        self.assertFalse(_can_attack(owned, snap["territory_map"], "RED"))


if __name__ == "__main__":
    unittest.main()
