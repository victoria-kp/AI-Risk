"""Unit tests for tools/battle_sim.py"""

import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'pyrisk_vendor'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tools.battle_sim import simulate_battle


class TestOutputStructure(unittest.TestCase):
    """Tests for the return value format."""

    def test_returns_dict(self):
        result = simulate_battle(5, 3)
        self.assertIsInstance(result, dict)

    def test_has_all_keys(self):
        result = simulate_battle(5, 3)
        expected = {
            'win_probability',
            'expected_attacker_remaining',
            'expected_defender_remaining',
        }
        self.assertEqual(set(result.keys()), expected)

    def test_all_values_are_floats(self):
        result = simulate_battle(5, 3)
        for key, val in result.items():
            self.assertIsInstance(val, float, f'{key} is not a float')

    def test_win_probability_between_0_and_1(self):
        result = simulate_battle(5, 3)
        self.assertGreaterEqual(result['win_probability'], 0.0)
        self.assertLessEqual(result['win_probability'], 1.0)

    def test_remaining_troops_non_negative(self):
        result = simulate_battle(5, 3)
        self.assertGreaterEqual(result['expected_attacker_remaining'], 0.0)
        self.assertGreaterEqual(result['expected_defender_remaining'], 0.0)


class TestObviousOutcomes(unittest.TestCase):
    """Tests for matchups with known outcomes."""

    def test_overwhelming_attacker_wins(self):
        result = simulate_battle(10, 1, num_simulations=2000)
        self.assertGreater(result['win_probability'], 0.95)

    def test_single_attacker_cannot_win(self):
        """1 attacker must keep 1 behind, so 0 dice rolled — always loses."""
        result = simulate_battle(1, 10)
        self.assertEqual(result['win_probability'], 0.0)

    def test_single_attacker_vs_single_defender(self):
        """1 vs 1: attacker can never attack (must keep 1 behind)."""
        result = simulate_battle(1, 1)
        self.assertEqual(result['win_probability'], 0.0)

    def test_hopeless_attack(self):
        result = simulate_battle(2, 10, num_simulations=2000)
        self.assertLess(result['win_probability'], 0.01)

    def test_large_advantage(self):
        result = simulate_battle(20, 2, num_simulations=2000)
        self.assertGreater(result['win_probability'], 0.99)


class TestDefenderAdvantage(unittest.TestCase):
    """Defender wins ties in Risk, so equal troops should favor the defender."""

    def test_3v3(self):
        result = simulate_battle(3, 3, num_simulations=5000)
        self.assertLess(result['win_probability'], 0.50)

    def test_5v5(self):
        result = simulate_battle(5, 5, num_simulations=5000)
        self.assertLess(result['win_probability'], 0.50)

    def test_8v8(self):
        result = simulate_battle(8, 8, num_simulations=5000)
        self.assertLess(result['win_probability'], 0.50)

    def test_10v10(self):
        result = simulate_battle(10, 10, num_simulations=5000)
        self.assertLess(result['win_probability'], 0.50)


class TestMonotonicity(unittest.TestCase):
    """More attackers should mean higher win probability."""

    def test_increasing_attackers(self):
        defender = 5
        prev_prob = 0.0
        for atk in [2, 3, 5, 8, 10, 15]:
            result = simulate_battle(atk, defender, num_simulations=5000)
            prob = result['win_probability']
            self.assertGreaterEqual(
                prob, prev_prob - 0.03,
                f'{atk} vs {defender}: prob {prob} < previous {prev_prob}'
            )
            prev_prob = prob

    def test_increasing_defenders_lowers_win_rate(self):
        attacker = 8
        prev_prob = 1.0
        for dfn in [1, 3, 5, 8, 10]:
            result = simulate_battle(attacker, dfn, num_simulations=5000)
            prob = result['win_probability']
            self.assertLessEqual(
                prob, prev_prob + 0.03,
                f'{attacker} vs {dfn}: prob {prob} > previous {prev_prob}'
            )
            prev_prob = prob


class TestExpectedRemaining(unittest.TestCase):
    """Sanity checks on expected remaining troop counts."""

    def test_attacker_remaining_decreases_with_stronger_defender(self):
        prev_remaining = 10.0
        for dfn in [1, 3, 5, 7]:
            result = simulate_battle(10, dfn, num_simulations=5000)
            remaining = result['expected_attacker_remaining']
            self.assertLessEqual(
                remaining, prev_remaining + 0.5,
                f'10 vs {dfn}: remaining {remaining} > previous {prev_remaining}'
            )
            prev_remaining = remaining

    def test_defender_remaining_zero_on_certain_win(self):
        """If attacker almost always wins, defender remaining should be near 0."""
        result = simulate_battle(20, 1, num_simulations=2000)
        self.assertLess(result['expected_defender_remaining'], 0.1)

    def test_attacker_remaining_zero_on_certain_loss(self):
        """If attacker always loses, attacker remaining should be 0."""
        result = simulate_battle(1, 10)
        self.assertEqual(result['expected_attacker_remaining'], 0.0)

    def test_attacker_remaining_less_than_starting(self):
        """Surviving attackers should always be less than starting count."""
        for atk in [3, 5, 10]:
            result = simulate_battle(atk, 3, num_simulations=2000)
            self.assertLess(result['expected_attacker_remaining'], atk)


class TestNumSimulations(unittest.TestCase):
    """Tests for the num_simulations parameter."""

    def test_different_simulation_counts(self):
        """Should work with various simulation counts."""
        for n in [10, 100, 1000]:
            result = simulate_battle(5, 3, num_simulations=n)
            self.assertIn('win_probability', result)

    def test_higher_sims_more_stable(self):
        """Higher simulation counts should produce less variance."""
        spreads = {}
        for n_sims in [100, 5000]:
            probs = []
            for _ in range(10):
                r = simulate_battle(5, 3, num_simulations=n_sims)
                probs.append(r['win_probability'])
            spreads[n_sims] = max(probs) - min(probs)
        self.assertLess(
            spreads[5000], spreads[100] + 0.01,
            'Higher sim count should have less spread'
        )

    def test_default_simulations(self):
        """Should work without specifying num_simulations."""
        result = simulate_battle(5, 3)
        self.assertIn('win_probability', result)


class TestKwargsIgnored(unittest.TestCase):
    """The function accepts **kwargs for compatibility with the tool interface."""

    def test_extra_kwargs_ignored(self):
        result = simulate_battle(5, 3, game=None, player=None)
        self.assertIn('win_probability', result)


if __name__ == '__main__':
    unittest.main()
