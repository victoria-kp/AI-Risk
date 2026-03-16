"""Monte Carlo simulation of Risk combat.

Wraps pyrisk's AI.simulate() to reuse the engine's own dice logic and returns
win probability and expected remaining troops for both sides.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'pyrisk_vendor'))

from ai import AI


def simulate_battle(attacking, defending, num_simulations=1000, **kwargs):
    """Simulate combat between attacking and defending armies.

    Args:
        attacking: number of troops in attacking territory
        defending: number of troops in defending territory
        num_simulations: number of Monte Carlo runs

    Returns:
        {
            "win_probability": float,
            "expected_attacker_remaining": float,
            "expected_defender_remaining": float,
        }
    """
    # Clear cache so num_simulations is respected
    AI._sim_cache.pop((attacking, defending), None)

    win_prob, avg_atk_survive, avg_def_survive = AI.simulate(
        attacking, defending, tests=num_simulations
    )

    return {
        "win_probability": round(win_prob, 3),
        "expected_attacker_remaining": round(avg_atk_survive, 1),
        "expected_defender_remaining": round(avg_def_survive, 1),
    }
