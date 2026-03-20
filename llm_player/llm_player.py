"""pyrisk AI wrapper that uses the LangGraph pipeline.

Subclass of pyrisk's AI class. The full LangGraph pipeline runs once
during reinforce() — computing reinforcement, attack, and movement
decisions from the current board state. attack() and freemove() read
cached results from that single pipeline invocation.

initial_placement() uses the LLM via decide_placement (separate from
the per-turn pipeline, since it's called many times during setup).

Methods:
- start(): initialize ModelBackend
- initial_placement(): LLM-driven territory selection
- reinforce(available): run pipeline, return {territory: count} dict
- attack(): yield (src, target, None, None) tuples from cached decisions
- freemove(): return (src, target, count) or None from cached decision
"""

from ai import AI
from llm_player.model import ModelBackend
from llm_player.graph import run_turn
from llm_player.nodes.decide_placement import decide_placement


class LLMPlayer(AI):
    """AI player that uses an LLM via LangGraph pipeline."""

    def start(self):
        """Initialize the model backend."""
        self.model = ModelBackend()
        self._cached_result = None

    def initial_placement(self, empty, remaining):
        """Use the LLM to pick a territory to claim or reinforce."""
        result = decide_placement({
            "game": self.game,
            "player": self.player,
            "model": self.model,
            "empty": empty,
            "remaining": remaining,
        })
        return result["placement_decision"]

    def reinforce(self, available):
        """Run the full LangGraph pipeline and return reinforcement decisions."""
        self._cached_result = run_turn(
            self.game, self.player, self.model,
            reinforcements_available=available,
        )
        return self._cached_result["reinforcement_decision"]

    def attack(self):
        """Yield attack tuples from cached pipeline results."""
        if self._cached_result is None:
            return
        for attack in self._cached_result.get("attack_decisions", []):
            yield (attack["src"], attack["target"], None, None)

    def freemove(self):
        """Return freemove tuple from cached pipeline results."""
        if self._cached_result is None:
            return None
        decision = self._cached_result.get("movement_decision")
        if decision is None:
            return None
        return (decision["src"], decision["target"], decision["count"])
