"""LLMPlayer subclass that logs every decision for data collection.

Used by both data/run_benchmark.py (off-policy Gemini data) and
data/collect_on_policy.py (on-policy trained model data).
"""

from llm_player.llm_player import LLMPlayer
from llm_player.nodes.decide_placement import decide_placement
from llm_player.graph import run_turn


class BenchmarkLLMPlayer(LLMPlayer):
    """LLMPlayer that records every decision for logging."""

    def start(self):
        super().start()
        self.turn_log = []  # filled by the benchmark runner

    def initial_placement(self, empty, remaining):
        log_idx_before = len(self.model.call_log)
        result = decide_placement({
            "game": self.game,
            "player": self.player,
            "model": self.model,
            "empty": empty,
            "remaining": remaining,
        })
        log_idx_after = len(self.model.call_log)

        # Extract prompt/response from call_log
        calls = self.model.call_log[log_idx_before:log_idx_after]
        prompt = calls[0]["prompt"] if calls else ""
        response = calls[0]["response"] if calls else ""

        self.turn_log.append({
            "turn": self.game.turn,
            "phase": "placement",
            "prompt": prompt,
            "response": response,
            "decision": result["placement_decision"],
            "fallback": result["placement_fallback"],
            "board_snapshot": self._snapshot(),
        })
        return result["placement_decision"]

    def reinforce(self, available):
        log_idx_before = len(self.model.call_log)
        self._cached_result = run_turn(
            self.game, self.player, self.model,
            reinforcements_available=available,
        )
        log_idx_after = len(self.model.call_log)
        calls = self.model.call_log[log_idx_before:log_idx_after]

        # Log reinforcements
        reinf_calls = [c for c in calls if c["caller"].startswith("reinforcements")]
        self.turn_log.append({
            "turn": self.game.turn,
            "phase": "reinforcements",
            "prompt": reinf_calls[0]["prompt"] if reinf_calls else "",
            "response": reinf_calls[-1]["response"] if reinf_calls else "",
            "decision": self._cached_result["reinforcement_decision"],
            "fallback": self._cached_result.get("reinforcement_fallback", False),
            "board_snapshot": self._snapshot(),
        })

        # Log attacks
        atk_calls = [c for c in calls if c["caller"].startswith("attacks")]
        self.turn_log.append({
            "turn": self.game.turn,
            "phase": "attacks",
            "prompt": atk_calls[0]["prompt"] if atk_calls else "",
            "response": atk_calls[-1]["response"] if atk_calls else "",
            "decision": self._cached_result["attack_decisions"],
            "fallback": self._cached_result.get("attack_fallback", False),
            "board_snapshot": self._snapshot(),
        })

        # Log movement
        mov_calls = [c for c in calls if c["caller"].startswith("movement")]
        self.turn_log.append({
            "turn": self.game.turn,
            "phase": "movement",
            "prompt": mov_calls[0]["prompt"] if mov_calls else "",
            "response": mov_calls[-1]["response"] if mov_calls else "",
            "decision": self._cached_result.get("movement_decision"),
            "fallback": self._cached_result.get("movement_fallback", False),
            "board_snapshot": self._snapshot(),
        })

        return self._cached_result["reinforcement_decision"]

    def _snapshot(self):
        """Full board snapshot for KPIs and GRPO training.

        Captures complete game state so the reward function can
        validate decisions (owned territories, adjacency, forces).
        """
        my_territories = list(self.player.territories)
        my_territory_names = [t.name for t in my_territories]
        border_territories = [t.name for t in my_territories if t.border]

        # Full territory map: every territory with owner, forces, continent
        territory_map = {}
        for t in self.game.world.territories.values():
            territory_map[t.name] = {
                "owner": t.owner.name if t.owner else None,
                "forces": t.forces,
                "continent": t.area.name,
                "adjacent": [a.name for a in t.connect],
            }

        # Per-player summary
        players = {}
        for p in self.game.players.values():
            p_territories = list(p.territories)
            players[p.name] = {
                "alive": p.alive,
                "territories": len(p_territories),
                "forces": sum(t.forces for t in p_territories),
                "continents": [a.name for a in p.areas],
            }

        return {
            "player_name": self.player.name,
            "owned_territories": my_territory_names,
            "border_territories": border_territories,
            "total_forces": sum(t.forces for t in my_territories),
            "continents": [a.name for a in self.player.areas],
            "reinforcements": self.player.reinforcements,
            "territory_map": territory_map,
            "players": players,
            "turn": self.game.turn,
        }
