"""Hybrid Risk player: BetterAI for placement/movement, LLM for reinforce/attack.

Extends BetterAI so it inherits:
  - initial_placement() — continent-priority territory selection
  - freemove() — inland-to-border troop movement

Overrides:
  - reinforce() — LLM decides where to place troops
  - attack() — LLM picks attacks from a numbered menu

Direct model.generate() calls. Falls back to BetterAI logic if LLM parsing fails.
"""

import json
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'pyrisk_vendor'))

from ai.better import BetterAI
from llm_player.model import ModelBackend
from llm_player.decision_menus import (
    build_reinforce_prompt,
    build_attack_menu,
    build_attack_prompt,
    parse_reinforcements,
    parse_attack_indices,
)

LOG = logging.getLogger("llm_player.hybrid")


class HybridPlayer(BetterAI):
    """BetterAI for placement/movement, LLM for reinforce/attack."""

    def start(self):
        """Initialize BetterAI state + model backend."""
        super().start()
        self.model = ModelBackend()
        self.turn_log = []

    def _snapshot(self) -> dict:
        """Build board_snapshot dict from live game state."""
        player = self.player
        game = self.game
        world = self.world

        owned = [t.name for t in player.territories]
        borders = [t.name for t in player.territories if t.border]
        continents = [a.name for a in player.areas]
        total_forces = sum(t.forces for t in player.territories)

        territory_map = {}
        for t in world.territories.values():
            territory_map[t.name] = {
                "owner": t.owner.name if t.owner else None,
                "forces": t.forces,
                "continent": t.area.name,
                "adjacent": [a.name for a in t.connect],
            }

        players_info = {}
        for p in game.players.values():
            players_info[p.name] = {
                "alive": p.alive,
                "territories": p.territory_count,
                "forces": p.forces,
                "continents": [a.name for a in p.areas],
            }

        return {
            "player_name": player.name,
            "owned_territories": owned,
            "border_territories": borders,
            "total_forces": total_forces,
            "continents": continents,
            "territory_map": territory_map,
            "players": players_info,
        }

    def reinforce(self, available):
        """Use LLM to decide reinforcements, fallback to BetterAI."""
        snapshot = self._snapshot()
        prompt = build_reinforce_prompt(snapshot, available)

        raw_output = self.model.generate(
            prompt, max_tokens=256, temperature=0.3, caller="reinforce"
        ) or ""

        owned = set(snapshot["owned_territories"])
        decision = parse_reinforcements(raw_output, available, owned)

        if decision is not None:
            # Check sum — if less than available, distribute remainder via BetterAI
            placed = sum(decision.values())
            if placed < available:
                remainder = available - placed
                fallback = super().reinforce(remainder)
                for t, count in fallback.items():
                    t_name = t.name if hasattr(t, 'name') else str(t)
                    decision[t_name] = decision.get(t_name, 0) + count

            self.turn_log.append({
                "phase": "reinforcements",
                "prompt": prompt,
                "response": raw_output,
                "decision": decision,
                "fallback": False,
                "snapshot": snapshot,
            })
            return decision

        # Fallback to BetterAI
        LOG.warning("Reinforce parse failed, falling back to BetterAI")
        fallback_result = super().reinforce(available)
        # Convert territory objects to names
        result = {}
        for t, count in fallback_result.items():
            t_name = t.name if hasattr(t, 'name') else str(t)
            result[t_name] = count

        self.turn_log.append({
            "phase": "reinforcements",
            "prompt": prompt,
            "response": raw_output,
            "decision": result,
            "fallback": True,
            "snapshot": snapshot,
        })
        return result

    def attack(self):
        """Use LLM to pick attacks from menu, fallback to BetterAI."""
        snapshot = self._snapshot()
        menu = build_attack_menu(snapshot)

        if not menu:
            self.turn_log.append({
                "phase": "attacks",
                "prompt": "",
                "response": "",
                "decision": [],
                "fallback": False,
                "snapshot": snapshot,
                "attack_menu": [],
            })
            return

        prompt = build_attack_prompt(snapshot, menu)
        raw_output = self.model.generate(
            prompt, max_tokens=256, temperature=0.3, caller="attack"
        ) or ""

        indices = parse_attack_indices(raw_output, len(menu))

        if indices is not None:
            attacks = []
            for idx in indices:
                opt = menu[idx - 1]
                attacks.append({
                    "src": opt["src"],
                    "target": opt["target"],
                    "idx": idx,
                })

            self.turn_log.append({
                "phase": "attacks",
                "prompt": prompt,
                "response": raw_output,
                "decision": attacks,
                "fallback": False,
                "snapshot": snapshot,
                "attack_menu": menu,
            })

            for atk in attacks:
                yield (atk["src"], atk["target"], None, None)
            return

        # Fallback to BetterAI
        LOG.warning("Attack parse failed, falling back to BetterAI")
        self.turn_log.append({
            "phase": "attacks",
            "prompt": prompt,
            "response": raw_output,
            "decision": [],
            "fallback": True,
            "snapshot": snapshot,
            "attack_menu": menu,
        })
        yield from super().attack()
