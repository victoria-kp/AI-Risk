"""Collect SFT training data from heuristic AI games.

Runs BetterAI, AlAI, and ChronAI games and captures reinforce + attack
decisions with board snapshots. Uses menu-based prompts (no tool calls).

Usage:
    python data/collect_heuristic_data.py --games 200 --ai all
    python data/collect_heuristic_data.py --games 200 --ai better
"""

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'pyrisk_vendor'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from game import Game
from world import CONNECT, MAP, KEY, AREAS
from ai.better import BetterAI
from ai.chron import ChronAI
from ai.stupid import StupidAI
from llm_player.decision_menus import (
    build_reinforce_prompt,
    build_attack_menu,
    build_attack_prompt,
    map_attack_decisions_to_indices,
)


# ── Logging wrapper for BetterAI ─────────────────────────────────────

class LoggingBetterAI(BetterAI):
    """BetterAI wrapper that logs decisions and board snapshots."""

    def start(self):
        super().start()
        self.turn_log = []

    def _snapshot(self):
        """Create board snapshot dict from live game state."""
        player = self.player
        game = self.game

        owned = [t for t in game.world.territories.values()
                 if t.owner == player]
        borders = [t for t in owned
                   if any(adj.owner != player
                          for adj in t.connect if adj.owner is not None)]

        territory_map = {}
        for name, t in game.world.territories.items():
            continent = "Unknown"
            for area in game.world.areas.values():
                if t in area.territories:
                    continent = area.name
                    break
            territory_map[name] = {
                "owner": t.owner.name if t.owner else None,
                "forces": t.forces,
                "continent": continent,
                "adjacent": [adj.name for adj in t.connect],
            }

        continents = []
        for area in game.world.areas.values():
            if all(t.owner == player for t in area.territories):
                continents.append(area.name)

        players = {}
        for pname, p in game.players.items():
            p_territories = [t for t in game.world.territories.values()
                             if t.owner == p]
            p_continents = []
            for area in game.world.areas.values():
                if all(t.owner == p for t in area.territories):
                    p_continents.append(area.name)
            players[pname] = {
                "alive": p.alive,
                "territories": len(p_territories),
                "forces": sum(t.forces for t in p_territories),
                "continents": p_continents,
            }

        return {
            "player_name": player.name,
            "owned_territories": [t.name for t in owned],
            "border_territories": [t.name for t in borders],
            "total_forces": sum(t.forces for t in owned),
            "continents": continents,
            "reinforcements": 0,
            "territory_map": territory_map,
            "players": players,
            "turn": game.turn,
        }

    def reinforce(self, available):
        snapshot = self._snapshot()
        snapshot["reinforcements"] = available

        result = super().reinforce(available)

        named = {}
        for t, count in result.items():
            name = t.name if hasattr(t, 'name') else str(t)
            named[name] = named.get(name, 0) + count

        self.turn_log.append({
            "turn": self.game.turn,
            "phase": "reinforcements",
            "decision": named,
            "board_snapshot": snapshot,
            "available": available,
        })

        return result

    def attack(self):
        snapshot = self._snapshot()

        attacks_list = list(super().attack())

        attack_decisions = []
        for item in attacks_list:
            src, target = item[0], item[1]
            src_name = src.name if hasattr(src, 'name') else str(src)
            tgt_name = target.name if hasattr(target, 'name') else str(target)
            src_forces = snapshot["territory_map"].get(src_name, {}).get("forces", 0)
            attack_decisions.append({
                "src": src_name,
                "target": tgt_name,
                "count": max(1, src_forces - 1),
            })

        self.turn_log.append({
            "turn": self.game.turn,
            "phase": "attacks",
            "decision": attack_decisions,
            "board_snapshot": snapshot,
        })

        for attack in attacks_list:
            yield attack

    def freemove(self):
        snapshot = self._snapshot()
        result = super().freemove()

        if result:
            src, target, count = result
            movement = {
                "src": src.name if hasattr(src, 'name') else str(src),
                "target": target.name if hasattr(target, 'name') else str(target),
                "count": count,
            }
        else:
            movement = None

        self.turn_log.append({
            "turn": self.game.turn,
            "phase": "movement",
            "decision": movement,
            "board_snapshot": snapshot,
        })

        return result


class LoggingChronAI(ChronAI):
    """ChronAI wrapper that logs decisions and board snapshots."""

    def start(self):
        super().start()
        self.turn_log = []

    _snapshot = LoggingBetterAI._snapshot

    def reinforce(self, available):
        snapshot = self._snapshot()
        snapshot["reinforcements"] = available

        result = super().reinforce(available)

        named = {}
        for t, count in result.items():
            name = t.name if hasattr(t, 'name') else str(t)
            named[name] = named.get(name, 0) + count

        self.turn_log.append({
            "turn": self.game.turn,
            "phase": "reinforcements",
            "decision": named,
            "board_snapshot": snapshot,
            "available": available,
        })

        return result

    def attack(self):
        snapshot = self._snapshot()

        attacks_list = list(super().attack())

        attack_decisions = []
        for item in attacks_list:
            src, target = item[0], item[1]
            src_name = src.name if hasattr(src, 'name') else str(src)
            tgt_name = target.name if hasattr(target, 'name') else str(target)
            src_forces = snapshot["territory_map"].get(src_name, {}).get("forces", 0)
            attack_decisions.append({
                "src": src_name,
                "target": tgt_name,
                "count": max(1, src_forces - 1),
            })

        self.turn_log.append({
            "turn": self.game.turn,
            "phase": "attacks",
            "decision": attack_decisions,
            "board_snapshot": snapshot,
        })

        for attack in attacks_list:
            yield attack

    def freemove(self):
        snapshot = self._snapshot()
        result = super().freemove()

        if result:
            src, target, count = result
            movement = {
                "src": src.name if hasattr(src, 'name') else str(src),
                "target": target.name if hasattr(target, 'name') else str(target),
                "count": count,
            }
        else:
            movement = None

        self.turn_log.append({
            "turn": self.game.turn,
            "phase": "movement",
            "decision": movement,
            "board_snapshot": snapshot,
        })

        return result


# ── Bridge text generators ───────────────────────────────────────────

def hybrid_bridge_reinforce(decision, snapshot):
    """1-2 sentence reasoning for reinforcement decision."""
    borders = set(snapshot.get("border_territories", []))
    territory_map = snapshot.get("territory_map", {})
    player_name = snapshot.get("player_name", "LLM")

    targets = list(decision.keys())

    if not targets:
        return "No troops to place."

    main_target = max(targets, key=lambda t: decision[t])
    main_count = decision[main_target]

    if main_target in borders:
        info = territory_map.get(main_target, {})
        enemy_forces = []
        for adj in info.get("adjacent", []):
            adj_info = territory_map.get(adj, {})
            if adj_info.get("owner") and adj_info["owner"] != player_name:
                enemy_forces.append(f"{adj}({adj_info.get('forces', 0)})")
        if enemy_forces:
            return (f"Concentrating {main_count} troops on {main_target} "
                    f"which borders {', '.join(enemy_forces[:2])}.")

    if len(targets) == 1:
        return f"Placing all {main_count} troops on {main_target} for concentration."
    return (f"Reinforcing {main_target} with {main_count} troops "
            f"and spreading remainder across {len(targets)-1} other territories.")


def hybrid_bridge_attack(indices, menu):
    """1-2 sentence reasoning for attack decision."""
    if not indices:
        return "No attacks with good enough odds this turn."

    chosen = [menu[i - 1] for i in indices]
    first = chosen[0]
    src_f = first["src_forces"]
    tgt_f = first["tgt_forces"]

    if len(chosen) == 1:
        return (f"Attacking {first['target']} from {first['src']} "
                f"with {src_f} vs {tgt_f} troops.")
    return (f"Launching {len(chosen)} attacks, starting with "
            f"{first['src']} ({src_f}) -> {first['target']} ({tgt_f}).")


def build_hybrid_completion(phase, decision, snapshot, menu=None, indices=None):
    """Build synthetic completion: bridge text + JSON."""
    parts = []

    if phase == "reinforcements":
        bridge = hybrid_bridge_reinforce(decision, snapshot)
        json_data = {"reinforcements": decision}
    elif phase == "attacks":
        bridge = hybrid_bridge_attack(indices or [], menu or [])
        json_data = {"attacks": indices or []}
    else:
        return ""

    parts.append(bridge)
    json_str = json.dumps(json_data)
    parts.append(f"```json\n{json_str}\n```")
    return "\n\n".join(parts)


# ── Game running ─────────────────────────────────────────────────────

AI_CLASSES = {
    "better": LoggingBetterAI,
    "chron": LoggingChronAI,
}


def run_game(ai_class, seed=None):
    """Run one game: heuristic AI vs StupidAI vs StupidAI."""
    if seed is not None:
        random.seed(seed)

    game = Game(curses=False, connect=CONNECT, cmap=MAP, ckey=KEY, areas=AREAS)
    game.add_player("LLM", ai_class)
    game.add_player("OPP", StupidAI)
    game.add_player("FILL", StupidAI)

    winner = game.play()
    llm_player = game.players["LLM"]

    return winner, llm_player


# ── Data collection ──────────────────────────────────────────────────

def collect_hybrid(num_games, output_path, ai_names, seed=42):
    """Run games and convert decisions to hybrid SFT training data.

    Only generates reinforcement + attack entries (no placement/movement).
    Uses menu-based prompts from decision_menus.py.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    all_entries = []
    stats = {
        "games": 0, "wins": 0,
        "reinforcements": 0, "attacks": 0, "attacks_real": 0,
        "attacks_empty": 0,
    }

    for ai_name in ai_names:
        ai_class = AI_CLASSES[ai_name]
        print(f"\n--- {ai_name.upper()} AI ({num_games} games) ---")

        for i in range(num_games):
            seed_offset = {"better": 0, "chron": 1000}.get(ai_name, 0)
            game_seed = seed + i + seed_offset
            try:
                winner, llm_player = run_game(ai_class, seed=game_seed)
            except Exception as e:
                print(f"  Game {i+1}: ERROR - {e}")
                continue

            llm_won = (winner == "LLM")
            outcome = "win" if llm_won else "loss"
            stats["games"] += 1
            if llm_won:
                stats["wins"] += 1

            status = "WIN" if llm_won else "LOSS"
            n_decisions = len(llm_player.ai.turn_log)
            print(f"  Game {i+1:>3}/{num_games}: {status:<4} "
                  f"({n_decisions} decisions)")

            for entry in llm_player.ai.turn_log:
                phase = entry["phase"]
                decision = entry["decision"]
                snapshot = entry["board_snapshot"]

                if phase == "reinforcements":
                    available = entry.get("available", 3)
                    prompt = build_reinforce_prompt(snapshot, available)
                    completion = build_hybrid_completion(
                        phase, decision, snapshot)

                    stats["reinforcements"] += 1
                    all_entries.append({
                        "prompt": prompt,
                        "response": completion,
                        "phase": phase,
                        "board_snapshot": snapshot,
                        "available": available,
                        "outcome": outcome,
                        "data_source": f"heuristic_{ai_name}",
                    })

                elif phase == "attacks":
                    menu = build_attack_menu(snapshot)

                    if not menu:
                        prompt = build_attack_prompt(snapshot, [])
                        indices = []
                    else:
                        indices = map_attack_decisions_to_indices(
                            decision, menu)
                        prompt = build_attack_prompt(snapshot, menu)

                    completion = build_hybrid_completion(
                        phase, decision, snapshot,
                        menu=menu, indices=indices)

                    stats["attacks"] += 1
                    if indices:
                        stats["attacks_real"] += 1
                    else:
                        stats["attacks_empty"] += 1

                    all_entries.append({
                        "prompt": prompt,
                        "response": completion,
                        "phase": phase,
                        "board_snapshot": snapshot,
                        "attack_menu": menu,
                        "outcome": outcome,
                        "data_source": f"heuristic_{ai_name}",
                    })

    # Write output
    with open(output_path, "w") as f:
        for entry in all_entries:
            f.write(json.dumps(entry) + "\n")

    # Print stats
    win_rate = stats["wins"] / stats["games"] if stats["games"] else 0
    print(f"\n=== Data Collection ===")
    print(f"Games:          {stats['games']} ({stats['wins']} wins, "
          f"{win_rate:.0%} win rate)")
    print(f"Total entries:  {len(all_entries)}")
    print(f"  Reinforcements: {stats['reinforcements']}")
    print(f"  Attacks:        {stats['attacks']} "
          f"({stats['attacks_real']} real, {stats['attacks_empty']} empty)")
    print(f"Output:         {output_path}")

    summary_path = output_path.replace(".jsonl", "_summary.json")
    with open(summary_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Summary:        {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Collect SFT training data from heuristic AI games"
    )
    parser.add_argument("--games", type=int, default=50,
                        help="Number of games per AI type (default: 50)")
    parser.add_argument("--output", type=str,
                        default="data/hybrid_data/turns.jsonl",
                        help="Output path (default: data/hybrid_data/turns.jsonl)")
    parser.add_argument("--ai", type=str, default="both",
                        choices=["better", "chron", "both"],
                        help="Which AI to use (default: both)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    args = parser.parse_args()

    if args.ai == "both":
        ai_names = ["better", "chron"]
    else:
        ai_names = [args.ai]

    collect_hybrid(args.games, args.output, ai_names, args.seed)
