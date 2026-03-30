"""Collect SFT training data from heuristic AI games.

Runs BetterAI and ChronAI games and captures decisions with board snapshots.
Generates synthetic LLM-style completions (tool call + strategic bridge + JSON)
for SFT training.

BetterAI has hard-coded continent strategy. ChronAI uses pathfinding and
adaptive strategy (defensive when strong, aggressive when weak). Both win
much more often than Gemini Flash Lite (the original training data source,
30% win rate vs StupidAI). Training on these decisions teaches the model:
- Prioritize a continent and focus there
- Attack when you have force advantage
- Move inland troops to borders
- Adapt strategy based on board position

The synthetic completion format matches what the LLM should output:
  <tool_call>tool_name(args)</tool_call>
  <tool_result>...</tool_result>
  Strategic reasoning (2-3 sentences referencing tool results).
  ```json
  {"decision": ...}
  ```

Original game data is NOT modified. Output goes to a new file.

Usage:
    python data/collect_heuristic_data.py
    python data/collect_heuristic_data.py --games 200 --ai both
    python data/collect_heuristic_data.py --games 100 --ai chron
"""

import argparse
import collections
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
from risk_env.state_serializer import serialize_game_state
from tools.battle_sim import simulate_battle
from tools.threat_analyzer import analyze_threats_from_snapshot
from tools.position_evaluator import evaluate_position_from_snapshot
from risk_env.tool_interface import _format_result


# ── Prompt templates (exact copies from llm_player/nodes/) ────────────

REINFORCEMENT_PROMPT = """You are playing Risk. Here is the current board state:

{board_summary}

You have {available} reinforcement troops to place on your territories.

You may optionally call tools before deciding. Available tools:
- battle_sim(attacking=N, defending=N) — simulate battle odds
- threat_analyzer() — analyze threats to your territories
- position_evaluator() — evaluate your overall board position

To call a tool, write: <tool_call>tool_name(args)</tool_call>

After any tool analysis, output your final decision as JSON:
```json
{{"reinforcements": {{"TerritoryName": count, ...}}}}
```

Rules:
- Troop counts must sum to exactly {available}
- You can only place troops on territories you own
- Place troops strategically: prioritize border territories under threat"""

ATTACK_PROMPT = """You are playing Risk. Here is the current board state:

{board_summary}

Decide which attacks to execute this turn. You may attack 0 or more times.

You may optionally call tools before deciding. Available tools:
- battle_sim(attacking=N, defending=N) — simulate battle odds
- threat_analyzer() — analyze threats to your territories
- position_evaluator() — evaluate your overall board position

To call a tool, write: <tool_call>tool_name(args)</tool_call>

After any tool analysis, output your final decision as JSON:
```json
{{"attacks": [{{"src": "YourTerritory", "target": "EnemyTerritory"}}, ...]}}
```

Or to skip attacking:
```json
{{"attacks": []}}
```

Rules:
- src must be a territory you own with more than 1 troop
- target must be adjacent to src and owned by an enemy
- You can list multiple attacks; they execute in order"""

MOVEMENT_PROMPT = """You are playing Risk. Here is the current board state:

{board_summary}

You may make one free troop movement: move troops from one of your territories
to another connected friendly territory.

You may optionally call tools before deciding. Available tools:
- battle_sim(attacking=N, defending=N) — simulate battle odds
- threat_analyzer() — analyze threats to your territories
- position_evaluator() — evaluate your overall board position

To call a tool, write: <tool_call>tool_name(args)</tool_call>

After any tool analysis, output your final decision as JSON:
```json
{{"movement": {{"src": "YourTerritory", "target": "YourTerritory", "count": N}}}}
```

Or to skip movement:
```json
{{"movement": null}}
```

Rules:
- Both src and target must be territories you own
- src and target must be adjacent
- count must be between 1 and (src troops - 1) — you must leave at least 1 troop behind"""


# ── Logging wrapper for BetterAI ─────────────────────────────────────

class LoggingBetterAI(BetterAI):
    """BetterAI wrapper that logs decisions and board snapshots."""

    def start(self):
        super().start()
        self.turn_log = []

    def _snapshot(self):
        """Create board snapshot matching BenchmarkLLMPlayer format."""
        player = self.player
        game = self.game

        owned = [t for t in game.world.territories.values()
                 if t.owner == player]
        borders = [t for t in owned
                   if any(adj.owner != player
                          for adj in t.connect if adj.owner is not None)]

        territory_map = {}
        for name, t in game.world.territories.items():
            # Find continent
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

        # Detect controlled continents
        continents = []
        for area in game.world.areas.values():
            if all(t.owner == player for t in area.territories):
                continents.append(area.name)

        # Players info
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

    def _board_summary(self):
        """Get serialized board state text (same as LLM prompt)."""
        return serialize_game_state(self.game, self.player)

    def reinforce(self, available):
        snapshot = self._snapshot()
        snapshot["reinforcements"] = available
        board_summary = self._board_summary()

        result = super().reinforce(available)

        # Convert Territory objects to names
        named = {}
        for t, count in result.items():
            name = t.name if hasattr(t, 'name') else str(t)
            named[name] = named.get(name, 0) + count

        self.turn_log.append({
            "turn": self.game.turn,
            "phase": "reinforcements",
            "decision": named,
            "board_snapshot": snapshot,
            "board_summary": board_summary,
            "available": available,
        })

        return result

    def attack(self):
        snapshot = self._snapshot()
        board_summary = self._board_summary()

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
            "board_summary": board_summary,
        })

        for attack in attacks_list:
            yield attack

    def freemove(self):
        snapshot = self._snapshot()
        board_summary = self._board_summary()

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
            "board_summary": board_summary,
        })

        return result


class LoggingChronAI(ChronAI):
    """ChronAI wrapper that logs decisions and board snapshots.

    Reuses the same _snapshot and _board_summary methods as LoggingBetterAI.
    """

    def start(self):
        super().start()
        self.turn_log = []

    # Reuse snapshot/summary from LoggingBetterAI
    _snapshot = LoggingBetterAI._snapshot
    _board_summary = LoggingBetterAI._board_summary

    def reinforce(self, available):
        snapshot = self._snapshot()
        snapshot["reinforcements"] = available
        board_summary = self._board_summary()

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
            "board_summary": board_summary,
            "available": available,
        })

        return result

    def attack(self):
        snapshot = self._snapshot()
        board_summary = self._board_summary()

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
            "board_summary": board_summary,
        })

        for attack in attacks_list:
            yield attack

    def freemove(self):
        snapshot = self._snapshot()
        board_summary = self._board_summary()

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
            "board_summary": board_summary,
        })

        return result


# ── Tool call + result generation ────────────────────────────────────

def build_tool_prefix_reinforcements(snapshot):
    """Generate threat_analyzer() call + result."""
    try:
        threats = analyze_threats_from_snapshot(snapshot)
    except Exception:
        return None
    if not threats:
        return None
    call = "<tool_call>threat_analyzer()</tool_call>"
    result_text = _format_result(threats[:5])
    return f"{call}\n<tool_result>\n{result_text}\n</tool_result>"


def build_tool_prefix_attacks(attacks, snapshot):
    """Generate battle_sim() or threat_analyzer() call + result."""
    if attacks:
        first = attacks[0]
        territory_map = snapshot.get("territory_map", {})
        src_forces = territory_map.get(first["src"], {}).get("forces", 0)
        tgt_forces = territory_map.get(first["target"], {}).get("forces", 0)
        if src_forces > 1 and tgt_forces > 0:
            try:
                sim_result = simulate_battle(src_forces, tgt_forces,
                                             num_simulations=1000)
                call = (f"<tool_call>battle_sim(attacking={src_forces}, "
                        f"defending={tgt_forces})</tool_call>")
                result_text = _format_result(sim_result)
                return f"{call}\n<tool_result>\n{result_text}\n</tool_result>"
            except Exception:
                pass
    # Fallback: threat_analyzer
    try:
        threats = analyze_threats_from_snapshot(snapshot)
    except Exception:
        return None
    if not threats:
        return None
    call = "<tool_call>threat_analyzer()</tool_call>"
    result_text = _format_result(threats[:5])
    return f"{call}\n<tool_result>\n{result_text}\n</tool_result>"


def build_tool_prefix_movement(snapshot):
    """Generate position_evaluator() call + result."""
    try:
        position = evaluate_position_from_snapshot(snapshot)
    except Exception:
        return None
    if not position:
        return None
    call = "<tool_call>position_evaluator()</tool_call>"
    result_text = _format_result(position)
    return f"{call}\n<tool_result>\n{result_text}\n</tool_result>"


# ── Strategic bridge generation ──────────────────────────────────────

def _get_continent_progress(snapshot):
    """Get continent progress from snapshot."""
    territory_map = snapshot["territory_map"]
    player_name = snapshot["player_name"]

    BONUSES = {
        "Africa": 3, "Asia": 7, "Australia": 2,
        "Europe": 5, "North America": 5, "South America": 2,
    }

    continents = {}
    for name, info in territory_map.items():
        cont = info.get("continent", "Unknown")
        if cont not in continents:
            continents[cont] = {
                "name": cont, "total": 0, "owned": 0,
                "bonus": BONUSES.get(cont, 0), "missing": [],
            }
        continents[cont]["total"] += 1
        if info.get("owner") == player_name:
            continents[cont]["owned"] += 1
        else:
            continents[cont]["missing"].append(name)

    for c in continents.values():
        c["remaining"] = c["total"] - c["owned"]

    return list(continents.values())


def bridge_reinforcements(decision, snapshot):
    """2-3 sentence strategic bridge for reinforcement decision."""
    territory_map = snapshot["territory_map"]
    borders = set(snapshot["border_territories"])
    continent_info = _get_continent_progress(snapshot)

    targets = list(decision.keys())
    border_targets = [t for t in targets if t in borders]

    parts = []

    # Mention continent strategy
    near_complete = [c for c in continent_info
                     if 0 < c["remaining"] <= 2]
    complete = [c for c in continent_info if c["remaining"] == 0]

    if complete:
        c = complete[0]
        parts.append(f"I control {c['name']} (+{c['bonus']} bonus).")
    if near_complete:
        c = near_complete[0]
        parts.append(
            f"{c['name']} is {c['owned']}/{c['total']} — "
            f"need {c['remaining']} more for +{c['bonus']} bonus."
        )

    # Mention border defense
    if border_targets:
        max_threat = 0
        most_threatened = border_targets[0]
        for t in border_targets:
            info = territory_map.get(t, {})
            enemy_adj = sum(
                territory_map.get(adj, {}).get("forces", 0)
                for adj in info.get("adjacent", [])
                if territory_map.get(adj, {}).get("owner") != snapshot["player_name"]
                and territory_map.get(adj, {}).get("owner") is not None
            )
            if enemy_adj > max_threat:
                max_threat = enemy_adj
                most_threatened = t
        parts.append(
            f"Reinforcing {most_threatened} which faces "
            f"{max_threat} enemy troops adjacent."
        )
    else:
        parts.append("Concentrating troops on key positions.")

    return " ".join(parts) if parts else "Reinforcing priority territories."


def bridge_attacks(attacks, snapshot):
    """2-3 sentence strategic bridge for attack decision."""
    if not attacks:
        return "No attacks with sufficient force advantage this turn."

    territory_map = snapshot["territory_map"]
    continent_info = _get_continent_progress(snapshot)

    first = attacks[0]
    src_forces = territory_map.get(first["src"], {}).get("forces", 0)
    tgt_forces = territory_map.get(first["target"], {}).get("forces", 0)

    parts = []

    # Check if attack completes a continent
    target_continent = territory_map.get(first["target"], {}).get("continent", "")
    for c in continent_info:
        if c["name"] == target_continent and c["remaining"] == 1:
            parts.append(
                f"Taking {first['target']} completes {c['name']} "
                f"for +{c['bonus']} bonus."
            )
            break

    ratio_str = f"{src_forces} vs {tgt_forces}"
    if src_forces > tgt_forces * 2:
        parts.append(
            f"Attacking {first['target']} from {first['src']} "
            f"({ratio_str}) — overwhelming advantage."
        )
    elif src_forces > tgt_forces:
        parts.append(
            f"Attacking {first['target']} from {first['src']} "
            f"({ratio_str}) — favorable odds."
        )
    else:
        parts.append(
            f"Attacking {first['target']} from {first['src']} "
            f"({ratio_str})."
        )

    if len(attacks) > 1:
        n = len(attacks) - 1
        parts.append(f"Also targeting {n} more position{'s' if n > 1 else ''}.")

    return " ".join(parts)


def bridge_movement(movement, snapshot):
    """2-3 sentence strategic bridge for movement decision."""
    if movement is None:
        borders = set(snapshot["border_territories"])
        owned = set(snapshot["owned_territories"])
        inland = owned - borders
        if not inland:
            return "All territories are on the border. No useful movement available."
        return "Current positions are adequate. No movement needed."

    borders = set(snapshot["border_territories"])
    src = movement["src"]
    target = movement["target"]
    count = movement["count"]

    src_is_border = src in borders
    tgt_is_border = target in borders

    if not src_is_border and tgt_is_border:
        return (
            f"Moving {count} troops from {src} (interior) to "
            f"{target} (border) to strengthen defenses."
        )
    elif src_is_border and tgt_is_border:
        return (
            f"Redistributing {count} troops from {src} to "
            f"{target} to balance border defenses."
        )
    else:
        return f"Moving {count} troops from {src} to {target} to consolidate forces."


# ── Completion builder ───────────────────────────────────────────────

def build_completion(phase, decision, snapshot):
    """Build synthetic completion: tool call + strategic bridge + JSON."""
    parts = []

    # 1. Tool call + result
    if phase == "reinforcements":
        tool = build_tool_prefix_reinforcements(snapshot)
    elif phase == "attacks":
        tool = build_tool_prefix_attacks(decision, snapshot)
    elif phase == "movement":
        tool = build_tool_prefix_movement(snapshot)
    else:
        tool = None

    if tool:
        parts.append(tool)

    # 2. Strategic bridge (2-3 sentences)
    if phase == "reinforcements":
        bridge = bridge_reinforcements(decision, snapshot)
    elif phase == "attacks":
        bridge = bridge_attacks(decision, snapshot)
    elif phase == "movement":
        bridge = bridge_movement(decision, snapshot)
    else:
        bridge = ""

    if bridge:
        parts.append(bridge)

    # 3. JSON decision
    if phase == "reinforcements":
        json_data = {"reinforcements": decision}
    elif phase == "attacks":
        json_data = {"attacks": decision if decision else []}
    elif phase == "movement":
        json_data = {"movement": decision}
    else:
        json_data = {}

    json_str = json.dumps(json_data, indent=2)
    parts.append(f"```json\n{json_str}\n```")

    return "\n\n".join(parts)


# ── Prompt builder ───────────────────────────────────────────────────

def build_prompt(phase, board_summary, available=None):
    """Build the prompt the LLM would see for this phase."""
    if phase == "reinforcements":
        return REINFORCEMENT_PROMPT.format(
            board_summary=board_summary, available=available
        )
    elif phase == "attacks":
        return ATTACK_PROMPT.format(board_summary=board_summary)
    elif phase == "movement":
        return MOVEMENT_PROMPT.format(board_summary=board_summary)
    return ""


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


# ── Main ─────────────────────────────────────────────────────────────

def collect(num_games, output_path, ai_names, seed=42):
    """Run games and convert decisions to SFT training data.

    Args:
        num_games: games per AI type.
        output_path: where to write turns.jsonl.
        ai_names: list of AI names to use (e.g. ["better", "chron"]).
        seed: random seed.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    all_entries = []
    stats = {
        "games": 0, "wins": 0,
        "reinforcements": 0, "attacks": 0, "attacks_real": 0,
        "attacks_empty": 0, "movement": 0, "movement_real": 0,
        "movement_null": 0, "tool_prefixes": 0,
    }

    for ai_name in ai_names:
        ai_class = AI_CLASSES[ai_name]
        print(f"\n--- {ai_name.upper()} AI ({num_games} games) ---")

        for i in range(num_games):
            game_seed = seed + i + (1000 if ai_name == "chron" else 0)
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

            # Convert each decision to a training entry
            for entry in llm_player.ai.turn_log:
                phase = entry["phase"]
                decision = entry["decision"]
                snapshot = entry["board_snapshot"]
                board_summary = entry["board_summary"]

                # Build prompt (same as LLM would see)
                available = entry.get("available")
                prompt = build_prompt(phase, board_summary, available)

                # Build synthetic completion
                response = build_completion(phase, decision, snapshot)

                # Track stats
                if phase == "reinforcements":
                    stats["reinforcements"] += 1
                elif phase == "attacks":
                    stats["attacks"] += 1
                    if decision:
                        stats["attacks_real"] += 1
                    else:
                        stats["attacks_empty"] += 1
                elif phase == "movement":
                    stats["movement"] += 1
                    if decision:
                        stats["movement_real"] += 1
                    else:
                        stats["movement_null"] += 1

                if "<tool_call>" in response:
                    stats["tool_prefixes"] += 1

                all_entries.append({
                    "prompt": prompt,
                    "response": response,
                    "phase": phase,
                    "board_snapshot": snapshot,
                    "outcome": outcome,
                    "fallback": False,
                    "data_source": f"heuristic_{ai_name}",
                })

    # Write output
    with open(output_path, "w") as f:
        for entry in all_entries:
            f.write(json.dumps(entry) + "\n")

    # Print stats
    win_rate = stats["wins"] / stats["games"] if stats["games"] else 0
    print(f"\n=== Heuristic Data Collection ===")
    print(f"Games:          {stats['games']} ({stats['wins']} wins, "
          f"{win_rate:.0%} win rate)")
    print(f"Total entries:  {len(all_entries)}")
    print(f"  Reinforcements: {stats['reinforcements']}")
    print(f"  Attacks:        {stats['attacks']} "
          f"({stats['attacks_real']} real, {stats['attacks_empty']} empty)")
    print(f"  Movement:       {stats['movement']} "
          f"({stats['movement_real']} real, {stats['movement_null']} null)")
    print(f"  Tool prefixes:  {stats['tool_prefixes']}")
    print(f"Output:         {output_path}")

    # Write summary
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
                        default="data/heuristic_results/turns.jsonl",
                        help="Output path (default: data/heuristic_results/turns.jsonl)")
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

    collect(args.games, args.output, ai_names, args.seed)
