"""Benchmark Gemini LLMPlayer against pyrisk's built-in AIs.

Runs N games per matchup (LLM vs each AI), collecting:
  1. Turn-level training data (turns.jsonl) — prompts, responses, decisions
  2. Game-level KPI summary (summary.json) — win rates, call counts, etc.

Supports resuming: if turns.jsonl already exists, skips completed games
and appends new ones. summary.json is rebuilt from all data at the end.

Usage:
    python data/run_benchmark.py --games-per-matchup 1   # smoke test
    python data/run_benchmark.py --games-per-matchup 20  # full benchmark
    python data/run_benchmark.py --games-per-matchup 5 --output-dir data/my_run/
"""

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'pyrisk_vendor'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from game import Game
from world import CONNECT, MAP, KEY, AREAS
from ai.stupid import StupidAI
from ai.better import BetterAI
from ai.al import AlAI
from ai.chron import ChronAI
from llm_player.llm_player import LLMPlayer
from llm_player.nodes.decide_placement import decide_placement
from llm_player.graph import run_turn


# ── Benchmark LLMPlayer subclass ─────────────────────────────────────

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


# ── Matchup definitions ──────────────────────────────────────────────

MATCHUPS = [
    ("vs_StupidAI", StupidAI),
    ("vs_BetterAI", BetterAI),
    ("vs_AlAI", AlAI),
    ("vs_ChronAI", ChronAI),
]


# ── Run one game ─────────────────────────────────────────────────────

def run_game(opponent_class, seed=None):
    """Run a single 3-player game: LLM + opponent + StupidAI.

    Returns (winner_name, game, llm_player, duration_s).
    """
    if seed is not None:
        random.seed(seed)

    game = Game(curses=False, connect=CONNECT, cmap=MAP, ckey=KEY, areas=AREAS)
    game.add_player("LLM", BenchmarkLLMPlayer)
    game.add_player("OPP", opponent_class)
    game.add_player("FILL", StupidAI)

    start = time.time()
    winner = game.play()
    duration = time.time() - start

    return winner, game, game.players["LLM"], duration


# ── Resume support ───────────────────────────────────────────────────

def _load_existing(turns_path):
    """Load existing turns.jsonl and return (completed_games, all_game_records).

    completed_games: dict of {matchup_name: set of game_ids}
    all_game_records: list of per-game dicts reconstructed from turns data
    """
    completed_games = {}
    game_data = {}

    if not os.path.exists(turns_path):
        return completed_games, []

    with open(turns_path) as f:
        for line in f:
            entry = json.loads(line)
            matchup = entry["matchup"]
            gid = entry["game_id"]

            if matchup not in completed_games:
                completed_games[matchup] = set()
            completed_games[matchup].add(gid)

            if gid not in game_data:
                game_data[gid] = {
                    "game_id": gid,
                    "matchup": matchup,
                    "llm_won": entry["outcome"] == "win",
                    "winner": "LLM" if entry["outcome"] == "win" else "OPP/FILL",
                    "total_turns": 0,
                    "fallback_count": 0,
                    "decision_count": 0,
                }
            game_data[gid]["total_turns"] = max(
                game_data[gid]["total_turns"],
                entry["board_snapshot"]["turn"]
            )
            game_data[gid]["decision_count"] += 1
            if entry["fallback"]:
                game_data[gid]["fallback_count"] += 1

    return completed_games, list(game_data.values())


# ── Main benchmark ───────────────────────────────────────────────────

def run_benchmark(games_per_matchup, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    turns_path = os.path.join(output_dir, "turns.jsonl")
    summary_path = os.path.join(output_dir, "summary.json")

    # Check for existing data to resume from
    completed_games, existing_game_records = _load_existing(turns_path)
    if existing_game_records:
        next_game_id = max(g["game_id"] for g in existing_game_records) + 1
    else:
        next_game_id = 0

    all_games = list(existing_game_records)
    game_id = next_game_id

    total_existing = sum(len(ids) for ids in completed_games.values())
    total_needed = len(MATCHUPS) * games_per_matchup

    print(f"=== Benchmark: Gemini vs pyrisk AIs ===")
    print(f"Model: gemini-2.5-flash-lite")
    print(f"Games per matchup: {games_per_matchup}")
    if total_existing > 0:
        print(f"Resuming: {total_existing} games already completed")
    print()

    with open(turns_path, "a") as turns_file:
        for matchup_name, opponent_class in MATCHUPS:
            done = len(completed_games.get(matchup_name, set()))
            remaining = games_per_matchup - done

            if remaining <= 0:
                print(f"--- {matchup_name}: {done}/{games_per_matchup} already done, skipping ---\n")
                continue

            wins = 0
            matchup_games = []

            if done > 0:
                print(f"--- {matchup_name} ({remaining} remaining, {done} already done) ---")
            else:
                print(f"--- {matchup_name} ({games_per_matchup} games) ---")

            for i in range(remaining):
                seed = int(time.time() * 1000) % (2**31) + game_id
                winner, game, llm_player, duration = run_game(
                    opponent_class, seed=seed
                )
                llm_won = (winner == "LLM")
                if llm_won:
                    wins += 1

                # Count fallbacks from turn log
                fallback_count = sum(
                    1 for entry in llm_player.ai.turn_log if entry["fallback"]
                )

                # Game-level stats
                game_record = {
                    "game_id": game_id,
                    "matchup": matchup_name,
                    "winner": winner,
                    "llm_won": llm_won,
                    "total_turns": game.turn,
                    "api_calls": llm_player.ai.model.call_count,
                    "fallback_count": fallback_count,
                    "calls_by_node": dict(llm_player.ai.model.call_counts_by_caller),
                    "duration_s": round(duration, 1),
                }
                matchup_games.append(game_record)
                all_games.append(game_record)

                # Write turn-level data (backfill outcome)
                outcome = "win" if llm_won else "loss"
                for entry in llm_player.ai.turn_log:
                    turn_record = {
                        "game_id": game_id,
                        "matchup": matchup_name,
                        "turn": entry["turn"],
                        "phase": entry["phase"],
                        "prompt": entry["prompt"],
                        "response": entry["response"],
                        "decision": entry["decision"],
                        "fallback": entry["fallback"],
                        "board_snapshot": entry["board_snapshot"],
                        "outcome": outcome,
                    }
                    turns_file.write(json.dumps(turn_record) + "\n")
                turns_file.flush()

                game_num = done + i + 1
                status = "LLM wins" if llm_won else f"{winner} wins"
                print(
                    f"  Game {game_num:>2}/{games_per_matchup}: {status:<14} "
                    f"({game.turn} turns, {llm_player.ai.model.call_count} calls, "
                    f"{duration:.0f}s)"
                )
                game_id += 1

            # Matchup summary (new games only for display)
            if matchup_games:
                losses = remaining - wins
                win_rate = wins / remaining
                print(f"  This run: LLM {wins}-{losses} ({win_rate:.1%} win rate)\n")

    # Build summary from ALL data (existing + new)
    all_matchup_stats = {}
    for matchup_name, _ in MATCHUPS:
        mg = [g for g in all_games if g["matchup"] == matchup_name]
        if not mg:
            continue
        wins = sum(1 for g in mg if g.get("llm_won", g.get("winner") == "LLM"))
        losses = len(mg) - wins
        all_matchup_stats[matchup_name] = {
            "wins": wins,
            "losses": losses,
            "win_rate": round(wins / len(mg), 3),
            "avg_turns": round(sum(g["total_turns"] for g in mg) / len(mg), 1),
            "avg_fallbacks": round(sum(g["fallback_count"] for g in mg) / len(mg), 1),
        }
        # Only include api_calls/duration if available (not for resumed games)
        if "api_calls" in mg[0]:
            new_mg = [g for g in mg if "api_calls" in g]
            if new_mg:
                all_matchup_stats[matchup_name]["avg_api_calls"] = round(
                    sum(g["api_calls"] for g in new_mg) / len(new_mg), 1)
                all_matchup_stats[matchup_name]["avg_duration_s"] = round(
                    sum(g["duration_s"] for g in new_mg) / len(new_mg), 1)

    summary = {
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": "gemini-2.5-flash-lite",
        "games_per_matchup": games_per_matchup,
        "total_decisions": sum(1 for _ in open(turns_path)),
        "matchups": all_matchup_stats,
        "games": [g for g in all_games if "api_calls" in g],
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Print final results
    print("=== Final Results ===")
    for name, stats in all_matchup_stats.items():
        print(f"  {name:<15} {stats['win_rate']:.1%} ({stats['wins']}-{stats['losses']})")

    with open(turns_path) as f:
        total_decisions = sum(1 for _ in f)

    print(f"\nSaved: {turns_path} ({total_decisions} decisions)")
    print(f"Saved: {summary_path}")


# ── CLI ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Gemini LLMPlayer vs pyrisk AIs")
    parser.add_argument(
        "--games-per-matchup", type=int, default=20,
        help="Number of games per matchup (default: 20)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/benchmark_results",
        help="Output directory for results (default: data/benchmark_results)",
    )
    args = parser.parse_args()

    run_benchmark(args.games_per_matchup, args.output_dir)
