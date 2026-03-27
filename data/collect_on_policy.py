"""Collect on-policy training data by playing Risk with the trained model.

Plays N games per matchup using the PEFT-trained Qwen model, collecting
turn-level data in the same format as data/benchmark_results/turns.jsonl.

The data is saved to data/on_policy_results/ by default and includes a
"data_source": "on_policy" field to distinguish it from off-policy data.

Supports resuming: if turns.jsonl already exists, skips completed games.

Usage:
    # On Colab (after training)
    python data/collect_on_policy.py --model-path ./risk_grpo_output --games-per-matchup 5

    # Smoke test (1 game)
    python data/collect_on_policy.py --model-path ./risk_grpo_output --games-per-matchup 1

    # Specific matchups only
    python data/collect_on_policy.py --model-path ./risk_grpo_output --matchups vs_StupidAI vs_BetterAI
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
from llm_player.benchmark_player import BenchmarkLLMPlayer


# ── Matchup definitions ──────────────────────────────────────────────

MATCHUPS = {
    "vs_StupidAI": StupidAI,
    "vs_BetterAI": BetterAI,
    "vs_AlAI": AlAI,
    "vs_ChronAI": ChronAI,
}


# ── Run one game ─────────────────────────────────────────────────────

def run_game(opponent_class, seed=None):
    """Run a single 3-player game: trained model + opponent + StupidAI.

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
    """Load existing turns.jsonl and return (completed_games, all_game_records)."""
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


# ── Main collection ──────────────────────────────────────────────────

def collect(model_path, games_per_matchup, output_dir, matchup_names):
    os.makedirs(output_dir, exist_ok=True)
    turns_path = os.path.join(output_dir, "turns.jsonl")
    summary_path = os.path.join(output_dir, "summary.json")

    # Set env var so ModelBackend auto-detects backend.
    # Only convert to absolute path if it's a local directory (not a HF model name).
    if os.path.exists(model_path):
        os.environ["RISK_MODEL_PATH"] = os.path.abspath(model_path)
    else:
        os.environ["RISK_MODEL_PATH"] = model_path

    # Filter to requested matchups
    matchups = [(name, MATCHUPS[name]) for name in matchup_names
                if name in MATCHUPS]
    if not matchups:
        print(f"Error: no valid matchups in {matchup_names}")
        print(f"Available: {list(MATCHUPS.keys())}")
        return

    # Check for existing data to resume from
    completed_games, existing_game_records = _load_existing(turns_path)
    if existing_game_records:
        next_game_id = max(g["game_id"] for g in existing_game_records) + 1
    else:
        next_game_id = 0

    all_games = list(existing_game_records)
    game_id = next_game_id

    total_existing = sum(len(ids) for ids in completed_games.values())

    print(f"=== On-Policy Data Collection ===")
    print(f"Model:        {model_path}")
    print(f"Output:       {output_dir}")
    print(f"Games/matchup: {games_per_matchup}")
    print(f"Matchups:     {[m[0] for m in matchups]}")
    if total_existing > 0:
        print(f"Resuming: {total_existing} games already completed")
    print()

    with open(turns_path, "a") as turns_file:
        for matchup_name, opponent_class in matchups:
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
                try:
                    winner, game, llm_player, duration = run_game(
                        opponent_class, seed=seed
                    )
                except Exception as e:
                    print(f"  Game {done + i + 1}: ERROR - {e}")
                    game_id += 1
                    continue

                llm_won = (winner == "LLM")
                if llm_won:
                    wins += 1

                fallback_count = sum(
                    1 for entry in llm_player.ai.turn_log if entry["fallback"]
                )

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

                # Write turn-level data
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
                        "data_source": "on_policy",
                    }
                    turns_file.write(json.dumps(turn_record) + "\n")
                turns_file.flush()

                game_num = done + i + 1
                status = "LLM wins" if llm_won else f"{winner} wins"
                print(
                    f"  Game {game_num:>2}/{games_per_matchup}: {status:<14} "
                    f"({game.turn} turns, {llm_player.ai.model.call_count} calls, "
                    f"{fallback_count} fallbacks, {duration:.0f}s)"
                )
                game_id += 1

            if matchup_games:
                losses = remaining - wins
                win_rate = wins / remaining if remaining > 0 else 0
                print(f"  This run: LLM {wins}-{losses} ({win_rate:.1%} win rate)\n")

    # Build summary from ALL data
    all_matchup_stats = {}
    for matchup_name, _ in matchups:
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
        new_mg = [g for g in mg if "api_calls" in g]
        if new_mg:
            all_matchup_stats[matchup_name]["avg_api_calls"] = round(
                sum(g["api_calls"] for g in new_mg) / len(new_mg), 1)
            all_matchup_stats[matchup_name]["avg_duration_s"] = round(
                sum(g["duration_s"] for g in new_mg) / len(new_mg), 1)

    summary = {
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": f"peft:{os.path.abspath(model_path)}",
        "data_source": "on_policy",
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
    parser = argparse.ArgumentParser(
        description="Collect on-policy training data using the trained PEFT model"
    )
    parser.add_argument(
        "--model-path", type=str,
        default=os.environ.get("RISK_MODEL_PATH", "risk_grpo_output"),
        help="Path to PEFT adapter directory (default: risk_grpo_output)",
    )
    parser.add_argument(
        "--games-per-matchup", type=int, default=5,
        help="Number of games per matchup (default: 5)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/on_policy_results",
        help="Output directory for results (default: data/on_policy_results)",
    )
    parser.add_argument(
        "--matchups", type=str, nargs="+",
        default=list(MATCHUPS.keys()),
        help="Which matchups to run (default: all)",
    )
    args = parser.parse_args()

    collect(args.model_path, args.games_per_matchup,
            args.output_dir, args.matchups)
