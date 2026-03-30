"""Run HybridPlayer games and save turn logs.

Usage:
    python analysis/run_benchmark.py --games 5 --output results/gemini_benchmark
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
from ai.stupid import StupidAI
from llm_player.hybrid_player import HybridPlayer


def run_game(seed=None):
    """Run one game: HybridPlayer vs StupidAI x2."""
    if seed is not None:
        random.seed(seed)

    game = Game(curses=False, connect=CONNECT, cmap=MAP, ckey=KEY, areas=AREAS)
    game.add_player("LLM", HybridPlayer)
    game.add_player("OPP", StupidAI)
    game.add_player("FILL", StupidAI)

    winner = game.play()
    llm_player = game.players["LLM"]

    return winner, llm_player


def main():
    parser = argparse.ArgumentParser(description="Run HybridPlayer benchmark")
    parser.add_argument("--games", type=int, default=5)
    parser.add_argument("--output", type=str, default="results/benchmark")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    output_path = os.path.join(args.output, "turns.jsonl")

    all_entries = []
    wins = 0

    for i in range(args.games):
        game_seed = args.seed + i
        print(f"Game {i+1}/{args.games} (seed={game_seed})...", end=" ", flush=True)

        try:
            winner, llm_player = run_game(seed=game_seed)
        except Exception as e:
            print(f"ERROR: {e}")
            continue

        llm_won = (winner == "LLM")
        if llm_won:
            wins += 1
        outcome = "win" if llm_won else "loss"

        n_turns = len(llm_player.ai.turn_log)
        fallbacks = sum(1 for t in llm_player.ai.turn_log if t.get("fallback"))
        print(f"{'WIN' if llm_won else 'LOSS'} "
              f"({n_turns} decisions, {fallbacks} fallbacks)")

        for entry in llm_player.ai.turn_log:
            entry["outcome"] = outcome
            entry["game_id"] = i
            all_entries.append(entry)

    # Write results
    with open(output_path, "w") as f:
        for entry in all_entries:
            f.write(json.dumps(entry) + "\n")

    # Summary
    print(f"\n=== Results ===")
    print(f"Games: {args.games}")
    print(f"Wins:  {wins}/{args.games} ({wins/args.games:.0%})")
    print(f"Total decisions: {len(all_entries)}")
    print(f"Output: {output_path}")

    summary = {"games": args.games, "wins": wins,
               "win_rate": wins / args.games if args.games else 0,
               "total_decisions": len(all_entries)}
    summary_path = os.path.join(args.output, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
