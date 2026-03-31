"""Compare benchmark results across all models.

Scans results/ for benchmark folders, computes metrics from turns.jsonl,
and prints a comparison table.

Usage:
    python analysis/compare_results.py
    python analysis/compare_results.py --reward   # include reward scores
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'pyrisk_vendor'))

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')


def analyze_benchmark(turns_path, compute_rewards=False):
    """Analyze a turns.jsonl file and return metrics dict."""
    entries = []
    with open(turns_path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    if not entries:
        return None

    # Game-level stats
    games = {}
    for e in entries:
        gid = e.get('game_id', 0)
        if gid not in games:
            games[gid] = {'outcome': e.get('outcome', '?'), 'turns': 0,
                          'fallbacks': 0, 'fb_reinf': 0, 'fb_atk': 0,
                          'llm_reinf': 0, 'llm_atk': 0}
        g = games[gid]
        g['turns'] += 1
        phase = e.get('phase', '')
        if e.get('fallback'):
            g['fallbacks'] += 1
            if phase == 'reinforcements':
                g['fb_reinf'] += 1
            else:
                g['fb_atk'] += 1
        else:
            if phase == 'reinforcements':
                g['llm_reinf'] += 1
            else:
                g['llm_atk'] += 1

    n_games = len(games)
    wins = sum(1 for g in games.values() if g['outcome'] == 'win')
    total_turns = sum(g['turns'] for g in games.values())
    total_fallbacks = sum(g['fallbacks'] for g in games.values())
    total_fb_reinf = sum(g['fb_reinf'] for g in games.values())
    total_fb_atk = sum(g['fb_atk'] for g in games.values())

    win_games = [g for g in games.values() if g['outcome'] == 'win']
    loss_games = [g for g in games.values() if g['outcome'] != 'win']

    metrics = {
        'games': n_games,
        'wins': wins,
        'win_rate': wins / n_games if n_games else 0,
        'total_decisions': total_turns,
        'total_fallbacks': total_fallbacks,
        'fallback_rate': total_fallbacks / total_turns if total_turns else 0,
        'fb_reinf': total_fb_reinf,
        'fb_atk': total_fb_atk,
        'avg_turns_win': (sum(g['turns'] for g in win_games) / len(win_games)
                          if win_games else 0),
        'avg_turns_loss': (sum(g['turns'] for g in loss_games) / len(loss_games)
                           if loss_games else 0),
        'avg_fb_win': (sum(g['fallbacks'] for g in win_games) / len(win_games)
                       if win_games else 0),
        'avg_fb_loss': (sum(g['fallbacks'] for g in loss_games) / len(loss_games)
                        if loss_games else 0),
    }

    # Per-game details
    metrics['game_details'] = []
    for gid, g in sorted(games.items()):
        metrics['game_details'].append({
            'game_id': gid,
            'outcome': g['outcome'],
            'turns': g['turns'],
            'fallbacks': g['fallbacks'],
        })

    # Reward scores
    if compute_rewards:
        from training.reward_hybrid import compute_reward as compute_reward_hybrid
        rewards = {'reinforcements': [], 'attacks': [], 'all': []}
        llm_rewards = []
        fb_rewards = []

        for e in entries:
            phase = e.get('phase', '')
            if phase not in ('reinforcements', 'attacks'):
                continue
            snapshot = e.get('board_snapshot') or e.get('snapshot', {})
            r = compute_reward_hybrid(
                e.get('response', ''), phase, snapshot,
                available=e.get('available', 3),
                attack_menu=e.get('attack_menu', []),
            )
            rewards[phase].append(r)
            rewards['all'].append(r)
            if e.get('fallback'):
                fb_rewards.append(r)
            else:
                llm_rewards.append(r)

        for key in ['reinforcements', 'attacks', 'all']:
            vals = rewards[key]
            if vals:
                metrics[f'reward_{key}'] = sum(vals) / len(vals)
            else:
                metrics[f'reward_{key}'] = None

        metrics['reward_llm_only'] = (sum(llm_rewards) / len(llm_rewards)
                                       if llm_rewards else None)
        metrics['reward_fb_only'] = (sum(fb_rewards) / len(fb_rewards)
                                      if fb_rewards else None)

    return metrics


def print_table(all_metrics, labels):
    """Print a comparison table."""
    col_w = max(18, max(len(l) for l in labels) + 2)

    def header():
        label_row = ''.join(l.rjust(col_w) for l in labels)
        print(f"  {'':30s}{label_row}")
        print('  ' + '-' * (30 + col_w * len(labels)))

    def row(name, key, fmt='pct'):
        vals = []
        for m in all_metrics:
            v = m.get(key)
            if v is None:
                vals.append('N/A')
            elif fmt == 'pct':
                vals.append(f'{v:.0%}')
            elif fmt == 'f2':
                vals.append(f'{v:.3f}')
            elif fmt == 'f1':
                vals.append(f'{v:.1f}')
            elif fmt == 'd':
                vals.append(f'{v}')
            else:
                vals.append(str(v))
        val_str = ''.join(v.rjust(col_w) for v in vals)
        print(f'  {name:30s}{val_str}')

    print()
    print('=' * (32 + col_w * len(labels)))
    header()

    print('\n  GAME RESULTS')
    row('Games', 'games', 'd')
    row('Wins', 'wins', 'd')
    row('Win rate', 'win_rate', 'pct')

    print('\n  DECISIONS')
    row('Total decisions', 'total_decisions', 'd')
    row('Total fallbacks', 'total_fallbacks', 'd')
    row('Fallback rate', 'fallback_rate', 'pct')
    row('  Reinforce fallbacks', 'fb_reinf', 'd')
    row('  Attack fallbacks', 'fb_atk', 'd')

    print('\n  GAME LENGTH')
    row('Avg turns (wins)', 'avg_turns_win', 'f1')
    row('Avg turns (losses)', 'avg_turns_loss', 'f1')
    row('Avg fallbacks (wins)', 'avg_fb_win', 'f1')
    row('Avg fallbacks (losses)', 'avg_fb_loss', 'f1')

    if any('reward_all' in m for m in all_metrics):
        print('\n  REWARD SCORES (v2 weights)')
        row('Overall reward', 'reward_all', 'f2')
        row('Reinforce reward', 'reward_reinforcements', 'f2')
        row('Attack reward', 'reward_attacks', 'f2')
        row('LLM-only reward', 'reward_llm_only', 'f2')
        row('Fallback-only reward', 'reward_fb_only', 'f2')

    print('=' * (32 + col_w * len(labels)))
    print()


def main():
    parser = argparse.ArgumentParser(description='Compare benchmark results')
    parser.add_argument('--reward', action='store_true',
                        help='Compute reward scores (slower)')
    parser.add_argument('--results-dir', type=str, default=RESULTS_DIR,
                        help=f'Results directory (default: {RESULTS_DIR})')
    args = parser.parse_args()

    results_dir = os.path.abspath(args.results_dir)
    if not os.path.isdir(results_dir):
        print(f'Results directory not found: {results_dir}')
        sys.exit(1)

    # Find all benchmark folders
    benchmarks = []
    for name in sorted(os.listdir(results_dir)):
        turns_path = os.path.join(results_dir, name, 'turns.jsonl')
        if os.path.isfile(turns_path):
            benchmarks.append((name, turns_path))

    if not benchmarks:
        print('No benchmark results found.')
        sys.exit(1)

    labels = []
    all_metrics = []
    for name, path in benchmarks:
        print(f'Analyzing {name}...')
        metrics = analyze_benchmark(path, compute_rewards=args.reward)
        if metrics:
            labels.append(name)
            all_metrics.append(metrics)

    print_table(all_metrics, labels)


if __name__ == '__main__':
    main()
