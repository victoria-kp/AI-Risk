"""Human player for pyrisk using input() for decisions.

Works in Jupyter notebooks: renders the board with draw_board(),
prints available options, and collects decisions via input().
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'pyrisk_vendor'))

from ai import AI


class HumanAI(AI):
    """Human-controlled AI that uses input() for all decisions."""

    def start(self):
        # Only initialize if not already set by the notebook
        if not hasattr(self, 'render_fn'):
            self.render_fn = None

    def _render(self, **kwargs):
        if self.render_fn:
            from IPython.display import clear_output
            clear_output(wait=False)
            self.render_fn(**kwargs)
            sys.stdout.flush()

    # ── Initial placement ────────────────────────────────────────────

    def initial_placement(self, empty, remaining):
        self._render(highlight=[t.name for t in empty] if empty else None)

        if empty:
            print(f"\n--- PLACEMENT (claiming) | {remaining} troops remaining ---")
            print("Unclaimed territories:")
            sorted_empty = sorted(empty, key=lambda t: (t.area.name, t.name))
            for i, t in enumerate(sorted_empty):
                print(f"  {i:>2}. {t.name} ({t.area.name})")
            idx = _input_int(f"Pick territory [0-{len(sorted_empty)-1}]: ",
                             0, len(sorted_empty) - 1)
            return sorted_empty[idx].name
        else:
            print(f"\n--- PLACEMENT (reinforcing) | {remaining} troops remaining ---")
            owned = sorted(self.player.territories, key=lambda t: t.name)
            for i, t in enumerate(owned):
                print(f"  {i:>2}. {t.name} ({t.forces} troops)")
            idx = _input_int(f"Reinforce which territory [0-{len(owned)-1}]: ",
                             0, len(owned) - 1)
            return owned[idx].name

    # ── Reinforcement ────────────────────────────────────────────────

    def reinforce(self, available):
        self._render()

        print(f"\n--- REINFORCEMENT | {available} troops to deploy ---")
        owned = sorted(self.player.territories, key=lambda t: t.name)
        for i, t in enumerate(owned):
            border_mark = " [border]" if t.border else ""
            print(f"  {i:>2}. {t.name} ({t.forces} troops){border_mark}")

        result = {}
        remaining = available
        while remaining > 0:
            print(f"\n  {remaining} troops left to place.")
            idx = _input_int(f"  Territory [0-{len(owned)-1}]: ", 0, len(owned) - 1)
            max_troops = remaining
            count = _input_int(f"  Troops [1-{max_troops}]: ", 1, max_troops)
            name = owned[idx].name
            result[name] = result.get(name, 0) + count
            remaining -= count

        print("  Reinforcements:", {k: v for k, v in result.items() if v > 0})
        return result

    # ── Attack ───────────────────────────────────────────────────────

    def attack(self):
        attacks = []
        while True:
            self._render()

            # Find territories that can attack
            can_attack = []
            for t in sorted(self.player.territories, key=lambda t: t.name):
                if t.forces > 1:
                    enemies = [a for a in t.connect if a.owner != self.player]
                    if enemies:
                        can_attack.append((t, enemies))

            if not can_attack:
                print("\n  No territories can attack.")
                break

            print(f"\n--- ATTACK (type 'done' to stop) ---")
            for i, (t, enemies) in enumerate(can_attack):
                enemy_str = ", ".join(
                    f"{e.name}({e.owner.name},{e.forces}f)" for e in enemies
                )
                print(f"  {i:>2}. {t.name} ({t.forces} troops) -> {enemy_str}")

            src_input = input("  Attack from [number or 'done']: ").strip()
            if src_input.lower() == 'done':
                break
            try:
                src_idx = int(src_input)
                if not 0 <= src_idx < len(can_attack):
                    print("  Invalid number.")
                    continue
            except ValueError:
                print("  Invalid input.")
                continue

            src_t, enemies = can_attack[src_idx]

            if len(enemies) == 1:
                target_t = enemies[0]
                print(f"  Target: {target_t.name} ({target_t.forces} troops)")
            else:
                print("  Targets:")
                for j, e in enumerate(enemies):
                    print(f"    {j}. {e.name} ({e.owner.name}, {e.forces} troops)")
                tgt_idx = _input_int(f"  Attack target [0-{len(enemies)-1}]: ",
                                     0, len(enemies) - 1)
                target_t = enemies[tgt_idx]

            attacks.append((src_t.name, target_t.name, None, None))

        return attacks

    # ── Free move ────────────────────────────────────────────────────

    def freemove(self):
        self._render()

        # Find territories that can move troops
        can_move = []
        for t in sorted(self.player.territories, key=lambda t: t.name):
            if t.forces > 1:
                friendly = [a for a in t.connect if a.owner == self.player]
                if friendly:
                    can_move.append((t, friendly))

        if not can_move:
            print("\n  No free moves available.")
            return None

        print(f"\n--- FREE MOVE (type 'skip' to skip) ---")
        for i, (t, friendly) in enumerate(can_move):
            targets_str = ", ".join(f"{f.name}({f.forces}f)" for f in friendly)
            print(f"  {i:>2}. {t.name} ({t.forces} troops) -> {targets_str}")

        src_input = input("  Move from [number or 'skip']: ").strip()
        if src_input.lower() == 'skip':
            return None
        try:
            src_idx = int(src_input)
            if not 0 <= src_idx < len(can_move):
                print("  Invalid number.")
                return None
        except ValueError:
            print("  Invalid input.")
            return None

        src_t, friendly = can_move[src_idx]

        if len(friendly) == 1:
            target_t = friendly[0]
            print(f"  Target: {target_t.name}")
        else:
            print("  Targets:")
            for j, f in enumerate(friendly):
                print(f"    {j}. {f.name} ({f.forces} troops)")
            tgt_idx = _input_int(f"  Move to [0-{len(friendly)-1}]: ",
                                 0, len(friendly) - 1)
            target_t = friendly[tgt_idx]

        max_count = src_t.forces - 1
        count = _input_int(f"  Troops to move [1-{max_count}]: ", 1, max_count)

        return (src_t.name, target_t.name, count)


# ── Input helpers ────────────────────────────────────────────────────

def _input_int(prompt, min_val, max_val):
    """Prompt until a valid integer in [min_val, max_val] is entered."""
    while True:
        try:
            val = int(input(prompt).strip())
            if min_val <= val <= max_val:
                return val
            print(f"  Please enter a number between {min_val} and {max_val}.")
        except ValueError:
            print(f"  Please enter a number between {min_val} and {max_val}.")
