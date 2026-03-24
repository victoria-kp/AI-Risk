"""Matplotlib Risk board renderer.

Draws filled territory regions colored by continent, with yellow boundary
lines, owner-colored troop circles, and dotted lines for cross-ocean
connections.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'pyrisk_vendor'))

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import numpy as np
from world import MAP, KEY, AREAS, CONNECT

plt.ioff()

# ── Parse ASCII MAP into grid and territory data ─────────────────────

_MAP_LINES = MAP.strip('\n').split('\n')
_MAP_HEIGHT = len(_MAP_LINES)
_MAP_WIDTH = max(len(line) for line in _MAP_LINES)

_TERRITORY_NAMES = sorted(KEY.values())
_NAME_TO_IDX = {name: idx + 1 for idx, name in enumerate(_TERRITORY_NAMES)}

_GRID = np.zeros((_MAP_HEIGHT, _MAP_WIDTH), dtype=np.int8)
for j, line in enumerate(_MAP_LINES):
    for i, c in enumerate(line):
        if c in KEY:
            _GRID[j][i] = _NAME_TO_IDX[KEY[c]]

# Territory centroids
TERRITORY_COORDS = {}
for name, idx in _NAME_TO_IDX.items():
    ys, xs = np.where(_GRID == idx)
    if len(xs) > 0:
        cx = xs.mean()
        cy = (_MAP_HEIGHT - 1) - ys.mean()
        TERRITORY_COORDS[name] = (cx, cy)

# Territory -> continent lookup
TERRITORY_TO_CONTINENT = {}
for continent_name, (_, territories) in AREAS.items():
    for t in territories:
        TERRITORY_TO_CONTINENT[t] = continent_name

# Per-territory boundary segments
_TERRITORY_BOUNDARY_SEGMENTS = {}

for j in range(_MAP_HEIGHT):
    for i in range(_MAP_WIDTH):
        idx = _GRID[j][i]
        if idx == 0:
            continue
        name = _TERRITORY_NAMES[idx - 1]
        if name not in _TERRITORY_BOUNDARY_SEGMENTS:
            _TERRITORY_BOUNDARY_SEGMENTS[name] = []
        yf = (_MAP_HEIGHT - 1) - j

        for dj, di, seg in [
            (0, 1,  ((i + 0.5, yf - 0.5), (i + 0.5, yf + 0.5))),
            (0, -1, ((i - 0.5, yf - 0.5), (i - 0.5, yf + 0.5))),
            (-1, 0, ((i - 0.5, yf + 0.5), (i + 0.5, yf + 0.5))),
            (1, 0,  ((i - 0.5, yf - 0.5), (i + 0.5, yf - 0.5))),
        ]:
            nj, ni = j + dj, i + di
            if 0 <= nj < _MAP_HEIGHT and 0 <= ni < _MAP_WIDTH:
                neighbor = _GRID[nj][ni]
            else:
                neighbor = 0
            if neighbor != idx:
                _TERRITORY_BOUNDARY_SEGMENTS[name].append(seg)

for name in _TERRITORY_BOUNDARY_SEGMENTS:
    _TERRITORY_BOUNDARY_SEGMENTS[name] = list(set(_TERRITORY_BOUNDARY_SEGMENTS[name]))

# All boundary segments (deduplicated)
_ALL_BOUNDARY_SEGMENTS = set()
for segs in _TERRITORY_BOUNDARY_SEGMENTS.values():
    _ALL_BOUNDARY_SEGMENTS.update(segs)
_ALL_BOUNDARY_SEGMENTS = list(_ALL_BOUNDARY_SEGMENTS)

# ── Cross-ocean connections ──────────────────────────────────────────

def _parse_all_edges():
    edges = set()
    for line in CONNECT.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split('--')]
        for i in range(len(parts) - 1):
            a, b = parts[i], parts[i + 1]
            edges.add(tuple(sorted([a, b])))
    return edges

def _territories_touch(name_a, name_b):
    idx_a, idx_b = _NAME_TO_IDX[name_a], _NAME_TO_IDX[name_b]
    ys_b, xs_b = np.where(_GRID == idx_b)
    cells_b = set(zip(ys_b.tolist(), xs_b.tolist()))
    ys_a, xs_a = np.where(_GRID == idx_a)
    for y, x in zip(ys_a.tolist(), xs_a.tolist()):
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            if (y + dy, x + dx) in cells_b:
                return True
    return False

_ALL_EDGES = _parse_all_edges()
CROSS_OCEAN = [(a, b) for a, b in _ALL_EDGES if not _territories_touch(a, b)]


# ── Colors ───────────────────────────────────────────────────────────

# Continent fill colors (muted/pastel shades for the territory regions)
CONTINENT_FILL_COLORS = {
    'North America':  '#C0392B',  # muted red
    'South America':  '#E67E22',  # muted orange
    'Africa':         '#D4A04A',  # muted yellow/gold
    'Europe':         '#2E86C1',  # muted blue
    'Asia':           '#27AE60',  # muted green
    'Australia':      '#8E44AD',  # muted purple
}

# Continent label colors (lighter versions for text overlay)
CONTINENT_LABEL_COLORS = {
    'North America':  '#E6B0AA',
    'South America':  '#F5CBA7',
    'Africa':         '#F9E79F',
    'Europe':         '#AED6F1',
    'Asia':           '#A9DFBF',
    'Australia':      '#D2B4DE',
}

# Player/owner circle colors (bright, distinct from continent fills)
PLAYER_COLORS = {
    0: '#F1C40F',   # bright yellow
    1: '#E74C3C',   # bright red
    2: '#95A5A6',   # silver/gray
    3: '#3498DB',   # bright blue
    4: '#2ECC71',   # bright green
    5: '#E67E22',   # bright orange
}

UNCLAIMED_COLOR = '#3a3a4e'
BG_COLOR = '#1a1a2e'
BOUNDARY_COLOR = '#DAA520'  # golden yellow for all boundaries


# ── Main draw function ───────────────────────────────────────────────

def draw_board(game, current_player=None, highlight=None):
    """Render the full Risk board as a matplotlib figure.

    Territory fill = continent color.
    Troop circle fill = owner color.
    All boundaries = yellow.
    Cross-ocean connections = dotted white lines.
    """
    if highlight is None:
        highlight = []
    highlight_set = set(highlight)

    # Build player -> color mapping for circles
    player_circle_color = {}
    for i, name in enumerate(game.turn_order):
        player_circle_color[name] = mcolors.to_rgb(PLAYER_COLORS.get(i, '#FFFFFF'))

    bg_rgb = mcolors.to_rgb(BG_COLOR)
    unclaimed_rgb = mcolors.to_rgb(UNCLAIMED_COLOR)

    # Territory fill = always continent color (regardless of ownership)
    idx_to_color = {0: bg_rgb}
    for name, idx in _NAME_TO_IDX.items():
        continent = TERRITORY_TO_CONTINENT.get(name)
        if continent:
            idx_to_color[idx] = mcolors.to_rgb(CONTINENT_FILL_COLORS.get(continent, '#555555'))
        else:
            idx_to_color[idx] = unclaimed_rgb

    # Build RGB image
    img = np.full((_MAP_HEIGHT, _MAP_WIDTH, 3), bg_rgb)
    for idx, rgb in idx_to_color.items():
        mask = _GRID == idx
        img[mask] = rgb

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(24, 13), dpi=150)
    ax.imshow(img, extent=[-0.5, _MAP_WIDTH - 0.5, -0.5, _MAP_HEIGHT - 0.5],
              aspect='equal', interpolation='nearest')
    ax.axis('off')
    fig.patch.set_facecolor(BG_COLOR)

    # Draw all boundary lines in yellow
    for (x1, y1), (x2, y2) in _ALL_BOUNDARY_SEGMENTS:
        ax.plot([x1, x2], [y1, y2], color=BOUNDARY_COLOR, linewidth=0.8,
                zorder=2, solid_capstyle='butt')

    # Draw cross-ocean dotted lines
    for a, b in CROSS_OCEAN:
        if a in TERRITORY_COORDS and b in TERRITORY_COORDS:
            xa, ya = TERRITORY_COORDS[a]
            xb, yb = TERRITORY_COORDS[b]
            ax.plot([xa, xb], [ya, yb], color='white', linewidth=1.5,
                    linestyle='--', alpha=0.7, zorder=3)

    # Highlight borders (brighter/thicker yellow)
    for name in highlight_set:
        if name in _TERRITORY_BOUNDARY_SEGMENTS:
            for (x1, y1), (x2, y2) in _TERRITORY_BOUNDARY_SEGMENTS[name]:
                ax.plot([x1, x2], [y1, y2], color='#FFFF00', linewidth=2.5,
                        zorder=6, solid_capstyle='butt')

    # (Continent labels removed for cleaner look)

    # Troop circles (owner-colored) and territory names
    for name, (cx, cy) in TERRITORY_COORDS.items():
        territory = game.world.territories.get(name)
        if territory is None:
            continue

        forces = territory.forces if territory.owner else 0
        if forces > 0:
            owner_color = player_circle_color.get(territory.owner.name, (0.5, 0.5, 0.5))
            # Dark outline ring
            outline = plt.Circle((cx, cy), 1.15, color='black', alpha=0.5, zorder=4)
            ax.add_patch(outline)
            # Owner-colored circle
            circle = plt.Circle((cx, cy), 1.0, color=owner_color, zorder=5)
            ax.add_patch(circle)
            # Troop count text
            ax.text(cx, cy, str(forces), fontsize=28, ha='center', va='center',
                    color='white', fontweight='bold', zorder=6,
                    path_effects=[_text_outline()])

        short = _short_name(name)
        ax.text(cx, cy - 1.3, short, fontsize=15, ha='center', va='top',
                color='#CCCCDD', zorder=5)

    # Legend (player colors)
    legend_handles = []
    for name in game.turn_order:
        p = game.players[name]
        color_hex = PLAYER_COLORS.get(game.turn_order.index(name), '#FFFFFF')
        label = f"{name} ({p.territory_count}t, {p.forces}f)"
        if not p.alive:
            label += " [dead]"
        legend_handles.append(mpatches.Patch(color=color_hex, label=label))
    ax.legend(handles=legend_handles, loc='lower left', fontsize=22,
              facecolor=BG_COLOR, edgecolor='#444466', labelcolor='white',
              framealpha=0.9)

    # Title
    title = f"Turn {game.turn}"
    if current_player:
        title += f" — {current_player}'s turn"
    ax.set_title(title, color='white', fontsize=36, fontweight='bold')

    plt.tight_layout()

    from IPython.display import display
    display(fig)
    plt.close(fig)


def _text_outline():
    """Return a path effect for dark text outline."""
    import matplotlib.patheffects as pe
    return pe.withStroke(linewidth=2, foreground='black')


def _short_name(name):
    """Abbreviate long territory names for display."""
    abbreviations = {
        'Northwest Territories': 'NW Terr',
        'Western United States': 'W US',
        'Eastern United States': 'E US',
        'Western Europe': 'W Europe',
        'Northern Europe': 'N Europe',
        'Southern Europe': 'S Europe',
        'South East Asia': 'SE Asia',
        'Western Australia': 'W Aust',
        'Eastern Australia': 'E Aust',
        'North Africa': 'N Africa',
        'East Africa': 'E Africa',
        'South Africa': 'S Africa',
        'Middle East': 'Mid East',
        'New Guinea': 'N Guinea',
    }
    return abbreviations.get(name, name)
