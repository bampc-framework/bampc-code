"""Shared matplotlib styling for sweep summary plots (headless Agg).

Import only from sweep ``plots.py`` modules -- matplotlib stays out of
the library import path.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

BG = "#fcfcfb"
BLUE = "#2a78d6"
GREEN = "#1baf7a"
AMBER = "#eda100"
RED = "#e34948"
TEAL = "#0fb5c4"
PURPLE = "#7a3fb0"
INK = "#0b0b0b"
MUTED = "#52514e"
TICK = "#898781"
SPINE = "#c3c2b7"
GRID = "#e1e0d9"


def style_axis(ax: plt.Axes) -> None:
    """Shared muted styling for one axis."""
    ax.set_facecolor(BG)
    ax.tick_params(colors=TICK)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(SPINE)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
