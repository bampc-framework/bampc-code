"""A clean, submission-ready bar figure of the friction_id headline result.

Distinct from ``analysis.py``'s own ``headline_fig`` (which uses this repo's
muted internal debugging palette): this targets a paper figure directly --
white background, high-contrast colorblind-safe palette, serif-free clean
type, vector PDF output alongside a PNG preview. Reuses ``analysis.load_all``
for the data rather than re-deriving anything.

Two panels stacked in one narrow column, one bar per (algo, arm): average
miss distance over a campaign above the best (minimum) miss distance --
the two numbers the experiment exists to compare across arms. Median
across seeds; error bars are the IQR (25th-75th percentile), the same
robust-to-non-determinism convention ``analysis.py`` uses.

Run (from the repo root)::

    ...plot_paper --version curling_fr3/friction_id/v1
    ...plot_paper --version curling_fr3/friction_id/v1 --mode quick

where ``...plot_paper`` is::

    uv run python -m experiments.friction_id.plot_paper
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from experiments.friction_id.analysis import (  # noqa: E402
    EXPERIMENTS_ROOT,
    arm_order,
    load_all,
)

# A qualitative, colorblind-safe triple (Wong 2011 palette).
ARM_COLOR = {
    "wrong_estimate": "#E69F00",  # orange
    "wide_hedge": "#56B4E9",      # sky blue
    "adaptive": "#009E73",        # bluish green
}
ARM_LABEL = {
    "wrong_estimate": "Wrong estimate",
    "wide_hedge": "Wide hedge",
    "adaptive": "Adaptive (ours)",
}
ALGO_LABEL = {"mppi": "MPPI", "ps": "PS", "cem": "CEM"}


def _style() -> None:
    """Minimal, print-friendly rcParams -- set once, scoped to this script."""
    plt.rcParams.update({
        "font.size": 10,
        "font.family": "sans-serif",
        "axes.edgecolor": "#333333",
        "axes.labelcolor": "#111111",
        "text.color": "#111111",
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "axes.linewidth": 0.8,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })


def _bars(ax: plt.Axes, runs: dict, key: str, ylabel: str, arms: list[str]
           ) -> None:
    """One panel: grouped bars over algos, one group of bars per arm."""
    algos = list(runs)
    width = 0.8 / max(len(arms), 1)
    x = np.arange(len(algos))
    for i, arm in enumerate(arms):
        vals = [
            runs[algo][arm][key] if arm in runs[algo] else np.array([np.nan])
            for algo in algos
        ]
        centers = np.array([np.nanmedian(v) for v in vals])
        los = np.array([np.nanpercentile(v, 25) for v in vals])
        his = np.array([np.nanpercentile(v, 75) for v in vals])
        offset = (i - (len(arms) - 1) / 2) * width
        ax.bar(
            x + offset, centers, width=width * 0.92,
            yerr=np.vstack([centers - los, his - centers]),
            color=ARM_COLOR[arm], label=ARM_LABEL[arm],
            capsize=2.5, linewidth=0.6, edgecolor="white",
            error_kw={"linewidth": 0.9, "ecolor": "#333333"},
        )
    ax.set_xticks(x)
    ax.set_xticklabels([ALGO_LABEL.get(a, a) for a in algos])
    ax.set_ylabel(ylabel)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, color="#dddddd", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.margins(x=0.08)


def headline_paper_fig(runs: dict, out_path: Path) -> None:
    """Two-panel bar figure, stacked: average miss above best miss.

    A single narrow column (one figure width, two short panels) rather than
    a wide side-by-side pair -- fits a paper's column width directly, and
    each panel is shorter than a standalone plot since only the bars, not
    axis chrome, need the vertical space.
    """
    arms = arm_order(runs)
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(3.4, 3.7), sharex=True,
        gridspec_kw={"hspace": 0.45, "height_ratios": [1, 1]},
    )
    _bars(ax1, runs, "avg_miss", "Average miss (m)", arms)
    ax1.set_title("(a) Average over campaign", fontsize=9)
    ax1.tick_params(axis="x", labelbottom=False)
    _bars(ax2, runs, "best_miss", "Best miss (m)", arms)
    ax2.set_title("(b) Best over campaign", fontsize=9)
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", ncol=3, frameon=False,
        bbox_to_anchor=(0.5, 1.04), fontsize=7.5, columnspacing=1.0,
        handlelength=1.4, handletextpad=0.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Load the version's results and write the paper figure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--version", required=True,
        help="Path relative to experiments/, e.g. curling_fr3/friction_id/v1.",
    )
    parser.add_argument(
        "--mode", choices=["full", "quick", "smoke"], default="full",
        help="Which results/ subtree to read (see run.py --mode).",
    )
    args = parser.parse_args()

    _style()
    results = EXPERIMENTS_ROOT / args.version / "results"
    if args.mode != "full":
        results = results / args.mode
    runs = load_all(results)
    out_path = results / "headline_paper"
    headline_paper_fig(runs, out_path)
    print(f"wrote {out_path.with_suffix('.pdf')} and .png")


if __name__ == "__main__":
    main()
