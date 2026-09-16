"""Plots for the plan-time scaling benchmark. Reads ``run.py``'s CSVs only.

For each sweep, writes two variants -- single-log (log-x, linear-y) and
log-log (log-x, log-y) -- so the two can be compared side by side. Every
plot is annotated with horizontal dashed lines at the 10/20/30 Hz
real-time replan budgets (100/50/33.3 ms); crossing values are reported
in the paper's table, not burned into the figure.

Run (from the repo root)::

    uv run python -m experiments.scaling.analysis
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent / "results"

BG = "#ffffff"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
TEAL = "#0f8fa3"
MUTED = "#52514e"
TICK = "#898781"
SPINE = "#c3c2b7"
GRID = "#e1e0d9"

PRACTICE_COLORS = {
    "push_fr3": BLUE,
    "balance_fr3": ORANGE,
    "curling_fr3": TEAL,
}

# (label, budget in ms) -- the replan deadline at each control rate.
BUDGETS_HZ = (
    ("30 Hz", 1000.0 / 30),
    ("20 Hz", 1000.0 / 20),
    ("10 Hz", 1000.0 / 10),
)


def _style_axis(ax: plt.Axes) -> None:
    ax.set_facecolor(BG)
    ax.tick_params(colors=TICK, labelsize=10.5)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(SPINE)
        ax.spines[spine].set_linewidth(1.2)
    ax.yaxis.grid(True, color=GRID, linewidth=0.9)
    ax.set_axisbelow(True)


def _read_csv(path: Path) -> list[dict]:
    with path.open() as f:
        rows = []
        for row in csv.DictReader(f):
            rows.append({
                k: v if k == "task" else float(v) for k, v in row.items()
            })
        return rows


def _add_horizontal_budgets(
    ax: plt.Axes, ymin: float, ymax: float, *, logy: bool,
) -> None:
    """Draw budget lines, nudging label positions apart when they'd overlap.

    30/20/10 Hz are only 16.7 ms apart (33.3/50/100 ms) -- close enough
    that at a wide y-range their text collides. Labels are placed in a
    normalized (log for a log axis) y-coordinate with a minimum spacing,
    then mapped back to data coordinates.
    """

    def _norm(v: float) -> float:
        if logy:
            return (math.log(v) - math.log(ymin)) / (
                math.log(ymax) - math.log(ymin)
            )
        return (v - ymin) / (ymax - ymin)

    def _denorm(f: float) -> float:
        if logy:
            return math.exp(
                math.log(ymin) + f * (math.log(ymax) - math.log(ymin))
            )
        return ymin + f * (ymax - ymin)

    present = [
        (label, ms) for label, ms in BUDGETS_HZ if ymin <= ms <= ymax
    ]
    order = sorted(present, key=lambda t: _norm(t[1]))
    min_gap = 0.06
    placed: list[tuple[str, float, float]] = []
    prev_f = None
    for label, ms in order:
        f = _norm(ms)
        if prev_f is not None and f - prev_f < min_gap:
            f = prev_f + min_gap
        placed.append((label, ms, f))
        prev_f = f

    for label, ms, f in placed:
        ax.axhline(ms, color=MUTED, linestyle="--", linewidth=1.3,
                    alpha=0.75, zorder=1)
        ax.annotate(
            label,
            xy=(1, _denorm(f)),
            xycoords=("axes fraction", "data"),
            xytext=(4, 0),
            textcoords="offset points",
            va="center", ha="left", fontsize=9.5, color=MUTED,
            fontweight="bold",
        )


def _plot(
    series: list[tuple[str, str, list[dict]]],
    x_key: str,
    xlabel: str,
    out_path: Path,
    *,
    logy: bool,
) -> None:
    """One figure; ``series`` is ``[(label, color, rows), ...]``."""
    fig, ax = plt.subplots(figsize=(6.8, 5.3), facecolor=BG)

    all_x: list[float] = []
    all_lo: list[float] = []
    all_hi: list[float] = []
    for label, color, rows in series:
        x = [r[x_key] for r in rows]
        mean_ms = [r["mean_ms"] for r in rows]
        std_ms = [r["std_ms"] for r in rows]
        lo = [m - s for m, s in zip(mean_ms, std_ms, strict=True)]
        hi = [m + s for m, s in zip(mean_ms, std_ms, strict=True)]
        ax.plot(
            x, mean_ms, color=color, marker="o", markersize=6.5,
            markeredgecolor=BG, markeredgewidth=1.3,
            linewidth=2.6, label=label, zorder=3,
        )
        ax.fill_between(x, lo, hi, color=color, alpha=0.20, linewidth=0,
                         zorder=2)
        all_x += x
        all_lo += lo
        all_hi += hi

    ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
        ymin, ymax = min(all_lo) * 0.85, max(all_hi) * 1.15
    else:
        ymin, ymax = 0.0, max(all_hi) * 1.1
    ax.set_ylim(ymin, ymax)

    _add_horizontal_budgets(ax, ymin, ymax, logy=logy)

    ax.set_xlabel(xlabel, color=MUTED, fontsize=12)
    ax.set_ylabel("plan time per replan (ms)", color=MUTED, fontsize=12)
    _style_axis(ax)
    if len(series) > 1:
        legend = ax.legend(
            frameon=True, fontsize=9,
            loc="lower right" if logy else "upper right",
            handlelength=1.4, handletextpad=0.5, labelspacing=0.3,
            borderpad=0.4, facecolor=BG, edgecolor=SPINE,
            framealpha=0.95,
        )
        for text in legend.get_texts():
            text.set_color(MUTED)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, facecolor=BG)
    plt.close(fig)


def _plot_both_scales(
    series: list[tuple[str, str, list[dict]]],
    x_key: str,
    xlabel: str,
    stem: str,
    results: Path,
) -> None:
    _plot(
        series, x_key, xlabel,
        results / f"{stem}_singlelog.pdf", logy=False,
    )
    _plot(
        series, x_key, xlabel,
        results / f"{stem}_loglog.pdf", logy=True,
    )


def _by_task(rows: list[dict]) -> list[tuple[str, str, list[dict]]]:
    """One series per task present in ``rows``.

    A task missing from the CSV is skipped rather than drawn empty -- the
    runner's ``--tasks`` can time any subset.
    """
    series = []
    for name, color in PRACTICE_COLORS.items():
        task_rows = [r for r in rows if r["task"] == name]
        if task_rows:
            series.append((name, color, task_rows))
    return series


def main() -> None:
    """Read every CSV and write single-log + log-log variants of each plot."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["full", "smoke"], default="full")
    args = parser.parse_args()
    results = RESULTS_DIR if args.mode == "full" else RESULTS_DIR / args.mode
    _plot_both_scales(
        [("push", BLUE, _read_csv(results / "nworld_sweep.csv"))],
        x_key="nworld",
        xlabel="nworld",
        results=results,
        stem="plan_time_vs_samples",
    )
    _plot_both_scales(
        [("push", BLUE, _read_csv(results / "horizon_sweep.csv"))],
        x_key="horizon_steps",
        xlabel="horizon (steps)",
        results=results,
        stem="plan_time_vs_horizon",
    )

    _plot_both_scales(
        [("push, H=1", BLUE, _read_csv(results / "isolated_samples.csv"))],
        x_key="nworld",
        xlabel="nworld",
        results=results,
        stem="plan_time_vs_samples_isolated",
    )
    _plot_both_scales(
        [("push, S=1", BLUE, _read_csv(results / "isolated_horizon.csv"))],
        x_key="horizon_steps",
        xlabel="horizon (steps)",
        results=results,
        stem="plan_time_vs_horizon_isolated",
    )

    _plot_both_scales(
        _by_task(_read_csv(results / "practice_samples_sweep.csv")),
        x_key="nworld",
        xlabel="nworld",
        results=results,
        stem="plan_time_vs_samples_practice",
    )
    _plot_both_scales(
        _by_task(_read_csv(results / "practice_horizon_sweep.csv")),
        x_key="horizon_steps",
        xlabel="horizon (steps)",
        results=results,
        stem="plan_time_vs_horizon_practice",
    )

    for stem in (
        "plan_time_vs_samples", "plan_time_vs_horizon",
        "plan_time_vs_samples_isolated", "plan_time_vs_horizon_isolated",
        "plan_time_vs_samples_practice", "plan_time_vs_horizon_practice",
    ):
        for suffix in ("singlelog", "loglog"):
            print(f"wrote {results / f'{stem}_{suffix}.pdf'}")


if __name__ == "__main__":
    main()
