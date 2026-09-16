"""Cross-rate x cross-truth-level model-mismatch summary, one parameter.

Each ``<param>_truth_<rate>hz_v1`` sibling under a task's ``model_mismatch/``
answers "does sensitivity to a wrong ``<param>`` belief hold across a couple
of plausible real-world ``<param>`` values, AT THIS RATE" (see
``plot_truth_grid.py``). This summarizes the whole family for one parameter
at once: one panel per truth level (as ``plot_truth_grid.py``), but now
overlaying every discovered rate as its own line in each panel, so "does the
truth-level story also hold across replan rates" is one figure instead of a
manual cross-check across N separate ones.

Each rate leg runs at its own real-time sample budget (see a family's
``<param>_truth_<rate>hz_v1`` headers), so a line here differs from its
neighbours in replan rate *and* in samples per plan. That is deliberate --
it is what each rate would actually deploy on one GPU -- but it means this
figure answers "does the story hold across replan rates as operated", not
"...at fixed compute".

Rate dirs are discovered by name (``<param>_truth_<digits>hz_v1``) rather
than hardcoded, so a new rate leg shows up automatically. A rate whose
``results/`` (or ``results/<mode>/``) doesn't exist yet is skipped with a
console note, not an error -- this is meant to summarize whatever is
complete, not block on the slowest leg.

Run (from the repo root)::

    ...plot_truth_grid_summary --task push_fr3 --param mass
    ...plot_truth_grid_summary --task balance_fr3 --param mass --mode quick

where ``...plot_truth_grid_summary`` is::

    uv run python -m experiments.model_mismatch.plot_truth_grid_summary
"""

from __future__ import annotations

import argparse
import re
import string
from pathlib import Path

import numpy as np

from experiments.common.dr.plot_style import (
    BLUE,
    GREEN,
    GRID,
    INK,
    PURPLE,
    RED,
    TEAL,
    plt,
    style_axis,
)
from experiments.model_mismatch.analysis import (
    EXPERIMENTS_ROOT,
    default_rot_scale,
    load_truth_grid,
    run_bias_params,
)
from experiments.model_mismatch.plot_replan_sensitivity import facet

RATE_COLORS = [BLUE, TEAL, PURPLE, RED, GREEN]
TITLE_SIZE = 13
LABEL_SIZE = 11
TICK_SIZE = 9.5
LEGEND_SIZE = 10.5
SUPTITLE_SIZE = 15


def _discover_rate_dirs(
    task_dir: Path, param: str
) -> list[tuple[float, Path]]:
    """Every ``<param>_truth_<rate>hz_v1`` dir for this param, by rate."""
    pattern = re.compile(rf"^{re.escape(param)}_truth_(\d+(?:\.\d+)?)hz_v1$")
    out = []
    for d in task_dir.iterdir():
        m = pattern.match(d.name)
        if m and d.is_dir():
            out.append((float(m.group(1)), d))
    return sorted(out)


def _truth_label(dirname: str) -> str:
    """``"truth_1.2"`` -> ``"1.2x truth"``."""
    return f"{dirname[len('truth_'):]}x truth"


def main() -> None:  # noqa: PLR0912
    """Discover every rate leg for ``--task``/``--param`` and plot them."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task", required=True, choices=["push_fr3", "balance_fr3"],
    )
    parser.add_argument(
        "--param", required=True,
        help="Swept parameter, e.g. mass, friction, rolling_friction.",
    )
    parser.add_argument(
        "--mode", choices=["full", "smoke", "quick"], default="full",
        help="Which results/ subtree to read on each rate leg.",
    )
    args = parser.parse_args()

    task_dir = EXPERIMENTS_ROOT / args.task / "model_mismatch"
    rate_dirs = _discover_rate_dirs(task_dir, args.param)
    if not rate_dirs:
        raise SystemExit(
            f"no {args.param}_truth_<rate>hz_v1 dirs under {task_dir}"
        )

    per_rate = []
    for rate, d in rate_dirs:
        results = d / "results"
        if args.mode != "full":
            results = results / args.mode
        if not results.is_dir():
            print(f"  (skip {d.name}: no results/ for --mode {args.mode} yet)")
            continue
        try:
            grid = load_truth_grid(results)
        except SystemExit:
            print(f"  (skip {d.name}: results/ present but empty)")
            continue
        per_rate.append((rate, grid))
    if not per_rate:
        raise SystemExit("no complete rate legs found")

    truth_names = sorted(
        {t for _, grid in per_rate for t in grid},
        key=lambda n: float(n[len("truth_"):]),
    )
    any_runs = next(iter(per_rate[0][1].values()))
    bias_params = run_bias_params(any_runs)
    rot_scale = default_rot_scale(any_runs)

    out_dir = task_dir if args.mode == "full" else task_dir / args.mode
    out_dir.mkdir(exist_ok=True)
    stats = (("median", ""), ("mean", "_meanstd"))
    for stat, suffix in stats:
        out = out_dir / f"truth_grid_summary_{args.param}{suffix}"
        _plot(
            per_rate, truth_names, args.param, bias_params, rot_scale, stat,
            out,
        )


def _plot(
    per_rate: list[tuple[float, dict]],
    truth_names: list[str],
    param: str,
    bias_params: tuple[str, ...],
    rot_scale: float,
    stat: str,
    out: Path,
) -> None:
    """One panel per truth level, one line per rate."""
    # facets[(rate, truth_name)] -- reuses facet()'s own drop_catastrophic()
    # filtering, so one rate's basin-flip can't set the scale here either.
    facets = {}
    for rate, grid in per_rate:
        for truth_name in truth_names:
            runs = grid.get(truth_name)
            if runs is None:
                continue
            facets[(rate, truth_name)] = facet(
                runs, param, bias_params, rot_scale, stat
            )
    center_vals = [
        v for d in facets.values() for v in d["centers"] if np.isfinite(v)
    ]
    ymax = 1.2 * max(center_vals) if center_vals else 1.0

    fig, axes = plt.subplots(
        1, len(truth_names), figsize=(4.2 * len(truth_names), 4.5),
        sharey=True,
    )
    axes = np.atleast_1d(axes)
    rate_labels = [f"{rate:g} Hz" for rate, _ in per_rate]
    for i, (ax, truth_name) in enumerate(zip(axes, truth_names)):
        style_axis(ax)
        ax.axvline(1.0, color=GRID, linewidth=1.4, zorder=0)
        notes = []
        for (rate, _), color, label in zip(per_rate, RATE_COLORS, rate_labels):
            d = facets.get((rate, truth_name))
            if d is None or d["xs"].size == 0:
                continue
            yerr = np.abs(
                np.vstack([d["centers"] - d["los"], d["his"] - d["centers"]])
            )
            ax.errorbar(
                d["xs"], d["centers"], yerr=yerr, color=color,
                marker="o", ms=6, markeredgecolor="white",
                markeredgewidth=0.5, capsize=3.5, capthick=1.2,
                elinewidth=1.2, linewidth=1.9, label=label,
            )
            notes += [f"{label}{n}" for n in d["catastrophic_notes"]]
            for x, fr in zip(d["xs"], d["fails"]):
                if fr > 0:
                    notes.append(f"{label}@{x:g}x: {fr:.0%} failed")
        if notes:
            ax.text(
                0.97, 0.97, "\n".join(notes), transform=ax.transAxes,
                ha="right", va="top", fontsize=6.5, color=RED,
            )
        ax.set_xscale("log")
        ax.set_ylim(0.0, ymax)
        ax.tick_params(labelsize=TICK_SIZE)
        ax.set_xlabel(f"{param} multiplier (x that truth level)",
                       fontsize=LABEL_SIZE, color=INK)
        ax.set_title(
            f"({string.ascii_lowercase[i]}) {_truth_label(truth_name)}",
            fontsize=TITLE_SIZE, color=INK, pad=9,
        )
    axes[0].set_ylabel(
        "final accumulated pose error [m$\\cdot$s]",
        fontsize=LABEL_SIZE, color=INK,
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.0),
        ncol=len(per_rate), fontsize=LEGEND_SIZE, frameon=False,
        handlelength=1.6, columnspacing=1.6,
    )
    spread = "mean $\\pm$ std" if stat == "mean" else "median + IQR"
    fig.suptitle(
        f"{param}-mismatch sensitivity across rates and real-world "
        f"{param} levels ({spread})",
        fontsize=SUPTITLE_SIZE, color=INK, y=1.12,
    )
    fig.tight_layout()
    for ext, dpi in ((".png", 300), (".pdf", None)):
        fig.savefig(out.with_suffix(ext), dpi=dpi, bbox_inches="tight")
        print(f"wrote {out.with_suffix(ext)}")
    plt.close(fig)


if __name__ == "__main__":
    main()
