"""Cross-frequency bias-sensitivity comparison, push_fr3 replan-rate family.

One-off comparison across ``push_fr3/model_mismatch/replan_{2,5,10}hz_v1``
(not axis-generic like ``run.py``/``analysis.py`` -- task and leg names are
hardcoded). One subplot per replan rate, sharing a y-axis, each showing final
accumulated pose error vs. bias multiplier for every swept parameter, colored
as in :mod:`experiments.model_mismatch.analysis`. A leg with no ``results/``
yet, or an arm within a leg that hasn't finished (no ``tracking.csv``), is
silently omitted rather than erroring. Failed episodes (task_success == 0)
are excluded from the magnitude statistic and their failure rate annotated
instead -- see ``analysis.py``'s ``valid_mask``/``failure_rate``.

The legs compared are listed in ``LEGS``; a rate not named there is left
out of the comparison.

Styled for a conference figure: one shared legend instead of one per panel,
larger print-legible type, and both a raster (PNG, 300dpi) and vector (PDF)
export per stat.

Run (from the repo root)::

    uv run python -m experiments.model_mismatch.plot_replan_sensitivity
"""

from __future__ import annotations

import string

import numpy as np

from experiments.common.dr.plot_style import GRID, INK, MUTED, plt, style_axis
from experiments.model_mismatch.analysis import (
    EXPERIMENTS_ROOT,
    PARAM_COLOR,
    default_rot_scale,
    drop_catastrophic,
    failure_rate,
    final_accum_valid,
    load_all,
    run_bias_params,
    run_plot_params,
)
from experiments.model_mismatch.harness import parse_arm

LEGS = [
    ("replan_2hz_v1", "2 Hz"),
    ("replan_5hz_v1", "5 Hz"),
    ("replan_10hz_v1", "10 Hz"),
]
TASK_DIR = EXPERIMENTS_ROOT / "push_fr3" / "model_mismatch"
OUT = TASK_DIR / "replan_sensitivity"
OUT_MEANSTD = TASK_DIR / "replan_sensitivity_meanstd"

TITLE_SIZE = 13
LABEL_SIZE = 11
TICK_SIZE = 9.5
LEGEND_SIZE = 10.5
SUPTITLE_SIZE = 15


def facet(
    runs: dict,
    param: str,
    bias_params: tuple[str, ...],
    rot_scale: float,
    stat: str = "median",
) -> dict[str, np.ndarray]:
    """Centre +/- spread of final accumulated error vs. multiplier, one param.

    ``stat="median"`` centres on the median with an IQR spread (robust to the
    bimodal MJWarp flyaways); ``stat="mean"`` centres on the mean with a
    +/- std spread, kept only as the explicit comparison that justifies the
    median (a std reports how many runs flipped basin, not the noise in one).
    Failed episodes (task_success == 0 -- for some arms a numerically-invalid
    rollout, not "worse control", see :func:`experiments.model_mismatch.
    analysis.valid_mask`) are excluded from both; ``failure_rate()`` is
    returned per point instead of being baked into the magnitude.
    """
    nominal = (
        final_accum_valid(runs["nominal"], rot_scale)
        if "nominal" in runs else None
    )
    xs, centers, los, his, fails = [], [], [], [], []

    def stats(f: np.ndarray) -> tuple[float, float, float]:
        if stat == "mean":
            c, sd = np.nanmean(f), np.nanstd(f)
            return c, c - sd, c + sd
        c = np.nanmedian(f)
        return c, np.nanpercentile(f, 25), np.nanpercentile(f, 75)

    for name, data in runs.items():
        kind, params, mult = parse_arm(name, bias_params)
        if kind == "biased" and len(params) == 1 and params[0] == param:
            c, lo, hi = stats(final_accum_valid(data, rot_scale))
            xs.append(mult)
            centers.append(c)
            los.append(lo)
            his.append(hi)
            fails.append(failure_rate(data))
    if nominal is not None:  # the true model sits at multiplier 1.0
        c, lo, hi = stats(nominal)
        xs.append(1.0)
        centers.append(c)
        los.append(lo)
        his.append(hi)
        fails.append(failure_rate(runs["nominal"]))
    order = np.argsort(xs)
    xs_k, centers_k, los_k, his_k, fails_k, notes = drop_catastrophic(
        np.asarray(xs)[order], np.asarray(centers)[order],
        np.asarray(los)[order], np.asarray(his)[order],
        np.asarray(fails)[order],
    )
    return {
        "xs": xs_k, "centers": centers_k, "los": los_k, "his": his_k,
        "fails": fails_k, "catastrophic_notes": notes,
    }


def plot_sensitivity(
    legs: list[tuple[str, dict]],
    bias_params: tuple[str, ...],
    rot_scale: float,
    stat: str,
    out,
    title: str = "Model-mismatch sensitivity across replan rates",
    plot_params: tuple[str, ...] | None = None,
) -> None:
    """One subplot per leg (a replan rate, a truth level, ...), one y-axis.

    Styled for print: one legend shared across panels (not one per panel),
    a light reference line at the true model (multiplier 1.0), panel
    letters for text cross-referencing, and larger type throughout. ``title``
    defaults to this module's own replan-rate framing; a caller comparing a
    different kind of leg (e.g. truth-friction levels) should pass its own.

    ``plot_params`` (default: all of ``bias_params``) restricts which lines
    are drawn -- see ``analysis.bias_sensitivity``'s same parameter for the
    rationale (a swept-but-flat parameter, e.g. push_fr3's impratio, can be
    omitted from the figure without touching the underlying data).
    ``bias_params`` still classifies every arm name via ``facet``.
    """
    plot_params = bias_params if plot_params is None else plot_params
    facets = {
        (label, param): facet(runs, param, bias_params, rot_scale, stat)
        for label, runs in legs for param in plot_params
    }
    # Scale off the centres (medians/means) only, never the upper spread --
    # a handful of catastrophic (basin-flipped) episodes can push an IQR's
    # 75th percentile far off scale even when under half the episodes for
    # that arm diverged, which would otherwise flatten every healthy point
    # in the figure to a line. A spread that still runs past the axis after
    # that is simply clipped, not reported -- only failure_rate() is.
    # A center itself in a different regime is already gone by this point
    # (facet() -> drop_catastrophic()), so it can't set the scale either --
    # these figures answer "how sensitive is a REASONABLE mismatch."
    center_vals = [
        v for d in facets.values() for v in d["centers"] if np.isfinite(v)
    ]
    ymax = 1.2 * max(center_vals) if center_vals else 1.0

    fig, axes = plt.subplots(
        1, len(legs), figsize=(4.0 * len(legs), 4.3), sharey=True,
    )
    axes = np.atleast_1d(axes)
    for i, (ax, (label, runs)) in enumerate(zip(axes, legs)):
        style_axis(ax)
        ax.axvline(1.0, color=GRID, linewidth=1.4, zorder=0)
        fail_notes = []
        for param in plot_params:
            d = facets[(label, param)]
            yerr = np.abs(
                np.vstack([d["centers"] - d["los"], d["his"] - d["centers"]])
            )
            ax.errorbar(
                d["xs"], d["centers"], yerr=yerr, color=PARAM_COLOR[param],
                marker="o", ms=6.5, markeredgecolor="white",
                markeredgewidth=0.6, capsize=4, capthick=1.3, elinewidth=1.3,
                linewidth=2.1, label=param,
            )
            for x, fr in zip(d["xs"], d["fails"]):
                if fr > 0:
                    fail_notes.append(f"{param}@{x:g}x: {fr:.0%} failed")
            fail_notes += [f"{param}{n}" for n in d["catastrophic_notes"]]
        if fail_notes:
            ax.text(
                0.97, 0.97, "\n".join(fail_notes),
                transform=ax.transAxes, ha="right", va="top", fontsize=6.5,
                color=MUTED,
            )
        ax.set_xscale("log")
        ax.set_ylim(0.0, ymax)
        ax.tick_params(labelsize=TICK_SIZE)
        ax.set_title(
            f"({string.ascii_lowercase[i]}) {label}",
            fontsize=TITLE_SIZE, color=INK, pad=9,
        )
    axes[0].set_ylabel(
        "final accumulated pose error [m$\\cdot$s]",
        fontsize=LABEL_SIZE, color=INK,
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.0),
        ncol=len(bias_params), fontsize=LEGEND_SIZE, frameon=False,
        handlelength=1.6, columnspacing=1.6,
    )
    spread = "mean $\\pm$ std" if stat == "mean" else "median + IQR"
    fig.suptitle(
        f"{title} ({spread})", fontsize=SUPTITLE_SIZE, color=INK, y=1.1,
    )
    fig.supxlabel(
        f"{'/'.join(plot_params)} multiplier ($\\times$ truth, "
        f"rot-scale {rot_scale:g} m/rad)",
        fontsize=LABEL_SIZE, color=INK, y=-0.02,
    )
    fig.tight_layout()
    for ext, dpi in ((".png", 300), (".pdf", None)):
        fig.savefig(out.with_suffix(ext), dpi=dpi, bbox_inches="tight")
        print(f"wrote {out.with_suffix(ext)}")
    plt.close(fig)


def main() -> None:
    """Load whatever replan-rate legs have results and plot side by side."""
    legs, first_dir = [], None
    for name, label in LEGS:
        results = TASK_DIR / name / "results"
        if not results.is_dir():
            print(f"  (skip {name}: no results/ yet)")
            continue
        legs.append((label, load_all(results)))
        if first_dir is None:
            first_dir = TASK_DIR / name
    if not legs:
        raise SystemExit(f"no replan-rate legs with results under {TASK_DIR}")

    bias_params = run_bias_params(legs[0][1])
    plot_params = run_plot_params(legs[0][1], bias_params, first_dir)
    rot_scale = default_rot_scale(legs[0][1])

    plot_sensitivity(
        legs, bias_params, rot_scale, "median", OUT,
        plot_params=plot_params,
    )
    plot_sensitivity(
        legs, bias_params, rot_scale, "mean", OUT_MEANSTD,
        plot_params=plot_params,
    )


if __name__ == "__main__":
    main()
