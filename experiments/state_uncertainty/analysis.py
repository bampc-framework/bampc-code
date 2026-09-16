"""Figures and metrics for a state-uncertainty performance sweep.

Reads what ``run.py`` wrote and owns **every** derived number, so a metric can
change without re-running the GPU. Shared across tasks (Push-FR3, Balance-FR3,
...); a task's own manifest (``data["manifest"]["task"]``) drives the handful
of task-specific branches -- see below.

**The score is goal-to-object pose error on SO(3) x R^3.** Position and
rotation are measured in their own spaces and combined by one explicit scale
in metres per radian -- never summed as if radians and metres were
commensurable. ``--rot-scale`` sets it; the default is the run's own
``w_orient / w_pos``, i.e. the rate the planner was optimising under, so the
score and the controller agree unless the rate is changed on purpose. A task
with no orientation goal (Balance-FR3) logs an all-NaN ``orient_err`` column,
which degrades every metric here to position-only automatically -- see
``load_variant``'s ``rotation()`` helper. The scale appears in the title of
every figure that depends on it.

1. ``pose_error_accumulated`` -- the headline. A controller briefly worse
   that then recovers is not the same as one steadily worse, and only the
   integral separates them.
2. ``pose_error_components`` -- the same two channels on their OWN axes, in
   native units. Skipped for a position-only task (nothing to show).
3. ``cost_over_time`` / ``cost_accumulated`` -- **diagnostic, not the score.**
   Cost is what the planner optimised and carries shaping terms that exist to
   help the planner rather than to measure the task.
4. ``estimate_error`` -- the error of the state each arm handed its planner,
   under the same metric. Without this a score difference cannot be
   attributed: it separates "this estimator is better" from "this planner
   coped better".

   Read the ensemble arms carefully here. What they hand the planner *is* the
   raw reading -- the cloud is drawn around it inside the planner -- so they
   sit exactly on the ``naive`` line by construction, not by coincidence. An
   ensemble does not improve the point estimate; it hedges over it.

**Median and IQR over the episodes per cell, never mean and std.** These runs
are bimodal -- MJWarp's non-determinism makes trajectories fall into a few
discrete basins rather than diffuse -- so
a standard deviation reports how many runs flipped basin, not how noisy any
one run is.

**Push-FR3 only**: a fixed reachable-box failure concept (``REACH_X``/
``REACH_Y``, "the block left the arm's reach") drives the mean+/-std facets'
failure exclusion, ``failures_table`` and ``lost_block_table``. No other task
defines an equivalent geometric failure mode today, so these are gated behind
``task == "push_fr3"`` in ``main``, not run unconditionally.

The summary table is **paired**: every arm is differenced against the oracle
at the same ``(seed, repeat)``, which cancels the scenario difficulty and the
basin draw that dominate the raw spread.

``--version`` is a path relative to ``experiments/``, e.g.
``push_fr3/state_uncertainty/t-ou/scale0.6-tau2.0-warmup50``.

Run (from the repo root)::

    uv run python -m experiments.state_uncertainty.analysis \
        --version push_fr3/state_uncertainty/t-ou/scale0.6-tau2.0-warmup50
    ...analysis --version <version> --mode smoke
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from experiments.common.uncertainty.plot_style import (
    AMBER,
    BLUE,
    GREEN,
    INK,
    MUTED,
    PURPLE,
    RED,
    TEAL,
    plt,
    style_axis,
)

EXPERIMENTS_ROOT = Path(__file__).resolve().parent.parent
ALGO_ORDER = ("ps", "cem", "mppi")

# Related arms share a hue and split on linestyle, so a figure reads as a
# handful of comparisons rather than a dozen unrelated curves. The full set
# is listed so a config that adds an arm still colours it; arms not present
# in a given run are skipped.
ARM_STYLE = {
    "oracle": (INK, "-", "oracle (true state)"),
    "naive": (RED, "-", "naive (raw reading)"),
    "point": (AMBER, "-", "point (Kalman)"),
    "point_debiased": (AMBER, "--", "point, debiased (privileged)"),
    "point_fast": (GREEN, "-", "point, faster rate"),
    # Same filter as `point`, but meas_cov keeps adapting online from each
    # step's innovation instead of staying fixed after warm-up. TEAL, not
    # amber: it isn't a sizing variant of `point`, it's a different filter.
    "point_adaptive": (TEAL, "-", "point, online-adaptive R"),
    # Dotted and BLUE, not amber: it shares the ensembles' engine shape, and
    # its whole job is to be read against them.
    "point_narrow": (BLUE, ":", "point, ensemble shape (no spread)"),
    "ensemble_exact": (BLUE, "-", "ensemble, sigma exact"),
    "ensemble_wide": (BLUE, "--", "ensemble, sigma overstated"),
    "ensemble_wider": (BLUE, "-.", "ensemble, sigma 2.5x"),
    # push_t_fr3_velocity_v1's clean sigma ladder (plain arms, no risk fan).
    # Reuses wide/wider's dash pattern since that pair isn't in this run's
    # roster -- same "increasing sigma" convention, cleaner multipliers.
    "ensemble_double": (BLUE, "--", "ensemble, sigma 2x"),
    "ensemble_triple": (BLUE, "-.", "ensemble, sigma 3x"),
    # Centred on the raw reading, not a filter -- RED like naive, since
    # that's the estimate family this reads against, not the BLUE sigma
    # ladder above (which centres on the Kalman filter).
    "ensemble_fixed_naive": (RED, "--", "ensemble, fixed spread on naive"),
    "ensemble_fixed_naive_double": (
        RED, "-.", "ensemble, fixed spread x2 on naive"),
    "ensemble_fixed_naive_triple": (
        RED, ":", "ensemble, fixed spread x3 on naive"),
    # Member-count ablation: ensemble_exact (1x sigma) at a smaller R than
    # the sweep's own belief.num_domains default -- PURPLE, not BLUE, so it
    # never collides with the sigma ladder's BLUE dash pattern when both
    # appear in the same plot (R and sigma are orthogonal axes here).
    "ensemble_double_r4": (PURPLE, ":", "ensemble, R=4, sigma 2x"),
    "ensemble_double_r8": (PURPLE, "--", "ensemble, R=8, sigma 2x"),
    "ensemble_double_r16": (PURPLE, "-.", "ensemble, R=16, sigma 2x"),
    "ensemble_double_r32": (PURPLE, "-", "ensemble, R=32, sigma 2x"),
    # Same R=32 as above, but at 3x sigma -- kept PURPLE (same R-ladder
    # family) with a distinct dash-dot-dot pattern since "-"/":"/"--"/"-."
    # are already claimed by the double-sigma R-ladder above.
    "ensemble_triple_r32": (
        PURPLE, (0, (3, 1, 1, 1)), "ensemble, R=32, sigma 3x"
    ),
    "ensemble_triple_r32-risk_average": (
        PURPLE, (0, (3, 1, 1, 1)), "ensemble, R=32, sigma 3x"
    ),
    # Same R-ladder family at 5x sigma -- a fourth pattern, distinct
    # from the three above.
    "ensemble_quintuple_r32": (
        PURPLE, (0, (1, 1)), "ensemble, R=32, sigma 5x"
    ),
    "ensemble_octuple_r32": (
        PURPLE, (0, (5, 1)), "ensemble, R=32, sigma 8x"
    ),
    # Risk-fanned variants of the triple_r32 high-noise ensemble (which
    # runs average risk) -- TEAL, like the other risk-fan pairs
    # above, since it's a planner-risk ablation, not a sizing variant.
    "ensemble_triple_r32-risk_cvar_0.5": (
        TEAL, "-", "ensemble, R=32, sigma 3x, CVaR (worst half)"),
    "ensemble_triple_r32-risk_inverse_cvar_0.5": (
        TEAL, "--", "ensemble, R=32, sigma 3x, inverse CVaR (best half)"),
    "ensemble_triple_r32-risk_cvar_0.2": (
        TEAL, ":", "ensemble, R=32, sigma 3x, CVaR (worst 20%)"),
    "ensemble_triple_r32-risk_inverse_cvar_0.2": (
        TEAL, "-.", "ensemble, R=32, sigma 3x, inverse CVaR (best 20%)"),
    "ensemble_triple_r32-risk_worstcase": (
        TEAL, (0, (1, 1)), "ensemble, R=32, sigma 3x, worst case"),
    "ensemble_triple_r32-risk_bestcase": (
        TEAL, (0, (5, 1)), "ensemble, R=32, sigma 3x, best case"),
    # Spread by the filter's INNOVATION covariance instead of its posterior
    # -- TEAL, like point_adaptive, since it shares that filter's adapt_rate
    # rather than being a sigma-sizing variant of ensemble_exact.
    "ensemble_adaptive_small": (TEAL, ":", "ensemble, adaptive R (small)"),
    "ensemble_adaptive_medium": (TEAL, "--", "ensemble, adaptive R (medium)"),
    "ensemble_adaptive_big": (TEAL, "-.", "ensemble, adaptive R (big)"),
    # Risk-fanned variants (loader.py's `_fan_risk`): TEAL, since every BLUE
    # linestyle is already spoken for by exact/wide/wider/point_narrow.
    "ensemble_double-risk_cvar_0.5": (
        TEAL, "-", "ensemble, sigma 2x, CVaR (worst half)"),
    "ensemble_double-risk_inverse_cvar_0.5": (
        TEAL, "--", "ensemble, sigma 2x, inverse CVaR (best half)"),
    "ensemble_exact-risk_average": (BLUE, "-", "ensemble exact, average"),
    "ensemble_exact-risk_worstcase": (BLUE, "--", "ensemble exact, worst case"),
    "ensemble_exact-risk_cvar_0.25": (BLUE, ":", "ensemble exact, cvar 0.25"),
    "ensemble_exact-risk_cvar_0.5": (BLUE, "-.", "ensemble exact, cvar 0.5"),
    "ensemble_wide-risk_average": (TEAL, "-", "ensemble wide, average"),
    "ensemble_wide-risk_worstcase": (TEAL, "--", "ensemble wide, worst case"),
    "ensemble_wide-risk_cvar_0.25": (TEAL, ":", "ensemble wide, cvar 0.25"),
    "ensemble_wide-risk_cvar_0.5": (TEAL, "-.", "ensemble wide, cvar 0.5"),
    # Green, not blue: it shares the ensembles' spread but forces one contact
    # mode, so it reads against ensemble_exact rather than beside it.
    "ensemble_collapsed": (GREEN, "-", "ensemble, mode-collapsed"),
    # push_t_settle_v1: settle_steps ablation on naive/point/ensemble_exact.
    # Same color per arm kind as above, linestyle encodes settle_steps.
    "naive-settle_0": (RED, ":", "naive, settle=0"),
    "naive-settle_1": (RED, "--", "naive, settle=1"),
    "naive-settle_3": (RED, "-", "naive, settle=3"),
    "point-settle_0": (AMBER, ":", "point, settle=0"),
    "point-settle_1": (AMBER, "--", "point, settle=1"),
    "point-settle_3": (AMBER, "-", "point, settle=3"),
    "ensemble_exact-settle_0": (BLUE, ":", "ensemble exact, settle=0"),
    "ensemble_exact-settle_1": (BLUE, "--", "ensemble exact, settle=1"),
    "ensemble_exact-settle_3": (BLUE, "-", "ensemble exact, settle=3"),
}

# Push-FR3 only: the region the arm can still recover a block from. Pushed
# outside it the block is gone -- the arm cannot get behind it -- so the
# episode accumulates cost for the rest of its duration. Those episodes are
# not noise and not instability; they are a distinct failure mode that has to
# be counted, not averaged into the cost.
REACH_X = (0.20, 0.80)
REACH_Y = (-0.40, 0.40)

# Every band figure caps its shared y-axis at this multiple of the worst
# arm's central curve (median, or mean on the mean+/-std foils). Enough
# headroom to still show the shape of a typical spread band, low enough that
# a diverging arm clips instead of setting the scale for everyone.
#
# The cap is taken off the central curve, not off the spread, because more
# than one arm can run away at once: on a Push-FR3 t-ou cell the `point` IQR
# reaches 370 m and `naive` 21 m while every median stays under 0.5 m, so a
# "second-worst spread" rule is still a runaway and caps nothing. A central
# curve is robust however many arms diverge.
CAP_HEADROOM = 1.6


def load_variant(run_dir: Path) -> dict | None:
    """Read one ``results/<algo>/<arm>/`` into ``(episode, step)`` arrays.

    Rows are grouped into episodes by ``(seed, repeat)``. Returns ``None``
    when the directory has not been run.
    """
    csv_path = run_dir / "tracking.csv"
    if not csv_path.is_file():
        return None
    d = np.genfromtxt(csv_path, delimiter=",", names=True)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    n_ep = manifest["num_seeds"] * manifest["repeats"]
    n_step = manifest["num_chunks"]
    if d.size != n_ep * n_step:
        raise ValueError(
            f"{csv_path}: {d.size} rows is not {n_ep} episodes x {n_step}"
        )

    def grid(name: str) -> np.ndarray:
        return d[name].reshape(n_ep, n_step)

    # Cheapest guard against a row-ordering mix-up, which would leave every
    # curve plausible and meaningless.
    seeds = grid("seed")
    if not np.all(seeds == seeds[:, :1]):
        raise ValueError(f"{csv_path}: rows are not grouped by episode")

    def rotation(name: str) -> np.ndarray | None:
        """A rotation channel, or ``None`` when the task has no orientation.

        A position-only task (Balance-FR3) logs ``orient_err = None``, which
        arrives here as an all-NaN column. Returning ``None`` lets the metric
        degrade to position-only and say so, rather than propagating NaN
        through every figure and table.
        """
        col = grid(name)
        return None if np.all(np.isnan(col)) else col

    resolved = json.loads((run_dir / "resolved.json").read_text())
    # Episodes that went non-finite (a truth solver divergence writes NaN into
    # the block state for the rest of the run). They are dropped -- not
    # averaged -- from every metric here, and counted so the exclusion is
    # visible.
    pos = grid("pos_err")
    valid = np.isfinite(pos).all(axis=1)
    time_col = grid("time")
    # Episode 0 is not guaranteed valid -- if it diverged, its logged time
    # row is NaN too, which fed a NaN axis range straight into every plot
    # sharing this variant's time axis. Every episode has the same chunk
    # schedule, so any valid one works.
    time_row = time_col[valid][0] if valid.any() else time_col[0]
    return {
        "manifest": manifest,
        "config": resolved["config"],
        "valid": valid,
        "n_dropped": int((~valid).sum()),
        "time": time_row,
        "cost": grid("cost_total"),
        "pos_err": grid("pos_err"),
        "orient_err": rotation("orient_err"),
        "block_x": grid("block_x"),
        "block_y": grid("block_y"),
        "est_pos_err": grid("est_pos_err"),
        "est_rot_err": rotation("est_rot_err"),
        "obs_pos_err": grid("obs_pos_err"),
        "seed": seeds[:, 0].astype(int),
        "repeat": grid("repeat")[:, 0].astype(int),
    }


def load_all(results: Path) -> dict[tuple[str, str], dict]:
    """Every ``(algo, arm)`` variant present under ``results``."""
    out = {}
    for algo_dir in sorted(results.iterdir()):
        if not algo_dir.is_dir() or algo_dir.name not in ALGO_ORDER:
            continue
        for arm_dir in sorted(algo_dir.iterdir()):
            if not arm_dir.is_dir():
                continue
            data = load_variant(arm_dir)
            if data is not None:
                out[(algo_dir.name, arm_dir.name)] = data
    if not out:
        raise SystemExit(f"no runs found under {results}")
    return out


# Used only when a run's config does not name the cost weights. The real
# default is read per-run from resolved.json -- see `default_rot_scale`.
FALLBACK_ROT_SCALE = 0.0857


def default_rot_scale(runs: dict) -> float:
    """Metres-per-radian implied by the run's own cost weights.

    Deliberately not a constant: ``w_orient / w_pos`` is the exchange rate the
    planner was *optimising* under, so scoring at that rate makes the report
    and the controller agree unless the rate is overridden on purpose.
    """
    for data in runs.values():
        tp = data["config"].get("task_params", {})
        if tp.get("w_pos") and tp.get("w_orient") is not None:
            return float(tp["w_orient"]) / float(tp["w_pos"])
    return FALLBACK_ROT_SCALE


def has_rotation(runs: dict) -> bool:
    """True when every variant logged an orientation channel."""
    return all(d["orient_err"] is not None for d in runs.values())


def pose_error(data: dict, rot_scale: float) -> np.ndarray:
    """Goal-to-object pose error in metres, ``(episode, step)``.

    A product metric on **SO(3) x R^3**: the two channels are measured in
    their own spaces and combined by one explicit scale, rather than being
    summed as if radians and metres were commensurable. Position-only when
    the task has no orientation goal (Balance-FR3).
    """
    if data["orient_err"] is None:
        return data["pos_err"]
    return data["pos_err"] + rot_scale * data["orient_err"]


def estimate_error(data: dict, rot_scale: float) -> np.ndarray:
    """The same metric applied to estimate-vs-truth, ``(episode, step)``."""
    if data["est_rot_err"] is None:
        return data["est_pos_err"]
    return data["est_pos_err"] + rot_scale * data["est_rot_err"]


def accumulate(data: dict, values: np.ndarray) -> np.ndarray:
    """Running time-integral of a per-replan series, ``(episode, step)``.

    Takes ``replan_dt`` from the variant's **own** manifest, never a shared
    one: arms may replan at different rates, and a 20 Hz arm integrated with
    a 10 Hz step would read twice its true total.
    """
    return np.cumsum(values, axis=1) * data["manifest"]["replan_dt"]


def improvement(data: dict, ref: dict, rot_scale: float) -> np.ndarray:
    """Accumulated pose error SAVED versus a reference, ``(episode, step)``.

    ``accumulate(ref) - accumulate(data)`` paired per episode, so **higher is
    better**. The reference may replan at a different rate, so its
    accumulated curve is interpolated onto this arm's time grid first.
    """
    acc = accumulate(data, pose_error(data, rot_scale))
    acc_ref = accumulate(ref, pose_error(ref, rot_scale))
    t, t_ref = data["time"], ref["time"]
    if len(t) != len(t_ref) or not np.allclose(t, t_ref):
        acc_ref = np.stack([np.interp(t, t_ref, row) for row in acc_ref])
    return acc_ref - acc


def central(values: np.ndarray, stat: str) -> np.ndarray:
    """The line ``band`` draws: the mean for ``"mean"``, else the median.

    Shared with the y-axis cap so the limit is always computed from exactly
    the curve that is plotted.
    """
    if stat == "mean":
        return np.nanmean(values, axis=0)
    return np.percentile(values, 50, axis=0)


NO_BOUNDS = (math.inf, -math.inf)


def cap_bounds(values: np.ndarray, stat: str,
               so_far: tuple[float, float]) -> tuple[float, float]:
    """Running ``(min, max)`` of the central curve, for the y-axis cap."""
    mid = central(values, stat)
    lo, hi = so_far
    return (min(lo, float(np.nanmin(mid))),
            max(hi, float(np.nanmax(mid))))


def apply_cap(axes, bounds: tuple[float, float], stat: str) -> str:
    """Clamp a shared y-axis to the central-curve envelope plus headroom.

    Both ends, not just the top: a spread band runs away downwards too, and
    an arm whose lower quartile dives sets the scale just as badly as one
    whose upper quartile climbs.

    Each end is scaled relative to zero rather than padded by a share of the
    span, so a series that merely grazes below zero gets a floor just under
    zero instead of a quarter-panel of empty space.

    Returns the suptitle suffix, or ``""`` when there was nothing to cap.
    """
    lo, hi = bounds
    if not (math.isfinite(lo) and math.isfinite(hi)) or hi <= lo:
        return ""
    top = hi * CAP_HEADROOM if hi > 0.0 else hi / CAP_HEADROOM
    bottom = lo * CAP_HEADROOM if lo < 0.0 else lo / CAP_HEADROOM
    for ax in axes:
        ax.set_ylim(bottom=bottom, top=top)
    line = "mean" if stat == "mean" else "median"
    return f"  (y capped to the arms' {line}s; wider bands clip)"


def band(ax, t, values, color, style, label, stat="median", keep=None) -> None:
    """Central line over episodes with a spread band.

    ``stat="median"`` (default) draws the median with an inter-quartile band;
    ``stat="mean"`` draws the mean with a +/- std band, kept only as the foil
    that shows why the median is reported (these populations are bimodal).
    """
    if keep is not None:
        values = values[keep]
    mid = central(values, stat)
    if stat == "mean":
        sd = np.nanstd(values, axis=0)
        lo, hi = mid - sd, mid + sd
    else:
        lo, hi = np.percentile(values, [25, 75], axis=0)
    ax.fill_between(t, lo, hi, color=color, alpha=0.13, linewidth=0)
    ax.plot(t, mid, color=color, linestyle=style, linewidth=1.4, label=label)


def facet(runs, title, ylabel, series, out: Path, arms=None,
          stat="median", failure_fn=None) -> None:
    """One row of algo panels sharing a y-axis, one line per arm.

    ``failure_fn(data) -> bool array`` names a task-specific failure concept
    (Push-FR3's lost-block box, via ``failure_mask``); when given, the mean
    +/- std facets drop those episodes first (their runaway error otherwise
    blows the axis out) and report the excluded count per arm in the corner.
    ``None`` (the default, and every non-Push-FR3 task today) skips this --
    the median facets always keep every episode regardless, since the median
    is robust to the same flyaways.

    The shared y-axis is capped at ``CAP_HEADROOM`` x the worst arm's central
    curve, so a diverging arm's band clips instead of setting the scale.
    """
    algos = [a for a in ALGO_ORDER if any(k[0] == a for k in runs)]
    fig, axes = plt.subplots(
        1, len(algos), figsize=(4.6 * len(algos), 3.9), sharey=True
    )
    axes = np.atleast_1d(axes)
    # Envelope of every arm's plotted line, across every panel.
    bounds = NO_BOUNDS
    for ax, algo in zip(axes, algos):
        style_axis(ax)
        fails: list[tuple[str, int]] = []
        n_total = 0
        for arm, (color, ls, label) in ARM_STYLE.items():
            if arms is not None and arm not in arms:
                continue
            data = runs.get((algo, arm))
            if data is None:
                continue
            keep = data["valid"]
            if stat == "mean" and failure_fn is not None:
                bad = failure_fn(data)
                keep = keep & ~bad
                n_total = int(data["valid"].sum())
                excluded = int((data["valid"] & bad).sum())
                if excluded:
                    fails.append((arm, excluded))
            values = series(data)
            band(ax, data["time"], values, color, ls, label,
                 stat=stat, keep=keep)
            bounds = cap_bounds(values[keep], stat, bounds)
        ax.set_title(algo, color=INK, fontsize=10)
        ax.set_xlabel("time [s]", color=MUTED)
        if stat == "mean" and failure_fn is not None:
            txt = (
                f"failures excluded (of {n_total}):\n"
                + "\n".join(f"  {a}: {n}" for a, n in fails)
                if fails else "no failures excluded"
            )
            ax.text(
                0.02, 0.98, txt, transform=ax.transAxes, va="top", ha="left",
                fontsize=6, family="monospace", color=MUTED,
                bbox=dict(facecolor="white", edgecolor=MUTED, alpha=0.8,
                          boxstyle="round,pad=0.3"),
            )
    # After plotting, never before -- an autoscale from a later draw would
    # otherwise discard the limit.
    note = apply_cap(axes, bounds, stat)
    axes[0].set_ylabel(ylabel, color=MUTED)
    # Outside the axes: capping the y-range packs the curves into the full
    # panel height, leaving an inset legend nowhere to sit without covering
    # them.
    axes[-1].legend(
        frameon=False, labelcolor=MUTED, fontsize=7.5,
        loc="center left", bbox_to_anchor=(1.02, 0.5),
    )
    title += note
    fig.suptitle(title, color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.name}")


# Contrasts that hold the engine shape FIXED, so exactly one thing differs.
# Comparing an ensemble arm against the oracle does NOT: it changes the
# belief *and* trades control samples, and the sample count alone moves the
# cost. Only these pairs support a causal read. Rows are emitted only when
# both arms are present in a run, so a run without an arm silently skips it.
CONTRASTS = (
    ("oracle", "naive", "state knowledge", "S=full both"),
    ("naive", "point", "Kalman filter", "S=full both"),
    ("point", "point_debiased", "debiasing", "S=full both"),
    ("point", "point_adaptive", "online-adaptive R", "S=full both"),
    ("point_adaptive", "ensemble_adaptive_small",
     "adaptive belief vs point, R=4", "nworld matched"),
    ("point_adaptive", "ensemble_adaptive_medium",
     "adaptive belief vs point, R=16", "nworld matched"),
    ("point_adaptive", "ensemble_adaptive_big",
     "adaptive belief vs point, R=64", "nworld matched"),
    ("point", "ensemble_exact", "belief vs point", "nworld matched"),
    ("point_narrow", "ensemble_exact", "belief alone", "R matched, S matched"),
    ("point", "point_narrow", "samples full->split", "same centre"),
    ("ensemble_exact", "ensemble_wide", "belief width exact->wide",
     "S=split both"),
    ("ensemble_wide", "ensemble_wider", "belief width wide->wider",
     "S=split both"),
    ("point", "point_fast", "rate up", "S=full both"),
    ("ensemble_collapsed", "ensemble_exact", "contact-mode diversity",
     "same shape, spread"),
    ("point", "ensemble_collapsed", "state spread, one mode",
     "nworld matched"),
    # push_t_fr3_velocity_v1's clean sigma ladder and fixed-on-naive family.
    ("point", "ensemble_double", "belief vs point, sigma 2x",
     "nworld matched"),
    ("point", "ensemble_triple", "belief vs point, sigma 3x",
     "nworld matched"),
    ("ensemble_exact", "ensemble_double", "belief width exact->double",
     "S=split both"),
    ("ensemble_double", "ensemble_triple", "belief width double->triple",
     "S=split both"),
    ("naive", "ensemble_fixed_naive", "fixed hedge vs naive",
     "nworld matched"),
    ("ensemble_fixed_naive", "ensemble_fixed_naive_double",
     "fixed hedge width", "S=split both"),
    ("ensemble_fixed_naive", "ensemble_exact",
     "fixed-on-naive vs filtered posterior", "same shape"),
)
BOOTSTRAP = 20000
# Below this many episodes a bootstrap CI is degenerate and the sign test can
# only return 1.0, so the statistics columns are suppressed rather than
# printed as a fake null. A quick tuning run (1 episode) lands here.
MIN_EPISODES_FOR_STATS = 8


def sign_test(x: np.ndarray) -> float:
    """Two-sided exact binomial sign test that the median of ``x`` is 0.

    Distribution-free, which matters here: the paired differences are not
    remotely normal (bimodal basins), so a t-test would understate the tails.
    """
    n = int((x != 0).sum())
    k = int((x > 0).sum())
    tail = sum(math.comb(n, i) for i in range(min(k, n - k) + 1)) / 2**n
    return float(min(1.0, 2.0 * tail))


def paired_stats(
    a: np.ndarray, b: np.ndarray, rng
) -> tuple[float, float, float, float]:
    """``(median %, ci_lo, ci_hi, p)`` for b relative to a, episode-paired."""
    rel = (b - a) / a * 100.0
    rel = rel[np.isfinite(rel)]
    boot = np.median(rng.choice(rel, (BOOTSTRAP, len(rel))), axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(np.median(rel)), float(lo), float(hi), sign_test(rel)


def contrast_table(runs, rot_scale: float) -> str:
    """Shape-matched contrasts with bootstrap CIs and a sign test.

    This harness has its own run-to-run movement on trajectory metrics,
    so a difference inside that band is unresolved no matter how suggestive
    the median looks. The Bonferroni bar
    is computed from the contrasts actually **emitted**, not the table's
    length: a run that mutes an arm drops every contrast needing it.
    """
    rng = np.random.default_rng(0)
    emitted = [
        (algo, c)
        for algo in ALGO_ORDER
        for c in CONTRASTS
        if runs.get((algo, c[0])) is not None
        and runs.get((algo, c[1])) is not None
    ]
    alpha = 0.05 / max(len(emitted), 1)
    n_ep = min(d["cost"].shape[0] for d in runs.values())
    stats_ok = n_ep >= MIN_EPISODES_FOR_STATS

    header = "| algo | contrast | shape | B vs A |"
    rule = "| --- | --- | --- | --- |"
    if stats_ok:
        header += " 95% CI | sign p | resolved? |"
        rule += " --- | --- | --- |"
    lines = [header, rule]

    for algo, (name_a, name_b, label, shape) in emitted:
        run_a, run_b = runs[(algo, name_a)], runs[(algo, name_b)]
        mask = run_a["valid"] & run_b["valid"]
        final_a = accumulate(run_a, pose_error(run_a, rot_scale))[:, -1][mask]
        final_b = accumulate(run_b, pose_error(run_b, rot_scale))[:, -1][mask]
        row = f"| {algo} | {label} | {shape} |"
        if stats_ok:
            med, lo, hi, p = paired_stats(final_a, final_b, rng)
            verdict = "**yes**" if p < alpha else "no"
            row += (
                f" {med:+.1f}% | [{lo:+.1f}, {hi:+.1f}]% | {p:.3f} | "
                f"{verdict} |"
            )
        else:
            rel = float(np.median((final_b - final_a) / final_a * 100.0))
            row += f" {rel:+.1f}% |"
        lines.append(row)

    lines.append("")
    if stats_ok:
        lines.append(
            f"Negative = B cheaper than A. `resolved?` is Bonferroni-"
            f"corrected over the {len(emitted)} contrasts this run emitted "
            f"(p < {alpha:.4f})."
        )
    else:
        lines.append(
            f"Negative = B cheaper than A. **{n_ep} episode(s) per cell — "
            "no confidence interval or significance test is possible, and "
            "none is shown.** Read the *direction* of a knob only, never "
            "the magnitude."
        )
    return "\n".join(lines)


def summary_table(runs, rot_scale: float) -> str:
    """Final accumulated pose error per (algo, arm), paired against the oracle.

    The pairing is what makes the numbers readable: raw spread across
    episodes is dominated by which scenario and which basin a run landed in,
    both shared by every arm at the same ``(seed, repeat)`` and therefore
    cancelling in the difference.
    """
    lines = [
        "| algo | arm | final accum. pose err [m*s], median [IQR] | "
        "vs oracle (paired median) |",
        "| --- | --- | --- | --- |",
    ]
    dropped: list[tuple[str, str, int]] = []
    for algo in ALGO_ORDER:
        base = runs.get((algo, "oracle"))
        ref = (
            None if base is None
            else accumulate(base, pose_error(base, rot_scale))[:, -1]
        )
        base_valid = None if base is None else base["valid"]
        for arm in ARM_STYLE:
            data = runs.get((algo, arm))
            if data is None:
                continue
            if data["n_dropped"]:
                dropped.append((algo, arm, data["n_dropped"]))
            valid = data["valid"]
            final = accumulate(data, pose_error(data, rot_scale))[:, -1]
            q25, q50, q75 = np.percentile(final[valid], [25, 50, 75])
            if arm == "oracle" or ref is None:
                delta = "--"
            else:
                m = valid & base_valid
                rel = np.median((final[m] - ref[m]) / ref[m]) * 100.0
                delta = f"{rel:+.1f}%"
            lines.append(
                f"| {algo} | {arm} | {q50:.1f} [{q25:.1f}, {q75:.1f}] | "
                f"{delta} |"
            )
    if dropped:
        lines.append("")
        lines.append(
            "Diverged episodes dropped (non-finite): "
            + ", ".join(f"{a}/{arm} {n}" for a, arm, n in dropped)
            + "."
        )
    return "\n".join(lines)


def components(runs, out: Path) -> None:
    """Position and rotation on their OWN axes, never combined.

    A no-op (returns immediately) for a position-only task -- there is
    nothing to split.

    Each row shares one y-axis across the algo panels (``sharey="row"``) and
    is capped like every other band figure here -- see ``CAP_HEADROOM`` -- so
    the algos stay comparable and a runaway arm's band clips instead of
    flattening every informative arm onto the floor. Position and rotation
    are capped independently: they are different units.
    """
    algos = [a for a in ALGO_ORDER if any(k[0] == a for k in runs)]
    if not has_rotation(runs):
        return
    fig, axes = plt.subplots(
        2, len(algos), figsize=(4.6 * len(algos), 6.4), sharex=True,
        sharey="row", squeeze=False,
    )
    channels = (
        ("pos_err", "position error [m]"),
        ("orient_err", "rotation error [rad]"),
    )
    # Per row, the envelope of every arm's median curve.
    row_bounds = [NO_BOUNDS for _ in channels]
    for col, algo in enumerate(algos):
        for row, (key, ylabel) in enumerate(channels):
            ax = axes[row][col]
            style_axis(ax)
            for arm, (color, ls, label) in ARM_STYLE.items():
                data = runs.get((algo, arm))
                if data is None:
                    continue
                keep = data["valid"]
                band(ax, data["time"], data[key], color, ls, label,
                     keep=keep)
                row_bounds[row] = cap_bounds(
                    data[key][keep], "median", row_bounds[row]
                )
            if row == 0:
                ax.set_title(algo, color=INK, fontsize=10)
            else:
                ax.set_xlabel("time [s]", color=MUTED)
            if col == 0:
                ax.set_ylabel(ylabel, color=MUTED)
    # After plotting, never before -- an autoscale triggered by a later draw
    # would otherwise discard the limit.
    note = ""
    for row, bounds in enumerate(row_bounds):
        note = apply_cap(axes[row], bounds, "median") or note
    axes[0][-1].legend(
        frameon=False, labelcolor=MUTED, fontsize=7.5,
        loc="center left", bbox_to_anchor=(1.02, 0.5),
    )
    title = "pose error by channel -- native units, never summed here" + note
    fig.suptitle(title, color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.name}")


def excursion(data: dict) -> np.ndarray:
    """Push-FR3 only: how far outside the reachable box the block is.

    ``(episode, step)``. Positive means out of reach.
    """
    bx, by = data["block_x"], data["block_y"]
    return np.maximum.reduce([
        REACH_X[0] - bx, bx - REACH_X[1],
        REACH_Y[0] - by, by - REACH_Y[1],
    ])


def failure_mask(data: dict) -> np.ndarray:
    """Push-FR3 only: episodes that ended with the block outside the box."""
    return excursion(data)[:, -1] > 0.0


def failures_table(runs: dict) -> str:
    """Push-FR3 only: per (algo, arm) count of lost-block failures."""
    lines = [
        "## Failures (block lost -- ended outside the reach box)",
        "",
        "Excluded from the mean+/-std pose figures; the median figures keep "
        "them.",
        "",
        "| algo | arm | failed | of |",
        "| --- | --- | --- | --- |",
    ]
    for algo in ALGO_ORDER:
        for arm in ARM_STYLE:
            data = runs.get((algo, arm))
            if data is None:
                continue
            bad = failure_mask(data)[data["valid"]]
            lines.append(f"| {algo} | {arm} | {int(bad.sum())} | {len(bad)} |")
    return "\n".join(lines)


def lost_block_table(runs) -> str:
    """Push-FR3 only: how often each arm pushed the block out of reach.

    Measured on the *truth*, so it is what actually happened, not what any
    arm believed.
    """
    lines = [
        "| algo | arm | **lost** (out at the end) | ever left | steps out | "
        "worst excursion |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for algo in ALGO_ORDER:
        for arm in ARM_STYLE:
            data = runs.get((algo, arm))
            if data is None:
                continue
            exc = excursion(data)[data["valid"]]
            out = exc > 0.0
            lost = out[:, -1]
            lines.append(
                f"| {algo} | {arm} | {lost.mean():.0%} | "
                f"{out.any(axis=1).mean():.0%} | {out.mean():.0%} | "
                f"{max(exc.max(), 0.0) * 1e3:.0f} mm |"
            )
    lines.append("")
    lines.append(
        f"Reachable box: x {REACH_X}, y {REACH_Y}. **Read this beside the "
        "pose-error table, never instead of it** — the two measure different "
        "failures."
    )
    return "\n".join(lines)


def estimator_table(runs, rot_scale: float) -> str:
    """Mean error of the state each arm handed its planner.

    Averaged over algos: the estimator does not know which planner is behind
    it. Position and rotation get their own columns in native units ("--"
    when the task has none); the combined column applies the headline scale.
    """
    lines = [
        "| arm | obs pos [mm] | est pos [mm] | est rot [deg] | "
        "est combined [mm] |",
        "| --- | --- | --- | --- | --- |",
    ]
    for arm in ARM_STYLE:
        obs, pos, rot, comb = [], [], [], []
        for (_, name), data in runs.items():
            if name != arm:
                continue
            obs.append(np.nanmean(data["obs_pos_err"]))
            pos.append(np.nanmean(data["est_pos_err"]))
            comb.append(np.nanmean(estimate_error(data, rot_scale)))
            if data["est_rot_err"] is not None:
                rot.append(np.nanmean(data["est_rot_err"]))
        if not obs:
            continue
        rot_cell = f"{np.degrees(np.mean(rot)):.2f}" if rot else "--"
        lines.append(
            f"| {arm} | {np.mean(obs) * 1e3:.2f} | {np.mean(pos) * 1e3:.2f} | "
            f"{rot_cell} | {np.mean(comb) * 1e3:.2f} |"
        )
    return "\n".join(lines)


def vs_oracle_facet(runs, rot_scale: float, out: Path) -> None:
    """Accumulated pose error ABOVE the oracle, paired per episode.

    Paired against the oracle: ``0`` is the oracle, HIGHER is worse, and the
    gap is what imperfect state costs. Pairing cancels the shared
    scenario/basin variance, so the bands are narrower than the raw
    accumulated-error facet's.
    """
    algos = [a for a in ALGO_ORDER if any(k[0] == a for k in runs)]
    fig, axes = plt.subplots(
        1, len(algos), figsize=(4.6 * len(algos), 3.9), sharey=True
    )
    axes = np.atleast_1d(axes)
    bounds = NO_BOUNDS
    for ax, algo in zip(axes, algos):
        style_axis(ax)
        ref = runs.get((algo, "oracle"))
        if ref is None:
            ax.set_title(f"{algo}  (no oracle)", color=INK, fontsize=10)
            continue
        ax.axhline(0.0, color=INK, ls=":", linewidth=1.0)
        for arm, (color, ls, label) in ARM_STYLE.items():
            if arm == "oracle":
                continue  # the reference itself: the y=0 line
            data = runs.get((algo, arm))
            if data is None:
                continue
            excess = -improvement(data, ref, rot_scale)
            keep = data["valid"] & ref["valid"]
            band(ax, data["time"], excess, color, ls, label, keep=keep)
            bounds = cap_bounds(excess[keep], "median", bounds)
        ax.set_title(algo, color=INK, fontsize=10)
        ax.set_xlabel("time [s]", color=MUTED)
    note = apply_cap(axes, bounds, "median")
    axes[0].set_ylabel("pose error above oracle [m*s]", color=MUTED)
    axes[-1].legend(
        frameon=False, labelcolor=MUTED, fontsize=7.5,
        loc="center left", bbox_to_anchor=(1.02, 0.5),
    )
    fig.suptitle(
        "accumulated pose error above the oracle (paired; LOWER = better; "
        "y=0 is the oracle ceiling)" + note,
        color=INK, fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.name}")


def main() -> None:
    """Build every figure and print the tables."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--version", required=True,
        help="Path relative to experiments/, e.g. "
        "push_fr3/state_uncertainty/t-ou/scale0.6-tau2.0-warmup50.",
    )
    parser.add_argument(
        "--mode", choices=["full", "smoke", "quick"], default="full",
        help="which results/ subtree to read (quick/smoke are reduced runs).",
    )
    parser.add_argument(
        "--rot-scale", type=float, default=None,
        help="Metres per radian for the SO(3)xR^3 pose metric. Default: the "
        "run's own w_orient/w_pos, i.e. the rate the planner optimised "
        "under. 0 scores position alone.",
    )
    parser.add_argument(
        "--algo", action="append", default=None,
        help="Only load these algos (repeatable). Use it to plot a finished "
        "block while another is still running, so a half-done algo does not "
        "draw an empty panel.",
    )
    args = parser.parse_args()

    results = EXPERIMENTS_ROOT / args.version / "results"
    if args.mode != "full":
        results = results / args.mode
    runs = load_all(results)
    if args.algo:
        runs = {k: v for k, v in runs.items() if k[0] in args.algo}
        if not runs:
            raise SystemExit(
                f"no variants for algos {args.algo} under {results}"
            )
    task = next(iter(runs.values()))["manifest"]["task"]
    # The lost-block/stuck-episode machinery assumes Push-FR3's fixed
    # reachable-box geometry; no other task defines an equivalent concept.
    is_push_fr3 = task == "push_fr3"

    figures = results / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    rates = sorted({d["manifest"]["replan_dt"] for d in runs.values()})
    rot_scale = (
        default_rot_scale(runs) if args.rot_scale is None else args.rot_scale
    )
    src = (
        "from w_orient/w_pos" if args.rot_scale is None else "from --rot-scale"
    )
    rotless = "" if has_rotation(runs) else "  [no rotation channel logged]"
    print(
        f"loaded {len(runs)} variants from {results}  (task={task})"
        + (f"  (mixed replan rates: {rates})" if len(rates) > 1 else "")
        + f"\nrot_scale = {rot_scale:.4f} m/rad ({src}){rotless}"
    )
    dropped = [(k, v["n_dropped"]) for k, v in runs.items() if v["n_dropped"]]
    if dropped:
        print("dropped diverged (non-finite) episodes, excluded from metrics:")
        for (algo, arm), n in dropped:
            print(f"  {algo}/{arm}: {n}")
    # The scale rides in every title that depends on it: a pose-error plot is
    # not interpretable without knowing what a radian was charged.
    tag = (
        f"pose error, SO(3)xR^3 @ {rot_scale:.3f} m/rad"
        if has_rotation(runs)
        else "pose error (position only -- task has no orientation goal)"
    )
    failure_fn = failure_mask if is_push_fr3 else None

    facet(
        runs,
        f"accumulated {tag}  (the headline: area under the curve)",
        "integral of pose error [m*s]",
        lambda d: accumulate(d, pose_error(d, rot_scale)),
        figures / "pose_error_accumulated.png",
    )
    # Mean +/- std versions of the two pose figures, as the explicit
    # comparison that justifies reporting the median. Read them beside the
    # median figures, never instead of.
    facet(
        runs,
        f"goal-to-object {tag}  (MEAN +/- STD -- comparison; median is "
        "the reported statistic)",
        "pose error [m]",
        lambda d: pose_error(d, rot_scale),
        figures / "pose_error_meanstd.png",
        stat="mean", failure_fn=failure_fn,
    )
    facet(
        runs,
        f"accumulated {tag}  (MEAN +/- STD -- comparison; median is "
        "the reported statistic)",
        "integral of pose error [m*s]",
        lambda d: accumulate(d, pose_error(d, rot_scale)),
        figures / "pose_error_accumulated_meanstd.png",
        stat="mean", failure_fn=failure_fn,
    )
    components(runs, figures / "pose_error_components.png")
    facet(
        runs,
        "achieved cost (DIAGNOSTIC -- what the planner optimised, not the "
        "score)",
        "running cost",
        lambda d: d["cost"],
        figures / "cost_over_time.png",
    )
    facet(
        runs,
        "accumulated cost (DIAGNOSTIC -- see pose_error_accumulated for the "
        "score)",
        "integral of cost [cost*s]",
        lambda d: accumulate(d, d["cost"]),
        figures / "cost_accumulated.png",
    )
    facet(
        runs,
        f"error of the state handed to the planner ({tag})",
        "estimate error [m]",
        lambda d: estimate_error(d, rot_scale),
        figures / "estimate_error.png",
    )
    vs_oracle_facet(runs, rot_scale, figures / "pose_error_vs_oracle.png")

    table = (
        f"# Results\n\nScored on goal-to-object pose error, SO(3)xR^3 at "
        f"**{rot_scale:.4f} m/rad**.\n\n"
        f"## Shape-matched contrasts (the causal read)\n\n"
        f"{contrast_table(runs, rot_scale)}\n\n"
        f"## Accumulated pose error, all arms vs the oracle\n\n"
        f"{summary_table(runs, rot_scale)}\n\n"
    )
    if is_push_fr3:
        table += (
            f"{failures_table(runs)}\n\n"
            f"## Lost the block (out of the arm's reach)\n\n"
            f"{lost_block_table(runs)}\n\n"
        )
    table += f"## Estimator quality\n\n{estimator_table(runs, rot_scale)}\n"
    (results / "summary.md").write_text(table)
    print(f"\n{table}")


if __name__ == "__main__":
    main()
