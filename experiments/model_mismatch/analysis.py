"""Figures and metrics for the model_mismatch sweep.

Reads what ``run.py`` wrote and owns **every** derived number, so a metric can
change without re-running the GPU.

The score is goal-to-object pose error on SO(3) x R^3: position and rotation
measured in their own spaces and combined by one explicit ``--rot-scale`` in
metres per radian (default: the run's own ``w_orient / w_pos``). Median and IQR
over the episodes per cell, **never mean and std** -- these runs are bimodal
(MJWarp non-determinism), so a std reports how many runs flipped basin, not
the noise within one.

Figures:

1. **accumulated pose error vs time** -- one band per arm, ``nominal`` the
   reference. A wrong model that recovers is not the same as one steadily worse.
2. **bias sensitivity** -- final accumulated error vs multiplier, one facet per
   parameter, with ``nominal`` and the wide-DR arms as references. The headline
   "does DR beat a wrong model".
3. **disagreement** -- ``dom_cost_cv`` over time for each ``wide_dr*`` arm, plus
   a contact-split summary (CV | contact vs CV | free-flight). Wide arms only.

Plus a paired contrast table (each wide arm vs its matching biased arms and vs
nominal).

Run (from the repo root)::

    ...analysis --version push_fr3/model_mismatch/mass_truth_10hz_v1
    ...analysis --version <version> --mode smoke

where ``...analysis`` is::

    uv run python -m experiments.model_mismatch.analysis
"""

from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path

import numpy as np

from experiments.common.dr.plot_style import (
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
from experiments.model_mismatch.harness import (
    BIAS_PARAMS,
    load_run_config,
    parse_arm,
)

EXPERIMENTS_ROOT = Path(__file__).resolve().parents[1]
PARAM_COLOR = {
    "mass": RED,
    "friction": AMBER,
    "impratio": GREEN,
    "rolling_friction": TEAL,
    "solimp_dmin": PURPLE,
    "wall_friction": AMBER,
    "wall_solimp_dmin": PURPLE,
}
MULTI_COLOR = PURPLE  # a multi-parameter (all_*) biased arm
FALLBACK_ROT_SCALE = 0.1
# A center more than this many times a facet's own median is treated as a
# divergence (basin flip / instability) rather than "a large but reasonable
# mismatch" -- these figures answer the latter question, so such points
# are dropped from the line and the y-scale entirely (not just clipped)
# and named in a text note instead, see drop_catastrophic().
CATASTROPHIC_RATIO = 100.0

# Peg-FR3 mount-uncertainty arms (not the push/balance DR grammar): a fixed
# small set with their own hues, so `arm_color` need not go through `parse_arm`.
PEG_ARMS = {
    "oracle": (INK, "-"),        # knows the true grasp (ceiling)
    "point": (RED, "-"),         # believes the peg is where commanded
    "hedge": (BLUE, "-"),        # hedges over a mount cloud
    "hedge_wide": (BLUE, "--"),
}


def load_variant(run_dir: Path) -> dict | None:
    """Read one ``results/<algo>/<arm>/`` into ``(episode, step)`` arrays."""
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

    seeds = grid("seed")
    if not np.all(seeds == seeds[:, :1]):
        raise ValueError(f"{csv_path}: rows are not grouped by episode")

    def rotation(name: str) -> np.ndarray | None:
        col = grid(name)
        return None if np.all(np.isnan(col)) else col

    resolved = json.loads((run_dir / "resolved.json").read_text())
    names = d.dtype.names

    # episodes.csv is (n_ep,) -- one row per episode, same seed/repeat order
    # as tracking.csv's episode blocks (both come from the same nested loop
    # in run.py). "success" is absent (old, pre-flag runs) or all-NaN (a task
    # with no task_success condition, e.g. Push-FR3) -> None.
    ep = np.atleast_1d(
        np.genfromtxt(run_dir / "episodes.csv", delimiter=",", names=True)
    )
    success = None
    if ep.dtype.names and "success" in ep.dtype.names:
        s = ep["success"].astype(float)
        if not np.all(np.isnan(s)):
            success = s

    return {
        "manifest": manifest,
        "config": resolved["config"],
        "time": grid("time")[0],
        "cost": grid("cost_total"),
        "pos_err": grid("pos_err"),
        "orient_err": rotation("orient_err"),
        "had_contact": grid("had_contact"),
        "success": success,
        "dom_cost_cv": grid("dom_cost_cv"),
        "dom_cost_perplexity": grid("dom_cost_perplexity"),
        # Peg-FR3 only: peg<->wall contact force, the jam/failure signal.
        "wall_force": grid("wall_force") if "wall_force" in names else None,
        "seed": seeds[:, 0].astype(int),
        "repeat": grid("repeat")[:, 0].astype(int),
    }


def load_all(results: Path) -> dict[str, dict]:
    """Every arm present under ``results/ps/`` (ps-only sweep)."""
    out: dict[str, dict] = {}
    for algo_dir in sorted(p for p in results.iterdir() if p.is_dir()):
        for arm_dir in sorted(p for p in algo_dir.iterdir() if p.is_dir()):
            data = load_variant(arm_dir)
            if data is not None:
                out[arm_dir.name] = data
    if not out:
        raise SystemExit(f"no runs found under {results}")
    return out


def is_peg_grid(results: Path) -> bool:
    """Whether ``results/`` is a ``peg_grid_variants`` layout.

    A grid config nests one level deeper (``<algo>/clearance_<c>/<arm-or-
    noise_<n>/hedge>``) than the ordinary ``<algo>/<arm>`` sweep, so it needs
    its own loader (:func:`load_peg_grid`) rather than :func:`load_all`.
    """
    return any(
        d.name.startswith("clearance_")
        for algo_dir in results.iterdir() if algo_dir.is_dir()
        for d in algo_dir.iterdir() if d.is_dir()
    )


# Second-level directory prefix -> (axis key, display label, compact-label
# letter). ``peg_grid_variants`` picks the prefix from ``truth_mode``:
# "translation" writes "offset_", "tilt" writes "tilt_", "combined" writes
# "combined_" (where the numeric suffix is a severity INDEX, not a physical
# magnitude -- see peg_grid_variants). One results/ tree only ever has one
# of these three, since truth_mode is fixed per experiment config.
_TRUE_AXIS_PREFIXES: dict[str, tuple[str, str, str]] = {
    "offset_": ("offset", "offset (m)", "o"),
    "tilt_": ("tilt", "tilt (rad)", "t"),
    "combined_": ("combined", "severity index", "s"),
}

# Third-level (hedge sub-arm) directory prefix -> cell-name builder. Each of
# peg_grid_variants's three hedge families ("matched"/"deploy"/"tiltdeploy")
# writes its own prefix; ``_parse_hedge_cell`` reads the resulting cell name
# back into ``(domains, value, kind)``.
_HEDGE_AXIS_PREFIXES = ("ratio_", "spread_", "tiltspread_")


def _load_hedge_axis_dir(axis_dir: Path) -> dict[str, dict]:
    """Every ``domains_<d>`` cell under one hedge sub-arm directory.

    ``axis_dir.name`` is ``ratio_<r>`` ("matched"), ``spread_<s>``
    ("deploy"), or ``tiltspread_<t>`` ("tiltdeploy") -- see
    ``peg_grid_variants``. Keyed by ``"hedge_r<domains>_x<ratio>"`` /
    ``"hedge_r<domains>_s<spread*1000:g>mm"`` /
    ``"hedge_r<domains>_t<trot:g>rad"``, the convention
    :func:`_parse_hedge_cell` reads back.
    """
    name = axis_dir.name
    if name.startswith("ratio_"):
        value = float(name[len("ratio_"):])
        cell_name = lambda d: f"hedge_r{d}_x{value:g}"  # noqa: E731
    elif name.startswith("spread_"):
        value = float(name[len("spread_"):])
        cell_name = lambda d: f"hedge_r{d}_s{value * 1000:g}mm"  # noqa: E731
    elif name.startswith("tiltspread_"):
        value = float(name[len("tiltspread_"):])
        cell_name = lambda d: f"hedge_r{d}_t{value:g}rad"  # noqa: E731
    else:
        return {}
    out: dict[str, dict] = {}
    for d_dir in sorted(p for p in axis_dir.iterdir() if p.is_dir()):
        if not d_dir.name.startswith("domains_"):
            continue
        domains = int(d_dir.name[len("domains_"):])
        data = load_variant(d_dir)
        if data is not None:
            out[cell_name(domains)] = data
    return out


def load_peg_grid(
    results: Path,
) -> tuple[dict[tuple[float, float, str], dict], str]:
    """Every ``(clearance, true_value, cell)`` cell of a peg misalignment grid.

    ``cell`` is ``"point"``/``"oracle"`` (their spec doesn't depend on the
    hedge sub-arm) or ``"hedge_r<domains>_<x|s|t><value>[mm|rad]"`` for one
    of the three hedge families -- see ``peg_grid_variants``. Also returns
    the true-error axis key (``"offset"`` / ``"tilt"`` / ``"combined"``,
    from whichever second-level prefix is actually present) so callers can
    label figures correctly instead of assuming "offset". Skips
    ``allocation_probe/`` entirely -- it isn't a ``clearance_*`` directory,
    so the outer loop's prefix check already excludes it.
    """
    out: dict[tuple[float, float, str], dict] = {}
    axis_key: str | None = None
    for algo_dir in sorted(p for p in results.iterdir() if p.is_dir()):
        for c_dir in sorted(p for p in algo_dir.iterdir() if p.is_dir()):
            if not c_dir.name.startswith("clearance_"):
                continue
            clearance = float(c_dir.name[len("clearance_"):])
            for t_dir in sorted(p for p in c_dir.iterdir() if p.is_dir()):
                prefix = next(
                    (p for p in _TRUE_AXIS_PREFIXES
                     if t_dir.name.startswith(p)),
                    None,
                )
                if prefix is None:
                    continue
                axis_key = _TRUE_AXIS_PREFIXES[prefix][0]
                true_val = float(t_dir.name[len(prefix):])
                for sub in sorted(p for p in t_dir.iterdir() if p.is_dir()):
                    if sub.name.startswith(_HEDGE_AXIS_PREFIXES):
                        for cell, data in _load_hedge_axis_dir(sub).items():
                            out[(clearance, true_val, cell)] = data
                        continue
                    data = load_variant(sub)
                    if data is not None:
                        out[(clearance, true_val, sub.name)] = data
    if not out:
        raise SystemExit(f"no grid runs found under {results}")
    return out, (axis_key or "offset")


def load_truth_grid(results: Path) -> dict[str, dict[str, dict]]:
    """``{truth_<mult>: {arm: data}}`` for a ``truth_grid`` bias_target leg.

    One level deeper than :func:`load_all` (``<algo>/truth_<mult>/<arm>``,
    from :func:`experiments.model_mismatch.harness.friction_truth_grid_
    variants`) -- each ``truth_<mult>`` key holds its own ordinary arm-keyed
    dict, reusable with every plain-``load_all`` helper (``run_bias_params``,
    ``default_rot_scale``, ``final_accum_valid``, ...).
    """
    out: dict[str, dict[str, dict]] = {}
    for algo_dir in sorted(p for p in results.iterdir() if p.is_dir()):
        for t_dir in sorted(p for p in algo_dir.iterdir() if p.is_dir()):
            if not t_dir.name.startswith("truth_"):
                continue
            arms = {}
            for arm_dir in sorted(p for p in t_dir.iterdir() if p.is_dir()):
                data = load_variant(arm_dir)
                if data is not None:
                    arms[arm_dir.name] = data
            if arms:
                out[t_dir.name] = arms
    if not out:
        raise SystemExit(f"no truth-grid runs found under {results}")
    return out


def run_bias_params(runs: dict) -> tuple[str, ...]:
    """The swept parameter set for this sweep (from any run's config).

    Reads it off the resolved config; falls back to the Push-FR3 triple so
    results written before ``bias_params`` existed still load.
    """
    for data in runs.values():
        bp = data["config"].get("bias_params")
        if bp:
            return tuple(bp)
    return BIAS_PARAMS


def run_plot_params(
    runs: dict, bias_params: tuple[str, ...], version_dir: Path | None = None
) -> tuple[str, ...]:
    """Which of ``bias_params`` to actually draw a sensitivity facet for.

    ``plot_bias_params`` is figures-only (see ``RunConfig.plot_bias_params``)
    -- it changes no physics, so unlike every other config field this
    function prefers the LIVE ``version_dir/config.yaml`` over what's frozen
    in ``resolved.json`` (a run's own snapshot, taken at run time): setting
    it after the sweep already ran should not require re-running the GPU
    just to update a figure. Falls back to each run's own resolved snapshot
    (for a leg whose live config already had it, or an archived leg with no
    ``version_dir``), then to every swept parameter.
    """
    if version_dir is not None:
        config_path = version_dir / "config.yaml"
        if config_path.is_file():
            pp = load_run_config(config_path).plot_bias_params
            if pp:
                return tuple(pp)
    for data in runs.values():
        pp = data["config"].get("plot_bias_params")
        if pp:
            return tuple(pp)
    return bias_params


def default_rot_scale(runs: dict) -> float:
    """Metres-per-radian implied by the run's own cost weights."""
    for data in runs.values():
        tp = data["config"].get("task_params", {})
        if tp.get("w_pos") and tp.get("w_orient") is not None:
            return float(tp["w_orient"]) / float(tp["w_pos"])
    return FALLBACK_ROT_SCALE


def pose_error(data: dict, rot_scale: float) -> np.ndarray:
    """Goal-to-object pose error in metres, ``(episode, step)``."""
    if data["orient_err"] is None:
        return data["pos_err"]
    return data["pos_err"] + rot_scale * data["orient_err"]


def accumulate(data: dict, values: np.ndarray) -> np.ndarray:
    """Running time-integral of a per-replan series, ``(episode, step)``."""
    return np.cumsum(values, axis=1) * data["manifest"]["replan_dt"]


def arm_color(name: str, bias_params: tuple[str, ...]) -> tuple[str, str]:
    """``(color, linestyle)`` for one arm, grouped by family."""
    if name in PEG_ARMS:
        return PEG_ARMS[name]
    kind, params, mult = parse_arm(name, bias_params)
    if kind == "nominal":
        return INK, "-"
    if kind == "wide":
        # Bare wide_dr (all params) solid; a per-parameter wide arm dashed.
        return BLUE, "-" if len(params) == len(bias_params) else "--"
    if kind == "mixed":
        # Per-param multipliers differ (e.g. mass_2.0+friction_0.5), so there
        # is no single "softer/harder than truth" direction to dash by --
        # dash-dot keeps it visually distinct from all_*'s solid/dotted.
        return MULTI_COLOR, "-."
    # Biased: dash by direction (softer / harder than truth). A multi-parameter
    # (all_*) arm gets its own hue; a single-parameter arm hues by parameter.
    if len(params) > 1:
        color = MULTI_COLOR
    else:
        color = PARAM_COLOR.get(params[0], MUTED)
    return color, "-" if mult >= 1.0 else ":"


def band(ax, t, values, color, style, label, alpha=0.15, stat="median") -> None:
    """Central line + spread band over the episode axis.

    ``stat`` picks the pair: ``"median"`` -> median with an IQR band (the
    default, robust to the bimodal flyaways), ``"mean"`` -> mean with a
    +/- std band (shown only as the explicit comparison that justifies the
    median: a std reports how many runs flipped basin, not the noise in one).
    """
    if stat == "mean":
        center = np.nanmean(values, axis=0)
        sd = np.nanstd(values, axis=0)
        lo, hi = center - sd, center + sd
    else:
        center = np.nanmedian(values, axis=0)
        lo = np.nanpercentile(values, 25, axis=0)
        hi = np.nanpercentile(values, 75, axis=0)
    ax.fill_between(t, lo, hi, color=color, alpha=alpha, linewidth=0)
    ax.plot(t, center, color=color, linestyle=style, label=label, linewidth=1.8)


def center_spread(values: np.ndarray, stat: str = "median") -> tuple:
    """Scalar center + (lo, hi) bound, ``stat`` convention as :func:`band`."""
    if stat == "mean":
        c = np.nanmean(values)
        sd = np.nanstd(values)
        return c, c - sd, c + sd
    return np.nanmedian(values), *np.nanpercentile(values, [25, 75])


def drop_catastrophic(
    xs: np.ndarray, centers: np.ndarray, los: np.ndarray, his: np.ndarray,
    fails: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
    """Drop points beyond ``CATASTROPHIC_RATIO`` x this facet's median.

    A center that far off the rest of the facet is treated as a divergence
    rather than a reasonable mismatch, and reported as a text note instead.
    A single basin-flipped point a few multipliers out otherwise sets the
    y-scale for the ENTIRE figure, flattening every reasonable-range point
    -- the actual question these sensitivity figures answer -- to an
    invisible line. The median is
    computed off every finite center in the facet first (robust to the one
    or two catastrophic points themselves), so this only fires on points
    that are quantitatively a different regime, not just the largest of an
    otherwise-smooth curve.
    """
    centers = np.asarray(centers, dtype=float)
    finite = centers[np.isfinite(centers)]
    baseline = np.nanmedian(finite) if finite.size else np.nan
    if not np.isfinite(baseline) or baseline <= 0:
        keep = np.ones_like(centers, dtype=bool)
    else:
        bound = CATASTROPHIC_RATIO * baseline
        keep = ~np.isfinite(centers) | (centers <= bound)
    notes = [
        f"@{x:g}x: {c:.3g} (off-scale, omitted)"
        for x, c, k in zip(xs, centers, keep) if np.isfinite(c) and not k
    ]
    idx = np.where(keep)[0]
    return (
        np.asarray(xs)[idx], centers[idx], np.asarray(los)[idx],
        np.asarray(his)[idx], np.asarray(fails)[idx], notes,
    )


def _bias_facets(
    runs: dict,
    plot_params: tuple[str, ...],
    bias_params: tuple[str, ...],
    rot_scale: float,
    nominal: np.ndarray | None,
    stat: str,
) -> tuple:
    """Per-facet biased points + the shared y-scale (biased + nominal only).

    Centers only feed the y-scale, never the upper error-bar bound -- a
    handful of catastrophic (basin-flipped) episodes can push a 75th
    percentile (or a mean+std under stat="mean") far off scale even when
    under half the episodes for that arm diverged, which would otherwise
    flatten every healthy point to a line. A spread that still runs past
    the axis after that is simply clipped, not reported -- only
    failure_rate() is. Failed episodes are already excluded above (see
    final_accum_valid); failure_rate() is reported alongside, not baked
    into the magnitude statistic. A center itself in a different regime
    (not just a wide spread -- see :func:`drop_catastrophic`) is dropped
    from the facet entirely and named in ``facets[param]
    ["catastrophic_notes"]`` instead: these figures answer "how sensitive
    is a REASONABLE mismatch," and one basin-flipped arm at the far end of
    the sweep otherwise sets the scale for every other point.
    """
    facets, scale_vals = {}, []
    for param in plot_params:
        xs, meds, los, his, fails = [], [], [], [], []
        for name, data in runs.items():
            kind, params, mult = parse_arm(name, bias_params)
            # Single-parameter arms only; a multi-param (all_*) arm has no
            # single multiplier axis to sit on.
            if kind == "biased" and len(params) == 1 and params[0] == param:
                c, lo, hi = center_spread(
                    final_accum_valid(data, rot_scale), stat
                )
                xs.append(mult)
                meds.append(c)
                los.append(lo)
                his.append(hi)
                fails.append(failure_rate(data))
        if nominal is not None:  # the true model sits at multiplier 1.0
            c, lo, hi = center_spread(nominal, stat)
            xs.append(1.0)
            meds.append(c)
            los.append(lo)
            his.append(hi)
            fails.append(failure_rate(runs["nominal"]))
        order = np.argsort(xs)
        xs_k, meds_k, los_k, his_k, fails_k, notes = drop_catastrophic(
            np.asarray(xs)[order], np.asarray(meds)[order],
            np.asarray(los)[order], np.asarray(his)[order],
            np.asarray(fails)[order],
        )
        facets[param] = {
            "xs": xs_k, "meds": meds_k, "los": los_k, "his": his_k,
            "fails": fails_k, "catastrophic_notes": notes,
        }
        scale_vals += list(meds_k)
    finite = [v for v in scale_vals if np.isfinite(v)]
    return facets, (1.2 * max(finite) if finite else 1.0)


def accumulated_error(
    runs: dict,
    out: Path,
    rot_scale: float,
    bias_params: tuple[str, ...],
    stat: str = "median",
) -> None:
    """Accumulated pose error vs time, one band per arm.

    ``stat`` is passed to :func:`band` (median+IQR or mean+/-std).
    """
    fig, ax = plt.subplots(figsize=(9, 5.5))
    style_axis(ax)
    fail_notes = []
    for name, data in runs.items():
        color, style = arm_color(name, bias_params)
        mask = valid_mask(data)
        acc = accumulate(data, pose_error(data, rot_scale))[mask]
        if acc.size == 0:
            acc = accumulate(data, pose_error(data, rot_scale))
        band(ax, data["time"], acc, color, style, name, stat=stat)
        fr = failure_rate(data)
        if fr > 0:
            fail_notes.append(f"{name}: {fr:.0%} failed (excluded)")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("accumulated pose error [m*s]")
    spread = "mean +/- std" if stat == "mean" else "median + IQR"
    ax.set_title(
        f"Accumulated pose error ({spread}, rot-scale {rot_scale:g} m/rad)"
    )
    if fail_notes:
        ax.text(
            0.99, 0.02, "\n".join(fail_notes), transform=ax.transAxes,
            ha="right", va="bottom", fontsize=6.5, color=RED,
        )
    ax.legend(fontsize=7, ncol=2, loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"  wrote {out}")


def final_accum(data: dict, rot_scale: float) -> np.ndarray:
    """Per-episode accumulated pose error at the end, ``(episode,)``."""
    return accumulate(data, pose_error(data, rot_scale))[:, -1]


def valid_mask(data: dict) -> np.ndarray:
    """Episode mask excluding task_success failures, ``(episode,)``.

    A failure here is not "the controller did a bit worse" -- at the
    extreme end of a bias arm it can be a numerically-invalid rollout (the
    object goes airborne, loses contact, and position error grows without
    a physical bound). Mixing that into a continuous magnitude statistic
    answers "how bad is the simulator once it has broken" instead of "how
    bad is the control" -- exclude it and report failure_rate() alongside
    instead. Tasks with no task_success condition (Push-FR3) mark every
    episode valid.
    """
    s = data.get("success")
    if s is None:
        return np.ones(data["pos_err"].shape[0], dtype=bool)
    return s == 1.0


def final_accum_valid(data: dict, rot_scale: float) -> np.ndarray:
    """:func:`final_accum`, task_success failures excluded.

    Falls back to every episode if none are valid -- better to show an arm
    that failed everywhere (with failure_rate() reporting 1.0 alongside)
    than crash on an empty array.
    """
    values = final_accum(data, rot_scale)
    mask = valid_mask(data)
    return values[mask] if mask.any() else values


def failure_rate(data: dict) -> float:
    """Fraction of episodes task_success flags as failed.

    ``0.0`` if no such condition exists for this task.
    """
    s = data.get("success")
    if s is None:
        return 0.0
    known = s[~np.isnan(s)]
    return float(np.mean(known == 0.0)) if known.size else 0.0


def bias_sensitivity(
    runs: dict,
    out: Path,
    rot_scale: float,
    bias_params: tuple[str, ...],
    plot_params: tuple[str, ...] | None = None,
    stat: str = "median",
) -> None:
    """Final accumulated error vs multiplier, one facet per parameter.

    Failed episodes (task_success == 0, see :func:`valid_mask`) are
    excluded from the magnitude statistic -- for some arms these are
    numerically-invalid rollouts, not "worse control" -- and each point's
    failure_rate() is annotated instead. The y-axis is scaled to the
    biased + nominal arms (the comparison of interest). A wide-DR reference
    that lands off that scale is simply not drawn, so one diverged arm
    cannot flatten every healthy curve to a line.

    ``plot_params`` (default: all of ``bias_params``) is which facets to
    actually draw -- a subset lets a config omit a swept-but-uninformative
    parameter from the figure without dropping its arms from the data or
    from every OTHER figure.
    ``bias_params`` still classifies every arm name (so an omitted
    parameter's arms are simply never selected into a facet, not
    misparsed).

    ``stat`` picks the point + error bar, same convention as :func:`band`:
    ``"median"`` (default) -> median with an IQR bar, ``"mean"`` -> mean
    with a +/- std bar.
    """
    plot_params = bias_params if plot_params is None else plot_params
    fig, axes = plt.subplots(
        1, len(plot_params), figsize=(4.2 * len(plot_params), 4.5),
        sharey=True,
    )
    axes = np.atleast_1d(axes)
    nominal = (
        final_accum_valid(runs["nominal"], rot_scale)
        if "nominal" in runs else None
    )

    facets, ymax = _bias_facets(
        runs, plot_params, bias_params, rot_scale, nominal, stat
    )

    for ax, param in zip(axes, plot_params):
        style_axis(ax)
        d = facets[param]
        yerr = np.abs(np.vstack([d["meds"] - d["los"], d["his"] - d["meds"]]))
        ax.errorbar(
            d["xs"], d["meds"], yerr=yerr, color=PARAM_COLOR[param],
            marker="o", capsize=3, linewidth=1.6, label="biased single model",
        )
        notes = [
            f"{param}@{x:g}x: {fr:.0%} failed (excluded)"
            for x, fr in zip(d["xs"], d["fails"]) if fr > 0
        ]
        notes += [f"{param}{n}" for n in d["catastrophic_notes"]]
        for wide, style in ((f"wide_dr_{param}", "--"), ("wide_dr", ":")):
            if wide not in runs:
                continue
            m = center_spread(final_accum_valid(runs[wide], rot_scale), stat)[0]
            if np.isfinite(m) and m <= ymax:
                ax.axhline(m, color=BLUE, linestyle=style, linewidth=1.4,
                           label=wide)
            wfr = failure_rate(runs[wide])
            if wfr > 0:
                notes.append(f"{wide}: {wfr:.0%} failed (excluded)")
        if notes:
            ax.text(0.5, 0.97, "\n".join(notes), transform=ax.transAxes,
                    ha="center", va="top", fontsize=6.5, color=RED)
        ax.set_xscale("log")
        ax.set_ylim(0.0, ymax)
        ax.set_xlabel(f"{param} multiplier (x truth)")
        ax.set_title(param)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("final accumulated pose error [m*s]")
    spread = "mean +/- std" if stat == "mean" else "median + IQR"
    fig.suptitle(f"Does wide DR beat a wrong model? ({spread})")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"  wrote {out}")


def disagreement_fig(
    runs: dict, out: Path, bias_params: tuple[str, ...]
) -> None:
    """Cross-domain cost CV over time for the wide-DR arms."""
    wide = {
        n: d for n, d in runs.items()
        if parse_arm(n, bias_params)[0] == "wide"
    }
    if not wide:
        print("  (no wide-DR arms; skipping disagreement figure)")
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    style_axis(ax)
    for name, data in wide.items():
        color, style = arm_color(name, bias_params)
        band(ax, data["time"], data["dom_cost_cv"], color, style, name)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("cross-domain cost CV [-]")
    ax.set_title("How much the domains disagree about the executed action")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"  wrote {out}")


def disagreement_table(runs: dict, bias_params: tuple[str, ...]) -> str:
    """CV split by contact vs free-flight, per wide-DR arm."""
    lines = [
        "cross-domain disagreement (median CV, contact vs free-flight)",
        f"{'arm':<20} {'CV|contact':>12} {'CV|free':>12} {'perplexity':>12}",
    ]
    for name, data in runs.items():
        if parse_arm(name, bias_params)[0] != "wide":
            continue
        cv = data["dom_cost_cv"]
        contact = data["had_contact"].astype(bool)
        cv_c = np.nanmedian(cv[contact]) if contact.any() else np.nan
        cv_f = np.nanmedian(cv[~contact]) if (~contact).any() else np.nan
        perp = np.nanmedian(data["dom_cost_perplexity"])
        lines.append(
            f"{name:<20} {cv_c:>12.4f} {cv_f:>12.4f} {perp:>12.3f}"
        )
    return "\n".join(lines)


def success_rate_table(runs: dict) -> str | None:
    """Per-arm episode success rate, from ``episodes.csv``'s ``success``.

    ``None`` if no arm in this run carries a success condition (e.g. a
    Push-FR3 sweep, where ``task_success`` is unimplemented and every
    episode's value is empty/NaN).
    """
    rows = []
    for name, data in runs.items():
        s = data.get("success")
        if s is None:
            continue
        valid = s[~np.isnan(s)]
        if valid.size == 0:
            continue
        rows.append((name, float(np.mean(valid)), int(valid.size)))
    if not rows:
        return None
    lines = [
        "episode success rate (episodes.csv's success: AND-reduced over "
        "replans for balance/balance_fr3, OR-reduced for peg_fr3/flip_fr3)",
        f"{'arm':<20} {'success rate':>14} {'n episodes':>12}",
    ]
    for name, rate, n in rows:
        lines.append(f"{name:<20} {rate:>14.1%} {n:>12d}")
    return "\n".join(lines)


def sign_test(x: np.ndarray) -> float:
    """Two-sided exact binomial p-value that ``x`` is centred on zero."""
    x = x[~np.isnan(x)]
    n = int(np.sum(x != 0.0))
    if n == 0:
        return 1.0
    k = int(np.sum(x > 0.0))
    tail = sum(comb(n, i) for i in range(min(k, n - k) + 1))
    return float(min(1.0, 2.0 * tail / 2.0**n))


def paired_delta(a: dict, b: dict, rot_scale: float) -> np.ndarray:
    """Per-episode final-accum difference ``a - b`` (same seed x repeat)."""
    if not (np.array_equal(a["seed"], b["seed"])
            and np.array_equal(a["repeat"], b["repeat"])):
        raise ValueError("arms are not paired by (seed, repeat)")
    return final_accum(a, rot_scale) - final_accum(b, rot_scale)


def contrast_table(
    runs: dict, rot_scale: float, bias_params: tuple[str, ...]
) -> str:
    """Each wide arm vs its matching biased arms and vs nominal (paired)."""
    lines = [
        "paired contrasts on final accumulated pose error (median delta %, "
        "delta<0 = first arm better)",
        f"{'contrast':<38} {'median %':>10} {'sign p':>8}",
    ]
    pairs: list[tuple[str, str]] = []
    if "wide_dr" in runs and "nominal" in runs:
        pairs.append(("wide_dr", "nominal"))
    for param in bias_params:
        wide = f"wide_dr_{param}"
        if wide not in runs:
            continue
        for name in runs:
            kind, params, _ = parse_arm(name, bias_params)
            if kind == "biased" and len(params) == 1 and params[0] == param:
                pairs.append((wide, name))
    # Multi-parameter arms vs the true model: all_* (biased, shared mult) and
    # mixed (biased, per-param mult) both have len(params) > 1.
    if "nominal" in runs:
        for name in runs:
            kind, params, _ = parse_arm(name, bias_params)
            if kind in ("biased", "mixed") and len(params) > 1:
                pairs.append((name, "nominal"))
    for lhs, rhs in pairs:
        delta = paired_delta(runs[lhs], runs[rhs], rot_scale)
        base = np.nanmedian(final_accum(runs[rhs], rot_scale))
        pct = 100.0 * np.nanmedian(delta) / base if base else np.nan
        p = sign_test(delta)
        lines.append(f"{lhs + ' vs ' + rhs:<38} {pct:>10.1f} {p:>8.3f}")
    return "\n".join(lines)


def wall_force_fig(runs: dict, out: Path, bias_params: tuple[str, ...]) -> None:
    """Peg<->wall contact force over time, one band per arm (median + IQR).

    The failure signal: a confidently-wrong mount (``point``) jams the peg on a
    socket wall, so its force sits above the ``oracle`` and ``hedge`` arms.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    style_axis(ax)
    for name, data in runs.items():
        if data.get("wall_force") is None:
            continue
        color, style = arm_color(name, bias_params)
        band(ax, data["time"], data["wall_force"], color, style, name)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("peg<->wall contact force [N]")
    ax.set_title("Peg-wall force (median + IQR): a wrong mount jams on a wall")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"  wrote {out}")


# Peg contrasts (paired): the cost of ignoring the grasp, the benefit of
# hedging, and the residual gap to the oracle ceiling.
PEG_CONTRASTS = (("point", "oracle"), ("hedge", "point"), ("hedge", "oracle"))


def peg_contrast_table(runs: dict, rot_scale: float) -> str:
    """Paired peg contrasts on final accumulated pose error."""
    lines = [
        "paired contrasts on final accumulated pose error (median delta %, "
        "delta<0 = first arm better)",
        f"{'contrast':<24} {'median %':>10} {'sign p':>8}",
    ]
    for lhs, rhs in PEG_CONTRASTS:
        if lhs not in runs or rhs not in runs:
            continue
        delta = paired_delta(runs[lhs], runs[rhs], rot_scale)
        base = np.nanmedian(final_accum(runs[rhs], rot_scale))
        pct = 100.0 * np.nanmedian(delta) / base if base else np.nan
        lines.append(
            f"{lhs + ' vs ' + rhs:<24} {pct:>10.1f} {sign_test(delta):>8.3f}"
        )
    return "\n".join(lines)


def peg_summary_table(runs: dict, rot_scale: float) -> str:
    """Per-arm final pose error and peg-wall force (median [IQR])."""
    lines = [
        "per-arm summary (median [IQR] over episodes)",
        f"{'arm':<12} {'pose err [m*s]':>22} {'wall force [N]':>22} "
        f"{'max force [N]':>14}",
    ]
    for arm in PEG_ARMS:
        data = runs.get(arm)
        if data is None:
            continue
        f = final_accum(data, rot_scale)
        pe = (f"{np.nanmedian(f):.2f} [{np.nanpercentile(f, 25):.2f}, "
              f"{np.nanpercentile(f, 75):.2f}]")
        wf = data.get("wall_force")
        if wf is None:
            wcell, wmax = "--", "--"
        else:
            wcell = (f"{np.nanmedian(wf):.2f} [{np.nanpercentile(wf, 25):.2f}, "
                     f"{np.nanpercentile(wf, 75):.2f}]")
            wmax = f"{np.nanmax(wf):.2f}"
        lines.append(f"{arm:<12} {pe:>22} {wcell:>22} {wmax:>14}")
    return "\n".join(lines)


def _cell_rate(data: dict | None) -> tuple[float | None, int]:
    """``(success rate, n episodes)`` for one grid cell, ``None`` if n/a."""
    if data is None:
        return None, 0
    s = data.get("success")
    if s is None:
        return None, 0
    valid = s[~np.isnan(s)]
    if valid.size == 0:
        return None, 0
    return float(np.mean(valid)), int(valid.size)


# One color per hedge sub-arm curve (RED/INK are point/oracle's, reserved).
_RATIO_PALETTE = (BLUE, MULTI_COLOR, AMBER, TEAL)

# hedge sub-arm kind -> (single-letter cell infix, display label template).
_HEDGE_KINDS = {
    "ratio": ("x", "ratio {v:g}x"),
    "spread": ("s", "spread {v:g}mm"),
    "tiltspread": ("t", "tiltspread {v:g}rad"),
}


def _parse_hedge_cell(cell: str) -> tuple[int, float, str]:
    """``(domains, value, kind)`` from a hedge sub-arm cell name.

    ``cell`` is ``"hedge_r<domains>_<infix><value>[mm|rad]"``; ``kind`` is
    one of ``"ratio"``/``"spread"``/``"tiltspread"`` -- see
    ``peg_grid_variants``'s three hedge sub-arm families and
    :func:`_load_hedge_axis_dir`'s cell-name convention.
    """
    body = cell[len("hedge_r"):]
    r_str, rest = body.split("_", 1)
    domains = int(r_str)
    for kind, (infix, _) in _HEDGE_KINDS.items():
        if rest.startswith(infix):
            value_str = rest[len(infix):]
            if value_str.endswith("mm"):
                value_str = value_str[:-2]
            elif value_str.endswith("rad"):
                value_str = value_str[:-3]
            return domains, float(value_str), kind
    raise ValueError(f"unrecognized hedge cell: {cell!r}")


def _hedge_label(value: float, kind: str) -> str:
    """Display label for one hedge sub-arm value, e.g. ``"ratio 2x"``."""
    return _HEDGE_KINDS[kind][1].format(v=value)


def peg_grid_report(
    runs: dict[tuple[float, float, str], dict],
    results: Path,
    axis_key: str = "offset",
) -> None:
    """Success rate vs. domains, one facet per ``(clearance, true_value)``.

    Each hedge sub-arm (``kind``, ``value``) is a curve (rate vs. R);
    ``point``/``oracle`` (whose rate doesn't depend on the hedge sub-arm)
    are flat reference lines -- reads directly as "does hedging close the
    point-oracle gap, does more domain coverage help at a fixed
    control-sample count, and does the answer change with the true error or
    the fit clearance." ``axis_key`` (``"offset"``/``"tilt"``/``"combined"``,
    from :func:`load_peg_grid`) picks the axis label -- the true-value axis
    is a translation offset, a tilt magnitude, or a combined-truth severity
    index depending on which ``truth_mode`` this grid was run under.
    """
    _, axis_display, _ = _TRUE_AXIS_PREFIXES[f"{axis_key}_"]
    clearances = sorted({c for c, _, _ in runs})
    true_vals = sorted({t for _, t, _ in runs})
    fig, axes = plt.subplots(
        len(clearances), len(true_vals),
        figsize=(4.2 * len(true_vals), 4.0 * len(clearances)),
        sharey=True, squeeze=False,
    )
    table = [
        "peg misalignment grid: episode success rate",
        f"{'clearance':>10} {axis_key:>8} {'cell':>24} {'rate':>8} {'n':>6}",
    ]
    kinds = sorted({
        (_parse_hedge_cell(cell)[2], _parse_hedge_cell(cell)[1])
        for _, _, cell in runs if cell.startswith("hedge_")
    })
    for i, c in enumerate(clearances):
        for j, t in enumerate(true_vals):
            ax = axes[i][j]
            style_axis(ax)
            cell_domains_kind = {
                cell: _parse_hedge_cell(cell)
                for cc, tt, cell in runs
                if cc == c and tt == t and cell.startswith("hedge_")
            }
            for k, (kind, value) in enumerate(kinds):
                domain_cells = sorted(
                    (domains, cell)
                    for cell, (domains, v, kd) in cell_domains_kind.items()
                    if kd == kind and v == value
                )
                if not domain_cells:
                    continue
                xs, ys = [], []
                for domains, cell in domain_cells:
                    rate, count = _cell_rate(runs.get((c, t, cell)))
                    xs.append(domains)
                    ys.append(rate)
                    if count:
                        table.append(
                            f"{c:>10g} {t:>8g} {cell:>24} {rate:>8.1%} "
                            f"{count:>6d}"
                        )
                ax.plot(
                    xs, ys, color=_RATIO_PALETTE[k % len(_RATIO_PALETTE)],
                    marker="o", linewidth=1.8,
                    label=_hedge_label(value, kind),
                )
            for arm, color in (("point", RED), ("oracle", INK)):
                rate, count = _cell_rate(runs.get((c, t, arm)))
                if count == 0:
                    continue
                ax.axhline(rate, color=color, linestyle="--", linewidth=1.4,
                           label=arm)
                table.append(
                    f"{c:>10g} {t:>8g} {arm:>24} {rate:>8.1%} {count:>6d}"
                )
            ax.set_ylim(0.0, 1.0)
            ax.set_xlabel("domains (R)")
            ax.set_title(f"clearance {c:g}, {axis_key} {t:g}")
            if j == 0:
                ax.set_ylabel("episode success rate")
            ax.legend(fontsize=6)
    fig.suptitle(
        "Does hedging over the mount misalignment prevent jamming, and does "
        f"more domain coverage help at fixed control samples? ({axis_display})"
    )
    fig.tight_layout()
    out = results / "success_grid.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"  wrote {out}")
    print("\n" + "\n".join(table))


def peg_grid_heatmaps(
    runs: dict[tuple[float, float, str], dict],
    results: Path,
    axis_key: str = "offset",
) -> None:
    """One ``clearance x true_value`` success-rate heatmap per arm, one figure.

    Companion to :func:`peg_grid_table`'s single wide grid -- there each row
    is a ``(clearance, true_value)`` cell and arms are columns (reads as
    "how do arms compare in THIS cell"); here each arm gets its own small
    panel (reads as "where does THIS arm fail"). ``axis_key`` labels the
    true-value axis -- see :func:`peg_grid_report`.
    """
    clearances = sorted({c for c, _, _ in runs})
    true_vals = sorted({t for _, t, _ in runs})
    kinds = sorted({
        (_parse_hedge_cell(cell)[2], _parse_hedge_cell(cell)[1])
        for _, _, cell in runs if cell.startswith("hedge_")
    })
    domains = sorted({
        _parse_hedge_cell(cell)[0]
        for _, _, cell in runs if cell.startswith("hedge_")
    })

    def grid_for(arm: str) -> np.ndarray:
        g = np.full((len(clearances), len(true_vals)), np.nan)
        for i, c in enumerate(clearances):
            for j, t in enumerate(true_vals):
                rate, count = _cell_rate(runs.get((c, t, arm)))
                if count:
                    g[i, j] = rate
        return g

    def draw(ax, arm: str, title: str):
        g = grid_for(arm)
        im = ax.imshow(g, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")
        ax.set_xticks(range(len(true_vals)))
        ax.set_xticklabels([f"{t:g}" for t in true_vals], fontsize=6)
        ax.set_yticks(range(len(clearances)))
        ax.set_yticklabels([f"{c:g}" for c in clearances], fontsize=6)
        for i in range(len(clearances)):
            for j in range(len(true_vals)):
                v = g[i, j]
                if np.isnan(v):
                    continue
                color = "black" if 0.3 < v < 0.85 else "white"
                ax.text(
                    j, i, f"{v:.0%}", ha="center", va="center",
                    fontsize=5.5, color=color,
                )
        ax.set_title(title, fontsize=8)
        return im

    # Row 0: point, oracle (rest of the row unused). Rows 1..: one per
    # (kind, value), domains across columns -- so every hedge sub-arm lands
    # at (kind/value, domains), matching peg_grid_report's grouping. The
    # three hedge families ("matched"/"deploy"/"tiltdeploy") land on
    # different rows even at the same domains column, so they stay visually
    # distinct rather than merged -- deploy vs tiltdeploy directly
    # comparable this way.
    ncols = max(2, len(domains))
    nrows = 1 + len(kinds)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(2.6 * ncols, 2.4 * nrows), squeeze=False,
    )

    im = draw(axes[0][0], "point", "point")
    draw(axes[0][1], "oracle", "oracle")
    for j in range(2, ncols):
        axes[0][j].axis("off")
    _cell_suffix = {"ratio": "", "spread": "mm", "tiltspread": "rad"}
    for k_i, (kind, value) in enumerate(kinds):
        infix = _HEDGE_KINDS[kind][0]
        for d_i, d in enumerate(domains):
            cell = f"hedge_r{d}_{infix}{value:g}{_cell_suffix[kind]}"
            im = draw(
                axes[1 + k_i][d_i], cell,
                f"{kind} R={d} {_hedge_label(value, kind)}",
            )
        for d_i in range(len(domains), ncols):
            axes[1 + k_i][d_i].axis("off")

    for row in axes:
        row[0].set_ylabel("clearance", fontsize=7)
    for ax in axes[-1][:len(domains) or ncols]:
        ax.set_xlabel(axis_key, fontsize=7)

    fig.suptitle("peg misalignment grid: success rate per arm")
    fig.tight_layout()
    fig.colorbar(
        im, ax=axes.ravel().tolist(), label="episode success rate",
        shrink=0.6,
    )
    out = results / "success_heatmaps.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"  wrote {out}")


def peg_grid_table(
    runs: dict[tuple[float, float, str], dict],
    results: Path,
    axis_key: str = "offset",
) -> None:
    """Success rate as a ``(clearance, true_value)`` x arm heatmap-table.

    Complements :func:`peg_grid_report`'s domains-vs-kind curves with the
    exact numbers in one scannable grid -- most of a wide clearance x
    true_value sweep saturates at 0%/100%, where a curve's shape carries
    little information but the precise rate still does. ``axis_key`` labels
    the true-value axis and its row-label letter -- see
    :func:`peg_grid_report`.
    """
    _, _, axis_letter = _TRUE_AXIS_PREFIXES[f"{axis_key}_"]
    clearances = sorted({c for c, _, _ in runs})
    true_vals = sorted({t for _, t, _ in runs})
    kinds = sorted({
        (_parse_hedge_cell(cell)[2], _parse_hedge_cell(cell)[1])
        for _, _, cell in runs if cell.startswith("hedge_")
    })
    domains = sorted({
        _parse_hedge_cell(cell)[0]
        for _, _, cell in runs if cell.startswith("hedge_")
    })
    _cell_suffix = {"ratio": "", "spread": "mm", "tiltspread": "rad"}
    columns = ["point", "oracle"] + [
        f"hedge_r{d}_{_HEDGE_KINDS[kind][0]}{value:g}{_cell_suffix[kind]}"
        for kind, value in kinds for d in domains
    ]
    col_labels = ["point", "oracle"] + [
        f"R={d}\n{_hedge_label(value, kind)}"
        for kind, value in kinds for d in domains
    ]
    rows = [(c, t) for c in clearances for t in true_vals]
    row_labels = [f"c={c:g}\n{axis_letter}={t:g}" for c, t in rows]

    grid = np.full((len(rows), len(columns)), np.nan)
    for i, (c, t) in enumerate(rows):
        for j, col in enumerate(columns):
            rate, count = _cell_rate(runs.get((c, t, col)))
            if count:
                grid[i, j] = rate

    fig, ax = plt.subplots(
        figsize=(1.1 * len(columns) + 1.5, 0.32 * len(rows) + 1.5)
    )
    im = ax.imshow(grid, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels(col_labels, fontsize=7)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(row_labels, fontsize=7)
    ax.set_xticks(np.arange(-0.5, len(columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(rows), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0)
    ax.tick_params(which="minor", length=0)
    for i in range(len(rows)):
        for j in range(len(columns)):
            v = grid[i, j]
            if np.isnan(v):
                continue
            color = "black" if 0.3 < v < 0.85 else "white"
            ax.text(
                j, i, f"{v:.0%}", ha="center", va="center", fontsize=6.5,
                color=color,
            )
    fig.colorbar(im, ax=ax, label="episode success rate", shrink=0.6)
    ax.set_title("peg misalignment grid: episode success rate")
    fig.tight_layout()
    out = results / "success_table.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"  wrote {out}")


def peg_report(runs: dict, results: Path, rot_scale: float) -> None:
    """Figures + tables for a Peg-FR3 mount-uncertainty sweep."""
    # bias_params is unused by the peg arms; pass the default so arm_color's
    # signature is satisfied (it short-circuits on the peg names).
    bp = BIAS_PARAMS
    accumulated_error(runs, results / "accumulated_error.png", rot_scale, bp)
    wall_force_fig(runs, results / "wall_force.png", bp)
    print("\n" + peg_summary_table(runs, rot_scale))
    print("\n" + peg_contrast_table(runs, rot_scale))
    sr = success_rate_table(runs)
    if sr is not None:
        print("\n" + sr)


def main() -> None:
    """Load a run's results and emit every figure and table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--version", required=True,
        help="Path relative to experiments/, e.g. "
        "push_fr3/model_mismatch/mass_truth_10hz_v1.",
    )
    parser.add_argument(
        "--mode", choices=["full", "smoke", "quick"], default="full"
    )
    parser.add_argument(
        "--rot-scale", type=float, default=None,
        help="Metres per radian for the pose metric (default: w_orient/w_pos).",
    )
    parser.add_argument(
        "--exclude-dr", action="store_true",
        help="Drop the wide_dr* arms and plot only nominal + biased "
        "single-model arms (wrong physics params). Writes to a models_only/ "
        "subdir so the full figures are kept.",
    )
    args = parser.parse_args()

    results = EXPERIMENTS_ROOT / args.version / "results"
    if args.mode != "full":
        results = results / args.mode
    if is_peg_grid(results):
        grid_runs, axis_key = load_peg_grid(results)
        print(f"loaded {len(grid_runs)} grid cells (axis: {axis_key})")
        peg_grid_report(grid_runs, results, axis_key)
        peg_grid_table(grid_runs, results, axis_key)
        peg_grid_heatmaps(grid_runs, results, axis_key)
        return
    runs = load_all(results)
    rot_scale = (
        args.rot_scale if args.rot_scale is not None
        else default_rot_scale(runs)
    )
    # Peg-FR3's mount-uncertainty runs use a different arm grammar (oracle/
    # point/hedge) and their own figures. A peg_fr3 run using the ordinary
    # <param>_<mult>/all_<mult> grammar (e.g. wall_friction mismatch) instead
    # falls through to the generic path below like push/balance.
    task = next(iter(runs.values()))["config"]["task"]
    if task == "peg_fr3" and set(runs) & set(PEG_ARMS):
        print(f"loaded {len(runs)} peg arms; rot-scale {rot_scale:g} m/rad")
        peg_report(runs, results, rot_scale)
        return
    bias_params = run_bias_params(runs)
    out_dir = results
    if args.exclude_dr:
        runs = {
            n: d for n, d in runs.items()
            if parse_arm(n, bias_params)[0] != "wide"
        }
        out_dir = results / "models_only"
        out_dir.mkdir(exist_ok=True)
    print(
        f"loaded {len(runs)} arms; rot-scale {rot_scale:g} m/rad; "
        f"params {list(bias_params)}"
    )
    accumulated_error(
        runs, out_dir / "accumulated_error.png", rot_scale, bias_params
    )
    accumulated_error(
        runs, out_dir / "accumulated_error_meanstd.png", rot_scale,
        bias_params, stat="mean",
    )
    if task == "peg_fr3":
        wall_force_fig(runs, out_dir / "wall_force.png", bias_params)
    bias_sensitivity(
        runs, out_dir / "bias_sensitivity.png", rot_scale, bias_params,
        run_plot_params(runs, bias_params, EXPERIMENTS_ROOT / args.version),
    )
    bias_sensitivity(
        runs, out_dir / "bias_sensitivity_meanstd.png", rot_scale,
        bias_params,
        run_plot_params(runs, bias_params, EXPERIMENTS_ROOT / args.version),
        stat="mean",
    )
    disagreement_fig(runs, out_dir / "disagreement.png", bias_params)
    print("\n" + disagreement_table(runs, bias_params))
    print("\n" + contrast_table(runs, rot_scale, bias_params))
    sr = success_rate_table(runs)
    if sr is not None:
        print("\n" + sr)


if __name__ == "__main__":
    main()
