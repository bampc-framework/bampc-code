r"""Plot real-world Push-T-FR3 total pose error over time: point vs ensemble.

Reads every episode CSV under
``output/real_world/{point,ensemble}_T_<ALGO>/*.csv`` (columns include ``t``,
``pos_err``, ``rot_err``; one file per seed/episode),
resamples each onto a shared time grid, and plots the *total* pose error
over time for both estimator arms, one figure pair per algorithm found.
Writes two figures per algo, always both: median +/- IQR and mean +/- std.
Any directory with "backup" in its name is skipped.

Total error is ``pos_err + ROT_TO_M * rot_err``: rotation error (rad) is
scaled by the object's encapsulating-sphere radius so it lands in the same
length units as position error before the two are summed -- a heuristic for
"how far a point on the object's surface moves", not an exact quantity.

Also writes two paper-ready figures (one CEM|PS panel pair per summary
statistic) as vector PDF: minimal chrome, serif/STIX type sized for a
double-column figure, one shared legend -- meant to drop straight into a
LaTeX ``\includegraphics``, not for exploring the data (use the per-algo
PNGs above for that).

Plus one more paper-ready figure: translation (top row) | rotation (bottom
row) x one column per algo, all in a single figure with one shared legend
on top, instead of one ``ROT_TO_M``-merged total -- the merge hides that
rotation error stays noisy throughout while translation converges. Rotation
is in degrees on a fixed ``[0, 180]`` axis (the geodesic quaternion angle's
true range), not autoscaled radians -- radians next to meters isn't
independently interpretable, and autoscaling only to what the band spans
hides how close individual episodes get to the theoretical worst case.

Each algo's column can also overlay 1-2 raw single-episode traces, labeled
generically "Example 1", "Example 2", ... in the shared legend (one
linestyle per episode, shared between both rows, so a viewer can pair a
translation trace with its rotation trace by eye) -- a median/IQR or
mean/std band averages away exactly the per-episode behavior worth seeing:

* ``OVERSHOOT_ALGO`` (default ``"PS"``): rotation overshoots back and forth
  while translation stays poorly reduced because of it. Picked by
  :func:`_pick_overshoot_episodes`, not, say, a one-off stuck flip that
  recovers position fine, or a wobbly correction that still converges.
* ``DRIFT_ALGO`` (default ``"CEM"``): rotation stays low the whole episode
  while translation just drifts off, tracking the arm's own aggregate
  trend. Picked by :func:`_pick_drift_episodes`.

Run (from the repo root)::

    uv run python scripts/push_fr3/plot_real_world_pose_error.py

Writes, under ``output/results/push_fr3/``:
``<algo>/real_world_pose_error_{median_iqr,mean_std}.png`` and
``paper/pose_error_{median_iqr,mean_std,components_median_iqr,
components_mean_std}.pdf``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from bampc import OUTPUT_DIR

_SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = OUTPUT_DIR / "real_world"
RESULTS_DIR = OUTPUT_DIR / "results" / _SCRIPT_DIR.name

ARM_NAMES = ("point", "ensemble")
# Categorical slots 1 (blue) / 2 (orange) -- validated adjacent-pair CVD
# separation, see the dataviz skill's palette reference.
COLORS = {"point": "#2a78d6", "ensemble": "#eb6834"}
BG, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9"
ROT_TO_M = 0.1  # encapsulating-sphere radius [m], rad -> m heuristic
ALGO_LABELS = {"CEM": "CEM", "PS": "Predictive Sampling"}
ARM_LABELS = {"point": "Point estimate", "ensemble": "Ensemble"}
# Which algo's component-figure column gets 1-2 overshoot example-episode
# traces overlaid: PS is where the EE was observed overshooting back and
# forth while correcting a local rotation error.
OVERSHOOT_ALGO = "PS"
NUM_EXAMPLE_EPISODES = 2
# Minimum swing (degrees) for _swing_amplitudes to call a direction change a
# real reversal rather than sensor-noise-level jitter.
OVERSHOOT_MIN_AMPLITUDE_DEG = 15.0
# Rotation error (deg) below which an episode counts as "converged" -- only
# swings after the *first* such crossing are scored, so a big early value
# settling down for the first time isn't mistaken for overshoot.
OVERSHOOT_CONVERGED_DEG = 30.0
# Rotation error (deg) an episode must end below to be eligible at all: a
# stuck failure (e.g. an unrecovered ~180 deg flip) is not an overshoot.
OVERSHOOT_STUCK_DEG = 90.0
# Which algo's component-figure column gets 1-2 drift example-episode
# traces overlaid: CEM's point estimate is where translation error was
# observed to just drift off, tracking the arm's own aggregate trend,
# while rotation stays under control throughout.
DRIFT_ALGO = "CEM"
# Rotation error (deg) an episode must never exceed to be eligible: the
# point of a drift example is a *clean* translation drift, not a rotation
# excursion muddying the story.
DRIFT_ROT_CAP_DEG = 45.0
# Distinct from COLORS["point"] so an example trace doesn't read as another
# arm; low alpha keeps it clearly a background reference, not a headline
# curve. Linestyle cycles per episode (same episode = same style in both
# the translation and rotation panel) so the two can be paired by eye.
EXAMPLE_COLOR = "#1b4f9c"
EXAMPLE_ALPHA = 0.55
EXAMPLE_LINESTYLES = ("--", ":", "-.", (0, (3, 1, 1, 1)))

# Print-oriented rcParams: serif/STIX type (reads as Times without needing a
# LaTeX install), sized for a double-column figure that gets shrunk to
# ~3.3-3.5in per panel in the final layout -- font sizes are chosen to still
# be legible at that scale, not at the pixel size this renders in a viewer.
PAPER_RC = {
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.4,
    "pdf.fonttype": 42,  # embed as real (editable/searchable) text, not curves
    "ps.fonttype": 42,
}
PAPER_INK, PAPER_MUTED, PAPER_GRID = "#000000", "#3a3a3a", "#d8d8d8"


def discover_algos() -> list[str]:
    """Algo names with a non-backup ``point_T_<ALGO>`` and matching ensemble."""
    algos = []
    for path in sorted(DATA_DIR.glob("point_T_*")):
        if not path.is_dir() or "backup" in path.name:
            continue
        algo = path.name.removeprefix("point_T_")
        ensemble_dir = DATA_DIR / f"ensemble_T_{algo}"
        if ensemble_dir.is_dir() and "backup" not in ensemble_dir.name:
            algos.append(algo)
    return algos


def load_arm_components(
    arm_dir: Path, grid: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Resample every episode's pos_err/rot_err onto ``grid``.

    Returns ``(pos, rot)``, each ``(N, T)``.
    """
    paths = sorted(arm_dir.glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"no episodes under {arm_dir}")
    pos = np.full((len(paths), len(grid)), np.nan)
    rot = np.full((len(paths), len(grid)), np.nan)
    for i, path in enumerate(paths):
        d = np.genfromtxt(path, delimiter=",", names=True)
        t = d["t"]
        in_range = grid <= t[-1]
        pos[i, in_range] = np.interp(grid[in_range], t, d["pos_err"])
        rot[i, in_range] = np.interp(grid[in_range], t, d["rot_err"])
    return pos, rot


def band_median_iqr(ax, t, err, label, color):
    """Median-over-episodes curve with an IQR band."""
    median = np.nanmedian(err, axis=0)
    q1 = np.nanpercentile(err, 25, axis=0)
    q3 = np.nanpercentile(err, 75, axis=0)
    ax.fill_between(t, q1, q3, color=color, alpha=0.22, linewidth=0)
    ax.plot(t, median, color=color, linewidth=1.6, label=label)


def band_mean_std(ax, t, err, label, color):
    """Mean-over-episodes curve with a +/-1 std band, clipped at zero."""
    mean = np.nanmean(err, axis=0)
    std = np.nanstd(err, axis=0)
    ax.fill_between(
        t, np.maximum(mean - std, 0.0), mean + std, color=color, alpha=0.22,
        linewidth=0,
    )
    ax.plot(t, mean, color=color, linewidth=1.6, label=label)


def style_axis(ax) -> None:
    """Shared BG/spine/grid styling for a totals or per-component axis."""
    ax.set_facecolor(BG)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.xaxis.grid(False)


def style_axis_paper(ax) -> None:
    """Minimal print chrome: no facecolor tint, thin muted spines/grid."""
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(PAPER_MUTED)
        ax.spines[spine].set_linewidth(0.8)
    ax.tick_params(colors=PAPER_MUTED, width=0.8)
    ax.yaxis.grid(True, color=PAPER_GRID, linewidth=0.6)
    ax.xaxis.grid(False)


def draw_paper_figure(
    by_algo: dict[str, dict[str, np.ndarray]], grid: np.ndarray, band_fn,
    out: Path,
) -> None:
    """One CEM|PS panel pair, vector PDF, for direct use in the paper."""
    algos = list(by_algo)
    with plt.rc_context(PAPER_RC):
        fig, axes = plt.subplots(
            1, len(algos), figsize=(6.5, 2.5), sharey=True,
        )
        fig.patch.set_facecolor("white")

        for ax, algo in zip(axes, algos, strict=True):
            ax.set_facecolor("white")
            style_axis_paper(ax)
            totals = by_algo[algo]
            for name in ARM_NAMES:
                band_fn(ax, grid, totals[name], ARM_LABELS[name], COLORS[name])
            ax.set_xlabel("Time [s]", color=PAPER_INK)
            ax.set_title(
                f"{ALGO_LABELS.get(algo, algo)} (n={totals['point'].shape[0]})",
                color=PAPER_INK,
            )
            ax.set_ylim(bottom=0.0)

        axes[0].set_ylabel("Total pose error [m]", color=PAPER_INK)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.06),
            ncol=len(labels), frameon=False, labelcolor=PAPER_INK,
        )
        fig.tight_layout()

        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"wrote {out}")


def draw_figure(
    totals: dict[str, np.ndarray], grid: np.ndarray, band_fn, summary: str,
    algo: str, out: Path,
) -> None:
    """One total-pose-error figure using ``band_fn`` for the summary curve."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.patch.set_facecolor(BG)
    style_axis(ax)

    for name in ARM_NAMES:
        band_fn(ax, grid, totals[name], name, COLORS[name])

    ax.set_ylabel("total pose error  [m + "
                   f"{ROT_TO_M:g}*rad]", color=MUTED)
    ax.set_xlabel("time [s]", color=MUTED)
    ax.set_ylim(bottom=0.0)
    ax.legend(frameon=False, labelcolor=MUTED, fontsize=9)

    ax.set_title(
        f"real-world Push-T-FR3 ({algo}): total pose error over time, "
        f"point vs ensemble  (point n={totals['point'].shape[0]}, "
        f"ensemble n={totals['ensemble'].shape[0]}, {summary}, "
        f"rot scaled by {ROT_TO_M:g} m sphere radius)",
        color=INK, fontsize=10,
    )
    fig.tight_layout()

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def _swing_amplitudes(
    deg: np.ndarray, min_amplitude_deg: float
) -> list[float]:
    """Amplitude (deg) of each reversal-to-reversal swing in ``deg``.

    A standard zigzag/swing decomposition: a swing only ends once the
    signal has moved back by at least ``min_amplitude_deg`` from its most
    recent extreme, so sensor-noise-level jitter doesn't fragment it into
    spurious swings.
    """
    valid = deg[~np.isnan(deg)]
    if len(valid) < 3:
        return []
    amps: list[float] = []
    direction = 0  # 0 = undecided, +/-1 = currently rising/falling
    extreme = valid[0]
    swing_start = valid[0]
    for v in valid[1:]:
        delta = v - extreme
        if direction >= 0 and delta < -min_amplitude_deg:
            amps.append(abs(extreme - swing_start))
            swing_start, direction, extreme = extreme, -1, v
        elif direction <= 0 and delta > min_amplitude_deg:
            amps.append(abs(extreme - swing_start))
            swing_start, direction, extreme = extreme, 1, v
        elif (direction == 1 and v > extreme) or (
            direction == -1 and v < extreme
        ):
            extreme = v
    amps.append(abs(extreme - swing_start))
    return amps


def _pick_overshoot_episodes(
    pos_err: np.ndarray,
    rot_err: np.ndarray,
    k: int,
    min_amplitude_deg: float = OVERSHOOT_MIN_AMPLITUDE_DEG,
) -> list[int]:
    """Indices of the ``k`` episodes matching a reported failure pattern.

    Rotation converges, then overshoots back up by a lot, while translation
    stays poorly reduced because of it. Two things this deliberately is
    *not*: a noisy initial decay from a large starting error (excluded by
    only scoring swings *after* the first time rotation error drops below
    ``OVERSHOOT_CONVERGED_DEG`` -- otherwise "137 deg settling down for the
    first time" scores as a huge, spurious "overshoot"), and a stuck
    failure like an unrecovered ~180 deg flip (excluded by requiring the
    episode end below ``OVERSHOOT_STUCK_DEG`` and swing back down at least
    once after its post-convergence peak). Ranks the survivors by
    ``max_swing_amplitude * mean(pos_err over the final quarter)``:
    amplitude alone would also surface episodes that still recovered
    position fine, and late position error alone would surface plain
    failures.
    """
    scores = np.zeros(len(rot_err))
    tail = max(1, pos_err.shape[1] // 4)
    late_pos = np.nanmean(pos_err[:, -tail:], axis=1)
    for i, row in enumerate(rot_err):
        valid = np.degrees(row[~np.isnan(row)])
        below = np.flatnonzero(valid < OVERSHOOT_CONVERGED_DEG)
        if len(below) == 0:
            continue  # never converges -- no baseline to overshoot from
        post = valid[below[0]:]
        final = np.nanmean(post[-max(1, len(post) // 10):])
        if final > OVERSHOOT_STUCK_DEG:
            continue  # ends near the failure ceiling -- stuck, not swinging
        amps = _swing_amplitudes(post, min_amplitude_deg)
        if len(amps) < 2:
            continue  # no full up-then-down cycle after convergence
        scores[i] = max(amps) * late_pos[i]
    return list(np.argsort(scores)[::-1][:k])


def _pick_drift_episodes(
    pos_err: np.ndarray,
    rot_err: np.ndarray,
    k: int,
    rot_cap_deg: float = DRIFT_ROT_CAP_DEG,
) -> list[int]:
    """Indices of the ``k`` episodes matching a different reported pattern.

    Rotation stays low for the entire episode while translation just drifts
    off, tracking the arm's own aggregate trend rather than standing out as
    an outlier. Filters out any episode whose rotation error ever exceeds
    ``rot_cap_deg`` -- the point of these examples is a clean translation
    drift, not a rotation excursion -- then ranks the survivors by how much
    translation error grows from the episode's first quarter to its last.
    """
    tail = max(1, pos_err.shape[1] // 4)
    early = np.nanmean(pos_err[:, :tail], axis=1)
    late = np.nanmean(pos_err[:, -tail:], axis=1)
    drift = late - early
    max_rot_deg = np.degrees(np.nanmax(rot_err, axis=1))
    drift = np.where(max_rot_deg <= rot_cap_deg, drift, -np.inf)
    return list(np.argsort(drift)[::-1][:k])


def draw_paper_component_figure(
    by_algo_pos: dict[str, dict[str, np.ndarray]],
    by_algo_rot: dict[str, dict[str, np.ndarray]],
    grid: np.ndarray,
    band_fn,
    out: Path,
    example_traces: (
        dict[str, list[tuple[np.ndarray, np.ndarray]]] | None
    ) = None,
) -> None:
    """Translation (top row) | rotation (bottom row) x one column per algo.

    Same point-vs-ensemble comparison as :func:`draw_paper_figure`, but
    without collapsing the two error components into one ``ROT_TO_M``-scaled
    total. Rotation is shown in degrees on a fixed ``[0, 180]`` axis (the
    geodesic quaternion angle's true range) rather than autoscaled radians:
    autoscaling only to what the median/IQR (or mean/std) band spans hides
    how close individual episodes actually get to the theoretical worst
    case, and radians next to meters isn't independently interpretable
    either way. Y-axes share scale within a row, so the algos are directly
    comparable per component.

    ``example_traces[algo]``, when given, overlays 1-2 raw single-episode
    traces on that algo's column in both rows -- a band is a summary and,
    by construction, averages away exactly the per-episode behavior those
    traces are meant to show (e.g. :func:`_pick_overshoot_episodes` for
    ``OVERSHOOT_ALGO``, :func:`_pick_drift_episodes` for ``DRIFT_ALGO``).
    Each episode gets its own linestyle, consistent between the translation
    and rotation rows, so a viewer can pair "this episode's translation
    trace" with "this episode's rotation trace" by eye. The legend calls
    them "Example 1", "Example 2", ... generically -- the linestyle, not
    the number, is what a reader matches between panels; which actual
    episode that is differs per algo column.
    """
    algos = list(by_algo_pos)
    example_traces = example_traces or {}
    max_examples = max((len(v) for v in example_traces.values()), default=0)
    # (ylabel, per-algo data, rad->plotted-unit scale, fixed (bottom, top))
    rows = (
        ("Translation error [m]", by_algo_pos, 1.0, None),
        ("Rotation error [deg]", by_algo_rot, 180.0 / np.pi, (0.0, 180.0)),
    )

    with plt.rc_context(PAPER_RC):
        fig, axes = plt.subplots(
            2, len(algos), figsize=(6.5, 4.8), sharey="row", squeeze=False,
        )
        fig.patch.set_facecolor("white")

        for r, (ylabel, by_algo, scale, ylim) in enumerate(rows):
            is_rotation = scale != 1.0
            for c, algo in enumerate(algos):
                ax = axes[r, c]
                ax.set_facecolor("white")
                style_axis_paper(ax)
                by_arm = by_algo[algo]
                for name in ARM_NAMES:
                    band_fn(
                        ax, grid, by_arm[name] * scale, ARM_LABELS[name],
                        COLORS[name],
                    )
                for i, (pos_row, rot_row) in enumerate(
                    example_traces.get(algo, [])
                ):
                    row_data = (rot_row if is_rotation else pos_row) * scale
                    ax.plot(
                        grid, row_data, color=EXAMPLE_COLOR,
                        alpha=EXAMPLE_ALPHA, linewidth=1.1,
                        linestyle=(
                            EXAMPLE_LINESTYLES[i % len(EXAMPLE_LINESTYLES)]
                        ),
                    )

                if r == 0:
                    ax.set_title(
                        f"{ALGO_LABELS.get(algo, algo)} "
                        f"(n={by_arm['point'].shape[0]})",
                        color=PAPER_INK,
                    )
                if r == len(rows) - 1:
                    ax.set_xlabel("Time [s]", color=PAPER_INK)
                if c == 0:
                    ax.set_ylabel(ylabel, color=PAPER_INK)
                if ylim is not None:
                    ax.set_ylim(*ylim)
                else:
                    # Must come after all plotting: an axis with no artists
                    # yet defaults to a (0, 1) view, and set_ylim locks in
                    # whatever top is current at the time (and disables
                    # further autoscaling) -- called this early it silently
                    # stretches the panel to (0, 1) regardless of its real
                    # data range.
                    ax.set_ylim(bottom=0.0)

        handles = [
            Line2D([0], [0], color=COLORS[n], lw=1.6, label=ARM_LABELS[n])
            for n in ARM_NAMES
        ]
        handles += [
            Line2D(
                [0], [0], color=EXAMPLE_COLOR, lw=1.1, alpha=EXAMPLE_ALPHA,
                linestyle=EXAMPLE_LINESTYLES[i % len(EXAMPLE_LINESTYLES)],
                label=f"Example {i + 1}",
            )
            for i in range(max_examples)
        ]
        fig.legend(
            handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.06),
            ncol=len(handles), frameon=False, labelcolor=PAPER_INK,
        )
        fig.tight_layout()

        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"wrote {out}")


def main() -> None:
    """Per algo found, load both arms and write the two summary figures."""
    dt = 0.1
    grid = np.arange(0.0, 20.0 + dt, dt)

    algos = discover_algos()
    if not algos:
        raise FileNotFoundError(
            f"no point_T_*/ensemble_T_* pairs under {DATA_DIR}"
        )

    by_algo, by_algo_pos, by_algo_rot, example_traces = {}, {}, {}, {}
    for algo in algos:
        pos_by_arm, rot_by_arm = {}, {}
        for name in ARM_NAMES:
            pos_by_arm[name], rot_by_arm[name] = load_arm_components(
                DATA_DIR / f"{name}_T_{algo}", grid
            )
        by_algo_pos[algo], by_algo_rot[algo] = pos_by_arm, rot_by_arm
        totals = {
            name: pos_by_arm[name] + ROT_TO_M * rot_by_arm[name]
            for name in ARM_NAMES
        }
        by_algo[algo] = totals

        out_dir = RESULTS_DIR / algo
        draw_figure(
            totals, grid, band_median_iqr, "median +/- IQR", algo,
            out_dir / "real_world_pose_error_median_iqr.png",
        )
        draw_figure(
            totals, grid, band_mean_std, "mean +/- std", algo,
            out_dir / "real_world_pose_error_mean_std.png",
        )

        if algo == OVERSHOOT_ALGO:
            idx = _pick_overshoot_episodes(
                pos_by_arm["point"], rot_by_arm["point"], NUM_EXAMPLE_EPISODES
            )
            example_traces[algo] = [
                (pos_by_arm["point"][i], rot_by_arm["point"][i]) for i in idx
            ]
        elif algo == DRIFT_ALGO:
            idx = _pick_drift_episodes(
                pos_by_arm["point"], rot_by_arm["point"], NUM_EXAMPLE_EPISODES
            )
            example_traces[algo] = [
                (pos_by_arm["point"][i], rot_by_arm["point"][i]) for i in idx
            ]

    paper_dir = RESULTS_DIR / "paper"
    draw_paper_figure(
        by_algo, grid, band_median_iqr, paper_dir / "pose_error_median_iqr.pdf",
    )
    draw_paper_figure(
        by_algo, grid, band_mean_std, paper_dir / "pose_error_mean_std.pdf",
    )
    draw_paper_component_figure(
        by_algo_pos, by_algo_rot, grid, band_median_iqr,
        paper_dir / "pose_error_components_median_iqr.pdf", example_traces,
    )
    draw_paper_component_figure(
        by_algo_pos, by_algo_rot, grid, band_mean_std,
        paper_dir / "pose_error_components_mean_std.pdf", example_traces,
    )


if __name__ == "__main__":
    main()
