r"""Plot real-world Push-T-FR3 position and orientation error separately.

Same data and estimator comparison as ``plot_real_world_pose_error.py``
(``output/real_world/{point,ensemble}_T_<ALGO>/*.csv``, columns ``t``,
``pos_err``, ``rot_err``), but instead of collapsing position and rotation
into one heuristic "total pose error" scalar, each is plotted in its own
panel -- position error [m] on the left, orientation error [rad] on the
right -- point vs ensemble on both, one figure pair per algorithm found.
Writes two figures per algo, always both: median +/- IQR and mean +/- std.
Any directory with "backup" in its name is skipped.

Also writes two paper-ready figures (one 2x(CEM|PS) grid per summary
statistic, rows = position/orientation) as vector PDF: minimal chrome,
serif/STIX type sized for a double-column figure, one shared legend --
meant to drop straight into a LaTeX ``\includegraphics``, not for exploring
the data (use the per-algo PNGs above for that).

Run (from the repo root)::

    uv run python scripts/push_fr3/plot_real_world_pose_error_split.py

Writes, under ``output/results/push_fr3/``:
``<algo>/real_world_pose_error_split_{median_iqr,mean_std}.png`` and
``paper/pose_error_split_{median_iqr,mean_std}.pdf``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from bampc import OUTPUT_DIR

_SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = OUTPUT_DIR / "real_world"
RESULTS_DIR = OUTPUT_DIR / "results" / _SCRIPT_DIR.name

ARM_NAMES = ("point", "ensemble")
METRICS = ("pos", "rot")
METRIC_LABELS = {"pos": "Position error [m]", "rot": "Orientation error [rad]"}
# Categorical slots 1 (blue) / 2 (orange) -- validated adjacent-pair CVD
# separation, see the dataviz skill's palette reference.
COLORS = {"point": "#2a78d6", "ensemble": "#eb6834"}
BG, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9"
ALGO_LABELS = {"CEM": "CEM", "PS": "Predictive Sampling"}
ARM_LABELS = {"point": "Point estimate", "ensemble": "Ensemble"}

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


def load_arm(arm_dir: Path, grid: np.ndarray) -> dict[str, np.ndarray]:
    """Resample every episode's pos/rot error onto ``grid``. Each ``(N, T)``."""
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
    return {"pos": pos, "rot": rot}


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
    """Apply the shared dark plot styling to one axis."""
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


def draw_figure(
    by_arm: dict[str, dict[str, np.ndarray]], grid: np.ndarray, band_fn,
    summary: str, algo: str, out: Path,
) -> None:
    """Position (left) and orientation (right) error, side by side."""
    n_point = by_arm["point"]["pos"].shape[0]
    n_ensemble = by_arm["ensemble"]["pos"].shape[0]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.patch.set_facecolor(BG)

    for ax, metric in zip(axes, METRICS, strict=True):
        style_axis(ax)
        for name in ARM_NAMES:
            band_fn(
                ax, grid, by_arm[name][metric], ARM_LABELS[name], COLORS[name],
            )
        ax.set_ylabel(METRIC_LABELS[metric], color=MUTED)
        ax.set_xlabel("time [s]", color=MUTED)
        ax.set_ylim(bottom=0.0)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.04), ncol=2,
        frameon=False, labelcolor=MUTED,
    )
    fig.suptitle(
        f"real-world Push-T-FR3 ({algo}): position/orientation error over "
        f"time, point vs ensemble  (point n={n_point}, "
        f"ensemble n={n_ensemble}, {summary})",
        color=INK, fontsize=10, y=1.1,
    )
    fig.tight_layout()

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def draw_paper_figure(
    by_algo: dict[str, dict[str, dict[str, np.ndarray]]], grid: np.ndarray,
    band_fn, out: Path,
) -> None:
    """A 2x(CEM|PS) grid: rows position/orientation, cols algos."""
    algos = list(by_algo)
    with plt.rc_context(PAPER_RC):
        fig, axes = plt.subplots(
            2, len(algos), figsize=(6.5, 4.5), sharex=True, sharey="row",
        )
        fig.patch.set_facecolor("white")

        for col, algo in enumerate(algos):
            by_arm = by_algo[algo]
            n_point = by_arm["point"]["pos"].shape[0]
            for row, metric in enumerate(METRICS):
                ax = axes[row, col]
                ax.set_facecolor("white")
                style_axis_paper(ax)
                for name in ARM_NAMES:
                    band_fn(
                        ax, grid, by_arm[name][metric], ARM_LABELS[name],
                        COLORS[name],
                    )
                ax.set_ylim(bottom=0.0)
                if row == 0:
                    ax.set_title(
                        f"{ALGO_LABELS.get(algo, algo)} (n={n_point})",
                        color=PAPER_INK,
                    )
                if row == len(METRICS) - 1:
                    ax.set_xlabel("Time [s]", color=PAPER_INK)
                if col == 0:
                    ax.set_ylabel(METRIC_LABELS[metric], color=PAPER_INK)

        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(
            handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.04),
            ncol=len(labels), frameon=False, labelcolor=PAPER_INK,
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

    by_algo = {}
    for algo in algos:
        by_arm = {
            name: load_arm(DATA_DIR / f"{name}_T_{algo}", grid)
            for name in ARM_NAMES
        }
        by_algo[algo] = by_arm
        out_dir = RESULTS_DIR / algo
        draw_figure(
            by_arm, grid, band_median_iqr, "median +/- IQR", algo,
            out_dir / "real_world_pose_error_split_median_iqr.png",
        )
        draw_figure(
            by_arm, grid, band_mean_std, "mean +/- std", algo,
            out_dir / "real_world_pose_error_split_mean_std.png",
        )

    paper_dir = RESULTS_DIR / "paper"
    draw_paper_figure(
        by_algo, grid, band_median_iqr,
        paper_dir / "pose_error_split_median_iqr.pdf",
    )
    draw_paper_figure(
        by_algo, grid, band_mean_std,
        paper_dir / "pose_error_split_mean_std.pdf",
    )


if __name__ == "__main__":
    main()
