"""Plot one determinism_check.py CSV: run divergence + contact modes.

Three stacked panels sharing a time axis:

1. block position error -- each run's distance from the across-run consensus,
   mean over runs with a +/-1 std band;
2. the same for block rotation;
3. one dot row per run, coloured by contact mode at each replan.

There is no ground truth here (every run is the *same* scenario), so "error"
means divergence from the run consensus -- symmetric, with no privileged
reference run.

Run (from the repo root)::

    uv run python scripts/probing/plot_determinism.py \
        output/results/push_fr3_determinism/determinism_joint_warp_s512.csv

Writes a PNG beside the CSV. Re-plotting is free; re-running the probe costs
GPU minutes, which is why this reads the CSV rather than the simulator.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bampc.sim.viewer import contact_palette  # noqa: E402
from experiments.common.dr.plot_style import (  # noqa: E402
    AMBER,
    BG,
    BLUE,
    GREEN,
    INK,
    MUTED,
    RED,
    plt,
    style_axis,
)

MODE_NAMES = ("none", "pusher", "table", "both")


def load(path: Path):
    """Read the CSV into ``(times, xpos, xquat, modes)`` shaped by run.

    Returns ``times`` ``(T,)``, ``xpos`` ``(R, T, 3)``, ``xquat``
    ``(R, T, 4)``, ``modes`` ``(R, T)``.
    """
    d = np.genfromtxt(path, delimiter=",", names=True)
    reps = d["repeat"].astype(int)
    R = reps.max() + 1
    T = len(reps) // R
    if len(reps) != R * T:
        raise ValueError(f"{path}: {len(reps)} rows is not {R} runs x {T}")
    # Cheapest guard against a row-ordering mix-up, which would leave every
    # curve plausible-looking and meaningless.
    if not np.all(reps.reshape(R, T) == np.arange(R)[:, None]):
        raise ValueError(f"{path}: rows are not grouped by repeat")

    def col(*names):
        return np.stack([d[n] for n in names], axis=-1).reshape(R, T, -1)

    return (
        d["time"].reshape(R, T)[0],
        col("block_x", "block_y", "block_z"),
        col("block_qw", "block_qx", "block_qy", "block_qz"),
        d["contact_mode"].reshape(R, T).astype(int),
    )


def quat_angle(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Geodesic angle between two quaternion arrays, broadcasting.

    ``4*atan2(|a - s*b|, |a + s*b|)`` rather than ``2*arccos(|dot|)``: arccos
    is vertical at 1, so two *identical* quaternions dot to ``|q|^2`` and a
    1-ulp shortfall reads as ~3e-8 rad -- a floor that would hide the
    bit-identical prefix this figure exists to show. Same formula as
    ``determinism_check.py:pairwise_angle``; duplicated so this file stands
    alone.
    """
    s = np.where(np.sum(a * b, axis=-1) < 0.0, -1.0, 1.0)[..., None]
    return 4.0 * np.arctan2(
        np.linalg.norm(a - s * b, axis=-1),
        np.linalg.norm(a + s * b, axis=-1),
    )


def consensus_errors(xpos: np.ndarray, xquat: np.ndarray):
    """Per-run distance from the across-run consensus. ``(R, T)`` each."""
    pos_err = np.linalg.norm(xpos - xpos.mean(axis=0), axis=-1)

    # Chordal L2 mean: q and -q are the same rotation, so align every run to
    # run 0's hemisphere before averaging, then renormalize. Exact enough at
    # these spreads.
    s = np.where(np.sum(xquat * xquat[0], axis=-1) < 0.0, -1.0, 1.0)[..., None]
    aligned = xquat * s
    mean_q = aligned.mean(axis=0)
    mean_q /= np.linalg.norm(mean_q, axis=-1, keepdims=True)
    return pos_err, quat_angle(aligned, mean_q)


def band(ax, t, err, label, color):
    """Mean-over-runs curve with a +/-1 std band, clipped at zero."""
    mean, std = err.mean(axis=0), err.std(axis=0)
    ax.fill_between(
        t, np.maximum(mean - std, 0.0), mean + std, color=color, alpha=0.22,
        linewidth=0,
    )
    ax.plot(t, mean, color=color, linewidth=1.3, label=label)


def variant(path: Path) -> str:
    """Short label from a filename like ``determinism_free_warp_s256``."""
    parts = path.stem.replace("determinism_", "").split("_")
    if parts[-1].startswith("s"):
        parts = parts[:-1]
    return "/".join(parts)


def compare(paths: list[Path], out: Path) -> None:
    """Overlay several runs: divergence growth, and the basin structure.

    Two panels, because the two findings are different in kind:

    * how fast each variant diverges (log y -- the variants span ~25x, which
      a linear axis would squash into the free block alone). Mean lines only:
      four overlapping std bands are unreadable, and the band is misleading
      here anyway (see the second panel for why).
    * where each individual repeat *lands*. Divergence is not diffusion --
      runs fall into a few discrete basins, several often bit-identical --
      so a strip of per-repeat endpoints says more than any summary spread.
    """
    fig, (ax_t, ax_b) = plt.subplots(
        2, 1, figsize=(9, 6), gridspec_kw={"height_ratios": [3, 2]}
    )
    fig.patch.set_facecolor(BG)
    colors = [BLUE, GREEN, AMBER, RED]
    # While the runs are bit-identical, x - mean(x) is not exactly 0 -- the
    # mean of R identical float64s rounds -- leaving ~1e-16 of pure rounding
    # hash that a log axis renders as if it were signal. Mask it.
    floor = 1e-12

    for i, path in enumerate(paths):
        t, xpos, xquat, _ = load(path)
        pos_err, _ = consensus_errors(xpos, xquat)
        mean = pos_err.mean(axis=0)
        c = colors[i % len(colors)]
        # Median over the last second: these curves oscillate by ~4x between
        # adjacent steps, so a single endpoint sample is not a summary.
        tail = float(np.median(mean[t >= t[-1] - 1.0]))
        ax_t.plot(
            t, np.where(mean > floor, mean, np.nan), color=c, linewidth=1.3,
            label=f"{variant(path)}  (last 1 s median: {tail * 100:.1f} cm)",
        )
        # Endpoint of every repeat relative to run 0: identical values
        # stacking on one x is exactly the basin structure being shown.
        final = np.linalg.norm(xpos - xpos[0], axis=-1)[:, -1]
        ax_b.scatter(
            np.where(final > 0, final, np.nan), np.full(len(final), i),
            color=c, s=40, alpha=0.65, edgecolors="none",
        )

    for ax in (ax_t, ax_b):
        style_axis(ax)
    ax_t.set_yscale("log")
    ax_t.set_ylabel("mean consensus error [m]", color=MUTED)
    ax_t.set_xlabel("time [s]", color=MUTED)
    ax_t.legend(frameon=False, labelcolor=MUTED, fontsize=8)
    ax_t.set_title(
        "divergence between identical runs  "
        "(mean distance from the run consensus; gaps = bit-identical)",
        color=INK, fontsize=10,
    )

    ax_b.set_xscale("log")
    ax_b.set_yticks(range(len(paths)))
    ax_b.set_yticklabels([variant(p) for p in paths], color=MUTED, fontsize=8)
    ax_b.invert_yaxis()
    ax_b.yaxis.grid(False)
    ax_b.xaxis.grid(True, color="#e1e0d9", linewidth=0.8)
    ax_b.set_xlabel(
        "final distance from run 0 [m]  (one dot per repeat; "
        "dots sharing an x are the same basin)", color=MUTED, fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=BG, bbox_inches="tight")
    print(f"wrote {out}")


def main() -> None:
    """Load the CSV, draw the three panels, write the PNG."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("csv", type=Path, nargs="+",
                   help="one CSV for the per-run figure; several to compare")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--plan-freq", type=float, default=10.0)
    p.add_argument("--every", type=int, default=1, help="thin the dot rows")
    p.add_argument(
        "--yscale", default="linear", choices=["linear", "symlog", "log"],
        help=(
            "linear reads correctly for mean+/-std. symlog/log expose the "
            "early growth from machine precision, but wherever std > mean "
            "the band clips to 0 and floods every decade -- read the mean "
            "line, not the band, on those scales."
        ),
    )
    args = p.parse_args()

    if len(args.csv) > 1:
        compare(
            args.csv,
            args.out or args.csv[0].parent / "determinism_compare.png",
        )
        return
    args.csv = args.csv[0]

    t, xpos, xquat, modes = load(args.csv)
    R = xpos.shape[0]
    pos_err, rot_err = consensus_errors(xpos, xquat)

    # Contact mode *at replan* -- that is the state the planner acted on, so
    # the raster samples replan instants rather than thinning arbitrarily.
    dt = float(np.median(np.diff(t)))
    stride = max(int(round((1.0 / args.plan_freq) / dt)), 1) * max(
        args.every, 1
    )
    idx = np.arange(0, len(t), stride)

    fig, axes = plt.subplots(
        3, 1, figsize=(9, 7), sharex=True,
        gridspec_kw={"height_ratios": [3, 3, 2]},
    )
    fig.patch.set_facecolor(BG)
    for ax in axes:
        style_axis(ax)

    band(axes[0], t, pos_err, "mean over runs", BLUE)
    axes[0].set_ylabel("position error [m]", color=MUTED)
    band(axes[1], t, rot_err, "mean over runs", BLUE)
    axes[1].set_ylabel("rotation error [rad]", color=MUTED)
    for ax in axes[:2]:
        ax.set_yscale(args.yscale, **(
            {"linthresh": 1e-12} if args.yscale == "symlog" else {}
        ))
        # The error is a norm, so nothing is ever negative. Without this,
        # symlog draws its whole mirrored negative half -- a dozen empty
        # decades that squash the real data into a sliver.
        ax.set_ylim(bottom=0.0 if args.yscale != "log" else None)
        ax.legend(frameon=False, labelcolor=MUTED, fontsize=8)

    palette = contact_palette(1.0)
    axes[2].scatter(
        np.tile(t[idx], R),
        np.repeat(np.arange(R), len(idx)),
        c=palette[modes[:, idx]].reshape(-1, 4),
        s=14,
        marker="o",
    )
    axes[2].set_ylabel("run", color=MUTED)
    axes[2].set_xlabel("time [s]", color=MUTED)
    axes[2].set_yticks(range(R))
    axes[2].invert_yaxis()
    axes[2].yaxis.grid(False)
    # All four entries even when only some occur -- the legend documents the
    # encoding, which is shared with the viewer's belief ghosts.
    axes[2].legend(
        handles=[
            plt.Line2D([], [], marker="o", linestyle="", color=palette[m],
                       label=f"{m} {MODE_NAMES[m]}")
            for m in range(4)
        ],
        frameon=False, labelcolor=MUTED, fontsize=8, ncol=4,
        loc="upper center", bbox_to_anchor=(0.5, -0.35),
    )

    axes[0].set_title(
        f"{args.csv.stem}  --  {R} identical runs, "
        f"contact mode at {args.plan_freq:g} Hz replan",
        color=INK, fontsize=10,
    )
    fig.tight_layout()
    out = args.out or args.csv.with_suffix(".png")
    fig.savefig(out, dpi=150, facecolor=BG, bbox_inches="tight")
    print(f"wrote {out}  ({R} runs x {len(t)} steps, {len(idx)} dots/run)")


if __name__ == "__main__":
    main()
