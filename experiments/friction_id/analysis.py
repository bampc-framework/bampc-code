"""Figures and metrics for the friction_id sweep.

Reads what ``run.py`` wrote and owns **every** derived number, so a metric
can change without re-running the GPU. Score is per-attempt puck-to-house
miss distance (metres) and whether the shot scored; median and IQR across
seeds (repo convention, robust to MJWarp's per-run bimodal
non-determinism), never mean and std.

Writes one **headline** figure -- median ``avg_miss``/``best_miss`` per
``(algo, arm)`` across seeds, with IQR error bars -- plus a
success-rate/avg-miss/best-miss summary table.

Run (from the repo root)::

    ...analysis --version curling_fr3/friction_id/v1
    ...analysis --version curling_fr3/friction_id/v1 --mode smoke

where ``...analysis`` is::

    uv run python -m experiments.friction_id.analysis
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.common.dr.plot_style import (
    AMBER,
    BG,
    BLUE,
    INK,
    MUTED,
    plt,
    style_axis,
)

EXPERIMENTS_ROOT = Path(__file__).resolve().parents[1]

ARM_COLOR = {
    "wrong_estimate": AMBER,
    "wide_hedge": MUTED,
    "adaptive": BLUE,
}
ARM_LABEL = {
    "wrong_estimate": "wrong estimate (fixed)",
    "wide_hedge": "wide hedge (static)",
    "adaptive": "adaptive (belief-collapse)",
}


def load_variant(run_dir: Path) -> dict | None:
    """Read one ``results/<algo>/<arm>/`` dir into (seed, attempt) arrays."""
    csv_path = run_dir / "tracking.csv"
    if not csv_path.is_file():
        return None
    d = np.atleast_1d(np.genfromtxt(
        csv_path, delimiter=",", names=True, dtype=None, encoding="utf-8"
    ))
    manifest = json.loads((run_dir / "manifest.json").read_text())
    n_seeds, n_attempts = manifest["num_seeds"], manifest["attempts"]
    if d.size != n_seeds * n_attempts:
        raise ValueError(
            f"{csv_path}: {d.size} rows is not {n_seeds} seeds x "
            f"{n_attempts} attempts"
        )

    def grid(name: str) -> np.ndarray:
        return d[name].reshape(n_seeds, n_attempts)

    seeds = grid("seed")
    if not np.all(seeds == seeds[:, :1]):
        raise ValueError(f"{csv_path}: rows are not grouped by seed")

    ep = np.atleast_1d(np.genfromtxt(
        run_dir / "episodes.csv", delimiter=",", names=True, dtype=None,
        encoding="utf-8",
    ))
    return {
        "manifest": manifest,
        "flip_every": manifest["flip_every"],
        "attempt": grid("attempt")[0].astype(int),
        "truth_mu": grid("truth_mu").astype(float),
        "belief_mean": grid("belief_mean").astype(float),
        "miss": grid("miss").astype(float),
        "success": grid("success").astype(bool),
        "stage_r": grid("stage_r").astype(int),
        "stage_s": grid("stage_s").astype(int),
        "avg_miss": ep["avg_miss"].astype(float),
        "best_miss": ep["best_miss"].astype(float),
        "success_rate": ep["success_rate"].astype(float),
    }


def load_all(results: Path) -> dict[str, dict[str, dict]]:
    """``{algo: {arm: data}}`` for every ``results/<algo>/<arm>/`` present."""
    out: dict[str, dict[str, dict]] = {}
    for algo_dir in sorted(p for p in results.iterdir() if p.is_dir()):
        arms = {}
        for arm_dir in sorted(p for p in algo_dir.iterdir() if p.is_dir()):
            data = load_variant(arm_dir)
            if data is not None:
                arms[arm_dir.name] = data
        if arms:
            out[algo_dir.name] = arms
    if not out:
        raise SystemExit(f"no runs found under {results}")
    return out


def arm_order(runs: dict[str, dict[str, dict]]) -> list[str]:
    """Every arm present, in ``ARM_COLOR``'s fixed order."""
    present = {arm for arms in runs.values() for arm in arms}
    return [a for a in ARM_COLOR if a in present]


def headline_fig(runs: dict[str, dict[str, dict]], out_path: Path) -> None:
    """Grouped bars: median ``avg_miss``/``best_miss`` per (algo, arm)."""
    algos = list(runs)
    arms = arm_order(runs)
    width = 0.8 / max(len(arms), 1)
    x = np.arange(len(algos))
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(4.0 * max(len(algos), 1) + 2, 4), facecolor=BG
    )
    for ax, key, title in (
        (ax1, "avg_miss", "average miss over the campaign (m)"),
        (ax2, "best_miss", "best miss over the campaign (m)"),
    ):
        style_axis(ax)
        for i, arm in enumerate(arms):
            vals = [
                runs[algo][arm][key] if arm in runs[algo]
                else np.array([np.nan])
                for algo in algos
            ]
            centers = np.array([np.nanmedian(v) for v in vals])
            los = np.array([np.nanpercentile(v, 25) for v in vals])
            his = np.array([np.nanpercentile(v, 75) for v in vals])
            offset = (i - (len(arms) - 1) / 2) * width
            ax.bar(
                x + offset, centers, width=width,
                yerr=np.vstack([centers - los, his - centers]),
                color=ARM_COLOR[arm], label=ARM_LABEL[arm], capsize=3,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(algos)
        ax.set_title(title, color=INK, fontsize=10)
    ax1.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=BG)
    plt.close(fig)


def summary_table(runs: dict[str, dict[str, dict]]) -> str:
    """One row per ``(algo, arm)``: median success rate / avg / best miss."""
    header = (
        f"{'algo':>6} {'arm':>16} {'success_rate':>13} "
        f"{'avg_miss':>9} {'best_miss':>10}"
    )
    lines = [header]
    for algo, arms in runs.items():
        for arm in arm_order({algo: arms}):
            d = arms[arm]
            lines.append(
                f"{algo:>6} {arm:>16} "
                f"{np.nanmedian(d['success_rate']):>13.2f} "
                f"{np.nanmedian(d['avg_miss']):>9.4f} "
                f"{np.nanmedian(d['best_miss']):>10.4f}"
            )
    return "\n".join(lines)


def main() -> None:
    """Load the version's results and write every figure + the table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--version", required=True,
        help="Path relative to experiments/, e.g. curling_fr3/friction_id/v1.",
    )
    parser.add_argument(
        "--mode", choices=["full", "quick", "smoke"], default="full"
    )
    args = parser.parse_args()

    results = EXPERIMENTS_ROOT / args.version / "results"
    if args.mode != "full":
        results = results / args.mode
    runs = load_all(results)
    n = sum(len(arms) for arms in runs.values())
    print(f"loaded {n} (algo, arm) variants across {len(runs)} algos")

    headline_fig(runs, results / "headline.png")
    print("\n" + summary_table(runs))


if __name__ == "__main__":
    main()
