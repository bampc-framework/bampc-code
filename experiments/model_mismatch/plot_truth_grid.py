r"""Cross-truth-level model-mismatch comparison, any single-parameter grid.

Reuses ``plot_replan_sensitivity``'s ``plot_sensitivity`` unchanged, with
truth levels standing in for replan rates as the per-panel "leg", for any
``bias_target: truth_grid`` config. One subplot per truth level, sharing a
y-axis, showing final
accumulated pose error vs. the arm's multiplier RELATIVE TO THAT LEVEL's
truth (not the global nominal -- see ``harness.truth_grid_spec``). The
question this answers: does model-mismatch sensitivity to a wrong belief
look the same regardless of what the real value actually is?

Run (from the repo root)::

    ...plot_truth_grid --version \\
        balance_fr3/model_mismatch/rolling_friction_truth_10hz_v1
    ...plot_truth_grid --version push_fr3/model_mismatch/mass_truth_10hz_v1 \\
        --mode quick

where ``...plot_truth_grid`` is::

    uv run python -m experiments.model_mismatch.plot_truth_grid
"""

from __future__ import annotations

import argparse

from experiments.model_mismatch.analysis import (
    EXPERIMENTS_ROOT,
    default_rot_scale,
    load_truth_grid,
    run_bias_params,
)
from experiments.model_mismatch.plot_replan_sensitivity import plot_sensitivity


def _truth_label(dirname: str) -> str:
    """``"truth_1.2"`` -> ``"1.2x truth"``."""
    return f"{dirname[len('truth_'):]}x truth"


def main() -> None:
    """Load every truth-level cell for ``--version`` and plot side by side."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--version", required=True,
        help="Path relative to experiments/, e.g. "
        "balance_fr3/model_mismatch/rolling_friction_truth_10hz_v1.",
    )
    parser.add_argument(
        "--mode", choices=["full", "smoke", "quick"], default="full",
        help="Which results/ subtree to read (see run.py --mode).",
    )
    parser.add_argument(
        "--title", default=None,
        help="Figure title. Default: derived from the swept parameter.",
    )
    args = parser.parse_args()

    leg_dir = EXPERIMENTS_ROOT / args.version
    results = leg_dir / "results"
    if args.mode != "full":
        results = results / args.mode
    out = results / "truth_grid_sensitivity"
    out_meanstd = results / "truth_grid_sensitivity_meanstd"

    grid = load_truth_grid(results)
    legs = [
        (_truth_label(name), grid[name])
        for name in sorted(grid, key=lambda n: float(n[len("truth_"):]))
    ]
    bias_params = run_bias_params(legs[0][1])
    rot_scale = default_rot_scale(legs[0][1])
    param = bias_params[0] if len(bias_params) == 1 else "/".join(bias_params)
    title = (
        args.title
        or f"{param}-mismatch sensitivity across real-world {param} levels"
    )

    plot_sensitivity(legs, bias_params, rot_scale, "median", out, title)
    plot_sensitivity(legs, bias_params, rot_scale, "mean", out_meanstd, title)
    print(f"wrote {out}.png / {out_meanstd}.png (+ .pdf)")


if __name__ == "__main__":
    main()
