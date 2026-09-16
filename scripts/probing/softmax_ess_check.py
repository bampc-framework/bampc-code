"""Is MPPI's softmax actually averaging, or is it an argmin in disguise?

MPPI weights samples by ``w ~ exp(-(J - J.min()) / temperature)``. That target
is defined by the temperature and the sampling noise alone, so **more samples
should just be a better estimate of the same update** -- unlike PS (argmin) and
CEM (top-k), whose targets are defined in terms of the drawn set and therefore
move with the sample count. That makes MPPI the one algorithm that needs no
adjustment when arms run at different ``num_samples``.

That argument only holds if the weights are actually spread. The effective
sample size

    ESS = 1 / sum(w^2)

says how many samples the update really averages: ``S`` when the weights are
uniform, **1** when one sample takes all the mass. At ESS ~ 1 the softmax is a
hard argmin, MPPI inherits PS's dependence on the sample count, and any claim
that it smooths *more* than CEM is backwards.

The temperature has to be read against the **cost scale**, which is a property
of the task's weights, not of MPPI -- so it goes stale silently whenever the
cost is retuned. This prints both, at each arm's sample count.

Run (from the repo root)::

    uv run python scripts/probing/softmax_ess_check.py
    uv run python scripts/probing/softmax_ess_check.py --steps 8 \
        --version t-ou/scale0.6-tau2.0-warmup50
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bampc.planner import build_planner  # noqa: E402
from bampc.planner.base import StateSnapshot  # noqa: E402
from bampc.rollout import WarpRolloutEngine  # noqa: E402
from experiments.common.uncertainty.loader import (  # noqa: E402
    load_run_config,
)

SWEEP = "experiments/push_fr3/state_uncertainty"
# The two engine shapes in play: a full-batch point arm and a narrow ensemble.
ARMS = ("point", "ensemble_exact")


def ess_at(cfg, arm: str, steps: int, lam: float | None = None) -> dict:
    """Plan a few steps at one arm's shape and summarise its softmax."""
    r, s = cfg.engine_shape(arm)
    task = cfg.make_prediction_task()
    engine = WarpRolloutEngine(task, num_samples=s, num_randomizations=r)
    pc = replace(cfg.planner, algo="mppi", num_samples=s)
    if lam is not None:
        pc = replace(pc, temperature=lam)
    planner = build_planner(pc, task, engine)
    md = cfg.load_start(task, 0)
    params = planner.init_params()

    ess, spread, scale = [], [], []
    for _ in range(steps):
        state = StateSnapshot(
            qpos=md.qpos.copy(), qvel=md.qvel.copy(), time=0.0
        )
        params, info = planner.optimize(state, params)
        # Recompute the weights the update just used, from its own costs.
        c = info.sample_costs
        w = np.exp(-(c - c.min()) / planner.temperature)
        w /= w.sum()
        ess.append(1.0 / np.sum(w**2))
        spread.append(float(c.max() - c.min()))
        scale.append(float(np.abs(c).mean()))
    return {
        "R": r, "S": s,
        "ess": float(np.median(ess)),
        "spread": float(np.median(spread)),
        "scale": float(np.median(scale)),
    }


def main() -> None:
    """Report the effective sample size at each arm's shape."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--version", default="t-ou/scale0.6-tau2.0-warmup50",
        help="Version under experiments/push_fr3/state_uncertainty/.",
    )
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument(
        "--temperature", type=float, action="append", default=None,
        help="Try this temperature instead of the config's (repeatable). "
             "Use it to pick a value from the ESS rather than by feel.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    cfg = load_run_config(root / SWEEP / args.version / "config.yaml")
    lams = args.temperature or [cfg.planner.temperature]
    print(f"\nconfig temperature = {cfg.planner.temperature:g}\n")
    print(f"{'lambda':>8s} {'arm':16s} {'S':>6s} {'ESS':>9s} {'ESS/S':>8s} "
          f"{'cost spread':>12s} {'cost scale':>11s}  verdict")

    for lam, arm in ((x, a) for x in lams for a in ARMS):
        m = ess_at(cfg, arm, args.steps, lam)
        frac = m["ess"] / m["S"]
        verdict = (
            "ARGMIN -- softmax inert" if m["ess"] < 2.0
            else "very peaked" if frac < 0.01
            else "selective" if frac < 0.25
            else "near-uniform -- barely discriminates"
        )
        print(f"{lam:8g} {arm:16s} {m['S']:6d} {m['ess']:9.2f} {frac:8.2%} "
              f"{m['spread']:12.3e} {m['scale']:11.3e}  {verdict}")

    print(
        "\nESS is how many samples the update really averages. At ~1 the\n"
        "softmax is a hard argmin: MPPI then inherits PS's dependence on the\n"
        "sample count, and any claim that it smooths more than CEM is\n"
        "backwards. The fix is to raise the temperature toward the cost\n"
        "spread, not to change the algorithm."
    )


if __name__ == "__main__":
    main()
