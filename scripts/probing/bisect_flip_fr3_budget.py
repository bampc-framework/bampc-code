"""Bisect the real-time R=1 sample budget for Flip-FR3's planner profile.

Mirrors `scripts/probing/bisect_noise_grid_budget.py` (same real-time
criterion: mean_ms + 2*std_ms <= period_ms), but built directly from the
plain named profiles (`configs/planner/flip_fr3.yaml`,
`configs/reward/flip_fr3.yaml`, `configs/numerics/flip_fr3.yaml`) instead
of an `experiments/*/state_uncertainty` sweep config -- Flip-FR3 has no such
sweep yet.

The printed budget goes to the real-robot launcher's `--sample-budget`
(`scripts/ros/run_planner_node_flip_fr3.py`), which resolves it per
`--domains` via `bampc.planner.engine_shape`. It does **not** belong
in `configs/planner/flip_fr3.yaml`: a profile is shared across machines and
this number is not. It must be measured on the actual GPU -- do not
hand-guess one.

It is only valid for the shape it was measured at. `PLAN_RATE_HZ` matches the
launcher's `--plan-rate` default, and the profile's `settle_steps` is baked
into the timing (every settle step is another physics step in every world),
so changing either one means re-running this with a matching `PLAN_RATE_HZ`
or profile.

Run::

    uv run python scripts/probing/bisect_flip_fr3_budget.py
"""

from __future__ import annotations

import gc
import time
from dataclasses import replace

import numpy as np

from bampc.config import planner as planner_profiles
from bampc.config import reward as reward_profiles
from bampc.config.numerics import load as load_numerics
from bampc.planner import PlannerConfig, build_planner
from bampc.planner.base import StateSnapshot
from bampc.rollout import WarpRolloutEngine
from bampc.task.flip_fr3 import FlipFr3

PLAN_RATE_HZ = 10.0
REALTIME_MARGIN_STD = 2.0
BUDGET_SEARCH_BOUNDS = (64, 65536)
REPS, WARMUP = 40, 8


def _time_one(num_samples: int) -> tuple[float, float]:
    numerics = load_numerics("flip_fr3")
    reward_kw = reward_profiles.load("flip_fr3")
    planner_kw = planner_profiles.load("flip_fr3")
    base_planner_cfg = PlannerConfig(algo="ps", **planner_kw)
    task = FlipFr3(
        model_config=numerics,
        ctrl_range_scale=base_planner_cfg.ctrl_range_scale,
        **reward_kw,
    )
    engine = WarpRolloutEngine(
        task, num_samples=num_samples, num_randomizations=1
    )
    planner_cfg = replace(base_planner_cfg, num_samples=num_samples)
    planner = build_planner(planner_cfg, task, engine)
    state = StateSnapshot(
        qpos=np.zeros(task.mj_model.nq, np.float64),
        qvel=np.zeros(task.mj_model.nv, np.float64),
    )
    params = planner.init_params()
    for _ in range(WARMUP):
        params, _ = planner.optimize(state, params)
    times_ms = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        params, _ = planner.optimize(state, params)
        times_ms.append((time.perf_counter() - t0) * 1e3)
    del planner, engine, task
    gc.collect()
    return float(np.mean(times_ms)), float(np.std(times_ms))


def main() -> None:
    """Bisect and print the real-time-capable budget."""
    period_ms = 1000.0 / PLAN_RATE_HZ
    print(f"period budget: {period_ms:.2f} ms  (plan_rate_hz={PLAN_RATE_HZ})")

    lo, hi = BUDGET_SEARCH_BOUNDS

    def _probe(n: int) -> bool:
        mean_ms, std_ms = _time_one(n)
        ok = mean_ms + REALTIME_MARGIN_STD * std_ms <= period_ms
        print(
            f"  num_samples={n:>6}  {mean_ms:8.3f} +- {std_ms:6.3f} ms  "
            f"{'OK' if ok else 'too slow'}"
        )
        return ok

    if not _probe(lo):
        print(f"WARNING: not real-time capable even at num_samples={lo}")
        return
    if _probe(hi):
        print(f"fits real-time even at the search ceiling num_samples={hi}")
        return

    best = lo
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if _probe(mid):
            lo, best = mid, mid
        else:
            hi = mid
    print(f"\nreal-time budget: {best}")
    print(f"pass it as --sample-budget {best} to "
          "scripts/ros/run_planner_node_flip_fr3.py")


if __name__ == "__main__":
    main()
