"""Bisect the real-time R=1 sample budget for the t-ou/ grid's shape.

Mirrors `experiments/scaling/run.py`'s `_find_realtime_budget` (same
real-time criterion: mean_ms + 2*std_ms <= period_ms), but builds the task
and planner straight from the actual sweep config via
`experiments.common.uncertainty`, matching production exactly (plan_horizon,
settle_steps, numerics, reward weights) instead of the scaling harness's own
separate task-construction path.

Run::

    uv run python scripts/probing/bisect_noise_grid_budget.py
"""

from __future__ import annotations

import gc
import time
from dataclasses import replace

import numpy as np

from bampc.planner import build_planner
from bampc.planner.base import StateSnapshot
from bampc.rollout import WarpRolloutEngine
from experiments.common.uncertainty.loader import load_run_config

CONFIG = (
    "experiments/push_fr3/state_uncertainty/t-ou/scale0.6-tau2.0-warmup50/"
    "config.yaml"
)
REALTIME_MARGIN_STD = 2.0
BUDGET_SEARCH_BOUNDS = (64, 65536)
REPS, WARMUP = 40, 8


def _time_one(num_samples: int) -> tuple[float, float]:
    cfg = load_run_config(CONFIG)
    task = cfg.make_prediction_task()
    engine = WarpRolloutEngine(
        task, num_samples=num_samples, num_randomizations=1
    )
    planner_cfg = replace(cfg.planner, num_samples=num_samples)
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
    cfg = load_run_config(CONFIG)
    period_ms = 1000.0 / cfg.plan_freq_hz
    print(
        f"period budget: {period_ms:.2f} ms "
        f"(plan_freq_hz={cfg.plan_freq_hz})"
    )

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


if __name__ == "__main__":
    main()
