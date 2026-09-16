"""Plan-time scaling benchmark: writes data only, ``analysis.py`` plots it.

Measures wall time of one real ``planner.optimize()`` call (stage + rollout,
i.e. everything ``SamplingPlanner.optimize`` times per replan --
``bampc/planner/base.py``) against two axes: total sample count
``nworld = num_samples * num_randomizations`` (flattened, so ``R``/``S`` are
not tracked separately) and planning horizon in solver steps. Each point
runs in its own subprocess, and results are written as CSV.

Three families of sweep, all over the same ``(task, num_samples,
horizon_steps)`` timing primitive:

* **representative** -- generic ``Push``, other axis fixed at a realistic
  mid-range value (H=50 steps / S=4096).
* **isolated** -- generic ``Push``, other axis fixed at its degenerate
  minimum (H=1 step / S=1), to separate each axis's marginal effect from
  the multiplicative coupling ``plan_time ~= floor + H * per_step_cost(S)``.
  Not strictly "more correct" than the representative curves: S=1 removes
  batch parallelism, so it understates the real H-slope at any realistic
  batch width; H=1 mostly measures the launch/sync/copyback floor rather
  than resolving the S-effect.
* **practice** -- the FR3 (arm) tasks, each built from the matching
  ``experiments/<task>/model_mismatch/<version>/config.yaml`` (task_params,
  numerics profile, planner profile + overrides -- the exact setup that
  sweep will run), other axis fixed at that config's own value. Shows what
  the idealized ``Push`` scaling laws above mean for the heavier models
  actually used. ``curling_fr3`` is the exception: it has no
  ``model_mismatch`` sweep, so it is built from its own
  ``configs/{planner,numerics,reward}/curling.yaml`` profiles instead.

``--tasks`` restricts the practice and budget passes to a subset; the CSVs
they write are merged, so a subset run replaces only its own tasks' rows
and leaves the rest on disk. ``--skip-generic`` skips the two ``Push``
families entirely.

A final pass bisection-searches, per practice task, the largest
``num_samples`` whose ``mean_ms + 2*std_ms`` still fits that task's own
``plan_freq_hz`` replan period -- the real-time-capable sampling budget --
and (with ``--apply-budget``) writes it into that task's
``planner.overrides.sample_budget`` in its ``model_mismatch`` config.

Run (from the repo root)::

    uv run python -m experiments.scaling.run
    uv run python -m experiments.scaling.run --mode smoke
    uv run python -m experiments.scaling.run --apply-budget  # patch configs
    uv run python -m experiments.scaling.run --skip-generic \
        --tasks push_fr3 balance_fr3 curling_fr3
"""

from __future__ import annotations

import argparse
import csv
import gc
import re
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from bampc.config import planner as planner_profiles
from bampc.config import reward as reward_profiles
from bampc.config import scenarios
from bampc.config.numerics import load as load_numerics
from bampc.planner import PlannerConfig, StateSnapshot, build_planner
from bampc.rollout import WarpRolloutEngine
from bampc.task.curling_fr3 import CurlingFr3
from bampc.task.push import Push
from experiments.model_mismatch.harness import load_run_config

RESULTS_DIR = Path(__file__).resolve().parent / "results"
EXPERIMENTS_ROOT = Path(__file__).resolve().parent.parent
# A task's model_mismatch versions all share one planner/prediction/
# task_params setup, so which one is named here only picks the replan rate
# the budget pass measures against: 10 Hz for push_fr3, 20 Hz for
# balance_fr3.
PRACTICE_CONFIGS = {
    "push_fr3": EXPERIMENTS_ROOT / "push_fr3" / "model_mismatch"
    / "mass_truth_10hz_v1" / "config.yaml",
    "balance_fr3": EXPERIMENTS_ROOT / "balance_fr3" / "model_mismatch"
    / "mass_truth_20hz_v1" / "config.yaml",
}
# Curling-FR3 has no model_mismatch sweep to read a rate off.
CURLING_PLAN_FREQ_HZ = 10.0  # scripts/curling/friction_id.py's PLAN_FREQ
REALTIME_MARGIN_STD = 2.0
BUDGET_SEARCH_BOUNDS = (64, 65536)

SAMPLES_GRID = tuple(
    int(n) for n in sorted(set(np.geomspace(256, 65536, 10).round()))
)
HORIZON_GRID = tuple(
    int(h) for h in sorted(set(np.geomspace(10, 350, 8).round()))
)
FIXED_HORIZON_STEPS = 50
FIXED_SAMPLES = 4096


# --------------------------------------------------------------------- #
# Task registry -- (task, base PlannerConfig). ``push`` is a fixed
# representative/isolated stand-in; the FR3 "practice" tasks are built
# straight from their model_mismatch config below.
# --------------------------------------------------------------------- #


def _build_push():
    task = Push(shape="t")
    config = PlannerConfig(
        algo="ps", risk="worstcase", noise_level=0.2, num_samples=128,
        plan_horizon=0.5, num_knots=5, spline_type="zero",
    )
    return task, config


def _build_curling():
    """Curling-FR3 from its own profiles -- it has no model_mismatch sweep.

    No randomizer: this benchmark is ``R=1`` everywhere.
    """
    bank = scenarios.load("curling_circle")
    task = CurlingFr3(
        shape=bank.shape,
        scale=bank.scale,
        goal_xy=bank.goal_xy,
        model_config=load_numerics("curling"),
        **reward_profiles.load("curling"),
    )
    config = PlannerConfig(algo="mppi", **planner_profiles.load("curling"))
    return task, config


def _build_practice(task_name: str):
    """Task + PlannerConfig, built like model_mismatch would build them.

    ``algo`` is already resolved via that config's ``overrides:``.
    """
    cfg = load_run_config(PRACTICE_CONFIGS[task_name])
    return cfg.make_prediction_task(), cfg.planner


TASKS = {
    "push": _build_push,
    "push_fr3": lambda: _build_practice("push_fr3"),
    "balance_fr3": lambda: _build_practice("balance_fr3"),
    "curling_fr3": _build_curling,
}
PRACTICE_TASKS = ("push_fr3", "balance_fr3", "curling_fr3")


def _plan_freq_hz(task_name: str) -> float:
    """That task's replan rate, the real-time bar the budget pass uses."""
    if task_name == "curling_fr3":
        return CURLING_PLAN_FREQ_HZ
    return load_run_config(PRACTICE_CONFIGS[task_name]).plan_freq_hz


def _fixed_point(task_name: str) -> tuple[int, int]:
    """That task's own example ``(num_samples, horizon_steps)``."""
    task, config = TASKS[task_name]()
    return config.num_samples, int(round(config.plan_horizon / task.dt))


def _time_one(
    task_name: str, num_samples: int, horizon_steps: int, reps: int,
    warmup: int,
) -> tuple[float, float]:
    """Mean/std wall-ms of one ``planner.optimize()`` call.

    ``num_randomizations=1``, so ``nworld == num_samples`` -- the domain axis
    stays flat, matching every other knob in this benchmark.
    """
    task, base_config = TASKS[task_name]()
    config = replace(
        base_config,
        num_samples=num_samples,
        plan_horizon=horizon_steps * task.dt,
    )
    engine = WarpRolloutEngine(
        task, num_samples=num_samples, num_randomizations=1
    )
    planner = build_planner(config, task, engine)
    state = StateSnapshot(
        qpos=np.zeros(task.mj_model.nq, np.float64),
        qvel=np.zeros(task.mj_model.nv, np.float64),
    )
    params = planner.init_params()

    for _ in range(warmup):
        params, _ = planner.optimize(state, params)
    times_ms = []
    for _ in range(reps):
        t0 = time.perf_counter()
        params, info = planner.optimize(state, params)
        times_ms.append((time.perf_counter() - t0) * 1e3)
    del planner, engine, task
    gc.collect()
    return float(np.mean(times_ms)), float(np.std(times_ms))


def _run_subprocess(
    task_name: str, num_samples: int, horizon_steps: int, reps: int,
    warmup: int,
) -> tuple[float, float]:
    """Time one point in a fresh subprocess; NaN on OOM/failure.

    Device memory never accumulates across configs -- a high-``nworld`` point
    can OOM the dense contact Jacobian if run in the same process as prior
    points.
    """
    out = subprocess.run(
        [
            sys.executable, __file__, "--single",
            "--task", task_name,
            "--num-samples", str(num_samples),
            "--horizon-steps", str(horizon_steps),
            "--reps", str(reps),
            "--warmup", str(warmup),
        ],
        capture_output=True, text=True, check=False,
    )
    for line in out.stdout.splitlines():
        parts = line.split()
        if len(parts) == 2:
            try:
                return float(parts[0]), float(parts[1])
            except ValueError:
                pass
    return float("nan"), float("nan")


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _merge_csv(path: Path, rows: list[dict], tasks: tuple[str, ...]) -> None:
    """Write ``rows``, keeping any on-disk rows for tasks not in ``tasks``.

    A ``--tasks`` subset run must not drop the tasks it did not time from
    the figures that read this CSV.
    """
    kept: list[dict] = []
    if path.exists():
        with path.open(newline="") as f:
            kept = [r for r in csv.DictReader(f) if r["task"] not in tasks]
    _write_csv(path, rows + kept)


def _sweep_over_samples(
    task_name: str, horizon_steps: int, reps: int, warmup: int,
    grid=SAMPLES_GRID,
) -> list[dict]:
    rows = []
    for n in grid:
        mean_ms, std_ms = _run_subprocess(
            task_name, n, horizon_steps, reps, warmup
        )
        print(f"{task_name:>12} samples={n:>6} horizon={horizon_steps:>4}"
              f"steps  {mean_ms:8.3f} +- {std_ms:6.3f} ms")
        rows.append({
            "task": task_name,
            "nworld": n,
            "num_samples": n,
            "horizon_steps": horizon_steps,
            "mean_ms": mean_ms,
            "std_ms": std_ms,
        })
    return rows


def _sweep_over_horizon(
    task_name: str, num_samples: int, reps: int, warmup: int,
    grid=HORIZON_GRID,
) -> list[dict]:
    rows = []
    for h in grid:
        mean_ms, std_ms = _run_subprocess(
            task_name, num_samples, h, reps, warmup
        )
        print(f"{task_name:>12} samples={num_samples:>6} horizon={h:>4}"
              f"steps  {mean_ms:8.3f} +- {std_ms:6.3f} ms")
        rows.append({
            "task": task_name,
            "horizon_steps": h,
            "num_samples": num_samples,
            "nworld": num_samples,
            "mean_ms": mean_ms,
            "std_ms": std_ms,
        })
    return rows


def _find_realtime_budget(
    task_name: str, horizon_steps: int, period_ms: float, reps: int,
    warmup: int,
) -> tuple[int, float, float]:
    """Largest real-time-capable ``num_samples`` in ``BUDGET_SEARCH_BOUNDS``.

    "Real-time" means ``mean_ms + REALTIME_MARGIN_STD*std_ms <= period_ms``.
    Bisects (assuming plan time is non-decreasing in ``num_samples``)
    rather than reading the coarse ``SAMPLES_GRID`` off the sweep above --
    that grid is too sparse to land on a precise, real-time-optimal
    budget.
    """
    lo, hi = BUDGET_SEARCH_BOUNDS

    def _probe(n: int) -> tuple[bool, float, float]:
        mean_ms, std_ms = _run_subprocess(
            task_name, n, horizon_steps, reps, warmup
        )
        ok = mean_ms + REALTIME_MARGIN_STD * std_ms <= period_ms
        return ok, mean_ms, std_ms

    lo_ok, lo_mean, lo_std = _probe(lo)
    if not lo_ok:
        print(f"  WARNING: {task_name} not real-time capable even at "
              f"num_samples={lo} ({lo_mean:.2f}+-{lo_std:.2f} ms vs "
              f"{period_ms:.2f} ms budget)")
        return lo, lo_mean, lo_std

    hi_ok, hi_mean, hi_std = _probe(hi)
    if hi_ok:
        print(f"  {task_name} fits real-time even at the search ceiling "
              f"num_samples={hi} ({hi_mean:.2f}+-{hi_std:.2f} ms)")
        return hi, hi_mean, hi_std

    best_n, best_mean, best_std = lo, lo_mean, lo_std
    while hi - lo > 1:
        mid = (lo + hi) // 2
        ok, mean_ms, std_ms = _probe(mid)
        print(f"  {task_name} bisect num_samples={mid:>6}  "
              f"{mean_ms:8.3f} +- {std_ms:6.3f} ms  "
              f"{'OK' if ok else 'too slow'}")
        if ok:
            lo, best_n, best_mean, best_std = mid, mid, mean_ms, std_ms
        else:
            hi = mid
    return best_n, best_mean, best_std


def _patch_budget(path: Path, num_samples: int) -> None:
    """Rewrite only the ``sample_budget:`` value in a model_mismatch config.

    It lives inside the inline ``planner.overrides`` mapping. A full YAML
    round-trip would strip the config's hand-written comments.
    """
    text = path.read_text()
    new_text, count = re.subn(
        r"(sample_budget:\s*)\d+", rf"\g<1>{num_samples}", text
    )
    if count != 1:
        raise ValueError(
            f"expected exactly one 'sample_budget:' value in {path}, "
            f"found {count}"
        )
    path.write_text(new_text)
    print(f"  patched {path} -> sample_budget: {num_samples}")


def main() -> None:
    """Run every sweep, or one subprocess-mode point with ``--single``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--single", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--task", default="push", choices=list(TASKS))
    parser.add_argument("--num-samples", type=int)
    parser.add_argument("--horizon-steps", type=int)
    parser.add_argument("--reps", type=int, default=40)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument(
        "--mode", choices=["full", "smoke"], default="full",
        help="smoke = reps 5, warmup 2, written to results/smoke/",
    )
    parser.add_argument(
        "--tasks", nargs="+", default=list(PRACTICE_TASKS),
        choices=list(PRACTICE_TASKS), metavar="TASK",
        help="practice tasks to time (default: all); their rows replace "
             "only their own in the practice/budget CSVs",
    )
    parser.add_argument(
        "--skip-generic", action="store_true",
        help="skip the representative and isolated generic-Push sweeps, "
             "leaving their CSVs untouched",
    )
    parser.add_argument(
        "--apply-budget", action="store_true",
        help="patch the determined real-time budget into each practice "
             "task's model_mismatch config.yaml (off by default so a plain "
             "run never mutates tracked files)",
    )
    args = parser.parse_args()

    if args.single:
        # Subprocess mode: one point, print "mean_ms std_ms" and exit.
        mean_ms, std_ms = _time_one(
            args.task, args.num_samples, args.horizon_steps, args.reps,
            args.warmup,
        )
        print(f"{mean_ms:.4f} {std_ms:.4f}")
        return

    reps, warmup = args.reps, args.warmup
    results = RESULTS_DIR
    if args.mode == "smoke":
        reps, warmup, results = 5, 2, RESULTS_DIR / "smoke"
    results.mkdir(parents=True, exist_ok=True)
    tasks = tuple(args.tasks)

    if args.skip_generic:
        print("=== skipping the generic-Push sweeps (--skip-generic) ===")
    else:
        _run_generic_sweeps(reps, warmup, results)

    print(f"\n=== practice: nworld sweep ({' / '.join(tasks)}) ===")
    practice_samples_rows: list[dict] = []
    for name in tasks:
        _, horizon_steps = _fixed_point(name)
        practice_samples_rows += _sweep_over_samples(
            name, horizon_steps, reps, warmup
        )
    _merge_csv(results / "practice_samples_sweep.csv",
               practice_samples_rows, tasks)

    print(f"\n=== practice: horizon sweep ({' / '.join(tasks)}) ===")
    practice_horizon_rows: list[dict] = []
    for name in tasks:
        num_samples, _ = _fixed_point(name)
        practice_horizon_rows += _sweep_over_horizon(
            name, num_samples, reps, warmup
        )
    _merge_csv(results / "practice_horizon_sweep.csv",
               practice_horizon_rows, tasks)

    print(f"\n=== real-time budget ({' / '.join(tasks)}) ===")
    budget_rows: list[dict] = []
    for name in tasks:
        plan_freq_hz = _plan_freq_hz(name)
        _, horizon_steps = _fixed_point(name)
        period_ms = 1000.0 / plan_freq_hz
        n, mean_ms, std_ms = _find_realtime_budget(
            name, horizon_steps, period_ms, reps, warmup
        )
        print(f"{name:>12} plan_freq_hz={plan_freq_hz:>5.1f}  "
              f"period={period_ms:7.2f}ms  budget={n:>6}  "
              f"{mean_ms:8.3f} +- {std_ms:6.3f} ms")
        budget_rows.append({
            "task": name,
            "plan_freq_hz": plan_freq_hz,
            "period_ms": period_ms,
            "num_samples": n,
            "mean_ms": mean_ms,
            "std_ms": std_ms,
        })
        if args.apply_budget:
            if name == "curling_fr3":
                print("  curling_fr3 has no model_mismatch config to patch")
            else:
                _patch_budget(PRACTICE_CONFIGS[name], n)
    _merge_csv(results / "realtime_budget.csv", budget_rows, tasks)


def _run_generic_sweeps(reps: int, warmup: int, results: Path) -> None:
    """The representative and isolated generic-``Push`` sweeps."""
    print("=== representative: nworld sweep "
          f"(push, horizon={FIXED_HORIZON_STEPS} steps) ===")
    rows = _sweep_over_samples("push", FIXED_HORIZON_STEPS, reps, warmup)
    _write_csv(results / "nworld_sweep.csv", rows)

    print(f"\n=== representative: horizon sweep "
          f"(push, num_samples={FIXED_SAMPLES}) ===")
    rows = _sweep_over_horizon("push", FIXED_SAMPLES, reps, warmup)
    _write_csv(results / "horizon_sweep.csv", rows)

    print("\n=== isolated: nworld sweep (push, horizon=1 step) ===")
    rows = _sweep_over_samples("push", 1, reps, warmup)
    _write_csv(results / "isolated_samples.csv", rows)

    print("\n=== isolated: horizon sweep (push, num_samples=1) ===")
    rows = _sweep_over_horizon("push", 1, reps, warmup)
    _write_csv(results / "isolated_horizon.csv", rows)


if __name__ == "__main__":
    main()
