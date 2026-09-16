"""Interactive Push -- plain regime (no DR, no state uncertainty).

The base task, for tuning rewards and studying a single/multi-parameter
**planner** model mismatch (``--mismatch``): the planner rolls out a wrong
model while the viewer sims the truth. See ``examples/mismatch.py``.

Run::

    uv run python examples/simple/push.py mppi
    uv run python examples/simple/push.py ps --scenario 3
    uv run python examples/simple/push.py cem --mismatch friction=0.5 mass=2.0
"""

from __future__ import annotations

import argparse

from bampc.config import planner as planner_profiles
from bampc.config.numerics import load as load_numerics
from bampc.dr import DomainRandomizer
from bampc.planner import PlannerConfig, build_planner
from bampc.rollout import WarpRolloutEngine
from bampc.sim import run_interactive
from bampc.task.push import Push
from examples.flags import add_run_args
from examples.latency import add_latency_args
from examples.mismatch import add_mismatch_arg, mismatch_spec
from examples.scenario import add_scenario_args, initial_state, resolve_scenario
from examples.status import CostStatus

# --------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------- #

parser = argparse.ArgumentParser(
    description="Interactive Push (plain).",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "algorithm", nargs="?", default="ps", choices=["mppi", "ps", "cem"],
    help="Sampling algorithm.",
)
add_scenario_args(parser, "push")
add_mismatch_arg(parser)
add_run_args(parser)
add_latency_args(parser)
args = parser.parse_args()

# --------------------------------------------------------------------- #
# Config: numerics / task / planner  (from the shared global configs)
# --------------------------------------------------------------------- #

NUMERICS = load_numerics("push")

BANK, SCENARIO, SHAPE, SCALE, GOAL_DRIFT = resolve_scenario(args, "push")

PLANNER_KW = planner_profiles.load("push", algo=args.algorithm)
REWARD_KW = {}  # Push has none of its own.
PLANNER_KW["settle_steps"] = args.settle_steps
PLANNER = PlannerConfig(algo=args.algorithm, **PLANNER_KW)

PLAN_FREQ = 30
RECORDING = False

# --------------------------------------------------------------------- #
# Build: task -> (optional single-model mismatch) -> engine -> planner
# --------------------------------------------------------------------- #

task = Push(shape=SHAPE, scale=SCALE, goal_drift=GOAL_DRIFT,
            model_config=NUMERICS, **REWARD_KW)

# --mismatch builds a single fixed WRONG model (R=1) for the planner's engine;
# the viewer keeps stepping the pristine task.mj_model, so this is a genuine
# planner-vs-reality mismatch, not randomization.
spec = mismatch_spec(task, args.mismatch)
randomizer = DomainRandomizer(task, 1, spec) if spec else None
engine = WarpRolloutEngine(
    task,
    num_samples=PLANNER.num_samples,
    num_randomizations=1,
    randomizer=randomizer,
    record_traces=True,
)
planner = build_planner(PLANNER, task, engine)

# --------------------------------------------------------------------- #
# Initial state -- from the frozen scenario bank
# --------------------------------------------------------------------- #

mj_model = task.mj_model
mj_data = initial_state(task, BANK, SCENARIO)

# --------------------------------------------------------------------- #
# Viewer
# --------------------------------------------------------------------- #


cost_status = CostStatus(task)


if args.mismatch:
    print(f"planner mismatch: {args.mismatch}  (viewer sims the truth)")

run_interactive(
    planner,
    mj_model,
    mj_data,
    frequency=PLAN_FREQ,
    show_traces=True,
    show_endpoints=True,
    max_traces=30,
    status_callback=cost_status,
    plan_lag_steps=args.plan_lag_steps,
    compensate_latency=args.compensate_latency,
    duration=args.duration,
    record=RECORDING,
    record_format="gif",
    record_camera="main",
    record_name=f"push_{args.algorithm}",
)
