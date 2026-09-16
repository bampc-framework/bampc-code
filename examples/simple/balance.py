"""Interactive Balance -- plain regime (no DR, no state uncertainty).

Tilt a plate (2-DOF gimbal) to slide a block to a plate-relative goal. For
tuning rewards and single/multi-parameter planner mismatch (``--mismatch``).

Run::

    uv run python examples/simple/balance.py mppi
    uv run python examples/simple/balance.py cem --shape square --scenario 4
    uv run python examples/simple/balance.py ps --mismatch friction=0.4
"""

from __future__ import annotations

import argparse

from bampc.config import planner as planner_profiles
from bampc.config import reward as reward_profiles
from bampc.config.numerics import load as load_numerics
from bampc.dr import DomainRandomizer
from bampc.planner import PlannerConfig, build_planner
from bampc.rollout import WarpRolloutEngine
from bampc.sim import run_interactive
from bampc.task.balance import Balance
from examples.flags import add_run_args
from examples.latency import add_latency_args
from examples.mismatch import add_mismatch_arg, mismatch_spec
from examples.scenario import add_scenario_args, initial_state, resolve_scenario
from examples.status import CostStatus

# --------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------- #

parser = argparse.ArgumentParser(
    description="Interactive Balance (plain).",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "algorithm", nargs="?", default="ps", choices=["mppi", "ps", "cem"],
    help="Sampling algorithm.",
)
add_scenario_args(parser, "balance")
add_mismatch_arg(parser)
add_run_args(parser)
add_latency_args(parser)
args = parser.parse_args()

# --------------------------------------------------------------------- #
# Config: numerics / task / planner  (from the shared global configs)
# --------------------------------------------------------------------- #

NUMERICS = load_numerics("balance")

BANK, SCENARIO, SHAPE, SCALE, GOAL_DRIFT = resolve_scenario(args, "balance")

PLANNER_KW = planner_profiles.load("balance")
REWARD_KW = reward_profiles.load("balance")
PLANNER_KW["settle_steps"] = args.settle_steps
PLANNER = PlannerConfig(algo=args.algorithm, **PLANNER_KW)

PLAN_FREQ = 15
RECORDING = True

# --------------------------------------------------------------------- #
# Build: task -> (optional single-model mismatch) -> engine -> planner
# --------------------------------------------------------------------- #

task = Balance(shape=SHAPE, scale=SCALE,
               goal_drift=GOAL_DRIFT, model_config=NUMERICS, **REWARD_KW)

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
    show_endpoints=True,
    trace_idxs=[0],
    max_traces=1,
    status_callback=cost_status,
    plan_lag_steps=args.plan_lag_steps,
    compensate_latency=args.compensate_latency,
    duration=args.duration,
    record=RECORDING,
    record_format="gif",
    record_camera="main",
    record_name=f"balance_{args.algorithm}",
)
