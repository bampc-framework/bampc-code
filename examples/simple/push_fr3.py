"""Interactive Push-FR3 -- plain regime (no DR, no state uncertainty).

Two orthogonal axes (see ``bampc.task.push_fr3``):

* ``--sampling {task,joint}`` -- 2-D EE twist via IK (default) or joint vels.
* ``--manipulation {joint,free}`` -- 3-DOF constrained block (default) or free.

For tuning rewards and single/multi-parameter planner mismatch (``--mismatch``).

Run::

    uv run python examples/simple/push_fr3.py mppi
    uv run python examples/simple/push_fr3.py ps --manipulation free
    uv run python examples/simple/push_fr3.py cem --mismatch mass=2.0
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
from bampc.task.push_fr3 import PushFr3
from examples.flags import add_run_args
from examples.latency import add_latency_args
from examples.mismatch import add_mismatch_arg, mismatch_spec
from examples.scenario import add_scenario_args, initial_state, resolve_scenario
from examples.status import CostStatus

# --------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------- #

parser = argparse.ArgumentParser(
    description="Interactive Push FR3 (plain).",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "algorithm", nargs="?", default="ps", choices=["mppi", "ps", "cem"],
    help="Sampling algorithm.",
)
parser.add_argument(
    "--sampling", default="task", choices=["task", "joint"],
    help="Control space the planner samples in.",
)
parser.add_argument(
    "--manipulation", default="joint", choices=["joint", "free"],
    help="Block DOF: 3-DOF constrained, or a 6-DOF free body.",
)
add_scenario_args(parser, "push_fr3_t")
add_mismatch_arg(parser)
add_run_args(parser)
add_latency_args(parser)
args = parser.parse_args()

# --------------------------------------------------------------------- #
# Config: numerics / task / planner  (from the shared global configs)
# --------------------------------------------------------------------- #

NUMERICS = load_numerics(f"fr3_{args.manipulation}")

BANK, SCENARIO, SHAPE, SCALE, GOAL_DRIFT = resolve_scenario(args, "push_fr3")

PLANNER_KW = planner_profiles.load(
    "push_fr3", sampling=args.sampling, manipulation=args.manipulation
)
REWARD_KW = reward_profiles.load(
    "push_fr3", sampling=args.sampling, manipulation=args.manipulation
)
PLANNER_KW["settle_steps"] = args.settle_steps

TASK = dict(
    sampling_space=args.sampling,
    manipulation_type=args.manipulation,
    shape=SHAPE,
    scale=SCALE,
    trace_sites=["ee_site"],
    goal_drift=GOAL_DRIFT,
    **REWARD_KW,
)

PLANNER = PlannerConfig(algo=args.algorithm, **PLANNER_KW)

PLAN_FREQ = 10
RECORDING = False

# --------------------------------------------------------------------- #
# Build: task -> (optional single-model mismatch) -> engine -> planner
# --------------------------------------------------------------------- #

task = PushFr3(**TASK, model_config=NUMERICS)

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
    max_traces=0,
    status_callback=cost_status,
    duration=args.duration,
    plan_lag_steps=args.plan_lag_steps,
    compensate_latency=args.compensate_latency,
    record=RECORDING,
    record_format="gif",
    record_camera="main",
    record_name=f"push_fr3_{args.algorithm}",
)
