"""Interactive Flip-FR3 -- plain regime (no DR, no state uncertainty).

An FR3 pusher tips a free box (default: a real YCB cracker box) onto a
target face. ``--sampling {task,joint}`` -- full 6-D EE twist via IK
(default: joint) or joint vels; unlike Push-FR3/Balance-FR3's ``task``,
Flip's is un-regulated (see ``bampc.task.flip_fr3``'s module
docstring). For tuning rewards and single/multi-parameter planner mismatch
(``--mismatch``).

Run::

    uv run python examples/simple/flip_fr3.py mppi
    uv run python examples/simple/flip_fr3.py ps --no-wall
    uv run python examples/simple/flip_fr3.py cem --mismatch friction=0.4
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
from bampc.task.flip_fr3 import FlipFr3
from examples.flags import add_run_args
from examples.latency import add_latency_args
from examples.mismatch import add_mismatch_arg, mismatch_spec
from examples.scenario import add_scenario_args, initial_state, resolve_scenario
from examples.status import CostStatus

# --------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------- #

parser = argparse.ArgumentParser(
    description="Interactive Flip-FR3 (plain).",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "algorithm", nargs="?", default="ps", choices=["mppi", "ps", "cem"],
    help="Sampling algorithm.",
)
parser.add_argument(
    "--sampling", default="joint", choices=["task", "joint"],
    help="Control space the planner samples in.",
)
add_scenario_args(parser, "flip_fr3", drift=False)
parser.add_argument(
    "--no-wall", dest="wall", action="store_false",
    help="Drop the static wall behind the box (on by default -- free-space "
    "was observed not to sustain a push at all). Placement is a "
    "provisional constant, see bampc.task.flip_fr3.",
)
parser.add_argument(
    "--arm-kv",
    type=float,
    default=None,
    help="Override the arm's velocity-actuator gain (FlipFr3's own arm_kv, "
    "default: the shared fr3_arm.xml value, kv=50). Unlike --mismatch, "
    "this patches the one model both the planner and the viewer use (no "
    "separate truth), so it's for sanity-checking a candidate kv in sim "
    "before trying it against the real robot -- see "
    "scripts/ros/run_planner_node_flip_fr3.py's --arm-kv docstring.",
)
add_mismatch_arg(parser)
add_run_args(parser)
add_latency_args(parser)
args = parser.parse_args()

# --------------------------------------------------------------------- #
# Config: numerics / task / planner  (from the shared global configs)
# --------------------------------------------------------------------- #

NUMERICS = load_numerics("flip_fr3")

BANK, SCENARIO, SHAPE, SCALE, GOAL_DRIFT = resolve_scenario(
    args, "flip_fr3", drift=False
)

PLANNER_KW = planner_profiles.load("flip_fr3", sampling=args.sampling)
REWARD_KW = reward_profiles.load("flip_fr3", sampling=args.sampling)
PLANNER_KW["settle_steps"] = args.settle_steps
PLANNER = PlannerConfig(algo=args.algorithm, **PLANNER_KW)

PLAN_FREQ = 20
RECORDING = True

# --------------------------------------------------------------------- #
# Build: task -> (optional single-model mismatch) -> engine -> planner
# --------------------------------------------------------------------- #

task = FlipFr3(
    shape=SHAPE, scale=SCALE, wall=args.wall, arm_kv=args.arm_kv,
    sampling_space=args.sampling,
    goal_drift=GOAL_DRIFT, model_config=NUMERICS,
    ctrl_range_scale=PLANNER.ctrl_range_scale, **REWARD_KW,
)

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
    show_sites=True,
    show_endpoints=True,
    max_traces=10,
    status_callback=cost_status,
    plan_lag_steps=args.plan_lag_steps,
    compensate_latency=args.compensate_latency,
    duration=args.duration,
    record=RECORDING,
    record_camera="main",
    record_name=f"flip_fr3_{args.algorithm}",
    record_format="gif",
)
