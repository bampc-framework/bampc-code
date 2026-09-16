"""Interactive Balance-FR3 -- plain regime (no DR, no state uncertainty).

The FR3 tilts a plate on its end effector to slide a free block to a goal that
rides the plate. For tuning rewards and single/multi-parameter planner mismatch
(``--mismatch``).

Run::

    uv run python examples/simple/balance_fr3.py mppi --sampling joint
    uv run python examples/simple/balance_fr3.py cem --scenario 4
    uv run python examples/simple/balance_fr3.py mppi --mismatch mass=2.0
    uv run python examples/simple/balance_fr3.py mppi --rolling-sensor
    uv run python examples/simple/balance_fr3.py mppi --sampling joint \
        --shape circle --drift-shape circle --drift-radius 0.08
    uv run python examples/simple/balance_fr3.py mppi --sampling joint \
        --shape circle --random-goal --random-margin 0.05 \
        --random-threshold 0.03
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
from bampc.task.balance_fr3 import BalanceFr3, RandomWaypoint
from examples.flags import add_run_args
from examples.latency import add_latency_args
from examples.mismatch import add_mismatch_arg, mismatch_spec
from examples.rolling_sensor import (
    add_rolling_sensor_args,
    build_rolling_observer,
)
from examples.scenario import add_scenario_args, initial_state, resolve_scenario
from examples.status import CostStatus

# --------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------- #

parser = argparse.ArgumentParser(
    description="Interactive Balance FR3 (plain).",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "algorithm", nargs="?", default="mppi", choices=["mppi", "ps", "cem"],
    help="Sampling algorithm.",
)
parser.add_argument(
    "--sampling", default="joint", choices=["task", "joint"],
    help="Control space the planner samples in.",
)
add_scenario_args(parser, "balance_fr3")
parser.add_argument(
    "--goal-xy", type=float, nargs=2, default=None, metavar=("X", "Y"),
    help="Override the reward profile's goal center. Default: the "
    "profile's own value, unless --drift-shape circle recenters it onto "
    "the plate's origin (see below).",
)
parser.add_argument(
    "--random-goal", action="store_true",
    help="Jump the goal to a new random on-plate point whenever the "
    "block gets within --random-threshold of it (overrides any drift).",
)
parser.add_argument(
    "--random-margin", type=float, default=0.02,
    help="Extra keep-out distance from the plate edge for sampled goals "
    "(m), on top of the shape's own footprint (added automatically).",
)
parser.add_argument(
    "--random-threshold", type=float, default=0.03,
    help="Block-to-goal distance that triggers a jump to a new goal (m).",
)
parser.add_argument(
    "--random-seed", type=int, default=0,
    help="RNG seed for random-goal sampling.",
)
parser.add_argument(
    "--arm-kv",
    type=float,
    default=None,
    help="Override the arm's velocity-actuator gain (BalanceFr3's own "
    "arm_kv, default: the shared fr3_arm.xml value, kv=50). Unlike "
    "--mismatch, this patches the one model both the planner and the "
    "viewer use (no separate truth), so it's for sanity-checking a "
    "candidate kv in sim before trying it against the real robot -- see "
    "BalanceFr3.arm_kv's docstring.",
)
add_mismatch_arg(parser)
add_rolling_sensor_args(parser)
add_run_args(parser)
add_latency_args(parser)
args = parser.parse_args()

# --------------------------------------------------------------------- #
# Config: numerics / task / planner  (from the shared global configs)
# --------------------------------------------------------------------- #

NUMERICS = load_numerics("balance_fr3")

BANK, SCENARIO, SHAPE, SCALE, GOAL_DRIFT = resolve_scenario(
    args, "balance_fr3"
)
if args.rolling_sensor and SHAPE != "sphere":
    parser.error("--rolling-sensor requires --shape sphere")

PLANNER_KW = planner_profiles.load("balance_fr3", sampling=args.sampling)
REWARD_KW = reward_profiles.load("balance_fr3", sampling=args.sampling)
if args.goal_xy is not None:
    REWARD_KW["goal_xy"] = tuple(args.goal_xy)
elif args.drift_shape == "circle":
    # A circular orbit is "around the centre of the plate" by definition --
    # the profile's own goal_xy is only off-center to give a Lissajous
    # wobble room, which doesn't apply here.
    REWARD_KW["goal_xy"] = (0.0, 0.0)

RANDOM_GOAL = None
if args.random_goal:
    GOAL_DRIFT = None
    RANDOM_GOAL = RandomWaypoint(
        margin=args.random_margin,
        jump_threshold=args.random_threshold,
        seed=args.random_seed,
    )

TASK = dict(
    sampling_space=args.sampling,
    shape=SHAPE,
    scale=SCALE,
    goal_drift=GOAL_DRIFT,
    random_goal=RANDOM_GOAL,
    **REWARD_KW,
)

PLANNER_KW["settle_steps"] = args.settle_steps
PLANNER = PlannerConfig(algo=args.algorithm, **PLANNER_KW)

PLAN_FREQ = 30
RECORDING = False

# --------------------------------------------------------------------- #
# Build: task -> (optional single-model mismatch) -> engine -> planner
# --------------------------------------------------------------------- #

task = BalanceFr3(
    **TASK,
    model_config=NUMERICS,
    arm_kv=args.arm_kv,
    ctrl_range_scale=PLANNER.ctrl_range_scale,
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


# --rolling-sensor: the planner sees the ball through the real robot's
# position-only chain (examples.rolling_sensor); otherwise the true state.
observer, kalman = None, None
if args.rolling_sensor:
    observer, kalman = build_rolling_observer(
        task, args, mj_data, 1.0 / PLAN_FREQ
    )
    print(f"rolling sensor: noise={args.noise}  filter={args.filter}")

cost_status = CostStatus(task, filt=kalman)

if args.mismatch:
    print(f"planner mismatch: {args.mismatch}  (viewer sims the truth)")

run_interactive(
    planner,
    mj_model,
    mj_data,
    frequency=PLAN_FREQ,
    show_endpoints=True,
    show_belief=False,
    trace_idxs=[0],
    max_traces=0,
    status_callback=cost_status,
    observer=observer,
    plan_lag_steps=args.plan_lag_steps,
    compensate_latency=args.compensate_latency,
    duration=args.duration,
    record=RECORDING,
    record_format="gif",
    record_camera="main",
    record_name=f"balance_fr3_{args.algorithm}",
)
