"""Interactive Curling-FR3 -- plain regime (no DR, no state uncertainty).

Shoot a puck out of the launch box and land it in the house. The EE is held
inside the box by a hard velocity barrier in the IK, and the house sits well
outside it, so the release is structural rather than something the planner
has to be taught.

Unlike push_fr3 there are no ``--sampling`` / ``--manipulation`` axes: task
space and a free puck are the only options this task has.

Run::

    uv run python examples/simple/curling.py mppi
    uv run python examples/simple/curling.py cem --scenario 3
    uv run python examples/simple/curling.py ps --mismatch friction=2.0
"""

from __future__ import annotations

import argparse
import dataclasses
import math

from bampc.config import planner as planner_profiles
from bampc.config import reward as reward_profiles
from bampc.config.numerics import load as load_numerics
from bampc.dr import DomainRandomizer
from bampc.planner import PlannerConfig, build_planner
from bampc.rollout import WarpRolloutEngine
from bampc.sim import run_interactive
from bampc.task.curling_fr3 import (
    DEFAULT_LAUNCH_BOX,
    SHAPES,
    CurlingFr3,
)
from examples.flags import add_run_args
from examples.latency import add_latency_args
from examples.mismatch import add_mismatch_arg, mismatch_spec
from examples.scenario import add_scenario_args, initial_state, resolve_scenario
from examples.status import CostStatus

# --------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------- #

parser = argparse.ArgumentParser(
    description="Interactive Curling FR3 (plain).",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "algorithm", nargs="?", default="mppi", choices=["mppi", "ps", "cem"],
    help="Sampling algorithm.",
)
add_scenario_args(
    parser, "curling_circle", drift=False, shapes=SHAPES
)
add_mismatch_arg(parser)
parser.add_argument(
    "--surface-z-offset",
    type=float,
    default=0.0,
    help="Shift the modelled lane (m, world z) to preview a physically "
    "thicker or thinner real surface before running it live. Puck resting "
    "height, house height and ee_z_target all move with it.",
)
parser.add_argument(
    "--goal-x",
    type=float,
    default=None,
    help="Override the house's world-x distance (m). y stays the bank's "
    "value. Default: the scenario bank's goal_xy.",
)
parser.add_argument(
    "--launch-length",
    type=float,
    default=None,
    help="Override the launch box's length in x (m); the near edge stays "
    f"fixed at {DEFAULT_LAUNCH_BOX[0]}. Default: the task's launch box.",
)
parser.add_argument(
    "--rotate90",
    action="store_true",
    help="Experimental: rotate the whole lane 90 deg about z (about the "
    "robot base) so the arm pushes sideways -- shoulder rotation -- "
    "instead of forward -- arm extension. Pair with "
    "'--bank curling_circle_rot90' for a matching start-state bank; "
    "--goal-x/--launch-length are not rotate90-aware, so don't combine "
    "them with this flag. Omit for the exact current (unrotated) "
    "behavior.",
)
parser.add_argument(
    "--lane-shift",
    type=float,
    nargs=2,
    default=None,
    metavar=("DX", "DY"),
    help="Translate the whole rotated lane (launch box, goal, and this "
    "run's scenario start state) by (dx, dy) world metres, e.g. to try "
    "the sideways push at a different distance/offset from the base "
    "without hand-editing a new scenario bank. Only takes effect with "
    "--rotate90 (a no-op, with a warning, otherwise).",
)
add_run_args(
    parser, settle_default=None,
    settle_note=" Default: the planner profile's value, which is non-zero"
    " here -- the puck rests at exactly zero penetration and would"
    " otherwise free-fall on step 1.",
)
add_latency_args(parser)
args = parser.parse_args()

# --------------------------------------------------------------------- #
# Config: numerics / task / planner  (from the shared global configs)
# --------------------------------------------------------------------- #

NUMERICS = load_numerics("curling")

BANK, SCENARIO, SHAPE, SCALE, GOAL_DRIFT = resolve_scenario(
    args, "curling", drift=False
)

# --rotate90 needs a bank rotated to match: the default bank's frozen states
# are for the unrotated box and would silently spawn puck/EE/goal in the
# wrong place while still looking like a normal run.
if args.rotate90 and BANK.name != "curling_circle_rot90":
    parser.error(
        "--rotate90 needs a matching rotated scenario bank -- pass "
        f"'--bank curling_circle_rot90' (got --bank {args.bank!r}, whose "
        "start states were frozen for the unrotated lane)."
    )

PLANNER_KW = planner_profiles.load("curling")
REWARD_KW = reward_profiles.load("curling")
if args.settle_steps is not None:
    PLANNER_KW["settle_steps"] = args.settle_steps

goal_xy = BANK.goal_xy
if args.goal_x is not None:
    goal_xy = (args.goal_x, goal_xy[1])
launch_box = DEFAULT_LAUNCH_BOX
if args.launch_length is not None:
    x_lo, _, y_lo, y_hi = DEFAULT_LAUNCH_BOX
    launch_box = (x_lo, x_lo + args.launch_length, y_lo, y_hi)

# --rotate90: rotate launch box + lane visuals 90 deg CCW about the robot
# base. The box's y-range is symmetric about 0, so swapping x/y is an exact
# rotation. goal_xy already comes from the rotated bank.
lane_xy, lane_yaw, downrange_axis = None, 0.0, 0
if args.rotate90:
    x_lo, x_hi, y_lo, y_hi = launch_box
    launch_box = (y_lo, y_hi, x_lo, x_hi)
    lane_xy = (0.0, 0.90)  # scene.xml's default ground pos (0.90, 0.0), rotated
    lane_yaw = math.pi / 2
    downrange_axis = 1

    # --lane-shift: all four of launch_box/lane_xy/goal_xy and the frozen
    # block_xy/ee_xy must move together, or the puck/EE start outside the
    # shifted box and set_initial_state clamps. dataclasses.replace returns a
    # NEW Scenario: BANK came from scenarios.load()'s @cache, so mutating it
    # would corrupt it for every other caller in this process.
    if args.lane_shift is not None:
        dx, dy = args.lane_shift
        x_lo, x_hi, y_lo, y_hi = launch_box
        launch_box = (x_lo + dx, x_hi + dx, y_lo + dy, y_hi + dy)
        lane_xy = (lane_xy[0] + dx, lane_xy[1] + dy)
        goal_xy = (goal_xy[0] + dx, goal_xy[1] + dy)
        bx, by = SCENARIO.start["block_xy"]
        ex, ey = SCENARIO.start["ee_xy"]
        SCENARIO = dataclasses.replace(
            SCENARIO,
            start={
                **SCENARIO.start,
                "block_xy": [bx + dx, by + dy],
                "ee_xy": [ex + dx, ey + dy],
            },
        )
elif args.lane_shift is not None:
    print("WARNING: --lane-shift has no effect without --rotate90; ignored.")

TASK = dict(
    shape=SHAPE,
    scale=SCALE,
    goal_xy=goal_xy,
    launch_box=launch_box,
    lane_xy=lane_xy,
    lane_yaw=lane_yaw,
    downrange_axis=downrange_axis,
    surface_z_offset=args.surface_z_offset,
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

task = CurlingFr3(**TASK, model_config=NUMERICS)

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
    record_name=f"curling_{args.algorithm}",
)
