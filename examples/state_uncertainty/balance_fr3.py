"""Balance-FR3 with a noisy ball-pose estimate -- state-uncertainty regime.

The FR3 tilts a plate to balance a free ball, but the planner reads the ball
through a **sensor** (one noisy reading per plan step) and a **Kalman
filter**, then plans over a belief cloud spread by that filter's posterior.
``--estimator`` is the comparison switch: cloud vs single estimate.

The arm's joint encoders stay trusted, so pick a ``--noise`` preset with no
``joint_jitter`` term (``pose``, ``pose-biased``, ``pose-ou``).

``--mismatch`` (see ``examples.mismatch``) makes the planner roll out a
deliberately-wrong model while the viewer steps the nominal one, orthogonal
to the belief cloud. ``rolling_friction`` is the closest proxy to a velocity
mismatch.

``--rolling-sensor`` (sphere only) swaps the generic pose-and-twist filter
for the real robot's chain: a position-only reading, then a Kalman filter or
a raw finite difference for linear velocity, then the no-slip constraint
(``ros.adapters.balance_fr3._rolling_angular_velocity``) for angular
velocity -- the two must share one source or they contradict each other.
Pairs with ``--estimator ensemble_fixed --noise pose-rolling-hedge``.

Run::

    uv run python examples/state_uncertainty/balance_fr3.py mppi
    uv run python examples/state_uncertainty/balance_fr3.py mppi \
        --estimator point
    uv run python examples/state_uncertainty/balance_fr3.py mppi \
        --estimator ensemble_fixed --mismatch rolling_friction=0.3
    uv run python examples/state_uncertainty/balance_fr3.py mppi \
        --rolling-sensor --filter none --estimator ensemble_fixed \
        --noise pose-rolling-hedge --mismatch rolling_friction=0.3
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
from bampc.task.balance_fr3 import BalanceFr3
from examples.flags import add_run_args
from examples.latency import add_latency_args
from examples.mismatch import add_mismatch_arg, mismatch_spec
from examples.rolling_sensor import rolling_observer
from examples.scenario import add_scenario_args, initial_state, resolve_scenario
from examples.state_uncertainty.common import (
    add_uncertainty_args,
    build_belief,
    build_filter,
    build_sensor,
    make_observer,
    warm_up,
)
from examples.status import GREEN, RESET, CostStatus

# --------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------- #

parser = argparse.ArgumentParser(
    description="Balance FR3 with state noise.",
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
add_uncertainty_args(parser)
add_mismatch_arg(parser)
parser.add_argument(
    "--rolling-sensor",
    action="store_true",
    help="Sphere only: replace the generic Kalman-on-pose-and-twist filter "
    "with the real robot's exact chain (see module docstring). --filter "
    "kalman/none and --velocity-source still select how the point "
    "estimate's linear velocity is computed here (default: kalman, same "
    "as the generic path); --observe-velocity is ignored -- whatever "
    "twist the sensor draws is always overwritten.",
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
add_run_args(parser, settle_default=5)
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

TASK = dict(
    sampling_space=args.sampling,
    shape=SHAPE,
    scale=SCALE,
    goal_drift=GOAL_DRIFT,
    **REWARD_KW,
)

NUM_RANDOMIZATIONS = args.domains

PLANNER_KW["settle_steps"] = args.settle_steps
PLANNER = PlannerConfig(algo=args.algorithm, **PLANNER_KW)

PLAN_FREQ = 20
RECORDING = False

# --------------------------------------------------------------------- #
# Build: task -> uncertainty -> engine -> planner
# --------------------------------------------------------------------- #

task = BalanceFr3(
    **TASK,
    model_config=NUMERICS,
    arm_kv=args.arm_kv,
    ctrl_range_scale=PLANNER.ctrl_range_scale,
)

# The SENSOR draws one reading of the truth per plan step (R=1); the filter
# turns the history into an estimate; the BELIEF fans it into R domains.
sensor, sensor_noise = build_sensor(task, args)
if args.rolling_sensor:
    # point_filter is unused directly (the rolling observer below reads
    # `kalman` itself), but warm_up still calibrates+replays through it,
    # so build the same (point, kalman) == (k, k) pair build_filter's
    # "kalman" branch returns for the generic path -- or (None, None)
    # under --filter none, where the no-slip chain uses a raw finite
    # difference instead (see _build_rolling_observer).
    point_filter, kalman = (
        build_filter(task, args, sensor_noise, 1.0 / PLAN_FREQ)
        if args.filter == "kalman"
        else (None, None)
    )
else:
    point_filter, kalman = build_filter(
        task, args, sensor_noise, 1.0 / PLAN_FREQ
    )
uncertainty, belief_how = build_belief(task, args, kalman)
# A single planner-model mismatch (same wrong value in every belief domain,
# orthogonal to the belief cloud itself), mirroring simple/balance_fr3.py --
# the viewer keeps stepping the nominal task.mj_model, so this is a genuine
# planner-vs-reality mismatch, not a DR spread.
mismatch = mismatch_spec(task, args.mismatch)
randomizer = (
    DomainRandomizer(task, NUM_RANDOMIZATIONS, mismatch) if mismatch else None
)
engine = WarpRolloutEngine(
    task,
    num_samples=PLANNER.num_samples,
    num_randomizations=NUM_RANDOMIZATIONS,
    randomizer=randomizer,
    record_traces=True,
    record_initial_state=True,
)
planner = build_planner(PLANNER, task, engine, state_uncertainty=uncertainty)

# --------------------------------------------------------------------- #
# Initial state -- from the frozen scenario bank
# --------------------------------------------------------------------- #

mj_model = task.mj_model
mj_data = initial_state(task, BANK, SCENARIO)

# --------------------------------------------------------------------- #
# Viewer
# --------------------------------------------------------------------- #


cost_status = CostStatus(task, filt=kalman)

warm_up(
    task, sensor, point_filter, kalman, mj_data, args.warmup_steps,
    1.0 / PLAN_FREQ,
)

filter_desc = (
    f"rolling-sensor ({'kalman' if args.filter == 'kalman' else 'FD'} + "
    "no-slip)"
    if args.rolling_sensor
    else args.filter
)
print(
    f"{GREEN}"
    f"noise={args.noise} x{args.noise_scale}  filter={filter_desc}  "
    f"R={NUM_RANDOMIZATIONS}  settle={args.settle_steps}  "
    f"warmup={args.warmup_steps}\nbelief: {belief_how}"
    f"{RESET}"
)
if args.mismatch:
    print(f"planner mismatch: {args.mismatch}  (viewer sims the truth)")

observer = (
    rolling_observer(task, sensor, 1.0 / PLAN_FREQ, kalman=kalman)
    if args.rolling_sensor
    else make_observer(sensor, point_filter, 1.0 / PLAN_FREQ)
)

run_interactive(
    planner,
    mj_model,
    mj_data,
    frequency=PLAN_FREQ,
    show_endpoints=True,
    show_belief=True,
    max_traces=1,
    status_callback=cost_status,
    observer=observer,
    plan_lag_steps=args.plan_lag_steps,
    compensate_latency=args.compensate_latency,
    duration=args.duration,
    record=RECORDING,
    record_format="gif",
    record_camera="main",
    record_name=f"balance_fr3_uncertain_{args.estimator}",
)
