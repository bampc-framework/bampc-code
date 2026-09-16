"""Push with a noisy state estimate -- state-uncertainty regime.

The same task as ``simple/push.py``, but the planner no longer knows exactly
where the block is. A **sensor** takes one noisy reading per plan step and the
planner plans over a belief cloud around it; ``--estimator`` decides whether it
sees the whole cloud (ensemble) or just its centre (point).

Push's block is PLANAR and ``PoseKalman`` covers free-joint objects only, so
there is no filter here -- the cloud is spread at the sensor's stated width.
``push_fr3.py`` is the one that mirrors the sweeps end to end (filter,
posterior-width cloud, warm-up); this is the same idea on a simpler task.

Run::

    uv run python examples/state_uncertainty/push.py mppi
    uv run python examples/state_uncertainty/push.py mppi --estimator point
    uv run python examples/state_uncertainty/push.py ps --noise pose-biased \
        --noise-scale 4
"""

from __future__ import annotations

import argparse

from bampc.config import planner as planner_profiles
from bampc.config.numerics import load as load_numerics
from bampc.planner import PlannerConfig, build_planner
from bampc.rollout import WarpRolloutEngine
from bampc.sim import run_interactive
from bampc.task.push import Push
from examples.flags import add_run_args
from examples.latency import add_latency_args
from examples.scenario import add_scenario_args, initial_state, resolve_scenario
from examples.state_uncertainty.common import (
    add_uncertainty_args,
    build_belief,
    build_filter,
    build_sensor,
    make_observer,
    warm_up,
)
from examples.status import CostStatus

# --------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------- #

parser = argparse.ArgumentParser(
    description="Push with state noise.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "algorithm", nargs="?", default="ps", choices=["mppi", "ps", "cem"],
    help="Sampling algorithm.",
)
add_scenario_args(parser, "push")
add_uncertainty_args(parser)
add_run_args(parser, settle_default=5)
add_latency_args(parser)
args = parser.parse_args()

# --------------------------------------------------------------------- #
# Config: numerics / task / planner  (from the shared global configs)
# --------------------------------------------------------------------- #

NUMERICS = load_numerics("push")

BANK, SCENARIO, SHAPE, SCALE, GOAL_DRIFT = resolve_scenario(args, "push")

# Hold R FIXED across --estimator values: shrinking it for the point arm
# would change compute and information at once. --domains 1 is the naive
# arm -- one noisy estimate, believed.
NUM_RANDOMIZATIONS = args.domains

PLANNER_KW = planner_profiles.load("push", algo=args.algorithm)
REWARD_KW = {}  # Push has none of its own.
# Average over beliefs -- this regime's own choice; simple/ and
# domain_randomization/ use the profile's worstcase.
PLANNER_KW["risk"] = "average"
PLANNER_KW["settle_steps"] = args.settle_steps
PLANNER = PlannerConfig(algo=args.algorithm, **PLANNER_KW)

PLAN_FREQ = 30
RECORDING = False

# --------------------------------------------------------------------- #
# Build: task -> uncertainty -> engine -> planner
# --------------------------------------------------------------------- #

task = Push(shape=SHAPE, scale=SCALE, goal_drift=GOAL_DRIFT,
            model_config=NUMERICS, **REWARD_KW)

# No DomainRandomizer: this example varies the *state*, not the model.
# The SENSOR draws ONE reading of the truth per plan step (R=1); the BELIEF fans
# it into R domains. joints_ok=False: no arm, so drop the encoder terms.
sensor, sensor_noise = build_sensor(task, args, joints_ok=False)
point_filter, kalman = build_filter(task, args, sensor_noise, 1.0 / PLAN_FREQ)
uncertainty, belief_how = build_belief(task, args, kalman, joints_ok=False)
engine = WarpRolloutEngine(
    task,
    num_samples=PLANNER.num_samples,
    num_randomizations=NUM_RANDOMIZATIONS,
    record_traces=True,
    record_initial_state=True,  # colors the belief ghosts by contact mode
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

print(
    f"noise={args.noise} x{args.noise_scale}  "
    f"R={NUM_RANDOMIZATIONS}  settle={args.settle_steps}\n"
    f"belief: {belief_how}"
)

run_interactive(
    planner,
    mj_model,
    mj_data,
    frequency=PLAN_FREQ,
    show_traces=False,
    show_belief=True,
    max_traces=1,
    status_callback=cost_status,
    observer=make_observer(sensor, point_filter, 1.0 / PLAN_FREQ),
    plan_lag_steps=args.plan_lag_steps,
    compensate_latency=args.compensate_latency,
    duration=args.duration,
    record=RECORDING,
    record_format="gif",
    record_camera="main",
    record_name=f"push_uncertain_{args.estimator}",
)
