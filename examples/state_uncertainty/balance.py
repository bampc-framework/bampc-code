"""Balance with a noisy block-position estimate -- state-uncertainty regime.

Tilt a plate to slide a block to a plate-relative goal, but the planner reads
the block through a **sensor** (one noisy reading per plan step) and plans over
a belief cloud around it; ``--estimator`` picks the whole cloud (ensemble) or
its centre (point). The belief ghosts show where each domain thinks the block
is right now.

Run::

    uv run python examples/state_uncertainty/balance.py mppi
    uv run python examples/state_uncertainty/balance.py cem --estimator point
    uv run python examples/state_uncertainty/balance.py ps --noise pose \
        --noise-scale 3
"""

from __future__ import annotations

import argparse

from bampc.config import planner as planner_profiles
from bampc.config import reward as reward_profiles
from bampc.config.numerics import load as load_numerics
from bampc.planner import PlannerConfig, build_planner
from bampc.rollout import WarpRolloutEngine
from bampc.sim import run_interactive
from bampc.task.balance import Balance
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
    description="Balance with state noise.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "algorithm", nargs="?", default="ps", choices=["mppi", "ps", "cem"],
    help="Sampling algorithm.",
)
add_scenario_args(parser, "balance")
add_uncertainty_args(parser)
add_run_args(parser, settle_default=5)
add_latency_args(parser)
args = parser.parse_args()

# --------------------------------------------------------------------- #
# Config: numerics / task / planner  (from the shared global configs)
# --------------------------------------------------------------------- #

NUMERICS = load_numerics("balance")

BANK, SCENARIO, SHAPE, SCALE, GOAL_DRIFT = resolve_scenario(args, "balance")

NUM_RANDOMIZATIONS = args.domains

PLANNER_KW = planner_profiles.load("balance")
REWARD_KW = reward_profiles.load("balance")
PLANNER_KW["settle_steps"] = args.settle_steps
PLANNER = PlannerConfig(algo=args.algorithm, **PLANNER_KW)

PLAN_FREQ = 15
RECORDING = False

# --------------------------------------------------------------------- #
# Build: task -> uncertainty -> engine -> planner
# --------------------------------------------------------------------- #

task = Balance(shape=SHAPE, scale=SCALE,
               goal_drift=GOAL_DRIFT, model_config=NUMERICS, **REWARD_KW)

# joints_ok=False: no arm here, so the sensor drops the encoder terms.
sensor, sensor_noise = build_sensor(task, args, joints_ok=False)
point_filter, kalman = build_filter(task, args, sensor_noise, 1.0 / PLAN_FREQ)
uncertainty, belief_how = build_belief(task, args, kalman, joints_ok=False)
engine = WarpRolloutEngine(
    task,
    num_samples=PLANNER.num_samples,
    num_randomizations=NUM_RANDOMIZATIONS,
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

print(
    f"noise={args.noise} x{args.noise_scale}  R={NUM_RANDOMIZATIONS}  "
    f"settle={args.settle_steps}\nbelief: {belief_how}"
)

run_interactive(
    planner,
    mj_model,
    mj_data,
    frequency=PLAN_FREQ,
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
    record_name=f"balance_uncertain_{args.estimator}",
)
