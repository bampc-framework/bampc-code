"""Interactive Balance-FR3 -- domain-randomization regime.

The FR3 tilts a plate to slide a free block to a goal that rides the plate. The
planner rolls out ``--domains`` randomized worlds aggregated by ``--risk``.
``--rolling-sensor`` hands it the real robot's position-only ball estimate
instead of the true state (see ``examples.rolling_sensor``).

Run::

    uv run python examples/domain_randomization/balance_fr3.py mppi
    uv run python examples/domain_randomization/balance_fr3.py mppi \
        --rolling-sensor --filter none
    uv run python examples/domain_randomization/balance_fr3.py cem --risk cvar
    uv run python examples/domain_randomization/balance_fr3.py mppi --domains 12
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
from examples.flags import add_dr_args, add_run_args
from examples.latency import add_latency_args
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
    description="Interactive Balance FR3 (DR).",
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
add_dr_args(parser, domains=10)
add_scenario_args(parser, "balance_fr3")
add_run_args(parser)
add_latency_args(parser)
add_rolling_sensor_args(parser)
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
if args.risk is not None:
    PLANNER_KW["risk"] = args.risk
PLANNER_KW["settle_steps"] = args.settle_steps

TASK = dict(
    sampling_space=args.sampling,
    shape=SHAPE,
    scale=SCALE,
    goal_drift=GOAL_DRIFT,
    **REWARD_KW,
)

PLANNER = PlannerConfig(algo=args.algorithm, **PLANNER_KW)

NUM_RANDOMIZATIONS = args.domains
PLAN_FREQ = 20
RECORDING = True

# --------------------------------------------------------------------- #
# Domain randomization. Rolling friction bites on the sphere (condim=6);
# "body" targets the block's child geoms by id, so it works for any --shape.
# --------------------------------------------------------------------- #

RANDOMIZATION_SPEC = {
    "body": {"block": {"rolling_friction": (0.0001, 0.001)}},
    # "body": {"block": {"friction": (0.2, 0.6)}},
    # "body": {"block": {"mass": (0.1, 0.4)}},
}

# --------------------------------------------------------------------- #
# Build: task -> randomizer -> engine -> planner
# --------------------------------------------------------------------- #

task = BalanceFr3(**TASK, model_config=NUMERICS)
randomizer = DomainRandomizer(
    task, num_randomizations=NUM_RANDOMIZATIONS, spec=RANDOMIZATION_SPEC, seed=0
)
engine = WarpRolloutEngine(
    task,
    num_samples=PLANNER.num_samples,
    num_randomizations=NUM_RANDOMIZATIONS,
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

print(f"DR: risk={PLANNER.risk}  R={NUM_RANDOMIZATIONS}")

run_interactive(
    planner,
    mj_model,
    mj_data,
    frequency=PLAN_FREQ,
    show_endpoints=True,
    trace_idxs=[0],
    max_traces=1,
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
