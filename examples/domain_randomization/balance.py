"""Interactive Balance -- domain-randomization regime.

Tilt a plate to slide a block to a plate-relative goal, with the planner rolling
out ``--domains`` randomized worlds aggregated by ``--risk``. Tune the DR ranges
and the risk here.

Run::

    uv run python examples/domain_randomization/balance.py mppi
    uv run python examples/domain_randomization/balance.py cem --risk cvar
    uv run python examples/domain_randomization/balance.py ps --scenario 2
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
from examples.flags import add_dr_args, add_run_args
from examples.latency import add_latency_args
from examples.scenario import add_scenario_args, initial_state, resolve_scenario
from examples.status import CostStatus

# --------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------- #

parser = argparse.ArgumentParser(
    description="Interactive Balance (DR).",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "algorithm", nargs="?", default="ps", choices=["mppi", "ps", "cem"],
    help="Sampling algorithm.",
)
add_dr_args(parser, domains=6)
add_scenario_args(parser, "balance")
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
if args.risk is not None:
    PLANNER_KW["risk"] = args.risk
PLANNER_KW["settle_steps"] = args.settle_steps
PLANNER = PlannerConfig(algo=args.algorithm, **PLANNER_KW)

NUM_RANDOMIZATIONS = args.domains
PLAN_FREQ = 15
RECORDING = True

# --------------------------------------------------------------------- #
# Domain randomization. "body" targets a body's child geoms by id, so it
# works no matter which --shape is active.
# --------------------------------------------------------------------- #

RANDOMIZATION_SPEC = {
    "body": {"block": {"friction": (0.2, 0.6)}},
    # "body": {"block": {"mass": (0.1, 1.4)}},
}

# --------------------------------------------------------------------- #
# Build: task -> randomizer -> engine -> planner
# --------------------------------------------------------------------- #

task = Balance(shape=SHAPE, scale=SCALE,
               goal_drift=GOAL_DRIFT, model_config=NUMERICS, **REWARD_KW)
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


cost_status = CostStatus(task)


print(f"DR: risk={PLANNER.risk}  R={NUM_RANDOMIZATIONS}")

run_interactive(
    planner,
    mj_model,
    mj_data,
    frequency=PLAN_FREQ,
    show_endpoints=True,
    show_domain_traces=False,
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
