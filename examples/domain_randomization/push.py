"""Interactive Push -- domain-randomization regime.

The base task plus DR: the planner rolls out ``--domains`` worlds with a
randomized model (``RANDOMIZATION_SPEC``) and aggregates their costs with a
``--risk`` strategy. Tune the DR ranges and the risk here.

Run::

    uv run python examples/domain_randomization/push.py mppi
    uv run python examples/domain_randomization/push.py ps --risk cvar
    uv run python examples/domain_randomization/push.py cem --scenario 3
"""

from __future__ import annotations

import argparse

from bampc.config import planner as planner_profiles
from bampc.config.numerics import load as load_numerics
from bampc.dr import DomainRandomizer
from bampc.planner import PlannerConfig, build_planner
from bampc.rollout import WarpRolloutEngine
from bampc.sim import run_interactive
from bampc.task.push import Push
from examples.flags import add_dr_args, add_run_args
from examples.latency import add_latency_args
from examples.scenario import add_scenario_args, initial_state, resolve_scenario
from examples.status import CostStatus

# --------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------- #

parser = argparse.ArgumentParser(
    description="Interactive Push (DR).",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "algorithm", nargs="?", default="ps", choices=["mppi", "ps", "cem"],
    help="Sampling algorithm.",
)
add_dr_args(parser, domains=8)
add_scenario_args(parser, "push")
add_run_args(parser)
add_latency_args(parser)
args = parser.parse_args()

# --------------------------------------------------------------------- #
# Config: numerics / task / planner  (from the shared global configs)
# --------------------------------------------------------------------- #

NUMERICS = load_numerics("push")

BANK, SCENARIO, SHAPE, SCALE, GOAL_DRIFT = resolve_scenario(args, "push")

PLANNER_KW = planner_profiles.load("push", algo=args.algorithm)
REWARD_KW = {}  # Push has none of its own.
if args.risk is not None:
    PLANNER_KW["risk"] = args.risk
PLANNER_KW["settle_steps"] = args.settle_steps
PLANNER = PlannerConfig(algo=args.algorithm, **PLANNER_KW)

NUM_RANDOMIZATIONS = args.domains
PLAN_FREQ = 30
RECORDING = True

# --------------------------------------------------------------------- #
# Domain randomization -- see bampc/dr/randomizer.py. "body" targets
# a body's child geoms by id, so it works for any --shape.
# --------------------------------------------------------------------- #

RANDOMIZATION_SPEC = {
    "body": {"block": {"friction": (0.2, 1.2)}},
    # "body": {"block": {"mass": (0.1, 1.4)}},
    # "geom": {"__all__": {"friction": (0.2, 1.5)}},
}

# --------------------------------------------------------------------- #
# Build: task -> randomizer -> engine -> planner
# --------------------------------------------------------------------- #

task = Push(shape=SHAPE, scale=SCALE, goal_drift=GOAL_DRIFT,
            model_config=NUMERICS, **REWARD_KW)
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
    show_traces=True,
    show_domain_traces=True,
    show_endpoints=True,
    max_traces=30,
    status_callback=cost_status,
    plan_lag_steps=args.plan_lag_steps,
    compensate_latency=args.compensate_latency,
    duration=args.duration,
    record=RECORDING,
    record_format="gif",
    record_camera="main",
    record_name=f"push_{args.algorithm}",
)
