"""Interactive Push-FR3 -- domain-randomization regime.

Two orthogonal axes (see ``bampc.task.push_fr3``):

* ``--sampling {task,joint}`` -- 2-D EE twist via IK (default) or joint vels.
* ``--manipulation {joint,free}`` -- 3-DOF constrained block (default) or free.

The planner rolls out ``--domains`` randomized worlds aggregated by ``--risk``.

Run::

    uv run python examples/domain_randomization/push_fr3.py mppi
    uv run python examples/domain_randomization/push_fr3.py ps --risk cvar
    uv run python examples/domain_randomization/push_fr3.py cem --domains 16
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
from examples.flags import add_dr_args, add_run_args
from examples.latency import add_latency_args
from examples.scenario import add_scenario_args, initial_state, resolve_scenario
from examples.status import CostStatus

# --------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------- #

parser = argparse.ArgumentParser(
    description="Interactive Push FR3 (DR).",
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
add_dr_args(parser, domains=12)
add_scenario_args(parser, "push_fr3_t")
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
if args.risk is not None:
    PLANNER_KW["risk"] = args.risk
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

NUM_RANDOMIZATIONS = args.domains
PLAN_FREQ = 10
RECORDING = False

# --------------------------------------------------------------------- #
# Domain randomization. "body" targets a body's child geoms by id, so it
# works for any --shape (and any --manipulation block model).
# --------------------------------------------------------------------- #

RANDOMIZATION_SPEC = {
    "body": {"block": {"friction": (0.2, 1.2)}},
    # "body": {"block": {"mass": (0.15, 3.0)}},
    # "geom": {"ee": {"margin": (-0.02, 0.02)}},
}

# --------------------------------------------------------------------- #
# Build: task -> randomizer -> engine -> planner
# --------------------------------------------------------------------- #

task = PushFr3(**TASK, model_config=NUMERICS)
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
    trace_idxs=[0],
    max_traces=1,
    status_callback=cost_status,
    plan_lag_steps=args.plan_lag_steps,
    compensate_latency=args.compensate_latency,
    duration=args.duration,
    record=RECORDING,
    record_format="gif",
    record_camera="main",
    record_name=f"push_fr3_{args.algorithm}",
)
