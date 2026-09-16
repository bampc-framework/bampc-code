r"""Interactive Peg-FR3 -- plain regime (no DR, no state hedging).

Insert a rigidly-mounted box peg into a square socket. For tuning rewards, the
fit (``--clearance``), and a single **planner** model mismatch on the grasp: the
peg's mount pose in the wrist frame.

Peg is a special case -- its uncertain quantity is the mount (a static model
parameter), not a free-body pose or the ``block`` params the generic
``--mismatch`` covers. ``--mount-mismatch`` sets the mount offset the PLANNER
believes (m, along x) while the viewer sims the true, nominal mount: a
confidently-wrong planner that jams the peg on a wall or misses the hole.

Run::

    uv run python examples/simple/peg_fr3.py cem
    uv run python examples/simple/peg_fr3.py cem --clearance 0.001 --scenario 3
    uv run python examples/simple/peg_fr3.py cem --mount-mismatch 0.004
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
from bampc.task.peg_fr3 import PegFr3
from examples.flags import add_run_args
from examples.latency import add_latency_args
from examples.scenario import add_scenario_args, initial_state, resolve_scenario
from examples.status import CostStatus

# --------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------- #

parser = argparse.ArgumentParser(
    description="Interactive Peg FR3 (plain).",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "algorithm", nargs="?", default="cem", choices=["cem", "ps"],
    help="No mppi: its softmax mean blurs across the narrow feasible funnel.",
)
parser.add_argument(
    "--sampling", default="joint", choices=["task", "joint"],
    help="Control space the planner samples in.",
)
parser.add_argument(
    "--clearance", type=float, default=0.002,
    help="Radial gap between a peg face and the facing hole wall (m).",
)
parser.add_argument(
    "--mount-mismatch", type=float, default=0.0,
    help="Mount offset the PLANNER believes (m, along x); the viewer stays "
    "nominal. A single-model grasp mismatch.",
)
add_scenario_args(parser, "peg_fr3", geometry=False, drift=False)
parser.add_argument(
    "--num-samples", type=int, default=None,
    help="Override the planner profile's sample count S.",
)
add_run_args(parser)
add_latency_args(parser)
args = parser.parse_args()

# --------------------------------------------------------------------- #
# Config: numerics / task / planner  (from the shared global configs)
# --------------------------------------------------------------------- #

NUMERICS = load_numerics("peg_fr3")

BANK, SCENARIO, _, _, GOAL_DRIFT = resolve_scenario(
    args, "peg_fr3", geometry=False, drift=False
)

PLANNER_KW = planner_profiles.load("peg_fr3", sampling=args.sampling)
REWARD_KW = reward_profiles.load("peg_fr3", sampling=args.sampling)
if args.num_samples is not None:
    PLANNER_KW["num_samples"] = args.num_samples
PLANNER_KW["settle_steps"] = args.settle_steps

task = PegFr3(
    sampling_space=args.sampling,
    clearance=args.clearance,
    goal_drift=GOAL_DRIFT,
    model_config=NUMERICS,
    **REWARD_KW,
)

PLANNER = PlannerConfig(algo=args.algorithm, **PLANNER_KW)

PLAN_FREQ = 20
RECORDING = False

# --------------------------------------------------------------------- #
# Build: task -> (optional planner mount mismatch) -> engine -> planner
# --------------------------------------------------------------------- #

# The mount offset is a real model field on the peg (a kinematic child of the
# wrist), so an R=1 DR spec makes only the planner's engine wrong; the viewer
# steps the nominal task.mj_model.
spec = (
    {"body": {"peg_body": {"pos_x": args.mount_mismatch}}}
    if args.mount_mismatch != 0.0
    else {}
)
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


if args.mount_mismatch != 0.0:
    print(f"planner mount mismatch: {args.mount_mismatch} m "
          "(viewer sims the nominal mount)")

run_interactive(
    planner,
    mj_model,
    mj_data,
    frequency=PLAN_FREQ,
    show_traces=False,
    show_endpoints=False,
    trace_idxs=[0],
    max_traces=1,
    status_callback=cost_status,
    plan_lag_steps=args.plan_lag_steps,
    compensate_latency=args.compensate_latency,
    duration=args.duration,
    record=RECORDING,
    record_format="gif",
    record_camera="main",
    record_name=f"peg_fr3_{args.algorithm}",
)
