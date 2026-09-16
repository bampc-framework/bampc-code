"""Interactive Curling-FR3 -- domain-randomization regime.

Curling is the one open-loop-after-release task: contact breaks at the
launch-box edge and the puck slides to rest with no further control, so an
ice-friction error compounds over the whole slide instead of being absorbed
by the next replan. That makes it where risk-aware aggregation has most to
say. The planner rolls out ``--domains`` worlds aggregated by ``--risk``.

As in ``simple/curling.py`` there are no ``--sampling``/``--manipulation``
axes -- task space and a free puck are curling's only options.

Run::

    uv run python examples/domain_randomization/curling.py mppi
    uv run python examples/domain_randomization/curling.py ps --risk cvar
    uv run python examples/domain_randomization/curling.py cem --domains 16
"""

from __future__ import annotations

import argparse

import numpy as np

from bampc.config import planner as planner_profiles
from bampc.config import reward as reward_profiles
from bampc.config.numerics import load as load_numerics
from bampc.dr import DomainRandomizer
from bampc.planner import PlannerConfig, build_planner
from bampc.rollout import WarpRolloutEngine
from bampc.sim import run_interactive
from bampc.task.curling_fr3 import SHAPES, CurlingFr3
from examples.flags import add_dr_args, add_run_args
from examples.latency import add_latency_args
from examples.scenario import add_scenario_args, initial_state, resolve_scenario
from examples.status import CostStatus

# --------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------- #

parser = argparse.ArgumentParser(
    description="Interactive Curling FR3 (DR).",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "algorithm", nargs="?", default="ps", choices=["mppi", "ps", "cem"],
    help="Sampling algorithm.",
)
add_dr_args(parser, domains=8)
add_scenario_args(
    parser, "curling_circle", drift=False, shapes=SHAPES
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

PLANNER_KW = planner_profiles.load("curling")
REWARD_KW = reward_profiles.load("curling")
if args.risk is not None:
    PLANNER_KW["risk"] = args.risk
if args.settle_steps is not None:
    PLANNER_KW["settle_steps"] = args.settle_steps

TASK = dict(
    shape=SHAPE,
    scale=SCALE,
    goal_xy=BANK.goal_xy,
    trace_sites=["ee_site"],
    goal_drift=GOAL_DRIFT,
    **REWARD_KW,
)

PLANNER = PlannerConfig(algo=args.algorithm, **PLANNER_KW)

NUM_RANDOMIZATIONS = args.domains
PLAN_FREQ = 10
RECORDING = True

# --------------------------------------------------------------------- #
# Domain randomization: ice friction, drawn BELOW AND ABOVE the 0.08
# baseline, so the planner is exposed to both an over- and an
# under-estimate -- the whole point of a risk strategy is to be pulled by
# both tails at once, not just the safe one.
#
# The one-sided version of this (randomize the puck only) does not work:
# MuJoCo takes the elementwise MAX of a contacting pair, and the lane
# ("wood" class) sits at a fixed 0.08, so any puck draw below 0.08 is
# silently floored back up to 0.08 by the lane -- a wasted domain, not an
# underestimate. Fix: draw ONE per-domain value and write it to BOTH the
# puck (body "block") and the lane (geom "ground"), so puck == lane == the
# draw and the max is exactly that value in both directions. DR's explicit
# per-domain grid (a length-R array, see dr/README.md) makes the two draws
# identical rather than independently sampled.
#
# The "uniform mu" failure mode in configs/numerics/curling.yaml's header
# was measured on the OLD box lane; the lane is now a plane and re-running
# those pairs comes back clean down to mu=0.02, so this range is safe.
# --------------------------------------------------------------------- #

_FRICTION_RNG = np.random.default_rng(0)
_FRICTION_DRAWS = _FRICTION_RNG.uniform(
    0.06, 0.18, size=NUM_RANDOMIZATIONS
).tolist()

RANDOMIZATION_SPEC = {
    "body": {"block": {"friction": _FRICTION_DRAWS}},
    "geom": {"ground": {"friction": _FRICTION_DRAWS}},
}

# --------------------------------------------------------------------- #
# Build: task -> randomizer -> engine -> planner
# --------------------------------------------------------------------- #

task = CurlingFr3(**TASK, model_config=NUMERICS)
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
    record_name=f"curling_{args.algorithm}",
)
