"""Push-FR3 with a noisy state estimate -- state-uncertainty regime.

The same task as ``simple/push_fr3.py``, but the planner reads the block
through a **sensor** (one noisy reading per plan step) and a **Kalman
filter**, then plans over a belief cloud spread by that filter's own
posterior -- the arrangement ``experiments/push_fr3/state_uncertainty``
measures, so what you drive here is what the sweep scores. ``--estimator``
is the comparison switch: plan over the cloud, or over its centre.

The belief ghosts show where each domain thinks the block is right now,
colored by contact mode (grey none / yellow pusher / blue table / green
both).

Run::

    uv run python examples/state_uncertainty/push_fr3.py ps --estimator point
    uv run python examples/state_uncertainty/push_fr3.py ps --estimator ensemble

    # no filter at all -- believe every reading (the sweeps' `naive`):
    uv run python examples/state_uncertainty/push_fr3.py ps --filter none \
        --estimator point

    # MPPI's softmax is inert at the default temperature (it is an argmin);
    # drive both to see whether a real softmax helps here:
    uv run python examples/state_uncertainty/push_fr3.py mppi --temperature 0.3
"""

from __future__ import annotations

import argparse

from bampc.config import planner as planner_profiles
from bampc.config import reward as reward_profiles
from bampc.config.numerics import load as load_numerics
from bampc.planner import PlannerConfig, build_planner
from bampc.rollout import WarpRolloutEngine
from bampc.sim import run_interactive
from bampc.task.push_fr3 import PushFr3
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
    description="Push FR3 with state noise.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "algorithm", nargs="?", default="mppi", choices=["mppi", "ps", "cem"],
    help="Sampling algorithm.",
)
parser.add_argument(
    "--sampling", default="task", choices=["task", "joint"],
    help="Control space the planner samples in.",
)
parser.add_argument(
    "--manipulation", default="free", choices=["joint", "free"],
    help="Block DOF: 3-DOF constrained, or a 6-DOF free body.",
)
# Geometry and drift come from the bank, so this drives the same block and
# moving goal the sweeps measure against.
add_scenario_args(parser, "push_fr3_t")
# MPPI only. The default makes the softmax inert here (a hard argmin): the
# cost spread across samples swamps it. The right value depends on the COST
# SCALE, so re-check with scripts/probing/softmax_ess_check.py after any
# cost-weight change.
parser.add_argument(
    "--temperature", type=float, default=0.01,
    help="MPPI softmax temperature. Ignored by ps/cem.",
)
add_uncertainty_args(parser)
add_run_args(parser, settle_default=5)
add_latency_args(parser)
args = parser.parse_args()

# --------------------------------------------------------------------- #
# Config: numerics / task / planner
# --------------------------------------------------------------------- #

NUMERICS = load_numerics(f"fr3_{args.manipulation}")

BANK, SCENARIO, SHAPE, SCALE, GOAL_DRIFT = resolve_scenario(
    args, "push_fr3"
)

PLANNER_KW = planner_profiles.load(
    "push_fr3", sampling=args.sampling, manipulation=args.manipulation
)
REWARD_KW = reward_profiles.load(
    "push_fr3", sampling=args.sampling, manipulation=args.manipulation
)
# Matches push_fr3/state_uncertainty's own override on top of the same
# profile -- this regime mirrors what that sweep measures.
PLANNER_KW["plan_horizon"] = 0.4

TASK = dict(
    sampling_space=args.sampling,
    manipulation_type=args.manipulation,
    shape=SHAPE,
    scale=SCALE,
    # From the bank always: it is the centre the drift swings about.
    goal_xy=BANK.goal_xy,
    trace_sites=["ee_site"],
    goal_drift=GOAL_DRIFT,
    **REWARD_KW,
)

# The belief cloud needs domains to live in. Hold R FIXED across --estimator
# values: shrinking it for the point arm would change compute and information
# at once. The sweeps go further and hold R*S fixed; here S is shared.
NUM_RANDOMIZATIONS = args.domains

PLANNER_KW["temperature"] = args.temperature
PLANNER_KW["settle_steps"] = args.settle_steps
PLANNER = PlannerConfig(algo=args.algorithm, **PLANNER_KW)

PLAN_FREQ = 10
RECORDING = False

# --------------------------------------------------------------------- #
# Build: task -> uncertainty -> engine -> planner
# --------------------------------------------------------------------- #

task = PushFr3(**TASK, model_config=NUMERICS)

# No DomainRandomizer: this example varies the *state*, not the model.
# Two stages, as in the sweeps: the SENSOR draws one reading per plan step
# (R=1) and the filter estimates from the history; the BELIEF fans that
# estimate into R domains. Drawing R readings off the truth instead would put
# their mean noise/sqrt(R) from it -- a sampling artefact, not uncertainty.
sensor, sensor_noise = build_sensor(task, args)
point_filter, kalman = build_filter(task, args, sensor_noise, 1.0 / PLAN_FREQ)
uncertainty, belief_how = build_belief(task, args, kalman)
engine = WarpRolloutEngine(
    task,
    num_samples=PLANNER.num_samples,
    num_randomizations=NUM_RANDOMIZATIONS,
    record_traces=True,
    record_initial_state=True,  # colors the belief ghosts by contact mode
)
planner = build_planner(
    PLANNER, task, engine, state_uncertainty=uncertainty
)

# --------------------------------------------------------------------- #
# Initial state
# --------------------------------------------------------------------- #

mj_model = task.mj_model
mj_data = initial_state(task, BANK, SCENARIO)

# --------------------------------------------------------------------- #
# Viewer
# --------------------------------------------------------------------- #

cost_status = CostStatus(task, filt=kalman)


# Calibrate the filter on the still start state so it earns its measurement
# covariance rather than being handed the sensor's true magnitudes. Must run
# AFTER the initial state is set.
warm_up(
    task, sensor, point_filter, kalman, mj_data, args.warmup_steps,
    1.0 / PLAN_FREQ,
)

print(
    f"noise={args.noise} x{args.noise_scale}  filter={args.filter}  "
    f"R={NUM_RANDOMIZATIONS}  settle={args.settle_steps}  "
    f"warmup={args.warmup_steps}  temp={args.temperature:g}\n"
    f"belief: {belief_how}\n"
    f"start: bank scenario {args.scenario} (phase {mj_data.time:g}s)"
)

run_interactive(
    planner,
    mj_model,
    mj_data,
    frequency=PLAN_FREQ,
    show_traces=False,
    show_domain_traces=False,
    show_endpoints=False,
    show_belief=True,
    trace_idxs=[0],
    max_traces=1,
    status_callback=cost_status,
    observer=make_observer(sensor, point_filter, 1.0 / PLAN_FREQ),
    plan_lag_steps=args.plan_lag_steps,
    compensate_latency=args.compensate_latency,
    duration=args.duration,
    record=RECORDING,
    record_format="gif",
    record_camera="main",
    record_name=f"push_fr3_uncertain_{args.estimator}",
)
