r"""Peg-FR3 under grasp uncertainty -- the special state-uncertainty case.

Peg has no free observable object, so the usual sensor->filter->belief
machinery (``common.py``) does not apply. Its uncertain quantity is the
**mount** -- the peg's pose in the wrist frame, a static model parameter --
so point-vs-ensemble is framed over the mount instead:

* ``--estimator point`` -- plan on the nominal mount (R=1).
* ``--estimator ensemble`` -- roll out ``--domains`` mounts from an isotropic
  SE(3) Gaussian (std ``--noise`` m / ``--noise-rot`` rad), aggregated by
  ``--risk``.

``--true-offset`` sets the std of the REAL mount error, sampled once and
applied to the viewer's peg -- what the ensemble must be robust to and what
the point arm gets wrong. A scenario's own ``mount_offset`` overrides it.

Run::

    uv run python examples/state_uncertainty/peg_fr3.py cem \
        --estimator point --true-offset 0.003
    uv run python examples/state_uncertainty/peg_fr3.py cem \
        --estimator ensemble --noise 0.004 --true-offset 0.003
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
from examples.flags import add_dr_args, add_run_args
from examples.latency import add_latency_args
from examples.scenario import add_scenario_args, initial_state, resolve_scenario
from examples.status import CostStatus

PEG_LEN = 0.12

# --------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------- #

parser = argparse.ArgumentParser(
    description="Peg FR3 grasp uncertainty.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "algorithm", nargs="?", default="cem", choices=["cem", "ps"],
    help="No mppi: its softmax mean blurs across the narrow feasible funnel.",
)
parser.add_argument(
    "--sampling", default="task", choices=["task", "joint"],
    help="Control space the planner samples in.",
)
parser.add_argument(
    "--estimator", default="ensemble", choices=["point", "ensemble"],
    help="point = one nominal mount (R=1); ensemble = a mount cloud (R>1).",
)
parser.add_argument(
    "--noise", type=float, default=0.004,
    help="Std (m) of the SE(3) Gaussian the ensemble hedges over.",
)
parser.add_argument("--noise-rot", type=float, default=None)
parser.add_argument(
    "--true-offset", type=float, default=None,
    help="Std (m) of the REAL mount error's SE(3) Gaussian, sampled once. "
    "An explicit value here always wins; default is the scenario's frozen "
    "mount_offset if it has one, else 0.003.",
)
add_dr_args(
    parser, domains=12, domains_help="Mount-cloud size (ensemble)."
)
parser.add_argument("--clearance", type=float, default=0.002)
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

IS_ENSEMBLE = args.estimator == "ensemble"
NUM_RANDOMIZATIONS = args.domains if IS_ENSEMBLE else 1

PLANNER_KW = planner_profiles.load("peg_fr3", sampling=args.sampling)
REWARD_KW = reward_profiles.load("peg_fr3", sampling=args.sampling)
if args.risk is not None:
    PLANNER_KW["risk"] = args.risk
if args.num_samples is not None:
    PLANNER_KW["num_samples"] = args.num_samples
PLANNER_KW["settle_steps"] = args.settle_steps
PLANNER = PlannerConfig(algo=args.algorithm, **PLANNER_KW)

PLAN_FREQ = 20
RECORDING = True

# --------------------------------------------------------------------- #
# The belief: a cloud of mount poses (ensemble) or the nominal mount (point).
# peg_body is a kinematic child of the wrist, so MJWarp reads these per world.
# --------------------------------------------------------------------- #

_ROT = args.noise_rot if args.noise_rot is not None else args.noise / PEG_LEN
BELIEF_SPEC = (
    {"body": {"peg_body": {
        "pos_x": {"std": args.noise},
        "pos_y": {"std": args.noise},
        "pos_z": {"std": args.noise},
        "rot_x": {"std": _ROT},
        "rot_y": {"std": _ROT},
        "rot_z": {"std": _ROT},
    }}}
    if IS_ENSEMBLE
    else {}
)

# --------------------------------------------------------------------- #
# Build: task -> belief (mount DR) -> engine -> planner
# --------------------------------------------------------------------- #

_TASK_KW = dict(
    sampling_space=args.sampling,
    clearance=args.clearance,
    goal_drift=GOAL_DRIFT,
    model_config=NUMERICS,
    **REWARD_KW,
)
task = PegFr3(**_TASK_KW)  # pristine -- the planner plans against this

# The REAL grasp error, sampled once and baked into a SEPARATE task instance
# (truth_task) so the planner's own model (task) stays nominal for the point
# arm -- it must not secretly know the true offset. Precedence: an explicit
# --true-offset always wins; otherwise a scenario-provided mount_offset
# (fixed); otherwise 0.003.
truth_task = PegFr3(**_TASK_KW)
mount_offset = SCENARIO.start.get("mount_offset")
if args.true_offset is not None:
    _TRUE_ROT = args.true_offset / PEG_LEN
    TRUE_SPEC = (
        {"body": {"peg_body": {
            "pos_x": {"std": args.true_offset},
            "pos_y": {"std": args.true_offset},
            "pos_z": {"std": args.true_offset},
            "rot_x": {"std": _TRUE_ROT},
            "rot_y": {"std": _TRUE_ROT},
            "rot_z": {"std": _TRUE_ROT},
        }}}
        if args.true_offset != 0.0
        else None
    )
elif mount_offset is not None:
    TRUE_SPEC = {"body": {"peg_body": {
        "pos_x": mount_offset["pos"][0],
        "pos_y": mount_offset["pos"][1],
        "pos_z": mount_offset["pos"][2],
        "rot_x": mount_offset["rot"][0],
        "rot_y": mount_offset["rot"][1],
        "rot_z": mount_offset["rot"][2],
    }}}
else:
    _TRUE_ROT = 0.003 / PEG_LEN
    TRUE_SPEC = {"body": {"peg_body": {
        "pos_x": {"std": 0.003},
        "pos_y": {"std": 0.003},
        "pos_z": {"std": 0.003},
        "rot_x": {"std": _TRUE_ROT},
        "rot_y": {"std": _TRUE_ROT},
        "rot_z": {"std": _TRUE_ROT},
    }}}

if TRUE_SPEC is not None:
    DomainRandomizer(
        truth_task, 1, TRUE_SPEC, seed=SCENARIO.index
    ).apply_to_mj_model(truth_task.mj_model)

randomizer = (
    DomainRandomizer(task, NUM_RANDOMIZATIONS, BELIEF_SPEC)
    if BELIEF_SPEC else None
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
# Initial state -- from the frozen scenario bank, against the TRUE kinematics
# --------------------------------------------------------------------- #

mj_model = truth_task.mj_model
mj_data = initial_state(truth_task, BANK, SCENARIO)

# --------------------------------------------------------------------- #
# Viewer
# --------------------------------------------------------------------- #


cost_status = CostStatus(truth_task)


_true_offset_desc = (
    args.true_offset if args.true_offset is not None
    else "scenario" if mount_offset is not None
    else 0.003
)
print(f"estimator={args.estimator}  R={NUM_RANDOMIZATIONS}  "
      f"risk={PLANNER.risk}  noise={args.noise}  "
      f"true_offset={_true_offset_desc}")

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
    record_name=f"peg_fr3_uncertain_{args.estimator}",
)
