"""Real-robot planner node for Flip-FR3: GPU planning, its own OS process.

Builds the task/planner from the same named profiles an example or a sweep
would (``configs/planner``, ``configs/reward``, ``configs/numerics``), then
runs :class:`~bampc.ros.planner_node.PlannerNode` with the Flip-FR3
state reader (:mod:`bampc.ros.adapters.flip_fr3`). Run
``run_control_node_flip_fr3.py`` as a **separate process** alongside this
one -- see ``planner_node.py``'s module docstring for why they must not
share a process.

``FlipFr3`` has no ``sampling_space`` axis at all (unlike Push-FR3/
Balance-FR3): the sampled action is already the 7 joint velocities, so this
pipeline's command writer
(:func:`bampc.ros.adapters.flip_fr3.make_command_writer`) publishes
a 7-wide ``JointJog``, the same as Balance-FR3's joint-space mode.

``--tf-frame`` is **required** -- unlike Balance-FR3's fixed
``sphere_centre`` default, Flip's tracked box has no single canonical
marker/frame name across setups.

The goal is a static target *orientation* baked into the model (see
``FlipFr3``'s docstring), not a drifting xy point -- there is no
``--goal-xy``/``--goal-drift`` here, unlike ``run_planner_node_balance_fr3.py``.

The moment the box is upright and aligned (``task.task_success``, both
``orient_tol`` and ``upright_tol`` satisfied), the robot is zeroed
immediately (:class:`~bampc.ros.planner_node.PlannerNode`'s
``success_fn`` hook) -- under the default ``--interactive`` this also ends
the rollout and returns to the idle prompt, so a finished flip never keeps
being replanned.

``--domains`` (R) and the real-time sample budget: by default each domain
gets the profile's ``num_samples``, so ``nworld = R * num_samples`` and
raising R raises the total. The profile states no budget on purpose -- what
fits a replan period depends on the deployment GPU and on ``settle_steps``,
so it is measured, not shared. Measure yours with
``scripts/probing/bisect_flip_fr3_budget.py`` and pass it as
``--sample-budget``: S is then derived from R by
``bampc.planner.engine_shape`` (``budget // R``), so raising
``--domains`` alone shrinks S instead of exceeding the budget -- any split
is allowed, including one whose S falls below the profile's recommended
``num_samples``. ``--samples`` overrides S explicitly and bypasses the
derivation entirely.

``--filter`` (``kalman``/``passthrough``) selects the box's point-estimate
filter independently of ``--estimator``'s belief-spread choice -- it
defaults to ``passthrough`` under ``--estimator naive`` (else ``kalman``),
but can be set either way regardless of ``--estimator``, e.g. ``--filter
passthrough --estimator ensemble_fixed`` runs a naive (unfiltered) point
estimate hedged by a fixed ensemble. ``--estimator ensemble`` needs
``--filter kalman`` (it spreads the filter's own posterior covariance);
combining it with ``--filter passthrough`` falls back to ``point`` with a
warning.

``--interactive`` defaults on: the planner idles until Enter starts a
rollout, Enter again stops it early (or task success stops it
automatically), and ``h`` + Enter sends the arm home.

``--home-on-start`` defaults on: the arm is sent home once, automatically,
right after the node starts.

Run::

    uv run python scripts/ros/run_planner_node_flip_fr3.py ps \
        --tf-frame box_marker
    uv run python scripts/ros/run_planner_node_flip_fr3.py ps \
        --tf-frame box_marker --domains 8
    uv run python scripts/ros/run_planner_node_flip_fr3.py ps \
        --tf-frame box_marker --domains 8 --samples 512
"""

from __future__ import annotations

import argparse
import dataclasses
import functools

import rclpy

from bampc.config import planner as planner_profiles
from bampc.config import reward as reward_profiles
from bampc.config.numerics import load as load_numerics
from bampc.planner import PlannerConfig, build_planner, engine_shape
from bampc.rollout import WarpRolloutEngine
from bampc.ros.adapters import flip_fr3 as flip_fr3_adapter
from bampc.ros.planner_node import PlannerNode
from bampc.task.common.shapes import list_shapes
from bampc.task.flip_fr3 import FlipFr3
from bampc.uncertainty import (
    Ensemble,
    Passthrough,
    PosteriorGaussian,
    StateUncertainty,
    presets,
)

parser = argparse.ArgumentParser(
    description="Flip-FR3 planner node (real robot, joint-velocity jog).",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "algorithm", nargs="?", default="ps", choices=["mppi", "ps", "cem"]
)
parser.add_argument("--shape", default="cracker_box", choices=list_shapes())
parser.add_argument("--scale", type=float, default=1.0)
parser.add_argument(
    "--no-wall", dest="wall", action="store_false",
    help="Drop the static wall behind the box (on by default -- see "
    "bampc.task.flip_fr3).",
)
parser.add_argument(
    "--arm-kv",
    type=float,
    default=None,
    help="Override the arm's velocity-actuator gain (FlipFr3's own arm_kv, "
    "default: the shared fr3_arm.xml value, kv=50) on this run's planning "
    "model only. At kv=50 the simulated joints under-track a commanded "
    "velocity by up to ~18%%, which accounted for most of an observed "
    "Balance-FR3 sim-vs-real gap (a kv sweep found it stable up to "
    "~kv=8000, plateauing by ~500-1000) -- that sweep "
    "was against Balance-FR3, not this task, so treat it as a starting "
    "point to re-sweep here, not a calibrated value.",
)
parser.add_argument("--risk", default=None, help="Override the profile's risk")
parser.add_argument("--plan-rate", type=float, default=10.0)
parser.add_argument("--control-rate", type=float, default=50.0)
parser.add_argument("--spline-topic", default="/bampc/trajectory")
parser.add_argument(
    "--tf-frame",
    required=True,
    help="TF frame of the tracked box. No default -- pass the frame name "
    "your tracker publishes for this setup.",
)
parser.add_argument("--base-frame", default="fr3_link0")
parser.add_argument(
    "--joint-topic",
    default="/franka_robot_state_broadcaster/measured_joint_states",
)
parser.add_argument(
    "--viewer", action="store_true", help="Show the CPU debug viewer"
)
parser.add_argument(
    "--compensate-latency",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Predict the state forward by the measured plan duration before "
    "optimizing, so the plan lands on the state the robot will actually "
    "be in rather than the one just read",
)
parser.add_argument(
    "--estimator",
    default="point",
    choices=["point", "ensemble", "ensemble_fixed", "naive"],
    help="point = single Kalman-filtered estimate; ensemble = a belief "
    "cloud spread from the tracking filter's own posterior "
    "(PosteriorGaussian); ensemble_fixed = a belief cloud spread by a "
    "fixed, named noise preset (--noise/--sigma-scale/"
    "--twist-sigma-scale), independent of the filter's own claimed "
    "covariance; naive = no filter at all -- the raw TF reading is used "
    "exactly as read, and --filter defaults to passthrough (see --filter "
    "below). R (--domains) is held fixed across all four for a fair "
    "comparison.",
)
parser.add_argument(
    "--filter",
    default=None,
    choices=["kalman", "passthrough"],
    help="Filter producing the box's point-estimate pose/twist -- "
    "independent of --estimator's belief-spread choice, so e.g. --filter "
    "passthrough --estimator ensemble_fixed --noise pose runs a naive "
    "(unfiltered) point estimate still hedged by a fixed ensemble. "
    "Defaults to passthrough when --estimator naive (else kalman). "
    "--estimator ensemble needs --filter kalman (it spreads the filter's "
    "own posterior covariance); combining it with --filter passthrough "
    "falls back to point with a warning.",
)
parser.add_argument(
    "--noise",
    default="pose",
    choices=presets.names(),
    help="Preset used only by --estimator ensemble_fixed's spread "
    "(configs/noise/*.yaml). Ignored otherwise.",
)
parser.add_argument(
    "--domains",
    type=int,
    default=8,
    help="R, the number of belief domains. S (samples/domain) is the "
    "profile's num_samples, so nworld = R * num_samples and nothing checks "
    "it against your replan period -- pass a measured --sample-budget to "
    "have S derived from R instead. See this script's module docstring.",
)
parser.add_argument(
    "--samples",
    type=int,
    default=None,
    help="Explicit override of S, samples/domain -- when given, used "
    "exactly as passed (bypasses the sample_budget derivation entirely), "
    "so nworld = --domains * --samples at any split you choose.",
)
parser.add_argument(
    "--sample-budget",
    type=int,
    default=None,
    help="The real-time-safe max total nworld for THIS machine, measured "
    "with scripts/probing/bisect_flip_fr3_budget.py at this --plan-rate "
    "and --settle-steps. Given, S = budget // --domains at any split; "
    "omitted, S is just num_samples per domain. "
    "Ignored when --samples is given.",
)
parser.add_argument(
    "--settle-steps",
    type=int,
    default=None,
    help="Override settle_steps from the flip_fr3 planner profile -- "
    "zero-control steps each rollout world takes before the planned "
    "horizon, letting a perturbed state's interpenetration resolve first.",
)
parser.add_argument("--sigma-scale", type=float, default=1.0)
parser.add_argument(
    "--twist-sigma-scale",
    type=float,
    default=1.0,
    help="Under --estimator ensemble_fixed only: multiplies the fixed "
    "preset's velocity (twist) magnitude independently of --sigma-scale's "
    "pose magnitude. Ignored otherwise.",
)
parser.add_argument(
    "--record", action="store_true", help="Record the CPU viewer to MP4"
)
parser.add_argument("--record-camera", default="main")
parser.add_argument("--record-name", default="real_world")
parser.add_argument("--record-dir", default=None)
parser.add_argument(
    "--record-size",
    type=int,
    nargs=2,
    default=(480, 640),
    metavar=("HEIGHT", "WIDTH"),
    help="Capped by the model's offscreen framebuffer (640x480 unless "
    "the XML sets <visual><global offwidth=.. offheight=../></visual>)",
)
parser.add_argument(
    "--collect-data",
    action="store_true",
    help="Press-Enter episodic data collection: roll out for "
    "--episode-duration seconds (or until task success), save a CSV, "
    "prompt again.",
)
parser.add_argument(
    "--interactive",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Idle until Enter starts a rollout; press Enter again to stop "
    "it, or it stops on its own once the box is upright and aligned "
    "(no --episode-duration limit unless --collect-data is also set); "
    "'h' + Enter sends the arm home. --no-interactive publishes "
    "continuously from startup, holding zero once the task succeeds.",
)
parser.add_argument(
    "--home-on-start",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Send the arm home once, automatically, right after startup -- "
    "the same move 'h' + Enter triggers, just without waiting for the "
    "keypress. Requires --interactive or --collect-data (on by default "
    "here) so the planning loop stays idle until homing finishes.",
)
parser.add_argument("--episode-duration", type=float, default=20.0)
parser.add_argument("--data-dir", default=None)
args = parser.parse_args()
if args.filter is None:
    args.filter = "passthrough" if args.estimator == "naive" else "kalman"

NUMERICS = load_numerics("flip_fr3")
PLANNER_KW = planner_profiles.load("flip_fr3")
REWARD_KW = reward_profiles.load("flip_fr3")
if args.risk is not None:
    PLANNER_KW["risk"] = args.risk
PLANNER = PlannerConfig(algo=args.algorithm, **PLANNER_KW)
if args.sample_budget is not None:
    PLANNER = dataclasses.replace(PLANNER, sample_budget=args.sample_budget)
if args.settle_steps is not None:
    PLANNER = dataclasses.replace(PLANNER, settle_steps=args.settle_steps)

if args.samples is not None:
    domains, samples = args.domains, args.samples
else:
    domains, samples = engine_shape(PLANNER, args.domains)
# PLANNER.num_samples must match the engine's actual num_samples -- the
# planner's own sample_knots/update_params index knots by it, so a stale
# profile value here (vs. the engine built with `samples` below) is a
# shape mismatch that only surfaces as an out-of-bounds index at runtime.
PLANNER = dataclasses.replace(PLANNER, num_samples=samples)

task = FlipFr3(
    shape=args.shape, scale=args.scale, wall=args.wall,
    model_config=NUMERICS, arm_kv=args.arm_kv,
    ctrl_range_scale=PLANNER.ctrl_range_scale, **REWARD_KW,
)
engine = WarpRolloutEngine(
    task,
    num_samples=samples,
    num_randomizations=domains,
    record_initial_state=True,  # state_uncertainty is always set below,
    # and FlipFr3 declares endpoint_body=["block"], so optimize() always
    # reads engine.current_body_pose.
)
planner = build_planner(PLANNER, task, engine)

state_reader_factory = functools.partial(
    flip_fr3_adapter.make_state_reader,
    joint_topic=args.joint_topic,
    tf_frame=args.tf_frame,
    base_frame=args.base_frame,
    filt_kind=args.filter,
)
home_mover_factory = functools.partial(
    flip_fr3_adapter.make_home_mover, q_home=task.q_home
)

rclpy.init()
node = PlannerNode(
    planner,
    state_reader_factory,
    plan_rate=args.plan_rate,
    control_rate=args.control_rate,
    spline_topic=args.spline_topic,
    show_viewer=args.viewer,
    compensate_latency=args.compensate_latency,
    record=args.record,
    record_camera=args.record_camera,
    record_name=args.record_name,
    record_dir=args.record_dir,
    record_size=tuple(args.record_size),
    collect_data=args.collect_data,
    episode_duration=args.episode_duration,
    data_dir=args.data_dir,
    data_label=f"{args.shape}_{args.estimator}",
    home_mover_factory=home_mover_factory,
    interactive=args.interactive,
    home_on_start=args.home_on_start,
    success_fn=task.task_success,
)

# state_uncertainty needs the tracking filter, which only exists once
# PlannerNode.__init__ has built the state reader -- set on the plain
# attribute afterward (optimize() reads it fresh every call, so this is
# safe). The planning thread has already started; a benign few-tick race
# means the very first ticks may run as a point estimate before this
# lands, which is harmless since point is the safer of the two anyway.
noise = []
if args.estimator == "ensemble":
    filt = getattr(node.state_reader, "filt", None)
    if filt is None or isinstance(filt, Passthrough):
        node.get_logger().warn(
            "estimator=ensemble needs a Kalman filter's own posterior "
            "(--filter kalman), but none was built -- falling back to "
            "point. For a passthrough/naive point estimate with an "
            "ensemble spread, use --estimator ensemble_fixed instead, "
            "which doesn't depend on the filter's covariance."
        )
    else:
        noise = [PosteriorGaussian(filt, args.sigma_scale)]
elif args.estimator == "ensemble_fixed":
    noise = presets.build(
        args.noise, args.sigma_scale, joints=False, bias=False,
        twist=True, twist_scale=args.twist_sigma_scale,
    )
node.planner.state_uncertainty = StateUncertainty(
    task, domains, noise, estimator=Ensemble(), seed=0
)

try:
    rclpy.spin(node)
finally:
    node.destroy_node()
    rclpy.shutdown()
