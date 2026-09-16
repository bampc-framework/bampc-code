"""Real-robot planner node for Balance-FR3: GPU planning, its own OS process.

Builds the task/planner from the same named profiles an example or a sweep
would (``configs/planner``, ``configs/reward``, ``configs/numerics``), then
runs :class:`~bampc.ros.planner_node.PlannerNode` with the Balance-FR3
state reader (:mod:`bampc.ros.adapters.balance_fr3`). Run
``run_control_node_balance_fr3.py`` as a **separate process** alongside this
one -- see ``planner_node.py``'s module docstring for why they must not
share a process.

Unlike ``run_planner_node_push_fr3.py``, ``sampling_space`` is fixed to
``"joint"`` here, not a CLI choice: this pipeline's command writer
(:func:`bampc.ros.adapters.balance_fr3.make_command_writer`) publishes a
7-wide ``JointJog``, which only matches a 7-wide joint-velocity action.
``"task"`` sampling (``nu=2``) would need a Twist-based writer instead, like
Push-FR3's -- out of scope here.

``--tf-frame`` defaults to
:data:`bampc.ros.adapters.balance_fr3.SPHERE_TF_FRAME`
(``sphere_centre``); pass it explicitly (e.g.
:data:`~bampc.ros.adapters.balance_fr3.DEFAULT_TF_FRAME` for the old
fiducial marker) to track something else instead -- ``sphere_centre`` is
then ignored entirely.

``--domains`` defaults to 8 and, whenever left above 1 (the CLI default
included), also defaults ``--estimator`` to ``ensemble_fixed`` and
``--noise`` to ``pose-rolling-hedge-fixed`` -- otherwise a plain
``--domains N`` (N>1) would spend GPU budget on belief domains that carry
no spread at all (``--estimator``/``--noise`` otherwise default to
``point``/``pose``, i.e. no ensemble). Pass either flag explicitly to
override; ``--domains 1`` keeps the old ``point``/``pose`` defaults.

``--filter`` (``kalman``/``passthrough``) selects the ball's point-estimate
filter independently of ``--estimator``'s belief-spread choice -- it
defaults to ``passthrough`` under ``--estimator naive`` (else ``kalman``),
but can be set either way regardless of ``--estimator``, e.g. ``--filter
passthrough --estimator ensemble_fixed --noise pose-rolling-hedge-fixed``
runs a naive (unfiltered) point estimate hedged by the preset's fixed
belief spread. The tracked sphere's angular velocity itself is unaffected by
this choice either way -- it always comes from the no-slip rolling
constraint (see
:func:`bampc.ros.adapters.balance_fr3._rolling_angular_velocity`),
never from this filter. ``--estimator ensemble`` needs ``--filter kalman``
(it spreads the filter's own posterior covariance); combining it with
``--filter passthrough`` falls back to ``point`` with a warning.

``--interactive`` defaults on: the planner idles until Enter starts a
rollout, Enter again stops it early, and ``h`` + Enter sends the arm home
-- no CSV, no ``--episode-duration`` limit (that only applies under
``--collect-data``, which implies this gate regardless of this flag).
``--no-interactive`` publishes continuously from startup instead, as this
script did before this flag existed.

``--home-on-start`` defaults on: the arm is sent home once, automatically,
right after the node starts -- the same move 'h' + Enter triggers,
without waiting for the keypress. Requires ``--interactive`` or
``--collect-data`` (both raise otherwise) so the planner stays idle until
homing finishes.

``--goal-drift`` defaults on, using the small, slow ``_GOAL_DRIFT`` below so
the goal doesn't sit static -- ``--no-goal-drift`` freezes it for a specific
test, and also recenters ``goal_xy`` on the plate's own origin (unless
``--goal-xy`` is given explicitly), since the profile's off-center default
exists only to leave the drift radius headroom. Its amplitude was picked
with real headroom under
:class:`~bampc.task.balance_fr3.BalanceFr3`'s own plate-boundary check
(itself derived from the model's actual, possibly non-square, plate
half-extents -- see that check's comment for why a hardcoded number there
already went stale once).

``--drift-shape circle`` swaps ``_GOAL_DRIFT``'s bounded quasi-periodic
wobble for a true circle of ``--drift-radius`` (one frequency,
``--drift-freq``'s first value) -- a sine and a pi/2-phase-shifted sine on
the same rate is a sine/cosine pair, so this is the same
:class:`~bampc.task.base.GoalDrift` math, just a different parameter
choice. Also recenters ``goal_xy`` on the plate's origin unless
``--goal-xy`` is given, same reasoning as ``--no-goal-drift`` above.
``--drift-radius``/``--drift-freq`` also override ``_GOAL_DRIFT``'s own
values under the default ``lissajous`` shape.

``--drift-shape random`` replaces smooth drift with a goal that jumps to a
new random on-plate point once the block gets within ``--random-threshold``
of it (``--random-margin`` is *extra* keep-out clearance beyond the tracked
shape's own footprint -- BalanceFr3 adds the shape's radius automatically,
so a puck's own edge never lands at the physical plate boundary --
``--random-seed`` seeds the RNG) -- unlike ``GoalDrift``, this needs live
block-position feedback each tick, so it's a separate stateful
:class:`~bampc.task.balance_fr3.RandomWaypoint`, not a ``GoalDrift``
parameter choice. Also recenters ``goal_xy`` on the plate's origin unless
``--goal-xy`` is given, same reasoning as ``--drift-shape circle`` above.

Run::

    uv run python scripts/ros/run_planner_node_balance_fr3.py ps
    uv run python scripts/ros/run_planner_node_balance_fr3.py ps \
        --tf-frame marker_72212
    uv run python scripts/ros/run_planner_node_balance_fr3.py ps \
        --no-goal-drift
    uv run python scripts/ros/run_planner_node_balance_fr3.py ps \
        --shape circle --drift-shape circle --drift-radius 0.08
    uv run python scripts/ros/run_planner_node_balance_fr3.py ps \
        --shape circle --tf-frame marker_72211 --drift-shape random \
        --random-margin 0.05 --random-threshold 0.03
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import math

import rclpy

from bampc.config import planner as planner_profiles
from bampc.config import reward as reward_profiles
from bampc.config.numerics import load as load_numerics
from bampc.planner import PlannerConfig, build_planner
from bampc.rollout import WarpRolloutEngine
from bampc.ros.adapters import balance_fr3 as balance_fr3_adapter
from bampc.ros.planner_node import PlannerNode
from bampc.task.balance_fr3 import BalanceFr3, RandomWaypoint
from bampc.task.base import GoalDrift
from bampc.task.common.shapes import list_shapes
from bampc.uncertainty import (
    Ensemble,
    Passthrough,
    PosteriorGaussian,
    StateUncertainty,
    presets,
)
from bampc.uncertainty.noise import AngularVelocityScale

# Small, slow drift around the reward profile's goal_xy=(0.05, 0.04) --
# BalanceFr3 itself raises if |goal_xy| + radius leaves the plate (checked
# against the model's true, possibly non-square, half-extents), but that
# check is a hard boundary, not a "this is a sensible amplitude" guarantee.
# Keep real headroom under it by hand: at the current plate (0.195 x 0.25,
# margin 0.05 for the block -> safe (0.145, 0.20) per axis, see
# balance_fr3.py) this leaves roughly a third of the available room
# unused on each axis, so a shape/scale/plate change that shrinks the
# margin doesn't put a "default" run right at the edge. yaw_amp=0 -- the
# default sphere has w_orient=0, so a nonzero goal yaw would drift for no
# reason; set it explicitly if driving an oriented shape.
_GOAL_DRIFT = GoalDrift(
    radius_xy=(0.03, 0.05), freq_xy=(0.015, 0.02), yaw_amp=0.0
)

parser = argparse.ArgumentParser(
    description="Balance-FR3 planner node (real robot, joint-velocity jog).",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "algorithm", nargs="?", default="ps", choices=["mppi", "ps", "cem"]
)
parser.add_argument("--shape", default="sphere", choices=list_shapes())
parser.add_argument("--scale", type=float, default=1.0)
parser.add_argument(
    "--arm-kv",
    type=float,
    default=None,
    help="Override the arm's velocity-actuator gain (BalanceFr3's own "
    "arm_kv, default: the shared fr3_arm.xml value, kv=50) on this run's "
    "planning model only. At kv=50 the simulated joints under-track a "
    "commanded velocity by up to ~18%%, which accounted for most of an "
    "observed Balance-FR3 sim-vs-real gap (a kv sweep found it stable "
    "up to ~kv=8000, plateauing by ~500-1000).",
)
parser.add_argument(
    "--goal-xy", type=float, nargs=2, default=None, metavar=("X", "Y")
)
parser.add_argument(
    "--goal-drift",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Drift the goal in a bounded quasi-periodic path around goal_xy "
    "(see _GOAL_DRIFT above) instead of holding it static. --no-goal-drift "
    "freezes the goal for a specific test.",
)
parser.add_argument(
    "--drift-shape", default="lissajous",
    choices=["lissajous", "circle", "random"],
    help="'lissajous' (default): _GOAL_DRIFT's bounded quasi-periodic "
    "wobble. 'circle': a true circle of --drift-radius, one frequency "
    "(--drift-freq's first value), centered on the plate's origin unless "
    "--goal-xy is given. 'random': jump to a new random on-plate point "
    "whenever the block gets within --random-threshold of it (see "
    "--random-margin/--random-threshold/--random-seed below).",
)
parser.add_argument(
    "--drift-radius", type=float, default=None,
    help="Goal-drift xy radius override (default: _GOAL_DRIFT's). Unused "
    "by --drift-shape random.",
)
parser.add_argument(
    "--drift-freq", type=float, nargs=2, default=None, metavar=("FX", "FY"),
    help="Goal-drift xy frequencies [Hz] override (default: _GOAL_DRIFT's). "
    "--drift-shape circle uses only FX; unused by --drift-shape random.",
)
parser.add_argument(
    "--random-margin", type=float, default=0.02,
    help="--drift-shape random: extra keep-out distance from the plate "
    "edge for sampled goals (m), on top of the shape's own footprint "
    "(added automatically).",
)
parser.add_argument(
    "--random-threshold", type=float, default=0.03,
    help="--drift-shape random: block-to-goal distance that triggers a "
    "jump to a new goal (m).",
)
parser.add_argument(
    "--random-seed", type=int, default=0,
    help="--drift-shape random: RNG seed for goal sampling.",
)
parser.add_argument("--risk", default=None, help="Override the profile's risk")
parser.add_argument("--plan-rate", type=float, default=10.0)
parser.add_argument("--control-rate", type=float, default=50.0)
parser.add_argument("--spline-topic", default="/bampc/trajectory")
parser.add_argument(
    "--tf-frame",
    default=None,
    help="TF frame of the tracked block. Defaults to sphere_centre; pass "
    "e.g. --tf-frame marker_72211 to track a fiducial marker instead "
    "(sphere_centre is then ignored).",
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
    default=None,
    choices=["point", "ensemble", "ensemble_fixed", "naive"],
    help="point = single Kalman-filtered estimate; ensemble = a belief "
    "cloud spread from the tracking filter's own posterior "
    "(PosteriorGaussian); ensemble_fixed = a belief cloud spread by a "
    "fixed, named noise preset (--noise/--sigma-scale/"
    "--twist-sigma-scale), independent of the filter's own claimed "
    "covariance; naive = no filter at all -- the raw TF reading is used "
    "exactly as read, and --filter defaults to passthrough (see --filter "
    "below). R (--domains) is held fixed across all four for a fair "
    "comparison. Defaults to ensemble_fixed when --domains > 1 (else "
    "point) -- see --domains.",
)
parser.add_argument(
    "--filter",
    default=None,
    choices=["kalman", "passthrough"],
    help="Filter producing the ball's point-estimate pose/linear-velocity "
    "(bampc.uncertainty.filter.PoseKalman or Passthrough) -- "
    "independent of --estimator's belief-spread choice, so e.g. "
    "--filter passthrough --estimator ensemble_fixed --noise "
    "pose-rolling-hedge-fixed runs a naive (unfiltered) point estimate "
    "still hedged by the preset's fixed belief spread (the tracked "
    "sphere's "
    "angular velocity is unaffected either way -- it always comes from "
    "the no-slip rolling constraint, never this filter, see "
    "adapters.balance_fr3._rolling_angular_velocity). Defaults to "
    "passthrough when --estimator naive (else kalman).",
)
parser.add_argument(
    "--noise",
    default=None,
    choices=presets.names(),
    help="Preset (configs/noise/*.yaml) used by --estimator "
    "ensemble_fixed's spread. Any angular_velocity_scale term in "
    "it is a deterministic per-tick correction rather than a spread, so "
    "it is applied regardless of --estimator -- every other term is "
    "ignored outside ensemble_fixed. Defaults to pose-rolling-hedge-fixed "
    "when --domains > 1 (else pose) -- see --domains.",
)
parser.add_argument(
    "--domains",
    type=int,
    default=8,
    help="R, the number of belief domains (nworld = R * num_samples -- "
    "this multiplies real per-tick GPU cost over R=1, verify plan time "
    "stays inside --plan-rate's budget). R > 1 also switches the "
    "--estimator/--noise defaults above to ensemble_fixed/"
    "pose-rolling-hedge-fixed, so an unadorned --domains N (N>1) actually "
    "spreads the belief instead of silently planning "
    "with zero belief spread; pass --estimator/--noise explicitly to "
    "override.",
)
parser.add_argument(
    "--samples",
    type=int,
    default=None,
    help="Override S, num_samples from the balance_fr3 planner profile. "
    "Combine with --domains to hold nworld=R*S at any split.",
)
parser.add_argument(
    "--settle-steps",
    type=int,
    default=None,
    help="Override settle_steps from the balance_fr3 planner profile -- "
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
    "--episode-duration seconds, save a CSV, prompt again.",
)
parser.add_argument(
    "--interactive",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Idle until Enter starts a rollout; press Enter again to stop "
    "it (runs indefinitely otherwise -- no --episode-duration limit "
    "unless --collect-data is also set); 'h' + Enter sends the arm "
    "home. --no-interactive publishes continuously from startup, as "
    "before.",
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
parser.add_argument(
    "--data-schema",
    choices=["full", "position"],
    default="position",
    help="collect_data CSV columns: 'position' (default here) logs "
    "just t + block-center + goal-center, all balance_fr3 needs; "
    "'full' matches push_fr3's schema (pos_err/rot_err + EE/block "
    "pose).",
)
args = parser.parse_args()
if args.estimator is None:
    args.estimator = "ensemble_fixed" if args.domains > 1 else "point"
if args.noise is None:
    args.noise = (
        "pose-rolling-hedge-fixed" if args.domains > 1 else "pose"
    )
if args.filter is None:
    args.filter = "passthrough" if args.estimator == "naive" else "kalman"

NUMERICS = load_numerics("balance_fr3")
PLANNER_KW = planner_profiles.load("balance_fr3", sampling="joint")
REWARD_KW = reward_profiles.load("balance_fr3", sampling="joint")
if args.risk is not None:
    PLANNER_KW["risk"] = args.risk
PLANNER = PlannerConfig(algo=args.algorithm, **PLANNER_KW)
if args.samples is not None:
    PLANNER = dataclasses.replace(PLANNER, num_samples=args.samples)
if args.settle_steps is not None:
    PLANNER = dataclasses.replace(PLANNER, settle_steps=args.settle_steps)
if args.goal_xy is not None:
    # REWARD_KW already carries the profile's own goal_xy (unlike
    # push_fr3's reward profile, balance_fr3's includes it -- see that
    # YAML's header comment) -- override in place rather than also passing
    # goal_xy= explicitly below, which would collide with **REWARD_KW.
    REWARD_KW["goal_xy"] = tuple(args.goal_xy)
elif not args.goal_drift or args.drift_shape in ("circle", "random"):
    # The profile's goal_xy=(0.05, 0.04) is only off-center to leave
    # _GOAL_DRIFT headroom under the plate-boundary check above -- with
    # drift off, or a circle/random mode (both "around the centre of the
    # plate" by definition), there's no reason to start off-center, so
    # default to the plate's own origin (BalanceFr3's constructor default)
    # instead.
    REWARD_KW["goal_xy"] = (0.0, 0.0)

RANDOM_GOAL = None
if args.drift_shape == "random":
    GOAL_DRIFT = None
    RANDOM_GOAL = RandomWaypoint(
        margin=args.random_margin,
        jump_threshold=args.random_threshold,
        seed=args.random_seed,
    )
elif args.drift_shape == "circle":
    # A sine and a pi/2-phase-shifted sine on the same rate is a
    # sine/cosine pair -- a true circle, not a new GoalDrift branch.
    radius = args.drift_radius if args.drift_radius is not None else 0.08
    freq = args.drift_freq[0] if args.drift_freq is not None else 0.03
    GOAL_DRIFT = GoalDrift(
        radius_xy=(radius, radius), freq_xy=(freq, freq), yaw_amp=0.0,
        phase=(0.0, math.pi / 2, 0.0),
    )
elif args.drift_radius is not None or args.drift_freq is not None:
    GOAL_DRIFT = dataclasses.replace(
        _GOAL_DRIFT,
        radius_xy=(
            (args.drift_radius, args.drift_radius)
            if args.drift_radius is not None else _GOAL_DRIFT.radius_xy
        ),
        freq_xy=(
            tuple(args.drift_freq) if args.drift_freq is not None
            else _GOAL_DRIFT.freq_xy
        ),
    )
else:
    GOAL_DRIFT = _GOAL_DRIFT

task = BalanceFr3(
    sampling_space="joint",
    shape=args.shape,
    scale=args.scale,
    model_config=NUMERICS,
    goal_drift=GOAL_DRIFT if args.goal_drift else None,
    random_goal=RANDOM_GOAL if args.goal_drift else None,
    arm_kv=args.arm_kv,
    ctrl_range_scale=PLANNER.ctrl_range_scale,
    **REWARD_KW,
)
engine = WarpRolloutEngine(
    task,
    num_samples=PLANNER.num_samples,
    num_randomizations=args.domains,
    record_initial_state=True,  # state_uncertainty is always set below,
    # and BalanceFr3 declares endpoint_body="block", so optimize() always
    # reads engine.current_body_pose.
)
planner = build_planner(PLANNER, task, engine)

state_reader_factory = functools.partial(
    balance_fr3_adapter.make_state_reader,
    joint_topic=args.joint_topic,
    tf_frame=args.tf_frame,
    base_frame=args.base_frame,
    filt_kind=args.filter,
)
home_mover_factory = functools.partial(
    balance_fr3_adapter.make_home_mover, q_home=task.q_home
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
    data_schema=args.data_schema,
    home_mover_factory=home_mover_factory,
    interactive=args.interactive,
    home_on_start=args.home_on_start,
)

# state_uncertainty needs the tracking filter, which only exists once
# PlannerNode.__init__ has built the state reader -- set on the plain
# attribute afterward (optimize() reads it fresh every call, so this is
# safe). The planning thread has already started; a benign few-tick race
# means the very first ticks may run as a point estimate before this
# lands, which is harmless since point is the safer of the two anyway.
# AngularVelocityScale (see its docstring) is a deterministic
# per-tick correction, not a spread -- it means the same thing whether
# `noise` below ends up tiling one estimate R times (point/naive) or
# spreading a real cloud (ensemble/ensemble_fixed), so it is pulled out of
# --noise and applied unconditionally rather than gated on --estimator
# like the rest of this preset.
hedge = [
    n for n in presets.build(
        args.noise, args.sigma_scale, joints=False, bias=False,
        twist=True, twist_scale=args.twist_sigma_scale,
    )
    if isinstance(n, AngularVelocityScale)
]
noise = list(hedge)
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
        noise = [PosteriorGaussian(filt, args.sigma_scale), *hedge]
elif args.estimator == "ensemble_fixed":
    # Already includes any AngularVelocityScale term in --noise --
    # presets.build returns the whole preset, so `hedge` would just
    # duplicate it here.
    noise = presets.build(
        args.noise, args.sigma_scale, joints=False, bias=False,
        twist=True, twist_scale=args.twist_sigma_scale,
    )
node.planner.state_uncertainty = StateUncertainty(
    task, args.domains, noise, estimator=Ensemble(), seed=0
)

try:
    rclpy.spin(node)
finally:
    node.destroy_node()
    rclpy.shutdown()
