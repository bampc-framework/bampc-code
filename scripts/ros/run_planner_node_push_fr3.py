"""Real-robot planner node for Push-FR3: GPU planning, its own OS process.

Builds the task/planner from the same named profiles an example or a sweep
would (``configs/planner``, ``configs/reward``, ``configs/numerics``), then
runs :class:`~bampc.ros.planner_node.PlannerNode` with the Push-FR3
state reader (:mod:`bampc.ros.adapters.push_fr3`). Run
``run_control_node_push_fr3.py`` as a **separate process** alongside this one
-- see ``planner_node.py``'s module docstring for why they must not share a
process.

Run::

    uv run python scripts/ros/run_planner_node_push_fr3.py ps \
        --tf-frame objectPushT_MuJoCo
"""

from __future__ import annotations

import argparse
import dataclasses
import functools

import rclpy

from bampc.config import planner as planner_profiles
from bampc.config import reward as reward_profiles
from bampc.config.numerics import load as load_numerics
from bampc.dr import DomainRandomizer
from bampc.planner import PlannerConfig, build_planner
from bampc.rollout import WarpRolloutEngine
from bampc.ros.adapters import push_fr3 as push_fr3_adapter
from bampc.ros.planner_node import PlannerNode
from bampc.task.base import GoalDrift
from bampc.task.common.shapes import list_shapes
from bampc.task.push_fr3 import PushFr3
from bampc.uncertainty import (
    Ensemble,
    PosteriorGaussian,
    StateUncertainty,
    presets,
)

# Same tuned drift shape as configs/scenarios/push_fr3_t.yaml's goal_drift,
# reused as a real-robot default -- not loaded from the scenario bank itself
# (that bank is sim-scenario machinery, unrelated to a real run).
_GOAL_DRIFT = GoalDrift(
    radius_xy=(0.04, 0.10), freq_xy=(0.01, 0.02), yaw_amp=0.5, yaw_freq=0.05
)

# Fixed home pose (base frame), sent via MoveIt when 'h' is pressed at the
# idle prompt. Quat is [w, x, y, z]. Edit directly for a different setup.
_HOME_POS = (0.55, 0.2, 0.039)
_HOME_QUAT = (0.0, 1.0, 0.0, 0.0)

parser = argparse.ArgumentParser(
    description="Push-FR3 planner node (real robot).",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "algorithm", nargs="?", default="ps", choices=["mppi", "ps", "cem"]
)
parser.add_argument("--sampling", default="task", choices=["task", "joint"])
parser.add_argument(
    "--manipulation", default="free", choices=["joint", "free"]
)
parser.add_argument("--shape", default="t", choices=list_shapes())
parser.add_argument("--scale", type=float, default=1.0)
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
    "--drift-radius", type=float, nargs=2, default=None, metavar=("RX", "RY"),
    help="Goal-drift xy radius override (default: _GOAL_DRIFT's).",
)
parser.add_argument(
    "--drift-freq", type=float, nargs=2, default=None, metavar=("FX", "FY"),
    help="Goal-drift xy frequencies [Hz] override (default: _GOAL_DRIFT's) "
    "-- this is what controls drift speed, independent of --drift-radius.",
)
parser.add_argument("--risk", default=None, help="Override the profile's risk")
parser.add_argument("--plan-rate", type=float, default=10.0)
parser.add_argument("--control-rate", type=float, default=50.0)
parser.add_argument("--spline-topic", default="/bampc/trajectory")
parser.add_argument(
    "--tf-frame", required=True, help="TF frame of the tracked block"
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
    "--show-belief",
    action="store_true",
    help="Draw one translucent ghost per belief domain in the debug viewer "
    "(needs --viewer or --record) -- the real-robot counterpart of "
    "examples/state_uncertainty's show_belief, showing where each domain "
    "of --estimator's belief cloud currently thinks the block is.",
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
    choices=["point", "ensemble", "ensemble_fixed"],
    help="point = single filtered estimate; ensemble = a belief cloud "
    "spread from the tracking filter's own posterior (PosteriorGaussian); "
    "ensemble_fixed = a belief cloud spread by a fixed, named noise preset "
    "(--noise/--sigma-scale/--twist-sigma-scale), independent of the "
    "filter's own claimed covariance. R (--domains) is held fixed across "
    "all three for a fair comparison.",
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
    default=16,
    help="R, the number of belief/randomization domains (nworld = R * "
    "num_samples -- this multiplies real per-tick GPU cost over R=1, "
    "verify plan time stays inside --plan-rate's budget). Shared by "
    "--estimator's belief cloud and --friction-range/--mass-range's "
    "domain randomization when set.",
)
parser.add_argument(
    "--friction-range",
    type=float,
    nargs=2,
    default=None,
    metavar=("LOW", "HIGH"),
    help="Domain-randomize the block's sliding friction uniformly over "
    "[LOW, HIGH] across the R=--domains worlds (model/DR axis, "
    "independent of --estimator's state axis -- both can be on at "
    "once). Unset (default): no friction randomization, same block "
    "friction (models/push_fr3/free/scene.xml's tblock class, 0.5) in "
    "every world.",
)
parser.add_argument(
    "--mass-range",
    type=float,
    nargs=2,
    default=None,
    metavar=("LOW", "HIGH"),
    help="Domain-randomize the block's mass (kg) uniformly over [LOW, "
    "HIGH] across the same R=--domains worlds -- combines with "
    "--friction-range into one DR spec if both are set. Unset "
    "(default): no mass randomization, the shape's own geometry-derived "
    "mass (e.g. models/shapes/t.xml's geoms sum to ~0.21 kg) in every "
    "world.",
)
parser.add_argument(
    "--samples",
    type=int,
    default=None,
    help="Override S, num_samples from the push_fr3 planner profile "
    "(400 by default). Combine with --domains to hold nworld=R*S at any "
    "split, e.g. --domains 1 --samples 6400 vs --domains 16 (--samples "
    "unset, default 400) both give nworld=6400.",
)
parser.add_argument(
    "--settle-steps",
    type=int,
    default=None,
    help="Override settle_steps from the push_fr3 planner profile (0 by "
    "default) -- zero-control steps each rollout world takes before the "
    "planned horizon, letting a perturbed state's interpenetration "
    "resolve first.",
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
parser.add_argument("--episode-duration", type=float, default=20.0)
parser.add_argument("--data-dir", default=None)
parser.add_argument(
    "--teleop",
    action="store_true",
    help="Jog the EE in x/y from the keyboard (w/a/s/d, space/x to stop) "
    "while --collect-data idles waiting for Enter. Needs --sampling task "
    "(nu=2). Puts the terminal in cbreak mode for the duration of the run.",
)
parser.add_argument(
    "--teleop-speed",
    type=float,
    default=0.06,
    help="Commanded EE speed (m/s) per jog keypress, clamped to the task's "
    "own action bounds.",
)
parser.add_argument(
    "--teleop-pulse",
    type=float,
    default=0.3,
    help="Seconds of nonzero command per jog keypress before it decays to "
    "zero (rest of the published trajectory is zero-padded).",
)
args = parser.parse_args()

NUMERICS = load_numerics(f"fr3_{args.manipulation}")
PLANNER_KW = planner_profiles.load(
    "push_fr3", sampling=args.sampling, manipulation=args.manipulation
)
REWARD_KW = reward_profiles.load(
    "push_fr3", sampling=args.sampling, manipulation=args.manipulation
)
if args.risk is not None:
    PLANNER_KW["risk"] = args.risk
PLANNER = PlannerConfig(algo=args.algorithm, **PLANNER_KW)
if args.samples is not None:
    PLANNER = dataclasses.replace(PLANNER, num_samples=args.samples)
if args.settle_steps is not None:
    PLANNER = dataclasses.replace(PLANNER, settle_steps=args.settle_steps)

GOAL_DRIFT = _GOAL_DRIFT
if args.drift_radius is not None or args.drift_freq is not None:
    GOAL_DRIFT = dataclasses.replace(
        _GOAL_DRIFT,
        radius_xy=tuple(args.drift_radius)
        if args.drift_radius is not None else _GOAL_DRIFT.radius_xy,
        freq_xy=tuple(args.drift_freq)
        if args.drift_freq is not None else _GOAL_DRIFT.freq_xy,
    )

task = PushFr3(
    sampling_space=args.sampling,
    manipulation_type=args.manipulation,
    shape=args.shape,
    scale=args.scale,
    goal_xy=tuple(args.goal_xy) if args.goal_xy else None,
    goal_drift=GOAL_DRIFT if args.goal_drift else None,
    trace_sites=["ee_site"],
    model_config=NUMERICS,
    **REWARD_KW,
)
randomizer = None
BLOCK_DR: dict = {}
if args.friction_range is not None:
    BLOCK_DR["friction"] = tuple(args.friction_range)
if args.mass_range is not None:
    BLOCK_DR["mass"] = tuple(args.mass_range)
if BLOCK_DR:
    randomizer = DomainRandomizer(
        task,
        num_randomizations=args.domains,
        spec={"body": {"block": BLOCK_DR}},
        seed=0,
    )
engine = WarpRolloutEngine(
    task,
    num_samples=PLANNER.num_samples,
    num_randomizations=args.domains,
    randomizer=randomizer,
    record_initial_state=True,  # state_uncertainty is always set below,
    # and PushFr3 declares endpoint_body="block", so optimize() always
    # reads engine.current_body_pose -- see examples/state_uncertainty/
    # push_fr3.py for the same requirement.
)
planner = build_planner(PLANNER, task, engine)

state_reader_factory = functools.partial(
    push_fr3_adapter.make_state_reader,
    joint_topic=args.joint_topic,
    tf_frame=args.tf_frame,
    base_frame=args.base_frame,
)
home_mover_factory = functools.partial(
    push_fr3_adapter.make_home_mover,
    home_pos=_HOME_POS,
    home_quat=_HOME_QUAT,
)

rclpy.init()
node = PlannerNode(
    planner,
    state_reader_factory,
    plan_rate=args.plan_rate,
    control_rate=args.control_rate,
    spline_topic=args.spline_topic,
    show_viewer=args.viewer,
    show_belief=args.show_belief,
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
    teleop=args.teleop,
    teleop_speed=args.teleop_speed,
    teleop_pulse=args.teleop_pulse,
    home_mover_factory=home_mover_factory,
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
    if filt is None:
        node.get_logger().warn(
            "estimator=ensemble but the object's pose layout is planar "
            "(no PoseKalman) -- falling back to point"
        )
    else:
        noise = [PosteriorGaussian(filt, args.sigma_scale)]
elif args.estimator == "ensemble_fixed":
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
