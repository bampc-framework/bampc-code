"""Debug node for Curling-FR3: render the live state pipeline, no planner.

Builds the exact same state-reader factory ``run_planner_node_curling.py``
would (:mod:`bampc.ros.adapters.push_fr3`, which Curling-FR3 reuses --
both are a 2-D EE twist over the same ``pusher`` end effector) and feeds it
into :class:`~bampc.ros.state_viewer_node.StateViewerNode`: a CPU
passive MuJoCo viewer, no GPU, no planner, no rollout engine, and **nothing
published to the robot**.

Run this first, on real TF/JointState. What to check before trusting the
planner node with the arm:

* The puck sits *on* the lane, not through or above it. Curling's lane
  collides as a plane whose top face is z = 0.006, and the puck's resting
  height is what ``CurlingFr3`` derives ``ee_z_target`` from -- a tracker
  z-offset here becomes a push above or below the puck's centre of mass,
  which is the failure the numerics profile's header is about.
* The puck starts inside the launch box (x in [0.45, 0.70],
  y in [-0.15, 0.15]); this prints the box so you can compare.
* The arm pose matches the real one, i.e. the joint mapping is right.
* The house is where it physically is -- pass ``--goal-xy`` to move it.

Run::

    uv run python scripts/ros/run_state_viewer_node_curling.py \
        --tf-frame puck
"""

from __future__ import annotations

import argparse
import functools

import rclpy

from bampc.config import reward as reward_profiles
from bampc.config import scenarios
from bampc.config.numerics import load as load_numerics
from bampc.ros.adapters import push_fr3 as fr3_adapter
from bampc.ros.state_viewer_node import StateViewerNode
from bampc.task.curling_fr3 import DEFAULT_LAUNCH_BOX, CurlingFr3

parser = argparse.ArgumentParser(
    description="Curling-FR3 state-pipeline viewer (real robot, no planner).",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--scenario-bank", default="curling_circle")
parser.add_argument(
    "--goal-xy", type=float, nargs=2, default=None, metavar=("X", "Y"),
    help="Override the house's planar position (m, world) to match the "
    "real lane.",
)
parser.add_argument(
    "--surface-z-offset", type=float, default=0.0,
    help="Shift the modelled lane (m, world z) to match a physically "
    "thicker or thinner real surface. Puck resting height, house height "
    "and ee_z_target all move with it.",
)
parser.add_argument(
    "--launch-length", type=float, default=None,
    help="Override the launch box's length in x (m); the near edge stays "
    f"fixed at {DEFAULT_LAUNCH_BOX[0]}. Default: the task's launch box.",
)
parser.add_argument("--rate", type=float, default=30.0)
parser.add_argument(
    "--tf-frame", required=True, help="TF frame of the tracked puck"
)
parser.add_argument("--base-frame", default="fr3_link0")
parser.add_argument(
    "--joint-topic",
    default="/franka_robot_state_broadcaster/measured_joint_states",
)
args = parser.parse_args()

bank = scenarios.load(args.scenario_bank)
launch_box = DEFAULT_LAUNCH_BOX
if args.launch_length is not None:
    x_lo, _, y_lo, y_hi = DEFAULT_LAUNCH_BOX
    launch_box = (x_lo, x_lo + args.launch_length, y_lo, y_hi)
task = CurlingFr3(
    shape=bank.shape,
    scale=bank.scale,
    goal_xy=tuple(args.goal_xy) if args.goal_xy else bank.goal_xy,
    launch_box=launch_box,
    surface_z_offset=args.surface_z_offset,
    trace_sites=[],
    model_config=load_numerics("curling"),
    **reward_profiles.load("curling"),
)

x_lo, x_hi, y_lo, y_hi = task.launch_box
print(f"launch box: x [{x_lo:.3f}, {x_hi:.3f}]  y [{y_lo:.3f}, {y_hi:.3f}]")
print(f"surface z-offset: {args.surface_z_offset:+.4f} m")
print(f"puck resting height / EE push height: {task.ee_z_target:.4f} m")

state_reader_factory = functools.partial(
    fr3_adapter.make_state_reader,
    joint_topic=args.joint_topic,
    tf_frame=args.tf_frame,
    base_frame=args.base_frame,
    pose_correction=False,  # a disc has no Push-T silhouette ambiguity
)

rclpy.init()
node = StateViewerNode(task, state_reader_factory, rate=args.rate)
try:
    rclpy.spin(node)
finally:
    node.destroy_node()
    rclpy.shutdown()
