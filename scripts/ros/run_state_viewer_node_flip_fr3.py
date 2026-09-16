"""Debug node for Flip-FR3: render the live state pipeline, no planner.

Builds the exact same state-reader factory ``run_planner_node_flip_fr3.py``
would (:mod:`bampc.ros.adapters.flip_fr3`) and feeds it into
:class:`~bampc.ros.state_viewer_node.StateViewerNode` -- a CPU passive
MuJoCo viewer with no GPU, no planner, no rollout engine. Use this first, on
real TF/JointState, to check the tracking/encoder pipeline (right frame,
right quaternion convention, right joint mapping) before ever starting the
planner node.

``--tf-frame`` is required -- pass the TF frame name your tracker publishes
for the box in this setup.

Run::

    uv run python scripts/ros/run_state_viewer_node_flip_fr3.py \
        --tf-frame box_marker
"""

from __future__ import annotations

import argparse
import functools

import rclpy

from bampc.ros.adapters import flip_fr3 as flip_fr3_adapter
from bampc.ros.state_viewer_node import StateViewerNode
from bampc.task.common.shapes import list_shapes
from bampc.task.flip_fr3 import FlipFr3

parser = argparse.ArgumentParser(
    description="Flip-FR3 state-pipeline viewer (real robot, no planner).",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--shape", default="cracker_box", choices=list_shapes())
parser.add_argument("--scale", type=float, default=1.0)
parser.add_argument(
    "--no-wall", dest="wall", action="store_false",
    help="Drop the static wall behind the box (on by default -- see "
    "bampc.task.flip_fr3).",
)
parser.add_argument("--rate", type=float, default=30.0)
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
args = parser.parse_args()

task = FlipFr3(shape=args.shape, scale=args.scale, wall=args.wall)

state_reader_factory = functools.partial(
    flip_fr3_adapter.make_state_reader,
    joint_topic=args.joint_topic,
    tf_frame=args.tf_frame,
    base_frame=args.base_frame,
)

rclpy.init()
node = StateViewerNode(task, state_reader_factory, rate=args.rate)
try:
    rclpy.spin(node)
finally:
    node.destroy_node()
    rclpy.shutdown()
