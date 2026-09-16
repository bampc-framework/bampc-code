"""Debug node for Push-FR3: render the live state pipeline, no planner.

Builds the exact same state-reader factory ``run_planner_node_push_fr3.py``
would
(:mod:`bampc.ros.adapters.push_fr3`) and feeds it into
:class:`~bampc.ros.state_viewer_node.StateViewerNode` -- a CPU passive
MuJoCo viewer with no GPU, no planner, no rollout engine. Use this first, on
real TF/JointState, to check the tracking/encoder pipeline (right frame,
right quaternion convention, right joint mapping) before ever starting the
planner node.

Run::

    uv run python scripts/ros/run_state_viewer_node_push_fr3.py \
        --tf-frame objectPushT_MuJoCo
"""

from __future__ import annotations

import argparse
import functools

import rclpy

from bampc.ros.adapters import push_fr3 as push_fr3_adapter
from bampc.ros.state_viewer_node import StateViewerNode
from bampc.task.common.shapes import list_shapes
from bampc.task.push_fr3 import PushFr3

parser = argparse.ArgumentParser(
    description="Push-FR3 state-pipeline viewer (real robot, no planner).",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--sampling", default="task", choices=["task", "joint"])
parser.add_argument(
    "--manipulation", default="free", choices=["joint", "free"]
)
parser.add_argument("--shape", default="t", choices=list_shapes())
parser.add_argument("--scale", type=float, default=1.0)
parser.add_argument("--rate", type=float, default=30.0)
parser.add_argument(
    "--tf-frame", required=True, help="TF frame of the tracked block"
)
parser.add_argument("--base-frame", default="fr3_link0")
parser.add_argument(
    "--joint-topic",
    default="/franka_robot_state_broadcaster/measured_joint_states",
)
args = parser.parse_args()

task = PushFr3(
    sampling_space=args.sampling,
    manipulation_type=args.manipulation,
    shape=args.shape,
    scale=args.scale,
    trace_sites=[],
)

state_reader_factory = functools.partial(
    push_fr3_adapter.make_state_reader,
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
