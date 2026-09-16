"""Debug node for Balance-FR3: render the live state pipeline, no planner.

Builds the exact same state-reader factory ``run_planner_node_balance_fr3.py``
would (:mod:`bampc.ros.adapters.balance_fr3`) and feeds it into
:class:`~bampc.ros.state_viewer_node.StateViewerNode` -- a CPU passive
MuJoCo viewer with no GPU, no planner, no rollout engine. Use this first, on
real TF/JointState, to check the tracking/encoder pipeline (right frame,
right quaternion convention, right joint mapping) before ever starting the
planner node or the joint-jog sine-wave script.

``--tf-frame`` defaults to
:data:`bampc.ros.adapters.balance_fr3.SPHERE_TF_FRAME`
(``sphere_centre``); pass it explicitly (e.g.
:data:`~bampc.ros.adapters.balance_fr3.DEFAULT_TF_FRAME` for the old
fiducial marker) to track something else instead -- ``sphere_centre`` is
then ignored entirely.

Run::

    uv run python scripts/ros/run_state_viewer_node_balance_fr3.py
"""

from __future__ import annotations

import argparse
import functools

import rclpy

from bampc.ros.adapters import balance_fr3 as balance_fr3_adapter
from bampc.ros.state_viewer_node import StateViewerNode
from bampc.task.balance_fr3 import BalanceFr3
from bampc.task.common.shapes import list_shapes

parser = argparse.ArgumentParser(
    description="Balance-FR3 state-pipeline viewer (real robot, no planner).",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--shape", default="sphere", choices=list_shapes())
parser.add_argument("--scale", type=float, default=1.0)
parser.add_argument("--rate", type=float, default=30.0)
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
args = parser.parse_args()

task = BalanceFr3(
    sampling_space="joint",
    shape=args.shape,
    scale=args.scale,
)

state_reader_factory = functools.partial(
    balance_fr3_adapter.make_state_reader,
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
