"""Real-robot control node for Balance-FR3: trajectory replay, its own process.

Entirely task-agnostic infrastructure (:class:`~bampc.ros.control_node.
ControlNode`) wired to the Balance-FR3 command writer
(:mod:`bampc.ros.adapters.balance_fr3`), which publishes a
``control_msgs/JointJog`` to MoveIt Servo -- direct joint-velocity control,
no IK. Run ``run_planner_node_balance_fr3.py`` as a **separate process**
alongside this one.

``--deadman`` (default on) requires pressing Enter in this terminal to arm
real commands and pressing it again to disarm -- see
:class:`~bampc.ros.control_node.ControlNode`'s docstring for why
this is a toggle rather than a true continuous-hold deadman. Starts
disarmed, so nothing moves until you press Enter once.

Run::

    uv run python scripts/ros/run_control_node_balance_fr3.py
"""

from __future__ import annotations

import argparse
import functools

import rclpy

from bampc.ros.adapters import balance_fr3 as balance_fr3_adapter
from bampc.ros.control_node import ControlNode

parser = argparse.ArgumentParser(
    description="Balance-FR3 control node (real robot, joint-velocity jog).",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--control-rate", type=float, default=100.0)
parser.add_argument("--spline-topic", default="/bampc/trajectory")
parser.add_argument(
    "--watchdog-timeout",
    type=float,
    default=0.5,
    help="Zero the command if no trajectory arrives within this many "
    "seconds -- pick a small multiple of the planner's replan period.",
)
parser.add_argument("--joint-topic", default="/servo_node/delta_joint_cmds")
parser.add_argument(
    "--dry-run",
    action="store_true",
    help="Publish zero joint velocities instead of the planned command -- "
    "the planner/trajectory pipeline still runs for real, the robot just "
    "never moves. Pair with `run_planner_node_balance_fr3.py --viewer` (or "
    "run_state_viewer_node_balance_fr3.py) to sanity-check a rollout on a "
    "CPU MuJoCo viewer before trusting it to actually drive the arm.",
)
parser.add_argument(
    "--deadman",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Require pressing Enter in this terminal to arm real commands, "
    "and again to disarm (publish zeros) -- a safety toggle independent "
    "of Ctrl+C. Starts disarmed. --no-deadman disables it, e.g. for "
    "automated testing.",
)
args = parser.parse_args()

command_writer_factory = functools.partial(
    balance_fr3_adapter.make_command_writer, joint_topic=args.joint_topic
)

rclpy.init()
node = ControlNode(
    command_writer_factory,
    control_rate=args.control_rate,
    spline_topic=args.spline_topic,
    watchdog_timeout=args.watchdog_timeout,
    dry_run=args.dry_run,
    deadman=args.deadman,
)
try:
    rclpy.spin(node)
finally:
    node.destroy_node()
    rclpy.shutdown()
