"""Real-robot control node for 2-D EE twist tasks: trajectory replay.

Entirely task-agnostic infrastructure (:class:`~bampc.ros.control_node.
ControlNode`) wired to the Push-FR3 command writer
(:mod:`bampc.ros.adapters.push_fr3`), which publishes a
``TwistStamped`` to MoveIt Servo. **Curling-FR3 runs this one too**, despite
the name: the writer only ever sees ``u[0]``/``u[1]`` and knows nothing about
the task, so every task whose action is a 2-D EE twist shares it (the same
reuse ``run_planner_node_curling.py`` already makes of the adapter itself).
Balance-FR3 and Flip-FR3 need their own only because they command joint
velocities. Run the matching ``run_planner_node*.py`` as a **separate
process** alongside this one.

Run::

    uv run python scripts/ros/run_control_node_push_fr3.py
    uv run python scripts/ros/run_control_node_push_fr3.py --deadman  # curling
"""

from __future__ import annotations

import argparse
import functools

import rclpy

from bampc.ros.adapters import push_fr3 as push_fr3_adapter
from bampc.ros.control_node import ControlNode

parser = argparse.ArgumentParser(
    description="Push-FR3 control node (real robot).",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--control-rate", type=float, default=50.0)
parser.add_argument("--spline-topic", default="/bampc/trajectory")
parser.add_argument(
    "--watchdog-timeout",
    type=float,
    default=0.5,
    help="Zero the command if no trajectory arrives within this many "
    "seconds -- pick a small multiple of the planner's replan period.",
)
parser.add_argument("--twist-topic", default="/servo_node/delta_twist_cmds")
parser.add_argument("--frame-id", default="fr3_link0")
parser.add_argument(
    "--dry-run",
    action="store_true",
    help="Publish zero twists instead of the planned command -- the "
    "planner/trajectory pipeline still runs for real, the robot just "
    "never moves. Pair with `run_planner_node_push_fr3.py --viewer` (or "
    "run_state_viewer_node_push_fr3.py) to sanity-check a rollout on a CPU "
    "MuJoCo viewer before trusting it to actually drive the arm.",
)
parser.add_argument(
    "--deadman",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Gate every command on an arm/disarm toggle driven by Enter in "
    "*this* terminal; starts disarmed. Worth having for a task that "
    "commands a deliberate high-speed strike (Curling-FR3 runs at "
    "max_lin_vel 0.8 m/s), where the planner's first published horizon is "
    "not something to discover by watching the arm take it.",
)
args = parser.parse_args()

command_writer_factory = functools.partial(
    push_fr3_adapter.make_command_writer,
    twist_topic=args.twist_topic,
    frame_id=args.frame_id,
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
