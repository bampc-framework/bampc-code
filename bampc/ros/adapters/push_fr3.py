"""Push-FR3 adapter: state in from JointState + TF, commands out to Servo.

The concrete instance of the two hooks :mod:`bampc.ros.adapters`
declares, so :class:`~bampc.ros.planner_node.PlannerNode` and
:class:`~bampc.ros.control_node.ControlNode` stay task-agnostic. A
future
Balance / Peg-FR3 adapter implements the same two functions with different
internals; nothing in the nodes changes.
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING

import numpy as np

from bampc.planner.base import StateSnapshot
from bampc.task.base import quat_mul
from bampc.task.common.ik import quat_rotate
from bampc.uncertainty.filter import PoseKalman
from bampc.uncertainty.noise import _dof_of_qpos

try:
    import rclpy
    from geometry_msgs.msg import Pose, TwistStamped
    from moveit_msgs.action import MoveGroup
    from moveit_msgs.msg import (
        BoundingVolume,
        Constraints,
        MotionPlanRequest,
        OrientationConstraint,
        PlanningOptions,
        PositionConstraint,
    )
    from moveit_msgs.srv import ServoCommandType
    from rclpy.action import ActionClient
    from rclpy.qos import qos_profile_sensor_data
    from rclpy.time import Time
    from sensor_msgs.msg import JointState
    from shape_msgs.msg import SolidPrimitive
    from std_srvs.srv import SetBool
    from tf2_ros import Buffer, TransformListener

    _HAS_ROS = True
except ImportError:  # ROS not installed; keep the module importable.
    _HAS_ROS = False

if TYPE_CHECKING:
    from rclpy.node import Node

    from bampc.ros.adapters import CommandWriter, HomeMover, StateReader
    from bampc.task.push_fr3 import PushFr3

_ARM_JOINTS = [f"fr3_joint{i}" for i in range(1, 8)]

# Push-T + FoundationPose only: the tracked pose is occasionally the
# ambiguous 180-degree-about-local-X solution (the T's silhouette is
# symmetric under that flip), reported with local Z pointing down instead
# of up. See make_state_reader's docstring.
_FLIP_X180 = np.array([0.0, 1.0, 0.0, 0.0])
_IDENTITY_QUAT = np.array([1.0, 0.0, 0.0, 0.0])


def _yaw_of(quat_wxyz: np.ndarray) -> float:
    """Yaw about world Z from a ``[w, x, y, z]`` quaternion."""
    w, x, y, z = quat_wxyz
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def _correct_free_pose(
    pos: np.ndarray,
    quat: np.ndarray,
    read_state: dict,
    marker_z_offset: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the Push-T FoundationPose z-offset and latched symmetric flip.

    See ``make_state_reader``'s docstring. ``read_state["flip_corr"]`` is
    latched from the first call and reused for the rest of the rollout.
    """
    if read_state["flip_corr"] is None:
        z_world = quat_rotate(quat, np.array([0.0, 0.0, 1.0]))
        read_state["flip_corr"] = (
            _FLIP_X180 if z_world[2] < 0.0 else _IDENTITY_QUAT
        )
    quat = quat_mul(quat, read_state["flip_corr"])
    pos = pos + quat_rotate(quat, np.array([0.0, 0.0, marker_z_offset]))
    return pos, quat


def make_state_reader(  # noqa: PLR0915
    node: Node,
    task: PushFr3,
    *,
    joint_topic: str = "/franka_robot_state_broadcaster/measured_joint_states",
    tf_frame: str,
    base_frame: str = "fr3_link0",
    pos_std: float = 0.005,
    rot_std: float = 0.02,
    accel_std: float = 0.2,
    ang_accel_std: float = 1.0,
    marker_z_offset: float = -0.0, # was 0.025
    pose_correction: bool = True,
) -> StateReader:
    """Build a Push-FR3 :class:`~bampc.ros.adapters.StateReader`.

    Owns the arm's ``JointState`` subscription, a TF listener for the tracked
    block, and (free-jointed block only) a
    :class:`~bampc.uncertainty.filter.PoseKalman` filling in the
    block's twist -- a TF pose lookup is exactly that filter's "pose-only
    sensor" case (``observe_twist=False``). ``pos_std``/``rot_std`` seed its
    measurement covariance; tune them to the tracker's real noise.

    **Push-T + FoundationPose specific**, and skipped entirely on a planar
    block (no z DOF): FoundationPose's object-frame origin sits
    ``marker_z_offset`` along the object's local +Z from the free joint's body
    origin, so the raw pose is corrected before being written to ``qpos``.
    Separately, the T's silhouette is symmetric under a 180-degree rotation
    about local X, so the tracker sometimes locks onto that solution. The flip
    is constant for one rollout, so it is latched from the **first** reading
    rather than re-detected each tick. Another shape needs a different
    ``marker_z_offset`` and may have no such ambiguity.

    Args:
        node: The owning ROS node (subscriptions/TF listener attach to it).
        task: The constructed ``PushFr3`` task -- supplies ``mj_model`` and
            ``object_pose_qpos``.
        joint_topic: Franka broadcaster's joint-state topic.
        tf_frame: TF frame of the tracked block.
        base_frame: TF frame ``tf_frame`` is looked up relative to.
        pos_std: Initial position measurement std (m) for the filter.
        rot_std: Initial rotation measurement std (rad) for the filter.
        accel_std: Process noise -- unmodelled linear acceleration (m/s^2).
        ang_accel_std: Process noise -- unmodelled angular acceleration
            (rad/s^2).
        marker_z_offset: Distance (m) from FoundationPose's object-frame
            origin to the free joint's body origin along local +Z. Push-T
            specific; see above.
        pose_correction: Apply the Push-T offset + X180-flip correction
            described above. Turn it off for a tracked object that is not
            the T slab -- a disc has no such silhouette ambiguity, and
            latching a spurious flip from one reading would corrupt every
            later one. Otherwise this reader is shape-agnostic, which is
            why Curling-FR3 reuses it rather than copying it.

    Returns:
        A zero-arg closure returning the latest ``StateSnapshot``.
    """
    if not _HAS_ROS:
        raise RuntimeError("rclpy is not available; source a ROS 2 install")

    mj_model = task.mj_model
    qpos_adr = np.array([mj_model.joint(n).qposadr[0] for n in _ARM_JOINTS])
    qvel_adr = np.array([mj_model.joint(n).dofadr[0] for n in _ARM_JOINTS])

    layout = task.object_pose_qpos
    filt = None
    if not layout.is_planar:
        filt = PoseKalman(
            layout,
            _dof_of_qpos(mj_model, layout.adr),
            pos_std=pos_std,
            rot_std=rot_std,
            accel_std=accel_std,
            ang_accel_std=ang_accel_std,
            observe_twist=False,
        )

    lock = threading.Lock()
    latest = {"pos": None, "vel": None}

    def on_joint_state(msg: JointState) -> None:
        pos = dict(zip(msg.name, msg.position, strict=False))
        vel = dict(zip(msg.name, msg.velocity, strict=False))
        with lock:
            latest["pos"] = np.array([pos[n] for n in _ARM_JOINTS])
            latest["vel"] = np.array([vel.get(n, 0.0) for n in _ARM_JOINTS])

    node.create_subscription(
        JointState, joint_topic, on_joint_state, qos_profile_sensor_data
    )
    tf_buffer = Buffer()
    TransformListener(tf_buffer, node)

    start = time.monotonic()
    read_state = {"t": None, "flip_corr": None}

    def read() -> StateSnapshot:
        with lock:
            arm_pos, arm_vel = latest["pos"], latest["vel"]
        if arm_pos is None:
            raise RuntimeError(f"no JointState received yet on {joint_topic!r}")

        qpos = np.zeros(mj_model.nq)
        qvel = np.zeros(mj_model.nv)
        qpos[qpos_adr] = arm_pos
        qvel[qvel_adr] = arm_vel

        tf = tf_buffer.lookup_transform(base_frame, tf_frame, Time())
        p, q = tf.transform.translation, tf.transform.rotation
        pos = np.array([p.x, p.y, p.z])
        quat = np.array([q.w, q.x, q.y, q.z])  # ROS xyzw -> MuJoCo wxyz

        if layout.is_planar:
            qpos[layout.x_adr] = pos[0]
            qpos[layout.y_adr] = pos[1]
            qpos[layout.yaw_adr] = _yaw_of(quat)
        else:
            if pose_correction:
                pos, quat = _correct_free_pose(
                    pos, quat, read_state, marker_z_offset
                )
            a = layout.adr
            qpos[a : a + 3] = pos
            qpos[a + 3 : a + 7] = quat
            # TF gives pose only -- the object's twist slot stays zero until
            # the filter below fills it in.

        now = time.monotonic()
        dt = 0.0 if read_state["t"] is None else now - read_state["t"]
        read_state["t"] = now
        if filt is not None:
            qpos, qvel = filt.update(qpos, qvel, dt)

        return StateSnapshot(qpos=qpos, qvel=qvel, time=now - start)

    # Exposed (not part of the StateReader contract) so a caller can build a
    # belief ensemble from this exact filter's live posterior instead of a
    # second, out-of-sync one -- see run_planner_node_push_fr3.py. None if the
    # object's pose layout is planar (no PoseKalman was built above).
    read.filt = filt
    return read


def make_command_writer(
    node: Node,
    *,
    twist_topic: str = "/servo_node/delta_twist_cmds",
    frame_id: str = "fr3_link0",
    switch_command_type_srv: str = "/servo_node/switch_command_type",
) -> CommandWriter:
    """Build a Push-FR3 :class:`~bampc.ros.adapters.CommandWriter`.

    Wraps a 2-D EE twist (``sampling_space="task"``) into a ``TwistStamped``
    published to MoveIt Servo, which runs its own IK on the real robot -- so
    ``task.control_map_host`` (this package's own IK, used inside rollouts for
    cost-consistency) is not needed on this path.

    Also switches Servo into ``TWIST`` command mode once, here, before
    returning the writer: the current ``servo_node`` accepts multiple input
    types (joint jog / twist / pose) gated by its ``switch_command_type``
    service (``moveit_msgs/srv/ServoCommandType``), and won't act on
    ``TwistStamped`` messages published to ``twist_topic`` until told to --
    silently, with no error on the publishing side. This node is the only
    ``TwistStamped`` publisher for the whole session (planner-driven
    rollouts and teleop jogging both funnel through it), so setting the
    mode once here covers both.
    """
    if not _HAS_ROS:
        raise RuntimeError("rclpy is not available; source a ROS 2 install")

    pub = node.create_publisher(TwistStamped, twist_topic, 10)

    switch = node.create_client(ServoCommandType, switch_command_type_srv)
    if switch.wait_for_service(timeout_sec=5.0):
        fut = switch.call_async(ServoCommandType.Request(command_type=1))
        rclpy.spin_until_future_complete(node, fut, timeout_sec=5.0)
        result = fut.result() if fut.done() else None
        if result is None or not result.success:
            node.get_logger().warn(
                "failed to switch Servo to TWIST command mode -- it may "
                "ignore all commands on this topic"
            )
    else:
        node.get_logger().warn(
            f"{switch_command_type_srv} unavailable; Servo command type "
            "not set -- it may ignore all commands on this topic"
        )

    def write(u: np.ndarray) -> None:
        msg = TwistStamped()
        msg.header.stamp = node.get_clock().now().to_msg()
        msg.header.frame_id = frame_id
        msg.twist.linear.x = float(u[0])
        msg.twist.linear.y = float(u[1])
        pub.publish(msg)

    return write


def make_home_mover(  # noqa: PLR0913, PLR0915
    node: Node,
    *,
    home_pos: tuple[float, float, float],
    home_quat: tuple[float, float, float, float],
    ee_link: str = "fr3_pusher_tcp",
    frame_id: str = "fr3_link0",
    planning_group: str = "fr3_arm",
    move_action: str = "/move_action",
    pause_servo_srv: str = "/servo_node/pause_servo",
    pos_tol: float = 0.005,
    ori_tol: float = 0.05,
    vel_scale: float = 0.2,
    acc_scale: float = 0.2,
    servo_settle: float = 0.2,
    resume_settle: float = 1.5,
    plan_timeout: float = 30.0,
) -> HomeMover:
    """Build a Push-FR3 :class:`~bampc.ros.adapters.HomeMover`.

    Pauses MoveIt Servo -- it owns the twist-command stream while the planner
    runs, so it must yield control -- then plans and executes a **Cartesian**
    pose goal (position + orientation constraints, not a joint-space goal) via
    ``/move_action``, then resumes Servo.

    Args:
        node: The owning ROS node (the action/service clients attach here).
        home_pos: Target EE position ``(x, y, z)`` in ``frame_id``.
        home_quat: Target EE orientation ``[w, x, y, z]`` in ``frame_id``.
        ee_link: Link the pose goal is expressed for.
        frame_id: Frame the pose goal and constraints are expressed in.
        planning_group: MoveIt planning group name.
        move_action: MoveIt's ``MoveGroup`` action name.
        pause_servo_srv: Servo's ``pause_servo`` (``SetBool``) service name
            -- ``data=True`` pauses, ``data=False`` resumes.
        pos_tol: Position constraint sphere radius (m).
        ori_tol: Orientation constraint tolerance per axis (rad).
        vel_scale: ``max_velocity_scaling_factor`` for the home move.
        acc_scale: ``max_acceleration_scaling_factor`` for the home move.
        servo_settle: Seconds to wait after pausing Servo before planning.
        resume_settle: Seconds to wait after the move finishes before
            resuming Servo -- longer than ``servo_settle`` because this
            side is the one racing the controller: it can report the
            action goal "succeeded" a beat before it has internally
            released it and started accepting Servo's topic-streamed
            points again, and losing that race drops Servo's first
            commands silently (no error, the arm just doesn't move).
        plan_timeout: Seconds to wait for the plan+execute to finish.

    Returns:
        A zero-arg closure that runs the sequence and returns success. Runs
        for the length of a MoveIt plan+execute -- call it off the ROS
        executor thread.
    """
    if not _HAS_ROS:
        raise RuntimeError("rclpy is not available; source a ROS 2 install")

    move = ActionClient(node, MoveGroup, move_action)
    pause_servo = node.create_client(SetBool, pause_servo_srv)

    def _await(fut, timeout: float) -> bool:
        t0 = time.monotonic()
        while not fut.done():
            if time.monotonic() - t0 > timeout:
                node.get_logger().error("home: future timed out")
                return False
            time.sleep(0.01)
        return True

    def _set_paused(paused: bool, timeout: float = 2.0) -> None:
        if not pause_servo.wait_for_service(timeout_sec=timeout):
            node.get_logger().warn(f"home: {pause_servo_srv} unavailable")
            return
        fut = pause_servo.call_async(SetBool.Request(data=paused))
        _await(fut, timeout=5.0)

    def _build_goal():
        req = MotionPlanRequest()
        req.group_name = planning_group
        req.num_planning_attempts = 10
        req.allowed_planning_time = 5.0
        req.max_velocity_scaling_factor = vel_scale
        req.max_acceleration_scaling_factor = acc_scale

        pc = PositionConstraint()
        pc.header.frame_id = frame_id
        pc.link_name = ee_link
        pc.weight = 1.0
        sphere = SolidPrimitive()
        sphere.type = SolidPrimitive.SPHERE
        sphere.dimensions = [pos_tol]
        bv = BoundingVolume()
        bv.primitives.append(sphere)
        p = Pose()
        p.position.x, p.position.y, p.position.z = home_pos
        p.orientation.w = 1.0  # region pose; identity is fine
        bv.primitive_poses.append(p)
        pc.constraint_region = bv

        oc = OrientationConstraint()
        oc.header.frame_id = frame_id
        oc.link_name = ee_link
        oc.orientation.w, oc.orientation.x = home_quat[0], home_quat[1]
        oc.orientation.y, oc.orientation.z = home_quat[2], home_quat[3]
        oc.absolute_x_axis_tolerance = ori_tol
        oc.absolute_y_axis_tolerance = ori_tol
        oc.absolute_z_axis_tolerance = ori_tol
        oc.weight = 1.0

        c = Constraints()
        c.position_constraints.append(pc)
        c.orientation_constraints.append(oc)
        req.goal_constraints.append(c)

        goal = MoveGroup.Goal()
        goal.request = req
        goal.planning_options = PlanningOptions()
        goal.planning_options.plan_only = False
        return goal

    def home() -> bool:
        _set_paused(True)
        time.sleep(servo_settle)
        try:
            if not move.wait_for_server(timeout_sec=5.0):
                node.get_logger().error(f"home: {move_action} unavailable")
                return False
            send_fut = move.send_goal_async(_build_goal())
            if not _await(send_fut, timeout=10.0):
                return False
            gh = send_fut.result()
            if gh is None or not gh.accepted:
                node.get_logger().error("home: goal rejected")
                return False
            res_fut = gh.get_result_async()
            if not _await(res_fut, timeout=plan_timeout):
                return False
            result = res_fut.result()
            return bool(result and result.result.error_code.val == 1)
        finally:
            # The controller can report the action goal "succeeded" a beat
            # before it has internally released it and started accepting
            # Servo's topic-streamed points again -- resuming immediately
            # risks Servo's first commands landing in that gap and being
            # silently dropped.
            time.sleep(resume_settle)
            _set_paused(False)

    return home
