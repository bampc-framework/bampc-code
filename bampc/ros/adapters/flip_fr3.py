"""Flip-FR3 adapter: state in from JointState + TF, joint jog out.

The concrete instance of the hooks :mod:`bampc.ros.adapters`
declares, so :class:`~bampc.ros.planner_node.PlannerNode` and
:class:`~bampc.ros.control_node.ControlNode` stay task-agnostic.
Mirrors :mod:`bampc.ros.adapters.balance_fr3`'s structure: ``FlipFr3``
has no ``sampling_space`` axis at all -- the sampled action is already the 7
joint velocities (``control_map_host`` is the identity), so this adapter
commands the real robot with direct joint velocities ("joint jog") instead
of wrapping a 2-D EE twist for Servo's own IK.

Unlike Balance-FR3's sphere, the tracked box has real, TF-observable
orientation, so none of that adapter's no-slip rolling-velocity workaround
applies here -- a plain :class:`~bampc.uncertainty.filter.PoseKalman`
on the raw TF reading (as Push-FR3's free-block branch uses) is the right
model.
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING, Literal

import numpy as np

from bampc.planner.base import StateSnapshot
from bampc.uncertainty.filter import Passthrough, PoseKalman
from bampc.uncertainty.noise import _dof_of_qpos

try:
    import rclpy
    from control_msgs.msg import JointJog
    from moveit_msgs.action import MoveGroup
    from moveit_msgs.msg import (
        Constraints,
        JointConstraint,
        MotionPlanRequest,
        PlanningOptions,
    )
    from moveit_msgs.srv import ServoCommandType
    from rclpy.action import ActionClient
    from rclpy.qos import qos_profile_sensor_data
    from rclpy.time import Time
    from sensor_msgs.msg import JointState
    from std_srvs.srv import SetBool
    from tf2_ros import Buffer, TransformListener

    _HAS_ROS = True
except ImportError:  # ROS not installed; keep the module importable.
    _HAS_ROS = False

if TYPE_CHECKING:
    from rclpy.node import Node

    from bampc.ros.adapters import CommandWriter, HomeMover, StateReader
    from bampc.task.flip_fr3 import FlipFr3

_ARM_JOINTS = [f"fr3_joint{i}" for i in range(1, 8)]


def make_state_reader(
    node: Node,
    task: FlipFr3,
    *,
    joint_topic: str = "/franka_robot_state_broadcaster/measured_joint_states",
    tf_frame: str,
    base_frame: str = "fr3_link0",
    pos_std: float = 0.005,
    rot_std: float = 0.02,
    accel_std: float = 0.2,
    ang_accel_std: float = 1.0,
    filt_kind: Literal["kalman", "passthrough"] = "kalman",
) -> StateReader:
    """Build a Flip-FR3 :class:`~bampc.ros.adapters.StateReader`.

    Same structure as ``adapters.push_fr3.make_state_reader``'s free-block
    branch: the arm's ``JointState`` subscription, a TF listener for the box,
    and a filter filling in its twist from a pose-only reading. No
    marker-offset or silhouette-flip correction (those are Push-T +
    FoundationPose specific).

    Args:
        node: The owning ROS node (subscriptions/TF listener attach to it).
        task: The constructed ``FlipFr3`` task -- supplies ``mj_model`` and
            ``object_pose_qpos``.
        joint_topic: Franka broadcaster's joint-state topic.
        tf_frame: TF frame of the tracked box. No default -- unlike
            Balance-FR3's fixed ``sphere_centre``, Flip's box marker/frame
            name isn't a single canonical value across setups.
        base_frame: TF frame ``tf_frame`` is looked up relative to.
        pos_std: Initial position measurement std (m) for the filter.
        rot_std: Initial rotation measurement std (rad) for the filter.
        accel_std: Process noise -- unmodelled linear acceleration (m/s^2).
        ang_accel_std: Process noise -- unmodelled angular acceleration
            (rad/s^2).
        filt_kind: ``"kalman"`` (default) fills the box's twist in via
            :class:`~bampc.uncertainty.filter.PoseKalman` on the raw
            TF reading. ``"passthrough"`` skips filtering entirely --
            :class:`~bampc.uncertainty.filter.Passthrough`, "the
            naive baseline" per its own docstring: the raw TF pose is used
            exactly as read, and the box's twist stays zero (TF alone
            carries no velocity, and nothing fills it in).

    Returns:
        A zero-arg closure returning the latest ``StateSnapshot``.
    """
    if not _HAS_ROS:
        raise RuntimeError("rclpy is not available; source a ROS 2 install")

    mj_model = task.mj_model
    qpos_adr = np.array([mj_model.joint(n).qposadr[0] for n in _ARM_JOINTS])
    qvel_adr = np.array([mj_model.joint(n).dofadr[0] for n in _ARM_JOINTS])

    layout = task.object_pose_qpos
    dof_adr = _dof_of_qpos(mj_model, layout.adr)
    if filt_kind == "passthrough":
        filt = Passthrough()
    elif filt_kind == "kalman":
        filt = PoseKalman(
            layout,
            dof_adr,
            pos_std=pos_std,
            rot_std=rot_std,
            accel_std=accel_std,
            ang_accel_std=ang_accel_std,
            observe_twist=False,
        )
    else:
        raise ValueError(f"unknown filt_kind {filt_kind!r}")

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
    read_state = {"t": None}

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
        a = layout.adr
        qpos[a : a + 3] = [p.x, p.y, p.z]
        qpos[a + 3 : a + 7] = [q.w, q.x, q.y, q.z]  # ROS xyzw -> MuJoCo wxyz
        # TF gives pose only -- the box's twist slot stays zero until the
        # filter below fills it in.

        now = time.monotonic()
        dt = 0.0 if read_state["t"] is None else now - read_state["t"]
        read_state["t"] = now
        qpos, qvel = filt.update(qpos, qvel, dt)

        return StateSnapshot(qpos=qpos, qvel=qvel, time=now - start)

    # Exposed (not part of the StateReader contract) so a caller can build a
    # belief ensemble from this exact filter's live posterior instead of a
    # second, out-of-sync one -- see run_planner_node_flip_fr3.py.
    read.filt = filt
    return read


def make_command_writer(
    node: Node,
    *,
    joint_topic: str = "/servo_node/delta_joint_cmds",
    joint_names: list[str] = _ARM_JOINTS,
    switch_command_type_srv: str = "/servo_node/switch_command_type",
) -> CommandWriter:
    """Build a Flip-FR3 :class:`~bampc.ros.adapters.CommandWriter`.

    Wraps the 7 sampled joint velocities (``control_map_host`` is the
    identity for ``FlipFr3`` -- it has no ``sampling_space`` axis) into a
    ``control_msgs/JointJog`` published to MoveIt Servo's joint-jog input,
    no IK anywhere in the loop.

    Also switches Servo into ``JOINT_JOG`` command mode once, here, before
    returning the writer -- the same reasoning as
    ``adapters.push_fr3.make_command_writer``: ``servo_node`` ignores
    messages on a topic that doesn't match its current mode, silently, with
    no error on the publishing side.
    """
    if not _HAS_ROS:
        raise RuntimeError("rclpy is not available; source a ROS 2 install")

    pub = node.create_publisher(JointJog, joint_topic, 10)

    switch = node.create_client(ServoCommandType, switch_command_type_srv)
    if switch.wait_for_service(timeout_sec=5.0):
        fut = switch.call_async(
            ServoCommandType.Request(command_type=ServoCommandType.Request.JOINT_JOG)
        )
        rclpy.spin_until_future_complete(node, fut, timeout_sec=5.0)
        result = fut.result() if fut.done() else None
        if result is None or not result.success:
            node.get_logger().warn(
                "failed to switch Servo to JOINT_JOG command mode -- it may "
                "ignore all commands on this topic"
            )
    else:
        node.get_logger().warn(
            f"{switch_command_type_srv} unavailable; Servo command type "
            "not set -- it may ignore all commands on this topic"
        )

    def write(u: np.ndarray) -> None:
        msg = JointJog()
        msg.header.stamp = node.get_clock().now().to_msg()
        msg.joint_names = list(joint_names)
        msg.velocities = [float(x) for x in u]
        pub.publish(msg)

    return write


def make_home_mover(  # noqa: PLR0913, PLR0915
    node: Node,
    *,
    q_home: np.ndarray,
    joint_names: list[str] = _ARM_JOINTS,
    planning_group: str = "fr3_arm",
    move_action: str = "/move_action",
    pause_servo_srv: str = "/servo_node/pause_servo",
    tol: float = 0.01,
    vel_scale: float = 0.2,
    acc_scale: float = 0.2,
    servo_settle: float = 0.2,
    resume_settle: float = 1.5,
    plan_timeout: float = 30.0,
) -> HomeMover:
    """Build a Flip-FR3 :class:`~bampc.ros.adapters.HomeMover`.

    Same pause/plan+execute/resume sequence as
    ``adapters.push_fr3.make_home_mover``, but the MoveIt goal is
    **joint-space** (a ``JointConstraint`` per arm joint) rather than a
    Cartesian pose: ``FlipFr3.q_home`` is already available and this task has
    no Cartesian EE target.

    Args:
        node: The owning ROS node (the action/service clients attach here).
        q_home: Target arm joint angles (rad), length 7, in ``joint_names``
            order -- pass ``task.q_home`` so the real-robot home move can
            never drift from the arm-home cost anchor used in rollouts.
        joint_names: Arm joint names, in ``q_home``'s order.
        planning_group: MoveIt planning group name.
        move_action: MoveIt's ``MoveGroup`` action name.
        pause_servo_srv: Servo's ``pause_servo`` (``SetBool``) service name
            -- ``data=True`` pauses, ``data=False`` resumes.
        tol: Per-joint position tolerance (rad), above and below.
        vel_scale: ``max_velocity_scaling_factor`` for the home move.
        acc_scale: ``max_acceleration_scaling_factor`` for the home move.
        servo_settle: Seconds to wait after pausing Servo before planning.
        resume_settle: Seconds to wait after the move finishes before
            resuming Servo -- see ``push_fr3.make_home_mover``'s docstring
            for why this is longer than ``servo_settle``.
        plan_timeout: Seconds to wait for the plan+execute to finish.

    Returns:
        A zero-arg closure that runs the sequence and returns success. Runs
        for the length of a MoveIt plan+execute -- call it off the ROS
        executor thread.
    """
    if not _HAS_ROS:
        raise RuntimeError("rclpy is not available; source a ROS 2 install")

    q_home = np.asarray(q_home, dtype=float)
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

        c = Constraints()
        for name, q in zip(joint_names, q_home, strict=True):
            jc = JointConstraint()
            jc.joint_name = name
            jc.position = float(q)
            jc.tolerance_above = tol
            jc.tolerance_below = tol
            jc.weight = 1.0
            c.joint_constraints.append(jc)
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
            time.sleep(resume_settle)
            _set_paused(False)

    return home
