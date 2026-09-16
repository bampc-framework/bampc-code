"""Push with an FR3 arm: two control spaces x two block manipulation types.

The pusher is the FR3 end-effector. Two orthogonal axes, both
velocity-actuated:

* ``sampling_space`` — ``"joint"`` samples the 7 joint velocities directly
  (``nu=7``), ``"task"`` samples a 2-D end-effector twist mapped each
  rollout step to joint velocities by a damped least-squares IK (``nu=2``).
* ``manipulation_type`` — ``"joint"`` models the block on three
  constrained planar joints (3-DOF), ``"free"`` as a free rigid body
  (6-DOF). Block geometry comes from ``bampc.task.common.shapes``.

The cost is block position + geodesic orientation error to the goal, an
EE->attractor-site term and a safety-zone step penalty, plus posture
regularizers (arxiv 2307.09105) that stabilize joint-space sampling, where
-- unlike task space -- no IK keeps the arm well-conditioned: an EE term
(point-down + push height) and an arm-home term anchoring the 7-DOF arm's
redundant null-space against drift into limits and singularities.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, Literal

import mujoco
import numpy as np
import warp as wp
from mujoco_warp._src.math import quat_sub, rot_vec_quat

from bampc import MODELS_DIR
from bampc.task.base import (
    ContactProbe,
    GoalDrift,
    ModelConfig,
    ObjectPose,
    Task,
    body_geom_ids,
)
from bampc.task.common.cholesky import vec6
from bampc.task.common.ee import attach_ee
from bampc.task.common.ik import (
    damped_ls,
    damped_ls_host,
    ee_jacobian_dof,
    quat_rotate,
    solve_task_pose_ik,
)
from bampc.task.common.shapes import (
    graft_shape,
    push_fr3_contact_budget,
    shape_com_xy,
    shape_min_z,
)

# Fallback solver options, used only when the caller names none; the tuned
# values live in configs/numerics/fr3_free.yaml and fr3_joint.yaml. NOT the
# tuned "joint" setup, which wants eulerdamp=False -- the caller names the
# profile, this task does not pick one per manipulation_type.
_BASELINE = ModelConfig(
    timestep=0.02,
    solver="Newton",
    integrator="implicitfast",
    cone="elliptic",
    jacobian="dense",
    impratio=1.0,
    iterations=5,
    ls_iterations=8,
    eulerdamp=True,
    warmstart=True,
)

#: Block models this task supports. Public so config loaders can validate
#: per-manipulation numerics against it.
MANIPULATION_TYPES = ("free", "joint")


@wp.kernel
def _fr3_cost_kernel(
    sensordata: wp.array2d(dtype=wp.float32),
    pos_adr: wp.int32,
    orient_adr: wp.int32,
    safety_adr: wp.int32,
    t1_adr: wp.int32,
    t2_adr: wp.int32,
    t3_adr: wp.int32,
    eez_adr: wp.int32,
    eepos_adr: wp.int32,
    blk_adr: wp.int32,
    goal_adr: wp.int32,
    w_pos: wp.float32,
    w_orient: wp.float32,
    w_attract: wp.float32,
    w_align: wp.float32,
    w_safety: wp.float32,
    safety_thresh: wp.float32,
    w_ee_orient: wp.float32,
    w_ee_height: wp.float32,
    ee_z_target: wp.float32,
    qpos: wp.array2d(dtype=wp.float32),
    arm_qposadr: wp.array(dtype=wp.int32),
    q_home: wp.array(dtype=wp.float32),
    w_arm_home: wp.float32,
    scale: wp.float32,
    cost: wp.array(dtype=wp.float32),
):
    """Block pose error + EE->block shaping + safety + EE/arm posture.

    ``w_align`` keeps the block between the EE and the goal, so the EE
    approaches from the pushing side instead of the nearest side.

    ``w_ee_orient``/``w_ee_height`` regularize the EE to a point-down stick at
    push height (posture task-space IK enforces for free, but joint-space
    sampling must be told about). ``w_arm_home`` anchors the arm's redundant
    null-space to the home config so joint-space sampling can't drift into
    joint limits / singularities.
    """
    i = wp.tid()

    px = sensordata[i, pos_adr + 0]
    py = sensordata[i, pos_adr + 1]
    pz = sensordata[i, pos_adr + 2]
    pos_cost = wp.sqrt(px * px + py * py + pz * pz)

    qw = sensordata[i, orient_adr + 0]
    qx = sensordata[i, orient_adr + 1]
    qy = sensordata[i, orient_adr + 2]
    qz = sensordata[i, orient_adr + 3]
    qv = wp.sqrt(qx * qx + qy * qy + qz * qz)
    orient_cost = 2.0 * wp.atan2(qv, wp.abs(qw))

    b1x = sensordata[i, t1_adr + 0]
    b1y = sensordata[i, t1_adr + 1]
    b1z = sensordata[i, t1_adr + 2]
    b2x = sensordata[i, t2_adr + 0]
    b2y = sensordata[i, t2_adr + 1]
    b2z = sensordata[i, t2_adr + 2]
    b3x = sensordata[i, t3_adr + 0]
    b3y = sensordata[i, t3_adr + 1]
    b3z = sensordata[i, t3_adr + 2]
    attract = (
        wp.sqrt(b1x * b1x + b1y * b1y + b1z * b1z)
        + wp.sqrt(b2x * b2x + b2y * b2y + b2z * b2z)
        + wp.sqrt(b3x * b3x + b3y * b3y + b3z * b3z)
    )

    # Push-align: cos(alpha) + 1 for alpha between the EE->block and
    # goal->block vectors (planar). Zero when the block sits between the EE
    # and the goal, 2 when the EE is on the goal side. Both guards matter:
    # at the goal there is no push direction to align with.
    rx = sensordata[i, eepos_adr + 0] - sensordata[i, blk_adr + 0]
    ry = sensordata[i, eepos_adr + 1] - sensordata[i, blk_adr + 1]
    gx = sensordata[i, goal_adr + 0] - sensordata[i, blk_adr + 0]
    gy = sensordata[i, goal_adr + 1] - sensordata[i, blk_adr + 1]
    rn = wp.sqrt(rx * rx + ry * ry)
    gn = wp.sqrt(gx * gx + gy * gy)
    align = wp.float32(0.0)
    if rn > 1.0e-6 and gn > 1.0e-6:
        align = (rx * gx + ry * gy) / (rn * gn) + 1.0

    sx = sensordata[i, safety_adr + 0]
    sy = sensordata[i, safety_adr + 1]
    sz = sensordata[i, safety_adr + 2]
    safety = 0.0
    if wp.sqrt(sx * sx + sy * sy + sz * sz) > safety_thresh:
        safety = 1.0

    # EE point-down: world z-axis of the stick vs (0, 0, -1). Distance grows
    # with tilt (roll/pitch) but is invariant to spin about the stick axis.
    zx = sensordata[i, eez_adr + 0]
    zy = sensordata[i, eez_adr + 1]
    zz = sensordata[i, eez_adr + 2]
    ee_orient_cost = wp.sqrt(zx * zx + zy * zy + (zz + 1.0) * (zz + 1.0))

    # EE height: keep the stick tip at the push height above the table.
    ee_z = sensordata[i, eepos_adr + 2]
    ee_height_cost = wp.abs(ee_z - ee_z_target)

    # Arm-home posture: pull the 7 arm joints back toward the home config so
    # the redundant null-space can't drift into joint limits / singularities.
    arm_sq = wp.float32(0.0)
    for k in range(7):
        dq = qpos[i, arm_qposadr[k]] - q_home[k]
        arm_sq += dq * dq
    arm_home_cost = wp.sqrt(arm_sq)

    running = (
        w_pos * pos_cost
        + w_orient * orient_cost
        + w_attract * attract
        + w_align * align
        + w_safety * safety
        + w_ee_orient * ee_orient_cost
        + w_ee_height * ee_height_cost
        + w_arm_home * arm_home_cost
    )
    cost[i] = cost[i] + scale * running


class _Fr3Cost:
    """Launches :func:`_fr3_cost_kernel` with cached addresses and weights."""

    def __init__(
        self,
        adrs: tuple[int, int, int, int, int, int, int, int, int, int],
        w_pos: float,
        w_orient: float,
        w_attract: float,
        w_align: float,
        w_safety: float,
        safety_thresh: float,
        w_ee_orient: float,
        w_ee_height: float,
        ee_z_target: float,
        arm_qposadr: np.ndarray,
        q_home: np.ndarray,
        w_arm_home: float,
    ) -> None:
        self.adrs = tuple(int(a) for a in adrs)
        self.w_pos = float(w_pos)
        self.w_orient = float(w_orient)
        self.w_attract = float(w_attract)
        self.w_align = float(w_align)
        self.w_safety = float(w_safety)
        self.safety_thresh = float(safety_thresh)
        self.w_ee_orient = float(w_ee_orient)
        self.w_ee_height = float(w_ee_height)
        self.ee_z_target = float(ee_z_target)
        self.w_arm_home = float(w_arm_home)
        # Allocated at build time (under the engine's ScopedDevice), never in
        # accumulate() which runs during graph capture.
        self._arm_qposadr = wp.array(
            np.asarray(arm_qposadr, dtype=np.int32), dtype=wp.int32
        )
        self._q_home = wp.array(
            np.asarray(q_home, dtype=np.float32), dtype=wp.float32
        )

    def accumulate(self, d, cost: wp.array, scale: float) -> None:
        """Add ``scale * running_cost(d)`` into ``cost`` (per world)."""
        pa, oa, sa, t1, t2, t3, eez, eepos, blk, goal = self.adrs
        wp.launch(
            _fr3_cost_kernel,
            dim=cost.shape[0],
            inputs=[
                d.sensordata,
                wp.int32(pa),
                wp.int32(oa),
                wp.int32(sa),
                wp.int32(t1),
                wp.int32(t2),
                wp.int32(t3),
                wp.int32(eez),
                wp.int32(eepos),
                wp.int32(blk),
                wp.int32(goal),
                wp.float32(self.w_pos),
                wp.float32(self.w_orient),
                wp.float32(self.w_attract),
                wp.float32(self.w_align),
                wp.float32(self.w_safety),
                wp.float32(self.safety_thresh),
                wp.float32(self.w_ee_orient),
                wp.float32(self.w_ee_height),
                wp.float32(self.ee_z_target),
                d.qpos,
                self._arm_qposadr,
                self._q_home,
                wp.float32(self.w_arm_home),
                wp.float32(scale),
                cost,
            ],
        )


@wp.kernel
def _fr3_ik_kernel(
    body_parentid: wp.array(dtype=wp.int32),
    body_rootid: wp.array(dtype=wp.int32),
    dof_bodyid: wp.array(dtype=wp.int32),
    body_isdofancestor: wp.array2d(dtype=wp.int32),
    subtree_com: wp.array2d(dtype=wp.vec3),
    cdof: wp.array2d(dtype=wp.spatial_vector),
    xpos: wp.array2d(dtype=wp.vec3),
    xquat: wp.array2d(dtype=wp.quat),
    arm_dof: wp.array(dtype=wp.int32),
    ee_body: wp.int32,
    goal_quat: wp.quat,
    z_target: wp.float32,
    lam: wp.float32,
    controls: wp.array3d(dtype=wp.float32),
    t: wp.int32,
    ctrl: wp.array2d(dtype=wp.float32),
):
    """Damped-LS task-space IK per world: 2-D twist -> 7 joint velocities.

    Device twin of ``control_map_host``.
    """
    w = wp.tid()
    point = xpos[w, ee_body]
    ee_quat = xquat[w, ee_body]

    jmat = ee_jacobian_dof(
        body_parentid,
        body_rootid,
        dof_bodyid,
        body_isdofancestor,
        subtree_com,
        cdof,
        point,
        ee_body,
        arm_dof,
        w,
    )

    # Twist: [vx, vy, z-regulation, world-frame point-down orientation error].
    e_world = rot_vec_quat(quat_sub(goal_quat, ee_quat), ee_quat)
    b = vec6()
    b[0] = controls[w, t, 0]
    b[1] = controls[w, t, 1]
    b[2] = z_target - point[2]
    b[3] = e_world[0]
    b[4] = e_world[1]
    b[5] = e_world[2]

    dq = damped_ls(jmat, b, lam)
    for c in range(7):
        ctrl[w, c] = dq[c]


class _Fr3IK:
    """Applier: launches :func:`_fr3_ik_kernel` to write ``d.ctrl`` per step."""

    def __init__(
        self,
        arm_dofadr: np.ndarray,
        ee_body: int,
        goal_quat: np.ndarray,
        z_target: float,
        lam: float,
    ) -> None:
        # Allocated here (build time, under the engine's ScopedDevice) -- never
        # inside apply(), which runs during graph capture.
        self._arm_dof = wp.array(
            np.asarray(arm_dofadr, dtype=np.int32), dtype=wp.int32
        )
        self.ee_body = int(ee_body)
        g = [float(x) for x in goal_quat]  # [w, x, y, z]
        self._goal = wp.quat(g[0], g[1], g[2], g[3])
        self.z_target = float(z_target)
        self.lam = float(lam)

    def apply(self, m, d, controls: wp.array, t: int) -> None:
        """Write ``d.ctrl`` from the staged 2-D twist at step ``t``."""
        wp.launch(
            _fr3_ik_kernel,
            dim=controls.shape[0],
            inputs=[
                m.body_parentid,
                m.body_rootid,
                m.dof_bodyid,
                m.body_isdofancestor,
                d.subtree_com,
                d.cdof,
                d.xpos,
                d.xquat,
                self._arm_dof,
                wp.int32(self.ee_body),
                self._goal,
                wp.float32(self.z_target),
                wp.float32(self.lam),
                controls,
                wp.int32(t),
                d.ctrl,
            ],
        )


def _build_model(
    path,
    manipulation_type: str,
    shape: str,
    scale: float,
    bite: float,
    goal_xy: tuple[float, float] | None = None,
) -> tuple[mujoco.MjModel, mujoco.MjSpec]:
    """Graft ``shape`` onto block/goal and place them at their resting z.

    For ``"joint"``, also anchors the 3 block joints at the shape's COM.
    For ``"free"``, rewrites the "home" keyframe's block qpos to match
    (it's authored for the original T and would otherwise be stale).
    Returns ``(model, spec)``: the spec is kept so the viewer can extend
    a copy of it (e.g. mocap ghost bodies) without redoing composition.
    """
    spec = mujoco.MjSpec.from_file(str(path))
    attach_ee(spec, "pusher")  # pusher EE onto the shared arm's attach site

    shape_body = graft_shape(
        spec, "block", shape, scale, default=spec.find_default("tblock")
    )
    graft_shape(
        spec,
        "goal",
        shape,
        scale,
        default=spec.find_default("goal"),
        include_sites=False,
    )

    ground_body = spec.body("ground")
    ground_geom = spec.geom("ground")
    table_top_z = ground_body.pos[2] + ground_geom.pos[2] + ground_geom.size[2]
    bite_effective = bite if manipulation_type == "joint" else 0.0
    z = table_top_z - shape_min_z(shape_body, scale) - bite_effective

    block_pos = list(spec.body("block").pos)
    block_pos[2] = z
    spec.body("block").pos = block_pos
    goal_pos = list(spec.body("goal").pos)
    goal_pos[2] = z
    # The goal is a world mocap, and goal_mocap_pose drifts as an offset from
    # wherever it starts, so this moves the drift centre with it. None keeps
    # the scene's authored position.
    if goal_xy is not None:
        goal_pos[0], goal_pos[1] = float(goal_xy[0]), float(goal_xy[1])
    spec.body("goal").pos = goal_pos

    if manipulation_type == "joint":
        com_x, com_y = shape_com_xy(shape_body, scale)
        anchor = [com_x, com_y, 0.0]
        spec.joint("block_x").pos = anchor
        spec.joint("block_y").pos = anchor
        spec.joint("block_yaw").pos = anchor
    else:
        key = spec.key("home")
        qpos = list(key.qpos)
        qpos[0:3] = block_pos
        key.qpos = qpos

    return spec.compile(), spec


class PushFr3(Task):
    """Push driven by an FR3 arm (task-space or joint-space sampling)."""

    def __init__(
        self,
        *,
        trace_sites: Sequence[str] | None = ("ee_site", "attractor_site1"),
        sampling_space: Literal["task", "joint"] = "task",
        manipulation_type: Literal["free", "joint"] = "joint",
        shape: str = "t",
        scale: float = 1.0,
        goal_xy: tuple[float, float] | None = None,
        bite: float = 0.0001,
        max_lin_vel: float = 0.15,
        w_pos: float = 35.0,
        w_orient: float = 3.0,
        w_attract: float = 0.01,
        w_align: float = 0.0,
        w_safety: float = 1.0,
        safety_thresh: float = 0.5,
        w_ee_orient: float = 1.0,
        w_ee_height: float = 15.0,
        w_arm_home: float = 2.0,
        terminal_scale: float = 10.0,
        model_config: ModelConfig | None = None,
        goal_drift: GoalDrift | None = None,
    ) -> None:
        """Compose ``shape`` at ``scale`` onto the block/goal bodies.

        Args:
            trace_sites: Sites to trace for visualization.
            sampling_space: ``"task"`` (nu=2, EE x/y vel, IK-mapped) or
                ``"joint"`` (nu=7, joint velocities sampled directly).
            manipulation_type: ``"free"`` (6-DOF block) or ``"joint"``
                (3-DOF slide/slide/hinge block).
            shape: Name under ``models/shapes/`` (see
                ``bampc.task.common.shapes.list_shapes``).
            scale: Uniform size scale factor.
            goal_xy: World ``(x, y)`` of the goal mocap; ``None`` keeps the
                scene's authored position. ``goal_drift`` is an offset from
                wherever the goal starts, so this moves the drift centre too.
                Supplied by the scenario bank in sweeps.
            bite: Resting penetration depth for ``manipulation_type=
                "joint"`` (no vertical DOF, so this sets its fixed height);
                ignored for ``"free"``, which just rests via gravity.
            max_lin_vel: Task-space velocity bound (m/s) when sampling in
                task space.
            w_pos: Weight on the block->goal position error.
            w_orient: Weight on the block->goal geodesic orientation error.
            w_attract: Weight on the EE->attractor-site term (anti-local-
                minima contact guidance).
            w_align: Weight on the push-align term ``cos(alpha) + 1``, alpha
                being the angle between the EE->block and goal->block vectors.
                Zero when the block sits between EE and goal, so the EE
                circles to the pushing side rather than the nearest side.
            w_safety: Weight on the safety-zone step penalty.
            safety_thresh: EE-to-goal distance (m) beyond which the safety
                penalty fires.
            terminal_scale: Terminal cost = ``terminal_scale`` x the running
                pos+orient terms (posture/attract/safety are running-only).
            w_ee_orient: Weight on the EE point-down penalty (world z-axis
                vs ``(0, 0, -1)``). Stabilizes joint-space sampling, where no
                IK keeps the stick vertical; ~0 under task-space IK.
            w_ee_height: Weight on the EE push-height penalty (|z - push
                height|). Same rationale as ``w_ee_orient``.
            w_arm_home: Weight on the arm-home posture penalty (``||q_arm -
                q_home||``), anchoring the redundant null-space against drift
                into limits or singularities. No effect without a "home"
                keyframe; set 0 under task-space sampling, where the IK
                already bounds the drift.
            model_config: Solver/integrator overrides layered on
                :data:`_BASELINE`.
            goal_drift: Optional moving-goal spec (``joint`` or ``free``
                block). Unvalidated: a large radius can push the goal out
                of the arm's reach, which is a legitimate setup here.
        """
        self.sampling_space = sampling_space
        self.manipulation_type = manipulation_type

        if sampling_space == "task":
            nu: int | None = 2
            ctrl_limits = {
                "u_min": np.array([-max_lin_vel, -max_lin_vel]),
                "u_max": np.array([max_lin_vel, max_lin_vel]),
            }
        else:
            nu, ctrl_limits = None, None

        if manipulation_type not in MANIPULATION_TYPES:
            raise ValueError(f"Unknown manipulation_type: {manipulation_type}")

        path = MODELS_DIR / "push_fr3" / manipulation_type / "scene.xml"
        mj_model, mj_spec = _build_model(
            path, manipulation_type, shape, scale, bite, goal_xy
        )
        self.goal_xy = goal_xy

        super().__init__(
            mj_model,
            mj_spec=mj_spec,
            trace_sites=trace_sites,
            nu=nu,
            ctrl_limits=ctrl_limits,
            endpoint_body="block",
            model_config=_BASELINE.merged_with(model_config),
            contact_budget=push_fr3_contact_budget(shape),
            goal_drift=goal_drift,
        )

        # Cost-sensor addresses (non-contiguous; cache from the model),
        # grouped by the cost term that reads them.
        self._adr_pos, self._adr_orient, self._adr_safety = (
            self._sensor_adr("position"),
            self._sensor_adr("orientation"),
            self._sensor_adr("safety"),
        )
        self._adr_t1, self._adr_t2, self._adr_t3 = (
            self._sensor_adr("ee_t1"),
            self._sensor_adr("ee_t2"),
            self._sensor_adr("ee_t3"),
        )
        self._adr_ee_zaxis, self._adr_ee_pos = (
            self._sensor_adr("ee_zaxis"),
            self._sensor_adr("ee_position_world"),
        )
        # World-frame block/goal positions, for the push-align term.
        self._adr_blk, self._adr_goal = (
            self._sensor_adr("position_world"),
            self._sensor_adr("goal_position_world"),
        )

        # Running-cost weights, shared by kernel
        # + host twin. All constructor args so each scenario can tune them.
        self.w_pos = float(w_pos)
        self.w_orient = float(w_orient)
        self.w_attract = float(w_attract)
        self.w_align = float(w_align)
        self.w_safety = float(w_safety)
        self.safety_thresh = float(safety_thresh)
        # EE-posture regularization (paper-inspired Cee_align); zero to disable.
        self.w_ee_orient, self.w_ee_height = (
            float(w_ee_orient), float(w_ee_height)
        )
        # Arm-home posture: anchors the redundant null-space (anti-drift).
        self.w_arm_home = float(w_arm_home)
        self.terminal_scale = float(terminal_scale)  # terminal = scale*running

        # Block qpos addresses, for set_initial_state. "free" rides a single
        # free joint (xyz + quat); "joint" rides slide/slide/hinge, whose
        # qpos is a displacement from the block's XML anchor --
        # _block_anchor_xy converts set_initial_state's world-frame
        # block_xy into that displacement so both manipulation types take
        # world-frame coordinates.
        if manipulation_type == "free":
            self._block_free_adr = self._joint_qposadr("block")
        else:
            self._block_x_adr = self._joint_qposadr("block_x")
            self._block_y_adr = self._joint_qposadr("block_y")
            self._block_yaw_adr = self._joint_qposadr("block_yaw")
            block_bid = mj_model.body("block").id
            self._block_anchor_xy = mj_model.body_pos[block_bid, :2].copy()

        # Task-space IK bookkeeping (host control map + arm placement). Cached
        # regardless of sampling_space: set_initial_state(ee_pos=...) needs the
        # arm IK even when sampling joint velocities.
        self.ee_body_id = int(mj_model.body("ee_frame").id)
        arm_joints = [f"fr3_joint{i}" for i in range(1, 8)]
        jids = [mj_model.joint(n).id for n in arm_joints]
        self.arm_qposadr = mj_model.jnt_qposadr[jids].astype(int)
        self.arm_dofadr = mj_model.jnt_dofadr[jids].astype(int)
        self.joint_limits = mj_model.jnt_range[jids]  # (7, 2)
        self._jacp = np.zeros((3, mj_model.nv))  # mj_jacBody scratch
        self._jacr = np.zeros((3, mj_model.nv))
        # Point-down EE goal orientation [w, x, y, z] and table-height z target.
        self.goal_quat_ee = np.array([0.0, 0.7071, 0.7071, 0.0])
        self.ee_z_target = 0.039
        self.ik_damping = 1e-3
        kid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_KEY, "home")
        self.q_home = (
            mj_model.key_qpos[kid][self.arm_qposadr] if kid != -1 else None
        )

    def _sensor_adr(self, name: str) -> int:
        """Address of a named sensor's first scalar in ``sensordata``."""
        sid = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_SENSOR, name
        )
        return int(self.mj_model.sensor_adr[sid])

    def _joint_qposadr(self, name: str) -> int:
        """First qpos address of a named joint."""
        return int(self.mj_model.jnt_qposadr[self.mj_model.joint(name).id])

    @property
    def object_pose_qpos(self) -> ObjectPose:
        """Block pose layout, which differs by ``manipulation_type``."""
        if self.manipulation_type == "free":
            return ObjectPose(kind="free", adr=self._block_free_adr)
        return ObjectPose(
            kind="planar",
            x_adr=self._block_x_adr,
            y_adr=self._block_y_adr,
            yaw_adr=self._block_yaw_adr,
        )

    @property
    def contact_probes(self) -> tuple[ContactProbe, ...]:
        """Pusher-block and block-table contact.

        The pusher side covers both the ``ee`` sphere and the arm box above
        it, since either can be what actually touches the block.
        """
        block = body_geom_ids(self.mj_model, "block")
        return (
            ContactProbe(
                name="pusher_block",
                geoms_a=body_geom_ids(self.mj_model, "ee_frame", "pusher"),
                geoms_b=block,
                bit=0,
            ),
            ContactProbe(
                name="block_table",
                geoms_a=block,
                geoms_b=body_geom_ids(self.mj_model, "ground"),
                bit=1,
            ),
        )

    @property
    def _cost_adrs(
        self,
    ) -> tuple[int, int, int, int, int, int, int, int, int, int]:
        return (
            self._adr_pos,
            self._adr_orient,
            self._adr_safety,
            self._adr_t1,
            self._adr_t2,
            self._adr_t3,
            self._adr_ee_zaxis,
            self._adr_ee_pos,
            self._adr_blk,
            self._adr_goal,
        )

    def _arm_home_ref(self) -> tuple[np.ndarray, float]:
        """Home reference + effective weight (0 with no 'home' keyframe)."""
        if self.q_home is None:
            return np.zeros(7), 0.0
        return np.asarray(self.q_home), self.w_arm_home

    def build_cost_kernel(self) -> _Fr3Cost:  # noqa: D102
        q_home, w_arm_home = self._arm_home_ref()
        return _Fr3Cost(
            self._cost_adrs,
            self.w_pos,
            self.w_orient,
            self.w_attract,
            self.w_align,
            self.w_safety,
            self.safety_thresh,
            self.w_ee_orient,
            self.w_ee_height,
            self.ee_z_target,
            self.arm_qposadr,
            q_home,
            w_arm_home,
        )

    def build_terminal_cost_kernel(self) -> _Fr3Cost:  # noqa: D102
        # Same expression as the running cost, scaled by terminal_scale.
        # Posture (like attract/safety) is a per-step regularizer, not a
        # terminal objective, so it is zeroed here.
        s = self.terminal_scale
        q_home, _ = self._arm_home_ref()
        return _Fr3Cost(
            self._cost_adrs,
            s * self.w_pos,
            s * self.w_orient,
            0 * self.w_attract,
            0 * self.w_align,
            0 * self.w_safety,
            self.safety_thresh,
            0.0,
            0.0,
            self.ee_z_target,
            self.arm_qposadr,
            q_home,
            0.0,
        )

    def cost_components(self, mj_data: mujoco.MjData) -> dict[str, float]:
        """Host twin of the running cost, split per weighted term."""
        s = mj_data.sensordata
        p = s[self._adr_pos : self._adr_pos + 3]
        pos_cost = float(np.linalg.norm(p))

        q = s[self._adr_orient : self._adr_orient + 4]
        qv = float(np.linalg.norm(q[1:4]))
        orient_cost = 2.0 * math.atan2(qv, abs(float(q[0])))

        attract = 0.0
        for adr in (self._adr_t1, self._adr_t2, self._adr_t3):
            attract += float(np.linalg.norm(s[adr : adr + 3]))

        r = s[self._adr_ee_pos : self._adr_ee_pos + 2] - (
            s[self._adr_blk : self._adr_blk + 2]
        )
        g = s[self._adr_goal : self._adr_goal + 2] - (
            s[self._adr_blk : self._adr_blk + 2]
        )
        rn, gn = float(np.linalg.norm(r)), float(np.linalg.norm(g))
        align = 0.0
        if rn > 1e-6 and gn > 1e-6:
            align = float(np.dot(r, g)) / (rn * gn) + 1.0

        sd = s[self._adr_safety : self._adr_safety + 3]
        safety = 1.0 if float(np.linalg.norm(sd)) > self.safety_thresh else 0.0

        z = s[self._adr_ee_zaxis : self._adr_ee_zaxis + 3]
        ee_orient_cost = float(
            math.sqrt(z[0] ** 2 + z[1] ** 2 + (z[2] + 1.0) ** 2)
        )
        ee_z = float(s[self._adr_ee_pos + 2])
        ee_height_cost = abs(ee_z - self.ee_z_target)

        if self.q_home is None:
            arm_home_cost = 0.0
        else:
            dq = mj_data.qpos[self.arm_qposadr] - np.asarray(self.q_home)
            arm_home_cost = float(np.linalg.norm(dq))

        pos = self.w_pos * pos_cost
        orient = self.w_orient * orient_cost
        att = self.w_attract * attract
        alg = self.w_align * align
        saf = self.w_safety * safety
        ee_orient = self.w_ee_orient * ee_orient_cost
        ee_height = self.w_ee_height * ee_height_cost
        arm_home = self.w_arm_home * arm_home_cost
        return {
            "pos": pos,
            "orient": orient,
            "attract": att,
            "align": alg,
            "safety": saf,
            "ee_orient": ee_orient,
            "ee_height": ee_height,
            "arm_home": arm_home,
            "total": (
                pos
                + orient
                + att
                + alg
                + saf
                + ee_orient
                + ee_height
                + arm_home
            ),
        }

    def pose_error(self, mj_data: mujoco.MjData) -> tuple[float, float]:
        """Raw (unweighted) linear (m) and rotational (rad) block-to-goal error.

        Same sensors as :meth:`cost_components`'s ``pos``/``orient`` terms,
        before the ``w_pos``/``w_orient`` weighting. Weighted cost is the
        wrong unit for logging/analysis (a component can be traded against
        another without the total moving).
        """
        s = mj_data.sensordata
        p = s[self._adr_pos : self._adr_pos + 3]
        pos_err = float(np.linalg.norm(p))
        q = s[self._adr_orient : self._adr_orient + 4]
        qv = float(np.linalg.norm(q[1:4]))
        rot_err = 2.0 * math.atan2(qv, abs(float(q[0])))
        return pos_err, rot_err

    def running_cost_reference(
        self, mj_data: mujoco.MjData, u: object = None
    ) -> float:
        """Eager numpy running cost (sum of :meth:`cost_components`)."""
        return self.cost_components(mj_data)["total"]

    def terminal_cost_reference(self, mj_data: mujoco.MjData) -> float:
        """Eager numpy terminal cost: the running cost, scaled uniformly.

        Not the terminal kernel's expression --
        :meth:`build_terminal_cost_kernel` keeps only ``w_pos``/``w_orient``
        and zeroes the per-step regularizers, which this scales instead.
        """
        return self.terminal_scale * self.running_cost_reference(mj_data)

    def build_control_map_kernel(self) -> Any | None:  # noqa: D102
        if self.sampling_space == "joint":
            return None  # joint velocities sampled directly -> write ctrl
        return _Fr3IK(
            self.arm_dofadr,
            self.ee_body_id,
            self.goal_quat_ee,
            self.ee_z_target,
            self.ik_damping,
        )

    def control_map_host(
        self, mj_data: mujoco.MjData, u: np.ndarray
    ) -> np.ndarray:
        """Map a control to the 7 joint-velocity actuators (host).

        Joint sampling is identity. Task sampling maps the 2-D EE twist to
        joint velocities by a damped least-squares IK on the EE Jacobian,
        regulating z to table height and orientation to point down.
        """
        if self.sampling_space == "joint":
            return np.asarray(u)

        m = self.mj_model
        mujoco.mj_jacBody(m, mj_data, self._jacp, self._jacr, self.ee_body_id)
        jac = np.vstack([self._jacp, self._jacr])[:, self.arm_dofadr]  # (6, 7)

        ee_pos = mj_data.xpos[self.ee_body_id]
        ee_quat = mj_data.xquat[self.ee_body_id]
        e_rot = self._orientation_error_world(ee_quat)
        twist = np.array(
            [
                u[0],
                u[1],
                self.ee_z_target - ee_pos[2],  # z-height regulation
                e_rot[0],
                e_rot[1],
                e_rot[2],
            ]
        )
        return damped_ls_host(jac, twist, self.ik_damping)

    def _orientation_error_world(self, ee_quat: np.ndarray) -> np.ndarray:
        """World-frame rotation vector from ``ee_quat`` to the goal."""
        res = np.zeros(3)
        mujoco.mju_subQuat(res, self.goal_quat_ee, ee_quat)
        return quat_rotate(ee_quat, res)

    def _solve_ik(
        self,
        target_pos: np.ndarray,
        target_quat: np.ndarray,
        q_seed: np.ndarray | None = None,
        max_iters: int = 200,
        pos_tol: float = 1e-4,
        rot_tol: float = 1e-3,
        step_scale: float = 0.5,
    ) -> tuple[np.ndarray, bool]:
        """Damped least-squares IK for the EE pose; see `solve_task_pose_ik`."""
        return solve_task_pose_ik(
            self,
            target_pos,
            target_quat,
            q_seed,
            max_iters=max_iters,
            pos_tol=pos_tol,
            rot_tol=rot_tol,
            step_scale=step_scale,
        )

    def set_initial_state(
        self,
        mj_data: mujoco.MjData,
        *,
        ee_pos: np.ndarray | None = None,
        ee_quat: np.ndarray | None = None,
        block_xy: np.ndarray | None = None,
        block_yaw: float | None = None,
    ) -> None:
        """Overlay a start pose on ``mj_data`` (host), then forward.

        Writes only the block pose for the active ``manipulation_type``;
        pass only what you want to override.

        Args:
            mj_data: MjData to mutate; caller has loaded a keyframe first.
            ee_pos: Desired EE position (m, world); triggers IK.
            ee_quat: Desired EE quaternion [w, x, y, z].
            block_xy: Desired block planar position (x, y) in world
                coordinates, meters -- the same convention in both
                ``manipulation_type``s, even though ``"joint"``'s slide
                joints internally store a displacement from the block's
                XML anchor.
            block_yaw: Desired block yaw around world Z, radians.
        """
        if ee_pos is not None:
            ee_pos = np.asarray(ee_pos, dtype=np.float64)
            tgt_quat = (
                self.goal_quat_ee
                if ee_quat is None
                else np.asarray(ee_quat, dtype=np.float64)
            )
            q_sol, converged = self._solve_ik(ee_pos, tgt_quat)
            if not converged:
                print(
                    "WARNING: PushFr3 IK did not converge for "
                    f"ee_pos={ee_pos.tolist()}; using best-effort solution."
                )
            mj_data.qpos[self.arm_qposadr] = q_sol
        elif ee_quat is not None:
            raise ValueError("ee_quat given without ee_pos; specify ee_pos too")

        if self.manipulation_type == "free":
            adr = self._block_free_adr
            if block_xy is not None:
                mj_data.qpos[adr : adr + 2] = np.asarray(
                    block_xy, dtype=np.float64
                )
            if block_yaw is not None:
                half = 0.5 * float(block_yaw)
                mj_data.qpos[adr + 3 : adr + 7] = [
                    np.cos(half), 0.0, 0.0, np.sin(half)
                ]
        else:  # "joint": block_x/block_y slides + block_yaw hinge
            if block_xy is not None:
                # Slide-joint qpos is a displacement from the block's XML
                # anchor, not an absolute position -- subtract it so
                # block_xy means the same world-frame coordinate as the
                # "free" branch.
                offset = (
                    np.asarray(block_xy, dtype=np.float64)
                    - self._block_anchor_xy
                )
                mj_data.qpos[self._block_x_adr] = offset[0]
                mj_data.qpos[self._block_y_adr] = offset[1]
            if block_yaw is not None:
                mj_data.qpos[self._block_yaw_adr] = float(block_yaw)

        mujoco.mj_forward(self.mj_model, mj_data)
