"""Balance with an FR3 arm: tilt the EE plate to slide a block to a goal.

The FR3 holds a plate on its end effector (models/balance_fr3 + the plate EE)
and tilts it -- via its 7 joints -- to slide a free 6-DOF block to a goal that
rides the plate. It reuses Push-FR3's ``sampling_space`` axis:

* ``"joint"`` samples the 7 joint velocities directly (``nu=7``).
* ``"task"`` samples a 2-D plate-tilt command (roll/pitch angular velocity)
  mapped each rollout step to joint velocities by the shared damped-LS IK
  (``bampc.task.common.ik``), which regulates EE position + yaw and
  leaves roll/pitch to the command.

Unlike Push-FR3 there is no ``manipulation_type`` axis -- the object is always
the 6-DOF free block. The cost is block->goal position error (this is
balancing), plus an arm-home posture term that stabilizes joint-space sampling
(keeps the redundant arm near the level home). The goal rides the plate and the
block is placed on the plate exactly as in ``bampc.task.balance``.

``contact_budget`` uses ``balance_fr3_contact_budget``, derived (not reused
from the planar toy ``Balance``, which under-sized ``nj_per_env``) with
``scripts/probing/probe_task.py``.
"""

from __future__ import annotations

import math
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
    quat_mul,
)
from bampc.task.common.cholesky import vec6
from bampc.task.common.ee import attach_ee
from bampc.task.common.ik import (
    damped_ls,
    damped_ls_host,
    ee_jacobian_dof,
    quat_rotate,
)
from bampc.task.common.shapes import (
    balance_fr3_contact_budget,
    graft_shape,
    shape_max_xy_radius,
    shape_min_z,
)

# Fallback solver options, used only when the caller names none. The tuned
# values live with the caller -- configs/numerics/balance_fr3.yaml. Same family
# as Push-FR3's free block (elliptic cone, implicit integrator, fine timestep).
_BASELINE = ModelConfig(
    timestep=0.01,
    solver="Newton",
    integrator="implicitfast",
    cone="elliptic",
    jacobian="dense",
    impratio=1.0,
    iterations=16,
    ls_iterations=20,
    eulerdamp=True,
    warmstart=True,
)


@wp.kernel
def _balance_fr3_cost_kernel(
    sensordata: wp.array2d(dtype=wp.float32),
    ctrl: wp.array2d(dtype=wp.float32),
    pos_adr: wp.int32,
    orient_adr: wp.int32,
    w_pos: wp.float32,
    w_orient: wp.float32,
    w_ctrl: wp.float32,
    qpos: wp.array2d(dtype=wp.float32),
    arm_qposadr: wp.array(dtype=wp.int32),
    q_home: wp.array(dtype=wp.float32),
    w_arm_home: wp.float32,
    scale: wp.float32,
    cost: wp.array(dtype=wp.float32),
):
    """Block->goal pose error + control effort + arm-home posture.

    ``w_arm_home`` anchors the 7 arm joints to the level home config so
    joint-space sampling can't drift the plate far from level / into limits.
    """
    i = wp.tid()

    ex = sensordata[i, pos_adr + 0]
    ey = sensordata[i, pos_adr + 1]
    ez = sensordata[i, pos_adr + 2]
    pos_cost = ex * ex + ey * ey + ez * ez

    qw = sensordata[i, orient_adr + 0]
    qx = sensordata[i, orient_adr + 1]
    qy = sensordata[i, orient_adr + 2]
    qz = sensordata[i, orient_adr + 3]
    qv = wp.sqrt(qx * qx + qy * qy + qz * qz)
    orient_cost = 2.0 * wp.atan2(qv, wp.abs(qw))

    ctrl_cost = wp.float32(0.0)
    for k in range(ctrl.shape[1]):
        ctrl_cost += ctrl[i, k] * ctrl[i, k]

    arm_sq = wp.float32(0.0)
    for k in range(7):
        dq = qpos[i, arm_qposadr[k]] - q_home[k]
        arm_sq += dq * dq
    arm_home_cost = wp.sqrt(arm_sq)

    running = (
        w_pos * pos_cost
        + w_orient * orient_cost
        + w_ctrl * ctrl_cost
        + w_arm_home * arm_home_cost
    )
    cost[i] = cost[i] + scale * running


class _BalanceFr3Cost:
    """Launches :func:`_balance_fr3_cost_kernel` with cached addrs/weights."""

    def __init__(
        self,
        pos_adr: int,
        orient_adr: int,
        w_pos: float,
        w_orient: float,
        w_ctrl: float,
        arm_qposadr: np.ndarray,
        q_home: np.ndarray,
        w_arm_home: float,
    ) -> None:
        self.pos_adr = int(pos_adr)
        self.orient_adr = int(orient_adr)
        self.w_pos = float(w_pos)
        self.w_orient = float(w_orient)
        self.w_ctrl = float(w_ctrl)
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
        wp.launch(
            _balance_fr3_cost_kernel,
            dim=cost.shape[0],
            inputs=[
                d.sensordata,
                d.ctrl,
                wp.int32(self.pos_adr),
                wp.int32(self.orient_adr),
                wp.float32(self.w_pos),
                wp.float32(self.w_orient),
                wp.float32(self.w_ctrl),
                d.qpos,
                self._arm_qposadr,
                self._q_home,
                wp.float32(self.w_arm_home),
                wp.float32(scale),
                cost,
            ],
        )


@wp.kernel
def _balance_fr3_ik_kernel(
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
    target_pos: wp.vec3,
    level_quat: wp.quat,
    k_pos: wp.float32,
    k_yaw: wp.float32,
    lam: wp.float32,
    controls: wp.array3d(dtype=wp.float32),
    t: wp.int32,
    ctrl: wp.array2d(dtype=wp.float32),
):
    """Damped-LS task-space IK per world: 2-D plate tilt -> 7 joint velocities.

    Commands roll/pitch (``b[3:5]``); regulates EE position (``b[0:3]``) and yaw
    (``b[5]``, z-component of the world orientation error to level). Device twin
    of :meth:`BalanceFr3.control_map_host`.
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

    e_world = rot_vec_quat(quat_sub(level_quat, ee_quat), ee_quat)
    b = vec6()
    b[0] = k_pos * (target_pos[0] - point[0])
    b[1] = k_pos * (target_pos[1] - point[1])
    b[2] = k_pos * (target_pos[2] - point[2])
    b[3] = controls[w, t, 0]  # commanded roll rate (about world x)
    b[4] = controls[w, t, 1]  # commanded pitch rate (about world y)
    b[5] = k_yaw * e_world[2]  # regulate yaw only

    dq = damped_ls(jmat, b, lam)
    for c in range(7):
        ctrl[w, c] = dq[c]


class _BalanceFr3IK:
    """Applier: launches :func:`_balance_fr3_ik_kernel` to write ``d.ctrl``."""

    def __init__(
        self,
        arm_dofadr: np.ndarray,
        ee_body: int,
        target_pos: np.ndarray,
        level_quat: np.ndarray,
        k_pos: float,
        k_yaw: float,
        lam: float,
    ) -> None:
        # Allocated here (build time, under the engine's ScopedDevice) -- never
        # inside apply(), which runs during graph capture.
        self._arm_dof = wp.array(
            np.asarray(arm_dofadr, dtype=np.int32), dtype=wp.int32
        )
        self.ee_body = int(ee_body)
        p = [float(x) for x in target_pos]
        self._target = wp.vec3(p[0], p[1], p[2])
        g = [float(x) for x in level_quat]  # [w, x, y, z]
        self._level = wp.quat(g[0], g[1], g[2], g[3])
        self.k_pos = float(k_pos)
        self.k_yaw = float(k_yaw)
        self.lam = float(lam)

    def apply(self, m, d, controls: wp.array, t: int) -> None:
        """Write ``d.ctrl`` from the staged 2-D tilt command at step ``t``."""
        wp.launch(
            _balance_fr3_ik_kernel,
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
                self._target,
                self._level,
                wp.float32(self.k_pos),
                wp.float32(self.k_yaw),
                wp.float32(self.lam),
                controls,
                wp.int32(t),
                d.ctrl,
            ],
        )


def _build_model(
    shape: str, scale: float
) -> tuple[mujoco.MjModel, mujoco.MjSpec, mujoco.MjsBody]:
    """Attach the plate EE and graft ``shape`` onto the block/goal bodies.

    Block placement is left to :meth:`BalanceFr3.set_initial_state`, which needs
    the live plate pose (from the arm), so only the shape geometry is grafted
    here. Returns ``(model, spec, shape_body)``.
    """
    spec = mujoco.MjSpec.from_file(
        str(MODELS_DIR / "balance_fr3" / "scene.xml")
    )
    attach_ee(spec, "plate")  # plate EE onto the shared arm's attach site
    shape_body = graft_shape(
        spec, "block", shape, scale, default=spec.find_default("block")
    )
    graft_shape(
        spec,
        "goal",
        shape,
        scale,
        default=spec.find_default("goal"),
        include_sites=False,
    )
    return spec.compile(), spec, shape_body


class RandomWaypoint:
    """Goal jumps to a new random on-plate point once the block gets close.

    Stateful (current target + RNG), unlike :class:`GoalDrift` -- reset()
    per episode the same way ``uncertainty/filter.py``'s filters are.
    """

    def __init__(
        self,
        margin: float = 0.02,
        jump_threshold: float = 0.03,
        seed: int = 0,
    ) -> None:
        """``margin``/``jump_threshold`` in meters; ``seed`` for the RNG.

        ``margin`` is *extra* clearance beyond the tracked shape's own
        footprint -- ``BalanceFr3`` adds ``shape_max_xy_radius`` on top of
        it, so this is measured from the shape's edge, not its center.
        """
        self.margin = float(margin)
        self.jump_threshold = float(jump_threshold)
        self._rng = np.random.default_rng(seed)
        self._current: np.ndarray | None = None

    def reset(self) -> None:
        """Forget the current target so the next :meth:`target` resamples."""
        self._current = None

    def target(
        self, safe_x: float, safe_y: float, block_xy: np.ndarray
    ) -> np.ndarray:
        """Current goal ``(x, y)``, resampling if ``block_xy`` is close."""
        if self._current is not None:
            if np.linalg.norm(block_xy - self._current) < self.jump_threshold:
                self._current = None
        if self._current is None:
            cand = None
            for _ in range(20):  # avoid landing right back near the block
                cand = self._rng.uniform([-safe_x, -safe_y], [safe_x, safe_y])
                if np.linalg.norm(block_xy - cand) >= self.jump_threshold:
                    break
            self._current = cand
        return self._current


class BalanceFr3(Task):
    """FR3 tilts an EE plate to slide a free block to a goal on the plate."""

    def __init__(  # noqa: PLR0915
        self,
        *,
        sampling_space: Literal["task", "joint"] = "task",
        shape: str = "sphere",
        scale: float = 1.0,
        goal_xy: tuple[float, float] = (0.0, 0.0),
        bite: float = 0.0001,
        max_tilt_vel: float = 1.0,
        w_pos: float = 1000.0,
        w_orient: float = 0.0,
        w_ctrl: float = 0.0,
        w_arm_home: float = 1.0,
        terminal_scale: float = 5.0,
        k_pos: float = 1.0,
        k_yaw: float = 1.0,
        model_config: ModelConfig | None = None,
        goal_drift: GoalDrift | None = None,
        random_goal: RandomWaypoint | None = None,
        lost_margin: float = 0.16,
        arm_kv: float | None = None,
        ctrl_range_scale: float = 1.0,
    ) -> None:
        """Compose the FR3 + plate EE + grafted ``shape`` at ``scale``.

        Args:
            sampling_space: ``"task"`` (nu=2, plate roll/pitch rate, IK-mapped)
                or ``"joint"`` (nu=7, joint velocities sampled directly).
            shape: Block/goal shape under ``models/shapes/``.
            scale: Uniform size scale factor.
            goal_xy: Target ``(x, y)`` in the plate's local frame (from its
                center); the goal rides the plate's tilt.
            bite: Resting penetration depth when placing the block (contact
                stability), used by ``set_initial_state``.
            max_tilt_vel: Task-space plate-tilt bound (rad/s) per axis.
            w_pos: Weight on the squared block->goal position error.
            w_orient: Weight on the block->goal geodesic orientation error
                (0 default -- a ball has no meaningful orientation; raise for
                oriented shapes).
            w_ctrl: Weight on the squared joint-velocity control effort.
            w_arm_home: Weight on the arm-home posture penalty (||q_arm -
                q_home||). Anchors the redundant arm so joint-space sampling
                stays near the level home. No effect without a "home" keyframe;
                set 0 for task-space sampling (the IK already bounds the arm).
            terminal_scale: Terminal cost = ``terminal_scale`` x the running
                pos(+orient) terms (control/posture are running-only).
            k_pos: IK gain regulating the EE toward its home position (task).
            k_yaw: IK gain regulating the EE yaw toward level (task).
            model_config: Solver/integrator overrides layered on ``_BASELINE``.
            goal_drift: Optional moving-goal spec (yaw ignored -- distance
                cost); ``|goal_xy| + radius`` must stay on the plate.
                Mutually exclusive with ``random_goal``.
            random_goal: Optional random-waypoint goal: jumps to a new
                random on-plate point once the block gets within its
                ``jump_threshold``. Its ``margin`` is extra clearance on
                top of the shape's own footprint (added automatically via
                ``shape_max_xy_radius``). Mutually exclusive with
                ``goal_drift``.
            lost_margin: How far (m) the block's plate-local height may drop
                below rest before ``task_success`` reports it lost. Sized to
                the current shape/scale -- re-derive by hand if either
                changes, the same way ``ContactBudget`` is.
            arm_kv: Overrides the arm's velocity-actuator gain on this
                task's own compiled model only -- other FR3 tasks compile
                their own from the same XML and are unaffected. ``None``
                leaves the XML's gain untouched. At the XML default the
                simulated joints visibly under-track a commanded velocity
                (worse on higher-inertia base joints), which accounted for
                most of an observed Balance-FR3 sim-vs-real gap. Re-sweep
                on the real robot before changing it.
            ctrl_range_scale: ``"joint"`` sampling only -- shrinks the raw
                MJCF actuator ctrlrange to this fraction before it becomes the
                planner's sampling bound (see
                ``configs/planner/balance_fr3.yaml``). ``1.0`` is a no-op:
                every arm actuator here is already ctrllimited, so this only
                makes that bound *narrowable*. Ignored under ``"task"``
                sampling, whose bound is ``max_tilt_vel``, in a different
                unit.
        """
        self.sampling_space = sampling_space

        if sampling_space == "task":
            nu: int | None = 2
            ctrl_limits = {
                "u_min": np.array([-max_tilt_vel, -max_tilt_vel]),
                "u_max": np.array([max_tilt_vel, max_tilt_vel]),
            }
        else:
            nu, ctrl_limits = None, None

        mj_model, mj_spec, shape_body = _build_model(shape, scale)

        if arm_kv is not None:
            # Patches this task's own compiled mj_model only -- fr3_arm.xml
            # on disk is untouched, and every other FR3 task compiles its
            # own separate model from it, so they're unaffected.
            mj_model.actuator_gainprm[:, 0] = arm_kv
            mj_model.actuator_biasprm[:, 2] = -arm_kv

        if sampling_space == "joint":
            # Task.__init__ derives u_min/u_max straight from
            # mj_model.actuator_ctrlrange -- ignoring `ctrl_limits` entirely
            # -- whenever `nu` is left None (see its docstring). Passing
            # `nu` explicitly (same as FlipFr3 does) is what makes
            # `ctrl_limits` take effect at all; at ctrl_range_scale=1.0 this
            # is numerically identical to the old None-fallback (every arm
            # actuator here is ctrllimited), so it's a no-op by value, not
            # just by construction, at the default.
            nu = mj_model.nu
            ctrl_limits = {
                "u_min": mj_model.actuator_ctrlrange[:, 0] * ctrl_range_scale,
                "u_max": mj_model.actuator_ctrlrange[:, 1] * ctrl_range_scale,
            }

        super().__init__(
            mj_model,
            mj_spec=mj_spec,
            endpoint_body=["block"],
            nu=nu,
            ctrl_limits=ctrl_limits,
            model_config=_BASELINE.merged_with(model_config),
            contact_budget=balance_fr3_contact_budget(shape),
            goal_drift=goal_drift,
        )
        self.goal_xy = (float(goal_xy[0]), float(goal_xy[1]))

        # Keep the drifting goal on the plate: read the true (possibly
        # non-square) plate half-extents from the model instead of a
        # hardcoded literal, less the default block half-width 0.05, so the
        # goal may travel most of the plate. A hardcoded number here already
        # went stale once (plate resized from 0.2x0.2 to 0.195x0.25 without
        # updating it) and would silently let a drift target run off the
        # physical edge instead of raising. Larger --scale shrinks the true
        # margin; the default shapes fit.
        plate_geom = mj_spec.geom("plate")
        safe_x, safe_y = np.asarray(plate_geom.size[:2]) - 0.05
        rx, ry = self.goal_drift.radius_xy
        if abs(self.goal_xy[0]) + abs(rx) > safe_x or (
            abs(self.goal_xy[1]) + abs(ry) > safe_y
        ):
            raise ValueError(
                f"goal_xy {self.goal_xy} + drift radius {(rx, ry)} leaves "
                f"the plate (|center|+|radius| must be <= "
                f"{(float(safe_x), float(safe_y))} m, per axis)"
            )

        self.random_goal = random_goal
        if random_goal is not None:
            if self.goal_drift.enabled:
                raise ValueError(
                    "random_goal and an active goal_drift are mutually "
                    "exclusive -- pass at most one"
                )
            # random_goal.margin is *extra* clearance on top of the shape's
            # own footprint (shape_max_xy_radius) -- without subtracting the
            # footprint too, a goal sampled at the edge of the safe box
            # would put the shape's own edge, not just its center, right at
            # the physical plate boundary. This is why a default margin
            # that happened to equal a shape's radius (e.g. the 0.05 m
            # circle puck) looked "right at the edge": it left zero true
            # clearance for that shape.
            keepout = (
                shape_max_xy_radius(shape_body, scale) + random_goal.margin
            )
            rand_safe = np.asarray(plate_geom.size[:2]) - keepout
            if rand_safe[0] <= 0.0 or rand_safe[1] <= 0.0:
                raise ValueError(
                    f"random_goal keep-out {keepout} m (shape footprint + "
                    f"margin) leaves no room on the plate (half-extents "
                    f"{tuple(np.asarray(plate_geom.size[:2]))} m)"
                )
            self._random_safe_xy = (float(rand_safe[0]), float(rand_safe[1]))

        # Cost-sensor addresses (block pose in the goal frame).
        self._adr_pos = self._sensor_adr("position")
        self._adr_orient = self._sensor_adr("orientation")

        self.w_pos = float(w_pos)
        self.w_orient = float(w_orient)
        self.w_ctrl = float(w_ctrl)
        self.w_arm_home = float(w_arm_home)
        self.terminal_scale = float(terminal_scale)

        # Block free-joint qpos + plate frame (goal rides it, block rests on).
        self._block_adr = self._joint_qposadr("block")
        self._plate_body_id = int(mj_model.body("plate_body").id)
        plate_top_local = float(plate_geom.pos[2] + plate_geom.size[2])
        self._rest_z_local = (
            plate_top_local - shape_min_z(shape_body, scale) - float(bite)
        )
        self.lost_margin = float(lost_margin)

        # Arm bookkeeping (IK + arm-home posture).
        self.ee_body_id = int(mj_model.body("ee_frame").id)
        arm_joints = [f"fr3_joint{i}" for i in range(1, 8)]
        jids = [mj_model.joint(n).id for n in arm_joints]
        self.arm_qposadr = mj_model.jnt_qposadr[jids].astype(int)
        self.arm_dofadr = mj_model.jnt_dofadr[jids].astype(int)
        self.joint_limits = mj_model.jnt_range[jids]  # (7, 2)
        self._jacp = np.zeros((3, mj_model.nv))  # mj_jacBody scratch
        self._jacr = np.zeros((3, mj_model.nv))
        self.ik_damping = 1e-3
        self.k_pos = float(k_pos)
        self.k_yaw = float(k_yaw)

        # Home reference: arm posture + the EE pose the task-space IK regulates
        # to (position + level orientation), taken from the "home" keyframe.
        kid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if kid != -1:
            self.q_home = mj_model.key_qpos[kid][self.arm_qposadr]
            scratch = mujoco.MjData(mj_model)
            scratch.qpos[:] = mj_model.key_qpos[kid]
            mujoco.mj_forward(mj_model, scratch)
            self.ee_home_pos = scratch.xpos[self.ee_body_id].copy()
            self.ee_home_quat = scratch.xquat[self.ee_body_id].copy()
        else:
            self.q_home = None
            self.ee_home_pos = np.zeros(3)
            self.ee_home_quat = np.array([1.0, 0.0, 0.0, 0.0])

    def _sensor_adr(self, name: str) -> int:
        """Address of a named sensor's first scalar in ``sensordata``."""
        sid = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        return int(self.mj_model.sensor_adr[sid])

    def _joint_qposadr(self, name: str) -> int:
        """First qpos address of a named joint."""
        return int(self.mj_model.jnt_qposadr[self.mj_model.joint(name).id])

    @property
    def object_pose_qpos(self) -> ObjectPose:
        """The free block's pose location in ``qpos`` (uncertainty hook)."""
        return ObjectPose(kind="free", adr=self._block_adr)

    @property
    def contact_probes(self) -> tuple[ContactProbe, ...]:
        """Block<->plate contact (the only real pair; arm geoms don't pair)."""
        return (
            ContactProbe(
                name="block_plate",
                geoms_a=body_geom_ids(self.mj_model, "block"),
                geoms_b=body_geom_ids(self.mj_model, "plate_body"),
                bit=0,
            ),
        )

    def _block_local_xyz(self, mj_data: mujoco.MjData) -> np.ndarray:
        """Block position in the plate's *local* frame (robust to tilt)."""
        plate_pos = mj_data.xpos[self._plate_body_id]
        plate_quat = mj_data.xquat[self._plate_body_id]
        inv_quat = np.empty(4)
        mujoco.mju_negQuat(inv_quat, plate_quat)
        world_off = (
            mj_data.qpos[self._block_adr:self._block_adr + 3] - plate_pos
        )
        local = np.empty(3)
        mujoco.mju_rotVecQuat(local, world_off, inv_quat)
        return local

    def _block_local_z(self, mj_data: mujoco.MjData) -> float:
        """Block height in the plate's *local* frame (robust to tilt)."""
        return float(self._block_local_xyz(mj_data)[2])

    def task_success(self, mj_data: mujoco.MjData) -> bool:
        """``False`` once the block has dropped ``lost_margin`` off rest."""
        floor = self._rest_z_local - self.lost_margin
        return self._block_local_z(mj_data) >= floor

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

    def _arm_home_ref(self) -> tuple[np.ndarray, float]:
        """Home reference + effective weight (0 with no 'home' keyframe)."""
        if self.q_home is None:
            return np.zeros(7), 0.0
        return np.asarray(self.q_home), self.w_arm_home

    def build_cost_kernel(self) -> _BalanceFr3Cost:  # noqa: D102
        q_home, w_arm_home = self._arm_home_ref()
        return _BalanceFr3Cost(
            self._adr_pos,
            self._adr_orient,
            self.w_pos,
            self.w_orient,
            self.w_ctrl,
            self.arm_qposadr,
            q_home,
            w_arm_home,
        )

    def build_terminal_cost_kernel(self) -> _BalanceFr3Cost:  # noqa: D102
        # Same pose expression, scaled; control/posture are running-only.
        s = self.terminal_scale
        q_home, _ = self._arm_home_ref()
        return _BalanceFr3Cost(
            self._adr_pos,
            self._adr_orient,
            s * self.w_pos,
            s * self.w_orient,
            0.0,
            self.arm_qposadr,
            q_home,
            0.0,
        )

    def build_control_map_kernel(self) -> Any | None:  # noqa: D102
        if self.sampling_space == "joint":
            return None  # joint velocities sampled directly -> write ctrl
        return _BalanceFr3IK(
            self.arm_dofadr,
            self.ee_body_id,
            self.ee_home_pos,
            self.ee_home_quat,
            self.k_pos,
            self.k_yaw,
            self.ik_damping,
        )

    def control_map_host(
        self, mj_data: mujoco.MjData, u: np.ndarray
    ) -> np.ndarray:
        """Map a control to the 7 joint-velocity actuators (host).

        Joint sampling is identity. Task sampling maps the 2-D plate-tilt
        command to joint velocities by the shared damped-LS IK, regulating the
        EE toward its home position and level yaw.
        """
        if self.sampling_space == "joint":
            return np.asarray(u)

        m = self.mj_model
        mujoco.mj_jacBody(m, mj_data, self._jacp, self._jacr, self.ee_body_id)
        jac = np.vstack([self._jacp, self._jacr])[:, self.arm_dofadr]  # (6, 7)

        ee_pos = mj_data.xpos[self.ee_body_id]
        ee_quat = mj_data.xquat[self.ee_body_id]
        res = np.zeros(3)
        mujoco.mju_subQuat(res, self.ee_home_quat, ee_quat)
        e_world = quat_rotate(ee_quat, res)
        twist = np.array(
            [
                self.k_pos * (self.ee_home_pos[0] - ee_pos[0]),
                self.k_pos * (self.ee_home_pos[1] - ee_pos[1]),
                self.k_pos * (self.ee_home_pos[2] - ee_pos[2]),
                u[0],
                u[1],
                self.k_yaw * e_world[2],
            ]
        )
        return damped_ls_host(jac, twist, self.ik_damping)

    def cost_components(self, mj_data: mujoco.MjData) -> dict[str, float]:
        """Host twin of the running cost, split per weighted term."""
        s = mj_data.sensordata
        p = s[self._adr_pos : self._adr_pos + 3]
        pos = self.w_pos * float(np.dot(p, p))

        q = s[self._adr_orient : self._adr_orient + 4]
        qv = float(np.linalg.norm(q[1:4]))
        orient = self.w_orient * (2.0 * math.atan2(qv, abs(float(q[0]))))

        ctrl = self.w_ctrl * float(np.dot(mj_data.ctrl, mj_data.ctrl))

        if self.q_home is None:
            arm_home = 0.0
        else:
            dq = mj_data.qpos[self.arm_qposadr] - np.asarray(self.q_home)
            arm_home = self.w_arm_home * float(np.linalg.norm(dq))

        return {
            "pos": pos,
            "orient": orient,
            "ctrl": ctrl,
            "arm_home": arm_home,
            "total": pos + orient + ctrl + arm_home,
        }

    def running_cost_reference(
        self, mj_data: mujoco.MjData, u: object = None
    ) -> float:
        """Eager numpy running cost (sum of :meth:`cost_components`)."""
        return self.cost_components(mj_data)["total"]

    def terminal_cost_reference(self, mj_data: mujoco.MjData) -> float:
        """Eager numpy terminal cost (``terminal_scale * pose terms``)."""
        c = self.cost_components(mj_data)
        return self.terminal_scale * (c["pos"] + c["orient"])

    def _goal_local(
        self, t: float, mj_data: mujoco.MjData | None = None
    ) -> np.ndarray:
        """Plate-local goal ``(x, y, z)`` at time ``t`` (center + drift).

        With ``random_goal`` set, ignores ``t``/``goal_xy``/``goal_drift``
        entirely and instead jumps to a new random on-plate point once the
        live block (read from ``mj_data``) gets close to the current one.
        """
        if self.random_goal is not None:
            block_xy = self._block_local_xyz(mj_data)[:2]
            x, y = self.random_goal.target(*self._random_safe_xy, block_xy)
            return np.array([x, y, self._rest_z_local], dtype=np.float64)
        dx, dy, _ = self.goal_drift.offset(t)  # yaw ignored: distance cost
        return np.array(
            [self.goal_xy[0] + dx, self.goal_xy[1] + dy, self._rest_z_local],
            dtype=np.float64,
        )

    def goal_mocap_pose(
        self,
        t: float,
        mj_data: mujoco.MjData,
        base_pos: np.ndarray,
        base_quat: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """World goal pose = live plate frame ∘ (plate-local goal + drift).

        Always returns a pose (not gated on ``goal_drift``): the goal is a world
        mocap, so it must ride the plate's tilt each frame -- with zero drift
        this pins it to the plate-relative target. ``base_*`` are unused.
        """
        plate_pos = np.asarray(mj_data.xpos[self._plate_body_id], np.float64)
        plate_quat = np.asarray(mj_data.xquat[self._plate_body_id], np.float64)
        world_off = np.zeros(3)
        mujoco.mju_rotVecQuat(
            world_off, self._goal_local(t, mj_data), plate_quat
        )
        return plate_pos + world_off, plate_quat.copy()

    def set_initial_state(
        self,
        mj_data: mujoco.MjData,
        *,
        block_xy: tuple[float, float] = (0.0, 0.0),
        block_yaw: float = 0.0,
    ) -> None:
        """Place the block on the plate (host); the arm gives the tilt.

        The caller loads the "home" keyframe first (level plate). This reads the
        live plate frame and rests the block on it at plate-local ``block_xy``.

        Args:
            mj_data: MjData to mutate; caller has loaded a keyframe first.
            block_xy: Block ``(x, y)`` in the plate's local frame, meters.
            block_yaw: Block yaw about the plate's local normal, radians.
        """
        mujoco.mj_forward(self.mj_model, mj_data)
        plate_pos = mj_data.xpos[self._plate_body_id].copy()
        plate_quat = mj_data.xquat[self._plate_body_id].copy()

        local = np.array(
            [block_xy[0], block_xy[1], self._rest_z_local], dtype=np.float64
        )
        world_offset = np.zeros(3)
        mujoco.mju_rotVecQuat(world_offset, local, plate_quat)
        block_quat = quat_mul(
            plate_quat,
            np.array(
                [np.cos(0.5 * block_yaw), 0.0, 0.0, np.sin(0.5 * block_yaw)]
            ),
        )

        adr = self._block_adr
        mj_data.qpos[adr : adr + 3] = plate_pos + world_offset
        mj_data.qpos[adr + 3 : adr + 7] = block_quat

        goal_mid = self.goal_mocap_id
        if goal_mid is not None:
            goal_off = np.zeros(3)
            mujoco.mju_rotVecQuat(
                goal_off, self._goal_local(0.0, mj_data), plate_quat
            )
            mj_data.mocap_pos[goal_mid] = plate_pos + goal_off
            mj_data.mocap_quat[goal_mid] = plate_quat
        mujoco.mj_forward(self.mj_model, mj_data)
