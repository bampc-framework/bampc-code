"""Balance: tilt a plate to slide a block to a plate-relative goal.

The simplest balancing task: a plate floating in mid-air, tilted about two
range-limited hinges (roll then pitch), with a free-rigid-body block resting
on top that must be slid -- by gravity, contact friction, and tilting alone --
to a goal position defined *relative to the plate*. Block geometry comes
from ``bampc.task.common.shapes`` (``shape``, ``scale``), same as
``Push``. The goal is a plain child body of the plate (not a mocap -- MuJoCo
mocap bodies must be direct children of the world body, so this is what
lets the goal marker track the tilt for free via ordinary forward
kinematics) and the cost
sensor reads the block's position directly in that moving frame.
"""

from __future__ import annotations

import mujoco
import numpy as np
import warp as wp

from bampc import MODELS_DIR
from bampc.task.base import GoalDrift, ModelConfig, ObjectPose, Task
from bampc.task.common.shapes import (
    balance_contact_budget,
    graft_shape,
    shape_min_z,
)

# Override per-sweep via ``Balance(model_config=ModelConfig(...))``. Same
# family as PushFr3's "free" baseline: a free rigid body settling onto a
# surface benefits from the softer elliptic cone / implicit integrator.
_BASELINE = ModelConfig(
    timestep=0.005,
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


@wp.kernel
def _balance_cost_kernel(
    sensordata: wp.array2d(dtype=wp.float32),
    ctrl: wp.array2d(dtype=wp.float32),
    w_pos: wp.float32,
    w_ctrl: wp.float32,
    scale: wp.float32,
    cost: wp.array(dtype=wp.float32),
):
    """Squared distance to the goal plus a quadratic control-effort term."""
    i = wp.tid()

    ex = sensordata[i, 0]
    ey = sensordata[i, 1]
    ez = sensordata[i, 2]
    pos_cost = ex * ex + ey * ey + ez * ez

    u_roll = ctrl[i, 0]
    u_pitch = ctrl[i, 1]
    ctrl_cost = u_roll * u_roll + u_pitch * u_pitch

    running = w_pos * pos_cost + w_ctrl * ctrl_cost
    cost[i] = cost[i] + scale * running


class _BalanceCost:
    """Launches :func:`_balance_cost_kernel` with this task's weights."""

    def __init__(self, w_pos: float, w_ctrl: float = 0.0) -> None:
        self.w_pos = float(w_pos)
        self.w_ctrl = float(w_ctrl)

    def accumulate(self, d, cost: wp.array, scale: float) -> None:
        """Add ``scale * running_cost(d)`` into ``cost`` (per world)."""
        wp.launch(
            _balance_cost_kernel,
            dim=cost.shape[0],
            inputs=[
                d.sensordata,
                d.ctrl,
                wp.float32(self.w_pos),
                wp.float32(self.w_ctrl),
                wp.float32(scale),
                cost,
            ],
        )


class Balance(Task):
    """Tilt a plate (2-DOF gimbal) to slide a block to a goal on the plate."""

    def __init__(
        self,
        *,
        shape: str = "circle",
        scale: float = 1.0,
        bite: float = 0.0001,
        goal_xy: tuple[float, float] = (0.0, 0.0),
        model_config: ModelConfig | None = None,
        goal_drift: GoalDrift | None = None,
        lost_margin: float = 0.16,
    ) -> None:
        """Compose ``shape`` at ``scale`` onto the block/goal bodies.

        Args:
            shape: Name under ``models/shapes/`` (see
                ``bampc.task.common.shapes.list_shapes``).
            scale: Uniform size scale factor.
            bite: Resting penetration depth (contact-solver stability).
            goal_xy: Target position ``(x, y)`` in the plate's local frame,
                measured from the plate's center.
            model_config: Solver/integrator overrides layered on
                ``_BASELINE``.
            goal_drift: Optional moving-goal spec (yaw ignored -- the cost is
                distance only). ``|goal_xy| + radius`` must stay on the plate.
            lost_margin: How far (m) the block's plate-local height may drop
                below rest before ``task_success`` reports it lost. Sized to
                the current shape/scale (default: 2 sphere diameters at
                ``scale=1``) -- re-derive by hand if either changes, the same
                way ``ContactBudget`` is.
        """
        path = str(MODELS_DIR / "balance" / "scene.xml")
        spec = mujoco.MjSpec.from_file(path)
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

        plate_geom = spec.geom("plate")
        plate_top_z = plate_geom.pos[2] + plate_geom.size[2]
        z = plate_top_z - shape_min_z(shape_body, scale) - bite

        # The goal is now a world-body mocap; place it at the level-plate
        # WORLD pose (overwritten each frame by goal_mocap_pose /
        # set_initial_state, which track the live tilt + any drift).
        gimbal_z = spec.body("gimbal_roll").pos[2]
        spec.body("goal").pos = [goal_xy[0], goal_xy[1], gimbal_z + z]
        # Level-plate placeholder; the "home" start pose is (re)placed each
        # episode by set_initial_state, which accounts for the live tilt.
        spec.body("block").pos = [0.0, 0.0, gimbal_z + z]

        mj_model = spec.compile()
        super().__init__(
            mj_model,
            mj_spec=spec,
            endpoint_body=["block", "plate"],
            model_config=_BASELINE.merged_with(model_config),
            contact_budget=balance_contact_budget(shape),
            goal_drift=goal_drift,
        )
        self.goal_xy = (float(goal_xy[0]), float(goal_xy[1]))
        # Keep the drifting goal on the plate: plate half-extent 0.2, less
        # roughly a block half-width and a margin.
        rx, ry = self.goal_drift.radius_xy
        safe = 0.12
        if abs(self.goal_xy[0]) + abs(rx) > safe or (
            abs(self.goal_xy[1]) + abs(ry) > safe
        ):
            raise ValueError(
                f"goal_xy {self.goal_xy} + drift radius {(rx, ry)} leaves the "
                f"plate (|center|+|radius| must be <= {safe} m per axis)"
            )
        # Local-frame height (above the plate origin) a block rests at;
        # reused by set_initial_state to place the block under any tilt.
        self._rest_z_local = float(z)
        self._block_adr = self._joint_qposadr("block")
        self._roll_adr = self._joint_qposadr("tilt_roll")
        self._pitch_adr = self._joint_qposadr("tilt_pitch")
        self._plate_body_id = int(mj_model.body("plate").id)
        self.lost_margin = float(lost_margin)

        self.w_pos = 1000.0
        # Quadratic control-effort penalty (running cost only -- the
        # terminal state has no associated future control) that discourages
        # large, sudden tilt commands.
        self.w_ctrl = 2.0

    def _joint_qposadr(self, name: str) -> int:
        """First qpos address of a named joint."""
        return int(self.mj_model.jnt_qposadr[self.mj_model.joint(name).id])

    @property
    def object_pose_qpos(self) -> ObjectPose:
        """The free block's pose location in ``qpos`` (uncertainty hook)."""
        return ObjectPose(kind="free", adr=self._block_adr)

    def _block_local_z(self, mj_data: mujoco.MjData) -> float:
        """Block height in the plate's *local* frame (robust to tilt)."""
        plate_pos = mj_data.xpos[self._plate_body_id]
        plate_quat = mj_data.xquat[self._plate_body_id]
        inv_quat = np.empty(4)
        mujoco.mju_negQuat(inv_quat, plate_quat)
        world_off = (
            mj_data.qpos[self._block_adr:self._block_adr + 3] - plate_pos
        )
        local = np.empty(3)
        mujoco.mju_rotVecQuat(local, world_off, inv_quat)
        return float(local[2])

    def task_success(self, mj_data: mujoco.MjData) -> bool:
        """``False`` once the block has dropped ``lost_margin`` off rest."""
        floor = self._rest_z_local - self.lost_margin
        return self._block_local_z(mj_data) >= floor

    def build_cost_kernel(self) -> _BalanceCost:  # noqa: D102
        return _BalanceCost(self.w_pos, self.w_ctrl)

    def build_terminal_cost_kernel(self) -> _BalanceCost:  # noqa: D102
        return _BalanceCost(5.0 * self.w_pos)

    def cost_components(self, mj_data: mujoco.MjData) -> dict[str, float]:
        """Host twin of the running cost, split per weighted term."""
        ex, ey, ez = (float(v) for v in mj_data.sensordata[0:3])
        pos = self.w_pos * (ex * ex + ey * ey + ez * ez)
        u_roll, u_pitch = float(mj_data.ctrl[0]), float(mj_data.ctrl[1])
        ctrl = self.w_ctrl * (u_roll * u_roll + u_pitch * u_pitch)
        return {"pos": pos, "ctrl": ctrl, "total": pos + ctrl}

    def running_cost_reference(
        self, mj_data: mujoco.MjData, u: object = None
    ) -> float:
        """Eager numpy running cost (sum of :meth:`cost_components`)."""
        return self.cost_components(mj_data)["total"]

    def terminal_cost_reference(self, mj_data: mujoco.MjData) -> float:
        """Eager numpy terminal cost; same expression as the running cost.

        Unscaled: this task has no ``terminal_scale``, and its terminal
        kernel uses its own fixed weights rather than a multiple of the
        running ones.
        """
        return self.running_cost_reference(mj_data)

    def _goal_local(self, t: float) -> np.ndarray:
        """Plate-local goal ``(x, y, z)`` at time ``t`` (center + drift)."""
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

        Always returns a pose (not gated on ``goal_drift``): the goal is a
        world mocap, so it must be repositioned each frame to follow the
        plate's tilt -- with zero drift this reproduces the old plate-child
        marker exactly. ``base_*`` are unused (the pose is state-derived).
        """
        plate_pos = np.asarray(mj_data.xpos[self._plate_body_id], np.float64)
        plate_quat = np.asarray(mj_data.xquat[self._plate_body_id], np.float64)
        world_off = np.zeros(3)
        mujoco.mju_rotVecQuat(world_off, self._goal_local(t), plate_quat)
        return plate_pos + world_off, plate_quat.copy()

    def set_initial_state(
        self,
        mj_data: mujoco.MjData,
        *,
        tilt: tuple[float, float] = (0.0, 0.0),
        block_xy: tuple[float, float] = (0.0, 0.0),
        block_yaw: float = 0.0,
    ) -> None:
        """Set the plate's tilt and place the block relative to it (host).

        Args:
            mj_data: MjData to mutate; caller has loaded a keyframe first.
            tilt: ``(roll, pitch)`` hinge angles, radians.
            block_xy: Block position ``(x, y)`` in the plate's local frame
                (i.e. a translation from the plate origin), meters.
            block_yaw: Block yaw about the plate's local normal, radians.
        """
        mj_data.qpos[self._roll_adr] = float(tilt[0])
        mj_data.qpos[self._pitch_adr] = float(tilt[1])
        mujoco.mj_forward(self.mj_model, mj_data)

        plate_pos = mj_data.xpos[self._plate_body_id].copy()
        plate_quat = mj_data.xquat[self._plate_body_id].copy()

        local = np.array(
            [block_xy[0], block_xy[1], self._rest_z_local], dtype=np.float64
        )
        world_offset = np.zeros(3)
        mujoco.mju_rotVecQuat(world_offset, local, plate_quat)

        half = 0.5 * float(block_yaw)
        yaw_quat = np.array([np.cos(half), 0.0, 0.0, np.sin(half)])
        block_quat = np.zeros(4)
        mujoco.mju_mulQuat(block_quat, plate_quat, yaw_quat)

        adr = self._block_adr
        mj_data.qpos[adr : adr + 3] = plate_pos + world_offset
        mj_data.qpos[adr + 3 : adr + 7] = block_quat

        # Place the goal mocap at the (level-tilt) plate-relative target so a
        # non-viewer reset (headless) starts with a correct goal frame.
        goal_mid = self.goal_mocap_id
        if goal_mid is not None:
            goal_off = np.zeros(3)
            mujoco.mju_rotVecQuat(goal_off, self._goal_local(0.0), plate_quat)
            mj_data.mocap_pos[goal_mid] = plate_pos + goal_off
            mj_data.mocap_quat[goal_mid] = plate_quat
        mujoco.mj_forward(self.mj_model, mj_data)
