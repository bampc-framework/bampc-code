"""Push: push a block to a target pose (2-D pusher, direct actuators).

The simplest task: 2 actuators sampled directly, friction domain
randomization, a cost on block position + orientation + a pusher-near-block
shaping term. Block geometry comes from ``bampc.task.common.shapes``
(``shape``, ``scale``); sensors give the block pose relative to the goal.
"""

from __future__ import annotations

import math

import mujoco
import warp as wp

from bampc import MODELS_DIR
from bampc.task.base import (
    ContactProbe,
    GoalDrift,
    ModelConfig,
    ObjectPose,
    Task,
    body_geom_ids,
)
from bampc.task.common.shapes import (
    graft_shape,
    push_contact_budget,
    shape_com_xy,
    shape_min_z,
)

# Override per-sweep via ``Push(model_config=ModelConfig(...))``.
_BASELINE = ModelConfig(
    timestep=0.01,
    solver="Newton",
    integrator="Euler",
    cone="pyramidal",
    jacobian="auto",
    impratio=1.0,
    iterations=1,
    ls_iterations=4,
    eulerdamp=True,
    warmstart=True,
)


@wp.kernel
def _push_cost_kernel(
    sensordata: wp.array2d(dtype=wp.float32),
    w_pos: wp.float32,
    w_orient: wp.float32,
    w_close: wp.float32,
    scale: wp.float32,
    cost: wp.array(dtype=wp.float32),
):
    """SE(2) pose error to the goal plus a pusher-near-block shaping term."""
    i = wp.tid()

    ex = sensordata[i, 0]
    ey = sensordata[i, 1]
    pos_cost = ex * ex + ey * ey

    qw = sensordata[i, 3]
    qx = sensordata[i, 4]
    qy = sensordata[i, 5]
    qz = sensordata[i, 6]
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = wp.atan2(siny_cosp, cosy_cosp)
    orient_cost = yaw * yaw

    cx = sensordata[i, 7]
    cy = sensordata[i, 8]
    cz = sensordata[i, 9]
    close_cost = cx * cx + cy * cy + cz * cz

    running = w_pos * pos_cost + w_orient * orient_cost + w_close * close_cost
    cost[i] = cost[i] + scale * running


class _PushCost:
    """Launches :func:`_push_cost_kernel` with this task's weights."""

    def __init__(self, w_pos: float, w_orient: float, w_close: float) -> None:
        self.w_pos = float(w_pos)
        self.w_orient = float(w_orient)
        self.w_close = float(w_close)

    def accumulate(self, d, cost: wp.array, scale: float) -> None:
        """Add ``scale * running_cost(d)`` into ``cost`` (per world)."""
        wp.launch(
            _push_cost_kernel,
            dim=cost.shape[0],
            inputs=[
                d.sensordata,
                wp.float32(self.w_pos),
                wp.float32(self.w_orient),
                wp.float32(self.w_close),
                wp.float32(scale),
                cost,
            ],
        )


@wp.kernel
def _push_terminal_kernel(
    sensordata: wp.array2d(dtype=wp.float32),
    w_pos: wp.float32,
    w_orient: wp.float32,
    scale: wp.float32,
    cost: wp.array(dtype=wp.float32),
):
    """Terminal SE(2) pose error to the goal (no shaping term)."""
    i = wp.tid()

    ex = sensordata[i, 0]
    ey = sensordata[i, 1]
    pos_cost = ex * ex + ey * ey

    qw = sensordata[i, 3]
    qx = sensordata[i, 4]
    qy = sensordata[i, 5]
    qz = sensordata[i, 6]
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = wp.atan2(siny_cosp, cosy_cosp)
    orient_cost = yaw * yaw

    terminal = w_pos * pos_cost + w_orient * orient_cost
    cost[i] = cost[i] + scale * terminal


class _PushTerminalCost:
    """Launches :func:`_push_terminal_kernel` with the terminal weights."""

    def __init__(self, w_pos: float, w_orient: float) -> None:
        self.w_pos = float(w_pos)
        self.w_orient = float(w_orient)

    def accumulate(self, d, cost: wp.array, scale: float) -> None:
        """Add ``scale * terminal_cost(d)`` into ``cost`` (per world)."""
        wp.launch(
            _push_terminal_kernel,
            dim=cost.shape[0],
            inputs=[
                d.sensordata,
                wp.float32(self.w_pos),
                wp.float32(self.w_orient),
                wp.float32(scale),
                cost,
            ],
        )


class Push(Task):
    """Planar push task with a sphere pusher and a slide/slide/hinge block."""

    def __init__(
        self,
        *,
        shape: str = "t",
        scale: float = 1.0,
        bite: float = 0.0003,
        model_config: ModelConfig | None = None,
        goal_drift: GoalDrift | None = None,
    ) -> None:
        """Compose ``shape`` at ``scale`` onto the block/goal bodies.

        Args:
            shape: Name under ``models/shapes/`` (see
                ``bampc.task.common.shapes.list_shapes``).
            scale: Uniform size scale factor.
            bite: Resting penetration depth (contact-solver stability); the
                block has no vertical DOF, so this is its fixed height.
            model_config: Solver/integrator overrides layered on
                ``_BASELINE``.
            goal_drift: Optional moving-goal spec; its xy radius must keep
                the goal on the table (``|radius| <= 0.35`` m).
        """
        path = str(MODELS_DIR / "push" / "scene.xml")
        spec = mujoco.MjSpec.from_file(path)
        shape_body = graft_shape(spec, "block", shape, scale)
        graft_shape(
            spec,
            "goal",
            shape,
            scale,
            default=spec.find_default("goal"),
            include_sites=False,
        )

        z = -shape_min_z(shape_body, scale) - bite
        spec.body("block").pos = [0.0, 0.0, z]
        spec.body("goal").pos = [0.0, 0.0, z]

        com_x, com_y = shape_com_xy(shape_body, scale)
        anchor = [com_x, com_y, 0.0]
        spec.joint("block_x").pos = anchor
        spec.joint("block_y").pos = anchor
        spec.joint("block_yaw").pos = anchor

        mj_model = spec.compile()
        super().__init__(
            mj_model,
            mj_spec=spec,
            trace_sites=["pusher"],
            endpoint_body="block",
            model_config=_BASELINE.merged_with(model_config),
            contact_budget=push_contact_budget(shape),
            goal_drift=goal_drift,
        )
        # Keep the drifting goal on the table (planar pusher, no robot reach).
        rx, ry = self.goal_drift.radius_xy
        if max(abs(rx), abs(ry)) > 0.35:
            raise ValueError(
                f"goal_drift radius {(rx, ry)} leaves the table "
                "(|radius| must be <= 0.35 m)"
            )
        # Shared by the device kernel and the host reference below so the
        # live debug print matches the GPU exactly.
        self.w_pos = 1000.0
        self.w_orient = 1000.0
        self.w_close = 50.0

    def build_cost_kernel(self) -> _PushCost:  # noqa: D102
        return _PushCost(self.w_pos, self.w_orient, self.w_close)

    @property
    def object_pose_qpos(self) -> ObjectPose:
        """Block pose: slide/slide/hinge, so planar (xy + yaw only)."""
        qadr = self.mj_model.jnt_qposadr
        return ObjectPose(
            kind="planar",
            x_adr=int(qadr[self.mj_model.joint("block_x").id]),
            y_adr=int(qadr[self.mj_model.joint("block_y").id]),
            yaw_adr=int(qadr[self.mj_model.joint("block_yaw").id]),
        )

    @property
    def contact_probes(self) -> tuple[ContactProbe, ...]:
        """Pusher-block contact only.

        Push is a purely planar model -- the block rides slide/slide/hinge
        joints with no table body -- so there is no block-table pair to track.
        """
        return (
            ContactProbe(
                name="pusher_block",
                geoms_a=body_geom_ids(self.mj_model, "pusher"),
                geoms_b=body_geom_ids(self.mj_model, "block"),
                bit=0,
            ),
        )

    def cost_components(self, mj_data: mujoco.MjData) -> dict[str, float]:
        """Host twin of the running cost, split per weighted term."""
        s = mj_data.sensordata

        ex, ey = float(s[0]), float(s[1])
        pos_cost = ex * ex + ey * ey

        qw, qx, qy, qz = float(s[3]), float(s[4]), float(s[5]), float(s[6])
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        orient_cost = yaw * yaw

        cx, cy, cz = float(s[7]), float(s[8]), float(s[9])
        close_cost = cx * cx + cy * cy + cz * cz

        pos = self.w_pos * pos_cost
        orient = self.w_orient * orient_cost
        close = self.w_close * close_cost
        return {
            "pos": pos,
            "orient": orient,
            "close": close,
            "total": pos + orient + close,
        }

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

    def build_terminal_cost_kernel(self) -> _PushTerminalCost:  # noqa: D102
        return _PushTerminalCost(w_pos=2000.0, w_orient=2000.0)
