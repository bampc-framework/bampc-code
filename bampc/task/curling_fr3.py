"""Curling with an FR3 arm: shoot a puck down a lane at a target region.

The first task here that is **open-loop after release**. The EE is confined
to a launch rectangle by a hard velocity barrier inside the IK; the house
(goal region) sits outside that rectangle, so contact necessarily breaks and
the puck slides ballistically to rest with no further control authority.

That structure is the point. In every other task the controller stays in
contact and absorbs its own errors on the next replan, which partly hides
both uncertainty axes. Here a model error (ice friction) or a belief error
(puck pose at release) compounds over the whole slide and cannot be
corrected, so risk aggregation over domains has something real to bite on.

Two consequences fall out of that and both are load-bearing:

* The planning horizon must cover the **whole flight**, not the next
  contact. A horizon shorter than the slide cannot see where the puck ends
  up, so the planner just greedily approaches and stalls at the box wall.
* The terminal cost must charge for **speed as well as distance**. With
  distance alone the planner learns to fly the puck through the house at the
  terminal instant and score zero.

No ``sampling_space`` / ``manipulation_type`` axes: task-space sampling only
(a box constraint on EE *pose* cannot be expressed in joint-velocity
sampling), and a free 6-DOF puck only (isotropic Coulomb drag; per-axis joint
``frictionloss`` would make the range depend on shot direction).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

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
    curling_contact_budget,
    graft_shape,
    shape_min_z,
)

# Fallback solver options, used only when the caller names none; the tuned
# values live in configs/numerics/curling.yaml -- read its header before
# changing `cone` or `timestep`. `cone` is NOT free to change: the elliptic
# cone cannot solve the puck's flat contact at all. This fallback keeps the
# conservative timestep, since naming no profile opts out of that evidence.
_BASELINE = ModelConfig(
    timestep=0.01,
    solver="Newton",
    integrator="implicitfast",
    cone="pyramidal",
    jacobian="dense",
    impratio=1.0,
    iterations=10,
    ls_iterations=16,
    eulerdamp=True,
    warmstart=True,
)

#: Puck shapes this task supports (see ``curling_contact_budget``).
SHAPES = ("circle", "sphere", "square")

#: ``(x_lo, x_hi, y_lo, y_hi)`` the EE is confined to, in world metres.
#: Callers overriding the box (e.g. a ``--launch-length`` CLI flag) should
#: keep ``x_lo`` fixed and only move ``x_hi``, matching the real lane's near
#: edge staying put.
DEFAULT_LAUNCH_BOX = (0.45, 0.70, -0.15, 0.15)


@wp.kernel
def _curling_cost_kernel(  # noqa: PLR0915
    sensordata: wp.array2d(dtype=wp.float32),
    pos_adr: wp.int32,
    vel_adr: wp.int32,
    t1_adr: wp.int32,
    t2_adr: wp.int32,
    t3_adr: wp.int32,
    eez_adr: wp.int32,
    eepos_adr: wp.int32,
    blk_adr: wp.int32,
    goal_adr: wp.int32,
    w_pos: wp.float32,
    w_stop: wp.float32,
    w_attract: wp.float32,
    w_align: wp.float32,
    w_lat_vel: wp.float32,
    w_box: wp.float32,
    x_lo: wp.float32,
    x_hi: wp.float32,
    y_lo: wp.float32,
    y_hi: wp.float32,
    w_ee_orient: wp.float32,
    w_ee_height: wp.float32,
    ee_z_target: wp.float32,
    qpos: wp.array2d(dtype=wp.float32),
    arm_qposadr: wp.array(dtype=wp.int32),
    q_home: wp.array(dtype=wp.float32),
    w_arm_home: wp.float32,
    w_post_release: wp.float32,
    release_x: wp.float32,
    downrange_axis: wp.int32,
    ee_linvel_adr: wp.int32,
    ee_angvel_adr: wp.int32,
    scale: wp.float32,
    cost: wp.array(dtype=wp.float32),
):
    """Accumulate ``scale * running_cost`` into ``cost[w]``, one world/tid."""
    i = wp.tid()

    # 1. puck -> house, PLANAR. The house is a flat region on the lane, so the
    #    z component of the sensor (goal body sits at the puck's resting
    #    height) carries no information worth charging for.
    px = sensordata[i, pos_adr + 0]
    py = sensordata[i, pos_adr + 1]
    pos_cost = wp.sqrt(px * px + py * py)

    # 2. puck speed. Terminal-only in practice (w_stop is 0 in the running
    #    instance): without it a shot that passes THROUGH the house at the
    #    final step scores a perfect zero on term 1.
    vx = sensordata[i, vel_adr + 0]
    vy = sensordata[i, vel_adr + 1]
    vz = sensordata[i, vel_adr + 2]
    stop_cost = wp.sqrt(vx * vx + vy * vy + vz * vz)

    # 3. attract: EE to the puck's three rim sites -- gets the EE onto the
    #    puck at all, before any of the shot terms mean anything.
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

    # 4. shot-align: cos(alpha)+1 between EE->puck and house->puck (planar).
    #    Zero when the EE is directly behind the puck on the house line.
    ee_x = sensordata[i, eepos_adr + 0]
    ee_y = sensordata[i, eepos_adr + 1]
    rx = ee_x - sensordata[i, blk_adr + 0]
    ry = ee_y - sensordata[i, blk_adr + 1]
    gx = sensordata[i, goal_adr + 0] - sensordata[i, blk_adr + 0]
    gy = sensordata[i, goal_adr + 1] - sensordata[i, blk_adr + 1]
    rn = wp.sqrt(rx * rx + ry * ry)
    gn = wp.sqrt(gx * gx + gy * gy)
    align = wp.float32(0.0)
    if rn > 1.0e-6 and gn > 1.0e-6:
        align = (rx * gx + ry * gy) / (rn * gn) + 1.0

    # 4b. lateral puck velocity: the component of the puck's planar velocity
    #     perpendicular to the puck->house shot line (2-D cross product,
    #     |v x u_hat| = |vx*gy - vy*gx| / gn). Zero for a puck sliding
    #     straight down the line, however fast -- it only picks up
    #     side-slip/jitter from an off-axis or glancing push, which is what
    #     actually separates a clean push from a chaotic one, without
    #     fighting the forward (on-axis) component the shot itself needs.
    lat_cost = wp.float32(0.0)
    if gn > 1.0e-6:
        lat_cost = wp.abs(vx * gy - vy * gx) / gn

    # 5. launch-box margin, hinge on each face. This does NOT enforce the box
    #    -- the IK barrier does, exactly. It exists because that barrier maps
    #    a whole cone of samples onto identical rollouts, which drags the
    #    sampler's mean into the wall; this gives the mean somewhere to sit.
    box = (
        wp.max(ee_x - x_hi, 0.0)
        + wp.max(x_lo - ee_x, 0.0)
        + wp.max(ee_y - y_hi, 0.0)
        + wp.max(y_lo - ee_y, 0.0)
    )

    # 6. EE point-down: world z-axis of the stick vs (0, 0, -1).
    zx = sensordata[i, eez_adr + 0]
    zy = sensordata[i, eez_adr + 1]
    zz = sensordata[i, eez_adr + 2]
    ee_orient_cost = wp.sqrt(zx * zx + zy * zy + (zz + 1.0) * (zz + 1.0))

    # 7. EE height.
    ee_height_cost = wp.abs(sensordata[i, eepos_adr + 2] - ee_z_target)

    # 8. arm-home posture: anchors the 7-DOF arm's redundant null-space.
    arm_sq = wp.float32(0.0)
    for k in range(7):
        dq = qpos[i, arm_qposadr[k]] - q_home[k]
        arm_sq += dq * dq
    arm_home_cost = wp.sqrt(arm_sq)

    # 9. post-release stillness: once the puck's trailing edge clears the
    #    box (`release_x` includes the puck radius) the shot is open-loop,
    #    so charge the EE's own twist instead of letting attract/align chase
    #    a puck that is structurally gone. Twist, not joint velocity: two
    #    joints cancelling at the EE would hide under a joint-space measure.
    released = wp.float32(0.0)
    if sensordata[i, blk_adr + downrange_axis] > release_x:
        released = wp.float32(1.0)
    lvx = sensordata[i, ee_linvel_adr + 0]
    lvy = sensordata[i, ee_linvel_adr + 1]
    lvz = sensordata[i, ee_linvel_adr + 2]
    avx = sensordata[i, ee_angvel_adr + 0]
    avy = sensordata[i, ee_angvel_adr + 1]
    avz = sensordata[i, ee_angvel_adr + 2]
    twist_sq = (
        lvx * lvx
        + lvy * lvy
        + lvz * lvz
        + avx * avx
        + avy * avy
        + avz * avz
    )
    post_release_cost = released * wp.sqrt(twist_sq)

    # Approach-shaping terms only mean something pre-release -- once the
    # puck is gone the EE cannot act on any of them, so gate them off with
    # the same ``released`` flag rather than leave them fighting
    # ``post_release_cost`` for the sampler's attention. A no-op on the
    # terminal-cost instance, where these six are already built at weight 0.
    shaping = (
        w_attract * attract
        + w_align * align
        + w_lat_vel * lat_cost
        + w_box * box
        + w_ee_orient * ee_orient_cost
        + w_ee_height * ee_height_cost
        + w_arm_home * arm_home_cost
    )

    running = (
        w_pos * pos_cost
        + w_stop * stop_cost
        + (1.0 - released) * shaping
        + w_post_release * post_release_cost
    )
    cost[i] = cost[i] + scale * running


class _CurlingCost:
    """Evaluator: launches :func:`_curling_cost_kernel` per rollout step."""

    def __init__(
        self,
        adrs: tuple[int, int, int, int, int, int, int, int, int],
        w_pos: float,
        w_stop: float,
        w_attract: float,
        w_align: float,
        w_lat_vel: float,
        w_box: float,
        launch_box: tuple[float, float, float, float],
        w_ee_orient: float,
        w_ee_height: float,
        ee_z_target: float,
        arm_qposadr: np.ndarray,
        q_home: np.ndarray,
        w_arm_home: float,
        w_post_release: float,
        release_x: float,
        downrange_axis: int,
        ee_linvel_adr: int,
        ee_angvel_adr: int,
    ) -> None:
        self.adrs = adrs
        self.w_pos = w_pos
        self.w_stop = w_stop
        self.w_attract = w_attract
        self.w_align = w_align
        self.w_lat_vel = w_lat_vel
        self.w_box = w_box
        self.launch_box = launch_box
        self.w_ee_orient = w_ee_orient
        self.w_ee_height = w_ee_height
        self.ee_z_target = ee_z_target
        self.w_arm_home = w_arm_home
        self.w_post_release = w_post_release
        self.release_x = release_x
        self.downrange_axis = downrange_axis
        self.ee_linvel_adr = ee_linvel_adr
        self.ee_angvel_adr = ee_angvel_adr
        # Allocated at build time (under the engine's ScopedDevice), never in
        # accumulate() which runs during graph capture.
        self._arm_qposadr = wp.array(
            np.asarray(arm_qposadr, np.int32), dtype=wp.int32
        )
        self._q_home = wp.array(
            np.asarray(q_home, np.float32), dtype=wp.float32
        )

    def accumulate(self, d, cost: wp.array, scale: float) -> None:
        """Add this step's weighted running cost into ``cost``."""
        pa, va, t1, t2, t3, eez, eepos, blk, goal = self.adrs
        x_lo, x_hi, y_lo, y_hi = self.launch_box
        wp.launch(
            _curling_cost_kernel,
            dim=cost.shape[0],
            inputs=[
                d.sensordata,
                wp.int32(pa),
                wp.int32(va),
                wp.int32(t1),
                wp.int32(t2),
                wp.int32(t3),
                wp.int32(eez),
                wp.int32(eepos),
                wp.int32(blk),
                wp.int32(goal),
                wp.float32(self.w_pos),
                wp.float32(self.w_stop),
                wp.float32(self.w_attract),
                wp.float32(self.w_align),
                wp.float32(self.w_lat_vel),
                wp.float32(self.w_box),
                wp.float32(x_lo),
                wp.float32(x_hi),
                wp.float32(y_lo),
                wp.float32(y_hi),
                wp.float32(self.w_ee_orient),
                wp.float32(self.w_ee_height),
                wp.float32(self.ee_z_target),
                d.qpos,
                self._arm_qposadr,
                self._q_home,
                wp.float32(self.w_arm_home),
                wp.float32(self.w_post_release),
                wp.float32(self.release_x),
                wp.int32(self.downrange_axis),
                wp.int32(self.ee_linvel_adr),
                wp.int32(self.ee_angvel_adr),
                wp.float32(scale),
                cost,
            ],
        )


@wp.kernel
def _curling_ik_kernel(
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
    x_lo: wp.float32,
    x_hi: wp.float32,
    y_lo: wp.float32,
    y_hi: wp.float32,
    k_box: wp.float32,
    lam: wp.float32,
    controls: wp.array3d(dtype=wp.float32),
    t: wp.int32,
    ctrl: wp.array2d(dtype=wp.float32),
):
    """Damped-LS task-space IK per world, with the launch-box barrier.

    Device twin of ``control_map_host``. THE TWO MUST STAY IN SYNC: this one
    governs what the planner predicts, the host one what the viewer and the
    real robot do. Clamping in only one makes the planner's model wrong.
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

    # Launch-box velocity barrier. Saturating, and branch-free: x_lo <= x_hi
    # guarantees lo <= hi for every point, wherever the EE is. Inside the box
    # both bounds are slack and the command passes through untouched; at a
    # face the outward bound collapses to 0; outside, both point inward and it
    # is pulled back at k_box m/s per metre of excursion.
    vx = wp.clamp(
        controls[w, t, 0], k_box * (x_lo - point[0]), k_box * (x_hi - point[0])
    )
    vy = wp.clamp(
        controls[w, t, 1], k_box * (y_lo - point[1]), k_box * (y_hi - point[1])
    )

    # Twist: [vx, vy, z-regulation, world-frame point-down orientation error].
    e_world = rot_vec_quat(quat_sub(goal_quat, ee_quat), ee_quat)
    b = vec6()
    b[0] = vx
    b[1] = vy
    b[2] = z_target - point[2]
    b[3] = e_world[0]
    b[4] = e_world[1]
    b[5] = e_world[2]

    dq = damped_ls(jmat, b, lam)
    for c in range(7):
        ctrl[w, c] = dq[c]


class _CurlingIK:
    """Applier: launches :func:`_curling_ik_kernel` to write ``d.ctrl``."""

    def __init__(
        self,
        arm_dofadr: np.ndarray,
        ee_body: int,
        goal_quat: np.ndarray,
        z_target: float,
        launch_box: tuple[float, float, float, float],
        box_gain: float,
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
        self.launch_box = launch_box
        self.box_gain = float(box_gain)
        self.lam = float(lam)

    def apply(self, m, d, controls: wp.array, t: int) -> None:
        """Write ``d.ctrl`` from the staged 2-D twist at step ``t``."""
        x_lo, x_hi, y_lo, y_hi = self.launch_box
        wp.launch(
            _curling_ik_kernel,
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
                wp.float32(x_lo),
                wp.float32(x_hi),
                wp.float32(y_lo),
                wp.float32(y_hi),
                wp.float32(self.box_gain),
                wp.float32(self.lam),
                controls,
                wp.int32(t),
                d.ctrl,
            ],
        )


def _build_model(
    path,
    shape: str,
    scale: float,
    goal_xy: tuple[float, float] | None = None,
    surface_z_offset: float = 0.0,
    launch_box: tuple[float, float, float, float] = DEFAULT_LAUNCH_BOX,
    lane_xy: tuple[float, float] | None = None,
    lane_yaw: float = 0.0,
) -> tuple[mujoco.MjModel, mujoco.MjSpec]:
    """Graft ``shape`` onto the puck and seat everything on the lane.

    Only the ``block`` body is grafted. The house is a target *region*, not a
    ghost copy of the puck, so it is authored in the scene XML instead -- but
    the goal body is still lifted to the puck's resting height (which keeps
    the ``position`` sensor purely planar), so its discs are pushed back down
    onto the lane by the same offset.

    ``surface_z_offset`` shifts the ``ground`` body in world z before
    anything else is derived from it, so a physically thicker or thinner
    real surface (a different table, mat or ice sheet) moves the puck's
    resting height, the house and ``ee_z_target`` together, from one number.

    ``lane_xy``/``lane_yaw`` move/rotate the same ``ground`` body in world
    x/y -- the lane slab, collision plane and launch-region marker are all
    authored in its local frame, so rotating the body reorients them as a
    rigid unit with no further edits. Neither touches ``goal_xy`` or
    ``launch_box``, which the caller must move to match by hand.

    Returns ``(model, spec)``: the spec is kept so the viewer can extend a
    copy of it (e.g. mocap ghost bodies) without redoing composition.
    """
    spec = mujoco.MjSpec.from_file(str(path))
    attach_ee(spec, "pusher")  # pusher EE onto the shared arm's attach site

    shape_body = graft_shape(
        spec, "block", shape, scale, default=spec.find_default("stone")
    )

    # "ground" is a PLANE (see the scene XML for why), so its contact surface
    # is its pos -- there is no half-thickness to add the way a box slab would
    # need. The visual slab is a separate geom and is not consulted here.
    ground_body = spec.body("ground")
    ground_pos = list(ground_body.pos)
    if lane_xy is not None:
        ground_pos[0], ground_pos[1] = float(lane_xy[0]), float(lane_xy[1])
    ground_pos[2] += surface_z_offset
    ground_body.pos = ground_pos
    if lane_yaw:
        ground_body.quat = [
            math.cos(lane_yaw / 2.0), 0.0, 0.0, math.sin(lane_yaw / 2.0)
        ]
    ground_geom = spec.geom("ground")
    lane_top_z = ground_body.pos[2] + ground_geom.pos[2]
    z = lane_top_z - shape_min_z(shape_body, scale)

    # Keep the visual launch-region marker in sync with the barrier box, so
    # overriding launch_box needs no scene-XML edit. launch_box is world-
    # axis-aligned; region.pos is a LOCAL offset, so under lane_yaw != 0 it
    # must be un-rotated into the body frame, and the geom needs a
    # counter-quat for its size to still read as world-x/y half-extents.
    x_lo, x_hi, y_lo, y_hi = launch_box
    ground_quat = np.asarray(ground_body.quat, dtype=np.float64)
    ground_quat_conj = ground_quat * np.array([1.0, -1.0, -1.0, -1.0])
    world_offset = np.array([
        (x_lo + x_hi) / 2 - ground_pos[0],
        (y_lo + y_hi) / 2 - ground_pos[1],
        0.0,
    ])
    local_offset = quat_rotate(ground_quat_conj, world_offset)
    region = spec.geom("launch_region")
    region.pos = [local_offset[0], local_offset[1], region.pos[2]]
    region.size = [(x_hi - x_lo) / 2, (y_hi - y_lo) / 2, region.size[2]]
    region.quat = list(ground_quat_conj)

    block_pos = list(spec.body("block").pos)
    block_pos[2] = z
    spec.body("block").pos = block_pos

    goal_pos = list(spec.body("goal").pos)
    goal_pos[2] = z
    if goal_xy is not None:
        goal_pos[0], goal_pos[1] = float(goal_xy[0]), float(goal_xy[1])
    spec.body("goal").pos = goal_pos
    for name in ("house", "button"):
        geom = spec.geom(name)
        gpos = list(geom.pos)
        gpos[2] = lane_top_z + 0.0005 - z
        geom.pos = gpos

    key = spec.key("home")
    qpos = list(key.qpos)
    qpos[0:3] = block_pos
    key.qpos = qpos

    return spec.compile(), spec


class CurlingFr3(Task):
    """Shoot a puck from a launch box to a target region down the lane."""

    def __init__(
        self,
        *,
        trace_sites: Sequence[str] | None = ("ee_site",),
        shape: str = "circle",
        scale: float = 1.0,
        goal_xy: tuple[float, float] | None = None,
        surface_z_offset: float = 0.0,
        launch_box: tuple[
            float, float, float, float
        ] = DEFAULT_LAUNCH_BOX,
        lane_xy: tuple[float, float] | None = None,
        lane_yaw: float = 0.0,
        downrange_axis: int = 0,
        box_gain: float = 10.0,
        house_radius: float = 0.08,
        rest_speed: float = 0.02,
        max_lin_vel: float = 0.8,
        w_pos: float = 50.0,
        w_stop: float = 20.0,
        w_attract: float = 0.15,
        w_align: float = 0.3,
        w_lat_vel: float = 2.0,
        w_box: float = 2.0,
        w_ee_orient: float = 0.0,
        w_ee_height: float = 0.0,
        w_arm_home: float = 0.0,
        w_post_release: float = 0.0,
        terminal_scale: float = 20.0,
        arm_kv: float | None = None,
        model_config: ModelConfig | None = None,
        goal_drift: GoalDrift | None = None,
    ) -> None:
        """Build the task.

        Args:
            trace_sites: Sites the viewer draws rollout traces for.
            shape: Puck geometry from the shape library; see :data:`SHAPES`.
            scale: Uniform scale applied to the shape fragment.
            goal_xy: Override the house's planar position (m, world).
            surface_z_offset: Shift the lane (and everything derived from
                it -- puck resting height, house height, ``ee_z_target``) in
                world z, to match a physically thicker or thinner real
                surface. 0.0 leaves the modelled lane height unchanged.
            launch_box: ``(x_lo, x_hi, y_lo, y_hi)`` the EE is confined to,
                in world metres. The ``launch_region`` visual geom in the
                scene XML is synced to this automatically.
            lane_xy: Override the ``ground`` body's world (x, y), moving the
                lane's slab, collision plane and launch-region visuals as a
                rigid unit. **Does not** move ``launch_box``, ``goal_xy`` or
                the scenario bank's start states -- those are independent and
                the caller must keep them in sync (see ``--rotate90`` in
                ``examples/simple/curling.py``).
            lane_yaw: Rotate the ``ground`` body about world z (rad). The
                lane's geoms are authored in the body's local frame, so this
                reorients all of them together with no scene-XML edit.
            downrange_axis: World axis (0 = x, 1 = y) the release gate is
                measured along. Pair with ``lane_yaw=pi/2`` and a matching
                rotated ``launch_box``/``goal_xy``/scenario bank for a
                90 deg-rotated lane.
            box_gain: Barrier pull-back rate (1/s); 10.0 is a ~0.1 s time
                constant. The barrier is *saturating*, not a wall -- the clamp
                only zeroes the outward command and cannot apply an impulse,
                so an EE arriving at full speed overshoots by a few mm before
                being pulled back. Raise to tighten, at the cost of a stiffer
                IK command near the faces.
            house_radius: Scoring radius of the house, for ``task_success``.
            rest_speed: Speed below which the puck counts as stopped (m/s).
            max_lin_vel: Sampling bound on the EE twist (m/s). Do not raise
                past 0.8 without re-running ``scripts/curling/range_check.py``
                -- above v0 = 1.0 m/s the puck's contact degrades silently.
            w_pos: Weight on planar puck-to-house distance.
            w_stop: Weight on puck speed; applied in the terminal cost only.
            w_attract: Weight on the EE-to-rim-site term.
            w_align: Weight on the shot-alignment term.
            w_lat_vel: Weight on the puck's lateral (off-shot-axis) velocity,
                penalizing side-slip from a glancing push without fighting the
                forward component. Zero in the terminal cost, like
                ``w_attract``/``w_align``/``w_box``: a push-quality shaping
                term, not a property of the finished shot.
            w_box: Weight on the soft launch-box margin (anti-bias only; the
                barrier is what actually enforces the box).
            w_ee_orient: Weight on the EE point-down term.
            w_ee_height: Weight on the EE height term.
            w_arm_home: Weight on the arm-home posture term.
            w_post_release: Weight on the EE's own Cartesian twist once the
                puck's trailing edge clears the launch box (``release_x``
                includes the puck radius). Past that the shot is open-loop, so
                nothing is gained by the sampler chasing it. Zero in the
                terminal cost.
            terminal_scale: Multiplier on ``w_pos`` in the terminal cost.
            arm_kv: Overrides the arm's velocity-actuator gain on this task's
                own compiled model only. ``None`` leaves the XML's gain
                untouched. See ``BalanceFr3.arm_kv``.
            model_config: Solver options, merged over :data:`_BASELINE`.
            goal_drift: Optional moving-goal schedule.
        """
        if shape not in SHAPES:
            raise ValueError(f"Unknown shape for curling: {shape!r}")
        x_lo, x_hi, y_lo, y_hi = (float(v) for v in launch_box)
        if x_lo >= x_hi or y_lo >= y_hi:
            raise ValueError(
                f"launch_box must be (x_lo<x_hi, y_lo<y_hi): {launch_box}"
            )
        self.launch_box = (x_lo, x_hi, y_lo, y_hi)
        self.downrange_axis = int(downrange_axis)
        self.box_gain = float(box_gain)
        self.house_radius = float(house_radius)
        self.rest_speed = float(rest_speed)
        self.shape = shape
        self.scale = float(scale)

        self.w_pos = float(w_pos)
        self.w_stop = float(w_stop)
        self.w_attract = float(w_attract)
        self.w_align = float(w_align)
        self.w_lat_vel = float(w_lat_vel)
        self.w_box = float(w_box)
        self.w_ee_orient = float(w_ee_orient)
        self.w_ee_height = float(w_ee_height)
        self.w_arm_home = float(w_arm_home)
        self.w_post_release = float(w_post_release)
        self.terminal_scale = float(terminal_scale)

        path = MODELS_DIR / "curling" / "scene.xml"
        mj_model, mj_spec = _build_model(
            path,
            shape,
            scale,
            goal_xy,
            surface_z_offset,
            self.launch_box,
            lane_xy,
            lane_yaw,
        )

        if arm_kv is not None:
            # Patches this task's own compiled mj_model only -- fr3_arm.xml
            # on disk is untouched, and every other FR3 task compiles its
            # own separate model from it, so they're unaffected.
            mj_model.actuator_gainprm[:, 0] = arm_kv
            mj_model.actuator_biasprm[:, 2] = -arm_kv

        self.goal_xy = goal_xy
        self.surface_z_offset = float(surface_z_offset)
        self.lane_yaw = float(lane_yaw)

        super().__init__(
            mj_model,
            mj_spec=mj_spec,
            trace_sites=trace_sites,
            nu=2,
            ctrl_limits={
                "u_min": np.array([-max_lin_vel, -max_lin_vel]),
                "u_max": np.array([max_lin_vel, max_lin_vel]),
            },
            endpoint_body="block",
            model_config=_BASELINE.merged_with(model_config),
            contact_budget=curling_contact_budget(shape),
            goal_drift=goal_drift,
        )

        self._bind_sensors()
        self._bind_arm(mj_model)

    def _bind_sensors(self) -> None:
        """Cache the sensordata / qpos addresses the cost kernel reads."""
        self._adr_pos = self._sensor_adr("position")
        self._adr_vel = self._sensor_adr("block_linvel")
        self._adr_t1 = self._sensor_adr("ee_t1")
        self._adr_t2 = self._sensor_adr("ee_t2")
        self._adr_t3 = self._sensor_adr("ee_t3")
        self._adr_ee_zaxis = self._sensor_adr("ee_zaxis")
        self._adr_ee_pos = self._sensor_adr("ee_position_world")
        self._adr_blk = self._sensor_adr("position_world")
        self._adr_goal = self._sensor_adr("goal_position_world")
        self._adr_ee_linvel = self._sensor_adr("ee_linvel")
        self._adr_ee_angvel = self._sensor_adr("ee_angvel")

        m = self.mj_model
        self._block_free_adr = self._joint_qposadr("block")
        self._block_dofadr = int(m.jnt_dofadr[m.joint("block").id])

        # Bounding-sphere radius, so `release_x` gates on the puck's
        # TRAILING edge clearing the box, not its centre -- matching the
        # real-robot release latch in run_planner_node_curling.py (which
        # pads the same way, for the same reason).
        puck = m.body("block")
        g0, gn = int(puck.geomadr[0]), int(puck.geomnum[0])
        self.puck_radius = float(np.max(m.geom_rbound[g0 : g0 + gn]))
        # `release_x` is measured along `self.downrange_axis` (0 = world-x,
        # the default lane heading; 1 = world-y, an experimental 90
        # deg-rotated lane) -- despite the name, it is not always literally
        # world-x. `launch_box` is always `(x_lo, x_hi, y_lo, y_hi)`, so its
        # "hi" bound on that axis is index 1 (x_hi) or 3 (y_hi).
        hi_idx = 1 if self.downrange_axis == 0 else 3
        self.release_x = self.launch_box[hi_idx] + self.puck_radius

    def _bind_arm(self, mj_model: mujoco.MjModel) -> None:
        """Cache arm/IK handles and the puck-derived push height.

        ``solve_task_pose_ik`` keys on these attribute names, so naming them
        exactly like the other FR3 tasks is what gives this task host-side IK
        placement for free.
        """
        self.ee_body_id = int(mj_model.body("ee_frame").id)
        jids = [mj_model.joint(f"fr3_joint{i}").id for i in range(1, 8)]
        self.arm_qposadr = mj_model.jnt_qposadr[jids].astype(int)
        self.arm_dofadr = mj_model.jnt_dofadr[jids].astype(int)
        self.joint_limits = mj_model.jnt_range[jids]  # (7, 2)
        self._jacp = np.zeros((3, mj_model.nv))
        self._jacr = np.zeros((3, mj_model.nv))
        self.goal_quat_ee = np.array([0.0, 0.7071, 0.7071, 0.0])  # point-down
        self.ik_damping = 1e-3

        kid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_KEY, "home")
        self.q_home = (
            mj_model.key_qpos[kid][self.arm_qposadr] if kid != -1 else None
        )

        # The static "home" keyframe was solved for the DEFAULT (lane_yaw=0)
        # box location -- ``solve_task_pose_ik``/``_solve_ik`` fall back to
        # ``q_home`` as the IK seed for every per-episode placement
        # (``set_initial_state``), so a stale seed left pointing at the old
        # box gets used to reach a target possibly on the opposite side of
        # the workspace once ``lane_yaw`` rotates the box elsewhere. fr3_joint1
        # (index 0) rotates about the base's world z-axis, so adding
        # `lane_yaw` re-aims the arm's IK seed the same way the box moved.
        # Not an exact solution -- goal_quat_ee is not itself rotated -- just
        # a better seed. Clipped to the joint's range so an extreme lane_yaw
        # cannot seed the IK outside it.
        if self.q_home is not None and self.lane_yaw:
            self.q_home = self.q_home.copy()
            self.q_home[0] = np.clip(
                self.q_home[0] + self.lane_yaw,
                self.joint_limits[0, 0],
                self.joint_limits[0, 1],
            )

        # Push through the puck's centre of mass, not push_fr3's fixed 0.039.
        # Every shape here is vertically centred on its body origin, so the
        # resting body height IS the CoM height, and it differs per shape.
        # Pushing above it applies a tipping torque that drives the leading
        # edge into the lane -- the degraded contact configs/numerics/
        # curling.yaml's header is about.
        self.ee_z_target = float(
            mj_model.key_qpos[kid][self._block_free_adr + 2]
            if kid != -1
            else mj_model.body("block").pos[2]
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
        """Puck pose layout (always a free joint here)."""
        return ObjectPose(kind="free", adr=self._block_free_adr)

    @property
    def contact_probes(self) -> tuple[ContactProbe, ...]:
        """Pusher-puck and puck-lane contact.

        The pusher side covers both the ``ee`` sphere and the arm box above
        it, since either can be what actually touches the puck.
        """
        puck = body_geom_ids(self.mj_model, "block")
        return (
            ContactProbe(
                name="pusher_puck",
                geoms_a=body_geom_ids(self.mj_model, "ee_frame", "pusher"),
                geoms_b=puck,
                bit=0,
            ),
            ContactProbe(
                name="puck_lane",
                geoms_a=puck,
                geoms_b=body_geom_ids(self.mj_model, "ground"),
                bit=1,
            ),
        )

    def task_success(self, mj_data: mujoco.MjData) -> bool:
        """True once the puck is stopped inside the house.

        Both halves matter: a puck still travelling through the house has not
        scored, it is mid-flight.
        """
        s = mj_data.sensordata
        dist = float(
            np.linalg.norm(s[self._adr_pos : self._adr_pos + 2])
        )
        speed = float(
            np.linalg.norm(s[self._adr_vel : self._adr_vel + 3])
        )
        return dist <= self.house_radius and speed <= self.rest_speed

    @property
    def _cost_adrs(
        self,
    ) -> tuple[int, int, int, int, int, int, int, int, int]:
        return (
            self._adr_pos,
            self._adr_vel,
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

    def build_cost_kernel(self) -> _CurlingCost:  # noqa: D102
        q_home, w_arm_home = self._arm_home_ref()
        return _CurlingCost(
            self._cost_adrs,
            self.w_pos,
            0.0,  # w_stop: terminal only -- see build_terminal_cost_kernel
            self.w_attract,
            self.w_align,
            self.w_lat_vel,
            self.w_box,
            self.launch_box,
            self.w_ee_orient,
            self.w_ee_height,
            self.ee_z_target,
            self.arm_qposadr,
            q_home,
            w_arm_home,
            self.w_post_release,
            self.release_x,
            self.downrange_axis,
            self._adr_ee_linvel,
            self._adr_ee_angvel,
        )

    def build_terminal_cost_kernel(self) -> _CurlingCost:  # noqa: D102
        # Where the shot is actually judged: distance scaled up, plus the
        # at-rest term. Attract/align/lat_vel/box/posture/post_release are
        # per-step regularizers that shape the approach, not properties of a
        # finished shot, so they are zeroed here (w_stop already charges
        # terminal residual speed, lateral component included, so
        # re-charging it via w_lat_vel here would double-count).
        s = self.terminal_scale
        q_home, _ = self._arm_home_ref()
        return _CurlingCost(
            self._cost_adrs,
            s * self.w_pos,
            self.w_stop,
            0.0,
            0.0,
            0.0,
            0.0,
            self.launch_box,
            0.0,
            0.0,
            self.ee_z_target,
            self.arm_qposadr,
            q_home,
            0.0,
            0.0,
            self.release_x,
            self.downrange_axis,
            self._adr_ee_linvel,
            self._adr_ee_angvel,
        )

    def cost_components(self, mj_data: mujoco.MjData) -> dict[str, float]:
        """Host twin of the running cost, split per weighted term."""
        s = mj_data.sensordata
        pos_cost = float(np.linalg.norm(s[self._adr_pos : self._adr_pos + 2]))
        stop_cost = float(np.linalg.norm(s[self._adr_vel : self._adr_vel + 3]))

        attract = 0.0
        for adr in (self._adr_t1, self._adr_t2, self._adr_t3):
            attract += float(np.linalg.norm(s[adr : adr + 3]))

        blk = s[self._adr_blk : self._adr_blk + 2]
        r = s[self._adr_ee_pos : self._adr_ee_pos + 2] - blk
        g = s[self._adr_goal : self._adr_goal + 2] - blk
        rn, gn = float(np.linalg.norm(r)), float(np.linalg.norm(g))
        align = 0.0
        if rn > 1e-6 and gn > 1e-6:
            align = float(np.dot(r, g)) / (rn * gn) + 1.0

        vx = float(s[self._adr_vel + 0])
        vy = float(s[self._adr_vel + 1])
        lat_cost = 0.0
        if gn > 1e-6:
            lat_cost = abs(vx * g[1] - vy * g[0]) / gn

        x_lo, x_hi, y_lo, y_hi = self.launch_box
        ee_x = float(s[self._adr_ee_pos + 0])
        ee_y = float(s[self._adr_ee_pos + 1])
        box = (
            max(ee_x - x_hi, 0.0)
            + max(x_lo - ee_x, 0.0)
            + max(ee_y - y_hi, 0.0)
            + max(y_lo - ee_y, 0.0)
        )

        z = s[self._adr_ee_zaxis : self._adr_ee_zaxis + 3]
        ee_orient_cost = float(
            math.sqrt(z[0] ** 2 + z[1] ** 2 + (z[2] + 1.0) ** 2)
        )
        ee_height_cost = abs(float(s[self._adr_ee_pos + 2]) - self.ee_z_target)

        if self.q_home is None:
            arm_home_cost = 0.0
        else:
            dq = mj_data.qpos[self.arm_qposadr] - np.asarray(self.q_home)
            arm_home_cost = float(np.linalg.norm(dq))

        released = (
            float(s[self._adr_blk + self.downrange_axis]) > self.release_x
        )
        post_release_cost = 0.0
        if released:
            twist = np.concatenate(
                [
                    s[self._adr_ee_linvel : self._adr_ee_linvel + 3],
                    s[self._adr_ee_angvel : self._adr_ee_angvel + 3],
                ]
            )
            post_release_cost = float(np.linalg.norm(twist))

        # Approach-shaping terms gate off once released, matching the
        # kernel: they only mean something pre-release, and left on they'd
        # fight post_release_cost for the sampler's attention.
        gate = 0.0 if released else 1.0

        # w_stop is terminal-only, so it is absent from this running twin.
        out = {
            "pos": self.w_pos * pos_cost,
            "speed": stop_cost,
            "attract": gate * self.w_attract * attract,
            "align": gate * self.w_align * align,
            "lat_vel": gate * self.w_lat_vel * lat_cost,
            "box": gate * self.w_box * box,
            "ee_orient": gate * self.w_ee_orient * ee_orient_cost,
            "ee_height": gate * self.w_ee_height * ee_height_cost,
            "arm_home": gate * self.w_arm_home * arm_home_cost,
            "post_release": self.w_post_release * post_release_cost,
        }
        out["total"] = sum(v for k, v in out.items() if k != "speed")
        return out

    def pose_error(self, mj_data: mujoco.MjData) -> tuple[float, float]:
        """Raw planar puck-to-house distance (m) and puck speed (m/s).

        The pair a curling shot is actually judged on, unweighted -- weighted
        cost is the wrong unit for logging (a component can be traded against
        another without the total moving). Note the second element is a speed,
        not the rotation error the other FR3 tasks return here: a disc's yaw
        does not matter, and whether it has stopped does.
        """
        s = mj_data.sensordata
        dist = float(np.linalg.norm(s[self._adr_pos : self._adr_pos + 2]))
        speed = float(np.linalg.norm(s[self._adr_vel : self._adr_vel + 3]))
        return dist, speed

    def running_cost_reference(
        self, mj_data: mujoco.MjData, u: object = None
    ) -> float:
        """Eager numpy running cost (sum of :meth:`cost_components`)."""
        return self.cost_components(mj_data)["total"]

    def terminal_cost_reference(self, mj_data: mujoco.MjData) -> float:
        """Eager numpy terminal cost (distance + at-rest, both reweighted)."""
        c = self.cost_components(mj_data)
        return self.terminal_scale * c["pos"] + self.w_stop * c["speed"]

    def build_control_map_kernel(self) -> Any | None:  # noqa: D102
        return _CurlingIK(
            self.arm_dofadr,
            self.ee_body_id,
            self.goal_quat_ee,
            self.ee_z_target,
            self.launch_box,
            self.box_gain,
            self.ik_damping,
        )

    def clamp_to_launch_box(
        self, ee_xy: np.ndarray, u: np.ndarray
    ) -> np.ndarray:
        """Host twin of the kernel's launch-box velocity barrier.

        Args:
            ee_xy: Current EE planar position (m, world).
            u: Commanded 2-D EE twist (m/s).

        Returns:
            The twist with each axis saturated so the EE cannot leave the box.
        """
        x_lo, x_hi, y_lo, y_hi = self.launch_box
        k = self.box_gain
        return np.array(
            [
                np.clip(u[0], k * (x_lo - ee_xy[0]), k * (x_hi - ee_xy[0])),
                np.clip(u[1], k * (y_lo - ee_xy[1]), k * (y_hi - ee_xy[1])),
            ]
        )

    def control_map_host(
        self, mj_data: mujoco.MjData, u: np.ndarray
    ) -> np.ndarray:
        """Map a 2-D EE twist to the 7 joint-velocity actuators (host).

        Damped least-squares IK on the EE Jacobian, regulating z to launch
        height and orientation to point down, after the launch-box barrier.
        """
        m = self.mj_model
        mujoco.mj_jacBody(m, mj_data, self._jacp, self._jacr, self.ee_body_id)
        jac = np.vstack([self._jacp, self._jacr])[:, self.arm_dofadr]  # (6, 7)

        ee_pos = mj_data.xpos[self.ee_body_id]
        ee_quat = mj_data.xquat[self.ee_body_id]
        v = self.clamp_to_launch_box(ee_pos[:2], np.asarray(u, dtype=float))
        e_rot = self._orientation_error_world(ee_quat)
        twist = np.array(
            [
                v[0],
                v[1],
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
        clamp_ee: bool = True,
    ) -> None:
        """Overlay a start pose on ``mj_data`` (host), then forward.

        ``ee_pos`` is clamped into the launch box before the IK runs: the
        barrier would otherwise yank the arm on the first step of the
        episode, which reads as a planner failure rather than a bad start
        state.

        Args:
            mj_data: MjData to mutate; caller has loaded a keyframe first.
            ee_pos: Desired EE position (m, world); triggers IK.
            ee_quat: Desired EE quaternion [w, x, y, z].
            block_xy: Desired puck planar position (x, y) in world metres.
            block_yaw: Desired puck yaw around world Z, radians.
            clamp_ee: Keep ``ee_pos`` inside the launch box. Leave it on for
                anything that starts an episode. Turn it off only to place
                the EE somewhere deliberately outside the box for
                measurement -- ``scripts/probing/make_scenarios.py`` bisects
                the standoff from the puck centre outwards, and a clamp
                partway through would make it search a flattened function
                and report a gap it never actually measured.
        """
        if ee_pos is not None:
            ee_pos = np.asarray(ee_pos, dtype=np.float64).copy()
            x_lo, x_hi, y_lo, y_hi = self.launch_box
            clamped = np.array(
                [np.clip(ee_pos[0], x_lo, x_hi), np.clip(ee_pos[1], y_lo, y_hi)]
            )
            if clamp_ee and not np.allclose(clamped, ee_pos[:2]):
                print(
                    "WARNING: CurlingFr3 start EE "
                    f"{ee_pos[:2].tolist()} is outside the launch box "
                    f"{self.launch_box}; clamped to {clamped.tolist()}."
                )
                ee_pos[:2] = clamped
            tgt_quat = (
                self.goal_quat_ee
                if ee_quat is None
                else np.asarray(ee_quat, dtype=np.float64)
            )
            q_sol, converged = self._solve_ik(ee_pos, tgt_quat)
            if not converged:
                print(
                    "WARNING: CurlingFr3 IK did not converge for "
                    f"ee_pos={ee_pos.tolist()}; using best-effort solution."
                )
            mj_data.qpos[self.arm_qposadr] = q_sol
        elif ee_quat is not None:
            raise ValueError("ee_quat given without ee_pos; specify ee_pos too")

        adr = self._block_free_adr
        if block_xy is not None:
            mj_data.qpos[adr : adr + 2] = np.asarray(
                block_xy, dtype=np.float64
            )
        if block_yaw is not None:
            half = 0.5 * float(block_yaw)
            mj_data.qpos[adr + 3 : adr + 7] = [
                np.cos(half),
                0.0,
                0.0,
                np.sin(half),
            ]

        mujoco.mj_forward(self.mj_model, mj_data)
