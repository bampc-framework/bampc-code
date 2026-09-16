"""FlipFr3: an FR3 pusher tips a free box onto a target face.

The block starts **lying on its side** (``_LYING_QUAT``, baked into the
compiled default and the "home" keyframe by ``_build_model``) -- the pusher
must genuinely tip it upright, not just nudge an already-standing box.

Unlike ``PushFr3``/``BalanceFr3``, which sample 2 of 6 twist rows and
regulate the rest to a constant EE pose, ``sampling_space="task"`` here is a
full un-regulated SE(3) twist: this task's push height and orientation vary
per scenario, so there is no constant to regulate the other rows to.
``"joint"`` (the default) samples the 7 joint velocities directly.

``w_upright`` exists because the geodesic term alone is nearly flat almost
everywhere: tipping the box upright on the *wrong* face barely reduces the
distance to the one target quaternion, so the planner gets no credit for a
real step toward success. The upright term scores "standing on any face"
regardless of heading; ``w_orient`` still owns the fine final alignment.

A static wall behind the box (``wall=True``, the default) gives the push
something to react against -- in free space the pusher touches the box and
backs off without sustaining it. ``wall=False`` is a comparison fallback.
Placement and size are fixed constants (``_WALL_POS``/``_WALL_SIZE``).
"""

from __future__ import annotations

from typing import Literal

import mujoco
import numpy as np
import warp as wp
from mujoco_warp._src.support import contact_force_fn
from mujoco_warp._src.types import vec5

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
    solve_task_pose_ik,
)
from bampc.task.common.shapes import (
    flip_fr3_contact_budget,
    graft_shape,
    shape_center,
    shape_min_z,
)

# Fallback solver options, used only when the caller names none. The tuned
# values live with the caller (configs/numerics/flip_fr3.yaml -- forked from
# Push-FR3's "free" profile rather than shared with it, since Flip's box-vs-
# wall impact needs its own GPU-stability tuning; see that file's header).
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

# Wall placement/size -- fixed constants, tuned by hand against the frozen
# scenario (configs/scenarios/flip_fr3.yaml). Edit these directly; there is
# no per-call override.
_WALL_POS = (0.55, 0.33, 0.0)  # m, world xyz (body frame is axis-aligned)
_WALL_SIZE = (0.3775, 0.01, 0.1275)  # m, half-extents (thin, wide, tall)

# Initial block orientation: lying on its largest face, long axis along world
# y to match this scenario's push direction (configs/scenarios/flip_fr3.yaml
# offsets ee_pos from block_xy only in y), so a push along +/-y tips it about
# a pivot along world x. Composed from two 90 deg turns -- about local y, then
# about world z -- and verified numerically, NOT hand-derived. If the push
# direction changes, recompute from those two quats rather than editing these
# numbers by hand.
_LYING_QUAT = (0.5, 0.5, -0.5, 0.5)  # [w, x, y, z]


@wp.kernel
def _flip_fr3_cost_kernel(
    sensordata: wp.array2d(dtype=wp.float32),
    ctrl: wp.array2d(dtype=wp.float32),
    orient_adr: wp.int32,
    zaxis_adr: wp.int32,
    t1_adr: wp.int32,
    t2_adr: wp.int32,
    t3_adr: wp.int32,
    eez_adr: wp.int32,
    w_orient: wp.float32,
    w_upright: wp.float32,
    w_attract: wp.float32,
    w_ctrl: wp.float32,
    w_ee_orient: wp.float32,
    qpos: wp.array2d(dtype=wp.float32),
    arm_qposadr: wp.array(dtype=wp.int32),
    q_home: wp.array(dtype=wp.float32),
    w_arm_home: wp.float32,
    scale: wp.float32,
    cost: wp.array(dtype=wp.float32),
):
    """Block->goal orientation error + upright + EE->block attract + arm/ctrl.

    No position/contact/align/safety shaping terms -- this task rewards the
    box being on the target face, not the tipping motion or approach.
    """
    i = wp.tid()

    qw = sensordata[i, orient_adr + 0]
    qx = sensordata[i, orient_adr + 1]
    qy = sensordata[i, orient_adr + 2]
    qz = sensordata[i, orient_adr + 3]
    qv = wp.sqrt(qx * qx + qy * qy + qz * qz)
    orient_cost = 2.0 * wp.atan2(qv, wp.abs(qw))

    # 0 when the block's own z-axis points straight up (standing on any
    # face), up to 2 upside-down -- yaw-independent, unlike orient_cost.
    upright_cost = 1.0 - sensordata[i, zaxis_adr + 2]

    b1x = sensordata[i, t1_adr + 0]
    b1y = sensordata[i, t1_adr + 1]
    b1z = sensordata[i, t1_adr + 2]
    b2x = sensordata[i, t2_adr + 0]
    b2y = sensordata[i, t2_adr + 1]
    b2z = sensordata[i, t2_adr + 2]
    b3x = sensordata[i, t3_adr + 0]
    b3y = sensordata[i, t3_adr + 1]
    b3z = sensordata[i, t3_adr + 2]
    attract_cost = (
        wp.sqrt(b1x * b1x + b1y * b1y + b1z * b1z)
        + wp.sqrt(b2x * b2x + b2y * b2y + b2z * b2z)
        + wp.sqrt(b3x * b3x + b3y * b3y + b3z * b3z)
    )

    ctrl_cost = wp.float32(0.0)
    for k in range(ctrl.shape[1]):
        ctrl_cost += ctrl[i, k] * ctrl[i, k]

    # EE point-down: world z-axis of the pusher vs (0, 0, -1), same
    # mechanism as PushFr3's w_ee_orient. Distance grows with tilt
    # (roll/pitch) but is invariant to spin about the pusher's own axis.
    zx = sensordata[i, eez_adr + 0]
    zy = sensordata[i, eez_adr + 1]
    zz = sensordata[i, eez_adr + 2]
    ee_orient_cost = wp.sqrt(zx * zx + zy * zy + (zz + 1.0) * (zz + 1.0))

    arm_sq = wp.float32(0.0)
    for k in range(7):
        dq = qpos[i, arm_qposadr[k]] - q_home[k]
        arm_sq += dq * dq
    arm_home_cost = wp.sqrt(arm_sq)

    running = (
        w_orient * orient_cost
        + w_upright * upright_cost
        + w_attract * attract_cost
        + w_ctrl * ctrl_cost
        + w_ee_orient * ee_orient_cost
        + w_arm_home * arm_home_cost
    )
    cost[i] = cost[i] + scale * running


@wp.func
def _is_pusher_block(
    pair: wp.vec2i,
    ee_geom: wp.array(dtype=wp.int32),
    block_geom: wp.array(dtype=wp.int32),
) -> wp.int32:
    """1 when this contact pairs any pusher geom with a block geom.

    ``ee_geom`` is the whole pusher body group, not just the tip, so pushing
    the block with the shaft cannot dodge the force penalty.
    """
    g0 = pair[0]
    g1 = pair[1]
    g0_is_ee = wp.int32(0)
    g1_is_ee = wp.int32(0)
    for k in range(ee_geom.shape[0]):
        if g0 == ee_geom[k]:
            g0_is_ee = 1
        if g1 == ee_geom[k]:
            g1_is_ee = 1
    other = wp.int32(-1)
    if g0_is_ee == 1 and g1_is_ee == 0:
        other = g1
    elif g1_is_ee == 1 and g0_is_ee == 0:
        other = g0
    else:
        return wp.int32(0)
    for k in range(block_geom.shape[0]):
        if other == block_geom[k]:
            return wp.int32(1)
    return wp.int32(0)


@wp.kernel
def _flip_fr3_pusher_force_kernel(
    opt_cone: wp.int32,
    con_frame: wp.array(dtype=wp.mat33),
    con_friction: wp.array(dtype=vec5),
    con_dim: wp.array(dtype=wp.int32),
    con_efc_address: wp.array2d(dtype=wp.int32),
    con_worldid: wp.array(dtype=wp.int32),
    con_geom: wp.array(dtype=wp.vec2i),
    efc_force: wp.array2d(dtype=wp.float32),
    njmax: wp.int32,
    nacon: wp.array(dtype=wp.int32),
    ee_geom: wp.array(dtype=wp.int32),
    block_geom: wp.array(dtype=wp.int32),
    force_safe: wp.float32,
    peak_excess: wp.array(dtype=wp.float32),
):
    """Track the trajectory's peak pusher<->block force excess.

    Unlike PegFr3's wall-force term (any contact there is unwanted scraping),
    tipping the box *requires* sustained force -- so integrating ``excess**2``
    every step would punish how long the push lasted rather than how hard it
    got, and made the planner avoid contact entirely. Tracking the peak and
    charging it once at the terminal step matches what hardware safety needs
    (a spike tripping the collision reflex).

    Launched over the whole contact pool: MJWarp keeps contacts in one global
    ``naconmax`` array tagged by ``worldid``, so this is a scatter
    (``wp.atomic_max``), not a per-world accumulate.
    """
    c = wp.tid()
    if c >= nacon[0]:
        return
    if _is_pusher_block(con_geom[c], ee_geom, block_geom) == 0:
        return
    worldid = con_worldid[c]
    force = contact_force_fn(
        opt_cone,
        con_frame,
        con_friction,
        con_dim,
        con_efc_address,
        efc_force,
        njmax,
        nacon,
        worldid,
        c,
        False,  # contact frame: force[0] is the normal component
    )
    excess = wp.max(wp.abs(force[0]) - force_safe, 0.0)
    wp.atomic_max(peak_excess, worldid, excess)


@wp.kernel
def _flip_fr3_peak_force_kernel(
    peak_excess: wp.array(dtype=wp.float32),
    w_force: wp.float32,
    cost: wp.array(dtype=wp.float32),
):
    """Charge ``w_force * peak_excess**2`` once, then reset.

    ``peak_excess`` is shared between the running and terminal cost
    evaluators (``FlipFr3._peak_excess``, allocated once in ``set_nworld``)
    and persists across ``H`` captured steps -- so it must be zeroed back to
    0 here, at the one point per rollout it's read, rather than by a
    separate reset launch.
    """
    w = wp.tid()
    e = peak_excess[w]
    cost[w] = cost[w] + w_force * e * e
    peak_excess[w] = 0.0


@wp.kernel
def _flip_fr3_ik_kernel(
    body_parentid: wp.array(dtype=wp.int32),
    body_rootid: wp.array(dtype=wp.int32),
    dof_bodyid: wp.array(dtype=wp.int32),
    body_isdofancestor: wp.array2d(dtype=wp.int32),
    subtree_com: wp.array2d(dtype=wp.vec3),
    cdof: wp.array2d(dtype=wp.spatial_vector),
    xpos: wp.array2d(dtype=wp.vec3),
    arm_dof: wp.array(dtype=wp.int32),
    ee_body: wp.int32,
    v_max: wp.float32,
    w_max: wp.float32,
    lam: wp.float32,
    controls: wp.array3d(dtype=wp.float32),
    t: wp.int32,
    ctrl: wp.array2d(dtype=wp.float32),
):
    """Damped-LS task-space IK per world: 6-D twist -> 7 joint velocities.

    All six rows are commanded -- nothing is regulated, unlike Push-FR3 and
    Balance-FR3 (see the module docstring). Controls arrive normalized to
    [-1, 1] and are scaled to physical rates here. Device twin of
    :meth:`FlipFr3.control_map_host`.
    """
    w = wp.tid()
    point = xpos[w, ee_body]

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

    b = vec6()
    b[0] = v_max * controls[w, t, 0]
    b[1] = v_max * controls[w, t, 1]
    b[2] = v_max * controls[w, t, 2]
    b[3] = w_max * controls[w, t, 3]
    b[4] = w_max * controls[w, t, 4]
    b[5] = w_max * controls[w, t, 5]

    dq = damped_ls(jmat, b, lam)
    for c in range(7):
        ctrl[w, c] = dq[c]


class _FlipFr3IK:
    """Applier: launches :func:`_flip_fr3_ik_kernel` to write ``d.ctrl``."""

    def __init__(
        self,
        arm_dofadr: np.ndarray,
        ee_body: int,
        max_lin_vel: float,
        max_ang_vel: float,
        lam: float,
    ) -> None:
        # Allocated here (build time, under the engine's ScopedDevice) -- never
        # inside apply(), which runs during graph capture.
        self._arm_dof = wp.array(
            np.asarray(arm_dofadr, dtype=np.int32), dtype=wp.int32
        )
        self.ee_body = int(ee_body)
        self.v_max = float(max_lin_vel)
        self.w_max = float(max_ang_vel)
        self.lam = float(lam)

    def apply(self, m, d, controls: wp.array, t: int) -> None:
        """Write ``d.ctrl`` from the staged 6-D twist at step ``t``."""
        wp.launch(
            _flip_fr3_ik_kernel,
            dim=controls.shape[0],
            inputs=[
                m.body_parentid,
                m.body_rootid,
                m.dof_bodyid,
                m.body_isdofancestor,
                d.subtree_com,
                d.cdof,
                d.xpos,
                self._arm_dof,
                wp.int32(self.ee_body),
                wp.float32(self.v_max),
                wp.float32(self.w_max),
                wp.float32(self.lam),
                controls,
                wp.int32(t),
                d.ctrl,
            ],
        )


class _FlipFr3Cost:
    """Launches the sensor-based kernel plus the pusher<->block force one."""

    def __init__(
        self,
        orient_adr: int,
        zaxis_adr: int,
        t_adrs: tuple[int, int, int],
        eez_adr: int,
        w_orient: float,
        w_upright: float,
        w_attract: float,
        w_ctrl: float,
        w_ee_orient: float,
        arm_qposadr: np.ndarray,
        q_home: np.ndarray,
        w_arm_home: float,
        ee_geoms: tuple[int, ...],
        block_geoms: tuple[int, ...],
        w_force: float,
        force_safe: float,
        cone: int,
        peak_excess: wp.array,
        is_terminal: bool,
    ) -> None:
        self.orient_adr = int(orient_adr)
        self.zaxis_adr = int(zaxis_adr)
        self.t1_adr, self.t2_adr, self.t3_adr = (int(a) for a in t_adrs)
        self.eez_adr = int(eez_adr)
        self.w_orient = float(w_orient)
        self.w_upright = float(w_upright)
        self.w_attract = float(w_attract)
        self.w_ctrl = float(w_ctrl)
        self.w_ee_orient = float(w_ee_orient)
        self.w_arm_home = float(w_arm_home)
        self.w_force = float(w_force)
        self.force_safe = float(force_safe)
        # Solver cone is a host-side int on the device model and is never
        # randomized, so it is safe to bake in here.
        self.cone = int(cone)
        # Shared with the running/terminal counterpart of this evaluator
        # (FlipFr3.build_cost_kernel / build_terminal_cost_kernel both read
        # FlipFr3._peak_excess, allocated once in set_nworld): running scans
        # contacts and atomic_maxes into it every step, terminal reads +
        # charges + resets it once. See _flip_fr3_peak_force_kernel.
        self._peak_excess = peak_excess
        self.is_terminal = bool(is_terminal)
        # Allocated at build time (under the engine's ScopedDevice), never in
        # accumulate() which runs during graph capture.
        self._arm_qposadr = wp.array(
            np.asarray(arm_qposadr, dtype=np.int32), dtype=wp.int32
        )
        self._q_home = wp.array(
            np.asarray(q_home, dtype=np.float32), dtype=wp.float32
        )
        # The whole pusher body (tip sphere + connecting shaft), not just the
        # tip -- see _is_pusher_block's docstring for why the shaft must be
        # charged too.
        self._ee_geom = wp.array(
            np.asarray(ee_geoms, dtype=np.int32), dtype=wp.int32
        )
        self._block_geom = wp.array(
            np.asarray(block_geoms, dtype=np.int32), dtype=wp.int32
        )

    def accumulate(self, d, cost: wp.array, scale: float) -> None:
        """Add ``scale * running_cost(d)`` into ``cost`` (per world)."""
        wp.launch(
            _flip_fr3_cost_kernel,
            dim=cost.shape[0],
            inputs=[
                d.sensordata,
                d.ctrl,
                wp.int32(self.orient_adr),
                wp.int32(self.zaxis_adr),
                wp.int32(self.t1_adr),
                wp.int32(self.t2_adr),
                wp.int32(self.t3_adr),
                wp.int32(self.eez_adr),
                wp.float32(self.w_orient),
                wp.float32(self.w_upright),
                wp.float32(self.w_attract),
                wp.float32(self.w_ctrl),
                wp.float32(self.w_ee_orient),
                d.qpos,
                self._arm_qposadr,
                self._q_home,
                wp.float32(self.w_arm_home),
                wp.float32(scale),
                cost,
            ],
        )
        if self.w_force == 0.0:
            return
        if self.is_terminal:
            # Read this rollout's peak excess (accumulated by the running
            # evaluator's launch below, over all H steps), charge it once,
            # and reset the shared buffer for the next rollout's replay.
            wp.launch(
                _flip_fr3_peak_force_kernel,
                dim=cost.shape[0],
                inputs=[self._peak_excess, wp.float32(self.w_force), cost],
            )
            return
        wp.launch(
            _flip_fr3_pusher_force_kernel,
            dim=d.contact.dist.shape[0],  # the whole pool; gated on nacon
            inputs=[
                wp.int32(self.cone),
                d.contact.frame,
                d.contact.friction,
                d.contact.dim,
                d.contact.efc_address,
                d.contact.worldid,
                d.contact.geom,
                d.efc.force,
                wp.int32(d.njmax),
                d.nacon,
                self._ee_geom,
                self._block_geom,
                wp.float32(self.force_safe),
                self._peak_excess,
            ],
        )


def _build_model(
    shape: str, scale: float, wall: bool
) -> tuple[mujoco.MjModel, mujoco.MjSpec, mujoco.MjsBody]:
    """Attach the pusher EE, graft ``shape``, place it, and add the wall.

    Returns ``(model, spec, shape_body)``.
    """
    spec = mujoco.MjSpec.from_file(str(MODELS_DIR / "flip_fr3" / "scene.xml"))
    attach_ee(spec, "pusher")  # pusher EE onto the shared arm's attach site

    shape_body = graft_shape(
        spec,
        "block",
        shape,
        scale,
        default=spec.find_default("block"),
        include_sites=False,
    )
    # All 3 attractor sites at the block's own center of mass, not spread
    # across corners/faces the way the shape fragment authors them (Push's
    # multi-approach-direction convention) -- a push needs to be aimed at
    # the box's middle, not the centroid of 3 scattered points, or the
    # planner can find a push angle that scores well on w_attract without
    # actually being a good tipping angle. Keeps each site's own name/size,
    # only repositions.
    center = shape_center(shape_body, scale)
    block_body = spec.body("block")
    for s in shape_body.sites:
        block_body.add_site(name=s.name, pos=center, size=s.size * scale)
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

    # The goal represents the box standing upright on its target face --
    # resting height assumes the identity (standing) orientation, exactly
    # as shape_min_z computes it.
    z_standing = table_top_z - shape_min_z(shape_body, scale)
    goal_pos = list(spec.body("goal").pos)
    goal_pos[2] = z_standing
    spec.body("goal").pos = goal_pos

    # The block starts lying on its side (_LYING_QUAT), so its resting
    # height is a different, rotation-aware calc: the world-frame vertical
    # half-extent of the (box-shaped) collision geom under that rotation,
    # via the standard rotated-AABB formula |R| @ half_extent.
    collision_geom = next(g for g in shape_body.geoms if g.name == "collision")
    half_extent = np.asarray(collision_geom.size, dtype=np.float64) * scale
    geom_pos = np.asarray(collision_geom.pos, dtype=np.float64) * scale
    lying_mat = np.zeros(9)
    mujoco.mju_quat2Mat(lying_mat, np.array(_LYING_QUAT))
    lying_r = lying_mat.reshape(3, 3)
    world_half_extent = np.abs(lying_r) @ half_extent
    geom_center_z = float((lying_r @ geom_pos)[2])
    z_lying = table_top_z + world_half_extent[2] - geom_center_z

    block_pos = list(spec.body("block").pos)
    block_pos[2] = z_lying
    spec.body("block").pos = block_pos
    spec.body("block").quat = list(_LYING_QUAT)

    key = spec.key("home")
    key_qpos = list(key.qpos)
    key_qpos[0:3] = block_pos
    key_qpos[3:7] = list(_LYING_QUAT)
    key.qpos = key_qpos

    if wall:
        wall_body = spec.body("wall")
        wall_body.pos = list(_WALL_POS)
        wall_body.add_geom(
            spec.find_default("wood"),
            name="wall_wall",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=list(_WALL_SIZE),
            pos=[0.0, 0.0, _WALL_SIZE[2]],
        )

    return spec.compile(), spec, shape_body


class FlipFr3(Task):
    """FR3 pusher tips a free box onto a target face."""

    def __init__(
        self,
        *,
        shape: str = "cracker_box",
        scale: float = 1.0,
        wall: bool = True,
        sampling_space: Literal["task", "joint"] = "joint",
        max_lin_vel: float = 0.15,
        max_ang_vel: float = 1.0,
        ctrl_range_scale: float = 1.0,
        w_orient: float = 1000.0,
        w_upright: float = 0.0,
        w_attract: float = 0.01,
        w_ctrl: float = 0.0,
        w_ee_orient: float = 1.0,
        w_arm_home: float = 1.0,
        w_force: float = 0.0,
        force_safe: float = 20.0,
        terminal_scale: float = 5.0,
        orient_tol: float = 0.3,
        upright_tol: float = 0.35,
        model_config: ModelConfig | None = None,
        goal_drift: GoalDrift | None = None,
        arm_kv: float | None = None,
    ) -> None:
        """Compose the FR3 + pusher EE + grafted ``shape`` at ``scale``.

        Args:
            shape: Block/goal shape under ``models/shapes/`` (only
                ``"cracker_box"`` has a contact budget -- see
                ``bampc.task.common.shapes.flip_fr3_contact_budget``).
            scale: Uniform size scale factor.
            wall: Add a static wall behind the box for the push to react
                against (on by default; in free space the push does not
                sustain). Placement and size are the fixed constants
                ``_WALL_POS``/``_WALL_SIZE`` -- edit those, not the XML.
            sampling_space: ``"task"`` samples a normalized full SE(3) twist
                (``nu=6``); ``"joint"`` (default) samples the 7 joint
                velocities directly. ``"task"`` is untuned here.
            max_lin_vel: Linear rate (m/s) a twist component of 1.0 maps to
                under ``sampling_space="task"``; ignored under ``"joint"``.
            max_ang_vel: Angular rate (rad/s) a twist component of 1.0 maps
                to under ``sampling_space="task"``; ignored under
                ``"joint"``.
            ctrl_range_scale: Shrinks the raw MJCF actuator ctrlrange (up
                to +-5.26 rad/s on the wrist) to this fraction before it
                becomes the planner's sampling bound. Only meaningful under
                ``sampling_space="joint"``, where it is the only ceiling on
                how fast a sampled trajectory can ask the arm to move; a
                no-op under ``"task"``. Tightened in
                ``configs/planner/flip_fr3.yaml`` for real-robot runs, where
                an unconstrained wrist command saturates the joint's real
                +-12 Nm torque budget under contact.
            w_orient: Weight on the block->goal geodesic orientation error.
            w_upright: Weight on tilt off vertical (``1 - block_zaxis.z``).
                Gives gradient toward tipping regardless of face/yaw; see
                the module docstring. Nonzero in
                ``configs/reward/flip_fr3.yaml``.
            w_attract: Weight on the EE->attractor-site term. Flip's 3 sites
                all sit at the block's centre of mass (see ``_build_model``),
                unlike Push's, which spread over corners and faces: Flip only
                pushes from one side, so scattering them would let a mediocre
                push angle score well by averaging out.
            w_ctrl: Weight on the squared joint-velocity control effort.
            w_ee_orient: Weight on the EE point-down penalty (``ee_zaxis``
                sensor vs ``(0, 0, -1)``). Preferred over ``w_arm_home`` for
                keeping the redundant arm sane, since it constrains the
                pusher tip's orientation rather than just joint angles --
                especially under ``sampling_space="task"``.
            w_arm_home: Weight on the arm-home posture penalty (``||q_arm -
                q_home||``). No effect without a "home" keyframe. Left low
                in favour of ``w_ee_orient``.
            w_force: Weight on a *peak* hinge penalty on pusher<->block
                contact force above ``force_safe``, charged **once** per
                rollout on the peak excess (``atomic_max`` in the running
                evaluator, charged and reset by the terminal one).
                Deliberately not integrated over the horizon: tipping needs
                sustained contact, and charging every step made the planner
                avoid committing to contact at all rather than avoiding
                *spikes*, which is what hardware safety actually needs.
            force_safe: Contact-force budget (N) for ``w_force``'s hinge.
                Untuned placeholder -- calibrate from real contact-force
                data before enabling ``w_force``.
            terminal_scale: Scales **every** running weight at the final
                state, so the terminal snapshot cannot be won on box
                orientation alone while the pusher has already disengaged.
                ``w_force`` is exempt: it is already terminal-only.
            orient_tol: ``task_success`` threshold (rad) on the block->goal
                geodesic orientation error.
            upright_tol: ``task_success`` threshold (rad) on tilt off
                vertical, independent of yaw -- "standing on some face",
                not necessarily the target one.
            model_config: Solver/integrator overrides layered on
                ``_BASELINE``.
            goal_drift: Optional moving-goal spec (default static).
                Unvalidated: this goal is an orientation, not a
                support-relative position.
            arm_kv: Overrides the arm's velocity-actuator gain on this
                task's own compiled model only; ``None`` leaves the XML's
                gain untouched. See ``BalanceFr3.arm_kv`` -- that sweep was
                run against Balance-FR3, so treat it as a prior here, not a
                calibrated value, until re-swept.
        """
        mj_model, mj_spec, shape_body = _build_model(shape, scale, wall)

        if arm_kv is not None:
            # Patches this task's own compiled mj_model only -- fr3_arm.xml
            # on disk is untouched, and every other FR3 task compiles its
            # own separate model from it, so they're unaffected.
            mj_model.actuator_gainprm[:, 0] = arm_kv
            mj_model.actuator_biasprm[:, 2] = -arm_kv

        self.sampling_space = sampling_space
        if sampling_space == "task":
            # Normalized channels -- see the module docstring.
            nu: int | None = 6
            ctrl_limits = {"u_min": -np.ones(6), "u_max": np.ones(6)}
        else:
            lo = mj_model.actuator_ctrlrange[:, 0] * ctrl_range_scale
            hi = mj_model.actuator_ctrlrange[:, 1] * ctrl_range_scale
            nu, ctrl_limits = mj_model.nu, {"u_min": lo, "u_max": hi}

        super().__init__(
            mj_model,
            mj_spec=mj_spec,
            endpoint_body=["block"],
            nu=nu,
            ctrl_limits=ctrl_limits,
            model_config=_BASELINE.merged_with(model_config),
            contact_budget=flip_fr3_contact_budget(shape),
            goal_drift=goal_drift,
        )

        self._adr_orient = self._sensor_adr("orientation")
        self._adr_t1, self._adr_t2, self._adr_t3 = (
            self._sensor_adr("ee_t1"),
            self._sensor_adr("ee_t2"),
            self._sensor_adr("ee_t3"),
        )
        self._adr_block_zaxis = self._sensor_adr("block_zaxis")
        self._adr_ee_zaxis = self._sensor_adr("ee_zaxis")

        self.w_orient = float(w_orient)
        self.w_upright = float(w_upright)
        self.w_attract = float(w_attract)
        self.w_ctrl = float(w_ctrl)
        self.w_ee_orient = float(w_ee_orient)
        self.w_arm_home = float(w_arm_home)
        self.w_force = float(w_force)
        self.force_safe = float(force_safe)
        # Allocated by set_nworld (engine calls it once nworld is known,
        # before build_cost_kernel/build_terminal_cost_kernel) -- shared
        # peak-tracking scratch, see _FlipFr3Cost / _flip_fr3_peak_force_kernel.
        self._peak_excess: wp.array | None = None
        self.max_lin_vel = float(max_lin_vel)
        self.max_ang_vel = float(max_ang_vel)
        self.terminal_scale = float(terminal_scale)
        self.orient_tol = float(orient_tol)
        self.upright_tol = float(upright_tol)

        self._block_adr = self._joint_qposadr("block")
        # w_force's hinge penalty + pusher_block_force's contact filter: the
        # whole pusher body (tip sphere + connecting shaft), not just the
        # named "ee" tip -- narrowing to the tip alone let the planner push
        # the block with the shaft to dodge the force penalty entirely.
        self._ee_geom_ids = body_geom_ids(mj_model, "ee_frame", "pusher")
        self._block_geom_ids = body_geom_ids(mj_model, "block")

        # Arm bookkeeping (IK for set_initial_state + arm-home posture).
        self.ee_body_id = int(mj_model.body("ee_frame").id)
        arm_joints = [f"fr3_joint{i}" for i in range(1, 8)]
        jids = [mj_model.joint(n).id for n in arm_joints]
        self.arm_qposadr = mj_model.jnt_qposadr[jids].astype(int)
        self.arm_dofadr = mj_model.jnt_dofadr[jids].astype(int)
        self.joint_limits = mj_model.jnt_range[jids]  # (7, 2)
        # control_map_host's IK scratch (only used under sampling_space="task").
        self._jacp = np.zeros((3, mj_model.nv))
        self._jacr = np.zeros((3, mj_model.nv))
        # Point-down EE goal orientation [w, x, y, z], same convention as
        # PushFr3. No ee_z_target here: unlike Push (a fixed table-height
        # push), Flip's scenarios specify the EE's full xyz start pose
        # directly -- push height genuinely varies with the box's lying
        # geometry, so there's no one sane task-level default to fall back to.
        self.goal_quat_ee = np.array([0.0, 0.7071, 0.7071, 0.0])
        self.ik_damping = 1e-3
        kid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_KEY, "home")
        self.q_home = (
            mj_model.key_qpos[kid][self.arm_qposadr] if kid != -1 else None
        )

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
        """Pusher-block and block-table contact."""
        return (
            ContactProbe(
                name="pusher_block",
                geoms_a=self._ee_geom_ids,
                geoms_b=self._block_geom_ids,
                bit=0,
            ),
            ContactProbe(
                name="block_table",
                geoms_a=self._block_geom_ids,
                geoms_b=body_geom_ids(self.mj_model, "ground"),
                bit=1,
            ),
        )

    @property
    def _t_adrs(self) -> tuple[int, int, int]:
        return (self._adr_t1, self._adr_t2, self._adr_t3)

    def task_success(self, mj_data: mujoco.MjData) -> bool:
        """True once the block is both aligned to the goal and upright.

        Two independently-checked geometric conditions, both required:

        - "aligned": the block->goal geodesic angle (the ``orientation``
          sensor -- the same quantity ``w_orient`` penalizes) under
          ``orient_tol``.
        - "upright": the block's tilt off vertical (``block_zaxis`` sensor,
          the world-frame direction its own z-axis points -- (0, 0, 1)
          exactly when standing on any face) under ``upright_tol``.
          Independent of yaw, so it stays true for "standing but not yet at
          the target face's exact heading" -- a meaningful intermediate
          state a pose-alignment threshold alone can't distinguish from
          "still lying down."
        """
        s = mj_data.sensordata
        q = s[self._adr_orient : self._adr_orient + 4]
        qv = float(np.linalg.norm(q[1:4]))
        orient_err = 2.0 * float(np.arctan2(qv, abs(float(q[0]))))

        z = s[self._adr_block_zaxis : self._adr_block_zaxis + 3]
        tilt = float(np.arccos(np.clip(float(z[2]), -1.0, 1.0)))

        return orient_err < self.orient_tol and tilt < self.upright_tol

    def _arm_home_ref(self) -> tuple[np.ndarray, float]:
        """Home reference + effective weight (0 with no 'home' keyframe)."""
        if self.q_home is None:
            return np.zeros(7), 0.0
        return np.asarray(self.q_home), self.w_arm_home

    def set_nworld(self, nworld: int) -> None:  # noqa: D102
        self._peak_excess = wp.zeros(int(nworld), dtype=wp.float32)

    def build_cost_kernel(self) -> _FlipFr3Cost:  # noqa: D102
        q_home, w_arm_home = self._arm_home_ref()
        return _FlipFr3Cost(
            self._adr_orient,
            self._adr_block_zaxis,
            self._t_adrs,
            self._adr_ee_zaxis,
            self.w_orient,
            self.w_upright,
            self.w_attract,
            self.w_ctrl,
            self.w_ee_orient,
            self.arm_qposadr,
            q_home,
            w_arm_home,
            self._ee_geom_ids,
            self._block_geom_ids,
            self.w_force,
            self.force_safe,
            int(self.mj_model.opt.cone),
            self._peak_excess,
            False,
        )

    def build_terminal_cost_kernel(self) -> _FlipFr3Cost:  # noqa: D102
        # EVERY weight is scaled, not just w_orient -- see terminal_scale in
        # __init__. w_force is untouched: it is already terminal-only.
        s = self.terminal_scale
        q_home, w_arm_home = self._arm_home_ref()
        return _FlipFr3Cost(
            self._adr_orient,
            self._adr_block_zaxis,
            self._t_adrs,
            self._adr_ee_zaxis,
            s * self.w_orient,
            s * self.w_upright,
            s * self.w_attract,
            s * self.w_ctrl,
            s * self.w_ee_orient,
            self.arm_qposadr,
            q_home,
            s * w_arm_home,
            self._ee_geom_ids,
            self._block_geom_ids,
            self.w_force,
            self.force_safe,
            int(self.mj_model.opt.cone),
            self._peak_excess,
            True,
        )

    def build_control_map_kernel(self) -> _FlipFr3IK | None:  # noqa: D102
        if self.sampling_space == "joint":
            return None  # joint velocities sampled directly -> write ctrl
        return _FlipFr3IK(
            self.arm_dofadr,
            self.ee_body_id,
            self.max_lin_vel,
            self.max_ang_vel,
            self.ik_damping,
        )

    def control_map_host(
        self, mj_data: mujoco.MjData, u: np.ndarray
    ) -> np.ndarray:
        """Map a control to the 7 joint-velocity actuators (host).

        Joint sampling is identity. Task sampling scales the normalized 6-D
        twist to physical rates and solves the same damped-LS system the
        device kernel does.
        """
        if self.sampling_space == "joint":
            return np.asarray(u)

        m = self.mj_model
        mujoco.mj_jacBody(m, mj_data, self._jacp, self._jacr, self.ee_body_id)
        jac = np.vstack([self._jacp, self._jacr])[:, self.arm_dofadr]  # (6, 7)
        twist = np.concatenate(
            [
                self.max_lin_vel * np.asarray(u[:3]),
                self.max_ang_vel * np.asarray(u[3:6]),
            ]
        )
        return damped_ls_host(jac, twist, self.ik_damping)

    def cost_components(self, mj_data: mujoco.MjData) -> dict[str, float]:
        """Host twin of the running cost, split per weighted term."""
        s = mj_data.sensordata
        q = s[self._adr_orient : self._adr_orient + 4]
        qv = float(np.linalg.norm(q[1:4]))
        orient = self.w_orient * (2.0 * np.arctan2(qv, abs(float(q[0]))))

        zz = float(s[self._adr_block_zaxis + 2])
        upright = self.w_upright * (1.0 - zz)

        attract_raw = 0.0
        for adr in self._t_adrs:
            attract_raw += float(np.linalg.norm(s[adr : adr + 3]))
        attract = self.w_attract * attract_raw

        ctrl = self.w_ctrl * float(np.dot(mj_data.ctrl, mj_data.ctrl))

        zx, zy, zz_ee = s[self._adr_ee_zaxis : self._adr_ee_zaxis + 3]
        ee_orient = self.w_ee_orient * float(
            np.sqrt(zx * zx + zy * zy + (zz_ee + 1.0) ** 2)
        )

        if self.q_home is None:
            arm_home = 0.0
        else:
            dq = mj_data.qpos[self.arm_qposadr] - np.asarray(self.q_home)
            arm_home = self.w_arm_home * float(np.linalg.norm(dq))

        return {
            "orient": orient,
            "upright": upright,
            "attract": attract,
            "ctrl": ctrl,
            "ee_orient": ee_orient,
            "arm_home": arm_home,
            "total": orient + upright + attract + ctrl + ee_orient + arm_home,
        }

    def pusher_block_force(self, mj_data: mujoco.MjData) -> float:
        """Total pusher<->block contact normal force (N), host-side.

        Diagnostic only (not a cost term) -- host twin of peg_fr3.wall_force's
        pattern, over the whole pusher body (tip sphere + connecting shaft,
        ``_ee_geom_ids``) to match ``w_force``'s hinge penalty and
        ``contact_probes``'s tagging convention -- narrowing to just the tip
        let contact through the shaft go unmeasured.
        """
        total = 0.0
        wrench = np.zeros(6)
        for i in range(mj_data.ncon):
            c = mj_data.contact[i]
            pair = (int(c.geom1), int(c.geom2))
            g0_is_ee = pair[0] in self._ee_geom_ids
            g1_is_ee = pair[1] in self._ee_geom_ids
            if g0_is_ee == g1_is_ee:  # neither, or both (never happens)
                continue
            other = pair[1] if g0_is_ee else pair[0]
            if other not in self._block_geom_ids:
                continue
            mujoco.mj_contactForce(self.mj_model, mj_data, i, wrench)
            total += abs(float(wrench[0]))
        return total

    def running_cost_reference(
        self, mj_data: mujoco.MjData, u: object = None
    ) -> float:
        """Eager numpy running cost (sum of :meth:`cost_components`)."""
        return self.cost_components(mj_data)["total"]

    def terminal_cost_reference(self, mj_data: mujoco.MjData) -> float:
        """Eager numpy terminal cost: the orientation term only, scaled.

        Narrower than the terminal kernel, which scales every weight and
        also charges ``w_force`` -- see :meth:`build_terminal_cost_kernel`.
        Read this as "how far from flipped", not as the kernel's value.
        """
        return self.terminal_scale * self.cost_components(mj_data)["orient"]

    def _solve_ik(
        self,
        target_pos: np.ndarray,
        target_quat: np.ndarray,
        q_seed: np.ndarray | None = None,
    ) -> tuple[np.ndarray, bool]:
        """Damped least-squares IK for the EE pose; see `solve_task_pose_ik`."""
        return solve_task_pose_ik(self, target_pos, target_quat, q_seed)

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

        Mirrors ``PushFr3.set_initial_state``'s "free" branch: pass only
        what you want to override.

        Args:
            mj_data: MjData to mutate; caller has loaded a keyframe first.
            ee_pos: Desired EE position (m, world); triggers IK.
            ee_quat: Desired EE quaternion [w, x, y, z].
            block_xy: Desired block planar position (x, y) in world
                coordinates, meters.
            block_yaw: Rotation around world Z, radians, composed on top of
                the fixed lying tilt (``_LYING_QUAT``) -- 0 reproduces the
                "home" keyframe's lying pose as-is; nonzero spins which way
                the lying box points, it does not stand it up.
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
                    "WARNING: FlipFr3 IK did not converge for "
                    f"ee_pos={ee_pos.tolist()}; using best-effort solution."
                )
            mj_data.qpos[self.arm_qposadr] = q_sol
        elif ee_quat is not None:
            raise ValueError("ee_quat given without ee_pos; specify ee_pos too")

        adr = self._block_adr
        if block_xy is not None:
            mj_data.qpos[adr : adr + 2] = np.asarray(block_xy, dtype=np.float64)
        if block_yaw is not None:
            half = 0.5 * float(block_yaw)
            yaw_quat = np.array([np.cos(half), 0.0, 0.0, np.sin(half)])
            lying_quat = np.array(_LYING_QUAT)
            out = np.zeros(4)
            mujoco.mju_mulQuat(out, yaw_quat, lying_quat)
            mj_data.qpos[adr + 3 : adr + 7] = out

        mujoco.mj_forward(self.mj_model, mj_data)
