"""Peg insertion with an FR3 arm: put a box peg into a square socket.

The FR3 carries a peg rigidly mounted on its end effector and inserts it into
a socket on the ground. ``sampling_space="joint"`` samples the 7 joint
velocities; ``"task"`` samples a **full SE(3) twist** (``nu=6``), every row
commanded, because a peg has to be steered in all six.

**Task-space controls are normalized to [-1, 1], not physical units**, unlike
Push-FR3 and Balance-FR3; the kernel scales them by
``max_lin_vel``/``max_ang_vel``. A 6-D twist mixes m/s with rad/s, and
``PlannerConfig.noise_level``/``init_std`` are *scalars* applied isotropically
across all ``nu`` channels, so with physical units one noise level is
necessarily wrong for half the twist.

The cost is peg->goal pose error plus a penalty on contact force between the peg
and the socket **walls**. Position error is measured in the goal's own frame, so
its z is the insertion axis, and ``z_scale`` down-weights it: without that the
planner buys cheap cost by matching depth before it has found the hole.

NOTE: ``peg_fr3_contact_budget`` is hand-sized, not derived. Re-derive it with
scripts/probing/probe_task.py before any long/headless run.
"""

from __future__ import annotations

from typing import Any, Literal

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
    Task,
    body_geom_ids,
)
from bampc.task.common.cholesky import vec6
from bampc.task.common.ee import attach_ee
from bampc.task.common.ik import (
    damped_ls,
    damped_ls_host,
    ee_jacobian_dof,
    solve_pose_ik,
)
from bampc.task.common.shapes import peg_fr3_contact_budget

# Fallback solver options, used only when the caller names none. The tuned
# values live with the caller -- configs/numerics/peg_fr3.yaml. Same family as
# Push-FR3's free block: insertion is contact-rich, so it wants the fine
# timestep and the higher iteration counts, not the joint profile's.
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
def _peg_fr3_cost_kernel(
    sensordata: wp.array2d(dtype=wp.float32),
    ctrl: wp.array2d(dtype=wp.float32),
    pos_adr: wp.int32,
    orient_adr: wp.int32,
    w_pos: wp.float32,
    z_scale: wp.float32,
    w_orient: wp.float32,
    w_ctrl: wp.float32,
    qpos: wp.array2d(dtype=wp.float32),
    arm_qposadr: wp.array(dtype=wp.int32),
    q_home: wp.array(dtype=wp.float32),
    w_arm_home: wp.float32,
    scale: wp.float32,
    cost: wp.array(dtype=wp.float32),
):
    """Peg->goal pose error + control + arm-home posture.

    The position sensor is peg-relative-to-goal, and the goal frame's z is the
    insertion axis, so ``z_scale`` separates "am I over the hole" (lateral)
    from "am I deep enough" (axial) without any frame math here.
    """
    i = wp.tid()

    ex = sensordata[i, pos_adr + 0]
    ey = sensordata[i, pos_adr + 1]
    ez = sensordata[i, pos_adr + 2]
    lateral = wp.sqrt(ex * ex + ey * ey)
    axial = wp.abs(ez)
    pos_cost = lateral + z_scale * axial

    qw = sensordata[i, orient_adr + 0]
    qx = sensordata[i, orient_adr + 1]
    qy = sensordata[i, orient_adr + 2]
    qz = sensordata[i, orient_adr + 3]
    qv = wp.sqrt(qx * qx + qy * qy + qz * qz)
    orient_cost = 2.0 * wp.atan2(qv, wp.abs(qw))

    ctrl_cost = wp.float32(0.0)
    for k in range(ctrl.shape[1]):
        ctrl_cost += ctrl[i, k] * ctrl[i, k]

    # Arm-home posture: keeps the redundant null-space off the joint limits,
    # which joint-space sampling has no other reason to avoid.
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


@wp.func
def _is_peg_wall(
    pair: wp.vec2i, peg_geom: wp.int32, wall_geom: wp.array(dtype=wp.int32)
) -> wp.int32:
    """1 when this contact pairs the peg with one of the socket walls."""
    g0 = pair[0]
    g1 = pair[1]
    other = wp.int32(-1)
    if g0 == peg_geom:
        other = g1
    elif g1 == peg_geom:
        other = g0
    else:
        return wp.int32(0)
    for k in range(wall_geom.shape[0]):
        if other == wall_geom[k]:
            return wp.int32(1)
    return wp.int32(0)


@wp.kernel
def _peg_wall_force_kernel(
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
    peg_geom: wp.int32,
    wall_geom: wp.array(dtype=wp.int32),
    w_force: wp.float32,
    scale: wp.float32,
    cost: wp.array(dtype=wp.float32),
):
    """Add the peg<->wall normal force into each world's running cost.

    Launched over the *contact pool*, not over worlds: MJWarp keeps contacts in
    one global ``naconmax`` array tagged by ``worldid``, with no per-world
    offset, so a per-world launch would have to scan the whole pool. That makes
    this a scatter, hence ``wp.atomic_add`` rather than the ``cost[i] = cost[i]
    + ...`` the per-world kernels use.

    Only walls count. Seating the peg on the socket floor is the goal the pose
    term drives toward, so charging for that contact would fight it.
    """
    c = wp.tid()
    # Must precede contact_force_fn: it indexes contact_dim/efc_address before
    # its own bounds check, and that check is `<= nacon` rather than `<`.
    if c >= nacon[0]:
        return
    if _is_peg_wall(con_geom[c], peg_geom, wall_geom) == 0:
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
    wp.atomic_add(cost, worldid, scale * w_force * wp.abs(force[0]))


class _PegFr3Cost:
    """Launches the pose kernel plus the peg<->wall force kernel."""

    def __init__(
        self,
        pos_adr: int,
        orient_adr: int,
        w_pos: float,
        z_scale: float,
        w_orient: float,
        w_ctrl: float,
        arm_qposadr: np.ndarray,
        q_home: np.ndarray,
        w_arm_home: float,
        peg_geom: int,
        wall_geoms: tuple[int, ...],
        w_force: float,
        cone: int,
    ) -> None:
        self.pos_adr = int(pos_adr)
        self.orient_adr = int(orient_adr)
        self.w_pos = float(w_pos)
        self.z_scale = float(z_scale)
        self.w_orient = float(w_orient)
        self.w_ctrl = float(w_ctrl)
        self.w_arm_home = float(w_arm_home)
        self.peg_geom = int(peg_geom)
        self.w_force = float(w_force)
        # Solver cone is a host-side int on the device model and is never
        # randomized, so it is safe to bake in here.
        self.cone = int(cone)
        # Allocated at build time (under the engine's ScopedDevice), never in
        # accumulate() which runs during graph capture.
        self._arm_qposadr = wp.array(
            np.asarray(arm_qposadr, dtype=np.int32), dtype=wp.int32
        )
        self._q_home = wp.array(
            np.asarray(q_home, dtype=np.float32), dtype=wp.float32
        )
        self._wall_geom = wp.array(
            np.asarray(wall_geoms, dtype=np.int32), dtype=wp.int32
        )

    def accumulate(self, d, cost: wp.array, scale: float) -> None:
        """Add ``scale * running_cost(d)`` into ``cost`` (per world)."""
        wp.launch(
            _peg_fr3_cost_kernel,
            dim=cost.shape[0],
            inputs=[
                d.sensordata,
                d.ctrl,
                wp.int32(self.pos_adr),
                wp.int32(self.orient_adr),
                wp.float32(self.w_pos),
                wp.float32(self.z_scale),
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
        if self.w_force == 0.0:
            return
        wp.launch(
            _peg_wall_force_kernel,
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
                wp.int32(self.peg_geom),
                self._wall_geom,
                wp.float32(self.w_force),
                wp.float32(scale),
                cost,
            ],
        )


@wp.kernel
def _peg_fr3_ik_kernel(
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
    v_max: wp.float32,
    w_max: wp.float32,
    lam: wp.float32,
    controls: wp.array3d(dtype=wp.float32),
    t: wp.int32,
    ctrl: wp.array2d(dtype=wp.float32),
):
    """Damped-LS task-space IK per world: 6-D twist -> 7 joint velocities.

    All six rows are commanded -- nothing is regulated, unlike Push-FR3 and
    Balance-FR3. Controls arrive normalized to [-1, 1] and are scaled to
    physical rates here (see the module docstring for why). Device twin of
    :meth:`PegFr3.control_map_host`.
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


class _PegFr3IK:
    """Applier: launches :func:`_peg_fr3_ik_kernel` to write ``d.ctrl``."""

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
            _peg_fr3_ik_kernel,
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
                wp.float32(self.v_max),
                wp.float32(self.w_max),
                wp.float32(self.lam),
                controls,
                wp.int32(t),
                d.ctrl,
            ],
        )


def build_peg_spec(
    *,
    peg_half: float = 0.012,
    peg_len: float = 0.12,
    clearance: float = 0.002,
    socket_outer: float = 0.05,
    socket_depth: float = 0.06,
    base_thick: float = 0.01,
    socket_xy: tuple[float, float] = (0.5, 0.0),
) -> tuple[mujoco.MjSpec, dict[str, float]]:
    """Load the scene, attach the peg EE, and size peg + socket together.

    Clearance is one number and both halves of the fit are derived from it, so
    the peg and the hole it must enter cannot drift apart the way two
    hand-typed XML numbers would. The sizes in the XML are a self-consistent
    default only.

    Returns:
        ``(spec, geometry)`` -- geometry carries the derived world-frame
        dimensions the task and the probing scripts both need.
    """
    if clearance <= 0.0:
        raise ValueError(f"clearance must be positive, got {clearance}")
    hole_half = peg_half + clearance
    wall_half = 0.5 * (socket_outer - hole_half)
    if wall_half <= 0.0:
        raise ValueError(
            f"hole (half {hole_half}) does not fit inside a socket of half "
            f"width {socket_outer}"
        )

    spec = mujoco.MjSpec.from_file(str(MODELS_DIR / "peg_fr3" / "scene.xml"))
    attach_ee(spec, "peg")  # peg EE onto the shared arm's attach site

    # Peg. Density, not mass, is what stays fixed when it is resized, so read
    # it off the authored geom before overwriting the size.
    peg = spec.geom("peg")
    authored = 8.0 * peg.size[0] * peg.size[1] * peg.size[2]
    density = float(peg.mass) / authored
    peg.size = [peg_half, peg_half, 0.5 * peg_len]
    peg.pos = [0.0, 0.0, 0.5 * peg_len]
    peg.mass = density * 8.0 * peg_half * peg_half * (0.5 * peg_len)
    spec.site("peg_tip").pos = [0.0, 0.0, peg_len]

    # Socket. Body origin sits on the floor, so local z is height above ground.
    base_half = 0.5 * base_thick
    depth_half = 0.5 * socket_depth
    wall_z = base_thick + depth_half
    wall_offset = hole_half + wall_half

    spec.body("socket").pos = [socket_xy[0], socket_xy[1], 0.0]
    spec.geom("socket_base").size = [socket_outer, socket_outer, base_half]
    spec.geom("socket_base").pos = [0.0, 0.0, base_half]
    for name, sign in (("socket_xn", -1.0), ("socket_xp", 1.0)):
        g = spec.geom(name)
        g.size = [wall_half, socket_outer, depth_half]
        g.pos = [sign * wall_offset, 0.0, wall_z]
    for name, sign in (("socket_yn", -1.0), ("socket_yp", 1.0)):
        g = spec.geom(name)
        g.size = [hole_half, wall_half, depth_half]
        g.pos = [0.0, sign * wall_offset, wall_z]

    hole_bottom_z = base_thick
    hole_top_z = base_thick + socket_depth
    spec.site("hole").pos = [0.0, 0.0, hole_top_z]

    # Goal ghost: the peg at its fully seated pose.
    spec.body("goal").pos = [
        socket_xy[0], socket_xy[1], hole_bottom_z + peg_len
    ]
    spec.geom("goal_peg").size = [peg_half, peg_half, 0.5 * peg_len]
    spec.geom("goal_peg").pos = [0.0, 0.0, 0.5 * peg_len]

    geometry = {
        "peg_half": peg_half,
        "peg_len": peg_len,
        "hole_half": hole_half,
        "clearance": clearance,
        "socket_x": float(socket_xy[0]),
        "socket_y": float(socket_xy[1]),
        "hole_top_z": hole_top_z,
        "hole_bottom_z": hole_bottom_z,
        "depth": socket_depth,
    }
    return spec, geometry


class PegFr3(Task):
    """FR3 inserting a rigidly mounted box peg into a square socket."""

    # Peg pointing down with its faces square to the hole: a 180 deg rotation
    # about world x, so peg local +z -> world -z and local +x -> world +x.
    peg_down_quat = np.array([0.0, 1.0, 0.0, 0.0])

    def __init__(
        self,
        *,
        sampling_space: Literal["task", "joint"] = "task",
        clearance: float = 0.002,
        peg_half: float = 0.012,
        peg_len: float = 0.12,
        socket_xy: tuple[float, float] = (0.5, 0.0),
        max_lin_vel: float = 0.15,
        max_ang_vel: float = 1.0,
        w_pos: float = 20.0,
        z_scale: float = 0.3,
        w_orient: float = 2.0,
        w_ctrl: float = 0.0,
        w_arm_home: float = 0.0,
        w_force: float = 0.02,
        terminal_scale: float = 5.0,
        model_config: ModelConfig | None = None,
        goal_drift: GoalDrift | None = None,
    ) -> None:
        """Build the task.

        Args:
            sampling_space: ``"task"`` samples a normalized 6-D SE(3) twist,
                ``"joint"`` the 7 joint velocities.
            clearance: radial gap between a peg face and the facing hole wall.
            peg_half: peg box half-width.
            peg_len: peg length from the flange.
            socket_xy: socket centre in world xy.
            max_lin_vel: linear rate a task-space control of 1.0 maps to (m/s).
            max_ang_vel: angular rate a control of 1.0 maps to (rad/s).
            w_pos: weight on peg->goal position error.
            z_scale: weight on the *insertion-axis* component of that error,
                relative to the lateral one. Below 1 so the planner cannot buy
                cheap cost by matching depth before it is over the hole.
            w_orient: weight on peg->goal orientation error.
            w_ctrl: weight on squared actuator command.
            w_arm_home: weight on the arm-home posture term (joint sampling
                needs it; task-space IK keeps the arm sane for free).
            w_force: weight on peg<->wall contact normal force.
            terminal_scale: terminal cost multiplier.
            model_config: solver options; falls back to ``_BASELINE``.
            goal_drift: moving-goal spec, or None.
        """
        self.sampling_space = sampling_space
        if sampling_space == "task":
            # Normalized channels -- see the module docstring.
            nu: int | None = 6
            ctrl_limits = {
                "u_min": -np.ones(6),
                "u_max": np.ones(6),
            }
        else:
            nu, ctrl_limits = None, None

        spec, geometry = build_peg_spec(
            peg_half=peg_half,
            peg_len=peg_len,
            clearance=clearance,
            socket_xy=socket_xy,
        )
        mj_model = spec.compile()
        self.geometry = geometry

        super().__init__(
            mj_model,
            mj_spec=spec,
            trace_sites=("peg_tip",),
            endpoint_body="peg_body",
            nu=nu,
            ctrl_limits=ctrl_limits,
            model_config=_BASELINE.merged_with(model_config),
            contact_budget=peg_fr3_contact_budget(),
            goal_drift=goal_drift,
        )

        self._adr_pos = self._sensor_adr("position")
        self._adr_orient = self._sensor_adr("orientation")
        self._adr_tip_hole = self._sensor_adr("tip_hole")
        self.w_pos = float(w_pos)
        self.z_scale = float(z_scale)
        self.w_orient = float(w_orient)
        self.w_ctrl = float(w_ctrl)
        self.w_arm_home = float(w_arm_home)
        self.w_force = float(w_force)
        self.terminal_scale = float(terminal_scale)
        self.max_lin_vel = float(max_lin_vel)
        self.max_ang_vel = float(max_ang_vel)

        self.peg_geom_id = int(mj_model.geom("peg").id)
        self.wall_geom_ids = tuple(
            int(mj_model.geom(n).id)
            for n in ("socket_xn", "socket_xp", "socket_yn", "socket_yp")
        )
        self.base_geom_id = int(mj_model.geom("socket_base").id)
        self.tip_site_id = int(mj_model.site("peg_tip").id)

        # Arm bookkeeping (IK + arm-home posture) -- identical in PushFr3.
        self.ee_body_id = int(mj_model.body("ee_frame").id)
        self.peg_body_id = int(mj_model.body("peg_body").id)
        arm_joints = [f"fr3_joint{i}" for i in range(1, 8)]
        jids = [mj_model.joint(n).id for n in arm_joints]
        self.arm_qposadr = mj_model.jnt_qposadr[jids].astype(int)
        self.arm_dofadr = mj_model.jnt_dofadr[jids].astype(int)
        self.joint_limits = mj_model.jnt_range[jids]  # (7, 2)
        self._jacp = np.zeros((3, mj_model.nv))  # mj_jacBody scratch
        self._jacr = np.zeros((3, mj_model.nv))
        self.ik_damping = 1e-3

        kid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_KEY, "home")
        self.q_home = (
            mj_model.key_qpos[kid][self.arm_qposadr]
            if kid != -1
            else np.zeros(7)
        )

    def _sensor_adr(self, name: str) -> int:
        sid = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        return int(self.mj_model.sensor_adr[sid])

    @property
    def contact_probes(self) -> tuple[ContactProbe, ...]:
        """Peg-vs-walls and peg-vs-floor, tracked separately.

        Separate bits because they mean opposite things: wall contact is the
        failure mode the force penalty charges for, floor contact is success.
        """
        peg = body_geom_ids(self.mj_model, "peg_body")
        return (
            ContactProbe(
                name="peg_wall", geoms_a=peg, geoms_b=self.wall_geom_ids, bit=0
            ),
            ContactProbe(
                name="peg_base",
                geoms_a=peg,
                geoms_b=(self.base_geom_id,),
                bit=1,
            ),
        )

    def build_cost_kernel(self) -> _PegFr3Cost:  # noqa: D102
        return self._cost(1.0)

    def build_terminal_cost_kernel(self) -> _PegFr3Cost:  # noqa: D102
        return self._cost(self.terminal_scale)

    def _cost(self, gain: float) -> _PegFr3Cost:
        """Cost evaluator with every weight multiplied by ``gain``."""
        return _PegFr3Cost(
            self._adr_pos,
            self._adr_orient,
            gain * self.w_pos,
            self.z_scale,
            gain * self.w_orient,
            gain * self.w_ctrl,
            self.arm_qposadr,
            self.q_home,
            gain * self.w_arm_home,
            self.peg_geom_id,
            self.wall_geom_ids,
            gain * self.w_force,
            int(self.mj_model.opt.cone),
        )

    def build_control_map_kernel(self) -> Any | None:  # noqa: D102
        if self.sampling_space == "joint":
            return None  # joint velocities sampled directly -> write ctrl
        return _PegFr3IK(
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
        twist to physical rates and solves the same damped-LS system the device
        kernel does.
        """
        if self.sampling_space == "joint":
            return np.asarray(u)

        m = self.mj_model
        mujoco.mj_jacBody(m, mj_data, self._jacp, self._jacr, self.ee_body_id)
        jac = np.vstack([self._jacp, self._jacr])[:, self.arm_dofadr]  # (6, 7)
        twist = np.concatenate(
            [self.max_lin_vel * np.asarray(u[:3]),
             self.max_ang_vel * np.asarray(u[3:6])]
        )
        return damped_ls_host(jac, twist, self.ik_damping)

    def tip_target(self, height: float) -> np.ndarray:
        """World position of the peg tip ``height`` above the hole mouth.

        Negative heights are inside the hole; ``-depth`` is fully seated.
        """
        g = self.geometry
        return np.array(
            [g["socket_x"], g["socket_y"], g["hole_top_z"] + height]
        )

    def set_initial_state(
        self,
        mj_data: mujoco.MjData,
        *,
        tip_pos: np.ndarray | None = None,
        tip_quat: np.ndarray | None = None,
        height: float | None = None,
    ) -> None:
        """Place the arm from a task-space (6-D) peg-tip pose, then forward.

        The controlled point is the peg *tip*, not the flange -- that is the
        pose that means something for insertion.

        Args:
            mj_data: MjData to mutate; caller has loaded a keyframe first.
            tip_pos: Desired world position of the peg tip.
            tip_quat: Desired peg orientation [w, x, y, z]; defaults to
                pointing straight down with faces square to the hole.
            height: Shorthand for a ``tip_pos`` on the hole axis at this height
                above the mouth. Mutually exclusive with ``tip_pos``.
        """
        if tip_pos is not None and height is not None:
            raise ValueError("give tip_pos or height, not both")
        if height is not None:
            tip_pos = self.tip_target(height)
        if tip_pos is None:
            if tip_quat is not None:
                raise ValueError("tip_quat given without a position")
            mujoco.mj_forward(self.mj_model, mj_data)
            return

        target_quat = (
            self.peg_down_quat
            if tip_quat is None
            else np.asarray(tip_quat, dtype=np.float64)
        )
        q_sol, converged = solve_pose_ik(
            self.mj_model,
            np.asarray(tip_pos, dtype=np.float64),
            target_quat,
            ee_body_id=self.peg_body_id,
            arm_qposadr=self.arm_qposadr,
            arm_dofadr=self.arm_dofadr,
            joint_limits=self.joint_limits,
            q_seed=self.q_home,
            site_id=self.tip_site_id,
            damping=self.ik_damping,
            max_step=0.2,
            posture_gain=0.05,
        )
        if not converged:
            print(
                "WARNING: PegFr3 IK did not converge for "
                f"tip_pos={np.asarray(tip_pos).tolist()}; using best-effort "
                "solution."
            )
        mj_data.qpos[self.arm_qposadr] = q_sol
        mujoco.mj_forward(self.mj_model, mj_data)

    def cost_components(self, mj_data: mujoco.MjData) -> dict[str, float]:
        """Per-term running cost of the current host state (debug/status)."""
        sd = mj_data.sensordata
        e = sd[self._adr_pos : self._adr_pos + 3]
        lateral = float(np.linalg.norm(e[:2]))
        axial = float(abs(e[2]))
        q = sd[self._adr_orient : self._adr_orient + 4]
        orient = float(
            2.0 * np.arctan2(np.linalg.norm(q[1:4]), abs(q[0]))
        )
        ctrl = float(np.sum(np.asarray(mj_data.ctrl) ** 2))
        arm = float(
            np.linalg.norm(mj_data.qpos[self.arm_qposadr] - self.q_home)
        )
        force = self.wall_force(mj_data)
        out = {
            "lateral": self.w_pos * lateral,
            "axial": self.w_pos * self.z_scale * axial,
            "orient": self.w_orient * orient,
            "ctrl": self.w_ctrl * ctrl,
            "arm_home": self.w_arm_home * arm,
            "force": self.w_force * force,
        }
        out["total"] = float(sum(out.values()))
        return out

    def task_success(self, mj_data: mujoco.MjData) -> bool:
        """``True`` once the peg tip is within 8 mm of full seat depth.

        Uses the ``tip_hole`` sensor (peg tip relative to the fixed hole
        mouth site) rather than ``position`` (peg body relative to the
        *goal mocap*, which tracks whatever pose is currently commanded) --
        ``tip_hole`` is the purpose-built, purely geometric approach-error
        sensor. Negative z is inside the hole (``tip_target``'s convention);
        ``-depth`` is fully seated. 8 mm covers both ordinary IK/solver
        residual (``set_initial_state(height=-depth)`` itself only converges
        to within ~0.1 mm of the exact target) and a deliberate margin --
        full bottom-out is not required for the insertion to count as
        successful.
        """
        tip_z = mj_data.sensordata[self._adr_tip_hole + 2]
        return float(tip_z) <= -self.geometry["depth"] + 0.008

    def wall_force(self, mj_data: mujoco.MjData) -> float:
        """Total peg<->wall contact normal force (N), host twin of the kernel.

        Uses MuJoCo's own ``mj_contactForce``, which is the host counterpart of
        MJWarp's ``contact_force_fn``: both return the contact-frame wrench
        whose first component is the normal force.
        """
        total = 0.0
        wrench = np.zeros(6)
        for i in range(mj_data.ncon):
            c = mj_data.contact[i]
            pair = (int(c.geom1), int(c.geom2))
            if self.peg_geom_id not in pair:
                continue
            other = pair[0] if pair[1] == self.peg_geom_id else pair[1]
            if other not in self.wall_geom_ids:
                continue
            mujoco.mj_contactForce(self.mj_model, mj_data, i, wrench)
            total += abs(float(wrench[0]))
        return total

    def running_cost_reference(
        self, mj_data: mujoco.MjData, u: np.ndarray | None = None
    ) -> float:
        """Numpy reference for the device running cost."""
        return self.cost_components(mj_data)["total"]

    def terminal_cost_reference(self, mj_data: mujoco.MjData) -> float:
        """Numpy reference for the device terminal cost."""
        return self.terminal_scale * self.running_cost_reference(mj_data)
