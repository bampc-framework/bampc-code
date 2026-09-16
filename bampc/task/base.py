"""Task interface for the native-Warp MPC stack.

A ``Task`` owns the MuJoCo model (host ``MjModel``) and declares everything the
rollout engine needs that is *problem-specific*:

* the **cost** as a Warp kernel (device, accumulated over the horizon), with an
  optional numpy **reference** cost for tests/debugging;
* an optional **control-mapping** kernel (device), e.g. task-space ->
  joint-space IK evaluated each rollout step, plus its host twin for the real
  robot;
* the **domain-randomization** knobs it exposes;
* contact/constraint **budgets** for ``make_data``;
* host helpers for setting the initial state (sim + real bring-up).

Generic machinery (sampling, risk, splines, the rollout loop) lives elsewhere
and never needs to be subclassed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, fields, replace
from typing import Any

import mujoco
import numpy as np

# String -> MuJoCo enum tables for the options MuJoCo Warp supports. PGS solver,
# the island flag, and the MJX-only `<custom>`/`<size>` knobs are deliberately
# absent: MJWarp does not support them (see mujoco_warp _src/io.py).
_SOLVER = {
    "CG": mujoco.mjtSolver.mjSOL_CG,
    "Newton": mujoco.mjtSolver.mjSOL_NEWTON,
}
_INTEGRATOR = {
    "Euler": mujoco.mjtIntegrator.mjINT_EULER,
    "RK4": mujoco.mjtIntegrator.mjINT_RK4,
    "implicitfast": mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
    "implicit": mujoco.mjtIntegrator.mjINT_IMPLICIT,
}
_CONE = {
    "pyramidal": mujoco.mjtCone.mjCONE_PYRAMIDAL,
    "elliptic": mujoco.mjtCone.mjCONE_ELLIPTIC,
}
_JACOBIAN = {
    "dense": mujoco.mjtJacobian.mjJAC_DENSE,
    "sparse": mujoco.mjtJacobian.mjJAC_SPARSE,
    "auto": mujoco.mjtJacobian.mjJAC_AUTO,
}
_DSBL_EULERDAMP = int(mujoco.mjtDisableBit.mjDSBL_EULERDAMP)
_DSBL_WARMSTART = int(mujoco.mjtDisableBit.mjDSBL_WARMSTART)


def _enum(table: dict, name: str, key: str) -> int:
    """Map an XML-style option string to its MuJoCo enum, or raise."""
    try:
        return int(table[key])
    except KeyError:
        valid = ", ".join(table)
        raise ValueError(
            f"unsupported {name} {key!r}; valid: {valid}"
        ) from None


def _set_disable_bit(opt: Any, bit: int, enabled: bool) -> None:
    """Toggle a ``disableflags`` bit (``enabled`` clears it, else sets it)."""
    if enabled:
        opt.disableflags &= ~bit
    else:
        opt.disableflags |= bit


@dataclass(frozen=True)
class ModelConfig:
    """Solver / integrator options written onto ``mj_model.opt`` pre-rollout.

    The numeric ``<option>`` settings live here instead of in the XML, so a
    sweep can vary the solver/integrator without a model file per combination.
    Every field defaults to ``None`` = "inherit the task baseline"; a sweep
    overrides only what it varies (e.g. ``ModelConfig(solver="CG")``). Strings
    use MuJoCo's XML spelling and only values MuJoCo Warp supports are accepted.
    Contact buffer sizing is *not* here -- it is fixed per task via
    :class:`ContactBudget`.
    """

    timestep: float | None = None
    solver: str | None = None  # "CG" | "Newton"
    integrator: str | None = None  # Euler | RK4 | implicitfast | implicit
    cone: str | None = None  # "pyramidal" | "elliptic"
    jacobian: str | None = None  # "dense" | "sparse" | "auto"
    impratio: float | None = None
    iterations: int | None = None
    ls_iterations: int | None = None
    tolerance: float | None = None
    ls_tolerance: float | None = None
    eulerdamp: bool | None = None  # passive Euler damping in the implicit step
    warmstart: bool | None = None  # constraint-solver warm start

    def merged_with(self, override: ModelConfig | None) -> ModelConfig:
        """Return a copy with ``override``'s non-None fields taking precedence.

        Used to layer a sweep override on top of a task's fully-populated
        baseline; ``None`` fields in ``override`` leave the baseline untouched.
        """
        if override is None:
            return self
        changes = {
            f.name: getattr(override, f.name)
            for f in fields(self)
            if getattr(override, f.name) is not None
        }
        return replace(self, **changes)

    def apply_to(self, mj_model: mujoco.MjModel) -> None:
        """Write the non-None options onto ``mj_model.opt`` in place."""
        opt = mj_model.opt
        if self.timestep is not None:
            opt.timestep = float(self.timestep)
        if self.impratio is not None:
            opt.impratio = float(self.impratio)
        if self.iterations is not None:
            opt.iterations = int(self.iterations)
        if self.ls_iterations is not None:
            opt.ls_iterations = int(self.ls_iterations)
        if self.tolerance is not None:
            opt.tolerance = float(self.tolerance)
        if self.ls_tolerance is not None:
            opt.ls_tolerance = float(self.ls_tolerance)
        if self.solver is not None:
            opt.solver = _enum(_SOLVER, "solver", self.solver)
        if self.integrator is not None:
            opt.integrator = _enum(_INTEGRATOR, "integrator", self.integrator)
        if self.cone is not None:
            opt.cone = _enum(_CONE, "cone", self.cone)
        if self.jacobian is not None:
            opt.jacobian = _enum(_JACOBIAN, "jacobian", self.jacobian)
        if self.eulerdamp is not None:
            _set_disable_bit(opt, _DSBL_EULERDAMP, self.eulerdamp)
        if self.warmstart is not None:
            _set_disable_bit(opt, _DSBL_WARMSTART, self.warmstart)


@dataclass(frozen=True)
class ContactBudget:
    """Per-environment contact / constraint buffer sizes.

    ``ncon_per_env``/``nj_per_env`` are **per-world** budgets (MJWarp batches
    constraint arrays as ``(nworld, njmax)``); ``nac_per_env`` sizes the
    **global** contact pool ``naconmax`` and is the only one the engine
    multiplies by ``nworld`` in ``make_data``. ``0`` means "use the MJWarp
    default". Load-bearing: under-sizing silently drops contacts rather than
    failing -- re-derive with ``scripts/probing/probe_task.py`` after a model
    change.
    """

    ncon_per_env: int = 0
    nac_per_env: int = 0
    nj_per_env: int = 0


@dataclass(frozen=True)
class ObjectPose:
    """Where a tracked body's pose lives in ``qpos``.

    Tells a state-uncertainty noise model how to perturb a pose without
    knowing which task it came from. Two layouts exist:

    * ``kind="free"`` -- one free joint: ``qpos[adr:adr+3]`` is world xyz and
      ``qpos[adr+3:adr+7]`` the ``[w, x, y, z]`` quaternion. Full SO(3).
    * ``kind="planar"`` -- slide/slide/hinge: ``x_adr``/``y_adr``/``yaw_adr``
      hold xy and yaw. Only yaw is representable, so out-of-plane noise is an
      error, not something to silently drop.

    Planar slide qpos is a *displacement* from the body's XML anchor, but an
    additive delta is unaffected by that offset, so noise needs no correction.
    """

    kind: str  # "free" | "planar"
    adr: int = -1  # free: first qpos address
    x_adr: int = -1  # planar
    y_adr: int = -1
    yaw_adr: int = -1

    @property
    def is_planar(self) -> bool:
        """True when only yaw is representable."""
        return self.kind == "planar"


@dataclass(frozen=True)
class ContactProbe:
    """A named pair of geom groups whose contact is tracked.

    Geom ids are resolved once at task construction. ``bit`` is the probe's
    slot in the per-world contact bitmask the rollout engine reports.
    """

    name: str
    geoms_a: tuple[int, ...]
    geoms_b: tuple[int, ...]
    bit: int


@dataclass(frozen=True)
class GoalDrift:
    """Bounded quasi-periodic drift of the goal pose over time.

    Two incommensurate sinusoids in xy plus a bounded yaw sinusoid, so the
    goal keeps moving instead of letting the block settle. Each axis is a pure
    sine of amplitude = radius starting at 0, so it joins the static goal
    continuously. ``enabled`` is False when every amplitude is zero.
    """

    radius_xy: tuple[float, float] = (0.0, 0.0)
    freq_xy: tuple[float, float] = (0.11, 0.17)  # Hz, incommensurate
    yaw_amp: float = 0.0
    yaw_freq: float = 0.07  # Hz
    phase: tuple[float, float, float] = (0.0, 0.0, 0.0)

    @property
    def enabled(self) -> bool:
        """True if any amplitude is nonzero (otherwise a static goal)."""
        return self.radius_xy != (0.0, 0.0) or self.yaw_amp != 0.0

    def offset(self, t: float) -> tuple[float, float, float]:
        """``(dx, dy, dyaw)`` drift from the base goal at time ``t`` (s)."""
        rx, ry = self.radius_xy
        fx, fy = self.freq_xy
        px, py, pw = self.phase
        dx = rx * np.sin(2.0 * np.pi * fx * t + px)
        dy = ry * np.sin(2.0 * np.pi * fy * t + py)
        dyaw = self.yaw_amp * np.sin(2.0 * np.pi * self.yaw_freq * t + pw)
        return float(dx), float(dy), float(dyaw)


def body_geom_ids(
    mj_model: mujoco.MjModel, *body_names: str
) -> tuple[int, ...]:
    """Geom ids belonging to the named bodies.

    By body, not by geom name: the shape library grafts differently-named
    geoms per shape (``block_puck`` vs ``block_bottom``/``block_top`` …), so
    naming geoms directly would break whenever ``--shape`` changes.
    """
    ids: list[int] = []
    for name in body_names:
        bid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            continue
        start = int(mj_model.body_geomadr[bid])
        ids.extend(range(start, start + int(mj_model.body_geomnum[bid])))
    return tuple(ids)


def yaw_quat(yaw: float) -> np.ndarray:
    """Unit quaternion ``[w, x, y, z]`` for a rotation ``yaw`` about +z."""
    half = 0.5 * float(yaw)
    return np.array([np.cos(half), 0.0, 0.0, np.sin(half)])


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product ``a * b`` of two ``[w, x, y, z]`` quaternions."""
    out = np.zeros(4)
    a64 = np.asarray(a, np.float64)
    b64 = np.asarray(b, np.float64)
    mujoco.mju_mulQuat(out, a64, b64)
    return out


class Task(ABC):
    """Abstract optimal-control problem: dynamics + cost + control mapping."""

    def __init__(
        self,
        mj_model: mujoco.MjModel,
        *,
        trace_sites: Sequence[str] | None = None,
        nu: int | None = None,
        ctrl_limits: dict[str, np.ndarray] | None = None,
        contact_budget: ContactBudget | None = None,
        mj_spec: mujoco.MjSpec | None = None,
        endpoint_body: str | Sequence[str] | None = None,
        model_config: ModelConfig | None = None,
        goal_drift: GoalDrift | None = None,
    ) -> None:
        """Store the model and resolve shared bookkeeping.

        Args:
            mj_model: Compiled host MuJoCo model defining the dynamics.
            mj_spec: The (pre-``compile()``) spec ``mj_model`` was built
                from, kept so the viewer can extend a copy of it (e.g. to
                add mocap ghost bodies) without re-deriving the model.
            model_config: Solver/integrator options to write onto
                ``mj_model.opt`` before anything reads it (notably ``dt``). The
                numeric ``<option>`` settings live here, not in the XML, so a
                sweep can vary them per-task without a model file per combo.
            trace_sites: Site names to record over the horizon (viewer).
            nu: Number of *sampling-space* controls. ``None`` uses every
                actuator (raw actuator-space sampling); set explicitly when a
                control mapper expands a reduced sampling space (e.g. 2-D
                task-space velocity -> 7 joint velocities).
            ctrl_limits: ``{"u_min", "u_max"}`` sampling-space bounds. Required
                when ``nu`` is given; else derived from actuator ranges.
            contact_budget: Contact/constraint buffer sizing for ``make_data``.
            endpoint_body: Body name(s) whose predicted final pose the viewer
                visualizes as translucent mocap "ghosts"; each body's geoms are
                cloned, so each shows its own object. Accepts a single name or a
                list (e.g. block + end-effector). ``None`` disables the pose
                ghosts (the viewer falls back to a generic trace marker).
            goal_drift: Optional moving-goal spec (default static). Push and
                Balance validate it against their support surface; Push-FR3
                does not bound it (an out-of-reach goal is a valid setup).
        """
        assert isinstance(mj_model, mujoco.MjModel)
        self.mj_model = mj_model
        self.mj_spec = mj_spec
        # Normalize to a list; a bare string is one body, None is no ghosts.
        if endpoint_body is None:
            self.endpoint_bodies: list[str] = []
        elif isinstance(endpoint_body, str):
            self.endpoint_bodies = [endpoint_body]
        else:
            self.endpoint_bodies = list(endpoint_body)
        # Apply solver/integrator options before anything reads opt (e.g. dt).
        if model_config is not None:
            model_config.apply_to(mj_model)
        self.model_config = model_config
        self.dt: float = float(mj_model.opt.timestep)
        self.contact_budget = contact_budget or ContactBudget()
        # A moving goal (default: static). Push/Balance validate it against
        # their support surface in their own __init__; Push-FR3 does not.
        self.goal_drift = goal_drift or GoalDrift()

        self.nu = int(nu) if nu is not None else int(mj_model.nu)
        if nu is None:
            lo = mj_model.actuator_ctrlrange[:, 0]
            hi = mj_model.actuator_ctrlrange[:, 1]
            limited = mj_model.actuator_ctrllimited.astype(bool)
            self.u_min = np.where(limited, lo, -np.inf)
            self.u_max = np.where(limited, hi, np.inf)
        else:
            if ctrl_limits is None:
                raise ValueError("'ctrl_limits' is required when 'nu' is set")
            self.u_min = np.asarray(ctrl_limits["u_min"], dtype=np.float64)
            self.u_max = np.asarray(ctrl_limits["u_max"], dtype=np.float64)

        trace_sites = trace_sites or []
        self.trace_site_ids = np.array(
            [mj_model.site(name).id for name in trace_sites], dtype=np.int32
        )

    def set_nworld(self, nworld: int) -> None:
        """Optional hook: told the batch width before the kernels are built.

        Default no-op. Override only if the task needs a persistent
        per-world scratch buffer (e.g. tracking a max across the rollout)
        allocated once, before graph capture, since ``nworld`` isn't known
        inside :meth:`build_cost_kernel` otherwise (see
        ``FlipFr3.set_nworld``).
        """
        return None

    @abstractmethod
    def build_cost_kernel(self) -> Any:
        """Return a cost evaluator exposing ``accumulate(d, cost, scale)``.

        ``accumulate`` launches a Warp kernel that reads batched device state
        for world ``w`` and adds ``scale * running_cost`` into ``cost[w]``. The
        engine calls it per step with ``scale = dt``, so the running cost
        integrates entirely on device; the terminal term is added separately
        (see :meth:`build_terminal_cost_kernel`).
        """

    def build_terminal_cost_kernel(self) -> Any | None:
        """Return a terminal cost evaluator, or ``None``.

        Same interface as :meth:`build_cost_kernel`'s evaluator. The engine
        calls it once after the final step with ``scale = 1`` to *add* a
        distinct terminal term (e.g. heavier pose weights, no shaping) on top
        of the running cost. ``None`` (default) reuses the running-cost kernel,
        i.e. the terminal cost equals the running cost at the final state.
        """
        return None

    def build_control_map_kernel(self) -> Any | None:
        """Return a control applier with ``apply(m, d, controls, t)``, or None.

        ``apply`` writes ``d.ctrl`` from the staged ``controls`` buffer (shape
        ``(nworld, H, nu)``) at step ``t``, optionally through a state-dependent
        mapping (e.g. Jacobian IK). Returning ``None`` (default) makes the
        engine copy ``controls[:, t]`` straight into ``d.ctrl`` — correct when
        the sampling space is already actuator space.
        """
        return None

    def control_map_host(
        self, mj_data: mujoco.MjData, u: np.ndarray
    ) -> np.ndarray:
        """Host twin of :meth:`build_control_map_kernel` for the real robot.

        Runs in the low-frequency control loop (ROS node / sim-to-sim driver),
        not in the batched rollout. Defaults to identity.
        """
        return np.asarray(u)

    def set_initial_state(self, mj_data: mujoco.MjData, **kwargs: Any) -> None:
        """Mutate ``mj_data`` in place to a desired start state (host).

        Overlay on top of a keyframe load (IK for the arm, T-block pose, …).
        Defaults to a no-op.
        """

    @property
    def object_pose_qpos(self) -> ObjectPose | None:
        """Where the manipulated object's pose sits in ``qpos``.

        Read by ``bampc.uncertainty`` to perturb the pose without
        knowing the task. ``None`` (default) means the task has no such
        object, and
        pose noise on it raises.
        """
        return None

    @property
    def contact_probes(self) -> tuple[ContactProbe, ...]:
        """Geom-group pairs whose contact the engine tracks per world.

        Empty (default) disables contact tracking entirely.
        """
        return ()

    def task_success(self, mj_data: mujoco.MjData) -> bool | None:
        """Whether the current state counts as this task's success condition.

        Instantaneous, not episode-level -- a caller iterating replans
        decides how to reduce a sequence of these into one outcome (e.g. AND
        for a condition that must hold throughout, OR for one that just
        needs to be reached once). ``None`` (default) means this task
        defines no such condition.
        """
        return None

    @property
    def goal_mocap_id(self) -> int | None:
        """Mocap slot of the ``goal`` body, or ``None`` if it isn't a mocap."""
        gid = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "goal"
        )
        if gid < 0:
            return None
        mid = int(self.mj_model.body_mocapid[gid])
        return mid if mid >= 0 else None

    def goal_mocap_pose(
        self,
        t: float,
        mj_data: mujoco.MjData,
        base_pos: np.ndarray,
        base_quat: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """World ``(pos, quat)`` of the drifting goal mocap at time ``t``.

        Default: a world-frame offset from the static goal pose
        (``base_pos``/``base_quat`` captured at reset) -- correct for tasks
        whose goal is a world mocap (Push, Push-FR3). ``None`` when drift is
        disabled. Tasks with a support-relative goal (Balance) override this.
        """
        if not self.goal_drift.enabled:
            return None
        dx, dy, dyaw = self.goal_drift.offset(t)
        pos = np.asarray(base_pos, np.float64) + np.array([dx, dy, 0.0])
        quat = quat_mul(base_quat, yaw_quat(dyaw))
        return pos, quat

    def running_cost_reference(
        self, mj_data: mujoco.MjData, u: np.ndarray
    ) -> float:
        """Eager numpy running cost for one host state.

        A host-side readout, not a guaranteed twin of the device kernel --
        each override documents the expression it actually evaluates.
        """
        raise NotImplementedError

    def terminal_cost_reference(self, mj_data: mujoco.MjData) -> float:
        """Eager numpy terminal cost for one host state.

        Same caveat as :meth:`running_cost_reference`: see the override.
        """
        raise NotImplementedError