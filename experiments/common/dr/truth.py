"""Ground-truth backends: the "real world" a sweep measures against.

Two selectable backends (``truth.backend`` in the config, via
:func:`make_truth`):

* ``warp``: a single-domain MJWarp engine. Given the same ``ModelConfig`` as
  the prediction engines it shares their float32 precision, which isolates
  the swept parameter's own effect; given a different one it becomes a
  solver/timestep robustness check with precision held fixed.
* ``cpu``: plain CPU MuJoCo (float64), which carries a genuine sim-to-sim
  gap versus the float32 predictions on top of any ``ModelConfig`` gap.

Both expose the same interface (``set_true_domain``/``reset``/``state``/
``step``/``has_contact``). The task handed in carries the truth's physics,
so neither backend needs to know about ``ModelConfig`` itself.
"""

from __future__ import annotations

import mujoco
import numpy as np

from bampc.dr import DomainRandomizer
from bampc.rollout import WarpRolloutEngine
from bampc.task.base import Task


def _body_geom_ids(mj_model: mujoco.MjModel, body: str) -> set[int]:
    """Geom ids of every geom on ``body`` (by id -- names vary by shape)."""
    bid = mj_model.body(body).id
    return {
        gid
        for gid in range(mj_model.ngeom)
        if mj_model.geom_bodyid[gid] == bid
    }


def pusher_block_geom_ids(
    mj_model: mujoco.MjModel,
) -> tuple[set[int], set[int]]:
    """Pusher vs. block geom-id sets (Push contact tagging)."""
    pid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "pusher")
    return {pid}, _body_geom_ids(mj_model, "block")


def ee_block_geom_ids(
    mj_model: mujoco.MjModel,
) -> tuple[set[int], set[int]]:
    """Pusher-tip vs. block geom-id sets (Push-FR3 contact tagging).

    ``ncon > 0`` can't tell you the *pusher* touched the block -- a free
    block rests on the ground, pinning ``ncon`` nonzero regardless.
    """
    ee_gid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "ee")
    return {ee_gid}, _body_geom_ids(mj_model, "block")


def plate_block_geom_ids(
    mj_model: mujoco.MjModel,
) -> tuple[set[int], set[int]]:
    """Plate vs. block geom-id sets (Balance contact tagging)."""
    plate_gid = mujoco.mj_name2id(
        mj_model, mujoco.mjtObj.mjOBJ_GEOM, "plate"
    )
    return {plate_gid}, _body_geom_ids(mj_model, "block")


def peg_wall_geom_ids(
    mj_model: mujoco.MjModel,
) -> tuple[set[int], set[int]]:
    """Peg vs. socket-wall geom-id sets (Peg-FR3 contact tagging).

    Walls only -- seating the peg on the socket floor is success, so charging
    for peg<->floor contact would fight the goal. Mirrors the peg cost kernel.
    """
    peg_gid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "peg")
    walls = {
        mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, n)
        for n in ("socket_xn", "socket_xp", "socket_yn", "socket_yp")
    }
    return {peg_gid}, walls


def _contact_hit(
    g1: np.ndarray,
    g2: np.ndarray,
    contact_pair: tuple[set[int], set[int]],
) -> bool:
    """True if any ``(g1, g2)`` contact pairs the two geom-id sets."""
    a, b = (list(s) for s in contact_pair)
    hit_ab = np.isin(g1, a) & np.isin(g2, b)
    hit_ba = np.isin(g2, a) & np.isin(g1, b)
    return bool(np.any(hit_ab | hit_ba))


class WarpTruth:
    """Single-domain MJWarp ground truth.

    A single-domain, single-sample engine built once (graph capture is
    expensive); ``set_true_domain`` pushes fresh values via
    ``update_randomizations`` (no recapture). Never mutates ``mj_model``,
    so specs may safely be resolved against this backend's task.
    """

    def __init__(
        self,
        task: Task,
        spec: dict,
        contact_pair: tuple[set[int], set[int]] | None = None,
    ) -> None:
        """Build the engine with ``spec`` as the initial true domain."""
        self.task = task
        self.contact_pair = contact_pair
        randomizer = DomainRandomizer(task, 1, spec)
        self.engine = WarpRolloutEngine(
            task,
            num_samples=1,
            num_randomizations=1,
            randomizer=randomizer,
        )
        self.engine.build_graph(horizon=1)

    @property
    def dt(self) -> float:
        """This backend's physics timestep."""
        return self.task.dt

    def set_true_domain(self, spec: dict) -> None:
        """Push new true values into the model (no recapture)."""
        self.engine.update_randomizations(spec)

    def reset(self, md: mujoco.MjData) -> None:
        """Load an episode's start state from a posed host MjData."""
        nq, nv = self.task.mj_model.nq, self.task.mj_model.nv
        self.engine.set_initial_state(
            qpos=md.qpos[:nq].copy(), qvel=md.qvel[:nv].copy(), time=0.0
        )

    def state(self) -> tuple[np.ndarray, np.ndarray, float]:
        """Current ``(qpos, qvel, time)``."""
        d = self.engine.d
        return (
            d.qpos.numpy()[0].copy(),
            d.qvel.numpy()[0].copy(),
            float(d.time.numpy()[0]),
        )

    def step(self, u: np.ndarray) -> None:
        """Apply a sampling-space control and step once."""
        self.engine.set_controls(np.asarray(u, np.float32).reshape(1, 1, -1))
        self.engine.rollout()

    def has_contact(self) -> bool:
        """True if the last step had a ``contact_pair`` contact."""
        if self.contact_pair is None:
            return False
        n = int(self.engine.d.nacon.numpy()[0])
        if n == 0:
            return False
        geoms = self.engine.d.contact.geom.numpy()[:n]
        return _contact_hit(geoms[:, 0], geoms[:, 1], self.contact_pair)


class CpuTruth:
    """Ground truth stepped on plain CPU MuJoCo (float64).

    Same interface as :class:`WarpTruth`, but MuJoCo's CPU pipeline runs at
    higher precision than the float32 MJWarp predictions -- so this backend
    deliberately carries a genuine sim-to-sim gap.

    It **mutates its own** ``mj_model`` in place (``set_true_domain`` bakes
    the value in before stepping), which is why it must be handed a task
    instance of its own, and why domain specs must never be resolved
    against it -- ``mass_multiplier`` would compound seed over seed. Build
    specs against a pristine reference task instead.
    """

    def __init__(
        self,
        task: Task,
        spec: dict,
        contact_pair: tuple[set[int], set[int]] | None = None,
    ) -> None:
        """Build the CPU truth and bake ``spec`` into its model."""
        self.task = task
        self.contact_pair = contact_pair
        self.data: mujoco.MjData | None = None
        self.set_true_domain(spec)

    @property
    def dt(self) -> float:
        """This backend's physics timestep."""
        return self.task.dt

    def set_true_domain(self, spec: dict) -> None:
        """Bake the true value into the model before stepping.

        Reuses ``DomainRandomizer``'s field resolution (a single fake
        domain) instead of re-deriving which model field/index a spec
        maps to. ``apply_to_mj_model`` writes MuJoCo's own units, which
        matters for ``opt`` params the device stores transformed.
        """
        DomainRandomizer(self.task, 1, spec).apply_to_mj_model(
            self.task.mj_model
        )

    def reset(self, md: mujoco.MjData) -> None:
        """Load an episode's start state from a posed host MjData."""
        self.data = md

    def state(self) -> tuple[np.ndarray, np.ndarray, float]:
        """Current ``(qpos, qvel, time)``."""
        nq, nv = self.task.mj_model.nq, self.task.mj_model.nv
        return (
            self.data.qpos[:nq].copy(),
            self.data.qvel[:nv].copy(),
            float(self.data.time),
        )

    def step(self, u: np.ndarray) -> None:
        """Map a sampling-space control to actuators and step once."""
        self.data.ctrl[:] = self.task.control_map_host(self.data, u)
        mujoco.mj_step(self.task.mj_model, self.data)

    def has_contact(self) -> bool:
        """True if the last step had a ``contact_pair`` contact."""
        if self.contact_pair is None:
            return False
        n = self.data.ncon
        if n == 0:
            return False
        return _contact_hit(
            self.data.contact.geom1[:n],
            self.data.contact.geom2[:n],
            self.contact_pair,
        )


_TRUTH_BACKENDS = {"warp": WarpTruth, "cpu": CpuTruth}


def make_truth(
    backend: str,
    task: Task,
    spec: dict,
    contact_pair: tuple[set[int], set[int]] | None = None,
) -> WarpTruth | CpuTruth:
    """Build the ground-truth backend named by ``backend`` (warp|cpu)."""
    try:
        cls = _TRUTH_BACKENDS[backend]
    except KeyError:
        raise ValueError(f"unknown truth backend: {backend!r}") from None
    return cls(task, spec, contact_pair)
