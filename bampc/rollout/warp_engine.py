"""Concrete batched rollout engine on MuJoCo Warp.

Owns the device model/data and the cost/control/trace buffers, captures one
``H``-step rollout into a CUDA graph, and replays it each plan step. See
:class:`bampc.rollout.engine.RolloutEngine` for the world-layout and
CUDA-graph contract.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import mujoco_warp as mjw
import numpy as np
import warp as wp

from bampc.dr import get_path, set_path
from bampc.rollout.engine import (
    PredictedState,
    RolloutEngine,
    RolloutResult,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from bampc.dr.randomizer import DomainRandomizer
    from bampc.task.base import Task


@wp.kernel
def _zero(cost: wp.array(dtype=wp.float32)):
    """Reset the per-world cost accumulator."""
    cost[wp.tid()] = 0.0


@wp.kernel
def _write_ctrl(
    controls: wp.array3d(dtype=wp.float32),
    t: wp.int32,
    ctrl: wp.array2d(dtype=wp.float32),
):
    """Default control applier: copy ``controls[:, t]`` into ``d.ctrl``."""
    w = wp.tid()
    for j in range(controls.shape[2]):
        ctrl[w, j] = controls[w, t, j]


@wp.kernel
def _zero_ctrl(ctrl: wp.array2d(dtype=wp.float32)):
    """Clear ``d.ctrl``.

    This means "hold still" only because every task here is **velocity**
    actuated (and the FR3's links carry ``gravcomp``). On a torque-actuated
    task zero ctrl is zero torque, and a settle would drop the arm under
    gravity rather than holding it.
    """
    w, j = wp.tid()
    ctrl[w, j] = 0.0


@wp.kernel
def _record_body_pose(
    xpos: wp.array2d(dtype=wp.vec3),
    xquat: wp.array2d(dtype=wp.quat),
    body_ids: wp.array(dtype=wp.int32),
    out_pos: wp.array2d(dtype=wp.vec3),
    out_quat: wp.array2d(dtype=wp.quat),
):
    """Snapshot tracked bodies' world pose into ``out_*``.

    Needed because ``d.xpos`` after a rollout holds the *end*-of-horizon
    pose; this captures the pose the rollout started from.
    """
    w = wp.tid()
    for k in range(body_ids.shape[0]):
        out_pos[w, k] = xpos[w, body_ids[k]]
        out_quat[w, k] = xquat[w, body_ids[k]]


@wp.kernel
def _zero_int(flags: wp.array(dtype=wp.int32)):
    """Reset the per-world contact bitmask."""
    flags[wp.tid()] = 0


@wp.kernel
def _contact_flags(
    nacon: wp.array(dtype=wp.int32),
    con_geom: wp.array(dtype=wp.vec2i),
    con_worldid: wp.array(dtype=wp.int32),
    con_dist: wp.array(dtype=wp.float32),
    probe_geom: wp.array2d(dtype=wp.int32),
    probe_side: wp.array2d(dtype=wp.int32),
    flags: wp.array(dtype=wp.int32),
):
    """OR a bit per matched probe into ``flags[world]``.

    ``probe_geom[p]`` lists every geom involved in probe ``p`` and
    ``probe_side[p]`` marks each as group A (0) or B (1); a contact counts
    when its two geoms land on opposite sides. Contacts are gated on
    ``dist <= 0``: MuJoCo also emits contacts within ``margin``, which carry
    no force and are not what "in contact" means here.
    """
    c = wp.tid()
    if c >= nacon[0]:
        return
    if con_dist[c] > 0.0:
        return
    pair = con_geom[c]
    g0, g1 = pair[0], pair[1]
    for p in range(probe_geom.shape[0]):
        # Explicitly typed: Warp forbids mutating an inferred constant
        # inside a dynamic loop.
        s0 = wp.int32(-1)
        s1 = wp.int32(-1)
        for k in range(probe_geom.shape[1]):
            g = probe_geom[p, k]
            if g < 0:
                break
            if g == g0:
                s0 = probe_side[p, k]
            if g == g1:
                s1 = probe_side[p, k]
        if s0 >= 0 and s1 >= 0 and s0 != s1:
            wp.atomic_or(flags, con_worldid[c], wp.int32(1) << wp.int32(p))


@wp.kernel
def _record_sites(
    site_xpos: wp.array2d(dtype=wp.vec3),
    site_ids: wp.array(dtype=wp.int32),
    t: wp.int32,
    out: wp.array3d(dtype=wp.vec3),
):
    """Copy traced site world positions into ``out[:, t]``."""
    w = wp.tid()
    for k in range(site_ids.shape[0]):
        out[w, t, k] = site_xpos[w, site_ids[k]]


@wp.kernel
def _record_state(
    qpos: wp.array2d(dtype=wp.float32),
    qvel: wp.array2d(dtype=wp.float32),
    t: wp.int32,
    out_qpos: wp.array3d(dtype=wp.float32),
    out_qvel: wp.array3d(dtype=wp.float32),
):
    """Copy every world's full ``qpos``/``qvel`` into ``out_*[:, t]``.

    Unlike ``_record_sites``, this records *every* sample (not a curated
    subset) since the samples worth keeping aren't known until costs come
    back; :meth:`WarpRolloutEngine.extract_predicted_state` subsets after.
    """
    w = wp.tid()
    for i in range(qpos.shape[1]):
        out_qpos[w, t, i] = qpos[w, i]
    for i in range(qvel.shape[1]):
        out_qvel[w, t, i] = qvel[w, i]


@wp.kernel
def _gather_state(
    qpos: wp.array3d(dtype=wp.float32),
    qvel: wp.array3d(dtype=wp.float32),
    world_idx: wp.array(dtype=wp.int32),
    out_qpos: wp.array3d(dtype=wp.float32),
    out_qvel: wp.array3d(dtype=wp.float32),
):
    """Copy selected worlds' full-horizon state into a compact buffer."""
    i = wp.tid()
    w = world_idx[i]
    for t in range(qpos.shape[1]):
        for j in range(qpos.shape[2]):
            out_qpos[i, t, j] = qpos[w, t, j]
        for j in range(qvel.shape[2]):
            out_qvel[i, t, j] = qvel[w, t, j]


class WarpRolloutEngine(RolloutEngine):
    """Batched MJWarp rollout with a captured CUDA graph."""

    def __init__(
        self,
        task: Task,
        num_samples: int,
        num_randomizations: int,
        randomizer: DomainRandomizer | None = None,
        record_traces: bool = False,
        record_predictions: bool = False,
        record_initial_state: bool = False,
        device: str = "cuda:0",
    ) -> None:
        """See :class:`RolloutEngine`; ``device`` selects the Warp device."""
        super().__init__(
            task,
            num_samples,
            num_randomizations,
            randomizer,
            record_traces,
            record_predictions,
            record_initial_state,
        )
        self.device = device
        self._graph = None
        self._settle_graph = None
        self._settle_steps = 0
        self._rand_arrays: dict[str, wp.array] = {}
        self._t0 = 0.0
        self._contact_flags = None
        self._probes = ()
        self._init_pos = None
        self._init_quat = None
        self._tracked_bodies: tuple[str, ...] = ()

    def build_graph(  # noqa: D102
        self, horizon: int, settle_steps: int = 0
    ) -> None:
        self.H = int(horizon)
        self._settle_steps = max(int(settle_steps), 0)
        N = self.nworld        # randomizations x samples
        self._budget = N       # fixed for the graph's lifetime (set_allocation)
        mjm = self.task.mj_model
        cb = self.task.contact_budget

        with wp.ScopedDevice(self.device):
            self.m = mjw.put_model(mjm)
            # make_data sizing has two flavours (see mujoco_warp make_data):
            # nconmax/njmax are PER-WORLD (constraint arrays are batched
            # (nworld, njmax); the dense efc.J is (nworld, njmax_pad, nv_pad)),
            # while naconmax is the GLOBAL contact pool across all worlds. So
            # only the pool scales with N; multiplying njmax by N would size
            # every world's Jacobian for the whole batch -> O(N^2) memory.
            self.d = mjw.make_data(
                mjm,
                nworld=N,
                nconmax=cb.ncon_per_env or None,
                njmax=cb.nj_per_env or None,
                naconmax=cb.nac_per_env * N or None,
            )

            # Promote randomized model fields to per-world (N, …) arrays so they
            # can vary per domain. Allocated once; updated via wp.copy after
            # capture because graph capture freezes pointers.
            if self.randomizer is not None:
                for field in self.randomizer.randomized_fields():
                    # Fields may be nested (global solver options live on
                    # `m.opt`), so address them by dotted path.
                    a = get_path(self.m, field)
                    tiled = np.repeat(a.numpy(), N, axis=0)
                    new = wp.array(tiled, dtype=a.dtype)
                    set_path(self.m, field, new)
                    self._rand_arrays[field] = new

            self._controls = wp.zeros(
                (N, self.H, self.task.nu), dtype=wp.float32
            )
            self._cost = wp.zeros(N, dtype=wp.float32)
            self.task.set_nworld(N)
            self._cost_fn = self.task.build_cost_kernel()
            self._terminal_fn = self.task.build_terminal_cost_kernel()
            self._applier = self.task.build_control_map_kernel()

            self._alloc_record_buffers(N, mjm)
            self._alloc_initial_state_buffers(N)

            mjw.forward(self.m, self.d)

            # Capture the rollout; fall back to eager replay if unavailable.
            try:
                with wp.ScopedCapture() as cap:
                    self._rollout_body()
                self._graph = cap.graph
            except Exception as e:  # noqa: BLE001
                print(f"[graph capture unavailable: {e}] using eager replay")
                self._graph = None

            # Settle is its own graph: it runs before the rollout, not inside
            # it, so it can't share the capture.
            if self._settle_steps > 0:
                try:
                    with wp.ScopedCapture() as cap:
                        self._settle_body()
                    self._settle_graph = cap.graph
                except Exception as e:  # noqa: BLE001
                    print(f"[settle capture unavailable: {e}] eager replay")
                    self._settle_graph = None

        # Populate the device fields with the initial per-domain values once
        # (capture-safe wp.copy). Later changes go through update_randomizations
        # without a recapture, not per plan step.
        self.resample_randomizations()

    def _rollout_body(self) -> None:
        """One ``H``-step rollout: control -> step -> cost (+ traces)."""
        N, H, dt = self.nworld, self.H, self.task.dt
        mjw.forward(self.m, self.d)
        # Snapshot what this rollout starts from, reusing the forward above
        # (contacts + tracked body poses; ~0.03 ms). No-op unless the engine
        # was built with record_initial_state.
        self._record_initial_state()
        # zero control cost buffers
        wp.launch(_zero, dim=N, inputs=[self._cost])
        # overwrite init state site traces and predicted state
        if self._traces is not None:
            wp.launch(
                _record_sites,
                dim=N,
                inputs=[self.d.site_xpos, self._site_ids, 0, self._traces],
            )
        if self._pred_qpos is not None:
            wp.launch(
                _record_state,
                dim=N,
                inputs=[
                    self.d.qpos, self.d.qvel, 0,
                    self._pred_qpos, self._pred_qvel,
                ],
            )
        for t in range(H):
            # control mapping 
            if self._applier is None:
                wp.launch(
                    _write_ctrl,
                    dim=N,
                    inputs=[self._controls, t, self.d.ctrl],
                )
            else:
                self._applier.apply(self.m, self.d, self._controls, t)
            mjw.step(self.m, self.d)
            self._cost_fn.accumulate(self.d, self._cost, dt)
            if t == H - 1:
                # Terminal term: a separate kernel if the task defines one,
                # else reuse the running cost at the final state.
                terminal = self._terminal_fn or self._cost_fn
                terminal.accumulate(self.d, self._cost, 1.0)
            if self._traces is not None:
                wp.launch(
                    _record_sites,
                    dim=N,
                    inputs=[
                        self.d.site_xpos, self._site_ids, t + 1, self._traces
                    ],
                )
            if self._pred_qpos is not None:
                wp.launch(
                    _record_state,
                    dim=N,
                    inputs=[
                        self.d.qpos, self.d.qvel, t + 1,
                        self._pred_qpos, self._pred_qvel,
                    ],
                )

    def _alloc_record_buffers(self, n: int, mjm) -> None:
        """Allocate the optional trace / predicted-state buffers."""
        self._n_traced = len(self.task.trace_site_ids)
        if self.record_traces and self._n_traced > 0:
            self._site_ids = wp.array(
                self.task.trace_site_ids, dtype=wp.int32
            )
            self._traces = wp.zeros(
                (n, self.H + 1, self._n_traced), dtype=wp.vec3
            )
        else:
            self._traces = None

        if self.record_predictions:
            self._pred_qpos = wp.zeros(
                (n, self.H + 1, mjm.nq), dtype=wp.float32
            )
            self._pred_qvel = wp.zeros(
                (n, self.H + 1, mjm.nv), dtype=wp.float32
            )
        else:
            self._pred_qpos = None
            self._pred_qvel = None

    def _alloc_initial_state_buffers(self, n: int) -> None:
        """Allocate the initial-state snapshot buffers (poses + contacts)."""
        self._tracked_bodies = tuple(self.task.endpoint_bodies)
        if self.record_initial_state and self._tracked_bodies:
            ids = [self.task.mj_model.body(b).id for b in self._tracked_bodies]
            self._body_ids = wp.array(np.array(ids, np.int32), dtype=wp.int32)
            self._init_pos = wp.zeros((n, len(ids)), dtype=wp.vec3)
            self._init_quat = wp.zeros((n, len(ids)), dtype=wp.quat)
        else:
            self._init_pos = None
            self._init_quat = None
        self._build_contact_probes()

    def _record_initial_state(self) -> None:
        """Snapshot contacts + tracked body poses from the current device state.

        Launched inside the rollout graph right after its opening ``forward``,
        which is the one moment the device state *is* the state the rollout
        starts from -- so this reuses work the rollout already does instead of
        paying for a second ``forward``.
        """
        n = self.nworld
        if self._contact_flags is not None:
            wp.launch(_zero_int, dim=n, inputs=[self._contact_flags])
            wp.launch(
                _contact_flags,
                dim=self.d.contact.dist.shape[0],
                inputs=[
                    self.d.nacon,
                    self.d.contact.geom,
                    self.d.contact.worldid,
                    self.d.contact.dist,
                    self._probe_geom,
                    self._probe_side,
                    self._contact_flags,
                ],
            )
        if self._init_pos is not None:
            wp.launch(
                _record_body_pose,
                dim=n,
                inputs=[
                    self.d.xpos, self.d.xquat, self._body_ids,
                    self._init_pos, self._init_quat,
                ],
            )

    def _build_contact_probes(self) -> None:
        """Pack the task's contact probes into padded device arrays.

        One row per probe listing every geom it involves, plus a parallel
        row marking each geom as group A (0) or B (1). Rows are padded with
        ``-1`` so a single rectangular array covers probes of different sizes.
        """
        self._probes = (
            tuple(self.task.contact_probes)
            if self.record_initial_state
            else ()
        )
        if not self._probes:
            self._contact_flags = None
            return
        width = max(len(p.geoms_a) + len(p.geoms_b) for p in self._probes)
        geom = np.full((len(self._probes), width), -1, np.int32)
        side = np.full((len(self._probes), width), -1, np.int32)
        for i, p in enumerate(self._probes):
            ids = list(p.geoms_a) + list(p.geoms_b)
            geom[i, : len(ids)] = ids
            side[i, : len(ids)] = [0] * len(p.geoms_a) + [1] * len(p.geoms_b)
        self._probe_geom = wp.array(geom, dtype=wp.int32)
        self._probe_side = wp.array(side, dtype=wp.int32)
        self._contact_flags = wp.zeros(self.nworld, dtype=wp.int32)

    def _settle_body(self) -> None:
        """Step with zero control to resolve interpenetration."""
        N = self.nworld
        # Zero every actuator, not just the task's sampling-space width:
        # d.ctrl is (nworld, mj_model.nu), which is wider than task.nu
        # whenever sampling is task-space (IK-mapped) rather than joint-space.
        wp.launch(
            _zero_ctrl, dim=(N, self.d.ctrl.shape[1]), inputs=[self.d.ctrl]
        )
        for _ in range(self._settle_steps):
            mjw.step(self.m, self.d)

    def settle(self) -> None:  # noqa: D102
        if self._settle_steps <= 0:
            return  # nothing to do: the rollout starts from the upload
        with wp.ScopedDevice(self.device):
            if self._settle_graph is not None:
                wp.capture_launch(self._settle_graph)
            else:
                self._settle_body()
            # Settling advances the clock; the planner's spline is indexed by
            # absolute time, so put it back rather than leaking the offset.
            wp.copy(
                self.d.time,
                wp.array(
                    np.full(self.nworld, self._t0, np.float32),
                    dtype=wp.float32,
                ),
            )

    def refresh_initial_state(self) -> None:
        """Recompute the initial-state snapshot **outside** a rollout.

        Only for off-hot-path callers (probes, checks) that want
        :meth:`contact_modes` without running a rollout. ``set_initial_state``
        only copies memory, so collision must be re-run first or the contact
        arrays still describe the last state stepped -- which the rollout path
        gets for free off its own opening ``forward``, and this does not.
        """
        if self._contact_flags is None and self._init_pos is None:
            return
        with wp.ScopedDevice(self.device):
            # kinematics+collision, not a full forward: the snapshot needs
            # body poses and the contact pool, not dynamics.
            mjw.kinematics(self.m, self.d)
            mjw.collision(self.m, self.d)
            self._record_initial_state()

    def contact_modes(self) -> np.ndarray | None:  # noqa: D102
        if self._contact_flags is None:
            return None
        with wp.ScopedDevice(self.device):
            flags = self._contact_flags.numpy()
        # Within a domain every sample shares the initial state, so the first
        # sample's mask speaks for the domain.
        return flags.reshape(
            self.num_randomizations, self.num_samples
        )[:, 0].astype(np.uint8)

    def set_allocation(
        self, num_randomizations: int, num_samples: int
    ) -> None:
        """Change the ``(R, S)`` split in place; ``R * S`` must stay fixed.

        No device buffers change size or get reassigned -- they're already
        sized for the fixed ``nworld`` budget from :meth:`build_graph`, and
        every device call reads ``num_randomizations``/``num_samples``
        fresh. This only updates that bookkeeping; callers still need to
        follow up with :meth:`resample_randomizations` (new ``R`` means a
        fresh domain draw) and keep the planner's ``num_samples`` in sync
        (see ``bampc.allocation.AllocationController``).
        """
        r, s = int(num_randomizations), int(num_samples)
        if r * s != self._budget:
            raise ValueError(
                f"allocation {r}x{s}={r * s} does not match the fixed "
                f"budget of {self._budget} worlds"
            )
        self.num_randomizations = r
        self.num_samples = s

    def set_initial_state(  # noqa: D102
        self,
        qpos: np.ndarray,
        qvel: np.ndarray,
        mocap_pos: np.ndarray | None = None,
        mocap_quat: np.ndarray | None = None,
        time: float = 0.0,
    ) -> None:
        N = self.nworld
        self._t0 = float(time)
        R, S = self.num_randomizations, self.num_samples
        with wp.ScopedDevice(self.device):
            wp.copy(self.d.qpos, _tile(qpos, N, wp.float32, R, S))
            wp.copy(self.d.qvel, _tile(qvel, N, wp.float32, R, S))
            if mocap_pos is not None and self.d.mocap_pos.shape[1] > 0:
                wp.copy(self.d.mocap_pos, _tile(mocap_pos, N, wp.vec3))
            if mocap_quat is not None and self.d.mocap_quat.shape[1] > 0:
                wp.copy(self.d.mocap_quat, _tile(mocap_quat, N, wp.quat))
            wp.copy(
                self.d.time,
                wp.array(np.full(N, time, np.float32), dtype=wp.float32),
            )

    def set_controls(self, controls: np.ndarray) -> None:  # noqa: D102
        # (S, H, nu) -> (N, H, nu): repeat the sample block over domains so
        # world w = r*S + s uses controls[s].
        tiled = np.tile(
            np.asarray(controls, np.float32), (self.num_randomizations, 1, 1)
        )
        with wp.ScopedDevice(self.device):
            wp.copy(self._controls, wp.array(tiled, dtype=wp.float32))

    def resample_randomizations(self) -> None:  # noqa: D102
        if self.randomizer is None:
            return
        overrides = self.randomizer.sample()
        self.last_overrides = overrides
        with wp.ScopedDevice(self.device):
            for field, values in overrides.items():
                dst = self._rand_arrays[field]
                # (R, …) -> (N, …): every sample of a domain shares its params.
                tiled = np.repeat(values, self.num_samples, axis=0)
                wp.copy(dst, wp.array(tiled, dtype=dst.dtype))

    def rollout(self) -> RolloutResult:  # noqa: D102
        with wp.ScopedDevice(self.device):
            if self._graph is not None:
                wp.capture_launch(self._graph)
            else:
                self._rollout_body()
            # Device-scoped, not wp.synchronize()'s global all-streams
            # barrier -- this runs every rollout, and a global sync would
            # needlessly drain unrelated GPU work (another engine, a
            # viewer) sharing the process's CUDA context.
            wp.synchronize_device(self.device)
            costs = self._cost.numpy().reshape(
                self.num_randomizations, self.num_samples
            )
            traces = None
            if self._traces is not None:
                traces = self._traces.numpy().reshape(
                    self.num_randomizations,
                    self.num_samples,
                    self.H + 1,
                    self._n_traced,
                    3,
                )
        return RolloutResult(costs=costs, trace_sites=traces)

    def extract_predicted_state(  # noqa: D102
        self, indices: Sequence[int]
    ) -> PredictedState:
        if self._pred_qpos is None:
            raise RuntimeError(
                "engine was not built with record_predictions=True"
            )
        R, S = self.num_randomizations, self.num_samples
        idx = list(indices)
        # Only the requested sample(s), across every domain, are worth a
        # PCIe trip -- gather on device first so `.numpy()` copies back
        # `R * len(idx)` worlds instead of all `R * S`.
        world_idx = np.array(
            [r * S + s for r in range(R) for s in idx], dtype=np.int32
        )
        nq, nv = self._pred_qpos.shape[2], self._pred_qvel.shape[2]
        with wp.ScopedDevice(self.device):
            out_qpos = wp.zeros((len(world_idx), self.H + 1, nq), wp.float32)
            out_qvel = wp.zeros((len(world_idx), self.H + 1, nv), wp.float32)
            wp.launch(
                _gather_state,
                dim=len(world_idx),
                inputs=[
                    self._pred_qpos,
                    self._pred_qvel,
                    wp.array(world_idx, dtype=wp.int32),
                    out_qpos,
                    out_qvel,
                ],
            )
            qpos = out_qpos.numpy().reshape(R, len(idx), self.H + 1, nq)
            qvel = out_qvel.numpy().reshape(R, len(idx), self.H + 1, nv)
        return PredictedState(
            qpos=qpos, qvel=qvel, t0=self._t0, dt=self.task.dt
        )

    def final_body_pose(  # noqa: D102
        self, body_name: str
    ) -> tuple[np.ndarray, np.ndarray]:
        bid = self.task.mj_model.body(body_name).id
        R, S = self.num_randomizations, self.num_samples
        pos = self.d.xpos.numpy()[:, bid].reshape(R, S, 3)
        quat = self.d.xquat.numpy()[:, bid].reshape(R, S, 4)
        return pos, quat

    def current_body_pose(  # noqa: D102
        self, body_name: str
    ) -> tuple[np.ndarray, np.ndarray]:
        if self._init_pos is None:
            raise RuntimeError(
                "engine was not built with record_initial_state=True"
            )
        k = self._tracked_bodies.index(body_name)
        R, S = self.num_randomizations, self.num_samples
        with wp.ScopedDevice(self.device):
            pos = self._init_pos.numpy()[:, k].reshape(R, S, 3)
            quat = self._init_quat.numpy()[:, k].reshape(R, S, 4)
        # Every sample in a domain shares the initial state, so sample 0
        # speaks for the domain.
        return pos[:, 0], quat[:, 0]


def _tile(
    value: np.ndarray, n: int, dtype, num_domains: int = 0, num_samples: int = 0
) -> wp.array:
    """Stage a host array onto the ``(n, …)`` device layout.

    Two input shapes, distinguished by the leading axis:

    * ``(k, …)`` — one state for everything, broadcast to all ``n`` worlds.
    * ``(R, k, …)`` — one state *per domain*, repeated ``num_samples`` times
      so world ``r*S + s`` gets domain ``r`` (the same expansion
      ``resample_randomizations`` uses for randomized model fields).
    """
    value = np.asarray(value, np.float32)
    if num_domains and value.ndim >= 2 and value.shape[0] == num_domains:
        tiled = np.repeat(value, num_samples, axis=0)
    else:
        tiled = np.repeat(value[None], n, axis=0)
    return wp.array(tiled, dtype=dtype)