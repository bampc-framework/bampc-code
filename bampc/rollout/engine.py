"""Batched MJWarp rollout engine.

This is the only heavyweight, GPU-bound, CUDA-graph-captured component. It
owns the device model/data and the cost/control/trace buffers, and exposes a
small imperative API the host planner calls each plan step.

World layout
------------
There is exactly one flat batch axis, ``nworld = num_randomizations *
num_samples``. We fix the layout

    world index  w = r * num_samples + s          (r = domain, s = sample)

so that:

* **controls** depend only on ``s`` -> a ``(num_samples, H, nu)`` host array is
  tiled with ``np.tile(controls, (R, 1, 1))`` into the ``(nworld, H, nu)``
  device buffer;
* **randomized fields** depend only on ``r`` -> a ``(R, …)`` per-domain array is
  expanded with ``np.repeat(values, num_samples, axis=0)`` into the ``(nworld,
  …)`` device field;
* **costs** are accumulated over the horizon *on device* (the cost kernel sums
  into ``cost[w]``) and come back ``(nworld,)``, reshaped to
  ``(R, num_samples)`` for the risk strategy. Only this small scalar tensor
  crosses PCIe — the full per-world state never leaves the GPU.

CUDA-graph discipline
---------------------
Buffers are allocated once with a fixed ``nworld``. After ``build_graph`` the
inputs (initial state, controls, randomized fields) may only be updated via
``wp.copy`` into the existing arrays — never reassigned. The public setters
below enforce this for callers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from bampc.dr.randomizer import DomainRandomizer, RandomizationSpec
    from bampc.task.base import Task


@dataclass
class RolloutResult:
    """Outputs of one batched rollout, copied back to the host.

    Attributes:
        costs: Per-domain total costs (horizon summed on device, terminal term
            included), shape ``(num_randomizations, num_samples)``.
        trace_sites: Optional site traces for visualization, shape
            ``(num_randomizations, num_samples, H+1, num_sites, 3)``; ``None``
            when trace recording is disabled (the real-time path).
    """

    costs: np.ndarray
    trace_sites: np.ndarray | None = None


@dataclass
class PredictedState:
    """Full-horizon state prediction for selected samples, across domains.

    Produced by :meth:`RolloutEngine.extract_predicted_state` once the caller
    knows which samples are worth a full-state trace. Used to score domain
    prediction accuracy against a later observed chunk (see
    :mod:`bampc.tracking`).

    Attributes:
        qpos: Predicted configuration per domain/selected-sample/step, shape
            ``(num_randomizations, len(indices), H+1, nq)``.
        qvel: Predicted velocity, shape
            ``(num_randomizations, len(indices), H+1, nv)``.
        t0: Sim time the rollout started from (``state.time`` at that plan
            step).
        dt: Per-step time increment (``task.dt``), so a chunk of length ``L``
            spans ``[t0, t0 + L * dt]``.
    """

    qpos: np.ndarray
    qvel: np.ndarray
    t0: float
    dt: float


class RolloutEngine(ABC):
    """Owns the batched device simulation and the captured rollout graph."""

    def __init__(
        self,
        task: Task,
        num_samples: int,
        num_randomizations: int,
        randomizer: DomainRandomizer | None = None,
        record_traces: bool = False,
        record_predictions: bool = False,
        record_initial_state: bool = False,
    ) -> None:
        """Configure batch sizes; ``build_graph`` allocates the device state.

        Args:
            task: The task (model, cost kernel, control-map kernel, budgets).
            num_samples: Control sequences sampled per plan step (S).
            num_randomizations: Domains evaluated per sample (R).
            randomizer: Domain randomizer; ``None`` disables randomization.
            record_traces: Record site traces (viewer only — adds a per-step
                write and a larger copyback; keep off for real-time planning).
            record_predictions: Record full ``qpos``/``qvel`` every step for
                every sample, across domains (a per-step write, kept on device
                until :meth:`extract_predicted_state` pulls a subset back).
                Off by default; enable for domain-prediction tracking.
            record_initial_state: Snapshot the state each rollout *starts*
                from -- which ``contact_probes`` are touching
                (:meth:`contact_modes`) and where the ``endpoint_bodies`` are
                (:meth:`current_body_pose`). Recorded **inside** the rollout
                graph, right after its opening ``forward``, so it reuses work
                the rollout already does; a separate ``forward`` costs two
                orders of magnitude more. Off by default: it allocates
                buffers and must be decided before graph capture.
        """
        self.task = task
        self.num_samples = int(num_samples)
        self.num_randomizations = max(int(num_randomizations), 1)
        self.randomizer = randomizer
        self.record_traces = bool(record_traces)
        self.record_predictions = bool(record_predictions)
        self.record_initial_state = bool(record_initial_state)
        self.last_overrides: dict[str, np.ndarray] | None = None

    @property
    def nworld(self) -> int:
        """Total parallel environments, ``num_randomizations * num_samples``."""
        return self.num_randomizations * self.num_samples

    @abstractmethod
    def build_graph(self, horizon: int, settle_steps: int = 0) -> None:
        """Allocate device buffers and capture the rollout into a CUDA graph.

        Args:
            horizon: Number of control steps ``H`` per rollout (fixed for the
                lifetime of the captured graph).
            settle_steps: Zero-control steps captured into a *separate* graph
                and replayed by :meth:`settle` before each rollout. ``0``
                (default) captures nothing.

        Builds the device model/data (with ``make_data`` sized by the task's
        contact budget * ``nworld``), the cost/control/trace buffers, the task's
        cost and control-map kernels, and captures one ``H``-step rollout for
        replay.
        """

    @abstractmethod
    def set_initial_state(
        self,
        qpos: np.ndarray,
        qvel: np.ndarray,
        mocap_pos: np.ndarray | None = None,
        mocap_quat: np.ndarray | None = None,
        time: float = 0.0,
    ) -> None:
        """Upload the current state into every world (``wp.copy``).

        ``qpos``/``qvel`` may be ``(nq,)``/``(nv,)`` — one state broadcast
        everywhere, the default — or ``(R, nq)``/``(R, nv)`` to give each
        domain its own state, which is how ``bampc.uncertainty`` fans a
        belief cloud across the domain axis. ``mocap_*`` is always broadcast:
        the goal is commanded, not estimated.
        """

    def settle(self) -> None:
        """Step with zero control to resolve interpenetration, then re-time.

        Runs between :meth:`set_initial_state` and :meth:`rollout`. Perturbed
        or measured states can start with bodies overlapping; settling lets
        the constraint solver push them apart, at the cost of the rollout no
        longer starting from exactly the uploaded state. No-op by default.
        """

    def contact_modes(self) -> np.ndarray | None:
        """Per-domain contact bitmask ``(R,)``, or ``None`` if untracked.

        Bit ``p`` is set when the task's ``contact_probes[p]`` geom groups
        are touching. Describes the state the rollout **started** from, and
        is recorded during it -- so read it *after* :meth:`rollout`.
        """
        return None

    def current_body_pose(
        self, body_name: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """Per-domain pose ``(R, 3)`` / ``(R, 4)`` a rollout started from.

        The twin of :meth:`final_body_pose` for the *initial* state (after
        any settling). Like :meth:`contact_modes` it is recorded inside the
        rollout, so read it *after* :meth:`rollout`.
        """
        raise NotImplementedError

    @abstractmethod
    def set_controls(self, controls: np.ndarray) -> None:
        """Stage the sampling-space controls for this rollout.

        Args:
            controls: Dense controls per sample, shape ``(num_samples, H, nu)``.
                Tiled across domains into the ``(nworld, H, nu)`` device buffer.
        """

    @abstractmethod
    def resample_randomizations(self) -> None:
        """Draw fresh per-domain fields and ``wp.copy`` them into the model.

        No-op when no randomizer is attached. Safe to call after graph capture.
        Called once at build time (and by :meth:`update_randomizations`), not
        per plan step — concrete dict values would otherwise re-upload identical
        data every step. Implementations must also stash the draw as
        ``self.last_overrides`` (``{field: (num_randomizations, *shape)}``),
        the only host-side record of which value each domain actually got.
        """

    def update_randomizations(self, spec: RandomizationSpec) -> None:
        """Change DR ranges/values online and re-upload — no graph recapture.

        Merges ``spec`` into the attached randomizer (only already-declared
        fields; a new model field raises, since the per-world device arrays are
        fixed at capture time) and pushes fresh per-domain values into the
        device fields. The captured graph is untouched, so this is safe to call
        between plan steps in an experiment loop.
        """
        if self.randomizer is None:
            raise RuntimeError("no randomizer attached to update")
        self.randomizer.update(spec)
        self.resample_randomizations()

    @abstractmethod
    def rollout(self) -> RolloutResult:
        """Replay the graph; copy costs (and traces) back to host."""

    @abstractmethod
    def extract_predicted_state(
        self, indices: Sequence[int]
    ) -> PredictedState:
        """Pull the recorded full-horizon state for selected samples.

        Call right after :meth:`rollout`, once the sample indices worth a
        full-state trace are known (they generally depend on that rollout's
        own costs, e.g. the winning knot sequence — so they can't be chosen
        in advance). Requires the engine to have been built with
        ``record_predictions=True``.

        Args:
            indices: Sample indices (0 <= i < num_samples) to extract, across
                every domain.

        Returns:
            A :class:`PredictedState` with ``qpos``/``qvel`` shape
            ``(num_randomizations, len(indices), H+1, nq/nv)``.
        """

    @abstractmethod
    def final_body_pose(
        self, body_name: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """World pose of a body at the end of the last rollout, per world.

        Must be called right after :meth:`rollout` (reads the live ``d``, which
        holds every world's final state). Used by the viewer to render endpoint
        ghosts with orientation.

        Returns:
            ``(pos, quat)`` of shape ``(R, S, 3)`` and ``(R, S, 4)`` (quat is
            MuJoCo ``w, x, y, z``).
        """