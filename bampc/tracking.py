"""Per-replan domain-prediction tracking (chunk vs. chunk).

Scores how well each DR domain predicted the *executed* control's real
outcome, one replan at a time. Only the sample that actually became
``params.mean`` (the one really applied between replans) is a valid
ground-truth comparison — an arbitrary sample's rollout used a different
control than what was really executed, so diffing it against reality would
conflate physics error with control mismatch.

Since receding-horizon replanning supersedes a prediction long before its
full horizon plays out, only the chunk actually executed since the last
replan (length ``L``, generally << ``H``) is ever validated; the rest of the
horizon is discarded unread.
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from bampc.planner.base import StateSnapshot
    from bampc.rollout.engine import PredictedState


def chunk_error_components(
    predicted: PredictedState,
    obs_qpos: np.ndarray,
    obs_qvel: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-domain qpos and qvel chunk error, kept apart.

    Kept apart because they carry different units (m^2 / rad^2 vs their
    per-second squares); any 1:1 sum of them is an arbitrary weighting, so
    the caller picks one instead of inheriting it.

    Args:
        predicted: A single tracked sample's prediction (``qpos``/``qvel``
            carry a length-1 sample axis, e.g. predictive sampling's winning
            knot sequence), shape ``(R, 1, H+1, nq/nv)``.
        obs_qpos: Observed configuration chunk, shape ``(L, nq)``.
        obs_qvel: Observed velocity chunk, shape ``(L, nv)``.

    Returns:
        ``(err_qpos, err_qvel)``, each shape ``(R,)``.
    """
    # predicted.qpos[:, 0, 0, :] is the *pre*-step state optimize() was
    # called with; obs_qpos[0] is only ever appended *after* a step (see
    # PredictionTracker.observe_tick). So predicted index i+1 (state after
    # step i+1) aligns with obs_qpos[i] -- index 0 must be skipped, or every
    # comparison is off by one whole physics step.
    max_len = predicted.qpos.shape[2] - 1  # H
    length = min(obs_qpos.shape[0], max_len)
    pred_qpos = predicted.qpos[:, 0, 1 : length + 1, :]  # (R, L, nq)
    pred_qvel = predicted.qvel[:, 0, 1 : length + 1, :]  # (R, L, nv)
    obs_q = obs_qpos[:length][None]  # (1, L, nq), broadcasts against R
    err_qpos = np.sum((pred_qpos - obs_q) ** 2, axis=(1, 2))
    err_qvel = np.sum((pred_qvel - obs_qvel[:length][None]) ** 2, axis=(1, 2))
    return err_qpos, err_qvel


def chunk_error(
    predicted: PredictedState,
    obs_qpos: np.ndarray,
    obs_qvel: np.ndarray,
) -> np.ndarray:
    """Per-domain summed squared error between one predicted/observed chunk.

    The 1:1 sum of :func:`chunk_error_components`. See it for the args.

    Returns:
        Per-domain error, shape ``(R,)``.
    """
    err_qpos, err_qvel = chunk_error_components(predicted, obs_qpos, obs_qvel)
    return err_qpos + err_qvel


class PredictionTracker:
    """Chunk-vs-chunk domain accuracy, one score per replan, in a window."""

    def __init__(self, window: int = 50) -> None:
        """Set up an empty pending prediction and chunk buffer.

        Args:
            window: Max number of past per-replan ``(R,)`` errors to keep.
        """
        self._pending: PredictedState | None = None
        self._chunk_qpos: list[np.ndarray] = []
        self._chunk_qvel: list[np.ndarray] = []
        self.window: deque[np.ndarray] = deque(maxlen=window)

    def reset(self) -> None:
        """Drop the pending prediction, open chunk, and error history.

        Call after anything that breaks domain identity (e.g. an
        :class:`~bampc.allocation.AllocationController` stage
        switch redraws every domain from scratch) -- otherwise the next
        :meth:`on_replan` scores the new chunk against a stale prediction
        shaped for the old domain count.
        """
        self._pending = None
        self._chunk_qpos = []
        self._chunk_qvel = []
        self.window.clear()

    def observe_tick(self, state: StateSnapshot) -> None:
        """Append the real state at this physics step to the open chunk."""
        self._chunk_qpos.append(np.asarray(state.qpos))
        self._chunk_qvel.append(np.asarray(state.qvel))

    def on_replan(
        self, predicted_state: PredictedState | None
    ) -> np.ndarray | None:
        """Score the just-closed chunk against the prior pending prediction.

        Args:
            predicted_state: This replan's fresh prediction (stored as
                pending for the *next* call); typically
                ``PlanInfo.predicted_state``.

        Returns:
            Per-domain ``(R,)`` summed squared error over the closed chunk,
            or ``None`` before the first full cycle (no pending prediction
            yet, or nothing was observed since it was set).
        """
        error = None
        if self._pending is not None and self._chunk_qpos:
            obs_qpos = np.stack(self._chunk_qpos)  # (L, nq)
            obs_qvel = np.stack(self._chunk_qvel)  # (L, nv)
            error = chunk_error(self._pending, obs_qpos, obs_qvel)
            self.window.append(error)

        self._chunk_qpos = []
        self._chunk_qvel = []
        self._pending = predicted_state
        return error
