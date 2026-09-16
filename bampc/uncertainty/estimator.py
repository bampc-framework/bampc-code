"""Estimators: what the planner *does* with a belief cloud.

The noise models say what the sensor does to you; an estimator says what you
do about it. The seam is shape-preserving ``(R, nq) -> (R, nq)``, so nothing
downstream -- engine, planner, risk strategy, viewer -- learns that estimators
exist. Only what fills the rows changes.

* :class:`Ensemble` keeps the R draws distinct: plan over the cloud and let
  the risk strategy aggregate, the domain-randomization-analogous arm.
* :class:`Mean` / :class:`WeightedMean` / :class:`Draw` collapse the cloud to
  one state and broadcast it: a point estimator runs *before* planning, and
  the planner is uncertainty-blind.

Every estimator still hands the planner a **wrong** state -- the sensor noise
is real in all of them. They differ only in whether the spread survives.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

    from bampc.task.base import ObjectPose


def quat_mean(quats: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted chordal-L2 mean of ``[w, x, y, z]`` quaternions.

    The leading eigenvector of ``sum_i w_i q_i q_i^T``. This is sign-safe by
    construction: ``q`` and ``-q`` are the same rotation and contribute the
    same outer product, so a mixed-sign cloud averages correctly where a
    componentwise mean would cancel toward zero.
    """
    m = (quats * w[:, None]).T @ quats
    vals, vecs = np.linalg.eigh(m)
    q = vecs[:, int(np.argmax(vals))]
    if q[0] < 0.0:  # canonical hemisphere, so the sign is reproducible
        q = -q
    return q / np.linalg.norm(q)


def circular_mean(angles: np.ndarray, w: np.ndarray) -> float:
    """Weighted mean of angles (rad), via ``atan2(sum sin, sum cos)``.

    A plain average of a cloud straddling +/-pi collapses toward 0 -- a
    failure that looks like a tuning result rather than a bug.
    """
    return float(
        np.arctan2(np.sum(w * np.sin(angles)), np.sum(w * np.cos(angles)))
    )


def pose_mean(
    qpos: np.ndarray, layout: ObjectPose | None, w: np.ndarray
) -> np.ndarray:
    """Weighted mean of ``(R, nq)`` belief states, respecting the manifold.

    Everything averages componentwise except the object's orientation, which
    lives on a circle (planar yaw) or a sphere (free-joint quaternion). One
    helper so a new estimator can't get this subtly wrong.
    """
    mean = w @ qpos
    if layout is None:
        return mean
    if layout.is_planar:
        mean[layout.yaw_adr] = circular_mean(qpos[:, layout.yaw_adr], w)
    else:
        a = layout.adr
        mean[a + 3 : a + 7] = quat_mean(qpos[:, a + 3 : a + 7], w)
    return mean


class Estimator(ABC):
    """Reduces belief samples to what the planner plans with."""

    @abstractmethod
    def __call__(
        self,
        qpos: np.ndarray,
        qvel: np.ndarray,
        layout: ObjectPose | None,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Map ``(R, nq)``/``(R, nv)`` samples to the same shapes."""


class Ensemble(Estimator):
    """Keep every sample: plan over the cloud (the DR-analogous arm)."""

    def __call__(self, qpos, qvel, layout, rng):  # noqa: D102
        return qpos, qvel


class WeightedMean(Estimator):
    """Collapse to a weighted mean, broadcast to every domain.

    Args:
        weights: ``(qpos, qvel) -> (R,)`` scores, normalized internally.
            ``None`` (default) means uniform, which makes this exactly
            :class:`Mean`.

    The intended non-uniform source is the domain-likelihood posterior the
    sweeps already use (``softmax(-E_r / 2 sigma^2)`` over chunk prediction
    error, fed by ``bampc.tracking``). That is a stateful feedback
    loop from executed motion -- a particle filter -- and is deliberately
    not built here.
    This signature is shaped to accept it later.
    """

    def __init__(
        self,
        weights: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
    ) -> None:
        """Store the weight source (``None`` = uniform)."""
        self.weights = weights

    def __call__(self, qpos, qvel, layout, rng):  # noqa: D102
        n = qpos.shape[0]
        if self.weights is None:
            w = np.full(n, 1.0 / n)
        else:
            w = np.asarray(self.weights(qpos, qvel), dtype=float)
            if w.shape != (n,):
                raise ValueError(
                    f"weights must be ({n},), got {w.shape}"
                )
            if np.any(w < 0.0) or not np.any(w > 0.0):
                raise ValueError("weights must be non-negative and not all 0")
            w = w / w.sum()
        mean_q = pose_mean(qpos, layout, w)
        mean_v = w @ qvel
        return np.tile(mean_q, (n, 1)), np.tile(mean_v, (n, 1))


class Mean(WeightedMean):
    """Collapse to the unweighted mean, broadcast to every domain."""

    def __init__(self) -> None:
        """Uniform weights."""
        super().__init__(weights=None)


class Draw(Estimator):
    """Believe one raw sample exactly -- the naive baseline."""

    def __init__(self, index: int | None = None) -> None:
        """Pick row ``index``, or a uniformly random row when ``None``."""
        self.index = index

    def __call__(self, qpos, qvel, layout, rng):  # noqa: D102
        n = qpos.shape[0]
        i = rng.integers(n) if self.index is None else self.index
        return np.tile(qpos[i], (n, 1)), np.tile(qvel[i], (n, 1))
