"""A deterministic sigma-point belief: the unscented set of the estimate.

The counterpart to a random ensemble for one question: is a *structured* set
reproducing the posterior's moments exactly worth more than the same number of
random draws? :class:`SigmaPointBelief` places the unscented-transform sigma
points of a filter's pose posterior instead of sampling it, so the ``R = 2n+1``
members reproduce the posterior **mean and covariance** with no Monte-Carlo
error. The pose delta is world-frame, matching
:func:`~bampc.uncertainty.noise._apply_pose`.

It spreads the **pose only**, exactly as
:class:`~bampc.uncertainty.noise.PosteriorGaussian` does, so a sigma
arm and an ensemble arm differ only in *how* the spread is represented.

A constant-velocity :class:`~bampc.uncertainty.filter.PoseKalman`'s
pose block is diagonal, so the sigma set is axis-aligned -- it differs from a
random cloud in being deterministic and symmetric, not in capturing
correlations. The factorization is kept general (a symmetric matrix square
root) so it stays correct if the filter gains cross-axis terms.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from bampc.uncertainty.noise import _apply_pose
from bampc.uncertainty.state import StateUncertainty

if TYPE_CHECKING:
    from bampc.planner.base import StateSnapshot
    from bampc.task.base import Task
    from bampc.uncertainty.estimator import Estimator
    from bampc.uncertainty.filter import PoseKalman


class SigmaPointBelief(StateUncertainty):
    """The unscented sigma set of a filter's pose posterior, as a belief cloud.

    Args:
        task / num_randomizations / estimator / seed: as
            :class:`~bampc.uncertainty.state.StateUncertainty`.
            ``num_randomizations`` **must** be ``2n+1`` (``n`` the pose DoF,
            6 for a free block), or construction raises -- a sigma set has a
            fixed size.
        filt: The filter whose pose posterior is read live (its mean is the
            estimate, its covariance sizes the set), exactly as
            :class:`~bampc.uncertainty.noise.PosteriorGaussian`
            reads its std. Feed it some readings first.
        alpha / kappa: Unscented spread parameters. The point distance is
            ``sqrt(n + lambda)`` with ``lambda = alpha^2 (n + kappa) - n``;
            the default ``alpha=1, kappa=3-n`` is the classic Gaussian choice
            (``lambda = 3-n``, distance ``sqrt(3)``). The moment reproduction
            is exact for any choice; only the placement scale changes.

    Domain 0 is the unperturbed estimate (the sigma-set centre), so it matches
    ``StateUncertainty(include_truth=False)``'s point arm and is a free
    reference. Free-block only -- the filter it reads is free-block only.
    """

    def __init__(
        self,
        task: Task,
        num_randomizations: int,
        filt: PoseKalman,
        estimator: Estimator | None = None,
        seed: int = 0,
        alpha: float = 1.0,
        kappa: float | None = None,
    ) -> None:
        """Fix the sigma-set size and spread; the covariance is read live."""
        super().__init__(
            task, num_randomizations, [], estimator, seed,
            include_truth=False,
        )
        if self.layout is None or self.layout.is_planar:
            raise ValueError(
                "SigmaPointBelief covers free-joint objects; a planar object "
                "has no full pose posterior to place a sigma set on"
            )
        self.filt = filt
        self._n = 6  # free-block pose DoF: [dp(3), dtheta(3)]
        expected = 2 * self._n + 1
        if self.num_randomizations != expected:
            raise ValueError(
                f"SigmaPointBelief needs num_randomizations = 2n+1 = "
                f"{expected} for n={self._n} pose DoF, got "
                f"{self.num_randomizations}"
            )
        self.alpha = float(alpha)
        self.kappa = float(3 - self._n) if kappa is None else float(kappa)
        self._lam = self.alpha**2 * (self._n + self.kappa) - self._n

    def weights(self) -> tuple[np.ndarray, np.ndarray]:
        """Unscented mean/covariance weights ``(Wm, Wc)``, each ``(2n+1,)``.

        Not used by the identifiability softmax (a uniform prior over domains),
        but exposed so a moment check -- and a later UKF update -- can verify
        the set reproduces the posterior.
        """
        n, lam = self._n, self._lam
        wm = np.full(2 * n + 1, 1.0 / (2.0 * (n + lam)))
        wc = wm.copy()
        wm[0] = lam / (n + lam)
        wc[0] = wm[0] + (1.0 - self.alpha**2 + 2.0)  # beta = 2 (Gaussian)
        return wm, wc

    def _pose_sqrt(self) -> np.ndarray:
        """Symmetric square root of the pose posterior block, ``(6, 6)``.

        ``eigh`` rather than Cholesky so a semidefinite covariance (a channel
        the filter is certain about) does not raise; negative eigenvalues from
        round-off are clipped to zero.
        """
        n = self._n
        p6 = self.filt.posterior_cov()[:n, :n]
        w, v = np.linalg.eigh(0.5 * (p6 + p6.T))
        return (v * np.sqrt(np.clip(w, 0.0, None))) @ v.T

    def sample(
        self, state: StateSnapshot
    ) -> tuple[np.ndarray, np.ndarray]:
        """Place the sigma set around the estimate and reduce it."""
        r, n = self.num_randomizations, self._n
        qpos = np.tile(np.asarray(state.qpos, dtype=float), (r, 1))
        qvel = np.tile(np.asarray(state.qvel, dtype=float), (r, 1))

        # Tangent-space offsets: row 0 the centre, rows 1..2n the +/- columns
        # of sqrt((n+lambda) * P) -- exactly the unscented sigma points.
        c = np.sqrt(self._n + self._lam)
        cols = c * self._pose_sqrt()  # (n, n): column j is direction j
        delta = np.zeros((r, n))
        for j in range(n):
            delta[2 * j + 1] = cols[:, j]
            delta[2 * j + 2] = -cols[:, j]
        # [dp(3), dtheta(3)] -> a world-frame pose delta on every row.
        _apply_pose(qpos, self.layout, delta[:, 0:3], delta[:, 3:6])

        out = self.estimator(qpos, qvel, self.layout, self.rng)
        self.last_sample = out
        return out
