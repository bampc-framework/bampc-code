"""Particle-style belief over a scalar DR parameter, scored by tracking error.

Each domain ``r`` carries a candidate value of some randomized parameter
(e.g. actuator ``kv``); :mod:`bampc.tracking` scores how well each
domain's predictions matched reality. :class:`DomainBelief` turns that
per-domain error into a softmax weight, giving a weighted posterior
mean/std over the parameter without ever touching the device or the DR
spec itself — callers use the posterior to re-center/narrow a
:class:`bampc.allocation.SpecFn` (see ``allocation.py``).
"""

from __future__ import annotations

import numpy as np

# Floor on aggregated error before inverting it for "relative" weighting, so
# a domain landing on (near-)exact zero error doesn't divide by zero -- it
# still ends up carrying ~all the weight after normalizing, same as softmax
# would with a very cold temperature.
_MIN_RELATIVE_ERROR = 1e-9


class DomainBelief:
    """Weighted posterior over one scalar DR parameter, from chunk error."""

    def __init__(
        self,
        temperature: float,
        mean: float,
        std: float,
        weighting: str = "softmax",
    ) -> None:
        """Seed the posterior with the initial (pre-evidence) guess.

        Args:
            temperature: Softmax temperature converting aggregated per-domain
                error into a weight; larger values weight domains more
                evenly, smaller values concentrate weight on the
                lowest-error domain(s) faster. Scales with the chunk error's
                own units (position/velocity squared error), so it needs
                tuning per task/numerics -- see the module's calling example.
                Unused when ``weighting="relative"``.
            mean: Initial posterior mean, used until the first ``update``
                (typically the initial DR range's midpoint).
            std: Initial posterior std, used until the first ``update``
                (typically the initial DR range's half-width).
            weighting: ``"softmax"`` (default, see ``temperature``) or
                ``"relative"`` -- weight each domain by its own error alone
                (``1/error``, normalized), with no temperature to tune.
                Softmax weighs domains *against each other* (a temperature
                sets how much a gap between them matters); relative weighs
                each domain only against its own error, so two equally-bad
                domains always split the weight evenly regardless of how
                good or bad the best domain is. Whichever is picked feeds
                both :attr:`mean`/:attr:`std` and, through them, anything
                that reads those -- so pick per caller: a caller that only
                displays them (e.g. ``allocation.WorstHalfPolicy``'s
                display-only ``belief``) can use either freely, but one
                whose actual narrowing decision reads them (e.g.
                :class:`bampc.allocation.BeliefCollapsePolicy`)
                changes behavior if switched.
        """
        if weighting not in ("softmax", "relative"):
            raise ValueError(
                f"weighting must be 'softmax' or 'relative', got {weighting!r}"
            )
        self.temperature = float(temperature)
        self.mean = float(mean)
        self.std = float(std)
        self.weighting = weighting
        self._prior_mean = float(mean)
        self._prior_std = float(std)
        self.weights: np.ndarray | None = None
        self.values: np.ndarray | None = None
        self.best_error: float | None = None
        self.errors: np.ndarray | None = None

    def update(
        self, values: np.ndarray, error_window: list[np.ndarray]
    ) -> None:
        """Recompute the posterior from the current domains' recent errors.

        Args:
            values: This parameter's current per-domain value, shape ``(R,)``
                -- e.g. ``engine.last_overrides["actuator_gainprm"][:, i, 0]``.
            error_window: Recent per-replan ``(R,)`` chunk errors, oldest
                first (``tracking.PredictionTracker.window``). Must all share
                ``values``'s ``R`` -- the caller must clear the window on any
                domain-identity change (a ladder stage switch redraws every
                domain from scratch).
        """
        if not error_window:
            return
        agg = np.mean(np.stack(error_window), axis=0)  # (R,)
        if self.weighting == "softmax":
            z = -agg / self.temperature
            z = z - np.max(z)
            w = np.exp(z)
        else:
            w = 1.0 / np.maximum(agg, _MIN_RELATIVE_ERROR)
        w = w / np.sum(w)
        self.weights = w
        self.values = values
        self.mean = float(np.sum(w * values))
        self.std = float(np.sqrt(np.sum(w * (values - self.mean) ** 2)))
        self.best_error = float(np.min(agg))
        self.errors = agg

    def collapsed(self, std_threshold: float) -> bool:
        """Whether the posterior has narrowed enough to trust a re-center."""
        return self.weights is not None and self.std < std_threshold

    def excluded_frac(self, reject_ratio: float) -> float:
        """Fraction of domains this evidence can clearly rule out.

        A domain is ruled out when its aggregated error is ``reject_ratio``
        times the best domain's or worse. In a squared-distance error a ratio
        of 4 means "missed twice as far as the best guess did".

        This reads the errors themselves rather than the posterior they were
        softmaxed into, which makes it the honest test of whether an
        observation *separated* the hypotheses. :meth:`collapsed` cannot
        answer that: ``std`` is small both when one domain won outright and
        when the grid it was measured over had already been narrowed to a
        span where nothing disagrees, and it shrinks with ``temperature``
        regardless of what the evidence said.
        """
        if self.errors is None or self.errors.size == 0:
            return 0.0
        best = float(np.min(self.errors))
        return float(np.mean(self.errors > best * reject_ratio))

    def mismatched(self, error_threshold: float) -> bool:
        """Whether even the best-fitting domain predicts reality badly.

        Unlike :meth:`collapsed`, this reads the error's absolute size rather
        than the posterior's shape, so it still works once the belief has
        narrowed to a single domain (where ``std`` is 0 by construction and
        carries no information). A trip means no current hypothesis explains
        what happened -- the truth has moved outside the sampled range.
        """
        return self.best_error is not None and self.best_error > error_threshold

    def reset_prior(self) -> None:
        """Restore the constructor's mean/std, re-spanning the initial band.

        Drops the accumulated posterior so a caller's ``SpecFn`` redraws the
        full prior range on the next stage switch.
        """
        self.mean = self._prior_mean
        self.std = self._prior_std
        self.weights = None
        self.values = None
        self.best_error = None
        self.errors = None
