"""Cross-entropy method (host algorithm)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from bampc.planner.base import SamplingParams, SamplingPlanner

if TYPE_CHECKING:
    from bampc.risk import RiskStrategy
    from bampc.rollout.engine import RolloutEngine
    from bampc.spline import InterpMethod
    from bampc.task.base import Task


class CEM(SamplingPlanner):
    """Refit a diagonal Gaussian to the elite fraction of rollouts."""

    def __init__(
        self,
        task: Task,
        engine: RolloutEngine,
        *,
        risk_strategy: RiskStrategy,
        num_samples: int,
        init_std: float,
        elite_frac: float = 0.1,
        num_elite: int | None = None,
        min_std: float = 1e-3,
        adapt_covariance: bool = False,
        plan_horizon: float = 1.0,
        num_knots: int = 4,
        spline_type: InterpMethod = "zero",
        seed: int = 0,
        track_predictions: bool = False,
        **kwargs,
    ) -> None:
        """See :class:`SamplingPlanner`; adds the elite-refit hyperparameters.

        ``init_std`` is the initial per-knot sampling std, ``elite_frac`` the
        fraction of samples kept as elites each update, and ``min_std`` a
        floor so sampling never collapses to zero. ``num_elite`` overrides
        ``elite_frac`` with an absolute count -- see :meth:`_elite_count` for
        when that is the one you want. ``adapt_covariance``, if
        True (standard CEM), refits the std from the elite samples every
        update; if False, it stays fixed at ``init_std`` and only the mean
        is refit. ``track_predictions`` extracts the best (lowest-cost)
        sample's full-horizon state prediction each plan step (into
        ``self._predicted_state`` / ``PlanInfo.predicted_state``); requires
        the engine to have been built with ``record_predictions=True``.
        """
        super().__init__(
            task,
            engine,
            risk_strategy=risk_strategy,
            plan_horizon=plan_horizon,
            num_samples=num_samples,
            num_knots=num_knots,
            spline_type=spline_type,
            seed=seed,
            **kwargs,
        )
        if not 0.0 < elite_frac <= 1.0:
            raise ValueError("elite_frac must be in (0, 1]")
        if num_elite is not None and num_elite < 1:
            raise ValueError("num_elite must be >= 1")
        self.init_std = float(init_std)
        self.elite_frac = float(elite_frac)
        self.fixed_num_elite = None if num_elite is None else int(num_elite)
        self.min_std = float(min_std)
        self.adapt_covariance = bool(adapt_covariance)
        self.track_predictions = bool(track_predictions)
        self.num_elite = self._elite_count(num_samples)
        self.std = np.full((self.num_knots, self.task.nu), self.init_std)

    def _elite_count(self, num_samples: int) -> int:
        """How many elites to refit from, at this sample count.

        ``num_elite`` fixes the count; otherwise it is ``elite_frac`` of the
        samples. The two differ in *what stays constant as the sample count
        changes*, which matters whenever runs at different ``num_samples`` are
        compared:

        * A fixed **fraction** holds selectivity constant (the top 5% either
          way) and lets the elite count move with the samples -- so the
          variance of the mean-over-elites moves with it too.
        * A fixed **count** holds that variance constant and lets selectivity
          move instead (13 of 4096 is far greedier than 13 of 256).

        Neither is sample-count-invariant; the choice is which one moves. Fix
        the count when the comparison is about *selection*, so a difference
        cannot be a noisier refit instead.
        """
        if self.fixed_num_elite is not None:
            return min(self.fixed_num_elite, num_samples)
        return max(1, int(round(num_samples * self.elite_frac)))

    def set_num_samples(self, num_samples: int) -> None:  # noqa: D102
        super().set_num_samples(num_samples)
        self.num_elite = self._elite_count(self.num_samples)

    def sample_knots(self, params: SamplingParams) -> np.ndarray:  # noqa: D102
        noise = self.rng.standard_normal(
            (self.num_samples, self.num_knots, self.task.nu)
        )
        knots = params.mean + self.std * noise
        # Keep the incumbent mean as a candidate, so cost never increases.
        knots[0] = params.mean
        return knots

    def update_params(
        self,
        params: SamplingParams,
        sample_costs: np.ndarray,
        knots: np.ndarray,
    ) -> SamplingParams:
        """Refit the mean (and, if enabled, the std) to the elites."""
        order = np.argsort(sample_costs)
        elite = knots[order[: self.num_elite]]
        mean = elite.mean(axis=0)
        if self.adapt_covariance:
            self.std = np.maximum(elite.std(axis=0), self.min_std)
        if self.track_predictions:
            self._predicted_state = self.engine.extract_predicted_state(
                [int(order[0])]
            )
        return SamplingParams(tk=params.tk, mean=mean)
