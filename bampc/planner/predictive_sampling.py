"""Predictive sampling (host algorithm)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from bampc.planner.base import SamplingParams, SamplingPlanner

if TYPE_CHECKING:
    from bampc.risk import RiskStrategy
    from bampc.rollout.engine import RolloutEngine
    from bampc.spline import InterpMethod
    from bampc.task.base import Task


class PredictiveSampling(SamplingPlanner):
    """Take the lowest-cost rollout each iteration (arxiv 2212.00541)."""

    def __init__(
        self,
        task: Task,
        engine: RolloutEngine,
        *,
        risk_strategy: RiskStrategy,
        num_samples: int,
        noise_level: float,
        plan_horizon: float = 1.0,
        num_knots: int = 4,
        spline_type: InterpMethod = "zero",
        seed: int = 0,
        track_predictions: bool = False,
        **kwargs,
    ) -> None:
        """See :class:`SamplingPlanner`; adds ``noise_level``.

        ``track_predictions`` extracts the winning knot sequence's full-
        horizon state prediction each plan step (into
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
        self.noise_level = float(noise_level)
        self.track_predictions = bool(track_predictions)

    def sample_knots(self, params: SamplingParams) -> np.ndarray:  # noqa: D102
        noise = self.rng.standard_normal(
            (self.num_samples, self.num_knots, self.task.nu)
        )
        knots = params.mean + self.noise_level * noise
        # Keep the incumbent mean as sample 0, so cost never increases.
        knots[0] = params.mean
        return knots

    def update_params(
        self,
        params: SamplingParams,
        sample_costs: np.ndarray,
        knots: np.ndarray,
    ) -> SamplingParams:
        """Pick the lowest-cost sample as the new mean (predictive sampling)."""
        best = int(np.argmin(sample_costs))
        if self.track_predictions:
            self._predicted_state = self.engine.extract_predicted_state(
                [best]
            )
        return SamplingParams(tk=params.tk, mean=knots[best])