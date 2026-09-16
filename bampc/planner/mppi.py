"""Model-predictive path integral control (host algorithm)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from bampc.planner.base import SamplingParams, SamplingPlanner

if TYPE_CHECKING:
    from bampc.risk import RiskStrategy
    from bampc.rollout.engine import RolloutEngine
    from bampc.spline import InterpMethod
    from bampc.task.base import Task


class MPPI(SamplingPlanner):
    """Exponentially-weighted average over rollouts (MPPI-generic)."""

    def __init__(
        self,
        task: Task,
        engine: RolloutEngine,
        *,
        risk_strategy: RiskStrategy,
        num_samples: int,
        noise_level: float,
        temperature: float,
        plan_horizon: float = 1.0,
        num_knots: int = 4,
        spline_type: InterpMethod = "zero",
        seed: int = 0,
        **kwargs,
    ) -> None:
        """See :class:`SamplingPlanner`; adds noise_level + temperature."""
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
        self.temperature = float(temperature)

    def sample_knots(self, params: SamplingParams) -> np.ndarray:  # noqa: D102
        noise = self.rng.standard_normal(
            (self.num_samples, self.num_knots, self.task.nu)
        )
        return params.mean + self.noise_level * noise

    def update_params(
        self,
        params: SamplingParams,
        sample_costs: np.ndarray,
        knots: np.ndarray,
    ) -> SamplingParams:
        """Softmax-weighted mean of the sampled knots (MPPI update)."""
        z = -(sample_costs - sample_costs.min()) / self.temperature
        weights = np.exp(z)
        weights /= weights.sum()
        mean = np.einsum("s,skn->kn", weights, knots)
        return SamplingParams(tk=params.tk, mean=mean)