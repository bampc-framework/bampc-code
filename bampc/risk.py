"""Risk strategies: reduce per-domain costs to a single robust objective.

Backend-agnostic host numpy, operating on the randomization axis (axis 0)
of the cost array.

Convention
----------
``combine_costs`` reduces the randomization axis (axis 0): it takes per-domain
total costs of shape ``(num_randomizations, num_samples)`` and returns one
scalar cost per sample, ``(num_samples,)``. The horizon is already summed on
device inside the cost kernel, so only this small tensor is copied back.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class RiskStrategy(ABC):
    """How to combine costs across domain randomizations."""

    @abstractmethod
    def combine_costs(self, costs: np.ndarray) -> np.ndarray:
        """Reduce the randomization axis.

        Args:
            costs: Per-domain total rollout costs, shape
                ``(randomizations, samples)``.

        Returns:
            Combined cost per sample, shape ``(samples,)``.
        """


class AverageCost(RiskStrategy):
    """Expected cost over randomizations (the RL-style default)."""

    def combine_costs(self, costs: np.ndarray) -> np.ndarray:  # noqa: D102
        return np.mean(costs, axis=0)


class WorstCase(RiskStrategy):
    """Pessimistic max cost over randomizations."""

    def combine_costs(self, costs: np.ndarray) -> np.ndarray:  # noqa: D102
        return np.max(costs, axis=0)


class BestCase(RiskStrategy):
    """Optimistic min cost over randomizations."""

    def combine_costs(self, costs: np.ndarray) -> np.ndarray:  # noqa: D102
        return np.min(costs, axis=0)


class ExponentialWeightedAverage(RiskStrategy):
    """Softmax-weighted average; ``gamma > 0`` is risk-averse."""

    def __init__(self, gamma: float) -> None:
        """Set the risk-aversion parameter ``gamma``."""
        self.gamma = gamma

    def combine_costs(self, costs: np.ndarray) -> np.ndarray:  # noqa: D102
        z = self.gamma * costs
        z = z - np.max(z, axis=0, keepdims=True)
        w = np.exp(z)
        w = w / np.sum(w, axis=0, keepdims=True)
        return np.sum(w * costs, axis=0)


class ValueAtRisk(RiskStrategy):
    """Cost at the ``(1 - alpha)`` quantile across randomizations."""

    def __init__(self, alpha: float) -> None:
        """Set the quantile level ``alpha``."""
        self.alpha = alpha

    def combine_costs(self, costs: np.ndarray) -> np.ndarray:  # noqa: D102
        return np.quantile(costs, 1.0 - self.alpha, axis=0)


class ConditionalValueAtRisk(RiskStrategy):
    """Expected cost in the tail beyond the ``(1 - alpha)`` quantile."""

    def __init__(self, alpha: float) -> None:
        """Set the quantile level ``alpha``."""
        self.alpha = alpha

    def combine_costs(self, costs: np.ndarray) -> np.ndarray:  # noqa: D102
        quant = np.quantile(costs, 1.0 - self.alpha, axis=0, keepdims=True)
        masked = np.where(costs >= quant, costs, np.nan)
        return np.nanmean(masked, axis=0)


class InverseConditionalValueAtRisk(RiskStrategy):
    """Expected cost in the best ``alpha`` fraction (mirror of CVaR)."""

    def __init__(self, alpha: float) -> None:
        """Set the quantile level ``alpha``."""
        self.alpha = alpha

    def combine_costs(self, costs: np.ndarray) -> np.ndarray:  # noqa: D102
        quant = np.quantile(costs, self.alpha, axis=0, keepdims=True)
        masked = np.where(costs <= quant, costs, np.nan)
        return np.nanmean(masked, axis=0)