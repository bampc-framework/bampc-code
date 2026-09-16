"""Serializable planner hyperparameters, and the factories that use them.

One dataclass naming everything a sampling planner needs, plus
``build_planner`` / ``build_risk`` to turn it into live objects. Lets an
example, a script and a sweep config all describe a planner the same way
instead of each repeating an ``algo``-dispatch chain.
"""

from __future__ import annotations

from dataclasses import dataclass

from bampc.planner.base import SamplingPlanner
from bampc.planner.cem import CEM
from bampc.planner.mppi import MPPI
from bampc.planner.predictive_sampling import PredictiveSampling
from bampc.risk import (
    AverageCost,
    BestCase,
    ConditionalValueAtRisk,
    ExponentialWeightedAverage,
    InverseConditionalValueAtRisk,
    RiskStrategy,
    ValueAtRisk,
    WorstCase,
)


@dataclass(frozen=True)
class PlannerConfig:
    """Serializable planner hyperparameters (one ``planner`` YAML file)."""

    algo: str  # ps | mppi | cem
    # average | worstcase | bestcase | cvar | var | ewa | inverse_cvar
    risk: str
    noise_level: float
    # Per-domain sample count S, used flatly when sample_budget is None --
    # see engine_shape(). A profile's value is a RECOMMENDATION, not a
    # floor: a count verified to make that task work sim-to-sim here, not a
    # budget for your hardware -- how much nworld fits a replan period
    # depends on the GPU and on settle_steps (each one is another physics
    # step in every world). See configs/README.md.
    num_samples: int
    plan_horizon: float
    num_knots: int
    spline_type: str
    # Total nworld to split across R domains as S = sample_budget // R --
    # see engine_shape(). None (every plain
    # example's and every profile's default) means S is just num_samples,
    # flatly, regardless of R. It is measured per machine, so it is never a
    # profile value: only a caller opts in, either a sweep config's
    # `planner.overrides.sample_budget` or a ROS launcher's --sample-budget.
    sample_budget: int | None = None
    temperature: float = 0.01  # mppi
    init_std: float = 0.3  # cem
    elite_frac: float = 0.1  # cem
    # cem: absolute elite count, overriding elite_frac. Set it when runs at
    # different num_samples are compared -- a fraction keeps selectivity fixed
    # and lets the refit get noisier, a count does the reverse.
    num_elite: int | None = None
    min_std: float = 1e-3  # cem
    adapt_covariance: bool = False  # cem
    risk_alpha: float = 0.1  # cvar / var
    risk_gamma: float = 1.0  # ewa
    settle_steps: int = 0  # zero-control steps before each rollout
    # Extract the actual best-cost rolled-out sample's prediction, instead of
    # nothing (mppi) or a mismatched one (cem's old track_predictions bug).
    # Does NOT change what gets executed -- that stays the algorithm's own
    # (possibly averaged) mean. No-op for ps either way (already true by
    # construction) -- see SamplingPlanner.__init__.
    best_sample_execution: bool = False
    # Shrinks a task's raw MJCF actuator ctrlrange to this fraction before
    # it becomes the sampling bound (see FlipFr3/BalanceFr3's own
    # ctrl_range_scale docstring for why this belongs to the planner's
    # exploration range, not the task's reward). 1.0 (default) is a no-op,
    # keeping the raw actuator range.
    ctrl_range_scale: float = 1.0


def engine_shape(p: PlannerConfig, num_randomizations: int) -> tuple[int, int]:
    """``(R, S)``: R domains, S = ``sample_budget // R``.

    S falls back to ``num_samples`` flatly when ``sample_budget`` is
    ``None`` (every plain example's default -- growing R there does not
    shrink S). Any split is allowed: ``num_samples`` is a recommendation,
    not a floor (``configs/README.md``), so a large R may legitimately take
    S below it -- only ``S >= 1`` is enforced. Every caller that fans a
    fixed total budget across a variable domain count must go through this
    rather than dividing ``sample_budget`` itself, so the split can't
    silently drift between call sites.
    """
    r = int(num_randomizations)
    if p.sample_budget is None:
        return r, p.num_samples
    return r, max(p.sample_budget // r, 1)


_RISK_FACTORIES = {
    "average": lambda p: AverageCost(),
    "worstcase": lambda p: WorstCase(),
    "bestcase": lambda p: BestCase(),
    "cvar": lambda p: ConditionalValueAtRisk(p.risk_alpha),
    "inverse_cvar": lambda p: InverseConditionalValueAtRisk(p.risk_alpha),
    "var": lambda p: ValueAtRisk(p.risk_alpha),
    "ewa": lambda p: ExponentialWeightedAverage(p.risk_gamma),
}


def build_risk(p: PlannerConfig) -> RiskStrategy:
    """Build the risk strategy named by ``planner.risk``."""
    try:
        return _RISK_FACTORIES[p.risk](p)
    except KeyError:
        raise ValueError(f"unknown risk strategy: {p.risk!r}") from None


def build_planner(
    p: PlannerConfig,
    task,
    engine,
    *,
    track_predictions: bool = False,
    state_uncertainty=None,
    seed: int = 0,
) -> SamplingPlanner:
    """Build the planner named by ``p.algo``, with its risk strategy.

    Args:
        p: The hyperparameters.
        task: Task being planned for.
        engine: Rollout engine the planner drives.
        track_predictions: Keep per-domain predicted trajectories (not
            supported by MPPI, which raises).
        state_uncertainty: A ``bampc.uncertainty.StateUncertainty``, or
            ``None``. A live object rather than config, hence a keyword
            argument -- unlike ``settle_steps``, which is serializable and
            lives on :class:`PlannerConfig`.
        seed: Sampling seed.

    Returns:
        A ready planner.
    """
    common = {
        "risk_strategy": build_risk(p),
        "num_samples": p.num_samples,
        "plan_horizon": p.plan_horizon,
        "num_knots": p.num_knots,
        "spline_type": p.spline_type,
        "seed": seed,
        "state_uncertainty": state_uncertainty,
        "settle_steps": p.settle_steps,
        "best_sample_execution": p.best_sample_execution,
    }
    if p.algo == "ps":
        return PredictiveSampling(
            task,
            engine,
            noise_level=p.noise_level,
            track_predictions=track_predictions,
            **common,
        )
    if p.algo == "mppi":
        if track_predictions:
            raise ValueError("MPPI does not support track_predictions")
        return MPPI(
            task,
            engine,
            noise_level=p.noise_level,
            temperature=p.temperature,
            **common,
        )
    if p.algo == "cem":
        return CEM(
            task,
            engine,
            init_std=p.init_std,
            elite_frac=p.elite_frac,
            num_elite=p.num_elite,
            min_std=p.min_std,
            adapt_covariance=p.adapt_covariance,
            track_predictions=track_predictions,
            **common,
        )
    raise ValueError(f"unknown planner algo: {p.algo!r}")
