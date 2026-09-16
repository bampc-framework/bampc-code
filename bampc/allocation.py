"""Adaptive domain/sample budget allocation.

The engine operates on one flat ``nworld = R * S`` axis (see
``rollout.engine``); nothing device-side cares how that total splits between
randomizations ``R`` and samples ``S``. This module turns that fact into a
switchable *stage* concept: given a starting ``(R, S)``, precompute every
``(R, S)`` pair whose product equals the same budget ``N`` (a "ladder"), and
let an :class:`AllocationPolicy` pick a stage index each replan. The current
fixed-``(R, S)`` behavior is the trivial one-stage case.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:
    from bampc.belief import DomainBelief
    from bampc.dr.randomizer import RandomizationSpec
    from bampc.planner.base import SamplingPlanner
    from bampc.rollout.warp_engine import WarpRolloutEngine

_MIN_STAGES_WARN = 4


def divisor_ladder(
    num_randomizations: int, num_samples: int
) -> list[tuple[int, int]]:
    """Every ``(R, S)`` split of the fixed budget ``N = R0 * S0``.

    ``R`` must divide ``N`` exactly, so the ladder is ``N``'s divisors paired
    with their quotient, ascending by ``R``. How rich it is depends entirely
    on how ``N`` factors -- a prime-ish ``N`` gives almost no intermediate
    stages.

    Args:
        num_randomizations: Starting ``R`` (need not be a ladder endpoint).
        num_samples: Starting ``S``.

    Returns:
        ``(R, S)`` pairs, ascending by ``R``; always includes ``(1, N)`` and
        ``(N, 1)``.
    """
    n = int(num_randomizations) * int(num_samples)
    divs = [d for d in range(1, int(n**0.5) + 1) if n % d == 0]
    divs = sorted({d for d in divs} | {n // d for d in divs})
    ladder = [(d, n // d) for d in divs]
    if len(ladder) <= _MIN_STAGES_WARN:
        warnings.warn(
            f"budget N={n} (from {num_randomizations}x{num_samples}) factors "
            f"poorly -- only {len(ladder)} stage(s) available: {ladder}. "
            "Pick a more composite starting split (e.g. a power of 2) for a "
            "usable ladder.",
            stacklevel=2,
        )
    return ladder


@dataclass
class AllocationContext:
    """Everything an :class:`AllocationPolicy` might read to decide.

    A given policy only reads the field(s) it cares about; the rest stay at
    their default. Keeping one shared context means the driver loop's call
    site never changes when the policy is swapped.

    Attributes:
        nudge: Manual stage delta (``-1``/``0``/``+1``), e.g. from a keypress.
        error_window: Recent per-domain prediction errors (oldest first),
            each shape ``(R,)`` -- see ``tracking.PredictionTracker.window``.
            ``None`` until a tracker has run at least one replan.
    """

    nudge: int = 0
    error_window: list[np.ndarray] | None = None


class AllocationPolicy(ABC):
    """Decide which ladder stage to be at, given the current one and signal."""

    @abstractmethod
    def decide(
        self, current_idx: int, num_stages: int, context: AllocationContext
    ) -> int:
        """Return the stage index to move to (``current_idx`` = no change)."""


class ManualStagePolicy(AllocationPolicy):
    """Move by ``context.nudge`` stages, clamped to the ladder's ends."""

    def decide(  # noqa: D102
        self, current_idx: int, num_stages: int, context: AllocationContext
    ) -> int:
        return int(np.clip(current_idx + context.nudge, 0, num_stages - 1))


class BeliefCollapsePolicy(AllocationPolicy):
    """Shift toward more samples as a :class:`DomainBelief` narrows.

    Reads ``context.error_window`` each call to refresh the belief's
    posterior over one randomized field's per-domain value, then moves one
    stage toward smaller ``R`` (more ``S``) once the posterior has collapsed.

    With ``widen_threshold`` set it also runs the reverse: if even the
    best-fitting domain predicts reality badly, the truth has moved outside
    the sampled range, so the belief is reset to its prior and the ladder
    jumps back to the wide stage to re-span it. Left unset (the default) the
    policy only ever narrows.

    Two guards, because narrowing discards domains irreversibly (the ladder
    cannot step back without a full widen). ``confirmations`` requires the
    posterior to land in the same place more than once, against a single
    unlucky measurement. ``reject_ratio`` requires the evidence to have
    actually *excluded* domains rather than merely being narrow -- see
    :meth:`DomainBelief.excluded_frac` for why ``std`` alone cannot say that.
    """

    def __init__(
        self,
        engine: WarpRolloutEngine,
        belief: DomainBelief,
        std_threshold: float,
        field: str = "actuator_gainprm",
        component: int = 0,
        entity: int = 0,
        widen_threshold: float | None = None,
        home_idx: int | None = None,
        confirmations: int = 1,
        reject_ratio: float | None = None,
        min_excluded: float = 0.5,
    ) -> None:
        """Bind the engine/belief this policy reads and its switch bars.

        Args:
            engine: Source of ``last_overrides`` -- the per-domain values the
                belief scores against ``context.error_window``.
            belief: Posterior to update each call.
            std_threshold: Collapse bar, in the parameter's own units.
            field: Randomized model field to read from ``last_overrides``.
            component: Column of that field's last axis to read, e.g. 0 for
                sliding friction in ``geom_friction[:, entity, component]``.
            entity: Index along that field's entity axis. Only an
                ``"__all__"``-targeted param writes every entity and so reads
                correctly at the default 0. A spec naming one geom/body
                writes only that column; every other stays at the baseline,
                which would hand the belief a constant and collapse it
                instantly on a value it never randomized.
            widen_threshold: Bar on ``belief.best_error`` above which the
                range is judged wrong and re-spanned. ``None`` disables
                widening entirely.
            home_idx: Stage to jump back to when widening. ``None`` latches
                the first stage this policy is asked about, so the ladder's
                starting index need not be known before
                :meth:`AllocationController.build` has computed it.
            confirmations: Consecutive collapsed updates whose posterior
                means must agree to within ``std_threshold`` before
                narrowing. A disagreeing update restarts the count.
            reject_ratio: How much worse than the best domain a domain's
                error must be to count as excluded, as a multiple. ``None``
                (the default) skips the check and narrows on ``std`` alone.
            min_excluded: Fraction of domains that must be excluded before a
                narrowing is applied. Only read when ``reject_ratio`` is set.
        """
        self.engine = engine
        self.belief = belief
        self.std_threshold = std_threshold
        self.field = field
        self.component = component
        self.entity = entity
        self.widen_threshold = widen_threshold
        self.home_idx = home_idx
        self.confirmations = max(int(confirmations), 1)
        self.reject_ratio = reject_ratio
        self.min_excluded = float(min_excluded)
        self.streak = 0
        self.last_mean: float | None = None

    def decide(  # noqa: D102
        self, current_idx: int, num_stages: int, context: AllocationContext
    ) -> int:
        if self.home_idx is None:
            self.home_idx = current_idx
        if not context.error_window or self.engine.last_overrides is None:
            return current_idx
        values = self.engine.last_overrides[self.field][
            :, self.entity, self.component
        ]
        self.belief.update(values, context.error_window)
        if self.widen_threshold is not None and self.belief.mismatched(
            self.widen_threshold
        ):
            # Before returning, not after: maybe_switch calls spec_fn as soon
            # as this returns, and it must read the restored prior rather than
            # the posterior that just failed to explain reality.
            self.belief.reset_prior()
            self.streak, self.last_mean = 0, None
            return self.home_idx
        if not self.belief.collapsed(self.std_threshold):
            self.streak, self.last_mean = 0, None
            return current_idx
        if (
            self.reject_ratio is not None
            and self.belief.excluded_frac(self.reject_ratio)
            < self.min_excluded
        ):
            # A narrow posterior over domains that all fit about equally is
            # not knowledge -- it is a grid that no longer spans anything
            # worth telling apart. Hold the stage and keep hedging.
            self.streak, self.last_mean = 0, None
            return current_idx

        agrees = (
            self.last_mean is not None
            and abs(self.belief.mean - self.last_mean) <= self.std_threshold
        )
        self.streak = self.streak + 1 if agrees else 1
        self.last_mean = self.belief.mean
        if self.streak < self.confirmations:
            return current_idx
        self.streak = 0
        return current_idx - 1


class SpecFn(Protocol):
    """Rebuild a randomization spec for a new domain count ``R``."""

    def __call__(self, num_randomizations: int) -> RandomizationSpec:
        """Return a spec whose grid-valued entries have length ``R``."""
        ...


class WorstHalfPolicy(AllocationPolicy):
    """Narrow by keeping the better half of domains, not a softmax posterior.

    A simpler alternative to :class:`BeliefCollapsePolicy` for interactive
    use: every call reads only THIS shot's raw per-domain error -- no
    running confirmations, no reject-ratio gate, no persistent weighting.
    Sort domains by error, keep the better half, and propose moving exactly
    one ladder stage toward more samples, with the hedge bracket set to
    ``[min, max]`` of the survivors' current values -- the same 1/mu-spaced
    grid a ``SpecFn`` always draws, just over a caller-chosen span instead of
    one derived from a posterior mean/std.

    Unlike :class:`BeliefCollapsePolicy` this mutates ``spec_fn`` directly
    from :meth:`decide` -- there is no persistent belief for ``spec_fn`` to
    read, so the bracket has to be written somewhere. To decline a decision,
    snapshot ``dict(spec_fn.bracket)`` beforehand and restore it verbatim,
    exactly as a caller would snapshot ``belief.__dict__`` for the other
    policy.

    ``widen_threshold`` mirrors :class:`BeliefCollapsePolicy`'s. ``None``
    disables widening -- the policy only ever narrows.
    """

    def __init__(
        self,
        engine: WarpRolloutEngine,
        spec_fn: SpecFn,
        widen_threshold: float | None = None,
        home_idx: int | None = None,
        field: str = "actuator_gainprm",
        component: int = 0,
        entity: int = 0,
        belief: DomainBelief | None = None,
    ) -> None:
        """Bind the engine/spec_fn this policy reads and mutates.

        Args:
            engine: Source of ``last_overrides`` -- the per-domain values
                this policy ranks by error.
            spec_fn: The ``SpecFn`` whose ``.bracket`` this policy writes
                directly on every narrow/widen decision (see class
                docstring). Must be a manually-driven bracket, e.g.
                ``hedge.make_manual_spec_fn`` -- not ``hedge.make_spec_fn``,
                whose bracket is instead derived from a ``DomainBelief``.
            widen_threshold: Bar on the best domain's error above which the
                range is judged wrong and re-spanned. ``None`` disables
                widening entirely.
            home_idx: Stage to jump back to when widening. ``None`` latches
                the first stage this policy is asked about.
            field: Randomized model field to read from ``last_overrides``.
            component: Column of that field's last axis to read.
            entity: Index along that field's entity axis (see
                :class:`BeliefCollapsePolicy`'s own note on this).
            belief: Display-only. Its ``update``/``reset_prior`` are still
                called so a caller can keep printing ``hedge.report``'s
                columns; this policy's decision never reads it.
        """
        self.engine = engine
        self.spec_fn = spec_fn
        self.widen_threshold = widen_threshold
        self.home_idx = home_idx
        self.field = field
        self.component = component
        self.entity = entity
        self.belief = belief

    def decide(  # noqa: D102
        self, current_idx: int, num_stages: int, context: AllocationContext
    ) -> int:
        if self.home_idx is None:
            self.home_idx = current_idx
        if not context.error_window or self.engine.last_overrides is None:
            return current_idx
        values = self.engine.last_overrides[self.field][
            :, self.entity, self.component
        ]
        error = np.mean(np.stack(context.error_window), axis=0)
        if self.belief is not None:
            self.belief.update(values, context.error_window)
        if (
            self.widen_threshold is not None
            and float(np.min(error)) > self.widen_threshold
        ):
            if self.belief is not None:
                self.belief.reset_prior()
            self.spec_fn.reset()
            return self.home_idx
        if current_idx == 0:
            # Already at the finest split (R=1) -- nothing left to halve.
            return current_idx
        keep = max(values.shape[0] // 2, 1)
        order = np.argsort(error)
        survivors = values[order[:keep]]
        self.spec_fn.bracket["lo"] = float(np.min(survivors))
        self.spec_fn.bracket["hi"] = float(np.max(survivors))
        return current_idx - 1


class AllocationController:
    """Owns the ladder + current stage; applies switches across the stack.

    A switch touches three collaborators, in this order: the engine's
    ``(R, S)`` (buffers are already sized for the fixed budget, so this is
    just bookkeeping -- no CUDA-graph rebuild), the planner's ``num_samples``
    (and anything an algorithm derives from it, e.g. CEM's elite count), and
    the domain randomizer, which is fully *redrawn* at the new ``R`` (no
    attempt to preserve domain identity across a switch -- see module
    docstring in ``dr.randomizer`` for why grid-valued spec entries need
    ``spec_fn`` to regenerate at the new length).
    """

    def __init__(
        self,
        engine: WarpRolloutEngine,
        planner: SamplingPlanner,
        stages: list[tuple[int, int]],
        start_idx: int,
        policy: AllocationPolicy,
        spec_fn: SpecFn | None = None,
    ) -> None:
        """Bind a precomputed ladder and starting stage to a policy.

        Args:
            engine: The rollout engine whose allocation gets switched.
            planner: The planner sharing the engine's sample count.
            stages: Ladder from :func:`divisor_ladder`.
            start_idx: Index into ``stages`` matching the engine/planner's
                current ``(R, S)``.
            policy: Decides the next stage each call to :meth:`maybe_switch`.
            spec_fn: Rebuilds the randomization spec at a new ``R`` (only
                needed if the engine has a randomizer with grid-valued
                entries; range-valued entries redraw at any ``R`` for free).
        """
        self.engine = engine
        self.planner = planner
        self.stages = stages
        self.current_idx = start_idx
        self.policy = policy
        self.spec_fn = spec_fn

    @classmethod
    def build(
        cls,
        engine: WarpRolloutEngine,
        planner: SamplingPlanner,
        policy: AllocationPolicy,
        spec_fn: SpecFn | None = None,
    ) -> AllocationController:
        """Derive the ladder from the engine's current ``(R, S)``."""
        r0, s0 = engine.num_randomizations, engine.num_samples
        stages = divisor_ladder(r0, s0)
        start_idx = stages.index((r0, s0))
        return cls(engine, planner, stages, start_idx, policy, spec_fn)

    @property
    def current(self) -> tuple[int, int]:
        """The active ``(R, S)``."""
        return self.stages[self.current_idx]

    def maybe_switch(self, context: AllocationContext) -> bool:
        """Ask the policy for a stage and apply it if it changed.

        Returns:
            ``True`` if the stage (and therefore ``(R, S)``) changed.
        """
        num_stages = len(self.stages)
        new_idx = self.policy.decide(self.current_idx, num_stages, context)
        return self.apply_stage(new_idx)

    def apply_stage(self, new_idx: int) -> bool:
        """Move to ``new_idx``, applying the switch across the stack.

        Split out of :meth:`maybe_switch` so a caller can put something
        between the decision and the switch -- a confirmation prompt, say.
        Such a caller drives the policy itself (``policy.decide(...)``) and
        passes the result here.

        Returns:
            ``True`` if the stage (and therefore ``(R, S)``) changed.
        """
        new_idx = int(np.clip(new_idx, 0, len(self.stages) - 1))
        if new_idx == self.current_idx:
            return False

        num_randomizations, num_samples = self.stages[new_idx]
        self.engine.set_allocation(num_randomizations, num_samples)
        self.planner.set_num_samples(num_samples)
        if self.engine.randomizer is not None and self.spec_fn is not None:
            self.engine.randomizer.num_randomizations = num_randomizations
            self.engine.randomizer.update(self.spec_fn(num_randomizations))
        self.engine.resample_randomizations()

        self.current_idx = new_idx
        return True
