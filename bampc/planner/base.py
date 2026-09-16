"""Host-side sampling planner: the receding-horizon control loop.

The planner is a stateful, synchronous, **viewer-free** object. It holds the
current spline parameters between calls and drives the rollout engine. This is
the unit a ROS 2 node owns (see :mod:`bampc.ros.planner_node`): the node
feeds it a :class:`StateSnapshot`, calls :meth:`optimize` on a planning timer,
and queries :meth:`get_action` on a faster control timer.

Three rates, deliberately decoupled for sim-to-real:

1. **plan rate** (GPU): :meth:`optimize` — sample, roll out, aggregate risk,
   update params.
2. **control rate** (host): :meth:`get_action` — a cheap spline query, no GPU.
3. **actuator rate** (host): ``task.control_map_host`` — applied by the driver.

Subclasses implement only :meth:`sample_knots` and :meth:`update_params`; the
:meth:`optimize` template owns the wiring.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import mujoco
import numpy as np

from bampc import spline

if TYPE_CHECKING:
    from bampc.risk import RiskStrategy
    from bampc.rollout.engine import PredictedState, RolloutEngine
    from bampc.spline import InterpMethod
    from bampc.task.base import Task
    from bampc.uncertainty import StateUncertainty


@dataclass
class StateSnapshot:
    """A current-state estimate handed to the planner.

    Decoupled from any simulator: a ROS node fills this from its state
    estimator; an interactive driver fills it from ``mj_data``.
    """

    qpos: np.ndarray
    qvel: np.ndarray
    time: float = 0.0
    mocap_pos: np.ndarray | None = None
    mocap_quat: np.ndarray | None = None


@dataclass
class SamplingParams:
    """Policy parameters: the control-spline knot distribution.

    Attributes:
        tk: Knot times, shape ``(num_knots,)``.
        mean: Mean knots, shape ``(num_knots, nu)``.
    """

    tk: np.ndarray
    mean: np.ndarray


@dataclass
class PlanInfo:
    """Diagnostics from a plan step (for logging / visualization)."""

    sample_costs: np.ndarray  # (num_samples,)
    knots: np.ndarray  # (num_samples, num_knots, nu)
    trace_sites: np.ndarray | None = None
    predicted_state: PredictedState | None = None
    timings: dict | None = None  # {"stage", "rollout"} seconds
    contact_modes: np.ndarray | None = None  # (R,) per-domain contact bitmask
    # {body: ((R, 3) pos, (R, 4) quat)} where each domain believes the body
    # is, captured after settling and *before* the rollout overwrites it.
    belief_pose: dict[str, tuple[np.ndarray, np.ndarray]] | None = None
    # (R, S) per-domain rollout costs, before the risk strategy reduces the
    # domain axis. Lets a caller read how much the domains disagree about a
    # sample's cost (the model-mismatch sweep's disagreement metric); ``None``
    # for callers that never look.
    domain_costs: np.ndarray | None = None


class SamplingPlanner(ABC):
    """Generic sampling-based MPC loop over a Warp rollout engine."""

    def __init__(
        self,
        task: Task,
        engine: RolloutEngine,
        *,
        risk_strategy: RiskStrategy,
        plan_horizon: float,
        num_samples: int,
        num_knots: int = 4,
        spline_type: InterpMethod = "zero",
        seed: int = 0,
        state_uncertainty: StateUncertainty | None = None,
        settle_steps: int = 0,
        best_sample_execution: bool = False,
    ) -> None:
        """Wire the planner to its task, engine and risk strategy.

        Args:
            task: Defines costs, control mapping and bounds.
            engine: Batched rollout backend (already constructed with matching
                ``num_samples`` / ``num_randomizations``).
            risk_strategy: Reduces the per-domain cost axis.
            plan_horizon: Planning horizon in seconds.
            num_samples: Control sequences sampled per iteration.
            num_knots: Spline knots in the control parameterization.
            spline_type: Knot interpolation method.
            seed: Seed for the control-noise RNG.
            state_uncertainty: Fans the state estimate into one belief per
                domain. ``None`` (default) uploads the exact state to every
                world, as before.
            settle_steps: Zero-control steps run before each rollout to
                resolve interpenetration.
            best_sample_execution: Extract the lowest-cost sample's
                full-horizon prediction into ``PlanInfo.predicted_state``
                (needs the engine built with ``record_predictions=True``), so
                every algorithm can be scored against reality the same way.
                Only :class:`PredictiveSampling` guarantees its committed mean
                *is* the best sample; MPPI's is a softmax-weighted average and
                CEM's an elite average, so those two otherwise have no single
                simulated rollout to score against. Does **not** change what
                is executed. Safe to decouple only because the one caller
                (Curling-FR3's friction ID) scores a replan strictly after
                release, where the puck is ballistic and no control choice
                changes the outcome -- the prediction only has to come from
                *some* real rollout. A caller needing the executed action to
                match the scored rollout would need a different mechanism.
        """
        self.task = task
        self.engine = engine
        self.risk = risk_strategy
        self.plan_horizon = float(plan_horizon)
        self.num_samples = int(num_samples)
        self.num_knots = int(num_knots)
        self.spline_type: InterpMethod = spline_type
        self.dt = task.dt
        self.ctrl_steps = int(round(self.plan_horizon / self.dt))
        self.rng = np.random.default_rng(seed)
        self._predicted_state: PredictedState | None = None
        self.state_uncertainty = state_uncertainty
        self.settle_steps = int(settle_steps)
        self.best_sample_execution = bool(best_sample_execution)
        self._predict_data: mujoco.MjData | None = None
        self.engine.build_graph(
            horizon=self.ctrl_steps, settle_steps=self.settle_steps
        )

    def init_params(
        self, initial_knots: np.ndarray | None = None
    ) -> SamplingParams:
        """Create the initial spline parameters."""
        if initial_knots is None:
            mean = np.zeros((self.num_knots, self.task.nu))
        else:
            mean = np.asarray(initial_knots, dtype=float)
        if mean.shape != (self.num_knots, self.task.nu):
            raise ValueError(
                f"initial_knots must be {(self.num_knots, self.task.nu)}, "
                f"got {mean.shape}"
            )
        tk = np.linspace(0.0, self.plan_horizon, self.num_knots)
        return SamplingParams(tk=tk, mean=mean)

    def optimize(
        self,
        state: StateSnapshot,
        params: SamplingParams,
        dt_lag: float = 0.0,
    ) -> tuple[SamplingParams, PlanInfo]:
        """Run one receding-horizon plan step.

        Warm-starts the spline to ``state.time``, then: sample knots ->
        interpolate -> roll out (with randomization) -> combine risk ->
        update params.

        Args:
            state: Current state estimate.
            params: Previous spline parameters (warm start).
            dt_lag: Predicted planning-compute latency (s), default 0. When
                positive, ``state`` is dead-reckoned forward by this much
                under the control the previous plan is still commanding (see
                :meth:`_predict_forward`) -- what the real system does while
                this call computes -- so the plan lands on the state it will
                apply to rather than the one that was read.

        Returns:
            Updated parameters and this step's diagnostics.
        """
        if dt_lag > 0.0:
            state = self._predict_forward(state, params, dt_lag)

        # Warm start: shift knot times to the current clock, re-evaluate the old
        # spline at the new times (clamped to avoid extrapolation).
        new_tk = (
            np.linspace(0.0, self.plan_horizon, self.num_knots) + state.time
        )
        clamped = np.clip(new_tk, params.tk[0], params.tk[-1])
        new_mean = spline.interp(
            self.spline_type, clamped, params.tk, params.mean[None, ...]
        )[0]
        params = replace(params, tk=new_tk, mean=new_mean)

        # State estimate -> one belief per domain. With no uncertainty model
        # this is the exact state, broadcast as before.
        if self.state_uncertainty is None:
            qpos, qvel = state.qpos, state.qvel
        else:
            qpos, qvel = self.state_uncertainty.sample(state)

        t = time.perf_counter()
        self.engine.set_initial_state(
            qpos=qpos,
            qvel=qvel,
            mocap_pos=state.mocap_pos,
            mocap_quat=state.mocap_quat,
            time=state.time,
        )
        # Resolve interpenetration before rolling out (no-op when
        # settle_steps is 0).
        self.engine.settle()
        stage_t = time.perf_counter() - t

        # Query at the same absolute times the engine applies each rollout
        # step at (new_tk[0] + t*dt for t in 0..ctrl_steps-1), not spread
        # across the full [new_tk[0], new_tk[-1]] span -- those differ
        # whenever ctrl_steps > 1, since plan_horizon / (ctrl_steps - 1) !=
        # dt = plan_horizon / ctrl_steps.
        tq = new_tk[0] + np.arange(self.ctrl_steps) * self.dt
        knots = self.sample_knots(params)  # (S, num_knots, nu)
        knots = np.clip(knots, self.task.u_min, self.task.u_max)
        controls = spline.interp(self.spline_type, tq, new_tk, knots)

        # Host-side device staging (alloc + H2D copies), no GPU sync.
        # Randomizations are uploaded once at build time (and on explicit
        # update_randomizations), not per step.
        t = time.perf_counter()
        self.engine.set_controls(controls)
        stage_t += time.perf_counter() - t

        # GPU rollout: graph replay + sync + copyback.
        t = time.perf_counter()
        result = self.engine.rollout()
        rollout_t = time.perf_counter() - t

        # The initial-state snapshot is recorded *during* the rollout (off its
        # opening forward), so it reads back only now. It still describes the
        # state the rollout started from.
        contact_modes = self.engine.contact_modes()
        belief_pose = None
        if self.state_uncertainty is not None and self.task.endpoint_bodies:
            belief_pose = {
                name: self.engine.current_body_pose(name)
                for name in self.task.endpoint_bodies
            }

        # (R, S) -> (S,): reduce the domain axis via the risk strategy.
        sample_costs = self.risk.combine_costs(result.costs)

        params = self.update_params(params, sample_costs, knots)
        if self.best_sample_execution and self.engine.record_predictions:
            best = int(np.argmin(sample_costs))
            self._predicted_state = self.engine.extract_predicted_state([best])
        info = PlanInfo(
            sample_costs=sample_costs,
            knots=knots,
            trace_sites=result.trace_sites,
            predicted_state=self._predicted_state,
            timings={"stage": stage_t, "rollout": rollout_t},
            contact_modes=contact_modes,
            belief_pose=belief_pose,
            domain_costs=result.costs,
        )
        return params, info

    def _predict_forward(
        self, state: StateSnapshot, params: SamplingParams, dt_lag: float
    ) -> StateSnapshot:
        """Dead-reckon ``state`` forward by ``dt_lag`` under ``params``.

        Plain host MuJoCo stepping on ``self.task.mj_model`` -- one state,
        not the batched Warp rollout. Algorithm-agnostic by necessity: the
        committed mean generally isn't any single already-rolled-out
        sample's trajectory (MPPI's is a weighted average over the whole
        sample set; CEM's is an elite average, coinciding with one rollout
        only when ``num_elite == 1``; only PredictiveSampling's always
        does), so there is usually no GPU trace to read this off instead.
        ``best_sample_execution`` doesn't change that -- it only affects
        which rollout gets recorded as ``predicted_state``, not what
        ``update_params`` returns as the committed mean. Uses the
        planner's own nominal model, not a "true"/mismatched one -- the
        real system has no such oracle either, so a mismatched planner
        mispredicts its own latency too, by design.
        """
        # Reused across calls (this runs every plan step on the sim-to-real
        # latency path) instead of reallocating; mj_resetData restores it to
        # exactly what a fresh MjData() would start from, so a state field
        # left unset below (e.g. no mocap) can't leak a previous call's
        # value.
        if self._predict_data is None:
            self._predict_data = mujoco.MjData(self.task.mj_model)
        mj_data = self._predict_data
        mujoco.mj_resetData(self.task.mj_model, mj_data)
        mj_data.qpos[:] = state.qpos
        mj_data.qvel[:] = state.qvel
        mj_data.time = state.time
        if state.mocap_pos is not None:
            mj_data.mocap_pos[:] = state.mocap_pos
        if state.mocap_quat is not None:
            mj_data.mocap_quat[:] = state.mocap_quat
        mujoco.mj_forward(self.task.mj_model, mj_data)

        for _ in range(max(1, round(dt_lag / self.dt))):
            u = self.get_action(params, mj_data.time)
            mj_data.ctrl[:] = self.task.control_map_host(mj_data, u)
            mujoco.mj_step(self.task.mj_model, mj_data)

        return StateSnapshot(
            qpos=mj_data.qpos.copy(),
            qvel=mj_data.qvel.copy(),
            time=mj_data.time,
            mocap_pos=state.mocap_pos,
            mocap_quat=state.mocap_quat,
        )

    def get_action(self, params: SamplingParams, t: float) -> np.ndarray:
        """Sampling-space control at time ``t`` (real-robot control path)."""
        return spline.query(self.spline_type, t, params.tk, params.mean)

    def set_num_samples(self, num_samples: int) -> None:
        """Change the sampling count in place (an adaptive-budget switch).

        Base implementation just updates the attribute; subclasses that
        derive anything from ``num_samples`` at construction (e.g. CEM's
        elite count) override this to also refresh it. Policy state
        (``params.mean``/``std``) doesn't depend on ``num_samples``, so a
        switch never needs a warm-start reset.
        """
        self.num_samples = int(num_samples)

    @abstractmethod
    def sample_knots(self, params: SamplingParams) -> np.ndarray:
        """Sample ``(num_samples, num_knots, nu)`` knots from the policy."""

    @abstractmethod
    def update_params(
        self,
        params: SamplingParams,
        sample_costs: np.ndarray,
        knots: np.ndarray,
    ) -> SamplingParams:
        """Update the policy from per-sample costs and the sampled knots."""