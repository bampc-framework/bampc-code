"""Config schema + task-behaviour registry for the state-uncertainty sweeps.

Data lives in YAML (parsed by :mod:`experiments.common.uncertainty.loader`
into a :class:`RunConfig`); behaviour that can't serialize -- building the
task, sampling the achieved cost -- lives here, dispatched by
``RunConfig.task``.

Deliberately not shared with the model axis (``experiments/model_mismatch/``):
the two axes measure different things, and a change made for one must not
silently move the other's numbers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np

from bampc.config import scenarios
from bampc.planner import PlannerConfig, engine_shape
from bampc.task.balance import Balance
from bampc.task.balance_fr3 import BalanceFr3
from bampc.task.base import ModelConfig, Task
from bampc.task.push import Push
from bampc.task.push_fr3 import PushFr3


def arm_base_name(name: str) -> str:
    """Strip a risk-/settle-fan suffix (``loader._fan_risk``/``_fan_settle``).

    A fanned name (``ensemble_triple_r32-risk_cvar_0.5``,
    ``point-settle_1``) carries no sizing information of its own -- only
    ``planner.risk``/``risk_alpha``/``settle_steps`` differ from its base
    arm -- so every ``arm_overrides`` lookup (sizing, rate, compute lag)
    resolves through this, the same way :func:`build_arm` already resolves
    estimator kind through it.
    """
    return name.split("-risk_", 1)[0].split("-settle_", 1)[0]


@dataclass(frozen=True)
class SensorConfig:
    """The sensor every arm reads through.

    One draw per replan, from a named preset in
    :mod:`bampc.uncertainty.presets`. As it is the *same* preset at the
    *same* scale in every arm, one ``(seed, repeat)`` names one observation
    sequence shared across arms -- which is what makes them paired.

    ``observe_velocity: false`` is the real camera: it reports the block's
    pose and nothing else, so the twist is erased from the reading and every
    arm has to derive it from the pose sequence. It covers the **tracked
    object only** -- the arm's joint velocities stay observed, because real
    encoders do report them and report them reliably. Leaving it ``true``
    keeps the older, more generous sensor that hands out a noisy but genuine
    velocity.

    ``twist_estimator`` says how the arms that do *not* filter the pose
    (``naive``, the ensembles) recover that twist. It must be the **same**
    estimator the point arms use, or the comparison between them silently
    becomes differencing-vs-Kalman rather than raw-vs-filtered pose:
    differencing amplifies pose noise by ``sqrt(2)*pos_std/dt`` while a
    Kalman filter smooths it. ``finite_difference`` is kept so that effect
    can be measured on purpose instead of suffered by accident.

    ``bias_directions`` turns the sensor's xy bias by a quarter turn per
    episode, cycling every ``n``. ``1`` (the default) is the older behaviour:
    one fixed direction for every episode of every seed, so a direction that
    happens to flatter an arm -- one that nudges the block toward the goal,
    say -- is baked into every number. ``4`` balances it instead.

    It **divides the ``repeats`` axis rather than multiplying episodes**:
    direction is ``repeat % n``, so it is free. The price is that ``repeats``
    exists to sample MJWarp's bimodal basin-hopping, and at ``repeats: 8``
    with 4 directions only 2 draws per (seed, direction) are left for that.
    """

    preset: str = "full"
    scale: float = 1.0
    observe_velocity: bool = True
    twist_estimator: str = "kalman"  # kalman | finite_difference
    bias_directions: int = 1
    # Seconds between random re-orientations of the sensor's xy bias WITHIN an
    # episode. 0 (default) keeps the per-episode `bias_directions` rotation;
    # > 0 replaces it with a bias that jumps to a random one of the four
    # compass directions every `bias_wander_period` seconds -- a
    # miscalibration whose direction drifts over time. The sequence is
    # deterministic per (seed, repeat) and drawn from a private RNG, so the
    # arms stay paired.
    bias_wander_period: float = 0.0
    # Live override of the preset's OU correlation time (seconds), applied to
    # every SE3OrnsteinUhlenbeck term in build_arm -- the sweep-config twin of
    # the examples' `--noise-tau`. `None` (default) leaves the preset's own
    # tau untouched.
    tau: float | None = None

    def __post_init__(self) -> None:
        """Reject an unknown ``twist_estimator`` or a bad direction count.

        Validated here rather than at the call site because the call site
        would have to pick a default, and silently falling back to Kalman on
        a typo like ``finite-difference`` is exactly the misconfiguration
        this experiment cannot afford.
        """
        allowed = ("kalman", "finite_difference")
        if self.twist_estimator not in allowed:
            raise ValueError(
                f"unknown twist_estimator {self.twist_estimator!r}; "
                f"expected one of {list(allowed)}"
            )
        if self.bias_directions < 1:
            raise ValueError(
                f"bias_directions must be >= 1, got {self.bias_directions}"
            )


@dataclass(frozen=True)
class BeliefConfig:
    """The cloud an ensemble arm plans over.

    The two ``sigma`` values multiply the sensor's magnitudes to give the
    *assumed* uncertainty -- the whole point of splitting sensor from belief.
    ``ensemble_exact`` states the noise exactly, ``ensemble_wide`` overstates
    it, which is what an engineer does when they do not trust their
    calibration. The bias terms are always dropped: a belief is the zero-mean
    spread *around* the reading, and a bias baked into it would just move the
    cloud somewhere else.
    """

    num_domains: int = 16
    sigma_scale: float = 1.0
    wide_sigma_scale: float = 2.0
    wider_sigma_scale: float = 3.0
    double_sigma_scale: float = 2.0
    triple_sigma_scale: float = 3.0
    # Two wider rungs of the same ladder, at the double/triple tier's R=32:
    # spread far enough that the cloud should start missing the true state.
    quintuple_sigma_scale: float = 5.0
    octuple_sigma_scale: float = 8.0
    # ensemble_fixed's spread: a fixed, named preset -- decoupled from
    # cfg.sensor.preset on purpose, and kept to a stateless preset on
    # purpose (Arm.reset never reseeds belief.noise; see build_arm).
    fixed_noise: str = "pose"
    fixed_sigma_scale: float = 1.0
    fixed_double_sigma_scale: float = 2.0
    fixed_triple_sigma_scale: float = 3.0
    fixed_twist_sigma_scale: float = 1.0
    estimator: str = "ensemble"

    def sigma_for(self, arm: str) -> float:
        """The sigma multiplier this arm's cloud is drawn at."""
        by_arm = {
            "ensemble_wide": self.wide_sigma_scale,
            "ensemble_wider": self.wider_sigma_scale,
            "ensemble_double": self.double_sigma_scale,
            "ensemble_triple": self.triple_sigma_scale,
            "ensemble_fixed": self.fixed_sigma_scale,
            "ensemble_fixed_naive": self.fixed_sigma_scale,
            "ensemble_fixed_naive_double": self.fixed_double_sigma_scale,
            "ensemble_fixed_naive_triple": self.fixed_triple_sigma_scale,
            "ensemble_double_r4": self.double_sigma_scale,
            "ensemble_double_r8": self.double_sigma_scale,
            "ensemble_double_r16": self.double_sigma_scale,
            "ensemble_double_r32": self.double_sigma_scale,
            "ensemble_triple_r32": self.triple_sigma_scale,
            "ensemble_quintuple_r32": self.quintuple_sigma_scale,
            "ensemble_octuple_r32": self.octuple_sigma_scale,
        }
        return by_arm.get(arm, self.sigma_scale)


@dataclass(frozen=True)
class FilterConfig:
    """The Kalman filter every non-oracle arm derives its estimate with.

    ``accel_std`` / ``ang_accel_std`` are the constant-velocity model's process
    noise -- a free design choice, not information about the world.

    ``warmup_steps`` is how many readings of the (still) start state the filter
    gets before the episode's first plan, and it is what lets the filter
    **measure** its measurement covariance rather than be handed the sensor's
    true one. The block really is at rest there, so the sample variance of the
    readings *is* the measurement variance; an estimate taken mid-episode would
    instead absorb the constant-velocity model's error during contact and come
    out too large. It is also bias-immune -- variance about the sample mean
    does not see a constant offset -- so a debiased arm calibrates to the same
    covariance as its biased twin and the two stay paired.

    Accuracy is set by ``warmup_steps`` alone: the relative error on a variance
    from ``N`` samples is ``sqrt(2 / (N - 1))``, so 5 readings is 71% and 50 is
    about 20% (~10% on the std). It costs no GPU time -- the truth does not
    step and nothing is planned -- so there is little reason to be stingy.

    ``0`` disables warm-up and leaves the filter with the sensor's true
    magnitudes, i.e. the older **privileged** filter. Reachable on purpose,
    never by default.

    ``adapt_rate`` is a *different* knob from ``warmup_steps``: warm-up sets
    ``meas_cov`` once, from a stationary window before the episode starts;
    ``adapt_rate`` (passed to :class:`~bampc.uncertainty.filter.
    PoseKalman`) keeps adjusting it online from each step's innovation for
    the rest of the episode. ``0`` (the default) leaves every arm's filter
    exactly as before; only an arm that asks for it (``point_adaptive``,
    see :mod:`experiments.common.uncertainty.arms`) is built with a nonzero
    value.

    ``adapt_ceiling`` caps the adapted variance at this multiple of the
    reference variance ``set_measurement_std`` last established (warm-up's,
    ordinarily) -- what keeps ``adapt_rate``'s positive-feedback loop (a bad
    step inflates R, which weakens the next correction, which can produce
    another bad step) from running away. Was a hardcoded ``PoseKalman``
    default (200x); exposed here so a sweep can retune it rather than
    inherit whatever the first occlusion pilot happened to need.
    """

    accel_std: float = 0.2
    ang_accel_std: float = 1.0
    adapt_ceiling: float = 200.0
    warmup_steps: int = 0
    adapt_rate: float = 0.0


@dataclass(frozen=True)
class RunConfig:
    """A resolved experiment config."""

    name: str
    task: str  # push | balance | push_fr3
    truth_backend: str  # warp | cpu
    planner: PlannerConfig
    scenario_bank: str
    num_seeds: int
    repeats: int
    rollout_time: float
    plan_freq_hz: float
    # Per-arm overrides, ``{arm: {plan_freq_hz, num_samples, num_domains,
    # settle_steps}}``. The escape hatch for an arm that deliberately leaves
    # the shared budget (`point_fast` replans twice as often on a narrower
    # batch; `point_narrow` takes the ensembles' num_domains) or the shared
    # planner (`oracle` drops settle its perfect state does not need). Adding
    # a control arm is a config line rather than a code change.
    arm_overrides: dict[str, dict[str, Any]]
    task_params: dict[str, Any]
    sensor: SensorConfig
    belief: BeliefConfig
    kalman: FilterConfig
    arms: tuple[str, ...]
    algos: tuple[str, ...]
    prediction_model_config: ModelConfig | None = None
    truth_model_config: ModelConfig | None = None
    # Whether every arm but `oracle` pays one replan interval of planning
    # compute lag, counteracted by forward prediction -- `plan_step` in
    # `episode.py` passing `dt_lag=replan_dt` into `planner.optimize`, the
    # same `SamplingPlanner._predict_forward` dead-reckoning (real `mj_step`
    # under the still-executing plan) the ROS deployment uses, not a
    # standalone approximation. `True` (default) is the realistic setting --
    # a real system can't act on a just-observed state instantaneously.
    # Oracle is never affected by this flag, on either setting: it has no
    # sensor and models the zero-latency upper bound by definition.
    # `arm_overrides` can flip it per arm, same as `plan_freq_hz`, for an
    # ablation that isolates lag from everything
    # else an arm varies.
    compute_lag: bool = True
    # Risk-strategy grid for the arm(s) named in `risk_arms` (e.g.
    # `ensemble_exact`, `ensemble_wide`) -- pairs of `(risk_name, alpha)`,
    # `alpha=None` where the strategy doesn't use one (average, worstcase).
    # `None` (default) is a no-op: every arm runs once, at the planner
    # profile's own `risk`. `run_variants` fans each named arm across the
    # grid, renaming it (e.g. `ensemble_exact-risk_cvar_0.25`) rather than
    # adding a directory level, so an unfanned arm's loader needs no change.
    risk_grid: tuple[tuple[str, float | None], ...] | None = None
    risk_arms: tuple[str, ...] | None = None
    # Settle-steps grid for the arm(s) named in `settle_arms` -- same fan
    # pattern as `risk_grid`/`risk_arms` (renames rather than adding a
    # directory level). `None` (default) is a no-op: every arm runs once, at
    # the planner's own `settle_steps`.
    settle_grid: tuple[int, ...] | None = None
    settle_arms: tuple[str, ...] | None = None
    # Set as the config fans; each variant carries one value and clears the
    # list it came from.
    arm: str | None = None
    # First bank scenario to run. Lets a one-seed tuning run choose *which*
    # scenario instead of always getting 0.
    seed_offset: int = 0

    @property
    def bank(self) -> scenarios.ScenarioBank:
        """The frozen start states, loaded once.

        The bank owns ``shape`` / ``scale`` / ``goal_drift``; the config is
        forbidden from restating them (``loader`` raises), so a retuned drift
        cannot silently apply to some runs and not others.
        """
        return scenarios.load(self.scenario_bank)

    def plan_freq_for(self, arm: str | None) -> float:
        """This arm's replan rate (Hz), defaulting to the shared one."""
        return float(
            self.arm_overrides.get(arm_base_name(arm or ""), {}).get(
                "plan_freq_hz", self.plan_freq_hz
            )
        )

    def compute_lag_for(self, arm: str | None) -> bool:
        """Whether this arm pays compute lag, defaulting to the shared flag."""
        return bool(
            self.arm_overrides.get(arm_base_name(arm or ""), {}).get(
                "compute_lag", self.compute_lag
            )
        )

    def replan_counts(
        self, dt: float, arm: str | None = None
    ) -> tuple[int, int]:
        """``(steps_per_replan, num_replans)`` for one episode at ``dt``.

        The replan interval is a duration, so each role takes its own step
        count from its own timestep; only ``num_replans`` is shared. It is
        also **per arm** -- a fast arm replans on its own clock.
        """
        replan_period = 1.0 / self.plan_freq_for(arm)
        if replan_period > self.planner.plan_horizon:
            raise ValueError(
                f"replan_period {replan_period:g}s (plan_freq_hz="
                f"{1.0 / replan_period:g}, arm={arm!r}) exceeds "
                f"plan_horizon {self.planner.plan_horizon:g}s -- execution "
                "would run past the last optimized rollout and coast on "
                "the spline's clamped extrapolation"
            )
        steps_per_replan = max(int(round(replan_period / dt)), 1)
        num_replans = max(int(round(self.rollout_time / replan_period)), 1)
        return steps_per_replan, num_replans

    def engine_shape(self, arm: str) -> tuple[int, int]:
        """``(num_randomizations, num_samples)`` for one arm.

        Absent an override both shapes multiply out to
        ``planner.sample_budget`` (via ``bampc.planner.engine_shape``,
        which allows any split), so every arm gets identical GPU work and an
        ensemble arm buys belief resolution with control samples -- a trade
        that has to be quoted alongside any result. A high-R arm's S may
        land below the profile's recommended ``num_samples``; that is the
        trade being measured, not an error.

        An arm named in ``arm_overrides`` may leave that budget deliberately.
        ``resolved.json``'s ``conditions`` records the real ``nworld``, so a
        run that does so cannot hide it.

        ``num_domains`` in ``arm_overrides`` puts a non-ensemble arm on the
        ensembles' shape -- what ``point_narrow`` needs to be a control for
        their sample count. Without it the domain count would key off the
        arm's *name*, which would make "is this arm fanned?" a spelling
        question. Looked up by :func:`arm_base_name`, so a risk-/settle-fanned
        arm inherits its base arm's override rather than falling back to the
        un-overridden default.
        """
        override = self.arm_overrides.get(arm_base_name(arm), {})
        r = int(
            override.get(
                "num_domains",
                self.belief.num_domains if arm.startswith("ensemble") else 1,
            )
        )
        if "num_samples" in override:
            return r, int(override["num_samples"])
        return engine_shape(self.planner, r)

    def make_task(self, model_config: ModelConfig | None = None) -> Task:
        """Build this condition's task (dispatched by ``task``)."""
        return _behavior(self.task).make_task(self, model_config)

    def make_prediction_task(self) -> Task:
        """Task carrying the prediction physics (what the planner rolls out)."""
        return self.make_task(self.prediction_model_config)

    def make_truth_task(self) -> Task:
        """Task carrying the truth physics (falls back to prediction)."""
        return self.make_task(
            self.truth_model_config
            if self.truth_model_config is not None
            else self.prediction_model_config
        )

    def load_start(self, task: Task, index: int) -> mujoco.MjData:
        """Host ``MjData`` posed at scenario ``index`` of the bank."""
        bank = self.bank
        return scenarios.pose(bank, task, bank[index])

    def tracking_sample(
        self,
        task: Task,
        qpos: np.ndarray,
        qvel: np.ndarray,
        goal_mocap: tuple[np.ndarray, np.ndarray] | None,
        had_contact: bool,
    ) -> dict[str, Any]:
        """Achieved cost and pose error at the current *true* state.

        Logged once per replan. The cost is the task's own running cost
        against the live (drifting) goal, which is the headline metric --
        pose error is kept beside it because it is free and reads in metres.
        """
        return _behavior(self.task).tracking_sample(
            self, task, qpos, qvel, goal_mocap, had_contact
        )


# --------------------------------------------------------------------- #
# Task-behaviour registry
# --------------------------------------------------------------------- #


def _apply_goal_mocap(
    md: mujoco.MjData, goal_mocap: tuple[np.ndarray, np.ndarray] | None
) -> None:
    """Write drifting-goal mocap arrays into ``md`` before scoring."""
    if goal_mocap is None:
        return
    md.mocap_pos[:] = goal_mocap[0]
    md.mocap_quat[:] = goal_mocap[1]


def _yaw_of(quat: np.ndarray) -> float:
    """Yaw (rotation about +z) of a ``[w, x, y, z]`` quaternion."""
    w, x, y, z = (float(v) for v in quat)
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def _goal_world_pose(
    md: mujoco.MjData,
    task: Task,
    goal_mocap: tuple[np.ndarray, np.ndarray] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """World ``(pos, quat)`` of the goal mocap in ``md`` (drifted if any)."""
    mid = task.goal_mocap_id
    if goal_mocap is not None:
        return goal_mocap[0][mid], goal_mocap[1][mid]
    return md.mocap_pos[mid].copy(), md.mocap_quat[mid].copy()


def _scored_state(
    task: Task,
    qpos: np.ndarray,
    qvel: np.ndarray,
    goal_mocap: tuple[np.ndarray, np.ndarray] | None,
) -> mujoco.MjData:
    """A forwarded scratch ``MjData`` at ``(qpos, qvel)`` with the goal set."""
    md = mujoco.MjData(task.mj_model)
    md.qpos[: len(qpos)] = qpos
    md.qvel[: len(qvel)] = qvel
    _apply_goal_mocap(md, goal_mocap)
    mujoco.mj_forward(task.mj_model, md)
    return md


def _row(
    task: Task,
    scratch: mujoco.MjData,
    goal_mocap: tuple[np.ndarray, np.ndarray] | None,
    pos_err: float,
    orient_err: float | None,
    had_contact: bool,
) -> dict[str, Any]:
    """The columns every task logs, given its own error definitions."""
    bid = task.mj_model.body("block").id
    gpos, gquat = _goal_world_pose(scratch, task, goal_mocap)
    costs = task.cost_components(scratch)
    outcome = task.task_success(scratch)
    return {
        "block_x": float(scratch.xpos[bid][0]),
        "block_y": float(scratch.xpos[bid][1]),
        "block_yaw": _yaw_of(scratch.xquat[bid]),
        "goal_x": float(gpos[0]),
        "goal_y": float(gpos[1]),
        "goal_yaw": _yaw_of(gquat),
        "pos_err": pos_err,
        "orient_err": orient_err,
        **{f"cost_{k}": float(v) for k, v in costs.items()},
        "had_contact": bool(had_contact),
        # None (not int(None)) when the task has no success condition; int
        # rather than bool so a numeric-column reader (np.genfromtxt) reads
        # a missing value as NaN instead of misparsing "True"/"False" text.
        "task_success": None if outcome is None else int(outcome),
    }


class _BalanceBehavior:
    """Balance: tilt a plate to slide a block to a plate-relative goal."""

    @staticmethod
    def make_task(
        cfg: RunConfig, model_config: ModelConfig | None = None
    ) -> Balance:
        """Build the Balance task, geometry and drift from the bank."""
        bank = cfg.bank
        return Balance(
            shape=bank.shape,
            scale=bank.scale,
            goal_xy=bank.goal_xy,
            goal_drift=bank.goal_drift,
            model_config=model_config,
        )

    @staticmethod
    def tracking_sample(
        cfg: RunConfig,
        task: Balance,
        qpos: np.ndarray,
        qvel: np.ndarray,
        goal_mocap: tuple[np.ndarray, np.ndarray] | None,
        had_contact: bool,
    ) -> dict[str, Any]:
        """Block-to-goal distance and achieved cost right now."""
        scratch = _scored_state(task, qpos, qvel, goal_mocap)
        pos_err = float(np.linalg.norm(scratch.sensordata[:3]))
        return _row(task, scratch, goal_mocap, pos_err, None, had_contact)


class _Fr3Behavior:
    """Push-FR3: an EE pushes a block to a goal pose (joint/free block)."""

    @staticmethod
    def make_task(
        cfg: RunConfig, model_config: ModelConfig | None = None
    ) -> PushFr3:
        """Build the PushFr3 task for this condition.

        Cost weights are config-driven because they are not incidental:
        the tuned reward profile retunes several per ``sampling_space``, and
        the library defaults match neither. Only keys the config names
        override the default. Geometry and drift come from the bank.
        """
        tp = cfg.task_params
        bank = cfg.bank
        weights = {
            k: tp[k]
            for k in (
                "w_pos", "w_orient", "w_attract", "w_align", "w_safety",
                "safety_thresh", "w_ee_orient", "w_ee_height",
                "w_arm_home", "terminal_scale",
            )
            if k in tp
        }
        return PushFr3(
            sampling_space=tp.get("sampling_space", "task"),
            manipulation_type=tp["manipulation_type"],
            shape=bank.shape,
            scale=bank.scale,
            goal_xy=bank.goal_xy,
            max_lin_vel=tp.get("max_lin_vel", 0.25),
            goal_drift=bank.goal_drift,
            model_config=model_config,
            **weights,
        )

    @staticmethod
    def tracking_sample(
        cfg: RunConfig,
        task: PushFr3,
        qpos: np.ndarray,
        qvel: np.ndarray,
        goal_mocap: tuple[np.ndarray, np.ndarray] | None,
        had_contact: bool,
    ) -> dict[str, Any]:
        """Block-pose error vs the goal, and achieved cost right now."""
        scratch = _scored_state(task, qpos, qvel, goal_mocap)
        c = task.cost_components(scratch)
        return _row(
            task, scratch, goal_mocap,
            c["pos"] / task.w_pos, c["orient"] / task.w_orient, had_contact,
        )


class _PushBehavior:
    """Push: a planar pusher shoves a block onto a goal pose at the origin."""

    @staticmethod
    def make_task(
        cfg: RunConfig, model_config: ModelConfig | None = None
    ) -> Push:
        """Build the Push task, geometry and drift from the bank."""
        bank = cfg.bank
        return Push(
            shape=bank.shape,
            scale=bank.scale,
            goal_drift=bank.goal_drift,
            model_config=model_config,
        )

    @staticmethod
    def tracking_sample(
        cfg: RunConfig,
        task: Push,
        qpos: np.ndarray,
        qvel: np.ndarray,
        goal_mocap: tuple[np.ndarray, np.ndarray] | None,
        had_contact: bool,
    ) -> dict[str, Any]:
        """Block-to-goal error and achieved cost right now."""
        scratch = _scored_state(task, qpos, qvel, goal_mocap)
        s = scratch.sensordata
        return _row(
            task, scratch, goal_mocap,
            float(np.linalg.norm(s[:2])), abs(_yaw_of(s[3:7])), had_contact,
        )


class _BalanceFr3Behavior:
    """Balance-FR3: an FR3 tilts an EE plate to slide a free ball to a goal."""

    @staticmethod
    def make_task(
        cfg: RunConfig, model_config: ModelConfig | None = None
    ) -> BalanceFr3:
        """Build the BalanceFr3 task for this condition.

        Weights are config-driven (the library defaults differ from the tuned
        joint-space reward profile); only keys the config
        names override. Geometry, goal and drift come from the bank.
        """
        tp = cfg.task_params
        bank = cfg.bank
        weights = {
            k: tp[k]
            for k in (
                "w_pos", "w_orient", "w_ctrl", "w_arm_home",
                "terminal_scale", "max_tilt_vel",
            )
            if k in tp
        }
        return BalanceFr3(
            sampling_space=tp.get("sampling_space", "joint"),
            shape=bank.shape,
            scale=bank.scale,
            goal_xy=bank.goal_xy,
            goal_drift=bank.goal_drift,
            model_config=model_config,
            **weights,
        )

    @staticmethod
    def tracking_sample(
        cfg: RunConfig,
        task: BalanceFr3,
        qpos: np.ndarray,
        qvel: np.ndarray,
        goal_mocap: tuple[np.ndarray, np.ndarray] | None,
        had_contact: bool,
    ) -> dict[str, Any]:
        """Ball-to-goal distance and achieved cost right now (position-only)."""
        scratch = _scored_state(task, qpos, qvel, goal_mocap)
        pos = scratch.sensordata[task._adr_pos : task._adr_pos + 3]
        pos_err = float(np.linalg.norm(pos))
        return _row(task, scratch, goal_mocap, pos_err, None, had_contact)


_BEHAVIORS = {
    "push": _PushBehavior,
    "balance": _BalanceBehavior,
    "push_fr3": _Fr3Behavior,
    "balance_fr3": _BalanceFr3Behavior,
}


def _behavior(task: str):
    """Look up the behaviour class for a task name."""
    try:
        return _BEHAVIORS[task]
    except KeyError:
        raise ValueError(f"unknown task: {task!r}") from None
