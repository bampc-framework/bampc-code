"""Closed-loop episodes that produce raw records, not metrics.

Every derived number (accumulated cost, mode entropy, paired deltas) is
analysis' job; an episode only reports what happened, so the metric can change
without re-running the GPU.

The truth and the prediction engines run at their own timesteps, so the
replan interval is a *duration*, not a step count: each side takes however
many of its own steps fill ``1/plan_freq_hz``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bampc.config.scenarios import Scenario
from bampc.planner.base import (
    PlanInfo,
    SamplingParams,
    SamplingPlanner,
    StateSnapshot,
)
from bampc.task.base import Task
from bampc.uncertainty import PoseKalman
from bampc.uncertainty.estimator import quat_mean
from bampc.uncertainty.noise import quat_mul_batch, quat_to_rotvec
from experiments.common.uncertainty.arms import Arm
from experiments.common.uncertainty.goal import (
    GoalDriver,
    drift_phase_time,
    snapshot,
)
from experiments.common.uncertainty.setups import RunConfig
from experiments.common.uncertainty.truth import CpuTruth, WarpTruth

# Four contact bitmasks (pusher/table bits), the same encoding the viewer's
# contact_palette uses.
NUM_MODES = 4


@dataclass
class EpisodeResult:
    """Raw outcome of one episode. Carries no derived metrics."""

    rows: list[dict]
    contact_fraction: float
    success: bool | None


def truth_stride(dt_pred: float, dt_truth: float) -> int:
    """Truth steps per prediction step, as an exact integer.

    The two sides only line up if the truth's clock divides the prediction's.
    """
    k = dt_pred / dt_truth
    if k < 1.0 or abs(k - round(k)) > 1e-9:
        raise ValueError(
            f"prediction dt {dt_pred:g} must be a positive integer multiple "
            f"of truth dt {dt_truth:g} (got ratio {k:g}); the truth has to "
            "be at least as fine as the model it is judging"
        )
    return int(round(k))


def pose_error(
    task: Task, qpos: np.ndarray, truth_qpos: np.ndarray
) -> tuple[float, float]:
    """``(position, rotation)`` error of one state estimate vs the truth.

    Measured on the tracked object only -- that is what the sensor corrupts
    and what the estimators disagree about. Returns ``(0.0, 0.0)`` for a task
    with no registered object.
    """
    layout = task.object_pose_qpos
    if layout is None:
        return 0.0, 0.0
    if layout.is_planar:
        d = np.array([
            qpos[layout.x_adr] - truth_qpos[layout.x_adr],
            qpos[layout.y_adr] - truth_qpos[layout.y_adr],
        ])
        dyaw = float(qpos[layout.yaw_adr] - truth_qpos[layout.yaw_adr])
        return float(np.linalg.norm(d)), abs(dyaw)
    a = layout.adr
    pos = float(np.linalg.norm(qpos[a : a + 3] - truth_qpos[a : a + 3]))
    q, qt = qpos[a + 3 : a + 7], truth_qpos[a + 3 : a + 7]
    inv = qt * np.array([1.0, -1.0, -1.0, -1.0])
    # Hamilton product inv * q, inlined to keep this dependency-free.
    rel = np.array([
        inv[0] * q[0] - inv[1:] @ q[1:],
        *(inv[0] * q[1:] + q[0] * inv[1:] + np.cross(inv[1:], q[1:])),
    ])
    return pos, float(np.linalg.norm(quat_to_rotvec(rel[None])[0]))


def domain_cost_trace(info: PlanInfo) -> np.ndarray | None:
    """Per-domain cost of the WINNING sample, raw -- not a derived statistic.

    The same winning-sample column :mod:`experiments.model_mismatch`'s
    ``disagreement`` reduces to a CV/perplexity; this keeps the raw ``(R,)``
    vector instead, so how the domains disagree is available for analysis to
    decide, not fixed at collection time. ``None`` when there is no domain
    axis to disagree over (``R == 1``, a point arm) or the planner did not
    report per-domain costs.
    """
    dc = info.domain_costs
    if dc is None or dc.shape[0] <= 1:
        return None
    s = int(np.argmin(info.sample_costs))
    return np.asarray(dc[:, s], dtype=float)


def kalman_trace(kf: PoseKalman) -> dict[str, float]:
    """This step's measurement-std diagonal, per channel.

    The same quantity ``scripts/probing/adaptive_kalman_check.py`` validates
    against a synthetic noise burst, recorded here from the live closed
    loop so a filter's (adaptive or fixed) actual trajectory through an
    episode can be checked against where the noise model's own regime
    switches land.
    """
    std = np.sqrt(np.diag(kf.meas_cov))
    labels = ["pos_x", "pos_y", "pos_z", "rot_x", "rot_y", "rot_z"]
    if kf.observe_twist:
        labels += [
            "lin_vel_x", "lin_vel_y", "lin_vel_z",
            "ang_vel_x", "ang_vel_y", "ang_vel_z",
        ]
    return {
        f"meas_std_{name}": float(v)
        for name, v in zip(labels, std, strict=True)
    }


def estimate_state(
    task: Task,
    arm: Arm,
    qpos: np.ndarray,
    qvel: np.ndarray,
    t0: float,
    replan_dt: float,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float]]:
    """Draw one observation and reduce it to this replan's state estimate.

    ``(qpos, qvel)`` unchanged, and ``obs_err = (0.0, 0.0)``, for the oracle
    (no sensor). Otherwise the sensor is sampled once and either passed
    through (an ensemble's own belief fans it, not this) or folded into
    ``arm.filt``.
    """
    if arm.sensor is None:
        return qpos, qvel, (0.0, 0.0)
    obs_qpos, obs_qvel = arm.sensor.sample(
        StateSnapshot(qpos=qpos, qvel=qvel, time=t0)
    )
    obs_qpos, obs_qvel = obs_qpos[0], obs_qvel[0]
    obs_err = pose_error(task, obs_qpos, qpos)
    if arm.filt is None:
        # Ensemble arm: the planner's own StateUncertainty fans this single
        # observation into the belief cloud. Under a pose-only sensor it is
        # NOT None -- it holds a differentiator, which fills in the twist
        # and leaves the pose alone, so the cloud still spreads around the
        # raw reading.
        return obs_qpos, obs_qvel, obs_err
    est_qpos, est_qvel = arm.filt.update(obs_qpos, obs_qvel, replan_dt)
    return est_qpos, est_qvel, obs_err


def plan_step(
    driver: GoalDriver,
    planner: SamplingPlanner,
    arm: Arm,
    est_qpos: np.ndarray,
    est_qvel: np.ndarray,
    t0: float,
    replan_dt: float,
    params: SamplingParams,
    compute_lag: bool,
) -> tuple[SamplingParams, SamplingParams, PlanInfo]:
    """One compute-lag-aware plan step.

    Returns ``(exec_params, params, info)`` -- the params to actually
    execute THIS interval, the params carried into the NEXT call, and the
    diagnostics from whichever ``optimize`` call ran.

    The oracle has no sensor and no compute lag, regardless of
    ``compute_lag``: it plans on the true state now and executes the result
    now, ``exec_params is params``. Every other arm pays ``replan_dt`` of
    planning compute before a plan can take effect, exactly as a real
    system would, UNLESS ``compute_lag`` is ``False`` (an ablation escape
    hatch, ``RunConfig.compute_lag``/``arm_overrides`` -- ``True`` by
    default): ``params`` walking in is last interval's plan, already
    targeted (forward-predicted, back then) for right now, so it becomes
    ``exec_params``; what is computed THIS interval is forward-predicted
    past the compute lag via ``planner.optimize``'s own ``dt_lag`` --
    :meth:`~bampc.planner.base.SamplingPlanner._predict_forward`,
    the SAME mechanism the real robot deployment uses
    (:mod:`bampc.ros.planner_node`): it dead-reckons the estimate
    forward under ``params`` (the plan still in effect) with real
    ``mj_step``s on the planner's own model, not a kinematic guess, so it
    sees whatever the model predicts happens on contact too. Only the goal
    is evaluated at the future time explicitly here, since
    ``_predict_forward`` holds mocap fixed through its internal stepping.
    """
    if arm.sensor is None or not compute_lag:
        params, info = planner.optimize(
            snapshot(driver, est_qpos, est_qvel, t0), params
        )
        return params, params, info
    exec_params = params
    mocap = driver.mocap(t0 + replan_dt, est_qpos, est_qvel)
    mocap_pos, mocap_quat = mocap if mocap is not None else (None, None)
    state = StateSnapshot(
        qpos=est_qpos, qvel=est_qvel, time=t0,
        mocap_pos=mocap_pos, mocap_quat=mocap_quat,
    )
    params, info = planner.optimize(state, params, dt_lag=replan_dt)
    return exec_params, params, info


def warm_up(
    task: Task,
    arm: Arm,
    qpos: np.ndarray,
    qvel: np.ndarray,
    t0: float,
    steps: int,
    dt: float,
) -> None:
    """Calibrate and prime an arm's filter on the still start state.

    The window a real system already has: the camera runs before the
    controller does. Three stages, in order --

    1. **Collect** ``steps`` observations through the arm's own sensor. The
       truth never moves and nothing is planned, so this is host-side numpy.
    2. **Calibrate**: the block is at rest, so the sample variance of those
       readings *is* the measurement variance. The filter is handed that
       instead of the sensor's true magnitudes, which is the privilege this
       removes. Variance about the sample mean ignores a constant offset, so a
       biased and a debiased arm calibrate identically and stay paired.
    3. **Replay** the same readings through the filter, so the estimate and
       its covariance start converged rather than at the uninformative prior.

    A no-op for the oracle (no sensor), for an arm with no Kalman filter, and
    for ``steps <= 0`` -- which leaves the filter on whatever covariance it was
    built with.
    """
    kalman = arm.kalman
    if arm.sensor is None or kalman is None or steps <= 0:
        return

    snapshot = StateSnapshot(qpos=qpos, qvel=qvel, time=t0)
    readings = [arm.sensor.sample(snapshot) for _ in range(steps)]

    a = task.object_pose_qpos.adr
    poses = np.array([q[0] for q, _ in readings])
    # Rotation residuals live in the tangent space about the cloud's own mean,
    # which is what makes this a rotation std rather than a quaternion one.
    quats = poses[:, a + 3 : a + 7]
    n = len(quats)
    inv = quat_mean(quats, np.full(n, 1.0 / n)) * np.array([1.0, -1, -1, -1])
    resid = quat_to_rotvec(quat_mul_batch(np.tile(inv, (n, 1)), quats))
    stds = {
        "pos_std": poses[:, a : a + 3].std(axis=0, ddof=1),
        "rot_std": resid.std(axis=0, ddof=1),
    }
    if kalman.observe_twist:
        # The block is at rest, so a reported twist is pure measurement
        # noise and its spread is the covariance of that channel. Without
        # this the two velocity stds would fall to set_measurement_std's
        # floor and the filter would believe an unmeasured twist absolutely.
        d = kalman.dof_adr
        twists = np.array([v[0, d : d + 6] for _, v in readings])
        stds["lin_vel_std"] = twists[:, :3].std(axis=0, ddof=1)
        stds["ang_vel_std"] = twists[:, 3:].std(axis=0, ddof=1)
    kalman.set_measurement_std(**stds)

    for obs_qpos, obs_qvel in readings:
        arm.filt.update(obs_qpos[0], obs_qvel[0], dt)


def mode_counts(modes: np.ndarray | None, num_domains: int) -> dict[str, int]:
    """Histogram the per-domain contact bitmasks into four columns.

    Raw counts, not an entropy: analysis owns every derived number. A point
    arm has one domain, so its row is one-hot -- which keeps the CSV schema
    identical across arms rather than sprouting nullable columns.
    """
    if modes is None:
        return {f"mode_{m}": 0 for m in range(NUM_MODES)}
    counts = np.bincount(
        np.asarray(modes, int)[:num_domains], minlength=NUM_MODES
    )
    return {f"mode_{m}": int(counts[m]) for m in range(NUM_MODES)}


def run_performance_episode(
    cfg: RunConfig,
    task: Task,
    planner: SamplingPlanner,
    truth: WarpTruth | CpuTruth,
    arm: Arm,
    seed: int,
    repeat: int,
    scenario: Scenario,
) -> EpisodeResult:
    """One closed-loop episode under one arm's estimation stack.

    The loop per replan: read the truth, draw ONE observation through the
    arm's sensor, reduce it (filter, or fan into a belief cloud), plan, then
    execute the chunk open-loop.

    For every arm but the oracle, planning is not instantaneous by default
    (``cfg.compute_lag``/``arm_overrides``, ``True`` unless overridden): a
    real system spends ``replan_dt`` computing a plan, during which the
    PREVIOUS plan keeps running. So what executes this interval is what was
    computed last interval (forward-predicted, back then, for right now);
    what is computed this interval is forward-predicted past the compute
    lag and targets ``t0 + replan_dt`` -- when it will actually take over --
    not ``t0``. The oracle has no sensor and no compute lag either,
    regardless of the flag: it plans on the true state and the plan takes
    effect immediately.

    Args:
        cfg: Resolved run config.
        task: The prediction task (what the planner rolls out).
        planner: Already built on an engine of ``cfg.engine_shape(arm.name)``.
        truth: Ground-truth backend.
        arm: The estimation stack; reset here.
        seed: Scenario index, and half the RNG key.
        repeat: Which identical repeat this is -- the other half of the key,
            so the 8 repeats explore MJWarp's non-determinism rather than
            replaying one trajectory.
        scenario: The frozen start state.

    Returns:
        One row per replan, the fraction of replans carrying contact, and
        the episode's success outcome (``None`` if the task defines none) --
        AND-reduced across the episode, since every task on this axis
        (Balance/BalanceFr3's ball-on-plate) is a condition that must hold
        throughout rather than a one-time achievement.
    """
    dt_pred, dt_truth = task.dt, truth.dt
    k = truth_stride(dt_pred, dt_truth)
    # Every rate below is the ARM's, not the config's shared one: a fast arm
    # replans on its own clock. The filter's dt in particular must match --
    # PoseKalman integrates its constant-velocity prediction over it, so a
    # 20 Hz arm handed 0.1 s would predict twice as far as it should and
    # quietly degrade. It is also FiniteDifference's lever arm: point_fast
    # differences over 0.05 s and so doubles its velocity noise. That is a
    # real cost of replanning faster off a pose-only sensor, not an artifact.
    replan_dt = 1.0 / cfg.plan_freq_for(arm.name)
    steps_pred, num_replans = cfg.replan_counts(dt_pred, arm.name)
    steps_truth, _ = cfg.replan_counts(dt_truth, arm.name)
    if steps_truth != steps_pred * k:
        raise ValueError(
            f"replan chunk does not divide: {steps_truth} truth steps vs "
            f"{steps_pred} prediction steps x stride {k}"
        )

    planner.rng = np.random.default_rng([seed, repeat])
    arm.reset(seed, repeat)
    planner.state_uncertainty = arm.belief
    num_domains = cfg.engine_shape(arm.name)[0]

    truth.reset(cfg.load_start(task, seed))
    driver = GoalDriver(task, drift_phase_time(task, scenario))
    # For any arm but the oracle, step 0's chunk executes THIS unoptimized
    # guess (the compute-lag pipeline has nothing older to fall back on
    # yet) -- a real system's own first control interval, before its first
    # plan has landed, is no different.
    params = planner.init_params()

    # Before the first plan, so the filter enters the episode calibrated and
    # converged rather than on an uninformative prior. Consumes the sensor's
    # RNG stream, which every arm draws from identically -- the arms stay
    # paired, they just start further into the same sequence.
    warm_up(task, arm, *truth.state(), cfg.kalman.warmup_steps, replan_dt)

    rows: list[dict] = []
    replans_with_contact = 0
    success: bool | None = None

    for step in range(num_replans):
        qpos, qvel, t0 = truth.state()

        # --- the arm's view of the world ------------------------------- #
        est_qpos, est_qvel, obs_err = estimate_state(
            task, arm, qpos, qvel, t0, replan_dt
        )
        est_err = pose_error(task, est_qpos, qpos)

        # --- plan on the estimate, but log against the truth ------------ #
        exec_params, params, info = plan_step(
            driver, planner, arm, est_qpos, est_qvel, t0, replan_dt,
            params, cfg.compute_lag_for(arm.name),
        )
        goal = driver.mocap(t0, qpos, qvel)
        modes = mode_counts(planner.engine.contact_modes(), num_domains)

        # --- execute the chunk open-loop -------------------------------- #
        had_contact = False
        for _ in range(steps_truth):
            _, _, t_now = truth.state()
            truth.step(planner.get_action(exec_params, t_now))
            had_contact |= truth.has_contact()
        replans_with_contact += had_contact

        sample = cfg.tracking_sample(task, qpos, qvel, goal, had_contact)
        raw_success = sample.get("task_success")
        if raw_success is not None:
            step_success = bool(raw_success)
            success = (
                step_success if success is None else success and step_success
            )

        row = {
            "seed": seed,
            "repeat": repeat,
            "step": step,
            "time": t0,
            **sample,
            **modes,
            "obs_pos_err": obs_err[0],
            "obs_rot_err": obs_err[1],
            "est_pos_err": est_err[0],
            "est_rot_err": est_err[1],
        }
        # Domain-cost disagreement: PS only for now -- CEM's own elite
        # selection isn't the same argmin-of-risk-combined-cost read, so
        # wiring it in wants its own check first rather than a guessed
        # column. `domain_cost_trace` itself is algo-agnostic; only the
        # write is gated.
        if cfg.planner.algo == "ps":
            dc = domain_cost_trace(info)
            if dc is not None:
                row.update({
                    f"dom_cost_{i}": float(v) for i, v in enumerate(dc)
                })
        # Sigma evolution: every arm with a Kalman filter behind it (point,
        # its variants, and every ensemble -- see `Arm.kalman`), not just
        # `point_adaptive`, so a fixed filter's flat trace is visible right
        # beside an adaptive one's.
        if arm.kalman is not None:
            row.update(kalman_trace(arm.kalman))
        rows.append(row)

    return EpisodeResult(
        rows=rows,
        contact_fraction=replans_with_contact / num_replans,
        success=success,
    )
