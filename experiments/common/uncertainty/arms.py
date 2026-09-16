"""Arms: what each condition knows about the state, and what it does with it.

Every arm sees the world through the same two-stage pipeline --

    truth  --sensor (ONE draw)-->  observation z  -->  estimate  --> planner

-- and differs only in the last two steps. That split is load-bearing. If the
belief were drawn straight from the truth ``R`` times instead, its mean would
sit ``noise/sqrt(R)`` from the truth: ``R`` independent sensor readings that a
point estimate never gets, so the ensemble would win on a gift rather than on
method. See :mod:`bampc.uncertainty`.

Under a pose-only sensor (``sensor.observe_velocity: false``) the reading
carries no object twist at all, so every arm below except ``oracle`` gains a
velocity-estimation step: the same Kalman filter infers it from the pose
sequence, and ``naive`` takes only that velocity while keeping its raw pose.

The arms:

* ``oracle`` -- no sensor at all. Plans on the true state. The upper bound.
* ``naive`` -- believes each reading as it arrives.
* ``point`` -- a Kalman filter over the reading history.
* ``point_adaptive`` -- the same filter, but its measurement covariance is
  not fixed after warm-up: each step folds that step's innovation back in
  online (an EWMA of the innovation-based estimator, ``kalman.adapt_rate``),
  so it can widen mid-episode when the sensor actually gets worse (e.g.
  camera occlusion) instead of carrying a single stationary-start-of-episode
  measurement for the whole rollout. See :class:`~bampc.uncertainty.
  filter.PoseKalman`'s docstring for the mechanism.
* ``point_debiased`` -- the same filter on an unbiased sensor. **Privileged**:
  a Kalman filter converges to ``truth + bias`` and cannot shed a systematic
  offset, so this arm is handed knowledge no estimator could earn.
* ``point_fast`` -- the same filter, replanning twice as often on a narrower
  batch (``arm_overrides`` in the config). Tests whether a shorter replanning
  window pays.
* ``point_narrow`` -- the same filter and the same single estimate, but at the
  *ensembles'* engine shape: ``R`` identical copies, so it pays their control-
  sample count without gaining a belief. The control that separates the two.
* ``ensemble_exact`` / ``ensemble_wide`` -- the same filter again, but the
  planner rolls out ``R`` particles drawn around its estimate at its own
  **posterior** covariance, taken at face value or deliberately overstated.
* ``ensemble_fixed`` -- the same filter's estimate centres the cloud, but the
  planner rolls out ``R`` particles spread by a **fixed, named noise preset**
  (``belief.fixed_noise``, pose and twist independently scaled by
  ``belief.fixed_sigma_scale``/``fixed_twist_sigma_scale``) instead of
  anything the filter claims about its own uncertainty -- for when the
  filter's posterior itself may be miscalibrated (e.g. a biased/lagging
  estimate under a contact-model or sensor-latency issue) but a risk-averse
  planner should still hedge against state error by an operator-trusted
  magnitude.
* ``ensemble_fixed_naive`` / ``_naive_double`` -- ``ensemble_fixed`` with the
  centre swapped to the RAW reading (no filter at all), so hedging is tested
  without depending on a good filter to hedge around. The two differ only in
  how large that fixed magnitude is (``fixed_sigma_scale`` vs
  ``fixed_double_sigma_scale``).
* ``ensemble_adaptive_small`` / ``_medium`` / ``_big`` -- the same filter and
  ``adapt_rate`` as ``point_adaptive``, but the planner rolls out ``R``
  particles drawn around its estimate at that filter's **innovation**
  covariance (``S = HPH^T + R``, :class:`~bampc.uncertainty.
  noise.InnovationGaussian``) rather than its posterior -- it reacts to
  ``adapt_rate``'s inflation one step sooner, since the posterior only
  catches up once a predict/update cycle has propagated it. The three names
  differ only in ``num_domains`` (``arm_overrides`` in the config), holding
  ``nworld`` fixed the same way ``ensemble_exact`` does.
* ``ensemble_collapsed`` -- ``ensemble_exact`` with its contact-mode diversity
  removed: every member is rejection-sampled onto the point estimate's own
  settled contact mode, so it keeps the state spread but not the mode coverage.
  Against ``ensemble_exact`` it isolates whether an ensemble's gain is contact-
  mode coverage or merely state spread (:class:`CollapsedEnsemble`).
* ``ensemble_sigma`` -- ``ensemble_exact`` with a *deterministic* cloud: the
  unscented sigma set of the posterior instead of R random draws, so the
  members reproduce the posterior mean and covariance exactly. Against
  ``ensemble_exact`` it isolates structured-vs-random representation of the
  same belief (:class:`SigmaPointBelief`); needs ``num_domains = 2n+1 = 13``.

So ``point`` and ``ensemble_exact`` share an estimator, a centre and (under
the equal-``nworld`` rule) a GPU budget, but not a sample count.
``point_narrow`` splits that in two: against ``ensemble_exact`` it isolates
the **belief**, against ``point`` it isolates the **sample count**, and the
two add up to the headline contrast.

Nothing in a belief reads a sensor magnitude any more, and with
``kalman.warmup_steps > 0`` the filter measures its own measurement covariance
from a window of readings taken before the episode starts, rather than being
handed the sensor's true one. Both are the same principle: an arm may only use
what it could work out for itself.

An arm's *name* selects its estimator here; its *sizing* (rate, samples) comes
from the config, so a new control arm is a config line plus a name in
:data:`ARM_KINDS`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bampc.task.base import Task
from bampc.uncertainty import (
    CollapsedEnsemble,
    Draw,
    Ensemble,
    FiniteDifference,
    InnovationGaussian,
    Mean,
    Passthrough,
    PointFilter,
    PoseKalman,
    PosteriorGaussian,
    SE3Bias,
    SE3BiasCompassWander,
    SE3OrnsteinUhlenbeck,
    SigmaPointBelief,
    StateUncertainty,
    TwistOnly,
    TwistUnobserved,
    WeightedMean,
    presets,
)
from bampc.uncertainty.noise import (
    _dof_of_qpos,
    rotate_bias_xy,
)
from experiments.common.uncertainty.setups import RunConfig, arm_base_name

# Arm name -> which estimator it uses. Sizing lives in the config, so two
# arms of the same kind differ only in rate/samples (`point` vs `point_fast`).
ARM_KINDS = {
    "oracle": "oracle",
    "naive": "passthrough",
    "point": "kalman",
    "point_adaptive": "kalman_adaptive",
    "point_debiased": "kalman",
    "point_fast": "kalman",
    "point_narrow": "kalman_narrow",
    "ensemble_exact": "ensemble",
    "ensemble_wide": "ensemble",
    "ensemble_wider": "ensemble",
    "ensemble_double": "ensemble",
    "ensemble_triple": "ensemble",
    # Member-count ablation: `ensemble_double` (2x sigma) at a smaller R
    # than `belief.num_domains`'s default -- set via that same name's
    # `arm_overrides.num_domains`, so `engine_shape` gives it a smaller R at
    # a correspondingly larger S, same total nworld. Not a distinct
    # estimator, just `ensemble_double` at a different point on the R/S
    # tradeoff -- named separately only so several R's can coexist as
    # different arms in one sweep/plot.
    "ensemble_double_r4": "ensemble",
    "ensemble_double_r8": "ensemble",
    "ensemble_double_r16": "ensemble",
    # R=32 needs a sample_budget that fits 32 * S -- check a cell's own
    # budget and real-time cap before adding this arm to it.
    "ensemble_double_r32": "ensemble",
    # Same R=32 as ensemble_double_r32, but at 3x filter-posterior sigma
    # instead of 2x (belief.triple_sigma_scale) -- isolates hedge WIDTH from
    # hedge SIZE at a fixed member count.
    "ensemble_triple_r32": "ensemble",
    # Same R=32 again, at 5x sigma -- the next rung of the width ladder.
    "ensemble_quintuple_r32": "ensemble",
    # And at 8x, the widest rung.
    "ensemble_octuple_r32": "ensemble",
    "ensemble_fixed": "ensemble_fixed",
    "ensemble_fixed_naive": "ensemble_fixed_naive",
    "ensemble_fixed_naive_double": "ensemble_fixed_naive",
    "ensemble_fixed_naive_triple": "ensemble_fixed_naive",
    "ensemble_adaptive_small": "ensemble_adaptive",
    "ensemble_adaptive_medium": "ensemble_adaptive",
    "ensemble_adaptive_big": "ensemble_adaptive",
    "ensemble_collapsed": "collapsed",
    "ensemble_sigma": "sigma",
}
ARMS = tuple(ARM_KINDS)

_ESTIMATORS = {
    "ensemble": Ensemble,
    "mean": Mean,
    "draw": Draw,
    "weighted_mean": WeightedMean,
}

# Distinct RNG salts, so reseeding the sensor cannot accidentally hand the
# belief or the planner the same stream.
_SALT_SENSOR = 0x5E_5074
_SALT_BELIEF = 0xBE_11EF


@dataclass
class Arm:
    """One condition's live estimation stack.

    Attributes:
        name: The arm's name.
        sensor: Draws one observation per replan, or ``None`` for the oracle.
        belief: Fans that observation into ``R`` particles, or ``None`` for a
            point arm (the planner then gets a single state broadcast).
        filt: Folds the reading history into one estimate.
        bias0: The sensor's original ``SE3Bias`` entries with their list
            indices, so :meth:`reset` can turn them without compounding.
        bias_directions: How many quarter turns the bias cycles through.

    An ensemble arm carries **both**: the same :class:`PoseKalman` instance is
    its ``filt`` and the source of its ``belief``'s spread, so the cloud is
    centred on the filtered estimate and sized by that filter's posterior.
    Only ``naive`` keeps a raw pose.
    """

    name: str
    sensor: StateUncertainty | None
    belief: StateUncertainty | None
    filt: PointFilter | None
    # (index into sensor.noise, the ORIGINAL entry) for every SE3Bias the
    # sensor carries. reset() rotates from these rather than from whatever is
    # currently in the list, so directions cannot compound across episodes.
    # Empty for the oracle and for any debiased arm, which has no bias to turn.
    bias0: tuple[tuple[int, SE3Bias], ...] = ()
    bias_directions: int = 1

    @property
    def kalman(self) -> PoseKalman | None:
        """This arm's :class:`PoseKalman`, if it has one to calibrate.

        Unwraps :class:`TwistOnly`, so warm-up can configure the filter of an
        arm that only takes a velocity from it.
        """
        filt = self.filt
        if isinstance(filt, TwistOnly):
            filt = filt.inner
        return filt if isinstance(filt, PoseKalman) else None

    def reset(self, seed: int, repeat: int) -> None:
        """Start a fresh episode: turn the bias, reseed, forget the estimate.

        The bias direction is ``repeat % bias_directions``, so it is balanced
        within each seed at no cost in episodes. Turning it does **not**
        disturb the RNG stream -- ``SE3Bias`` draws no random numbers -- so
        every arm at one ``(seed, repeat)`` still sees the identical Gaussian
        realization and the arms stay paired.

        A wandering bias (``SE3BiasCompassWander``) is handled instead by
        reseeding its own private RNG below; ``bias0`` is empty in that case
        (the wander wrapper is not an ``SE3Bias``), so the quarter-turn loop is
        a no-op. Its reseed also draws no numbers from the sensor RNG, so the
        pairing property holds exactly as for the fixed bias.
        """
        for idx, original in self.bias0:
            self.sensor.noise[idx] = rotate_bias_xy(
                original, repeat % self.bias_directions
            )
        if self.sensor is not None:
            for n in self.sensor.noise:
                if hasattr(n, "reseed"):
                    n.reseed(seed, repeat)
            self.sensor.rng = np.random.default_rng(
                [seed, repeat, _SALT_SENSOR]
            )
        if self.belief is not None:
            self.belief.rng = np.random.default_rng(
                [seed, repeat, _SALT_BELIEF]
            )
        if self.filt is not None:
            self.filt.reset()


def _has_arm(task: Task) -> bool:
    """True when the task has FR3 joints for the encoder-noise terms."""
    return any(
        "fr3_joint" in (task.mj_model.joint(j).name or "")
        for j in range(task.mj_model.njnt)
    )


def _apply_tau_override(noise: list, tau: float | None) -> None:
    """Live-override every SE3OrnsteinUhlenbeck term's tau, in place.

    Lets a sweep config A/B the OU correlation time the same way the
    examples' ``--noise-tau`` does; the term is a non-frozen dataclass, so
    this mutates rather than rebuilds the noise list. A no-op when ``tau``
    is ``None`` (the preset's own value stands).
    """
    if tau is None:
        return
    for n in noise:
        if isinstance(n, SE3OrnsteinUhlenbeck):
            n.tau = float(tau)


def build_arm(  # noqa: PLR0911, PLR0912, PLR0915
    cfg: RunConfig, name: str, task: Task
) -> Arm:
    """Build the estimation stack for one arm.

    Args:
        cfg: The resolved run config.
        name: One of :data:`ARMS`.
        task: The *prediction* task -- supplies the model and pose layout the
            noise models and the filter address the state through.

    Returns:
        The arm, already reset.

    Raises:
        ValueError: Unknown arm name.
    """
    # A risk- or settle-fanned name (`ensemble_exact-risk_cvar_0.25`,
    # `point-settle_1`, see loader.py's `_fan_risk`/`_fan_settle`) carries no
    # estimator information of its own -- only `cfg.planner.risk`/
    # `risk_alpha`/`settle_steps` differ -- so it looks itself up by its base
    # name for every purpose below (sizing, debias check, sensor).
    base_name = arm_base_name(name)
    if base_name not in ARM_KINDS:
        raise ValueError(f"unknown arm {name!r}; available: {list(ARMS)}")
    kind = ARM_KINDS[base_name]

    joints = _has_arm(task)
    if kind == "oracle":
        return Arm(name, None, None, None)

    # Debiasing is implemented as building the sensor WITHOUT its bias terms,
    # not as subtracting an offset afterwards. That is exact rather than
    # approximate -- on a free block an inverse rotation applied after the
    # Gaussian does not commute back past it -- and since SE3Bias draws no
    # random numbers, the debiased arm sees the identical Gaussian
    # realization as its biased twin. A perfectly paired comparison.
    debias = base_name.endswith("_debiased")
    # A pose-only sensor stops perturbing the object's twist AND erases it --
    # leaving it alone would hand every arm the true velocity for free.
    observe_vel = cfg.sensor.observe_velocity
    sensor_noise = presets.build(
        cfg.sensor.preset,
        cfg.sensor.scale,
        joints=joints,
        bias=not debias,
        twist=observe_vel,
    )
    if not observe_vel:
        sensor_noise.append(TwistUnobserved())
    _apply_tau_override(sensor_noise, cfg.sensor.tau)
    # A wandering bias replaces each fixed SE3Bias with a wrapper that turns
    # its direction to a random compass point every `bias_wander_period`
    # seconds. A debiased arm has no SE3Bias to wrap, so it is untouched and
    # stays the privileged, paired baseline. Once wrapped, no SE3Bias remains,
    # so `bias0` is empty and the per-episode quarter-turn is skipped.
    period = cfg.sensor.bias_wander_period
    if period > 0.0:
        for i, n in enumerate(sensor_noise):
            if isinstance(n, SE3Bias):
                sensor_noise[i] = SE3BiasCompassWander(n, period)
    sensor = StateUncertainty(task, 1, sensor_noise, seed=0)
    # Snapshot before anything rotates them. A debiased arm built with
    # bias=False has none, so rotation is a no-op there and it stays paired
    # with its biased twin. Empty too when the bias is wandering (wrapped).
    bias0 = tuple(
        (i, n) for i, n in enumerate(sensor_noise) if isinstance(n, SE3Bias)
    )
    dirs = cfg.sensor.bias_directions

    layout = task.object_pose_qpos
    # A planar layout has no free-joint address; both filters below raise on
    # one, so -1 is a marker that never gets used rather than a fallback.
    dof_adr = (
        -1 if layout.is_planar else _dof_of_qpos(task.mj_model, layout.adr)
    )
    # The measurement covariance the filter STARTS with. It is the sensor's
    # true magnitudes, i.e. a filter that was told the answer -- a privilege a
    # real one does not have. `kalman.warmup_steps > 0` overwrites it per
    # episode with a covariance measured from readings of the (still) start
    # state, so this is the value only when warm-up is switched off. That
    # makes the privileged filter reachable on purpose rather than by default.
    #
    # Only the process noise is a free choice either way. With a pose-only
    # sensor the two velocity stds are ignored (H drops those rows) and the
    # filter infers the twist from the pose sequence instead. SE3Bias
    # contributes nothing here, so a debiased arm gets the same stds.
    stds = presets.gaussian_stds(sensor_noise)

    def kalman(*, adapt_rate: float = 0.0) -> PoseKalman:
        """The pose+twist filter. One definition, used by every arm.

        ``adapt_rate`` defaults to ``0.0`` (disabled) for every caller except
        the ``kalman_adaptive`` branch below, so every existing arm's filter
        is unaffected.
        """
        return PoseKalman(
            layout,
            dof_adr,
            accel_std=cfg.kalman.accel_std,
            ang_accel_std=cfg.kalman.ang_accel_std,
            observe_twist=observe_vel,
            adapt_rate=adapt_rate,
            adapt_ceiling=cfg.kalman.adapt_ceiling,
            **stds,
        )

    def twist_estimator() -> PointFilter:
        """Fills in ONLY the twist a pose-only sensor never reported.

        The default wraps the *same* Kalman filter the point arms run, so
        velocity is a controlled constant across every arm and the pose
        passes through untouched. Without that, `naive`/ensemble vs `point`
        would compare differencing against Kalman smoothing rather than a raw
        pose against a filtered one -- and differencing amplifies pose noise
        by sqrt(2)*pos_std/dt, which at a high sensor scale swamps everything
        else. `finite_difference` keeps that older behaviour reachable so the
        effect can be measured deliberately.
        """
        if cfg.sensor.twist_estimator == "finite_difference":
            return FiniteDifference(layout, dof_adr)
        return TwistOnly(kalman())

    if kind == "kalman_narrow":
        # `point`'s estimate at the ENSEMBLE's engine shape: R identical
        # copies, so the planner sees the same state R times and spends the
        # shared nworld budget on S = budget/R control samples instead of the
        # full budget.
        #
        # It exists to split `point` vs `ensemble_exact` into its two halves.
        # An empty noise list is all it takes -- StateUncertainty.sample tiles
        # the state and applies nothing -- so the ONLY difference from
        # `ensemble_exact` is that the cloud has no spread, and the only
        # difference from `point` is the sample count.
        #
        # Averaging cost over R identical domains is very nearly a no-op;
        # what this arm actually changes is the sample count.
        return Arm(
            name,
            sensor,
            StateUncertainty(
                task,
                cfg.engine_shape(name)[0],
                [],
                estimator=Ensemble(),
                seed=0,
                include_truth=False,
            ),
            kalman(),
            bias0,
            dirs,
        )

    if kind == "kalman_adaptive":
        # Same filter as `point`, same warm-up, but `meas_cov` keeps moving
        # afterward: `cfg.kalman.adapt_rate` folds each step's innovation
        # back into it online (see `PoseKalman`'s docstring), so it is not
        # stuck with whatever a stationary window measured at t=0 for the
        # rest of the episode. Isolated to its own arm rather than a global
        # switch, so `point` stays the unchanged baseline.
        return Arm(
            name,
            sensor,
            None,
            kalman(adapt_rate=cfg.kalman.adapt_rate),
            bias0,
            dirs,
        )

    if kind in ("ensemble", "ensemble_adaptive"):
        # ONE filter, shared between the arm and its belief: the cloud is
        # centred on that filter's estimate and spread by that same filter's
        # own uncertainty. So an ensemble arm and `point` now differ in
        # exactly one thing -- whether the spread survives -- where before the
        # ensemble also carried a raw, unfiltered centre.
        #
        # `ensemble` spreads by the filter's POSTERIOR (its own claim about
        # its uncertainty; `sigma_for` takes that at face value (exact) or
        # deliberately overstates it (wide)). `ensemble_adaptive` instead
        # spreads by the filter's INNOVATION covariance (S = HPH^T + R) and
        # shares `point_adaptive`'s `adapt_rate` -- an inflated R shows up
        # here on the very reading that triggered it, where the posterior
        # only catches up once a predict/update cycle has propagated it
        # through P. Nothing here reads a sensor magnitude or the truth
        # either way, which is what makes this a method rather than a
        # privileged baseline.
        adaptive = kind == "ensemble_adaptive"
        k = kalman(adapt_rate=cfg.kalman.adapt_rate if adaptive else 0.0)
        spread_cls = InnovationGaussian if adaptive else PosteriorGaussian
        belief = StateUncertainty(
            task,
            # engine_shape, not cfg.belief.num_domains directly: the latter
            # ignores a per-arm `num_domains` override (arm_overrides), which
            # ensemble_adaptive_small/medium/big rely on for their R -- a
            # mismatch here desyncs the belief's particle count from the
            # engine's actual num_randomizations and crashes set_initial_state.
            cfg.engine_shape(name)[0],
            [spread_cls(k, cfg.belief.sigma_for(base_name))],
            estimator=_ESTIMATORS[cfg.belief.estimator](),
            seed=0,
            # Stays False: it would pin domain 0 to the *estimate*, which is
            # a different gift from the old truth one but still a gift.
            include_truth=False,
        )
        return Arm(name, sensor, belief, k, bias0, dirs)

    if kind == "ensemble_fixed":
        # Same filter as `point`/`ensemble_exact` -- it still centres the
        # cloud on the best available estimate -- but the SPREAD is a fixed,
        # named preset magnitude (belief.fixed_noise, pose and twist scaled
        # independently), completely decoupled from what this filter claims
        # about its own posterior. Exists for exactly the case where the
        # filter's own covariance may not be trustworthy but a risk-averse
        # planner should still hedge against state error by some magnitude
        # the operator chooses and trusts.
        k = kalman()
        fixed_noise = presets.build(
            cfg.belief.fixed_noise,
            cfg.belief.sigma_for(base_name),
            joints=False,
            bias=False,
            twist=True,
            twist_scale=cfg.belief.fixed_twist_sigma_scale,
        )
        # A stateful term (se3_ou) here would carry state
        # across episodes without ever being reseeded -- Arm.reset only
        # reseeds sensor.noise, not belief.noise. Fail loudly rather than
        # silently leak state between episodes.
        if any(hasattr(n, "reseed") for n in fixed_noise):
            raise ValueError(
                f"belief.fixed_noise={cfg.belief.fixed_noise!r} carries a "
                "per-episode-stateful term (se3_ou); "
                "ensemble_fixed requires a stateless preset"
            )
        belief = StateUncertainty(
            task,
            cfg.engine_shape(name)[0],
            fixed_noise,
            estimator=_ESTIMATORS[cfg.belief.estimator](),
            seed=0,
            include_truth=False,
        )
        return Arm(name, sensor, belief, k, bias0, dirs)

    if kind == "ensemble_fixed_naive":
        # `ensemble_fixed` with the centre swapped: the cloud sits on the
        # RAW reading (whatever `naive` itself uses -- `Passthrough` if the
        # sensor observes twist, `twist_estimator()` otherwise), not a
        # Kalman-filtered one. Decouples "does hedging help" from "does
        # hedging need a good filter to work" -- no filter tuning, no
        # warm-up, just a raw reading plus a magnitude the operator chose
        # and trusts. `ensemble_fixed_naive`/`_naive_double` share this
        # branch and differ only in `belief.sigma_for`'s multiplier
        # (`fixed_sigma_scale` vs `fixed_double_sigma_scale`).
        center = Passthrough() if observe_vel else twist_estimator()
        fixed_noise = presets.build(
            cfg.belief.fixed_noise,
            cfg.belief.sigma_for(base_name),
            joints=False,
            bias=False,
            twist=True,
            twist_scale=cfg.belief.fixed_twist_sigma_scale,
        )
        if any(hasattr(n, "reseed") for n in fixed_noise):
            raise ValueError(
                f"belief.fixed_noise={cfg.belief.fixed_noise!r} carries a "
                "per-episode-stateful term (se3_ou); "
                "ensemble_fixed_naive requires a stateless preset"
            )
        belief = StateUncertainty(
            task,
            cfg.engine_shape(name)[0],
            fixed_noise,
            estimator=_ESTIMATORS[cfg.belief.estimator](),
            seed=0,
            include_truth=False,
        )
        return Arm(name, sensor, belief, center, bias0, dirs)

    if kind == "collapsed":
        # `ensemble_exact` with its contact-mode diversity removed: the same
        # filter, the same posterior spread, but every member is rejection-
        # sampled onto the point estimate's own settled contact mode. State
        # spread survives; mode coverage does not. Against `ensemble_exact`
        # that isolates whether the ensemble's gain is contact-mode coverage
        # or just state spread. The engine (its mode oracle) is attached in
        # run.py's Stack, after it is built.
        k = kalman()
        belief = CollapsedEnsemble(
            task,
            cfg.belief.num_domains,
            [PosteriorGaussian(k, cfg.belief.sigma_for(base_name))],
            estimator=_ESTIMATORS[cfg.belief.estimator](),
            seed=0,
        )
        return Arm(name, sensor, belief, k, bias0, dirs)

    if kind == "sigma":
        # `ensemble_exact` with a DETERMINISTIC cloud: instead of drawing R
        # particles from the posterior, place its unscented sigma set, which
        # reproduces the posterior mean and covariance exactly with no
        # Monte-Carlo error. Same filter, same centre, same pose-only spread
        # -- so against `ensemble_exact` it isolates structured-vs-random
        # representation of the belief. num_domains must be 2n+1 = 13 (n=6
        # free-block pose); SigmaPointBelief raises otherwise.
        k = kalman()
        belief = SigmaPointBelief(
            task,
            cfg.belief.num_domains,
            filt=k,
            estimator=_ESTIMATORS[cfg.belief.estimator](),
            seed=0,
        )
        return Arm(name, sensor, belief, k, bias0, dirs)

    if kind == "passthrough":
        # Naive about the POSE, which is the axis under study -- not about
        # the velocity, which no arm observes and all of them must derive.
        return Arm(
            name,
            sensor,
            None,
            Passthrough() if observe_vel else twist_estimator(),
            bias0,
            dirs,
        )

    return Arm(name, sensor, None, kalman(), bias0, dirs)
