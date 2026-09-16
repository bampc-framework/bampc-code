"""Shared CLI wiring for the state-uncertainty examples.

One place for the estimation stack, mirroring the sweeps
(``experiments/state_uncertainty/``)::

    truth --sensor(R=1)--> ONE observation --filter--> estimate
                                                         |
                                              belief cloud around it (R)

The sensor draws **once**; the cloud spreads around the *filter's* estimate
at its own posterior width. Drawing R readings off the truth instead would
put their mean ``noise/sqrt(R)`` from it and flatter the ensemble.

Presets come from :mod:`bampc.uncertainty.presets`, so an example and
a sweep naming ``"pose-biased"`` get the same sensor. The sweeps' ``oracle``/
``naive``/``point_debiased`` baselines are deliberately not here.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable

import numpy as np

from bampc.planner.base import StateSnapshot
from bampc.task.base import Task
from bampc.uncertainty import (
    Ensemble,
    InnovationGaussian,
    PointFilter,
    PoseKalman,
    PosteriorGaussian,
    StateUncertainty,
    TwistOnly,
    TwistUnobserved,
    presets,
)
from bampc.uncertainty.estimator import quat_mean
from bampc.uncertainty.noise import (
    AngularVelocityScale,
    SE3OrnsteinUhlenbeck,
    _dof_of_qpos,
    quat_mul_batch,
    quat_to_rotvec,
)


def add_uncertainty_args(parser: argparse.ArgumentParser) -> None:
    """Add the sensor / filter / belief flags to an example's parser."""
    parser.add_argument(
        "--noise",
        default="pose-biased",
        choices=presets.names(),
        help="Sensor-noise preset (default: pose-biased, the sweeps').",
    )
    parser.add_argument(
        "--noise-scale",
        type=float,
        default=1.0,
        help="Multiplies every noise magnitude. Prefer editing the preset: "
        "a hand-picked level times a second knob stops being legible.",
    )
    parser.add_argument(
        "--noise-tau",
        type=float,
        default=None,
        help="Override the correlation time (s) of any se3_ou term in the "
        "preset, to A/B the noise variants live: 0 = i.i.d. (white), 0.5 = "
        "correlated, larger = longer memory. Default: the preset's own value.",
    )
    parser.add_argument(
        "--estimator",
        default="ensemble",
        choices=["ensemble", "point", "ensemble_fixed"],
        help="THE comparison switch. ensemble = plan over a cloud spread by "
        "the filter's posterior; ensemble_fixed = plan over a cloud spread "
        "by a fixed, named noise preset (--noise/--noise-scale/"
        "--sigma-scale/--twist-sigma-scale), independent of any filter's "
        "own claimed covariance; point = R identical copies of the "
        "estimate, so the planner sees one state (default: ensemble).",
    )
    parser.add_argument(
        "--domains",
        type=int,
        default=16,
        help="R, the number of belief domains. The examples hold S fixed, "
        "so raising this raises the total budget R*S; the sweeps instead "
        "hold R*S constant and trade S away for belief resolution.",
    )
    parser.add_argument(
        "--sigma-scale",
        type=float,
        default=1.0,
        help="Multiplies the belief's std (posterior or innovation, per "
        "--belief-source) when building the cloud. 1.0 takes the filter at "
        "its word; the sweeps' `ensemble_wide` uses 1.5 to ask whether "
        "overstating a calibrated belief hurts.",
    )
    parser.add_argument(
        "--twist-sigma-scale",
        type=float,
        default=1.0,
        help="Under --estimator ensemble_fixed only: multiplies the fixed "
        "preset's velocity (twist) magnitude independently of "
        "--sigma-scale's pose magnitude. Ignored otherwise.",
    )
    parser.add_argument(
        "--belief-source",
        default="posterior",
        choices=["posterior", "innovation"],
        help="What an ensemble's cloud is spread by, with --filter kalman: "
        "posterior = the filter's state covariance (default); innovation = "
        "its last measurement residual's covariance S = HPH^T + R, which "
        "reacts to a degrading sensor (--adapt-rate > 0) one step sooner. "
        "Ignored without a posterior to read: --estimator point or "
        "ensemble_fixed, --filter none, or a planar object.",
    )
    parser.add_argument(
        "--filter",
        default="kalman",
        choices=["kalman", "none"],
        help="kalman = the sweeps' point arms; none = believe each raw "
        "reading (their `naive`). With none, an ensemble falls back to a "
        "cloud at the sensor's stated width -- there is no posterior to "
        "spread. Ignored on a planar object, which has no filter either way.",
    )
    parser.add_argument(
        "--velocity-source",
        default="constant_velocity",
        choices=["constant_velocity", "finite_difference"],
        help="The Kalman filter's velocity model (see PoseKalman). "
        "constant_velocity: twist is itself a filtered, propagated state. "
        "finite_difference: twist is differenced from the filter's last two "
        "smoothed poses instead. Ignored under --filter none or a planar "
        "object.",
    )
    parser.add_argument(
        "--adapt-rate",
        type=float,
        default=0.0,
        help="Online, innovation-based measurement-covariance adaptation "
        "(0 = off, matching PoseKalman's own default). > 0 lets the filter "
        "widen mid-episode when the sensor degrades. Ignored under "
        "--filter none or a planar object, which has no filter.",
    )
    parser.add_argument(
        "--adapt-ceiling",
        type=float,
        default=200.0,
        help="Caps --adapt-rate's inflation at this multiple of the "
        "reference measurement variance (the warm-up's, ordinarily), "
        "preventing a runaway positive-feedback divergence loop. Ignored "
        "at --adapt-rate 0, or without a filter.",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=50,
        help="Readings of the still start state used to MEASURE the filter's "
        "measurement covariance before planning starts. 0 hands it the "
        "sensor's true magnitudes instead -- a privilege a real filter does "
        "not have. Ignored when there is no filter to warm up.",
    )
    parser.add_argument(
        "--observe-velocity",
        action="store_true",
        help="Let the sensor report the object's twist. Off by default, "
        "matching the sweeps: a camera reports pose and nothing measures "
        "velocity, so the filter must infer it from the pose sequence -- or, "
        "on a planar object (no filter), the planner simply never sees it.",
    )
    parser.add_argument(
        "--include-truth",
        action="store_true",
        help="Force domain 0 onto the true state. Debugging only -- it "
        "gives the ensemble a free correct domain and biases any "
        "estimator comparison.",
    )


def build_sensor(
    task: Task, args, joints_ok: bool = True
) -> tuple[StateUncertainty, list]:
    """The sensor: ONE noisy reading of the truth per plan step.

    ``R=1`` is what makes it a sensor rather than a belief -- see the module
    docstring. Returns the sampler and its noise list, which the filter needs
    to read its initial magnitudes from.
    """
    # joints_ok=False for Push, which has no arm: dropping the encoder terms
    # beats failing on a selector that matches nothing.
    noise = presets.build(args.noise, args.noise_scale, joints=joints_ok)
    # Live override of the OU correlation time, so the noise variants can be
    # A/B'd from the CLI without editing the preset (tau is mutable -- the term
    # is a non-frozen dataclass).
    if getattr(args, "noise_tau", None) is not None:
        for n in noise:
            if isinstance(n, SE3OrnsteinUhlenbeck):
                n.tau = float(args.noise_tau)
    if not args.observe_velocity:
        # Stop perturbing the object's twist AND erase it -- leaving it alone
        # would hand the planner the true velocity for free.
        noise.append(TwistUnobserved())
    return StateUncertainty(task, 1, noise, seed=0), noise


def build_filter(
    task: Task, args, noise: list, dt: float
) -> tuple[PointFilter | None, PoseKalman | None]:
    """Build the estimator the observer runs.

    Returns ``(point, kalman)``: what to run on each reading, and the
    underlying :class:`PoseKalman` that warm-up calibrates and a belief
    cloud reads its posterior from -- the *same* object under
    ``--filter kalman``.

    Under ``--filter none`` the raw pose is kept and only the twist is
    filled in, keeping velocity a controlled constant; differencing it
    would amplify pose noise by ``sqrt(2)*pos_std/dt`` and bury the naive
    arm for a reason unrelated to uncertainty.

    ``dt`` is the plan interval. Under ``finite_difference`` the process
    noise means a random-walk RATE, so ``accel_std ~ pos_std/sqrt(dt)``
    fixes the units; ``constant_velocity`` is unaffected.

    ``(None, None)`` for a **planar** object -- :class:`PoseKalman` covers
    free-joint objects only, so Push falls back to a cloud around the raw
    reading (see :func:`build_belief`).
    """
    layout = task.object_pose_qpos
    if layout.is_planar:
        return None, None
    stds = presets.gaussian_stds(noise)
    fd = args.velocity_source == "finite_difference"
    accel_std = stds["pos_std"] / dt**0.5 if fd else 0.2
    ang_accel_std = stds["rot_std"] / dt**0.5 if fd else 1.0

    def make() -> PoseKalman:
        return PoseKalman(
            layout,
            _dof_of_qpos(task.mj_model, layout.adr),
            accel_std=accel_std,
            ang_accel_std=ang_accel_std,
            observe_twist=args.observe_velocity,
            adapt_rate=args.adapt_rate,
            adapt_ceiling=args.adapt_ceiling,
            velocity_source=args.velocity_source,
            **stds,
        )

    if args.filter == "kalman":
        k = make()
        return k, k
    if args.observe_velocity:
        return None, None  # nothing to fill in: believe the whole reading
    k = make()
    return TwistOnly(k), k


def build_belief(
    task: Task, args, filt: PoseKalman | None, joints_ok: bool = True
) -> tuple[StateUncertainty, str]:
    """The cloud the planner plans over, centred on whatever it is handed.

    Returns the sampler and a one-line description of where its spread came
    from -- print it, the three sources are not otherwise distinguishable
    from the outside.

    ``point`` uses an empty noise list, so the planner effectively sees one
    state. The exception is an
    :class:`~bampc.uncertainty.noise.AngularVelocityScale`
    term, a deterministic per-tick correction rather than a spread, applied
    regardless of ``--estimator``.

    ``ensemble`` spreads it by the filter's posterior or innovation
    covariance (``--belief-source``) -- no sensor magnitude and no truth
    enters, which is what makes it a method rather than a privileged
    baseline. With no filter it falls back to the preset's own zero-mean
    terms, a weaker claim: that width is stated, not earned.

    ``ensemble_fixed`` spreads it by a fixed named preset regardless of
    ``--filter``, for when a filter's own claimed covariance is not
    trustworthy.
    """
    if args.estimator == "point":
        noise = [
            n for n in presets.build(
                args.noise, args.noise_scale, joints=joints_ok, bias=False,
            )
            if isinstance(n, AngularVelocityScale)
        ]
        how = f"point estimate, tiled x{args.domains}"
        if noise:
            how += f", angular-velocity hedge x{noise[0].factor:g}"
    elif args.estimator == "ensemble_fixed":
        noise = presets.build(
            args.noise,
            args.noise_scale * args.sigma_scale,
            joints=joints_ok,
            bias=False,
            twist=True,
            twist_scale=args.noise_scale * args.twist_sigma_scale,
        )
        how = (
            f"cloud at fixed preset {args.noise!r}, pose "
            f"x{args.sigma_scale:g} / twist x{args.twist_sigma_scale:g} "
            "(filter-independent)"
        )
    elif filt is not None and args.filter == "kalman":
        if args.belief_source == "innovation":
            noise = [InnovationGaussian(filt, args.sigma_scale)]
            how = f"cloud from the filter innovation cov x{args.sigma_scale:g}"
        else:
            noise = [PosteriorGaussian(filt, args.sigma_scale)]
            how = f"cloud from the filter posterior x{args.sigma_scale:g}"
    else:
        # Bias and twist terms are dropped: a belief is the zero-mean spread
        # *around* the reading, not a second offset, and nothing measured the
        # twist to spread it.
        noise = presets.build(
            args.noise,
            args.noise_scale * args.sigma_scale,
            joints=joints_ok,
            bias=False,
            twist=False,
        )
        how = f"cloud at the SENSOR width x{args.sigma_scale:g} (no filter)"
    return StateUncertainty(
        task,
        args.domains,
        noise,
        estimator=Ensemble(),
        seed=0,
        include_truth=args.include_truth,
    ), how


def make_observer(
    sensor: StateUncertainty, point: PointFilter | None, dt: float
) -> Callable[[StateSnapshot], StateSnapshot]:
    """Wrap sensor and filter into the viewer's ``observer`` hook.

    Called once per plan step with the truth; returns what the planner is
    told. The sensor draws exactly ONE reading here, so everything
    downstream is a belief about that reading, not a second look at truth.
    """

    def observe(state: StateSnapshot) -> StateSnapshot:
        qpos, qvel = sensor.sample(state)
        qpos, qvel = qpos[0], qvel[0]
        if point is not None:
            qpos, qvel = point.update(qpos, qvel, dt)
        return StateSnapshot(
            qpos=qpos,
            qvel=qvel,
            time=state.time,
            mocap_pos=state.mocap_pos,
            mocap_quat=state.mocap_quat,
        )

    return observe


def warm_up(
    task: Task,
    sensor: StateUncertainty,
    point: PointFilter | None,
    filt: PoseKalman | None,
    mj_data,
    steps: int,
    dt: float,
) -> None:
    """Measure the filter's measurement covariance on the still start state.

    The window a real system already has: the camera runs before the
    controller does. Collect ``steps`` readings, take their sample variance
    about their own mean as the measurement variance (valid because the object
    is at rest, and blind to a constant bias), then replay them so the
    estimate starts converged rather than at the uninformative prior.

    A no-op with no filter or ``steps <= 0``, which leaves the filter on the
    sensor's true magnitudes -- the privileged case, reachable on purpose.

    Mirrors ``experiments/common/uncertainty/episode.py``'s ``warm_up``; that
    one works off an ``Arm``, so the two are separate by a few lines. Change
    both.
    """
    if filt is None or steps <= 0:
        return
    snapshot = StateSnapshot(
        qpos=mj_data.qpos.copy(),
        qvel=mj_data.qvel.copy(),
        time=float(mj_data.time),
    )
    readings = [sensor.sample(snapshot) for _ in range(steps)]

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
    if filt.observe_twist:
        d = filt.dof_adr
        twists = np.array([v[0, d : d + 6] for _, v in readings])
        stds["lin_vel_std"] = twists[:, :3].std(axis=0, ddof=1)
        stds["ang_vel_std"] = twists[:, 3:].std(axis=0, ddof=1)
    filt.set_measurement_std(**stds)

    # Replay through whatever the observer will run, so a TwistOnly wrapper
    # sees the same history its inner filter just got calibrated on.
    for obs_qpos, obs_qvel in readings:
        point.update(obs_qpos[0], obs_qvel[0], dt)
