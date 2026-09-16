"""Point filters: one state estimate, carried across time.

The *temporal* counterpart to :mod:`~bampc.uncertainty.estimator`. An
estimator collapses a belief cloud at one instant and is stateless; a filter
holds a single estimate and folds each new observation into it, so it needs
``dt`` and a :meth:`PointFilter.reset` between episodes.

Every filter here consumes the raw observation and hands back a state of the
same shape, so a planner behind them is uncertainty-blind:

* :class:`Passthrough` -- believe the latest reading. The naive baseline.
* :class:`FiniteDifference` -- the reading's pose, velocity differenced from
  the last two RAW poses. No model, no covariance.
* :class:`PoseKalman` -- Kalman on the pose, with ``velocity_source``
  ``constant_velocity`` (twist is a propagated state) or
  ``finite_difference`` (twist differenced from the last two SMOOTHED poses).
* :class:`TwistOnly` -- wraps any of the above and keeps **only** its twist,
  so arms can share one velocity estimator without inheriting a pose
  correction they do not want.

**Pose-only sensing.** A camera reports pose, nothing measures velocity, so
with :class:`~bampc.uncertainty.noise.TwistUnobserved` the twist
arrives as zeros and filling it in is the filter's job. Joint velocities are
untouched either way; encoders do measure those.

Which filter fills it in is **not** free when arms are compared: differencing
amplifies pose noise by ``sqrt(2)*pos_std/dt`` while a Kalman filter smooths
it, so arms deriving velocity differently are no longer comparable on
anything else. :class:`TwistOnly` exists to hold that channel constant.

A **biased** sensor is exactly what none of them can fix: a Kalman filter
converges to ``truth + bias``, because the bias is indistinguishable from the
state it is estimating. Removing it takes knowledge of the bias itself, which
is why a debiased arm is a *privileged* baseline rather than a better filter.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Literal

import numpy as np

from bampc.uncertainty.noise import (
    quat_mul_batch,
    quat_to_rotvec,
    rotvec_to_quat,
)

if TYPE_CHECKING:
    from bampc.task.base import ObjectPose


class PointFilter(ABC):
    """Carries one state estimate across an episode."""

    def reset(self) -> None:
        """Forget the estimate. Called once per episode."""

    @abstractmethod
    def update(
        self, qpos: np.ndarray, qvel: np.ndarray, dt: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fold one observation in and return the current estimate."""


class Passthrough(PointFilter):
    """Believe the observation exactly -- no memory, no smoothing."""

    def update(self, qpos, qvel, dt):  # noqa: D102
        return np.array(qpos, dtype=float), np.array(qvel, dtype=float)


class FiniteDifference(PointFilter):
    """Object twist from two consecutive pose readings. Pose untouched.

    The minimal way to get velocity out of a pose-only sensor: no model, no
    smoothing, no covariance. The pose passes through exactly as observed, so
    this is *not* filtering -- it fills slots the sensor left empty, which is
    what lets a belief cloud on top of it stay centred on the raw reading.
    The price: differencing amplifies pose noise by ``sqrt(2) * pos_std / dt``,
    so a shorter replan interval gives a noisier velocity, not a better one.

    Args:
        layout: Where the object's pose sits in ``qpos``.
        dof_adr: Where its twist starts in ``qvel``.

    The first reading of an episode has nothing to difference against and
    yields zero velocity.
    """

    def __init__(self, layout: ObjectPose, dof_adr: int) -> None:
        """Store the layout; the pose history starts empty."""
        if layout.is_planar:
            raise NotImplementedError(
                "FiniteDifference covers free-joint objects; a planar object "
                "would need its x/y/yaw dofs resolved against the model"
            )
        self.layout = layout
        self.dof_adr = int(dof_adr)
        self.reset()

    def reset(self) -> None:
        """Drop the pose history; the next reading differences against zero."""
        self._pos: np.ndarray | None = None
        self._quat = np.array([1.0, 0.0, 0.0, 0.0])

    def update(self, qpos, qvel, dt):  # noqa: D102
        qpos = np.array(qpos, dtype=float)
        qvel = np.array(qvel, dtype=float)
        a, d = self.layout.adr, self.dof_adr
        pos = qpos[a : a + 3]
        quat = qpos[a + 3 : a + 7] / np.linalg.norm(qpos[a + 3 : a + 7])

        vel = np.zeros(6)
        if self._pos is not None:
            # MuJoCo stores a free joint's linear velocity in the WORLD frame
            # and its angular velocity in the BODY frame, and these two
            # expressions match that pairing: a world-frame difference, and a
            # relative rotation read in the previous body frame.
            vel[:3] = (pos - self._pos) / dt
            inv = self._quat * np.array([1.0, -1.0, -1.0, -1.0])
            vel[3:] = quat_to_rotvec(
                quat_mul_batch(inv[None], quat[None])
            )[0] / dt

        self._pos, self._quat = pos.copy(), quat.copy()
        qvel[d : d + 6] = vel
        return qpos, qvel


class TwistOnly(PointFilter):
    """Keep a filter's twist estimate; discard its pose correction.

    Turns any filter into a pure velocity estimator. The pose is returned
    exactly as it came in, so a belief cloud on top of this stays centred on
    the **raw reading** -- the basis of the two-stage sensor/belief split.

    Why it exists: with an unobserved twist every arm must derive velocity
    somehow, and letting them derive it differently turns a
    belief-vs-point-estimate comparison into a differencing-vs-Kalman one.
    Wrapping one shared estimator holds that channel constant.

    Args:
        inner: The filter to take a twist from.
    """

    def __init__(self, inner: PointFilter) -> None:
        """Wrap ``inner``."""
        self.inner = inner

    def reset(self) -> None:
        """Reset the wrapped filter."""
        self.inner.reset()

    def update(self, qpos, qvel, dt):  # noqa: D102
        _, est_vel = self.inner.update(qpos, qvel, dt)
        return np.array(qpos, dtype=float), est_vel


class PoseKalman(PointFilter):
    """Kalman filter on the object's pose, with a choice of velocity model.

    An error-state formulation: the estimate keeps the pose on its manifold
    (position in R^3, orientation as a unit quaternion) while the covariance
    lives on a vector of *errors*. With ``velocity_source="constant_velocity"``
    (default) that vector is 12-dim, ``[dp, dtheta, dv, dw]``: twist is itself
    a filtered, propagated state, and contacts perturb the object in ways the
    filter doesn't model, so acceleration is treated as process noise. With
    ``velocity_source="finite_difference"`` it is 6-dim, ``[dp, dtheta]``:
    there is no twist state to propagate or measure -- pose alone is
    Kalman-filtered (identical measurement update, ``adapt_rate``/
    ``adapt_ceiling``), and the returned twist is differenced from the last
    two SMOOTHED poses instead, the way :class:`FiniteDifference` differences
    raw ones.

    ``observe_twist=True`` reads both pose and twist (``H`` = identity).
    ``observe_twist=False`` reads pose alone (``H = [I_6  0]``); with
    ``velocity_source="constant_velocity"`` the model then *estimates* the
    twist rather than smoothing a reading of it -- the pose-only camera case.
    ``observe_twist=True`` with ``velocity_source="finite_difference"`` is
    rejected: measuring a twist the filter would immediately discard for a
    differenced one is a contradiction.

    Everything outside the object's own joint is passed through untouched:
    the arm is commanded and encoder-observed, and filtering it would blur
    which part of the state the comparison is about.

    Args:
        layout: Where the object's pose sits in ``qpos``.
        dof_adr: Where its twist starts in ``qvel``.
        pos_std: Measurement std on position (m).
        rot_std: Measurement std on orientation (rad).
        lin_vel_std: Measurement std on linear velocity (m/s). **Ignored**
            when ``observe_twist`` is ``False``.
        ang_vel_std: Measurement std on angular velocity (rad/s). Likewise.
        accel_std: Process noise. Under ``constant_velocity``, unmodelled
            linear acceleration (m/s^2) driving a double-integrated pose
            error. Under ``finite_difference`` (no twist state to integrate
            through), reinterpreted as a single-order random-walk rate
            directly on position (m/sqrt(s)).
        ang_accel_std: The ``accel_std`` pair, for orientation.
        observe_twist: Whether the sensor reports the object's velocity.
        velocity_source: ``"constant_velocity"`` (default) or
            ``"finite_difference"`` -- see above.

    Velocity channels are **dropped from** ``H`` rather than left tiny:
    :func:`~bampc.uncertainty.presets.gaussian_stds` floors an absent
    channel at ``1e-6``, and a 12x12 ``H`` would make the filter believe a
    zeroed velocity absolutely.

    The process-noise pair is not dimensionless -- with a twist measurement
    ``accel_std ~ lin_vel_std / dt`` balances prediction against the
    measurement; without one use ``sqrt(2) * pos_std / dt``. Re-check with
    ``scripts/probing/uncertainty_check.py`` after a change.

    ``adapt_rate`` (default ``0``, disabled) turns on innovation-based
    measurement-covariance adaptation so the filter can widen mid-episode --
    see :meth:`update` and ``adapt_ceiling``. After each :meth:`update`,
    ``last_innovation``/``last_innovation_cov`` hold that step's residual and
    innovation covariance, for a consistency check or for sizing a belief
    cloud from live filter uncertainty.
    """

    def __init__(
        self,
        layout: ObjectPose,
        dof_adr: int,
        *,
        pos_std: float,
        rot_std: float,
        lin_vel_std: float = 0.0,
        ang_vel_std: float = 0.0,
        accel_std: float = 0.2,
        ang_accel_std: float = 1.0,
        observe_twist: bool = True,
        adapt_rate: float = 0.0,
        adapt_ceiling: float = 200.0,
        velocity_source: Literal[
            "constant_velocity", "finite_difference"
        ] = "constant_velocity",
    ) -> None:
        """Store the layout and build the fixed noise covariances."""
        if layout.is_planar:
            raise NotImplementedError(
                "PoseKalman covers free-joint objects; a planar object has "
                "no quaternion and wants a plain linear filter"
            )
        if velocity_source not in ("constant_velocity", "finite_difference"):
            raise ValueError(f"unknown velocity_source {velocity_source!r}")
        if velocity_source == "finite_difference" and observe_twist:
            raise ValueError(
                "observe_twist=True measures a twist that "
                "velocity_source='finite_difference' would immediately "
                "discard for a differenced one -- use observe_twist=False"
            )
        self.layout = layout
        self.dof_adr = int(dof_adr)
        self.observe_twist = bool(observe_twist)
        self.velocity_source = velocity_source
        # 12-dim [dp, dtheta, dv, dw] under constant_velocity; 6-dim
        # [dp, dtheta] under finite_difference -- no twist state to hold.
        self._dim = 12 if velocity_source == "constant_velocity" else 6
        # H picks the measured rows out of the error state.
        self._h = (
            np.eye(self._dim) if self.observe_twist
            else np.eye(self._dim)[:6]
        )
        self.set_measurement_std(
            pos_std=pos_std,
            rot_std=rot_std,
            lin_vel_std=lin_vel_std,
            ang_vel_std=ang_vel_std,
        )
        self._accel_var = float(accel_std) ** 2
        self._ang_accel_var = float(ang_accel_std) ** 2
        self._adapt_rate = float(adapt_rate)
        self._adapt_ceiling = float(adapt_ceiling)
        self.last_innovation: np.ndarray | None = None
        self.last_innovation_cov: np.ndarray | None = None
        self.reset()

    def set_measurement_std(
        self,
        *,
        pos_std: float,
        rot_std: float,
        lin_vel_std: float = 0.0,
        ang_vel_std: float = 0.0,
    ) -> None:
        """Rebuild ``meas_cov`` from per-channel stds.

        Exists so a caller can hand the filter a covariance it *measured*
        rather than one it was told. Each std may be a scalar or a 3-vector,
        which is what lets a calibration report a per-axis result.

        Floored at ``1e-6`` on the std (``1e-12`` on the variance) for the
        same reason :func:`~bampc.uncertainty.presets.gaussian_stds`
        floors: a singular measurement covariance is not invertible, and a
        near-zero one
        reads as a perfect reading.
        """
        var = [
            np.broadcast_to(np.asarray(pos_std, dtype=float), (3,)) ** 2,
            np.broadcast_to(np.asarray(rot_std, dtype=float), (3,)) ** 2,
        ]
        if self.observe_twist:
            var += [
                np.broadcast_to(np.asarray(lin_vel_std, dtype=float), (3,))
                ** 2,
                np.broadcast_to(np.asarray(ang_vel_std, dtype=float), (3,))
                ** 2,
            ]
        base_var = np.maximum(np.concatenate(var), 1e-12)
        self.meas_cov = np.diag(base_var)
        # The reference `adapt_rate` inflation is capped against -- see
        # `update`. Re-anchored every time this is called, so a warm-up
        # measurement (or a later re-calibration) resets what "too inflated"
        # means, rather than comparing against whatever value the filter
        # happened to be constructed with.
        self._base_meas_var = base_var

    @property
    def is_adaptive(self) -> bool:
        """Whether ``adapt_rate > 0`` -- this filter's ``meas_cov`` moves."""
        return self._adapt_rate > 0.0

    def posterior_std(self) -> dict[str, np.ndarray]:
        """Per-axis std of the current estimate, in the tangent space.

        The square root of the covariance's diagonal. Under
        ``constant_velocity`` this is all four channels of the
        ``[dp, dtheta, dv, dw]`` error state (``pos``/``rot``/``lin_vel``/
        ``ang_vel``); under ``finite_difference`` there is no twist state, so
        only ``pos``/``rot`` are present. This is what the filter *claims*
        its own uncertainty is, which is the honest thing to build a belief
        cloud from -- see
        :class:`~bampc.uncertainty.noise.PosteriorGaussian`.

        Before the first :meth:`update` it reports the initial covariance,
        which is deliberately uninformative (1.0 in every channel). Feed the
        filter some readings first.
        """
        d = np.sqrt(np.diag(self._cov))
        out = {"pos": d[0:3], "rot": d[3:6]}
        if self._dim == 12:
            out["lin_vel"] = d[6:9]
            out["ang_vel"] = d[9:12]
        return out

    def posterior_cov(self) -> np.ndarray:
        """Full error-state covariance ``[dp, dtheta, dv, dw]``.

        ``12x12`` under ``constant_velocity``, ``6x6`` (``[dp, dtheta]``
        only) under ``finite_difference``. The whole matrix
        ``posterior_std`` takes the diagonal of, so a caller that needs the
        pose block with its correlations (a UKF sigma set) can factor
        ``[0:6, 0:6]`` rather than assume it diagonal. For this filter the
        pose block *is* diagonal, so the two agree there; the accessor
        exists so a downstream sigma set stays correct if the filter ever
        gains cross-axis terms.
        """
        return self._cov.copy()

    def reset(self) -> None:
        """Drop the estimate; the next observation initializes it."""
        self._pos: np.ndarray | None = None
        self._quat = np.array([1.0, 0.0, 0.0, 0.0])
        self._vel = (
            np.zeros(6) if self.velocity_source == "constant_velocity"
            else None
        )
        self._cov = np.eye(self._dim)

    def _process_cov(self, dt: float) -> np.ndarray:
        """Covariance added by one step of unmodelled motion.

        Under ``constant_velocity``, the standard block: an acceleration
        ``a`` held over ``dt`` moves the pose by ``a dt^2/2`` and the
        velocity by ``a dt``. Under ``finite_difference`` there is no
        velocity state to integrate through, so ``accel_std``/
        ``ang_accel_std`` are reinterpreted as a single-order random-walk
        rate directly on pose -- diffusive growth (``~dt``), not the
        double-integrated (``~dt^2``) growth of a held velocity.
        """
        lin, ang = self._accel_var, self._ang_accel_var
        if self.velocity_source == "constant_velocity":
            var = np.concatenate([
                np.full(3, lin * (dt**2 / 2.0) ** 2),
                np.full(3, ang * (dt**2 / 2.0) ** 2),
                np.full(3, lin * dt**2),
                np.full(3, ang * dt**2),
            ])
        else:
            var = np.concatenate([np.full(3, lin * dt), np.full(3, ang * dt)])
        return np.diag(var)

    def _transition(self, dt: float) -> np.ndarray:
        """Error-state transition.

        Under ``constant_velocity``, pose integrates the velocity error.
        Under ``finite_difference`` there is no velocity state to integrate
        through, so this is the identity -- a constant-position prediction.
        """
        f = np.eye(self._dim)
        if self.velocity_source == "constant_velocity":
            f[0:6, 6:12] = np.eye(6) * dt
        return f

    def _adopt(self, z_pos, z_quat, z_vel, cv) -> None:
        """First observation: adopt it rather than filter toward it.

        An unobserved velocity starts at zero and is learned from the next
        pose onward (or, under finite_difference, from the next pose's own
        differencing); the initial covariance carries that ignorance.
        Adopting a measurement means the uncertainty in what was adopted IS
        the measurement's -- so the measured rows take ``meas_cov`` and only
        the unmeasured ones keep :meth:`reset`'s deliberately vague 1.0.
        Without this the filter would claim 1 m of pose uncertainty after a
        reading it just believed outright, which is harmless to its own
        estimate but not to anything that asks it how sure it is (see
        :class:`~bampc.uncertainty.noise.PosteriorGaussian`).
        """
        self._pos, self._quat = z_pos.copy(), z_quat.copy()
        if cv:
            self._vel = z_vel.copy() if self.observe_twist else np.zeros(6)
        n = self._h.shape[0]
        self._cov[:n, :n] = self.meas_cov

    @staticmethod
    def _fd_velocity(prev_pos, prev_quat, pos, quat, dt) -> np.ndarray:
        """Twist from two consecutive (here: SMOOTHED) poses. See
        :class:`FiniteDifference`, whose math this mirrors.
        """  # noqa: D205
        vel = np.zeros(6)
        vel[:3] = (pos - prev_pos) / dt
        inv = prev_quat * np.array([1.0, -1.0, -1.0, -1.0])
        vel[3:] = quat_to_rotvec(quat_mul_batch(inv[None], quat[None]))[0] / dt
        return vel

    def update(self, qpos, qvel, dt):  # noqa: D102
        qpos = np.array(qpos, dtype=float)
        qvel = np.array(qvel, dtype=float)
        a, d = self.layout.adr, self.dof_adr
        z_pos, z_quat = qpos[a : a + 3], qpos[a + 3 : a + 7]
        z_vel = qvel[d : d + 6]
        cv = self.velocity_source == "constant_velocity"

        if self._pos is None:
            self._adopt(z_pos, z_quat, z_vel, cv)
            qvel[d : d + 6] = self._vel if cv else 0.0
            return qpos, qvel

        # Predict. Under constant_velocity, integrate the current velocity,
        # hold it constant. Under finite_difference there is no velocity
        # state to integrate through -- pose is held at its prior posterior
        # (a constant-position prediction), and that prior is stashed so the
        # eventual twist can be differenced against it below.
        if cv:
            self._pos = self._pos + self._vel[:3] * dt
            self._quat = quat_mul_batch(
                self._quat[None], rotvec_to_quat(self._vel[3:] * dt)
            )[0]
            self._quat /= np.linalg.norm(self._quat)
        else:
            prev_pos, prev_quat = self._pos.copy(), self._quat.copy()
        f = self._transition(dt)
        self._cov = f @ self._cov @ f.T + self._process_cov(dt)

        # Innovation. Orientation goes through the log map so the residual is
        # a rotation vector -- the tangent-space quantity the covariance is
        # expressed in.
        q_inv = self._quat * np.array([1.0, -1.0, -1.0, -1.0])
        y = [
            z_pos - self._pos,
            quat_to_rotvec(quat_mul_batch(q_inv[None], z_quat[None]))[0],
        ]
        if cv and self.observe_twist:
            y.append(z_vel - self._vel)
        y = np.concatenate(y)

        h = self._h
        pred_meas_cov = h @ self._cov @ h.T
        s = pred_meas_cov + self.meas_cov
        gain = np.linalg.solve(s.T, (self._cov @ h.T).T).T
        delta = gain @ y
        self._cov = (np.eye(self._dim) - gain @ h) @ self._cov
        self.last_innovation = y
        self.last_innovation_cov = s

        # Inject the correction back onto the manifold.
        self._pos = self._pos + delta[0:3]
        self._quat = quat_mul_batch(
            self._quat[None], rotvec_to_quat(delta[3:6])
        )[0]
        self._quat /= np.linalg.norm(self._quat)

        qpos[a : a + 3] = self._pos
        qpos[a + 3 : a + 7] = self._quat

        if cv:
            self._vel = self._vel + delta[6:12]
            qvel[d : d + 6] = self._vel
        else:
            qvel[d : d + 6] = self._fd_velocity(
                prev_pos, prev_quat, self._pos, self._quat, dt
            )

        if self._adapt_rate > 0.0:
            # Empirical measurement variance this step, per channel: the
            # innovation's own variance (y**2, the diagonal of y y^T) minus
            # the part the PREDICTED state uncertainty already accounts for.
            # What is left is this step's evidence about R itself, blended
            # into the running estimate with an EWMA rather than replacing
            # it outright, since a single step is a noisy estimate of a
            # covariance.
            r_hat = np.maximum(
                y**2 - np.diag(pred_meas_cov), 1e-12
            )
            diag = np.diag(self.meas_cov)
            new_diag = (
                1.0 - self._adapt_rate
            ) * diag + self._adapt_rate * r_hat
            # Capped, not free to grow: an inflated R lowers the gain,
            # weakening the next correction, which tends to produce another
            # large innovation and inflate R again -- unbounded, that loop
            # runs away. The cap keeps the gain bounded away from zero.
            # `_adapt_ceiling` multiplies the REFERENCE variance
            # `set_measurement_std` last established, not `diag` itself, so
            # it cannot ratchet upward with the filter's own history.
            ceiling = self._adapt_ceiling * self._base_meas_var
            self.meas_cov = np.diag(np.minimum(new_diag, ceiling))

        return qpos, qvel
