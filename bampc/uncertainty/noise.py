"""Sensor-noise models: what the state estimate gets wrong.

Composable frozen dataclasses, each declaring *what* it perturbs. None of
them knows about Warp, the engine, or the planner -- they take a nominal
``(qpos, qvel)`` and return ``R`` perturbed copies.

Pose noise uses the **SO(3) x R^3 product metric**: position and rotation are
drawn independently rather than as a coupled se(3) twist. Rotation is sampled
in the tangent space as a rotation vector and mapped onto the pose by the
exponential map, so the result is a valid rotation for any magnitude.

They compose in a list, so a shifted-mean Gaussian is just::

    [SE3Bias(pos=(0.005, 0.0, 0.0)), SE3Gaussian(pos_std=0.003)]
"""

from __future__ import annotations

import dataclasses
import fnmatch
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import mujoco
import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from bampc.task.base import ObjectPose

    # filter.py imports the quaternion helpers from here, so this direction
    # has to stay type-only or the two modules import each other.
    from bampc.uncertainty.filter import PoseKalman

# Below this rotation-vector magnitude, sin(t/2)/t is replaced by its Taylor
# limit 1/2 to avoid 0/0.
_SMALL_ANGLE = 1e-9


def _triple(
    value: float | Sequence[float], planar: bool, *, rot: bool, what: str
) -> np.ndarray:
    """Resolve a scalar-or-3-vector magnitude against the pose layout.

    A **scalar** is shorthand for "this much, wherever the object can move":
    on a planar object it means in-plane xy (or yaw, for rotation), never
    out-of-plane. An **explicit 3-vector** is taken literally, so a non-zero
    out-of-plane component on a planar object raises -- silently dropping it
    would read as a tuning result rather than a mistake.
    """
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        s = float(arr)
        if not planar:
            return np.full(3, s)
        return np.array([0.0, 0.0, s]) if rot else np.array([s, s, 0.0])
    if arr.shape != (3,):
        raise ValueError(
            f"{what} must be a scalar or 3 values, got {arr.shape}"
        )
    if planar:
        out_of_plane = arr[:2] if rot else arr[2:]
        if np.any(out_of_plane != 0.0):
            axes = "roll/pitch" if rot else "z"
            raise ValueError(
                f"planar object pose has no {axes} freedom, but {what}="
                f"{tuple(arr)} asks for it; set those components to zero"
            )
    return arr


def rotvec_to_quat(w: np.ndarray) -> np.ndarray:
    """Exponential map: rotation vectors ``(n, 3)`` -> quats ``(n, 4)``.

    Quaternions are MuJoCo's ``[w, x, y, z]``. Uses the Taylor limit near
    zero so tiny rotations stay finite.
    """
    w = np.atleast_2d(np.asarray(w, dtype=float))
    theta = np.linalg.norm(w, axis=1)
    half = 0.5 * theta
    # sin(theta/2)/theta, guarded at theta -> 0 (limit 1/2).
    scale = np.where(theta < _SMALL_ANGLE, 0.5, np.sin(half) / np.where(
        theta < _SMALL_ANGLE, 1.0, theta
    ))
    out = np.empty((w.shape[0], 4))
    out[:, 0] = np.cos(half)
    out[:, 1:] = w * scale[:, None]
    return out


def quat_to_rotvec(q: np.ndarray) -> np.ndarray:
    """Log map: quats ``(n, 4)`` -> rotation vectors ``(n, 3)``.

    Inverse of :func:`rotvec_to_quat`. Uses ``atan2(|v|, w)`` rather than
    ``arccos(w)`` because arccos is vertical at 1, so a near-identity
    rotation would read as a spurious ~1e-8 floor. ``q`` and ``-q`` are the
    same rotation; the negative-``w`` hemisphere is flipped first so the
    result is the short way round.
    """
    q = np.atleast_2d(np.asarray(q, dtype=float))
    q = q * np.where(q[:, :1] < 0.0, -1.0, 1.0)
    v = q[:, 1:]
    vn = np.linalg.norm(v, axis=1)
    theta = 2.0 * np.arctan2(vn, q[:, 0])
    # theta/sin(theta/2) == 2 at theta -> 0; vn is sin(theta/2).
    scale = np.where(vn < _SMALL_ANGLE, 2.0, theta / np.where(
        vn < _SMALL_ANGLE, 1.0, vn
    ))
    return v * scale[:, None]


def quat_mul_batch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product ``a * b`` for batches of ``[w, x, y, z]`` quats."""
    aw, ax, ay, az = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bw, bx, by, bz = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    return np.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=1,
    )


class StateNoise(ABC):
    """One perturbation applied to a batch of belief states."""

    @abstractmethod
    def apply(
        self,
        qpos: np.ndarray,
        qvel: np.ndarray,
        rng: np.random.Generator,
        model: mujoco.MjModel,
        layout: ObjectPose | None,
    ) -> None:
        """Perturb ``(R, nq)`` / ``(R, nv)`` arrays **in place**."""


def _apply_pose(
    qpos: np.ndarray,
    layout: ObjectPose | None,
    dpos: np.ndarray,
    drot: np.ndarray,
) -> None:
    """Add a world-frame pose delta to every row of ``qpos``, in place.

    ``dpos`` is ``(R, 3)`` metres and ``drot`` ``(R, 3)`` rotation vectors,
    already validated against the layout by :func:`_triple`. Rotation is
    applied as a **left** (world-frame) multiply, which is what makes the
    planar case consistent: a world yaw delta adds straight onto ``block_yaw``.
    """
    if layout is None:
        raise ValueError(
            "pose noise needs the task to define `object_pose_qpos`; this "
            "task has no manipulated object registered"
        )
    if layout.is_planar:
        qpos[:, layout.x_adr] += dpos[:, 0]
        qpos[:, layout.y_adr] += dpos[:, 1]
        qpos[:, layout.yaw_adr] += drot[:, 2]
        return

    a = layout.adr
    qpos[:, a : a + 3] += dpos
    q = quat_mul_batch(rotvec_to_quat(drot), qpos[:, a + 3 : a + 7])
    norm = np.linalg.norm(q, axis=1, keepdims=True)
    qpos[:, a + 3 : a + 7] = q / np.where(norm > 0.0, norm, 1.0)


@dataclass(frozen=True)
class SE3Gaussian(StateNoise):
    """Zero-mean pose noise, sampled in the tangent space.

    Args:
        pos_std: Position noise std (m). A scalar means "in every direction
            the object can translate" -- xy only on a planar object.
        rot_std: Rotation noise std (rad), as a rotation vector. A scalar
            means yaw only on a planar object, isotropic on a free one.
    """

    pos_std: float | Sequence[float] = 0.0
    rot_std: float | Sequence[float] = 0.0

    def apply(self, qpos, qvel, rng, model, layout) -> None:  # noqa: D102
        n = qpos.shape[0]
        planar = layout is not None and layout.is_planar
        sp = _triple(self.pos_std, planar, rot=False, what="pos_std")
        sr = _triple(self.rot_std, planar, rot=True, what="rot_std")
        dpos = rng.normal(0.0, 1.0, (n, 3)) * sp
        drot = rng.normal(0.0, 1.0, (n, 3)) * sr
        _apply_pose(qpos, layout, dpos, drot)


class PosteriorGaussian(StateNoise):
    """Pose spread taken from a filter's own posterior covariance.

    The belief a filter *claims* to have, turned into a cloud. Its magnitude
    is not fixed: it reads the filter's covariance on every call, so the cloud
    tightens as the estimate does (also why it is not a frozen dataclass).
    Nothing but the filter's own state enters it -- no sensor magnitude, no
    truth -- which is what makes an ensemble built on it a *method* rather
    than a privileged baseline.

    Args:
        filt: The filter to read. Feed it some readings first: before its
            first update its covariance is deliberately uninformative.
        scale: Multiplies the posterior std. ``1.0`` takes the filter at its
            word; above that deliberately overstates it.

    Spreads the **pose only**: under an unobserved twist the velocity block
    carries genuine ignorance, not a measured spread, and fanning across it
    would hedge over a number nothing measured. Diagonal -- exact for the pose
    block of a constant-velocity filter with per-axis block-diagonal ``Q``,
    ``R``, ``F``; it drops only the position-velocity correlation.
    """

    def __init__(self, filt: PoseKalman, scale: float = 1.0) -> None:
        """Hold a live reference to ``filt``."""
        self.filt = filt
        self.scale = float(scale)

    def apply(self, qpos, qvel, rng, model, layout) -> None:  # noqa: D102
        n = qpos.shape[0]
        planar = layout is not None and layout.is_planar
        std = self.filt.posterior_std()
        sp = _triple(
            std["pos"] * self.scale, planar, rot=False, what="pos_std"
        )
        sr = _triple(
            std["rot"] * self.scale, planar, rot=True, what="rot_std"
        )
        dpos = rng.normal(0.0, 1.0, (n, 3)) * sp
        drot = rng.normal(0.0, 1.0, (n, 3)) * sr
        _apply_pose(qpos, layout, dpos, drot)


class InnovationGaussian(StateNoise):
    """Pose spread taken from a filter's own innovation covariance.

    Where :class:`PosteriorGaussian` reads the covariance the filter *holds*,
    this reads ``S = H P H^T + R``, its most recent residual covariance. Under
    ``adapt_rate > 0`` an inflated ``R`` shows up here on the very reading
    that triggered it, while ``posterior_std`` only catches up after a
    predict/update cycle -- so a cloud spread by this reacts to a degrading
    sensor sooner.

    Args:
        filt: The filter to read. ``last_innovation_cov`` is ``None`` before
            its first :meth:`~PoseKalman.update` call -- feed it readings
            (warm-up, ordinarily) before sampling from this.
        scale: Multiplies the innovation std (std, not covariance), same
            convention as :class:`PosteriorGaussian`.

    Spreads the pose only -- see :class:`PosteriorGaussian`.
    """

    def __init__(self, filt: PoseKalman, scale: float = 1.0) -> None:
        """Hold a live reference to ``filt``."""
        self.filt = filt
        self.scale = float(scale)

    def apply(self, qpos, qvel, rng, model, layout) -> None:  # noqa: D102
        cov = self.filt.last_innovation_cov
        if cov is None:
            raise ValueError(
                "InnovationGaussian needs at least one filt.update() call "
                "before its first apply() -- warm_up() provides this."
            )
        n = qpos.shape[0]
        planar = layout is not None and layout.is_planar
        std = np.sqrt(np.diag(cov))
        sp = _triple(
            std[0:3] * self.scale, planar, rot=False, what="pos_std"
        )
        sr = _triple(
            std[3:6] * self.scale, planar, rot=True, what="rot_std"
        )
        dpos = rng.normal(0.0, 1.0, (n, 3)) * sp
        drot = rng.normal(0.0, 1.0, (n, 3)) * sr
        _apply_pose(qpos, layout, dpos, drot)


# Cache for `_dof_of_qpos`, keyed by (id(model), qadr) -- the joint scan is
# re-run once per model instead of every `apply()` tick.
_dof_cache: dict[tuple[int, int], int] = {}


def _dof_of_qpos(model: mujoco.MjModel, qadr: int) -> int:
    """First dof address of the joint whose qpos starts at ``qadr``.

    ``ObjectPose`` records qpos addresses only, and ``qpos`` and ``qvel``
    disagree in size for a free joint (7 vs 6), so the two cannot be indexed
    with one number. Looking the joint up here keeps the layout unchanged.
    """
    key = (id(model), qadr)
    cached = _dof_cache.get(key)
    if cached is not None:
        return cached
    for jid in range(model.njnt):
        if int(model.jnt_qposadr[jid]) == qadr:
            _dof_cache[key] = int(model.jnt_dofadr[jid])
            return _dof_cache[key]
    raise ValueError(f"no joint starts at qpos address {qadr}")


def _apply_twist(
    qvel: np.ndarray,
    model: mujoco.MjModel,
    layout: ObjectPose | None,
    dlin: np.ndarray,
    dang: np.ndarray,
) -> None:
    """Add a twist delta to every row of ``qvel``, in place.

    MuJoCo stores a free joint's linear velocity in the world frame and its
    angular velocity in the body frame. Both deltas here are isotropic
    Gaussians, which are frame-invariant, so no rotation is applied.
    """
    if layout is None:
        raise ValueError(
            "twist noise needs the task to define `object_pose_qpos`; this "
            "task has no manipulated object registered"
        )
    if layout.is_planar:
        qvel[:, _dof_of_qpos(model, layout.x_adr)] += dlin[:, 0]
        qvel[:, _dof_of_qpos(model, layout.y_adr)] += dlin[:, 1]
        qvel[:, _dof_of_qpos(model, layout.yaw_adr)] += dang[:, 2]
        return
    a = _dof_of_qpos(model, layout.adr)
    qvel[:, a : a + 3] += dlin
    qvel[:, a + 3 : a + 6] += dang


@dataclass(frozen=True)
class TwistGaussian(StateNoise):
    """Zero-mean noise on the object's 6-D velocity.

    The velocity counterpart to :class:`SE3Gaussian`. A camera reports pose;
    velocity is differentiated from it and is correspondingly noisy. Without
    this the object's velocity is observed *exactly*, which quietly makes any
    velocity-estimating filter pointless.

    Args:
        lin_std: Linear velocity noise std (m/s). A scalar means "in every
            direction the object can translate" -- xy only on a planar object.
        ang_std: Angular velocity noise std (rad/s). A scalar means yaw only
            on a planar object, isotropic on a free one.
    """

    lin_std: float | Sequence[float] = 0.0
    ang_std: float | Sequence[float] = 0.0

    def apply(self, qpos, qvel, rng, model, layout) -> None:  # noqa: D102
        n = qvel.shape[0]
        planar = layout is not None and layout.is_planar
        sl = _triple(self.lin_std, planar, rot=False, what="lin_std")
        sa = _triple(self.ang_std, planar, rot=True, what="ang_std")
        dlin = rng.normal(0.0, 1.0, (n, 3)) * sl
        dang = rng.normal(0.0, 1.0, (n, 3)) * sa
        _apply_twist(qvel, model, layout, dlin, dang)


def _scale_twist(
    qvel: np.ndarray,
    model: mujoco.MjModel,
    layout: ObjectPose | None,
    scale: np.ndarray,
) -> None:
    """Multiply every row's angular (and, if free, linear) velocity.

    A no-slip ``omega = (n_hat x v) / r`` rotated into the body frame is
    linear in ``v``, so scaling ``v`` before the conversion is **exactly**
    equivalent to scaling ``omega`` after it -- the same per-row ``scale``
    can be applied to the linear slot in place, with no need to re-derive
    omega from the plate normal and ball radius. That hedges the rolling
    motion consistently: a domain assuming the ball rolls 1.5x faster also
    assumes it translates 1.5x faster.

    Angular-only on a *planar* object: x/y are already free there rather
    than implied by a rolling constraint, so scaling them together would
    have no physical grounding.
    """
    if layout is None:
        raise ValueError(
            "AngularVelocityScale needs the task to define "
            "`object_pose_qpos`; this task has no manipulated object "
            "registered"
        )
    if layout.is_planar:
        qvel[:, _dof_of_qpos(model, layout.yaw_adr)] *= scale
        return
    a = _dof_of_qpos(model, layout.adr)
    qvel[:, a : a + 6] *= scale[:, None]


@dataclass(frozen=True)
class AngularVelocityScale(StateNoise):
    """A fixed multiplicative hedge on the object's rolling-motion magnitude.

    Under a rolling-without-slip inference (see
    ``ros.adapters.balance_fr3._rolling_angular_velocity``) the real angular
    velocity differs from the formula by a systematic factor, not by
    independent jitter -- so scaling the magnitude at fixed direction matches
    that failure mode where an additive Gaussian would not. On a free-jointed
    object the same factor also scales linear velocity (see
    :func:`_scale_twist` for why that is exact).

    Every domain gets the same factor, so this is a correction, not a spread:
    it means the same for a tiled point estimate as for an ensemble, and call
    sites apply it regardless of ``--estimator``.

    Args:
        factor: Multiplier on every domain's rolling-motion magnitude.
            ``1.0`` (default) is a no-op.
    """

    factor: float = 1.0

    def apply(self, qpos, qvel, rng, model, layout) -> None:  # noqa: D102
        n = qvel.shape[0]
        _scale_twist(qvel, model, layout, np.full(n, self.factor))


@dataclass(frozen=True)
class TwistUnobserved(StateNoise):
    """Erase the object's velocity -- a camera reports pose, not twist.

    Sets the slots to zero instead of perturbing them. That distinction is the
    whole point: leaving them alone would pass the *true* velocity straight
    through the sensor, which no camera does. Whatever consumes the
    observation has to derive velocity from the pose sequence itself (see
    :mod:`bampc.uncertainty.filter`).

    Only the tracked object is affected. Joint velocities stay as they are --
    a real arm's encoders do report them.
    """

    def apply(self, qpos, qvel, rng, model, layout) -> None:  # noqa: D102
        if layout is None:
            raise ValueError(
                "TwistUnobserved needs the task to define `object_pose_qpos`;"
                " this task has no manipulated object registered"
            )
        if layout.is_planar:
            for adr in (layout.x_adr, layout.y_adr, layout.yaw_adr):
                qvel[:, _dof_of_qpos(model, adr)] = 0.0
            return
        a = _dof_of_qpos(model, layout.adr)
        qvel[:, a : a + 6] = 0.0


@dataclass(frozen=True)
class SE3Bias(StateNoise):
    """Constant systematic pose offset -- a miscalibrated sensor.

    Identical to every domain, so on its own it just shifts the whole belief;
    its purpose is to compose with :class:`SE3Gaussian` into a shifted-mean
    distribution.

    Args:
        pos: Position offset (m).
        rot: Rotation offset as a rotation vector (rad).
    """

    pos: Sequence[float] = (0.0, 0.0, 0.0)
    rot: Sequence[float] = (0.0, 0.0, 0.0)

    def apply(self, qpos, qvel, rng, model, layout) -> None:  # noqa: D102
        n = qpos.shape[0]
        planar = layout is not None and layout.is_planar
        dpos = np.tile(_triple(self.pos, planar, rot=False, what="pos"), (n, 1))
        drot = np.tile(_triple(self.rot, planar, rot=True, what="rot"), (n, 1))
        _apply_pose(qpos, layout, dpos, drot)


def rotate_bias_xy(bias: SE3Bias, k: int) -> SE3Bias:
    """Turn a bias's xy offset by ``k`` quarter turns, magnitude unchanged.

    A fixed bias makes every episode sit at one arbitrary direction, so an arm
    flattered by that direction looks better than it is. Cycling the four
    compass directions makes direction a balanced factor instead.

    Args:
        bias: The offset to turn.
        k: Quarter turns counter-clockwise. Any integer; ``k`` and ``k + 4``
            give the same result.

    Returns:
        A new :class:`SE3Bias`; ``z`` and ``rot`` are untouched.

    Swaps components rather than applying a rotation matrix, so magnitude is
    preserved to the last bit and ``k=4`` returns the original exactly -- trig
    would give each direction its own rounding, an unexplained difference
    between conditions.
    """
    px, py, pz = _triple(bias.pos, planar=False, rot=False, what="pos")
    x, y = [(px, py), (-py, px), (-px, -py), (py, -px)][k % 4]
    return dataclasses.replace(bias, pos=(x, y, pz))


# Salt for the wander RNG, kept distinct from the sensor/belief salts so the
# direction stream cannot accidentally line up with a Gaussian one.
_SALT_WANDER = 0x5A_1AD5


class SE3BiasCompassWander(StateNoise):
    """A :class:`SE3Bias` whose xy direction jumps every ``period`` seconds.

    Models a miscalibration whose *direction* drifts over an episode. Wraps a
    base ``SE3Bias`` and, per ``period``-second window, turns it to one of the
    four compass directions, the choice drawn from a private per-episode RNG.

    Two invariants, both load-bearing for the paired-arm comparison: it draws
    from its **own** RNG, never the sensor RNG handed to :meth:`apply`, so a
    debiased arm sees the identical Gaussian realization; and the offset
    **magnitude is unchanged**, so only the direction wanders.

    Not a frozen dataclass: it holds the current time and a live RNG, set via
    :meth:`set_time` and :meth:`reseed`.
    """

    def __init__(
        self, base: SE3Bias, period: float = 4.0, max_windows: int = 512
    ) -> None:
        """Wrap ``base``; ``period`` is the seconds between direction jumps."""
        self.base = base
        self.period = float(period)
        self._max_windows = int(max_windows)
        self._time = 0.0
        # Benign default before the first reseed: k=0 everywhere -> the base
        # direction throughout, so a stray call is a plain fixed bias.
        self._choices = np.zeros(self._max_windows, dtype=int)

    def reseed(self, seed: int, repeat: int) -> None:
        """Draw a fresh compass choice per window for one episode.

        Deterministic in ``(seed, repeat)`` and independent of the sensor RNG,
        so every arm carrying this bias sees the identical direction sequence.
        """
        rng = np.random.default_rng([seed, repeat, _SALT_WANDER])
        self._choices = rng.integers(0, 4, size=self._max_windows)

    def set_time(self, t: float) -> None:
        """Record the current sim time (selects which window applies)."""
        self._time = float(t)

    def apply(self, qpos, qvel, rng, model, layout) -> None:  # noqa: D102
        w = int(self._time // self.period) if self.period > 0.0 else 0
        w = min(max(w, 0), self._max_windows - 1)
        rotate_bias_xy(self.base, int(self._choices[w])).apply(
            qpos, qvel, rng, model, layout
        )


# Salt for the OU RNG, distinct from the wander/sensor/belief salts so the
# correlated-error stream cannot line up with any other.
_SALT_OU = 0x0000_0075


@dataclass
class SE3OrnsteinUhlenbeck(StateNoise):
    """Temporally correlated zero-mean pose noise (Ornstein-Uhlenbeck).

    The colored counterpart to :class:`SE3Gaussian`: instead of an independent
    draw each call, the error decays toward zero with time constant ``tau`` and
    is re-excited, so consecutive readings share error the way a real pose
    tracker's does -- viewing geometry and partial occlusion change slowly as
    the object moves. Per channel::

        e_k = e_{k-1} exp(-dt/tau) + sigma sqrt(1 - exp(-2 dt/tau)) N(0, 1)

    The marginal std stays ``sigma`` for any ``tau``, and ``tau`` spans the two
    other pose models: ``tau -> 0`` is :class:`SE3Gaussian` (i.i.d.),
    ``tau -> inf`` a constant per-episode offset like :class:`SE3Bias`.

    Args:
        pos_std: Position noise std (m) -- the OU marginal. Scalar or 3-vector,
            resolved against the layout exactly like :class:`SE3Gaussian`.
        rot_std: Rotation noise std (rad) -- the OU marginal.
        tau: Correlation time (s). ``<= 0`` degenerates to i.i.d., which makes
            it one knob spanning white -> correlated -> near-bias.

    Not a frozen dataclass: it carries the current error and last sim time
    across calls, advanced via :meth:`set_time`, so ``tau`` is a true time
    constant and a faster-replanning arm samples the same process at finer
    ``dt``. Draws from its **own** RNG (like
    :class:`SE3BiasCompassWander`) so a debiased arm sees the identical
    stream and the arms stay paired. With ``dt <= 0`` it draws a fresh
    stationary sample, so a still-object calibration sees independent
    readings.
    """

    pos_std: float | Sequence[float] = 0.0
    rot_std: float | Sequence[float] = 0.0
    tau: float = 0.5

    def __post_init__(self) -> None:
        """Init the OU state and a deterministic default RNG."""
        self._t = 0.0
        self._t_prev: float | None = None
        self._e: np.ndarray | None = None
        # Deterministic, so an example run is reproducible without a reseed;
        # the sweep's Arm.reset overrides it per (seed, repeat).
        self._rng = np.random.default_rng(0)

    def reseed(self, seed: int, repeat: int) -> None:
        """Fresh private RNG and cleared state for one episode."""
        self._rng = np.random.default_rng([seed, repeat, _SALT_OU])
        self._t_prev = None
        self._e = None

    def set_time(self, t: float) -> None:
        """Record the current sim time (sets the OU step's ``dt``)."""
        self._t = float(t)

    def _sigma(self, layout: ObjectPose | None) -> np.ndarray:
        """Per-axis marginal std as a 6-vector ``[dp(3), dtheta(3)]``."""
        planar = layout is not None and layout.is_planar
        sp = _triple(self.pos_std, planar, rot=False, what="pos_std")
        sr = _triple(self.rot_std, planar, rot=True, what="rot_std")
        return np.concatenate([sp, sr])

    def apply(self, qpos, qvel, rng, model, layout) -> None:  # noqa: D102
        n = qpos.shape[0]
        sigma = self._sigma(layout)
        # One draw per call regardless of branch, so the RNG consumption is
        # constant and arms stay paired.
        fresh = self._rng.normal(0.0, 1.0, (n, 6)) * sigma
        if self._e is None or self._t_prev is None:
            e = fresh
        else:
            dt = self._t - self._t_prev
            a = np.exp(-dt / self.tau) if self.tau > 0.0 and dt > 0.0 else 0.0
            e = a * self._e + np.sqrt(max(1.0 - a * a, 0.0)) * fresh
        self._e = e
        self._t_prev = self._t
        _apply_pose(qpos, layout, e[:, 0:3], e[:, 3:6])


@dataclass(frozen=True)
class JointJitter(StateNoise):
    """Encoder noise on named joints.

    Args:
        joints: Joint names, or a glob matched against them
            (e.g. ``"fr3_joint*"``).
        qpos_std: Position noise std, in each joint's own unit (rad or m).
        qvel_std: Velocity noise std.
    """

    joints: str | Sequence[str] = "*"
    qpos_std: float = 0.0
    qvel_std: float = 0.0

    def __post_init__(self) -> None:
        """Init an empty per-model address cache (see :meth:`_addresses`)."""
        object.__setattr__(self, "_addr_cache", {})

    def apply(self, qpos, qvel, rng, model, layout) -> None:  # noqa: D102
        qadr, vadr = self._addresses(model)
        n = qpos.shape[0]
        if self.qpos_std:
            qpos[:, qadr] += rng.normal(0.0, self.qpos_std, (n, len(qadr)))
        if self.qvel_std:
            qvel[:, vadr] += rng.normal(0.0, self.qvel_std, (n, len(vadr)))

    def _addresses(
        self, model: mujoco.MjModel
    ) -> tuple[np.ndarray, np.ndarray]:
        """Resolve the joint selector to qpos / dof addresses, cached per model.

        Only single-DOF joints are eligible: adding scalar noise to one slot
        of a free joint's quaternion would leave it non-unit, and that is the
        pose noise models' job anyway.
        """
        cached = self._addr_cache.get(id(model))
        if cached is not None:
            return cached
        mj = mujoco
        names = (
            [self.joints] if isinstance(self.joints, str) else list(self.joints)
        )
        qadr, vadr = [], []
        for jid in range(model.njnt):
            name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, jid)
            if name is None:
                continue
            if not any(fnmatch.fnmatch(name, pat) for pat in names):
                continue
            jtype = model.jnt_type[jid]
            if jtype not in (mj.mjtJoint.mjJNT_HINGE, mj.mjtJoint.mjJNT_SLIDE):
                continue
            qadr.append(int(model.jnt_qposadr[jid]))
            vadr.append(int(model.jnt_dofadr[jid]))
        if not qadr:
            raise ValueError(
                f"JointJitter selector {self.joints!r} matched no "
                "single-DOF (hinge/slide) joint"
            )
        result = (np.array(qadr), np.array(vadr))
        self._addr_cache[id(model)] = result
        return result
