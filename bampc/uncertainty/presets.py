"""Named sensor-noise presets, loaded from ``configs/noise/*.yaml``.

Shared data the caller names, exactly like a numerics profile: an example and
a sweep that say ``"full"`` get the same sensor, so retuning one cannot
silently retune the other. Adding a preset is dropping a YAML in.

A preset is an **ordered list**, not a flat field dict, because order is
semantic: a bias before a Gaussian shifts the mean it spreads around. Each
entry carries a ``kind`` tag:

.. code-block:: yaml

    - {kind: se3_bias,     pos: [0.005, -0.003, 0.0], rot: [0, 0, 0.035]}
    - {kind: se3_gaussian, pos_std: 0.004, rot_std: 0.02}

Magnitudes are "plausible camera + encoder" at scale 1 and scale linearly with
``scale``. A planar block only has xy + yaw, so scalars are used throughout --
a scalar means "wherever the object can move"
(see :func:`bampc.uncertainty.noise._triple`).
"""

from __future__ import annotations

import dataclasses
from functools import cache

import numpy as np

from bampc import NOISE_DIR
from bampc._yaml import list_names, load_yaml
from bampc.uncertainty.noise import (
    AngularVelocityScale,
    JointJitter,
    SE3Bias,
    SE3Gaussian,
    SE3OrnsteinUhlenbeck,
    StateNoise,
    TwistGaussian,
    TwistUnobserved,
)

# YAML `kind` -> class.
_KINDS: dict[str, type[StateNoise]] = {
    "se3_gaussian": SE3Gaussian,
    "se3_ou": SE3OrnsteinUhlenbeck,
    "se3_bias": SE3Bias,
    "twist_gaussian": TwistGaussian,
    "joint_jitter": JointJitter,
    "twist_unobserved": TwistUnobserved,
    "angular_velocity_scale": AngularVelocityScale,
}

# Which fields `scale` multiplies. Everything else is left alone -- notably
# `JointJitter.joints`, a glob that must never be arithmetic'd.
_SCALABLE: dict[type[StateNoise], frozenset[str]] = {
    SE3Gaussian: frozenset({"pos_std", "rot_std"}),
    SE3OrnsteinUhlenbeck: frozenset({"pos_std", "rot_std"}),
    SE3Bias: frozenset({"pos", "rot"}),
    TwistGaussian: frozenset({"lin_std", "ang_std"}),
    JointJitter: frozenset({"qpos_std", "qvel_std"}),
    TwistUnobserved: frozenset(),
    AngularVelocityScale: frozenset(),
}
# Fields deliberately NOT scaled. Listed rather than implied, so the check
# below can tell "decided not to scale this" from "forgot about it".
_UNSCALED: dict[type[StateNoise], frozenset[str]] = {
    SE3Gaussian: frozenset(),
    # tau is a time constant, not a magnitude -- scale must not touch it.
    SE3OrnsteinUhlenbeck: frozenset({"tau"}),
    SE3Bias: frozenset(),
    TwistGaussian: frozenset(),
    JointJitter: frozenset({"joints"}),
    TwistUnobserved: frozenset(),
    # An absolute multiplier the caller picked, not a magnitude.
    AngularVelocityScale: frozenset({"factor"}),
}


def _check_scale_rules() -> None:
    """Every field of every noise class must be classified, at import.

    The one real hazard in loading presets from data: a field added to a
    noise class later would otherwise be silently scaled or silently skipped,
    and either way the YAML would keep loading and the magnitudes would be
    quietly wrong. Failing at import is the whole point of splitting
    :data:`_SCALABLE` from :data:`_UNSCALED` -- without this check, storing
    presets as data is strictly worse than the Python literals it replaced.
    """
    for cls in _KINDS.values():
        fields = {f.name for f in dataclasses.fields(cls)}
        classified = _SCALABLE[cls] | _UNSCALED[cls]
        missing = fields - classified
        if missing:
            raise RuntimeError(
                f"{cls.__name__} has unclassified field(s) {sorted(missing)}: "
                "add each to _SCALABLE (scale multiplies it) or _UNSCALED "
                "(it does not) in bampc/uncertainty/presets.py"
            )
        unknown = classified - fields
        if unknown:
            raise RuntimeError(
                f"{cls.__name__} scale rules name non-existent field(s) "
                f"{sorted(unknown)}"
            )


_check_scale_rules()


def _scaled(value, scale: float):
    """Multiply a magnitude by ``scale``, preserving scalar-vs-sequence."""
    if np.isscalar(value):
        return float(value) * scale
    return tuple(float(v) * scale for v in value)


def _build_one(entry: dict, scale: float) -> StateNoise:
    """Turn one YAML entry into a noise model."""
    if not isinstance(entry, dict) or "kind" not in entry:
        raise ValueError(
            f"noise entry must be a mapping with a `kind`, got {entry!r}"
        )
    spec = dict(entry)
    kind = spec.pop("kind")
    try:
        cls = _KINDS[kind]
    except KeyError:
        raise ValueError(
            f"unknown noise kind {kind!r}; available: {sorted(_KINDS)}"
        ) from None
    unknown = set(spec) - {f.name for f in dataclasses.fields(cls)}
    if unknown:
        raise ValueError(
            f"{kind}: unknown field(s) {sorted(unknown)}"
        )
    scalable = _SCALABLE[cls]
    return cls(**{
        k: _scaled(v, scale) if k in scalable else v for k, v in spec.items()
    })


def names() -> list[str]:
    """Every preset name available under ``configs/noise/``."""
    return list_names(NOISE_DIR)


@cache
def _raw(name: str) -> tuple[dict, ...]:
    """Read one preset file, cached. Returns the unscaled entries."""
    data = load_yaml(NOISE_DIR, name, "noise preset")
    if data is None:
        data = []
    if not isinstance(data, list):
        path = NOISE_DIR / f"{name}.yaml"
        raise ValueError(
            f"{path}: a preset is an ordered LIST of noise entries, got "
            f"{type(data).__name__}"
        )
    return tuple(data)


def build(
    name: str,
    scale: float = 1.0,
    *,
    joints: bool = True,
    bias: bool = True,
    twist: bool = True,
    twist_scale: float | None = None,
) -> list[StateNoise]:
    """Build a preset's noise list.

    Args:
        name: Preset name (see :func:`names`).
        scale: Multiplies every magnitude, except ``twist_gaussian``'s when
            ``twist_scale`` overrides it.
        joints: Keep :class:`~bampc.uncertainty.noise.JointJitter`
            terms. ``False`` for a task with no arm, where the selector
            would match nothing and raise.
        bias: Keep :class:`~bampc.uncertainty.noise.SE3Bias` terms.
            ``False`` gives the zero-mean part alone -- which is both what a
            *belief* around an observation should be, and (exactly, since
            ``SE3Bias`` draws no random numbers) what a perfectly debiased
            sensor produces.
        twist: Keep :class:`~bampc.uncertainty.noise.TwistGaussian`
            and :class:`~bampc.uncertainty.noise.AngularVelocityScale`
            terms. ``False`` when the object's velocity is not measured at
            all -- there is then nothing to add noise to. Note this only
            *stops perturbing* the twist; erasing it takes
            :class:`~bampc.uncertainty.noise.TwistUnobserved`.
        twist_scale: Multiplies ``twist_gaussian`` magnitudes instead of
            ``scale``, so a
            preset's pose and velocity terms can be sized independently
            (e.g. a fixed belief that hedges against a miscalibrated
            *velocity* estimate more than its pose). ``None`` (default)
            falls back to ``scale``, matching every existing caller's
            behavior exactly. Ignored by ``angular_velocity_scale``, whose
            ``factor`` is an absolute multiplier, not a magnitude.

    Returns:
        The noise models, in application order. A fresh list each call, so a
        caller may append to it (the sweep appends ``TwistUnobserved``).
    """
    ts = scale if twist_scale is None else twist_scale
    _twist_kinds = {"twist_gaussian"}
    noise = [
        _build_one(e, ts if e.get("kind") in _twist_kinds else scale)
        for e in _raw(name)
    ]
    if not joints:
        noise = [n for n in noise if not isinstance(n, JointJitter)]
    if not bias:
        noise = [n for n in noise if not isinstance(n, SE3Bias)]
    if not twist:
        noise = [
            n for n in noise
            if not isinstance(n, (TwistGaussian, AngularVelocityScale))
        ]
    return noise


def describe(noise: list[StateNoise]) -> list[dict]:
    """Expand a built noise list into JSON-able records.

    For provenance: a results directory that records only ``preset: full``
    loses its meaning the moment the preset file is edited. Recording the
    resolved magnitudes pins it to what it meant on the day it ran, the same
    way ``ModelConfig`` is expanded into ``resolved.json``.

    A model that no preset can name -- one whose magnitude is not a constant,
    like :class:`~bampc.uncertainty.noise.PosteriorGaussian` reading
    a live filter -- records its class name and whatever plain attributes it
    carries.
    There is no resolved magnitude to pin, because it does not have one.
    """
    inverse = {cls: kind for kind, cls in _KINDS.items()}
    out = []
    for n in noise:
        if type(n) in inverse and dataclasses.is_dataclass(n):
            out.append({"kind": inverse[type(n)], **dataclasses.asdict(n)})
            continue
        out.append({
            "kind": type(n).__name__,
            **{
                k: v
                for k, v in vars(n).items()
                if isinstance(v, (int, float, str, bool))
            },
        })
    return out


def gaussian_stds(noise: list[StateNoise], scale: float = 1.0) -> dict:
    """Total zero-mean std per channel, for a filter that knows its sensor.

    Independent terms add in quadrature. ``scale`` multiplies the result, so a
    filter can be handed a deliberately mis-stated covariance without
    rebuilding the noise list.

    Returns:
        ``pos_std`` / ``rot_std`` / ``lin_vel_std`` / ``ang_vel_std``, each a
        scalar. A zero channel is floored at a small positive value, since a
        Kalman filter cannot take a singular measurement covariance.

    The floor is why an *unmeasured* channel must be dropped from the filter's
    ``H`` (``PoseKalman(observe_twist=False)``) rather than left here: a
    near-zero measurement covariance reads as a perfect reading.
    """
    def biggest(v) -> float:
        """Largest component, so a 3-vector std collapses to one number."""
        return float(np.max(np.abs(np.asarray(v, dtype=float))))

    acc = {"pos_std": 0.0, "rot_std": 0.0,
           "lin_vel_std": 0.0, "ang_vel_std": 0.0}
    for n in noise:
        if isinstance(n, (SE3Gaussian, SE3OrnsteinUhlenbeck)):
            # The OU marginal std IS pos_std/rot_std, so a filter told its
            # sensor sees the same per-channel std as the i.i.d. case.
            acc["pos_std"] += biggest(n.pos_std) ** 2
            acc["rot_std"] += biggest(n.rot_std) ** 2
        elif isinstance(n, TwistGaussian):
            acc["lin_vel_std"] += biggest(n.lin_std) ** 2
            acc["ang_vel_std"] += biggest(n.ang_std) ** 2
    return {k: max(v**0.5 * scale, 1e-6) for k, v in acc.items()}
