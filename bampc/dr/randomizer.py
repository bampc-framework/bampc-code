"""Domain randomization for native MJWarp.

In MJWarp, randomizing a model parameter is literally overriding a batched
device field, e.g. ``m.dof_frictionloss`` becomes an ``(nworld, nv)`` array.
``DomainRandomizer`` owns a declarative **spec** (passed in), compiles it into
concrete model-field write targets, and draws per-domain values that the
rollout engine repeats across each domain's samples and ``wp.copy``s into the
live fields (capture-safe: never reassigned after the graph is captured).

Spec grammar: see ``bampc/dr/README.md``.

Update ranges online with :meth:`update` -- you may change values of
already-declared fields, but not introduce new ones, since the engine's
per-world arrays are fixed at graph-capture time.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import mujoco
import numpy as np

if TYPE_CHECKING:
    from bampc.task.base import Task

# value is a (lo, hi) range, an explicit per-domain grid, a scalar constant,
# or a {"std": s} Gaussian spec (optional "mean", default 0.0).
FieldValue = tuple[float, float] | Sequence[float] | float | dict[str, float]
RandomizationSpec = dict[str, dict[str, dict[str, FieldValue]]]

# Friendly param -> (model field, component index or None for a scalar field).
_GEOM_PARAMS = {
    "friction": ("geom_friction", 0),  # sliding friction
    "rolling_friction": ("geom_friction", 2),  # rolling resistance
    "margin": ("geom_margin", None),
    "solref_timeconst": ("geom_solref", 0),  # contact softness time constant
    "solimp_dmin": ("geom_solimp", 0),  # contact impedance at zero penetration
}
_JOINT_PARAMS = {
    "frictionloss": ("dof_frictionloss", None),
    "damping": ("dof_damping", None),
    "armature": ("dof_armature", None),
}
# A velocity actuator's servo gain (force = kv * (ctrl - qvel)). Writing
# gainprm alone would decouple it from biasprm's matching -kv term -- see the
# coupled write in _compile/_sample, mirroring body_mass/body_inertia.
_ACTUATOR_PARAMS = {"kv": ("actuator_gainprm", 0)}
# jnt_pos is indexed per-joint (not per-dof, unlike _JOINT_PARAMS above), so
# it's resolved separately in _resolve.
_JOINT_POS_PARAMS = {
    "pos_x": ("jnt_pos", 0),
    "pos_y": ("jnt_pos", 1),
}
# A body's mount pose in its parent's frame. Translation is component-wise like
# everything else; rotation is not -- see _BODY_ROT_PARAMS.
_BODY_POS_PARAMS = {
    "pos_x": ("body_pos", 0),
    "pos_y": ("body_pos", 1),
    "pos_z": ("body_pos", 2),
}
# Rotation-vector components (rad) of a mount tilt. These do NOT name a
# component of body_quat: three scalars are drawn and turned into one
# quaternion, because writing quaternion components independently produces
# neither a rotation nor a unit-norm value. Handled by _QuatTarget.
_BODY_ROT_PARAMS = {"rot_x": 0, "rot_y": 1, "rot_z": 2}
# Global solver options. Unlike every other param these live on `opt`, are
# scalar (no entity axis), and may be *stored differently* on the device than
# on the host: MJWarp keeps impratio as its inverse square root. So each entry
# names both paths plus the spec -> device transform, and spec values always
# stay in the friendly host unit (a plain impratio ratio).
# Friendly param -> (device path, host path, spec -> device value).
_OPT_PARAMS = {
    "impratio": (
        "opt.impratio_invsqrt",
        "opt.impratio",
        lambda v: 1.0 / np.sqrt(np.maximum(v, mujoco.mjMINVAL)),
    ),
}
_MJ_OBJ = {
    "geom": mujoco.mjtObj.mjOBJ_GEOM,
    "body": mujoco.mjtObj.mjOBJ_BODY,
    "joint": mujoco.mjtObj.mjOBJ_JOINT,
    "actuator": mujoco.mjtObj.mjOBJ_ACTUATOR,
}
_SINGLE_DOF = {
    int(mujoco.mjtJoint.mjJNT_SLIDE),
    int(mujoco.mjtJoint.mjJNT_HINGE),
}


class _Target:
    """One resolved write: where in a model field, and the per-domain values."""

    def __init__(
        self,
        field: str,
        index,
        comp: int | None,
        value: FieldValue,
        transform=None,
    ):
        self.field = field
        self.index = index  # int, list[int], or slice(None)
        self.comp = comp
        self.value = value
        # Spec -> device conversion; None when the two units agree.
        self.transform = transform

    def draw(self, rng: np.random.Generator, n: int) -> np.ndarray:
        """Return ``(n,)`` per-domain values for this target."""
        vals = self._draw_spec(rng, n)
        return vals if self.transform is None else self.transform(vals)

    def _draw_spec(self, rng: np.random.Generator, n: int) -> np.ndarray:
        """Draw ``(n,)`` values in the spec's own unit (pre-transform)."""
        return _draw(self.value, rng, n, self.field)

    def write(self, block: np.ndarray, vals: np.ndarray) -> None:
        """Write ``(n,)`` per-domain values into this target's slot."""
        if self.comp is None:
            block[:, self.index] = (
                vals[:, None] if _is_multi(self.index) else vals
            )
        elif _is_multi(self.index):
            block[:, self.index, self.comp] = vals[:, None]
        else:
            block[:, self.index, self.comp] = vals


class _QuatTarget:
    """A body's mount tilt: three rotation-vector draws, one quaternion write.

    Kept apart from :class:`_Target` because a quaternion is not addressable
    component-wise -- perturbing ``body_quat[k]`` alone is neither a rotation
    nor unit-norm. The three ``rot_*`` params of one body are merged here, drawn
    together, and composed onto the body's base quaternion.
    """

    field = "body_quat"
    transform = None

    def __init__(self, index: int, values: dict[int, FieldValue]):
        self.index = int(index)
        self.comp = None
        self.values = values  # rotation-vector component -> spec value

    def _draw_spec(self, rng: np.random.Generator, n: int) -> np.ndarray:
        """Draw the ``(n, 3)`` rotation vectors, in radians."""
        rotvec = np.zeros((n, 3))
        for comp, value in self.values.items():
            rotvec[:, comp] = _draw(value, rng, n, self.field)
        return rotvec

    def draw(self, rng: np.random.Generator, n: int) -> np.ndarray:
        """Return ``(n, 3)`` per-domain rotation vectors."""
        return self._draw_spec(rng, n)

    def write(self, block: np.ndarray, vals: np.ndarray) -> None:
        """Compose the tilt onto the base quaternion and write it back.

        Multiplied on the right (``q_base * q_delta``), so the perturbation is
        expressed in the body's own frame -- which is what a mount error is.
        """
        base = block[:, self.index, :]
        block[:, self.index, :] = _quat_mul(base, _rotvec_to_quat(vals))


def _draw(
    value: FieldValue, rng: np.random.Generator, n: int, field: str
) -> np.ndarray:
    """Draw ``(n,)`` values from a range, a grid, a scalar, or a Gaussian."""
    if isinstance(value, dict):
        return rng.normal(
            float(value.get("mean", 0.0)), float(value["std"]), size=n
        )
    if isinstance(value, tuple) and len(value) == 2:
        return rng.uniform(float(value[0]), float(value[1]), size=n)
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 0:
        return np.full(n, float(arr))
    if arr.shape != (n,):
        raise ValueError(
            f"grid for {field} must have length num_randomizations="
            f"{n}, got shape {arr.shape}"
        )
    return arr


def _rotvec_to_quat(rotvec: np.ndarray) -> np.ndarray:
    """``(n, 3)`` rotation vectors (rad) -> ``(n, 4)`` quats, ``[w, x, y, z]``.

    The small-angle limit is the identity, so a zero draw is exactly no tilt.
    """
    angle = np.linalg.norm(rotvec, axis=1)
    half = 0.5 * angle
    # sinc-style guard: at angle 0 the axis is undefined but sin(a/2)/a -> 1/2.
    scale = np.where(angle > 1e-12, np.sin(half) / np.where(
        angle > 1e-12, angle, 1.0
    ), 0.5)
    quat = np.empty((rotvec.shape[0], 4))
    quat[:, 0] = np.cos(half)
    quat[:, 1:] = rotvec * scale[:, None]
    return quat


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of two ``(n, 4)`` quaternion stacks, ``[w, x, y, z]``.

    Local, rather than reusing ``task.base.quat_mul``, to keep ``dr/`` free of
    ``task/`` imports -- domain randomization lives outside the Task by design.
    """
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


class DomainRandomizer:
    """Compile a randomization spec and produce per-domain field overrides."""

    def __init__(
        self,
        task: Task,
        num_randomizations: int,
        spec: RandomizationSpec,
        seed: int = 0,
    ) -> None:
        """Bind a spec to a task and validate it against the model.

        Args:
            task: The task whose ``mj_model`` defines the entities.
            num_randomizations: Number of distinct domains (R).
            spec: Randomization spec (see the module grammar).
            seed: Seed for the per-reset RNG used by range entries.
        """
        self.task = task
        self.num_randomizations = int(num_randomizations)
        self.spec: RandomizationSpec = spec
        self.rng = np.random.default_rng(seed)
        self._targets, self._bases = self._compile(spec)
        # Field set is frozen here; the engine allocates device arrays for it.
        self._frozen_fields = set(self._bases)

    def _compile(
        self, spec: RandomizationSpec
    ) -> tuple[list[_Target], dict[str, np.ndarray]]:
        """Resolve a spec into write targets + cached field bases (pure)."""
        mjm = self.task.mj_model
        targets: list[_Target | _QuatTarget] = []
        bases: dict[str, np.ndarray] = {}
        # body -> {rotvec component: value}; merged into one _QuatTarget below,
        # since three scalars must become a single quaternion write.
        tilts: dict[int, dict[int, FieldValue]] = {}
        for kind, entities in spec.items():
            for entity_name, params in entities.items():
                for param, value in params.items():
                    if kind == "body" and param in _BODY_ROT_PARAMS:
                        bid = _bid(mjm, entity_name)
                        comp = _BODY_ROT_PARAMS[param]
                        tilts.setdefault(bid, {})[comp] = value
                        bases.setdefault(
                            "body_quat", _field_base(mjm, "body_quat")
                        )
                        continue
                    field, index, comp, tf = _resolve(
                        mjm, kind, entity_name, param
                    )
                    if field not in bases:
                        bases[field] = _field_base(mjm, field)
                    if field == "body_mass" and "body_inertia" not in bases:
                        # mass writes also rescale the body's inertia
                        bases["body_inertia"] = np.asarray(
                            mjm.body_inertia
                        ).copy()
                    if (
                        field == "actuator_gainprm"
                        and "actuator_biasprm" not in bases
                    ):
                        # kv writes also keep the affine bias term coupled
                        # (biasprm[:, 2] = -gainprm[:, 0])
                        bases["actuator_biasprm"] = np.asarray(
                            mjm.actuator_biasprm
                        ).copy()
                    targets.append(_Target(field, index, comp, value, tf))
        targets += [_QuatTarget(bid, v) for bid, v in tilts.items()]
        return targets, bases

    def update(self, spec: RandomizationSpec) -> None:
        """Merge new ranges/values into the spec and re-resolve (online).

        Only values of already-declared fields may change; introducing a new
        model field raises, because the engine's per-world device arrays are
        fixed at graph-capture time.
        """
        merged = {k: {e: dict(p) for e, p in ents.items()}
                  for k, ents in self.spec.items()}
        for kind, entities in spec.items():
            for entity_name, params in entities.items():
                merged.setdefault(kind, {}).setdefault(entity_name, {}).update(
                    params
                )
        targets, bases = self._compile(merged)
        new = set(bases) - self._frozen_fields
        if new:
            raise ValueError(
                f"update introduces new model fields {new}; these need a graph "
                "rebuild. Declare them in the initial spec (any range)."
            )
        self.spec, self._targets, self._bases = merged, targets, bases

    def randomized_fields(self) -> set[str]:
        """Model fields this randomizer writes (engine pre-allocates these)."""
        return set(self._frozen_fields)

    def sample(self) -> dict[str, np.ndarray]:
        """Draw a fresh set of per-domain overrides.

        Returns:
            ``{field: (num_randomizations, *field_shape)}``. The engine repeats
            each row across that domain's samples and copies it into the field.
        """
        return self._sample()[0]

    def _sample(
        self,
    ) -> tuple[dict[str, np.ndarray], list[tuple[_Target, np.ndarray]]]:
        """Draw once; return device blocks plus each target's spec-unit draw.

        Two views of the *same* draw, so a host consumer and the device
        never see different numbers for one domain.
        """
        n = self.num_randomizations
        out = {
            f: np.repeat(b[None], n, axis=0).astype(np.float32)
            for f, b in self._bases.items()
        }
        drawn: list[tuple[_Target, np.ndarray]] = []
        for tgt in self._targets:
            spec_vals = tgt._draw_spec(self.rng, n)  # (n,) in spec units
            drawn.append((tgt, spec_vals))
            vals = (
                spec_vals
                if tgt.transform is None
                else tgt.transform(spec_vals)
            )
            tgt.write(out[tgt.field], vals)
            if tgt.field == "body_mass":
                # Inertia is linear in mass at fixed geometry: scale it by
                # the same ratio so the mass distribution keeps its shape.
                ratio = vals / self._bases["body_mass"][tgt.index]
                out["body_inertia"][:, tgt.index] *= ratio[:, None]
            if tgt.field == "actuator_gainprm":
                # Unlike mass/inertia this is a direct assignment, not a
                # ratio: the servo identity biasprm[2] = -kv holds exactly,
                # not relative to some independent baseline.
                if _is_multi(tgt.index):
                    out["actuator_biasprm"][:, tgt.index, 2] = -vals[:, None]
                else:
                    out["actuator_biasprm"][:, tgt.index, 2] = -vals
        return out, drawn

    def apply_to_mj_model(self, mjm) -> None:
        """Bake one domain's values straight onto a host ``MjModel``.

        For CPU consumers, which need MuJoCo's own units rather than the
        device layout: ``opt`` params are written to their host path
        untransformed, and everything else is copied field-for-field.
        Requires ``num_randomizations == 1``.
        """
        if self.num_randomizations != 1:
            raise ValueError(
                "apply_to_mj_model needs a single domain, got "
                f"{self.num_randomizations}"
            )
        out, drawn = self._sample()
        for field, values in out.items():
            host_path, is_opt = host_field(field)
            if not is_opt:
                getattr(mjm, host_path)[:] = values[0]
        for tgt, spec_vals in drawn:
            host_path, is_opt = host_field(tgt.field)
            if is_opt:
                set_path(mjm, host_path, float(spec_vals[0]))


def _is_multi(index) -> bool:
    """True if the index selects more than one entity (slice or list)."""
    return isinstance(index, (slice, list, np.ndarray))


def get_path(obj, path: str):
    """Read a dotted attribute path (``opt.impratio_invsqrt``)."""
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


def set_path(obj, path: str, value) -> None:
    """Write a dotted attribute path (``opt.impratio_invsqrt``)."""
    parts = path.split(".")
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)


# Device path -> (host path, spec -> device transform).
_OPT_BY_FIELD = {d: (h, t) for d, h, t in _OPT_PARAMS.values()}


def _field_base(mjm, field: str) -> np.ndarray:
    """Baseline values for a model field, in *device* units.

    Global ``opt`` fields are scalar on the host and may be stored
    transformed on the device, so they become a length-1 array carrying the
    converted value; every other field is copied as-is.
    """
    if field in _OPT_BY_FIELD:
        host_path, transform = _OPT_BY_FIELD[field]
        return np.atleast_1d(
            transform(np.asarray(get_path(mjm, host_path), dtype=float))
        )
    return np.asarray(getattr(mjm, field)).copy()


def host_field(field: str) -> tuple[str, bool]:
    """Map a device field to its host path, and whether it is an opt field."""
    if field in _OPT_BY_FIELD:
        return _OPT_BY_FIELD[field][0], True
    return field, False


def _resolve_joint(mjm, entity_name, param):
    """Resolve a ``joint`` spec entry (anchor xy, or a per-DOF field)."""
    jid = mujoco.mj_name2id(mjm, _MJ_OBJ["joint"], entity_name)
    if jid < 0:
        raise ValueError(f"joint {entity_name!r} not found")
    if param in _JOINT_POS_PARAMS:
        field, comp = _JOINT_POS_PARAMS[param]
        return field, jid, comp, None
    if param not in _JOINT_PARAMS:
        raise KeyError(f"unknown joint param {param!r}")
    field, comp = _JOINT_PARAMS[param]
    if int(mjm.jnt_type[jid]) not in _SINGLE_DOF:
        raise ValueError(
            f"joint {entity_name!r} is not single-DOF (slide/hinge)"
        )
    return field, int(mjm.jnt_dofadr[jid]), comp, None


def _resolve_actuator(mjm, entity_name, param):
    """Resolve an ``actuator`` spec entry (only ``kv`` today)."""
    if param not in _ACTUATOR_PARAMS:
        raise KeyError(f"unknown actuator param {param!r}")
    field, comp = _ACTUATOR_PARAMS[param]
    index = slice(None) if entity_name == "__all__" else _aid(
        mjm, entity_name
    )
    if param == "kv":
        idxs = range(mjm.nu) if index == slice(None) else [index]
        bad = [
            i for i in idxs
            if int(mjm.actuator_gaintype[i])
            != int(mujoco.mjtGain.mjGAIN_FIXED)
            or int(mjm.actuator_biastype[i])
            != int(mujoco.mjtBias.mjBIAS_AFFINE)
        ]
        if bad:
            raise ValueError(
                "kv-DR requires velocity-servo actuators (gaintype=fixed, "
                f"biastype=affine); actuator index(es) {bad} are not"
            )
    return field, index, comp, None


def _resolve_body(mjm, entity_name, param):
    """Resolve a ``body`` spec entry (mount pose, mass, or a geom param)."""
    bid = _bid(mjm, entity_name)
    if param in _BODY_POS_PARAMS:
        field, comp = _BODY_POS_PARAMS[param]
        return field, bid, comp, None
    if param == "mass":
        if float(mjm.body_mass[bid]) <= 0.0:
            raise ValueError(
                f"body {entity_name!r} has zero base mass; cannot "
                "derive an inertia scale"
            )
        return "body_mass", bid, None, None
    if param not in _GEOM_PARAMS:
        raise KeyError(f"unknown body param {param!r}")
    field, comp = _GEOM_PARAMS[param]
    start = int(mjm.body_geomadr[bid])
    count = int(mjm.body_geomnum[bid])
    return field, list(range(start, start + count)), comp, None


def _resolve(mjm, kind, entity_name, param):
    """Resolve ``(field, index, comp, transform)`` for one spec entry."""
    if kind == "opt":
        if param not in _OPT_PARAMS:
            raise KeyError(f"unknown opt param {param!r}")
        if entity_name != "__all__":
            raise ValueError(
                f"opt params are global; use '__all__', not {entity_name!r}"
            )
        device_path, _, transform = _OPT_PARAMS[param]
        return device_path, 0, None, transform
    if kind == "geom":
        if param not in _GEOM_PARAMS:
            raise KeyError(f"unknown geom param {param!r}")
        field, comp = _GEOM_PARAMS[param]
        index = slice(None) if entity_name == "__all__" else _gid(
            mjm, entity_name
        )
        return field, index, comp, None
    if kind == "joint":
        return _resolve_joint(mjm, entity_name, param)
    if kind == "body":
        return _resolve_body(mjm, entity_name, param)
    if kind == "actuator":
        return _resolve_actuator(mjm, entity_name, param)
    raise KeyError(f"unknown randomization kind {kind!r}")


def _gid(mjm, name: str) -> int:
    gid = mujoco.mj_name2id(mjm, _MJ_OBJ["geom"], name)
    if gid < 0:
        raise ValueError(f"geom {name!r} not found")
    return gid


def _aid(mjm, name: str) -> int:
    aid = mujoco.mj_name2id(mjm, _MJ_OBJ["actuator"], name)
    if aid < 0:
        raise ValueError(f"actuator {name!r} not found")
    return aid


def _bid(mjm, name: str) -> int:
    bid = mujoco.mj_name2id(mjm, _MJ_OBJ["body"], name)
    if bid < 0:
        raise ValueError(f"body {name!r} not found")
    return bid