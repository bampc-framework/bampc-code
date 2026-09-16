`DomainRandomizer`'s spec grammar (see `randomizer.py`): raw data, no wrapper
classes -- `{kind: {entity_name: {param: value}}}`.

- `kind` is `"geom"`, `"joint"`, `"body"`, `"opt"` or `"actuator"`.
- `entity_name` is a model entity name, or `"__all__"` for every entity of
  that kind (`opt` is global, so it takes `"__all__"` only).
- `param` is a friendly name:
  - geom: `"friction"` (sliding component, `geom_friction[:, 0]`),
    `"rolling_friction"`, `"margin"`, `"solref_timeconst"` /
    `"solimp_dmin"` (the contact solver's reference time-constant and
    minimum impedance -- how soft/springy a contact is).
  - joint: `"frictionloss"`, `"damping"`, `"armature"`, `"pos_x"` /
    `"pos_y"` (the joint anchor's xy, e.g. to mismatch a 3-DOF block's
    assumed COM).
  - opt: `"impratio"` (the solver's friction-to-normal impedance ratio --
    global, not per entity).
  - actuator: `"kv"` (a velocity actuator's servo gain, `force = kv *
    (ctrl - qvel)`; only actuators with `gaintype="fixed"` /
    `biastype="affine"` are accepted). `actuator_biasprm[:, 2]` is coupled
    to `-kv`, same coupled-write shape as body mass/inertia below.
  - body: the geom params, applied to the body's child geoms, plus
    `"mass"` (kg; `body_inertia` scales by the same `mass / base_mass`
    ratio, since inertia is linear in mass at fixed geometry -- the
    precomputed solver reference `*_invweight0` stays at baseline) and the
    body's **mount pose** in its parent's frame: `"pos_x"` / `"pos_y"` /
    `"pos_z"` (m) and `"rot_x"` / `"rot_y"` / `"rot_z"` (rad), e.g. to
    mismatch where a tool is bolted to a wrist. The three `rot_*` are drawn
    together, read as a rotation vector, and composed onto the body's base
    quaternion as one unit-norm write (see `_QuatTarget`) -- a quaternion
    can't be written one component at a time and stay a rotation. **A body
    whose only joint is free takes its pose from `qpos`, so randomizing
    these on it is a silent no-op** -- they bite on fixed bodies and
    hinge/slide children, where they set the joint frame's origin.
- `value` is one of: `(lo, hi)` (`Uniform(lo, hi)` per domain), a
  length-`num_randomizations` list/array (explicit per-domain grid), a
  scalar (fixed for every domain), or `{"std": s}` (optional `"mean"`,
  default `0.0`) for `Normal(mean, std)` per domain.

**Randomizing `friction`/`rolling_friction` on only one side of a contact
pair can be a silent no-op.** MuJoCo combines two contacting geoms'
friction via elementwise max (not `solmix` -- that only weights
`solref`/`solimp` averaging, and nothing in this repo sets `priority`
either, so max is always the rule in effect). If you randomize a geom's
friction downward but its counter-geom (the surface/tool it contacts)
keeps a fixed, higher nominal friction, the combined contact friction
stays clamped at the counter-geom's value regardless of what you wrote --
the spec applied cleanly, but the physics didn't change. Mirror the same
value onto every geom the randomized one contacts (see
`scripts/curling/range_check.py` for computing the effective post-max
friction directly).

Example:

```python
{"joint": {"block_x": {"frictionloss": (0.1, 1.8)},
           "block_yaw": {"frictionloss": [0.0, 0.3, 0.6, 1.0]}},  # grid
 "geom":  {"pusher": {"margin": (0.0, 0.01)},
           "__all__": {"friction": (0.2, 1.5)}}}
```
