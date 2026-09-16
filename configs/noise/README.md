Sensor-noise presets, named by `sensor.preset` in a sweep config or `--noise`
on an example. Built by `bampc/uncertainty/presets.py`; the models
themselves are in `uncertainty/noise.py`.

Each file is an **ordered** list, and order is semantic: a bias placed before a
Gaussian shifts the mean the Gaussian then spreads around. That is why a preset
is a list and not a mapping. Every entry is a mapping carrying a `kind`:

```yaml
- {kind: se3_bias,     pos: [0.005, -0.003, 0.0], rot: [0, 0, 0.035]}
- {kind: se3_gaussian, pos_std: 0.004, rot_std: 0.02}
```

A scalar `std` means "wherever the object can move" — a planar block only has
xy + yaw, so scalars are used throughout (see `noise._triple`).

## Kinds

Fields are listed with their defaults; every one is optional.

| `kind` | Fields | What it is |
| --- | --- | --- |
| `se3_gaussian` | `pos_std` 0.0, `rot_std` 0.0 | i.i.d. pose noise |
| `se3_ou` | `pos_std` 0.0, `rot_std` 0.0, `tau` 0.5 | Ornstein-Uhlenbeck pose noise, temporally correlated over `tau` seconds. Spans the other two: `tau`→0 is `se3_gaussian`, `tau`→∞ is `se3_bias` |
| `se3_bias` | `pos` (0,0,0), `rot` (0,0,0) | constant calibration offset; draws no random numbers |
| `twist_gaussian` | `lin_std` 0.0, `ang_std` 0.0 | i.i.d. velocity noise |
| `joint_jitter` | `joints` `"*"`, `qpos_std` 0.0, `qvel_std` 0.0 | encoder noise; `joints` is a glob, e.g. `"fr3_joint*"` |
| `twist_unobserved` | — | erases the twist rather than perturbing it |
| `angular_velocity_scale` | `factor` 1.0 | fixed multiplier on the object's rolling-motion magnitude, the same on every domain — a correction, not a spread, so call sites apply it under every estimator |

A multiplicative scale rather than a Gaussian is deliberate where a velocity is
*derived* (finite-differenced, or via a rolling constraint) rather than
measured — its error is a systematic factor, not additive jitter.

## Stateful terms

`se3_ou` carries state across a sequence of readings and must be reseeded per
episode. A **belief** cloud's spread preset must be stateless — which is why
`pose-ou-fixed.yaml` exists as the stateless twin of `pose-ou.yaml`, at the
magnitude the OU process settles to.

## `scale`

`presets.build(name, scale)` multiplies **magnitudes only**. Time constants
(`tau`), the `joints` glob and absolute multipliers (`factor`) are never
scaled — a multiplier the caller picked would lose its meaning.

Every field of every noise class must be classified as one or the other in
`presets.py` (`_SCALABLE` vs `_UNSCALED`); a field that is neither raises at
import, so a field added later cannot be silently scaled or silently skipped.

`build()` also takes `joints=` / `bias=` / `twist=` to drop whole families (a
task with no arm; a debiased sensor; an unmeasured velocity), and
`twist_scale=` to size pose and velocity terms independently.
