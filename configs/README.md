Five name-keyed **profile registries**. A caller names a profile and a loader
finds the file; adding a profile is dropping a YAML in. Directory constants
live in `bampc/__init__.py`.

| Directory | Holds | Loader | Named by |
| --- | --- | --- | --- |
| `numerics/` | Solver / integrator options written onto `mj_model.opt` | `config/numerics.py` → `ModelConfig` | the task name in examples, a sweep's `prediction.numerics` |
| `planner/` | Planner hyperparameters (samples, horizon, knots, risk) | `config/planner.py` → a flat dict | the task name, in examples and sweeps |
| `reward/` | Cost weights and task limits, spread into a `Task` constructor | `config/reward.py` → a flat dict | the task name |
| `scenarios/` | Frozen start states + one goal drift per task | `config/scenarios.py` → `ScenarioBank` | `--bank`, then `--scenario` picks the index |
| `noise/` | Sensor-noise presets (see `noise/README.md`) | `uncertainty/presets.py` | `--noise`, a sweep's `sensor.preset` |

## The one rule

**Name a profile; never retype its values.** That is what keeps an example and
a sweep on the same physics, geometry and sensor — retyped values have
silently drifted apart before. A results directory records the profile *name*,
so retuning a profile must not silently retune another task: prefer a new file
over editing a shared one.

## What is legal in a file

`numerics/` and `planner/` are validated against a dataclass, so the dataclass
is the authority on which keys exist and adding a field there admits it here
with no loader change:

- `numerics/` → every field of `ModelConfig` (`task/base.py`). **Omitting a
  field is not the same as choosing a MuJoCo default** — it loads as `None` and
  falls through to the task's own `_BASELINE`. Stating it here is the only way
  to control it from a profile.
- `planner/` → every field of `PlannerConfig` (`planner/config.py`) except
  `algo`, which is always a caller's run choice. A profile carries every field
  even where a given algorithm ignores it (only `cem.py` reads
  `init_std`/`elite_frac`/`num_elite`/`adapt_covariance`; only `mppi.py` reads
  `temperature`), so it stays a complete, swappable description.

**`planner/`'s `num_samples` is a recommendation, not a budget.** It is a
count verified to make that task work sim-to-sim on the machine it was tuned
on. How much total `nworld` actually fits inside a replan period is a property
of *your* GPU and of `settle_steps` (each settle step is another physics step
in every world), so no profile can state it. Measure your own with
`scripts/probing/bisect_*_budget.py` and supply it at the call site — a sweep's
`planner.overrides.sample_budget`, or a ROS launcher's `--sample-budget` —
rather than editing a profile, which would hand one machine's number to every
other caller.

`reward/` has no dataclass — its keys are whatever the task's constructor
accepts, and they genuinely vary per task. An unknown one already fails loudly
as a `TypeError` there. Note a key here must **not** also be passed explicitly
at a call site, or the spread raises.

## Per-axis branches

`planner/` and `reward/` profiles may carry the axes a task actually has. The
loader resolves them and returns one flat dict, so no caller re-implements the
merge:

- `by_sampling:` — `task` / `joint` keys, for a real difference between
  sampling spaces. A branch may nest `by_manipulation:` (`free` / `joint`) for
  the one task with both axes, `push_fr3`.
- `by_algo:` — algorithm-name keys plus a `default` (`planner/push.yaml` only).

## `configs/` vs `models/`

Kept apart on purpose. `configs/` is a registry resolved **by name**;
`models/` is the MJCF asset tree resolved **by relative path from inside the
MJCF**, so moving it would break mesh resolution. Neither ships in the wheel —
both resolve only from a source checkout.
