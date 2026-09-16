# state_uncertainty — reproducing a run

The *state* axis (domain randomization's twin, `experiments/model_mismatch/`
is the *model* axis): the rollout model is always correct, but the planner's
state comes from a noisy sensor and, for some arms, a filter or belief
ensemble built on it. `run.py` writes data only; `analysis.py` owns every
figure and number. One shared pair serves every task; run a variant with
`--version <task>/state_uncertainty/<name>` (path relative to
`experiments/`). Each task's version directories under
`<task>/state_uncertainty/` are its current runs.

## Naming convention: `<shape>-<noise>/<cell>`

A run's folder name encodes what varies — everything else (task, sampling
space, reward weights, budget, seeds x repeats) is held constant per task and
recorded in each run's own `config.yaml`.

- **shape** — which object, set by the scenario bank (`t` = the T block of
  `push_fr3_t`).
- **noise** — the sensor preset (`configs/noise/`, grammar in its
  `README.md`), e.g. `ou` for `pose-ou`.
- **cell** — one setting inside that family, e.g. `scale0.6-tau2.0-warmup50`
  for `sensor.scale`, `sensor.tau` and `kalman.warmup_steps`.
  `oracle-warmup50` holds the noise-free oracle arm that the other cells
  symlink in.

## Modes

`--mode quick --seed N` runs one full-length episode of one scenario, for
tuning; `--mode smoke` is a 2 s shape check whose numbers mean nothing.
Reduced runs land under `results/<mode>/`. Below 8 episodes `analysis.py`
suppresses the CI/p-value rather than printing a fake null.

## Gotchas

Estimation **arms** (`ARM_KINDS` in `common/uncertainty/arms.py`) x algos,
fanned into `results/<algo>/<arm>/`. An arm's *name* picks its estimator;
its *sizing* (rate, samples) comes from the config's `arm_overrides`, so a
new control arm is a config line plus a name in `ARM_KINDS`.

- **Rate is per arm, not global** (`point_fast` replans at 20 Hz) — the
  Kalman filter's `dt` must be the arm's own replan period, or its
  constant-velocity prediction integrates over the wrong interval.
- **Every arm but `oracle` pays one replan interval of compute lag by
  default** (`compute_lag: true`, `plan_step` in `episode.py`) — a real
  system can't act on a just-observed state instantaneously, so what
  executes an interval is what was computed last interval, forward-predicted
  past the lag to the moment it actually takes over via `planner.optimize`'s
  `dt_lag` — real `mj_step`s under the still-executing plan on the planner's
  own model, the same `SamplingPlanner._predict_forward` mechanism the ROS
  deployment uses, not a cheap kinematic stand-in. `arm_overrides: {<arm>:
  {compute_lag: false}}` disables it for one arm, for an ablation; `oracle` is never
  affected either way — it has no sensor and models the zero-latency upper
  bound by definition.
- The scenario bank owns `shape`/`scale`/`goal_drift`/`goal_xy`; a config
  restating any of them raises (`loader._BANK_OWNED`).
- Every arm reads one shared sensor stream per `(seed, repeat)` — paired,
  not independent. Verify on the raw draws
  (`scripts/probing/uncertainty_check.py`), not on `obs_pos_err`, which
  differences against a truth the arms have already driven apart.
- An ensemble's `est_pos_err` equals the raw observation error by
  construction (the cloud is built inside the planner) — it hedges the
  estimate, it does not improve it.
- **Score is pose error on SO(3) x R^3, not cost** (`--rot-scale`, defaults
  to the run's own `w_orient/w_pos`); cost is a diagnostic only. Check
  `pose_error_components` beside the headline — the combined score alone
  hides a filter trading rotation for position.
- **Debias** = building that arm's sensor without its `SE3Bias` term, not
  subtracting an offset after the fact.
- Preset names describe channel *coverage*, not magnitude — `full` is the
  *gentlest* preset (4 mm sigma). Don't size a sweep's noise off the name.
- `repeats:` != `num_seeds:` — one run samples a bimodal population, so
  analysis reports median +/- IQR, never mean +/- std.
- Only shape-matched contrasts (`CONTRASTS` in `analysis.py`) support a
  causal read — an ensemble-vs-oracle contrast also trades control samples,
  which moves cost on its own.
