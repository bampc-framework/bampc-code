# model_mismatch — reproducing a run

The *model* axis (domain randomization's twin, `experiments/state_uncertainty/`
is the *state* axis): the planner always sees the true state, but its rollout
model is deliberately wrong — one or more physics parameters offset from the
truth by a fixed multiplier. The truth simulator is always exactly nominal.
`run.py` writes data only; `analysis.py`, `plot_truth_grid.py` and
`plot_truth_grid_summary.py` own every figure and number (each version's
`run.sh` calls the last two). Run a
variant with `--version <task>/model_mismatch/<name>` (path relative to
`experiments/`) — one shared `run.py`/`analysis.py` here serves every
task. Each task's version directories under `<task>/model_mismatch/` are
its current runs.

## Arm grammar

Set by a config's `arms:` list, parsed by `harness.parse_arm`:

| arm name | meaning |
| --- | --- |
| `nominal` | the true model (R=1, the reference/ceiling) |
| `<param>_<mult>` | one of the config's `bias_params` off truth by `mult` (R=1) |
| `all_<mult>` | every `bias_params` entry off truth by `mult` at once (R=1) |
| `wide_dr[_<param>]` | uniform DR over all/one `bias_params`, Average risk (R=`num_domains`) — available for a DR-vs-mismatch comparison; not used by the current configs |

`bias_params` is a name into `harness._PARAM_SPEC` (`mass`, `friction`,
`rolling_friction`, `solimp_dmin`, `impratio` — targeting a task's `block`
body — plus `wall_friction`/`wall_solimp_dmin`, targeting Peg-FR3's socket
walls). Adding a new swept parameter means adding an entry there.

The archived Peg-FR3 sweep (`archive/peg_fr3/`) additionally has its own
mount-uncertainty arm names (`point` /
`oracle` / `hedge` / `hedge_wide`, a grasp-pose offset rather than a physics
mismatch).

## Which role gets the offset

By default an arm's spec lands on the *prediction* role (the planner's
rollout model is wrong; truth stays nominal). Setting
`bias_target: truth` in a config flips this: the planner keeps its nominal
model and the arm's offset lands on truth instead. Useful for a parameter
that's harder to pin down on the real robot than mass is (e.g. friction) --
the question becomes "how wrong can reality be before a certainty-equivalent
planner suffers," rather than "how wrong can the planner's belief be."
`nominal` is unaffected either way (both roles nominal).

A third value, `bias_target: truth_grid` (plus a `true_bias_grid` list and a
single-entry `bias_params`), combines both: truth is pinned at each grid
multiplier in turn, and the ordinary bias-arm grammar reruns relative to
THAT value instead of the global nominal (`truth_grid_spec`,
`truth_grid_variants`) -- one full mismatch sweep per real-world level, to
check whether the sweep's conclusions hold regardless of what reality's own
parameter actually is (every current config works this way: push_fr3 sweeps
mass and friction, balance_fr3 mass, friction and rolling_friction).
`nominal` here means "prediction knows this cell's true
value," not "prediction is at the global default" -- those only coincide at
the grid cell where `true_bias_mult == 1.0`.

Each config's `task_params.reward_profile` points at the shared
`configs/reward/<task>.yaml` profile rather than a private copy, so retuning
that profile (e.g. via the interactive examples) changes what these sweeps
measure against next time they're run — nothing here needs manual re-sync.

Any run stays analyzable from its own version directory, e.g.:

```bash
uv run python -m experiments.model_mismatch.analysis \
    --version push_fr3/model_mismatch/mass_truth_10hz_v1
```
