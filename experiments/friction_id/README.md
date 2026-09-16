# friction_id — reproducing a run

Curling-FR3's one open-loop uncertainty axis: the lane's (puck+ice)
friction is unknown to the planner and changes randomly partway through a
campaign of repeated shots. `run.py` writes data only; `analysis.py` owns
every figure and number. Run a variant with `--version
curling_fr3/friction_id/<name>` (path relative to `experiments/`).

## Arm grammar

Set by a config's `arms:` list, built by `harness.BUILDERS`:

| arm name | meaning |
| --- | --- |
| `wrong_estimate` | friction fixed at `wrong_mu`, R=1, never updates |
| `wide_hedge` | a static `linspace(mu_lo, mu_hi, domains)` grid, Average risk, R=`domains` — hedges but never narrows |
| `adaptive` | the belief-collapse ladder (`scripts/curling/friction_id.py`'s `build`, `bampc.belief.DomainBelief` + `bampc.allocation.BeliefCollapsePolicy`/`AllocationController`) — narrows toward the truth, re-widens if it moves |

Every arm's friction spec is mirrored onto both `body.block` and
`geom.ground`: MuJoCo combines two contacting geoms' friction via
elementwise max, so writing it to only one side of the puck/lane pair is a
silent no-op (`scripts/curling/hedge.py::make_spec_fn`'s docstring; the same
trap `experiments/model_mismatch/harness.py`'s `_counter_geom_names` works
around for every other task). Puck and lane are always tied to one value per
domain — never randomized independently, which would be a genuine
identifiability dead end (only `max(mu_puck, mu_table)` is observable from a
single slide).

## Campaign structure

One campaign is `attempts` shots (default 20) at a fixed scenario. The true
friction is hidden from the planner and re-drawn randomly every
`flip_every` attempts (`schedule.truth_schedule`) — the *same* draw sequence
is reused across every arm and algo for a given seed index, so a campaign's
comparison is apples-to-apples. `num_seeds` independent campaigns are run
per `(algo, arm)` cell to build a distribution robust to MJWarp's per-run
non-determinism.

Per-attempt scoring (matched-instant predicted-vs-observed puck position,
not final rest) reuses `scripts.curling.friction_id.shot` and
`scripts.curling.hedge.reject_reason` unmodified — read those two files
before changing how a shot is judged; the reasoning for every threshold is
recorded there, not here.

## Headline metrics

Per campaign: `avg_miss` (mean puck-to-house distance over all `attempts`
shots) and `best_miss` (the minimum), compared across arms by
`analysis.py`'s `headline_fig`.
