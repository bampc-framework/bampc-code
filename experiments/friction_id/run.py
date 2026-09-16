"""Compare three friction-handling arms across repeated open-loop shots.

Each campaign is ``cfg.attempts`` shots (default 20) at a lane whose true
friction is hidden from the planner and re-drawn randomly every
``cfg.flip_every`` attempts (see ``schedule.truth_schedule``) -- the SAME
draw sequence is reused across every arm and algorithm for a given seed, so
the comparison is apples-to-apples. Three arms (``experiments.friction_id.
harness``):

* ``wrong_estimate`` -- fixed, never-updated friction guess (R=1).
* ``wide_hedge`` -- a static wide grid, hedged but never narrowed (R=domains).
* ``adaptive`` -- the belief-collapse ladder
  (``scripts/curling/friction_id.py``), which narrows toward the truth and
  re-widens if it moves.

Per-attempt scoring reuses ``scripts.curling.friction_id.shot`` (matched-
instant prediction error) and ``scripts.curling.hedge.reject_reason``
unmodified -- this module only adds the seed/arm/algo fan, the truth
schedule, and CSV output. Writes data only; ``analysis.py`` makes every
figure. Results land in ``results/<algo>/<arm>/``.

``--version`` is a path relative to ``experiments/``, e.g.
``curling_fr3/friction_id/v1``.

Run (from the repo root)::

    ...run --version curling_fr3/friction_id/v1
    ...run --version curling_fr3/friction_id/v1 --mode smoke
    ...run --version curling_fr3/friction_id/v1 --mode quick --num-seeds 3
    ...run --version curling_fr3/friction_id/v1 --algo ps --arm adaptive

where ``...run`` is::

    uv run python -m experiments.friction_id.run
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import numpy as np
import warp as wp

from bampc.allocation import AllocationContext
from bampc.config import scenarios
from bampc.task.base import body_geom_ids
from experiments.common.dr.run_dir import (
    make_run_dir,
    write_csv,
    write_manifest,
)
from experiments.friction_id.harness import (
    BUILDERS,
    MODES,
    RunConfig,
    load_run_config,
)
from experiments.friction_id.schedule import truth_schedule
from scripts.curling import friction_id as friction_id_script
from scripts.curling.hedge import reject_reason

EXPERIMENTS_ROOT = Path(__file__).resolve().parents[1]


def run_campaign(
    cfg: RunConfig, algo: str, arm: str, seed: int
) -> tuple[list[dict], dict]:
    """Run one ``(algo, arm, seed)`` campaign of ``cfg.attempts`` shots.

    Returns ``(per-attempt rows, episode summary)``.
    """
    schedule = truth_schedule(
        cfg.attempts, cfg.flip_every, cfg.mu_lo, cfg.mu_hi, seed
    )
    campaign = BUILDERS[arm](cfg, algo, seed)
    task, planner = campaign.task, campaign.planner
    model = task.mj_model
    bank = scenarios.load(cfg.scenario_bank)
    scenario = bank[cfg.scenario]

    # Reality. Written only now: build_planner already captured the rollout
    # graph (a one-time device copy), so from here on the stepped model is a
    # truth the planner's domains never see -- see friction_id.run_attempts.
    truth_gids = [*body_geom_ids(model, "block"), int(model.geom("ground").id)]

    def set_truth(mu: float) -> None:
        model.geom_friction[truth_gids, 0] = mu

    steps_per_replan = max(
        int(round(
            (1.0 / friction_id_script.PLAN_FREQ) / model.opt.timestep
        )), 1,
    )
    replans = max(
        int(round(cfg.attempt_seconds * friction_id_script.PLAN_FREQ)), 1
    )

    rows: list[dict] = []
    for attempt in range(cfg.attempts):
        truth_mu = schedule[attempt]
        set_truth(truth_mu)
        md = scenarios.pose(bank, task, scenario)
        planner.rng = np.random.default_rng([seed, attempt])
        success, error, miss, _final_speed, peak_speed = (
            friction_id_script.shot(
                task, planner, model, md, replans, steps_per_replan,
                cfg.release_speed,
            )
        )
        reason = reject_reason(error, peak_speed)
        if reason is None and campaign.allocation is not None:
            campaign.allocation.maybe_switch(
                AllocationContext(error_window=[error])
            )
        belief = campaign.belief
        rows.append({
            "seed": seed,
            "algo": algo,
            "arm": arm,
            "attempt": attempt,
            "segment": attempt // cfg.flip_every,
            "truth_mu": truth_mu,
            "belief_mean": belief.mean if belief is not None else None,
            "belief_std": belief.std if belief is not None else None,
            "stage_r": planner.engine.num_randomizations,
            "stage_s": planner.engine.num_samples,
            "miss": float(miss),
            "success": int(bool(success)),
            "peak_speed": float(peak_speed),
            "reject_reason": reason or "",
        })

    misses = np.array([r["miss"] for r in rows])
    successes = np.array([r["success"] for r in rows])
    first_success = next(
        (r["attempt"] for r in rows if r["success"]), None
    )
    episode = {
        "seed": seed,
        "algo": algo,
        "arm": arm,
        "avg_miss": float(misses.mean()),
        "best_miss": float(misses.min()),
        "success_rate": float(successes.mean()),
        "first_success_attempt": first_success,
    }
    del campaign
    return rows, episode


def run_variant(
    cfg: RunConfig, config_dir: Path, algo: str, arm: str, subdir: str
) -> None:
    """Run every seed for one ``(algo, arm)`` condition."""
    print(
        f"\n{'=' * 70}\n{algo}/{arm}  attempts={cfg.attempts} "
        f"flip_every={cfg.flip_every}  seeds={cfg.num_seeds}\n{'=' * 70}"
    )
    rows: list[dict] = []
    episodes: list[dict] = []
    for seed in range(cfg.num_seeds):
        seed_rows, episode = run_campaign(cfg, algo, arm, seed)
        rows += seed_rows
        episodes.append(episode)
        print(
            f"  seed={seed}  avg_miss={episode['avg_miss']:.4f}  "
            f"best_miss={episode['best_miss']:.4f}  "
            f"success_rate={episode['success_rate']:.2f}"
        )
        # A fresh engine/planner is built per campaign; reclaim device
        # buffers and captured graphs before the next one -- see
        # experiments/model_mismatch/run.py's Stack.close for the same need.
        gc.collect()
        wp.synchronize()

    run_dir = make_run_dir(
        config_dir, cfg,
        sweep={
            "algo": algo, "arm": arm,
            "attempts": cfg.attempts, "flip_every": cfg.flip_every,
            "mu_lo": cfg.mu_lo, "mu_hi": cfg.mu_hi,
            "wrong_mu": cfg.wrong_mu, "domains": cfg.domains,
            "num_seeds": cfg.num_seeds,
        },
        subdir=subdir,
    )
    write_csv(run_dir / "tracking.csv", rows)
    write_csv(run_dir / "episodes.csv", episodes)
    write_manifest(run_dir, {
        "algo": algo,
        "arm": arm,
        "attempts": cfg.attempts,
        "flip_every": cfg.flip_every,
        "num_seeds": cfg.num_seeds,
    })
    print(f"  wrote {run_dir}")


def main() -> None:
    """Load the version's config and run every ``(algo, arm)`` variant."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--version", required=True,
        help="Path relative to experiments/, e.g. curling_fr3/friction_id/v1.",
    )
    parser.add_argument(
        "--mode", choices=["full", "quick", "smoke"], default="full",
        help="full = the real sweep; quick = fewer seeds (--num-seeds) at "
        "FULL attempts/flip_every, a faster statistical look; smoke = 1 "
        "seed x 4 attempts, a shape check only.",
    )
    parser.add_argument(
        "--num-seeds", type=int, default=3,
        help="Seed count for --mode quick (default 3).",
    )
    parser.add_argument(
        "--algo", action="append", default=None,
        help="Restrict to these algos (repeatable). Default: all in config.",
    )
    parser.add_argument(
        "--arm", action="append", default=None,
        help="Restrict to these arms (repeatable). Default: all in config.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip variants that already have a manifest.json (written "
        "last, so it marks a COMPLETE variant).",
    )
    args = parser.parse_args()

    config_dir = EXPERIMENTS_ROOT / args.version
    cfg = load_run_config(config_dir / "config.yaml")
    if args.mode == "smoke":
        cfg = MODES["smoke"](cfg)
    elif args.mode == "quick":
        cfg = MODES["quick"](cfg, args.num_seeds)

    prefix = [] if args.mode == "full" else [args.mode]
    for algo in cfg.algos:
        if args.algo and algo not in args.algo:
            continue
        for arm in cfg.arms:
            if args.arm and arm not in args.arm:
                continue
            subdir = "/".join(prefix + [algo, arm])
            if args.resume and (
                config_dir / "results" / subdir / "manifest.json"
            ).exists():
                print(f"  skip (done) {subdir}")
                continue
            run_variant(cfg, config_dir, algo, arm, subdir)


if __name__ == "__main__":
    main()
