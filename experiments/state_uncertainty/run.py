"""Control performance under state uncertainty (Push-FR3, Balance-FR3, ...).

Six arms, all at the **same GPU budget** (``nworld = R * S``):

===================  =============================================  ==  ====
arm                  plans on                                        R     S
===================  =============================================  ==  ====
``oracle``           the true state                                  1  full
``naive``            the raw noisy reading                           1  full
``point``            a Kalman estimate of the reading history        1  full
``point_debiased``   the same, on an unbiased sensor (privileged)    1  full
``ensemble_exact``   R particles around the reading, sigma stated    R  /R
``ensemble_wide``    R particles, sigma overstated                   R  /R
===================  =============================================  ==  ====

x {algos in config} x seeds x repeats. The repeats are not redundant: MJWarp
is not bit-reproducible and the divergence is discrete basin-hopping, so a
single run of a condition is a sample of a bimodal population.

Equal ``nworld`` means the ensembles genuinely trade control samples for
belief resolution. That is the real price of representing uncertainty and has
to be quoted with any result -- it is not a flaw in the comparison.

This writes **data only**; ``analysis.py`` makes every figure and every
derived number. Results land in the version's ``results/<algo>/<arm>/``.
``--version`` is a path relative to ``experiments/``, e.g.
``push_fr3/state_uncertainty/t-ou/scale0.6-tau2.0-warmup50``.

Run (from the repo root)::

    ...run --version <version>               # the real sweep
    ...run --version <version> --mode quick  # 1 seed x 1 trial
    ...run --version <version> --mode quick --algo ps --seed 3
    ...run --version <version> --algo ps --arm oracle --arm naive

where ``...run`` is::

    uv run python -m experiments.state_uncertainty.run
"""

from __future__ import annotations

import argparse
import gc
import time
from dataclasses import replace
from pathlib import Path

import warp as wp

from bampc.planner import build_planner
from bampc.rollout import WarpRolloutEngine
from bampc.uncertainty import presets
from experiments.common.uncertainty.arms import build_arm
from experiments.common.uncertainty.episode import (
    run_performance_episode,
    truth_stride,
)
from experiments.common.uncertainty.loader import (
    MODES,
    load_run_config,
    run_variants,
)
from experiments.common.uncertainty.run_dir import (
    make_run_dir,
    write_csv,
    write_manifest,
)
from experiments.common.uncertainty.setups import RunConfig
from experiments.common.uncertainty.truth import (
    ee_block_geom_ids,
    make_truth,
    plate_block_geom_ids,
    pusher_block_geom_ids,
)

EXPERIMENTS_ROOT = Path(__file__).resolve().parent.parent


def _contact_pair(cfg: RunConfig, mj_model):
    """Contact geom-id sets for this task's ground-truth tagging."""
    if cfg.task == "push_fr3":
        return ee_block_geom_ids(mj_model)
    if cfg.task == "push":
        return pusher_block_geom_ids(mj_model)
    return plate_block_geom_ids(mj_model)


class Stack:
    """Everything one variant needs, built once and reused across episodes.

    Graph capture costs seconds, so the planner, its engine and the truth are
    built once per variant and only *reseeded* per episode. Nothing that
    changes between episodes touches device memory shapes.
    """

    def __init__(self, cfg: RunConfig, arm_name: str) -> None:
        """Build the prediction task, engine, planner, truth and arm."""
        r, s = cfg.engine_shape(arm_name)
        self.task = cfg.make_prediction_task()
        self.engine = WarpRolloutEngine(
            self.task,
            num_samples=s,
            num_randomizations=r,
            # Gives engine.contact_modes() the per-domain bitmask the belief
            # states rolled out from. Recorded inside the rollout graph,
            # which is the only place it is affordable.
            record_initial_state=True,
        )
        # The config names one sample count; each arm spends its shared
        # budget differently, so the planner is built at the arm's own S.
        # Kept on the Stack so resolved.json records what actually ran
        # rather than the config's placeholder.
        pc = replace(cfg.planner, num_samples=s)
        # A per-arm settle override, so the oracle can skip a settle its
        # perfect (already-feasible) state has no interpenetration to resolve.
        # Recorded via self.planner_config below, so resolved.json shows the
        # real value per arm.
        override = cfg.arm_overrides.get(arm_name, {})
        if "settle_steps" in override:
            pc = replace(pc, settle_steps=int(override["settle_steps"]))
        self.planner_config = pc
        self.planner = build_planner(
            self.planner_config, self.task, self.engine
        )
        self.arm = build_arm(cfg, arm_name, self.task)
        # A belief that needs the rollout engine as a contact-mode oracle
        # (only CollapsedEnsemble today) gets it here. A no-op for every other
        # arm -- `arm.belief` is None or a plain StateUncertainty with no such
        # method -- so no existing variant's behaviour changes.
        if hasattr(self.arm.belief, "attach_engine"):
            self.arm.belief.attach_engine(self.engine)
        self.truth_task = cfg.make_truth_task()
        self.truth = make_truth(
            cfg.truth_backend,
            self.truth_task,
            spec={},
            contact_pair=_contact_pair(cfg, self.truth_task.mj_model),
        )
        self.shape = (r, s)

    def close(self) -> None:
        """Release device buffers and captured graphs before the next build.

        A reference cycle through the graph outlives refcounting, so without
        this the next variant's buffers co-exist with these.
        """
        del self.planner, self.engine, self.truth
        del self.task, self.truth_task
        gc.collect()
        wp.synchronize()


def run_variant(cfg: RunConfig, config_dir: Path, subdir: str) -> None:
    """Run every seed x repeat for one (algo, arm) condition."""
    arm_name = cfg.arm
    stack = Stack(cfg, arm_name)
    r, s = stack.shape
    bank = cfg.bank
    dt_pred, dt_truth = stack.task.dt, stack.truth_task.dt
    plan_freq = cfg.plan_freq_for(arm_name)
    steps_pred, num_replans = cfg.replan_counts(dt_pred, arm_name)

    run_dir = make_run_dir(
        config_dir,
        replace(cfg, planner=stack.planner_config),
        subdir=subdir,
        conditions={
            "algo": cfg.planner.algo,
            "arm": arm_name,
            "num_randomizations": r,
            "num_samples": s,
            "nworld": r * s,
            "plan_freq_hz": plan_freq,
            "compute_lag": cfg.compute_lag_for(arm_name),
            "scenario_bank": cfg.scenario_bank,
            "seeds": list(range(cfg.seed_offset, cfg.seed_offset
                                + cfg.num_seeds)),
            # The RESOLVED magnitudes, not just `preset: full, scale: 3.0`.
            # A preset file is editable, so a run that records only its name
            # loses its meaning the moment someone retunes it -- the same
            # reason ModelConfig is expanded rather than named here. Per arm,
            # so a debiased arm's dropped SE3Bias and an ensemble's widened
            # sigma are both visible.
            # The bias below is direction 0: resolved_noise is captured before
            # any episode runs, and Arm.reset turns it per episode. Recorded
            # so a results dir says the bias rotated rather than implying it
            # was fixed.
            "bias_directions": cfg.sensor.bias_directions,
            "resolved_noise": {
                "sensor": (
                    presets.describe(stack.arm.sensor.noise)
                    if stack.arm.sensor is not None
                    else None
                ),
                "belief": (
                    presets.describe(stack.arm.belief.noise)
                    if stack.arm.belief is not None
                    else None
                ),
            },
        },
    )
    print(
        f"\n{'=' * 70}\n{cfg.planner.algo}/{arm_name}  R={r} S={s} "
        f"(nworld={r * s})  {plan_freq:g}Hz  "
        f"dt_pred={dt_pred:g} dt_truth={dt_truth:g}\n{'=' * 70}"
    )

    rows: list[dict] = []
    episodes: list[dict] = []
    for seed in range(cfg.seed_offset, cfg.seed_offset + cfg.num_seeds):
        scenario = bank[seed]
        for repeat in range(cfg.repeats):
            t_start = time.perf_counter()
            result = run_performance_episode(
                cfg, stack.task, stack.planner, stack.truth, stack.arm,
                seed, repeat, scenario,
            )
            wall_s = time.perf_counter() - t_start
            rows += result.rows
            episodes.append({
                "seed": seed,
                "repeat": repeat,
                "contact_fraction": result.contact_fraction,
                "n_chunks": len(result.rows),
                "wall_s": round(wall_s, 2),
                # None (not int(None)) when the task defines no success
                # condition; int rather than bool for the csv/genfromtxt
                # round-trip.
                "success": (
                    None if result.success is None else int(result.success)
                ),
            })
        # One line per seed, not per repeat: 64 lines a variant is noise.
        last = episodes[-cfg.repeats:]
        contact = sum(e["contact_fraction"] for e in last) / len(last)
        print(
            f"  seed={seed}  {cfg.repeats} repeats  "
            f"contact={contact:.0%}  {sum(e['wall_s'] for e in last):.1f}s"
        )

    write_csv(run_dir / "tracking.csv", rows)
    write_csv(run_dir / "episodes.csv", episodes)
    write_manifest(
        run_dir,
        {
            "schema": 2,  # 2: episodes.csv/tracking.csv gained success
            "task": cfg.task,
            "algo": cfg.planner.algo,
            "arm": arm_name,
            "truth_backend": cfg.truth_backend,
            "num_randomizations": r,
            "num_samples": s,
            "dt_pred": dt_pred,
            "dt_truth": dt_truth,
            "truth_stride": truth_stride(dt_pred, dt_truth),
            "chunk_len": steps_pred,
            # This arm's own rate. Analysis MUST integrate with the
            # per-variant value, not a shared one.
            "plan_freq_hz": plan_freq,
            "replan_dt": 1.0 / plan_freq,
            "compute_lag": cfg.compute_lag_for(arm_name),
            "num_seeds": cfg.num_seeds,
            "repeats": cfg.repeats,
            "num_chunks": num_replans,
        },
    )
    stack.close()
    print(f"  wrote {run_dir}")


def main() -> None:
    """Load the version's config and run every (algo, arm) variant."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--version", required=True,
        help="Path relative to experiments/, e.g. "
        "push_fr3/state_uncertainty/t-ou/scale0.6-tau2.0-warmup50.",
    )
    parser.add_argument(
        "--mode", choices=["full", "smoke", "quick"], default="full",
        help="full = the real sweep; quick = 1 seed x 1 trial at FULL "
        "episode length, for tuning; smoke = 1 seed x 1 trial at "
        "2 s, a shape check whose numbers mean nothing.",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Which bank scenario to run in --mode quick (default 0).",
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
        help="Skip variants that already have a manifest.json -- which is "
        "written last, so it marks a COMPLETE variant. Lets an interrupted "
        "run continue without redoing finished work (and without overwriting "
        "good results with a fresh, non-identical draw). Assumes the config "
        "has not changed since; a completed variant is trusted as-is.",
    )
    parser.add_argument(
        "--sensor-scale", type=float, default=None,
        help="Override sensor.scale for a quick noise scan. Multiplies every "
        "preset magnitude (bias + Gaussian together).",
    )
    parser.add_argument(
        "--scenario-bank", default=None,
        help="Override the scenario bank for a quick probe (e.g. a no-drift "
        "bank). The start states must match the config's geometry.",
    )
    args = parser.parse_args()

    config_dir = EXPERIMENTS_ROOT / args.version
    cfg = load_run_config(config_dir / "config.yaml")
    # Scan overrides, applied before the mode transform. Meant for --mode quick
    # exploration; a reporting run edits the config so resolved.json records it.
    if args.sensor_scale is not None:
        cfg = replace(cfg, sensor=replace(cfg.sensor, scale=args.sensor_scale))
    if args.scenario_bank is not None:
        cfg = replace(cfg, scenario_bank=args.scenario_bank)
    if args.mode == "quick":
        cfg = MODES["quick"](cfg, args.seed)
    elif args.mode == "smoke":
        cfg = MODES["smoke"](cfg)

    for parts, variant in run_variants(cfg):
        if args.algo and variant.planner.algo not in args.algo:
            continue
        if args.arm and variant.arm not in args.arm:
            continue
        # Reduced runs are quarantined under results/<mode>/: one seed
        # written beside a real run would silently replace hours of results
        # with a single episode.
        prefix = [] if args.mode == "full" else [args.mode]
        subdir = "/".join(prefix + parts)
        if args.resume and (
            config_dir / "results" / subdir / "manifest.json"
        ).exists():
            print(f"  skip (done) {subdir}")
            continue
        run_variant(variant, config_dir, subdir)


if __name__ == "__main__":
    main()
