"""Control performance under a wrong rollout model.

Fans a config's arms x algos x seeds x repeats, at a fixed GPU budget
(``nworld = R * S``) per arm, while the planner always sees the true state
but rolls out a deliberately wrong model (the truth simulator is always
nominal by default -- see ``harness.py``'s module docstring). A config may
set ``bias_target: truth`` to flip this: the planner's model stays nominal
and the arm's offset lands on the truth instead, for parameters (e.g.
friction) that are less reliably known in reality than in simulation.
``bias_target: truth_grid`` (plus ``true_bias_grid``) does both: truth is
pinned at each grid value in turn and the ordinary arm grammar reruns
relative to THAT value, fanned by ``truth_grid_variants`` instead of
``run_variants`` -- "does model mismatch behave the same regardless of what
reality's own parameter actually is." Arm grammar is per ``parse_arm``:

* ``nominal`` -- the true model (the reference / oracle);
* ``<param>_<mult>`` -- one swept parameter off the truth by ``mult``, R=1;
* ``all_<mult>`` -- every one of the config's ``bias_params`` off the truth
  by ``mult`` at once, R=1;
* ``wide_dr`` / ``wide_dr_<param>`` -- uniform domain randomization over all,
  or one, of those parameters, Average risk (R=num_domains) -- not used by
  the current single-model-mismatch configs (``push_v1_final`` etc.), but
  still available.

Peg-FR3 additionally supports its own mount-uncertainty arm names
(``point``/``oracle``/``hedge``/``hedge_wide``, see ``_peg_arm_spec``) for
configs that use them. A config that also sets ``clearance_grid``
(``archive/peg_fr3/model_mismatch/peg_grid_v1``) fans over clearance x
``mount_noise`` instead of the ordinary (algo, arm) fan -- see
``peg_grid_variants``.

This writes **data only**; ``analysis.py`` makes every figure. Results land
in ``results/<algo>/<arm>/``.

``--version`` is a path relative to ``experiments/``, e.g.
``push_fr3/model_mismatch/mass_truth_10hz_v1``.

Run (from the repo root)::

    ...run --version push_fr3/model_mismatch/mass_truth_10hz_v1
    ...run --version <version> --mode quick --seed 3
    ...run --version <version> --algo ps --arm nominal --arm mass_2.0

where ``...run`` is::

    uv run python -m experiments.model_mismatch.run
"""

from __future__ import annotations

import argparse
import gc
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import warp as wp

from bampc.dr import DomainRandomizer
from bampc.planner import build_planner
from bampc.rollout import WarpRolloutEngine
from bampc.task.base import quat_mul
from experiments.common.dr.run_dir import (
    make_run_dir,
    write_csv,
    write_manifest,
)
from experiments.common.dr.truth import (
    ee_block_geom_ids,
    make_truth,
    peg_wall_geom_ids,
    plate_block_geom_ids,
)
from experiments.model_mismatch.episode import (
    run_mismatch_episode,
    truth_stride,
)
from experiments.model_mismatch.harness import (
    MODES,
    RunConfig,
    arm_spec,
    load_run_config,
    peg_grid_variants,
    run_variants,
    truth_grid_spec,
    truth_grid_variants,
)

EXPERIMENTS_ROOT = Path(__file__).resolve().parents[1]


class Stack:
    """Everything one variant needs, built once and reused across episodes.

    Graph capture costs seconds, so the engine (with the arm's randomizer),
    planner and truth are built once per variant and only reseeded per episode.
    """

    def __init__(self, cfg: RunConfig, arm_name: str) -> None:
        """Build the task, engine+randomizer, planner and (nominal) truth."""
        r, s = cfg.engine_shape(arm_name)
        # The plan task is pristine (only CpuTruth mutates a model), so the DR
        # spec's base values are read straight off it -- no compounding.
        self.task = cfg.make_prediction_task()
        # bias_target picks which role the arm's spec lands on. Default
        # "prediction" is every existing config's behaviour (planner wrong,
        # truth nominal); "truth" flips it (planner nominal, truth wrong) for
        # parameters truth itself is uncertain about, e.g. a real friction
        # coefficient; "truth_grid" does both at once -- truth pinned at this
        # variant's true_bias_mult, prediction biased relative to THAT (not
        # the global nominal) by the arm's own multiplier.
        if cfg.bias_target == "prediction":
            self.spec = arm_spec(arm_name, cfg, self.task)
        elif cfg.bias_target == "truth_grid":
            self.spec, _ = truth_grid_spec(
                arm_name, cfg.bias_params[0], self.task,
                cfg.true_bias_mult, cfg.bias_params,
            )
        else:
            self.spec = {}
        self.engine = WarpRolloutEngine(
            self.task,
            num_samples=s,
            num_randomizations=r,
            randomizer=DomainRandomizer(self.task, r, self.spec),
        )
        # The config names one sample count; each arm spends the shared budget
        # differently, so the planner is built at the arm's own S and that is
        # what resolved.json records.
        self.planner_config = replace(cfg.planner, num_samples=s)
        self.planner = build_planner(
            self.planner_config, self.task, self.engine
        )
        self.truth_task = cfg.make_truth_task()
        # Peg-FR3: the truth carries the REAL grasp/mount offset (the thing the
        # arms hedge over). Baked onto a fresh truth model, so it never
        # compounds.
        if cfg.task == "peg_fr3" and cfg.true_mount_offset != 0.0:
            bid = self.truth_task.mj_model.body("peg_body").id
            self.truth_task.mj_model.body_pos[bid, 0] += cfg.true_mount_offset
        # Truth-side tilt: the rotational twin of the offset write above,
        # composed onto body_quat (never overwritten -- a quaternion can't
        # be written one component at a time and stay a rotation) via the
        # same right-multiply-in-local-frame convention the DR randomizer's
        # own _QuatTarget uses (bampc/dr/randomizer.py).
        if cfg.task == "peg_fr3" and cfg.true_mount_tilt != 0.0:
            bid = self.truth_task.mj_model.body("peg_body").id
            half = 0.5 * cfg.true_mount_tilt
            q_delta = (
                np.array([np.cos(half), np.sin(half), 0.0, 0.0])
                if cfg.true_mount_tilt_axis == "x"
                else np.array([np.cos(half), 0.0, np.sin(half), 0.0])
            )
            q_before = np.array(self.truth_task.mj_model.body_quat[bid])
            q_after = quat_mul(q_before, q_delta)
            self.truth_task.mj_model.body_quat[bid] = q_after
            # A mistyped guard/field/body-id here would let a whole tilt
            # sweep complete and report plausible success numbers while
            # testing zero true tilt -- read the write back and fail loudly
            # rather than trusting it happened.
            assert not np.allclose(q_after, q_before, atol=1e-9), (
                "tilt injection was a no-op: body_quat unchanged for "
                f"true_mount_tilt={cfg.true_mount_tilt!r} axis="
                f"{cfg.true_mount_tilt_axis!r}"
            )
        # Push-FR3 tags pusher-tip vs block; Balance-FR3 plate vs block;
        # Peg-FR3 peg vs socket walls.
        contact_ids = {
            "balance_fr3": plate_block_geom_ids,
            "peg_fr3": peg_wall_geom_ids,
        }.get(cfg.task, ee_block_geom_ids)
        if cfg.bias_target == "truth":
            truth_spec = arm_spec(arm_name, cfg, self.truth_task)
        elif cfg.bias_target == "truth_grid":
            _, truth_spec = truth_grid_spec(
                arm_name, cfg.bias_params[0], self.truth_task,
                cfg.true_bias_mult, cfg.bias_params,
            )
        else:
            truth_spec = {}
        self.truth = make_truth(
            cfg.truth_backend,
            self.truth_task,
            spec=truth_spec,
            contact_pair=contact_ids(self.truth_task.mj_model),
        )
        self.shape = (r, s)

    def close(self) -> None:
        """Release device buffers and captured graphs before the next build."""
        del self.planner, self.engine, self.truth
        del self.task, self.truth_task
        gc.collect()
        wp.synchronize()


def run_variant(cfg: RunConfig, config_dir: Path, subdir: str) -> None:
    """Run every seed x repeat for one (algo, arm) condition."""
    arm_name = cfg.arm
    stack = Stack(cfg, arm_name)
    r, s = stack.shape
    dt_pred, dt_truth = stack.task.dt, stack.truth_task.dt
    steps_pred, num_replans = cfg.replan_counts(dt_pred)

    run_dir = make_run_dir(
        config_dir,
        replace(cfg, planner=stack.planner_config),
        sweep={
            "algo": cfg.planner.algo,
            "arm": arm_name,
            "num_randomizations": r,
            "num_samples": s,
            "nworld": r * s,
            # The RESOLVED DR spec (absolute magnitudes / ranges), so a results
            # dir records the physics that produced it, not just "mass_0.25".
            "dr_spec": stack.spec,
            "seeds": list(range(cfg.seed_offset,
                                cfg.seed_offset + cfg.num_seeds)),
        },
        subdir=subdir,
    )
    print(
        f"\n{'=' * 70}\n{cfg.planner.algo}/{arm_name}  R={r} S={s} "
        f"(nworld={r * s})  {cfg.plan_freq_hz:g}Hz  "
        f"dt_pred={dt_pred:g} dt_truth={dt_truth:g}\n{'=' * 70}"
    )

    rows: list[dict] = []
    episodes: list[dict] = []
    bank = cfg.bank
    for seed in range(cfg.seed_offset, cfg.seed_offset + cfg.num_seeds):
        scenario = bank[seed]
        for repeat in range(cfg.repeats):
            t_start = time.perf_counter()
            result = run_mismatch_episode(
                cfg, stack.task, stack.planner, stack.truth,
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
                # condition; int rather than bool for the same
                # csv/genfromtxt round-trip reason as tracking.csv's
                # had_contact/task_success.
                "success": (
                    None if result.success is None else int(result.success)
                ),
            })
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
            "plan_freq_hz": cfg.plan_freq_hz,
            "replan_dt": 1.0 / cfg.plan_freq_hz,
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
        "push_fr3/model_mismatch/mass_truth_10hz_v1.",
    )
    parser.add_argument(
        "--mode", choices=["full", "smoke", "quick"], default="full",
        help="full = the real sweep; quick = 1 seed x 1 trial at FULL episode "
        "length; smoke = 1 seed x 1 trial at 2 s, a shape check only.",
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
        help="Skip variants that already have a manifest.json (written last, "
        "so it marks a COMPLETE variant).",
    )
    args = parser.parse_args()

    config_dir = EXPERIMENTS_ROOT / args.version
    cfg = load_run_config(config_dir / "config.yaml")
    if args.mode == "quick":
        cfg = MODES["quick"](cfg, args.seed)
    elif args.mode == "smoke":
        cfg = MODES["smoke"](cfg)

    if cfg.true_bias_grid:
        variants = truth_grid_variants(cfg)
    elif cfg.clearance_grid:
        variants = peg_grid_variants(cfg)
    else:
        variants = run_variants(cfg)
    for parts, variant in variants:
        if args.algo and variant.planner.algo not in args.algo:
            continue
        if args.arm and variant.arm not in args.arm:
            continue
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
