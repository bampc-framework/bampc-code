"""Headless Push-T-FR3 rollout stills for paper figures.

FR3-armed counterpart of ``push_rollout.py`` (see its docstring for the
overall idea and ``_common.py`` for the shared rollout/capture machinery).
Runs closed-loop Push-FR3 rollouts from a handful of small, visually-distinct
perturbations of one scenario-bank start pose (drawn from the same
state-uncertainty noise machinery as ``examples/state_uncertainty/``), and
saves evenly-spaced PNG stills from the ``"main"`` camera already used for
Push-FR3 videos (``models/push_fr3/<manipulation>/scene.xml``). Each still
shows a handful of the planner's sampled control trajectories as faint
lines plus a translucent ghost of the block at each sample's predicted
final pose.

Defaults to the T-shaped scenario bank (``push_fr3_t``), task-space (IK)
sampling, and a 3-DOF constrained block, matching
``examples/simple/push_fr3.py``'s defaults.

Run::

    uv run python scripts/figures/push_fr3_rollout.py ps
    uv run python scripts/figures/push_fr3_rollout.py ps --manipulation free
"""

from __future__ import annotations

import argparse
from pathlib import Path

from bampc.config import planner as planner_profiles
from bampc.config import reward as reward_profiles
from bampc.config import scenarios
from bampc.config.numerics import load as load_numerics
from bampc.planner import PlannerConfig, build_planner
from bampc.rollout import WarpRolloutEngine
from bampc.sim.viewer import build_ghosts, select_trace_idxs
from bampc.task.push_fr3 import PushFr3
from scripts.figures._common import (
    FIGURE_DIR,
    GHOST_RGBA,
    draw_conditions,
    run_condition,
)

_PLAN_FREQ = 10  # Hz, matches examples/simple/push_fr3.py


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Headless Push-T-FR3 rollout stills for paper figures.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "algorithm", nargs="?", default="ps", choices=["mppi", "ps", "cem"],
        help="Sampling algorithm.",
    )
    parser.add_argument(
        "--sampling", default="task", choices=["task", "joint"],
        help="EE twist via IK (task) or direct joint velocities (joint).",
    )
    parser.add_argument(
        "--manipulation", default="joint", choices=["joint", "free"],
        help="3-DOF constrained block (joint) or 6-DOF free block (free).",
    )
    parser.add_argument(
        "--bank", default="push_fr3_t",
        choices=["push_fr3_t", "push_fr3_l"],
        help="Scenario bank (block shape).",
    )
    parser.add_argument(
        "--scenario", type=int, default=0,
        help="Scenario-bank index for the nominal start pose.",
    )
    parser.add_argument(
        "--num-conditions", type=int, default=3,
        help="Number of perturbed initial T poses to render.",
    )
    parser.add_argument(
        "--pos-std", type=float, default=0.03,
        help="Position std (m) of the initial-condition draw.",
    )
    parser.add_argument(
        "--rot-std", type=float, default=0.2,
        help="Yaw std (rad) of the initial-condition draw.",
    )
    parser.add_argument(
        "--condition-seed", type=int, default=0,
        help="Seed for drawing the initial conditions.",
    )
    parser.add_argument(
        "--num-samples-shown", type=int, default=6,
        help="Sampled-rollout lines + endpoint ghosts drawn per still.",
    )
    parser.add_argument(
        "--num-frames", type=int, default=5,
        help="Evenly-spaced stills saved per rollout.",
    )
    parser.add_argument(
        "--duration", type=float, default=10.0,
        help="Rollout length (sim seconds).",
    )
    parser.add_argument(
        "--camera", default="main",
        help="Camera name from the model XML.",
    )
    parser.add_argument(
        "--size", type=int, nargs=2, default=(480, 640), metavar=("H", "W"),
        help="Rendered frame size.",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=FIGURE_DIR / "push_fr3_rollout",
        help="Output directory.",
    )
    return parser.parse_args()


def main() -> None:
    """Build one Push-FR3 task/planner and render stills per condition."""
    args = _parse_args()

    numerics = load_numerics(f"fr3_{args.manipulation}")
    bank = scenarios.load(args.bank)
    if bank.task != "push_fr3":
        raise ValueError(f"--bank {args.bank!r} is not a push_fr3 bank")
    scenario = bank[args.scenario]

    planner_kw = planner_profiles.load(
        "push_fr3", sampling=args.sampling, manipulation=args.manipulation
    )
    reward_kw = reward_profiles.load(
        "push_fr3", sampling=args.sampling, manipulation=args.manipulation
    )
    task = PushFr3(
        sampling_space=args.sampling, manipulation_type=args.manipulation,
        shape=bank.shape, scale=bank.scale, trace_sites=["ee_site"],
        goal_drift=bank.goal_drift, model_config=numerics, **reward_kw,
    )
    planner_cfg = PlannerConfig(algo=args.algorithm, **planner_kw)
    engine = WarpRolloutEngine(
        task, num_samples=planner_cfg.num_samples, num_randomizations=1,
        record_traces=True,
    )
    planner = build_planner(planner_cfg, task, engine)

    nominal = scenarios.pose(bank, task, scenario)
    nominal.time = scenario.drift_phase_time
    qpos_cloud = draw_conditions(
        task, nominal, args.num_conditions, args.pos_std, args.rot_std,
        args.condition_seed,
    )

    idxs = select_trace_idxs(planner.num_samples, args.num_samples_shown, None)
    num_sites = len(task.trace_site_ids)
    vmodel, built = build_ghosts(
        task.mj_spec, task.endpoint_bodies,
        {"endpoint": (1, len(idxs), GHOST_RGBA)},
    )
    ghost_ids, _ = built["endpoint"]
    n_base_mocap = task.mj_model.nmocap
    if task.model_config is not None:
        task.model_config.apply_to(vmodel)

    print(f"conditions (pos_std={args.pos_std}, rot_std={args.rot_std}):")
    for i in range(args.num_conditions):
        dx = qpos_cloud[i] - nominal.qpos
        print(f"  condition {i}: dqpos (block joints) = {dx}")

    for i in range(args.num_conditions):
        out_dir = args.out_dir / f"condition_{i}"
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"running condition {i} -> {out_dir}")
        run_condition(
            planner, vmodel, n_base_mocap, qpos_cloud[i], nominal.qvel,
            ghost_ids, idxs, num_sites,
            plan_freq=_PLAN_FREQ, duration=args.duration,
            num_frames=args.num_frames, camera=args.camera, size=args.size,
            out_dir=out_dir,
        )

    print(f"done: {args.num_conditions} conditions x {args.num_frames} "
          f"frames under {args.out_dir}")


if __name__ == "__main__":
    main()
