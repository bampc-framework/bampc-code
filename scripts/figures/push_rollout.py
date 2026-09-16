"""Headless Push-T rollout stills for paper figures.

Runs closed-loop Push rollouts from a handful of small, visually-distinct
perturbations of one scenario-bank start pose (drawn from the same
state-uncertainty noise machinery as ``examples/state_uncertainty/``, not
picked from far-apart scenario-bank entries), and saves evenly-spaced PNG
stills from the ``"main"`` camera already used for Push-T videos
(``models/push/scene.xml``). Each still shows a handful of the planner's
sampled control trajectories as faint lines plus a translucent ghost of the
block at each sample's predicted final pose -- the same visualization
``run_interactive`` draws live, captured headless instead of recorded as
video. See ``push_fr3_rollout.py`` for the FR3-armed counterpart; the shared
rollout/capture machinery lives in ``_common.py``.

Run::

    uv run python scripts/figures/push_rollout.py ps
    uv run python scripts/figures/push_rollout.py ps --pos-std 0.02
"""

from __future__ import annotations

import argparse
from pathlib import Path

from bampc.config import planner as planner_profiles
from bampc.config import scenarios
from bampc.config.numerics import load as load_numerics
from bampc.planner import PlannerConfig, build_planner
from bampc.rollout import WarpRolloutEngine
from bampc.sim.viewer import build_ghosts, select_trace_idxs
from bampc.task.push import Push
from scripts.figures._common import (
    FIGURE_DIR,
    GHOST_RGBA,
    draw_conditions,
    run_condition,
)

_PLAN_FREQ = 30  # Hz, matches examples/simple/push.py


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Headless Push-T rollout stills for paper figures.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "algorithm", nargs="?", default="ps", choices=["mppi", "ps", "cem"],
        help="Sampling algorithm.",
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
        "--out-dir", type=Path, default=FIGURE_DIR / "push_rollout",
        help="Output directory.",
    )
    return parser.parse_args()


def main() -> None:
    """Build one Push task/planner and render stills per initial condition."""
    args = _parse_args()

    numerics = load_numerics("push")
    bank = scenarios.load("push")
    scenario = bank[args.scenario]

    task = Push(
        shape=bank.shape, scale=bank.scale, goal_drift=bank.goal_drift,
        model_config=numerics,
    )
    planner_kw = planner_profiles.load("push", algo=args.algorithm)
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
        print(f"  condition {i}: dqpos (block/pusher joints) = {dx}")

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
