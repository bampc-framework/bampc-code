"""How far do identical runs diverge? An MJWarp non-determinism probe.

Sweeps are not bit-reproducible -- same seed, same config, and the trajectory
still reaches **~9 cm apart by 10 s** -- blamed on float32 atomics in the
contact solver reordering. That claim is load-bearing (it is why an 8-seed
mean of a trajectory metric moves 3-20% between identical runs), so this probe
isolates it.

``configs/scenarios/*.yaml`` pins the start state and the goal drift, so it
can be measured directly: hold everything fixed and run one
scenario N times. Any difference between repeats is the simulator and
nothing else.

Run (from the repo root)::

    uv run python scripts/probing/determinism_check.py
    uv run python scripts/probing/determinism_check.py --truth cpu
    uv run python scripts/probing/determinism_check.py --bank push --index 3
    uv run python scripts/probing/determinism_check.py --samples 1024

Two sources of divergence, separated by ``--truth``:

* ``warp`` -- both compounded. The MJWarp truth steps differently *and* the
  planner's MJWarp rollouts return different costs, so it picks different
  controls. This is what a real closed-loop run does.
* ``cpu`` -- the truth is float64 CPU MuJoCo, which is deterministic, so any
  divergence left is purely the planner choosing different controls.

The difference between the two arms is the physics contribution.

Every repeat rebuilds the engine, planner and truth from scratch. Reusing one
captured CUDA graph would replay the same launches over the same memory, which
could be more repeatable than the separate-runs case the ~9 cm figure came
from -- and would then understate the spread. It costs ~10x the wall time.

``--samples`` is the knob worth sweeping: more worlds means more atomic
contention in the contact solver, so the spread should scale with it.
"""

from __future__ import annotations

import argparse
import csv
import gc
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
import warp as wp

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bampc import OUTPUT_DIR  # noqa: E402
from bampc.config import scenarios  # noqa: E402
from bampc.config.numerics import load as load_numerics  # noqa: E402
from bampc.planner import PlannerConfig, build_planner  # noqa: E402
from bampc.rollout import WarpRolloutEngine  # noqa: E402
from bampc.task.balance import Balance  # noqa: E402
from bampc.task.push import Push  # noqa: E402
from bampc.task.push_fr3 import PushFr3  # noqa: E402
from experiments.common.dr.episode import truth_stride  # noqa: E402
from experiments.common.dr.goal import GoalDriver, snapshot  # noqa: E402
from experiments.common.dr.pose import body_poses  # noqa: E402
from experiments.common.dr.truth import make_truth  # noqa: E402

# Contact-mode names, indexed by the bitmask. This is the SAME encoding as
# ``contact_palette`` in bampc/sim/viewer.py -- both read the order a
# task declares in ``Task.contact_probes`` (bit 0 pusher-object, bit 1
# object-table), so the ``contact_mode`` column below indexes that palette
# directly and a plot of it matches the viewer's belief-ghost colours.
MODE_NAMES = ("none", "pusher", "table", "both")

# Cost weights for Push-FR3's task-space sampling -- the library defaults
# match no tuned setup, and a controller
# that never touches the block would make this probe measure nothing.
_FR3_TASK_WEIGHTS = {
    "max_lin_vel": 0.2,
    "w_pos": 35.0,
    "w_orient": 3.0,
    "w_attract": 0.05,
    "w_align": 0.0,
    "w_safety": 1.0,
    "safety_thresh": 0.5,
    "w_ee_orient": 0.0,
    "w_ee_height": 0.0,
    "w_arm_home": 0.0,
    "terminal_scale": 5.0,
}

# Planner per task, from configs/planner/*.yaml with the sweep configs'
# overrides applied. `--samples`/`--algo` override these.
_PLANNERS = {
    "push": dict(
        algo="ps", risk="average", noise_level=0.2, num_samples=128,
        plan_horizon=0.5, num_knots=5, spline_type="zero",
    ),
    "balance": dict(
        algo="ps", risk="average", noise_level=0.2, num_samples=128,
        plan_horizon=0.4, num_knots=6, spline_type="cubic",
    ),
    "push_fr3": dict(
        algo="ps", risk="average", noise_level=0.3, num_samples=512,
        plan_horizon=0.6, num_knots=5, spline_type="cubic",
    ),
}


def build_task(bank: scenarios.ScenarioBank, manipulation: str):
    """The task this bank describes, at the bank's own shape and scale."""
    kw = dict(shape=bank.shape, scale=bank.scale, goal_drift=bank.goal_drift)
    if bank.task == "push":
        return Push(model_config=load_numerics("push"), **kw)
    if bank.task == "balance":
        return Balance(
            goal_xy=bank.goal_xy,
            model_config=load_numerics("balance"),
            **kw,
        )
    return PushFr3(
        sampling_space="task",
        manipulation_type=manipulation,
        model_config=load_numerics(f"fr3_{manipulation}"),
        **_FR3_TASK_WEIGHTS,
        **kw,
    )


def replan_counts(dt: float, plan_freq: float, duration: float):
    """``(steps_per_replan, num_replans)`` at ``dt``.

    The replan interval is a duration, so each role derives its own step
    count from its own timestep -- the same rule ``RunConfig.replan_counts``
    follows.
    """
    period = 1.0 / plan_freq
    return (
        max(int(round(period / dt)), 1),
        max(int(round(duration / period)), 1),
    )


def contact_mask(truth, probes) -> int:
    """Bitmask of which ``contact_probes`` are touching right now.

    Reads the truth's own contact pool, not ``engine.contact_modes()`` --
    that one reports the *planner's* initial state. Same access pattern as
    ``WarpTruth.has_contact``, generalized from one pair to every probe.

    The result indexes ``bampc.sim.viewer.contact_palette``; see
    ``MODE_NAMES``.
    """
    if hasattr(truth, "engine"):  # WarpTruth
        n = int(truth.engine.d.nacon.numpy()[0])
        if n == 0:
            return 0
        geoms = truth.engine.d.contact.geom.numpy()[:n]
        g1, g2 = geoms[:, 0], geoms[:, 1]
    else:  # CpuTruth
        n = truth.data.ncon
        if n == 0:
            return 0
        g1 = truth.data.contact.geom1[:n]
        g2 = truth.data.contact.geom2[:n]

    mask = 0
    for probe in probes:
        a, b = list(probe.geoms_a), list(probe.geoms_b)
        hit = (np.isin(g1, a) & np.isin(g2, b)) | (
            np.isin(g2, a) & np.isin(g1, b)
        )
        if bool(np.any(hit)):
            mask |= 1 << probe.bit
    return mask


def run_once(args, bank, scenario):
    """One full episode with everything rebuilt. Returns per-truth-step logs.

    Returns:
        ``(qpos, times, modes)`` -- ``(T, nq)``, ``(T,)``, ``(T,)``.
    """
    task = build_task(bank, args.manipulation)
    engine = WarpRolloutEngine(
        task,
        num_samples=args.samples,
        num_randomizations=1,
        randomizer=None,  # no DR: this probe varies the simulator, nothing else
    )
    cfg = PlannerConfig(
        **{**_PLANNERS[bank.task], "algo": args.algo,
           "num_samples": args.samples}
    )
    # Rebuilt per repeat, so `seed` gives a fresh identical RNG every time --
    # no manual planner.rng reset is needed (episode.py only does that
    # because it reuses one planner across seeds).
    planner = build_planner(cfg, task, engine, seed=args.seed)

    truth_task = build_task(bank, args.manipulation)
    truth = make_truth(args.truth, truth_task, {}, contact_pair=None)
    truth.reset(scenarios.pose(bank, truth_task, scenario))

    dt_pred, dt_truth = task.dt, truth.dt
    k = truth_stride(dt_pred, dt_truth)
    steps_pred, num_replans = replan_counts(
        dt_pred, args.plan_freq, args.duration
    )
    steps_truth = steps_pred * k

    driver = GoalDriver(task, scenario.drift_phase_time)
    probes = task.contact_probes
    params = planner.init_params()

    qpos_log, time_log, mode_log = [], [], []
    for _ in range(num_replans):
        qpos, qvel, t0 = truth.state()
        params, _ = planner.optimize(
            snapshot(driver, qpos, qvel, t0), params
        )
        for _ in range(steps_truth):
            _, _, t_now = truth.state()
            truth.step(planner.get_action(params, t_now))
            q, _, t_after = truth.state()
            qpos_log.append(q)
            time_log.append(t_after)
            mode_log.append(contact_mask(truth, probes))

    nq = task.mj_model.nq
    out = (
        np.stack(qpos_log)[:, :nq],
        np.asarray(time_log),
        np.asarray(mode_log, np.uint8),
    )

    # Release the graph + device buffers before the next repeat builds: a
    # reference cycle through the captured graph outlives refcounting.
    del planner, engine, truth, task, truth_task
    gc.collect()
    wp.synchronize()
    return out


def _offdiag_max(d: np.ndarray) -> np.ndarray:
    """Max over distinct pairs. ``(R, R, T)`` -> ``(T,)``."""
    d = d.copy()
    d[np.arange(d.shape[0]), np.arange(d.shape[0])] = 0.0
    return d.max(axis=(0, 1))


def pairwise_spread(x: np.ndarray) -> np.ndarray:
    """Max pairwise distance at each step. ``(R, T, 3)`` -> ``(T,)``."""
    return _offdiag_max(np.linalg.norm(x[:, None] - x[None, :], axis=-1))


def pairwise_angle(q: np.ndarray) -> np.ndarray:
    """Max pairwise quaternion geodesic angle. ``(R, T, 4)`` -> ``(T,)``.

    Uses ``4*atan2(|q1 - s*q2|, |q1 + s*q2|)`` rather than the textbook
    ``2*arccos(|dot|)``. The arccos form is useless at exactly the scale
    this probe cares about: two *identical* quaternions dot to ``|q|^2``,
    which is 1 only to within their normalization error, and arccos is
    vertical at 1 -- so a 1-ulp shortfall reads as ~3e-8 rad and becomes a
    floor that hides the first tens of milliseconds of real divergence.
    The atan2 form returns exactly 0 for identical inputs.
    """
    q1, q2 = q[:, None], q[None, :]
    s = np.where(np.sum(q1 * q2, axis=-1) < 0.0, -1.0, 1.0)[..., None]
    dm = np.linalg.norm(q1 - s * q2, axis=-1)
    dp = np.linalg.norm(q1 + s * q2, axis=-1)
    return _offdiag_max(4.0 * np.arctan2(dm, dp))


def disagreement(modes: np.ndarray) -> np.ndarray:
    """Repeats differing from the modal contact mode. ``(R, T)`` -> ``(T,)``."""
    counts = np.stack([(modes == m).sum(axis=0) for m in range(4)])
    return modes.shape[0] - counts.max(axis=0)


def first_above(values: np.ndarray, times: np.ndarray, thresh: float):
    """Time at which ``values`` first exceeds ``thresh`` (``None`` if never)."""
    hit = np.flatnonzero(values > thresh)
    return float(times[hit[0]]) if hit.size else None


def write_csv(path: Path, times, qpos_all, modes, xpos, xquat) -> None:
    """Long-format dump: one row per (repeat, step)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([
            "repeat", "step", "time", "contact_mode",
            "block_x", "block_y", "block_z",
            "block_qw", "block_qx", "block_qy", "block_qz",
        ])
        for r in range(qpos_all.shape[0]):
            for t in range(qpos_all.shape[1]):
                w.writerow([
                    r, t, f"{times[t]:.6f}", int(modes[r, t]),
                    *(f"{v:.9e}" for v in xpos[r, t]),
                    *(f"{v:.9e}" for v in xquat[r, t]),
                ])


def parse_args() -> argparse.Namespace:
    """CLI, with the per-task planner defaults filled in."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bank", default="push_fr3", choices=scenarios.list_banks())
    p.add_argument("--index", type=int, default=15)
    p.add_argument("--manipulation", default="joint", choices=["joint", "free"])
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--truth", default="warp", choices=["warp", "cpu"])
    p.add_argument("--duration", type=float, default=10.0)
    p.add_argument("--plan-freq", type=float, default=10.0)
    p.add_argument("--samples", type=int, default=None)
    p.add_argument("--algo", default=None, choices=["ps", "mppi", "cem"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--out", default=None, help="CSV path; default under output/results/"
    )
    args = p.parse_args()
    defaults = _PLANNERS[scenarios.load(args.bank).task]
    if args.samples is None:
        args.samples = defaults["num_samples"]
    if args.algo is None:
        args.algo = defaults["algo"]
    return args


def report(args, task, times, modes, pos_spread, rot_spread) -> None:
    """Print the sampled divergence table and the summary."""
    disagree = disagreement(modes)
    print(f"\n  {'t[s]':>6}  {'pos_spread[m]':>13}  {'rot_spread[rad]':>15}"
          f"  {'contact_disagree':>16}")
    for i in np.linspace(0, len(times) - 1, 12).astype(int):
        print(f"  {times[i]:6.2f}  {pos_spread[i]:13.3e}  "
              f"{rot_spread[i]:15.3e}  {disagree[i]:>10d}/{args.repeats}")

    t_split = first_above(pos_spread, times, 1e-9)
    t_contact = first_above(disagree.astype(float), times, 0.5)
    print("\nfirst state divergence   t = "
          + ("never" if t_split is None else f"{t_split:.3f} s"))
    print("first contact-mode split t = "
          + ("never" if t_contact is None else f"{t_contact:.3f} s"))

    occ = [float((modes == m).mean()) for m in range(4)]
    print("mode occupancy (mean over repeats): "
          + " | ".join(f"{m} {MODE_NAMES[m]} {occ[m]:.2f}" for m in range(4)))
    for probe in task.contact_probes:
        frac = ((modes >> probe.bit) & 1).mean(axis=1)
        print(f"contact fraction {probe.name}: "
              f"{frac.min():.2f} .. {frac.max():.2f} across repeats")
    print(f"final block pos spread: {pos_spread[-1] * 100:.2f} cm "
          f"({pos_spread[-1]:.3e} m)")


def main() -> None:
    """Run the repeats, report the spread, write the CSV."""
    args = parse_args()
    bank = scenarios.load(args.bank)
    scenario = bank[args.index]

    manip = f" manip={args.manipulation}" if bank.task == "push_fr3" else ""
    print(
        f"bank={args.bank!r} idx={args.index}{manip} truth={args.truth} "
        f"repeats={args.repeats} S={args.samples} algo={args.algo} "
        f"seed={args.seed} duration={args.duration:g}s"
    )
    print(f"start {scenario.start}  phase_time={scenario.drift_phase_time:g}s")

    runs = []
    for r in range(args.repeats):
        t0 = time.perf_counter()
        runs.append(run_once(args, bank, scenario))
        print(f"  repeat {r}: {time.perf_counter() - t0:.1f}s")

    lengths = {len(t) for _, t, _ in runs}
    if len(lengths) != 1:
        raise RuntimeError(f"repeats produced different lengths: {lengths}")

    qpos_all = np.stack([q for q, _, _ in runs])          # (R, T, nq)
    times = runs[0][1]                                     # (T,)
    modes = np.stack([m for _, _, m in runs])              # (R, T)

    # Kinematics replayed from qpos rather than read off xpos -- after a step
    # returns, d.qpos is the new configuration but d.xpos still holds the
    # previous one (see experiments/common/dr/pose.py).
    task = build_task(bank, args.manipulation)
    scratch = mujoco.MjData(task.mj_model)
    bid = task.mj_model.body("block").id
    xpos, xquat = body_poses(task.mj_model, scratch, qpos_all, bid)

    pos_spread = pairwise_spread(xpos)
    rot_spread = pairwise_angle(xquat)
    report(args, task, times, modes, pos_spread, rot_spread)

    # Manipulation is in the name because it is a different model, not a
    # different setting: without it a `--manipulation free` run silently
    # overwrites the joint results it should be compared against.
    tag = f"{args.manipulation}_" if bank.task == "push_fr3" else ""
    out = Path(args.out) if args.out else (
        OUTPUT_DIR / "results" / f"{args.bank}_determinism"
        / f"determinism_{tag}{args.truth}_s{args.samples}.csv"
    )
    write_csv(out, times, qpos_all, modes, xpos, xquat)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
