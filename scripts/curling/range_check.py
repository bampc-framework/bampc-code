"""How far does the curling puck slide, and is that number trustworthy?

The lane's contact model degrades **silently**. A flat cylinder resting on the
lane makes a 5-point coplanar contact; past a certain sliding speed the solver
stops resolving it, the puck sinks ~16 mm into the lane, and it stops
decelerating correctly. Nothing raises -- the rollout is just quietly wrong,
which is the worst failure mode for a task whose entire objective is where the
puck comes to rest.

So the physics has a *verified band*, not just a configuration, and this script
is what verifies it. Ground truth is the analytic Coulomb slide: a puck
launched at ``v0`` travels ``v0^2 / (2*mu*g)`` and stops after ``v0 / (mu*g)``.
A row is ok when the measured range matches that, the puck actually comes to
rest, and it never sank more than 5 mm.

Run (from the repo root)::

    uv run python scripts/curling/range_check.py
    uv run python scripts/curling/range_check.py --shape sphere
    uv run python scripts/curling/range_check.py --dt 0.02 --dt 0.01

Re-run it before raising ``max_lin_vel`` in ``configs/reward/curling.yaml``,
after changing ``timestep``/``cone`` in ``configs/numerics/curling.yaml``, or
after touching the ``stone``/``wood`` friction in the scene XML. The findings
those numbers came from are in ``configs/numerics/curling.yaml``'s header:
briefly, the elliptic cone cannot solve this contact at all, and lowering the
ice friction makes things worse rather than better, because the failure needs
sliding time to develop.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bampc.config.numerics import load as load_numerics  # noqa: E402
from bampc.task.curling_fr3 import CurlingFr3  # noqa: E402

SETTLE_T = 0.5  # s of zero control, to establish the lane contact first
SINK_TOL = 5e-3  # m; more than this and the contact has come apart
RANGE_TOL = 0.06  # m; measured-vs-analytic range agreement
REST_SPEED = 0.01  # m/s


def slide(
    task, v0: float, dt: float | None
) -> tuple[float, float | None, float]:
    """Settle, launch at ``v0`` along +x, return (range, t_stop, sink)."""
    m = task.mj_model
    if dt is not None:
        m.opt.timestep = dt
    adr = task._block_free_adr
    vadr = task._block_dofadr

    d = mujoco.MjData(m)
    kid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
    d.qpos[:] = m.key_qpos[kid]
    for _ in range(int(round(SETTLE_T / m.opt.timestep))):
        mujoco.mj_step(m, d)

    z_rest = float(d.qpos[adr + 2])
    d.qvel[:] = 0.0
    d.qvel[vadr] = v0
    x0, z_min, t0, t_stop = float(d.qpos[adr]), z_rest, d.time, None
    for _ in range(int(round(10.0 / m.opt.timestep))):
        mujoco.mj_step(m, d)
        z_min = min(z_min, float(d.qpos[adr + 2]))
        if np.linalg.norm(d.qvel[vadr : vadr + 3]) < REST_SPEED:
            t_stop = d.time - t0
            break
    return float(d.qpos[adr]) - x0, t_stop, z_rest - z_min


def main() -> None:
    """Measure the range curve and print it against the analytic slide."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", default="circle")
    parser.add_argument(
        "--dt",
        type=float,
        action="append",
        help="timestep(s) to test; default: the curling numerics profile's",
    )
    parser.add_argument(
        "--v0",
        type=float,
        nargs="+",
        default=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1],
        help="launch speeds to test (m/s)",
    )
    args = parser.parse_args()

    task = CurlingFr3(shape=args.shape, model_config=load_numerics("curling"))
    m = task.mj_model
    # MuJoCo combines a contacting pair's friction elementwise-max, so the
    # effective mu is the larger of the puck's and the lane's.
    puck_gid = [
        g for g in range(m.ngeom)
        if (m.geom(g).name or "").startswith("block_")
    ]
    mu = float(
        max([m.geom_friction[m.geom("ground").id, 0]]
            + [m.geom_friction[g, 0] for g in puck_gid])
    )
    a = mu * 9.81
    cone = mujoco.mjtCone(m.opt.cone).name.removeprefix("mjCONE_").lower()
    print(f"shape={args.shape}  mu={mu:.3f}  decel={a:.3f} m/s^2  cone={cone}")

    worst_ok = None
    for dt in args.dt or [None]:
        eff = dt if dt is not None else m.opt.timestep
        print(f"\n  dt={eff:.4f}")
        print("    v0     range    analytic   t_stop    sink")
        for v0 in args.v0:
            rng, ts, sink = slide(task, v0, dt)
            exact = v0 * v0 / (2.0 * a)
            ok = (
                ts is not None
                and sink < SINK_TOL
                and abs(rng - exact) < RANGE_TOL
            )
            if ok and dt is None:
                worst_ok = v0
            print(f"    {v0:4.1f}  {rng * 100:7.1f}cm {exact * 100:7.1f}cm  "
                  f"{'--' if ts is None else f'{ts:5.2f}s'}  "
                  f"{sink * 1000:6.2f}mm  {'ok' if ok else 'BROKEN'}")

    if worst_ok is not None:
        print(f"\nverified launch-speed ceiling: {worst_ok:.1f} m/s. "
              f"Keep max_lin_vel (configs/reward/curling.yaml) under it.")


if __name__ == "__main__":
    main()
