"""Regenerate the frozen scenario banks in ``configs/scenarios/*.yaml``.

Run with::

    uv run python scripts/probing/make_scenarios.py           # print YAML
    uv run python scripts/probing/make_scenarios.py --check    # validate only

Prints the ``scenarios:`` block for each task; paste it into the bank file
(the surrounding comments are hand-written and worth keeping). ``--check``
re-validates the banks already on disk instead, which is the cheap way to
catch a model change silently breaking a start state.

How the samples are drawn, and why not the way ``random_scenario`` does it:

* **Latin hypercube.** Each scenario owns one stratum per dimension, so 16
  scenarios cover the ranges evenly. Plain uniform draws clump, and a
  farthest-point spread does the opposite -- it piles onto the corners.
* **Clearance, not offset.** The manipulator standoff is bisected with
  ``mj_geomDistance`` to hit a sampled 6-25 mm *surface* gap, in the world
  frame. A fixed radial offset from the block's joint origin does not work:
  the T-block's geometry is anchored well off that origin, and Push's pusher
  body carries an XML ``pos="0 0.1 0.011"``, so ``root_x/root_y`` are not
  world coordinates at all. Measured, that combination starts the pusher
  *inside* the block for ~26% of draws.
* **Validated.** Every scenario is forwarded and checked for contact
  penetration at t=0, then stepped 0.25 s with zero control to confirm it
  does not fall through anything. Push-FR3 is checked against both
  manipulation types and keeps the larger standoff, so one bank serves both.

``mj_geomDistance`` is the bisection's *search* signal, never the pass/fail
one: on the FR3 models it returns exactly 0.0 for some clearly separated box
pairs (measured 0.0 for geoms 14 cm apart), so it cannot distinguish
"touching" from "far". That error is one-directional -- a spurious 0.0 reads
as "still overlapping" against a positive target, so the bisection pushes the
standoff further out and never inward. A few FR3 standoffs therefore overshoot
their sampled gap; none can land inside the block. The verdict comes from
``mj_forward`` contact depth, which is authoritative.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bampc.config import scenarios  # noqa: E402
from bampc.config.numerics import load as load_numerics  # noqa: E402
from bampc.task.balance import Balance  # noqa: E402
from bampc.task.balance_fr3 import BalanceFr3  # noqa: E402
from bampc.task.base import body_geom_ids  # noqa: E402
from bampc.task.curling_fr3 import CurlingFr3  # noqa: E402
from bampc.task.peg_fr3 import PegFr3  # noqa: E402
from bampc.task.push import Push  # noqa: E402
from bampc.task.push_fr3 import PushFr3  # noqa: E402

N = 16
PEN_TOL = 5e-4     # m; initial contact depth past this = interpenetration
SETTLE_T = 0.25    # s of zero-control settling
SETTLE_PEN = 5e-3  # m; soft contact legitimately sinks in while settling
GAP_RANGE = (0.006, 0.025)  # m; sampled manipulator-to-block surface gap

# Curling's bank stem. Not "curling": banks are named per shape, and this is
# both the makers key and what drift_freqs()/make_curling() load.
CURLING_BANK = "curling_circle"

# Peg-FR3's frozen mount-offset range: matches the magnitude examples/*/
# peg_fr3.py's --noise/--true-offset default to, so a scenario's baked-in
# error sits inside the range planners are tuned to hedge over by default.
PEG_LEN = 0.12  # m; same rot-from-pos scaling examples/*/peg_fr3.py uses
MOUNT_POS_RANGE = (-0.004, 0.004)  # m
MOUNT_ROT_RANGE = (-0.004 / PEG_LEN, 0.004 / PEG_LEN)  # rad

# Per-task drift, mirrored in the bank files' ``goal_drift:``.
def drift_freqs(name: str) -> list[float]:
    """The bank's own drift frequencies, for spreading the phase times.

    Read from the bank rather than restated here. These used to be a
    hardcoded table, and it drifted: push_fr3 sat at [0.06, 0.1, 0.07] long
    after its bank moved to [0.03, 0.06] + 0.05 yaw, so the phases were spread
    over half the real period. A copy of a value is a copy that goes stale.
    """
    d = scenarios.load(name).goal_drift
    return [*d.freq_xy, d.yaw_freq]


def worst_pen(m, d) -> float:
    """Deepest contact in ``d`` (0.0 if there are none)."""
    return float(min(d.contact.dist[: d.ncon])) if d.ncon else 0.0


def clearance(m, d, ga, gb) -> float:
    """Smallest surface gap between two geom groups (negative = overlap)."""
    return min(
        mujoco.mj_geomDistance(m, d, a, b, 1.0, None) for a in ga for b in gb
    )


def check(task, md, ga, gb, drift_tol: float) -> tuple:
    """``(ok, init_pen, gap, settle_drift, settle_pen)`` for a posed state."""
    m = task.mj_model
    mujoco.mj_forward(m, md)
    pen0 = worst_pen(m, md)
    gap = clearance(m, md, ga, gb)

    d = mujoco.MjData(m)
    d.qpos[:], d.qvel[:] = md.qpos, md.qvel
    mujoco.mj_forward(m, d)
    bid = m.body("block").id
    start = d.xpos[bid].copy()
    pen = pen0
    for _ in range(int(round(SETTLE_T / m.opt.timestep))):
        d.ctrl[:] = 0.0
        mujoco.mj_step(m, d)
        pen = min(pen, worst_pen(m, d))
    drift = float(np.linalg.norm(d.xpos[bid] - start))
    ok = pen0 > -PEN_TOL and pen > -SETTLE_PEN and drift < drift_tol
    return ok, pen0, gap, drift, pen


def lhs(rng, n: int, ranges) -> np.ndarray:
    """``(n, d, 2)`` Latin-hypercube cell bounds, one cell per scenario."""
    cells = np.empty((n, len(ranges), 2))
    for j, (lo, hi) in enumerate(ranges):
        edges = np.linspace(lo, hi, n + 1)
        order = rng.permutation(n)
        cells[:, j, 0], cells[:, j, 1] = edges[order], edges[order + 1]
    return cells


def bisect(gap_fn, target: float, lo: float, hi: float, iters: int = 36):
    """Smallest standoff in ``[lo, hi]`` whose surface gap reaches ``target``.

    ``lo`` must overlap, otherwise the bisection has nothing to bracket and
    would collapse to zero -- which is how a "standoff" ends up placing the
    pusher on top of the block.
    """
    if gap_fn(lo) >= target:
        raise RuntimeError(
            f"gap({lo}) already >= {target}: lo does not overlap"
        )
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if gap_fn(mid) < target:
            lo = mid
        else:
            hi = mid
    return hi


def r4(v) -> float:
    """Round to 0.1 mm -- the banks are meant to be read by a human."""
    return round(float(v), 4)


def make_push(check_only: bool):
    """Push: pusher on the far side of the block from the origin goal."""
    task = Push(shape="t", scale=1.0, model_config=load_numerics("push"))
    m = task.mj_model
    bg, pg = body_geom_ids(m, "block"), body_geom_ids(m, "pusher")
    adr = {n: m.joint(n).qposadr[0] for n in
           ("block_x", "block_y", "block_yaw", "root_x", "root_y")}
    p_off = m.body_pos[m.body("pusher").id][:2].copy()

    def build(bx, by, yaw, world_xy):
        md = mujoco.MjData(m)
        md.qpos[adr["block_x"]] = float(bx)
        md.qpos[adr["block_y"]] = float(by)
        md.qpos[adr["block_yaw"]] = float(yaw)
        md.qpos[adr["root_x"]] = float(world_xy[0] - p_off[0])
        md.qpos[adr["root_y"]] = float(world_xy[1] - p_off[1])
        return md

    if check_only:
        return _revalidate("push", task, bg, pg, 3e-3)

    rows, diags = [], []
    cells = lhs(np.random.default_rng(20260721), N,
                [(-0.15, 0.15), (-0.15, 0.15), (-0.5, 0.5), GAP_RANGE])
    rng = np.random.default_rng(1)
    for i in range(N):
        bx, by, yaw, gap = rng.uniform(cells[i, :, 0], cells[i, :, 1])
        norm = float(np.hypot(bx, by))
        u = np.array([bx, by]) / norm if norm > 1e-6 else np.array([0.0, 1.0])

        # Bisect outward from the block's centre of mass, so `lo` overlaps
        # whatever the shape's geometry does around its joint origin.
        c = build(bx, by, yaw, np.zeros(2))
        mujoco.mj_forward(m, c)
        centre = c.xipos[m.body("block").id][:2].copy()

        def gap_at(r, bx=bx, by=by, yaw=yaw, u=u, centre=centre):
            d = build(bx, by, yaw, centre + r * u)
            mujoco.mj_forward(m, d)
            return clearance(m, d, bg, pg)

        world = centre + bisect(gap_at, gap, 0.0, 0.35) * u
        md = build(bx, by, yaw, world)
        res = check(task, md, bg, pg, drift_tol=3e-3)
        if not res[0]:
            raise RuntimeError(f"push scenario {i} failed: {res[1:]}")
        rows.append({
            "block_xy": [r4(bx), r4(by)], "block_yaw": r4(yaw),
            "pusher_xy": [r4(world[0] - p_off[0]), r4(world[1] - p_off[1])],
        })
        diags.append(res[1:])
    return rows, diags


def make_balance(check_only: bool):
    """Balance: block on the plate, plate at a small initial tilt."""
    task = Balance(shape="circle", scale=1.0, goal_xy=(0.05, 0.06),
                   model_config=load_numerics("balance"))
    m = task.mj_model
    bg, pl = body_geom_ids(m, "block"), body_geom_ids(m, "plate")
    if check_only:
        return _revalidate("balance", task, bg, pl, 0.02)

    rows, diags = [], []
    cells = lhs(np.random.default_rng(20260722), N,
                [(-0.10, 0.10), (-0.10, 0.10), (-0.05, 0.05), (-0.05, 0.05)])
    rng = np.random.default_rng(2)
    for i in range(N):
        bx, by, roll, pitch = rng.uniform(cells[i, :, 0], cells[i, :, 1])
        md = mujoco.MjData(m)
        task.set_initial_state(md, tilt=(float(roll), float(pitch)),
                               block_xy=(float(bx), float(by)))
        # A tilted plate makes the ball slide, so the settle tolerance here
        # is 20 mm, not Push's 3 mm -- the tilt is the point.
        res = check(task, md, bg, pl, drift_tol=0.02)
        if not res[0]:
            raise RuntimeError(f"balance scenario {i} failed: {res[1:]}")
        rows.append({"block_xy": [r4(bx), r4(by)],
                     "tilt": [r4(roll), r4(pitch)]})
        diags.append(res[1:])
    return rows, diags


def make_balance_fr3(check_only: bool):
    """Balance-FR3: a sphere resting on the arm-held level plate.

    Sphere (not the planar-balance circle) so rolling friction bites. The arm
    starts at 'home' (level plate), so unlike planar Balance there is no tilt
    to excite the ball -- the block just rests at a plate-local ``block_xy``.
    """
    task = BalanceFr3(
        sampling_space="joint", shape="sphere", scale=1.0,
        goal_xy=(0.05, 0.04), model_config=load_numerics("balance_fr3"),
    )
    m = task.mj_model
    bg = body_geom_ids(m, "block")
    # The plate is a geom on the EE link, not its own body -- name it directly.
    pl = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "plate")}
    if check_only:
        return _revalidate("balance_fr3", task, bg, pl, 0.02)

    rows, diags = [], []
    cells = lhs(np.random.default_rng(20260724), N,
                [(-0.10, 0.10), (-0.10, 0.10)])
    rng = np.random.default_rng(4)
    kid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
    for i in range(N):
        bx, by = rng.uniform(cells[i, :, 0], cells[i, :, 1])
        md = mujoco.MjData(m)
        md.qpos[:] = m.key_qpos[kid]
        task.set_initial_state(md, block_xy=(float(bx), float(by)))
        # Level plate, so the ball barely drifts; keep a modest 20 mm tol.
        res = check(task, md, bg, pl, drift_tol=0.02)
        if not res[0]:
            raise RuntimeError(f"balance_fr3 scenario {i} failed: {res[1:]}")
        rows.append({"block_xy": [r4(bx), r4(by)]})
        diags.append(res[1:])
    return rows, diags


def make_peg_fr3(check_only: bool):
    """Peg-FR3: peg tip starting above and off-axis from the socket mouth.

    The arm holds the peg rigidly, so there is no free object to settle -- the
    'drift' check is the arm holding the tip near its commanded pose under zero
    control. Starts are off-axis above the mouth so the planner has to find the
    hole rather than descend straight in. Clearance-independent (world-frame
    poses above a fixed socket), so one bank serves every clearance config.

    Each scenario also carries a frozen ``mount_offset`` -- a small SE(3)
    perturbation of where the peg is bolted to the wrist (see
    ``bampc/dr/randomizer.py``'s body ``pos_x/y/z``/``rot_x/y/z``
    params), LHS-sampled on its own RNG stream so it never perturbs
    ``tip_pos``. This is what makes ``--true-offset`` reproducible per
    scenario in
    ``examples/domain_randomization/peg_fr3.py`` and
    ``examples/state_uncertainty/peg_fr3.py`` -- the *frozen*, catalog form
    of what those flags would otherwise draw at runtime. It plays no role in
    ``tip_pos`` validation: the mount is nominal for IK/clearance checks
    here, exactly as it is for the planner (which never knows its own true
    mount error).
    """
    task = PegFr3(sampling_space="task", clearance=0.002,
                  model_config=load_numerics("peg_fr3"))
    m = task.mj_model
    pg, walls = {task.peg_geom_id}, set(task.wall_geom_ids)
    kid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
    tip_sid = task.tip_site_id

    def validate(tip_pos: np.ndarray) -> tuple:
        md = mujoco.MjData(m)
        md.qpos[:] = m.key_qpos[kid]
        task.set_initial_state(md, tip_pos=tip_pos)
        mujoco.mj_forward(m, md)
        pen0 = worst_pen(m, md)
        gap = clearance(m, md, pg, walls)
        ik_err = float(np.linalg.norm(md.site_xpos[tip_sid] - tip_pos))
        d = mujoco.MjData(m)
        d.qpos[:], d.qvel[:] = md.qpos, md.qvel
        mujoco.mj_forward(m, d)
        start = d.site_xpos[tip_sid].copy()
        pen = pen0
        for _ in range(int(round(SETTLE_T / m.opt.timestep))):
            d.ctrl[:] = 0.0
            mujoco.mj_step(m, d)
            pen = min(pen, worst_pen(m, d))
        drift = float(np.linalg.norm(d.site_xpos[tip_sid] - start))
        ok = (pen0 > -PEN_TOL and pen > -SETTLE_PEN
              and ik_err < 0.01 and drift < 0.02)
        return ok, pen0, gap, drift, pen

    if check_only:
        bank = scenarios.load("peg_fr3")
        diags = []
        for s in bank.scenarios:
            res = validate(np.asarray(s.start["tip_pos"], float))
            if not res[0]:
                raise RuntimeError(f"peg_fr3 scenario {s.index}: {res[1:]}")
            mo = s.start.get("mount_offset")
            if mo is not None and (len(mo["pos"]) != 3 or len(mo["rot"]) != 3):
                raise RuntimeError(
                    f"peg_fr3 scenario {s.index}: malformed mount_offset {mo}"
                )
            diags.append(res[1:])
        return None, diags

    rows, diags = [], []
    cells = lhs(np.random.default_rng(20260725), N,
                [(-0.08, 0.08), (-0.08, 0.08), (0.12, 0.20)])
    rng = np.random.default_rng(5)
    # Independent RNG stream, so retuning/adding this never perturbs the
    # already-frozen tip_pos values above.
    mount_cells = lhs(np.random.default_rng(20260726), N,
                       [MOUNT_POS_RANGE, MOUNT_POS_RANGE, MOUNT_POS_RANGE,
                        MOUNT_ROT_RANGE, MOUNT_ROT_RANGE, MOUNT_ROT_RANGE])
    mount_rng = np.random.default_rng(6)
    for i in range(N):
        dx, dy, h = rng.uniform(cells[i, :, 0], cells[i, :, 1])
        tip = task.tip_target(float(h)) + np.array([dx, dy, 0.0])
        res = validate(tip)
        if not res[0]:
            raise RuntimeError(f"peg_fr3 scenario {i} failed: {res[1:]}")
        mpx, mpy, mpz, mrx, mry, mrz = mount_rng.uniform(
            mount_cells[i, :, 0], mount_cells[i, :, 1]
        )
        rows.append({
            "tip_pos": [r4(tip[0]), r4(tip[1]), r4(tip[2])],
            "mount_offset": {
                "pos": [r4(mpx), r4(mpy), r4(mpz)],
                "rot": [r4(mrx), r4(mry), r4(mrz)],
            },
        })
        diags.append(res[1:])
    return rows, diags


def _fr3_tasks():
    """One PushFr3 per manipulation type, at the bank's geometry."""
    bank = scenarios.load("push_fr3")
    return {
        mt: PushFr3(sampling_space="task", manipulation_type=mt,
                    shape=bank.shape, scale=bank.scale,
                    goal_xy=bank.goal_xy,
                    model_config=load_numerics(f"fr3_{mt}"))
        for mt in ("joint", "free")
    }


def make_push_fr3(check_only: bool):
    """Push-FR3: EE on the away-from-goal side, valid for both blocks."""
    tasks = _fr3_tasks()
    # From the bank, not restated: the EE is placed on the far side of the
    # block from the goal, so a generator that disagreed with the sweep about
    # where the goal is would lay out every start state facing the wrong way.
    goal_xy = np.asarray(scenarios.load("push_fr3").goal_xy, float)

    def build(mt, bx, by, yaw, exy):
        task = tasks[mt]
        m = task.mj_model
        md = mujoco.MjData(m)
        kid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
        md.qpos[:] = m.key_qpos[kid]
        task.set_initial_state(
            md, ee_pos=np.array([exy[0], exy[1], task.ee_z_target]),
            block_xy=np.array([bx, by]), block_yaw=yaw)
        return task, md

    def geoms(mt):
        m = tasks[mt].mj_model
        return body_geom_ids(m, "block"), body_geom_ids(m, "pusher")

    if check_only:
        bank = scenarios.load("push_fr3")
        out = []
        for mt in ("joint", "free"):
            for s in bank.scenarios:
                md = scenarios.pose(bank, tasks[mt], s)
                res = check(tasks[mt], md, *geoms(mt), drift_tol=3e-3)
                if not res[0]:
                    raise RuntimeError(f"push_fr3/{mt} {s.index}: {res[1:]}")
                out.append(res[1:])
        return None, out

    rows, diags = [], []
    cells = lhs(np.random.default_rng(20260723), N,
                [(0.44, 0.56), (-0.12, 0.12), (-0.6, 0.6),
                 (-0.4, 0.4), GAP_RANGE])
    rng = np.random.default_rng(3)
    for i in range(N):
        bx, by, yaw, jitter, gap = rng.uniform(cells[i, :, 0], cells[i, :, 1])
        away = np.array([bx, by]) - goal_xy
        ang = float(np.arctan2(away[1], away[0])) + jitter
        u = np.array([np.cos(ang), np.sin(ang)])

        jt = tasks["joint"]
        _, c = build("joint", bx, by, yaw, np.array([10.0, 10.0]))
        mujoco.mj_forward(jt.mj_model, c)
        centre = c.xipos[jt.mj_model.body("block").id][:2].copy()

        # Both manipulation types must clear, so keep the larger standoff.
        radii = []
        for mt in ("joint", "free"):
            bgm, pgm = geoms(mt)

            def gap_at(r, mt=mt, bgm=bgm, pgm=pgm):
                task, d = build(mt, bx, by, yaw, centre + r * u)
                mujoco.mj_forward(task.mj_model, d)
                return clearance(task.mj_model, d, bgm, pgm)

            radii.append(bisect(gap_at, gap, 0.0, 0.30, iters=26))
        exy = centre + max(radii) * u

        for mt in ("joint", "free"):
            task, md = build(mt, bx, by, yaw, exy)
            res = check(task, md, *geoms(mt), drift_tol=3e-3)
            if not res[0]:
                raise RuntimeError(f"push_fr3/{mt} {i} failed: {res[1:]}")
            diags.append(res[1:])
        rows.append({"block_xy": [r4(bx), r4(by)], "block_yaw": r4(yaw),
                     "ee_xy": [r4(exy[0]), r4(exy[1])]})
    return rows, diags


def _curling_task():
    """One CurlingFr3 at the bank's own geometry."""
    bank = scenarios.load(CURLING_BANK)
    return CurlingFr3(
        shape=bank.shape, scale=bank.scale, goal_xy=bank.goal_xy,
        model_config=load_numerics("curling"))


def make_curling(check_only: bool):
    """Curling: puck near the box front, EE behind it on the house line.

    Same mold as ``make_push_fr3`` minus the manipulation axis, plus one
    constraint that task does not have: the bisected EE standoff must land
    inside the launch box, or the barrier would drag the arm on step 1.
    """
    task = _curling_task()
    bank = scenarios.load(CURLING_BANK)
    goal_xy = np.asarray(bank.goal_xy, float)
    x_lo, x_hi, y_lo, y_hi = task.launch_box

    def geoms():
        m = task.mj_model
        return (body_geom_ids(m, "block"),
                body_geom_ids(m, "ee_frame", "pusher"))

    def build(bx, by, yaw, exy, clamp_ee=False):
        # clamp_ee defaults OFF here: the bisection below deliberately probes
        # standoffs from the puck centre outwards, most of which start outside
        # the launch box. Clamping those would flatten the function it is
        # searching and hand back a gap that was never measured at that
        # radius. The final row is re-checked against the box explicitly.
        m = task.mj_model
        md = mujoco.MjData(m)
        kid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
        md.qpos[:] = m.key_qpos[kid]
        task.set_initial_state(
            md, ee_pos=np.array([exy[0], exy[1], task.ee_z_target]),
            block_xy=np.array([bx, by]), block_yaw=yaw, clamp_ee=clamp_ee)
        return md

    if check_only:
        return _revalidate(CURLING_BANK, task, *geoms(), drift_tol=3e-3)

    rows, diags = [], []
    # Puck near the box's front face (its 5 cm radius puts it partly past
    # x_hi, which is fine -- the box constrains the EE, not the puck). Ranges
    # are kept narrow enough in y that the standoff cannot push the EE out of
    # the box sideways; that is asserted below rather than assumed.
    # x shifted +5 cm with the lane/launch box (models/curling/scene.xml);
    # keep this in lockstep by hand, same as CurlingFr3's launch_box default.
    cells = lhs(np.random.default_rng(20260907), N,
                [(0.55, 0.61), (-0.08, 0.08), (-0.6, 0.6),
                 (-0.3, 0.3), GAP_RANGE])
    rng = np.random.default_rng(11)
    bgm, pgm = geoms()
    for i in range(N):
        bx, by, yaw, jitter, gap = rng.uniform(cells[i, :, 0], cells[i, :, 1])
        away = np.array([bx, by]) - goal_xy
        ang = float(np.arctan2(away[1], away[0])) + jitter
        u = np.array([np.cos(ang), np.sin(ang)])

        c = build(bx, by, yaw, np.array([10.0, 10.0]))
        mujoco.mj_forward(task.mj_model, c)
        centre = c.xipos[task.mj_model.body("block").id][:2].copy()

        def gap_at(r, bx=bx, by=by, yaw=yaw, centre=centre, u=u):
            d = build(bx, by, yaw, centre + r * u)
            mujoco.mj_forward(task.mj_model, d)
            return clearance(task.mj_model, d, bgm, pgm)

        exy = centre + bisect(gap_at, gap, 0.0, 0.30, iters=26) * u
        if not (x_lo <= exy[0] <= x_hi and y_lo <= exy[1] <= y_hi):
            raise RuntimeError(
                f"curling {i}: EE standoff {exy.tolist()} left the launch box "
                f"{task.launch_box}; narrow the puck or jitter ranges.")

        md = build(bx, by, yaw, exy, clamp_ee=True)
        res = check(task, md, bgm, pgm, drift_tol=3e-3)
        if not res[0]:
            raise RuntimeError(f"curling {i} failed: {res[1:]}")
        diags.append(res[1:])
        rows.append({"block_xy": [r4(bx), r4(by)], "block_yaw": r4(yaw),
                     "ee_xy": [r4(exy[0]), r4(exy[1])]})
    return rows, diags


def _revalidate(name: str, task, ga, gb, drift_tol: float):
    """Re-check the bank already on disk against a freshly built task."""
    bank = scenarios.load(name)
    diags = []
    for s in bank.scenarios:
        md = scenarios.pose(bank, task, s)
        res = check(task, md, ga, gb, drift_tol)
        if not res[0]:
            raise RuntimeError(f"{name} scenario {s.index} failed: {res[1:]}")
        diags.append(res[1:])
    return None, diags


def phases(freqs, n: int) -> list[float]:
    """Drift phase times spread evenly over the slowest drift period.

    All-zero frequencies (a static goal, e.g. Peg-FR3) give phase 0 throughout.
    """
    positive = [f for f in freqs if f > 0]
    if not positive:
        return [0.0] * n
    period = 1.0 / min(positive)
    return [round(i * period / n, 3) for i in range(n)]


def main() -> None:
    """Generate (or re-validate) every bank and print the result."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate the banks on disk instead of generating new ones",
    )
    makers = {
        "push": make_push,
        "balance": make_balance,
        "push_fr3": make_push_fr3,
        "balance_fr3": make_balance_fr3,
        "peg_fr3": make_peg_fr3,
        # Keyed by BANK STEM, not task name -- main() feeds this key straight
        # to drift_freqs(), which loads a bank by it. The other FR3 entry
        # here gets that wrong: `make_scenarios.py --task push_fr3` raises,
        # because the banks on disk are push_fr3_t / push_fr3_l and there is
        # no push_fr3.yaml for either drift_freqs() or make_push_fr3() to
        # load. Don't copy that shape.
        CURLING_BANK: make_curling,
    }
    parser.add_argument(
        "--task",
        choices=list(makers),
        help="only this task (default: all). Speeds iterating on one bank.",
    )
    args = parser.parse_args()

    if args.task:
        makers = {args.task: makers[args.task]}
    for name, maker in makers.items():
        rows, diags = maker(args.check)
        print(f"\n### {name}")
        if rows is not None:
            ph = phases(drift_freqs(name), len(rows))
            print("scenarios:")
            for i, row in enumerate(rows):
                body = ", ".join(f"{k}: {v}" for k, v in row.items())
                print(f"  - {{{body}, drift_phase_time: {ph[i]}}}")
        a = np.array(diags, float)
        print(f"# {len(a)} states OK | init_pen worst {a[:, 0].min():.2e}"
              f" | gap {a[:, 1].min():+.4f}..{a[:, 1].max():+.4f}"
              f" | settle drift max {a[:, 2].max():.2e}"
              f" | settle_pen worst {a[:, 3].min():.2e}")


if __name__ == "__main__":
    main()
