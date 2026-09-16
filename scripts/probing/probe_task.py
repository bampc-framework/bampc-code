"""Probe the per-environment contact/constraint budget for each task.

Run with:
    uv run python scripts/probing/probe_task.py
    uv run python scripts/probing/probe_task.py --task push_fr3 --shape box

For each task this runs a short CPU-side MuJoCo simulation with random controls
and reports the peak ``mj_data.ncon`` and ``mj_data.nefc`` values, which map to
``nconmax``/``naconmax`` and ``njmax`` respectively. A safety factor gives the
recommended ``ContactBudget`` per environment (the engine multiplies these by
``nworld`` for ``make_data``). Adapted from the hydrax script of the same name.

These budgets matter for both correctness (under-sizing silently drops contacts)
and speed (over-sizing inflates buffers / per-step work), so re-run this when a
model changes.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence

import mujoco
import numpy as np

from bampc.config.numerics import load as load_numerics
from bampc.task.balance import Balance
from bampc.task.balance_fr3 import BalanceFr3
from bampc.task.common.shapes import list_shapes
from bampc.task.curling_fr3 import SHAPES as CURLING_SHAPES
from bampc.task.curling_fr3 import CurlingFr3
from bampc.task.flip_fr3 import (
    _LYING_QUAT,
    _WALL_POS,
    _WALL_SIZE,
    FlipFr3,
)
from bampc.task.peg_fr3 import PegFr3
from bampc.task.push import Push
from bampc.task.push_fr3 import PushFr3

# Random iid control per step is a poor stress test on its own -- a single
# unlucky RNG seed can under-report the true peak (see PROBE_SEEDS below), so
# every task is probed under several and the *max* across them is used.
PROBE_SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)

# Numerics per FR3 block model. Named because the task's fallback is not the
# tuned `joint` setup, and the budgets probed here must come from the physics
# that will actually run. Push/Balance fall back to values identical to their
# tuned ones.
MODEL_CONFIGS = {
    "joint": load_numerics("fr3_joint"),
    "free": load_numerics("fr3_free"),
    "balance_fr3": load_numerics("balance_fr3"),
    "peg_fr3": load_numerics("peg_fr3"),
    "flip_fr3": load_numerics("flip_fr3"),
    "curling": load_numerics("curling"),
}

# --------------------------------------------------------------------- #
# Seeding. Push/Balance graft their block at a computed resting height
# directly in the MJCF spec (see e.g. Balance.__init__), so a bare
# mj_resetData already starts in contact. BalanceFr3 and PegFr3 place their
# movable body via live forward kinematics instead -- confirmed empirically
# (both probed ncon=0 for the *entire* 1000-step random rollout under the
# default seed, before this fix): the block/peg never touches anything, so
# random control from qpos0 never exercises the contact-rich regime the
# budget needs to cover.
#
# PushFr3/FlipFr3 have the *same* problem one level up, and it went
# undetected longer because the block itself *is* in contact (resting on
# the table) at ``default_seed`` -- only the arm is missing it. Their bare
# ``qpos0`` (mj_resetData, all-zero joint angles) puts the arm ~0.86 m from
# the block/box, nowhere near the table -- confirmed empirically: peak ncon
# from a 1000-step random rollout was *identical* whether the arm started
# at qpos0 or at the "home" keyframe (~0.13-0.17 m away), because
# uncorrelated per-step iid control is a random walk in *velocity*, not a
# directed approach, and never reliably closes even a small gap in 1000
# steps (same lesson as ``_peg_fr3_seed`` below). So every PushFr3/FlipFr3
# entry used to probe only the block<->table resting contacts, never the
# EE<->block contact-rich regime pushing/flipping actually exercises, and
# (for FlipFr3) never the box<->wall contact the wall was added for --
# ``flip_fr3_contact_budget``'s docstring documents this gap directly.
# Fixed by seeding with the EE already overlapping the block/box, like
# ``_peg_fr3_seed`` seeds the peg near fully seated.
# --------------------------------------------------------------------- #


def default_seed(task) -> mujoco.MjData:
    """Bare model default (already resting in contact for these tasks)."""
    data = mujoco.MjData(task.mj_model)
    mujoco.mj_resetData(task.mj_model, data)
    return data


def _home_keyframe(task) -> mujoco.MjData:
    """Reset to the model's ``"home"`` keyframe (arm at its tuned pose).

    Forwards before returning so callers can immediately read ``xpos``/
    ``xquat`` (e.g. a block's resting pose) without a stale, all-zero
    kinematic cache.
    """
    data = mujoco.MjData(task.mj_model)
    kid = mujoco.mj_name2id(task.mj_model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(task.mj_model, data, kid)
    mujoco.mj_forward(task.mj_model, data)
    return data


def _balance_fr3_seed(task) -> mujoco.MjData:
    """Rest the block on the live plate frame, like every real caller does.

    ``set_initial_state``'s own docstring requires the "home" keyframe be
    loaded first.
    """
    data = _home_keyframe(task)
    task.set_initial_state(data)
    return data


def _peg_fr3_seed(task) -> mujoco.MjData:
    """Seed near fully seated, not the frozen scenarios' pre-insertion pose.

    The real start states put the peg tip *above* the socket mouth (the
    planner has to find the hole -- see ``configs/scenarios/peg_fr3.yaml``),
    but 1000 steps of uncorrelated random control essentially never drives a
    real insertion. The budget has to cover the inserted state (walls +
    base contact), so seed there directly and let random control wiggle
    around it.
    """
    data = _home_keyframe(task)
    task.set_initial_state(data, height=-0.9 * task.geometry["peg_len"])
    return data


def _push_fr3_seed(task) -> mujoco.MjData:
    """Overlap the EE with the block at push height, not just nearby.

    See the module comment above ``default_seed``: ``qpos0`` puts the arm
    far from the block and 1000 steps of iid random joint velocities never
    close that gap. Seeding the EE already inside the block, like
    ``_peg_fr3_seed`` seeds the peg near fully seated, guarantees every
    rollout actually exercises EE<->block contact instead of just the
    block resting on the table.
    """
    data = _home_keyframe(task)
    blk_id = mujoco.mj_name2id(task.mj_model, mujoco.mjtObj.mjOBJ_BODY, "block")
    block_xy = data.xpos[blk_id][:2].copy()
    task.set_initial_state(
        data,
        ee_pos=np.array([block_xy[0], block_xy[1], task.ee_z_target]),
        ee_quat=task.goal_quat_ee,
    )
    return data


def _flip_fr3_seed(task) -> mujoco.MjData:
    """Overlap the EE with the box, flush against the wall when present.

    Same rationale as ``_push_fr3_seed``. When the task has a wall, the
    box is additionally repositioned (still lying flat, still resting on
    the table -- only its Y shifts) so its leaning face sits right against
    it: the box<->wall contact regime the wall exists for, per
    ``flip_fr3_contact_budget``'s docstring, was never actually probed
    before this fix.
    """
    data = _home_keyframe(task)
    m = task.mj_model
    blk_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "block")
    block_xy = data.xpos[blk_id][:2].copy()
    has_wall = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "wall_wall") != -1
    if has_wall:
        gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "block_collision")
        lying_mat = np.zeros(9)
        mujoco.mju_quat2Mat(lying_mat, np.array(_LYING_QUAT))
        world_half_y = float(
            (np.abs(lying_mat.reshape(3, 3)) @ m.geom_size[gid])[1]
        )
        wall_face_y = _WALL_POS[1] - _WALL_SIZE[1]
        block_xy[1] = wall_face_y - world_half_y - 0.002
    block_z = float(data.xpos[blk_id][2])
    task.set_initial_state(
        data,
        ee_pos=np.array([block_xy[0], block_xy[1], block_z]),
        ee_quat=task.goal_quat_ee,
        block_xy=block_xy,
    )
    return data


def _curling_seed(task) -> mujoco.MjData:
    """Overlap the EE with the puck, inside the launch box.

    Same rationale as ``_push_fr3_seed``, plus one twist of curling's own:
    the seed is clamped into the launch box, so the probe measures the
    contact regime the task actually runs in rather than one where the
    barrier is saturated and fighting every random control.
    """
    data = _home_keyframe(task)
    m = task.mj_model
    blk_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "block")
    block_xy = data.xpos[blk_id][:2].copy()
    x_lo, x_hi, y_lo, y_hi = task.launch_box
    task.set_initial_state(
        data,
        ee_pos=np.array([
            float(np.clip(block_xy[0], x_lo, x_hi)),
            float(np.clip(block_xy[1], y_lo, y_hi)),
            task.ee_z_target,
        ]),
        ee_quat=task.goal_quat_ee,
    )
    return data


# Task name -> (zero-arg constructor, seed function). One constructor per
# (task, shape) combo; the two FR3 manipulation types differ in contacts
# (constrained joints vs a free rigid body), so probe both. sampling_space is
# irrelevant here (we drive actuators directly).
#
# "cracker_box" is excluded here -- it's a mesh-visual shape added for
# Flip-FR3 specifically (see its own registration below) and has no contact
# budget in push_contact_budget/balance_contact_budget/etc., so looping it
# through every other task would crash this script's main() on a plain
# `ValueError: no contact budget for shape 'cracker_box'`.
TASKS = {}
for _shape in list_shapes():
    if _shape == "cracker_box":
        continue
    TASKS[f"Push-{_shape}"] = (lambda s=_shape: Push(shape=s), default_seed)
    TASKS[f"PushFr3-joint-{_shape}"] = (
        lambda s=_shape: PushFr3(
            manipulation_type="joint", shape=s,
            model_config=MODEL_CONFIGS["joint"],
        ),
        _push_fr3_seed,
    )
    TASKS[f"PushFr3-free-{_shape}"] = (
        lambda s=_shape: PushFr3(
            manipulation_type="free", shape=s,
            model_config=MODEL_CONFIGS["free"],
        ),
        _push_fr3_seed,
    )
    TASKS[f"Balance-{_shape}"] = (
        lambda s=_shape: Balance(shape=s), default_seed
    )
    TASKS[f"BalanceFr3-{_shape}"] = (
        lambda s=_shape: BalanceFr3(
            shape=s, model_config=MODEL_CONFIGS["balance_fr3"],
        ),
        _balance_fr3_seed,
    )

# Flip-FR3: box placement is baked into the compiled spec at build time
# (same as Push/PushFr3's free block), so a bare reset already starts the
# box in contact with the table -- but not the EE with the box, hence
# _flip_fr3_seed (see its docstring), not default_seed. Probe both wall
# states: the wall adds box<->wall contacts on top of box<->table, and
# flip_fr3_contact_budget's single number must cover both.
for _shape in ("cracker_box",):
    TASKS[f"FlipFr3-{_shape}"] = (
        lambda s=_shape: FlipFr3(
            shape=s, wall=False, model_config=MODEL_CONFIGS["flip_fr3"],
        ),
        _flip_fr3_seed,
    )
    TASKS[f"FlipFr3-wall-{_shape}"] = (
        lambda s=_shape: FlipFr3(
            shape=s, wall=True, model_config=MODEL_CONFIGS["flip_fr3"],
        ),
        _flip_fr3_seed,
    )

# Curling-FR3: only the shapes curling_contact_budget covers, not every
# shape in the library -- the puck is a disc by definition and the T/L/etc.
# fragments have no budget here. Seeded like Push-FR3 (EE overlapping the
# puck), with the extra constraint that the seed must sit inside the launch
# box or the IK barrier fights the probe's random control the whole way.
for _shape in CURLING_SHAPES:
    TASKS[f"CurlingFr3-{_shape}"] = (
        lambda s=_shape: CurlingFr3(
            shape=s, model_config=MODEL_CONFIGS["curling"],
        ),
        _curling_seed,
    )

# PegFr3 has no `shape`/`scale` -- fixed peg/socket geometry -- so one entry.
TASKS["PegFr3"] = (
    lambda: PegFr3(model_config=MODEL_CONFIGS["peg_fr3"]), _peg_fr3_seed
)


def _probe_once(
    task, seed_fn: Callable, num_steps: int, seed: int
) -> tuple[int, int]:
    """One random CPU rollout; return (peak_ncon, peak_nefc)."""
    model = task.mj_model
    data = seed_fn(task)

    rng = np.random.default_rng(seed)
    lo = np.where(
        model.actuator_ctrllimited.astype(bool),
        model.actuator_ctrlrange[:, 0],
        -1.0,
    )
    hi = np.where(
        model.actuator_ctrllimited.astype(bool),
        model.actuator_ctrlrange[:, 1],
        1.0,
    )

    max_ncon = 0
    max_nefc = 0
    for _ in range(num_steps):
        data.ctrl[:] = rng.uniform(lo, hi)
        mujoco.mj_step(model, data)
        max_ncon = max(max_ncon, data.ncon)
        max_nefc = max(max_nefc, data.nefc)
    return max_ncon, max_nefc


def probe(
    task, seed_fn=default_seed, num_steps: int = 1000,
    safety_factor: float = 3.0, seed: int = 42,
    seeds: Sequence[int] | None = None,
) -> tuple[int, int, int, int]:
    """Random CPU rollout(s); return (peak_ncon, peak_nefc, rec_ncon, rec_nj).

    A single ``seed`` (the default) is one iid random rollout and can
    under-report the true peak. Pass ``seeds`` (plural) to
    probe several and take the max across them instead -- see
    ``PROBE_SEEDS``, used by this script's own ``main()``.
    """
    max_ncon = 0
    max_nefc = 0
    for s in (seeds if seeds is not None else (seed,)):
        ncon, nefc = _probe_once(task, seed_fn, num_steps, s)
        max_ncon = max(max_ncon, ncon)
        max_nefc = max(max_nefc, nefc)

    rec_ncon = int(np.ceil(max_ncon * safety_factor))
    rec_nj = int(np.ceil(max_nefc * safety_factor))
    return max_ncon, max_nefc, rec_ncon, rec_nj


def _filter_tasks(
    task_filter: str | None, shape_filter: str | None
) -> dict:
    """Registered tasks matching both filters, or all of ``TASKS`` if unset.

    Raises:
        ValueError: The filters match nothing -- lists what's available
            instead of silently printing an empty table.
    """
    out = {
        name: entry for name, entry in TASKS.items()
        if (task_filter is None or task_filter.lower() in name.lower())
        and (shape_filter is None or name.endswith(f"-{shape_filter}"))
    }
    if not out:
        raise ValueError(
            f"no task matches --task={task_filter!r} --shape={shape_filter!r}"
            f"; available: {sorted(TASKS)}"
        )
    return out


def main() -> None:
    """Probe every registered (or filtered) task, print recommended budgets."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        help="substring filter on task name (case-insensitive), e.g. "
        "'push_fr3' or 'balance'",
    )
    parser.add_argument("--shape", help="exact shape filter, e.g. 'box'")
    args = parser.parse_args()

    fmt = "{:<22} {:>6} {:>6} {:>14} {:>14} {:>12}"
    print(fmt.format("Task", "ncon", "nefc", "ncon/nac_env", "nj_env", "(x3)"))
    print("-" * 70)
    for name, (cls, seed_fn) in _filter_tasks(args.task, args.shape).items():
        task = cls()
        peak_ncon, peak_nefc, rec_ncon, rec_nj = probe(
            task, seed_fn, seeds=PROBE_SEEDS
        )
        print(
            fmt.format(
                name, peak_ncon, peak_nefc, rec_ncon, rec_nj, "safety=3.0"
            )
        )


if __name__ == "__main__":
    main()