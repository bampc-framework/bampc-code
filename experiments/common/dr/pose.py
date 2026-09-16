"""Body poses from configurations, on the host.

The rollout engine records ``qpos``/``qvel``, not body poses, so any
task-space metric (e.g. ``scripts/probing/determinism_check.py``'s pairwise
spread) needs the pose derived after the fact.

Reading ``d.xpos`` inside the rollout would be wrong, not just awkward:
``mjw.step`` is a forward pass plus an integrate, so once a step returns,
``d.qpos`` holds the new configuration while ``d.xpos`` still holds the
kinematics of the *previous* one. Recorded side by side they would sit one
step apart. Kinematics is a pure function of ``qpos``, so replaying it here
is exact and leaves the captured graph alone.
"""

from __future__ import annotations

import mujoco
import numpy as np


def body_poses(
    mj_model: mujoco.MjModel,
    scratch: mujoco.MjData,
    qpos: np.ndarray,
    body_id: int,
) -> tuple[np.ndarray, np.ndarray]:
    """World pose of one body at each of many configurations.

    Args:
        mj_model: Model to run kinematics against.
        scratch: Scratch data, mutated in place (never the live sim's).
        qpos: Configurations, shape ``(..., nq)``.
        body_id: Body whose pose to read.

    Returns:
        ``(xpos, xquat)`` of shape ``(..., 3)`` and ``(..., 4)``.
    """
    flat = np.asarray(qpos, np.float64).reshape(-1, mj_model.nq)
    xpos = np.empty((flat.shape[0], 3))
    xquat = np.empty((flat.shape[0], 4))
    for i, q in enumerate(flat):
        scratch.qpos[:] = q
        mujoco.mj_kinematics(mj_model, scratch)
        xpos[i] = scratch.xpos[body_id]
        xquat[i] = scratch.xquat[body_id]
    lead = np.shape(qpos)[:-1]
    return xpos.reshape(*lead, 3), xquat.reshape(*lead, 4)
