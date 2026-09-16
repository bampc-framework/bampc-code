"""Shared FR3 damped-least-squares IK: device kernels + numpy twin.

Every FR3 task maps a reduced task-space command to the 7 joint-velocity
actuators the same way -- assemble the 6x7 end-effector Jacobian, then solve the
damped system ``(J Jᵀ + λI) x = twist`` and back-substitute ``dq = Jᵀ x``. Only
the *twist* differs per task (which components are sampled vs regulated), so the
mechanism lives here and each task supplies its own 6-vector ``b``:

* Push commands EE vx/vy, regulates z-height + point-down orientation.
* Balance commands plate roll/pitch rate, regulates EE position + yaw.
* Peg commands the full SE(3) twist -- all six rows sampled, none regulated.

The solve is symmetric-positive-definite (λ>0), done with the branch-free 6x6
Cholesky in :mod:`bampc.task.common.cholesky`.

:func:`solve_pose_ik` is the host-only counterpart used to *place* the arm at a
pose (start states), rather than to track a twist each step.
"""

from __future__ import annotations

import mujoco
import numpy as np
import warp as wp
from mujoco_warp._src.support import jac_dof

from bampc.task.common.cholesky import chol_solve6, mat66, vec6

# 6x7 EE Jacobian (6 spatial rows, 7 actuated FR3 dofs) and a 7-vector of joint
# velocities -- the shapes the shared device funcs pass around.
mat67 = wp.types.matrix(shape=(6, 7), dtype=wp.float32)
vec7 = wp.types.vector(length=7, dtype=wp.float32)


@wp.func
def ee_jacobian_dof(
    body_parentid: wp.array(dtype=wp.int32),
    body_rootid: wp.array(dtype=wp.int32),
    dof_bodyid: wp.array(dtype=wp.int32),
    body_isdofancestor: wp.array2d(dtype=wp.int32),
    subtree_com: wp.array2d(dtype=wp.vec3),
    cdof: wp.array2d(dtype=wp.spatial_vector),
    point: wp.vec3,
    ee_body: wp.int32,
    arm_dof: wp.array(dtype=wp.int32),
    w: wp.int32,
) -> mat67:
    """6x7 EE Jacobian at ``point`` for world ``w`` (rows 0-2 lin, 3-5 rot)."""
    jmat = mat67()
    for c in range(7):
        jp, jr = jac_dof(
            body_parentid,
            body_rootid,
            dof_bodyid,
            body_isdofancestor,
            subtree_com,
            cdof,
            point,
            ee_body,
            arm_dof[c],
            w,
        )
        jmat[0, c] = jp[0]
        jmat[1, c] = jp[1]
        jmat[2, c] = jp[2]
        jmat[3, c] = jr[0]
        jmat[4, c] = jr[1]
        jmat[5, c] = jr[2]
    return jmat


@wp.func
def damped_ls(jmat: mat67, b: vec6, lam: wp.float32) -> vec7:
    """Damped-LS joint velocities: ``dq = Jᵀ (J Jᵀ + λI)⁻¹ b``."""
    amat = mat66()
    for i in range(6):
        for j in range(6):
            s = 0.0
            for c in range(7):
                s += jmat[i, c] * jmat[j, c]
            if i == j:
                s += lam
            amat[i, j] = s

    x = chol_solve6(amat, b)

    dq = vec7()
    for c in range(7):
        v = 0.0
        for r in range(6):
            v += jmat[r, c] * x[r]
        dq[c] = v
    return dq


def damped_ls_host(
    jac: np.ndarray, twist: np.ndarray, damping: float
) -> np.ndarray:
    """Numpy twin of :func:`damped_ls` for the host control loop.

    Args:
        jac: 6x7 EE Jacobian (rows 0-2 linear, 3-5 rotational).
        twist: desired 6-vector spatial velocity.
        damping: least-squares damping ``λ``.

    Returns:
        The 7 joint velocities ``Jᵀ (J Jᵀ + λI)⁻¹ twist``.
    """
    reg = damping * np.eye(6)
    return jac.T @ np.linalg.solve(jac @ jac.T + reg, twist)


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate 3-vector ``v`` by quaternion ``q`` ([w, x, y, z])."""
    qv = q[1:4]
    t = 2.0 * np.cross(qv, v)
    return v + q[0] * t + np.cross(qv, t)


def solve_pose_ik(
    mj_model: mujoco.MjModel,
    target_pos: np.ndarray,
    target_quat: np.ndarray,
    *,
    ee_body_id: int,
    arm_qposadr: np.ndarray,
    arm_dofadr: np.ndarray,
    joint_limits: np.ndarray,
    q_seed: np.ndarray,
    site_id: int | None = None,
    damping: float = 1e-3,
    max_iters: int = 200,
    pos_tol: float = 1e-4,
    rot_tol: float = 1e-3,
    step_scale: float = 0.5,
    max_step: float | None = None,
    posture_gain: float = 0.0,
) -> tuple[np.ndarray, bool]:
    """Damped least-squares IK for an FR3 EE pose (numpy, no scipy).

    Host-only: allocates its own scratch ``MjData`` and never touches the
    device. Shared by every FR3 task, which all cache the same bookkeeping
    under the same names.

    Args:
        mj_model: The task's model.
        target_pos: Desired world position of the controlled point.
        target_quat: Desired orientation [w, x, y, z].
        ee_body_id: Body whose frame is being placed.
        arm_qposadr: qpos addresses of the 7 arm joints.
        arm_dofadr: dof addresses of the 7 arm joints.
        joint_limits: ``(7, 2)`` joint ranges; the solution is clamped to them.
        q_seed: Starting joint configuration.
        site_id: Drive position at this site instead of the body origin -- for
            a tool, the tip is the point you actually want placed.
        damping: Least-squares damping ``λ``.
        max_iters: Iteration cap.
        pos_tol: Position tolerance (m) for convergence.
        rot_tol: Rotation tolerance (rad) for convergence.
        step_scale: Fraction of the full Newton step taken per iteration.
        max_step: Optional per-joint step clamp (rad), which keeps a large
            initial error from throwing the solve across a singularity.
        posture_gain: Optional null-space pull back toward ``q_seed``, so the
            redundant DOF drifts away from the joint limits rather than into
            them. ``0.0`` disables it.

    Returns:
        ``(q_solution, converged)`` for the 7 FR3 joints. A failed solve
        returns its best effort rather than raising -- the caller decides.
    """
    q = np.asarray(q_seed, dtype=np.float64).copy()
    scratch = mujoco.MjData(mj_model)
    q_min, q_max = joint_limits[:, 0], joint_limits[:, 1]
    jacp = np.zeros((3, mj_model.nv))
    jacr = np.zeros((3, mj_model.nv))

    converged = False
    for _ in range(max_iters):
        scratch.qpos[arm_qposadr] = q
        mujoco.mj_forward(mj_model, scratch)
        if site_id is None:
            point = scratch.xpos[ee_body_id]
        else:
            point = scratch.site_xpos[site_id]
        ee_quat = scratch.xquat[ee_body_id]

        pos_err = target_pos - point
        res = np.zeros(3)
        mujoco.mju_subQuat(res, target_quat, ee_quat)
        rot_err = quat_rotate(ee_quat, res)
        if (
            np.linalg.norm(pos_err) < pos_tol
            and np.linalg.norm(rot_err) < rot_tol
        ):
            converged = True
            break

        if site_id is None:
            mujoco.mj_jacBody(mj_model, scratch, jacp, jacr, ee_body_id)
        else:
            # Jacobian at the site: a wrist rotation swings a tool tip, and
            # taking it at the body origin would miss exactly that coupling.
            mujoco.mj_jacSite(mj_model, scratch, jacp, jacr, site_id)
        jac = np.vstack([jacp, jacr])[:, arm_dofadr]
        twist = np.concatenate([pos_err, rot_err])
        dq = step_scale * damped_ls_host(jac, twist, damping)
        if posture_gain:
            j_pinv = jac.T @ np.linalg.inv(
                jac @ jac.T + damping * np.eye(6)
            )
            null = np.eye(7) - j_pinv @ jac
            # Faded out as the task error shrinks. With damping, `null` is only
            # an approximate null-space projector, so a constant pull leaks
            # into task space and parks the solve a few tenths of a mm short of
            # the target -- enough to open a gap where a contact was wanted.
            # Full strength while far away (where it earns its keep by steering
            # the redundant DOF off the limits), zero on approach.
            fade = min(1.0, np.linalg.norm(pos_err) / max(10.0 * pos_tol, 1e-9))
            dq = dq + null @ (fade * posture_gain * (np.asarray(q_seed) - q))
        if max_step is not None:
            dq = np.clip(dq, -max_step, max_step)
        q = np.clip(q + dq, q_min, q_max)

    return q, converged


def solve_task_pose_ik(
    task, target_pos: np.ndarray, target_quat: np.ndarray, q_seed=None,
    **overrides,
) -> tuple[np.ndarray, bool]:
    """:func:`solve_pose_ik` over an FR3 task's own cached IK geometry.

    Defaults ``q_seed`` to ``task.q_home``. For the extra knobs
    (``site_id``/``max_step``/``posture_gain``) pass them as ``overrides`` --
    ``PegFr3`` needs those and calls :func:`solve_pose_ik` directly instead,
    since it also targets a different body (``peg_body_id``, not
    ``ee_body_id``).
    """
    if q_seed is None:
        if task.q_home is None:
            raise RuntimeError("no q_seed and no 'home' keyframe in model")
        q_seed = task.q_home
    return solve_pose_ik(
        task.mj_model,
        target_pos,
        target_quat,
        ee_body_id=task.ee_body_id,
        arm_qposadr=task.arm_qposadr,
        arm_dofadr=task.arm_dofadr,
        joint_limits=task.joint_limits,
        q_seed=q_seed,
        damping=task.ik_damping,
        **overrides,
    )
