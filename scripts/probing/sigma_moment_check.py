"""Check that a SigmaPointBelief reproduces its filter's pose posterior.

The unscented sigma set is only worth using if it actually carries the
posterior it claims to: the ``2n+1`` points, under the UKF weights, must
reproduce the filter's pose **mean and covariance** before any settle. This
verifies that on a primed
:class:`~bampc.uncertainty.filter.PoseKalman`, so a failure here is
caught before a GPU run reads meaning into the cloud.

Run::

    uv run python scripts/probing/sigma_moment_check.py
"""

from __future__ import annotations

import mujoco
import numpy as np

from bampc.config import numerics
from bampc.planner.base import StateSnapshot
from bampc.task.push_fr3 import PushFr3
from bampc.uncertainty import PoseKalman, SigmaPointBelief
from bampc.uncertainty.noise import quat_mul_batch, quat_to_rotvec


def main() -> None:
    """Prime a filter, place the sigma set, and check the two moments."""
    task = PushFr3(
        manipulation_type="free", model_config=numerics.load("fr3_free")
    )
    layout = task.object_pose_qpos
    a = layout.adr
    dof = next(
        int(task.mj_model.jnt_dofadr[j])
        for j in range(task.mj_model.njnt)
        if int(task.mj_model.jnt_qposadr[j]) == a
    )
    kf = PoseKalman(
        layout, dof, pos_std=0.01, rot_std=0.05, observe_twist=False
    )
    md = mujoco.MjData(task.mj_model)
    mujoco.mj_forward(task.mj_model, md)
    qpos, qvel = md.qpos.copy(), md.qvel.copy()
    for _ in range(10):
        kf.update(qpos, qvel, 0.1)
    p6 = kf.posterior_cov()[:6, :6]

    bel = SigmaPointBelief(task, 13, filt=kf, seed=0)
    qp, qv = bel.sample(StateSnapshot(qpos=qpos, qvel=qvel, time=0.0))

    # Tangent offsets of each member from the estimate (world-frame residual,
    # the convention _apply_pose placed them with).
    dp = qp[:, a : a + 3] - qpos[a : a + 3]
    conj = qpos[a + 3 : a + 7] * np.array([1.0, -1.0, -1.0, -1.0])
    inv = np.tile(conj, (13, 1))
    drot = quat_to_rotvec(quat_mul_batch(qp[:, a + 3 : a + 7], inv))
    x = np.concatenate([dp, drot], axis=1)
    wm, wc = bel.weights()
    mean = wm @ x
    cov = np.einsum("i,ij,ik->jk", wc, x - mean, x - mean)

    checks = {
        "members == 13": qp.shape[0] == 13,
        "mean ~ estimate": np.abs(mean).max() < 1e-9,
        "cov == posterior": np.allclose(cov, p6, atol=1e-9),
        "domain 0 == estimate": np.allclose(qp[0, a : a + 7], qpos[a : a + 7]),
        "velocity untouched": np.allclose(qv, np.tile(qvel, (13, 1))),
    }
    for name, ok in checks.items():
        print(f"  [{'ok' if ok else 'FAIL'}] {name}")
    print(f"  cov max|diff| = {np.abs(cov - p6).max():.2e}")
    if not all(checks.values()):
        raise SystemExit("sigma moment check FAILED")
    print("all checks passed")


if __name__ == "__main__":
    main()
