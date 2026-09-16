"""Check that PoseKalman's online-adaptive meas_cov tracks noise bursts.

``adapt_rate > 0`` folds each step's innovation back into ``meas_cov`` via
an EWMA (see :class:`~bampc.uncertainty.filter.PoseKalman`'s
docstring). This verifies the mechanism does what it claims on synthetic
data with a KNOWN noise schedule, independent of any GPU sweep or fitted
sensor model: feed the filter a stationary true pose corrupted by
CLEAR-level Gaussian position
noise, then a BURST of much larger noise, then back to CLEAR, and check that
the filter's own claimed measurement std actually rises during the burst and
decays back down afterward -- not just that it is nonzero.

Run::

    uv run python scripts/probing/adaptive_kalman_check.py
"""

from __future__ import annotations

import numpy as np

from bampc.config import numerics
from bampc.task.push_fr3 import PushFr3
from bampc.uncertainty import PoseKalman


def main() -> None:
    """Feed a synthetic CLEAR/BURST/CLEAR schedule; check adaptation."""
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

    rng = np.random.default_rng(0)
    dt = 0.1
    clear_std, burst_std = 0.002, 0.03  # position std (m)
    n_clear1, n_burst, n_clear2 = 60, 40, 80

    true_pos = np.array([0.4, 0.0, 0.05])
    true_quat = np.array([1.0, 0.0, 0.0, 0.0])

    kf = PoseKalman(
        layout,
        dof,
        pos_std=clear_std,
        rot_std=0.01,
        observe_twist=False,
        adapt_rate=0.1,
    )

    regime = ["clear"] * n_clear1 + ["burst"] * n_burst + ["clear"] * n_clear2
    meas_std_trace = []
    for r in regime:
        std = burst_std if r == "burst" else clear_std
        qpos = np.zeros(task.mj_model.nq)
        qpos[a : a + 3] = true_pos + rng.normal(scale=std, size=3)
        qpos[a + 3 : a + 7] = true_quat
        qvel = np.zeros(task.mj_model.nv)
        kf.update(qpos, qvel, dt)
        meas_std_trace.append(float(np.sqrt(np.diag(kf.meas_cov)[0])))

    meas_std_trace = np.array(meas_std_trace)
    clear1_end, burst_end = n_clear1, n_clear1 + n_burst
    pre_burst = meas_std_trace[clear1_end - 10 : clear1_end].mean()
    during_burst_end = meas_std_trace[burst_end - 5 : burst_end].mean()
    post_settle = meas_std_trace[-10:].mean()

    checks = {
        "inflates during burst": during_burst_end > 2.0 * pre_burst,
        "deflates back after burst": post_settle < 0.5 * during_burst_end,
        "post-burst settles near pre-burst level": (
            abs(post_settle - pre_burst) < 0.5 * pre_burst
        ),
    }
    print(f"  true clear_std = {clear_std}, true burst_std = {burst_std}")
    print(f"  pre-burst meas_std    = {pre_burst:.5f}")
    print(f"  end-of-burst meas_std = {during_burst_end:.5f}")
    print(f"  post-burst meas_std   = {post_settle:.5f}")
    for name, ok in checks.items():
        print(f"  [{'ok' if ok else 'FAIL'}] {name}")
    if not all(checks.values()):
        raise SystemExit("adaptive Kalman check FAILED")
    print("all checks passed")


if __name__ == "__main__":
    main()
