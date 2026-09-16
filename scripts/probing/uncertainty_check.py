"""Checks the state-uncertainty machinery against known-good answers.

Thirteen things, each of which has a silent failure mode:

1. ``quat_to_rotvec`` round-trips ``rotvec_to_quat``, including near-identity
   and near-pi rotations. The obvious ``arccos`` form of the log map is
   vertical at 1 and would fabricate a ~1e-8 floor.
2. ``PoseKalman`` reproduces a plain 12-state linear Kalman filter *exactly*
   on a rotation-free track. The manifold handling is what makes it look
   unusual; strip the rotation away and it must be the textbook filter.
3. It beats the raw observation on a constant-velocity track -- otherwise it
   is a no-op dressed up as an estimator.
4. Under a **biased** sensor it converges *to the bias* rather than removing
   it. This is the claim the debiased arm of
   ``experiments/push_fr3/state_uncertainty`` exists to test; if a plain
   Kalman filter could shed a bias, that arm would be meaningless.
5. The sweep's arms share one observation sequence per ``(seed, repeat)``,
   so they are *paired* -- an arm's advantage cannot be a luckier sensor.
   Checked on the raw draws rather than on the logged ``obs_pos_err``: that
   column is a difference against a truth the arms have already driven apart,
   so cancellation leaves ~1e-17 of rounding in it even when the draws are
   bit-identical. The debiased arm is required to differ, by exactly its bias.

The last three cover **pose-only sensing**, where the object's velocity is
never measured and has to be recovered from the pose sequence:

6. ``FiniteDifference`` returns the twist in MuJoCo's frames -- linear in the
   world frame, angular in the *body* frame. Checked on a track whose initial
   orientation is not the identity, so a world/body swap actually shows up;
   with a fixed rotation axis through the origin the two agree and the bug
   would hide. It is invisible in aggregate cost, which is what makes it
   dangerous.
7. A pose-only ``PoseKalman`` recovers the twist and beats raw differencing.
   Otherwise its constant-velocity model is not doing the estimating.
8. With ``observe_velocity: false`` the sensor leaves **exactly zero** object
   twist in the reading. Merely not perturbing it would pass the true
   velocity through, which is the whole failure this axis exists to avoid.
9. ``naive`` returns the pose **bit-identical** to the reading. It is the only
   arm that does not filter its pose, so this is what keeps naive naive; a
   leaked pose correction would still look perfectly reasonable.
10. Every arm at one replan rate derives the **same** object twist. Otherwise
    ensemble-vs-point compares differencing against Kalman smoothing rather
    than a raw pose against a filtered one.

The last two cover the **belief a filter claims**, which is what an ensemble
arm now plans over:

11. The cloud is centred on the filter's *estimate* (not the raw reading),
    spread by its *own* ``posterior_std``, and never wider than the filter's
    measurement sigma -- the arm deliberately does not clamp, so a covariance
    transient could otherwise hand the planner an enormous cloud.
12. Warm-up recovers the sensor's sigma without being told it, and does so
    identically for a biased and a debiased arm. This is what replaces
    handing the filter its true measurement covariance.
13. Turning the bias changes its **direction only** -- the magnitude holds to
    the last bit, ``k=4`` returns to the original, and the arms stay paired.
    A magnitude that drifted with direction would make the two move together
    and no table would separate them.

Run (from the repo root)::

    uv run python scripts/probing/uncertainty_check.py
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bampc.planner.base import StateSnapshot  # noqa: E402
from bampc.task.base import ObjectPose  # noqa: E402
from bampc.uncertainty import (  # noqa: E402
    FiniteDifference,
    PoseKalman,
    PosteriorGaussian,
    StateUncertainty,
    presets,
)
from bampc.uncertainty.noise import (  # noqa: E402
    _dof_of_qpos,
    quat_mul_batch,
    quat_to_rotvec,
    rotate_bias_xy,
    rotvec_to_quat,
)
from experiments.common.uncertainty.arms import (  # noqa: E402
    ARM_KINDS,
    build_arm,
)
from experiments.common.uncertainty.episode import warm_up  # noqa: E402
from experiments.common.uncertainty.loader import (  # noqa: E402
    load_run_config,
)

# One sensor, used by every check below.
POS_STD, ROT_STD, LV_STD, AV_STD = 0.004, 0.02, 0.02, 0.1
ACCEL_STD, ANG_ACCEL_STD = 0.2, 1.0
DT = 0.1  # the sweep's 10 Hz replan
LAYOUT = ObjectPose(kind="free", adr=0)
VEL = np.array([0.05, -0.02, 0.0])
OMEGA = np.array([0.0, 0.0, 0.3])
# A deliberately tilted starting orientation for the pose-only checks. With
# the identity here, body-frame and world-frame angular velocity would be the
# same vector and check 6 could not tell them apart.
Q0 = rotvec_to_quat(np.array([0.7, -0.4, 0.2]))[0]


def make_filter() -> PoseKalman:
    """A filter that knows its sensor exactly."""
    return PoseKalman(
        LAYOUT, 0, pos_std=POS_STD, rot_std=ROT_STD, lin_vel_std=LV_STD,
        ang_vel_std=AV_STD, accel_std=ACCEL_STD, ang_accel_std=ANG_ACCEL_STD,
    )


def observe(
    k: int, rng: np.random.Generator, bias: np.ndarray, spin: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One noisy reading of the constant-velocity track at step ``k``."""
    true_p = VEL * (k * DT)
    qpos, qvel = np.zeros(7), np.zeros(6)
    qpos[0:3] = true_p + bias + rng.normal(0.0, POS_STD, 3)
    if spin:
        true_q = rotvec_to_quat(OMEGA * (k * DT))[0]
        perturb = rotvec_to_quat(rng.normal(0.0, ROT_STD, 3))
        qpos[3:7] = quat_mul_batch(perturb, true_q[None])[0]
        qvel[3:6] = OMEGA + rng.normal(0.0, AV_STD, 3)
    else:
        qpos[3] = 1.0
    qvel[0:3] = VEL + rng.normal(0.0, LV_STD, 3)
    return qpos, qvel, true_p


def check_log_map() -> None:
    """Round-trip the exp/log maps over the full rotation range."""
    rng = np.random.default_rng(0)
    axes = rng.normal(size=(2000, 3))
    axes /= np.linalg.norm(axes, axis=1, keepdims=True)
    angles = np.concatenate([
        rng.uniform(0.0, np.pi, 1900),
        np.full(50, 1e-12),           # near identity
        np.full(50, np.pi - 1e-7),    # near pi
    ])
    w = axes * angles[:, None]
    q = rotvec_to_quat(w)
    err = np.linalg.norm(quat_to_rotvec(q) - w, axis=1).max()
    assert err < 1e-9, f"round-trip error {err:.3e}"
    # q and -q are the same rotation and must give the same log.
    assert np.allclose(quat_to_rotvec(q), quat_to_rotvec(-q))
    print(f"1. log/exp round-trip   max err {err:.2e}  (q/-q agree)")


def check_matches_linear_kf() -> None:
    """With no rotation, PoseKalman must BE the textbook linear filter."""
    n = 60
    meas_cov = np.diag(np.r_[
        [POS_STD**2] * 3, [ROT_STD**2] * 3,
        [LV_STD**2] * 3, [AV_STD**2] * 3,
    ])
    f = np.eye(12)
    f[0:6, 6:12] = np.eye(6) * DT
    q = np.diag(np.r_[
        [ACCEL_STD**2 * (DT**2 / 2) ** 2] * 3,
        [ANG_ACCEL_STD**2 * (DT**2 / 2) ** 2] * 3,
        [ACCEL_STD**2 * DT**2] * 3,
        [ANG_ACCEL_STD**2 * DT**2] * 3,
    ])

    kf = make_filter()
    rng = np.random.default_rng(3)
    x, cov, worst = None, np.eye(12), 0.0
    for k in range(n):
        qpos, qvel, _ = observe(k, rng, np.zeros(3), spin=False)
        z = np.r_[qpos[0:3], np.zeros(3), qvel[0:6]]
        if x is None:
            # Adopt the first reading, and with it the reading's own
            # covariance -- the same convention PoseKalman.update uses. The
            # textbook filter has no opinion here; the two just have to agree.
            x, cov = z.copy(), meas_cov.copy()
        else:
            x = f @ x
            cov = f @ cov @ f.T + q
            gain = cov @ np.linalg.inv(cov + meas_cov)
            x = x + gain @ (z - x)
            cov = (np.eye(12) - gain) @ cov
        mine_q, mine_v = kf.update(qpos, qvel, DT)
        got = np.r_[mine_q[0:3], np.zeros(3), mine_v[0:6]]
        worst = max(worst, float(np.abs(got - x).max()))
    assert worst < 1e-12, f"diverged from the linear KF by {worst:.3e}"
    print(f"2. vs textbook linear KF  max |diff| {worst:.2e}")


def track(bias: np.ndarray, seed: int = 1, n: int = 300) -> tuple:
    """Raw vs filtered error: ``(pos_raw, pos_kf, vel_raw, vel_kf)``."""
    rng = np.random.default_rng(seed)
    kf = make_filter()
    out = ([], [], [], [])
    for k in range(n):
        qpos, qvel, true_p = observe(k, rng, bias, spin=True)
        est_q, est_v = kf.update(qpos, qvel, DT)
        out[0].append(np.linalg.norm(qpos[0:3] - true_p))
        out[1].append(np.linalg.norm(est_q[0:3] - true_p))
        out[2].append(np.linalg.norm(qvel[0:3] - VEL))
        out[3].append(np.linalg.norm(est_v[0:3] - VEL))
    # Drop the transient: the filter adopts the first reading verbatim.
    return tuple(np.array(v)[50:] for v in out)


def check_beats_raw() -> None:
    """The filter must actually reduce the error it is given."""
    p_raw, p_kf, v_raw, v_kf = track(np.zeros(3))
    p_gain = p_raw.mean() / p_kf.mean()
    v_gain = v_raw.mean() / v_kf.mean()
    print(
        f"3. unbiased sensor      pos {p_raw.mean()*1e3:5.2f} -> "
        f"{p_kf.mean()*1e3:5.2f} mm ({p_gain:.1f}x)   "
        f"vel {v_raw.mean()*1e3:5.1f} -> {v_kf.mean()*1e3:5.1f} mm/s "
        f"({v_gain:.1f}x)"
    )
    assert p_gain > 1.2, f"position gain only {p_gain:.2f}x"
    assert v_gain > 1.0, f"velocity gain only {v_gain:.2f}x"


def check_cannot_shed_bias() -> None:
    """A Kalman filter converges to truth + bias. It cannot remove it."""
    bias = np.array([0.005, -0.003, 0.0])
    mag = float(np.linalg.norm(bias))
    _, p_kf, _, _ = track(bias)
    settled = float(p_kf[-50:].mean())
    print(
        f"4. biased sensor        kf settles at {settled*1e3:.2f} mm, "
        f"|bias| = {mag*1e3:.2f} mm -> the bias survives"
    )
    assert abs(settled - mag) < 0.5 * mag, (
        f"settled at {settled:.4f}, expected ~{mag:.4f}"
    )


def sweep_config(version: str = "t-ou/scale0.6-tau2.0-warmup50"):
    """A real sweep config -- that is the thing that has to be right."""
    root = Path(__file__).resolve().parents[2]
    return load_run_config(
        root / "experiments" / "push_fr3" / "state_uncertainty"
        / version / "config.yaml"
    )


def check_arms_are_paired() -> None:
    """Every arm must read the same sensor draws at one (seed, repeat).

    Compares the raw draws, not the logged error column.
    """
    cfg = sweep_config()
    task = cfg.make_prediction_task()
    state = StateSnapshot(
        qpos=np.zeros(task.mj_model.nq), qvel=np.zeros(task.mj_model.nv),
        time=0.0,
    )
    state.qpos[3] = 1.0  # unit quaternion for the free block

    draws = {}
    for name in cfg.arms:
        arm = build_arm(cfg, name, task)
        if arm.sensor is None:
            continue
        arm.reset(seed=3, repeat=2)
        draws[name] = [arm.sensor.sample(state)[0][0].copy() for _ in range(8)]

    ref_name = "naive"
    ref = draws[ref_name]
    for name, seq in draws.items():
        gap = max(float(np.abs(a - b).max()) for a, b in zip(ref, seq))
        if name.endswith("_debiased"):
            assert gap > 1e-4, f"{name} must differ from {ref_name}: its bias"
        else:
            assert gap == 0.0, f"{name} vs {ref_name} differ by {gap:.3e}"
    biased = [n for n in draws if n.endswith("_debiased")]
    print(
        f"5. arms paired          {len(draws) - len(biased)} arms bit-identical"
        f" to {ref_name}; {biased} differ by the bias, as intended"
    )


def pose_track(
    k: int, rng: np.random.Generator, pos_std: float = 0.0,
    rot_std: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Reading ``k`` of a constant-twist track, POSE ONLY.

    The orientation is ``Q0 * exp(OMEGA t)``, i.e. a constant *body-frame*
    angular velocity off a tilted start -- which is what makes the two frames
    distinguishable. The returned twist is zeros, as a pose-only sensor
    leaves it.
    """
    qpos, qvel = np.zeros(7), np.zeros(6)
    qpos[0:3] = VEL * (k * DT)
    q = quat_mul_batch(Q0[None], rotvec_to_quat(OMEGA * (k * DT)))[0]
    if pos_std:
        qpos[0:3] += rng.normal(0.0, pos_std, 3)
    if rot_std:
        perturb = rotvec_to_quat(rng.normal(0.0, rot_std, 3))
        q = quat_mul_batch(perturb, q[None])[0]
    qpos[3:7] = q
    return qpos, qvel


def check_finite_difference() -> None:
    """FD must return the twist in MuJoCo's frames, exactly."""
    fd = FiniteDifference(LAYOUT, 0)
    rng = np.random.default_rng(0)
    worst_lin = worst_ang = 0.0
    for k in range(20):
        qpos, qvel = pose_track(k, rng)
        _, est_v = fd.update(qpos, qvel, DT)
        if k == 0:
            assert not est_v.any(), "the first reading must give zero velocity"
            continue
        worst_lin = max(worst_lin, float(np.abs(est_v[0:3] - VEL).max()))
        worst_ang = max(worst_ang, float(np.abs(est_v[3:6] - OMEGA).max()))
    print(
        f"6. finite difference    max err  lin {worst_lin:.2e} m/s   "
        f"ang {worst_ang:.2e} rad/s  (body frame)"
    )
    assert worst_lin < 1e-11, f"linear off by {worst_lin:.3e}"
    assert worst_ang < 1e-11, (
        f"angular off by {worst_ang:.3e} -- a world/body frame swap?"
    )


def check_pose_only_kalman() -> None:
    """Without a twist reading, the filter must still recover the twist."""
    kf = PoseKalman(
        LAYOUT, 0, pos_std=POS_STD, rot_std=ROT_STD,
        accel_std=ACCEL_STD, ang_accel_std=ANG_ACCEL_STD,
        observe_twist=False,
    )
    fd = FiniteDifference(LAYOUT, 0)
    rng = np.random.default_rng(5)
    e_kf, e_fd = [], []
    for k in range(300):
        qpos, qvel = pose_track(k, rng, POS_STD, ROT_STD)
        _, v_kf = kf.update(qpos, qvel, DT)
        _, v_fd = fd.update(qpos, qvel, DT)
        if k < 50:  # drop the transient: it starts from zero velocity
            continue
        e_kf.append(np.linalg.norm(v_kf[0:3] - VEL))
        e_fd.append(np.linalg.norm(v_fd[0:3] - VEL))
    kf_err, fd_err = float(np.mean(e_kf)), float(np.mean(e_fd))
    print(
        f"7. pose-only Kalman     vel err {fd_err*1e3:5.1f} (fd) -> "
        f"{kf_err*1e3:5.1f} mm/s ({fd_err/kf_err:.1f}x), "
        f"|VEL| = {np.linalg.norm(VEL)*1e3:.0f} mm/s"
    )
    assert fd_err / kf_err > 2.0, f"only {fd_err / kf_err:.2f}x better than fd"
    # A loose bar on purpose. The sharp claim is the ratio above; this one
    # only says the estimate is a velocity rather than noise around zero, and
    # raw differencing (~1.8x |VEL|) fails it. The filter lands near what
    # check 3's *twist-observing* filter manages, which is the real headline:
    # the constant-velocity model recovers about as much from poses alone.
    assert kf_err < 0.5 * float(np.linalg.norm(VEL)), (
        f"kf velocity error {kf_err:.4f} is not tracking"
    )


def check_pose_only_sensor_hides_twist() -> None:
    """`observe_velocity: false` must ERASE the object twist, not keep it.

    Forced on rather than read off the config, so this tests the mechanism
    whatever the config says.
    """
    cfg = sweep_config()
    cfg = replace(
        cfg, sensor=replace(cfg.sensor, observe_velocity=False)
    )
    task = cfg.make_prediction_task()
    layout = task.object_pose_qpos
    adr = _dof_of_qpos(task.mj_model, layout.adr)

    # A state whose object really is moving, so zeros in the reading can only
    # come from the sensor erasing them.
    state = StateSnapshot(
        qpos=np.zeros(task.mj_model.nq),
        qvel=np.full(task.mj_model.nv, 0.37),
        time=0.0,
    )
    state.qpos[layout.adr + 3] = 1.0

    for name in cfg.arms:
        arm = build_arm(cfg, name, task)
        if arm.sensor is None:
            continue
        arm.reset(seed=1, repeat=0)
        obs_qvel = arm.sensor.sample(state)[1][0]
        twist = obs_qvel[adr : adr + 6]
        assert not twist.any(), f"{name} leaked object twist {twist}"
        others = np.delete(obs_qvel, np.arange(adr, adr + 6))
        assert others.any(), f"{name} zeroed more than the object's twist"
    print(
        "8. pose-only sensor     object twist erased in every arm; "
        "joint velocities untouched"
    )


def _arm_filters(cfg, task):
    """Every non-oracle arm's filter, reset, keyed by name."""
    out = {}
    for name in cfg.arms:
        arm = build_arm(cfg, name, task)
        if arm.filt is None:
            continue
        arm.reset(seed=0, repeat=0)
        out[name] = arm
    return out


def check_raw_pose_survives() -> None:
    """``naive`` must return the observed pose bit-identical.

    It is the only arm left that does not filter its pose -- the ensembles
    now plan around the filter's estimate on purpose -- so this is what keeps
    "naive" meaning naive. A pose correction leaking into it would still look
    perfectly reasonable, which is why it is asserted rather than assumed.
    """
    cfg = sweep_config()
    task = cfg.make_prediction_task()
    layout = task.object_pose_qpos
    a = layout.adr
    nq, nv = task.mj_model.nq, task.mj_model.nv
    rng = np.random.default_rng(11)

    raw_pose_arms = [n for n in cfg.arms if ARM_KINDS[n] == "passthrough"]
    arms = _arm_filters(cfg, task)
    worst = 0.0
    for k in range(20):
        qpos, qvel = np.zeros(nq), np.zeros(nv)
        qpos[a : a + 7] = pose_track(k, rng, POS_STD, ROT_STD)[0][0:7]
        for name in raw_pose_arms:
            est_q, _ = arms[name].filt.update(qpos, qvel, 0.1)
            worst = max(worst, float(np.abs(est_q - qpos).max()))
    print(
        f"9. raw pose survives    {raw_pose_arms} return the observed pose "
        f"exactly (max |diff| {worst:.1e})"
    )
    assert worst == 0.0, f"a pose correction leaked through: {worst:.3e}"


def check_twist_is_controlled() -> None:
    """Every arm at one replan rate must derive the SAME object twist.

    The confound this exists to remove: if `naive` and the ensembles
    finite-difference while the point arms filter, then
    ensemble-vs-point compares differencing against Kalman smoothing rather
    than a raw pose against a filtered one. Holding the twist constant is
    asserted here rather than assumed, because nothing else would catch it.

    ``point_fast`` is excluded on purpose -- it replans at 20 Hz, so its
    filter integrates over a different ``dt`` and *should* differ.
    """
    cfg = sweep_config()
    task = cfg.make_prediction_task()
    layout = task.object_pose_qpos
    a = layout.adr
    d = _dof_of_qpos(task.mj_model, layout.adr)
    nq, nv = task.mj_model.nq, task.mj_model.nv
    rng = np.random.default_rng(11)

    arms = _arm_filters(cfg, task)
    base_dt = 1.0 / cfg.plan_freq_hz
    same_rate = [
        n for n in arms if abs(cfg.plan_freq_for(n) - cfg.plan_freq_hz) < 1e-9
    ]
    twists: dict[str, list] = {n: [] for n in same_rate}
    for k in range(30):
        qpos, qvel = np.zeros(nq), np.zeros(nv)
        qpos[a : a + 7] = pose_track(k, rng, POS_STD, ROT_STD)[0][0:7]
        for name in same_rate:
            _, est_v = arms[name].filt.update(qpos, qvel, base_dt)
            twists[name].append(est_v[d : d + 6].copy())

    ref = twists["naive"]
    gaps = [
        float(np.abs(x - y).max())
        for seq in twists.values()
        for x, y in zip(ref, seq)
    ]
    worst = max(gaps)
    print(
        f"10. twist controlled     {len(same_rate)} arms at "
        f"{cfg.plan_freq_hz:g} Hz agree to {worst:.1e} "
        f"(estimator: {cfg.sensor.twist_estimator})"
    )
    if cfg.sensor.twist_estimator == "kalman":
        assert worst < 1e-12, (
            f"arms derive different twists (max {worst:.3e}) -- "
            "ensemble-vs-point is confounded by velocity estimation"
        )


def check_posterior_cloud() -> None:
    """An ensemble's cloud must be the filter's own posterior, and no wider.

    Three separate ways this can be silently wrong:

    * The cloud sits on the **raw reading** instead of the filtered estimate,
      which is the bug the whole redesign exists to remove -- it would look
      identical in every figure.
    * Its spread does not match ``posterior_std()``, so "the belief the filter
      claims" is not what the planner rolls out.
    * The spread exceeds the filter's own measurement sigma. It never can
      after a single update (``P_post = P (P + R)^-1 P <= R``), so this is the
      guard against a covariance transient handing the planner a cloud far
      wider than the reading it came from. The arm deliberately does not clamp
      -- a clamp would read a magnitude the filter is not entitled to -- so
      the property is asserted here instead.
    """
    cfg = sweep_config()
    task = cfg.make_prediction_task()
    layout = task.object_pose_qpos
    a = layout.adr
    nq, nv = task.mj_model.nq, task.mj_model.nv

    arm = build_arm(cfg, "ensemble_exact", task)
    arm.reset(seed=0, repeat=0)
    kalman = arm.kalman
    scale = cfg.belief.sigma_for("ensemble_exact")
    # A far bigger cloud than the sweep's R=16, so the sample statistics below
    # measure the noise model rather than 16 draws' luck.
    big = StateUncertainty(
        task, 4000, [PosteriorGaussian(kalman, scale)], seed=7
    )
    meas = np.sqrt(np.diag(kalman.meas_cov))
    rng = np.random.default_rng(11)

    worst_centre = worst_std = 0.0
    worst_ratio = 0.0
    for k in range(30):
        qpos, qvel = np.zeros(nq), np.zeros(nv)
        qpos[a : a + 7] = pose_track(k, rng, POS_STD, ROT_STD)[0][0:7]
        raw = qpos[a : a + 3].copy()
        est_q, est_v = arm.filt.update(qpos, qvel, DT)
        cloud, _ = big.sample(StateSnapshot(qpos=est_q, qvel=est_v, time=0.0))

        centre = cloud[:, a : a + 3].mean(axis=0)
        claimed = kalman.posterior_std()["pos"] * scale
        # 4000 draws put the sample mean within ~4 sigma/63 of the centre.
        worst_centre = max(
            worst_centre,
            float(np.abs(centre - est_q[a : a + 3]).max() / claimed.max()),
        )
        worst_std = max(
            worst_std,
            float(np.abs(cloud[:, a : a + 3].std(axis=0) / claimed - 1).max()),
        )
        worst_ratio = max(
            worst_ratio, float((claimed / (meas[0:3] * scale)).max())
        )
        if k:  # the first update adopts the reading, so they coincide
            assert np.abs(centre - raw).max() > 1e-6, (
                "the cloud is centred on the RAW reading, not the estimate"
            )

    print(
        f"11. posterior cloud      centre within {worst_centre:.3f} sigma, "
        f"spread within {worst_std:.1%} of posterior_std, "
        f"widest {worst_ratio:.2f}x the measurement sigma"
    )
    assert worst_centre < 0.1, "the cloud is not centred on the estimate"
    assert worst_std < 0.1, "the cloud's spread is not the claimed posterior"
    assert worst_ratio <= 1.0, (
        f"the cloud is {worst_ratio:.2f}x the measurement sigma -- a "
        "posterior cannot exceed it, so the filter is broken"
    )


def check_warmup_calibration() -> None:
    """Warm-up must recover the sensor's sigma without being told it.

    The filter is no longer handed ``presets.gaussian_stds``; it measures its
    own measurement covariance from readings of a state known to be still.
    This reports how well, across seeds, so ``kalman.warmup_steps`` can be
    chosen from the spread rather than from the ``sqrt(2/(N-1))`` estimate.

    Also asserts the calibration is **bias-immune**: a variance about the
    sample mean cannot see a constant offset, which is what keeps a debiased
    arm paired with its biased twin.
    """
    cfg = sweep_config()
    task = cfg.make_prediction_task()
    nq, nv = task.mj_model.nq, task.mj_model.nv
    qpos, qvel = np.zeros(nq), np.zeros(nv)
    qpos[task.object_pose_qpos.adr + 3] = 1.0
    n = cfg.kalman.warmup_steps

    truth = presets.gaussian_stds(
        build_arm(cfg, "point", task).sensor.noise
    )
    ratios: dict[str, list[float]] = {"point": [], "point_debiased": []}
    for name, out in ratios.items():
        for seed in range(40):
            arm = build_arm(cfg, name, task)
            arm.reset(seed=seed, repeat=0)
            warm_up(task, arm, qpos, qvel, 0.0, n, 1.0 / cfg.plan_freq_hz)
            got = np.sqrt(np.diag(arm.kalman.meas_cov))
            out.append(got[0:3].mean() / truth["pos_std"])

    lo, med, hi = np.percentile(ratios["point"], [5, 50, 95])
    print(
        f"12. warm-up calibration  N={n}: recovered pos sigma / true = "
        f"{med:.3f} [{lo:.3f}, {hi:.3f}] (5-95%), "
        f"predicted spread +/-{np.sqrt(2.0 / (n - 1)) / 2:.1%}"
    )
    assert 0.8 < med < 1.25, f"calibration is biased: median {med:.3f}"
    gap = float(
        np.abs(np.array(ratios["point"]) - ratios["point_debiased"]).max()
    )
    assert gap < 1e-12, (
        f"debiasing changed the calibration by {gap:.3e} -- a variance about "
        "the sample mean must not see a constant offset"
    )


def check_bias_rotation() -> None:
    """Turning the bias must change only its direction.

    Four ways this can be quietly wrong:

    * The magnitude drifts between directions, which would make direction and
      severity move together and no table would show it.
    * ``k`` does not cycle with period 4, so "four directions" is really
      three-and-a-bit.
    * The rotation perturbs the RNG stream, which would unpair the arms --
      the whole reason `(seed, repeat)` names one shared observation sequence.
    * It touches a debiased arm, which has no bias to turn and must stay
      bit-paired with its biased twin.
    """
    cfg = sweep_config()
    task = cfg.make_prediction_task()
    state = StateSnapshot(
        qpos=np.zeros(task.mj_model.nq), qvel=np.zeros(task.mj_model.nv),
        time=0.0,
    )
    state.qpos[task.object_pose_qpos.adr + 3] = 1.0

    arm = build_arm(cfg, "point", task)
    _, original = arm.bias0[0]
    mag0 = float(np.linalg.norm(np.asarray(original.pos, float)[:2]))
    seen, worst_mag = [], 0.0
    for k in range(5):
        turned = rotate_bias_xy(original, k)
        p = np.asarray(turned.pos, float)
        worst_mag = max(
            worst_mag, abs(float(np.linalg.norm(p[:2])) - mag0)
        )
        seen.append(tuple(p))
        assert p[2] == original.pos[2], "z moved"
        assert turned.rot == original.rot, "the yaw bias turned too"
    assert seen[4] == seen[0], "k=4 did not return to the original"
    assert len({s[:2] for s in seen[:4]}) == 4, "directions repeat within k<4"

    # Same (seed, repeat) across arms -> identical draws, whatever the bias.
    draws = {}
    for name in ("point", "naive", "ensemble_exact", "point_debiased"):
        a = build_arm(cfg, name, task)
        a.reset(seed=3, repeat=2)  # repeat 2 -> a half turn
        draws[name] = a.sensor.sample(state)[0][0].copy()
    for name in ("naive", "ensemble_exact"):
        gap = float(np.abs(draws[name] - draws["point"]).max())
        assert gap == 0.0, f"{name} unpaired from point by {gap:.3e}"
    deb = build_arm(cfg, "point_debiased", task)
    assert not deb.bias0, "a debiased arm should carry no bias to turn"

    print(
        f"13. bias rotation        4 distinct directions, |xy| stable to "
        f"{worst_mag:.1e} m, k=4 returns, arms stay paired"
    )


def main() -> None:
    """Run every check, loudly."""
    check_log_map()
    check_matches_linear_kf()
    check_beats_raw()
    check_cannot_shed_bias()
    check_arms_are_paired()
    check_finite_difference()
    check_pose_only_kalman()
    check_pose_only_sensor_hides_twist()
    check_raw_pose_survives()
    check_twist_is_controlled()
    check_posterior_cloud()
    check_warmup_calibration()
    check_bias_rotation()
    print("\nall state-uncertainty checks passed")


if __name__ == "__main__":
    main()
