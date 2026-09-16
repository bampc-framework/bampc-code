"""Balance-FR3's real-robot ball sensor, shared by all three example regimes.

The real node (``ros.adapters.balance_fr3.make_state_reader``) reads only the
sphere's position and derives the rest: linear velocity from a Kalman filter
or a raw finite difference, angular velocity from the no-slip rolling
constraint. :func:`rolling_observer` is that chain fed by a sim sensor.

``state_uncertainty/balance_fr3.py`` wires it into its belief stack;
``simple/`` and ``domain_randomization/`` opt in with ``--rolling-sensor``
via :func:`add_rolling_sensor_args` and :func:`build_rolling_observer`.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable

import mujoco

from bampc.planner.base import StateSnapshot
from bampc.ros.adapters.balance_fr3 import _rolling_angular_velocity
from bampc.task.balance_fr3 import BalanceFr3
from bampc.uncertainty import StateUncertainty, presets
from bampc.uncertainty.filter import FiniteDifference, PoseKalman
from bampc.uncertainty.noise import _dof_of_qpos
from examples.state_uncertainty.common import (
    add_uncertainty_args,
    build_filter,
    build_sensor,
    warm_up,
)


def rolling_observer(
    task: BalanceFr3,
    sensor: StateUncertainty,
    dt: float,
    kalman: PoseKalman | None = None,
) -> Callable[[StateSnapshot], StateSnapshot]:
    """The real robot's sphere chain, fed by the sim sensor instead of TF.

    ``kalman`` (or a raw finite difference when ``None``) gives linear
    velocity; the no-slip constraint gives angular velocity from *that same*
    linear velocity, or the rollout starts from a spin that contradicts its
    own forward speed.
    """
    mj_model = task.mj_model
    layout = task.object_pose_qpos
    dof_adr = _dof_of_qpos(mj_model, layout.adr)
    plate_body_id = mj_model.body("plate_body").id
    ball_radius = float(mj_model.geom("block_ball").size[0])
    kin_data = mujoco.MjData(mj_model)
    fd = None if kalman is not None else FiniteDifference(layout, dof_adr)

    def observe(state: StateSnapshot) -> StateSnapshot:
        qpos, qvel = sensor.sample(state)
        qpos, qvel = qpos[0], qvel[0]
        if kalman is not None:
            qpos, qvel = kalman.update(qpos, qvel, dt)
        else:
            qpos, qvel = fd.update(qpos, qvel, dt)
        lin_vel = qvel[dof_adr : dof_adr + 3]
        a = layout.adr
        qvel[dof_adr + 3 : dof_adr + 6] = _rolling_angular_velocity(
            mj_model,
            kin_data,
            plate_body_id,
            ball_radius,
            qpos,
            lin_vel,
            qpos[a + 3 : a + 7],
        )
        return StateSnapshot(
            qpos=qpos,
            qvel=qvel,
            time=state.time,
            mocap_pos=state.mocap_pos,
            mocap_quat=state.mocap_quat,
        )

    return observe


def add_rolling_sensor_args(parser: argparse.ArgumentParser) -> None:
    """Add ``--rolling-sensor`` and the sensor/filter flags it reads."""
    parser.add_argument(
        "--rolling-sensor", action="store_true",
        help="Sphere only: the planner sees the ball through the real "
        "robot's position-only sensor chain instead of the true state "
        "(see examples.rolling_sensor).",
    )
    parser.add_argument(
        "--noise", default="pose-rolling-hedge", choices=presets.names(),
        help="Sensor-noise preset for --rolling-sensor.",
    )
    parser.add_argument(
        "--noise-scale", type=float, default=1.0,
        help="Multiplies every --noise magnitude.",
    )
    parser.add_argument(
        "--filter", default="kalman", choices=["kalman", "none"],
        help="Linear velocity from a Kalman filter or a raw finite "
        "difference (the real node's kalman/passthrough).",
    )
    parser.add_argument(
        "--warmup-steps", type=int, default=50,
        help="Still readings used to measure the Kalman filter's "
        "measurement covariance before planning starts.",
    )


def build_rolling_observer(
    task: BalanceFr3, args: argparse.Namespace, mj_data, dt: float
) -> tuple[Callable[[StateSnapshot], StateSnapshot], PoseKalman | None]:
    """Build and warm up ``(observer, kalman)`` for ``--rolling-sensor``.

    Flags this example does not expose take the state-uncertainty example's
    defaults, so all three regimes build the same sensor.
    """
    defaults = argparse.ArgumentParser()
    add_uncertainty_args(defaults)
    full = defaults.parse_args([])
    vars(full).update(vars(args))

    sensor, noise = build_sensor(task, full)
    point, kalman = (
        build_filter(task, full, noise, dt)
        if full.filter == "kalman"
        else (None, None)
    )
    warm_up(task, sensor, point, kalman, mj_data, full.warmup_steps, dt)
    return rolling_observer(task, sensor, dt, kalman), kalman
