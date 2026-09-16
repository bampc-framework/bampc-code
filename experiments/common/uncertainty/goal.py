"""Drives a task's goal mocap through a headless episode.

The interactive viewer refreshes the goal mocap from ``task.goal_mocap_pose``
each frame; sweeps have no viewer, so this reproduces it and feeds the pose
to the planner via ``StateSnapshot`` (so the rollout cost tracks the live
goal).

Two contracts have to hold at once, which is why the pose is always asked
for rather than gated on drift being on:

* Push / Push-FR3 own a world mocap that is already at its compiled pose, so
  ``Task.goal_mocap_pose`` returns ``None`` with drift off and there is
  nothing to write.
* Balance's goal is a world mocap that must ride the plate's tilt, so
  ``Balance.goal_mocap_pose`` always returns a pose and reads the live plate
  frame off an ``MjData``.

So the task decides, and it is handed a forward-kinematics'd scratch state to
decide with.
"""

from __future__ import annotations

import mujoco
import numpy as np

from bampc.config.scenarios import Scenario
from bampc.planner.base import StateSnapshot
from bampc.task.base import Task


class GoalDriver:
    """Poses the goal mocap for one episode."""

    def __init__(self, task: Task, phase_time: float = 0.0) -> None:
        """Capture the base mocap pose the goal drifts around.

        ``phase_time`` shifts the drift clock (a time shift = a phase shift
        for the sinusoids), giving each seed a distinct excitation without a
        graph recapture -- see :func:`drift_phase_time`.
        """
        self.task = task
        self.mid = task.goal_mocap_id
        self.phase_time = float(phase_time)
        self._scratch = mujoco.MjData(task.mj_model)
        self.base_pos = self._scratch.mocap_pos.copy()
        self.base_quat = self._scratch.mocap_quat.copy()

    def mocap(
        self, t: float, qpos: np.ndarray, qvel: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Full mocap arrays at time ``t``, goal slot posed, or ``None``.

        ``None`` means "leave the mocap alone" -- either the task has no goal
        mocap, or its compiled pose is already right.

        Args:
            t: Sim time.
            qpos: Live configuration (a support-relative goal is posed off it).
            qvel: Live velocity.
        """
        if self.mid is None:
            return None
        self._scratch.qpos[:] = qpos
        self._scratch.qvel[:] = qvel
        mujoco.mj_forward(self.task.mj_model, self._scratch)
        pose = self.task.goal_mocap_pose(
            t + self.phase_time,
            self._scratch,
            self.base_pos[self.mid],
            self.base_quat[self.mid],
        )
        if pose is None:
            return None
        pos, quat = self.base_pos.copy(), self.base_quat.copy()
        pos[self.mid] = pose[0]
        if pose[1] is not None:
            quat[self.mid] = pose[1]
        return pos, quat


def drift_phase_time(task: Task, scenario: Scenario) -> float:
    """The scenario's frozen time shift into the goal drift.

    Comes from the bank rather than a per-seed random draw, so seed *i* meets
    the same stretch of the goal trajectory in every experiment on this task.
    A static goal ignores it.
    """
    if not task.goal_drift.enabled:
        return 0.0
    return float(scenario.drift_phase_time)


def snapshot(
    driver: GoalDriver, qpos: np.ndarray, qvel: np.ndarray, t: float
) -> StateSnapshot:
    """A state snapshot with the live goal mocap (if any) attached."""
    goal = driver.mocap(t, qpos, qvel)
    mocap_pos, mocap_quat = goal if goal is not None else (None, None)
    return StateSnapshot(
        qpos=qpos,
        qvel=qvel,
        time=t,
        mocap_pos=mocap_pos,
        mocap_quat=mocap_quat,
    )
