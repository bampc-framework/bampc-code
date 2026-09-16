"""Frozen start states, loaded from ``configs/scenarios/*.yaml``.

Experiments used to draw their initial conditions from a per-seed RNG, so
"seed 3" meant a different pose in every sweep and nothing was comparable
across them. A bank replaces that with a fixed, inspectable list: seed *i*
is scenario *i*, in every experiment on that task.

A bank carries three things:

* the **start states** -- one dict of task-specific fields per scenario,
  applied by :func:`pose`;
* the **goal drift**, one :class:`~bampc.task.base.GoalDrift` for the
  whole task, so excitation is never an uncontrolled variable;
* a per-scenario ``drift_phase_time``, spread evenly over the slowest drift
  period so the scenarios sample the goal trajectory uniformly.

Same pattern as ``configs/numerics/*.yaml``: shared data the caller
names, found via a directory constant. Regenerate with
``scripts/probing/make_scenarios.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from typing import Any

import mujoco
import numpy as np

from bampc import SCENARIOS_DIR
from bampc._yaml import list_names, load_yaml
from bampc.task.base import GoalDrift, Task


@dataclass(frozen=True)
class Scenario:
    """One frozen start state and its slice of the goal-drift trajectory."""

    index: int
    start: dict[str, Any]
    drift_phase_time: float


@dataclass(frozen=True)
class ScenarioBank:
    """Every frozen start state for one task."""

    name: str
    task: str
    shape: str
    scale: float
    goal_drift: GoalDrift
    goal_xy: tuple[float, float] | None
    scenarios: tuple[Scenario, ...]

    def __len__(self) -> int:
        """Number of scenarios in the bank."""
        return len(self.scenarios)

    def __getitem__(self, index: int) -> Scenario:
        """Scenario ``index``, with a message that names the bank."""
        if not 0 <= index < len(self.scenarios):
            raise IndexError(
                f"scenario {index} out of range for bank {self.name!r} "
                f"({len(self.scenarios)} scenarios)"
            )
        return self.scenarios[index]


def list_banks() -> list[str]:
    """Every bank name available under ``configs/scenarios/``."""
    return list_names(SCENARIOS_DIR)


@cache
def load(name: str) -> ScenarioBank:
    """Read one named bank.

    Args:
        name: Bank stem, e.g. ``"push"`` or ``"push_fr3"``.

    Returns:
        The bank. Frozen, so the cached instance is safe to share.

    Raises:
        ValueError: The bank does not exist or is missing a required key.
    """
    data = load_yaml(SCENARIOS_DIR, name, "scenario bank") or {}
    path = SCENARIOS_DIR / f"{name}.yaml"
    task = data.get("task")
    if task not in _POSERS:
        raise ValueError(
            f"{path}: task must be one of {sorted(_POSERS)}, got {task!r}"
        )
    rows = data.get("scenarios") or []
    if not rows:
        raise ValueError(f"{path}: no scenarios")

    drift = dict(data.get("goal_drift") or {})
    for key in ("radius_xy", "freq_xy", "phase"):
        if key in drift:
            drift[key] = tuple(drift[key])
    goal_xy = data.get("goal_xy")

    scenarios = []
    for i, row in enumerate(rows):
        start = dict(row)
        phase = float(start.pop("drift_phase_time", 0.0))
        scenarios.append(Scenario(i, start, phase))
    return ScenarioBank(
        name=name,
        task=task,
        shape=data.get("shape", "t"),
        scale=float(data.get("scale", 1.0)),
        goal_drift=GoalDrift(**drift),
        goal_xy=None if goal_xy is None else tuple(goal_xy),
        scenarios=tuple(scenarios),
    )


def pose(bank: ScenarioBank, task: Task, scenario: Scenario) -> mujoco.MjData:
    """Host ``MjData`` posed at ``scenario``'s start state.

    Args:
        bank: The bank ``scenario`` came from (names the posing rules).
        task: The task to pose -- must be built with the bank's ``shape``
            and ``scale``, since the start states were validated against
            that geometry.
        scenario: The start state to apply.

    Returns:
        A forwarded ``MjData``. Velocities are zero throughout.
    """
    return _POSERS[bank.task](task, scenario.start)


def _pose_push(task: Task, start: dict) -> mujoco.MjData:
    """Push: write the block and pusher joint coordinates straight in.

    ``pusher_xy`` is a joint coordinate, *not* a world position -- the
    pusher body carries an XML offset, so the world position is
    ``pusher_xy + body_pos[:2]``.
    """
    m = task.mj_model
    md = mujoco.MjData(m)
    bx, by = start["block_xy"]
    px, py = start["pusher_xy"]
    for name, value in (
        ("block_x", bx),
        ("block_y", by),
        ("block_yaw", start.get("block_yaw", 0.0)),
        ("root_x", px),
        ("root_y", py),
    ):
        md.qpos[m.joint(name).qposadr[0]] = float(value)
    mujoco.mj_forward(m, md)
    return md


def _pose_balance(task: Task, start: dict) -> mujoco.MjData:
    """Balance: tilt the plate, then place the block in the plate frame."""
    md = mujoco.MjData(task.mj_model)
    task.set_initial_state(
        md,
        tilt=tuple(start.get("tilt", (0.0, 0.0))),
        block_xy=tuple(start["block_xy"]),
        block_yaw=float(start.get("block_yaw", 0.0)),
    )
    return md


def _pose_push_fr3(task: Task, start: dict) -> mujoco.MjData:
    """Push-FR3: load 'home', pose the block, IK the EE to ``ee_xy``."""
    m = task.mj_model
    md = mujoco.MjData(m)
    kid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
    md.qpos[:] = m.key_qpos[kid]
    ex, ey = start["ee_xy"]
    task.set_initial_state(
        md,
        ee_pos=np.array([float(ex), float(ey), task.ee_z_target]),
        block_xy=np.asarray(start["block_xy"], dtype=np.float64),
        block_yaw=float(start.get("block_yaw", 0.0)),
    )
    return md


def _pose_balance_fr3(task: Task, start: dict) -> mujoco.MjData:
    """Balance-FR3: load 'home' (level plate), then rest the block on it."""
    m = task.mj_model
    md = mujoco.MjData(m)
    kid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
    md.qpos[:] = m.key_qpos[kid]
    task.set_initial_state(
        md,
        block_xy=tuple(start["block_xy"]),
        block_yaw=float(start.get("block_yaw", 0.0)),
    )
    return md


def _pose_peg_fr3(task: Task, start: dict) -> mujoco.MjData:
    """Peg-FR3: load 'home', then IK the peg tip to ``tip_pos`` (world)."""
    m = task.mj_model
    md = mujoco.MjData(m)
    kid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
    md.qpos[:] = m.key_qpos[kid]
    task.set_initial_state(
        md, tip_pos=np.asarray(start["tip_pos"], dtype=np.float64)
    )
    return md


def _pose_flip_fr3(task: Task, start: dict) -> mujoco.MjData:
    """Flip-FR3: load 'home', pose the block, IK the EE to ``ee_pos``.

    Unlike Push's ``ee_xy`` (a fixed table-height push), Flip requires the
    full xyz -- there's no one sane default push height across a box that's
    lying on its side, so every scenario states it explicitly.
    """
    m = task.mj_model
    md = mujoco.MjData(m)
    kid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
    md.qpos[:] = m.key_qpos[kid]
    task.set_initial_state(
        md,
        ee_pos=np.asarray(start["ee_pos"], dtype=np.float64),
        block_xy=np.asarray(start["block_xy"], dtype=np.float64),
        block_yaw=float(start.get("block_yaw", 0.0)),
    )
    return md


def _pose_curling(task: Task, start: dict) -> mujoco.MjData:
    """Curling: load 'home', pose the puck, IK the EE to ``ee_xy``.

    Same shape as ``_pose_push_fr3``, but ``ee_xy`` must land inside the
    task's launch box -- ``CurlingFr3.set_initial_state`` clamps it there and
    warns rather than starting an episode the barrier would immediately undo.
    """
    m = task.mj_model
    md = mujoco.MjData(m)
    kid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
    md.qpos[:] = m.key_qpos[kid]
    ex, ey = start["ee_xy"]
    task.set_initial_state(
        md,
        ee_pos=np.array([float(ex), float(ey), task.ee_z_target]),
        block_xy=np.asarray(start["block_xy"], dtype=np.float64),
        block_yaw=float(start.get("block_yaw", 0.0)),
    )
    return md


_POSERS = {
    "push": _pose_push,
    "balance": _pose_balance,
    "push_fr3": _pose_push_fr3,
    "balance_fr3": _pose_balance_fr3,
    "peg_fr3": _pose_peg_fr3,
    "flip_fr3": _pose_flip_fr3,
    "curling": _pose_curling,
}
