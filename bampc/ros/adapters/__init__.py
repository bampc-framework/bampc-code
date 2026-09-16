"""Per-task adapters: how to read a task's state, how to command it.

``PlannerNode`` and ``ControlNode`` are task-agnostic -- everything
task-specific (TF lookups, joint-state mapping, the state-estimation filter,
how a sampling-space action becomes a hardware command) lives in one adapter
module per task, injected as plain functions rather than branched on inside
the nodes. Adding a task means implementing the same hooks; nothing in the
nodes changes.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from rclpy.node import Node

    from bampc.planner.base import StateSnapshot
    from bampc.task.base import Task

#: Zero-arg closure returning the latest state estimate, called once per
#: planning tick (and, for a debug node, once per display tick).
StateReader = Callable[[], "StateSnapshot"]

#: Turns one sampling-space action row into a hardware-side publish.
CommandWriter = Callable[[np.ndarray], None]

#: Per-task factory: given the owning node (so it can attach subscriptions
#: / TF listeners to something that will actually be spun) and the task
#: (so it knows the model's joint/object layout), build a StateReader. This
#: is "a function defined per task that gets the current system state and
#: wires it into the planner" -- e.g. ``adapters.push_fr3.make_state_reader``.
StateReaderFactory = Callable[["Node", "Task"], StateReader]

#: Per-task factory: given the owning node, build a CommandWriter.
CommandWriterFactory = Callable[["Node"], CommandWriter]

#: Zero-arg closure that drives the arm to a fixed pose and blocks until
#: done (or timeout), returning success. Meant to be called off the ROS
#: executor thread -- it blocks for the length of a MoveIt plan+execute.
HomeMover = Callable[[], bool]

#: Per-task factory: given the owning node, build a HomeMover.
HomeMoverFactory = Callable[["Node"], HomeMover]

__all__ = [
    "CommandWriter",
    "CommandWriterFactory",
    "HomeMover",
    "HomeMoverFactory",
    "StateReader",
    "StateReaderFactory",
]
