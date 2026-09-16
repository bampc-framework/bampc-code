"""ROS 2 node mirroring PlannerNode's state pull-in, with no planner at all.

A debug tool: builds the exact same per-task state reader (see
:mod:`bampc.ros.adapters`)
:class:`~bampc.ros.planner_node.PlannerNode` would, from the same
factory function, on a plain ROS timer, and renders it in the CPU passive
viewer via :mod:`bampc.ros.debug_viewer`. No GPU, no
``optimize``, no publishing -- just the TF + joint-state + filter pipeline,
rendered live, so it can be checked before ever starting the planner.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from bampc.ros import debug_viewer

try:
    from rclpy.node import Node

    _HAS_RCLPY = True
except ImportError:  # ROS not installed; keep the module importable.
    Node = object  # type: ignore[assignment,misc]
    _HAS_RCLPY = False

if TYPE_CHECKING:
    from bampc.ros.adapters import StateReaderFactory
    from bampc.task.base import Task


class StateViewerNode(Node):
    """Render a task's live state pipeline with no planner running."""

    def __init__(
        self,
        task: Task,
        state_reader_factory: StateReaderFactory,
        *,
        rate: float,
    ) -> None:
        """Launch the viewer, build the state reader, start the display timer.

        Args:
            task: The task whose state pipeline is being checked -- supplies
                ``mj_model`` for the viewer and the layout the reader needs.
            state_reader_factory: The same per-task factory a
                ``PlannerNode`` would use (see
                :mod:`bampc.ros.adapters`),
                called as ``state_reader_factory(self, task)``.
            rate: Display refresh frequency (Hz).
        """
        if not _HAS_RCLPY:
            raise RuntimeError("rclpy is not available; source a ROS 2 install")
        super().__init__("bampc_state_viewer")
        self.state_reader = state_reader_factory(self, task)
        self._mj_model = task.mj_model
        self._ee_body_id = task.mj_model.body("ee_frame").id
        self._mj_data, self._handle = debug_viewer.launch(task.mj_model)
        self.create_timer(1.0 / rate, self._on_timer)

    def _on_timer(self) -> None:
        state = self.state_reader()
        debug_viewer.sync(self._mj_model, self._mj_data, self._handle, state)
        ee_z = self._mj_data.xpos[self._ee_body_id, 2]
        self.get_logger().info(f"ee_frame world z: {ee_z:.4f}")
