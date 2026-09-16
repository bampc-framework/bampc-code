"""ROS 2 node replaying a planner's published trajectory: the control half.

The peer of :mod:`bampc.ros.planner_node`, run as a **separate OS
process** so a slow plan step never delays a control tick. Entirely
task-agnostic: it holds no ``Task``, no ``mj_model``, no IK -- it only
indexes into whatever dense trajectory
(:mod:`bampc.ros.spline_msg`) was last published, by elapsed wall-clock
time, and hands each row to a per-task **command-writer factory** (see
:mod:`bampc.ros.adapters`) -- a function that takes this node (so it can
attach a publisher to something that actually gets spun) and returns the
closure that gets each row onto the wire. Task-specific kwargs (topic name,
frame id) are pre-bound by the caller, e.g.
``functools.partial(push_fr3.make_command_writer, twist_topic=...)``.

A **watchdog** gates every write: if no trajectory has arrived recently
enough to still be trustworthy, the command written is zero rather than an
indefinitely-extrapolated stale one. The same applies once a received
trajectory's own samples run out while still inside the watchdog window --
a stalled planner should stop the robot, not let it coast on stale intent.
"""

from __future__ import annotations

import select
import sys
import termios
import threading
import time
import tty
from typing import TYPE_CHECKING

import numpy as np

from bampc.ros.spline_msg import Float64MultiArray, unpack

try:
    from rclpy.node import Node

    _HAS_RCLPY = True
except ImportError:  # ROS not installed; keep the module importable.
    Node = object  # type: ignore[assignment,misc]
    _HAS_RCLPY = False

if TYPE_CHECKING:
    from bampc.ros.adapters import CommandWriterFactory


class ControlNode(Node):
    """Hold/replay the latest published trajectory on a fast timer."""

    def __init__(
        self,
        command_writer_factory: CommandWriterFactory,
        *,
        control_rate: float,
        spline_topic: str = "/bampc/trajectory",
        watchdog_timeout: float = 0.5,
        dry_run: bool = False,
        deadman: bool = False,
    ) -> None:
        """Wire the subscription, build the command writer, start the timer.

        Args:
            command_writer_factory: Per-task factory building the
                closure that turns one sampling-space action row into a
                hardware publish (see :mod:`bampc.ros.adapters`), called
                as ``command_writer_factory(self)``.
            control_rate: Timer frequency (Hz) at which the trajectory is
                indexed and the command writer is called.
            spline_topic: Topic the dense trajectory arrives on -- must
                match the planner node's.
            watchdog_timeout: Seconds since the last received trajectory
                after which the command writer is called with zeros instead
                of an indefinitely stale trajectory. Pick a small multiple
                of the planner's own replan period.
            dry_run: Always write zeros instead of the indexed action --
                the planner/trajectory pipeline still runs for real
                (visible on a :class:`~bampc.ros.debug_viewer` or
                :class:`~bampc.ros.state_viewer_node.StateViewerNode`),
                the robot just never receives a nonzero command. For
                sanity-checking a real-robot rollout before trusting it to
                actually move the arm.
            deadman: Require pressing Enter in this process's terminal to
                arm real commands, and pressing it again to disarm (write
                zeros) -- a manual safety toggle independent of Ctrl+C'ing
                the process. Starts disarmed. Not a true continuous-hold
                deadman (this terminal can't detect key release) -- it's a
                toggle using the same discrete-keypress mechanism
                :mod:`bampc.ros.planner_node`'s teleop reader uses.
        """
        if not _HAS_RCLPY:
            raise RuntimeError("rclpy is not available; source a ROS 2 install")
        super().__init__("bampc_control")
        self.dry_run = bool(dry_run)
        if self.dry_run:
            self.get_logger().warn(
                "dry_run=True: publishing zero commands, the robot will not "
                "move"
            )
        self.command_writer = command_writer_factory(self)
        self.watchdog_timeout = float(watchdog_timeout)

        self._lock = threading.Lock()
        self._traj: tuple[float, float, np.ndarray] | None = None
        self._recv_monotonic: float | None = None

        self.deadman = bool(deadman)
        self._deadman_lock = threading.Lock()
        self._armed = False
        self._deadman_stop = threading.Event()
        self._deadman_thread: threading.Thread | None = None
        if self.deadman:
            print(
                "\ndeadman: press Enter in this terminal to ARM real "
                "commands, press it again to DISARM (publish zeros)."
            )
            self._deadman_thread = threading.Thread(
                target=self._deadman_loop, daemon=True
            )
            self._deadman_thread.start()

        self.create_subscription(
            Float64MultiArray, spline_topic, self._on_spline, 10
        )
        self.create_timer(1.0 / control_rate, self._on_control_timer)

    def _deadman_loop(self) -> None:
        """Toggle ``self._armed`` on each Enter keypress in this terminal."""
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)  # not setraw: keeps Ctrl+C/SIGINT working
            while not self._deadman_stop.is_set():
                ready, _, _ = select.select([sys.stdin], [], [], 0.1)
                if not ready:
                    continue
                ch = sys.stdin.read(1)
                if ch not in ("\n", "\r"):
                    continue
                with self._deadman_lock:
                    self._armed = not self._armed
                    armed = self._armed
                if armed:
                    self.get_logger().warn("ARMED - publishing real commands")
                else:
                    self.get_logger().warn(
                        "DISARMED - publishing zero commands"
                    )
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def destroy_node(self) -> bool:
        """Stop the deadman reader thread before tearing down the node."""
        if self._deadman_thread is not None:
            self._deadman_stop.set()
            self._deadman_thread.join(timeout=1.0)
        return super().destroy_node()

    def _on_spline(self, msg: Float64MultiArray) -> None:
        """Cache the latest trajectory and the wall-clock time it arrived."""
        t0, dt, actions = unpack(msg)
        with self._lock:
            self._traj = (t0, dt, actions)
            self._recv_monotonic = time.monotonic()

    def _on_control_timer(self) -> None:
        """Index into the cached trajectory by elapsed time and write it out."""
        with self._lock:
            traj = self._traj
            recv = self._recv_monotonic
        if traj is None or recv is None:
            return  # nothing received yet -- nothing safe to command
        t0, dt, actions = traj
        now = time.monotonic()
        # Watchdog: liveness of the connection itself, keyed on when this
        # trajectory *arrived* -- a planner that's still publishing on time
        # but always slightly stale (see below) shouldn't trip this.
        if now - recv > self.watchdog_timeout:
            self.command_writer(np.zeros_like(actions[0]))
            return
        # Index: how far into the trajectory's own timeline we are, keyed
        # on ``t0`` (the planner's ``time.monotonic()`` reading when this
        # trajectory was computed to start from) rather than ``recv``. Using
        # ``recv`` instead would silently assume publish-to-subscribe
        # latency is zero; ``t0`` folds that latency into ``idx`` instead of
        # ignoring it. ``time.monotonic()`` is one kernel clock shared by
        # both processes, so ``t0`` from the planner's process and ``now``
        # here are directly comparable.
        idx = max(0.0, now - t0) / dt
        if idx >= actions.shape[0]:
            # The trajectory's own samples are exhausted (a replan hasn't
            # landed yet, but we're still inside the watchdog window) --
            # stop rather than hold the last sampled action indefinitely.
            self.command_writer(np.zeros_like(actions[0]))
            return
        idx_int = int(idx)
        with self._deadman_lock:
            armed = self._armed
        zero = self.dry_run or (self.deadman and not armed)
        action = np.zeros_like(actions[idx_int]) if zero else actions[idx_int]
        self.command_writer(action)
