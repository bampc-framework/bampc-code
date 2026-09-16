"""ROS 2 planner node: the planning half of a two-process real-robot split.

``PlannerNode`` runs the GPU-heavy ``planner.optimize()`` in its own
process and publishes the resulting trajectory for a lightweight peer
(:mod:`bampc.ros.control_node`) to replay on its own timer. **Two OS
processes, not two timers**: numpy/GPU work holds the GIL long enough to
starve a peer callback living in the same process.

Task-agnostic by construction -- TF, joint topics and command topics live in
a per-task state-reader factory (:mod:`bampc.ros.adapters`). What is
published is an already-interpolated dense trajectory
(:mod:`bampc.ros.spline_msg`), not spline knots, so the control node
needs no spline math.

``self._kin_data`` is refreshed every tick, including while idle. That is not
cosmetic: the drifting goal reaches the cost kernel through
``StateSnapshot.mocap_pos``/``mocap_quat``, so without it a real deployment
silently chases a static goal even with ``goal_drift`` configured.
"""

from __future__ import annotations

import csv
import select
import sys
import termios
import threading
import time
import tty
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import mujoco
import numpy as np

from bampc import OUTPUT_DIR, spline
from bampc.ros import debug_viewer
from bampc.ros.spline_msg import Float64MultiArray, pack
from bampc.sim.viewer import (
    BELIEF_RGBA,
    build_ghosts,
    contact_palette,
    domain_palette,
    save_video,
    set_ghost_contact_color,
    sync_endpoint_ghosts,
)

try:
    from rclpy.node import Node

    _HAS_RCLPY = True
except ImportError:  # ROS not installed; keep the module importable.
    Node = object  # type: ignore[assignment,misc]
    _HAS_RCLPY = False

if TYPE_CHECKING:
    from collections.abc import Callable

    from bampc.planner.base import (
        PlanInfo,
        SamplingPlanner,
        StateSnapshot,
    )
    from bampc.ros.adapters import HomeMoverFactory, StateReaderFactory

# Under the repo's one generated-artifact root, like sim/viewer.py.
_VIDEO_DIR = OUTPUT_DIR / "recordings"
_DATA_DIR = OUTPUT_DIR / "real_world"

# Opacity of an endpoint ghost, low enough that a fan of them stays readable
# where they bunch up. Matches scripts/curling/friction_id.py's viewer.
_GHOST_ALPHA = 0.3

# w/a/s/d jog the EE in +x/-y/-x/+y (base frame), space/x stop -- see
# PlannerNode's ``teleop`` kwarg and ``_teleop_publish``.
_TELEOP_KEYS = {
    "w": (1.0, 0.0),
    "s": (-1.0, 0.0),
    "d": (0.0, 1.0),
    "a": (0.0, -1.0),
    " ": (0.0, 0.0),
    "x": (0.0, 0.0),
}


class PlannerNode(Node):
    """Run a SamplingPlanner's plan step on its own thread, its own process."""

    def __init__(  # noqa: PLR0912, PLR0913, PLR0915
        self,
        planner: SamplingPlanner,
        state_reader_factory: StateReaderFactory,
        *,
        plan_rate: float,
        control_rate: float,
        spline_topic: str = "/bampc/trajectory",
        show_viewer: bool = False,
        show_belief: bool = False,
        initial_knots: np.ndarray | None = None,
        compensate_latency: bool = True,
        latency_ema_alpha: float = 0.3,
        record: bool = False,
        record_camera: str = "main",
        record_fps: float | None = None,
        record_name: str = "real_world",
        record_dir: str | Path | None = None,
        record_size: tuple[int, int] = (480, 640),
        collect_data: bool = False,
        episode_duration: float = 20.0,
        data_dir: str | Path | None = None,
        data_label: str = "episode",
        data_schema: Literal["full", "position"] = "full",
        teleop: bool = False,
        teleop_speed: float = 0.06,
        teleop_pulse: float = 0.3,
        home_mover_factory: HomeMoverFactory | None = None,
        interactive: bool = False,
        warm_start: bool = False,
        home_on_start: bool = False,
        success_fn: Callable[[mujoco.MjData], bool] | None = None,
        release_fn: Callable[[StateSnapshot], bool] | None = None,
        observer: Callable[[StateSnapshot, mujoco.MjData, PlanInfo], None]
        | None = None,
        status_fn: Callable[[StateSnapshot, mujoco.MjData, PlanInfo], None]
        | None = None,
        on_episode_end: Callable[[], None] | None = None,
        action_filter: Callable[[np.ndarray, mujoco.MjData], np.ndarray]
        | None = None,
        ghost_mode: Literal["belief", "endpoint"] = "belief",
    ) -> None:
        """Wire the publisher, build the state reader, start the plan thread.

        Args:
            planner: The constructed planner to drive.
            state_reader_factory: Per-task factory building the
                :class:`~bampc.planner.base.StateSnapshot` reader (see
                :mod:`bampc.ros.adapters`), called as
                ``state_reader_factory(self, planner.task)``.
            plan_rate: Replanning frequency (Hz) for the GPU ``optimize``
                call.
            control_rate: Sample spacing (Hz) of the dense trajectory this
                node publishes -- must match the peer control node's replay
                rate.
            spline_topic: Topic the dense trajectory is published on.
            show_viewer: Toggle a CPU passive MuJoCo viewer showing exactly
                what the state reader is feeding the planner, including the
                live drifting goal.
            show_belief: Draw one translucent mocap ghost per domain at that
                domain's believed object pose. Requires ``show_viewer`` or
                ``record`` and a task declaring ``endpoint_bodies``; raises
                otherwise.
            initial_knots: Optional initial spline knots (see
                ``SamplingPlanner.init_params``).
            compensate_latency: Predict the state forward by an EMA of past
                ``optimize()`` durations before planning, so the plan lands
                on the state the robot will be in rather than the one read.
            latency_ema_alpha: Smoothing factor for that EMA (0-1).
            record: Capture the viewer's state to an MP4, flushed on
                ``destroy_node()``. Independent of ``show_viewer``.
            record_camera: Model camera to render from. Required when
                ``record`` is set.
            record_fps: Playback frame rate. ``None`` uses ``plan_rate``
                (one frame per plan tick), mapping elapsed time 1:1.
            record_name: Filename stem (``<timestamp>_<record_name>.mp4``).
            record_dir: Output directory; defaults to ``videos/``.
            record_size: ``(height, width)`` of rendered frames. Capped by
                the model's offscreen framebuffer (MuJoCo defaults to
                480x640); exceeding it raises. Raise ``offwidth``/
                ``offheight`` in the XML's ``<visual><global>`` first.
            collect_data: Episodic mode -- idle until Enter, run for
                ``episode_duration`` logging one CSV row per tick, stop,
                save, prompt again.
            episode_duration: Seconds per collected episode.
            data_dir: Output directory for episode CSVs.
            data_label: Included in each filename so runs self-document.
            data_schema: CSV columns. ``"full"`` is ``t, pos_err, rot_err,
                ee_pos, ee_quat, blk_pos, blk_quat``; ``"position"`` is
                ``t`` plus block and goal centers, and requires a goal mocap
                body -- raises in ``__init__``, not on first save.
            teleop: With ``collect_data``, jog the EE in x/y from the
                keyboard while idle (w/a/s/d, space/x to stop). Needs a 2-D
                task-space action; raises otherwise. Puts the terminal in
                cbreak mode for the life of the node.
            teleop_speed: Commanded EE speed (m/s) per jog keypress.
            teleop_pulse: Seconds of nonzero command per jog keypress.
            home_mover_factory: Optional factory building a
                :class:`~bampc.ros.adapters.HomeMover`, wired to the
                ``h`` key at the idle prompt.
            interactive: ``collect_data``'s Enter/``h`` gate without the CSV
                or the duration limit. Implied by ``collect_data``.
            warm_start: Keep calling ``optimize()`` during the idle wait, so
                the knot mean already tracks the live state before Enter
                rather than catching up in the few replans an open-loop task
                gets. Nothing is published while idle either way.
            home_on_start: Home once right after construction. Requires
                ``collect_data`` or ``interactive`` (raises otherwise): the
                loop must stay idle until homing finishes, or a live MoveIt
                trajectory races Servo jog commands on the same joints.
            success_fn: Per-tick success check on ``self._kin_data``. On
                ``True`` the loop publishes zero and ends the episode (or
                latches zero, in plain continuous mode).
            release_fn: Per-tick check for "the robot's part is over, the
                episode is not". The first ``True`` zeroes the robot, homes
                it, and *coasts*: still reading, replanning and calling
                ``observer``, but no longer publishing. Curling-FR3 uses it
                for the puck leaving the launch box.
            observer: Per-tick callback (snapshot, ``_kin_data``,
                :class:`PlanInfo`) while a rollout is active, coasting
                included. Runs on the planning thread.
            status_fn: Same signature and thread as ``observer``, but fires
                on every tick ``optimize`` runs -- the idle ``warm_start``
                wait included. Use it for a live readout.
            on_episode_end: Callback run on the *input* thread between an
                episode stopping and the next idle prompt; stdin is in
                cbreak mode and owned by that thread, so it may prompt.
            action_filter: Last-chance transform on the dense
                ``(horizon, nu)`` block, applied in :meth:`_publish`.
                Curling-FR3 re-applies its launch-box velocity barrier here,
                which lives in the task's IK and is bypassed entirely when
                MoveIt Servo does the IK instead.
            ghost_mode: ``"belief"`` parks each ghost at that domain's
                believed current pose; ``"endpoint"`` at its predicted end
                of rollout -- the useful view for an open-loop task, where
                the spread across domains is the hedge itself.
        """
        if not _HAS_RCLPY:
            raise RuntimeError("rclpy is not available; source a ROS 2 install")
        if teleop and planner.task.nu != 2:
            raise ValueError(
                f"teleop needs a 2-D task-space action (sampling_space="
                f"'task'); this task has nu={planner.task.nu}"
            )
        super().__init__("bampc_planner")
        self.planner = planner
        self.state_reader = state_reader_factory(self, planner.task)
        self.plan_rate = float(plan_rate)
        self.control_rate = float(control_rate)
        self.params = planner.init_params(initial_knots)
        self.compensate_latency = bool(compensate_latency)
        self._latency_alpha = float(latency_ema_alpha)
        self._plan_duration = 1.0 / self.plan_rate
        self._pub = self.create_publisher(Float64MultiArray, spline_topic, 10)

        self._show_belief = bool(show_belief)
        self._ghost_mode = ghost_mode
        if self._show_belief and not (show_viewer or record):
            raise ValueError(
                "show_belief=True needs show_viewer or record -- otherwise "
                "nothing would render the belief ghosts"
            )
        if self._show_belief and not planner.task.endpoint_bodies:
            raise ValueError(
                "show_belief=True needs the task to declare endpoint_bodies "
                "(the belief ghosts clone their geometry from those bodies)"
            )

        mj_model = planner.task.mj_model
        self._belief_ids = self._belief_body_ids = None
        self._belief_R = 0
        self._belief_c_palette = None
        if self._show_belief:
            self._belief_R = planner.engine.num_randomizations
            # Endpoint ghosts are one-per-domain over a value grid, so hue
            # must run with the domain index to read as an ordered fan;
            # belief ghosts are all the same hypothesis and stay uniform.
            palette = (
                domain_palette(self._belief_R, alpha=_GHOST_ALPHA)
                if self._ghost_mode == "endpoint"
                else np.tile(BELIEF_RGBA, (self._belief_R, 1))
            )
            vmodel, built = build_ghosts(
                planner.task.mj_spec, planner.task.endpoint_bodies,
                {"belief": (self._belief_R, 1, palette)},
            )
            if planner.task.model_config is not None:
                planner.task.model_config.apply_to(vmodel)
            mj_model = vmodel
            self._belief_ids, self._belief_body_ids = built["belief"]
            if planner.task.contact_probes:
                if getattr(planner.engine, "record_initial_state", False):
                    self._belief_c_palette = contact_palette()
                else:
                    print(
                        "[note] task declares contact_probes but the engine "
                        "was built with record_initial_state=False -- "
                        "belief ghosts will stay a flat color, not contact "
                        "modes."
                    )
        self._viewer_model = mj_model

        self._kin_data = mujoco.MjData(mj_model)
        self._goal_mid = planner.task.goal_mocap_id
        self._goal_base_pos = self._goal_base_quat = None
        if self._goal_mid is not None:
            self._goal_base_pos = np.array(
                self._kin_data.mocap_pos[self._goal_mid]
            )
            self._goal_base_quat = np.array(
                self._kin_data.mocap_quat[self._goal_mid]
            )
        self._ee_body_id = mj_model.body("ee_frame").id
        self._block_body_id = mj_model.body("block").id

        self._viewer_handle = None
        if show_viewer:
            _, self._viewer_handle = debug_viewer.launch(
                mj_model, self._kin_data
            )

        self._recorder = None
        self._record_requested = bool(record)
        self._record_frames: list[np.ndarray] = []
        self._record_camera = record_camera
        self._record_fps = float(record_fps) if record_fps else self.plan_rate
        self._record_name = record_name
        self._record_dir = (
            Path(record_dir).expanduser() if record_dir else _VIDEO_DIR
        )
        self._record_size = record_size
        if record and mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_CAMERA, record_camera
        ) < 0:
            raise ValueError(
                f"record_camera {record_camera!r} is not in the model"
            )

        self.collect_data = bool(collect_data)
        self.interactive = bool(interactive)
        self._warm_start = bool(warm_start)
        self.episode_duration = float(episode_duration)
        self.data_dir = Path(data_dir) if data_dir else _DATA_DIR
        self.data_label = data_label
        if data_schema == "position" and self._goal_mid is None:
            raise ValueError(
                "data_schema='position' needs a goal mocap body ('goal' "
                "in the task's XML) to log a goal center; this task's "
                "goal_mocap_id is None"
            )
        self.data_schema = data_schema
        self.teleop = bool(teleop)
        self.teleop_speed = float(teleop_speed)
        self.teleop_pulse = float(teleop_pulse)
        self._home_mover = (
            home_mover_factory(self) if home_mover_factory else None
        )
        self._success_fn = success_fn
        self._release_fn = release_fn
        self._observer = observer
        self._status_fn = status_fn
        self._on_episode_end = on_episode_end
        self._action_filter = action_filter
        self._task_done = threading.Event()
        self._homing = threading.Event()
        # Set once release_fn fires: the robot is stopped and going home, but
        # the episode continues so the planner can keep predicting.
        self._coasting = threading.Event()
        self._episode_running = threading.Event()
        self._episode_done = threading.Event()
        self._episode_samples: list[tuple[float, ...]] = []
        self._episode_t0: float | None = None
        self._episode_wall_start = 0.0

        # Guards planner.optimize() against a concurrent allocation switch:
        # on_episode_end runs on the input thread (see its docstring) and,
        # for an interactive AllocationPolicy, calls apply_stage() there --
        # which mutates engine/planner/randomizer state that optimize()
        # reads mid-call. warm_start keeps optimize() ticking on the
        # planning thread even while idle (between episodes, exactly when
        # on_episode_end runs), so without this lock a switch applied
        # between optimize()'s sample_knots() and its rollout leaves the
        # sampled knots sized to the old num_samples but the rolled-out
        # costs sized to the new one.
        self._plan_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._planning_loop, daemon=True
        )
        self._thread.start()
        self._input_thread = None
        if self.collect_data or self.interactive:
            self._input_thread = threading.Thread(
                target=self._input_loop, daemon=True
            )
            self._input_thread.start()

        self.home_on_start = bool(home_on_start)
        if self.home_on_start and not (self.collect_data or self.interactive):
            raise ValueError(
                "home_on_start requires collect_data or interactive, so "
                "the planning loop stays idle (no Servo commands) until "
                "the home move finishes -- otherwise it would race a live "
                "MoveIt trajectory against Servo jog commands"
            )
        if self.home_on_start:
            self._on_home()

    def destroy_node(self) -> None:  # noqa: D102
        self._stop.set()
        self._thread.join(timeout=2.0)
        if self._recorder is not None and self._record_frames:
            path = save_video(
                self._record_frames,
                self._record_dir,
                self._record_name,
                self._record_fps,
            )
            n = len(self._record_frames)
            self.get_logger().info(f"saved {n} frames -> {path}")
        super().destroy_node()

    def _check_release(self, state: StateSnapshot) -> None:
        """Enter the coasting phase the first time ``release_fn`` fires.

        Stops the robot and sends it home, but leaves the episode running so
        the planning loop keeps replanning and feeding ``observer``.
        """
        if self._release_fn is None or self._coasting.is_set():
            return
        if not self._release_fn(state):
            return
        self.get_logger().info("release -- stopping robot, planner continues")
        self._coasting.set()
        self._on_home()

    def _sync_endpoint_ghosts(self, info: PlanInfo) -> None:
        """Park one ghost per domain at its predicted end-of-rollout pose.

        Reads the engine's device state directly rather than a ``PlanInfo``
        field: the rollout has already run, so ``d.xpos`` still holds each
        world's final pose. The sample index is the actual argmin of
        ``info.sample_costs`` -- not hardcoded to 0 -- since that is the one
        the scoring path reads (see ``SamplingPlanner.optimize``); it only
        coincides with raw sample 0 for PredictiveSampling.
        """
        best = int(np.argmin(info.sample_costs))
        for gid, name in zip(
            self._belief_ids, self.planner.task.endpoint_bodies, strict=True
        ):
            pos, quat = self.planner.engine.final_body_pose(name)
            sync_endpoint_ghosts(
                self._kin_data, gid[: self._belief_R], [best], pos, quat
            )
        if self._viewer_handle is not None:
            self._viewer_handle.sync()

    def _check_success(self) -> bool:
        """Handle a latched or freshly-detected task success.

        Returns:
            ``True`` if this tick should skip planning entirely (the robot
            has already been zeroed, either just now or on a prior tick).
        """
        if self._task_done.is_set():  # continuous mode, already succeeded
            self._publish_stop()
            return True
        if self._success_fn is not None and self._success_fn(self._kin_data):
            self.get_logger().info("task success -- stopping rollout")
            self._publish_stop()
            if self.collect_data or self.interactive:
                self._end_episode()
            else:
                self._task_done.set()
            return True
        return False

    def _planning_loop(self) -> None:  # noqa: PLR0912, PLR0915
        """Replan at ``plan_rate``, off the ROS executor, until stopped."""
        period = 1.0 / self.plan_rate
        if self._record_requested:
            # Built here, not in __init__: mujoco.Renderer owns a GL
            # context that is only valid on the thread that creates it,
            # and every render call below runs on this thread -- creating
            # it in __init__ (the main/ROS thread) instead left it current
            # nowhere this loop could see, and render() silently read
            # garbage GPU state (visible as full-frame noise in the mp4).
            self._recorder = debug_viewer.start_recording(
                self._viewer_model,
                self._record_camera,
                self._record_size,
            )
        while not self._stop.is_set():
            tick_start = time.monotonic()
            try:
                state = self.state_reader()
            except Exception as exc:
                # Transient sensor/TF dropout (e.g. DDS discovery still
                # catching up on this process's participant): retry next
                # tick instead of killing the planning thread, which would
                # otherwise need a full process restart (another Warp/CUDA
                # cold start).
                self.get_logger().warn(
                    f"state read failed ({exc}); retrying",
                    throttle_duration_sec=2.0,
                )
                time.sleep(max(0.0, period - (time.monotonic() - tick_start)))
                continue
            if self._goal_mid is not None:
                pose = self.planner.task.goal_mocap_pose(
                    state.time,
                    self._kin_data,
                    self._goal_base_pos,
                    self._goal_base_quat,
                )
                if pose is not None:
                    pos, quat = pose
                    state = replace(
                        state, mocap_pos=pos[None], mocap_quat=quat[None]
                    )

            # Refresh unconditionally -- including while idling between
            # collect_data episodes -- so the debug viewer (and the goal
            # drift it shows) stays live during the whole "press Enter"
            # wait, not just during an active rollout. With warm_start=True,
            # optimize() itself also runs continuously through that same
            # wait (below) -- not just this display refresh.
            debug_viewer.update(
                self._viewer_model, self._kin_data, state
            )
            if self._viewer_handle is not None:
                self._viewer_handle.sync()

            running = self._episode_running.is_set()
            warm = self.interactive and self._warm_start and not running
            if self.collect_data and not running:
                time.sleep(max(0.0, period - (time.monotonic() - tick_start)))
                continue
            if self.interactive and not running and not warm:
                time.sleep(max(0.0, period - (time.monotonic() - tick_start)))
                continue

            if running:
                if self._check_success():
                    time.sleep(
                        max(0.0, period - (time.monotonic() - tick_start))
                    )
                    continue
                self._check_release(state)

            # dt_lag's dead-reckoning assumes the read state is currently
            # being driven by the last published action -- false while warm
            # (nothing is published yet), so it would shift the state fed
            # to optimize() into a fictitious future.
            dt_lag = (
                self._plan_duration
                if self.compensate_latency and running
                else 0.0
            )
            opt_start = time.monotonic()
            with self._plan_lock:
                self.params, info = self.planner.optimize(
                    state, self.params, dt_lag=dt_lag
                )
            opt_elapsed = time.monotonic() - opt_start
            self._plan_duration += self._latency_alpha * (
                opt_elapsed - self._plan_duration
            )
            self.get_logger().info(
                f"optimize: {opt_elapsed * 1e3:.1f} ms (dt_lag ema: "
                f"{self._plan_duration * 1e3:.1f} ms, target period: "
                f"{1e3 / self.plan_rate:.1f} ms)",
                throttle_duration_sec=1.0,
            )

            if self._status_fn is not None:
                self._status_fn(state, self._kin_data, info)

            if running and self._observer is not None:
                self._observer(state, self._kin_data, info)

            if self._show_belief and self._ghost_mode == "endpoint":
                self._sync_endpoint_ghosts(info)
            elif self._show_belief and info.belief_pose is not None:
                # One ghost per domain at its believed (post-settle) pose --
                # mirrors bampc.sim.viewer.run_interactive's
                # show_belief block.
                for gid, name in zip(
                    self._belief_ids, self.planner.task.endpoint_bodies,
                    strict=True,
                ):
                    pos, quat = info.belief_pose[name]
                    sync_endpoint_ghosts(
                        self._kin_data, gid[: self._belief_R], [0],
                        pos[:, None], quat[:, None],
                    )
                if (
                    self._belief_c_palette is not None
                    and info.contact_modes is not None
                ):
                    for per_body in self._belief_body_ids:
                        set_ghost_contact_color(
                            self._viewer_model, per_body[: self._belief_R],
                            info.contact_modes, self._belief_c_palette,
                        )
                if self._viewer_handle is not None:
                    self._viewer_handle.sync()

            if self._recorder is not None:
                debug_viewer.capture_frame(
                    self._recorder,
                    self._kin_data,
                    self._record_camera,
                    self._record_frames,
                )

            # Coasting: the robot's part is over and it is on its way home,
            # so publishing now would fight the MoveIt trajectory. Everything
            # above still runs -- that is the point of the phase. Warm ticks
            # (running is False) never publish either, for the same reason
            # there's no active episode to execute yet.
            if running and not self._coasting.is_set():
                self._publish(state.time + dt_lag, time.monotonic())

            if self.collect_data:
                if self._episode_t0 is None:
                    self._episode_t0 = state.time
                    self._episode_wall_start = time.monotonic()
                    self.get_logger().info("rollout: first optimize tick")
                self._episode_samples.append(
                    self._episode_sample(state.time - self._episode_t0)
                )
                elapsed = time.monotonic() - self._episode_wall_start
                if elapsed >= self.episode_duration:
                    self._end_episode()

            time.sleep(max(0.0, period - (time.monotonic() - tick_start)))

    def _wait_while_homing(self) -> None:
        """Block until any in-progress home move finishes.

        ``self._homing`` is *set* while homing runs and *cleared* once it
        finishes (see ``_on_home``/``_home_worker``) -- the inverse of
        what ``Event.wait()`` blocks for, so this polls ``is_set()``
        instead of calling ``.wait()`` on it directly.
        """
        while self._homing.is_set() and not self._stop.is_set():
            time.sleep(0.05)

    def _await_idle_start(self) -> bool:
        """Block until Enter starts a rollout; ``h`` homes meanwhile.

        Returns:
            ``True`` once Enter is read, ``False`` if the node stopped
            first instead.
        """
        while not self._stop.is_set():
            ready, _, _ = select.select([sys.stdin], [], [], 0.1)
            if not ready:
                continue
            ch = sys.stdin.read(1)
            if ch in ("\n", "\r"):
                return True
            if ch.lower() == "h":
                self._on_home()
        return False

    def _await_episode_stop(self) -> None:
        """Block until the episode ends, by timeout or a second Enter.

        A ``collect_data`` duration timeout (handled in
        ``_planning_loop``) and a second Enter here both end the episode
        through the same ``_end_episode()`` -- whichever comes first.
        """
        while not self._episode_done.is_set() and not self._stop.is_set():
            ready, _, _ = select.select([sys.stdin], [], [], 0.1)
            if ready and sys.stdin.read(1) in ("\n", "\r"):
                self._end_episode()
                break
        self._episode_done.clear()

    def _print_idle_prompt(self) -> None:
        """Print the non-teleop idle prompt, worded for the active mode."""
        if self.collect_data:
            print(
                f"\nPress Enter to roll out for "
                f"{self.episode_duration:.0f}s (reposition the block, "
                "then press Enter), or 'h' + Enter to send the arm "
                "home..."
            )
        else:
            print(
                "\nPress Enter to start the rollout (press Enter again "
                "to stop it), or 'h' + Enter to send the arm home..."
            )

    def _gated_input_loop(self) -> None:
        """Non-teleop idle/rollout gate: Enter starts/stops, 'h' homes."""
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)  # not setraw: keeps Ctrl+C/SIGINT working
            while not self._stop.is_set():
                self._print_idle_prompt()
                if not self._await_idle_start():
                    break
                self._wait_while_homing()
                if self._stop.is_set():
                    break
                self.get_logger().info("starting rollout...")
                self._episode_running.set()
                self._await_episode_stop()
                if self._on_episode_end is not None:
                    # Only the state-mutating part of a switch (e.g.
                    # apply_stage()) actually needs mutual exclusion with
                    # optimize() -- but on_episode_end also blocks on the
                    # y/n read itself, so holding the lock for the whole
                    # call just pauses warm-start ticks while the human
                    # decides, which is harmless (nothing is mid-episode
                    # here) and far simpler than threading the lock through
                    # every prompt closure.
                    with self._plan_lock:
                        self._on_episode_end()
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def _input_loop(self) -> None:
        """Gate episodes on Enter, off the ROS executor, until stopped.

        Owns stdin exclusively: with ``teleop`` on, a second thread also
        reading raw keys would race this one byte-for-byte, so jog keys are
        handled inline here instead, sharing the one reader.
        """
        if not self.teleop:
            self._gated_input_loop()
            return

        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)  # not setraw: keeps Ctrl+C/SIGINT working
            while not self._stop.is_set():
                print(
                    f"\nw/a/s/d jog, space/x stop, h home, Enter to roll "
                    f"out for {self.episode_duration:.0f}s (reposition the "
                    "block, then press Enter)..."
                )
                while not self._stop.is_set():
                    ready, _, _ = select.select([sys.stdin], [], [], 0.1)
                    if not ready:
                        continue
                    ch = sys.stdin.read(1)
                    self.get_logger().info(f"key read: {ch!r}")
                    if ch in ("\n", "\r"):
                        break
                    if ch == "h":
                        self._on_home()
                    elif not self._homing.is_set():
                        self._teleop_publish(ch)
                if self._stop.is_set():
                    break
                self._wait_while_homing()
                if self._stop.is_set():
                    break
                self.get_logger().info("starting rollout...")
                self._episode_running.set()
                self._episode_done.wait()
                self._episode_done.clear()
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def _on_home(self) -> None:
        """Dispatch a blocking send-home to a background thread."""
        if self._home_mover is None:
            print("no home_mover_factory wired in on this run")
            return
        if self._homing.is_set():
            return
        self._homing.set()
        self._publish_stop()  # make sure Servo's last command is zero first
        threading.Thread(target=self._home_worker, daemon=True).start()

    def _home_worker(self) -> None:
        """Run the blocking home move, then clear the busy flag."""
        self.get_logger().info("sending robot home...")
        try:
            ok = self._home_mover()
        finally:
            self._homing.clear()
        self.get_logger().info("home: reached" if ok else "home: FAILED")
        print("ready again -- w/a/s/d jog, h home, Enter to roll out")

    def _teleop_publish(self, key: str) -> None:
        """Publish a short trajectory pulse for one jog keypress.

        The first ``teleop_pulse`` seconds of the published trajectory
        carry the commanded velocity, the rest is zero -- so a single
        publish decays on its own; holding a key relies on the terminal's
        own key-repeat to keep re-publishing pulses (a rough jog aid for
        repositioning, not precision teleop).
        """
        vel = _TELEOP_KEYS.get(key)
        if vel is None:
            return
        dt = 1.0 / self.control_rate
        horizon = max(1, round(self.planner.plan_horizon * self.control_rate))
        pulse = round(self.teleop_pulse * self.control_rate)
        pulse = min(horizon, max(1, pulse))
        actions = np.zeros((horizon, self.planner.task.nu))
        if any(vel):
            cmd = np.clip(
                np.array(vel) * self.teleop_speed,
                self.planner.task.u_min,
                self.planner.task.u_max,
            )
            actions[:pulse] = cmd
        self._pub.publish(pack(time.monotonic(), dt, actions))

    def _episode_sample(self, t: float) -> tuple[float, ...]:
        """Build one collect_data CSV row for the active ``data_schema``."""
        blk_pos = self._kin_data.xpos[self._block_body_id]
        if self.data_schema == "position":
            goal_pos = self._kin_data.mocap_pos[self._goal_mid]
            return (t, *blk_pos, *goal_pos)
        pos_err, rot_err = self.planner.task.pose_error(self._kin_data)
        ee_pos = self._kin_data.xpos[self._ee_body_id]
        ee_quat = self._kin_data.xquat[self._ee_body_id]
        blk_quat = self._kin_data.xquat[self._block_body_id]
        return (
            t, pos_err, rot_err, *ee_pos, *ee_quat, *blk_pos, *blk_quat,
        )

    def _end_episode(self) -> None:
        """Stop the robot, save the episode, and unblock the next Enter."""
        self._publish_stop()
        self._save_episode()
        self._episode_samples = []
        self._episode_t0 = None
        self._coasting.clear()
        self._episode_running.clear()
        self._episode_done.set()

    def _save_episode(self) -> None:
        """Write the just-finished episode's samples to a timestamped CSV."""
        if not self._episode_samples:
            return
        self.data_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = self.data_dir / f"{stamp}_{self.data_label}.csv"
        if self.data_schema == "position":
            header = [
                "t", "blk_x", "blk_y", "blk_z",
                "goal_x", "goal_y", "goal_z",
            ]
        else:
            header = [
                "t", "pos_err", "rot_err",
                "ee_x", "ee_y", "ee_z", "ee_qw", "ee_qx", "ee_qy", "ee_qz",
                "blk_x", "blk_y", "blk_z", "blk_qw", "blk_qx", "blk_qy",
                "blk_qz",
            ]
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(self._episode_samples)
        self.get_logger().info(
            f"saved {len(self._episode_samples)} samples -> {path}"
        )

    def _publish(self, t0_state: float, t0_wall: float) -> None:
        """Interpolate the current spline to a dense trajectory, publish it.

        ``t0_state`` (this node's ``state.time`` clock, same origin as
        ``self.params.tk``) anchors the *spline query* -- it must line up
        with the knot times the just-finished ``optimize()`` warm-started
        from. ``t0_wall`` (``time.monotonic()``, shared by every process on
        this host) is what goes out on the wire: it's the only clock
        :class:`~bampc.ros.control_node.ControlNode` can compare
        against its own ``time.monotonic()`` reads, since it never sees
        this node's ``state.time`` origin.
        """
        dt = 1.0 / self.control_rate
        horizon = max(1, round(self.planner.plan_horizon * self.control_rate))
        tq = t0_state + np.arange(horizon) * dt
        actions = spline.interp(
            self.planner.spline_type,
            tq,
            self.params.tk,
            self.params.mean[None, ...],
        )[0]
        if self._action_filter is not None:
            actions = self._action_filter(actions, self._kin_data)
        self._pub.publish(pack(t0_wall, dt, actions))

    def _publish_stop(self) -> None:
        """Publish an all-zero trajectory -- stop the robot immediately."""
        dt = 1.0 / self.control_rate
        horizon = max(1, round(self.planner.plan_horizon * self.control_rate))
        actions = np.zeros((horizon, self.planner.task.nu))
        self._pub.publish(pack(time.monotonic(), dt, actions))
