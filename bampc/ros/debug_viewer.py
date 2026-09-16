"""CPU passive-viewer helper: show what a state pipeline is reading, live.

Independent of the GPU rollout -- this drives *no* physics, only forward
kinematics (+ position-stage sensors), so it can display exactly the
:class:`~bampc.planner.base.StateSnapshot` a state reader produces
without simulating anything. Two call sites:
:class:`~bampc.ros.planner_node.PlannerNode` (optional, toggled,
also the one that uses :func:`start_recording`/:func:`capture_frame` for
real-robot rollout videos) and
:class:`~bampc.ros.state_viewer_node.StateViewerNode`
(its whole job) -- both just call the same state reader and feed its output
through :func:`sync`, so the tracking/encoder pipeline looks identical in
both places.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import mujoco
import mujoco.viewer
import numpy as np

if TYPE_CHECKING:
    from bampc.planner.base import StateSnapshot


def launch(
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData | None = None,
    show_sites: bool = False,
) -> tuple[mujoco.MjData, mujoco.viewer.Handle]:
    """Open a passive viewer on ``mj_data`` (a fresh one if not given).

    Passing an existing ``mj_data`` lets a caller share one synced buffer
    between the viewer window, off-screen recording (:func:`capture_frame`)
    and anything else reading body poses each tick (e.g. a real-robot
    planner node's goal tracking or pose-error logging).

    ``show_sites``: sites (attractor / EE / peg tip) are cost and IK
    reference frames, not geometry -- off by default, mirroring
    :func:`bampc.sim.viewer.run_interactive`'s ``show_sites``. Sites
    all sit in group 0; zeroing the sitegroup hides their spheres.
    """
    if mj_data is None:
        mj_data = mujoco.MjData(mj_model)
    handle = mujoco.viewer.launch_passive(mj_model, mj_data)
    if not show_sites:
        # Sites all sit in group 0; zeroing the group hides their spheres.
        # Locked like bampc.sim.viewer's own toggle -- the passive
        # viewer renders on its own thread.
        with handle.lock():
            handle.opt.sitegroup[:] = 0
    return mj_data, handle


def update(
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    state: StateSnapshot,
) -> None:
    """Write ``state`` into ``mj_data`` and refresh kinematics + sensors.

    Forward kinematics only -- there is nothing to simulate, only to
    display -- plus the position- and velocity-stage sensors (cheap, no
    dynamics), so a caller can read e.g. a task's block/goal sensors
    afterward without a display attached. Use :func:`sync` when there's a
    viewer window.

    The velocity stage is not optional: ``qvel`` is written here, so a task
    whose success predicate reads a ``framelinvel`` sensor (CurlingFr3's
    "puck has stopped") would otherwise see a permanent zero and fire on
    position alone.
    """
    mj_data.qpos[: mj_model.nq] = state.qpos
    mj_data.qvel[: mj_model.nv] = state.qvel
    mj_data.time = state.time
    if state.mocap_pos is not None:
        mj_data.mocap_pos[: len(state.mocap_pos)] = state.mocap_pos
    if state.mocap_quat is not None:
        mj_data.mocap_quat[: len(state.mocap_quat)] = state.mocap_quat
    mujoco.mj_kinematics(mj_model, mj_data)
    mujoco.mj_comPos(mj_model, mj_data)
    mujoco.mj_sensorPos(mj_model, mj_data)
    mujoco.mj_comVel(mj_model, mj_data)
    mujoco.mj_sensorVel(mj_model, mj_data)


def sync(
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    handle: mujoco.viewer.Handle,
    state: StateSnapshot,
) -> None:
    """:func:`update`, then refresh the viewer window."""
    update(mj_model, mj_data, state)
    handle.sync()


def start_recording(
    mj_model: mujoco.MjModel,
    camera: str,
    size: tuple[int, int] = (480, 640),
) -> mujoco.Renderer:
    """Open an off-screen renderer for capturing frames.

    Independent of the passive viewer window -- only needs ``mj_model`` and
    a named camera, mirroring :mod:`bampc.sim.viewer`'s
    ``record_camera`` contract so the real-robot recorder behaves the same
    way.
    """
    if mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_CAMERA, camera) < 0:
        raise ValueError(f"record_camera {camera!r} is not in the model")
    return mujoco.Renderer(mj_model, size[0], size[1])


def capture_frame(
    renderer: mujoco.Renderer,
    mj_data: mujoco.MjData,
    camera: str,
    frames: list[np.ndarray],
) -> None:
    """Render and append one off-screen frame from ``mj_data``."""
    renderer.update_scene(mj_data, camera=camera)
    frames.append(renderer.render().copy())
