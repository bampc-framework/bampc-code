"""Shared headless-rollout-stills machinery for ``scripts/figures/*``.

Every per-task script here (``push_rollout.py``, ``push_fr3_rollout.py``,
...) needs the same thing: draw a few small state-uncertainty perturbations
of one nominal start pose, then for each run a closed-loop rollout saving
evenly-spaced PNG stills with faint sample-trace lines and endpoint ghosts --
the same visualization ``run_interactive`` draws live, captured headless.
Only construction (which task, which scenario bank, which numerics/planner/
reward profile, which plan frequency) differs per task, so that stays in
each script; this module is the part that doesn't, mirroring how
``examples/scenario.py``/``examples/state_uncertainty/common.py`` share
plumbing across per-task example files.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

from bampc import OUTPUT_DIR
from bampc.planner.base import StateSnapshot
from bampc.sim.viewer import (
    append_line_traces,
    sync_endpoint_ghosts,
)
from bampc.uncertainty import StateUncertainty
from bampc.uncertainty.noise import SE3Gaussian

FIGURE_DIR = OUTPUT_DIR / "figures"
GHOST_RGBA = np.array([[0.1, 0.4, 0.95, 0.35]])  # translucent blue
# Faint white, matching run_interactive's single-domain default trace_color.
TRACE_RGBA = np.array([[1.0, 1.0, 1.0, 0.3]])


def draw_conditions(
    task,
    nominal: mujoco.MjData,
    num_conditions: int,
    pos_std: float,
    rot_std: float,
    seed: int,
) -> np.ndarray:
    """Perturbed initial ``qpos`` rows, position/yaw only (velocity stays 0).

    Uses ``SE3Gaussian`` directly rather than a named ``configs/noise/``
    preset: the presets also bundle ``TwistGaussian``, which would inject
    spurious initial velocity into what should be a resting block.
    """
    snap = StateSnapshot(
        qpos=nominal.qpos.copy(), qvel=nominal.qvel.copy(), time=nominal.time
    )
    su = StateUncertainty(
        task,
        num_randomizations=num_conditions,
        noise=[SE3Gaussian(pos_std=pos_std, rot_std=rot_std)],
        seed=seed,
    )
    qpos_cloud, _ = su.sample(snap)
    return qpos_cloud


def make_goal_updater(task, vdata: mujoco.MjData, n_base_mocap: int):
    """Return a closure that drives a drifting goal mocap by hand.

    Mocap bodies don't move under ``mj_step`` on their own, and Push /
    Push-FR3's default scenario banks drift the goal, so it has to be
    driven each step -- mirroring ``run_interactive``'s ``_update_goal``.
    """
    goal_mid = task.goal_mocap_id
    if goal_mid is not None and goal_mid >= n_base_mocap:
        goal_mid = None
    base_pos = base_quat = None
    if goal_mid is not None:
        base_pos = np.array(vdata.mocap_pos[goal_mid])
        base_quat = np.array(vdata.mocap_quat[goal_mid])

    def update_goal() -> None:
        if goal_mid is None:
            return
        pose = task.goal_mocap_pose(
            float(vdata.time), vdata, base_pos, base_quat
        )
        if pose is None:
            return
        pos, quat = pose
        vdata.mocap_pos[goal_mid] = pos
        if quat is not None:
            vdata.mocap_quat[goal_mid] = quat

    return update_goal


class StillCapture:
    """Evenly-spaced PNG stills of ``vdata``, rendered through ``renderer``.

    Capture times are ``num_frames`` evenly-spaced points over
    ``(start, start + duration]``, deliberately excluding the exact start --
    at ``t = 0`` no control has been applied yet, so every condition's first
    still would just show its (barely-perturbed) rest pose rather than the
    plan actually acting on it.

    Call :meth:`set_trace` after each replan with the current sample-line /
    ghost state, :meth:`drain` after every sim step, and :meth:`flush` once
    at the end (rounding can leave the last capture time just past the
    loop's exit condition).
    """

    def __init__(
        self,
        vdata: mujoco.MjData,
        renderer: mujoco.Renderer,
        scene_option: mujoco.MjvOption,
        camera: str,
        duration: float,
        num_frames: int,
        out_dir: Path,
    ) -> None:
        self.vdata = vdata
        self.renderer = renderer
        self.scene_option = scene_option
        self.camera = camera
        self.out_dir = out_dir
        self.start_time = float(vdata.time)
        self.times = self.start_time + duration * (
            np.arange(1, num_frames + 1) / num_frames
        )
        self.next_idx = 0
        self.trace: tuple | None = None

    def set_trace(self, trace: tuple) -> None:
        """Latest (trace_sites, idxs, num_sites, palette, width) to overlay."""
        self.trace = trace

    def _capture_one(self, idx: int) -> None:
        self.renderer.update_scene(
            self.vdata, camera=self.camera, scene_option=self.scene_option
        )
        if self.trace is not None:
            append_line_traces(self.renderer.scene, *self.trace)
        t = self.vdata.time - self.start_time
        path = self.out_dir / f"frame_{idx:02d}_t{t:05.2f}s.png"
        Image.fromarray(self.renderer.render()).save(path)

    def drain(self) -> None:
        """Capture every still whose scheduled time has now passed."""
        while self.next_idx < len(self.times) and (
            self.vdata.time >= self.times[self.next_idx]
        ):
            self._capture_one(self.next_idx)
            self.next_idx += 1

    def flush(self) -> None:
        """Capture any stills :meth:`drain` never reached."""
        while self.next_idx < len(self.times):
            self._capture_one(self.next_idx)
            self.next_idx += 1


def run_condition(
    planner,
    vmodel: mujoco.MjModel,
    n_base_mocap: int,
    qpos0: np.ndarray,
    qvel0: np.ndarray,
    ghost_ids: np.ndarray,
    idxs: list[int],
    num_sites: int,
    *,
    plan_freq: float,
    duration: float,
    num_frames: int,
    camera: str,
    size: tuple[int, int],
    out_dir: Path,
) -> None:
    """Roll out one initial condition and save its evenly-spaced stills.

    Mirrors ``run_interactive``'s core replan/step loop
    (``bampc/sim/viewer.py``), stripped of the window / allocation /
    belief-ghost bits: single domain, endpoint ghosts + sample-trace lines
    only, rendered headless instead of shown live.
    """
    vdata = mujoco.MjData(vmodel)
    vdata.qpos[:] = qpos0
    vdata.qvel[:] = qvel0
    mujoco.mj_forward(vmodel, vdata)

    task = planner.task
    sim_dt = vmodel.opt.timestep
    steps_per_replan = max(int(round(1.0 / plan_freq / sim_dt)), 1)
    params = planner.init_params()

    update_goal = make_goal_updater(task, vdata, n_base_mocap)
    update_goal()

    renderer = mujoco.Renderer(vmodel, size[0], size[1])
    scene_option = mujoco.MjvOption()
    scene_option.sitegroup[:] = 0
    still = StillCapture(
        vdata, renderer, scene_option, camera, duration, num_frames, out_dir
    )

    start_time = still.start_time
    while vdata.time - start_time < duration:
        state = StateSnapshot(
            qpos=vdata.qpos[: task.mj_model.nq].copy(),
            qvel=vdata.qvel[: task.mj_model.nv].copy(),
            time=float(vdata.time),
            mocap_pos=np.array(vdata.mocap_pos[:n_base_mocap]),
            mocap_quat=np.array(vdata.mocap_quat[:n_base_mocap]),
        )
        params, info = planner.optimize(state, params)

        if info.trace_sites is not None:
            still.set_trace(
                (info.trace_sites[0:1], idxs, num_sites, TRACE_RGBA, 3.0)
            )
        pos, quat = planner.engine.final_body_pose("block")
        sync_endpoint_ghosts(vdata, ghost_ids[0][:1], idxs, pos, quat)

        for _ in range(steps_per_replan):
            u = planner.get_action(params, float(vdata.time))
            vdata.ctrl[:] = task.control_map_host(vdata, u)
            mujoco.mj_step(vmodel, vdata)
            update_goal()
            still.drain()

        if vdata.time - start_time >= duration:
            break

    still.flush()
