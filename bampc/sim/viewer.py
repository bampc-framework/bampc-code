"""Interactive MuJoCo-viewer driver for a SamplingPlanner.

A thin, synchronous client of the planner (the real-robot counterpart is the ROS
node). Each frame: build a :class:`StateSnapshot` from ``mj_data`` -> plan ->
draw the enabled visualizations -> query the spline at the sim rate -> map to
actuators via ``task.control_map_host`` -> step the host simulation.

Three independent visualization toggles, all driven by the rollout's trace
sites ``(R, S, H+1, num_sites, 3)``:

* ``show_traces`` -- line traces of the selected sample rollouts (first
  domain only, single color).
* ``show_domain_traces`` -- the same traces for *every* domain, one color
  each (builds on ``show_traces``).
* ``show_endpoints`` -- a translucent mocap ghost at the **final** trace
  point of each selected rollout, per domain, so you can read off where each
  plan ends up under each randomization.

Select samples with explicit ``trace_idxs`` or a ``max_traces`` cap (spread
uniformly); the traced *sites* are whatever the task was built with.
"""

from __future__ import annotations

import colorsys
import shutil
import time
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import imageio
import mujoco
import mujoco.viewer
import numpy as np
from PIL import Image

from bampc import OUTPUT_DIR
from bampc.allocation import AllocationContext
from bampc.planner.base import StateSnapshot

if TYPE_CHECKING:
    from bampc.allocation import AllocationController
    from bampc.planner.base import SamplingPlanner
    from bampc.tracking import PredictionTracker

# GLFW key codes: down/up arrows nudge the active domain count down / up
# when an AllocationController is wired in.
_KEY_FEWER_DOMAINS = 264  # GLFW_KEY_DOWN
_KEY_MORE_DOMAINS = 265  # GLFW_KEY_UP

# Default output dir for recorded rollouts.
_VIDEO_DIR = OUTPUT_DIR / "recordings"

# Flat color for belief-cloud ghosts (see run_interactive's show_belief).
# Module-level so bampc.ros.planner_node can build the same ghosts
# for the real-robot debug viewer without redefining the color.
BELIEF_RGBA = np.array([1.0, 1.0, 0.0, 0.45], np.float32)


def domain_palette(num_domains: int, alpha: float = 0.2) -> np.ndarray:
    """Evenly-spaced HSV colors (RGBA) for distinguishing domains."""
    pal = np.zeros((num_domains, 4), np.float32)
    for d in range(num_domains):
        r, g, b = colorsys.hsv_to_rgb(d / max(num_domains, 1), 0.8, 0.95)
        pal[d] = (r, g, b, alpha)
    return pal


def select_trace_idxs(
    num_samples: int,
    max_traces: int | None,
    trace_idxs: Sequence[int] | None,
) -> list[int]:
    """Resolve which sample rollouts to visualize (explicit or uniform)."""
    if trace_idxs is not None:
        return [i for i in trace_idxs if i < num_samples]
    if max_traces is not None and max_traces < num_samples:
        return np.linspace(0, num_samples - 1, max_traces, dtype=int).tolist()
    return list(range(num_samples))


def _clone_body_geoms(
    ghost: mujoco.MjsBody, src: mujoco.MjsBody, rgba: np.ndarray
) -> None:
    """Copy ``src``'s geoms onto ``ghost`` as translucent, non-colliding viz.

    Copies the geometry-defining fields (type, size, pos, quat, plus mesh name
    and ``fromto`` when used), so the ghost matches *any* tracked object without
    task-specific code. Collision flags are cleared and the color overridden.
    """
    for g in src.geoms:
        ng = ghost.add_geom()
        ng.type = g.type
        ng.size = g.size
        ng.pos = g.pos
        ng.quat = g.quat
        ng.meshname = g.meshname  # no-op when empty; asset is in the same spec
        if not np.isnan(g.fromto[0]):  # capsule/cylinder defined endpoint-wise
            ng.fromto = g.fromto
        ng.rgba = list(rgba)
        ng.contype = 0
        ng.conaffinity = 0


def build_ghosts(
    mj_spec: mujoco.MjSpec,
    body_names: Sequence[str],
    groups: dict[str, tuple[int, int, np.ndarray]],
) -> tuple[mujoco.MjModel, dict[str, tuple[np.ndarray, np.ndarray]]]:
    """Extend a copy of ``mj_spec`` with one or more sets of mocap ghosts.

    Ghosts are laid out on three axes: one per tracked *body* (geoms cloned
    from it), fanned out over *domains* and *rollouts*. An empty
    ``body_names`` builds one slot of sphere markers instead.

    All groups are compiled into **one** model -- a spec can only be compiled
    once, which is why this takes every group at once rather than being called
    per group.

    ``num_domains`` must be the *maximum* count an adaptive ladder will ever
    use, not today's active count: the mocap body count is compiled in once
    and cannot change afterwards. :func:`set_ghost_domain_alpha` hides the
    unused tail at runtime instead.

    Args:
        mj_spec: The task's (already shape-composed) spec; copied so the
            task's own build spec is left untouched.
        body_names: Bodies whose geoms are cloned, one ghost group each. Empty
            builds a single sphere-marker group (position-only fallback).
        groups: ``{name: (num_domains, ghosts_per_domain, palette)}``, where
            ``palette`` is ``(num_domains, 4)`` RGBA.

    Returns:
        The compiled augmented model and, per group name, an
        ``(num_bodies, num_domains, ghosts_per_domain)`` array of mocap ids
        plus the matching array of body ids (for
        :func:`set_ghost_domain_alpha` / :func:`set_ghost_contact_color`).
        ``num_bodies`` is ``max(len(body_names), 1)``.
    """
    spec = mj_spec.copy()
    srcs = [spec.body(n) for n in body_names]
    num_bodies = max(len(srcs), 1)

    def decorate(
        body: mujoco.MjsBody, rgba: np.ndarray, src: mujoco.MjsBody | None
    ) -> None:
        if src is not None:
            _clone_body_geoms(body, src, rgba)
        else:  # no tracked body: a generic position marker
            g = body.add_geom()
            g.type = mujoco.mjtGeom.mjGEOM_SPHERE
            g.size = [0.01, 0.0, 0.0]
            g.rgba = list(rgba)
            g.contype = 0
            g.conaffinity = 0

    names: dict[str, np.ndarray] = {}
    for group, (num_domains, per_domain, palette) in groups.items():
        arr = np.empty((num_bodies, num_domains, per_domain), dtype=object)
        for bi in range(num_bodies):
            src = srcs[bi] if srcs else None
            for r in range(num_domains):
                for p in range(per_domain):
                    b = spec.worldbody.add_body()
                    b.name = f"ghost_{group}_{bi}_{r}_{p}"
                    b.mocap = True
                    decorate(b, palette[r], src)
                    arr[bi, r, p] = b.name
        names[group] = arr

    model = spec.compile()
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for group, arr in names.items():
        ids = np.empty(arr.shape, dtype=int)
        body_ids = np.empty_like(ids)
        for idx in np.ndindex(arr.shape):
            body = model.body(str(arr[idx]))
            ids[idx] = body.mocapid[0]
            body_ids[idx] = body.id
        out[group] = (ids, body_ids)
    return model, out


def set_ghost_domain_alpha(
    vmodel: mujoco.MjModel,
    ghost_body_ids: np.ndarray,
    active_domains: int,
    base_alpha: float,
) -> None:
    """Show ghosts for the first ``active_domains``, hide the rest.

    Visual-only (``geom_rgba`` alpha), so it's safe to call every frame
    without recompiling the model -- the mechanism an adaptive budget ladder
    uses to make "number of domains" look adjustable at runtime even though
    the mocap body count was fixed at :func:`build_endpoint_ghosts` time.

    Args:
        vmodel: The viewer's compiled model (mutated in place).
        ghost_body_ids: ``(num_domains, ghosts_per_domain)`` from
            :func:`build_endpoint_ghosts`.
        active_domains: How many domains (from index 0) should be visible.
        base_alpha: Alpha to restore for visible ghosts.
    """
    for r, row in enumerate(ghost_body_ids):
        alpha = base_alpha if r < active_domains else 0.0
        for bid in row:
            start = vmodel.body_geomadr[bid]
            count = vmodel.body_geomnum[bid]
            vmodel.geom_rgba[start : start + count, 3] = alpha


# Contact-mode names, indexed by bitmask (bit 0 pusher, bit 1 table).
_MODE_NAMES = ("none", "push", "table", "both")


def contact_palette(alpha: float = 0.45) -> np.ndarray:
    """RGBA per contact bitmask, indexed by the mask itself.

    Bit 0 is the pusher-object probe, bit 1 the object-table probe (the
    order a task declares in ``contact_probes``), so::

        0  no contact          grey
        1  pusher only         yellow
        2  table only          blue
        3  both -- pushing     green

    Replaces the per-domain hue on belief ghosts: with a whole cloud on
    screen, "is this domain in contact?" is the question worth a color.
    """
    return np.array(
        [
            [0.55, 0.55, 0.55, alpha],  # 0b00 floating / free
            [0.95, 0.85, 0.15, alpha],  # 0b01 pusher only
            [0.25, 0.45, 0.95, alpha],  # 0b10 resting on the table
            [0.20, 0.85, 0.35, alpha],  # 0b11 pushing on the table
        ],
        dtype=np.float32,
    )


def set_ghost_contact_color(
    vmodel: mujoco.MjModel,
    ghost_body_ids: np.ndarray,
    modes: np.ndarray,
    palette: np.ndarray,
) -> None:
    """Recolor ghosts by their domain's contact mode.

    Visual-only (``geom_rgba``), the same mechanism as
    :func:`set_ghost_domain_alpha`, so it is safe every frame without
    recompiling. Domains beyond ``len(modes)`` are left alone.

    Args:
        vmodel: The viewer's compiled model (mutated in place).
        ghost_body_ids: ``(num_domains, ghosts_per_domain)`` body ids.
        modes: ``(R,)`` contact bitmasks from ``engine.contact_modes()``.
        palette: ``(4, 4)`` RGBA from :func:`contact_palette`.
    """
    for r, row in enumerate(ghost_body_ids):
        if r >= len(modes):
            break
        rgba = palette[int(modes[r]) & 0b11]
        for bid in row:
            start = vmodel.body_geomadr[bid]
            count = vmodel.body_geomnum[bid]
            vmodel.geom_rgba[start : start + count] = rgba


def sync_endpoint_ghosts(
    vdata: mujoco.MjData,
    ghost_mocap_ids: np.ndarray,
    sample_idxs: Sequence[int],
    pos: np.ndarray,
    quat: np.ndarray | None = None,
) -> None:
    """Place each ghost at the final pose of its (domain, rollout).

    Args:
        vdata: Viewer data whose ``mocap_pos`` / ``mocap_quat`` is written.
        ghost_mocap_ids: ``(num_domains, len(sample_idxs))`` mocap ids.
        sample_idxs: Which sampled rollouts the ghosts follow.
        pos: Endpoint positions, shape ``(R, S, 3)``.
        quat: Optional endpoint orientations (MuJoCo w,x,y,z), shape
            ``(R, S, 4)``. When given, the ghost shows the predicted yaw; when
            ``None``, orientation is left at identity.
    """
    # ghost_mocap_ids is sized to the domain count at ghost-creation time;
    # an adaptive allocation switch (allocation.py) can shrink the live
    # domain count below that afterwards, so cap to pos/quat's actual R
    # rather than assume they still match.
    for r in range(min(ghost_mocap_ids.shape[0], pos.shape[0])):
        for p, i in enumerate(sample_idxs):
            mid = ghost_mocap_ids[r, p]
            vdata.mocap_pos[mid] = pos[r, i]
            if quat is not None:
                vdata.mocap_quat[mid] = quat[r, i]


def _draw_line_traces(
    viewer,
    trace_data: np.ndarray,
    sample_idxs: Sequence[int],
    num_sites: int,
    palette: np.ndarray,
    width: float,
) -> None:
    """Connect consecutive trace points into line geoms.

    ``trace_data`` has shape ``(D, S, H+1, num_sites, 3)`` where ``D`` is the
    number of domains being drawn (1 or R).
    """
    horizon = trace_data.shape[2] - 1
    ii = 0
    for k in range(num_sites):
        for d in range(trace_data.shape[0]):
            for i in sample_idxs:
                for j in range(horizon):
                    mujoco.mjv_connector(
                        viewer.user_scn.geoms[ii],
                        mujoco.mjtGeom.mjGEOM_LINE,
                        width,
                        trace_data[d, i, j, k],
                        trace_data[d, i, j + 1, k],
                    )
                    ii += 1


def append_line_traces(
    scene: mujoco.MjvScene,
    trace_data: np.ndarray,
    sample_idxs: Sequence[int],
    num_sites: int,
    palette: np.ndarray,
    width: float,
) -> None:
    """Append trace connectors onto an already-populated scene.

    The offscreen recorder's ``renderer.scene`` is rebuilt from the model each
    frame, so (unlike :func:`_draw_line_traces`, which overwrites a fixed
    user-scene block) this adds geoms *after* whatever ``update_scene`` placed,
    letting the recording show the same lines as the interactive window. Stops
    early if the scene's geom buffer fills. Shared with
    ``scripts/figures/push_rollout.py``, a second headless caller.

    ``trace_data`` has shape ``(D, S, H+1, num_sites, 3)``.
    """
    horizon = trace_data.shape[2] - 1
    for k in range(num_sites):
        for d in range(trace_data.shape[0]):
            for i in sample_idxs:
                for j in range(horizon):
                    if scene.ngeom >= scene.maxgeom:
                        return
                    g = scene.geoms[scene.ngeom]
                    mujoco.mjv_initGeom(
                        g,
                        mujoco.mjtGeom.mjGEOM_LINE,
                        np.zeros(3),
                        np.zeros(3),
                        np.eye(3).flatten(),
                        palette[d % len(palette)],
                    )
                    mujoco.mjv_connector(
                        g,
                        mujoco.mjtGeom.mjGEOM_LINE,
                        width,
                        trace_data[d, i, j, k],
                        trace_data[d, i, j + 1, k],
                    )
                    scene.ngeom += 1


def save_video(
    frames: list[np.ndarray],
    directory: Path,
    name: str,
    fps: int,
    fmt: Literal["mp4", "gif"] = "mp4",
) -> Path:
    """Write captured RGB frames to a timestamped video file.

    Frames are ``(H, W, 3)`` uint8 arrays. Returns the written path. Public
    (and reused by :mod:`bampc.ros.debug_viewer`) so the real-robot
    recorder writes the exact same file format as the sim.
    """
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = directory / f"{stamp}_{name}.{fmt}"
    if fmt == "gif":
        images = [Image.fromarray(f) for f in frames]
        images[0].save(
            path,
            save_all=True,
            append_images=images[1:],
            duration=int(round(1000 / fps)),
            loop=0,
        )
    else:
        # imageio's default "quality" (5/10) visibly compresses a rendered
        # scene's flat colors and hard edges; 10 asks ffmpeg for its own
        # near-lossless default (crf ~0) instead.
        with imageio.get_writer(path, fps=fps, quality=10) as writer:
            for frame in frames:
                writer.append_data(frame)
    return path


def run_interactive(  # noqa: PLR0912, PLR0915
    planner: SamplingPlanner,
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    *,
    frequency: float,
    initial_knots: np.ndarray | None = None,
    show_traces: bool = True,
    show_domain_traces: bool = False,
    show_endpoints: bool = False,
    show_belief: bool = False,
    show_sites: bool = False,
    max_traces: int | None = None,
    trace_idxs: Sequence[int] | None = None,
    trace_width: float = 3.0,
    trace_color: Sequence[float] = (1.0, 1.0, 1.0, 0.3),
    status_callback: Callable[[mujoco.MjData], str] | None = None,
    observer: Callable[[StateSnapshot], StateSnapshot] | None = None,
    allocation: AllocationController | None = None,
    tracker: PredictionTracker | None = None,
    plan_lag_steps: int = 0,
    compensate_latency: bool = False,
    duration: float | None = None,
    record: bool = False,
    record_camera: str | None = None,
    record_name: str = "rollout",
    record_fps: int = 30,
    record_size: tuple[int, int] = (480, 640),
    record_dir: str | Path | None = None,
    record_format: Literal["mp4", "gif"] = "mp4",
) -> None:
    """Run the interactive sim loop.

    Args:
        planner: A constructed planner (owns its rollout engine).
        mj_model: Host model for the visualized simulation.
        mj_data: Initial state.
        frequency: Requested replanning frequency (Hz).
        initial_knots: Optional warm-start knots at ``t = 0``.
        show_traces: Draw line traces of selected rollouts (first domain).
        show_domain_traces: Draw line traces for every domain, colored per
            domain. Implies ``show_traces``. Not supported together with
            ``allocation`` (line-geom count/coloring is fixed at launch).
        show_endpoints: Draw a mocap ghost of each of the task's
            ``endpoint_bodies`` at the final pose of every selected rollout,
            per domain, in one flat red. Falls back to a position-only sphere
            when the task sets no endpoint bodies. Domain spread reads from
            ghost *position*, not color.
        show_belief: Draw one ghost per domain at the pose that domain
            *believes* the object is in now -- the post-settle pose the
            planner actually rolls out from. Flat yellow, recolored by
            contact mode (see :func:`contact_palette`) when the task declares
            ``contact_probes``. One flat color per ghost GROUP rather than a
            per-domain rainbow: with both groups on, a shared rainbow made an
            endpoint and a belief ghost from the same domain slot
            indistinguishable. Independent of ``show_endpoints``.
        show_sites: Draw the model's sites, which MuJoCo renders as spheres.
            Off by default -- they are cost and IK reference frames, not
            geometry. Applies to the recording too.
        max_traces: Cap on visualized rollouts (uniformly spread); ignored if
            ``trace_idxs`` is given. With ``allocation`` this must stay ``<=``
            every ladder stage's sample count, so the ghosts-per-domain count
            need not change at runtime.
        trace_idxs: Explicit sample indices to visualize.
        trace_width: Line width (px) for traces.
        trace_color: RGBA for single-domain line traces.
        status_callback: Optional ``(mj_data) -> str``, appended to the status
            line each frame.
        observer: Optional ``(truth) -> estimate`` applied to the snapshot
            before it reaches the planner. Without it the planner sees the
            true state, so a ``StateUncertainty`` draws its domains straight
            from the truth -- ``R`` independent readings rather than a belief
            about one. Supply a sensor-plus-filter here for the sweeps'
            arrangement (see :mod:`examples.state_uncertainty.common`). The
            sim always steps the truth; only what the planner is told changes.
        allocation: Optional adaptive-budget controller. The up/down arrow
            keys then step the domain count through its ladder each replan;
            ghosts are pre-built for the ladder's max domain count and
            shown/hidden as the active count changes.
        tracker: Optional per-replan domain-prediction scorer (see
            :mod:`bampc.tracking`), feeding ``allocation`` its
            ``AllocationContext.error_window``. Needs the engine built with
            ``record_predictions=True`` *and* the planner with
            ``track_predictions=True``, or every chunk scores as ``None``.
            Reset on every stage switch, which redraws every domain.
        plan_lag_steps: Planning-latency model (0 = off). Step the sim this
            many times under the previous plan before a fresh one lands, so
            the action arrives on a state already ``plan_lag_steps * dt``
            stale.
        compensate_latency: Predict the state forward by ``plan_lag_steps *
            dt`` before ``optimize``, undoing that staleness. Lets a caller
            A/B the compensation at an exactly-known lag before relying on it
            against real, only-estimated latency.
        duration: Sim seconds before stopping. ``None`` runs until the window
            closes. Checked once per replan cycle, so the stop can overshoot
            by up to one cycle; recorded frames are gated on sim time, so
            video length is unaffected.
        record: Capture the rollout to a video, flushed on exit (Ctrl-C
            included).
        record_camera: Model camera to render from. Required when ``record``
            is set; raises otherwise.
        record_name: Filename stem
            (``<timestamp>_<record_name>.<record_format>``).
        record_fps: Target playback frame rate (frames are subsampled to it).
        record_size: ``(height, width)``. Must fit the model's off-screen
            buffer (MuJoCo default 480x640).
        record_dir: Output directory; defaults to ``output/recordings/``.
        record_format: ``"mp4"`` (default) or ``"gif"``.

    Any visualization requires the engine to have been built with
    ``record_traces=True``.
    """
    if allocation is not None and show_domain_traces:
        raise NotImplementedError(
            "show_domain_traces isn't supported together with allocation "
            "yet -- its line geoms are sized/colored for a fixed domain "
            "count at launch. Use show_endpoints instead."
        )
    if show_belief and getattr(planner, "state_uncertainty", None) is None:
        raise ValueError(
            "show_belief=True needs the planner to have a state_uncertainty; "
            "without one every domain holds the same state and the 'cloud' "
            "is a single point. Pass state_uncertainty= to build_planner."
        )
    if show_belief and not planner.task.endpoint_bodies:
        raise ValueError(
            "show_belief=True needs the task to declare endpoint_bodies (the "
            "belief ghosts clone their geometry from those bodies)"
        )
    if record and record_camera is None:
        raise ValueError(
            "record=True requires record_camera (a camera name defined in "
            "the model XML); none was given."
        )

    task = planner.task
    R = planner.engine.num_randomizations
    R_max = allocation.stages[-1][0] if allocation is not None else R
    H = planner.ctrl_steps
    num_sites = len(task.trace_site_ids)
    show_traces = show_traces or show_domain_traces

    idxs = select_trace_idxs(planner.num_samples, max_traces, trace_idxs)
    num_traces = len(idxs)

    # One hue per domain *slot*, used for line traces (show_domain_traces).
    # Sized to R_max so slot colors stay stable as the active domain count
    # changes.
    hues = domain_palette(R_max, alpha=0.2)

    # Ghosts use one flat color per GROUP instead of a per-domain rainbow --
    # endpoint vs belief is the distinction that matters when both are on
    # (see show_endpoints/show_belief's docstrings: "independent... both can
    # be on"), and a shared rainbow made the two indistinguishable at a
    # glance (same hue per domain slot, differing only in alpha). Domain-to-
    # domain spread is read from ghost POSITION now, not color.
    _ENDPOINT_RGBA = np.array([1.0, 0.0, 0.0, 0.2], np.float32)  # red
    ghost_palette = np.tile(_ENDPOINT_RGBA, (R_max, 1))

    # (c) endpoint ghosts live in the model, so build it up front. Geometry is
    # cloned from the task's tracked bodies (task.endpoint_bodies). Built for
    # R_max ghosts even if fewer are active now -- see build_endpoint_ghosts.
    # Both ghost sets must be declared before the single spec compile.
    groups: dict[str, tuple[int, int, np.ndarray]] = {}
    if show_endpoints:
        groups["endpoint"] = (R_max, num_traces, ghost_palette)
    if show_belief:
        # One per domain, not per rollout: a belief is a property of the
        # domain. Starts on the flat belief color; recolored per contact
        # mode below when the task declares probes.
        groups["belief"] = (R_max, 1, np.tile(BELIEF_RGBA, (R_max, 1)))

    ghost_ids = ghost_body_ids = None
    belief_ids = belief_body_ids = None
    if groups:
        vmodel, built = build_ghosts(
            task.mj_spec, task.endpoint_bodies, groups
        )
        if show_endpoints:
            ghost_ids, ghost_body_ids = built["endpoint"]
            if allocation is not None:
                for per_body in ghost_body_ids:
                    set_ghost_domain_alpha(vmodel, per_body, R, base_alpha=0.2)
        if show_belief:
            belief_ids, belief_body_ids = built["belief"]
            if allocation is not None:
                for per_body in belief_body_ids:
                    set_ghost_domain_alpha(vmodel, per_body, R, base_alpha=0.45)
    else:
        vmodel = mj_model

    # Contact coloring needs both a task that declares probes and an engine
    # built to track them; without either, belief ghosts keep their domain
    # hue. Warn rather than raise -- an uncolored cloud is still useful.
    tracked = getattr(planner.engine, "record_initial_state", False)
    c_palette = None
    if show_belief and task.contact_probes and not tracked:
        print(
            "[note] task declares contact_probes but the engine was built "
            "with record_initial_state=False -- belief ghosts will show domain "
            "hues, not contact modes."
        )
    elif task.contact_probes and tracked:
        c_palette = contact_palette()

    # build_endpoint_ghosts recompiles task.mj_spec, which is structure-only
    # (solver/integrator options live in the task's ModelConfig, not the XML).
    # Without re-applying it, vmodel reverts to MuJoCo's compile defaults
    # (e.g. pyramidal cone, dt 0.002) and the viewer would step different
    # physics than the planner. No-op when vmodel is mj_model.
    if task.model_config is not None:
        task.model_config.apply_to(vmodel)

    vdata = mujoco.MjData(vmodel)
    vdata.qpos[: mj_model.nq] = mj_data.qpos
    vdata.qvel[: mj_model.nv] = mj_data.qvel
    n_base_mocap = mj_model.nmocap
    if n_base_mocap > 0:
        vdata.mocap_pos[:n_base_mocap] = mj_data.mocap_pos[:n_base_mocap]
        vdata.mocap_quat[:n_base_mocap] = mj_data.mocap_quat[:n_base_mocap]
    mujoco.mj_forward(vmodel, vdata)

    # Moving-goal support: if the task drives a goal mocap, write its pose
    # into vdata each frame (it flows to the rollout cost via StateSnapshot's
    # mocap fields). goal_mocap_pose returns None when there is nothing to do
    # (e.g. a static Push/Push-FR3 goal), so this is a no-op unless enabled.
    goal_mid = task.goal_mocap_id
    if goal_mid is None or goal_mid >= n_base_mocap:
        goal_mid = None
        base_goal_pos = base_goal_quat = None
    else:
        base_goal_pos = np.array(vdata.mocap_pos[goal_mid])
        base_goal_quat = np.array(vdata.mocap_quat[goal_mid])

    def _update_goal() -> None:
        """Refresh the goal mocap from the task's time/state-driven pose."""
        if goal_mid is None:
            return
        pose = task.goal_mocap_pose(
            float(vdata.time), vdata, base_goal_pos, base_goal_quat
        )
        if pose is None:
            return
        pos, quat = pose
        vdata.mocap_pos[goal_mid] = pos
        if quat is not None:
            vdata.mocap_quat[goal_mid] = quat

    _update_goal()

    # Video capture (optional): render vmodel/vdata off-screen through the
    # named camera, subsampled to record_fps, and flush to a GIF on exit
    # (see the try/finally around the loop -- Ctrl-C still saves).
    record_frames: list[np.ndarray] = []
    renderer = None
    record_out = Path(record_dir) if record_dir is not None else _VIDEO_DIR
    record_next = [0.0]  # next sim time (s) at which to grab a frame
    # Latest line-trace draw args, replayed into each recorded frame (the
    # window's traces live in viewer.user_scn, which the offscreen renderer
    # can't see). None until the first replan draws traces.
    record_trace_state: list[tuple | None] = [None]
    # The offscreen renderer has its own visualization options, so the site
    # toggle has to be applied to it separately from the window's viewer.opt.
    record_scene_option = mujoco.MjvOption()
    if not show_sites:
        record_scene_option.sitegroup[:] = 0
    if record:
        if mujoco.mj_name2id(
            vmodel, mujoco.mjtObj.mjOBJ_CAMERA, record_camera
        ) < 0:
            raise ValueError(
                f"record_camera {record_camera!r} is not a camera in the "
                "model; define it in the XML."
            )
        renderer = mujoco.Renderer(vmodel, record_size[0], record_size[1])

    def _capture_frame() -> None:
        """Grab one off-screen frame once the record_fps interval elapsed."""
        if not record or vdata.time < record_next[0]:
            return
        renderer.update_scene(
            vdata, camera=record_camera, scene_option=record_scene_option
        )
        if record_trace_state[0] is not None:
            append_line_traces(renderer.scene, *record_trace_state[0])
        record_frames.append(renderer.render().copy())
        record_next[0] = vdata.time + 1.0 / record_fps

    sim_dt = vmodel.opt.timestep
    replan_period = 1.0 / frequency
    if replan_period > planner.plan_horizon:
        raise ValueError(
            f"replan_period {replan_period:g}s (frequency={frequency:g}Hz) "
            f"exceeds plan_horizon {planner.plan_horizon:g}s -- execution "
            "would run past the last optimized rollout and coast on the "
            "spline's clamped extrapolation"
        )
    steps_per_replan = max(int(round(replan_period / sim_dt)), 1)
    params = planner.init_params(initial_knots=initial_knots)

    # Pending manual stage nudge from the keyboard; the callback runs on the
    # viewer's own thread, so it only ever does this one plain assignment.
    pending_nudge = [0]

    def _on_key(keycode: int) -> None:
        if keycode == _KEY_MORE_DOMAINS:
            pending_nudge[0] += 1
        elif keycode == _KEY_FEWER_DOMAINS:
            pending_nudge[0] -= 1

    key_callback = _on_key if allocation is not None else None

    # Line-trace colors: per-domain hues (b) -- ghosts use a flat per-group
    # color instead now, see above -- or a single color (a).
    if show_domain_traces:
        line_palette = hues.copy()
        line_palette[:, 3] = 0.9
    else:
        line_palette = np.asarray([trace_color], np.float32)
    domains_drawn = R if show_domain_traces else 1

    with mujoco.viewer.launch_passive(
        vmodel, vdata, key_callback=key_callback
    ) as viewer:
        if not show_sites:
            # Sites all sit in group 0; zeroing the groups hides their
            # spheres. Traces (user_scn lines) and ghosts (mocap bodies with
            # geoms) don't go through sitegroup, so both still draw.
            with viewer.lock():
                viewer.opt.sitegroup[:] = 0
        if show_traces and num_sites > 0:
            n_lines = num_sites * domains_drawn * num_traces * H
            if n_lines > viewer.user_scn.maxgeom:
                raise RuntimeError(
                    f"need {n_lines} line geoms but viewer caps at "
                    f"{viewer.user_scn.maxgeom}; lower max_traces."
                )
            gi = 0
            for k in range(num_sites):
                for d in range(domains_drawn):
                    for _ in range(num_traces * H):
                        mujoco.mjv_initGeom(
                            viewer.user_scn.geoms[gi],
                            mujoco.mjtGeom.mjGEOM_LINE,
                            np.zeros(3),
                            np.zeros(3),
                            np.eye(3).flatten(),
                            line_palette[d % len(line_palette)],
                        )
                        gi += 1
            viewer.user_scn.ngeom = gi

        start_time = float(vdata.time)
        try:
            while viewer.is_running() and (
                duration is None or vdata.time - start_time < duration
            ):
                t0 = time.time()

                if allocation is not None:
                    nudge, pending_nudge[0] = pending_nudge[0], 0
                    error_window = (
                        list(tracker.window) if tracker is not None else None
                    )
                    context = AllocationContext(
                        nudge=nudge, error_window=error_window
                    )
                    if allocation.maybe_switch(context):
                        if tracker is not None:
                            tracker.reset()
                        R = allocation.current[0]
                        new_idxs = select_trace_idxs(
                            allocation.current[1], max_traces, trace_idxs
                        )
                        if len(new_idxs) != num_traces:
                            raise RuntimeError(
                                f"stage switch changed the visualized sample "
                                f"count ({num_traces} -> {len(new_idxs)}); the "
                                "ghost/line geoms are sized for a fixed count "
                                "-- pass a max_traces <= the ladder's smallest "
                                "sample count (e.g. 1)."
                            )
                        idxs = new_idxs
                        if show_endpoints:
                            for per_body in ghost_body_ids:
                                set_ghost_domain_alpha(
                                    vmodel, per_body, R, base_alpha=0.2
                                )

                _update_goal()
                state = StateSnapshot(
                    qpos=vdata.qpos[: mj_model.nq].copy(),
                    qvel=vdata.qvel[: mj_model.nv].copy(),
                    time=float(vdata.time),
                    mocap_pos=np.array(vdata.mocap_pos[:n_base_mocap]),
                    mocap_quat=np.array(vdata.mocap_quat[:n_base_mocap]),
                )
                # Planning-lag model: advance the world under the command still
                # in effect (old params) before the fresh plan below replaces
                # them, so the plan is optimized from an already-stale snapshot.
                for _ in range(plan_lag_steps):
                    u = planner.get_action(params, float(vdata.time))
                    vdata.ctrl[:] = task.control_map_host(vdata, u)
                    mujoco.mj_step(vmodel, vdata)
                    _update_goal()
                    viewer.sync()
                    _capture_frame()

                plan_start = time.time()
                if observer is not None:
                    state = observer(state)
                dt_lag = (
                    plan_lag_steps * task.dt if compensate_latency else 0.0
                )
                params, info = planner.optimize(state, params, dt_lag=dt_lag)
                plan_time = time.time() - plan_start
                if tracker is not None:
                    tracker.on_replan(info.predicted_state)

                plot_start = time.time()
                if (
                    show_traces
                    and num_sites > 0
                    and info.trace_sites is not None
                ):
                    td = info.trace_sites
                    td = td if show_domain_traces else td[0:1]
                    _draw_line_traces(
                        viewer, td, idxs, num_sites, line_palette, trace_width
                    )
                    record_trace_state[0] = (
                        td, idxs, num_sites, line_palette, trace_width
                    )
                if show_endpoints:
                    # Body pose gives position + yaw; trace points are a
                    # fallback (position only) and need recorded traces.
                    if task.endpoint_bodies:
                        for gid, name in zip(
                            ghost_ids, task.endpoint_bodies, strict=True
                        ):
                            pos, quat = planner.engine.final_body_pose(name)
                            sync_endpoint_ghosts(
                                vdata, gid[:R], idxs, pos, quat
                            )
                    elif info.trace_sites is not None:
                        pos = info.trace_sites[:, :, -1, 0]
                        sync_endpoint_ghosts(
                            vdata, ghost_ids[0][:R], idxs, pos
                        )
                if show_belief and info.belief_pose is not None:
                    # One ghost per domain at its believed (post-settle) pose.
                    for gid, name in zip(
                        belief_ids, task.endpoint_bodies, strict=True
                    ):
                        pos, quat = info.belief_pose[name]
                        sync_endpoint_ghosts(
                            vdata, gid[:R], [0], pos[:, None], quat[:, None]
                        )
                    if c_palette is not None and info.contact_modes is not None:
                        for per_body in belief_body_ids:
                            set_ghost_contact_color(
                                vmodel, per_body[:R], info.contact_modes,
                                c_palette,
                            )
                plot_time = time.time() - plot_start

                for _ in range(steps_per_replan):
                    u = planner.get_action(params, float(vdata.time))
                    vdata.ctrl[:] = task.control_map_host(vdata, u)
                    mujoco.mj_step(vmodel, vdata)
                    if tracker is not None:
                        tracker.observe_tick(
                            StateSnapshot(
                                qpos=vdata.qpos[: mj_model.nq].copy(),
                                qvel=vdata.qvel[: mj_model.nv].copy(),
                            )
                        )
                    _update_goal()
                    viewer.sync()
                    _capture_frame()

                elapsed = time.time() - t0
                step_dt = steps_per_replan * sim_dt
                if elapsed < step_dt:
                    time.sleep(step_dt - elapsed)

                rtr = step_dt / (time.time() - t0)
                tm = info.timings or {"stage": 0.0, "rollout": 0.0}
                status = (
                    f"sim time: {vdata.time:6.2f}s | "
                    f"realtime rate: {rtr:4.2f}x | "
                    f"plan: {plan_time * 1e3:6.1f}ms "
                    f"(stage {tm['stage'] * 1e3:5.1f} | "
                    f"gpu {tm['rollout'] * 1e3:6.1f}) | "
                    f"plot: {plot_time * 1e3:5.1f}ms"
                )
                if info.contact_modes is not None:
                    # How the belief cloud splits across contact modes --
                    # the same information the ghost colors carry.
                    counts = np.bincount(info.contact_modes, minlength=4)
                    status += " | contact " + "/".join(
                        f"{n}:{int(c)}"
                        for n, c in zip(_MODE_NAMES, counts, strict=True)
                        if c
                    )
                # Clip to the live terminal width: a wider line soft-wraps
                # onto an extra row, which the fixed one-row cursor math
                # below doesn't account for and desyncs the in-place refresh
                # into scrolling new lines.
                cols = shutil.get_terminal_size().columns
                if status_callback is None:
                    print(status[:cols], end="\r")
                else:
                    # Put the callback text on a *second* in-place line.
                    # \x1b[K clears stale chars; \x1b[1A returns to the
                    # status line to overwrite.
                    extra = status_callback(vdata)
                    print(
                        f"\r{status[:cols]}\x1b[K\n{extra[:cols]}\x1b[K\x1b[1A",
                        end="",
                    )
        except KeyboardInterrupt:
            # Ctrl-C mid-rollout is a normal stop: fall through to the finally
            # below, which flushes whatever frames were captured.
            pass
        finally:
            if renderer is not None:
                if record_frames:
                    path = save_video(
                        record_frames,
                        record_out,
                        record_name,
                        record_fps,
                        fmt=record_format,
                    )
                    print(f"\nsaved {len(record_frames)} frames -> {path}")
                renderer.close()

    # Preserve the last readout, dropping below the second line if present.
    print("\n" if status_callback is not None else "")