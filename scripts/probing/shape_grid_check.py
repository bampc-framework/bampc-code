"""Visual smoke test for the shape library: spawn every shape twice.

Each ``models/shapes/*.xml`` fragment is grafted (via
``bampc.task.common.shapes``) into two standalone bodies per shape --
one 6-DOF (free joint), one 3-DOF (slide/slide/hinge) -- laid out on a grid
(rows = DOF type, columns = shape) and dropped onto a ground plane, to
eyeball resting height/contact and that a shape survives a uniform
``--scale`` without clipping or floating.

Run::

    uv run python scripts/probing/shape_grid_check.py
    uv run python scripts/probing/shape_grid_check.py --scale 1.5
    uv run python scripts/probing/shape_grid_check.py --shapes t circle
"""

from __future__ import annotations

import argparse
import colorsys
import time

import mujoco
import mujoco.viewer
import numpy as np

from bampc.task.common.shapes import (
    SHAPES_DIR,
    geom_mass,
    list_shapes,
    shape_com_xy,
    shape_min_z,
)

SPACING = 0.28  # grid cell pitch (m)
BITE = 0.0001  # resting penetration, matches models/push/push.xml
DOF_ROWS = ("free", "joint")  # row 0: 6-DOF free body; row 1: 3-DOF block


def _palette(n: int) -> np.ndarray:
    """Evenly-spaced HSV colors (RGBA), one per shape column."""
    pal = np.zeros((n, 4), np.float32)
    for i in range(n):
        r, g, b = colorsys.hsv_to_rgb(i / max(n, 1), 0.65, 0.95)
        pal[i] = (r, g, b, 1.0)
    return pal


def _add_shape_instance(
    spec: mujoco.MjSpec,
    shape_body: mujoco.MjsBody,
    *,
    name: str,
    pos: tuple[float, float, float],
    dof: str,
    scale: float,
    rgba: np.ndarray,
) -> None:
    """Graft a scaled copy of ``shape_body`` onto a new free/joint body."""
    body = spec.worldbody.add_body(name=name, pos=pos)
    if dof == "free":
        body.add_freejoint()
        rgba = np.array([rgba[0], rgba[1], rgba[2], 0.6 * rgba[3]])
    else:
        com_x, com_y = shape_com_xy(shape_body, scale)
        anchor = [com_x, com_y, 0.0]
        body.add_joint(
            name=f"{name}_x",
            type=mujoco.mjtJoint.mjJNT_SLIDE,
            axis=[1, 0, 0],
            pos=anchor,
        )
        body.add_joint(
            name=f"{name}_y",
            type=mujoco.mjtJoint.mjJNT_SLIDE,
            axis=[0, 1, 0],
            pos=anchor,
        )
        body.add_joint(
            name=f"{name}_yaw",
            type=mujoco.mjtJoint.mjJNT_HINGE,
            axis=[0, 0, 1],
            pos=anchor,
        )

    for g in shape_body.geoms:
        ng = body.add_geom(type=g.type, size=g.size * scale, pos=g.pos * scale)
        ng.quat = g.quat
        if not np.isnan(g.fromto[0]):
            ng.fromto = g.fromto * scale
        ng.rgba = rgba
        # Explicit mass so the compiled model's inertia matches the mass
        # assumption shape_com_xy anchored the joints on (not just its
        # own density-based default, in case a fragment geom overrides it).
        ng.mass = geom_mass(g, scale)

    for s in shape_body.sites:
        ns = body.add_site(
            name=f"{name}_{s.name}", pos=s.pos * scale, size=s.size * scale
        )
        ns.rgba = [0.0, 0.0, 0.0, 1.0]


def build_model(shapes: list[str], scale: float) -> mujoco.MjModel:
    """Compose a ground plane plus a free/joint instance of every shape."""
    spec = mujoco.MjSpec()
    spec.worldbody.add_light(pos=[0, 0, 2])

    tex = spec.add_texture(name="groundplane")
    tex.type = mujoco.mjtTexture.mjTEXTURE_2D
    tex.builtin = mujoco.mjtBuiltin.mjBUILTIN_CHECKER
    tex.rgb1 = [0.2, 0.3, 0.4]
    tex.rgb2 = [0.1, 0.2, 0.3]
    tex.width = 300
    tex.height = 300
    mat = spec.add_material(name="groundplane")
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "groundplane"

    n_cols = len(shapes)
    span_x = (n_cols - 1) * SPACING
    span_y = (len(DOF_ROWS) - 1) * SPACING
    spec.worldbody.add_geom(
        name="floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[span_x + 0.4, span_y + 0.4, 0.125],
        material="groundplane",
    )

    palette = _palette(n_cols)
    for c, shape_name in enumerate(shapes):
        frag = mujoco.MjSpec.from_file(str(SHAPES_DIR / f"{shape_name}.xml"))
        shape_body = frag.body("shape")
        z = -shape_min_z(shape_body, scale) - BITE
        for r, dof in enumerate(DOF_ROWS):
            _add_shape_instance(
                spec,
                shape_body,
                name=f"{shape_name}_{dof}",
                pos=(c * SPACING, r * SPACING, z),
                dof=dof,
                scale=scale,
                rgba=palette[c],
            )

    return spec.compile()


def main() -> None:
    """Build the grid and drop it in the interactive viewer."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shapes",
        nargs="+",
        default=None,
        help="Shape names (models/shapes/<name>.xml); default: all of them.",
    )
    parser.add_argument(
        "--scale", type=float, default=1.0, help="Uniform size scale factor."
    )
    args = parser.parse_args()

    shapes = args.shapes or list_shapes()
    print(f"shapes: {shapes}  scale={args.scale}")
    rows = ", ".join(f"{r}={dof}" for r, dof in enumerate(DOF_ROWS))
    print(f"rows (y): {rows}")

    model = build_model(shapes, args.scale)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()
            mujoco.mj_step(model, data)
            viewer.sync()
            dt = model.opt.timestep - (time.time() - step_start)
            if dt > 0:
                time.sleep(dt)


if __name__ == "__main__":
    main()
