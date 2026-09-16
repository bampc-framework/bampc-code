"""Shape-library composition: graft a shape fragment onto a task body.

Fragments live under ``models/shapes/*.xml`` as a bare ``<body
name="shape">`` (geometry + optional attractor sites, no classes). Grafting
deletes a task body's existing geoms/sites and adds scaled copies from the
fragment instead, so ``Push``/``PushFr3`` can swap shapes at build time.
"""

from __future__ import annotations

import mujoco
import numpy as np

from bampc import MODELS_DIR
from bampc.task.base import ContactBudget

SHAPES_DIR = MODELS_DIR / "shapes"


def list_shapes() -> list[str]:
    """Every fragment name available under ``models/shapes/``."""
    return sorted(p.stem for p in SHAPES_DIR.glob("*.xml"))


def _geom_min_z(
    gtype: mujoco.mjtGeom, size: np.ndarray, pos: np.ndarray, fromto: np.ndarray
) -> float:
    """Lowest local-z point of a box/sphere/cylinder/capsule geom."""
    if not np.isnan(fromto[0]):
        return float(min(fromto[2], fromto[5]) - size[0])
    if gtype == mujoco.mjtGeom.mjGEOM_BOX:
        return float(pos[2] - size[2])
    if gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
        return float(pos[2] - size[0])
    if gtype in (mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_CAPSULE):
        return float(pos[2] - size[1])
    raise ValueError(f"unsupported geom type for bbox: {gtype}")


def shape_min_z(shape_body: mujoco.MjsBody, scale: float) -> float:
    """Lowest local-z point across a shape fragment's geoms.

    Mesh geoms are skipped -- they're visual-only (see ``graft_shape``); the
    envelope comes from the fragment's primitive collision geom(s).
    """
    return min(
        _geom_min_z(g.type, g.size * scale, g.pos * scale, g.fromto * scale)
        for g in shape_body.geoms
        if g.type != mujoco.mjtGeom.mjGEOM_MESH
    )


def _geom_xy_radius(
    gtype: mujoco.mjtGeom, size: np.ndarray, pos: np.ndarray, fromto: np.ndarray
) -> float:
    """Farthest horizontal reach of a box/sphere/cylinder/capsule geom.

    Conservative (circumscribed, ignores yaw-dependent footprint changes) --
    good enough for a keep-out margin, not for precise collision geometry.
    """
    if not np.isnan(fromto[0]):
        seg_xy = 0.5 * float(np.linalg.norm(fromto[3:5] - fromto[0:2]))
        center_xy = 0.5 * (fromto[0:2] + fromto[3:5])
        return float(np.linalg.norm(center_xy)) + seg_xy + float(size[0])
    if gtype == mujoco.mjtGeom.mjGEOM_BOX:
        return float(np.linalg.norm(pos[:2])) + float(np.hypot(*size[:2]))
    if gtype in (
        mujoco.mjtGeom.mjGEOM_SPHERE,
        mujoco.mjtGeom.mjGEOM_CYLINDER,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
    ):
        return float(np.linalg.norm(pos[:2])) + float(size[0])
    raise ValueError(f"unsupported geom type for bbox: {gtype}")


def shape_max_xy_radius(shape_body: mujoco.MjsBody, scale: float) -> float:
    """Farthest horizontal reach across a shape fragment's geoms.

    Mesh geoms are skipped, same as :func:`shape_min_z`. Used to keep a
    sampled goal's shape footprint (not just its center) off a boundary.
    """
    return max(
        _geom_xy_radius(g.type, g.size * scale, g.pos * scale, g.fromto * scale)
        for g in shape_body.geoms
        if g.type != mujoco.mjtGeom.mjGEOM_MESH
    )


def _geom_volume(
    gtype: mujoco.mjtGeom, size: np.ndarray, fromto: np.ndarray
) -> float:
    """Volume of a box/sphere/cylinder/capsule geom."""
    if not np.isnan(fromto[0]):
        length = float(np.linalg.norm(fromto[3:6] - fromto[:3]))
        r = size[0]
        return np.pi * r**2 * length + (4.0 / 3.0) * np.pi * r**3  # capsule
    if gtype == mujoco.mjtGeom.mjGEOM_BOX:
        return float(8.0 * size[0] * size[1] * size[2])
    if gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
        return float((4.0 / 3.0) * np.pi * size[0] ** 3)
    if gtype == mujoco.mjtGeom.mjGEOM_CYLINDER:
        return float(np.pi * size[0] ** 2 * (2.0 * size[1]))
    raise ValueError(f"unsupported geom type for volume: {gtype}")


def geom_mass(g: mujoco.MjsGeom, scale: float) -> float:
    """Mass of ``g`` at ``scale``, honoring an explicit ``mass``/``density``.

    An explicit ``mass`` is authored at nominal (``scale=1``) size and
    scaled by volume ratio so it stays physically consistent under scaling.
    """
    vol = _geom_volume(g.type, g.size * scale, g.fromto * scale)
    if not np.isnan(g.mass):
        vol_nominal = _geom_volume(g.type, g.size, g.fromto)
        return float(g.mass) * (vol / vol_nominal)
    return float(g.density * vol)


def shape_com_xy(
    shape_body: mujoco.MjsBody, scale: float
) -> tuple[float, float]:
    """Mass-weighted centroid (x, y) of a shape fragment's geoms.

    The 3-DOF ``block_x``/``block_y``/``block_yaw`` joints must anchor
    here, not at the body origin, or yawing couples into translation. Mesh
    geoms are skipped (visual-only, contribute no mass; see ``graft_shape``).
    """
    com = np.zeros(2)
    total = 0.0
    for g in shape_body.geoms:
        if g.type == mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mass = geom_mass(g, scale)
        pos = g.pos * scale
        if not np.isnan(g.fromto[0]):
            fromto = g.fromto * scale
            pos = 0.5 * (fromto[:3] + fromto[3:])
        com += mass * pos[:2]
        total += mass
    return float(com[0] / total), float(com[1] / total)


def _geom_extent(
    gtype: mujoco.mjtGeom,
    size: np.ndarray,
    pos: np.ndarray,
    fromto: np.ndarray,
    axis: int,
) -> tuple[float, float]:
    """``(min, max)`` local extent of a geom along ``axis`` (0=x, 1=y, 2=z).

    Box/cylinder/capsule half-extents are per-axis for a box, but a single
    radius (``size[0]``) off-axis for cylinder/capsule (their long axis is
    local z by MuJoCo convention when given as ``size`` rather than
    ``fromto``) -- so only the z case reads ``size[1]`` (half-length).
    """
    if not np.isnan(fromto[0]):
        a, b, r = fromto[axis], fromto[axis + 3], size[0]
        return float(min(a, b) - r), float(max(a, b) + r)
    if gtype == mujoco.mjtGeom.mjGEOM_BOX:
        return float(pos[axis] - size[axis]), float(pos[axis] + size[axis])
    if gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
        return float(pos[axis] - size[0]), float(pos[axis] + size[0])
    if gtype in (mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_CAPSULE):
        r = size[1] if axis == 2 else size[0]
        return float(pos[axis] - r), float(pos[axis] + r)
    raise ValueError(f"unsupported geom type for bbox: {gtype}")


def shape_center(
    shape_body: mujoco.MjsBody, scale: float
) -> tuple[float, float, float]:
    """Geometric center (bounding-box midpoint) of a shape fragment's geoms.

    Unweighted, unlike the mass-weighted :func:`shape_com_xy`: this re-centers
    attractor sites on the shape's middle rather than the fragment's authored
    positions. Mesh geoms are skipped, same as :func:`shape_min_z`.
    """
    center = []
    for axis in range(3):
        los, his = zip(*(
            _geom_extent(g.type, g.size * scale, g.pos * scale,
                         g.fromto * scale, axis)
            for g in shape_body.geoms
            if g.type != mujoco.mjtGeom.mjGEOM_MESH
        ))
        center.append(0.5 * (min(los) + max(his)))
    return tuple(center)


def _graft_mesh_geom(
    spec: mujoco.MjSpec,
    frag: mujoco.MjSpec,
    body: mujoco.MjsBody,
    body_name: str,
    shape: str,
    g: mujoco.MjsGeom,
    scale: float,
    default: mujoco.MjsDefault | None,
) -> None:
    """Graft a visual-only mesh geom, copying its mesh/texture/material.

    Assets are shape-scoped (``f"{shape}_..."``), not body-scoped, so
    grafting the same shape onto ``block`` and ``goal`` shares one copy
    instead of erroring on a duplicate name -- "add once" is guarded by
    scanning ``spec.meshes``/``texture``/``materials`` for the name
    (``spec.mesh(name)``-style by-name lookup returns ``None`` for a
    just-``add_material``-ed material in this MuJoCo build even though it
    exists, confirmed empirically; scanning the list is what actually
    works for all three asset kinds, so it's used uniformly here).
    Non-colliding (``contype``/``conaffinity`` forced to 0: the target
    body's ``default`` collision class would otherwise clobber that) and
    massless (the fragment's primitive collision geom supplies the body's
    real mass; see ``shape_min_z``/``shape_com_xy``, which skip mesh geoms).
    """
    fmesh = frag.mesh(g.meshname)
    mesh_name = f"{shape}_{g.meshname}"
    if not any(x.name == mesh_name for x in spec.meshes):
        spec.add_mesh(
            name=mesh_name,
            file=str((SHAPES_DIR / fmesh.file).resolve()),
            scale=[float(fmesh.scale[i]) * scale for i in range(3)],
        )

    material_name = None
    if g.material:
        fmat = frag.material(g.material)
        material_name = f"{shape}_{g.material}"
        if not any(x.name == material_name for x in spec.materials):
            tex_names = []
            for tex_name in fmat.textures:
                if not tex_name:
                    tex_names.append("")
                    continue
                new_tex_name = f"{shape}_{tex_name}"
                if not any(x.name == new_tex_name for x in spec.textures):
                    ftex = frag.texture(tex_name)
                    spec.add_texture(
                        name=new_tex_name,
                        type=ftex.type,
                        file=str((SHAPES_DIR / ftex.file).resolve()),
                    )
                tex_names.append(new_tex_name)
            spec.add_material(name=material_name, textures=tex_names)

    ng = body.add_geom(
        default,
        name=f"{body_name}_{g.name}",
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname=mesh_name,
        pos=g.pos * scale,
        mass=0.0,
    )
    ng.quat = g.quat
    ng.contype = 0
    ng.conaffinity = 0
    if material_name:
        ng.material = material_name


def graft_shape(
    spec: mujoco.MjSpec,
    body_name: str,
    shape: str,
    scale: float,
    *,
    default: mujoco.MjsDefault | None = None,
    include_sites: bool = True,
) -> mujoco.MjsBody:
    """Replace ``body_name``'s geoms/sites with a scaled copy of ``shape``.

    A mesh-type geom is grafted visual-only (see ``_graft_mesh_geom``); every
    other geom type is a scaled physical copy as before.

    Returns the shape fragment's body (for ``shape_min_z``/``shape_com_xy``).
    """
    frag = mujoco.MjSpec.from_file(str(SHAPES_DIR / f"{shape}.xml"))
    shape_body = frag.body("shape")

    body = spec.body(body_name)
    for g in list(body.geoms):
        spec.delete(g)
    for s in list(body.sites):
        spec.delete(s)

    for g in shape_body.geoms:
        if g.type == mujoco.mjtGeom.mjGEOM_MESH:
            _graft_mesh_geom(
                spec, frag, body, body_name, shape, g, scale, default
            )
            continue
        ng = body.add_geom(
            default,
            name=f"{body_name}_{g.name}",
            type=g.type,
            size=g.size * scale,
            pos=g.pos * scale,
        )
        ng.quat = g.quat
        if not np.isnan(g.fromto[0]):
            ng.fromto = g.fromto * scale
        ng.mass = geom_mass(g, scale)

    if include_sites:
        for s in shape_body.sites:
            body.add_site(name=s.name, pos=s.pos * scale, size=s.size * scale)

    return shape_body


# Per-shape contact/constraint budgets, from scripts/probing/probe_task.py
# (safety factor 3x); re-derive after changing a shape or its scale range.
def push_contact_budget(shape: str) -> ContactBudget:
    """Contact budget for ``Push``, switched on ``shape``."""
    match shape:
        case "t" | "l":
            return ContactBudget(ncon_per_env=24, nac_per_env=24, nj_per_env=96)
        case "i" | "a" | "c":
            return ContactBudget(
                ncon_per_env=36, nac_per_env=36, nj_per_env=144
            )
        case "r":
            return ContactBudget(
                ncon_per_env=63, nac_per_env=63, nj_per_env=252
            )
        case "square":
            return ContactBudget(ncon_per_env=12, nac_per_env=12, nj_per_env=48)
        case "circle":
            return ContactBudget(ncon_per_env=9, nac_per_env=9, nj_per_env=36)
        case "sphere":
            return ContactBudget(ncon_per_env=9, nac_per_env=9, nj_per_env=36)
        case _:
            raise ValueError(f"no contact budget for shape {shape!r}")


def push_fr3_contact_budget(shape: str) -> ContactBudget:
    """Contact budget for ``PushFr3``, switched on ``shape``.

    One budget covers both ``manipulation_type``s (the caller doesn't split
    on it), so each case takes the max across "joint"/"free". Shapes are
    grouped only where their measurements match: the budget scales with
    ``nworld``, so over-provisioning a cheap shape is real memory.
    """
    # shape -> (ncon_per_env == nac_per_env, nj_per_env).
    budgets = {
        "t": (42, 147), "l": (42, 147),
        "i": (57, 192), "a": (75, 246), "c": (54, 183),
        "r": (108, 345),
        "square": (27, 102),
        "circle": (33, 120), "sphere": (12, 57),
    }
    if shape not in budgets:
        raise ValueError(f"no contact budget for shape {shape!r}")
    ncon, nj = budgets[shape]
    return ContactBudget(ncon_per_env=ncon, nac_per_env=ncon, nj_per_env=nj)


def peg_fr3_contact_budget() -> ContactBudget:
    """Contact budget for ``PegFr3``.

    Hand-sized from the geometric ceiling (a box peg in a square socket can
    touch 4 walls plus the base, ~4 manifold points each), kept above the
    probed value as the safer of the two. Over-sizing only costs memory;
    under-sizing silently zeroes the wall-force penalty rather than failing.
    """
    return ContactBudget(ncon_per_env=60, nac_per_env=60, nj_per_env=240)


def balance_fr3_contact_budget(shape: str) -> ContactBudget:
    """Contact budget for ``BalanceFr3``, switched on ``shape``.

    **Not** interchangeable with ``balance_contact_budget``: the planar toy
    model has no FR3 joint-limit constraints, so reusing it under-sizes
    ``nj_per_env`` and silently drops constraint rows on every run.
    """
    match shape:
        case "t" | "l":
            # Grouped on the larger of the two probed values.
            return ContactBudget(
                ncon_per_env=27, nac_per_env=27, nj_per_env=183
            )
        case "i" | "a" | "c":
            return ContactBudget(
                ncon_per_env=36, nac_per_env=36, nj_per_env=237
            )
        case "r":
            return ContactBudget(
                ncon_per_env=60, nac_per_env=60, nj_per_env=381
            )
        case "square":
            return ContactBudget(
                ncon_per_env=12, nac_per_env=12, nj_per_env=93
            )
        case "circle":
            return ContactBudget(
                ncon_per_env=15, nac_per_env=15, nj_per_env=111
            )
        case "sphere":
            return ContactBudget(ncon_per_env=3, nac_per_env=3, nj_per_env=39)
        case _:
            raise ValueError(f"no contact budget for shape {shape!r}")


def balance_contact_budget(shape: str) -> ContactBudget:
    """Contact budget for ``Balance``, switched on ``shape``.

    Sized for the plate's ``condim=6`` block contacts (spin/roll active),
    so the counts run higher than the ``condim=3`` PushFr3 budgets above.
    """
    match shape:
        case "t" | "l":
            return ContactBudget(
                ncon_per_env=24, nac_per_env=24, nj_per_env=144
            )
        case "i" | "a" | "c":
            return ContactBudget(
                ncon_per_env=36, nac_per_env=36, nj_per_env=216
            )
        case "r":
            return ContactBudget(
                ncon_per_env=60, nac_per_env=60, nj_per_env=360
            )
        case "square":
            return ContactBudget(ncon_per_env=12, nac_per_env=12, nj_per_env=72)
        case "circle":
            return ContactBudget(ncon_per_env=15, nac_per_env=15, nj_per_env=90)
        case "sphere":
            return ContactBudget(ncon_per_env=3, nac_per_env=3, nj_per_env=18)
        case _:
            raise ValueError(f"no contact budget for shape {shape!r}")


def curling_contact_budget(shape: str) -> ContactBudget:
    """Contact budget for ``CurlingFr3``, switched on ``shape``.

    Do **not** borrow push_fr3's numbers: curling's house and launch region
    are visual-only (``contype=0``), but the scene is not uniformly smaller
    and the borrowed value under-sizes at least one shape. Re-probe after any
    lane geometry change -- **including a pure translation**, which is not
    budget-neutral (the EE tracks the puck's resting height, so a shift
    changes the arm posture and with it the contact count).
    """
    # shape -> (ncon_per_env == nac_per_env, nj_per_env).
    budgets = {"circle": (18, 93), "sphere": (15, 81), "square": (27, 129)}
    if shape not in budgets:
        raise ValueError(f"no contact budget for shape {shape!r}")
    ncon, nj = budgets[shape]
    return ContactBudget(ncon_per_env=ncon, nac_per_env=ncon, nj_per_env=nj)


def flip_fr3_contact_budget(shape: str) -> ContactBudget:
    """Contact budget for ``FlipFr3``, switched on ``shape``.

    One budget covers ``wall=True``/``False``: the wall does not raise the
    *simultaneous* contact ceiling above what EE + table already reach.
    """
    match shape:
        case "cracker_box":
            return ContactBudget(
                ncon_per_env=27, nac_per_env=27, nj_per_env=102
            )
        case _:
            raise ValueError(f"no contact budget for shape {shape!r}")
