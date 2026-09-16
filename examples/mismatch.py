"""Single/multi-parameter planner model mismatch for the examples.

The planner rolls out a deliberately-wrong model while the viewer sims the
truth: a :class:`~bampc.dr.DomainRandomizer` writes per-world
*device* arrays and never touches the host ``mj_model``, so an R=1 constant
spec makes only the planner wrong.

Keys: ``mass``, ``friction`` (sliding), ``rolling_friction``,
``solimp_dmin``, ``impratio`` -- all need a ``block`` body, so peg (which
has its own mount knobs) does not apply.
"""

from __future__ import annotations

import argparse

import mujoco
import numpy as np

from bampc.task.base import Task


def add_mismatch_arg(parser: argparse.ArgumentParser) -> None:
    """Add ``--mismatch KEY=MULT ...`` (repeatable, single or multi param)."""
    parser.add_argument(
        "--mismatch",
        nargs="*",
        default=[],
        metavar="KEY=MULT",
        help="Planner-model parameter multipliers vs the truth, e.g. "
        "`--mismatch mass=2.0 friction=0.5`. Keys: mass, friction, "
        "rolling_friction, solimp_dmin, impratio. The viewer stays nominal, so "
        "this is a genuine planner-vs-reality mismatch (one fixed model, R=1).",
    )


def _block_geom0(task: Task) -> int:
    """First geom id of the ``block`` body (off a pristine task)."""
    m = task.mj_model
    bid = m.body("block").id
    return next(g for g in range(m.ngeom) if m.geom_bodyid[g] == bid)


# key -> (base reader, spec fragment builder from an ABSOLUTE value).
_SPEC: dict = {
    "mass": (
        lambda t: float(t.mj_model.body("block").mass[0]),
        lambda v: {"body": {"block": {"mass": v}}},
    ),
    "friction": (
        lambda t: float(t.mj_model.geom_friction[_block_geom0(t), 0]),
        lambda v: {"body": {"block": {"friction": v}}},
    ),
    "rolling_friction": (
        lambda t: float(t.mj_model.geom_friction[_block_geom0(t), 2]),
        lambda v: {"body": {"block": {"rolling_friction": v}}},
    ),
    "solimp_dmin": (
        lambda t: float(t.mj_model.geom_solimp[_block_geom0(t), 0]),
        lambda v: {
            "body": {"block": {"solimp_dmin": float(
                np.clip(v, mujoco.mjMINVAL, 1.0 - 1e-6)
            )}}
        },
    ),
    "impratio": (
        lambda _t: 1.0,
        lambda v: {"opt": {"__all__": {"impratio": v}}},
    ),
}


def _merge(specs: list[dict]) -> dict:
    """Deep-merge DR-spec fragments (body/opt) into one nested dict."""
    out: dict = {}
    for spec in specs:
        for kind, entities in spec.items():
            dst = out.setdefault(kind, {})
            for entity, params in entities.items():
                dst.setdefault(entity, {}).update(params)
    return out


def mismatch_spec(task: Task, pairs: list[str]) -> dict:
    """Build a constant (R=1) DR spec of ``base * mult`` per ``KEY=MULT``.

    ``task`` must be pristine -- base values are read off it. Returns ``{}`` for
    an empty list, so the caller can build the engine plain.
    """
    frags = []
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"--mismatch expects KEY=MULT, got {pair!r}")
        key, mult = pair.split("=", 1)
        if key not in _SPEC:
            raise ValueError(
                f"unknown mismatch key {key!r}; choose from {list(_SPEC)}"
            )
        base, build = _SPEC[key]
        frags.append(build(base(task) * float(mult)))
    return _merge(frags)
