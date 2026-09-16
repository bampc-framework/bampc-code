"""Shared CLI plumbing for driving an example from a frozen scenario bank.

The bank is the one source of truth for start pose *and* goal drift:
``--bank``/``--scenario`` pick the file and the index, and the
shape/scale/drift-* flags are optional overrides on top of it.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Sequence

import mujoco

from bampc.config import scenarios
from bampc.config.scenarios import Scenario, ScenarioBank
from bampc.task.base import GoalDrift
from bampc.task.common.shapes import list_shapes
from examples.status import RESET, YELLOW

_DRIFT_FLAGS = ("drift_shape", "drift_radius", "drift_yaw", "drift_freq")


def _banks_for(task: str) -> list[str]:
    """Bank names whose task matches, so --bank cannot offer a wrong one."""
    return [n for n in scenarios.list_banks() if scenarios.load(n).task == task]


def add_scenario_args(
    parser: argparse.ArgumentParser,
    default_bank: str,
    *,
    geometry: bool = True,
    drift: bool = True,
    shapes: Sequence[str] | None = None,
) -> None:
    """Add ``--bank``/``--scenario`` plus optional per-field overrides.

    ``geometry=False`` drops ``--shape``/``--scale`` (peg has no shape
    library); ``shapes`` narrows ``--shape`` to what the task accepts.
    ``drift=False`` hides the ``--drift-*`` flags for a task whose goal is
    static -- they stay parseable, but warn and do nothing. Override flags
    default to ``None`` so "not given" differs from "the bank's own value".
    """
    parser.add_argument(
        "--bank", default=default_bank,
        choices=_banks_for(scenarios.load(default_bank).task),
        help=f"Scenario bank to drive (default: {default_bank}).",
    )
    parser.add_argument(
        "--scenario", type=int, default=0,
        help="Start-state index in the bank (default: 0).",
    )
    if geometry:
        parser.add_argument(
            "--shape", default=None,
            choices=list(shapes) if shapes is not None else list_shapes(),
            help="Block shape (default: the bank's).",
        )
        parser.add_argument(
            "--scale", type=float, default=None,
            help="Block scale (default: the bank's).",
        )
    # A static-goal task (peg socket, curling house, flip orientation) still
    # accepts these, so an old command line keeps working -- hidden from
    # --help, and resolve_scenario warns instead of animating the goal.
    def drift_help(text: str) -> str:
        return text if drift else argparse.SUPPRESS

    parser.add_argument(
        "--drift-shape", default="lissajous", choices=["lissajous", "circle"],
        help=drift_help(
            "'lissajous' (default): two independent xy sinusoids, amplitude "
            "= radius per axis. 'circle': a true circle of the given radius, "
            "driven by one frequency (only --drift-freq's first value)."
        ),
    )
    parser.add_argument(
        "--drift-radius", type=float, default=None,
        help=drift_help("Goal-drift xy radius override (default: bank's)."),
    )
    parser.add_argument(
        "--drift-yaw", type=float, default=None,
        help=drift_help("Goal-drift yaw amplitude override (default: bank's)."),
    )
    parser.add_argument(
        "--drift-freq", type=float, nargs=2, default=None,
        metavar=("FX", "FY"),
        help=drift_help(
            "Goal-drift xy frequencies [Hz] override (default: the bank's). "
            "--drift-shape circle uses only FX."
        ),
    )


def resolve_scenario(
    args: argparse.Namespace,
    task: str,
    *,
    geometry: bool = True,
    drift: bool = True,
) -> tuple[ScenarioBank, Scenario, str | None, float | None, GoalDrift | None]:
    """Resolve ``args`` (from :func:`add_scenario_args`) against its bank.

    Returns ``(bank, scenario, shape, scale, goal_drift)``. ``shape``/
    ``scale`` are ``None`` when ``geometry=False``; ``goal_drift`` is
    ``None`` when ``drift=False`` or every amplitude resolves to zero.
    """
    bank = scenarios.load(args.bank)
    if bank.task != task:
        raise ValueError(
            f"--bank {args.bank!r} is a {bank.task!r} bank, not {task!r}"
        )
    scenario = bank[args.scenario]
    if not drift:
        given = [
            f"--{f.replace('_', '-')}"
            for f in _DRIFT_FLAGS
            if getattr(args, f, None) not in (None, "lissajous")
        ]
        if given:
            print(
                f"{YELLOW}ignoring {' '.join(given)}: {task}'s goal is "
                f"static, so it does not drift.{RESET}",
                file=sys.stderr,
            )

    shape = scale = None
    if geometry:
        shape = args.shape if args.shape is not None else bank.shape
        scale = args.scale if args.scale is not None else bank.scale

    goal_drift = None
    if drift:
        d = bank.goal_drift
        radius = (
            args.drift_radius if args.drift_radius is not None
            else d.radius_xy[0]
        )
        yaw = args.drift_yaw if args.drift_yaw is not None else d.yaw_amp
        freq = (
            tuple(args.drift_freq) if args.drift_freq is not None
            else d.freq_xy
        )
        # circle: a sine and a pi/2-phase-shifted sine on a shared frequency
        # is a sine/cosine pair -- a true circle of radius (radius, radius),
        # not a new waveform. freq[1] is ignored (a circle has one rate).
        phase = (
            (0.0, math.pi / 2, 0.0) if args.drift_shape == "circle"
            else (0.0, 0.0, 0.0)
        )
        if args.drift_shape == "circle":
            freq = (freq[0], freq[0])
        goal_drift = (
            GoalDrift(
                radius_xy=(radius, radius), freq_xy=freq, yaw_amp=yaw,
                yaw_freq=d.yaw_freq, phase=phase,
            )
            if radius > 0.0 or yaw > 0.0
            else None
        )

    return bank, scenario, shape, scale, goal_drift


def initial_state(
    task, bank: ScenarioBank, scenario: Scenario
) -> mujoco.MjData:
    """Pose ``task`` at ``scenario``, clock started at its drift phase.

    Without setting ``mj_data.time`` every episode would start the drift at
    phase 0, the one thing the scenario's own ``drift_phase_time`` exists to
    vary.
    """
    mj_data = scenarios.pose(bank, task, scenario)
    mj_data.time = scenario.drift_phase_time
    return mj_data
