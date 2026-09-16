"""Run-length and domain-randomization CLI flags, shared by the examples.

Both groups were previously retyped verbatim in every example file.
"""

from __future__ import annotations

import argparse

_RISK_CHOICES = ["worstcase", "bestcase", "average", "cvar", "var", "ewa"]


def add_run_args(
    parser: argparse.ArgumentParser,
    *,
    settle_default: int | None = 0,
    settle_note: str = "",
) -> None:
    """Add ``--duration``/``--settle-steps``.

    ``settle_default=None`` defers to the planner profile's own value;
    ``settle_note`` appends a task-specific reason to the help.
    """
    parser.add_argument(
        "--duration", type=float, default=None,
        help="Stop the rollout after this many sim seconds (for equal-length "
        "recordings). Default: run until the viewer window closes.",
    )
    parser.add_argument(
        "--settle-steps", type=int, default=settle_default,
        help="Zero-control steps run before each rollout, to resolve any "
        "interpenetration a perturbed start state introduces." + settle_note,
    )


def add_dr_args(
    parser: argparse.ArgumentParser,
    *,
    domains: int,
    domains_help: str = "Number of randomized worlds (R).",
) -> None:
    """Add ``--risk``/``--domains``."""
    parser.add_argument(
        "--risk", default=None, choices=_RISK_CHOICES,
        help="How per-domain costs are aggregated into the robust objective. "
        "Default: the task config's.",
    )
    parser.add_argument(
        "--domains", type=int, default=domains, help=domains_help
    )
