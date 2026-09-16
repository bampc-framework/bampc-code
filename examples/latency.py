"""Planning-latency CLI flags, shared across every interactive example.

Thin wrapper around ``run_interactive``'s ``plan_lag_steps``/
``compensate_latency``. Off by default everywhere -- opt in per run.
"""

from __future__ import annotations

import argparse


def add_latency_args(parser: argparse.ArgumentParser) -> None:
    """Add ``--plan-lag-steps``/``--compensate-latency``."""
    parser.add_argument(
        "--plan-lag-steps", type=int, default=0,
        help="Simulate this many steps of planning latency (0 = off): "
        "the sim keeps running under the previous plan while the next "
        "one is being computed, so optimize() starts from an "
        "already-stale state.",
    )
    parser.add_argument(
        "--compensate-latency", action="store_true",
        help="Predict the state forward by --plan-lag-steps before "
        "planning, undoing the staleness it introduces. No effect "
        "without it.",
    )
