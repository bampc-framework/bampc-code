"""Shared, colorized cost-status line for the interactive examples.

The one place that turns a task's ``cost_components()`` dict into a status
line for ``run_interactive``.
"""

from __future__ import annotations

import sys

import numpy as np

from bampc.task.base import Task
from bampc.uncertainty import PoseKalman

_USE_COLOR = sys.stdout.isatty()

RED = "\033[31m" if _USE_COLOR else ""
GREEN = "\033[32m" if _USE_COLOR else ""
YELLOW = "\033[33m" if _USE_COLOR else ""
RESET = "\033[0m" if _USE_COLOR else ""


class CostStatus:
    """A ``status_callback`` for ``run_interactive``: live per-term cost.

    ``total`` is colored against an EMA of this episode's own totals, not a
    fixed threshold -- costs sit on very different scales across tasks, so
    green/red means "at-or-below / above what this run has typically been".
    ``alpha`` sets how fast that baseline tracks.

    An adaptive ``filt`` adds a live ``unc <std>`` segment (its claimed
    position uncertainty); a non-adaptive one or ``None`` shows nothing.
    """

    def __init__(
        self, task: Task, alpha: float = 0.05, filt: PoseKalman | None = None
    ) -> None:
        """Set up an empty baseline for ``task``'s cost components."""
        self._task = task
        self._alpha = alpha
        self._filt = filt
        self._baseline: float | None = None

    def __call__(self, data) -> str:
        """Format one status line, updating the running baseline."""
        c = self._task.cost_components(data)
        total = c["total"]
        self._baseline = (
            total if self._baseline is None
            else self._baseline + self._alpha * (total - self._baseline)
        )
        color = GREEN if total <= self._baseline else RED
        terms = " | ".join(
            f"{k} {v:7.3f}" for k, v in c.items() if k != "total"
        )
        # Instantaneous, not the sweeps' episode-level aggregation. None
        # (no success condition, e.g. Push) shows no segment at all.
        outcome = self._task.task_success(data)
        suffix = ""
        if outcome is not None:
            suffix = (
                f" {GREEN}success{RESET}" if outcome
                else f" {RED}failure{RESET}"
            )
        if self._filt is not None and self._filt.is_adaptive:
            unc = float(np.linalg.norm(self._filt.posterior_std()["pos"]))
            suffix += f" unc {unc:.4f}"
        # total leads the line: the viewer clips this string to the
        # terminal width, so a trailing segment is the first thing lost.
        return f"{color}total {total:7.3f}{RESET}{suffix} | cost: {terms}"
