"""The per-campaign true-friction schedule: one draw per segment of attempts.

Kept separate from ``harness``/``run`` so the same schedule is trivially
shared across every arm and algorithm for a given seed -- the comparison
between arms is only fair if all of them face the identical sequence of true
frictions, not independently-drawn ones.
"""

from __future__ import annotations

import numpy as np


def truth_schedule(
    attempts: int, flip_every: int, mu_lo: float, mu_hi: float, seed: int
) -> list[float]:
    """One randomly-drawn true ``mu`` per segment of ``flip_every`` attempts.

    ``ceil(attempts / flip_every)`` segments are drawn up front from
    ``Generator(PCG64(seed))``, uniform in ``[mu_lo, mu_hi]``; attempt ``k``
    reads segment ``k // flip_every``. Drawing every segment up front (rather
    than lazily as attempts run) keeps the schedule reproducible independent
    of how many attempts a campaign actually uses (a rejected/skipped attempt
    does not consume a draw).
    """
    num_segments = -(-attempts // flip_every)  # ceil division
    rng = np.random.default_rng(seed)
    draws = rng.uniform(mu_lo, mu_hi, size=num_segments)
    return [float(draws[k // flip_every]) for k in range(attempts)]
