"""Shared truth/prediction timestep alignment for the `dr` experiment tree.

The truth and the prediction engines run at their own timesteps, so the
replan interval is a *duration*, not a step count: each side takes however
many of its own steps fill ``1/plan_freq_hz``.
"""

from __future__ import annotations


def truth_stride(dt_pred: float, dt_truth: float) -> int:
    """Truth steps per prediction step, as an exact integer.

    The two sides only line up if the truth's clock divides the
    prediction's. Then observation ``i`` sits at ``(i+1)*dt_truth`` and
    prediction ``j`` at ``j*dt_pred``, so ``i = j*k - 1`` and
    ``obs[k-1::k]`` lands exactly on the prediction's ticks -- no
    interpolation, which matters because interpolation error would land
    straight in the quantity being measured.
    """
    k = dt_pred / dt_truth
    if k < 1.0 or abs(k - round(k)) > 1e-9:
        raise ValueError(
            f"prediction dt {dt_pred:g} must be a positive integer multiple "
            f"of truth dt {dt_truth:g} (got ratio {k:g}); the truth has to "
            "be at least as fine as the model it is judging"
        )
    return int(round(k))
