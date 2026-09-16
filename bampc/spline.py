"""Host-side spline interpolation for the control parameterization.

Controls are spline *knots*, not raw action sequences. The planner interpolates
knots to a dense control trajectory before handing it to the rollout engine, and
queries the same spline at the control rate via :func:`query` (the cheap,
GPU-free path used on the real robot).

All functions are numpy and share the signature::

    vals = interp(method, tq, tk, knots)

with ``tq`` shape ``(H,)``, ``tk`` shape ``(num_knots,)``, ``knots`` shape
``(B, num_knots, nu)`` and ``vals`` shape ``(B, H, nu)``.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

import numpy as np

InterpMethod = Literal["zero", "linear", "cubic"]


def interp(
    method: InterpMethod,
    tq: np.ndarray,
    tk: np.ndarray,
    knots: np.ndarray,
) -> np.ndarray:
    """Interpolate a batch of knot sequences at query times ``tq``.

    Args:
        method: ``"zero"`` (zero-order hold), ``"linear"`` or ``"cubic"``.
        tq: Query times, shape ``(H,)``.
        tk: Knot times, shape ``(num_knots,)``.
        knots: Knot values, shape ``(B, num_knots, nu)``.

    Returns:
        Interpolated controls, shape ``(B, H, nu)``.
    """
    if method == "zero":
        idx = np.searchsorted(tk, tq, side="right") - 1
        idx = np.clip(idx, 0, len(tk) - 1)
        return knots[:, idx, :]
    if method == "linear":
        # Clamp to knot domain (no extrapolation), then blend each query's
        # bracketing knot pair -- vectorized over the whole (B, H, nu)
        # batch at once, no per-sample/per-channel Python loop.
        tqc = np.clip(tq, tk[0], tk[-1])
        n = tk.shape[0]
        if n == 1:
            return np.broadcast_to(
                knots[:, :1, :], (knots.shape[0], tq.shape[0], knots.shape[2])
            ).copy()
        idx = np.clip(np.searchsorted(tk, tqc, side="right") - 1, 0, n - 2)
        t0, t1 = tk[idx], tk[idx + 1]
        frac = (tqc - t0) / (t1 - t0)
        y0, y1 = knots[:, idx, :], knots[:, idx + 1, :]
        return y0 + frac[None, :, None] * (y1 - y0)
    if method == "cubic":
        tqc = np.clip(tq, tk[0], tk[-1])
        y = np.moveaxis(knots, 1, -1)  # (B, nu, num_knots)
        out = _natural_cubic_eval(tqc, tk, y)  # (B, nu, H)
        return np.moveaxis(out, -1, 1)
    raise ValueError(f"unknown interpolation method: {method!r}")


@lru_cache(maxsize=32)
def _cubic_forward_sweep(
    tk_key: tuple[bytes, str],
) -> tuple[np.ndarray, np.ndarray]:
    """Tridiagonal forward-sweep coefficients for a natural cubic spline.

    These depend only on the knot *times* ``tk`` (via ``h = diff(tk)``),
    never on knot values -- ``optimize()`` holds ``tk`` fixed across every
    sampling iteration of one plan step, so caching this here (keyed on the
    exact bytes/dtype of ``tk``) avoids re-deriving it each iteration.
    """
    data, dtype = tk_key
    tk = np.frombuffer(data, dtype=dtype)
    h = np.diff(tk)
    sub, diag, sup = h[:-1], 2.0 * (h[:-1] + h[1:]), h[1:]
    k = diag.shape[0]
    cp = np.empty(k)
    denom = np.empty(k)
    denom[0] = diag[0]
    cp[0] = sup[0] / denom[0]
    for i in range(1, k):
        denom[i] = diag[i] - sub[i] * cp[i - 1]
        if i < k - 1:
            cp[i] = sup[i] / denom[i]
    return cp, denom


def _thomas(
    sub: np.ndarray, cp: np.ndarray, denom: np.ndarray, d: np.ndarray
) -> np.ndarray:
    """Solve a tridiagonal system given cached forward-sweep coefficients.

    ``sub`` (the sub-diagonal) is 1-D, length ``k``; ``sub[0]`` is unused
    (no entry outside the matrix). ``cp``/``denom`` come from
    :func:`_cubic_forward_sweep`. ``d`` is ``(..., k)``; returns ``x`` with
    the same shape.
    """
    k = denom.shape[0]
    dp = np.empty_like(d)
    dp[..., 0] = d[..., 0] / denom[0]
    for i in range(1, k):
        dp[..., i] = (d[..., i] - sub[i] * dp[..., i - 1]) / denom[i]
    x = np.empty_like(d)
    x[..., -1] = dp[..., -1]
    for i in range(k - 2, -1, -1):
        x[..., i] = dp[..., i] - cp[i] * x[..., i + 1]
    return x


def _natural_cubic_second_derivs(tk: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Second derivatives at each knot for a natural cubic spline.

    ``y`` is ``(..., N)`` (last axis = knots). Boundary second
    derivatives are zero (natural spline); with fewer than 3 knots
    there are no interior points to solve for, so all are zero.
    """
    n = tk.shape[0]
    m = np.zeros_like(y)
    if n < 3:
        return m
    h = np.diff(tk)
    dy = np.diff(y, axis=-1) / h
    rhs = 6.0 * (dy[..., 1:] - dy[..., :-1])
    cp, denom = _cubic_forward_sweep((tk.tobytes(), tk.dtype.str))
    m[..., 1:-1] = _thomas(h[:-1], cp, denom, rhs)
    return m


def _natural_cubic_eval(
    tqc: np.ndarray, tk: np.ndarray, y: np.ndarray
) -> np.ndarray:
    """Evaluate a natural cubic spline through ``(tk, y)`` at ``tqc``.

    ``y`` is ``(..., N)``; ``tqc`` (query times, already clamped to
    ``[tk[0], tk[-1]]``) has shape ``(H,)``. Returns ``(..., H)``.
    """
    n = tk.shape[0]
    if n == 1:
        return np.broadcast_to(y[..., :1], (*y.shape[:-1], tqc.shape[0]))
    h = np.diff(tk)
    m = _natural_cubic_second_derivs(tk, y)
    idx = np.clip(np.searchsorted(tk, tqc, side="right") - 1, 0, n - 2)
    hi = h[idx]
    left = tk[idx + 1] - tqc
    right = tqc - tk[idx]
    y0, y1 = y[..., idx], y[..., idx + 1]
    m0, m1 = m[..., idx], m[..., idx + 1]
    return (
        m0 * left**3 / (6.0 * hi)
        + m1 * right**3 / (6.0 * hi)
        + (y0 / hi - m0 * hi / 6.0) * left
        + (y1 / hi - m1 * hi / 6.0) * right
    )


def query(
    method: InterpMethod,
    t: float,
    tk: np.ndarray,
    knots: np.ndarray,
) -> np.ndarray:
    """Evaluate a single knot sequence at one time ``t``.

    Cheap control-rate path for ``SamplingPlanner.get_action``.

    Args:
        method: Interpolation method.
        t: Query time (scalar).
        tk: Knot times, shape ``(num_knots,)``.
        knots: Knot values, shape ``(num_knots, nu)``.

    Returns:
        Control action, shape ``(nu,)``.
    """
    return interp(method, np.array([t]), tk, knots[None, ...])[0, 0]