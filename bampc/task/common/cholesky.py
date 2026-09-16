"""In-kernel 6x6 SPD linear solve (Cholesky), for the task-space IK.

Warp has no inverse/solve above 4x4, so the damped least-squares IK's 6x6
system ``(J Jᵀ + λI) x = twist`` is solved with a hand-written Cholesky. The
system is symmetric positive-definite (λ>0), so no pivoting is needed -- branch
free and stable near kinematic singularities.
"""

from __future__ import annotations

import warp as wp

vec6 = wp.types.vector(length=6, dtype=wp.float32)
mat66 = wp.types.matrix(shape=(6, 6), dtype=wp.float32)


@wp.func
def chol_solve6(a: mat66, b: vec6) -> vec6:
    """Solve ``a x = b`` for a 6x6 SPD ``a`` via Cholesky (no pivoting).

    Factors ``a = L Lᵀ`` (lower triangular ``L``), then forward/back
    substitution. Loops have static or simple runtime bounds so Warp lowers
    them without dynamic allocation.
    """
    el = mat66()  # zero-initialized lower factor L

    # Cholesky factorization: a = L Lᵀ.
    for j in range(6):
        s = a[j, j]
        for k in range(j):
            s -= el[j, k] * el[j, k]
        d = wp.sqrt(s)
        el[j, j] = d
        for i in range(j + 1, 6):
            t = a[i, j]
            for k in range(j):
                t -= el[i, k] * el[j, k]
            el[i, j] = t / d

    # Forward solve L y = b.
    y = vec6()
    for i in range(6):
        s = b[i]
        for k in range(i):
            s -= el[i, k] * y[k]
        y[i] = s / el[i, i]

    # Back solve Lᵀ x = y (iterate i = 5..0; avoid negative-step range).
    x = vec6()
    for ii in range(6):
        i = 5 - ii
        s = y[i]
        for k in range(i + 1, 6):
            s -= el[k, i] * x[k]
        x[i] = s / el[i, i]

    return x