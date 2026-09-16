"""The friction hedge: its verified band, its grid, and how a shot is judged.

Shared by the sim identification loop (``friction_id.py``, which invents
reality with ``--truth-mu``) and the real-robot one
(``scripts/ros/run_planner_node_curling.py``, where reality is the actual
lane). Both narrow the same hedge against the same evidence, so the pieces
that decide *what a shot is worth* live here rather than in either caller --
the comments below record measurements that each cost a run to obtain, and
two copies of them would drift.

What is NOT here: how the evidence is produced. The sim script steps a truth
model and reads the puck every physics step; the ROS node reads a tracker at
the plan rate. They hand the same ``(R,)`` error vector to the same policy.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from bampc.belief import DomainBelief

# The prior band, and a hard clamp on every grid drawn from it.
#
# Wider than anything else in the repo uses: the curling DR example
# disagrees with itself (comment claims a verified [0.05, 0.14], code draws
# uniform(0.06, 0.18)), and neither end has been re-measured here. High mu is
# the solver-safe direction -- a shorter slide gives the flat contact less
# time to come apart -- so 0.2 is cheap. The 0.03 end is the one carrying
# risk, on two counts, neither of which raises:
#
#   * configs/numerics/curling.yaml's header has mu=0.04 breaking at v0=0.5,
#     but on the OLD box lane; the plane lane is claimed re-verified clean
#     down to mu=0.02 (examples/domain_randomization/curling.py:124-133).
#     This band trusts that claim without independently confirming it.
#   * At mu=0.03 a v0=0.78 shot slides ~1.0 m, which carries the puck past
#     the drawn lane slab (x in [0.35, 1.45]) and well past the house at
#     1.00. Contact is against an infinite plane so the physics holds, but
#     such a shot is unscoreable by construction.
#
# Lowering MU_LO also costs wall time, and silently breaks identification if
# ignored: slide time goes as v0/(mu*g), ~2.7 s at mu=0.03 against ~1.0 s at
# 0.08, so a caller's per-shot clock must grow with it. Measured: at 3.0 s
# the band below ~0.06 scores against a puck that has not finished sliding,
# and the posterior pins to MU_LO regardless of the truth. At 4.0 s the same
# band identifies the closest grid point correctly at truths 0.04 to 0.16.
#
# The low end looks like it should be horizon-bound and is NOT, which cost a
# round of wrong conclusions worth recording. Scoring a domain's horizon-end
# position against the puck's eventual RESTING place silently couples the
# two: below mu ~ 0.06 the real puck is still sliding at horizon end, so
# every domain under-predicts and the argmin goes to whichever guessed the
# lowest friction, whatever the truth was. That looked exactly like a hard
# horizon floor at mu ~ v0/(g*(horizon - t_release)) ~ 0.063, and it is not
# one -- it was the comparison being taken at two different instants.
# Scoring at a matched instant instead removes it. Measured, one shot per
# truth, grid spanning [0.03, 0.2], horizon 1.6 s:
#     truth  0.03 0.05 0.06 0.08 0.12 0.16 0.20  -> closest grid point
#     truth  0.04                                -> off by one (near-tie)
#
# What remains is that the residual scales with how far the puck travels --
# 0.007 m at mu=0.08 against 0.66 m at mu=0.03 -- so an ABSOLUTE widen
# threshold is implicitly a different bar at every friction. Below ~0.08 the
# converged residual and the mismatch residual overlap and no fixed value
# separates them: 0.01 thrashes while a bar loose enough to stop that never
# fires at all. Both measured. Normalising by the observed displacement was
# tried and REVERTED -- see `make_spec_fn`'s caller and the note in
# friction_id.shot.
#
# Re-run scripts/curling/range_check.py before widening this further.
MU_LO, MU_HI = 0.03, 0.2

# Replans after release to take the scored prediction from. The release
# replan itself is unusable -- the puck is still being accelerated, so the
# horizon-end spread across domains has not opened up yet (measured: it
# disagreed with every later replan, which all agreed with each other).
PRED_OFFSET = 1

# Puck speed past which the contact model is outside what
# scripts/curling/range_check.py verifies (it prints this ceiling). A shot
# that exceeds it has left the regime the numerics were measured in, so its
# slide is not evidence about friction and the attempt is discarded.
SPEED_CEILING = 1.1


def _spaced_grid(lo: float, hi: float, n: int) -> list[float]:
    """``n`` points uniform in 1/mu (see :func:`make_spec_fn`) over ``lo..hi``.

    ``n == 1`` returns the bracket's own midpoint rather than an endpoint
    (``linspace(lo, hi, 1)`` would return ``[lo]``). Shared by
    :func:`make_spec_fn` and :func:`make_manual_spec_fn` so the grid law
    itself cannot drift between the belief-driven and manually-driven
    brackets -- only where ``lo``/``hi`` come from differs.
    """
    if n == 1:
        return [(lo + hi) / 2.0]
    s = np.linspace(1.0 / hi, 1.0 / lo, n)
    return np.sort(1.0 / s).tolist()


def make_manual_spec_fn(mu_lo: float, mu_hi: float):
    """A ``SpecFn`` drawing the 1/mu grid from an externally-set bracket.

    Companion to :func:`make_spec_fn` for a policy that decides its own
    bracket directly -- e.g. ``allocation.WorstHalfPolicy``'s survivors'
    min/max -- rather than deriving one from a :class:`DomainBelief`
    posterior. The caller (normally the policy itself, see
    ``WorstHalfPolicy.decide``) writes ``spec_fn.bracket['lo'/'hi']`` before
    calling ``spec_fn(num_randomizations)``; ``reset()`` restores it to the
    full ``(mu_lo, mu_hi)`` prior, the manual-mode equivalent of
    :meth:`DomainBelief.reset_prior`.
    """
    bracket = {"lo": mu_lo, "hi": mu_hi}

    def friction_spec(num_randomizations: int) -> dict:
        grid = _spaced_grid(bracket["lo"], bracket["hi"], num_randomizations)
        return {
            "body": {"block": {"friction": grid}},
            "geom": {"ground": {"friction": grid}},
        }

    def reset() -> None:
        bracket["lo"], bracket["hi"] = mu_lo, mu_hi

    friction_spec.bracket = bracket
    friction_spec.reset = reset
    return friction_spec


def make_spec_fn(belief: DomainBelief, mu_lo: float, mu_hi: float):
    """A ``SpecFn`` drawing the friction grid from ``belief``'s live posterior.

    One grid, written to BOTH the puck and the lane. MuJoCo takes the
    elementwise max of a contacting pair, so a value written to only one side
    is silently floored by the other; and a ``(lo, hi)`` range on both sides
    would draw them *independently*, making the effective mu the max of two
    draws rather than the value this domain is supposed to be testing.

    The bracket bisects rather than snapping to the posterior. Plain
    +/-2 sigma looks right but is not: the terminal score separates domains
    by two or three orders of magnitude, so the softmax goes nearly one-hot,
    sigma falls to ~0.002, and the next grid spans four domains within 4 mm
    of each other -- an irreversible commit to one shot's answer, on a
    parameter the ladder cannot re-widen without discarding everything. So
    the width is floored at half the current bracket: sharp evidence halves
    the range, weak evidence (4 sigma wider than that) narrows less.
    """
    bracket = {"lo": mu_lo, "hi": mu_hi}

    def friction_spec(num_randomizations: int) -> dict:
        if belief.weights is None:
            # No posterior: either the first build or a just-reset widen.
            lo, hi = mu_lo, mu_hi
        else:
            width = max(
                4.0 * belief.std, (bracket["hi"] - bracket["lo"]) / 2.0
            )
            lo = max(belief.mean - width / 2.0, mu_lo)
            hi = min(belief.mean + width / 2.0, mu_hi)
        bracket["lo"], bracket["hi"] = lo, hi

        if num_randomizations == 1:
            # linspace(lo, hi, 1) is [lo], which would park the last surviving
            # domain on the band's edge instead of the posterior's centre.
            centre = (
                belief.mean if belief.weights is not None else (lo + hi) / 2
            )
            grid = [float(np.clip(centre, mu_lo, mu_hi))]
        else:
            # Uniform in 1/mu (proportional to stopping distance, d =
            # v0^2/(2*mu*g)), not mu itself. d's slope w.r.t. mu blows up at
            # low mu and flattens at high mu, so a mu-linear grid spaces the
            # domains' PREDICTED OUTCOMES very unevenly -- high-mu domains'
            # stopping points bunch up close together, exactly where
            # score_window's position error has the least to work with.
            # Spacing s = 1/mu evenly instead spaces the predicted distances
            # evenly, trading grid density from where mu is finely resolved
            # (low mu, already well-separated in outcome space) to where the
            # outcome needs it (high mu). Endpoints map back exactly
            # (1/(1/lo) == lo), so the (lo, hi) bracket above is unchanged.
            # Only a first-order model: below mu ~ 0.06 the puck is still
            # sliding at the scoring instant (see MU_LO's note above), so
            # its matched-instant position follows the truncated
            # v0*t - 0.5*mu*g*t^2, not the resting 1/mu law, and outcome
            # separation compresses again as mu -> 0 regardless of grid
            # spacing. That doesn't undo the high-mu fix, just means the
            # low-mu end of this grid isn't fully cashing in the spacing it
            # is given.
            grid = _spaced_grid(lo, hi, num_randomizations)
        return {
            "body": {"block": {"friction": grid}},
            "geom": {"ground": {"friction": grid}},
        }

    friction_spec.bracket = bracket
    return friction_spec


def describe_nonfinite(
    error: np.ndarray, values: np.ndarray, pred, adr: int, dof: int
) -> None:
    """Print which domain(s) diverged and their state leading up to it.

    Call this once :func:`reject_reason` has returned ``"non-finite"`` -- a
    diverged rollout is discarded as evidence about friction, but it is
    still evidence about the *solver*, and silently dropping it makes that
    invisible. Walks ``pred``'s trailing steps back from wherever ``qpos``
    first goes non-finite, one domain at a time.

    Args:
        error: The ``(R,)`` array :func:`reject_reason` flagged.
        values: This shot's per-domain friction value, shape ``(R,)`` --
            e.g. ``engine.last_overrides[field][:, entity, component]``.
        pred: The scored prediction (``shot.preds[shot.released_at +
            PRED_OFFSET]``), whose ``qpos``/``qvel`` this reads.
        adr: The puck's first qpos address.
        dof: The puck's first qvel address (its linear velocity, xy).
    """
    for r in np.where(~np.isfinite(error))[0]:
        qpos = pred.qpos[r, 0, :, adr : adr + 2]
        qvel = pred.qvel[r, 0, :, dof : dof + 2]
        ok = np.where(np.all(np.isfinite(qpos), axis=1))[0]
        last_ok = int(ok[-1]) if ok.size else -1
        print(f"    domain {r} (mu={values[r]:.4f}) diverged at step "
              f"{last_ok + 1}/{qpos.shape[0] - 1}")
        for k in range(max(last_ok - 2, 0), min(last_ok + 3, qpos.shape[0])):
            tag = "" if np.all(np.isfinite(qpos[k])) else "  <- NaN"
            print(f"      step {k:>3}  xy=({qpos[k, 0]: .4f}, "
                  f"{qpos[k, 1]: .4f})  v=({qvel[k, 0]: .4f}, "
                  f"{qvel[k, 1]: .4f}){tag}")


def reject_reason(error, peak_speed: float) -> str | None:
    """Why this attempt is not usable evidence, or ``None`` if it is.

    Note that the puck does NOT have to have come to rest. The score is taken
    at a matched instant inside the horizon, so a slow shot still sliding when
    the attempt's clock runs out is perfectly good data -- which matters,
    because those are exactly the low-friction shots the band's bottom end
    depends on.
    """
    if error is None:
        return "no-release"
    if not np.all(np.isfinite(error)):
        # A diverged rollout scores as NaN, and NaN silently disables the whole
        # policy: the widen and narrow tests are both comparisons, and every
        # comparison against NaN is False, so a single bad shot would freeze
        # the belief permanently with no route back.
        return "non-finite"
    if peak_speed > SPEED_CEILING:
        # range_check.py only verifies the contact model to 1.1 m/s. Past that
        # the slide is not physics, so it is not evidence about friction.
        return f"over-ceiling({peak_speed:.2f})"
    return None


def score_window(
    pred, adr: int, observed_at: Callable[[float], np.ndarray | None],
    num_instants: int = 1,
) -> np.ndarray | None:
    """Per-domain mean squared miss, averaged over a prediction's tail.

    Averages ``num_instants`` matched-instant readings taken from the TAIL of
    one already-integrated, horizon-length prediction (``pred``, e.g. the
    replan at ``released_at + PRED_OFFSET``) against the real position
    observed at each instant's own real time -- never a fresh, re-anchored
    short prediction.

    This is a deliberately different mechanism from the per-replan chunk
    error that ``tracking.PredictionTracker`` uses elsewhere (see this
    module's own docstring history / ``friction_id.py``'s module docstring),
    which was measured on this exact task and rejected: best-vs-worst domain
    spread only 1.1x, "worse than useless" for chunk position, versus
    158-3839x for the matched-instant terminal comparison this task actually
    uses. That failure comes from RE-ANCHORING a short prediction to the true
    state every ~0.1 s, which only integrates ~0.1 s of friction difference
    before scoring -- throwing away the whole-slide integration that is the
    only thing that makes friction observable here. Do not "fix" a weak
    signal by shortening ``pred``'s own horizon or re-anchoring mid-window:
    every instant in the window comes from the tail of the SAME long
    rollout, so each one still carries the full slide's accumulated
    inter-domain separation. Only the OBSERVATION side benefits from
    windowing (it averages down real-robot tracker/interpolation noise
    around the scoring instant); it does not touch the mu -> stopping-
    distance signal shrinkage at high friction, which is a separate,
    physical limit.

    Args:
        pred: A horizon-length prediction (``qpos`` shape ``(R, 1, H + 1,
            nq)``, plus ``t0``/``dt``) -- the single best-sample slice
            ``extract_predicted_state`` records under
            ``best_sample_execution``.
        adr: First qpos address of the puck's free joint.
        observed_at: Maps a real time to the puck's observed planar xy
            (shape ``(2,)``), or ``None`` if unavailable at that time.
        num_instants: Trailing prediction steps to average over, each its
            own independent matched-instant comparison. ``1`` reproduces the
            previous single-instant (``t_end``-only) behaviour exactly.

    Returns:
        Per-domain ``(R,)`` mean squared planar miss (m^2), or ``None`` if no
        instant in the window had a usable observation.
    """
    horizon_steps = pred.qpos.shape[2] - 1
    n = min(max(int(num_instants), 1), horizon_steps + 1)
    errors = []
    for k in range(n):
        step = horizon_steps - k
        t = float(pred.t0) + step * float(pred.dt)
        obs = observed_at(t)
        if obs is None:
            continue
        predicted_xy = pred.qpos[:, 0, step, adr : adr + 2]
        errors.append(np.sum((predicted_xy - obs) ** 2, axis=1))
    if not errors:
        return None
    return np.mean(errors, axis=0)


def report(attempt, belief, allocation, miss, outcome, reject_ratio) -> None:
    """Print one attempt's row of the run table."""
    # sqrt of the best domain's squared miss: a readable metres figure, and
    # the unit the widen threshold is set in. None right after a widen, which
    # clears the posterior along with the error that triggered it.
    best_m = (
        f"{np.sqrt(belief.best_error):.4f}"
        if belief.best_error is not None else "reset"
    )
    # How many domains this shot could rule out -- the precondition for
    # narrowing, and the number to look at when it refuses to narrow.
    if belief.errors is None:
        excl = "-"
    else:
        n = len(belief.errors)
        k = int(round(belief.excluded_frac(reject_ratio) * n))
        excl = f"{k}/{n}"
    print(f"{attempt:>3} {belief.mean:>7.4f} {belief.std:>6.4f} "
          f"{best_m:>8} {excl:>5} {miss:>7.3f} "
          f"{str(allocation.current):>10}  {outcome}")
    if belief.weights is not None:
        # Per-domain posterior weight from THIS shot's softmax, mu-sorted so
        # the fan's shape (peaked vs. flat, one mode vs. two) reads at a
        # glance -- the row above only ever showed the mean/std it collapsed
        # into, never how the evidence actually split across the grid.
        order = np.argsort(belief.values)
        pairs = "  ".join(
            f"{belief.values[i]:.3f}:{belief.weights[i]:.3f}"
            for i in order
        )
        print(f"      mu:w  {pairs}")


def table_header() -> str:
    """Column header matching :func:`report`'s row layout."""
    return (f"{'try':>3} {'mu_hat':>7} {'std':>6} {'best_m':>8} "
            f"{'excl':>5} {'miss_m':>7} {'stage':>10}  outcome")
