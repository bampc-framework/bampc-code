"""Real-robot planner node for Curling-FR3: shoot, coast, identify, repeat.

The hardware counterpart of ``scripts/curling/friction_id.py``. That script
invents reality with ``--truth-mu``; here reality is the actual lane, and the
only thing that changes is where the evidence comes from. Both narrow the
same hedge with the same policy over the same ``(R, S)`` ladder -- the shared
pieces live in ``scripts/curling/hedge.py`` so the two cannot drift.

Run ``run_control_node_push_fr3.py`` as a **separate process** alongside this
one (see
``planner_node.py``'s module docstring for why they must not share a
process). Curling's action is a 2-D EE twist, exactly like Push-FR3's
task-space mode, so it reuses that pipeline's ``TwistStamped`` writer and its
TF+JointState reader unchanged.

One shot, in four phases
------------------------
1. **Idle.** The arm sits at home -- on the launch box's near face, the edge
   closest to the robot, pulled ``--home-margin`` inside it. With
   ``--warm-start`` (default), the planner keeps replanning against the live
   arm/puck state through this whole wait -- and the colored per-domain
   ghosts keep tracking it -- so its mean is already converged by the time
   you press Enter, instead of a stale one from the previous shot having to
   catch up in the few replans before release. Nothing is sent to the robot
   during this wait either way. Press Enter to start the strike.
2. **Strike.** The planner drives the EE at the puck. The EE cannot leave the
   launch box: ``CurlingFr3`` enforces that with a velocity barrier inside its
   IK, and since MoveIt Servo does the IK on the real arm instead, that
   barrier is re-applied host-side here (``box_clamp`` below).
3. **Coast.** The instant the puck clears the box's far face the robot is
   zeroed and sent home -- contact is broken, so nothing it does now can
   affect the shot. **The planner keeps running.** It has to: the score is
   each domain's predicted puck position at the end of its horizon against
   where the puck really was at that same instant, so both the prediction and
   the observed track have to keep arriving after the arm has stopped.
4. **Judge.** The shot becomes one ``(R,)`` error vector, the policy proposes
   a narrowing or a widening with its reason and the ``(R, S)`` split it
   would move to, and you accept or decline with ``y``/``n``. Then back to 1.

Why the prompt exists
---------------------
A narrowing is irreversible short of throwing the whole posterior away, and
the policy's two guards (``--confirmations``, ``--reject-ratio``) are
thresholds tuned in simulation. On real ice a shot you know was bad -- a
bobbled release, someone bumping the lane -- is exactly the one you want to
refuse, and no bar can see that. Declining is a true no-op: the belief and
the confirmation streak are both restored.

``--tf-frame`` is **required**: the tracked puck has no canonical frame name
across setups. ``--mu-lo``/``--mu-hi`` default to the verified band and warn
rather than clamp outside it; read ``scripts/curling/hedge.py``'s header and
re-run ``scripts/curling/range_check.py`` before trusting a wider hedge.

Where widening will not save you
--------------------------------
``hedge.py``'s header records that the absolute ``--widen-at`` bar does not
separate a converged residual from a mismatch one below mu ~ 0.08. Driving
these hooks against a sim truth turned up the *other* end of the same
problem, which is worth knowing before trusting a real session:

    belief pinned at mu 0.078, truth flipped to 0.170, R=1
    shot miss 0.19-0.21 m  ->  matched-instant best error only 0.022-0.026 m

A fifth of a metre wrong on the lane, and the score that is supposed to
notice barely moves -- nowhere near the default ``--widen-at`` of 0.01 m^2
(a 10 cm miss), so no widen is ever offered and the belief stays wrong. The
likely reason is structural rather than a tuning miss: at high friction the
puck has nearly stopped by the time the scoring instant arrives, so there is
little slide left for the domains to disagree about. Low mu loses the signal
by sliding past the horizon; high mu loses it by stopping before it. If a
session gets stuck confidently wrong on grippy ice, decline nothing -- there
will be nothing to decline -- and restart with a hedge centred higher.

Run::

    # terminal 1
    uv run python scripts/ros/run_control_node_push_fr3.py --deadman
    # terminal 2
    uv run python scripts/ros/run_planner_node_curling.py --tf-frame puck
    uv run python scripts/ros/run_planner_node_curling.py --tf-frame puck \
        --viewer --domains 8 --mu-lo 0.05 --mu-hi 0.16
"""

from __future__ import annotations

import argparse
import copy
import functools
import math
import select
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import rclpy

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bampc.allocation import (  # noqa: E402
    AllocationContext,
    AllocationController,
    BeliefCollapsePolicy,
    WorstHalfPolicy,
)
from bampc.belief import DomainBelief  # noqa: E402
from bampc.config import planner as planner_profiles  # noqa: E402
from bampc.config import reward as reward_profiles  # noqa: E402
from bampc.config import scenarios  # noqa: E402
from bampc.config.numerics import load as load_numerics  # noqa: E402
from bampc.dr import DomainRandomizer  # noqa: E402
from bampc.planner import PlannerConfig, build_planner  # noqa: E402
from bampc.rollout import WarpRolloutEngine  # noqa: E402
from bampc.ros.adapters import push_fr3 as fr3_adapter  # noqa: E402
from bampc.ros.planner_node import PlannerNode  # noqa: E402
from bampc.task.curling_fr3 import (  # noqa: E402
    DEFAULT_LAUNCH_BOX,
    CurlingFr3,
)
from scripts.curling.hedge import (  # noqa: E402
    MU_HI,
    MU_LO,
    PRED_OFFSET,
    describe_nonfinite,
    make_manual_spec_fn,
    make_spec_fn,
    reject_reason,
    report,
    score_window,
    table_header,
)


def parse_args() -> argparse.Namespace:
    """CLI: the hedge/belief knobs, plus the usual real-robot plumbing."""
    p = argparse.ArgumentParser(
        description="Real-robot Curling-FR3 planner with online friction ID.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # --- task / planner -------------------------------------------------
    p.add_argument("--algo", default="ps", choices=["ps", "cem", "mppi"],
                   help="All three track predictions via the planner "
                        "profile's best_sample_execution: true (see "
                        "configs/planner/curling.yaml), which extracts the "
                        "best-cost rolled-out sample's prediction without "
                        "changing what gets executed -- mppi/cem keep their "
                        "own averaged mean for control.")
    p.add_argument("--scenario-bank", default="curling_circle",
                   help="Bank supplying shape/scale/goal_xy. Start states "
                        "come from the real puck, not from the bank.")
    p.add_argument("--goal-xy", type=float, nargs=2, default=None,
                   metavar=("X", "Y"),
                   help="Override the house's planar position (m, world) to "
                        "match where it physically sits on the real lane.")
    p.add_argument("--surface-z-offset", type=float, default=0.0,
                   help="Shift the modelled lane (m, world z) to match a "
                        "physically thicker or thinner real surface. Puck "
                        "resting height, house height and ee_z_target all "
                        "move with it.")
    p.add_argument("--launch-length", type=float, default=None,
                   help="Override the launch box's length in x (m); the "
                        f"near edge stays fixed at {DEFAULT_LAUNCH_BOX[0]}. "
                        "Default: the task's launch box. Not rotate90-aware "
                        "(edits the box's x-pair regardless).")
    p.add_argument("--rotate90", action="store_true",
                   help="Experimental: rotate the launch box (and the "
                        "release gate / home position / debug-viewer lane) "
                        "90 deg about z, so the arm pushes sideways -- "
                        "shoulder rotation -- instead of forward -- arm "
                        "extension. Unlike examples/simple/curling.py there "
                        "is no scenario-bank rotation to match: real puck/EE "
                        "start states come from the tracked sensors, not a "
                        "bank, but --goal-xy still needs to be set to "
                        "wherever the house physically sits once the box "
                        "moves -- pass it explicitly, since the bank's "
                        "default is for the unrotated lane. Omit this flag "
                        "for the exact current (unrotated) behavior.")
    p.add_argument("--lane-shift", type=float, nargs=2, default=None,
                   metavar=("DX", "DY"),
                   help="Translate the launch box (and the debug-viewer "
                        "lane visuals) by (dx, dy) world metres. Only takes "
                        "effect with --rotate90 (a no-op, with a warning, "
                        "otherwise). Does NOT shift --goal-xy -- that "
                        "always reflects the real house's own measured "
                        "position, independent of this.")
    p.add_argument("--samples", type=int, default=None,
                   help="Override the profile's S. Note the ladder is built "
                        "from R*S, so this changes every stage.")
    p.add_argument("--settle-steps", type=int, default=None,
                   help="Override the profile's settle steps.")
    p.add_argument("--arm-kv", type=float, default=None,
                   help="Override the arm's velocity-actuator gain "
                        "(CurlingFr3's own arm_kv, default: the shared "
                        "fr3_arm.xml value, kv=50) on this run's planning "
                        "model only. At kv=50 the simulated joints "
                        "under-track a commanded velocity by up to ~18%% "
                        "(worse on higher-inertia base joints), which "
                        "accounted for most of an observed Balance-FR3 "
                        "sim-vs-real gap (a kv sweep found it stable up "
                        "to ~kv=8000, plateauing by ~500-1000).")
    # --- the hedge ------------------------------------------------------
    p.add_argument("--domains", type=int, default=8, help="Starting R.")
    p.add_argument("--mu-lo", type=float, default=MU_LO,
                   help="Low end of the prior hedge. Outside the verified "
                        "band this warns rather than clamps.")
    p.add_argument("--mu-hi", type=float, default=MU_HI,
                   help="High end of the prior hedge.")
    p.add_argument("--temperature", type=float, default=1e-3,
                   help="Softmax temperature, in the error's units (m^2 of "
                        "matched-instant position miss).")
    p.add_argument("--collapse-at", type=float, default=0.02,
                   help="Narrow once the posterior std falls below this (mu).")
    p.add_argument("--widen-at", type=float, default=0.01,
                   help="Re-span once even the best domain's matched-instant "
                        "error exceeds this (m^2; 0.01 = a 10 cm miss).")
    p.add_argument("--reject-ratio", type=float, default=4.0,
                   help="A domain counts as excluded when its error is this "
                        "multiple of the best domain's.")
    p.add_argument("--confirmations", type=int, default=2,
                   help="Consecutive shots whose estimate must agree before "
                        "a narrowing is offered. Ignored under "
                        "--simple-hedge.")
    p.add_argument("--simple-hedge", action="store_true",
                   help="Use a simpler narrowing rule instead of "
                        "BeliefCollapsePolicy's softmax posterior: every "
                        "shot proposes keeping the better (lower-error) "
                        "half of the current domains and resampling the "
                        "1/mu grid between their min/max friction, one "
                        "ladder stage at a time (half R, double S) -- no "
                        "confirmations, no reject-ratio gate, no "
                        "persistent posterior. Still asks before every "
                        "narrow AND widen, same as the default policy -- "
                        "--collapse-at/--confirmations/--reject-ratio are "
                        "ignored in this mode; --widen-at still applies.")
    p.add_argument("--belief-weighting", choices=["softmax", "relative"],
                   default="softmax",
                   help="How the printed mu:w row weighs domains: "
                        "'softmax' (see --temperature) or 'relative' "
                        "(1/error, normalized -- no temperature to tune). "
                        "Only takes effect under --simple-hedge, where the "
                        "belief is display-only (WorstHalfPolicy's decision "
                        "reads raw error directly, never belief.mean/std); "
                        "outside --simple-hedge, belief.mean/std drive "
                        "BeliefCollapsePolicy's actual narrowing, so this "
                        "flag is ignored there and softmax is always used.")
    p.add_argument("--score-window", type=int, default=5,
                   help="Trailing prediction steps averaged per shot score "
                        "(see hedge.score_window). At the curling numerics "
                        "profile's dt=0.02s and the default --plan-rate "
                        "7.0 Hz (~0.143s between raw tracker samples), 5 "
                        "steps span ~0.10s -- enough to sometimes pull in "
                        "an adjacent raw tracker sample at the window's far "
                        "edge (Shot.observed_at only linearly interpolates "
                        "WITHIN one bracket of two raw samples, so several "
                        "instants inside the same bracket are a "
                        "deterministic function of that one interpolant, "
                        "not independent noise draws) without reaching back "
                        "far enough to notably shrink the accumulated "
                        "inter-domain separation. Not a large "
                        "averaging effect by design -- raise --score-window "
                        "and --plan-rate together if more is needed; 1 "
                        "reproduces the previous exact single-instant "
                        "behaviour.")
    p.add_argument("--seed", type=int, default=0)
    # --- shot geometry --------------------------------------------------
    p.add_argument("--home-margin", type=float, default=0.02,
                   help="How far inside the launch box's near face (the edge "
                        "closest to the robot) home sits, in metres. Parking "
                        "exactly on the face puts the EE where the barrier "
                        "has already saturated the inward direction.")
    p.add_argument("--release-margin", type=float, default=0.0,
                   help="Extra clearance (m) beyond the box's far face, on "
                        "top of the puck's own radius, before the shot counts "
                        "as away. Raise it if tracker noise trips release "
                        "early.")
    p.add_argument("--rest-speed", type=float, default=None,
                   help="Puck speed (m/s) counted as stopped. Default: the "
                        "task's own rest_speed.")
    # --- ROS ------------------------------------------------------------
    p.add_argument("--tf-frame", required=True,
                   help="TF frame of the tracked puck.")
    p.add_argument("--base-frame", default="fr3_link0")
    p.add_argument("--joint-topic",
                   default="/franka_robot_state_broadcaster/"
                           "measured_joint_states")
    p.add_argument("--spline-topic", default="/bampc/trajectory")
    p.add_argument("--plan-rate", type=float, default=7.0,
                   help="Hz. Curling's rollout is ~2.7x push_fr3's, so this "
                        "is lower than the usual 10; check the node's own "
                        "'optimize: N ms' log and lower it further if the "
                        "loop cannot keep up.")
    p.add_argument("--control-rate", type=float, default=50.0)
    p.add_argument("--episode-duration", type=float, default=10.0,
                   help="Backstop stop for one shot (s) from the first "
                        "optimize tick. Only has effect with --collect-data: "
                        "PlannerNode deliberately gives an interactive "
                        "rollout no time limit. A shot normally ends on its "
                        "own once the puck has stopped and the scoring "
                        "instant has passed; a strike that never releases "
                        "never ends on its own, so press Enter to stop it "
                        "and it is judged as skipped:no-release.")
    p.add_argument("--compensate-latency", default=True,
                   action=argparse.BooleanOptionalAction)
    p.add_argument("--warm-start", default=True,
                   action=argparse.BooleanOptionalAction,
                   help="Keep the planner (and the colored per-domain "
                        "ghosts) replanning against the live arm/puck state "
                        "through the whole 'press Enter' wait, instead of "
                        "freezing both at wherever the previous shot left "
                        "them. Nothing is published while idle either way -- "
                        "this only changes whether optimize() has already "
                        "converged its mean to the current state by the "
                        "time Enter is pressed, or has to catch up in the "
                        "few replans before release. Use --no-warm-start to "
                        "compare against the old frozen-idle behavior.")
    # --- viewer / recording ---------------------------------------------
    p.add_argument("--viewer", action="store_true",
                   help="Open the debug viewer with one translucent ghost "
                        "puck per domain, parked where that friction "
                        "hypothesis predicts this shot ends up. The hedge is "
                        "then visible as a fan down the lane, closing as the "
                        "belief narrows.")
    p.add_argument("--show-endpoints", default=True,
                   action=argparse.BooleanOptionalAction,
                   help="With --viewer, draw the per-domain endpoint ghosts. "
                        "Use --no-show-endpoints to open the viewer without "
                        "them.")
    p.add_argument("--record", action="store_true")
    p.add_argument("--record-camera", default="main")
    p.add_argument("--record-name", default="curling_real")
    p.add_argument("--record-dir", default=None)
    p.add_argument("--record-size", type=int, nargs=2, default=[480, 640])
    p.add_argument("--collect-data", action="store_true",
                   help="Log a per-shot CSV and enforce --episode-duration. "
                        "Note the 'full' schema's second column is named "
                        "rot_err but carries CurlingFr3.pose_error's second "
                        "value, which is the puck's speed -- a disc's yaw "
                        "does not matter, whether it has stopped does.")
    p.add_argument("--data-dir", default=None)
    return p.parse_args()


# --------------------------------------------------------------------- #
# One shot's record
# --------------------------------------------------------------------- #


@dataclass
class Shot:
    """Everything one attempt accumulates, reset before the next Enter."""

    adr: int = 0  # the puck's first qpos address, fixed for the run
    index: int = 0
    released_at: int | None = None
    t_end: float | None = None
    preds: list = field(default_factory=list)
    obs_t: list[float] = field(default_factory=list)
    obs_xy: list[np.ndarray] = field(default_factory=list)
    peak_speed: float = 0.0
    last_t: float = 0.0
    miss: float = float("nan")  # puck-to-house distance, latest tick

    def reset(self) -> None:
        """Clear the per-attempt record, keeping the attempt counter."""
        self.released_at = None
        self.t_end = None
        self.preds = []
        self.obs_t = []
        self.obs_xy = []
        self.peak_speed = 0.0
        self.last_t = 0.0
        self.miss = float("nan")

    def observed_at(self, t: float) -> np.ndarray | None:
        """The puck's planar position at time ``t``, linearly interpolated.

        The sim script samples the track every physics step and takes the
        nearest sample; here observations arrive at the plan rate (~7 Hz,
        140 ms apart), far too coarse for "nearest" -- the puck moves
        centimetres between samples.

        Linear is enough, which is worth recording because it looks like it
        should not be. Measured on a mu=0.12 shot at 7 Hz:

        * In the case that actually matters, it is *exact*. The scoring
          instant ``t_end`` is one horizon (1.6 s) past a post-release
          replan, and the slide had finished at 0.84 s -- so ``t_end`` lands
          on a stationary puck and every sample around it is the same point.
          Interpolation order is irrelevant whenever the slide fits inside
          the horizon, which is the whole band above mu ~ 0.06.
        * Where it does bite (low mu, slide outlasting the horizon) the
          sample GAP dominates, not the fit order: sweeping the moving part
          of that track, linear was 3.7 mm mean / 26 mm max against a dense
          reference, and a three-point quadratic -- tried, measured,
          reverted -- only reached 3.1 mm / 24 mm. Raise ``--plan-rate`` if
          that band needs to be sharper; a fancier fit is not the lever.
        """
        if len(self.obs_t) < 2:
            return None
        times = np.asarray(self.obs_t)
        if t <= times[0] or t >= times[-1]:
            return None
        j = int(np.searchsorted(times, t))
        t0, t1 = times[j - 1], times[j]
        w = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
        return (1.0 - w) * self.obs_xy[j - 1] + w * self.obs_xy[j]


def score(
    shot: Shot, adr: int, num_instants: int = 1
) -> np.ndarray | None:
    """Per-domain mean squared miss over a trailing window, or ``None``.

    Thin wrapper around :func:`hedge.score_window`, bound to this shot's
    stored best-sample prediction and its linearly-interpolated observation
    track (see :meth:`Shot.observed_at`). Compares at matched INSTANTs, not
    against wherever the puck eventually came to rest -- those coincide only
    when every domain's slide finishes inside the horizon; below mu ~ 0.06
    the real puck is still moving at horizon end, and scoring it against its
    later resting place makes every domain under-predict, handing the argmin
    to whichever guessed the lowest friction whatever the truth actually was.

    Absolute squared metres, deliberately NOT normalised by how far the puck
    travelled. Normalising looks right and measurably made things worse: it
    leaves the argmin untouched but acts as a per-shot temperature change.
    """
    if shot.released_at is None:
        return None
    idx = shot.released_at + PRED_OFFSET
    if idx >= len(shot.preds) or shot.preds[idx] is None:
        return None
    return score_window(shot.preds[idx], adr, shot.observed_at, num_instants)


# --------------------------------------------------------------------- #
# The prompt
# --------------------------------------------------------------------- #


def explain(belief, policy, args, num_domains: int, moved: int) -> str:
    """Why the policy decided what it did, in the units it decided in.

    Every test here is a pure read of the belief, checked in the same order
    :meth:`BeliefCollapsePolicy.decide` checks them -- except that ``moved``
    (the signed stage delta it returned) picks the branch first. It has to:
    ``decide`` zeroes its own confirmation streak on the way out of a
    successful narrowing, so re-running the gates afterwards would report the
    freshly cleared streak as the reason a narrowing that just happened had
    not happened.
    """
    if belief.best_error is None:
        return (f"best domain's miss exceeded widen-at {args.widen_at:.4f} "
                "m^2 -- no surviving hypothesis explains this shot, so the "
                "prior is re-spanned")
    frac = belief.excluded_frac(args.reject_ratio)
    k = int(round(frac * num_domains))
    if moved < 0:
        return (f"posterior collapsed (std {belief.std:.4f} < "
                f"{args.collapse_at:.4f}); this shot excluded "
                f"{k}/{num_domains} domains (>= {args.reject_ratio:.1f}x the "
                f"best error); {args.confirmations}/{args.confirmations} "
                "confirmations agree")
    if not belief.collapsed(args.collapse_at):
        return (f"posterior std {belief.std:.4f} still above collapse-at "
                f"{args.collapse_at:.4f} -- this shot did not pin the value "
                "down enough to spend range")
    if frac < policy.min_excluded:
        need = int(np.ceil(policy.min_excluded * num_domains))
        return (f"only {k}/{num_domains} domains excluded (need {need}) -- a "
                "narrow posterior over domains that all fit about equally is "
                "not knowledge")
    return (f"{policy.streak}/{args.confirmations} confirmations -- waiting "
            "for another shot to agree before committing")


def ask_yes_no(question: str) -> bool:
    """Read a single y/n from stdin, which is already in cbreak mode.

    Runs on the node's input thread, which owns stdin exclusively -- hence
    ``select`` rather than ``input()``, so a Ctrl+C still gets through.
    """
    print(f"{question} [y/n] ", end="", flush=True)
    while rclpy.ok():
        ready, _, _ = select.select([sys.stdin], [], [], 0.1)
        if not ready:
            continue
        ch = sys.stdin.read(1).lower()
        if ch in ("y", "n"):
            print(ch)
            return ch == "y"
    return False


def preview_bracket(spec_fn, num_randomizations: int) -> tuple[float, float]:
    """The hedge ``spec_fn`` would draw at ``num_randomizations``, undone.

    Calling ``spec_fn`` is how the real bracket gets computed, so previewing
    it by re-deriving the arithmetic would be a second copy that can rot.
    Instead call the real thing and put its state back: it is a pure function
    of the belief plus the bracket, so the call ``apply_stage`` makes moments
    later sees identical inputs and lands on the same numbers.
    """
    saved = dict(spec_fn.bracket)
    spec_fn(num_randomizations)
    preview = (spec_fn.bracket["lo"], spec_fn.bracket["hi"])
    spec_fn.bracket.update(saved)
    return preview


def make_prompt(shot, belief, policy, allocation, spec_fn, args, task):
    """Build the post-shot callback: score, explain, offer, apply."""
    dof = int(task.mj_model.joint("block").dofadr[0])

    def on_episode_end() -> None:
        error = score(shot, shot.adr, args.score_window)
        skipped = reject_reason(error, shot.peak_speed)

        if skipped is not None:
            report(shot.index, belief, allocation, shot.miss,
                   f"skipped:{skipped}", args.reject_ratio)
            if skipped == "non-finite":
                values = policy.engine.last_overrides[policy.field][
                    :, policy.entity, policy.component
                ]
                pred = shot.preds[shot.released_at + PRED_OFFSET]
                describe_nonfinite(error, values, pred, shot.adr, dof)
            shot.index += 1
            shot.reset()
            return

        # Snapshot before decide(): it mutates the belief (and, on the widen
        # path, discards the posterior outright), so declining has to be able
        # to put everything back.
        snap_belief = copy.deepcopy(belief.__dict__)
        snap_streak = (policy.streak, policy.last_mean)

        before = allocation.current_idx
        target = policy.decide(
            before, len(allocation.stages),
            AllocationContext(error_window=[error]),
        )
        cur_r, cur_s = allocation.current
        reason = explain(belief, policy, args, cur_r, target - before)
        lo, hi = spec_fn.bracket["lo"], spec_fn.bracket["hi"]

        report(shot.index, belief, allocation, shot.miss, "",
               args.reject_ratio)
        if target == before:
            # A widen that lands on the stage it is already at still did
            # something -- decide() discarded the posterior on its way past.
            # Labelling that "HOLD" would hide a real state change. The grid
            # itself needs no redraw: at the home stage it is already the
            # prior, which is exactly what the reset restored.
            widened_in_place = belief.best_error is None
            print(f"{'WIDEN ' if widened_in_place else 'HOLD  '}  {reason}")
            if widened_in_place:
                print(f"        ladder already at its widest "
                      f"(R={cur_r} x S={cur_s}); prior re-spanned in place, "
                      f"hedge back to [{lo:.4f}, {hi:.4f}]")
            else:
                print(f"        hedge [{lo:.4f}, {hi:.4f}] unchanged, "
                      f"split stays R={cur_r} x S={cur_s}")
        else:
            verb = "WIDEN " if target > before else "NARROW"
            new_r, new_s = allocation.stages[target]
            new_lo, new_hi = preview_bracket(spec_fn, new_r)
            print(f"{verb}  {reason}")
            print(f"        hedge [{lo:.4f}, {hi:.4f}] -> "
                  f"[{new_lo:.4f}, {new_hi:.4f}]")
            print(f"        next split: R={new_r} domains x S={new_s} samples"
                  f"   (now R={cur_r} x S={cur_s})")
            if ask_yes_no("apply?"):
                allocation.apply_stage(target)
                print(f"        applied -> {allocation.current}")
            else:
                belief.__dict__.update(snap_belief)
                policy.streak, policy.last_mean = snap_streak
                print("        declined -- belief and streak restored")

        shot.index += 1
        shot.reset()

    return on_episode_end


def make_simple_prompt(shot, belief, policy, allocation, spec_fn, args, task):
    """The ``--simple-hedge`` counterpart of :func:`make_prompt`.

    Same score/skip/report shape, but the decision comes from
    :class:`allocation.WorstHalfPolicy` -- no softmax posterior, no
    confirmations, no reject-ratio gate. ``policy.decide`` mutates
    ``spec_fn.bracket`` directly (and, on a widen, resets ``belief`` for
    display only) as it proposes, so a decline restores both, exactly as
    :func:`make_prompt` restores ``belief.__dict__``. Still asks before
    every narrow AND widen -- there is no automatic path in this mode.
    """
    dof = int(task.mj_model.joint("block").dofadr[0])

    def on_episode_end() -> None:
        error = score(shot, shot.adr, args.score_window)
        skipped = reject_reason(error, shot.peak_speed)

        if skipped is not None:
            report(shot.index, belief, allocation, shot.miss,
                   f"skipped:{skipped}", args.reject_ratio)
            if skipped == "non-finite":
                values = policy.engine.last_overrides[policy.field][
                    :, policy.entity, policy.component
                ]
                pred = shot.preds[shot.released_at + PRED_OFFSET]
                describe_nonfinite(error, values, pred, shot.adr, dof)
            shot.index += 1
            shot.reset()
            return

        snap_bracket = dict(spec_fn.bracket)
        snap_belief = copy.deepcopy(belief.__dict__)

        before = allocation.current_idx
        target = policy.decide(
            before, len(allocation.stages),
            AllocationContext(error_window=[error]),
        )
        cur_r, cur_s = allocation.current
        lo, hi = snap_bracket["lo"], snap_bracket["hi"]

        report(shot.index, belief, allocation, shot.miss, "",
               args.reject_ratio)
        if target == before:
            # A widen that lands on the stage it is already at still reset
            # the bracket on its way past -- see make_prompt's own note.
            widened_in_place = spec_fn.bracket != snap_bracket
            reason = (
                f"best domain's miss exceeded widen-at {args.widen_at:.4f} "
                "m^2 -- no surviving hypothesis explains this shot"
                if widened_in_place else
                "no domain excluded and nothing to widen from this shot"
            )
            print(f"{'WIDEN ' if widened_in_place else 'HOLD  '}  {reason}")
            if widened_in_place:
                print(f"        ladder already at its widest "
                      f"(R={cur_r} x S={cur_s}); prior re-spanned in place, "
                      f"hedge back to [{lo:.4f}, {hi:.4f}]")
            else:
                print(f"        hedge [{lo:.4f}, {hi:.4f}] unchanged, "
                      f"split stays R={cur_r} x S={cur_s}")
        else:
            widen = target > before
            new_r, new_s = allocation.stages[target]
            new_lo, new_hi = spec_fn.bracket["lo"], spec_fn.bracket["hi"]
            verb = "WIDEN " if widen else "NARROW"
            reason = (
                f"best domain's miss exceeded widen-at {args.widen_at:.4f} "
                "m^2 -- no surviving hypothesis explains this shot"
                if widen else
                "kept the better half of this shot's domains by "
                "matched-instant error"
            )
            print(f"{verb}  {reason}")
            print(f"        hedge [{lo:.4f}, {hi:.4f}] -> "
                  f"[{new_lo:.4f}, {new_hi:.4f}]")
            print(f"        next split: R={new_r} domains x S={new_s} samples"
                  f"   (now R={cur_r} x S={cur_s})")
            if ask_yes_no("apply?"):
                allocation.apply_stage(target)
                print(f"        applied -> {allocation.current}")
            else:
                spec_fn.bracket.update(snap_bracket)
                belief.__dict__.update(snap_belief)
                print("        declined -- hedge and belief restored")

        shot.index += 1
        shot.reset()

    return on_episode_end


# --------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------- #


def build(args: argparse.Namespace):
    """Assemble task, belief, engine, planner and ladder.

    Same order and the same knobs as ``friction_id.build`` -- deliberately,
    since the two are meant to be the same experiment with a different source
    of truth. Everything here is ROS-free, which is also what makes the shot
    logic testable without a robot.
    """
    if args.mu_lo < MU_LO or args.mu_hi > MU_HI:
        print(f"[warn] hedge [{args.mu_lo:.3f}, {args.mu_hi:.3f}] leaves the "
              f"verified band [{MU_LO}, {MU_HI}] -- read "
              "scripts/curling/hedge.py's header and re-run "
              "range_check.py before trusting the physics there.")

    bank = scenarios.load(args.scenario_bank)
    goal_xy = tuple(args.goal_xy) if args.goal_xy else bank.goal_xy
    launch_box = DEFAULT_LAUNCH_BOX
    if args.launch_length is not None:
        x_lo, _, y_lo, y_hi = DEFAULT_LAUNCH_BOX
        launch_box = (x_lo, x_lo + args.launch_length, y_lo, y_hi)

    # --rotate90: rotate the launch box (and, cosmetically, the debug-viewer
    # lane visuals) 90 deg CCW about the robot base. Unlike
    # examples/simple/curling.py there is no scenario bank to rotate to
    # match -- real start states come from the tracked sensors -- so goal_xy
    # is deliberately left alone here; the operator sets it to the house's
    # actual measured position via --goal-xy regardless of --rotate90.
    lane_xy, lane_yaw, downrange_axis = None, 0.0, 0
    if args.rotate90:
        if args.goal_xy is None:
            print("[warn] --rotate90 without --goal-xy: using the "
                  f"scenario bank's goal_xy={goal_xy}, which was measured "
                  "for the UNROTATED lane and almost certainly does not "
                  "match where the house physically sits now.")
        x_lo, x_hi, y_lo, y_hi = launch_box
        launch_box = (y_lo, y_hi, x_lo, x_hi)
        lane_xy = (0.0, 0.90)  # scene.xml's default ground pos, rotated
        lane_yaw = math.pi / 2
        downrange_axis = 1
        if args.lane_shift is not None:
            dx, dy = args.lane_shift
            x_lo, x_hi, y_lo, y_hi = launch_box
            launch_box = (x_lo + dx, x_hi + dx, y_lo + dy, y_hi + dy)
            lane_xy = (lane_xy[0] + dx, lane_xy[1] + dy)
    elif args.lane_shift is not None:
        print("[warn] --lane-shift has no effect without --rotate90; "
              "ignored.")

    task = CurlingFr3(
        shape=bank.shape,
        scale=bank.scale,
        goal_xy=goal_xy,
        launch_box=launch_box,
        lane_xy=lane_xy,
        lane_yaw=lane_yaw,
        downrange_axis=downrange_axis,
        surface_z_offset=args.surface_z_offset,
        trace_sites=["ee_site"],
        model_config=load_numerics("curling"),
        arm_kv=args.arm_kv,
        **reward_profiles.load("curling"),
    )
    profile = planner_profiles.load("curling")
    if args.samples is not None:
        profile["num_samples"] = args.samples
    if args.settle_steps is not None:
        profile["settle_steps"] = args.settle_steps
    planner_cfg = PlannerConfig(algo=args.algo, **profile)

    belief = DomainBelief(
        temperature=args.temperature,
        mean=(args.mu_lo + args.mu_hi) / 2,
        std=(args.mu_hi - args.mu_lo) / 2,
        weighting=args.belief_weighting if args.simple_hedge else "softmax",
    )
    if args.simple_hedge:
        spec_fn = make_manual_spec_fn(args.mu_lo, args.mu_hi)
    else:
        spec_fn = make_spec_fn(belief, args.mu_lo, args.mu_hi)

    randomizer = DomainRandomizer(
        task,
        num_randomizations=args.domains,
        spec=spec_fn(args.domains),
        seed=args.seed,
    )
    engine = WarpRolloutEngine(
        task,
        num_samples=planner_cfg.num_samples,
        num_randomizations=args.domains,
        randomizer=randomizer,
        record_predictions=True,
    )
    planner = build_planner(planner_cfg, task, engine, seed=args.seed)

    # puck == lane by construction, so either side reads the effective mu.
    ground = int(task.mj_model.geom("ground").id)
    if args.simple_hedge:
        policy = WorstHalfPolicy(
            engine,
            spec_fn,
            widen_threshold=args.widen_at,
            field="geom_friction",
            component=0,  # sliding friction
            entity=ground,
            belief=belief,  # display only -- see WorstHalfPolicy's docstring
        )
    else:
        policy = BeliefCollapsePolicy(
            engine,
            belief,
            std_threshold=args.collapse_at,
            field="geom_friction",
            component=0,  # sliding friction
            entity=ground,
            widen_threshold=args.widen_at,
            confirmations=args.confirmations,
            reject_ratio=args.reject_ratio,
        )
    allocation = AllocationController.build(
        engine, planner, policy, spec_fn=spec_fn
    )
    return task, planner, belief, policy, allocation, spec_fn


def make_hooks(task, args: argparse.Namespace):
    """The five per-shot closures the node drives, plus the shot record.

    Returns ``(shot, released, collect, shot_done, box_clamp, status_fn)``.
    Split out of :func:`main` so a test can drive them against a plain
    MuJoCo sim -- the node only ever calls these, so exercising them covers
    the phase machine without needing ROS.
    """
    adr = int(task.object_pose_qpos.adr)
    dof = int(task.mj_model.joint("block").dofadr[0])
    # task.release_x already accounts for downrange_axis and the puck's
    # bounding-sphere radius (CurlingFr3._bind_sensors); for a disc this
    # over-reads by its half-height. Deliberate: it only makes release fire
    # a few millimetres late, which is the safe direction -- an early
    # release would zero the arm while the puck is still against it.
    release_x = task.release_x + args.release_margin
    axis_name = "x" if task.downrange_axis == 0 else "y"
    rest_speed = (
        task.rest_speed if args.rest_speed is None else args.rest_speed
    )
    shot = Shot(adr=adr)
    print(f"release past {axis_name}={release_x:.3f} m (box far face "
          f"{task.release_x - task.puck_radius:.3f} + puck radius "
          f"{task.puck_radius:.3f} + margin {args.release_margin:.3f})")

    def released(state) -> bool:
        if shot.released_at is not None:
            return False
        if float(state.qpos[adr + task.downrange_axis]) <= release_x:
            return False
        # preds has not been appended for this tick yet (the observer runs
        # after optimize), so its current length IS this tick's index --
        # matching friction_id.shot's `released_at = k`.
        shot.released_at = len(shot.preds)
        return True

    # Manual note-taking aid only -- no automation, just visible on every
    # tick optimize() runs (PlannerNode's status_fn, unlike observer below,
    # also fires through the warm_start idle wait, before Enter is even
    # pressed) so a number can be written down before the puck is re-racked.
    # Throttled to ~1 Hz to match PlannerNode's own "optimize: N ms" log
    # rather than spamming one line per replan.
    last_print = [0.0]

    def status_fn(state, kin_data, info) -> None:
        if state.time - last_print[0] < 1.0:
            return
        last_print[0] = state.time
        print(f"  goal err: {task.pose_error(kin_data)[0]:.3f} m")

    def collect(state, kin_data, info) -> None:
        shot.last_t = float(state.time)
        shot.miss = task.pose_error(kin_data)[0]
        shot.obs_t.append(float(state.time))
        shot.obs_xy.append(np.asarray(state.qpos[adr : adr + 2]).copy())
        shot.preds.append(info.predicted_state)
        shot.peak_speed = max(
            shot.peak_speed, float(np.linalg.norm(state.qvel[dof : dof + 3]))
        )
        if shot.t_end is None and shot.released_at is not None:
            idx = shot.released_at + PRED_OFFSET
            if idx < len(shot.preds) and shot.preds[idx] is not None:
                pred = shot.preds[idx]
                shot.t_end = float(pred.t0) + (
                    pred.qpos.shape[2] - 1
                ) * float(pred.dt)

    def shot_done(kin_data) -> bool:
        """The shot is over once it has been scored AND the puck has stopped.

        Both halves matter. The score is taken at ``t_end``, inside the
        horizon, so the episode has to outlive that instant even when the
        puck stopped early; and a puck still sliding is still evidence being
        collected. ``task.pose_error``'s speed is honest here only because
        ``debug_viewer.update`` runs the velocity sensor stage.
        """
        if shot.t_end is None or shot.last_t < shot.t_end:
            return False
        return task.pose_error(kin_data)[1] <= rest_speed

    dt_ctrl = 1.0 / args.control_rate

    def box_clamp(actions: np.ndarray, kin_data) -> np.ndarray:
        """Re-apply the launch-box barrier that Servo's own IK would bypass.

        The device rollout clamps every step against that world's own EE
        position, so the host twin has to walk the horizon the same way --
        dead-reckoning the EE forward by the command it just clamped. Using
        one stale position for the whole horizon would over-clamp badly: at
        0.8 m/s the EE crosses the box in a quarter of a second.
        """
        ee_xy = np.asarray(kin_data.xpos[task.ee_body_id][:2], dtype=float)
        out = np.empty_like(actions)
        for i, u in enumerate(actions):
            v = task.clamp_to_launch_box(ee_xy, u)
            out[i] = v
            ee_xy = ee_xy + v * dt_ctrl
        return out

    return shot, released, collect, shot_done, box_clamp, status_fn


def main() -> int:
    """Build the stack, wire the node to its hooks, spin."""
    args = parse_args()
    task, planner, belief, policy, allocation, spec_fn = build(args)
    shot, released, collect, shot_done, box_clamp, status_fn = make_hooks(
        task, args
    )
    # Home sits on the box's near face along the downrange axis, centred on
    # the box's own midpoint along the OTHER axis -- not hardcoded to y=0,
    # since --lane-shift can move that midpoint away from the world origin.
    x_lo, x_hi, y_lo, y_hi = task.launch_box
    if task.downrange_axis == 0:
        near_face = x_lo
        home_xy = (x_lo + args.home_margin, (y_lo + y_hi) / 2.0)
    else:
        near_face = y_lo
        home_xy = ((x_lo + x_hi) / 2.0, y_lo + args.home_margin)

    state_reader_factory = functools.partial(
        fr3_adapter.make_state_reader,
        joint_topic=args.joint_topic,
        tf_frame=args.tf_frame,
        base_frame=args.base_frame,
        pose_correction=False,  # a disc has no Push-T silhouette ambiguity
    )
    home_mover_factory = functools.partial(
        fr3_adapter.make_home_mover,
        home_pos=(home_xy[0], home_xy[1], task.ee_z_target),
        home_quat=task.goal_quat_ee,
    )

    print(f"ladder (R, S): {allocation.stages}")
    print(f"start stage {allocation.current_idx}: {allocation.current}  "
          f"prior mu {belief.mean:.3f} +/- {belief.std:.3f}")
    print(f"home ({home_xy[0]:.3f}, {home_xy[1]:.3f}) m "
          f"(box near face {near_face:.3f})")
    print(f"surface z-offset: {args.surface_z_offset:+.4f} m  "
          f"ee_z_target: {task.ee_z_target:.4f} m")
    print(table_header())

    rclpy.init()
    node = PlannerNode(
        planner,
        state_reader_factory,
        plan_rate=args.plan_rate,
        control_rate=args.control_rate,
        spline_topic=args.spline_topic,
        show_viewer=args.viewer,
        show_belief=args.viewer and args.show_endpoints,
        ghost_mode="endpoint",
        compensate_latency=args.compensate_latency,
        record=args.record,
        record_camera=args.record_camera,
        record_name=args.record_name,
        record_dir=args.record_dir,
        record_size=tuple(args.record_size),
        collect_data=args.collect_data,
        episode_duration=args.episode_duration,
        data_dir=args.data_dir,
        data_label="curling",
        home_mover_factory=home_mover_factory,
        interactive=True,
        warm_start=args.warm_start,
        home_on_start=True,
        success_fn=shot_done,
        release_fn=released,
        observer=collect,
        status_fn=status_fn,
        on_episode_end=(make_simple_prompt if args.simple_hedge
                        else make_prompt)(
            shot, belief, policy, allocation, spec_fn, args, task
        ),
        action_filter=box_clamp,
    )

    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
