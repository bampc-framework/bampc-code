"""Identify the puck's friction from its own shots, and re-identify on change.

Curling is the one task where a model error cannot be absorbed by the next
replan: contact breaks at the launch box and the puck slides to rest open
loop. That makes each shot a free experiment -- every randomized domain
predicts a different resting place, and reality picks one.

So the loop here is: hedge wide over ice friction, take a shot, score each
domain on where it *said the puck would stop*, halve the range toward the
part that survives, and hand the freed budget to control samples via the
``(R, S)`` ladder (``bampc.allocation``). Repeat until a shot scores.
A narrowing cannot be undone short of discarding the whole posterior, so
each halving has to earn itself twice over: ``--confirmations`` shots must
agree, and the shot must have *excluded* domains (``--reject-ratio``) rather
than merely produced a narrow posterior. Those are different claims -- a
posterior collapses just as tightly when every hypothesis fits equally badly
as when one of them wins.

Narrowing is only half of it. Once the belief has collapsed, a *change* in
the real friction shows up as every surviving domain predicting badly at
once -- ``DomainBelief.best_error`` -- which re-spans the prior and sends the
ladder back to its wide stage. ``--flip-at`` injects exactly that: reality's
friction is rewritten mid-run, and the run should visibly re-converge.

Score the whole slide, not each replan
--------------------------------------
The signal is one number per shot: the winning sample's predicted puck
position at the end of the horizon, per domain, against where the puck
really was *at that same instant*. Matching the instant is the whole trick
-- comparing against the puck's eventual resting place instead looks
equivalent, and is, right up until a slide outlasts the horizon (see
MU_LO's comment). Measured alternatives, on this task, all failed -- they
are recorded here because each looks reasonable and costs a run to rule
out:

* Per-replan chunk error (``tracking.PredictionTracker``, which is what the
  Push-FR3 ``kv`` belief uses) is far too weak. A replan re-anchors the
  prediction to the true state every 0.1 s, and over one such chunk a
  factor-two friction difference moves the puck ~2 mm -- swamped by
  CPU-truth-vs-MJWarp discrepancy. Measured best-vs-worst domain spread was
  1.1x, versus 158-3839x for the terminal comparison below.
* Scoring chunk *position* is worse than useless: it pins to whichever end
  of the friction grid predicts the least motion, giving the same answer
  (a grid edge) whether the truth was 0.06 or 0.13.
* Chunk *velocity*/*speed* at least moves with the truth -- speed read
  0.089 / 0.127 / 0.140 for truths 0.06 / 0.09 / 0.13 -- but is biased high
  by ~0.03 and still only ~1.1x spread.

The terminal comparison picked the closest grid point to the truth at all
three of those truths. It works because friction is only observable once
integrated over the whole slide, which is exactly what a horizon-length
rollout does and what re-anchoring throws away.

``--score-window`` (default 1, i.e. this exact single-instant comparison)
optionally averages a few trailing instants of the SAME horizon-length
prediction instead of just the last one (``hedge.score_window``) -- not a
re-anchored chunk, so the whole-slide integration above still holds. It
exists here to let a real-robot windowing choice
(``run_planner_node_curling.py``'s own ``--score-window``, default 5) be
reproduced and studied in sim; sim's perfect ground truth has no tracker
noise for it to average down, so raising it here is not expected to change
anything on its own.

Watching it (``--view``)
------------------------
``--view`` opens a window and adds one translucent ghost puck per domain,
parked where that friction hypothesis predicts *this* shot ends up. The hedge
is then literally visible as a fan of pucks down the lane, and it closes as
the shot resolves. Measured on one shot at truth 0.12, hedge ``[0.03, 0.2]``:

    replan after release   ghosts span x = 0.66 .. 1.39 m   (74 cm of hedge)
    ... puck decelerating  the fan closes monotonically ...
    puck at rest           all eight ghosts within 1 mm

Ghost hue runs with the domain index and the grid is an ascending linspace,
so colour order *is* friction order -- the ghost furthest down the lane is
the lowest-mu hypothesis (measured: mu 0.03 -> x 1.395, mu 0.20 -> x 0.656,
strictly monotone). Watch the fan shrink between attempts as the belief
narrows, and snap wide again when the flip fires.

Physics is unaffected: the window steps a ghost-augmented *copy* of the
task's model (mocap bodies carry no DOF), and a --view run reproduces the
headless numbers exactly.

Run (from the repo root)::

    uv run python scripts/curling/friction_id.py
    uv run python scripts/curling/friction_id.py --truth-mu 0.12 --flip-mu 0.06
    uv run python scripts/curling/friction_id.py --attempts 20 --flip-at 6

    # the visual version: set the truth and the hedge, watch, repeat
    uv run python scripts/curling/friction_id.py --view --truth-mu 0.10 \
        --mu-lo 0.05 --mu-hi 0.16 --speed 0.3 --flip-at 4 --flip-mu 0.15

``--mu-lo``/``--mu-hi`` default to ``MU_LO``/``MU_HI``, the verified band.
Going outside it warns rather than clamps -- ``configs/numerics/curling.yaml``
documents that *lower* mu means a longer slide and a more fragile flat
contact, so down is the direction that wanders out of measured territory.
Re-run ``scripts/curling/range_check.py`` before trusting a wider hedge.
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

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
from bampc.planner.base import StateSnapshot  # noqa: E402
from bampc.rollout import WarpRolloutEngine  # noqa: E402
from bampc.sim.viewer import (  # noqa: E402
    build_ghosts,
    domain_palette,
    set_ghost_domain_alpha,
    sync_endpoint_ghosts,
)
from bampc.task.base import body_geom_ids  # noqa: E402
from bampc.task.curling_fr3 import CurlingFr3  # noqa: E402
from scripts.curling.hedge import (  # noqa: E402
    MU_HI,
    MU_LO,
    PRED_OFFSET,
    make_manual_spec_fn,
    make_spec_fn,
    reject_reason,
    report,
    score_window,
    table_header,
)

PLAN_FREQ = 10  # Hz, matching the curling examples

# Ghost-puck opacity in --view. Low enough that eight overlapping predictions
# stay readable where they bunch up at the high-friction end of the hedge.
GHOST_ALPHA = 0.3


def parse_args() -> argparse.Namespace:
    """CLI: truth values, attempt budget, and the two belief thresholds."""
    p = argparse.ArgumentParser(
        description="Online puck-friction identification for Curling-FR3.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--algo", default="ps", choices=["ps", "cem", "mppi"],
                   help="All three track predictions via the planner "
                        "profile's best_sample_execution: true (see "
                        "configs/planner/curling.yaml), which extracts the "
                        "best-cost rolled-out sample's prediction without "
                        "changing what gets executed -- mppi/cem keep their "
                        "own averaged mean for control.")
    p.add_argument("--domains", type=int, default=8, help="Starting R.")
    p.add_argument("--num-samples", type=int, default=None,
                   help="Starting S. Default: the curling planner "
                        "profile's num_samples.")
    p.add_argument("--scenario", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--attempts", type=int, default=16,
                   help="Each ladder step costs --confirmations shots.")
    p.add_argument("--attempt-seconds", type=float, default=4.0,
                   help="Sim seconds per shot. Must cover setup plus the "
                        "whole slide at the SLOWEST friction in the band -- "
                        "see MU_LO; a shot still moving when the clock runs "
                        "out is skipped, not scored.")
    p.add_argument("--truth-mu", type=float, default=0.12,
                   help="Reality's friction, hidden from the planner.")
    p.add_argument("--mu-lo", type=float, default=MU_LO,
                   help="Low end of the prior hedge. Outside the verified "
                        "band this warns rather than clamps -- read this "
                        "file's header first.")
    p.add_argument("--mu-hi", type=float, default=MU_HI,
                   help="High end of the prior hedge.")
    p.add_argument("--view", action="store_true",
                   help="Open a viewer window and watch each shot, with one "
                        "translucent ghost puck per domain showing where that "
                        "friction hypothesis predicts this shot ends up.")
    p.add_argument("--speed", type=float, default=0.25,
                   help="Sim seconds per wall second in --view. Below 1 is "
                        "slow motion. The planner needs ~0.13 s per replan "
                        "whatever this says, so it is an upper bound.")
    p.add_argument("--flip-mu", type=float, default=0.06,
                   help="What reality changes to at --flip-at.")
    p.add_argument("--flip-at", type=int, default=-1,
                   help="Attempt index at which to change reality's "
                        "friction; negative disables the flip.")
    p.add_argument("--temperature", type=float, default=1e-3,
                   help="Softmax temperature, in the error's units (m^2 of "
                        "matched-instant position miss).")
    p.add_argument("--collapse-at", type=float, default=0.02,
                   help="Narrow once the posterior std falls below this (mu). "
                        "The default is about one grid step at R=8 over the "
                        "full band.")
    p.add_argument("--widen-at", type=float, default=0.01,
                   help="Re-span once even the best domain's matched-instant "
                        "error exceeds this (m^2; 0.01 = a 10 cm miss). "
                        "Absolute, so it is implicitly a tighter bar at low "
                        "friction, where slides are long -- see MU_LO.")
    p.add_argument("--release-speed", type=float, default=0.05,
                   help="Puck speed (m/s) taken to mean the shot is away.")
    p.add_argument("--reject-ratio", type=float, default=4.0,
                   help="A domain counts as excluded when its error is this "
                        "multiple of the best domain's. The error is squared "
                        "metres, so 4 = 'missed twice as far'. Narrowing "
                        "additionally requires half the domains to be "
                        "excluded, so a shot that told the hypotheses apart "
                        "is a precondition for spending the range.")
    p.add_argument("--confirmations", type=int, default=2,
                   help="Consecutive shots whose estimate must agree before "
                        "narrowing. 1 commits to a single shot, which one "
                        "unlucky launch speed is enough to get wrong. "
                        "Ignored under --simple-hedge.")
    p.add_argument("--simple-hedge", action="store_true",
                   help="Use a simpler narrowing rule instead of "
                        "BeliefCollapsePolicy's softmax posterior: every "
                        "shot proposes keeping the better (lower-error) "
                        "half of the current domains and resampling the "
                        "1/mu grid between their min/max friction, one "
                        "ladder stage at a time (half R, double S) -- no "
                        "confirmations, no reject-ratio gate, no "
                        "persistent posterior. Also switches from "
                        "automatic narrow/widen to an interactive y/n "
                        "prompt after each shot (see "
                        "scripts/ros/run_planner_node_curling.py's own "
                        "prompt, which this mirrors) -- --collapse-at/"
                        "--confirmations/--reject-ratio are ignored in "
                        "this mode; --widen-at still applies.")
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
    p.add_argument("--score-window", type=int, default=1,
                   help="Trailing prediction steps averaged per shot score "
                        "(see hedge.score_window). Left at 1 (the previous "
                        "exact single-instant behaviour) by default: sim has "
                        "perfect ground truth and no tracker interpolation "
                        "noise for a window to average down, unlike the "
                        "real-robot node's --score-window "
                        "(scripts/ros/run_planner_node_curling.py). Change "
                        "this only to reproduce/study the real-robot "
                        "windowing effect in simulation, not as a default "
                        "sim improvement.")
    return p.parse_args()


def build(args: argparse.Namespace):
    """Assemble task, belief, engine, planner and ladder."""
    bank = scenarios.load("curling_circle")
    task = CurlingFr3(
        shape=bank.shape,
        scale=bank.scale,
        goal_xy=bank.goal_xy,
        trace_sites=["ee_site"],
        model_config=load_numerics("curling"),
        **reward_profiles.load("curling"),
    )
    profile = planner_profiles.load("curling")
    if args.num_samples is not None:
        profile["num_samples"] = args.num_samples
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
    return bank, task, planner, belief, allocation


class ViewerClosedError(Exception):
    """The window was closed mid-shot, so the run is abandoned."""


@dataclass
class View:
    """The viewer window, its ghost pucks, and slow-motion pacing.

    Physics runs on ``model``/``data`` rather than the task's own model: the
    ghosts are mocap bodies compiled into a *copy* of the task's spec, so the
    thing on screen has to be the thing being stepped. Mocap bodies carry no
    degrees of freedom, so ``nq``/``nv`` are unchanged and a scenario pose
    drops straight in.
    """

    viewer: object
    model: mujoco.MjModel
    data: mujoco.MjData
    mocap_ids: np.ndarray
    body_ids: np.ndarray
    body_name: str
    alpha: float
    step_wall: float
    _next: float = 0.0

    def show_endpoints(self, planner, sample_costs: np.ndarray) -> None:
        """Move each domain's ghost to where it thinks this shot ends up.

        One ghost per domain, at the scored sample -- the argmin of
        ``sample_costs`` (this step's ``info.sample_costs``), the same one
        `shot` reads. Not hardcoded to raw sample 0: that only coincides with
        the argmin for PredictiveSampling. Hue runs with the domain index,
        and the grid is an ascending linspace, so the colour order *is* the
        friction order: the low-mu end of the hedge is the ghost that slid
        furthest.
        """
        num_randomizations = planner.engine.num_randomizations
        best = int(np.argmin(sample_costs))
        pos, quat = planner.engine.final_body_pose(self.body_name)
        sync_endpoint_ghosts(
            self.data, self.mocap_ids[:num_randomizations], [best], pos, quat
        )
        # Ghost count is fixed at compile time; the ladder only ever narrows
        # from its home stage, so hiding the tail covers every stage reached.
        set_ghost_domain_alpha(
            self.model, self.body_ids, num_randomizations, self.alpha
        )

    def tick(self) -> None:
        """Draw one stepped frame, holding it to the requested speed."""
        if not self.viewer.is_running():
            raise ViewerClosedError
        self.viewer.sync()
        now = time.perf_counter()
        if now < self._next:
            time.sleep(self._next - now)
        # Catch up rather than accumulate debt: a replan takes far longer than
        # one frame's wall budget, and without this the loop would sprint
        # through the following steps trying to win the time back.
        self._next = max(now, self._next) + self.step_wall


def make_view(task, num_domains: int, alpha: float, step_wall: float):
    """Compile a ghost-augmented copy of the task's model and wrap it.

    Returns the parts needed to launch a window; the caller owns the
    ``launch_passive`` context and finishes the :class:`View`.
    """
    palette = domain_palette(num_domains, alpha=alpha)
    vmodel, built = build_ghosts(
        task.mj_spec,
        task.endpoint_bodies,
        {"endpoint": (num_domains, 1, palette)},
    )
    # build_ghosts recompiles the spec, which is structure-only -- the solver
    # and timestep live in the task's ModelConfig. Without this the window
    # would step MuJoCo's compile defaults (dt 0.002) instead of curling's.
    if task.model_config is not None:
        task.model_config.apply_to(vmodel)
    mocap_ids, body_ids = built["endpoint"]
    return vmodel, mocap_ids[0], body_ids[0], alpha, step_wall


def episode_data(bank, task, scenario, view: View | None):
    """A fresh, posed :class:`MjData` for one attempt.

    Headless gets a new one each time; the viewer reuses its own, since
    ``launch_passive`` is bound to the object it was handed.
    """
    posed = scenarios.pose(bank, task, scenario)
    if view is None:
        return posed
    md = view.data
    md.qpos[:] = posed.qpos
    md.qvel[:] = posed.qvel
    md.ctrl[:] = 0.0
    md.time = 0.0
    num_mocap = task.mj_model.nmocap
    if num_mocap:
        # The task's own mocaps (the house) keep their ids -- ghosts were
        # appended after them.
        md.mocap_pos[:num_mocap] = posed.mocap_pos[:num_mocap]
        md.mocap_quat[:num_mocap] = posed.mocap_quat[:num_mocap]
    mujoco.mj_forward(view.model, md)
    return md


def shot(
    task, planner, mj_model, md, replans, steps_per_replan, release_speed,
    view=None, score_window_size: int = 1,
):
    """Run one attempt to rest.

    Returns:
        ``(success, terminal_error, miss, final_speed, peak_speed)``.
        ``terminal_error`` is the per-domain ``(R,)`` squared distance between
        each domain's predicted puck position at the end of its horizon and
        where the puck really was at that same instant, or ``None`` if the
        shot never left the launch box.
    """
    adr = int(task.object_pose_qpos.adr)
    params = planner.init_params()
    preds, released_at, success = [], None, False
    obs_t: list[float] = []
    obs_xy: list[np.ndarray] = []
    peak_speed = 0.0
    for k in range(replans):
        state = StateSnapshot(
            qpos=md.qpos[: mj_model.nq].copy(),
            qvel=md.qvel[: mj_model.nv].copy(),
            time=float(md.time),
        )
        params, info = planner.optimize(state, params)
        preds.append(info.predicted_state)
        if view is not None:
            view.show_endpoints(planner, info.sample_costs)
        for _ in range(steps_per_replan):
            u = planner.get_action(params, float(md.time))
            md.ctrl[:] = task.control_map_host(md, u)
            mujoco.mj_step(mj_model, md)
            obs_t.append(float(md.time))
            obs_xy.append(md.qpos[adr : adr + 2].copy())
            if view is not None:
                view.tick()
        speed = task.pose_error(md)[1]
        peak_speed = max(peak_speed, speed)
        if released_at is None and speed > release_speed:
            released_at = k
        # Sticky: curling is a one-time achievement, so a shot that ever rests
        # in the house has scored (cf. peg_fr3 / flip_fr3).
        success = success or task.task_success(md)

    miss, final_speed = task.pose_error(md)
    idx = None if released_at is None else released_at + PRED_OFFSET
    if idx is None or idx >= len(preds) or preds[idx] is None:
        return success, None, miss, final_speed, peak_speed

    # Compare at matched INSTANTs, not against wherever the puck eventually
    # came to rest. Those coincide only when every domain's slide finishes
    # inside the horizon; below mu ~ 0.06 the real puck is still moving at
    # horizon end, and scoring it against its later resting place makes every
    # domain under-predict, handing the argmin to whichever guessed the
    # lowest friction whatever the truth actually was.
    times = np.asarray(obs_t)

    def observed_at(t: float) -> np.ndarray | None:
        # Nearest-sample, not linear interpolation: sim steps every physics
        # tick densely with perfect ground truth, so there is no coarse
        # tracker gap to interpolate across (contrast Shot.observed_at on
        # the real-robot side, in run_planner_node_curling.py).
        if not times.size:
            return None
        j = int(np.argmin(np.abs(times - t)))
        return obs_xy[j]

    # Absolute squared metres, deliberately NOT normalised by how far the
    # puck travelled. Normalising looks right -- it would make --widen-at one
    # bar across the whole band instead of an implicitly friction-dependent
    # one -- and measurably made things worse: dividing every domain by the
    # same per-shot scalar leaves the argmin untouched but acts as a per-shot
    # temperature change, and the 0.13 -> 0.08 flip went from recovering in
    # two attempts to never widening at all (stuck at 0.102 vs truth 0.080).
    error = score_window(preds[idx], adr, observed_at, score_window_size)
    return success, error, miss, final_speed, peak_speed


def prompt_switch(allocation, belief, error: np.ndarray) -> tuple[bool, str]:
    """Interactive twin of ``allocation.maybe_switch``, for ``--simple-hedge``.

    ``WorstHalfPolicy.decide`` mutates ``spec_fn.bracket`` (and, on a widen,
    ``belief``) as it proposes -- snapshot both first so a decline can
    restore them untouched, mirroring how
    ``run_planner_node_curling.make_prompt`` snapshots ``belief.__dict__``
    for ``BeliefCollapsePolicy``. Prompts for both a narrow AND a widen
    (unlike the automatic path this replaces), same as the real-robot node.
    """
    spec_fn = allocation.spec_fn
    before = allocation.current_idx
    snap_bracket = dict(spec_fn.bracket)
    snap_belief = copy.deepcopy(belief.__dict__)
    target = allocation.policy.decide(
        before, len(allocation.stages),
        AllocationContext(error_window=[error]),
    )
    if target == before:
        if spec_fn.bracket != snap_bracket:
            return False, " widen(reset in place, already wide)"
        return False, ""

    widen = target > before
    cur_r, cur_s = allocation.current
    new_r, new_s = allocation.stages[target]
    verb = "widen" if widen else "narrow"
    print(f"\n  {verb} proposed: R={cur_r}x{cur_s} -> R={new_r}x{new_s}   "
          f"hedge [{snap_bracket['lo']:.4f}, {snap_bracket['hi']:.4f}] -> "
          f"[{spec_fn.bracket['lo']:.4f}, {spec_fn.bracket['hi']:.4f}]")
    if input("  apply? [y/n] ").strip().lower() == "y":
        return allocation.apply_stage(target), ""
    spec_fn.bracket.update(snap_bracket)
    belief.__dict__.update(snap_belief)
    return False, " declined"


def current_grid(planner, entity: int) -> str:
    """The friction values the domains are currently testing, as one line."""
    overrides = planner.engine.last_overrides
    if overrides is None:
        return ""
    vals = overrides["geom_friction"][:, entity, 0]
    return "hedge: " + " ".join(f"{v:.4f}" for v in vals)


def run_attempts(args, bank, task, planner, belief, allocation, model, view):
    """Shoot until a shot scores, updating the belief once per shot."""
    scenario = bank[args.scenario]
    ground = int(model.geom("ground").id)

    # Reality. Written only now: build_planner is what captured the rollout
    # graph (mjw.put_model takes a one-time device copy), so from here on the
    # stepped model is a truth the planner's domains never see.
    truth_gids = [*body_geom_ids(model, "block"), ground]

    def set_truth(mu: float) -> None:
        model.geom_friction[truth_gids, 0] = mu

    set_truth(args.truth_mu)
    steps_per_replan = max(
        int(round((1.0 / PLAN_FREQ) / model.opt.timestep)), 1
    )
    replans = max(int(round(args.attempt_seconds * PLAN_FREQ)), 1)

    print(table_header())

    truth_mu, scored = args.truth_mu, False
    for attempt in range(args.attempts):
        if attempt == args.flip_at:
            truth_mu = args.flip_mu
            set_truth(truth_mu)
            print(f"--- reality's friction changed to {truth_mu:.3f} ---")
        if view is not None:
            print(current_grid(planner, ground))

        md = episode_data(bank, task, scenario, view)
        planner.rng = np.random.default_rng([args.seed, attempt])
        success, error, miss, _final_speed, peak_speed = shot(
            task, planner, model, md, replans, steps_per_replan,
            args.release_speed, view=view,
            score_window_size=args.score_window,
        )

        note = ""
        switched = False
        before = allocation.current_idx
        reason = reject_reason(error, peak_speed)
        if reason is not None:
            note = f" skipped:{reason}"
        elif args.simple_hedge:
            switched, note = prompt_switch(allocation, belief, error)
        else:
            switched = allocation.maybe_switch(
                AllocationContext(error_window=[error])
            )
            if not switched and belief.best_error is None:
                # reset_prior() cleared the posterior, so the widen fired --
                # but the ladder was already home, so no stage changed.
                note = " mismatch(already wide)"

        outcome = ("SCORED" if success else "") + note
        if switched:
            widened = allocation.current_idx > before
            verb = "WIDEN" if widened else "narrow"
            outcome += f" {verb}->{allocation.current}"
        report(attempt, belief, allocation, miss, outcome, args.reject_ratio)

        scored = scored or success
        # Keep going past a score while a flip is still ahead: the point of
        # that run is to watch the belief recover, not to stop at the first
        # success.
        if success and attempt >= args.flip_at:
            break

    print(f"\nfinal mu_hat {belief.mean:.4f} +/- {belief.std:.4f} "
          f"(truth {truth_mu:.3f}), stage {allocation.current}")
    return 0 if scored else 1


def main() -> int:
    """Build everything, open a window if asked, and run; exit code 0/1."""
    args = parse_args()
    if args.mu_lo < MU_LO or args.mu_hi > MU_HI:
        print(f"[warn] hedge [{args.mu_lo:.3f}, {args.mu_hi:.3f}] leaves the "
              f"verified band [{MU_LO}, {MU_HI}] -- read this file's header "
              "and re-run range_check.py before trusting the physics there.")
    bank, task, planner, belief, allocation = build(args)

    print(f"ladder (R, S): {allocation.stages}")
    print(f"start stage {allocation.current_idx}: {allocation.current}  "
          f"prior mu {belief.mean:.3f} +/- {belief.std:.3f}")
    print(f"truth mu {args.truth_mu:.3f}"
          + (f" -> {args.flip_mu:.3f} at attempt {args.flip_at}"
             if args.flip_at >= 0 else ""))

    common = (args, bank, task, planner, belief, allocation)
    if not args.view:
        return run_attempts(*common, task.mj_model, None)

    vmodel, mocap_ids, body_ids, alpha, step_wall = make_view(
        task, args.domains, GHOST_ALPHA,
        float(task.mj_model.opt.timestep) / max(args.speed, 1e-3),
    )
    vdata = mujoco.MjData(vmodel)
    with mujoco.viewer.launch_passive(vmodel, vdata) as viewer:
        view = View(
            viewer, vmodel, vdata, mocap_ids, body_ids,
            task.endpoint_bodies[0], alpha, step_wall,
        )
        try:
            code = run_attempts(*common, vmodel, view)
        except ViewerClosedError:
            print("\nwindow closed -- run abandoned")
            code = 1
    return code


if __name__ == "__main__":
    raise SystemExit(main())
