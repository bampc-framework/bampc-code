"""Closed-loop episodes for the model_mismatch sweep: raw records, no metrics.

Unlike the state axis there is **no sensor, filter, belief or warm-up** -- the
planner is handed the true state each replan and the mismatch lives entirely in
its rollout model (uploaded to the engine once, via a ``DomainRandomizer``).
Every derived number is analysis' job.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bampc.config.scenarios import Scenario
from bampc.planner.base import PlanInfo, SamplingPlanner
from bampc.task.base import Task
from experiments.common.dr.episode import truth_stride
from experiments.common.dr.goal import GoalDriver, snapshot
from experiments.common.dr.truth import CpuTruth, WarpTruth
from experiments.model_mismatch.harness import RunConfig


@dataclass
class EpisodeResult:
    """Raw outcome of one episode. Carries no derived metrics."""

    rows: list[dict]
    contact_fraction: float
    success: bool | None


def disagreement(info: PlanInfo) -> dict[str, float]:
    """How much the domains disagree about the executed action's cost.

    From the per-domain costs of the sample the planner most preferred
    (``argmin`` of the risk-combined costs), both measures are scale-free so
    they read the same regardless of the cost's magnitude:

    * ``dom_cost_cv`` -- coefficient of variation ``std/|mean|``; 0 means every
      domain agrees on the action's cost.
    * ``dom_cost_perplexity`` -- effective number of domains, ``exp`` of the
      entropy of ``softmax(-(c - min)/std)``; ``std`` is the temperature so
      there is no free knob. Ranges 1 (one domain dominates) to R (all agree).

    NaN for an R==1 arm, which has no domain axis to disagree over.
    """
    dc = info.domain_costs
    if dc is None or dc.shape[0] <= 1:
        return {"dom_cost_cv": np.nan, "dom_cost_perplexity": np.nan}
    s = int(np.argmin(info.sample_costs))
    c = np.asarray(dc[:, s], dtype=float)
    mean, std = float(c.mean()), float(c.std())
    cv = std / abs(mean) if mean != 0.0 else np.nan
    if std > 0.0:
        z = -(c - c.min()) / std
        z -= z.max()
        w = np.exp(z)
        w /= w.sum()
        entropy = float(-np.sum(w * np.log(w + 1e-12)))
        perplexity = float(np.exp(entropy))
    else:
        # Every domain agrees exactly -> a uniform weighting over all R.
        perplexity = float(len(c))
    return {"dom_cost_cv": cv, "dom_cost_perplexity": perplexity}


def run_mismatch_episode(
    cfg: RunConfig,
    task: Task,
    planner: SamplingPlanner,
    truth: WarpTruth | CpuTruth,
    seed: int,
    repeat: int,
    scenario: Scenario,
) -> EpisodeResult:
    """One closed-loop episode under a fixed (possibly wrong) rollout model.

    The loop per replan: read the truth, plan on it directly, execute the
    chunk open-loop, log the achieved cost and the cross-domain disagreement.

    Args:
        cfg: Resolved run config.
        task: The prediction task (what the planner rolls out).
        planner: Already built on an engine of ``cfg.engine_shape(arm)`` with
            the arm's ``DomainRandomizer`` attached.
        truth: Ground-truth backend, at nominal physics.
        seed: Scenario index, and half the RNG key.
        repeat: The other half of the key, so identical repeats explore
            MJWarp's non-determinism rather than replaying one trajectory.
        scenario: The frozen start state (carries the goal-drift phase).

    Returns:
        One row per replan, the fraction of replans carrying contact, and
        the episode's success outcome (``None`` if the task defines none).
        Peg-FR3/Flip-FR3 latch success the first time it's ever seen (OR
        across the episode -- insertion/flipping are one-time achievements);
        every other task latches *failure* the first time success is ever
        false (AND -- e.g. Balance/BalanceFr3's ball must stay on the plate
        throughout).
    """
    dt_pred, dt_truth = task.dt, truth.dt
    k = truth_stride(dt_pred, dt_truth)
    steps_pred, num_replans = cfg.replan_counts(dt_pred)
    steps_truth, _ = cfg.replan_counts(dt_truth)
    if steps_truth != steps_pred * k:
        raise ValueError(
            f"replan chunk does not divide: {steps_truth} truth steps vs "
            f"{steps_pred} prediction steps x stride {k}"
        )

    planner.rng = np.random.default_rng([seed, repeat])
    truth.reset(cfg.load_start(task, seed))
    phase = scenario.drift_phase_time if task.goal_drift.enabled else 0.0
    driver = GoalDriver(task, phase)
    params = planner.init_params()

    rows: list[dict] = []
    replans_with_contact = 0
    success: bool | None = None

    for step in range(num_replans):
        qpos, qvel, t0 = truth.state()

        # Plan on the TRUE state; model mismatch is on the engine already.
        params, info = planner.optimize(
            snapshot(driver, qpos, qvel, t0), params
        )
        goal = driver.mocap(t0, qpos, qvel)

        had_contact = False
        for _ in range(steps_truth):
            _, _, t_now = truth.state()
            truth.step(planner.get_action(params, t_now))
            had_contact |= truth.has_contact()
        replans_with_contact += had_contact

        sample = cfg.tracking_sample(task, qpos, qvel, goal, had_contact)
        raw_success = sample.get("task_success")
        if raw_success is not None:
            step_success = bool(raw_success)
            if success is None:
                success = step_success
            elif cfg.task in ("peg_fr3", "flip_fr3"):
                success = success or step_success  # sticky True: OR
            else:
                success = success and step_success  # sticky False: AND

        rows.append({
            "seed": seed,
            "repeat": repeat,
            "step": step,
            "time": t0,
            **sample,
            **disagreement(info),
        })

    return EpisodeResult(
        rows=rows,
        contact_fraction=replans_with_contact / num_replans,
        success=success,
    )
