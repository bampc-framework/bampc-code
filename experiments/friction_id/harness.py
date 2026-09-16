"""Config schema + task/engine/planner builders for the friction_id sweep.

Three arms share one task/profile construction (:func:`_base`) and differ
only in how the rollout model's friction spec is built and whether a
:class:`~bampc.belief.DomainBelief` + `AllocationController` runs on
top of it:

* ``wrong_estimate`` -- R=1, friction fixed at ``cfg.wrong_mu`` (the model's
  own hardcoded nominal by default) -- never updates.
* ``wide_hedge`` -- R=``cfg.domains``, a static linspace over ``[cfg.mu_lo,
  cfg.mu_hi]`` built once at start, Average risk -- hedges but never learns.
* ``adaptive`` -- delegates to ``scripts.curling.friction_id.build``, the
  existing belief-collapse ladder (``bampc.belief.DomainBelief`` +
  ``bampc.allocation.BeliefCollapsePolicy``/``AllocationController``),
  unmodified.

Every spec mirrors friction onto both ``body.block`` and ``geom.ground``
(see ``scripts/curling/hedge.py::make_spec_fn``'s docstring) -- MuJoCo
combines a contacting pair's friction via elementwise max, so writing it to
only one side of the puck/lane pair is a silent no-op, exactly the trap the
rest of this repo's DR sweeps (``experiments/model_mismatch/harness.py``'s
``_body_val``/``_counter_geom_names``) already work around for other tasks.
"""

from __future__ import annotations

import types
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import yaml

from bampc.allocation import AllocationController
from bampc.belief import DomainBelief
from bampc.config import planner as planner_profiles
from bampc.config import reward as reward_profiles
from bampc.config import scenarios
from bampc.config.numerics import load as load_numerics
from bampc.dr import DomainRandomizer
from bampc.planner import PlannerConfig, build_planner
from bampc.planner.base import SamplingPlanner
from bampc.rollout import WarpRolloutEngine
from bampc.task.curling_fr3 import CurlingFr3
from scripts.curling import friction_id as friction_id_script

ARMS = ("wrong_estimate", "wide_hedge", "adaptive")


@dataclass(frozen=True)
class RunConfig:
    """A resolved friction_id experiment config."""

    name: str
    scenario_bank: str
    scenario: int
    planner_profile: str
    algos: tuple[str, ...]
    arms: tuple[str, ...]
    attempts: int
    flip_every: int
    num_seeds: int
    domains: int
    mu_lo: float
    mu_hi: float
    wrong_mu: float
    attempt_seconds: float
    release_speed: float
    temperature: float
    collapse_at: float
    widen_at: float
    confirmations: int
    reject_ratio: float
    sample_budget: int
    seed_offset: int = 0


def load_run_config(path: str | Path) -> RunConfig:
    """Load and resolve a friction_id config file."""
    raw = yaml.safe_load(Path(path).read_text())
    return RunConfig(
        name=raw["name"],
        scenario_bank=raw["scenario_bank"],
        scenario=int(raw.get("scenario", 0)),
        planner_profile=raw.get("planner", "curling"),
        algos=tuple(raw["algos"]),
        arms=tuple(raw.get("arms", ARMS)),
        attempts=int(raw["attempts"]),
        flip_every=int(raw["flip_every"]),
        num_seeds=int(raw["num_seeds"]),
        domains=int(raw["domains"]),
        mu_lo=float(raw["mu_lo"]),
        mu_hi=float(raw["mu_hi"]),
        wrong_mu=float(raw["wrong_mu"]),
        attempt_seconds=float(raw["attempt_seconds"]),
        release_speed=float(raw["release_speed"]),
        temperature=float(raw["temperature"]),
        collapse_at=float(raw["collapse_at"]),
        widen_at=float(raw["widen_at"]),
        confirmations=int(raw["confirmations"]),
        reject_ratio=float(raw["reject_ratio"]),
        sample_budget=int(raw["sample_budget"]),
    )


def smoke(cfg: RunConfig) -> RunConfig:
    """Shape check: 1 seed, 4 attempts, flips every 2."""
    return replace(
        cfg, name=cfg.name + "-smoke", num_seeds=1, attempts=4, flip_every=2
    )


def quick(cfg: RunConfig, num_seeds: int = 3) -> RunConfig:
    """A faster statistical look: fewer seeds, FULL attempts/flip_every."""
    return replace(cfg, name=cfg.name + "-quick", num_seeds=num_seeds)


MODES = {"smoke": smoke, "quick": quick}


@dataclass
class Campaign:
    """Everything one (algo, arm, seed) campaign needs to run its attempts."""

    task: CurlingFr3
    planner: SamplingPlanner
    belief: DomainBelief | None
    allocation: AllocationController | None


def _base(cfg: RunConfig, algo: str, num_domains: int):
    """Build the task + planner config, S = ``sample_budget // num_domains``."""
    bank = scenarios.load(cfg.scenario_bank)
    task = CurlingFr3(
        shape=bank.shape,
        scale=bank.scale,
        goal_xy=bank.goal_xy,
        trace_sites=["ee_site"],
        model_config=load_numerics("curling"),
        **reward_profiles.load("curling"),
    )
    profile = planner_profiles.load(cfg.planner_profile)
    profile["num_samples"] = cfg.sample_budget // num_domains
    planner_cfg = PlannerConfig(algo=algo, **profile)
    return bank, task, planner_cfg


def build_wrong_estimate(cfg: RunConfig, algo: str, seed: int) -> Campaign:
    """R=1, friction fixed at ``cfg.wrong_mu`` -- never updates."""
    _bank, task, planner_cfg = _base(cfg, algo, 1)
    spec = {
        "body": {"block": {"friction": cfg.wrong_mu}},
        "geom": {"ground": {"friction": cfg.wrong_mu}},
    }
    randomizer = DomainRandomizer(
        task, num_randomizations=1, spec=spec, seed=seed
    )
    engine = WarpRolloutEngine(
        task,
        num_samples=planner_cfg.num_samples,
        num_randomizations=1,
        randomizer=randomizer,
        record_predictions=True,
    )
    planner = build_planner(planner_cfg, task, engine, seed=seed)
    return Campaign(task, planner, None, None)


def build_wide_hedge(cfg: RunConfig, algo: str, seed: int) -> Campaign:
    """R=``cfg.domains``, a static grid over the full band -- never narrows."""
    _bank, task, planner_cfg = _base(cfg, algo, cfg.domains)
    grid = np.linspace(cfg.mu_lo, cfg.mu_hi, cfg.domains).tolist()
    spec = {
        "body": {"block": {"friction": grid}},
        "geom": {"ground": {"friction": grid}},
    }
    randomizer = DomainRandomizer(
        task, num_randomizations=cfg.domains, spec=spec, seed=seed
    )
    engine = WarpRolloutEngine(
        task,
        num_samples=planner_cfg.num_samples,
        num_randomizations=cfg.domains,
        randomizer=randomizer,
        record_predictions=True,
    )
    planner = build_planner(planner_cfg, task, engine, seed=seed)
    return Campaign(task, planner, None, None)


def build_adaptive(cfg: RunConfig, algo: str, seed: int) -> Campaign:
    """The existing belief-collapse ladder, from the standalone script."""
    args = types.SimpleNamespace(
        algo=algo,
        domains=cfg.domains,
        seed=seed,
        mu_lo=cfg.mu_lo,
        mu_hi=cfg.mu_hi,
        temperature=cfg.temperature,
        collapse_at=cfg.collapse_at,
        widen_at=cfg.widen_at,
        confirmations=cfg.confirmations,
        reject_ratio=cfg.reject_ratio,
        simple_hedge=False,
        num_samples=cfg.sample_budget // cfg.domains,
    )
    _bank, task, planner, belief, allocation = friction_id_script.build(args)
    return Campaign(task, planner, belief, allocation)


BUILDERS = {
    "wrong_estimate": build_wrong_estimate,
    "wide_hedge": build_wide_hedge,
    "adaptive": build_adaptive,
}
