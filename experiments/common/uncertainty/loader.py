"""Load a YAML experiment config into a :class:`RunConfig`.

Resolves the imported ``planner`` file (relative to the config, plus optional
overrides) and each role's physics, then fans the config across its axes.

It carries no domain-randomization grammar, which belongs to the model axis.
What it adds is the **bank guard**: geometry and goal drift come from
``configs/scenarios/<bank>.yaml`` and a config that restates them raises,
because a retuned drift silently applying to some runs and not others is
exactly the trap frozen banks exist to close.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import yaml

from bampc.config import numerics, planner, reward
from bampc.task.base import ModelConfig
from experiments.common.uncertainty.arms import ARMS
from experiments.common.uncertainty.setups import (
    BeliefConfig,
    FilterConfig,
    PlannerConfig,
    RunConfig,
    SensorConfig,
)

_PLANNER_FIELDS = {f.name for f in dataclasses.fields(PlannerConfig)}
_MODEL_FIELDS = {f.name for f in dataclasses.fields(ModelConfig)}
_TASKS = ("push", "balance", "push_fr3", "balance_fr3")
_ALGOS = ("ps", "cem", "mppi")

# Owned by the scenario bank. A config naming any of these is rejected rather
# than silently overridden -- the bank's start states were validated against
# that geometry, and its drift is what makes seeds comparable across sweeps.
_BANK_OWNED = ("shape", "scale", "goal_drift", "goal_xy")


def _parse(path: Path) -> dict:
    """Parse a YAML or JSON file (chosen by extension)."""
    text = path.read_text()
    if path.suffix in (".yaml", ".yml"):
        return yaml.safe_load(text)
    if path.suffix == ".json":
        return json.loads(text)
    raise ValueError(f"unsupported config extension: {path.suffix!r}")


def _load_planner(
    spec: Any, sampling_space: str | None = None
) -> PlannerConfig:
    """Resolve ``planner`` (a bare profile name, or ``{profile, overrides}``).

    Axis resolution (``by_sampling``) lives in ``bampc.config.planner``
    itself; this only applies this sweep's own ``overrides`` on top.
    """
    if isinstance(spec, str):
        name, overrides = spec, {}
    else:
        name, overrides = spec["profile"], spec.get("overrides") or {}
    data = planner.load(name, sampling=sampling_space)
    data.update(overrides)
    unknown = set(data) - _PLANNER_FIELDS
    if unknown:
        raise ValueError(f"unknown planner fields: {sorted(unknown)}")
    return PlannerConfig(**data)


def _load_physics(
    config_path: Path, role_raw: dict, role: str
) -> ModelConfig | None:
    """Resolve one role's physics.

    Two spellings: ``numerics: <profile>`` names a shared profile from
    ``configs/numerics/`` (the normal case), while ``model_config:`` takes
    the fields
    inline or via ``{base, overrides}`` -- the escape hatch for a one-off that
    should not become a profile.
    """
    profile = role_raw.get("numerics")
    block = role_raw.get("model_config")
    if profile is not None and block:
        raise ValueError(f"{role}: give `numerics` or `model_config`, not both")
    if profile is not None:
        if not isinstance(profile, str):
            raise ValueError(
                f"{role} numerics must be a profile name, got "
                f"{type(profile).__name__}"
            )
        return numerics.load(profile)
    if not block:
        return None
    if "base" in block:
        data = _parse((config_path.parent / block["base"]).resolve())
        data.update(block.get("overrides") or {})
    else:
        data = block
    unknown = set(data) - _MODEL_FIELDS
    if unknown:
        raise ValueError(
            f"unknown {role} model_config fields: {sorted(unknown)}"
        )
    return ModelConfig(**data)


def _resolve_task_params(raw: dict) -> dict:
    """Pop ``reward_profile`` (if present) and merge it under the raw fields.

    Axis resolution (``by_sampling``/``by_manipulation``) lives in
    ``bampc.config.reward`` itself; the raw fields (including any
    override values) win over what it returns.
    """
    if "reward_profile" not in raw:
        return raw
    profile_data = reward.load(
        raw["reward_profile"],
        sampling=raw.get("sampling_space"),
        manipulation=raw.get("manipulation_type"),
    )
    rest = {k: v for k, v in raw.items() if k != "reward_profile"}
    return {**profile_data, **rest}


def _names(raw: Any, allowed: tuple[str, ...], what: str) -> tuple[str, ...]:
    """Validate a list of names against the allowed set, preserving order."""
    values = list(raw) if isinstance(raw, (list, tuple)) else [raw]
    if not values:
        raise ValueError(f"{what} list is empty")
    unknown = [v for v in values if v not in allowed]
    if unknown:
        raise ValueError(
            f"unknown {what}: {unknown}; available: {list(allowed)}"
        )
    if len(set(values)) != len(values):
        raise ValueError(f"duplicate {what}: {values}")
    return tuple(values)


# Per-arm escape hatches. The first three are SIZING/RATE (consumed in
# setups.py): `num_domains` puts a non-ensemble arm on the ensembles' engine
# shape, which is what makes `point_narrow` a control for their sample count
# rather than a differently-named point arm. `settle_steps` is a PLANNER
# behaviour knob (consumed in run.py's Stack), here so the oracle can skip a
# settle its perfect state has nothing to resolve.
_OVERRIDE_FIELDS = (
    "plan_freq_hz", "num_samples", "num_domains", "settle_steps",
    "compute_lag",
)


def _arm_overrides(raw: Any, arms: tuple[str, ...]) -> dict[str, dict]:
    """Validate ``arm_overrides`` against the arms this run actually uses.

    An override naming an arm that is not running is almost always a typo or
    a leftover, and silently ignoring it would mean a config that reads as if
    it changed something it did not.
    """
    if not raw:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(
            f"arm_overrides must be a mapping, got {type(raw).__name__}"
        )
    for arm, block in raw.items():
        if arm not in ARMS:
            raise ValueError(
                f"arm_overrides names unknown arm {arm!r}; "
                f"available: {list(ARMS)}"
            )
        if arm not in arms:
            raise ValueError(
                f"arm_overrides names {arm!r}, which is not in `arms:` -- "
                "it would have no effect"
            )
        unknown = set(block) - set(_OVERRIDE_FIELDS)
        if unknown:
            raise ValueError(
                f"arm_overrides[{arm!r}] has unknown fields "
                f"{sorted(unknown)}; allowed: {list(_OVERRIDE_FIELDS)}"
            )
    return {arm: dict(block) for arm, block in raw.items()}


def load_run_config(path: str | Path) -> RunConfig:
    """Load and resolve an experiment config file."""
    path = Path(path)
    raw = _parse(path)
    if raw["task"] not in _TASKS:
        raise ValueError(f"unknown task: {raw['task']!r}")

    task_params = raw.get("task_params", {})
    owned = [k for k in _BANK_OWNED if k in task_params]
    if owned:
        raise ValueError(
            f"{path}: {owned} come from the scenario bank "
            f"({raw['scenario_bank']!r}), not the config -- the bank's start "
            "states were validated against that geometry and its drift is "
            "what makes a seed mean the same thing across sweeps. Edit "
            "configs/scenarios/<bank>.yaml, or point at a different bank."
        )

    truth_raw = raw.get("truth") or {}
    backend = truth_raw.get("backend", "warp")
    if backend not in ("warp", "cpu"):
        raise ValueError(f"unknown truth backend: {backend!r}")

    sensor = SensorConfig(**(raw.get("sensor") or {}))
    belief = BeliefConfig(**(raw.get("belief") or {}))
    kalman = FilterConfig(**(raw.get("kalman") or {}))
    arms = _names(raw["arms"], ARMS, "arm")

    return RunConfig(
        name=raw["name"],
        task=raw["task"],
        truth_backend=backend,
        planner=_load_planner(
            raw["planner"], task_params.get("sampling_space")
        ),
        scenario_bank=raw["scenario_bank"],
        num_seeds=int(raw["num_seeds"]),
        repeats=int(raw["repeats"]),
        rollout_time=float(raw["rollout_time"]),
        plan_freq_hz=float(raw["plan_freq_hz"]),
        compute_lag=bool(raw.get("compute_lag", True)),
        task_params=_resolve_task_params(task_params),
        sensor=sensor,
        belief=belief,
        kalman=kalman,
        arms=arms,
        algos=_names(raw["algos"], _ALGOS, "algo"),
        arm_overrides=_arm_overrides(raw.get("arm_overrides"), arms),
        risk_grid=(
            None if raw.get("risk_grid") is None
            else tuple(
                (name, None if alpha is None else float(alpha))
                for name, alpha in raw["risk_grid"]
            )
        ),
        risk_arms=(
            None if raw.get("risk_arms") is None
            else tuple(raw["risk_arms"])
        ),
        settle_grid=(
            None if raw.get("settle_grid") is None
            else tuple(int(n) for n in raw["settle_grid"])
        ),
        settle_arms=(
            None if raw.get("settle_arms") is None
            else tuple(raw["settle_arms"])
        ),
        prediction_model_config=_load_physics(
            path, raw.get("prediction") or {}, "prediction"
        ),
        truth_model_config=_load_physics(path, truth_raw, "truth"),
    )


def smoke(cfg: RunConfig) -> RunConfig:
    """Shape check: 1 seed, 1 repeat, 2 s rollouts, ``-smoke`` name.

    Every arm and algo still runs -- the point is to shake out shapes and
    graph captures, which is exactly what dropping conditions would skip.
    The 2 s episodes make the *numbers* meaningless; use :func:`quick` to
    tune against.
    """
    return replace(
        cfg,
        name=cfg.name + "-smoke",
        num_seeds=1,
        repeats=1,
        rollout_time=2.0,
    )


def quick(cfg: RunConfig, seed: int = 0) -> RunConfig:
    """Tuning variant: one seed, one trial, **full-length** episodes.

    Unlike :func:`smoke` the rollout time is untouched, so the cost numbers
    mean something -- one episode of the real experiment per condition
    rather than a shape check.

    What it cannot do is resolve anything: a single episode carries no
    repeat spread, and MJWarp's basin-hopping means one draw is not the
    condition's typical behaviour. Use it to move a knob and see the
    direction, never to report an effect.
    """
    return replace(
        cfg,
        name=f"{cfg.name}-quick-s{seed}",
        num_seeds=1,
        repeats=1,
        seed_offset=seed,
    )


MODES = {"smoke": smoke, "quick": quick}


def _axis_algo(cfg: RunConfig) -> list[tuple[list[str], RunConfig]]:
    """Fan the planner algorithm -- ``results/<algo>/``."""
    return [
        (
            [algo],
            replace(
                cfg,
                name=f"{cfg.name}-{algo}",
                planner=replace(cfg.planner, algo=algo),
                algos=(algo,),
            ),
        )
        for algo in cfg.algos
    ]


def _axis_arm(cfg: RunConfig) -> list[tuple[list[str], RunConfig]]:
    """Fan the estimation arm -- ``results/<algo>/<arm>/``."""
    return [
        (
            [arm],
            replace(cfg, name=f"{cfg.name}-{arm}", arm=arm, arms=(arm,)),
        )
        for arm in cfg.arms
    ]


# Algo leads, so every arm of one planner sits together: the arms are the
# comparison, and a figure reads one algo at a time.
_AXES = (_axis_algo, _axis_arm)


def _risk_tag(risk_name: str, alpha: float | None) -> str:
    """``"average"`` or ``"cvar_0.25"`` -- the fanned arm's name suffix."""
    return risk_name if alpha is None else f"{risk_name}_{alpha:g}"


def _fan_risk(
    pairs: list[tuple[list[str], RunConfig]],
) -> list[tuple[list[str], RunConfig]]:
    """Fan each arm named in ``risk_arms`` across ``risk_grid``.

    Renames the arm (``ensemble_exact`` -> ``ensemble_exact-risk_cvar_0.25``)
    rather than adding a directory level, so an arm not in ``risk_arms``
    passes through untouched and the existing ``results/<algo>/<arm>/``
    loader needs no change -- a risk variant is just another arm name to it.
    """
    out: list[tuple[list[str], RunConfig]] = []
    for parts, cfg in pairs:
        if not cfg.risk_grid or cfg.arm not in (cfg.risk_arms or ()):
            out.append((parts, cfg))
            continue
        for risk_name, alpha in cfg.risk_grid:
            tag = _risk_tag(risk_name, alpha)
            arm_name = f"{cfg.arm}-risk_{tag}"
            planner = replace(
                cfg.planner,
                risk=risk_name,
                risk_alpha=(
                    cfg.planner.risk_alpha if alpha is None else alpha
                ),
            )
            variant = replace(
                cfg, name=f"{cfg.name}-risk_{tag}", arm=arm_name,
                planner=planner,
            )
            out.append(([*parts[:-1], arm_name], variant))
    return out


def _fan_settle(
    pairs: list[tuple[list[str], RunConfig]],
) -> list[tuple[list[str], RunConfig]]:
    """Fan each arm named in ``settle_arms`` across ``settle_grid``.

    Same rename-not-nest pattern as :func:`_fan_risk` (``point`` ->
    ``point-settle_1``), so an arm not in ``settle_arms`` passes through
    untouched.
    """
    out: list[tuple[list[str], RunConfig]] = []
    for parts, cfg in pairs:
        if not cfg.settle_grid or cfg.arm not in (cfg.settle_arms or ()):
            out.append((parts, cfg))
            continue
        for n in cfg.settle_grid:
            arm_name = f"{cfg.arm}-settle_{n}"
            variant = replace(
                cfg, name=f"{cfg.name}-settle_{n}", arm=arm_name,
                planner=replace(cfg.planner, settle_steps=n),
            )
            out.append(([*parts[:-1], arm_name], variant))
    return out


def run_variants(cfg: RunConfig) -> list[tuple[list[str], RunConfig]]:
    """Expand the fan-out axes into ``(subdir_parts, cfg)`` pairs.

    Fanning rather than copying a config into a v2 is the point: conditions
    that must differ in exactly one thing cannot drift apart if they are one
    file.
    """
    out: list[tuple[list[str], RunConfig]] = [([], cfg)]
    for axis in _AXES:
        out = [
            ([*parts, *sub], variant)
            for parts, base in out
            for sub, variant in axis(base)
        ]
    return _fan_settle(_fan_risk(out))
