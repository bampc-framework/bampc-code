"""Config schema + DR-spec/arm builders for the model_mismatch sweep.

The *model* axis measured with the state axis's achieved-error paradigm: the
planner always sees the true state, but its rollout model is deliberately
wrong -- or, with ``bias_target: truth`` (see ``RunConfig``), the planner's
model is the nominal one and the REAL world is what's offset, for a
parameter reality is less certain about than simulation is. Two arm
families:

* **biased** (R=1) -- one physics parameter (mass / friction / impratio) off
  the truth by a fixed multiplier;
* **wide DR** (R>1) -- uniform domain randomization over one or all three of
  those parameters, aggregated with Average risk.

Data lives in YAML (parsed by :func:`load_run_config`); behaviour that cannot
serialize -- building the task, resolving a DR spec, scoring the achieved
state -- lives here.

Deliberately not shared with ``experiments/common/uncertainty/`` (the state
axis): a change made for one axis must not silently move the other's numbers.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import yaml

from bampc.config import numerics, planner, reward, scenarios
from bampc.planner import PlannerConfig, engine_shape
from bampc.task.balance_fr3 import BalanceFr3
from bampc.task.base import ModelConfig, Task
from bampc.task.flip_fr3 import FlipFr3
from bampc.task.peg_fr3 import PegFr3
from bampc.task.push_fr3 import PushFr3

# Default swept parameters (Push-FR3). A config may name its own set via
# `bias_params:` -- the balance sweep uses rolling_friction / mass / friction /
# solimp_dmin. Each name maps to a DR spec through `_PARAM_SPEC` below.
BIAS_PARAMS = ("mass", "friction", "impratio")


# --------------------------------------------------------------------- #
# RunConfig
# --------------------------------------------------------------------- #


@dataclass(frozen=True)
class RunConfig:
    """A resolved model_mismatch experiment config."""

    name: str
    task: str  # push_fr3 | balance_fr3 | peg_fr3 | flip_fr3
    scenario_bank: str
    truth_backend: str  # warp | cpu
    planner: PlannerConfig
    num_seeds: int
    repeats: int
    rollout_time: float
    plan_freq_hz: float
    num_domains: int  # R for the wide_dr* arms
    wide_dr_range: tuple[float, float]  # multiplier bracket per param
    task_params: dict[str, Any]
    arms: tuple[str, ...]
    algos: tuple[str, ...]
    bias_params: tuple[str, ...] = BIAS_PARAMS  # the swept parameter set
    # A figures-only subset of bias_params to actually draw a sensitivity
    # facet for (analysis.py's bias_sensitivity/plot_sensitivity) -- lets a
    # config omit one facet while keeping that parameter a real arm, in the
    # data and in every other figure. None (default) means every
    # bias_params entry.
    plot_bias_params: tuple[str, ...] | None = None
    # Which role the arm's DR spec lands on: "prediction" (default -- the
    # planner's rollout model is wrong, truth stays nominal, as everywhere
    # else in this module) or "truth" (the planner keeps its nominal model,
    # the REAL world is offset instead -- e.g. a friction coefficient that
    # can't be measured as reliably as mass, so the sweep asks "how wrong can
    # reality be before a certainty-equivalent planner suffers"). A third
    # value, "truth_grid", combines both: truth is pinned per-cell at
    # `true_bias_mult` (see below) and the ordinary bias-arm grammar reruns
    # relative to THAT value instead of the global nominal -- "does model
    # mismatch behave the same regardless of what reality's own parameter
    # actually is." Only `truth_grid_variants` sets it up.
    bias_target: str = "prediction"
    # Truth-grid mode (bias_target == "truth_grid"): the multipliers (of the
    # global nominal) to pin truth's `bias_params[0]` at, one leg per value --
    # `truth_grid_variants` fans over this x `arms`. `true_bias_mult` carries
    # ONE cell's value per fanned variant; it is never set directly from
    # YAML. Parameter-agnostic (push_fr3's friction, balance_fr3's
    # rolling_friction, ... -- whatever `bias_params[0]` names).
    true_bias_grid: tuple[float, ...] | None = None
    true_bias_mult: float = 1.0
    # Peg-FR3 mount uncertainty (task == "peg_fr3"). The grasp/mount offset is a
    # static MODEL parameter (the peg's pose in the wrist frame), not a state:
    # `true_mount_offset` (m, along x) is baked into the TRUTH's peg mount, and
    # `mount_noise` (m) / `mount_noise_rot` (rad, default mount_noise/peg_len)
    # size the `hedge` arm's uniform cloud around the nominal (0-offset) mount.
    true_mount_offset: float = 0.0
    mount_noise: float = 0.0
    mount_noise_rot: float | None = None
    # Truth-side tilt (rad), the rotational twin of true_mount_offset -- a
    # real grasp/mount error has an angular component too, and jams
    # differently (binds at depth, not at the mouth). `true_mount_tilt_axis`
    # picks which of peg_body's local rotation axes carries it: "y" (default)
    # is co-planar with the x-translation offset (same x-z plane -- the easy,
    # correlated case); "x" is orthogonal (a direction the translation error
    # says nothing about -- the case where the hedge's symmetric rot_x/rot_y
    # cloud and the truth can actually disagree in direction). Both oracle
    # (which must "know" the true axis to cheat correctly) and the truth
    # injection in run.py's Stack.__init__ read this; the hedge cloud itself
    # doesn't need it since it already randomizes both axes symmetrically.
    true_mount_tilt: float = 0.0
    true_mount_tilt_axis: str = "y"
    # `truth_mode` picks what `peg_grid_variants` fans as the true-error
    # axis: "translation" (default -- true_offset_grid, today's exact
    # behaviour), "tilt" (true_tilt_grid, true_offset pinned at 0), or
    # "combined" (true_offset_grid x true_tilt_grid zipped by index --
    # MATCHED SEVERITY pairs, not a cross-product; a real grasp error isn't
    # correlated with itself at every combination, so this tests "does
    # hedging survive a realistic combined error" at a few severity levels,
    # not an exhaustive 2-D interaction sweep).
    truth_mode: str = "translation"
    # Peg-FR3 misalignment GRID (task == "peg_fr3", arms point/oracle/hedge):
    # when set, `peg_grid_variants` fans clearance x true_offset (x noise
    # ratio x domains, for hedge) instead of the ordinary (algo, arm) fan
    # `run_variants` does. `None` (every other config, including the plain
    # peg mount-uncertainty one) is a no-op.
    clearance_grid: tuple[float, ...] | None = None
    true_offset_grid: tuple[float, ...] | None = None
    # rad, fanned like true_offset_grid but for truth_mode "tilt"/"combined".
    true_tilt_grid: tuple[float, ...] | None = None
    # `hedge`'s noise is `ratio * true_offset` (or `ratio * true_tilt` for
    # mount_noise_rot, when truth_mode injects a tilt) for that cell, not an
    # absolute magnitude -- so it stays calibrated to whatever true error is
    # being tested, per cell. This is the "matched" hedge sub-arm.
    noise_ratio_grid: tuple[float, ...] | None = None
    # Absolute (not ratio-relative) hedge translation half-width (m) -- the
    # "deploy" sub-arm: a single number you could actually configure on
    # hardware without knowing the true offset. mount_noise_rot auto-derives
    # from it (n/peg_len, _peg_arm_spec's existing fallback) rather than
    # being set independently, so under a nonzero truth tilt this arm's tilt
    # coverage is a side effect of its translation spread, not calibrated to
    # the true tilt -- deliberately kept (not a bug) so it can be compared
    # against hedge_tilt_spread_grid below to see whether that coupling was
    # doing real work.
    hedge_spread_grid: tuple[float, ...] | None = None
    # Absolute hedge ROTATION half-width (rad), independent of translation
    # (mount_noise=0 for these cells) -- the "tiltdeploy" sub-arm, decoupled
    # from hedge_spread_grid's incidental tilt coverage.
    hedge_tilt_spread_grid: tuple[float, ...] | None = None
    # R values for `hedge`'s domains axis. Unlike every other fanned arm,
    # these do NOT split a fixed nworld budget -- see `hedge_fixed_samples`.
    hedge_domains_grid: tuple[int, ...] | None = None
    # R values for a ONE-CELL (middle clearance, middle true-error magnitude)
    # fixed-nworld probe: hedge_fixed_samples=False here, so S =
    # planner.sample_budget // R -- answers "does R vs S trade off at fixed
    # total compute," which hedge_domains_grid's fixed-S convention above
    # cannot (nworld grows with R there). None (default) adds nothing.
    allocation_probe_domains: tuple[int, ...] | None = None
    # Risk-strategy grid for `hedge`, pairs of (risk_name, alpha) -- alpha
    # `None` where the strategy doesn't use one (average, worstcase). `None`
    # (default) is a no-op: `peg_grid_variants` runs `hedge` once per
    # (clearance, true_offset, ratio, domains) at the planner profile's own
    # risk, as today.
    risk_grid: tuple[tuple[str, float | None], ...] | None = None
    # Set only by `peg_grid_variants` (never from YAML): when True, `hedge`
    # holds S at the planner profile's own num_samples instead of splitting
    # `planner.sample_budget` across R -- the domains axis buys R without
    # shrinking control samples, unlike the fixed-nworld convention every
    # other fanned arm uses.
    hedge_fixed_samples: bool = False
    prediction_model_config: ModelConfig | None = None
    truth_model_config: ModelConfig | None = None
    # Set as the config fans; each variant carries one value and clears the
    # list it came from.
    algo: str | None = None
    arm: str | None = None
    # First bank scenario to run (lets a one-seed quick run pick the scenario).
    seed_offset: int = 0

    @property
    def bank(self) -> scenarios.ScenarioBank:
        """The frozen start states (owns shape / scale / goal_drift)."""
        return scenarios.load(self.scenario_bank)

    def replan_counts(self, dt: float) -> tuple[int, int]:
        """``(steps_per_replan, num_replans)`` for one episode at ``dt``."""
        replan_period = 1.0 / self.plan_freq_hz
        if replan_period > self.planner.plan_horizon:
            raise ValueError(
                f"replan_period {replan_period:g}s (plan_freq_hz="
                f"{self.plan_freq_hz:g}) exceeds plan_horizon "
                f"{self.planner.plan_horizon:g}s -- execution would run "
                "past the last optimized rollout and coast on the spline's "
                "clamped extrapolation"
            )
        steps_per_replan = max(int(round(replan_period / dt)), 1)
        num_replans = max(int(round(self.rollout_time / replan_period)), 1)
        return steps_per_replan, num_replans

    def engine_shape(self, arm: str) -> tuple[int, int]:
        """``(num_randomizations, num_samples)`` for one arm.

        A fanned arm spends ``planner.sample_budget`` as ``R=num_domains``
        domains x ``S=sample_budget/R`` control samples (see
        ``bampc.planner.engine_shape`` -- any split is allowed, so S
        may land below the profile's recommended ``num_samples``); every
        other arm is R=1, S=sample_budget. The total ``nworld`` is the
        same either way, so a fanned arm buys domain coverage with control
        samples -- the fairness convention. Fanned arms are the wide-DR ones
        (push/balance) and the peg ``hedge`` cloud.

        Exception: when `hedge_fixed_samples` is set (only `peg_grid_variants`
        does this, for its domains-grid cells), S is pinned at the planner
        profile's own ``num_samples`` instead -- growing R there buys domain
        coverage WITHOUT shrinking control samples, a deliberate departure
        from the fixed-``nworld`` convention above.
        """
        fanned = arm.startswith("wide_dr") or arm.startswith("hedge")
        r = self.num_domains if fanned else 1
        p = self.planner
        if fanned and self.hedge_fixed_samples:
            p = replace(p, sample_budget=None)
        return engine_shape(p, r)

    def make_task(self, model_config: ModelConfig | None = None) -> Task:
        """Build the task (Push-FR3 or Balance-FR3) from bank + cost weights."""
        tp = self.task_params
        bank = self.bank

        def picked(keys: tuple[str, ...]) -> dict[str, Any]:
            return {k: tp[k] for k in keys if k in tp}

        if self.task == "push_fr3":
            return PushFr3(
                sampling_space=tp.get("sampling_space", "task"),
                manipulation_type=tp["manipulation_type"],
                shape=bank.shape,
                scale=bank.scale,
                goal_xy=bank.goal_xy,
                max_lin_vel=tp.get("max_lin_vel", 0.25),
                goal_drift=bank.goal_drift,
                model_config=model_config,
                **picked((
                    "w_pos", "w_orient", "w_attract", "w_align", "w_safety",
                    "safety_thresh", "w_ee_orient", "w_ee_height",
                    "w_arm_home", "terminal_scale",
                )),
            )
        if self.task == "balance_fr3":
            # BalanceFr3 has no manipulation_type (the block is always free),
            # and a smaller weight set than Push-FR3 -- only its keys pass.
            return BalanceFr3(
                sampling_space=tp.get("sampling_space", "task"),
                shape=bank.shape,
                scale=bank.scale,
                goal_xy=bank.goal_xy,
                goal_drift=bank.goal_drift,
                model_config=model_config,
                **picked((
                    "w_pos", "w_orient", "w_ctrl", "w_arm_home",
                    "terminal_scale", "max_tilt_vel",
                )),
            )
        if self.task == "peg_fr3":
            # PegFr3 has no shape library (build_peg_spec) and no goal_xy -- its
            # geometry knob is `clearance` and the goal is the static seated
            # pose. The bank's shape/scale are placeholders it never reads.
            return PegFr3(
                sampling_space=tp.get("sampling_space", "task"),
                clearance=tp.get("clearance", 0.002),
                peg_len=tp.get("peg_len", 0.12),
                max_lin_vel=tp.get("max_lin_vel", 0.15),
                max_ang_vel=tp.get("max_ang_vel", 1.0),
                goal_drift=bank.goal_drift,
                model_config=model_config,
                **picked((
                    "w_pos", "z_scale", "w_orient", "w_force",
                    "w_arm_home", "terminal_scale",
                )),
            )
        if self.task == "flip_fr3":
            # FlipFr3 has no sampling_space/manipulation_type/goal_xy axis --
            # always joint-space, goal pose baked into the model.
            return FlipFr3(
                shape=tp.get("shape", "cracker_box"),
                scale=bank.scale,
                wall=tp.get("wall", True),
                goal_drift=bank.goal_drift,
                model_config=model_config,
                **picked((
                    "w_orient", "w_upright", "w_attract", "w_ctrl",
                    "w_arm_home", "terminal_scale", "orient_tol",
                    "upright_tol",
                )),
            )
        raise ValueError(
            "model_mismatch supports push_fr3 / balance_fr3 / peg_fr3 / "
            f"flip_fr3, got {self.task!r}"
        )

    def make_prediction_task(self) -> Task:
        """Task carrying the prediction physics (what the planner rolls out)."""
        return self.make_task(self.prediction_model_config)

    def make_truth_task(self) -> Task:
        """Task carrying the truth physics (falls back to prediction)."""
        return self.make_task(
            self.truth_model_config
            if self.truth_model_config is not None
            else self.prediction_model_config
        )

    def load_start(self, task: Task, index: int) -> mujoco.MjData:
        """Host ``MjData`` posed at scenario ``index`` of the bank."""
        bank = self.bank
        return scenarios.pose(bank, task, bank[index])

    def tracking_sample(
        self,
        task: Task,
        qpos: np.ndarray,
        qvel: np.ndarray,
        goal_mocap: tuple[np.ndarray, np.ndarray] | None,
        had_contact: bool,
    ) -> dict[str, Any]:
        """Achieved cost and pose error at the current *true* state."""
        if self.task == "peg_fr3":
            return _peg_tracking_sample(task, qpos, qvel, goal_mocap,
                                        had_contact)
        if self.task == "flip_fr3":
            return _flip_tracking_sample(task, qpos, qvel, goal_mocap,
                                         had_contact)
        scratch = _scored_state(task, qpos, qvel, goal_mocap)
        c = task.cost_components(scratch)
        bid = task.mj_model.body("block").id
        gpos, gquat = _goal_world_pose(scratch, task, goal_mocap)
        outcome = task.task_success(scratch)
        return {
            "block_x": float(scratch.xpos[bid][0]),
            "block_y": float(scratch.xpos[bid][1]),
            "block_yaw": _yaw_of(scratch.xquat[bid]),
            "goal_x": float(gpos[0]),
            "goal_y": float(gpos[1]),
            "goal_yaw": _yaw_of(gquat),
            "pos_err": c["pos"] / task.w_pos,
            # w_orient == 0 (e.g. a ball has no orientation) -> NaN, which
            # analysis reads as "no rotation channel" -> position-only metric.
            "orient_err": (
                c["orient"] / task.w_orient if task.w_orient else np.nan
            ),
            **{f"cost_{k}": float(v) for k, v in c.items()},
            # Int, not bool: bool serializes to "True"/"False", which
            # np.genfromtxt reads back as NaN (and NaN.astype(bool) is True),
            # silently marking every replan a contact in analysis.
            "had_contact": int(bool(had_contact)),
            # None (not int(None)) when the task has no success condition --
            # csv.DictWriter writes None as an empty field, which
            # np.genfromtxt reads back as NaN on this otherwise-numeric
            # column, same convention as orient_err above. A raw bool would
            # hit the exact "True"/"False" text trap had_contact avoids.
            "task_success": None if outcome is None else int(outcome),
        }


# --------------------------------------------------------------------- #
# DR-spec + arm builders
# --------------------------------------------------------------------- #


def _block_geom_ids(task: Task) -> list[int]:
    """Geom ids of the block body, off a pristine task."""
    m = task.mj_model
    bid = m.body("block").id
    return [g for g in range(m.ngeom) if m.geom_bodyid[g] == bid]


def _base_mass(task: Task) -> float:
    """The block body's nominal mass (kg), off a pristine task."""
    return float(task.mj_model.body("block").mass[0])


def _base_friction(task: Task) -> float:
    """The block's nominal sliding friction, off a pristine task."""
    return float(task.mj_model.geom_friction[_block_geom_ids(task)[0], 0])


def _base_rolling_friction(task: Task) -> float:
    """The block's nominal rolling friction (geom_friction[:, 2])."""
    return float(task.mj_model.geom_friction[_block_geom_ids(task)[0], 2])


def _base_solimp_dmin(task: Task) -> float:
    """The block's nominal contact impedance at zero penetration."""
    return float(task.mj_model.geom_solimp[_block_geom_ids(task)[0], 0])


# Geoms a block can contact whose friction must move with it: `ground` /
# `plate` are the table/tray surface, `ee` the pusher tip, `pusher_shaft`
# the connecting shaft above it (Flip-FR3 counts both as the pusher for
# contact-force accounting, and the block does contact the shaft, not just
# the tip), `wall_wall` Flip-FR3's
# optional tipping wall. None collide with the decorative, non-colliding
# "floor" plane every scene also defines separately.
_COUNTER_GEOM_CANDIDATES = (
    "ground", "ee", "pusher_shaft", "wall_wall", "plate",
)


def _counter_geom_names(task: Task) -> tuple[str, ...]:
    """Geoms the block can contact whose friction must move with it.

    MuJoCo combines two contacting geoms' friction via elementwise max (no
    geom in this repo sets `priority`), so writing friction to only the
    block silently clamps any leg below the counter-geom's fixed value to a
    no-op. See ``scripts/curling/range_check.py`` for the same rule applied
    to curling's puck/lane pair.
    """
    names = []
    for name in _COUNTER_GEOM_CANDIDATES:
        try:
            task.mj_model.geom(name)
        except KeyError:
            continue
        names.append(name)
    return tuple(names)


def _body_val(param: str, value: float, task: Task) -> dict:
    """A ``{"body": {"block": {param: value}}}`` spec fragment.

    For ``friction``/``rolling_friction``, mirrors ``value`` onto every
    geom the block can contact (see ``_counter_geom_names``) so the swept
    value is the actual effective contact friction, not silently clamped
    by an un-randomized counter-geom.
    """
    spec: dict = {"body": {"block": {param: value}}}
    if param in ("friction", "rolling_friction"):
        counters = _counter_geom_names(task)
        if counters:
            spec["geom"] = {name: {param: value} for name in counters}
    return spec


def _solimp_dmin_val(param: str, value: float, task: Task) -> dict:
    """Like ``_body_val`` but clamped to the physical impedance range.

    ``solimp`` dmin/dmax live in ``[0, 1)``; base 0.8 x 2.0 would overshoot, so
    the harder arm saturates near-rigid instead of raising.
    """
    hi = 1.0 - 1e-6
    return _body_val(param, float(np.clip(value, mujoco.mjMINVAL, hi)), task)


_WALL_GEOMS = ("socket_xn", "socket_xp", "socket_yn", "socket_yp")


def _base_wall_friction(task: Task) -> float:
    """The socket walls' nominal sliding friction (shared by all four)."""
    gid = task.mj_model.geom(_WALL_GEOMS[0]).id
    return float(task.mj_model.geom_friction[gid, 0])


def _base_wall_solimp_dmin(task: Task) -> float:
    """The socket walls' nominal contact impedance at zero penetration."""
    gid = task.mj_model.geom(_WALL_GEOMS[0]).id
    return float(task.mj_model.geom_solimp[gid, 0])


def _wall_val(param: str, value: float) -> dict:
    """A spec fragment perturbing all four socket-wall geoms together.

    The walls share a body with the socket floor (``socket_base``), which
    must stay at truth -- so this targets the four wall geoms by name
    rather than the body. For ``friction``, also mirrors ``value`` onto
    the ``peg`` geom -- otherwise the peg's fixed nominal friction clamps
    any wall-friction leg below it to a no-op under MuJoCo's elementwise-
    max contact combination (see ``_counter_geom_names``).
    """
    spec = {"geom": {name: {param: value} for name in _WALL_GEOMS}}
    if param == "friction":
        spec["geom"]["peg"] = {"friction": value}
    return spec


def _wall_solimp_dmin_val(param: str, value: float) -> dict:
    """Like ``_wall_val`` but clamped to the physical impedance range."""
    hi = 1.0 - 1e-6
    return _wall_val(param, float(np.clip(value, mujoco.mjMINVAL, hi)))


# Each swept parameter: (base-value reader, spec builder from an ABSOLUTE
# value and the pristine task). `impratio` has nominal 1.0 and lives on the
# global opt block; the `wall_*` pair targets Peg-FR3's socket walls instead
# of a `block` body; the rest scale a base value read off a pristine block
# geom/body.
_PARAM_SPEC: dict = {
    "mass": (_base_mass, lambda v, t: _body_val("mass", v, t)),
    "friction": (_base_friction, lambda v, t: _body_val("friction", v, t)),
    "rolling_friction": (
        _base_rolling_friction,
        lambda v, t: _body_val("rolling_friction", v, t),
    ),
    "solimp_dmin": (
        _base_solimp_dmin, lambda v, t: _solimp_dmin_val("solimp_dmin", v, t)
    ),
    "impratio": (
        lambda _t: 1.0,
        lambda v, _t: {"opt": {"__all__": {"impratio": v}}},
    ),
    "wall_friction": (
        _base_wall_friction, lambda v, _t: _wall_val("friction", v)
    ),
    "wall_solimp_dmin": (
        _base_wall_solimp_dmin,
        lambda v, _t: _wall_solimp_dmin_val("solimp_dmin", v),
    ),
}


def _merge_specs(specs: list[dict]) -> dict:
    """Deep-merge DR-spec fragments (body/opt) into one nested dict."""
    out: dict[str, Any] = {}
    for spec in specs:
        for kind, entities in spec.items():
            dst = out.setdefault(kind, {})
            for entity, params in entities.items():
                dst.setdefault(entity, {}).update(params)
    return out


def biased_spec(
    params: tuple[str, ...], mult: float | tuple[float, ...], task: Task
) -> dict:
    """A DR spec for one or more parameters offset from truth.

    ``mult`` is either one multiplier broadcast to every param (a
    single-parameter or ``all_<mult>`` arm) or a tuple parallel to
    ``params`` giving each its own (a ``"mixed"`` arm, e.g.
    ``mass_2.0+friction_0.5``).

    ``task`` must be pristine (base values are read off it); resolving against
    a ``CpuTruth`` would compound seed over seed.
    """
    mults = mult if isinstance(mult, tuple) else (mult,) * len(params)
    frags = []
    for param, m in zip(params, mults, strict=True):
        base, build = _PARAM_SPEC[param]
        frags.append(build(base(task) * m, task))
    return _merge_specs(frags)


def wide_spec(
    params: tuple[str, ...], lo: float, hi: float, task: Task
) -> dict:
    """A uniform-DR spec: ``(base*lo, base*hi)`` per named parameter."""
    frags = []
    for param in params:
        base, build = _PARAM_SPEC[param]
        b = base(task)
        # A range fragment reuses the builder's structure with a (lo, hi) tuple;
        # solimp clamps each end.
        lo_spec, hi_spec = build(b * lo, task), build(b * hi, task)
        frags.append(_range_of(lo_spec, hi_spec))
    return _merge_specs(frags)


def _range_of(lo_spec: dict, hi_spec: dict) -> dict:
    """Combine two scalar fragments into one with ``(lo, hi)`` tuple values."""
    out: dict[str, Any] = {}
    for kind, entities in lo_spec.items():
        for entity, params in entities.items():
            for param, lo_v in params.items():
                hi_v = hi_spec[kind][entity][param]
                out.setdefault(kind, {}).setdefault(entity, {})[param] = (
                    lo_v, hi_v
                )
    return out


def _parse_mixed(
    name: str, bias_params: tuple[str, ...]
) -> tuple[tuple[str, ...], tuple[float, ...]]:
    """Split a ``+``-joined mixed arm name into parallel ``(params, mults)``.

    Each segment is the same ``<param>_<mult>`` form a single-parameter arm
    uses (e.g. ``mass_2.0+friction_0.5`` -> ``mass``, ``friction``).
    """
    params, mults = [], []
    for seg in name.split("+"):
        for param in bias_params:
            prefix = f"{param}_"
            if seg.startswith(prefix):
                params.append(param)
                mults.append(float(seg[len(prefix):]))
                break
        else:
            raise ValueError(
                f"unrecognized segment {seg!r} in mixed arm {name!r}"
            )
    if len(set(params)) != len(params):
        raise ValueError(f"duplicate parameter in mixed arm {name!r}")
    return tuple(params), tuple(mults)


def parse_arm(
    name: str, bias_params: tuple[str, ...] = BIAS_PARAMS
) -> tuple[str, tuple[str, ...], float | tuple[float, ...] | None]:
    """Classify an arm name into ``(kind, params, mult)``.

    ``kind`` is ``"nominal"`` / ``"biased"`` / ``"wide"`` / ``"mixed"``. A
    biased arm carries its param(s) and one multiplier shared by all of them;
    a wide arm has ``mult=None`` and its randomized set. ``all_<mult>`` and
    the bare ``wide_dr`` both expand to ``bias_params``. A mixed arm is
    several ``<param>_<mult>`` pieces joined by ``+`` (e.g.
    ``mass_2.0+friction_0.5``) -- ``mult`` is then a tuple parallel to
    ``params``, one multiplier per parameter, not a single shared value.
    """
    if name == "nominal":
        return "nominal", (), None
    if name == "wide_dr":
        return "wide", tuple(bias_params), None
    if name.startswith("wide_dr_"):
        param = name[len("wide_dr_"):]
        if param not in bias_params:
            raise ValueError(f"unknown wide-DR parameter in arm {name!r}")
        return "wide", (param,), None
    if name.startswith("all_"):
        return "biased", tuple(bias_params), float(name[len("all_"):])
    if "+" in name:
        params, mults = _parse_mixed(name, bias_params)
        return "mixed", params, mults
    for param in bias_params:
        prefix = f"{param}_"
        if name.startswith(prefix):
            return "biased", (param,), float(name[len(prefix):])
    raise ValueError(f"unrecognized arm name: {name!r}")


def _peg_arm_spec(name: str, cfg: RunConfig, task: Task) -> dict:
    """DR spec for a Peg-FR3 mount-uncertainty arm (point / oracle / hedge).

    The grasp offset is the peg's mount pose in its parent (wrist) frame -- a
    real model field on a kinematic child, so MJWarp reads it per world (a free
    body would ignore it). Body pos writes are ABSOLUTE, so each value offsets
    the nominal mount read off a pristine task (``peg_body.pos``, nominally 0).

    * ``point`` -- nominal mount: the certainty-equivalent baseline that
      believes the peg is where commanded (confidently wrong under an offset).
    * ``oracle`` -- the mount at ``true_mount_offset``: matches the truth.
    * ``hedge`` / ``hedge_wide`` -- a uniform cloud +/-``mount_noise`` (pos) and
      +/-``mount_noise_rot`` (tilt) around the nominal mount, R=num_domains.
    """
    base = task.mj_model.body("peg_body").pos
    bx, by = float(base[0]), float(base[1])
    if name == "point":
        return {}
    if name == "oracle":
        spec = {"pos_x": bx + cfg.true_mount_offset}
        if cfg.true_mount_tilt != 0.0:
            spec[f"rot_{cfg.true_mount_tilt_axis}"] = cfg.true_mount_tilt
        return {"body": {"peg_body": spec}}
    if name.startswith("hedge"):
        n = cfg.mount_noise
        nr = (
            cfg.mount_noise_rot
            if cfg.mount_noise_rot is not None
            else n / float(task.geometry["peg_len"])
        )
        if name == "hedge_wide":
            n, nr = 2.0 * n, 2.0 * nr
        elif name != "hedge":
            raise ValueError(f"unrecognized peg arm name: {name!r}")
        return {"body": {"peg_body": {
            "pos_x": (bx - n, bx + n),
            "pos_y": (by - n, by + n),
            "rot_x": (-nr, nr),
            "rot_y": (-nr, nr),
        }}}
    raise ValueError(f"unrecognized peg arm name: {name!r}")


_PEG_MOUNT_ARMS = frozenset({"point", "oracle", "hedge", "hedge_wide"})


def arm_spec(name: str, cfg: RunConfig, task: Task) -> dict:
    """The DR spec for one arm, built against a pristine ``task``."""
    if cfg.task == "peg_fr3" and name in _PEG_MOUNT_ARMS:
        return _peg_arm_spec(name, cfg, task)
    kind, params, mult = parse_arm(name, cfg.bias_params)
    if kind == "nominal":
        return {}
    if kind in ("biased", "mixed"):
        return biased_spec(params, mult, task)
    lo, hi = cfg.wide_dr_range
    return wide_spec(params, lo, hi, task)


def truth_grid_spec(
    arm_name: str,
    param: str,
    task: Task,
    true_mult: float,
    bias_params: tuple[str, ...],
) -> tuple[dict, dict]:
    """``(prediction_spec, truth_spec)`` for one truth-friction-grid cell.

    Truth is pinned at ``base(param) * true_mult`` (the REAL value for this
    cell); the arm's own multiplier (parsed the ordinary way) then scales
    PREDICTION relative to THAT value, not the global nominal -- so
    ``nominal`` here means "prediction knows this cell's true value", and a
    biased arm reuses the same relative bias factors as an ordinary
    ``friction_<mult>`` sweep, just anchored to a shifted truth instead of
    the pristine model. ``task`` must be pristine (mirrors ``biased_spec``).
    """
    base, build = _PARAM_SPEC[param]
    true_value = base(task) * true_mult
    truth_spec = build(true_value, task)
    kind, params, mult = parse_arm(arm_name, bias_params)
    if kind == "nominal":
        return truth_spec, truth_spec
    if kind == "biased" and params == (param,):
        return build(true_value * mult, task), truth_spec
    raise ValueError(
        f"truth-friction grid only supports 'nominal' or '{param}_<mult>' "
        f"arms, got {arm_name!r}"
    )


# --------------------------------------------------------------------- #
# Scoring helpers (achieved state -> pose error, shared with the episode)
# --------------------------------------------------------------------- #


def _apply_goal_mocap(
    md: mujoco.MjData, goal_mocap: tuple[np.ndarray, np.ndarray] | None
) -> None:
    """Write drifting-goal mocap arrays into ``md`` before scoring."""
    if goal_mocap is None:
        return
    md.mocap_pos[:] = goal_mocap[0]
    md.mocap_quat[:] = goal_mocap[1]


def _yaw_of(quat: np.ndarray) -> float:
    """Yaw (rotation about +z) of a ``[w, x, y, z]`` quaternion."""
    w, x, y, z = (float(v) for v in quat)
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def _goal_world_pose(
    md: mujoco.MjData,
    task: Task,
    goal_mocap: tuple[np.ndarray, np.ndarray] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """World ``(pos, quat)`` of the goal mocap in ``md`` (drifted if any)."""
    mid = task.goal_mocap_id
    if goal_mocap is not None:
        return goal_mocap[0][mid], goal_mocap[1][mid]
    return md.mocap_pos[mid].copy(), md.mocap_quat[mid].copy()


def _scored_state(
    task: Task,
    qpos: np.ndarray,
    qvel: np.ndarray,
    goal_mocap: tuple[np.ndarray, np.ndarray] | None,
) -> mujoco.MjData:
    """A forwarded scratch ``MjData`` at ``(qpos, qvel)`` with the goal set."""
    md = mujoco.MjData(task.mj_model)
    md.qpos[: len(qpos)] = qpos
    md.qvel[: len(qvel)] = qvel
    _apply_goal_mocap(md, goal_mocap)
    mujoco.mj_forward(task.mj_model, md)
    return md


def _peg_tracking_sample(
    task: Task,
    qpos: np.ndarray,
    qvel: np.ndarray,
    goal_mocap: tuple[np.ndarray, np.ndarray] | None,
    had_contact: bool,
) -> dict[str, Any]:
    """Peg->goal pose error, peg-wall force and cost at the true state.

    Peg-FR3 has no ``block`` body and its ``cost_components`` splits position
    into lateral/axial, so it takes its own sampler. ``wall_force`` is the
    failure signal (a confidently-wrong mount jams the peg on a wall);
    ``pos_err`` is the full peg->goal distance in the goal frame.
    """
    scratch = _scored_state(task, qpos, qvel, goal_mocap)
    c = task.cost_components(scratch)
    sd = scratch.sensordata
    a, ao = task._adr_pos, task._adr_orient
    e = sd[a : a + 3]
    q = sd[ao : ao + 4]
    pos_err = float(np.linalg.norm(e))
    orient_err = float(
        2.0 * np.arctan2(float(np.linalg.norm(q[1:4])), abs(float(q[0])))
    )
    tip = scratch.site_xpos[task.tip_site_id]
    outcome = task.task_success(scratch)
    return {
        "peg_x": float(tip[0]),
        "peg_y": float(tip[1]),
        "peg_z": float(tip[2]),
        "pos_err": pos_err,
        "orient_err": orient_err,
        "wall_force": float(task.wall_force(scratch)),
        **{f"cost_{k}": float(v) for k, v in c.items()},
        "had_contact": int(bool(had_contact)),
        "task_success": None if outcome is None else int(outcome),
    }


def _flip_tracking_sample(
    task: Task,
    qpos: np.ndarray,
    qvel: np.ndarray,
    goal_mocap: tuple[np.ndarray, np.ndarray] | None,
    had_contact: bool,
) -> dict[str, Any]:
    """Block->goal orientation error and cost at the true state.

    Flip-FR3's cost is orientation-only (no ``w_pos``/"pos" cost term -- the
    generic ``tracking_sample`` path would KeyError on it), so ``pos_err``
    is always ``0.0``: analysis' ``pose_error = pos_err + rot_scale *
    orient_err`` then collapses to the orientation term alone, which is the
    correct metric for a task with no position goal.
    """
    scratch = _scored_state(task, qpos, qvel, goal_mocap)
    c = task.cost_components(scratch)
    bid = task.mj_model.body("block").id
    gpos, gquat = _goal_world_pose(scratch, task, goal_mocap)
    outcome = task.task_success(scratch)
    return {
        "block_x": float(scratch.xpos[bid][0]),
        "block_y": float(scratch.xpos[bid][1]),
        "block_z": float(scratch.xpos[bid][2]),
        "goal_x": float(gpos[0]),
        "goal_y": float(gpos[1]),
        "goal_z": float(gpos[2]),
        "pos_err": 0.0,
        "orient_err": (
            c["orient"] / task.w_orient if task.w_orient else np.nan
        ),
        **{f"cost_{k}": float(v) for k, v in c.items()},
        "had_contact": int(bool(had_contact)),
        "task_success": None if outcome is None else int(outcome),
    }


# --------------------------------------------------------------------- #
# Loader
# --------------------------------------------------------------------- #

def _parse(path: Path) -> dict:
    """Parse a YAML config file."""
    return yaml.safe_load(path.read_text())


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
    allowed = {f.name for f in dataclasses.fields(PlannerConfig)}
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f"unknown planner fields: {sorted(unknown)}")
    return PlannerConfig(**data)


def _load_numerics(role_raw: dict) -> ModelConfig | None:
    """Resolve a role's ``numerics`` profile name (or ``None``)."""
    name = role_raw.get("numerics")
    return None if name is None else numerics.load(name)


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


def load_run_config(path: str | Path) -> RunConfig:
    """Load and resolve a model_mismatch config file."""
    path = Path(path)
    raw = _parse(path)
    prediction_raw = raw.get("prediction") or {}
    truth_raw = raw.get("truth") or {}
    sampling_space = (raw.get("task_params") or {}).get("sampling_space")
    bias_target = raw.get("bias_target", "prediction")
    if bias_target not in ("prediction", "truth", "truth_grid"):
        raise ValueError(
            "bias_target must be 'prediction', 'truth' or 'truth_grid', got "
            f"{bias_target!r}"
        )
    return RunConfig(
        name=raw["name"],
        task=raw["task"],
        scenario_bank=raw["scenario_bank"],
        truth_backend=truth_raw.get("backend", "warp"),
        planner=_load_planner(raw["planner"], sampling_space),
        num_seeds=int(raw["num_seeds"]),
        repeats=int(raw["repeats"]),
        rollout_time=float(raw["rollout_time"]),
        plan_freq_hz=float(raw["plan_freq_hz"]),
        num_domains=int(raw["num_domains"]),
        # Peg configs randomize a mount pose, not a multiplier bracket, so
        # wide_dr_range is optional there (defaults, unused by the peg arms).
        wide_dr_range=tuple(raw.get("wide_dr_range", (0.25, 2.0))),
        task_params=_resolve_task_params(raw.get("task_params", {})),
        arms=tuple(raw["arms"]),
        algos=tuple(raw["algos"]),
        bias_params=tuple(raw.get("bias_params", BIAS_PARAMS)),
        plot_bias_params=(
            None if raw.get("plot_bias_params") is None
            else tuple(raw["plot_bias_params"])
        ),
        bias_target=bias_target,
        true_bias_grid=(
            None if raw.get("true_bias_grid") is None
            else tuple(float(v) for v in raw["true_bias_grid"])
        ),
        true_mount_offset=float(raw.get("true_mount_offset", 0.0)),
        mount_noise=float(raw.get("mount_noise", 0.0)),
        mount_noise_rot=(
            None if raw.get("mount_noise_rot") is None
            else float(raw["mount_noise_rot"])
        ),
        true_mount_tilt=float(raw.get("true_mount_tilt", 0.0)),
        true_mount_tilt_axis=raw.get("true_mount_tilt_axis", "y"),
        truth_mode=raw.get("truth_mode", "translation"),
        clearance_grid=(
            None if raw.get("clearance_grid") is None
            else tuple(float(v) for v in raw["clearance_grid"])
        ),
        true_offset_grid=(
            None if raw.get("true_offset_grid") is None
            else tuple(float(v) for v in raw["true_offset_grid"])
        ),
        true_tilt_grid=(
            None if raw.get("true_tilt_grid") is None
            else tuple(float(v) for v in raw["true_tilt_grid"])
        ),
        noise_ratio_grid=(
            None if raw.get("noise_ratio_grid") is None
            else tuple(float(v) for v in raw["noise_ratio_grid"])
        ),
        hedge_spread_grid=(
            None if raw.get("hedge_spread_grid") is None
            else tuple(float(v) for v in raw["hedge_spread_grid"])
        ),
        hedge_tilt_spread_grid=(
            None if raw.get("hedge_tilt_spread_grid") is None
            else tuple(float(v) for v in raw["hedge_tilt_spread_grid"])
        ),
        hedge_domains_grid=(
            None if raw.get("hedge_domains_grid") is None
            else tuple(int(v) for v in raw["hedge_domains_grid"])
        ),
        allocation_probe_domains=(
            None if raw.get("allocation_probe_domains") is None
            else tuple(int(v) for v in raw["allocation_probe_domains"])
        ),
        risk_grid=(
            None if raw.get("risk_grid") is None
            else tuple(
                (name, None if alpha is None else float(alpha))
                for name, alpha in raw["risk_grid"]
            )
        ),
        prediction_model_config=_load_numerics(prediction_raw),
        truth_model_config=_load_numerics(truth_raw),
    )


def run_variants(cfg: RunConfig) -> list[tuple[list[str], RunConfig]]:
    """Fan the config into one ``(subdir_parts, cfg)`` per (algo, arm)."""
    out = []
    for algo in cfg.algos:
        for arm in cfg.arms:
            variant = replace(
                cfg,
                algo=algo,
                arm=arm,
                planner=replace(cfg.planner, algo=algo),
            )
            out.append(([algo, arm], variant))
    return out


def truth_grid_variants(
    cfg: RunConfig,
) -> list[tuple[list[str], RunConfig]]:
    """Fan a truth-grid config over ``true_bias_grid x arms``.

    One leg per ``true_bias_grid`` value: truth's ``bias_params[0]`` (a
    single-element tuple -- this mode sweeps exactly one parameter, e.g.
    push_fr3's friction or balance_fr3's rolling_friction) is pinned at
    ``base * true_mult`` for that leg, and ``cfg.arms`` (the ordinary
    ``nominal`` / ``<param>_<mult>`` grammar) reruns relative to THAT value
    via ``truth_grid_spec`` -- see ``Stack.__init__``.
    """
    if len(cfg.bias_params) != 1:
        raise ValueError(
            "a truth grid sweeps exactly one bias_params entry, got "
            f"{cfg.bias_params!r}"
        )
    out = []
    for algo in cfg.algos:
        for true_mult in cfg.true_bias_grid:
            for arm in cfg.arms:
                variant = replace(
                    cfg, algo=algo, arm=arm, true_bias_mult=true_mult,
                    planner=replace(cfg.planner, algo=algo),
                )
                out.append(([algo, f"truth_{true_mult:g}", arm], variant))
    return out


def peg_grid_variants(  # noqa: PLR0912
    cfg: RunConfig,
) -> list[tuple[list[str], RunConfig]]:
    """Fan a Peg-FR3 misalignment config over ``clearance x true-error x arm``.

    ``point``/``oracle`` run once per ``(clearance, true-error)`` cell at
    ``R=1, S=sample_budget`` -- the "full sampling budget" reference
    (``oracle``'s spec depends on the true error, ``point``'s doesn't, but
    the TRUTH each is tested against does, so both still need one run per
    cell). The true-error axis itself is picked by ``cfg.truth_mode``:

    * ``"translation"`` (default) -- ``true_offset_grid``, true tilt pinned
      at 0. Today's original behaviour, unchanged.
    * ``"tilt"`` -- ``true_tilt_grid``, true offset pinned at 0.
    * ``"combined"`` -- ``true_offset_grid`` x ``true_tilt_grid`` zipped by
      index (matched severity, NOT a cross-product): a real grasp error
      doesn't hit every offset/tilt combination independently, so this
      tests a few realistic (offset, tilt) severity levels together.

    ``hedge`` fans three independent sub-arm families per cell, all under
    ``hedge_domains_grid`` (R fixed per value; S held at the planner
    profile's own ``num_samples`` via ``hedge_fixed_samples`` -- growing R
    buys domain coverage without shrinking control samples):

    * "matched" (``noise_ratio_grid``) -- ``mount_noise = ratio *
      true_offset``, ``mount_noise_rot = ratio * true_tilt`` (or the
      ``n/peg_len`` fallback when true_tilt is 0) -- calibrated to
      whichever true axis this cell actually has.
    * "deploy" (``hedge_spread_grid``) -- a single ABSOLUTE translation
      half-width, the number a deployment would actually configure without
      knowing the true offset; rotation auto-derives from it.
    * "tiltdeploy" (``hedge_tilt_spread_grid``) -- an absolute ROTATION
      half-width set directly (``mount_noise=0``), decoupled from
      translation -- isolates whether tilt-hedging works on its own rather
      than riding on "deploy"'s incidental n/peg_len coverage.

    ``allocation_probe_domains``, if set, additionally emits one small,
    separate fixed-``nworld`` (``hedge_fixed_samples=False``) probe at the
    single middle clearance and middle true-error cell -- see the module's
    plan notes; it answers "does R vs S trade off at fixed total compute,"
    which every cell above (fixed S, growing nworld) cannot.
    """
    out = []
    for algo in cfg.algos:
        for clearance in cfg.clearance_grid:
            tp = {**cfg.task_params, "clearance": clearance}
            tag_c = f"clearance_{clearance:g}"

            if cfg.truth_mode == "translation":
                truth_cells = [
                    (f"offset_{o:g}", o, 0.0) for o in cfg.true_offset_grid
                ]
            elif cfg.truth_mode == "tilt":
                truth_cells = [
                    (f"tilt_{t:g}", 0.0, t) for t in cfg.true_tilt_grid
                ]
            elif cfg.truth_mode == "combined":
                if len(cfg.true_offset_grid) != len(cfg.true_tilt_grid):
                    raise ValueError(
                        "combined truth_mode needs true_offset_grid and "
                        "true_tilt_grid of equal length (matched severity)"
                    )
                truth_cells = [
                    (f"combined_{i}", o, t)
                    for i, (o, t) in enumerate(
                        zip(cfg.true_offset_grid, cfg.true_tilt_grid)
                    )
                ]
            else:
                raise ValueError(f"unknown truth_mode: {cfg.truth_mode!r}")

            for tag_t, true_offset, true_tilt in truth_cells:
                for arm in ("point", "oracle"):
                    if arm not in cfg.arms:
                        continue
                    variant = replace(
                        cfg, algo=algo, arm=arm, task_params=tp,
                        true_mount_offset=true_offset,
                        true_mount_tilt=true_tilt,
                        planner=replace(cfg.planner, algo=algo),
                    )
                    out.append(([algo, tag_c, tag_t, arm], variant))

                if "hedge" not in cfg.arms:
                    continue

                for ratio in (cfg.noise_ratio_grid or ()):
                    for domains in cfg.hedge_domains_grid:
                        variant = replace(
                            cfg, algo=algo, arm="hedge", task_params=tp,
                            true_mount_offset=true_offset,
                            true_mount_tilt=true_tilt,
                            mount_noise=ratio * true_offset,
                            mount_noise_rot=(
                                None if true_tilt == 0.0
                                else ratio * true_tilt
                            ),
                            num_domains=domains, hedge_fixed_samples=True,
                            planner=replace(cfg.planner, algo=algo),
                        )
                        out.append((
                            [algo, tag_c, tag_t, f"ratio_{ratio:g}",
                             f"domains_{domains}"],
                            variant,
                        ))

                for spread in (cfg.hedge_spread_grid or ()):
                    for domains in cfg.hedge_domains_grid:
                        variant = replace(
                            cfg, algo=algo, arm="hedge", task_params=tp,
                            true_mount_offset=true_offset,
                            true_mount_tilt=true_tilt,
                            mount_noise=spread, mount_noise_rot=None,
                            num_domains=domains, hedge_fixed_samples=True,
                            planner=replace(cfg.planner, algo=algo),
                        )
                        out.append((
                            [algo, tag_c, tag_t, f"spread_{spread:g}",
                             f"domains_{domains}"],
                            variant,
                        ))

                for trot in (cfg.hedge_tilt_spread_grid or ()):
                    for domains in cfg.hedge_domains_grid:
                        variant = replace(
                            cfg, algo=algo, arm="hedge", task_params=tp,
                            true_mount_offset=true_offset,
                            true_mount_tilt=true_tilt,
                            mount_noise=0.0, mount_noise_rot=trot,
                            num_domains=domains, hedge_fixed_samples=True,
                            planner=replace(cfg.planner, algo=algo),
                        )
                        out.append((
                            [algo, tag_c, tag_t, f"tiltspread_{trot:g}",
                             f"domains_{domains}"],
                            variant,
                        ))

            if cfg.allocation_probe_domains and "hedge" in cfg.arms:
                mid_clearance = cfg.clearance_grid[
                    len(cfg.clearance_grid) // 2
                ]
                if clearance != mid_clearance:
                    continue
                mid_tag, mid_offset, mid_tilt = truth_cells[
                    len(truth_cells) // 2
                ]
                for domains in cfg.allocation_probe_domains:
                    variant = replace(
                        cfg, algo=algo, arm="hedge", task_params=tp,
                        true_mount_offset=mid_offset,
                        true_mount_tilt=mid_tilt,
                        mount_noise=1.0 * mid_offset,
                        mount_noise_rot=(
                            None if mid_tilt == 0.0 else 1.0 * mid_tilt
                        ),
                        num_domains=domains, hedge_fixed_samples=False,
                        planner=replace(cfg.planner, algo=algo),
                    )
                    out.append((
                        [algo, "allocation_probe", mid_tag,
                         f"domains_{domains}"],
                        variant,
                    ))
    return out


def smoke(cfg: RunConfig) -> RunConfig:
    """Shape check: 1 seed, 1 repeat, 2 s rollouts."""
    return replace(
        cfg, name=cfg.name + "-smoke", num_seeds=1, repeats=1, rollout_time=2.0
    )


def quick(cfg: RunConfig, seed: int = 0) -> RunConfig:
    """Tuning run: 1 seed x 1 repeat at FULL episode length."""
    return replace(
        cfg,
        name=cfg.name + "-quick",
        num_seeds=1,
        repeats=1,
        seed_offset=seed,
    )


MODES = {"smoke": smoke, "quick": quick}
