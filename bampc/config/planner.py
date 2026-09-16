"""Named planner-hyperparameter profiles, loaded from ``configs/planner/``.

Mirrors ``numerics.py``'s name-a-profile pattern, but returns a plain dict
rather than a :class:`PlannerConfig`: a profile is always partial -- ``algo``
in particular is a run choice supplied by the caller (a CLI flag, an
experiment config), never the profile itself.

Keys in a profile YAML are ``PlannerConfig`` field names (minus ``algo``), so
adding a field to the dataclass admits it here with no change. A profile may
also carry the axes a task actually has:

* ``by_sampling:`` (``task``/``joint`` keys) -- exploration noise that
  genuinely differs by sampling space. A ``task``/``joint`` branch may itself
  carry a nested ``by_manipulation:`` (``free``/``joint`` keys), for the one
  task (push_fr3) with both axes.
* ``by_algo:`` (algo-name keys, plus a ``default``) -- tuning that depends on
  which search algorithm is active (only ``push`` today).

:func:`load` resolves both axes so every caller -- an example script, a
sweep config -- gets one flat, ready-to-spread dict back; none of them
re-implement this merge themselves.
"""

from __future__ import annotations

import dataclasses
from functools import cache

from bampc import PLANNER_DIR
from bampc._yaml import list_names, load_yaml
from bampc.planner.config import PlannerConfig

_FIELDS = {f.name for f in dataclasses.fields(PlannerConfig)} - {"algo"}


def list_profiles() -> list[str]:
    """Every profile name available under ``configs/planner/``."""
    return list_names(PLANNER_DIR)


@cache
def _load_raw(name: str) -> dict:
    """The profile's fields exactly as written, axes included, uncached-safe.

    Cached on the raw parse only -- :func:`load` always returns a fresh dict
    built from this, so a caller mutating its result (a common pattern here:
    ``PLANNER_KW["risk"] = args.risk``) can never corrupt the cache.
    """
    return load_yaml(PLANNER_DIR, name, "planner profile") or {}


def load(
    name: str,
    *,
    sampling: str | None = None,
    manipulation: str | None = None,
    algo: str | None = None,
) -> dict:
    """Read one named profile, resolved for one caller.

    Args:
        name: Profile stem, e.g. ``"push_fr3"``.
        sampling: ``"task"``/``"joint"``, for a profile with a
            ``by_sampling`` block; ``None`` to leave it unresolved (the
            axis's keys are dropped either way -- they are never valid
            ``PlannerConfig`` fields themselves).
        manipulation: ``"free"``/``"joint"``, for push_fr3's nested
            ``by_manipulation``; ``None`` otherwise.
        algo: The active search algorithm, for a profile with a ``by_algo``
            block (only ``push`` today); ``None`` otherwise.

    Returns:
        A fresh, flat dict (never includes ``algo``) -- safe for the caller
        to mutate.

    Raises:
        ValueError: The profile does not exist, or names an unknown field.
    """
    data = dict(_load_raw(name))
    by_sampling = data.pop("by_sampling", None)
    by_algo = data.pop("by_algo", None)
    if by_sampling and sampling is not None:
        branch = dict(by_sampling.get(sampling, {}))
        by_manipulation = branch.pop("by_manipulation", None)
        data.update(branch)
        if by_manipulation and manipulation is not None:
            data.update(by_manipulation.get(manipulation, {}))
    if by_algo and algo is not None:
        data.update(by_algo.get(algo, by_algo.get("default", {})))
    unknown = set(data) - _FIELDS
    if unknown:
        raise ValueError(f"{name}: unknown fields {sorted(unknown)}")
    return data
