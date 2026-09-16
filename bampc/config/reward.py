"""Named reward-weight profiles, loaded from ``configs/reward/``.

Mirrors ``planner.py``'s name-a-profile pattern, axis resolution included
(``by_sampling``, with push_fr3's nested ``by_manipulation``). Unlike
``numerics.py``/``planner.py``, there is no dataclass to validate keys
against -- reward kwargs are whatever each task constructor accepts, and
genuinely vary per task; an unknown one already fails loudly as a
``TypeError`` at the task constructor, so a second whitelist here would only
duplicate that check.
"""

from __future__ import annotations

from functools import cache

from bampc import REWARD_DIR
from bampc._yaml import list_names, load_yaml


def list_profiles() -> list[str]:
    """Every profile name available under ``configs/reward/``."""
    return list_names(REWARD_DIR)


@cache
def _load_raw(name: str) -> dict:
    """The profile's fields exactly as written, axes included, uncached-safe.

    Cached on the raw parse only -- :func:`load` always returns a fresh dict
    built from this, so a caller mutating its result can never corrupt the
    cache.
    """
    return load_yaml(REWARD_DIR, name, "reward profile") or {}


def load(
    name: str, *, sampling: str | None = None, manipulation: str | None = None
) -> dict:
    """Read one named profile, resolved for one caller.

    Args:
        name: Profile stem, e.g. ``"push_fr3"``.
        sampling: ``"task"``/``"joint"``, for a profile with a
            ``by_sampling`` block; ``None`` to leave it unresolved.
        manipulation: ``"free"``/``"joint"``, for push_fr3's nested
            ``by_manipulation``; ``None`` otherwise.

    Returns:
        A fresh, flat dict -- safe for the caller to mutate.

    Raises:
        ValueError: The profile does not exist.
    """
    data = dict(_load_raw(name))
    by_sampling = data.pop("by_sampling", None)
    if by_sampling and sampling is not None:
        branch = dict(by_sampling.get(sampling, {}))
        by_manipulation = branch.pop("by_manipulation", None)
        data.update(branch)
        if by_manipulation and manipulation is not None:
            data.update(by_manipulation.get(manipulation, {}))
    return data
