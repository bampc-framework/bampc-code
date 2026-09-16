"""Named solver/integrator profiles, loaded from ``configs/numerics/*.yaml``.

Numerics are declared by whoever runs the physics (an example, a script, a
sweep config), not by the Task. Rather than each caller retyping a
:class:`ModelConfig`, they name a profile and it is read from one file --
the same pattern the shape library uses for ``models/shapes/``.

Keys in a profile YAML are exactly ``ModelConfig`` field names, so adding a
field to the dataclass admits it here with no change.
"""

from __future__ import annotations

import dataclasses
from functools import cache

from bampc import NUMERICS_DIR
from bampc._yaml import list_names, load_yaml
from bampc.task.base import ModelConfig

_FIELDS = {f.name for f in dataclasses.fields(ModelConfig)}


def list_profiles() -> list[str]:
    """Every profile name available under ``configs/numerics/``."""
    return list_names(NUMERICS_DIR)


@cache
def load(name: str) -> ModelConfig:
    """Read one named profile into a :class:`ModelConfig`.

    Args:
        name: Profile stem, e.g. ``"push"`` or ``"fr3_joint"``.

    Returns:
        The profile's config. ``ModelConfig`` is frozen, so the cached
        instance is safe to share between callers.

    Raises:
        ValueError: The profile does not exist, or names an unknown field.
    """
    data = load_yaml(NUMERICS_DIR, name, "numerics profile") or {}
    unknown = set(data) - _FIELDS
    if unknown:
        path = NUMERICS_DIR / f"{name}.yaml"
        raise ValueError(f"{path}: unknown fields {sorted(unknown)}")
    return ModelConfig(**data)
