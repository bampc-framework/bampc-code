"""Shared YAML-profile-loading helpers.

Every named-profile registry in this codebase (numerics, scenarios,
planner, reward, noise) is a directory of ``<name>.yaml`` files with a
``list_*()`` that globs it and a raw loader that resolves ``name`` to a
path, checks it exists, and parses it. This module holds only that shared
part; each registry keeps its own public ``load()``/``list_profiles()``
wrapper, since what they return (a dataclass, a list, a dict with axes
resolved) genuinely differs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def list_names(directory: Path) -> list[str]:
    """Every profile/bank/preset stem available under ``directory``."""
    return sorted(p.stem for p in directory.glob("*.yaml"))


def load_yaml(directory: Path, name: str, kind: str) -> Any:
    """Parse ``<directory>/<name>.yaml``, raising a consistently-worded error.

    Args:
        directory: The registry's directory (e.g. ``NUMERICS_DIR``).
        name: Profile stem, e.g. ``"push"``.
        kind: Human-readable noun for the error message, e.g.
            ``"numerics profile"``.

    Returns:
        The parsed YAML document, ``None`` if the file is empty -- callers
        apply their own default (``or {}``, ``or []``) and type-check.

    Raises:
        ValueError: No such file under ``directory``.
    """
    path = directory / f"{name}.yaml"
    if not path.is_file():
        raise ValueError(
            f"unknown {kind} {name!r}; available: {list_names(directory)}"
        )
    return yaml.safe_load(path.read_text())
