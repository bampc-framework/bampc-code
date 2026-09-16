"""Run-directory + CSV helpers for the sweeps.

Results land next to their config: ``<config-dir>/results/`` (or
``results/<subdir>/`` for a Push-FR3 manipulation variant). A
``resolved.json`` snapshot -- the fully-resolved config, the sweep
description, the git commit and a timestamp -- is dropped alongside so any
result file is reproducible from its run directory alone.
"""

from __future__ import annotations

import csv
import dataclasses
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from experiments.common.uncertainty.setups import RunConfig

REPO_ROOT = Path(__file__).resolve().parents[2]


def make_run_dir(
    config_dir: Path,
    cfg: RunConfig,
    conditions: dict,
    *,
    subdir: str | None = None,
) -> Path:
    """Create ``<config_dir>/results[/subdir]/`` and dump ``resolved.json``.

    ``conditions`` records what this variant actually ran (its arm, algo and
    engine shape) beside the resolved config, so a results directory answers
    "what produced this?" without reading the runner.
    """
    run_dir = config_dir / "results"
    if subdir:
        run_dir = run_dir / subdir
    run_dir.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    resolved = {
        "config": dataclasses.asdict(cfg),
        "conditions": conditions,
        "git_commit": commit,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    (run_dir / "resolved.json").write_text(
        json.dumps(resolved, indent=2, default=_jsonable) + "\n"
    )
    return run_dir


def _jsonable(obj: Any) -> Any:
    """Fallback encoder for numpy scalars/arrays in the conditions dict."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return float(obj)


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write ``rows`` with the first row's keys as the header."""
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)



def write_manifest(run_dir: Path, manifest: dict) -> None:
    """Write ``manifest.json``: the contract analysis reads a run through."""
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=_jsonable) + "\n"
    )
