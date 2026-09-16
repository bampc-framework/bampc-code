"""Export a slim, standalone dataset + plot script for one sweep cell.

For sharing results outside the repo (e.g. a cloud drive) without handing
over the whole ``experiments/`` tree or its dependencies. Reads
what ``run.py`` wrote and ``analysis.py``'s own display metadata
(``ARM_STYLE``), and writes, per ``--out`` directory:

- ``<arm>.csv`` per arm -- only the columns ``pose_error_accumulated`` and
  ``pose_error_components`` need (``seed,repeat,step,time,pos_err,
  orient_err``), same row order as the source ``tracking.csv`` so a naive
  reshape by ``(num_seeds*repeats, num_chunks)`` recovers the same grid
  ``load_variant`` builds.
- ``meta.json`` -- task, cell name, sensor settings, ``rot_scale``, and each
  arm's display metadata (label/color/linestyle/replan_dt/...), so the
  standalone plot script needs no other file.
- ``plot.py`` -- copied in verbatim (see ``_PLOT_SCRIPT`` below): numpy +
  matplotlib only, no ``bampc``/``experiments`` import, so it runs
  after just ``pip install numpy matplotlib``.

Run (from the repo root)::

    uv run python -m experiments.state_uncertainty.export_shareable \
        --version push_fr3/state_uncertainty/t-ou/scale2.4-tau2.0-warmup50 \
        --out data/pushing/simulation_performance/scale2.4-tau2.0-warmup50
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.state_uncertainty.analysis import (
    ALGO_ORDER,
    ARM_STYLE,
    FALLBACK_ROT_SCALE,
)

EXPERIMENTS_ROOT = Path(__file__).resolve().parent.parent

# Only what pose_error_accumulated/pose_error_components read -- see
# analysis.py's load_variant. Kept in tracking.csv's own column order so a
# reader can genfromtxt this file exactly like the original.
COLUMNS = ("seed", "repeat", "step", "time", "pos_err", "orient_err")


def export(version: str, out: Path, mode: str = "full") -> None:
    """Write ``<arm>.csv`` + ``meta.json`` + ``plot.py`` under ``out``."""
    results = EXPERIMENTS_ROOT / version / "results"
    if mode != "full":
        results = results / mode
    out.mkdir(parents=True, exist_ok=True)

    arms_meta = []
    rot_scale = None
    sensor = None
    task = None
    cell = Path(version).name

    for algo_dir in sorted(results.iterdir()):
        if not algo_dir.is_dir() or algo_dir.name not in ALGO_ORDER:
            continue
        for arm_dir in sorted(algo_dir.iterdir()):
            csv_path = arm_dir / "tracking.csv"
            if not csv_path.is_file():
                continue
            arm = arm_dir.name
            manifest = json.loads((arm_dir / "manifest.json").read_text())
            resolved = json.loads((arm_dir / "resolved.json").read_text())
            task = manifest["task"]
            if sensor is None:
                sensor = resolved["config"].get("sensor", {})
            if rot_scale is None:
                tp = resolved["config"].get("task_params", {})
                if tp.get("w_pos") and tp.get("w_orient") is not None:
                    rot_scale = float(tp["w_orient"]) / float(tp["w_pos"])

            d = np.genfromtxt(csv_path, delimiter=",", names=True)
            style = ARM_STYLE.get(arm, (None, "-", arm))
            color, linestyle, label = style

            slim_path = out / f"{arm}.csv"
            with slim_path.open("w") as f:
                f.write(",".join(COLUMNS) + "\n")
                np.savetxt(
                    f, np.stack([d[c] for c in COLUMNS], axis=1),
                    delimiter=",", fmt="%.10g",
                )
            arms_meta.append({
                "name": arm,
                "algo": algo_dir.name,
                "label": label,
                "color": color,
                "linestyle": (
                    linestyle if isinstance(linestyle, str) else list(linestyle)
                ),
                "replan_dt": manifest["replan_dt"],
                "num_seeds": manifest["num_seeds"],
                "repeats": manifest["repeats"],
                "num_chunks": manifest["num_chunks"],
                "csv": slim_path.name,
            })
            print(f"  wrote {slim_path.relative_to(out.parent)}")

    if not arms_meta:
        raise SystemExit(f"no runs found under {results}")
    if rot_scale is None:
        rot_scale = FALLBACK_ROT_SCALE

    meta = {
        "task": task,
        "cell": cell,
        "sensor": sensor,
        "rot_scale": rot_scale,
        "arms": arms_meta,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"  wrote {(out / 'meta.json').relative_to(out.parent)}")

    (out / "plot.py").write_text(_PLOT_SCRIPT)
    print(f"  wrote {(out / 'plot.py').relative_to(out.parent)}")


# Standalone: numpy + matplotlib only. No bampc / experiments import
# -- this file travels with the data, outside the repo.
_PLOT_SCRIPT = '''"""Recreate pose_error_accumulated.png and
pose_error_components.png from the CSVs + meta.json in this directory.
Needs only numpy + matplotlib::

    pip install numpy matplotlib
    python plot.py
"""

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
META = json.loads((HERE / "meta.json").read_text())
ROT_SCALE = META["rot_scale"]


def load_arm(arm: dict):
    """seed/repeat/time/pos_err/orient_err as (episode, step) grids."""
    d = np.genfromtxt(HERE / arm["csv"], delimiter=",", names=True)
    n_ep = arm["num_seeds"] * arm["repeats"]
    n_step = arm["num_chunks"]

    def grid(name):
        return d[name].reshape(n_ep, n_step)

    pos_err = grid("pos_err")
    orient_err = grid("orient_err")
    valid = np.isfinite(pos_err).all(axis=1)
    time = grid("time")[valid][0] if valid.any() else grid("time")[0]
    return {
        "time": time,
        "pos_err": pos_err,
        "orient_err": orient_err,
        "valid": valid,
        "replan_dt": arm["replan_dt"],
    }


def pose_error(data):
    if np.all(np.isnan(data["orient_err"])):
        return data["pos_err"]
    return data["pos_err"] + ROT_SCALE * data["orient_err"]


def accumulate(data, values):
    return np.cumsum(values, axis=1) * data["replan_dt"]


def band(ax, t, values, color, style, label, keep):
    values = values[keep]
    lo, mid, hi = np.percentile(values, [25, 50, 75], axis=0)
    ax.fill_between(t, lo, hi, color=color, alpha=0.13, linewidth=0)
    ax.plot(t, mid, color=color, linestyle=style, linewidth=1.4, label=label)


def main():
    arms = {a["name"]: (a, load_arm(a)) for a in META["arms"]}

    # pose_error_accumulated.png
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for a, data in arms.values():
        ls = tuple(a["linestyle"]) if isinstance(a["linestyle"], list) \\
            else a["linestyle"]
        band(ax, data["time"], accumulate(data, pose_error(data)),
             a["color"], ls, a["label"], data["valid"])
    ax.set_xlabel("time [s]")
    ax.set_ylabel("integral of pose error [m*s]")
    ax.set_title(
        f"accumulated goal-to-object pose error, SO(3)xR^3 @ "
        f"{ROT_SCALE:.3f} m/rad  ({META['cell']})"
    )
    ax.legend(frameon=False, fontsize=7.5)
    fig.tight_layout()
    fig.savefig(HERE / "pose_error_accumulated.png", dpi=150)
    plt.close(fig)
    print("wrote pose_error_accumulated.png")

    # pose_error_components.png
    fig, axes = plt.subplots(2, 1, figsize=(6.4, 7.2), sharex=True)
    for a, data in arms.values():
        ls = tuple(a["linestyle"]) if isinstance(a["linestyle"], list) \\
            else a["linestyle"]
        band(axes[0], data["time"], data["pos_err"], a["color"], ls,
             a["label"], data["valid"])
        band(axes[1], data["time"], data["orient_err"], a["color"], ls,
             a["label"], data["valid"])
    axes[0].set_ylabel("position error [m]")
    axes[1].set_ylabel("rotation error [rad]")
    axes[1].set_xlabel("time [s]")
    axes[0].set_title(
        f"pose error by channel -- native units, never summed  "
        f"({META['cell']})"
    )
    axes[0].legend(frameon=False, fontsize=7.5)
    fig.tight_layout()
    fig.savefig(HERE / "pose_error_components.png", dpi=150)
    plt.close(fig)
    print("wrote pose_error_components.png")


if __name__ == "__main__":
    main()
'''


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--version", required=True,
        help="Path relative to experiments/, e.g. "
        "push_fr3/state_uncertainty/t-ou/scale2.4-tau2.0-warmup50.",
    )
    parser.add_argument(
        "--out", required=True, type=Path,
        help="Output directory for the shareable dataset.",
    )
    parser.add_argument(
        "--mode", choices=["full", "smoke", "quick"], default="full",
    )
    args = parser.parse_args()
    export(args.version, args.out, args.mode)


if __name__ == "__main__":
    main()
