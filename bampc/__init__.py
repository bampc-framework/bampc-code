"""Native MuJoCo Warp sampling-based MPC.

Architecture: a host-side **planner**
(sampling, risk aggregation, spline math — plain numpy) drives a device-side
**rollout engine** (batched MJWarp physics + cost kernels, CUDA-graph captured).
A **domain randomizer** owns the per-world randomized model fields. This split
is what keeps the algorithm classes tiny and makes the planner embeddable in a
ROS 2 node.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent

# `configs/` is a set of name-keyed profile registries; `models/` is the MJCF
# asset tree, resolved by relative path from inside the MJCF rather than by
# name, which is why it is not a registry. Neither ships in the wheel
# (`packages = ["bampc"]`), so both resolve only from a source
# checkout.
CONFIGS_DIR = ROOT.parent / "configs"
MODELS_DIR = ROOT.parent / "models"
NUMERICS_DIR = CONFIGS_DIR / "numerics"
SCENARIOS_DIR = CONFIGS_DIR / "scenarios"
NOISE_DIR = CONFIGS_DIR / "noise"
PLANNER_DIR = CONFIGS_DIR / "planner"
REWARD_DIR = CONFIGS_DIR / "reward"

# The one generated-artifact root: recordings, figures and sweep results all
# land under `output/`. Gitignored, and absent from the wheel like the two
# input trees above. Named here so nothing re-derives it from `__file__`.
OUTPUT_DIR = ROOT.parent / "output"

__all__ = [
    "CONFIGS_DIR",
    "MODELS_DIR",
    "NOISE_DIR",
    "NUMERICS_DIR",
    "OUTPUT_DIR",
    "PLANNER_DIR",
    "REWARD_DIR",
    "ROOT",
    "SCENARIOS_DIR",
]