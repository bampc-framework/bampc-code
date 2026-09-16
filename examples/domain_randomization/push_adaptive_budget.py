"""Interactive Push with an adaptive domain/sample budget.

Demonstrates ``bampc.allocation``: the flat ``nworld = R * S``
budget can split between more domains ``R`` (robustness) or more samples
``S`` (search depth) without any CUDA-graph rebuild. Press the down/up arrow
keys in the viewer to step down / up a precomputed ``(R, S)`` ladder; the
number of visible domain ghosts (translucent block outlines, one per
domain) tracks the switch live.

Uses :class:`ManualStagePolicy` and redraws the friction grid at the new
domain count on every switch, so "domain 0" is not preserved across one.

Run::

    uv run python examples/domain_randomization/push_adaptive_budget.py
"""

from __future__ import annotations

import mujoco
import numpy as np

from bampc.allocation import AllocationController, ManualStagePolicy
from bampc.config.numerics import load as load_numerics
from bampc.dr import DomainRandomizer
from bampc.dr.randomizer import RandomizationSpec
from bampc.planner import PlannerConfig, build_planner
from bampc.rollout import WarpRolloutEngine
from bampc.sim import run_interactive
from bampc.task.push import Push
from examples.status import CostStatus

# --------------------------------------------------------------------- #
# Config: numerics / task / planner
# --------------------------------------------------------------------- #

# No CLI: hardcoded to one algorithm and the default shape to keep this
# focused on the allocation mechanism -- see push.py for the full CLI.
NUMERICS = load_numerics("push")

TASK: dict = {}  # defaults: the 't' shape at scale 1, static goal

PLANNER = PlannerConfig(
    algo="mppi",
    risk="average",
    num_samples=128,
    plan_horizon=0.4,
    num_knots=5,
    spline_type="zero",
    noise_level=0.3,
    temperature=0.01,
)

NUM_RANDOMIZATIONS = 4  # N = 256 = 2**8 -> a clean 9-stage ladder
PLAN_FREQ = 30
DURATION = None  # sim seconds to run before stopping; None = until closed
PLAN_LAG_STEPS = 0  # planning-latency model (0 = off); see run_interactive
COMPENSATE_LATENCY = False  # forward-predict past PLAN_LAG_STEPS if set

# --------------------------------------------------------------------- #
# Domain randomization
# --------------------------------------------------------------------- #

# A function, not a dict: the allocation controller re-derives the grid at
# whatever domain count the new ladder stage asks for.


def friction_spec(num_randomizations: int) -> RandomizationSpec:
    """Sliding-friction grid, re-linspaced for the current domain count."""
    grid = np.linspace(0.05, 0.95, num_randomizations).tolist()
    return {"geom": {"floor": {"friction": grid}}}


# --------------------------------------------------------------------- #
# Build: task -> randomizer -> engine -> planner -> allocation
# --------------------------------------------------------------------- #

task = Push(**TASK, model_config=NUMERICS)
randomizer = DomainRandomizer(
    task,
    num_randomizations=NUM_RANDOMIZATIONS,
    spec=friction_spec(NUM_RANDOMIZATIONS),
    seed=0,
)
engine = WarpRolloutEngine(
    task,
    num_samples=PLANNER.num_samples,
    num_randomizations=NUM_RANDOMIZATIONS,
    randomizer=randomizer,
    record_traces=True,
)
planner = build_planner(PLANNER, task, engine)
allocation = AllocationController.build(
    engine, planner, ManualStagePolicy(), spec_fn=friction_spec
)

print("adaptive budget ladder (R, S):", allocation.stages)
print(f"starting at stage {allocation.current_idx}: {allocation.current}")
print("press up/down arrows in the viewer to step the domain/sample split")

# --------------------------------------------------------------------- #
# Initial state
# --------------------------------------------------------------------- #

mj_model = task.mj_model
mj_data = mujoco.MjData(mj_model)
mj_data.qpos = [0.1, 0.1, 1.3, 0.0, 0.0]

# --------------------------------------------------------------------- #
# Viewer
# --------------------------------------------------------------------- #

cost_status = CostStatus(task)

run_interactive(
    planner,
    mj_model,
    mj_data,
    frequency=PLAN_FREQ,
    show_traces=True,
    show_endpoints=True,
    trace_idxs=[0],
    max_traces=1,
    allocation=allocation,
    status_callback=cost_status,
    plan_lag_steps=PLAN_LAG_STEPS,
    compensate_latency=COMPENSATE_LATENCY,
    duration=DURATION,
)
