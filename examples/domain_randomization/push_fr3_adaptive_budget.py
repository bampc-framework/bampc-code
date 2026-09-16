"""Interactive Push-FR3 with an adaptive domain/sample budget.

FR3 counterpart of ``push_adaptive_budget.py`` -- see that module for the
mechanism. Press the down/up arrow keys in the viewer to step the ``(R, S)``
ladder; the visible block ghosts (one per domain) track the switch live.

Hardcoded to task-space sampling / 3-DOF joint manipulation / the ``t``
shape / predictive sampling, to keep the focus on allocation -- see
``simple/push_fr3.py`` for the full CLI.

Run::

    uv run python examples/domain_randomization/push_fr3_adaptive_budget.py
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
from bampc.task.push_fr3 import PushFr3
from examples.status import CostStatus

# --------------------------------------------------------------------- #
# Config: numerics / task / planner
# --------------------------------------------------------------------- #

# The joint block model wants `eulerdamp=False`; the task's fallback has it
# on, so the profile must be named rather than inherited.
NUMERICS = load_numerics("fr3_joint")

TASK = dict(
    sampling_space="task",
    manipulation_type="joint",
    max_lin_vel=0.25,
    trace_sites=["ee_site"],
)

PLANNER = PlannerConfig(
    algo="ps",
    risk="average",
    num_samples=128,
    plan_horizon=0.6,
    num_knots=5,
    spline_type="linear",
    noise_level=0.3,
)

NUM_RANDOMIZATIONS = 8  # N = 1024 = 2**10 -> a clean 11-stage ladder
PLAN_FREQ = 10
RECORDING = True
DURATION = None  # sim seconds to run before stopping; None = until closed
PLAN_LAG_STEPS = 0  # planning-latency model (0 = off); see run_interactive
COMPENSATE_LATENCY = False  # forward-predict past PLAN_LAG_STEPS if set

# --------------------------------------------------------------------- #
# Domain randomization
# --------------------------------------------------------------------- #

# A function, not a dict: the allocation controller re-derives the grid at
# whatever domain count the new ladder stage asks for.


def friction_spec(num_randomizations: int) -> RandomizationSpec:
    """Ground sliding-friction grid, re-linspaced for the domain count."""
    grid = np.linspace(0.2, 1.2, num_randomizations).tolist()
    return {"body": {"ground": {"friction": grid}}}


# --------------------------------------------------------------------- #
# Build: task -> randomizer -> engine -> planner -> allocation
# --------------------------------------------------------------------- #

task = PushFr3(**TASK, model_config=NUMERICS)
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
# Start from the 'home' keyframe, then overlay the EE and block start pose.
key_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_KEY, "home")
if key_id != -1:
    mj_data.qpos = mj_model.key_qpos[key_id]

task.set_initial_state(
    mj_data,
    ee_pos=np.array([0.3, -0.1, 0.045]),
    block_xy=np.array([0.45, 0.15]),
    block_yaw=0.6,
)

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
    status_callback=cost_status,
    allocation=allocation,
    plan_lag_steps=PLAN_LAG_STEPS,
    compensate_latency=COMPENSATE_LATENCY,
    duration=DURATION,
    record=RECORDING,
    record_format="gif",
    record_camera="main",
    record_name="push_fr3_adaptive_budget",
)
