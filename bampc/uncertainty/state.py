"""Per-domain state estimates: the belief the planner rolls out from.

The counterpart to :mod:`bampc.dr` on the *state* axis rather than the
model axis, and deliberately separate from it: a real robot's dominant error
is usually its state estimate, not its model parameters, and the two are
independent (both can be on at once).

Composes two orthogonal choices -- a **noise model** (what the sensor gets
wrong, :mod:`bampc.uncertainty.noise`) and an **estimator** (what
you do about it, :mod:`bampc.uncertainty.estimator`).
Simulator-agnostic by construction: it consumes a
:class:`~bampc.planner.base.StateSnapshot`, so a ROS node feeding
one from its state estimator uses the identical call.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from bampc.uncertainty.estimator import Ensemble

if TYPE_CHECKING:
    from collections.abc import Sequence

    from bampc.planner.base import StateSnapshot
    from bampc.task.base import Task
    from bampc.uncertainty.estimator import Estimator
    from bampc.uncertainty.noise import StateNoise


class StateUncertainty:
    """Fans one state estimate out into ``R`` per-domain beliefs.

    Args:
        task: Supplies the model and the object's pose layout.
        num_randomizations: ``R`` -- the domain axis the beliefs fill.
        noise: Noise models, applied in order to every domain.
        estimator: How the cloud is reduced before planning. Defaults to
            :class:`~bampc.uncertainty.estimator.Ensemble` (keep it
            all).
        seed: RNG seed.
        include_truth: Force domain 0 to the unperturbed state.

    ``include_truth`` defaults to ``False`` on purpose. It is a useful
    debugging aid, but it **rigs a comparison between estimators**: the
    ensemble arm would get one free correct domain that no point estimator
    ever sees, so part of any advantage it showed would be that gift rather
    than the method. Turn it on to look at something, off to measure it.
    """

    def __init__(
        self,
        task: Task,
        num_randomizations: int,
        noise: Sequence[StateNoise],
        estimator: Estimator | None = None,
        seed: int = 0,
        include_truth: bool = False,
    ) -> None:
        """Resolve the pose layout once; noise addresses resolve per call."""
        self.task = task
        self.num_randomizations = max(int(num_randomizations), 1)
        self.noise = list(noise)
        self.estimator = estimator if estimator is not None else Ensemble()
        self.rng = np.random.default_rng(seed)
        self.include_truth = bool(include_truth)
        self.layout = task.object_pose_qpos
        # The last cloud handed to the planner, cached for diagnostics that
        # need the exact states a rollout settled from (e.g. a settle-spread
        # probe). Inert: nothing in planning reads it, so no arm's numbers
        # change. ``None`` until the first :meth:`sample`.
        self.last_sample: tuple[np.ndarray, np.ndarray] | None = None

    def sample(
        self, state: StateSnapshot
    ) -> tuple[np.ndarray, np.ndarray]:
        """Draw ``(R, nq)`` / ``(R, nv)`` beliefs and reduce them.

        Returns the arrays the engine uploads per domain. With no noise and
        the default estimator this is just the input repeated ``R`` times,
        which is what ``set_initial_state`` would have broadcast anyway.
        """
        r = self.num_randomizations
        qpos = np.tile(np.asarray(state.qpos, dtype=float), (r, 1))
        qvel = np.tile(np.asarray(state.qvel, dtype=float), (r, 1))

        # Thread the sim time to any time-aware noise term (e.g. a wandering
        # bias). A no-op for every other term -- apply's signature is unchanged.
        t = float(getattr(state, "time", 0.0))
        for n in self.noise:
            if hasattr(n, "set_time"):
                n.set_time(t)
            n.apply(qpos, qvel, self.rng, self.task.mj_model, self.layout)

        if self.include_truth:
            qpos[0] = state.qpos
            qvel[0] = state.qvel

        out = self.estimator(qpos, qvel, self.layout, self.rng)
        self.last_sample = out
        return out
