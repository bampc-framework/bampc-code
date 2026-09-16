"""An ensemble with state spread but no contact-mode diversity.

The counterpart to a plain ensemble for one question: when a belief cloud
beats a point estimate, is that because it *covers several contact modes*, or
merely because it *spreads in state* within one? :class:`CollapsedEnsemble`
isolates the second -- it perturbs like an ordinary ensemble, then
rejection-samples so every member settles in the **same contact mode as the
point estimate**.

Unlike :class:`~bampc.uncertainty.state.StateUncertainty`, this is NOT
simulator-agnostic: deciding a candidate's contact mode needs physics, so it
borrows the rollout engine as a mode oracle
(``set_initial_state -> settle -> refresh_initial_state -> contact_modes``).
The engine must be built with ``record_initial_state=True`` and attached via
:meth:`attach_engine` before the first :meth:`sample`; unattached, it degrades
to a plain ensemble (no collapse), which keeps it safe to construct anywhere.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from bampc.uncertainty.state import StateUncertainty

if TYPE_CHECKING:
    from collections.abc import Sequence

    from bampc.planner.base import StateSnapshot
    from bampc.rollout.engine import RolloutEngine
    from bampc.task.base import Task
    from bampc.uncertainty.estimator import Estimator
    from bampc.uncertainty.noise import StateNoise


class CollapsedEnsemble(StateUncertainty):
    """A belief cloud collapsed onto the point estimate's contact mode.

    Args:
        task / num_randomizations / noise / estimator / seed: as
            :class:`~bampc.uncertainty.state.StateUncertainty`.
            ``noise`` is normally one
            :class:`~bampc.uncertainty.noise.PosteriorGaussian`, the
            same spread ``ensemble_exact`` uses.
        max_attempts: How many resample rounds to spend matching modes before
            collapsing the stragglers onto the estimate.

    Domain 0 is always the unperturbed estimate, so its settled mode is the
    match target and comes back for free as ``contact_modes()[0]``. A member
    that never matches after ``max_attempts`` becomes the estimate itself --
    mode-matched by construction; in the limit the arm degrades to a point
    estimate, the honest reading of "cannot spread inside this mode".
    """

    def __init__(
        self,
        task: Task,
        num_randomizations: int,
        noise: Sequence[StateNoise],
        estimator: Estimator | None = None,
        seed: int = 0,
        max_attempts: int = 8,
    ) -> None:
        """Build the ensemble; the engine oracle is attached later."""
        super().__init__(
            task, num_randomizations, noise, estimator, seed,
            include_truth=False,
        )
        self.max_attempts = int(max_attempts)
        self.engine: RolloutEngine | None = None
        # Diagnostics: how often a member could not be mode-matched and was
        # collapsed onto the estimate instead.
        self.fallbacks = 0
        self.replans = 0

    def attach_engine(self, engine: RolloutEngine) -> None:
        """Give it the rollout engine to use as a contact-mode oracle."""
        self.engine = engine

    def _modes(
        self, qpos: np.ndarray, qvel: np.ndarray, state: StateSnapshot
    ) -> np.ndarray:
        """Per-domain settled contact mask ``(R,)`` for candidate states."""
        self.engine.set_initial_state(
            qpos, qvel,
            mocap_pos=state.mocap_pos, mocap_quat=state.mocap_quat,
            time=state.time,
        )
        self.engine.settle()
        self.engine.refresh_initial_state()
        modes = self.engine.contact_modes()
        if modes is None:
            raise RuntimeError(
                "CollapsedEnsemble needs an engine built with "
                "record_initial_state=True to read contact modes"
            )
        return modes

    def _perturb(
        self, qpos: np.ndarray, qvel: np.ndarray, idx: np.ndarray,
        state: StateSnapshot,
    ) -> None:
        """Redraw domains ``idx`` as fresh perturbations of the estimate."""
        sub_q = np.tile(np.asarray(state.qpos, float), (len(idx), 1))
        sub_v = np.tile(np.asarray(state.qvel, float), (len(idx), 1))
        for n in self.noise:
            n.apply(sub_q, sub_v, self.rng, self.task.mj_model, self.layout)
        qpos[idx] = sub_q
        qvel[idx] = sub_v

    def sample(
        self, state: StateSnapshot
    ) -> tuple[np.ndarray, np.ndarray]:
        """Draw the cloud, then reject members off the estimate's mode."""
        r = self.num_randomizations
        qpos = np.tile(np.asarray(state.qpos, float), (r, 1))
        qvel = np.tile(np.asarray(state.qvel, float), (r, 1))
        for n in self.noise:
            n.apply(qpos, qvel, self.rng, self.task.mj_model, self.layout)
        # Domain 0 stays the estimate: the mode all others must match.
        qpos[0] = state.qpos
        qvel[0] = state.qvel

        if self.engine is not None:
            self.replans += 1
            for _ in range(self.max_attempts):
                modes = self._modes(qpos, qvel, state)
                bad = np.flatnonzero(modes[1:] != modes[0]) + 1
                if len(bad) == 0:
                    break
                self._perturb(qpos, qvel, bad, state)
            else:
                # Exhausted: collapse the stragglers onto the estimate.
                qpos[bad] = state.qpos
                qvel[bad] = state.qvel
                self.fallbacks += len(bad)

        return self.estimator(qpos, qvel, self.layout, self.rng)
