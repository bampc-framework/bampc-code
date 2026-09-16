"""State-estimate uncertainty: per-domain beliefs instead of one exact state.

Axes, composed by :class:`StateUncertainty`:

* **noise** -- what the sensor gets wrong
  (:mod:`~bampc.uncertainty.noise`), named in bulk by
  :mod:`~bampc.uncertainty.presets`.
* **estimator** -- what you do about a cloud at one instant
  (:mod:`~bampc.uncertainty.estimator`).
* **filter** -- what you do about a *sequence* of readings
  (:mod:`~bampc.uncertainty.filter`).

Separate from :mod:`bampc.dr`, which randomizes the *model*; both can be
active at once.

``StateUncertainty`` perturbs whatever state it is handed, which is what lets
one class play two roles: hand it the truth with ``R=1`` and it is a
**sensor** producing one observation; hand it that observation with ``R>1``
and it is a **belief** cloud around it. Fanning the truth directly into ``R``
draws instead would hand the planner ``R`` independent readings -- a mean
``noise/sqrt(R)`` from the truth that no real system gets.
"""

from bampc.uncertainty.collapsed import CollapsedEnsemble
from bampc.uncertainty.estimator import (
    Draw,
    Ensemble,
    Estimator,
    Mean,
    WeightedMean,
    pose_mean,
)
from bampc.uncertainty.filter import (
    FiniteDifference,
    Passthrough,
    PointFilter,
    PoseKalman,
    TwistOnly,
)
from bampc.uncertainty.noise import (
    InnovationGaussian,
    JointJitter,
    PosteriorGaussian,
    SE3Bias,
    SE3BiasCompassWander,
    SE3Gaussian,
    SE3OrnsteinUhlenbeck,
    StateNoise,
    TwistGaussian,
    TwistUnobserved,
)
from bampc.uncertainty.sigma import SigmaPointBelief
from bampc.uncertainty.state import StateUncertainty

__all__ = [
    "CollapsedEnsemble",
    "Draw",
    "Ensemble",
    "Estimator",
    "FiniteDifference",
    "InnovationGaussian",
    "JointJitter",
    "Mean",
    "Passthrough",
    "PointFilter",
    "PoseKalman",
    "PosteriorGaussian",
    "SE3Bias",
    "SE3BiasCompassWander",
    "SE3Gaussian",
    "SE3OrnsteinUhlenbeck",
    "SigmaPointBelief",
    "StateNoise",
    "StateUncertainty",
    "TwistGaussian",
    "TwistOnly",
    "TwistUnobserved",
    "WeightedMean",
    "pose_mean",
]
