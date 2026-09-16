"""Flat ``Float64MultiArray`` packing for the plan -> control hand-off.

The planner publishes an already-interpolated dense trajectory, not spline
knots, so the control node needs no spline math and no ``spline_type``
knowledge -- it only ever indexes into a plain array by elapsed time since
``t0``. ``t0`` is a ``time.monotonic()`` timestamp, not the planner's own
``state.time`` -- ``time.monotonic()`` is a single kernel clock shared by
every process on the host, which is what lets a separate control-node
process compare its own ``time.monotonic()`` reads against it; the
planner's ``state.time`` origin is private to whatever state reader
produced it and means nothing outside that process. Flat layout,
self-describing so ``unpack`` needs nothing but the message::

    [t0, dt, horizon, nu, u_0_0 .. u_0_{nu-1}, u_1_0 .. u_1_{nu-1}, ...]

No custom ``.msg``/ROS package build step: none exists in this repo, and a
documented flat array on a stock message type carries whatever ``nu`` the
task uses without one.
"""

from __future__ import annotations

import numpy as np

try:
    from std_msgs.msg import Float64MultiArray, MultiArrayDimension
except ImportError:  # ROS not installed; keep the module importable.
    Float64MultiArray = None  # type: ignore[assignment,misc]
    MultiArrayDimension = None  # type: ignore[assignment,misc]


def pack(t0: float, dt: float, actions: np.ndarray) -> Float64MultiArray:
    """Pack a dense ``(horizon, nu)`` trajectory into one flat message.

    ``t0`` must be a ``time.monotonic()`` reading (see module docstring),
    not any other clock/epoch.
    """
    actions = np.asarray(actions, dtype=np.float64)
    horizon, nu = actions.shape
    msg = Float64MultiArray()
    msg.layout.dim = [
        MultiArrayDimension(label="horizon", size=horizon, stride=nu),
        MultiArrayDimension(label="nu", size=nu, stride=1),
    ]
    msg.data = [float(t0), float(dt), float(horizon), float(nu)]
    msg.data.extend(actions.reshape(-1).tolist())
    return msg


def unpack(msg: Float64MultiArray) -> tuple[float, float, np.ndarray]:
    """Unpack a message into ``(t0, dt, actions)``, shape ``(horizon, nu)``."""
    data = np.asarray(msg.data, dtype=np.float64)
    t0, dt, horizon, nu = data[0], data[1], int(data[2]), int(data[3])
    actions = data[4:].reshape(horizon, nu)
    return float(t0), float(dt), actions
