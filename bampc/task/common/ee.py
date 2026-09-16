"""Attach an end effector onto the shared FR3 arm.

The arm (``models/fr3/fr3_arm.xml``) ends in a bare ``attachment_site``; the
end effector lives in its own XML model (``models/fr3/ee_<name>.xml``) and is
spliced onto that site at build time, mirroring how ``graft_shape`` grafts a
block. This keeps the arm common while each task picks its own EE (``pusher``
for Push, ``plate`` for Balance). Every EE fragment re-creates a body named
``ee_frame``, the IK / posture reference the rest of the code keys on.
"""

from __future__ import annotations

import mujoco

from bampc import MODELS_DIR


def attach_ee(spec: mujoco.MjSpec, name: str) -> mujoco.MjsBody:
    """Attach the ``ee_<name>`` end effector onto the arm's attachment site.

    Args:
        spec: arm spec exposing an ``attachment_site`` (the FR3 include).
        name: EE fragment stem, i.e. ``models/fr3/ee_<name>.xml``.

    Returns:
        The attached EE root body (in ``spec``).
    """
    child = mujoco.MjSpec.from_file(str(MODELS_DIR / "fr3" / f"ee_{name}.xml"))
    root = child.worldbody.bodies[0]  # single EE subtree, authored at the site
    # Empty prefix/suffix keeps the fragment's body/geom/site names.
    return spec.site("attachment_site").attach_body(root, "", "")
