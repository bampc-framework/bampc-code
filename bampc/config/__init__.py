"""Loaders for the name-keyed config registries under ``configs/``.

Host-side: a caller names a profile and the loader reads it from ``configs/`` --
solver/integrator profiles (:mod:`bampc.config.numerics`) and frozen
start-state banks (:mod:`bampc.config.scenarios`). Kept apart from the
task-dynamics classes in :mod:`bampc.task`, which define physics/cost.
"""
