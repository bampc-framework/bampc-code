"""Plan-time scaling benchmark (run via ``python -m experiments.scaling...``).

Standalone from the ``bampc`` library and from the accuracy-focused
``experiments.dr``/``experiments.uncertainty`` trees: measures how long one
``planner.optimize()`` call takes as a function of total sample count
(``nworld = R x S``) and planning horizon (solver steps). Not part of the
``bampc`` wheel.
"""
