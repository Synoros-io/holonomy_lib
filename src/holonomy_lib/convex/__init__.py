# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Synoros

"""holonomy_lib.convex: problem-level convex optimization primitives.

This package holds CONVEX PROBLEM interfaces ("minimize this objective")
as opposed to the raw iterative solvers in `holonomy_lib.algebra.solvers`
("run this Krylov method"). A primitive enters this package only with a
named downstream consumer (repo rule: no speculative features); the
docstring of each function records who consumes it.

Currently implemented:
  least_squares.regularized_lstsq: Tikhonov/ridge least squares
    `min ‖Ax − b‖² + λ‖x‖²` with solver dispatch (Cholesky normal
    equations / CG on the normal operator / LSQR with damping).
    Consumers: synoros-substrate's ridge sites (expectation_unit,
    shape_prediction, arbitrary_pairs, hierarchy_potential — all
    hand-rolled variants of this one problem as of 2026-07).

Planned (each gated on a consumer materializing):
  proximal operators, projections onto convex sets, accelerated
  first-order methods.
"""

from holonomy_lib.convex.least_squares import regularized_lstsq

__all__ = ["regularized_lstsq"]
