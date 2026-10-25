# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Synoros

"""holonomy_lib.tensor_calculus — multilinear algebra primitives.

Currently implemented:
  mode_product(T, A, axis)  — n-mode product T ×_k A on tensor T.
  mode_unfolding(T, axis)   — matricize T by bringing one axis to front.
  hosvd(T, ranks, mode)     — Truncated Higher-Order SVD.

Planned (per HANDOFF.md §6):
  CP / Tucker decompositions with ALS, Tensor-Train, manifold-valued
  tensor fields, Einstein notation helpers.
"""

from holonomy_lib.tensor_calculus.decomposition import (
    hosvd,
    mode_product,
    mode_unfolding,
)

__all__ = ["hosvd", "mode_product", "mode_unfolding"]
