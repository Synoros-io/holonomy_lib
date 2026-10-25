# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Synoros

"""holonomy_lib.hyperbolic — manifold-aware graph operations.

The primitives in this module take a manifold (e.g.
`LorentzManifold`) as an explicit dependency and operate on points
embedded in that manifold. This is the layer where graph algorithms
meet differential geometry: Fréchet (intrinsic) means,
parallel-transport-based similarity,
and the manifold heat kernel.

Manifold is passed in by dependency injection so primitives
generalize to other constant-curvature manifolds
(`KappaStereographicManifold`, future spherical etc.) without
rewrites.
"""

from holonomy_lib.hyperbolic.frechet_mean import frechet_mean
from holonomy_lib.hyperbolic.heat_kernel import hyperbolic_heat_kernel
from holonomy_lib.hyperbolic.manifold_inner import manifold_aware_inner

__all__ = [
    "frechet_mean",
    "hyperbolic_heat_kernel",
    "manifold_aware_inner",
]
