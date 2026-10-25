# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Synoros

"""Work in progress, shipped so it can be used and improved.

These primitives are unfinished. Each module's docstring describes what is
known to be wrong.
"""

from holonomy_lib.experimental.heterogeneous_kappa import HeterogeneousKappaManifold
from holonomy_lib.experimental.laplacian_eigenmaps import hyperbolic_laplacian_eigenmaps

__all__ = ["HeterogeneousKappaManifold", "hyperbolic_laplacian_eigenmaps"]
