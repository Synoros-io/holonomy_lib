# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Synoros

"""holonomy_lib.discrete_geometry — combinatorial curvatures and discrete
differential geometry on graphs / meshes.

This module exists because the continuous Riemannian primitives in
`holonomy_lib.manifolds` answer "what is the geometry of this smooth
manifold?", while many research problems instead ask "what is the
discrete geometry of this graph or simplicial complex?". The discrete
versions have their own well-developed theory — discrete Ricci flow,
combinatorial curvatures, etc. — that is computationally tractable in
a way the smooth versions are not.

Currently implemented:
  ricci.ollivier_ricci_curvature(A, alpha, ...) — Ollivier-Ricci on
    weighted graphs via Sinkhorn Wasserstein-1.

Planned:
  ricci.lin_lu_yau_curvature(A, ...)            — α → 1 limit (Liu-Lin-Yau 2011).
  ricci.forman_ricci_curvature(A, ...)          — Forman (2003) combinatorial.
  ricci.discrete_ricci_flow(A, n_steps, ...)    — iterative edge-weight flow.
  ricci.ricci_flow_with_surgery(A, ...)         — Sia-Jonckheere-Bogdan
    (2019) / Ni-Lin-Luo-Gao analog of Perelman's surgery for community
    decomposition.

References (program-level):
  Perelman, G. (2002, 2003). The entropy formula for the Ricci flow and
    its geometric applications; Ricci flow with surgery on
    three-manifolds; Finite extinction time. arXiv:math/0211159,
    arXiv:math/0303109, arXiv:math/0307245. The smooth-manifold
    inspiration; the discrete analogs in this module are computational
    cousins, not equivalents.
  Liu, F., Wang, X., Yau, S.-T., Zeng, W. (2017). A realization of
    Thurston's geometrization: discrete Ricci flow with surgery.
    arXiv:1709.08494. The most faithful computational analog of
    Perelman's surgery in 3-D, on simplicial complexes.
"""

from holonomy_lib.discrete_geometry.forman import (
    forman_ricci_augmented,
    forman_ricci_simple,
)
from holonomy_lib.discrete_geometry.ricci import (
    discrete_ricci_flow,
    ollivier_ricci_curvature,
    ricci_flow_with_surgery,
)

__all__ = [
    "discrete_ricci_flow",
    "forman_ricci_augmented",
    "forman_ricci_simple",
    "ollivier_ricci_curvature",
    "ricci_flow_with_surgery",
]
