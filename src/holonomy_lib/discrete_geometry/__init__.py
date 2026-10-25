# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Synoros

"""Discrete graph curvature and edge-weight Ricci flow.

Available primitives:
  ollivier_ricci_curvature: Ollivier-Ricci curvature on every node pair,
    from entropic (Sinkhorn) transport.
  forman_ricci_simple: weighted graph Forman curvature with unit vertices.
  forman_ricci_augmented: unit-weight Forman curvature with triangle terms.
  discrete_ricci_flow: curvature-driven edge-weight updates.
  ricci_flow_with_surgery: flow with threshold-based edge removal.

References:
  Ollivier (2009), Ricci curvature of Markov chains on metric spaces.
  Sreejith et al. (2016), Forman curvature for complex networks.
  Samal et al. (2018), Comparative analysis of two discretizations of
    Ricci curvature for complex networks.
  Ni et al. (2019), Community detection on networks with Ricci flow.
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
