# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Synoros

"""holonomy_lib.sheaf — cellular sheaves and sheaf Laplacians on graphs.

A **cellular sheaf** on a graph attaches a finite-dimensional vector
space (a "stalk") `F(σ)` to each simplex σ (node or edge) and a linear
**restriction map** `F_{τ ≤ σ}: F(τ) → F(σ)` for each face relation
τ ≤ σ. The sheaf is the data of those stalks + maps; the sheaf
Laplacian generalizes the graph Laplacian by capturing the
"disagreement" of node-stalk values once they are pushed up to each
incident edge stalk:

    L_F = δ^T δ,   δ_e(x)  =  F_{u ≤ e}(x_u) − F_{v ≤ e}(x_v)
                              for each edge e = (u, v).

When all stalks have dimension 1 and every restriction map is the
identity (i.e., the *trivial sheaf*), `L_F` reduces to the standard
combinatorial graph Laplacian. With higher-dimensional stalks and
non-trivial restriction maps it captures structure that a scalar
Laplacian cannot — orientation sheaves, signed cellular sheaves,
connection Laplacians, and the spectral domain that drives Neural
Sheaf Diffusion (Bodnar et al. 2022).

Currently exposed:
  GraphSheaf                  — dataclass holding edges + stalks + maps
  HeterogeneousGraphSheaf     — per-node (heterogeneous) stalk dimensions
  sheaf_coboundary(sheaf)     — δ as a (n_e·d_e, n_v·d_v) tensor
  sheaf_laplacian(sheaf)      — L_F = δ^T δ, (n_v·d_v, n_v·d_v), PSD
  sheaf_dirichlet_energy(sheaf, x)
                              — `x^T L_F x`, batched-first

Restricted to **node-edge sheaves on graphs** in v1; higher-dimensional
cellular sheaves on simplicial complexes (with 2-cells / faces and
the corresponding ∂_1 ∘ ∂_2 = 0 chain identity) is planned.

References:
  Hansen, J., Ghrist, R. (2019). Toward a spectral theory of cellular
    sheaves. Journal of Applied and Computational Topology 3:315–358.
    The canonical reference for sheaf Laplacians and their spectral
    theory.
  Bodnar, C., Di Giovanni, F., Chamberlain, B. P., Liò, P., Bronstein,
    M. M. (2022). Neural sheaf diffusion: A topological perspective on
    heterophily and oversmoothing in GNNs. NeurIPS 2022. Brought
    sheaf Laplacians into mainstream geometric deep learning.
  Curry, J. (2014). Sheaves, Cosheaves and Applications. PhD thesis,
    University of Pennsylvania. The applied-topology textbook on
    cellular sheaves; introduces the explicit cellular formulation
    that all the computational work uses.
"""

from holonomy_lib.sheaf.graph_sheaf import GraphSheaf, HeterogeneousGraphSheaf
from holonomy_lib.sheaf.laplacian import (
    sheaf_bottom_spectrum,
    sheaf_coboundary,
    sheaf_coboundary_sparse,
    sheaf_dirichlet_energy,
    sheaf_laplacian,
    sheaf_laplacian_sparse,
)

__all__ = [
    "GraphSheaf",
    "HeterogeneousGraphSheaf",
    "sheaf_bottom_spectrum",
    "sheaf_coboundary",
    "sheaf_coboundary_sparse",
    "sheaf_dirichlet_energy",
    "sheaf_laplacian",
    "sheaf_laplacian_sparse",
]
