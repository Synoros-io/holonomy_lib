# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Synoros

"""holonomy_lib.simplicial: simplicial complexes + boundary operators
+ Vietoris-Rips construction.

Used as the foundation for `holonomy_lib.topology` (Hodge Laplacians
and persistent homology). Two representations:

  DenseSimplicialComplex(simplices_by_dim, valid_mask, n_vertices, ...)
    Batched, with padded simplex tables and a validity mask. Boundary
    operators return dense `(B, n_{k-1}_max, n_k_max)` tensors.

  SparseSimplicialComplex(simplices_by_dim, n_vertices, ...)
    Single-instance (no batch dim). Boundary operators return
    `torch.sparse_csc_tensor` of shape `(n_{k-1}, n_k)`. Used by
    persistent homology, where the matrix-reduction kernel walks the
    sparse boundary column by column.

Construction:

  vietoris_rips_sparse(distances, max_radius, max_dim)
    Single `(n, n)` distance matrix → SparseSimplicialComplex.

  vietoris_rips_dense(distances, max_radius, max_dim, dtype=...)
    Batched `(B, n, n)` distance matrices → DenseSimplicialComplex
    with per-batch padded tables.

  pairwise_distances(points)
    Helper: Euclidean distance matrix from a `(n, d)` or `(B, n, d)`
    point cloud.

References:
  Munkres, J. R. (1984). Elements of Algebraic Topology. Westview
    Press. §1 — chain complex + boundary operator.
  Hausmann, J.-C. (1995). On the Vietoris-Rips complexes and a
    cohomology theory for metric spaces. In Prospects in Topology,
    Annals of Math Studies 138.
"""

from holonomy_lib.simplicial.complex import DenseSimplicialComplex
from holonomy_lib.simplicial.sparse_complex import SparseSimplicialComplex
from holonomy_lib.simplicial.vietoris_rips import (
    pairwise_distances,
    vietoris_rips_dense,
    vietoris_rips_sparse,
)

__all__ = [
    "DenseSimplicialComplex",
    "SparseSimplicialComplex",
    "pairwise_distances",
    "vietoris_rips_dense",
    "vietoris_rips_sparse",
]
