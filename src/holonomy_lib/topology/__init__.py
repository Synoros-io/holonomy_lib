# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Synoros

"""holonomy_lib.topology: Hodge Laplacians + persistent homology.

Built on `holonomy_lib.simplicial`. Two flavors of output:

  Hodge / Betti
    `hodge_laplacian(complex, k)`; the Hodge Laplacian on k-chains,
    `L_k = ∂_{k+1} ∂_{k+1}^T + ∂_k^T ∂_k`. Its kernel has dimension
    equal to the k-th Betti number.
    `betti_numbers(complex, max_dim)`; computes
    `β_k = n_k - rank(∂_k) - rank(∂_{k+1})` from float64 boundary ranks,
    giving `(β_0, β_1, …, β_max_dim)`. It does not form `L_k`.

  Persistent homology
    `persistence_diagrams(points_or_distances, max_dim)` returns the
    per-dim birth/death barcodes for a Vietoris-Rips filtration.
"""

from holonomy_lib.topology.hodge import betti_numbers, hodge_laplacian
from holonomy_lib.topology.persistent_homology import persistence_diagrams

__all__ = [
    "betti_numbers",
    "hodge_laplacian",
    "persistence_diagrams",
]
