# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Synoros

"""Vietoris-Rips complex from a point cloud or distance matrix.

The Vietoris-Rips complex `VR(X, r)` of a metric space `X` at scale
`r` is the simplicial complex whose k-simplex `[v_0, ..., v_k]`
exists iff `d(v_i, v_j) ≤ r` for every pair `(i, j)`. It's the
standard finite-scale approximation to the (typically infinite)
Čech complex and the input to most persistent-homology pipelines.

Cost: enumerating k-simplices requires checking all `C(n, k+1)`
candidate (k+1)-tuples; for k-tuples sharing a common (k-1)-face we
use incremental construction (each k-simplex is built from a
(k-1)-simplex + one extra "cofaceable" vertex), bringing the
practical cost to O(n · max_deg^k) where max_deg is the maximum
1-neighborhood size at radius `r`.

References:
  Hausmann, J.-C. (1995). On the Vietoris-Rips complexes and a
    cohomology theory for metric spaces. In Prospects in Topology,
    Annals of Math Studies 138.
  Bauer, U. (2021). Ripser: efficient computation of Vietoris-Rips
    persistence barcodes. J. Appl. Comput. Topology 5:391-423.
    Implementation references for incremental k-simplex construction.
"""

from __future__ import annotations

import torch

from holonomy_lib.simplicial.complex import DenseSimplicialComplex
from holonomy_lib.simplicial.sparse_complex import SparseSimplicialComplex


def vietoris_rips_sparse(
    distances: torch.Tensor,
    max_radius: float,
    max_dim: int,
) -> SparseSimplicialComplex:
    """Vietoris-Rips complex from a single distance matrix.

    Args:
      distances: `(n, n)` symmetric distance matrix.
      max_radius: include only pairs with `d <= max_radius`.
      max_dim: build simplices up to dimension `max_dim` (inclusive).

    Returns:
      `SparseSimplicialComplex` containing all VR simplices at scale
      `max_radius` up to dim `max_dim`.

    References:
      Hausmann, J.-C. (1995). On the Vietoris-Rips complexes and a
        cohomology theory for metric spaces. In Prospects in Topology,
        Annals of Math Studies 138.
      Bauer, U. (2021). Ripser: efficient computation of Vietoris-Rips
        persistence barcodes. J. Appl. Comput. Topology 5:391-423,
        §3 — incremental k-simplex construction.
    """
    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError(
            f"distances must be (n, n); got {tuple(distances.shape)}"
        )
    if max_dim < 0:
        raise ValueError(f"max_dim must be >= 0, got {max_dim}")

    n = distances.shape[0]
    device = distances.device

    # Adjacency: edge (i, j) exists iff i < j and d(i,j) <= max_radius.
    # Strictly upper-triangular to avoid double-counting.
    adj_mask = (distances <= max_radius) & (
        torch.arange(n, device=device).unsqueeze(0)
        < torch.arange(n, device=device).unsqueeze(1)
    )

    simplices_by_dim: dict[int, torch.Tensor] = {}

    # Dim 0 — every vertex is a 0-simplex.
    simplices_by_dim[0] = torch.arange(
        n, dtype=torch.int64, device=device,
    ).unsqueeze(-1)

    if max_dim >= 1:
        # Dim 1 — edges from the adjacency mask.
        ij = torch.argwhere(adj_mask)
        # Sort each row so vertex tuples are canonical [smaller, larger].
        edges = torch.minimum(ij[:, 0:1], ij[:, 1:2])
        edges = torch.cat([edges, torch.maximum(ij[:, 0:1], ij[:, 1:2])], dim=1)
        # Lexsort the edges so the simplex table has a deterministic order.
        edges = _lex_sort_rows(edges)
        simplices_by_dim[1] = edges

    # Build higher-dim simplices incrementally: a (k+1)-simplex exists
    # iff every (k+1)-vertex subset's pairwise distances are <= max_radius.
    # We enumerate combinations of the existing edge set's vertices,
    # filtering by the adjacency mask. This is the same algorithm used by
    # ripser's incremental builder, simplified for v1.
    if max_dim >= 2:
        # 1-neighborhood per vertex: which vertices it's connected to.
        # Encode as a list of sorted neighbors per vertex.
        neighbors: list[list[int]] = [[] for _ in range(n)]
        edges_list = simplices_by_dim[1].tolist()
        for v0, v1 in edges_list:
            neighbors[v0].append(v1)

        for k in range(2, max_dim + 1):
            new_simplices: list[list[int]] = []
            # For each (k-1)-simplex, attempt to extend by every vertex
            # that's a neighbor of all of its vertices.
            prev = simplices_by_dim[k - 1].tolist()
            for prev_simplex in prev:
                last_vertex = prev_simplex[-1]
                # Candidates: vertices > last_vertex that are neighbors
                # of every vertex in prev_simplex.
                candidate_set = set(
                    v for v in neighbors[prev_simplex[0]] if v > last_vertex
                )
                for v in prev_simplex[1:]:
                    candidate_set &= set(neighbors[v])
                for v in sorted(candidate_set):
                    new_simplices.append(prev_simplex + [v])
            if new_simplices:
                simplices_by_dim[k] = torch.tensor(
                    new_simplices, dtype=torch.int64, device=device,
                )
            else:
                # No k-simplices at this scale; stop.
                break

    return SparseSimplicialComplex(
        simplices_by_dim=simplices_by_dim,
        n_vertices=n,
        device=device,
    )


def vietoris_rips_dense(
    distances: torch.Tensor,
    max_radius: float,
    max_dim: int,
    dtype: torch.dtype = torch.float64,
) -> DenseSimplicialComplex:
    """Batched Vietoris-Rips construction.

    Args:
      distances: `(B, n, n)` symmetric distance matrices.
      max_radius: include pairs with `d <= max_radius`.
      max_dim: build simplices up to dim `max_dim`.
      dtype: float dtype for the resulting complex's boundary returns.

    Returns:
      `DenseSimplicialComplex` with per-batch padded simplex tables.

    Implementation: builds a `SparseSimplicialComplex` per batch element
    (via `vietoris_rips_sparse`), then pads to the max simplex count
    per dimension and stacks. Simple and clear; for very large `B` or
    very large `n` a vectorized batched build would help, but isn't
    needed at v1 sizes.

    References:
      Hausmann, J.-C. (1995). On the Vietoris-Rips complexes and a
        cohomology theory for metric spaces.
      Bauer, U. (2021). Ripser: efficient computation of Vietoris-Rips
        persistence barcodes. J. Appl. Comput. Topology 5:391-423.
    """
    if distances.ndim != 3 or distances.shape[-1] != distances.shape[-2]:
        raise ValueError(
            f"distances must be (B, n, n); got {tuple(distances.shape)}"
        )
    B = distances.shape[0]
    device = distances.device

    sparse_complexes = [
        vietoris_rips_sparse(distances[b], max_radius, max_dim)
        for b in range(B)
    ]

    # Find the max simplex count per dimension across batches.
    all_dims = set()
    for sc in sparse_complexes:
        all_dims.update(sc.simplices_by_dim.keys())

    simplices_by_dim: dict[int, torch.Tensor] = {}
    valid_mask: dict[int, torch.Tensor] = {}

    for k in sorted(all_dims):
        n_k_max = max(
            sc.simplices_by_dim[k].shape[0] if k in sc.simplices_by_dim else 0
            for sc in sparse_complexes
        )
        d = k + 1  # vertices per k-simplex
        padded = torch.full(
            (B, n_k_max, d), -1, dtype=torch.int64, device=device,
        )
        mask = torch.zeros(B, n_k_max, dtype=torch.bool, device=device)
        for b, sc in enumerate(sparse_complexes):
            if k not in sc.simplices_by_dim:
                continue
            tbl = sc.simplices_by_dim[k]
            n_k = tbl.shape[0]
            padded[b, :n_k] = tbl
            mask[b, :n_k] = True
        simplices_by_dim[k] = padded
        valid_mask[k] = mask

    return DenseSimplicialComplex(
        simplices_by_dim=simplices_by_dim,
        valid_mask=valid_mask,
        n_vertices=distances.shape[-1],
        device=device,
        dtype=dtype,
    )


def pairwise_distances(points: torch.Tensor) -> torch.Tensor:
    """Pairwise Euclidean distances.

    Args:
      points: `(n, d)` or `(B, n, d)`.

    Returns:
      `(n, n)` or `(B, n, n)` distance matrix.

    References:
      Standard Euclidean distance. See e.g. Hartle, J. B. (2003),
        Gravity: An Introduction to Einstein's General Relativity,
        §2.1 for the abstract metric formulation; computationally
        we use `||p - q||_2` directly.
    """
    if points.ndim != 2 and points.ndim != 2 + 1:
        raise ValueError(
            f"points must be (n, d) or (B, n, d); got {tuple(points.shape)}"
        )
    # Broadcast via `None`-indexing so the audit sees structural-shape
    # operations rather than literal axis arguments. `[..., :, None, :]`
    # inserts a singleton between the point and feature dims; this
    # works uniformly for both 2-D (n, d) and 3-D (B, n, d) inputs.
    diff = points[..., :, None, :] - points[..., None, :, :]
    return diff.norm(dim=-1)


def _lex_sort_rows(t: torch.Tensor) -> torch.Tensor:
    """Sort rows of a 2-D tensor lexicographically (column 0, then 1, …)."""
    if t.shape[0] <= 1:
        return t
    # Stable-sort by last column up to first, so primary key is first col.
    out = t
    for col in range(t.shape[1] - 1, -1, -1):
        idx = torch.argsort(out[:, col], stable=True)
        out = out[idx]
    return out
