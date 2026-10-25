# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Synoros

"""Vietoris-Rips complex from a point cloud or distance matrix.

The Vietoris-Rips complex `VR(X, r)` of a metric space `X` at scale
`r` is the simplicial complex whose k-simplex `[v_0, ..., v_k]`
exists iff `d(v_i, v_j) ≤ r` for every pair `(i, j)`. It's the
standard finite-scale approximation to the (typically infinite)
Čech complex and the input to most persistent-homology pipelines.

Construction extends each existing simplex by larger vertex indices.
Candidates are checked in configurable tiles; no batch, vertex, or simplex
is traversed individually in Python. Dimension progression is sequential
because a k-simplex is extended from a (k-1)-simplex. The worst case still
has combinatorial output size. Tiles bound candidate working memory, not
the size of the returned complex.

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

# One binary kibicandidate per tile bounds the temporary integer and bool
# tables to O(1024 * simplex_width) entries. This is a memory budget, not
# an accuracy parameter; callers can tune it for their device.
VR_CANDIDATE_TILE_DEFAULT = 1024


def _rips_tables(
    distances: torch.Tensor, max_radius: float, max_dim: int,
    candidate_tile_size: int,
) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    """Return packed (batch id, simplex) tensors for each dimension."""
    if distances.ndim != 2 + 1 or distances.shape[-1] != distances.shape[-2]:
        raise ValueError(
            f"distances must be (B, n, n); got {tuple(distances.shape)}"
        )
    if not distances.is_floating_point():
        raise ValueError("distances must have a real floating dtype")
    if not isinstance(max_dim, int) or max_dim < 0:
        raise ValueError(f"max_dim must be a nonnegative integer, got {max_dim}")
    if not isinstance(candidate_tile_size, int) or candidate_tile_size <= 0:
        raise ValueError("candidate_tile_size must be a positive integer")
    if max_radius < 0:
        raise ValueError("max_radius must be nonnegative")

    batch_size, n_vertices, _ = distances.shape
    device = distances.device
    vertex_ids = torch.arange(n_vertices, device=device)
    batch_ids = torch.arange(batch_size, device=device)
    tables = {0: (
        batch_ids.repeat_interleave(n_vertices),
        vertex_ids.expand(batch_size, -1).reshape(-1, 1),
    )}
    if max_dim == 0:
        return tables

    adjacency = distances <= max_radius
    edge_batch, edge_left, edge_right = torch.where(
        adjacency & (vertex_ids[:, None] < vertex_ids[None, :]),
    )
    tables[1] = (edge_batch, torch.stack((edge_left, edge_right), dim=-1))
    for dim in range(2, max_dim + 1):
        previous_batch, previous = tables[dim - 1]
        chunks: list[torch.Tensor] = []
        batch_chunks: list[torch.Tensor] = []
        # Each tile contains independent extension candidates. The strictly
        # larger final vertex gives every simplex one canonical parent.
        n_candidates = previous.shape[0] * n_vertices
        for start in range(0, n_candidates, candidate_tile_size):
            candidate_ids = torch.arange(
                start, min(start + candidate_tile_size, n_candidates),
                device=device,
            )
            parent_ids = candidate_ids // n_vertices
            extra_vertices = candidate_ids % n_vertices
            ordered = extra_vertices > previous[parent_ids, -1]
            parent_ids = parent_ids[ordered]
            extra_vertices = extra_vertices[ordered]
            parents = previous[parent_ids]
            batches = previous_batch[parent_ids]
            cofaceable = adjacency[
                batches[:, None], parents, extra_vertices[:, None],
            ].all(dim=-1)
            chunks.append(torch.cat((
                parents[cofaceable], extra_vertices[cofaceable, None],
            ), dim=-1))
            batch_chunks.append(batches[cofaceable])
        tables[dim] = (
            torch.cat(batch_chunks) if batch_chunks else batch_ids[:0],
            torch.cat(chunks) if chunks else torch.empty(
                (0, dim + 1), device=device, dtype=torch.int64,
            ),
        )
    return tables


def vietoris_rips_sparse(
    distances: torch.Tensor,
    max_radius: float,
    max_dim: int,
    *,
    candidate_tile_size: int = VR_CANDIDATE_TILE_DEFAULT,
) -> SparseSimplicialComplex:
    """Vietoris-Rips complex from a single distance matrix.

    Args:
      distances: `(n, n)` symmetric real distance matrix.
      max_radius: include only pairs with `d <= max_radius`.
      max_dim: build simplex tables through this dimension, including
        empty tables where no simplex exists.
      candidate_tile_size: maximum number of simplex-extension candidates
        evaluated at once. It bounds temporary storage, not output size.

    Returns:
      Single-instance complex on the input device.

    Notes:
      Dimensions are constructed sequentially because each extension uses
      the preceding dimension. Candidates within a tile are tensorized.

    References:
      Hausmann, J.-C. (1995). On the Vietoris-Rips complexes and a
        cohomology theory for metric spaces. In Prospects in Topology,
        Annals of Math Studies 138.
      Bauer, U. (2021). Ripser: efficient computation of Vietoris-Rips
        persistence barcodes. J. Appl. Comput. Topology 5:391-423.
    """
    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError(
            f"distances must be (n, n); got {tuple(distances.shape)}"
        )
    tables = _rips_tables(
        distances.unsqueeze(0), max_radius, max_dim, candidate_tile_size,
    )
    return SparseSimplicialComplex(
        simplices_by_dim={dim: table for dim, (_, table) in tables.items()},
        n_vertices=distances.shape[-1], device=distances.device,
    )


def vietoris_rips_dense(
    distances: torch.Tensor,
    max_radius: float,
    max_dim: int,
    dtype: torch.dtype | None = None,
    *,
    candidate_tile_size: int = VR_CANDIDATE_TILE_DEFAULT,
) -> DenseSimplicialComplex:
    """Batched Vietoris-Rips construction with padded simplex tables.

    Args:
      distances: `(B, n, n)` symmetric real distance matrices.
      max_radius: include pairs with `d <= max_radius`.
      max_dim: build tables through this dimension, including empty ones.
      dtype: boundary-operator dtype; defaults to the distances' dtype.
      candidate_tile_size: maximum number of simplex-extension candidates
        evaluated at once. It bounds temporary storage, not output size.

    Returns:
      A complex with per-batch padded simplex tables on the input device.
      Empty batches retain all requested dimensions.

    Notes:
      Batch members share tensor candidate evaluation and packing. Only
      dimensions and memory tiles are traversed in Python. Dimension
      progression is sequential because extensions require their parents.

    References:
      Hausmann, J.-C. (1995). On the Vietoris-Rips complexes and a
        cohomology theory for metric spaces.
      Bauer, U. (2021). Ripser: efficient computation of Vietoris-Rips
        persistence barcodes. J. Appl. Comput. Topology 5:391-423.
    """
    tables = _rips_tables(distances, max_radius, max_dim, candidate_tile_size)
    batch_size = distances.shape[0]
    device = distances.device
    simplices_by_dim: dict[int, torch.Tensor] = {}
    valid_mask: dict[int, torch.Tensor] = {}
    for dim, (batches, table) in tables.items():
        counts = torch.bincount(batches, minlength=batch_size)
        # Padding has a data-dependent width, so one size synchronization
        # per dimension is needed before allocating its output table.
        width = int(counts.max()) if batch_size else 0
        if dim == 0:
            width = distances.shape[-1]
        padded = torch.full(
            (batch_size, width, dim + 1), -1,
            dtype=torch.int64, device=device,
        )
        mask = torch.zeros(batch_size, width, dtype=torch.bool, device=device)
        offsets = counts.cumsum(0) - counts
        columns = torch.arange(table.shape[0], device=device) - offsets[batches]
        padded[batches, columns] = table
        mask[batches, columns] = True
        simplices_by_dim[dim] = padded
        valid_mask[dim] = mask
    return DenseSimplicialComplex(
        simplices_by_dim=simplices_by_dim, valid_mask=valid_mask,
        n_vertices=distances.shape[-1], device=device,
        dtype=distances.dtype if dtype is None else dtype,
    )


def pairwise_distances(points: torch.Tensor) -> torch.Tensor:
    """Pairwise Euclidean distances.

    Args:
      points: `(n, d)` or `(B, n, d)`.

    Returns:
      `(n, n)` or `(B, n, n)` distance matrix on the input device and dtype.

    Notes:
      Half and bfloat16 inputs use float32 intermediates because cdist's
      direct CPU kernel does not support these dtypes.

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
    if not points.is_floating_point():
        raise ValueError("points must have a real floating dtype")
    # cdist's CPU kernel only accepts float32/float64. Evaluate low-precision
    # inputs in float32 and return their original dtype on the same device.
    working = (
        points.float() if points.dtype in (torch.float16, torch.bfloat16) else points
    )
    # The direct kernel avoids an (B,n,n,d) difference tensor and the
    # matrix-multiplication identity's cancellation near coincident points.
    return torch.cdist(
        working, working, compute_mode="donot_use_mm_for_euclid_dist",
    ).to(points.dtype)
