# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Synoros

"""Persistent homology of Vietoris-Rips filtrations.

For a point cloud `X`, the Vietoris-Rips filtration is the family of
simplicial complexes `{VR(X, r) : r ≥ 0}` indexed by scale `r`. As
`r` grows, simplices are born; the persistent-homology pipeline
tracks which topological features (connected components, loops,
voids, ...) survive across scales.

This module exports a single public function

  persistence_diagrams(points, max_dim=2, max_radius=inf)
      → (diagrams, masks)

where `diagrams[k]` is a `(B, max_pairs_k, 2)` batched padded tensor
of `(birth, death)` pairs for the k-th persistence diagram, and
`masks[k]` is the matching `(B, max_pairs_k)` validity mask. Bars
that survive to the end of the filtration have `death = +inf`.

Two computational paths:

  H₀ — batched union-find on sorted filtration edges. Each merge
  records the death time of the "younger" component (the component
  whose latest vertex appeared later in the filtration, by
  convention). The persistent connected component that survives gets
  a `(birth, +inf)` bar.

  H_{1..max_dim} — Z/2 left-to-right reduction of the boundary
  matrix in filtration order (Edelsbrunner-Letscher-Zomorodian
  2002), via `_reduction.reduce_filtration`. Sequential within one
  filtration but batches across point clouds: the public function
  loops over batch elements, running one reduction each.

References:
  Edelsbrunner, H., Letscher, D., Zomorodian, A. (2002). Topological
    persistence and simplification. Discrete & Computational
    Geometry 28:511-533. Foundational PH paper.
  Cohen-Steiner, D., Edelsbrunner, H., Harer, J. (2007). Stability
    of persistence diagrams. Discrete & Computational Geometry
    37(1):103-120. Stability under perturbation of input data.
  Edelsbrunner, H., Harer, J. (2010). Computational Topology: An
    Introduction. AMS. Reference textbook.
  Bauer, U. (2021). Ripser: efficient computation of Vietoris-Rips
    persistence barcodes. J. Appl. Comput. Topology 5:391-423.
"""

from __future__ import annotations

import torch

from holonomy_lib.provenance import with_provenance
from holonomy_lib.simplicial import (
    pairwise_distances,
    vietoris_rips_sparse,
)
from holonomy_lib.topology._filtration import build_filtration
from holonomy_lib.topology._reduction import reduce_filtration


@with_provenance(
    "holonomy_lib.topology.persistence_diagrams", op_version="0.2",
)
def persistence_diagrams(
    points_or_distances: torch.Tensor,
    max_dim: int = 2,
    max_radius: float = float("inf"),
    input_is_distance: bool = False,
    reduction_backend: str = "python",
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Persistence diagrams for dims 0..max_dim of a Vietoris-Rips
    filtration.

    Args:
      points_or_distances: either `(B, n, d)` point clouds or
        `(B, n, n)` distance matrices. Set `input_is_distance=True`
        for the latter.
      max_dim: highest dim to compute. Common choices: 1 (clusters +
        loops) or 2 (clusters + loops + voids).
      max_radius: build the filtration only up to this scale. Bars
        born after this radius are not recorded. Default `+inf`
        builds the full filtration.
      input_is_distance: if True, `points_or_distances` is read as a
        distance matrix directly; otherwise pairwise Euclidean
        distances are computed.
      reduction_backend: `"python"` (default; CPython set columns,
        fast for typical inputs) or `"torch"` (LongTensor columns,
        device-agnostic — runs on the filtration's native device).
        See `_reduction.reduce_filtration` for the tradeoff.

    Returns:
      `(diagrams, masks)` where:
        diagrams[k]: `(B, max_pairs_k, 2)` float Tensor of
          `(birth, death)` pairs. Bars surviving to infinity have
          `death = +inf`.
        masks[k]: `(B, max_pairs_k)` bool Tensor; True for real
          pairs, False for padding.

      Both lists have length `max_dim + 1`.

    Notes:
      Cost is dominated by the matrix reduction for `max_dim ≥ 1`:
      O(B · m³) worst case where `m` is the number of simplices in
      the filtration, but typically much faster due to sparsity. For
      large point clouds, lower `max_radius` or `max_dim` to keep
      the simplex count manageable.

      Comparison vs `ripser`: tie-breaking on equal birth/death
      values may differ; the persistence diagrams agree under
      bottleneck distance.

    References:
      Edelsbrunner-Letscher-Zomorodian (2002).
      Cohen-Steiner-Edelsbrunner-Harer (2007) stability.
    """
    if max_dim < 0:
        raise ValueError(f"max_dim must be >= 0, got {max_dim}")
    if points_or_distances.ndim != 3:
        raise ValueError(
            f"points_or_distances must be 3-D (B, ...); got "
            f"{tuple(points_or_distances.shape)}"
        )
    B = points_or_distances.shape[0]

    if input_is_distance:
        if points_or_distances.shape[-1] != points_or_distances.shape[-2]:
            raise ValueError(
                f"input_is_distance=True requires (B, n, n); got "
                f"{tuple(points_or_distances.shape)}"
            )
        distances = points_or_distances
    else:
        distances = pairwise_distances(points_or_distances)

    # Per-batch lists of (birth, death) per dim.
    pairs_per_batch_per_dim: list[list[list[tuple[float, float]]]] = [
        [[] for _ in range(max_dim + 1)] for _ in range(B)
    ]

    for b in range(B):
        d_b = distances[b]
        n = d_b.shape[0]

        # H_0 — union-find on sorted filtration edges.
        h0_pairs = _persistent_h0(d_b, max_radius)
        pairs_per_batch_per_dim[b][0] = h0_pairs

        # H_{1..max_dim} — VR complex + matrix reduction. Skip the
        # reduction when max_dim == 0 (H_0 is already handled via
        # union-find above); for max_dim >= 1, the reduction itself
        # naturally produces empty diagrams when the point cloud has
        # too few vertices to form (max_dim+1)-simplices, so no
        # additional vertex-count guard is needed.
        if max_dim >= 1:
            complex = vietoris_rips_sparse(
                d_b, max_radius=max_radius, max_dim=max_dim + 1,
            )
            filt = build_filtration(d_b, complex)
            pairs_by_dim = reduce_filtration(filt, backend=reduction_backend)
            # The reduction returns H_0 as well, but our union-find
            # path produced cleaner H_0 pairs (no spurious
            # zero-length bars from tied filtration values); we
            # discard the reduction's H_0 and use ours.
            for k in range(1, max_dim + 1):
                pairs_per_batch_per_dim[b][k] = pairs_by_dim.get(k, [])

    # Pad per-dim into batched tensors.
    diagrams: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    for k in range(max_dim + 1):
        max_pairs_k = max(
            len(pairs_per_batch_per_dim[b][k]) for b in range(B)
        ) if B > 0 else 0
        if max_pairs_k == 0:
            diagrams.append(
                torch.zeros(B, 0, 2, dtype=distances.dtype,
                             device=distances.device)
            )
            masks.append(
                torch.zeros(B, 0, dtype=torch.bool, device=distances.device)
            )
            continue
        diag = torch.zeros(
            B, max_pairs_k, 2,
            dtype=distances.dtype, device=distances.device,
        )
        mask = torch.zeros(
            B, max_pairs_k, dtype=torch.bool, device=distances.device,
        )
        for b in range(B):
            for j, (birth, death) in enumerate(pairs_per_batch_per_dim[b][k]):
                diag[b, j, 0] = birth
                diag[b, j, 1] = death
                mask[b, j] = True
        diagrams.append(diag)
        masks.append(mask)

    return diagrams, masks


# ============================================================
# H_0 via batched union-find on sorted filtration edges
# ============================================================


def _persistent_h0(
    distances: torch.Tensor, max_radius: float,
) -> list[tuple[float, float]]:
    """Persistent H_0 for a single distance matrix.

    Returns a list of `(birth, death)` pairs. All `n` connected
    components are born at time 0 (the vertices). `n − 1` of them
    die when union-find merges them with an earlier component at
    some edge length. One component survives to infinity (the
    persistent connected component).
    """
    n = distances.shape[0]
    if n == 0:
        return []
    if n == 1:
        # Single vertex: one infinite bar.
        return [(0.0, float("inf"))]

    # Strictly upper-triangular edges (i, j) with i < j and d <= max_radius.
    iu, ju = torch.triu_indices(n, n, offset=1, device=distances.device)
    edge_lengths = distances[iu, ju]
    if max_radius != float("inf"):
        keep = edge_lengths <= max_radius
        iu, ju, edge_lengths = iu[keep], ju[keep], edge_lengths[keep]

    # Sort edges by length, stable for deterministic tie-breaking.
    sort_idx = torch.argsort(edge_lengths, stable=True)
    iu = iu[sort_idx].tolist()
    ju = ju[sort_idx].tolist()
    edge_lengths = edge_lengths[sort_idx].tolist()

    # Disjoint-set union with path compression + union-by-rank.
    parent = list(range(n))
    rank = [0] * n

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # path compression
            x = parent[x]
        return x

    pairs: list[tuple[float, float]] = []
    for u, v, length in zip(iu, ju, edge_lengths):
        ru, rv = find(u), find(v)
        if ru == rv:
            continue
        # Union: the component with the smaller rank (= "younger" by
        # convention) dies. For ties we pick the one with the larger
        # root index, matching ripser-style behavior.
        if rank[ru] < rank[rv]:
            parent[ru] = rv
        elif rank[ru] > rank[rv]:
            parent[rv] = ru
        else:
            parent[rv] = ru
            rank[ru] += 1
        # Record the death of the younger component. Both components
        # had birth = 0; the younger dies at this edge's length.
        pairs.append((0.0, length))

    # The persistent component survives to infinity.
    n_merged = len(pairs)
    n_surviving = n - n_merged
    for _ in range(n_surviving):
        pairs.append((0.0, float("inf")))

    return pairs
