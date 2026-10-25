# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Synoros

"""Persistent homology of Vietoris-Rips filtrations.

For a point cloud `X`, the Vietoris-Rips filtration is the family of
simplicial complexes `{VR(X, r) : r ≥ 0}` indexed by scale `r`. As
`r` grows, simplices are born; the persistent-homology pipeline
tracks which topological features (connected components, loops,
voids, ...) survive across scales.

This module exports a single public function

  persistence_diagrams(points_or_distances, max_dim=2, max_radius=inf,
      input_is_distance=False, reduction_backend="python")
      → (diagrams, masks)

where `diagrams[k]` is a `(B, max_pairs_k, 2)` batched padded tensor
of `(birth, death)` pairs for the k-th persistence diagram, and
`masks[k]` is the matching `(B, max_pairs_k)` validity mask. Bars
that survive to the end of the filtration have `death = +inf`. With a finite
`max_radius` the filtration ends at that radius, so bars still alive there
get `death = +inf` as well.

Two computational paths:

  H₀: sequential host union-find on sorted filtration edges, dispatched
  per batch member. All components are born at zero; each merge records
  one death and every remaining component gets a `(0, +inf)` bar.

  H_{1..max_dim}: Z/2 left-to-right reduction of the boundary
  matrix in filtration order (Edelsbrunner-Letscher-Zomorodian
  2002), via `_reduction.reduce_filtration`. Sequential within one
  filtration. Simplex enumeration is batched across point clouds, but the
  public function runs one reduction per batch element.

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

from typing import Literal

import torch

from holonomy_lib.simplicial import (
    SparseSimplicialComplex,
    pairwise_distances,
    vietoris_rips_dense,
)
from holonomy_lib.topology._filtration import build_filtration
from holonomy_lib.topology._reduction import reduce_filtration


def persistence_diagrams(
    points_or_distances: torch.Tensor,
    max_dim: int = 2,
    max_radius: float = float("inf"),
    input_is_distance: bool = False,
    reduction_backend: Literal["python", "torch"] = "python",
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Persistence diagrams for dims 0..max_dim of a Vietoris-Rips filtration.

    Args:
      points_or_distances: either `(B, n, d)` point clouds or
        `(B, n, n)` distance matrices. Set `input_is_distance=True`
        for the latter.
      max_dim: highest dim to compute. Common choices: 1 (clusters +
        loops) or 2 (clusters + loops + voids).
      max_radius: build the filtration only up to this scale. Bars
        born after this radius are not recorded, and bars still alive at
        this radius get `death = +inf`. Default `+inf` builds the full
        filtration.
      input_is_distance: if True, `points_or_distances` is read as a
        distance matrix directly; otherwise pairwise Euclidean
        distances are computed.
      reduction_backend: `"python"` (default; CPython set columns)
        or `"torch"` (LongTensor columns on the filtration's device).
        See `_reduction.reduce_filtration` for the tradeoff.

    Returns:
      `(diagrams, masks)` where:
        diagrams[k]: `(B, max_pairs_k, 2)` float Tensor of
          `(birth, death)` pairs. Bars surviving to infinity have
          `death = +inf`.
        masks[k]: `(B, max_pairs_k)` bool Tensor; True for real
          pairs, False for padding.

      Both lists have length `max_dim + 1`. H_0 retains one bar per input
      vertex, including zero-length bars when coincident vertices merge.
      Higher-dimensional diagrams omit zero-length bars.

    Notes:
      Cost is dominated by the matrix reduction for `max_dim ≥ 1`:
      O(B · m³) worst case where `m` is the number of simplices in
      the filtration, but typically much faster due to sparsity. For
      large point clouds, lower `max_radius` or `max_dim` to keep
      the simplex count manageable.

      Simplex construction is batched. Z/2 pivot reduction and H_0 union-find
      are sequential within each filtration: every pivot or component merge
      depends on previous ones. Ragged filtrations are dispatched separately.
      Python reduction is an explicit host boundary; torch reduction keeps
      columns on device but synchronizes for dependent pivot decisions. Bar
      coordinates are returned without an autograd connection to the inputs.

      Comparison vs `ripser`: tie-breaking on equal birth/death
      values may differ; the persistence diagrams agree under
      bottleneck distance.

    References:
      Edelsbrunner-Letscher-Zomorodian (2002).
      Cohen-Steiner-Edelsbrunner-Harer (2007) stability.
    """
    if points_or_distances.ndim != 3:
        raise ValueError(
            f"points_or_distances must be 3-D (B, ...); got "
            f"{tuple(points_or_distances.shape)}"
        )
    if not points_or_distances.is_floating_point():
        raise ValueError("points_or_distances must have real floating dtype")
    if max_dim < 0:
        raise ValueError(f"max_dim must be >= 0, got {max_dim}")
    if max_radius < 0:
        raise ValueError(f"max_radius must be >= 0, got {max_radius}")
    if reduction_backend not in ("python", "torch"):
        raise ValueError(f"unknown reduction_backend={reduction_backend!r}")
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

    # pairs_per_batch_per_dim[b][k] collects the (birth, death) pairs of batch
    # member b in dimension k.
    pairs_per_batch_per_dim: list[list[list[tuple[float, float]]]] = [
        [[] for _ in range(max_dim + 1)] for _ in range(B)
    ]

    # All batch members share one tensorized simplex enumeration. The reduction
    # is sequential, so it runs once per member on that member's filtration.
    complexes = vietoris_rips_dense(
        distances, max_radius=max_radius, max_dim=max_dim + 1,
    ) if max_dim >= 1 else None
    for b in range(B):
        d_b = distances[b]

        # H_0: union-find on sorted filtration edges.
        h0_pairs = _persistent_h0(d_b, max_radius)
        pairs_per_batch_per_dim[b][0] = h0_pairs

        # H_{1..max_dim}: build the VR complex and reduce its boundary matrix.
        # With max_dim == 0 there is no complex, so only union-find runs.
        # Otherwise the reduction runs on whatever simplices exist, and a
        # dimension with none adds no bars, so small clouds need no extra guard.
        if complexes is not None:
            complex = SparseSimplicialComplex(
                {k: table[b][complexes.valid_mask[k][b]]
                 for k, table in complexes.simplices_by_dim.items()},
                n_vertices=complexes.n_vertices, device=complexes.device,
            )
            filt = build_filtration(d_b, complex)
            pairs_by_dim = reduce_filtration(filt, backend=reduction_backend)
            # H_0 already came from union-find, so only dims 1..max_dim are
            # taken from this reduction.
            for k in range(1, max_dim + 1):
                pairs_per_batch_per_dim[b][k] = pairs_by_dim.get(k, [])

    # Pad per-dim into batched tensors.
    diagrams: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    for k in range(max_dim + 1):
        max_pairs_k = max(
            len(pairs_per_batch_per_dim[b][k]) for b in range(B)
        ) if B > 0 else 0
        # No member has a bar in this dimension. Return an empty (B, 0, 2)
        # tensor so callers can index every dimension the same way.
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
        # counts[b] is how many real pairs member b has in this dimension.
        counts = [len(member[k]) for member in pairs_per_batch_per_dim]
        # One host-to-device copy per dimension. Pairs are flattened in batch
        # order, so each (birth, death) stays whole.
        values = torch.tensor(
            [pair for member in pairs_per_batch_per_dim for pair in member[k]],
            dtype=distances.dtype, device=distances.device,
        )
        count_tensor = torch.tensor(counts, device=distances.device)
        # Row b's first counts[b] slots are real and the rest are padding.
        positions = torch.arange(max_pairs_k, device=distances.device)
        mask = positions < count_tensor[:, None]
        # Boolean-mask assignment fills the True slots row by row, which matches
        # the batch-order flattening of `values`.
        diag[mask] = values
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

    Takes an `(n, n)` distance matrix and the radius cap. Returns `n`
    `(birth, death)` pairs, one per vertex. Every component is born at 0.
    Each union-find merge kills one component at that edge length. A graph
    that is connected at the cap leaves one component alive to infinity, and
    a disconnected one leaves one essential bar per component. Edges are
    copied to the host in one bulk transfer. Union-find runs sequentially
    because each merge changes the partition that later edges see.
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

    # Sort edges by length. The stable sort breaks ties by upper-triangle
    # order, so the barcode is the same on every run.
    sort_idx = torch.argsort(edge_lengths, stable=True)
    sorted_u = iu[sort_idx].tolist()
    sorted_v = ju[sort_idx].tolist()
    sorted_lengths = edge_lengths[sort_idx].tolist()

    # Disjoint-set union with path compression + union-by-rank.
    parent = list(range(n))
    rank = [0] * n

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # path compression
            x = parent[x]
        return x

    pairs: list[tuple[float, float]] = []
    for u, v, length in zip(sorted_u, sorted_v, sorted_lengths, strict=True):
        ru, rv = find(u), find(v)
        if ru == rv:
            continue
        # All vertices are born at zero, so either representative gives
        # the same barcode. Rank only keeps the parent trees shallow.
        if rank[ru] < rank[rv]:
            parent[ru] = rv
        elif rank[ru] > rank[rv]:
            parent[rv] = ru
        else:
            parent[rv] = ru
            rank[ru] += 1
        # A successful merge kills one zero-born connected component.
        pairs.append((0.0, length))

    # Components that never merged survive to infinity. For a graph connected
    # at the cap, that is one component.
    n_merged = len(pairs)
    n_surviving = n - n_merged
    for _ in range(n_surviving):
        pairs.append((0.0, float("inf")))

    return pairs
