# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Synoros

"""Vietoris-Rips filtration construction for persistent homology.

A filtration is a totally-ordered sequence of simplices, each with
a birth time at which it joins the complex. For the Vietoris-Rips
filtration, the birth time of a simplex `σ` is the largest pairwise
distance among its vertices:

    birth(σ) = max_{i, j ∈ σ}  d(i, j).

Simplices with the same birth time are tie-broken by `(dim, input row
index)` so the ordering is total and deterministic. This module
returns the filtration in the order needed by the Z/2 boundary-matrix
reduction (`_reduction.py`): simplices listed in birth-time order,
with a per-simplex `(dim, original-index-within-dim)` tag so
boundary columns can be reconstructed.

References:
  Edelsbrunner, H., Harer, J. (2010). Computational Topology: An
    Introduction. American Mathematical Society. §VI.2; filtrations.
  Bauer, U. (2021). Ripser: efficient computation of Vietoris-Rips
    persistence barcodes. J. Appl. Comput. Topology 5:391-423.
    Filtration ordering conventions used here.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from holonomy_lib.simplicial import SparseSimplicialComplex


@dataclass(frozen=True)
class Filtration:
    """A Vietoris-Rips filtration with simplices in birth-time order.

    Attributes:
      complex: the underlying simplicial complex (sparse, single-instance).
      birth_times: `(n_total_simplices,)` float tensor, in filtration order.
      sorted_dims: `(n_total_simplices,)` int tensor; dim of each simplex.
      sorted_indices: `(n_total_simplices,)` int tensor; for the j-th
        simplex in filtration order, its row index within
        `complex.simplices_by_dim[sorted_dims[j]]`. Used to fetch the
        simplex's vertex tuple when building the boundary matrix.
    """

    complex: SparseSimplicialComplex
    birth_times: torch.Tensor
    sorted_dims: torch.Tensor
    sorted_indices: torch.Tensor


def build_filtration(
    distances: torch.Tensor,
    complex: SparseSimplicialComplex,
) -> Filtration:
    """Compute the birth time and sort order of every simplex in a VR complex.

    Args:
      distances: `(n, n)` pairwise distance matrix for the same vertex
        set as `complex`.
      complex: pre-built `SparseSimplicialComplex` (typically from
        `vietoris_rips_sparse`).

    Returns:
      A `Filtration` with simplices ordered by (birth_time, dim,
      input_row_index).

    References:
      Edelsbrunner, Harer (2010). Computational Topology: An Introduction.
        AMS, Chapter VI.2, filtrations.
    """
    # Gather one record per simplex across all dimensions into flat arrays,
    # so the sort below can order the whole complex at once.
    all_birth: list[torch.Tensor] = []
    all_dim: list[torch.Tensor] = []
    all_idx: list[torch.Tensor] = []

    for k in sorted(complex.simplices_by_dim.keys()):
        simplices = complex.simplices_by_dim[k]              # (n_k, k+1)
        n_k = simplices.shape[0]
        if n_k == 0:
            continue
        if k == 0:
            # A vertex's birth would be d(i, i) from the general formula below.
            # That is 0 for any distance matrix, so the gather is skipped.
            births = torch.zeros(n_k, dtype=distances.dtype,
                                  device=distances.device)
        else:
            # Birth is the largest distance among the simplex's vertices, which
            # is the max over its (k+1) x (k+1) submatrix of `distances`. Advanced
            # indexing builds all those submatrices at once:
            #   pairs[s, i, j] = distances[ simplices[s, i], simplices[s, j] ]
            v_i = simplices.unsqueeze(-1)                     # (n_k, k+1, 1)
            v_j = simplices.unsqueeze(-2)                     # (n_k, 1, k+1)
            # Broadcast to (n_k, k+1, k+1) of vertex pairs.
            sub = distances[v_i, v_j]                          # (n_k, k+1, k+1)
            births = sub.flatten(start_dim=1).max(dim=-1).values
        all_birth.append(births)
        all_dim.append(torch.full((n_k,), k, dtype=torch.int64,
                                    device=distances.device))
        all_idx.append(torch.arange(n_k, dtype=torch.int64,
                                      device=distances.device))

    # No simplices at all. Return empty tensors in the same dtypes as the
    # non-empty case, so callers never see a different type here.
    if not all_birth:
        empty = torch.zeros(0, dtype=distances.dtype, device=distances.device)
        empty_i = torch.zeros(0, dtype=torch.int64, device=distances.device)
        return Filtration(
            complex=complex, birth_times=empty,
            sorted_dims=empty_i, sorted_indices=empty_i,
        )

    birth_times = torch.cat(all_birth)
    dims = torch.cat(all_dim)
    indices = torch.cat(all_idx)

    # Sort by (birth_time, dim, row). A stable sort keeps the existing order
    # among ties, so we sort by the least significant key first. The last pass
    # sorts by birth time, and simplices with equal birth keep the (dim, row)
    # order that the passes before it left.
    order = torch.argsort(indices, stable=True)
    dims_sorted = dims[order]
    times_sorted = birth_times[order]
    indices_sorted = indices[order]

    order2 = torch.argsort(dims_sorted, stable=True)
    times_sorted = times_sorted[order2]
    indices_sorted = indices_sorted[order2]
    dims_sorted = dims_sorted[order2]

    order3 = torch.argsort(times_sorted, stable=True)
    times_sorted = times_sorted[order3]
    indices_sorted = indices_sorted[order3]
    dims_sorted = dims_sorted[order3]

    return Filtration(
        complex=complex,
        birth_times=times_sorted,
        sorted_dims=dims_sorted,
        sorted_indices=indices_sorted,
    )
