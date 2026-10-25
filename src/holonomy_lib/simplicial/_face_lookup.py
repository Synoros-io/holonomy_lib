# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Synoros

"""Shared face-enumeration + lookup helpers for the boundary operator.

For a simplex σ = [v_0, v_1, ..., v_k] (vertices sorted ascending),
the boundary is

    ∂σ = Σ_{i=0..k} (-1)^i · [v_0, ..., v̂_i, ..., v_k].

Building ∂_k as a matrix requires, for each k-simplex column j, locating
the (k-1)-simplex row index of each of its (k+1) faces. The two
representations (dense batched, sparse single-instance) share this
enumeration logic; only the matrix assembly differs.
"""

from __future__ import annotations

import torch


def enumerate_faces(simplex: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Enumerate oriented faces along the last simplex-table axis.

    Args:
      simplex: (..., k+1) int tensor of sorted vertex indices.

    Returns:
      faces: (..., k+1, k) int tensor; row i is the face obtained by
        dropping the i-th vertex.
      signs: (k+1,) tensor of `(-1)**i` values.

    References:
      Munkres, J. R. (1984). Elements of Algebraic Topology, §1.

    Notes:
      Assumes the last axis is sorted ascending. Dropping a vertex keeps
      the rest sorted, so each face has the same sorted form that
      lookup_rows matches against the stored tables.
    """
    width = simplex.shape[-1]
    positions = torch.arange(width, device=simplex.device)
    # Row i of `keep` lists every position except i, so indexing with it
    # drops vertex i from the simplex.
    keep = positions.expand(width, width)[
        ~torch.eye(width, dtype=torch.bool, device=simplex.device)
    ].reshape(width, width - 1)
    faces = simplex[..., keep]
    # Dropping vertex i carries the sign (-1)**i: +1 for even i, -1 for odd.
    signs = 1 - 2 * (positions % 2)
    return faces, signs


def lookup_rows(
    reference_rows: torch.Tensor, query_rows: torch.Tensor,
) -> torch.Tensor:
    """Return the reference index of each query row by an exact tensor join.

    Args:
      reference_rows: (n, d) unique integer keys.
      query_rows: (m, d) integer keys. Include batch ids in both tables
        when identical simplices in different batches must stay distinct.

    Returns:
      (m,) integer reference indices. A missing query raises ValueError.

    References:
      Munkres, J. R. (1984). Elements of Algebraic Topology, §1,
        for face incidence; the row join is an implementation detail.

    Notes:
      Duplicate reference rows are not supported. If two reference rows
      share a key, which of their indices comes back is unspecified.
    """
    n_reference = reference_rows.shape[0]
    # unique gives every distinct row one id, so a query row and its matching
    # reference row end up with the same id. The join uses whole rows instead
    # of a packed integer hash, because packing a vertex tuple into one integer
    # can overflow int64 when the vertex count is large. The batch id is one
    # of the columns, so the same simplex in two batch members stays distinct.
    keys, inverse = torch.unique(
        torch.cat((reference_rows, query_rows)), dim=0, return_inverse=True,
    )
    # Ids that no reference row claims stay -1.
    rows = torch.full(
        (keys.shape[0],), -1, dtype=torch.int64, device=reference_rows.device,
    )
    rows[inverse[:n_reference]] = torch.arange(
        n_reference, device=reference_rows.device,
    )
    found = rows[inverse[n_reference:]]
    # A -1 here means a query row never showed up among the reference rows.
    # For the boundary operator that means the complex is not closed under faces.
    if bool((found < 0).any()):
        raise ValueError(
            "Malformed complex: every face of a simplex must appear "
            "in the previous dimension's simplex table."
        )
    return found
