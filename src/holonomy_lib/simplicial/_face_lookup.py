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
    """
    width = simplex.shape[-1]
    positions = torch.arange(width, device=simplex.device)
    keep = positions.expand(width, width)[
        ~torch.eye(width, dtype=torch.bool, device=simplex.device)
    ].reshape(width, width - 1)
    faces = simplex[..., keep]
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
    """
    n_reference = reference_rows.shape[0]
    # Keeping complete rows avoids overflow from encoding a vertex tuple
    # as a positional integer hash. The shared ids also join across batches.
    keys, inverse = torch.unique(
        torch.cat((reference_rows, query_rows)), dim=0, return_inverse=True,
    )
    rows = torch.full(
        (keys.shape[0],), -1, dtype=torch.int64, device=reference_rows.device,
    )
    rows[inverse[:n_reference]] = torch.arange(
        n_reference, device=reference_rows.device,
    )
    found = rows[inverse[n_reference:]]
    if bool((found < 0).any()):
        raise ValueError(
            "Malformed complex: every face of a simplex must appear "
            "in the previous dimension's simplex table."
        )
    return found
