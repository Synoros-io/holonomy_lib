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
    """For a single k-simplex (1-D tensor of k+1 vertex indices), return
    its k+1 faces (each a (k,)-shape tensor) and their parity signs.

    Args:
      simplex: (k+1,) int tensor of sorted vertex indices.

    Returns:
      faces: (k+1, k) int tensor; row i is the face obtained by
        dropping the i-th vertex.
      signs: (k+1,) tensor of `(-1)**i` values.
    """
    k_plus_1 = simplex.shape[0]
    faces = torch.empty(
        (k_plus_1, k_plus_1 - 1), dtype=simplex.dtype, device=simplex.device,
    )
    for i in range(k_plus_1):
        # drop the i-th vertex
        mask = torch.ones(k_plus_1, dtype=torch.bool, device=simplex.device)
        mask[i] = False
        faces[i] = simplex[mask]
    # Signs: (-1)^0, (-1)^1, ..., (-1)^k
    signs = torch.tensor(
        [1 if (i % 2 == 0) else -1 for i in range(k_plus_1)],
        dtype=torch.int64, device=simplex.device,
    )
    return faces, signs


def build_simplex_index(simplices: torch.Tensor) -> dict[tuple[int, ...], int]:
    """Build a hash-table mapping sorted-vertex-tuple → row index.

    Args:
      simplices: (n, d) int tensor. Each row is a sorted simplex.

    Returns:
      Dict from `tuple(int, ...)` to row index in `simplices`.
    """
    out: dict[tuple[int, ...], int] = {}
    for j in range(simplices.shape[0]):
        out[tuple(simplices[j].tolist())] = j
    return out
