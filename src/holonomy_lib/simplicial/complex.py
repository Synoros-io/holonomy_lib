# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Synoros

"""DenseSimplicialComplex : batched simplicial complex with padded
dense simplex tables.

Used by Hodge Laplacians on small complexes (where batched dense
eigh is fast) and by tutorials / illustration code. For large
single-instance complexes (persistent homology), use
`SparseSimplicialComplex`.

Layout:
  `simplices_by_dim[k]` is an `(B, n_k_max, k+1)` int64 tensor.
  Each row holds a sorted tuple of vertex indices. Padding rows
  (beyond the valid count for that batch element) are filled with
  `-1` and masked off via `valid_mask[k]: (B, n_k_max) bool`.

  This shape makes boundary operators naturally batched: one
  `(B, n_{k-1}_max, n_k_max)` matrix per dim. Padding rows and columns
  stay zero because only valid simplices write entries into it.

References:
  Munkres, J. R. (1984). Elements of Algebraic Topology. Westview
    Press. §1 : simplicial complexes and chain complexes.
  Kolda, T. G., Bader, B. W. (2009). Tensor decompositions and
    applications. SIAM Review 51(3):455-500. (Tensor-layout
    conventions reused here.)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from holonomy_lib.simplicial._face_lookup import enumerate_faces, lookup_rows

if TYPE_CHECKING:
    from holonomy_lib.simplicial.sparse_complex import SparseSimplicialComplex


@dataclass(frozen=True)
class DenseSimplicialComplex:
    """Batched, padded simplicial complex.

    Attributes:
      simplices_by_dim: `dict[k, Tensor (B, n_k_max, k+1)]` of sorted
        vertex tuples. Padding rows are -1.
      valid_mask: `dict[k, Tensor (B, n_k_max) bool]` : True for real
        simplices, False for padding.
      n_vertices: vertex count (identical across batch elements).
      device, dtype: tensor placement + float dtype used for boundary
        operator returns.
    """

    simplices_by_dim: dict[int, torch.Tensor]
    valid_mask: dict[int, torch.Tensor]
    n_vertices: int
    device: torch.device = field(default=torch.device("cpu"))
    dtype: torch.dtype = field(default=torch.float64)

    @property
    def batch_size(self) -> int:
        """Leading batch dim. All dims must agree on this."""
        for tensor in self.simplices_by_dim.values():
            return tensor.shape[0]
        return 0

    @property
    def max_dim(self) -> int:
        """Highest dimension with a table, including an empty table."""
        return max(self.simplices_by_dim) if self.simplices_by_dim else -1

    def n_simplices(self, k: int) -> torch.Tensor:
        """Per-batch count of valid k-simplices: shape `(B,)`."""
        if k not in self.simplices_by_dim:
            return torch.zeros(self.batch_size, dtype=torch.int64,
                                device=self.device)
        return self.valid_mask[k].sum(dim=-1)

    def boundary(self, k: int) -> torch.Tensor:
        """Boundary operator `∂_k: C_k → C_{k-1}`.

        Returns a dense `(B, n_{k-1}_max, n_k_max)` matrix; entries
        `± 1` mark face → simplex incidences with the Koszul sign
        `(-1)^i` (i = dropped-vertex position). Padding rows/columns
        are zero.

        Args:
          k: simplex dimension. For k=0 the result is a `(B, 0, n_0_max)`
            zero tensor, since vertices have an empty boundary in the chain
            complex.

        References:
          Munkres, J. R. (1984). Elements of Algebraic Topology, §1.

        Notes:
          Assumes the complex is closed under faces. A face missing from
          the (k-1) table raises ValueError from the lookup.
        """
        if k == 0:
            # ∂_0 maps to the (-1)-chain group; conventionally zero.
            # Return a (B, 0, n_0_max) zero tensor so downstream
            # arithmetic (Hodge L_0 = ∂_0^T ∂_0 + ∂_1 ∂_1^T) reduces
            # cleanly to ∂_1 ∂_1^T.
            B = self.batch_size
            n_0_max = (
                self.simplices_by_dim[0].shape[1]
                if 0 in self.simplices_by_dim else 0
            )
            return torch.zeros(B, 0, n_0_max,
                                device=self.device, dtype=self.dtype)
        if k not in self.simplices_by_dim or k - 1 not in self.simplices_by_dim:
            raise ValueError(
                f"boundary(k={k}): complex lacks simplices at dim {k} "
                f"or {k - 1}"
            )

        B = self.batch_size
        k_simplices = self.simplices_by_dim[k]            # (B, n_k_max, k+1)
        prev_simplices = self.simplices_by_dim[k - 1]     # (B, n_{k-1}_max, k)
        n_k_max = k_simplices.shape[1]
        n_prev_max = prev_simplices.shape[1]
        k_valid = self.valid_mask[k]                      # (B, n_k_max)
        prev_valid = self.valid_mask[k - 1]

        D = torch.zeros(B, n_prev_max, n_k_max,
                         device=self.device, dtype=self.dtype)

        # Keys are (batch id, vertices), so a face in another batch member
        # cannot stand in for a missing face here. lookup_rows indexes only
        # the valid rows of the previous dimension, so prev_row maps that
        # index back to the padded row position.
        simplex_batch, simplex_col = torch.where(k_valid)  # padded columns
        prev_batch, prev_row = torch.where(prev_valid)
        faces, signs = enumerate_faces(k_simplices[k_valid])
        face_keys = torch.cat((
            simplex_batch[:, None, None].expand(-1, k + 1, 1), faces,
        ), dim=-1).reshape(-1, k + 1)
        prev_keys = torch.cat((prev_batch[:, None], prev_simplices[prev_valid]),
                              dim=-1)
        face_rows = prev_row[lookup_rows(prev_keys, face_keys)]
        # One entry per (simplex, face) pair, flattened simplex-major and
        # face-minor. repeat_interleave builds the index lists in that order,
        # and repeat builds the signs in the same order.
        D[simplex_batch.repeat_interleave(k + 1), face_rows,
          simplex_col.repeat_interleave(k + 1)] = signs.to(self.dtype).repeat(
              simplex_col.shape[0],
          )
        return D

    def to_sparse(self) -> SparseSimplicialComplex:
        """Convert to a single-instance sparse complex. Requires `B = 1`."""
        from holonomy_lib.simplicial.sparse_complex import SparseSimplicialComplex
        if self.batch_size != 1:
            raise ValueError(
                f"to_sparse() requires batch_size == 1, got {self.batch_size}"
            )
        sparse_simplices: dict[int, torch.Tensor] = {}
        for k, table in self.simplices_by_dim.items():
            valid = self.valid_mask[k][0]
            sparse_simplices[k] = table[0][valid]
        return SparseSimplicialComplex(
            simplices_by_dim=sparse_simplices,
            n_vertices=self.n_vertices,
            device=self.device,
        )
