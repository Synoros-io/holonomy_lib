# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Synoros

"""SparseSimplicialComplex : single-instance simplicial complex with
sparse-CSC boundary operators.

Used by persistent homology (where the matrix-reduction kernel walks
the sparse boundary matrix column by column) and by Hodge Laplacians
on large complexes where dense `(n, n)` doesn't fit.

No batch dimension: persistent homology operates on one complex at a
time (the batching for PH happens at a higher level, calling this
class once per point cloud). For batched small complexes, use
`DenseSimplicialComplex` and the dense Hodge path.

References:
  Saad, Y. (2003). Iterative Methods for Sparse Linear Systems, 2nd
    ed. SIAM. §3.4 : CSR and CSC formats.
  Edelsbrunner, H., Letscher, D., Zomorodian, A. (2002). Topological
    persistence and simplification. Discrete & Computational Geometry
    28:511-533. Original sparse-boundary reduction algorithm.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from holonomy_lib.simplicial._face_lookup import enumerate_faces, lookup_rows

if TYPE_CHECKING:
    from holonomy_lib.simplicial.complex import DenseSimplicialComplex


@dataclass(frozen=True)
class SparseSimplicialComplex:
    """Single-instance simplicial complex.

    Attributes:
      simplices_by_dim: `dict[k, Tensor (n_k, k+1)]` of sorted vertex
        tuples. No batch dim, no padding.
      n_vertices: vertex count.
      device: tensor placement.
    """

    simplices_by_dim: dict[int, torch.Tensor]
    n_vertices: int
    device: torch.device = field(default=torch.device("cpu"))

    @property
    def max_dim(self) -> int:
        """Highest dimension with a table, including an empty table."""
        return max(self.simplices_by_dim) if self.simplices_by_dim else -1

    def n_simplices(self, k: int) -> int:
        """Number of k-simplices. Returns 0 when there is no table for dimension k."""
        if k not in self.simplices_by_dim:
            return 0
        return self.simplices_by_dim[k].shape[0]

    def boundary(
        self, k: int, dtype: torch.dtype = torch.float64,
    ) -> torch.Tensor:
        """Sparse-CSC boundary operator `∂_k: C_k → C_{k-1}`.

        Returns a sparse-CSC `(n_{k-1}, n_k)` tensor; entries are
        `± 1` with the Koszul sign for each face → simplex incidence.

        Args:
          k: simplex dimension. `k = 0` returns an empty `(0, n_0)`
            sparse tensor (vertices have empty boundary).
          dtype: float dtype for the matrix values.

        References:
          Munkres, J. R. (1984). Elements of Algebraic Topology, §1.

        Notes:
          Assumes the complex is closed under faces. A face missing from
          the (k-1) table raises ValueError from the lookup.
        """
        if k == 0:
            n_0 = self.n_simplices(0)
            # No row indices and every column pointer is 0, so each of the
            # n_0 columns is empty.
            return torch.sparse_csc_tensor(
                ccol_indices=torch.zeros(n_0 + 1, dtype=torch.int64,
                                          device=self.device),
                row_indices=torch.zeros(0, dtype=torch.int64,
                                         device=self.device),
                values=torch.zeros(0, dtype=dtype, device=self.device),
                size=(0, n_0),
            )
        if k not in self.simplices_by_dim or k - 1 not in self.simplices_by_dim:
            raise ValueError(
                f"boundary(k={k}): complex lacks simplices at dim {k} "
                f"or {k - 1}"
            )

        k_simplices = self.simplices_by_dim[k]            # (n_k, k+1)
        prev_simplices = self.simplices_by_dim[k - 1]     # (n_{k-1}, k)
        n_k = k_simplices.shape[0]
        n_prev = prev_simplices.shape[0]

        faces, signs = enumerate_faces(k_simplices)
        rows = lookup_rows(prev_simplices, faces.reshape(-1, k))
        cols = torch.arange(n_k, device=self.device).repeat_interleave(k + 1)
        vals = signs.to(dtype).repeat(n_k)

        # Build the entries as COO, which takes (row, column) pairs directly.
        # coalesce sorts them by row then column, and to_sparse_csc then
        # compresses the column indices into the CSC layout.
        coo = torch.sparse_coo_tensor(
            indices=torch.stack([rows, cols]),
            values=vals,
            size=(n_prev, n_k),
        ).coalesce()
        return coo.to_sparse_csc()

    def to_dense(
        self, dtype: torch.dtype = torch.float64,
    ) -> DenseSimplicialComplex:
        """Convert to a batched-with-B=1 dense complex.

        Args:
          dtype: float dtype for boundary matrix returns in the dense
            complex.
        """
        from holonomy_lib.simplicial.complex import DenseSimplicialComplex
        dense_simplices: dict[int, torch.Tensor] = {}
        valid_masks: dict[int, torch.Tensor] = {}
        for k, table in self.simplices_by_dim.items():
            dense_simplices[k] = table.unsqueeze(0)         # (1, n_k, k+1)
            # A single complex has no padding, so every row is a real simplex.
            valid_masks[k] = torch.ones(
                1, table.shape[0], dtype=torch.bool, device=self.device,
            )
        return DenseSimplicialComplex(
            simplices_by_dim=dense_simplices,
            valid_mask=valid_masks,
            n_vertices=self.n_vertices,
            device=self.device,
            dtype=dtype,
        )
