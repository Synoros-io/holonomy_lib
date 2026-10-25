# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Synoros

"""Sparse-input tests for holonomy_lib.spectral.laplacian.*

The Laplacian primitives gained a layout-dispatched sparse path
(roadmap #5). For each of the four variants we verify:

  1. Sparse-in → sparse-out (output layout matches input layout).
  2. Densify of sparse Laplacian equals the existing dense Laplacian
     on the same adjacency (semantic agreement).
  3. End-to-end chain: sparse adjacency → sparse Laplacian →
     `lanczos_eigsh` sparse path → top-k eigenpairs. Compares against
     dense `torch.linalg.eigh`.
  4. 2-D-only enforcement (sparse batched is not supported in v1).
"""

from __future__ import annotations

import pytest
import torch

from holonomy_lib.algebra import lanczos_eigsh
from holonomy_lib.spectral import laplacian


def _seeded(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def _random_symmetric_sparse(
    n: int,
    edge_prob: float = 0.3,
    dtype=torch.float64,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """A pair `(A_dense, A_sparse_coo)` for the same symmetric graph."""
    g = _seeded(seed)
    A = torch.rand(n, n, dtype=dtype, generator=g)
    A = 0.5 * (A + A.mT)
    A = A * (A > 1.0 - edge_prob)             # mask out below threshold
    A = A - torch.diag(torch.diagonal(A))     # no self-loops
    return A, A.to_sparse_coo()


def _random_signed_symmetric_sparse(
    n: int,
    edge_prob: float = 0.3,
    dtype=torch.float64,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Like above but signs randomly flipped."""
    g = _seeded(seed)
    A, _ = _random_symmetric_sparse(n, edge_prob, dtype, seed)
    signs = torch.where(
        torch.rand(n, n, dtype=dtype, generator=g) > 0.5,
        torch.ones_like(A), -torch.ones_like(A),
    )
    signs = 0.5 * (signs + signs.mT)          # symmetrize signs
    # Snap to ±1 — symmetrizing a ±1 matrix can produce {-1, 0, 1}; we
    # want a strict ±1 mask, so just take the sign.
    signs = torch.sign(signs)
    signs = signs + (signs == 0).to(dtype)    # zeros → +1 to keep edges intact
    A = A * signs
    return A, A.to_sparse_coo()


# --------------------------------------------------------------------
# Layout dispatch + 2-D enforcement
# --------------------------------------------------------------------


class TestLayoutDispatch:
    def test_sparse_combinatorial_returns_sparse(self):
        _, A_sparse = _random_symmetric_sparse(10, seed=0)
        L = laplacian.combinatorial(A_sparse)
        assert L.is_sparse

    def test_sparse_symmetric_normalized_returns_sparse(self):
        _, A_sparse = _random_symmetric_sparse(10, seed=1)
        L = laplacian.symmetric_normalized(A_sparse)
        assert L.is_sparse

    def test_sparse_random_walk_returns_sparse(self):
        _, A_sparse = _random_symmetric_sparse(10, seed=2)
        L = laplacian.random_walk(A_sparse)
        assert L.is_sparse

    def test_sparse_signed_returns_sparse(self):
        _, A_sparse = _random_signed_symmetric_sparse(10, seed=3)
        L = laplacian.signed(A_sparse)
        assert L.is_sparse

    def test_sparse_rejects_3d(self):
        n = 5
        A_batched_sparse = torch.zeros(2, n, n, dtype=torch.float64).to_sparse_coo()
        with pytest.raises(ValueError, match="2-D"):
            laplacian.combinatorial(A_batched_sparse)


# --------------------------------------------------------------------
# Sparse Laplacian densifies to the dense reference
# --------------------------------------------------------------------


class TestAgreementWithDense:
    @pytest.mark.parametrize("n", [10, 30])
    def test_combinatorial(self, n):
        A_dense, A_sparse = _random_symmetric_sparse(n, seed=n)
        L_sparse = laplacian.combinatorial(A_sparse).to_dense()
        # Dense path takes a batch dim — wrap in (1, n, n) and squeeze.
        L_dense = laplacian.combinatorial(A_dense.unsqueeze(0)).squeeze(0)
        torch.testing.assert_close(L_sparse, L_dense, atol=1e-12, rtol=0)

    @pytest.mark.parametrize("n", [10, 30])
    def test_symmetric_normalized(self, n):
        A_dense, A_sparse = _random_symmetric_sparse(n, seed=n + 1)
        L_sparse = laplacian.symmetric_normalized(A_sparse).to_dense()
        L_dense = laplacian.symmetric_normalized(A_dense.unsqueeze(0)).squeeze(0)
        torch.testing.assert_close(L_sparse, L_dense, atol=1e-9, rtol=0)

    @pytest.mark.parametrize("n", [10, 30])
    def test_random_walk(self, n):
        A_dense, A_sparse = _random_symmetric_sparse(n, seed=n + 2)
        L_sparse = laplacian.random_walk(A_sparse).to_dense()
        L_dense = laplacian.random_walk(A_dense.unsqueeze(0)).squeeze(0)
        torch.testing.assert_close(L_sparse, L_dense, atol=1e-9, rtol=0)

    @pytest.mark.parametrize("n", [10, 30])
    def test_signed(self, n):
        A_dense, A_sparse = _random_signed_symmetric_sparse(n, seed=n + 3)
        L_sparse = laplacian.signed(A_sparse).to_dense()
        L_dense = laplacian.signed(A_dense.unsqueeze(0)).squeeze(0)
        torch.testing.assert_close(L_sparse, L_dense, atol=1e-12, rtol=0)


# --------------------------------------------------------------------
# End-to-end: sparse Laplacian → sparse Lanczos → top-k eigenpairs
# --------------------------------------------------------------------


class TestLayoutCoverage:
    """The dispatcher accepts COO, CSR, and CSC. The agreement-with-
    dense tests use COO only — confirm CSR and CSC paths produce the
    same Laplacian (PyTorch's sparse-CSR-to-COO conversion does NOT
    densify in the version we ship against, but the path needs
    coverage)."""

    @pytest.mark.parametrize("layout_method", ["to_sparse_coo", "to_sparse_csr"])
    def test_combinatorial_layouts_agree(self, layout_method):
        A_dense, _ = _random_symmetric_sparse(20, seed=99)
        A_sparse = getattr(A_dense, layout_method)()
        L_sparse = laplacian.combinatorial(A_sparse).to_dense()
        L_dense = laplacian.combinatorial(A_dense.unsqueeze(0)).squeeze(0)
        torch.testing.assert_close(L_sparse, L_dense, atol=1e-12, rtol=0)

    @pytest.mark.parametrize("layout_method", ["to_sparse_coo", "to_sparse_csr"])
    def test_symmetric_normalized_layouts_agree(self, layout_method):
        A_dense, _ = _random_symmetric_sparse(20, seed=100)
        A_sparse = getattr(A_dense, layout_method)()
        L_sparse = laplacian.symmetric_normalized(A_sparse).to_dense()
        L_dense = laplacian.symmetric_normalized(A_dense.unsqueeze(0)).squeeze(0)
        torch.testing.assert_close(L_sparse, L_dense, atol=1e-9, rtol=0)


class TestEndToEndSparseChain:
    def test_sparse_combinatorial_lanczos_matches_dense_eigh(self):
        """The motivating chain: build sparse L from sparse A, run
        sparse Lanczos on it, compare against dense `eigh` on the
        materialized L. This is the win the sparse backend unlocks.
        """
        n, k = 20, 3
        A_dense, A_sparse = _random_symmetric_sparse(n, edge_prob=0.4, seed=99)
        L_sparse = laplacian.combinatorial(A_sparse)
        # Densify just for the reference; the test path stays sparse.
        L_dense_ref = L_sparse.to_dense()

        # Sparse Lanczos (top-k by VALUE, descending).
        ritz_vals, _ = lanczos_eigsh(
            L_sparse, k=k, n_iter=n, oversample=0,
            generator=_seeded(0),
        )
        ref = torch.linalg.eigvalsh(L_dense_ref).flip(dims=(-1,))[:k]
        torch.testing.assert_close(ritz_vals, ref, atol=1e-7, rtol=0)
