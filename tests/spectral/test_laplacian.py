# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Synoros

"""Tests for holonomy_lib.spectral.laplacian."""

from __future__ import annotations

import pytest
import torch

from holonomy_lib.spectral import laplacian


def _seeded_generator(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def _random_undirected_unweighted(
    n: int, edge_prob: float, batch: int, seed: int,
    dtype=torch.float64,
) -> torch.Tensor:
    """Random Erdős-Rényi adjacency matrices, symmetric, {0, 1} entries.

    Diagonal is zero (no self-loops); off-diagonal entries are i.i.d.
    Bernoulli(`edge_prob`) and symmetrized.
    """
    g = _seeded_generator(seed)
    U = (torch.rand(batch, n, n, generator=g, dtype=dtype) < edge_prob).to(dtype)
    # Symmetrize: take upper triangle, mirror
    triu = torch.triu(U, diagonal=1)
    A = triu + triu.mT
    return A


def _random_signed_undirected(
    n: int, batch: int, seed: int,
    dtype=torch.float64,
) -> torch.Tensor:
    """Random symmetric adjacency with weights in [-1, 1], no self-loops."""
    g = _seeded_generator(seed)
    U = torch.rand(batch, n, n, generator=g, dtype=dtype) * 2 - 1
    triu = torch.triu(U, diagonal=1)
    return triu + triu.mT


# --------------------------------------------------------------------
# Input validation
# --------------------------------------------------------------------


class TestValidation:
    def test_rejects_1d(self):
        with pytest.raises(ValueError, match="at least"):
            laplacian.combinatorial(torch.zeros(5))

    def test_rejects_non_square(self):
        with pytest.raises(ValueError, match="square"):
            laplacian.combinatorial(torch.zeros(2, 4, 5))


# --------------------------------------------------------------------
# Shapes across B ∈ {0, 1, several} and constructor variants
# --------------------------------------------------------------------


@pytest.mark.parametrize("batch", [0, 1, 4])
@pytest.mark.parametrize("constructor", [
    laplacian.combinatorial,
    laplacian.symmetric_normalized,
    laplacian.random_walk,
    laplacian.signed,
])
class TestShapes:
    def test_shape_preserved(self, batch, constructor):
        n = 6
        A = _random_undirected_unweighted(n=n, edge_prob=0.4, batch=batch, seed=0)
        if constructor is laplacian.signed:
            # signed also works on signed inputs; use signed for variety
            A = _random_signed_undirected(n=n, batch=batch, seed=0)
        L = constructor(A)
        assert L.shape == (batch, n, n)


# --------------------------------------------------------------------
# Combinatorial Laplacian — properties
# --------------------------------------------------------------------


class TestCombinatorial:
    def test_psd(self):
        A = _random_undirected_unweighted(n=8, edge_prob=0.4, batch=3, seed=10)
        L = laplacian.combinatorial(A)
        eigvals = torch.linalg.eigvalsh(L)
        # Eigenvalues of PSD matrix are ≥ 0 (small numerical floor)
        assert (eigvals >= -1e-10).all()

    def test_row_sum_zero(self):
        """L · 1 = 0 for the combinatorial Laplacian."""
        A = _random_undirected_unweighted(n=7, edge_prob=0.5, batch=2, seed=11)
        L = laplacian.combinatorial(A)
        ones = torch.ones(2, 7, 1, dtype=L.dtype)
        zero = L @ ones
        torch.testing.assert_close(
            zero, torch.zeros_like(zero), atol=1e-12, rtol=0,
        )

    def test_symmetric(self):
        A = _random_undirected_unweighted(n=6, edge_prob=0.4, batch=2, seed=12)
        L = laplacian.combinatorial(A)
        torch.testing.assert_close(L, L.mT, atol=1e-12, rtol=0)

    def test_zero_eigenvalue_for_connected_graph(self):
        """The smallest eigenvalue of L is 0 for connected graphs."""
        # A complete-graph adjacency — definitely connected.
        n = 5
        A = (torch.ones(1, n, n, dtype=torch.float64)
             - torch.eye(n, dtype=torch.float64).unsqueeze(dim=0))
        L = laplacian.combinatorial(A)
        eigvals = torch.linalg.eigvalsh(L)
        assert abs(eigvals[0, 0].item()) < 1e-10


# --------------------------------------------------------------------
# Symmetric-normalized Laplacian — properties
# --------------------------------------------------------------------


class TestSymmetricNormalized:
    def test_psd(self):
        A = _random_undirected_unweighted(n=8, edge_prob=0.4, batch=3, seed=20)
        L = laplacian.symmetric_normalized(A)
        eigvals = torch.linalg.eigvalsh(L)
        assert (eigvals >= -1e-10).all()

    def test_spectrum_in_zero_two(self):
        """Eigenvalues of L_sym lie in [0, 2]."""
        A = _random_undirected_unweighted(n=8, edge_prob=0.4, batch=3, seed=21)
        L = laplacian.symmetric_normalized(A)
        eigvals = torch.linalg.eigvalsh(L)
        assert (eigvals >= -1e-10).all()
        assert (eigvals <= 2 + 1e-10).all()

    def test_symmetric(self):
        A = _random_undirected_unweighted(n=6, edge_prob=0.5, batch=2, seed=22)
        L = laplacian.symmetric_normalized(A)
        torch.testing.assert_close(L, L.mT, atol=1e-12, rtol=0)

    def test_handles_isolated_node(self):
        """Pseudoinverse convention: isolated node has L_sym row = e_i."""
        n = 5
        # Edges only among nodes 0..3; node 4 is isolated.
        A = torch.zeros(1, n, n, dtype=torch.float64)
        A[0, 0, 1] = A[0, 1, 0] = 1.0
        A[0, 1, 2] = A[0, 2, 1] = 1.0
        A[0, 2, 3] = A[0, 3, 2] = 1.0
        L = laplacian.symmetric_normalized(A)
        # Isolated row of L_sym: L_sym[4, :] should be [0, 0, 0, 0, 1] — i.e.,
        # since D^{-1/2}[4] = 0, that node contributes nothing to the off-diag
        # of L_sym, and L_sym[4,4] = 1 from the identity term.
        # Verify isolated diagonal and zero off-diagonal:
        assert torch.isclose(L[0, 4, 4], torch.tensor(1.0, dtype=L.dtype))
        assert torch.allclose(
            L[0, 4, :4], torch.zeros(4, dtype=L.dtype), atol=0,
        )


# --------------------------------------------------------------------
# Random-walk Laplacian — properties
# --------------------------------------------------------------------


class TestRandomWalk:
    def test_eigenvalues_same_as_sym(self):
        """L_rw and L_sym are similar (D^{1/2} L_rw D^{-1/2} = L_sym),
        so they share eigenvalues.
        """
        A = _random_undirected_unweighted(n=7, edge_prob=0.5, batch=3, seed=30)
        L_sym = laplacian.symmetric_normalized(A)
        L_rw = laplacian.random_walk(A)
        # Compare sorted eigenvalues. L_rw is real but not symmetric, so
        # use linalg.eigvals; sort by real part since values are real.
        ev_sym, _ = torch.sort(torch.linalg.eigvalsh(L_sym), dim=-1)
        ev_rw_complex = torch.linalg.eigvals(L_rw)
        # Imaginary parts should be ≈ 0
        assert ev_rw_complex.imag.abs().max() < 1e-10
        ev_rw, _ = torch.sort(ev_rw_complex.real, dim=-1)
        torch.testing.assert_close(ev_rw, ev_sym, atol=1e-9, rtol=1e-9)

    def test_row_sums_zero_on_non_isolated_rows(self):
        """For non-isolated rows, L_rw row sum is 0."""
        A = _random_undirected_unweighted(n=6, edge_prob=0.6, batch=2, seed=31)
        d = laplacian.degree(A)
        L = laplacian.random_walk(A)
        row_sums = L.sum(dim=-1)  # (B, n)
        # Where degree > 0, row sum is 0
        mask = d > 0
        zeros = torch.zeros_like(row_sums)
        torch.testing.assert_close(
            torch.where(mask, row_sums, zeros),
            zeros,
            atol=1e-12, rtol=0,
        )


# --------------------------------------------------------------------
# Signed Laplacian — properties
# --------------------------------------------------------------------


class TestSigned:
    def test_psd_even_with_negative_weights(self):
        """Kunegis (2010) Theorem 3.4: L^σ is PSD for any signed graph."""
        A = _random_signed_undirected(n=8, batch=3, seed=40)
        L = laplacian.signed(A)
        eigvals = torch.linalg.eigvalsh(L)
        assert (eigvals >= -1e-10).all(), f"min eigval {eigvals.min().item()}"

    def test_symmetric(self):
        A = _random_signed_undirected(n=6, batch=2, seed=41)
        L = laplacian.signed(A)
        torch.testing.assert_close(L, L.mT, atol=1e-12, rtol=0)

    def test_reduces_to_combinatorial_when_nonneg(self):
        """For non-negative weights, L^σ == L (combinatorial)."""
        A = _random_undirected_unweighted(n=6, edge_prob=0.5, batch=2, seed=42)
        L_signed = laplacian.signed(A)
        L_comb = laplacian.combinatorial(A)
        torch.testing.assert_close(L_signed, L_comb, atol=1e-12, rtol=0)

    def test_balanced_graph_has_zero_eigenvalue(self):
        """For a 2-clustering balanced signed graph (positive within
        clusters, negative across), L^σ has eigenvalue 0.

        Kunegis (2010), Theorem 3.4: the signed graph is balanced iff
        zero is in the spectrum of L^σ.
        """
        # Two clusters of 3 nodes each. Positive within, negative across.
        n = 6
        A = torch.zeros(1, n, n, dtype=torch.float64)
        for i in range(3):
            for j in range(3):
                if i != j:
                    A[0, i, j] = 1.0     # cluster 0–2 positive intra
                    A[0, i + 3, j + 3] = 1.0  # cluster 3–5 positive intra
        for i in range(3):
            for j in range(3, 6):
                A[0, i, j] = -1.0
                A[0, j, i] = -1.0
        L = laplacian.signed(A)
        eigvals = torch.linalg.eigvalsh(L)
        assert abs(eigvals[0, 0].item()) < 1e-9, (
            f"balanced signed graph should have zero eigenvalue, "
            f"got smallest {eigvals[0, 0].item()}"
        )


# --------------------------------------------------------------------
# Comparison against scipy.sparse.csgraph.laplacian
# --------------------------------------------------------------------


try:
    from scipy.sparse.csgraph import laplacian as _scipy_laplacian  # noqa: F401
    _HAVE_SCIPY = True
except ImportError:
    _HAVE_SCIPY = False


@pytest.mark.skipif(not _HAVE_SCIPY, reason="scipy not installed")
class TestAgainstScipy:
    """Cross-check against scipy.sparse.csgraph.laplacian.

    Scipy returns the combinatorial / symmetric-normalized / random-walk
    Laplacians via the `normed` and `form` parameters; we compare dense
    reconstructions.
    """

    def test_combinatorial_matches_scipy(self):
        import numpy as np
        from scipy.sparse.csgraph import laplacian as scipy_laplacian
        A = _random_undirected_unweighted(n=8, edge_prob=0.4, batch=1, seed=50)
        L_ours = laplacian.combinatorial(A)[0].numpy()
        L_scipy = scipy_laplacian(A[0].numpy(), normed=False)
        L_scipy_dense = (
            L_scipy.toarray() if hasattr(L_scipy, "toarray") else np.asarray(L_scipy)
        )
        np.testing.assert_allclose(L_ours, L_scipy_dense, atol=1e-12)

    def test_symmetric_normalized_matches_scipy(self):
        import numpy as np
        from scipy.sparse.csgraph import laplacian as scipy_laplacian
        A = _random_undirected_unweighted(n=8, edge_prob=0.5, batch=1, seed=51)
        L_ours = laplacian.symmetric_normalized(A)[0].numpy()
        L_scipy = scipy_laplacian(A[0].numpy(), normed=True)
        L_scipy_dense = (
            L_scipy.toarray() if hasattr(L_scipy, "toarray") else np.asarray(L_scipy)
        )
        np.testing.assert_allclose(L_ours, L_scipy_dense, atol=1e-12)
