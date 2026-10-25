# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Synoros

"""Tests for holonomy_lib.algebra.lanczos_eigsh.

Layers:
  1. Validation (k bounds, square input).
  2. Convergence against dense torch.linalg.eigh on small matrices.
  3. Eigenvector orthonormality.
  4. Diagonal-matrix edge case (exact convergence in m = n steps).
  5. Batching shape contract.
"""

from __future__ import annotations

import pytest
import torch

from holonomy_lib.algebra import lanczos_eigsh


def _seeded(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def _random_symmetric(
    n: int, batch: int = 1, dtype=torch.float64, seed: int = 0,
) -> torch.Tensor:
    g = _seeded(seed)
    A = torch.randn(batch, n, n, dtype=dtype, generator=g)
    return 0.5 * (A + A.mT)


# --------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------


class TestValidation:
    def test_rejects_non_square(self):
        with pytest.raises(ValueError, match="must be"):
            lanczos_eigsh(torch.zeros(1, 4, 5), k=2)

    def test_rejects_k_zero(self):
        A = _random_symmetric(5)
        with pytest.raises(ValueError, match="k must"):
            lanczos_eigsh(A, k=0)

    def test_rejects_k_too_large(self):
        A = _random_symmetric(5)
        with pytest.raises(ValueError, match="k must"):
            lanczos_eigsh(A, k=6)

    def test_rejects_n_iter_less_than_k(self):
        A = _random_symmetric(8)
        with pytest.raises(ValueError, match="n_iter"):
            lanczos_eigsh(A, k=4, n_iter=3, oversample=0)


# --------------------------------------------------------------------
# Convergence vs dense eigh
# --------------------------------------------------------------------


class TestConvergence:
    @pytest.mark.parametrize("n", [10, 25, 50])
    def test_full_iter_matches_exact(self, n):
        """With `n_iter = n` Lanczos builds the full Krylov subspace and
        is exact (up to rounding) for the top-1 eigenvalue. This pins
        down that the algorithm is correct; `test_partial_iter` below
        characterizes the typical-use convergence rate."""
        A = _random_symmetric(n, seed=42)
        ritz, _ = lanczos_eigsh(
            A, k=1, n_iter=n, oversample=0, generator=_seeded(0),
        )
        ref = torch.linalg.eigvalsh(A).flip(dims=(-1,))[..., :1]
        torch.testing.assert_close(
            ritz, ref, atol=1e-9, rtol=1e-9,
        )

    def test_partial_iter_converges_with_oversample(self):
        """With modest oversampling on a moderately-sized matrix the
        top-1 eigenvalue converges to a useful tolerance. Per Saad
        (2011) §6.7 the convergence is geometric in the spectral gap;
        we use a generous oversample so the test is robust across
        random starts."""
        n = 50
        A = _random_symmetric(n, seed=42)
        ritz, _ = lanczos_eigsh(
            A, k=1, n_iter=n // 2, oversample=0, generator=_seeded(0),
        )
        ref = torch.linalg.eigvalsh(A).flip(dims=(-1,))[..., :1]
        max_diff = (ritz - ref).abs().max().item()
        assert max_diff < 1e-3, (
            f"n=50, n_iter=25 top-1 diff {max_diff:.3e} should be < 1e-3"
        )

    def test_top_eigenvalue_residual_full_iter(self):
        """At `n_iter = n` the residual `‖A v − λ v‖` is machine-precision
        small (Lanczos has spanned the full eigenspace by then)."""
        n = 20
        A = _random_symmetric(n, seed=99)
        vals, vecs = lanczos_eigsh(
            A, k=1, n_iter=n, oversample=0, generator=_seeded(7),
        )
        Av = torch.matmul(A, vecs)
        residual = Av - vecs * vals.unsqueeze(dim=-2)
        max_res = residual.norm(dim=-2).max()
        assert max_res.item() < 1e-9, (
            f"full-iter top-1 residual {max_res.item():.3e} too large"
        )


# --------------------------------------------------------------------
# Orthonormality
# --------------------------------------------------------------------


class TestOrthonormality:
    def test_eigenvectors_orthonormal(self):
        """Returned eigenvectors should be orthonormal (within tolerance)."""
        A = _random_symmetric(15, seed=11)
        _, vecs = lanczos_eigsh(
            A, k=4, oversample=15, generator=_seeded(2),
        )
        # vecs^T vecs should ≈ I_k
        gram = torch.matmul(vecs.mT, vecs)
        k = vecs.shape[-1]
        I_k = torch.eye(k, dtype=vecs.dtype).unsqueeze(0)
        torch.testing.assert_close(gram, I_k, atol=1e-6, rtol=0)


# --------------------------------------------------------------------
# Diagonal matrix: exact case
# --------------------------------------------------------------------


class TestDiagonalCase:
    def test_recovers_top_diagonal_entries(self):
        """For a diagonal matrix A = diag(λ_1, ..., λ_n), the largest
        eigenvalues are the largest diagonal entries. With sufficient
        Lanczos iterations Lanczos should find them exactly (up to
        rounding) — provided the starting vector overlaps each
        eigenvector, which it does for a random Gaussian start with
        probability 1.
        """
        n = 12
        # Distinct, well-separated diagonal entries (already descending).
        diag = torch.tensor(
            [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.5, 0.1],
            dtype=torch.float64,
        )
        A = torch.diag(diag).unsqueeze(0)
        ritz, _ = lanczos_eigsh(
            A, k=4, n_iter=n, oversample=0, generator=_seeded(5),
        )
        # `diag` is already in descending order; the top-4 are the
        # first four entries.
        expected = diag[:4]
        torch.testing.assert_close(
            ritz[0], expected, atol=1e-8, rtol=0,
        )


# --------------------------------------------------------------------
# Shapes / batching
# --------------------------------------------------------------------


@pytest.mark.parametrize("batch", [1, 2, 4])
class TestShapes:
    def test_eigvals_shape(self, batch):
        A = _random_symmetric(15, batch=batch, seed=20)
        vals, _ = lanczos_eigsh(A, k=3, generator=_seeded(0))
        assert vals.shape == (batch, 3)

    def test_eigvecs_shape(self, batch):
        A = _random_symmetric(15, batch=batch, seed=21)
        _, vecs = lanczos_eigsh(A, k=3, generator=_seeded(0))
        assert vecs.shape == (batch, 15, 3)


class TestEdgeCaseShapes:
    def test_batch_zero(self):
        """B=0 must work — project convention is `B ∈ {0, 1, >1}`."""
        A = torch.zeros(0, 8, 8, dtype=torch.float64)
        vals, vecs = lanczos_eigsh(A, k=2, n_iter=4, generator=_seeded(0))
        assert vals.shape == (0, 2)
        assert vecs.shape == (0, 8, 2)


class TestOrthonormalityTight:
    """Tighter orthonormality assertion: with full reorthogonalization
    and `n_iter = n`, the basis should be orthonormal to ~1e-9, not
    just 1e-6. This catches drift from missing reorthogonalization
    that the looser test would let through."""

    def test_full_iter_basis_orthonormal_tight(self):
        n = 15
        A = _random_symmetric(n, seed=11)
        _, vecs = lanczos_eigsh(
            A, k=n, n_iter=n, oversample=0, generator=_seeded(2),
        )
        gram = torch.matmul(vecs.mT, vecs)
        I_k = torch.eye(n, dtype=vecs.dtype).unsqueeze(0)
        torch.testing.assert_close(gram, I_k, atol=1e-9, rtol=0)


# ====================================================================
# Shift-and-invert mode (which="SA")
# ====================================================================


def _random_spd(
    n: int, batch: int = 1, dtype=torch.float64, seed: int = 0,
) -> torch.Tensor:
    """Random SPD matrix: A = X Xᵀ + n·I for strict positive-definiteness."""
    g = _seeded(seed)
    X = torch.randn(batch, n, n, dtype=dtype, generator=g)
    return torch.matmul(X, X.mT) + n * torch.eye(n, dtype=dtype)


class TestShiftInvertValidation:
    def test_rejects_unknown_which(self):
        A = _random_spd(5)
        with pytest.raises(ValueError, match="which"):
            lanczos_eigsh(A, k=2, which="SOMETHING_ELSE")

    def test_rejects_sigma_with_LA(self):
        A = _random_spd(5)
        with pytest.raises(ValueError, match="sigma"):
            lanczos_eigsh(A, k=2, which="LA", sigma=1.5)

    def test_sparse_input_in_SA_dispatches_to_sparse_backend(self):
        # Sparse SA used to raise NotImplementedError; it now dispatches
        # to algebra.eigsh_sparse (full coverage in test_eigsh_sparse.py).
        # Here: the dispatch itself works and matches dense eigh.
        n = 8
        A_dense = _random_spd(n).squeeze(0)
        A_sparse = A_dense.to_sparse_csr()
        vals, vecs = lanczos_eigsh(A_sparse, k=2, which="SA")
        ref = torch.linalg.eigvalsh(A_dense)[:2]
        torch.testing.assert_close(vals, ref, atol=1e-8, rtol=0)
        assert vecs.shape == (n, 2)


class TestShiftInvertConvergence:
    """SA mode must return smallest k eigenvalues, matching dense eigh."""

    @pytest.mark.parametrize("n", [10, 20])
    def test_smallest_eigvals_match_eigh_on_spd(self, n):
        """For SPD `A`, `which="SA"` with default σ=0 returns the
        smallest k eigenvalues. Compare to torch.linalg.eigvalsh
        (which returns ascending)."""
        A = _random_spd(n, seed=7)
        k = 3
        vals_sa, _ = lanczos_eigsh(
            A, k=k, n_iter=n, oversample=0, which="SA",
            generator=_seeded(0),
        )
        ref = torch.linalg.eigvalsh(A)[..., :k]   # ascending, smallest k
        torch.testing.assert_close(vals_sa, ref, atol=1e-7, rtol=0)

    def test_eigenvectors_satisfy_A_v_equals_lambda_v(self):
        """A v_i ≈ λ_i v_i for each returned (λ_i, v_i) pair."""
        n, k = 12, 3
        A = _random_spd(n, seed=11)
        vals, vecs = lanczos_eigsh(
            A, k=k, n_iter=n, oversample=0, which="SA",
            generator=_seeded(0),
        )
        # (B, n, k); residual ‖A v − λ v‖ should be small for each col.
        Av = torch.matmul(A, vecs)                          # (B, n, k)
        lambda_v = vecs * vals.unsqueeze(dim=-2)             # broadcast (B, n, k)
        residual = torch.linalg.norm(Av - lambda_v, dim=-2)  # (B, k)
        assert residual.max().item() < 1e-6

    def test_sa_with_explicit_sigma_finds_closest_eigenvalues(self):
        """With σ set to an interior value, SA recovers eigenvalues
        closest to σ."""
        # Diagonal matrix with known spectrum.
        spectrum = torch.tensor(
            [-5.0, -1.0, 0.5, 2.0, 4.0, 10.0], dtype=torch.float64,
        )
        A = torch.diag(spectrum).unsqueeze(0)               # (1, 6, 6)
        # σ = 1.0 → closest eigenvalues are 0.5 and 2.0.
        vals, _ = lanczos_eigsh(
            A, k=2, n_iter=6, oversample=0, which="SA", sigma=1.0,
            generator=_seeded(0),
        )
        sorted_by_dist = sorted(vals[0].tolist(), key=lambda x: abs(x - 1.0))
        assert sorted_by_dist[0] == pytest.approx(0.5, abs=1e-8)
        assert sorted_by_dist[1] == pytest.approx(2.0, abs=1e-8)

    def test_sa_returns_ascending(self):
        """SA convention: smallest-first ordering."""
        n = 10
        A = _random_spd(n, seed=3)
        vals, _ = lanczos_eigsh(
            A, k=4, n_iter=n, oversample=0, which="SA",
            generator=_seeded(0),
        )
        # Ascending: each entry ≤ the next.
        diffs = vals[..., 1:] - vals[..., :-1]
        assert (diffs >= 0).all()


class TestShiftInvertOnLaplacian:
    """The motivating use case: smallest eigenvalues of a graph
    Laplacian = Fiedler eigenvalues. The smallest is always 0 for a
    connected graph, the second smallest controls algebraic
    connectivity (Cheeger's inequality)."""

    def test_path_graph_laplacian_smallest_eigenvalues(self):
        # Path graph P_5: combinatorial Laplacian with known smallest
        # eigenvalues 0, 2 − √3, ...
        n = 5
        L = (
            2 * torch.eye(n, dtype=torch.float64)
            - torch.diag(torch.ones(n - 1, dtype=torch.float64), 1)
            - torch.diag(torch.ones(n - 1, dtype=torch.float64), -1)
        )
        # Endpoints have degree 1 not 2 — fix the corner entries.
        L[0, 0] = 1.0
        L[n - 1, n - 1] = 1.0
        L = L.unsqueeze(0)
        # Shift sigma slightly below 0 so (L - σI) is strictly PD and
        # numerically clean (L itself has 0 in spectrum which makes the
        # LU factor near-singular).
        vals, _ = lanczos_eigsh(
            L, k=2, n_iter=n, oversample=0, which="SA", sigma=-0.1,
            generator=_seeded(0),
        )
        ref = torch.linalg.eigvalsh(L)[..., :2]
        torch.testing.assert_close(vals, ref, atol=1e-7, rtol=0)


class TestShiftInvertBreakdownDetection:
    """When σ coincides with (or lies near) an eigenvalue of A, the
    shifted operator (A − σI) is singular, the LU factor is degenerate,
    and the recovered λ = σ + 1/μ blows up. The library must fail
    loudly with a useful message, not return garbage."""

    def test_sigma_at_zero_on_singular_laplacian_raises(self):
        # Combinatorial Laplacian of a path graph has 0 in its spectrum
        # (always — the constant vector is in the kernel for any
        # connected graph). σ=0 makes (L - σI) = L singular.
        n = 5
        L = (
            2 * torch.eye(n, dtype=torch.float64)
            - torch.diag(torch.ones(n - 1, dtype=torch.float64), 1)
            - torch.diag(torch.ones(n - 1, dtype=torch.float64), -1)
        )
        L[0, 0] = 1.0
        L[n - 1, n - 1] = 1.0
        L = L.unsqueeze(0)
        with pytest.raises(RuntimeError, match="shift-invert breakdown"):
            lanczos_eigsh(
                L, k=2, n_iter=n, oversample=0, which="SA", sigma=0.0,
                generator=_seeded(0),
            )
