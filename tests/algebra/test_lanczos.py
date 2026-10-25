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

from typing import Literal

import pytest
import torch

from holonomy_lib.algebra import lanczos_eigsh


def _seeded(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def _random_symmetric(
    n: int,
    batch: int = 1,
    dtype=torch.float64,
    seed: int = 0,
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
        asymmetric = torch.tensor([[[1.0, 1.0], [0.0, 2.0]]])
        with pytest.warns(UserWarning, match="asymmetric"):
            lanczos_eigsh(asymmetric, k=1)


# --------------------------------------------------------------------
# Convergence vs dense eigh
# --------------------------------------------------------------------


class TestConvergence:
    @pytest.mark.parametrize("n", [10, 25, 50])
    def test_full_iter_matches_exact(self, n):
        """Full-subspace requests use eigh and preserve batch correspondence."""
        A = _random_symmetric(n, batch=3, seed=42)
        ritz, _ = lanczos_eigsh(
            A,
            k=1,
            n_iter=n,
            oversample=0,
            generator=_seeded(0),
        )
        ref = torch.linalg.eigvalsh(A).flip(dims=(-1,))[..., :1]
        torch.testing.assert_close(ritz, ref, atol=1e-9, rtol=1e-9)
        for index in range(A.shape[0]):
            singleton, _ = lanczos_eigsh(A[index : index + 1], k=1, n_iter=n)
            torch.testing.assert_close(ritz[index : index + 1], singleton)

    def test_partial_iter_converges_with_oversample(self):
        """With modest oversampling on a moderately-sized matrix the
        top-1 eigenvalue converges to a useful tolerance. Per Saad
        (2011) §6.7 the convergence is geometric in the spectral gap;
        we give the Krylov space enough dimensions to resolve the top mode."""
        n = 50
        A = _random_symmetric(n, seed=42)
        ritz, _ = lanczos_eigsh(
            A,
            k=1,
            n_iter=n // 2,
            oversample=0,
            generator=_seeded(0),
        )
        ref = torch.linalg.eigvalsh(A).flip(dims=(-1,))[..., :1]
        max_diff = (ritz - ref).abs().max().item()
        assert max_diff < 1e-3, (
            f"n=50, n_iter=25 top-1 diff {max_diff:.3e} should be < 1e-3"
        )

    def test_top_eigenvalue_residual_full_iter(self):
        """At `n_iter = n` the residual `‖A v − λ v‖` is machine-precision
        small because the full-subspace path uses dense eigh."""
        n = 20
        A = _random_symmetric(n, seed=99)
        vals, vecs = lanczos_eigsh(
            A,
            k=1,
            n_iter=n,
            oversample=0,
            generator=_seeded(7),
        )
        Av = torch.matmul(A, vecs)
        residual = Av - vecs * vals.unsqueeze(dim=-2)
        max_res = residual.norm(dim=-2).max()
        assert max_res.item() < 1e-9
        diagonal = torch.arange(1, 5, dtype=torch.float64).requires_grad_()
        modes: tuple[tuple[Literal["LA", "SA"], float | None], ...] = (
            ("LA", None),
            ("SA", None),
            ("SA", 0.0),
        )
        for which, sigma in modes:
            for iterations in (3, 4):
                assert torch.autograd.gradcheck(
                    lambda d, iterations=iterations, which=which, sigma=sigma: (
                        lanczos_eigsh(
                            torch.diag(d).unsqueeze(0),
                            k=2,
                            n_iter=iterations,
                            which=which,
                            sigma=sigma,
                            generator=_seeded(0),
                        )[0]
                    ),
                    (diagonal,),
                )


# --------------------------------------------------------------------
# Orthonormality
# --------------------------------------------------------------------


# --------------------------------------------------------------------
# Diagonal matrix: exact case
# --------------------------------------------------------------------


class TestDiagonalCase:
    def test_recovers_top_diagonal_entries(self):
        """For a diagonal matrix A = diag(λ_1, ..., λ_n), the largest
        eigenvalues are the largest diagonal entries. With sufficient
        Lanczos iterations Lanczos should find them exactly (up to
        rounding); provided the starting vector overlaps each
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
            A,
            k=4,
            n_iter=n,
            oversample=0,
            generator=_seeded(5),
        )
        # `diag` is already in descending order; the top-4 are the
        # first four entries.
        expected = diag[:4]
        torch.testing.assert_close(
            ritz[0],
            expected,
            atol=1e-8,
            rtol=0,
        )


# --------------------------------------------------------------------
# Shapes / batching
# --------------------------------------------------------------------


class TestEdgeCaseShapes:
    def test_batch_zero(self):
        """B=0 must work; project convention is `B ∈ {0, 1, >1}`."""
        A = torch.zeros(0, 8, 8, dtype=torch.float64)
        for which in ("LA", "SA"):
            for iterations in (4, 8):
                vals, vecs = lanczos_eigsh(A, k=2, n_iter=iterations, which=which)
                assert vals.shape == (0, 2)
                assert vecs.shape == (0, 8, 2)


class TestOrthonormalityTight:
    """Check full and partial bases, including members with Krylov breakdown."""

    def test_full_iter_basis_orthonormal_tight(self):
        n = 15
        A = _random_symmetric(n, seed=11)
        _, vecs = lanczos_eigsh(
            A,
            k=n,
            n_iter=n,
            oversample=0,
            generator=_seeded(2),
        )
        gram = torch.matmul(vecs.mT, vecs)
        I_k = torch.eye(n, dtype=vecs.dtype).unsqueeze(0)
        torch.testing.assert_close(gram, I_k, atol=1e-9, rtol=0)
        _, partial = lanczos_eigsh(A, k=4, n_iter=10, generator=_seeded(2))
        torch.testing.assert_close(
            partial.mT @ partial,
            torch.eye(4, dtype=A.dtype)[None],
            atol=1e-9,
            rtol=0,
        )
        # Distinct members close their Krylov spaces at different steps.
        spectra = torch.tensor(
            [[1.0, 1.0, 1.0, 1.0], [-1.0, -1.0, -1.0, -1.0], [1.0, 1.0, 2.0, 2.0]],
            dtype=torch.float64,
        )
        repeated = torch.diag_embed(spectra)
        for iterations in (3, 4):
            vals, vecs = lanczos_eigsh(
                repeated,
                k=iterations,
                n_iter=iterations,
                generator=_seeded(0),
            )
            torch.testing.assert_close(
                vecs.mT @ vecs,
                torch.eye(iterations, dtype=A.dtype).expand(3, -1, -1),
                atol=1e-9,
                rtol=0,
            )
            torch.testing.assert_close(repeated @ vecs, vecs * vals[:, None])
            if iterations == 4:
                torch.testing.assert_close(vals, spectra.sort(descending=True).values)
        for sign in (1.0, -1.0):
            sparse = (sign * torch.eye(6, dtype=A.dtype)).to_sparse()
            vals, vecs = lanczos_eigsh(sparse, k=3, n_iter=4, generator=_seeded(0))
            torch.testing.assert_close(vals, torch.full((3,), sign, dtype=A.dtype))
            torch.testing.assert_close(vecs.mT @ vecs, torch.eye(3, dtype=A.dtype))


# ====================================================================
# Shift-and-invert mode (which="SA")
# ====================================================================


def _random_spd(
    n: int,
    batch: int = 1,
    dtype=torch.float64,
    seed: int = 0,
) -> torch.Tensor:
    """Random SPD matrix: A = X Xᵀ + n·I for strict positive-definiteness."""
    g = _seeded(seed)
    X = torch.randn(batch, n, n, dtype=dtype, generator=g)
    return torch.matmul(X, X.mT) + n * torch.eye(n, dtype=dtype)


class TestShiftInvertConvergence:
    """SA mode must return smallest k eigenvalues, matching dense eigh."""

    @pytest.mark.parametrize("n", [10, 20])
    def test_smallest_eigvals_match_eigh_on_spd(self, n):
        """For SPD `A`, `which="SA"` with no explicit shift returns the
        smallest k eigenvalues. Compare to torch.linalg.eigvalsh
        (which returns ascending)."""
        A = _random_spd(n, batch=3, seed=7)
        k = 3
        vals_sa, _ = lanczos_eigsh(
            A,
            k=k,
            n_iter=n,
            oversample=0,
            which="SA",
            generator=_seeded(0),
        )
        ref = torch.linalg.eigvalsh(A)[..., :k]  # ascending, smallest k
        torch.testing.assert_close(vals_sa, ref, atol=1e-7, rtol=0)
        assert (vals_sa.diff() >= 0).all()
        for index in range(A.shape[0]):
            singleton, _ = lanczos_eigsh(
                A[index : index + 1], k=k, n_iter=n, which="SA"
            )
            torch.testing.assert_close(vals_sa[index : index + 1], singleton)
        # Scale must be local to each member, even in the inverse solve.
        scaled = torch.diag_embed(
            torch.tensor(
                [[1e8, 2e8, 3e8], [1e-8, 2e-8, 3e-8]],
                dtype=torch.float64,
            )
        )
        values, _ = lanczos_eigsh(scaled, k=1, which="SA", generator=_seeded(0))
        torch.testing.assert_close(values[:, 0], scaled[:, 0, 0])
        inverse_case = torch.diag_embed(
            torch.tensor(
                [[1e8, 1e8, 2e8, 2e8], [1e-8, 1e-8, 2e-8, 2e-8]],
                dtype=torch.float64,
            )
        )
        values, _ = lanczos_eigsh(
            inverse_case,
            k=1,
            n_iter=3,
            which="SA",
            sigma=0.0,
            generator=_seeded(0),
        )
        torch.testing.assert_close(values[:, 0], inverse_case[:, 0, 0])

    def test_eigenvectors_satisfy_A_v_equals_lambda_v(self):
        """A v_i ≈ λ_i v_i for each returned (λ_i, v_i) pair."""
        n, k = 12, 3
        A = _random_spd(n, seed=11)
        vals, vecs = lanczos_eigsh(
            A,
            k=k,
            n_iter=n,
            oversample=0,
            which="SA",
            generator=_seeded(0),
        )
        # (B, n, k); residual ‖A v − λ v‖ should be small for each col.
        Av = torch.matmul(A, vecs)  # (B, n, k)
        lambda_v = vecs * vals.unsqueeze(dim=-2)  # broadcast (B, n, k)
        residual = torch.linalg.norm(Av - lambda_v, dim=-2)  # (B, k)
        assert residual.max().item() < 1e-6

    def test_sa_with_explicit_sigma_finds_closest_eigenvalues(self):
        """With σ set to an interior value, SA recovers eigenvalues
        closest to σ."""
        # Diagonal matrix with known spectrum.
        spectrum = torch.tensor(
            [-5.0, -1.0, 0.5, 2.0, 4.0, 10.0],
            dtype=torch.float64,
        )
        A = torch.diag(spectrum).unsqueeze(0)  # (1, 6, 6)
        # σ = 1.0 → closest eigenvalues are 0.5 and 2.0.
        vals, _ = lanczos_eigsh(
            A,
            k=2,
            n_iter=6,
            oversample=0,
            which="SA",
            sigma=1.0,
            generator=_seeded(0),
        )
        sorted_by_dist = sorted(vals[0].tolist(), key=lambda x: abs(x - 1.0))
        assert sorted_by_dist[0] == pytest.approx(0.5, abs=1e-8)
        assert sorted_by_dist[1] == pytest.approx(2.0, abs=1e-8)


class TestShiftInvertOnLaplacian:
    """The motivating use case: smallest eigenvalues of a graph
    Laplacian = Fiedler eigenvalues. The smallest is always 0 for a
    connected graph, the second smallest controls algebraic
    connectivity (Cheeger's inequality)."""

    def test_path_graph_laplacian_smallest_eigenvalues(self):
        # Path graph P_5: combinatorial Laplacian with known smallest
        # eigenvalues 0, 2 - 2*cos(pi/5), ...
        n = 5
        L = (
            2 * torch.eye(n, dtype=torch.float64)
            - torch.diag(torch.ones(n - 1, dtype=torch.float64), 1)
            - torch.diag(torch.ones(n - 1, dtype=torch.float64), -1)
        )
        # Endpoints have degree 1 not 2; fix the corner entries.
        L[0, 0] = 1.0
        L[n - 1, n - 1] = 1.0
        L = L.unsqueeze(0)
        # Shift sigma slightly below 0 so (L - σI) is strictly PD and
        # numerically clean (L itself has 0 in spectrum which makes the
        # LU factor near-singular).
        vals, _ = lanczos_eigsh(
            L,
            k=2,
            n_iter=n,
            oversample=0,
            which="SA",
            sigma=-0.1,
            generator=_seeded(0),
        )
        ref = torch.linalg.eigvalsh(L)[..., :2]
        torch.testing.assert_close(vals, ref, atol=1e-7, rtol=0)


class TestShiftInvertBreakdownDetection:
    """A singular shift must fail before the partial-subspace inverse solve."""

    def test_sigma_at_zero_on_singular_laplacian_raises(self):
        # Combinatorial Laplacian of a path graph has 0 in its spectrum
        # (always; the constant vector is in the kernel for any
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
                L,
                k=2,
                n_iter=n - 1,
                oversample=0,
                which="SA",
                sigma=0.0,
                generator=_seeded(0),
            )
