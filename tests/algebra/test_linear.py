# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Synoros

"""Tests for holonomy_lib.algebra.linear.truncated_svd."""

from __future__ import annotations

import pytest
import torch

from holonomy_lib.algebra import truncated_svd


def _seeded_generator(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


# --------------------------------------------------------------------
# Input validation
# --------------------------------------------------------------------


class TestInputValidation:
    def test_rejects_r_zero(self):
        M = torch.randn(5, 7, dtype=torch.float64)
        for kwargs in ({"r": 0}, {"r": 2, "oversample": -1},
                       {"r": 2, "n_iter": -1}, {"r": 2, "oversample": 0.5}):
            with pytest.raises(ValueError):
                truncated_svd(M, **kwargs)
        with pytest.raises(ValueError):
            truncated_svd(torch.ones(7), r=1)
        with pytest.raises(ValueError):
            truncated_svd(torch.ones(2, 2, dtype=torch.int64), r=1)


# --------------------------------------------------------------------
# Correctness; exact mode
# --------------------------------------------------------------------


class TestExactMode:
    @pytest.mark.parametrize("batch", [0, 1, 3])
    def test_recovers_torch_svd_truncated(self, batch):
        """Exact-mode truncated_svd matches torch.linalg.svd truncated."""
        M = torch.randn(batch, 8, 6, dtype=torch.float64,
                        generator=_seeded_generator(10))
        r = 3
        U, S, Vh = truncated_svd(M, r=r, mode="exact")
        # Reference: torch.linalg.svd, then truncate
        U_ref, S_ref, Vh_ref = torch.linalg.svd(M, full_matrices=False)
        assert S.shape == (batch, r)
        assert (S >= 0).all()
        assert (S[..., :-1] >= S[..., 1:]).all()
        torch.testing.assert_close(S, S_ref[..., :r], atol=1e-12, rtol=0)
        # U, V have sign freedom in SVD; the rank-r reconstruction is invariant.
        M_ours = U @ torch.diag_embed(S) @ Vh
        M_ref = (U_ref[..., :, :r] @ torch.diag_embed(S_ref[..., :r])
                 @ Vh_ref[..., :r, :])
        torch.testing.assert_close(M_ours, M_ref, atol=1e-12, rtol=0)

    def test_eckart_young_exact_on_low_rank(self):
        """For a rank-r matrix M, truncated_svd(M, r) reconstructs M exactly."""
        m, n, r = 8, 6, 3
        U_true = torch.linalg.qr(
            torch.randn(2, m, r, dtype=torch.float64,
                        generator=_seeded_generator(20))
        )[0]
        S_true = torch.sort(
            torch.rand(2, r, dtype=torch.float64,
                       generator=_seeded_generator(21)),
            dim=-1, descending=True,
        )[0]
        V_true = torch.linalg.qr(
            torch.randn(2, n, r, dtype=torch.float64,
                        generator=_seeded_generator(22))
        )[0]
        M = U_true @ torch.diag_embed(S_true) @ V_true.mT
        # truncate at rank r; should recover M exactly
        U, S, Vh = truncated_svd(M, r=r, mode="exact")
        M_reconstructed = U @ torch.diag_embed(S) @ Vh
        torch.testing.assert_close(M_reconstructed, M, atol=1e-12, rtol=0)

    @pytest.mark.parametrize("mode", ["exact", "randomized"])
    def test_orthonormality(self, mode):
        """U has orthonormal columns, Vh has orthonormal rows."""
        M = torch.randn(2, 15, 10, dtype=torch.float64,
                        generator=_seeded_generator(30))
        U, _, Vh = truncated_svd(M, r=4, mode=mode,
                                  generator=_seeded_generator(61))
        UtU = U.mT @ U
        identity = torch.eye(4, dtype=torch.float64).expand_as(UtU)
        torch.testing.assert_close(UtU, identity, atol=1e-12, rtol=0)
        VVh = Vh @ Vh.mT
        torch.testing.assert_close(VVh, identity, atol=1e-12, rtol=0)


# --------------------------------------------------------------------
# Correctness; randomized mode
# --------------------------------------------------------------------


class TestRandomizedMode:
    def test_randomized_recovers_low_rank_exactly(self):
        """For a rank-r matrix, randomized SVD with adequate oversampling
        reconstructs it to high precision.

        Standard HMT theory: when the true rank ≤ r and oversample ≥ 2,
        the randomized projection captures the entire range with prob 1
        for Gaussian Ω. Subspace iteration tightens further.
        """
        m, n, r = 12, 10, 3
        # Generate a true rank-r matrix
        A = torch.randn(2, m, r, dtype=torch.float64,
                        generator=_seeded_generator(40))
        B = torch.randn(2, r, n, dtype=torch.float64,
                        generator=_seeded_generator(41))
        M = A @ B
        U, S, Vh = truncated_svd(M, r=r, mode="randomized",
                                  generator=_seeded_generator(42))
        M_recon = U @ torch.diag_embed(S) @ Vh
        rel_err = (
            torch.linalg.matrix_norm(M_recon - M, dim=(-2, -1))
            / torch.linalg.matrix_norm(M, dim=(-2, -1))
        )
        # With n_iter=2 and oversample=5, low-rank recovery should be essentially exact.
        assert (rel_err < 1e-10).all(), f"rel_err={rel_err}"
        # A full-rank imaginary identity exposes transpose/adjoint confusion
        # in both the power iteration and the projected small SVD.
        imaginary_identity = 1j * torch.eye(3, dtype=torch.complex128)
        u, singular, vh = truncated_svd(
            imaginary_identity, r=3, mode="randomized", oversample=0,
            generator=_seeded_generator(0),
        )
        reconstructed = (u * singular.unsqueeze(-2)) @ vh
        torch.testing.assert_close(reconstructed, imaginary_identity,
                                   atol=1e-12, rtol=0)


    def test_randomized_approximates_exact_on_decaying_spectrum(self):
        """For a matrix with a clear singular-value gap, randomized SVD
        approximates the exact truncation to small relative error.

        HMT (2011) Theorem 9.2: with n_iter=q subspace iterations, the
        approximation error is bounded by (1 + (σ_{r+1}/σ_r)^(2q+1) · poly(r,ℓ,n))
        times the optimal Eckart-Young error. For σ_{r+1}/σ_r = 0.5 and q=2,
        the gap term is 0.5^5 ≈ 0.03, which is well within our 1e-6 target.

        Scale-of-validity note (from the catalog): tighter convergence
        requires larger n_iter when σ_{r+1}/σ_r approaches 1.
        """
        m, n, r = 30, 20, 5
        # Geometrically decaying singular values with σ_{r+1}/σ_r = 0.5;
        # squarely in the regime the catalog defaults are tuned for.
        s = torch.tensor([0.5 ** i for i in range(min(m, n))],
                         dtype=torch.float64)
        U_true = torch.linalg.qr(
            torch.randn(m, m, dtype=torch.float64,
                        generator=_seeded_generator(50))
        )[0][:, :min(m, n)]
        V_true = torch.linalg.qr(
            torch.randn(n, n, dtype=torch.float64,
                        generator=_seeded_generator(51))
        )[0][:, :min(m, n)]
        M = U_true @ torch.diag_embed(s) @ V_true.mT

        # Exact reference
        U_ex, S_ex, Vh_ex = truncated_svd(M, r=r, mode="exact")
        M_ex = U_ex @ torch.diag_embed(S_ex) @ Vh_ex

        # Randomized
        U_r, S_r, Vh_r = truncated_svd(M, r=r, mode="randomized",
                                        generator=_seeded_generator(52))
        M_r = U_r @ torch.diag_embed(S_r) @ Vh_r

        # Both should be close to M; we compare against the exact truncation.
        rel_diff = (
            torch.linalg.matrix_norm(M_r - M_ex)
            / torch.linalg.matrix_norm(M_ex)
        )
        # Two power iterations + oversample=5 should comfortably beat 1e-6.
        assert rel_diff < 1e-6, f"rel_diff={rel_diff}"


    def test_randomized_falls_back_when_ell_too_large(self):
        """If r + oversample > min(m, n), the function silently falls back
        to exact (oversampling past full rank is meaningless).
        """
        m, n, r = 4, 5, 3
        # ell = r + 5 = 8 > min(m, n) = 4
        M = torch.randn(m, n, dtype=torch.float64,
                        generator=_seeded_generator(70))
        U, S, Vh = truncated_svd(M, r=r, mode="randomized")
        # Should still produce a valid rank-r decomposition.
        M_recon = U @ torch.diag_embed(S) @ Vh
        # And in this case, the exact truncation is recovered.
        U_ex, S_ex, Vh_ex = truncated_svd(M, r=r, mode="exact")
        M_ex = U_ex @ torch.diag_embed(S_ex) @ Vh_ex
        torch.testing.assert_close(M_recon, M_ex, atol=1e-12, rtol=0)
