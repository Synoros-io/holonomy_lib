# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Synoros

"""Tests for holonomy_lib.info_geometry.fisher.

Covers:
  1. Shape contract + validation.
  2. Closed-form correctness:
     - Categorical Fisher is diag(1/p).
     - Gaussian-mean Fisher is inv(Sigma).
  3. Natural gradient agrees with inv(F) @ grad (via solve, not inv).
  4. KL Taylor identity: KL(p ‖ p + δ) ≈ ½ δᵀ F(p) δ for small δ.
     This is the defining property of the Fisher metric (Amari 2016,
     §2.5) and ties Fisher back to the existing kl_divergence_*
     primitives.
"""

from __future__ import annotations

import pytest
import torch

from holonomy_lib.info_geometry import (
    fisher_information_categorical,
    fisher_information_gaussian_mean,
    kl_divergence_categorical,
    kl_divergence_gaussian,
    natural_gradient,
)


def _seeded(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def _random_simplex(
    batch: int, k: int, dtype=torch.float64, seed: int = 0,
) -> torch.Tensor:
    """Dirichlet-uniform-ish draws from the interior of the simplex."""
    g = _seeded(seed)
    raw = torch.rand(batch, k, dtype=dtype, generator=g) + 0.1
    return raw / raw.sum(dim=-1, keepdim=True)


def _random_spd(
    batch: int, d: int, dtype=torch.float64, seed: int = 0,
) -> torch.Tensor:
    g = _seeded(seed)
    X = torch.randn(batch, d, d, dtype=dtype, generator=g)
    return torch.matmul(X, X.mT) + d * torch.eye(d, dtype=dtype)


# --------------------------------------------------------------------
# Categorical Fisher
# --------------------------------------------------------------------


class TestCategoricalFisher:
    def test_shape(self):
        p = _random_simplex(batch=3, k=5)
        F = fisher_information_categorical(p)
        assert F.shape == (3, 5, 5)

    def test_is_diagonal(self):
        p = _random_simplex(batch=2, k=4)
        F = fisher_information_categorical(p)
        diag = torch.diagonal(F, dim1=-2, dim2=-1)
        F_offdiag = F - torch.diag_embed(diag)
        torch.testing.assert_close(
            F_offdiag, torch.zeros_like(F_offdiag), atol=1e-12, rtol=0,
        )

    def test_diagonal_entries_are_reciprocal_p(self):
        p = _random_simplex(batch=2, k=4, seed=1)
        F = fisher_information_categorical(p)
        diag = torch.diagonal(F, dim1=-2, dim2=-1)
        torch.testing.assert_close(
            diag, 1.0 / p, atol=1e-9, rtol=0,
        )

    def test_clamps_zero_entries(self):
        """An exact-zero entry should not blow up to inf; the numerical
        floor `1e-9` keeps the Fisher finite (and large). Verify the
        clamp is narrowly applied — non-zero entries must be untouched."""
        p = torch.tensor([[0.0, 0.5, 0.5]], dtype=torch.float64)
        F = fisher_information_categorical(p)
        assert torch.isfinite(F).all()
        # Diagonal entry for p_0 = 0 should be 1/clamp_floor = 1e9.
        assert F[0, 0, 0].item() == pytest.approx(1e9, rel=1e-3)
        # Non-zero entries must be the unclamped 1/p_i, not lumped
        # together with the clamped value.
        assert F[0, 1, 1].item() == pytest.approx(2.0, rel=1e-12)
        assert F[0, 2, 2].item() == pytest.approx(2.0, rel=1e-12)


# --------------------------------------------------------------------
# Gaussian-mean Fisher
# --------------------------------------------------------------------


class TestGaussianMeanFisher:
    def test_shape(self):
        Sigma = _random_spd(batch=3, d=4)
        F = fisher_information_gaussian_mean(Sigma)
        assert F.shape == (3, 4, 4)

    def test_matches_inv_sigma(self):
        Sigma = _random_spd(batch=2, d=5, seed=7)
        F = fisher_information_gaussian_mean(Sigma)
        ref = torch.linalg.inv(Sigma)
        torch.testing.assert_close(F, ref, atol=1e-9, rtol=0)

    def test_rejects_non_square(self):
        with pytest.raises(ValueError, match="must be"):
            fisher_information_gaussian_mean(torch.zeros(1, 3, 4))

    def test_identity_covariance_yields_identity_fisher(self):
        """N(μ, I) → g(μ) = I."""
        d = 4
        Sigma = torch.eye(d).unsqueeze(0).expand(2, d, d).clone()
        F = fisher_information_gaussian_mean(Sigma)
        I_batch = torch.eye(d).unsqueeze(0).expand(2, d, d)
        torch.testing.assert_close(F, I_batch, atol=1e-12, rtol=0)


# --------------------------------------------------------------------
# Natural gradient
# --------------------------------------------------------------------


class TestNaturalGradient:
    def test_shape(self):
        d = 5
        grad = torch.randn(3, d, dtype=torch.float64, generator=_seeded(0))
        F = _random_spd(batch=3, d=d, seed=1)
        ng = natural_gradient(grad, F)
        assert ng.shape == (3, d)

    def test_matches_inv_F_times_grad(self):
        d = 5
        grad = torch.randn(2, d, dtype=torch.float64, generator=_seeded(0))
        F = _random_spd(batch=2, d=d, seed=2)
        ng = natural_gradient(grad, F)
        ref = torch.matmul(
            torch.linalg.inv(F), grad.unsqueeze(dim=-1),
        ).squeeze(dim=-1)
        torch.testing.assert_close(ng, ref, atol=1e-9, rtol=0)

    def test_identity_fisher_passes_grad_through(self):
        d = 4
        grad = torch.randn(2, d, dtype=torch.float64, generator=_seeded(0))
        F = torch.eye(d, dtype=torch.float64).unsqueeze(0).expand(2, d, d)
        ng = natural_gradient(grad, F)
        torch.testing.assert_close(ng, grad, atol=1e-12, rtol=0)

    def test_rejects_dim_mismatch(self):
        grad = torch.randn(2, 5, dtype=torch.float64)
        F = _random_spd(batch=2, d=4, seed=0)
        with pytest.raises(ValueError, match="last dim"):
            natural_gradient(grad, F)

    def test_rejects_1d_fisher_matrix(self):
        """fisher_matrix must be at least 2-D; reject early with a
        clear error instead of a cryptic torch internal."""
        grad = torch.randn(3, dtype=torch.float64)
        F_bad = torch.randn(3, dtype=torch.float64)
        with pytest.raises(ValueError, match="at least 2 dims"):
            natural_gradient(grad, F_bad)


# --------------------------------------------------------------------
# KL Taylor identity: Fisher is the Hessian of KL at the diagonal.
# This is the defining property tying Fisher back to KL.
# --------------------------------------------------------------------


class TestKlTaylorIdentity:
    def test_categorical_kl_second_order_matches_fisher(self):
        """KL(p ‖ p + δ) ≈ ½ δᵀ F(p) δ for small δ ∈ T_p Δ.

        Take δ in the tangent space (Σ δ_i = 0), small enough that
        third-order terms are negligible, and check the ratio.
        """
        p = torch.tensor([[0.2, 0.3, 0.5]], dtype=torch.float64)
        eps = 1e-4
        # Tangent direction: must sum to 0 so p + δ stays on the simplex.
        delta = torch.tensor([[1.0, -1.0, 0.0]], dtype=torch.float64) * eps

        q = p + delta
        kl = kl_divergence_categorical(p, q)

        F = fisher_information_categorical(p)
        # ½ δᵀ F δ
        Fd = torch.matmul(F, delta.unsqueeze(dim=-1)).squeeze(dim=-1)
        quadratic = 0.5 * (delta * Fd).sum(dim=-1)

        # Third-order error scales as O(‖δ‖³) ≈ eps^3. With eps=1e-4
        # that's 1e-12; with quadratic ~ eps^2 = 1e-8 the relative
        # error should be ~ eps = 1e-4. Allow a generous 1e-3.
        rel_err = ((kl - quadratic) / quadratic).abs().item()
        assert rel_err < 1e-3, (
            f"KL Taylor expansion mismatch: rel_err={rel_err:.2e}"
        )

    def test_gaussian_kl_second_order_matches_mean_fisher(self):
        """KL(N(μ, Σ) ‖ N(μ + δμ, Σ)) ≈ ½ δμᵀ Σ^{−1} δμ at fixed Σ.

        Pin the covariance equal on both sides; only μ varies. Fisher
        for the mean parameter is Σ^{−1}.
        """
        d = 4
        Sigma = _random_spd(batch=1, d=d, seed=5)
        mu_p = torch.zeros(1, d, dtype=torch.float64)
        eps = 1e-3
        delta = torch.randn(1, d, dtype=torch.float64, generator=_seeded(7)) * eps
        mu_q = mu_p + delta

        kl = kl_divergence_gaussian(mu_p, Sigma, mu_q, Sigma)

        F = fisher_information_gaussian_mean(Sigma)
        Fd = torch.matmul(F, delta.unsqueeze(dim=-1)).squeeze(dim=-1)
        quadratic = 0.5 * (delta * Fd).sum(dim=-1)

        # At fixed Σ the second-order term is exact (KL is purely
        # quadratic in δμ — there is no third-order term). The error
        # is at numerical roundoff.
        torch.testing.assert_close(kl, quadratic, atol=1e-10, rtol=1e-8)
