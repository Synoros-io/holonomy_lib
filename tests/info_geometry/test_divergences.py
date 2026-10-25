# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Synoros

"""Tests for holonomy_lib.info_geometry.divergences.

Four layers:
  1. Identity: D(p, p) = 0 for all three divergences.
  2. Non-negativity (Gibbs / Bregman strict convexity).
  3. Closed-form sanity checks (Bregman with ½‖·‖² = squared Euclidean,
     KL between standard normals, Pinsker-style bounds).
  4. Validation (shape mismatches raise).
"""

from __future__ import annotations

import math

import pytest
import torch

from holonomy_lib.info_geometry import (
    bregman_divergence,
    kl_divergence_categorical,
    kl_divergence_gaussian,
)


def _seeded_generator(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


# --------------------------------------------------------------------
# Bregman divergence
# --------------------------------------------------------------------


def _half_norm_sq(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """F(x) = ½‖x‖²; ∇F(x) = x. Recovers squared-Euclidean Bregman."""
    return 0.5 * (x * x).sum(dim=-1), x


def _neg_entropy(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """F(x) = Σ x log x − x; ∇F(x) = log x. Recovers generalized KL."""
    log_x = torch.log(x.clamp(min=1e-9))
    return (x * log_x - x).sum(dim=-1), log_x


class TestBregman:
    def test_identity_self_distance_zero(self):
        p = torch.randn(4, 5, dtype=torch.float64, generator=_seeded_generator(0))
        d = bregman_divergence(p, p, _half_norm_sq)
        torch.testing.assert_close(
            d, torch.zeros_like(d), atol=1e-10, rtol=0,
        )

    def test_squared_euclidean_form(self):
        """With F = ½‖·‖², Bregman recovers ½‖p − q‖²."""
        g = _seeded_generator(1)
        p = torch.randn(4, 5, dtype=torch.float64, generator=g)
        q = torch.randn(4, 5, dtype=torch.float64, generator=g)
        d_bregman = bregman_divergence(p, q, _half_norm_sq)
        d_expected = 0.5 * ((p - q) ** 2).sum(dim=-1)
        torch.testing.assert_close(d_bregman, d_expected, atol=1e-12, rtol=0)

    def test_non_negative_on_random_inputs(self):
        """Bregman with strictly convex F is non-negative."""
        g = _seeded_generator(2)
        p = torch.randn(8, 6, dtype=torch.float64, generator=g)
        q = torch.randn(8, 6, dtype=torch.float64, generator=g)
        d = bregman_divergence(p, q, _half_norm_sq)
        assert (d >= -1e-10).all()

    def test_generalized_kl_matches_categorical_kl_on_simplex(self):
        """For p, q on the simplex (Σ x_i = 1), the generalized KL
        from F = Σ x log x − x reduces to the standard categorical KL.
        """
        g = _seeded_generator(3)
        # Build simplex distributions
        p_raw = torch.rand(4, 5, dtype=torch.float64, generator=g)
        q_raw = torch.rand(4, 5, dtype=torch.float64, generator=g)
        p = p_raw / p_raw.sum(dim=-1, keepdim=True)
        q = q_raw / q_raw.sum(dim=-1, keepdim=True)
        d_bregman = bregman_divergence(p, q, _neg_entropy)
        d_kl = kl_divergence_categorical(p, q)
        torch.testing.assert_close(d_bregman, d_kl, atol=1e-9, rtol=0)

    def test_shape_mismatch_raises(self):
        p = torch.randn(4, 5)
        q = torch.randn(4, 6)
        with pytest.raises(ValueError, match="matching"):
            bregman_divergence(p, q, _half_norm_sq)


# --------------------------------------------------------------------
# Categorical KL
# --------------------------------------------------------------------


class TestKLCategorical:
    def test_identity_self_distance_zero(self):
        g = _seeded_generator(10)
        p_raw = torch.rand(3, 4, dtype=torch.float64, generator=g)
        p = p_raw / p_raw.sum(dim=-1, keepdim=True)
        d = kl_divergence_categorical(p, p)
        torch.testing.assert_close(d, torch.zeros_like(d), atol=1e-12, rtol=0)

    def test_non_negative(self):
        """Gibbs inequality."""
        g = _seeded_generator(11)
        p_raw = torch.rand(8, 6, dtype=torch.float64, generator=g)
        q_raw = torch.rand(8, 6, dtype=torch.float64, generator=g)
        p = p_raw / p_raw.sum(dim=-1, keepdim=True)
        q = q_raw / q_raw.sum(dim=-1, keepdim=True)
        d = kl_divergence_categorical(p, q)
        assert (d >= -1e-12).all()

    def test_uniform_vs_delta(self):
        """KL(δ_0 ‖ uniform) on k=4 support = log 4 (≈ 1.386)."""
        p = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
        q = torch.full((1, 4), 0.25, dtype=torch.float64)
        d = kl_divergence_categorical(p, q)
        # ∑ p log(p/q) = 1·log(1/0.25) = log 4 (other terms have p=0)
        assert d[0].item() == pytest.approx(math.log(4.0), abs=1e-9)

    def test_zero_in_p_does_not_blow_up(self):
        """The 0·log 0 = 0 convention must hold."""
        p = torch.tensor([[0.5, 0.5, 0.0, 0.0]], dtype=torch.float64)
        q = torch.tensor([[0.25, 0.25, 0.25, 0.25]], dtype=torch.float64)
        d = kl_divergence_categorical(p, q)
        assert torch.isfinite(d).all()
        # Closed form: 2 · 0.5 log(0.5/0.25) = log 2.
        assert d[0].item() == pytest.approx(math.log(2.0), abs=1e-9)

    def test_zero_in_q_with_support_in_p_returns_inf(self):
        """If `q` has a zero where `p` has positive mass, KL diverges
        (supp(p) ⊄ supp(q); Gibbs's inequality bound is +inf). Regression
        for scrutiny-pass-3 bug: a previous symmetric clamp on `q`
        returned ~20 (= log(1e9)) instead of +inf."""
        p = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
        q = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
        d = kl_divergence_categorical(p, q)
        assert torch.isinf(d[0]).item() and d[0].item() > 0

    def test_zero_in_q_outside_support_of_p_is_finite(self):
        """If `q_i = 0` where `p_i = 0`, the 0·log convention applies
        and KL stays finite."""
        p = torch.tensor([[0.5, 0.5, 0.0]], dtype=torch.float64)
        q = torch.tensor([[0.5, 0.5, 0.0]], dtype=torch.float64)
        d = kl_divergence_categorical(p, q)
        assert d[0].item() == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------
# Gaussian KL
# --------------------------------------------------------------------


def _random_spd(d: int, batch: int, dtype=torch.float64, seed: int = 0):
    g = _seeded_generator(seed)
    A = torch.randn(batch, d, d, dtype=dtype, generator=g)
    # Wishart-style A Aᵀ + ε I → SPD and well-conditioned.
    return A @ A.mT + torch.eye(d, dtype=dtype).unsqueeze(0) * d


class TestKLGaussian:
    def test_identity_self_distance_zero(self):
        d = 4
        mu = torch.zeros(2, d, dtype=torch.float64)
        Sigma = _random_spd(d, batch=2, seed=20)
        kl = kl_divergence_gaussian(mu, Sigma, mu, Sigma)
        torch.testing.assert_close(
            kl, torch.zeros_like(kl), atol=1e-9, rtol=0,
        )

    def test_non_negative(self):
        d = 4
        mu_p = torch.randn(3, d, dtype=torch.float64,
                            generator=_seeded_generator(21))
        mu_q = torch.randn(3, d, dtype=torch.float64,
                            generator=_seeded_generator(22))
        Sigma_p = _random_spd(d, batch=3, seed=23)
        Sigma_q = _random_spd(d, batch=3, seed=24)
        kl = kl_divergence_gaussian(mu_p, Sigma_p, mu_q, Sigma_q)
        assert (kl >= -1e-9).all()

    def test_standard_normals_zero(self):
        """KL(N(0, I) ‖ N(0, I)) = 0."""
        d = 4
        mu = torch.zeros(1, d, dtype=torch.float64)
        Sigma = torch.eye(d, dtype=torch.float64).unsqueeze(0)
        kl = kl_divergence_gaussian(mu, Sigma, mu, Sigma)
        torch.testing.assert_close(kl, torch.zeros_like(kl), atol=1e-12, rtol=0)

    def test_mean_shift_only(self):
        """N(μ, I) vs N(0, I) ⇒ KL = ½‖μ‖²."""
        d = 3
        mu_p = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64)
        mu_q = torch.zeros(1, d, dtype=torch.float64)
        Sigma = torch.eye(d, dtype=torch.float64).unsqueeze(0)
        kl = kl_divergence_gaussian(mu_p, Sigma, mu_q, Sigma)
        expected = 0.5 * (mu_p * mu_p).sum(dim=-1)
        torch.testing.assert_close(kl, expected, atol=1e-12, rtol=0)

    def test_scalar_variance_1d(self):
        """1-D Gaussians: KL(N(μ_p, σ_p²) ‖ N(μ_q, σ_q²))
        = log(σ_q/σ_p) + (σ_p² + (μ_p − μ_q)²) / (2 σ_q²) − ½.
        """
        mu_p = torch.tensor([[0.5]], dtype=torch.float64)
        mu_q = torch.tensor([[1.5]], dtype=torch.float64)
        Sigma_p = torch.tensor([[[2.0]]], dtype=torch.float64)  # σ_p² = 2
        Sigma_q = torch.tensor([[[3.0]]], dtype=torch.float64)  # σ_q² = 3
        kl = kl_divergence_gaussian(mu_p, Sigma_p, mu_q, Sigma_q)
        sigma_p, sigma_q = math.sqrt(2.0), math.sqrt(3.0)
        expected = (
            math.log(sigma_q / sigma_p)
            + (2.0 + (0.5 - 1.5) ** 2) / (2.0 * 3.0)
            - 0.5
        )
        assert kl[0].item() == pytest.approx(expected, abs=1e-12)

    def test_shape_mismatch_raises(self):
        mu = torch.zeros(1, 4)
        bad_Sigma = torch.eye(5).unsqueeze(0)
        with pytest.raises(ValueError, match="Sigma_p"):
            kl_divergence_gaussian(mu, bad_Sigma, mu, torch.eye(4).unsqueeze(0))

    def test_mu_q_shape_mismatch_raises(self):
        """Regression: previously `mu_q` was unvalidated against `d`,
        producing opaque cholesky_solve errors. Now it raises a clear
        ValueError up front."""
        mu_p = torch.zeros(1, 4)
        mu_q_bad = torch.zeros(1, 5)
        Sigma = torch.eye(4).unsqueeze(0)
        with pytest.raises(ValueError, match="mu_q"):
            kl_divergence_gaussian(mu_p, Sigma, mu_q_bad, Sigma)
