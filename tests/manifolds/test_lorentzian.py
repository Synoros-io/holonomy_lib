# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Synoros

"""Tests for holonomy_lib.manifolds.lorentzian.LorentzianManifold.

Five layers:
  1. Construction validation.
  2. Shapes across B ∈ {0, 1, several}.
  3. Property tests; Minkowski-form algebra, causal classification
     (timelike / null / spacelike), proper time + distance.
  4. Autograd; linear ops in flat space are trivially smooth.
"""

from __future__ import annotations

import math

import pytest
import torch

from holonomy_lib.manifolds import LorentzianManifold


def _make_manifold(n=4, dtype=torch.float64):
    return LorentzianManifold(n=n, dtype=dtype)


def _seed(s: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(s)
    return g


# --------------------------------------------------------------------
# Construction
# --------------------------------------------------------------------


class TestConstruction:
    def test_rejects_n_too_small(self):
        with pytest.raises(ValueError, match="n"):
            LorentzianManifold(n=1)
        with pytest.raises(ValueError, match="n"):
            LorentzianManifold(n=0)
        with pytest.raises(ValueError, match="n"):
            LorentzianManifold(n=-2)


# --------------------------------------------------------------------
# Shapes
# --------------------------------------------------------------------


@pytest.mark.parametrize("batch", [0, 1, 4])
class TestShapes:
    def test_random_point(self, batch):
        mfd = _make_manifold()
        x = mfd.random_point(batch_size=batch, generator=_seed(0))
        assert x.shape == (batch, mfd.n)

    def test_origin(self, batch):
        mfd = _make_manifold()
        o = mfd.origin(batch_size=batch)
        assert o.shape == (batch, mfd.n)

    def test_interval(self, batch):
        mfd = _make_manifold()
        x = mfd.random_point(batch_size=batch, generator=_seed(1))
        y = mfd.random_point(batch_size=batch, generator=_seed(2))
        i_sq = mfd.interval_sq(x, y)
        assert i_sq.shape == (batch,)

    def test_causal_type(self, batch):
        mfd = _make_manifold()
        x = mfd.random_point(batch_size=batch, generator=_seed(3))
        y = mfd.random_point(batch_size=batch, generator=_seed(4))
        c = mfd.causal_type(x, y)
        assert c.shape == (batch,)
        assert c.dtype == torch.int64


# --------------------------------------------------------------------
# Minkowski algebra
# --------------------------------------------------------------------


class TestMinkowskiAlgebra:
    def test_inner_signature(self):
        """⟨e_0, e_0⟩_M = -1, ⟨e_i, e_i⟩_M = +1 for i ≥ 1."""
        mfd = _make_manifold(n=4)
        e = torch.eye(4, dtype=torch.float64)
        # Diagonal entries
        diag = torch.stack(
            [mfd.minkowski_inner(e[i:i+1], e[i:i+1]) for i in range(4)],
            dim=0,
        ).squeeze(-1)
        expected = torch.tensor([-1.0, 1.0, 1.0, 1.0], dtype=torch.float64)
        torch.testing.assert_close(diag, expected, atol=1e-12, rtol=0)

    def test_inner_symmetric(self):
        mfd = _make_manifold()
        u = torch.randn(4, mfd.n, dtype=mfd.dtype, generator=_seed(10))
        v = torch.randn(4, mfd.n, dtype=mfd.dtype, generator=_seed(11))
        torch.testing.assert_close(
            mfd.minkowski_inner(u, v),
            mfd.minkowski_inner(v, u),
            atol=1e-12, rtol=0,
        )

    def test_inner_bilinear(self):
        mfd = _make_manifold()
        u = torch.randn(3, mfd.n, dtype=mfd.dtype, generator=_seed(12))
        v = torch.randn(3, mfd.n, dtype=mfd.dtype, generator=_seed(13))
        w = torch.randn(3, mfd.n, dtype=mfd.dtype, generator=_seed(14))
        torch.testing.assert_close(
            mfd.minkowski_inner(u + v, w),
            mfd.minkowski_inner(u, w) + mfd.minkowski_inner(v, w),
            atol=1e-10, rtol=1e-10,
        )


    def test_interval_sq_translation_invariant(self):
        """⟨y - x, y - x⟩_M doesn't depend on a constant shift."""
        mfd = _make_manifold()
        x = torch.randn(3, mfd.n, dtype=mfd.dtype, generator=_seed(16))
        y = torch.randn(3, mfd.n, dtype=mfd.dtype, generator=_seed(17))
        shift = torch.randn(1, mfd.n, dtype=mfd.dtype, generator=_seed(18))
        i_sq = mfd.interval_sq(x, y)
        i_sq_shifted = mfd.interval_sq(x + shift, y + shift)
        torch.testing.assert_close(
            i_sq, i_sq_shifted, atol=1e-10, rtol=1e-10,
        )


# --------------------------------------------------------------------
# Causal classification
# --------------------------------------------------------------------


class TestCausalStructure:
    def test_future_timelike_pair(self):
        """A pair with (Δt)² > Σ(Δx_i)² and Δt > 0 is future-timelike."""
        mfd = _make_manifold(n=4)
        x = torch.zeros(1, 4, dtype=torch.float64)
        # Δ = (2, 0.5, 0.3, 0.2) — Δt² = 4, |Δx|² = 0.38, timelike ✓
        y = torch.tensor([[2.0, 0.5, 0.3, 0.2]], dtype=torch.float64)
        c = mfd.causal_type(x, y)
        assert c.item() == mfd.FUTURE_TIMELIKE

        mfd = _make_manifold(n=4)
        x = torch.zeros(1, 4, dtype=torch.float64)
        y = torch.tensor([[-2.0, 0.5, 0.3, 0.2]], dtype=torch.float64)
        c = mfd.causal_type(x, y)
        assert c.item() == mfd.PAST_TIMELIKE

        mfd = _make_manifold(n=4)
        x = torch.zeros(1, 4, dtype=torch.float64)
        y = torch.tensor([[0.5, 2.0, 1.0, 1.0]], dtype=torch.float64)
        c = mfd.causal_type(x, y)
        assert c.item() == mfd.SPACELIKE

        mfd = _make_manifold(n=4)
        x = torch.zeros(1, 4, dtype=torch.float64)
        # Δ = (√3, 1, 1, 1) — Δt² = 3, |Δx|² = 3, null ✓
        y = torch.tensor([[math.sqrt(3.0), 1.0, 1.0, 1.0]],
                          dtype=torch.float64)
        c = mfd.causal_type(x, y)
        assert c.item() == mfd.FUTURE_NULL

        mfd = _make_manifold(n=4)
        x = torch.zeros(1, 4, dtype=torch.float64)
        y = torch.tensor([[-math.sqrt(3.0), 1.0, 1.0, 1.0]],
                          dtype=torch.float64)
        c = mfd.causal_type(x, y)
        assert c.item() == mfd.PAST_NULL


# --------------------------------------------------------------------
# Proper time / distance
# --------------------------------------------------------------------


class TestProperMeasures:
    def test_proper_time_timelike(self):
        """τ = √(-interval_sq). For Δ=(2, 1, 0, 0): -i_sq = 4-1 = 3 ⇒ τ=√3."""
        mfd = _make_manifold(n=4)
        x = torch.zeros(1, 4, dtype=torch.float64)
        y = torch.tensor([[2.0, 1.0, 0.0, 0.0]], dtype=torch.float64)
        tau = mfd.proper_time(x, y)
        torch.testing.assert_close(
            tau, torch.tensor([math.sqrt(3.0)], dtype=torch.float64),
            atol=1e-12, rtol=0,
        )

        mfd = _make_manifold(n=4)
        x = torch.zeros(1, 4, dtype=torch.float64)
        y = torch.tensor([[0.5, 2.0, 0.0, 0.0]], dtype=torch.float64)
        s = mfd.proper_distance(x, y)
        torch.testing.assert_close(
            s, torch.tensor([math.sqrt(3.75)], dtype=torch.float64),
            atol=1e-12, rtol=0,
        )

    def test_proper_time_spacelike_is_nan(self):
        """proper_time on a spacelike pair returns NaN (type-mismatch)."""
        mfd = _make_manifold(n=4)
        x = torch.zeros(1, 4, dtype=torch.float64)
        y = torch.tensor([[0.5, 2.0, 1.0, 1.0]], dtype=torch.float64)
        tau = mfd.proper_time(x, y)
        assert torch.isnan(tau).all()

        mfd = _make_manifold(n=4)
        x = torch.zeros(1, 4, dtype=torch.float64)
        y = torch.tensor([[2.0, 0.5, 0.3, 0.2]], dtype=torch.float64)
        s = mfd.proper_distance(x, y)
        assert torch.isnan(s).all()


# --------------------------------------------------------------------
# Flat-space geodesics — exp, log, retraction are trivial
# --------------------------------------------------------------------


class TestFlatGeodesics:
    def test_exp_is_addition(self):
        mfd = _make_manifold()
        x = mfd.random_point(batch_size=3, generator=_seed(20))
        v = torch.randn(3, mfd.n, dtype=mfd.dtype, generator=_seed(21))
        out = mfd.exp(x, v)
        torch.testing.assert_close(out, x + v, atol=0, rtol=0)

        mfd = _make_manifold()
        x = mfd.random_point(batch_size=3, generator=_seed(24))
        y = mfd.random_point(batch_size=3, generator=_seed(25))
        torch.testing.assert_close(
            mfd.exp(x, mfd.log(x, y)), y, atol=1e-14, rtol=1e-14,
        )

    def test_log_is_subtraction(self):
        mfd = _make_manifold()
        x = mfd.random_point(batch_size=3, generator=_seed(22))
        y = mfd.random_point(batch_size=3, generator=_seed(23))
        torch.testing.assert_close(
            mfd.log(x, y), y - x, atol=0, rtol=0,
        )


    def test_projection_is_identity(self):
        mfd = _make_manifold()
        x = mfd.random_point(batch_size=3, generator=_seed(26))
        w = torch.randn(3, mfd.n, dtype=mfd.dtype, generator=_seed(27))
        torch.testing.assert_close(mfd.projection(x, w), w,
                                    atol=0, rtol=0)


# --------------------------------------------------------------------
# Autograd-finite — flat space ops are smooth everywhere
# --------------------------------------------------------------------


class TestAutogradFinite:
    def test_interval_sq_backward(self):
        mfd = _make_manifold(n=4)
        x = torch.randn(3, 4, dtype=torch.float64,
                        generator=_seed(30), requires_grad=True)
        y = mfd.random_point(batch_size=3, generator=_seed(31))
        i_sq = mfd.interval_sq(x, y)
        i_sq.sum().backward()
        assert torch.isfinite(x.grad).all()

        mfd = _make_manifold(n=4)
        # Force timelike: large Δt direction
        v = torch.tensor([2.0, 0.1, 0.0, 0.0], dtype=torch.float64,
                          requires_grad=True)
        x = torch.zeros(1, 4, dtype=torch.float64)
        y = x + v.unsqueeze(0)
        tau = mfd.proper_time(x, y)
        tau.sum().backward()
        assert torch.isfinite(v.grad).all()

        mfd = _make_manifold(n=4)
        v = torch.tensor([0.1, 2.0, 0.0, 0.0], dtype=torch.float64,
                          requires_grad=True)
        x = torch.zeros(1, 4, dtype=torch.float64)
        y = x + v.unsqueeze(0)
        s = mfd.proper_distance(x, y)
        s.sum().backward()
        assert torch.isfinite(v.grad).all()


# --------------------------------------------------------------------
# Curvature-tensor primitives (all zero in flat space)
# --------------------------------------------------------------------


class TestCurvatureTensors:
    """Flat Minkowski is Ricci-flat: metric is constant, Christoffel /
    Riemann / Ricci all identically zero. These tests pin the API +
    the flatness of our LorentzianManifold; subclasses for curved
    backgrounds (Schwarzschild, FLRW, etc.) would override and test
    against their known curvature expressions.
    """

    def test_metric_tensor_minkowski_signature(self):
        """g_μν = diag(-1, +1, +1, …, +1)."""
        mfd = _make_manifold(n=4)
        x = mfd.random_point(batch_size=3, generator=_seed(50))
        g = mfd.metric_tensor(x)
        assert g.shape == (3, 4, 4)
        # Diagonal entries: -1 at position (0,0), +1 elsewhere
        expected_diag = torch.tensor([-1.0, 1.0, 1.0, 1.0],
                                       dtype=torch.float64)
        for b in range(3):
            torch.testing.assert_close(
                torch.diagonal(g[b]), expected_diag,
                atol=1e-12, rtol=0,
            )
            # Off-diagonal: zero
            off = g[b] - torch.diag(torch.diagonal(g[b]))
            torch.testing.assert_close(
                off, torch.zeros_like(off), atol=1e-12, rtol=0,
            )

        mfd = _make_manifold(n=4)
        x = mfd.random_point(batch_size=1, generator=_seed(55))
        y = mfd.random_point(batch_size=1, generator=_seed(56))
        g_x = mfd.metric_tensor(x)
        g_y = mfd.metric_tensor(y)
        torch.testing.assert_close(g_x, g_y, atol=0, rtol=0)

    def test_christoffel_zero(self):
        mfd = _make_manifold(n=4)
        x = mfd.random_point(batch_size=3, generator=_seed(51))
        gamma = mfd.christoffel_symbols(x)
        assert gamma.shape == (3, 4, 4, 4)
        torch.testing.assert_close(
            gamma, torch.zeros_like(gamma), atol=0, rtol=0,
        )

        mfd = _make_manifold(n=4)
        x = mfd.random_point(batch_size=2, generator=_seed(52))
        R = mfd.riemann_tensor(x)
        assert R.shape == (2, 4, 4, 4, 4)
        torch.testing.assert_close(
            R, torch.zeros_like(R), atol=0, rtol=0,
        )

        mfd = _make_manifold(n=4)
        x = mfd.random_point(batch_size=2, generator=_seed(53))
        ricci = mfd.ricci_tensor(x)
        assert ricci.shape == (2, 4, 4)
        torch.testing.assert_close(
            ricci, torch.zeros_like(ricci), atol=0, rtol=0,
        )

        mfd = _make_manifold(n=4)
        x = mfd.random_point(batch_size=2, generator=_seed(54))
        scal = mfd.scalar_curvature(x)
        assert scal.shape == (2,)
        torch.testing.assert_close(
            scal, torch.zeros_like(scal), atol=0, rtol=0,
        )


    def test_minkowski_inner_matches_metric_contraction(self):
        """⟨u, v⟩_M = g_μν u^μ v^ν."""
        mfd = _make_manifold(n=4)
        x = mfd.random_point(batch_size=1, generator=_seed(57))
        u = torch.randn(1, 4, dtype=mfd.dtype, generator=_seed(58))
        v = torch.randn(1, 4, dtype=mfd.dtype, generator=_seed(59))
        # Via minkowski_inner
        ip_direct = mfd.minkowski_inner(u, v)
        # Via metric contraction
        g = mfd.metric_tensor(x)
        # g_μν u^μ v^ν = einsum
        ip_metric = torch.einsum("bij,bi,bj->b", g, u, v)
        torch.testing.assert_close(ip_direct, ip_metric,
                                    atol=1e-12, rtol=0)


# --------------------------------------------------------------------
# `inner` for API uniformity (so manifold-generic primitives don't
# crash when passed a LorentzianManifold). The metric is INDEFINITE
# (signed Minkowski form) — `inner(x, v, v)` can be any sign.
# --------------------------------------------------------------------


class TestInnerForApiUniformity:

    def test_inner_self_can_be_signed(self):
        """A timelike v has ⟨v, v⟩_M < 0; a spacelike v has > 0."""
        mfd = _make_manifold(n=4)
        x = mfd.origin(batch_size=1)
        # Pure timelike (Δt = 2, no spatial)
        v_time = torch.tensor([[2.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
        assert mfd.inner(x, v_time, v_time).item() < 0
        # Pure spacelike (no Δt, Δx = 1)
        v_space = torch.tensor([[0.0, 1.0, 0.0, 0.0]], dtype=torch.float64)
        assert mfd.inner(x, v_space, v_space).item() > 0


# --------------------------------------------------------------------
# Interop: LorentzianManifold as a component of ProductManifold
# --------------------------------------------------------------------
