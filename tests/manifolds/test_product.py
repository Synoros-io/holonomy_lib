# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Synoros

"""Tests for holonomy_lib.manifolds.product.ProductManifold."""

from __future__ import annotations

import pytest
import torch

from holonomy_lib.manifolds import (
    KappaStereographicManifold,
    LorentzManifold,
    ProductManifold,
)


def _seed(s: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(s)
    return g


def _make_mfd():
    """Standard test fixture: Euclidean × Hyperbolic (Lorentz)."""
    return ProductManifold([
        KappaStereographicManifold(n=3, kappa=0.0, dtype=torch.float64),
        LorentzManifold(n=3, k=-1.0, dtype=torch.float64),
    ])


# --------------------------------------------------------------------
# Construction
# --------------------------------------------------------------------


class TestConstruction:
    def test_rejects_empty(self):
        with pytest.raises(ValueError, match="at least one"):
            ProductManifold([])

    def test_rejects_negative_weight(self):
        with pytest.raises(ValueError, match="non-negative"):
            ProductManifold(
                [KappaStereographicManifold(n=3, kappa=0.0)],
                weights=[-1.0],
            )

    def test_rejects_mismatched_weights(self):
        with pytest.raises(ValueError, match="length"):
            ProductManifold(
                [KappaStereographicManifold(n=3, kappa=0.0)],
                weights=[1.0, 2.0],
            )

    def test_dim_and_ambient(self):
        mfd = _make_mfd()
        # 3 + 3 = 6 intrinsic, 3 + 4 = 7 ambient
        assert mfd.dim == 6
        assert mfd.ambient_dim == 7


# --------------------------------------------------------------------
# Pythagorean distance
# --------------------------------------------------------------------


class TestPythagoreanDistance:
    def test_distance_decomposes_per_component(self):
        """d²((x_E, x_H), (y_E, y_H)) = d_E² + d_H²."""
        mfd = _make_mfd()
        x = mfd.random_point(batch_size=4, generator=_seed(0))
        y = mfd.random_point(batch_size=4, generator=_seed(1))
        d_total = mfd.distance(x, y)
        mfd_e = mfd.manifolds[0]
        mfd_h = mfd.manifolds[1]
        d_e = mfd_e.distance(mfd.component(x, 0), mfd.component(y, 0))
        d_h = mfd_h.distance(mfd.component(x, 1), mfd.component(y, 1))
        expected = torch.sqrt(d_e * d_e + d_h * d_h)
        torch.testing.assert_close(d_total, expected, atol=1e-10, rtol=1e-10)

    def test_weighted_distance(self):
        """w-weighted Pythagorean: d² = Σ_i w_i · d_i²."""
        weights = [2.0, 0.5]
        mfd = ProductManifold(
            [
                KappaStereographicManifold(n=3, kappa=0.0, dtype=torch.float64),
                LorentzManifold(n=3, k=-1.0, dtype=torch.float64),
            ],
            weights=weights,
        )
        x = mfd.random_point(batch_size=3, generator=_seed(2))
        y = mfd.random_point(batch_size=3, generator=_seed(3))
        d_total = mfd.distance(x, y)
        d_e = mfd.manifolds[0].distance(
            mfd.component(x, 0), mfd.component(y, 0),
        )
        d_h = mfd.manifolds[1].distance(
            mfd.component(x, 1), mfd.component(y, 1),
        )
        expected = torch.sqrt(2.0 * d_e * d_e + 0.5 * d_h * d_h)
        torch.testing.assert_close(d_total, expected, atol=1e-10, rtol=1e-10)

    def test_distance_at_same_point(self):
        mfd = _make_mfd()
        x = mfd.random_point(batch_size=3, generator=_seed(4))
        d = mfd.distance(x, x)
        assert d.abs().max().item() < 1e-9

    def test_distance_symmetric(self):
        mfd = _make_mfd()
        x = mfd.random_point(batch_size=4, generator=_seed(5))
        y = mfd.random_point(batch_size=4, generator=_seed(6))
        torch.testing.assert_close(
            mfd.distance(x, y), mfd.distance(y, x),
            atol=1e-10, rtol=0,
        )


# --------------------------------------------------------------------
# Component delegation
# --------------------------------------------------------------------


class TestComponentDelegation:
    def test_exp_log_per_component(self):
        """log_x(y) and exp_x(v) decompose per submanifold."""
        mfd = _make_mfd()
        x = mfd.random_point(batch_size=3, generator=_seed(10))
        y = mfd.random_point(batch_size=3, generator=_seed(11))
        v = mfd.log(x, y)
        # log per-component
        v_e = mfd.manifolds[0].log(mfd.component(x, 0), mfd.component(y, 0))
        v_h = mfd.manifolds[1].log(mfd.component(x, 1), mfd.component(y, 1))
        torch.testing.assert_close(
            mfd.component(v, 0), v_e, atol=1e-12, rtol=0,
        )
        torch.testing.assert_close(
            mfd.component(v, 1), v_h, atol=1e-12, rtol=0,
        )
        # exp(x, log(x, y)) = y per-component (within float)
        y_back = mfd.exp(x, v)
        torch.testing.assert_close(y_back, y, atol=1e-9, rtol=1e-9)

    def test_inner_decomposes(self):
        """⟨u, v⟩_x = Σ w_i · ⟨u_i, v_i⟩_{x_i}."""
        mfd = _make_mfd()
        x = mfd.random_point(batch_size=3, generator=_seed(12))
        v_seed = torch.randn(3, mfd.ambient_dim, dtype=torch.float64,
                              generator=_seed(13))
        v = mfd.projection(x, v_seed)
        w_seed = torch.randn(3, mfd.ambient_dim, dtype=torch.float64,
                              generator=_seed(14))
        w = mfd.projection(x, w_seed)
        ip_total = mfd.inner(x, v, w)
        ip_e = mfd.manifolds[0].inner(
            mfd.component(x, 0), mfd.component(v, 0), mfd.component(w, 0),
        )
        ip_h = mfd.manifolds[1].inner(
            mfd.component(x, 1), mfd.component(v, 1), mfd.component(w, 1),
        )
        expected = ip_e + ip_h
        torch.testing.assert_close(
            ip_total, expected, atol=1e-12, rtol=0,
        )

    def test_is_on_manifold_requires_all_components(self):
        """A product point is on the manifold iff every component is."""
        mfd = _make_mfd()
        x = mfd.random_point(batch_size=3, generator=_seed(15))
        assert mfd.is_on_manifold(x).all()
        # Corrupt the Lorentz part (move it off the hyperboloid)
        x_bad = x.clone()
        x_bad[:, 3:] = torch.randn(3, 4, dtype=torch.float64,
                                   generator=_seed(16))
        # Some elements may coincidentally be on-manifold; not all.
        assert not mfd.is_on_manifold(x_bad).all()


# --------------------------------------------------------------------
# Tangent-at-origin (exp_0 / log_0)
# --------------------------------------------------------------------


class TestExpLogOrigin:
    def test_exp_0_log_0_inverse(self):
        mfd = _make_mfd()
        v = (torch.randn(3, mfd.dim, dtype=torch.float64,
                         generator=_seed(20)) * 0.1)
        y = mfd.exp_0(v)
        v_back = mfd.log_0(y)
        torch.testing.assert_close(v_back, v, atol=1e-9, rtol=1e-9)

    def test_exp_0_shape(self):
        mfd = _make_mfd()
        v = torch.randn(5, mfd.dim, dtype=torch.float64,
                        generator=_seed(21)) * 0.1
        y = mfd.exp_0(v)
        assert y.shape == (5, mfd.ambient_dim)


# --------------------------------------------------------------------
# Autograd-finite
# --------------------------------------------------------------------


class TestAutogradFinite:
    def test_distance_backward_at_same_point(self):
        """The classic d(x, x) = 0 → backward NaN trap: ProductManifold
        inherits the autograd-safe distance from each component, plus
        another _safe_sqrt over the sum-of-squares."""
        mfd = _make_mfd()
        v = (torch.randn(5, mfd.dim, dtype=torch.float64,
                         generator=_seed(30)) * 0.1)
        v.requires_grad_(True)
        T = mfd.exp_0(v)
        d = mfd.distance(T, T)  # all zeros — boundary case
        d.sum().backward()
        assert torch.isfinite(v.grad).all()

    def test_full_substrate_chain(self):
        """v → exp_0(v) → all-pairs distance → NLL → backward."""
        mfd = _make_mfd()
        v = (torch.randn(6, mfd.dim, dtype=torch.float64,
                         generator=_seed(31)) * 0.15)
        v.requires_grad_(True)
        T = mfd.exp_0(v)
        N = T.shape[0]
        Ti = T.unsqueeze(1).expand(N, N, mfd.ambient_dim).reshape(
            -1, mfd.ambient_dim,
        )
        Tj = T.unsqueeze(0).expand(N, N, mfd.ambient_dim).reshape(
            -1, mfd.ambient_dim,
        )
        d_all = mfd.distance(Ti, Tj).reshape(N, N)
        log_partition = torch.logsumexp(-d_all, dim=-1)
        target_d = d_all[torch.arange(N), (torch.arange(N) + 1) % N]
        loss = (target_d + log_partition).sum()
        loss.backward()
        assert torch.isfinite(v.grad).all()


# --------------------------------------------------------------------
# Provenance roundtrip
# --------------------------------------------------------------------


class TestProvenance:
    def test_signature_roundtrip(self):
        mfd = _make_mfd()
        sig = mfd._provenance_signature()
        mfd2 = ProductManifold._from_signature(sig)
        assert mfd2.dim == mfd.dim
        assert mfd2.ambient_dim == mfd.ambient_dim
        # Recomputed distance should match (within float roundoff)
        x = mfd.random_point(batch_size=2, generator=_seed(40))
        y = mfd.random_point(batch_size=2, generator=_seed(41))
        d_a = mfd.distance(x, y)
        d_b = mfd2.distance(x, y)
        torch.testing.assert_close(d_a, d_b, atol=1e-12, rtol=0)
