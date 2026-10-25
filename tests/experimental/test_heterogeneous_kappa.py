# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Synoros

"""Tests for holonomy_lib.experimental.heterogeneous_kappa.
HeterogeneousKappaManifold."""

from __future__ import annotations

import pytest
import torch

from holonomy_lib.experimental import HeterogeneousKappaManifold
from holonomy_lib.manifolds import KappaStereographicManifold


def _seed(s: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(s)
    return g


# --------------------------------------------------------------------
# Construction
# --------------------------------------------------------------------


class TestConstruction:
    def test_rejects_n_zero(self):
        with pytest.raises(ValueError, match="n"):
            HeterogeneousKappaManifold(n=0)

    def test_rejects_unknown_combiner(self):
        with pytest.raises(ValueError, match="combiner"):
            HeterogeneousKappaManifold(n=3, combiner="bogus")

    def test_accepts_callable_combiner(self):
        def custom(a, b):
            return (a + b) / 2
        mfd = HeterogeneousKappaManifold(n=3, combiner=custom)
        # Just check it doesn't reject
        assert mfd.combiner is custom

    def test_rejects_non_callable_non_string(self):
        with pytest.raises(TypeError, match="combiner"):
            HeterogeneousKappaManifold(n=3, combiner=42)


# --------------------------------------------------------------------
# Homogeneous case agrees with KappaStereographicManifold
# --------------------------------------------------------------------


class TestHomogeneousAgreement:
    """When all κ's are equal, results match `KappaStereographicManifold`
    at that κ. Pins the math against the established homogeneous case."""

    @pytest.mark.parametrize("k_val", [-1.0, -0.5, 0.5, 1.0])
    def test_exp_0_matches_homogeneous(self, k_val):
        het = HeterogeneousKappaManifold(n=3, dtype=torch.float64)
        homo = KappaStereographicManifold(
            n=3, kappa=k_val, dtype=torch.float64,
        )
        v = torch.randn(4, 3, dtype=torch.float64, generator=_seed(0)) * 0.2
        kappas = torch.full((4,), k_val, dtype=torch.float64)
        T_het = het.exp_0(v, kappas)
        T_homo = homo.exp_0(v)
        torch.testing.assert_close(T_het, T_homo, atol=1e-10, rtol=1e-10)

    @pytest.mark.parametrize("k_val", [-1.0, -0.5, 0.5, 1.0])
    def test_distance_matches_homogeneous(self, k_val):
        het = HeterogeneousKappaManifold(n=3, dtype=torch.float64)
        homo = KappaStereographicManifold(
            n=3, kappa=k_val, dtype=torch.float64,
        )
        v = torch.randn(4, 3, dtype=torch.float64, generator=_seed(1)) * 0.2
        kappas = torch.full((4,), k_val, dtype=torch.float64)
        x = het.exp_0(v, kappas)
        y = het.exp_0(
            v + 0.05 * torch.randn(4, 3, dtype=torch.float64,
                                    generator=_seed(2)),
            kappas,
        )
        d_het = het.distance(x, kappas, y, kappas)
        d_homo = homo.distance(x, y)
        torch.testing.assert_close(d_het, d_homo, atol=1e-9, rtol=1e-9)

    def test_log_0_matches_homogeneous(self):
        het = HeterogeneousKappaManifold(n=3, dtype=torch.float64)
        homo = KappaStereographicManifold(
            n=3, kappa=-1.0, dtype=torch.float64,
        )
        v = torch.randn(4, 3, dtype=torch.float64, generator=_seed(3)) * 0.2
        kappas = torch.full((4,), -1.0, dtype=torch.float64)
        x = het.exp_0(v, kappas)
        v_back_het = het.log_0(x, kappas)
        v_back_homo = homo.log_0(x)
        torch.testing.assert_close(
            v_back_het, v_back_homo, atol=1e-10, rtol=1e-10,
        )


# --------------------------------------------------------------------
# Heterogeneous case: per-point κ produces different geometry
# --------------------------------------------------------------------


class TestHeterogeneousBehavior:
    def test_different_kappas_give_different_embeddings(self):
        """Same tangent at origin, but different κ → different embedded
        point. This is the whole point of the class."""
        het = HeterogeneousKappaManifold(n=3, dtype=torch.float64)
        v = torch.tensor([[0.3, 0.0, 0.0]], dtype=torch.float64)
        # Try two different κ's on the same tangent
        T1 = het.exp_0(v, torch.tensor([-1.0], dtype=torch.float64))
        T2 = het.exp_0(v, torch.tensor([-0.5], dtype=torch.float64))
        # Different embedded points
        assert not torch.allclose(T1, T2)

    def test_distance_asymmetric_in_kappa(self):
        """d(x, κ_x, y, κ_y) is symmetric in (x, κ_x) ↔ (y, κ_y) (the
        combiner is commutative). But changing one κ changes the
        distance — that's the heterogeneous behavior."""
        het = HeterogeneousKappaManifold(n=3, dtype=torch.float64)
        x = torch.tensor([[0.1, 0.0, 0.0]], dtype=torch.float64)
        y = torch.tensor([[0.0, 0.1, 0.0]], dtype=torch.float64)
        kx = torch.tensor([-1.0], dtype=torch.float64)
        ky_a = torch.tensor([-1.0], dtype=torch.float64)
        ky_b = torch.tensor([-0.5], dtype=torch.float64)
        d_a = het.distance(x, kx, y, ky_a)
        d_b = het.distance(x, kx, y, ky_b)
        # Symmetry: pair swap shouldn't change anything
        d_a_swap = het.distance(y, ky_a, x, kx)
        torch.testing.assert_close(d_a, d_a_swap, atol=1e-12, rtol=0)
        # But changing κ_y to a different value DOES change distance
        assert not torch.allclose(d_a, d_b)

    def test_distance_at_same_point(self):
        het = HeterogeneousKappaManifold(n=3, dtype=torch.float64)
        x = torch.tensor([[0.1, -0.1, 0.05]], dtype=torch.float64)
        kappa = torch.tensor([-1.0], dtype=torch.float64)
        d = het.distance(x, kappa, x, kappa)
        assert d.abs().max().item() < 1e-9


# --------------------------------------------------------------------
# Autograd through both v and κ
# --------------------------------------------------------------------


class TestAutogradFinite:
    def test_grad_through_v_and_kappa(self):
        """Substrate-style: v and κ both learnable, loss depends on
        per-pair distance. Both gradients must be finite."""
        het = HeterogeneousKappaManifold(n=3, dtype=torch.float64)
        N = 6
        v = (torch.randn(N, 3, dtype=torch.float64,
                          generator=_seed(10)) * 0.15)
        v.requires_grad_(True)
        kappas = torch.nn.Parameter(
            torch.randn(N, dtype=torch.float64, generator=_seed(11)) * 0.5
            - 0.5  # bias toward hyperbolic
        )
        T = het.exp_0(v, kappas)
        # All-pairs distance with the corresponding κ's
        Ti = T.unsqueeze(1).expand(N, N, 3).reshape(-1, 3)
        Tj = T.unsqueeze(0).expand(N, N, 3).reshape(-1, 3)
        Ki = kappas.unsqueeze(1).expand(N, N).reshape(-1)
        Kj = kappas.unsqueeze(0).expand(N, N).reshape(-1)
        d = het.distance(Ti, Ki, Tj, Kj).reshape(N, N)
        loss = d.sum()
        loss.backward()
        assert torch.isfinite(v.grad).all()
        assert torch.isfinite(kappas.grad).all()
        # κ's gradient is non-trivial (the loss does depend on the κ's)
        assert kappas.grad.abs().max() > 0

    def test_grad_at_self_pair_kappa_only(self):
        """d(x, κ_x, x, κ_x) = 0 — gradient on κ should be 0 or near it
        (no asymmetry to learn from). Just verify finite."""
        het = HeterogeneousKappaManifold(n=3, dtype=torch.float64)
        kappa = torch.nn.Parameter(torch.tensor(-1.0, dtype=torch.float64))
        x = torch.tensor([[0.1, 0.0, 0.0]], dtype=torch.float64)
        d = het.distance(x, kappa.unsqueeze(0), x, kappa.unsqueeze(0))
        d.sum().backward()
        assert torch.isfinite(kappa.grad)


# --------------------------------------------------------------------
# Combiner behavior
# --------------------------------------------------------------------


class TestCombiner:
    def test_arithmetic_mean_default(self):
        """Default combiner is arithmetic mean. d(x, κ_x, y, κ_y) at
        opposite-sign κ's gives Euclidean-like distance (combined κ =
        0)."""
        het = HeterogeneousKappaManifold(n=3, dtype=torch.float64)
        x = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64)
        y = torch.tensor([[0.1, 0.0, 0.0]], dtype=torch.float64)
        # Opposite signs cancel under arithmetic mean
        kx = torch.tensor([-1.0], dtype=torch.float64)
        ky = torch.tensor([+1.0], dtype=torch.float64)
        d = het.distance(x, kx, y, ky)
        # κ_pair = 0 → Euclidean distance 2·‖y - x‖ (Bachmann normalization)
        expected = 2.0 * torch.linalg.vector_norm(y - x, dim=-1)
        torch.testing.assert_close(d, expected, atol=1e-9, rtol=1e-9)

    def test_custom_combiner(self):
        """User-supplied combiner overrides the built-in."""
        # Min of the two (an arbitrary asymmetric-in-spirit combiner)
        def min_combiner(a, b):
            return torch.minimum(a, b)
        het = HeterogeneousKappaManifold(
            n=3, combiner=min_combiner, dtype=torch.float64,
        )
        x = torch.tensor([[0.1, 0.0, 0.0]], dtype=torch.float64)
        y = torch.tensor([[0.0, 0.1, 0.0]], dtype=torch.float64)
        kx = torch.tensor([-1.0], dtype=torch.float64)
        ky = torch.tensor([-0.5], dtype=torch.float64)
        # min(-1, -0.5) = -1 (hyperbolic at κ=-1)
        # So the result should match the homogeneous distance at κ=-1
        homo = KappaStereographicManifold(
            n=3, kappa=-1.0, dtype=torch.float64,
        )
        d_het = het.distance(x, kx, y, ky)
        d_homo = homo.distance(x, y)
        torch.testing.assert_close(d_het, d_homo, atol=1e-10, rtol=1e-10)


# --------------------------------------------------------------------
# random_point + interop
# --------------------------------------------------------------------


class TestRandomPoint:
    def test_random_point_default_kappa(self):
        """When kappa is None, defaults to small standard-normal."""
        het = HeterogeneousKappaManifold(n=4, dtype=torch.float64)
        x = het.random_point(batch_size=5, generator=_seed(200))
        assert x.shape == (5, 4)
        assert torch.isfinite(x).all()

    def test_random_point_explicit_kappa(self):
        """kappa argument is honored when provided."""
        het = HeterogeneousKappaManifold(n=4, dtype=torch.float64)
        kappa = torch.tensor([-1.0, -0.5, +0.5, -1.0, +1.0],
                              dtype=torch.float64)
        x = het.random_point(batch_size=5, kappa=kappa, generator=_seed(201))
        assert x.shape == (5, 4)
        # All on manifold under their respective κ
        assert het.is_on_manifold(x, kappa).all()

    def test_random_point_rejects_mismatched_kappa(self):
        het = HeterogeneousKappaManifold(n=3, dtype=torch.float64)
        with pytest.raises(ValueError, match="shape"):
            het.random_point(
                batch_size=5, kappa=torch.tensor([1.0, 2.0]),
            )

    def test_random_point_rejects_negative_batch(self):
        het = HeterogeneousKappaManifold(n=3, dtype=torch.float64)
        with pytest.raises(ValueError, match="batch_size"):
            het.random_point(batch_size=-1)
