# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Synoros

"""Tests for holonomy_lib.hyperbolic.manifold_aware_inner."""

from __future__ import annotations

import pytest
import torch

from holonomy_lib.hyperbolic import manifold_aware_inner
from holonomy_lib.manifolds import LorentzManifold


def _seed(s: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(s)
    return g


@pytest.mark.parametrize("batch", [1, 4])
def test_shape(batch):
    mfd = LorentzManifold(n=3)
    x = mfd.random_point(batch_size=batch, generator=_seed(0))
    y = mfd.random_point(batch_size=batch, generator=_seed(1))
    out = manifold_aware_inner(x, y, mfd)
    assert out.shape == (batch,)


def test_symmetric():
    """⟨x, y⟩ = ⟨y, x⟩."""
    mfd = LorentzManifold(n=3)
    x = mfd.random_point(batch_size=4, generator=_seed(2))
    y = mfd.random_point(batch_size=4, generator=_seed(3))
    torch.testing.assert_close(
        manifold_aware_inner(x, y, mfd),
        manifold_aware_inner(y, x, mfd),
        atol=1e-12, rtol=0,
    )


def test_self_inner_is_distance_to_origin_squared():
    """⟨x, x⟩ = ‖log_o(x)‖² = d(o, x)² ≥ 0 with equality iff x = o."""
    mfd = LorentzManifold(n=3)
    x = mfd.random_point(batch_size=4, generator=_seed(4))
    self_ip = manifold_aware_inner(x, x, mfd)
    # All positive (and equal to d(o, x)²)
    assert (self_ip >= 0).all()
    # Check against direct distance computation
    origin = mfd.origin(batch_size=4)
    d = mfd.distance(origin, x)
    torch.testing.assert_close(
        self_ip, d * d, atol=1e-10, rtol=1e-10,
    )


def test_origin_inner_is_zero():
    """⟨o, x⟩ = ⟨log_o(o), log_o(x)⟩ = ⟨0, log_o(x)⟩ = 0."""
    mfd = LorentzManifold(n=3)
    x = mfd.random_point(batch_size=4, generator=_seed(5))
    o = mfd.origin(batch_size=4)
    out = manifold_aware_inner(o, x, mfd)
    torch.testing.assert_close(
        out, torch.zeros_like(out), atol=1e-12, rtol=0,
    )


def test_matches_euclidean_for_origin_neighborhood():
    """For x = exp_0(v) with small v, log_0(x) ≈ v, so the manifold-
    aware inner matches the Euclidean inner of the tangent v."""
    mfd = LorentzManifold(n=3)
    v_x = torch.randn(3, mfd.n, dtype=mfd.dtype, generator=_seed(6)) * 0.05
    v_y = torch.randn(3, mfd.n, dtype=mfd.dtype, generator=_seed(7)) * 0.05
    x = mfd.exp_0(v_x)
    y = mfd.exp_0(v_y)
    out = manifold_aware_inner(x, y, mfd)
    expected = (v_x * v_y).sum(dim=-1)
    torch.testing.assert_close(out, expected, atol=1e-6, rtol=1e-6)


# --------------------------------------------------------------------
# Autograd-finite
# --------------------------------------------------------------------


def test_manifold_aware_inner_backward_at_origin():
    """⟨origin, x⟩ backward must be finite — log_0(origin) is the
    boundary case where the old vector_norm path would NaN."""
    mfd = LorentzManifold(n=4)
    v = (torch.randn(3, 4, generator=_seed(20)) * 0.3)
    v.requires_grad_(True)
    x = mfd.exp_0(v)
    origin = mfd.origin(batch_size=3)
    out = manifold_aware_inner(origin, x, mfd)
    out.sum().backward()
    assert torch.isfinite(v.grad).all()


def test_manifold_aware_inner_backward_self():
    """⟨x, x⟩ backward finite — would NaN through log_0 if not safe."""
    mfd = LorentzManifold(n=4)
    v = torch.zeros(3, 4, requires_grad=True)
    x = mfd.exp_0(v)
    out = manifold_aware_inner(x, x, mfd)
    out.sum().backward()
    assert torch.isfinite(v.grad).all()


# --------------------------------------------------------------------
# Cross-manifold: manifold_aware_inner on KappaStereographicManifold
# --------------------------------------------------------------------


@pytest.mark.parametrize("k", [-1.0, -0.5, 0.0, 0.5, 1.0])
def test_works_on_kappa_stereographic(k):
    """manifold_aware_inner only depends on the manifold's `log_0`,
    so it works directly on KappaStereographicManifold without
    modification."""
    from holonomy_lib.manifolds import KappaStereographicManifold

    mfd = KappaStereographicManifold(n=3, kappa=k)
    x = mfd.random_point(batch_size=4, generator=_seed(80 + int(k * 10)))
    y = mfd.random_point(batch_size=4, generator=_seed(81 + int(k * 10)))
    out = manifold_aware_inner(x, y, mfd)
    assert out.shape == (4,)
    # Self-inner equals d(o, x)² on all branches
    self_ip = manifold_aware_inner(x, x, mfd)
    d = mfd.distance(mfd.origin(batch_size=4), x)
    torch.testing.assert_close(self_ip, d * d, atol=1e-9, rtol=1e-9)
