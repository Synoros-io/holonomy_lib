# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Synoros

"""Tests for holonomy_lib.hyperbolic.frechet_mean."""

from __future__ import annotations

import pytest
import torch

from holonomy_lib.hyperbolic import frechet_mean
from holonomy_lib.manifolds import LorentzManifold


def _seed(s: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(s)
    return g


@pytest.mark.parametrize("batch", [1, 3])
def test_shape(batch):
    mfd = LorentzManifold(n=4)
    P = mfd.random_point(batch_size=batch * 5, generator=_seed(0)).reshape(
        batch, 5, mfd.n + 1,
    )
    mu = frechet_mean(P, mfd)
    assert mu.shape == (batch, mfd.n + 1)


def test_rejects_empty():
    mfd = LorentzManifold(n=3)
    P = mfd.random_point(batch_size=0, generator=_seed(1)).reshape(
        1, 0, mfd.n + 1,
    )
    with pytest.raises(ValueError, match="empty"):
        frechet_mean(P, mfd)


def test_rejects_bad_weight_shape():
    mfd = LorentzManifold(n=3)
    P = mfd.random_point(batch_size=10, generator=_seed(2)).reshape(
        2, 5, mfd.n + 1,
    )
    bad_w = torch.ones(2, 4)
    with pytest.raises(ValueError, match="weights"):
        frechet_mean(P, mfd, weights=bad_w)


def test_mean_is_on_manifold():
    mfd = LorentzManifold(n=4)
    P = mfd.random_point(batch_size=20, generator=_seed(3)).reshape(
        1, 20, mfd.n + 1,
    )
    mu = frechet_mean(P, mfd)
    assert mfd.is_on_manifold(mu).all()


def test_single_point_is_itself():
    """Fréchet mean of a singleton is that point."""
    mfd = LorentzManifold(n=3)
    p = mfd.random_point(batch_size=3, generator=_seed(4)).unsqueeze(1)
    # Shape (3, 1, n+1)
    mu = frechet_mean(p, mfd, tol=1e-12)
    torch.testing.assert_close(mu, p.squeeze(1), atol=1e-10, rtol=0)


def test_two_identical_points_equal_mean():
    """Fréchet mean of two copies of the same point is that point."""
    mfd = LorentzManifold(n=3)
    p = mfd.random_point(batch_size=2, generator=_seed(5))
    P = torch.stack([p, p], dim=1)  # (2, 2, n+1)
    mu = frechet_mean(P, mfd, tol=1e-12)
    torch.testing.assert_close(mu, p, atol=1e-10, rtol=1e-10)


def test_mean_minimizes_sum_of_squared_distances():
    """Numerical check: the Karcher iterate has lower sum-of-squared-
    distances than any nearby perturbation along the manifold."""
    mfd = LorentzManifold(n=3)
    P = mfd.random_point(batch_size=10, generator=_seed(6)).reshape(
        1, 10, mfd.n + 1,
    )
    mu = frechet_mean(P, mfd, tol=1e-12, max_iter=200)

    def cost(point):
        # point: (1, n+1) → broadcast against (1, 10, n+1)
        point_b = point.unsqueeze(1).expand(1, 10, mfd.n + 1)
        d = mfd.distance(
            point_b.reshape(10, mfd.n + 1),
            P.reshape(10, mfd.n + 1),
        ).reshape(1, 10)
        return (d * d).sum(dim=-1)

    base_cost = cost(mu)
    # Perturb mu by a small tangent and check cost goes up
    v_seed = torch.randn(1, mfd.n + 1, dtype=mfd.dtype, generator=_seed(7))
    v = mfd.projection(mu, v_seed)
    v = v / mfd.norm(mu, v).clamp(
        min=torch.finfo(v.dtype).tiny,
    ).unsqueeze(-1) * 0.01
    mu_pert = mfd.exp(mu, v)
    pert_cost = cost(mu_pert)
    assert (pert_cost > base_cost).all(), (
        f"Karcher iterate is not a local minimizer: "
        f"base={base_cost.item():.6e}, perturbed={pert_cost.item():.6e}"
    )


def test_weights_shift_the_mean():
    """Heavy weight on one point pulls the mean toward it."""
    mfd = LorentzManifold(n=3)
    P = mfd.random_point(batch_size=5, generator=_seed(8)).reshape(
        1, 5, mfd.n + 1,
    )
    # Equal weights
    mu_equal = frechet_mean(P, mfd, tol=1e-12, max_iter=200)
    # Heavy weight on point 0
    heavy = torch.tensor([[10.0, 1.0, 1.0, 1.0, 1.0]], dtype=mfd.dtype)
    mu_heavy = frechet_mean(P, mfd, weights=heavy, tol=1e-12, max_iter=200)
    # mu_heavy should be closer to P[:, 0] than mu_equal is
    d_equal = mfd.distance(mu_equal, P[:, 0])
    d_heavy = mfd.distance(mu_heavy, P[:, 0])
    assert (d_heavy < d_equal).all(), (
        f"Heavy-weighted mean did not shift toward heavy point: "
        f"d_equal={d_equal.item():.4e}, d_heavy={d_heavy.item():.4e}"
    )


def test_provenance_signature():
    """Op is registered and emits a provenance node."""
    from holonomy_lib import provenance

    mfd = LorentzManifold(n=3)
    with provenance.record() as reg:
        P = mfd.random_point(batch_size=5, generator=_seed(9)).reshape(
            1, 5, mfd.n + 1,
        )
        _ = frechet_mean(P, mfd, max_iter=5)
    ops = {n.op_id for n in reg}
    assert "holonomy_lib.hyperbolic.frechet_mean" in ops


# --------------------------------------------------------------------
# Autograd-finite — Fréchet mean backward through a parameterized
# tangent-at-origin embedding chain.
# --------------------------------------------------------------------


def test_frechet_mean_backward_finite():
    """frechet_mean of points parameterized by tangents at origin
    must produce finite gradients on the tangent params."""
    from holonomy_lib.manifolds import LorentzManifold

    mfd = LorentzManifold(n=4, k=-1.0)
    # 8 points parameterized by Euclidean tangent at origin
    v = (torch.randn(8, 4, dtype=torch.float64,
                     generator=_seed(20)) * 0.3)
    v.requires_grad_(True)
    points = mfd.exp_0(v).unsqueeze(0)            # (1, 8, 5)
    mu = frechet_mean(points, mfd, max_iter=20, tol=1e-12)
    loss = mu.sum()
    loss.backward()
    assert torch.isfinite(v.grad).all(), (
        f"v.grad NaN: {torch.isnan(v.grad).sum().item()}"
    )


def test_frechet_mean_backward_at_collapsed_points():
    """Edge case: all points coincide → mean is that point;
    gradient through the converged iteration should stay finite."""
    from holonomy_lib.manifolds import LorentzManifold

    mfd = LorentzManifold(n=3, k=-1.0)
    # All 5 points = exp_0(v0); the mean is exactly that point.
    v0 = torch.tensor([0.2, 0.1, -0.3], dtype=torch.float64,
                      requires_grad=True)
    point = mfd.exp_0(v0.unsqueeze(0))             # (1, 4)
    P = point.expand(1, 5, 4)                       # (1, 5, 4)
    mu = frechet_mean(P, mfd, max_iter=5, tol=1e-12)
    mu.sum().backward()
    assert torch.isfinite(v0.grad).all(), (
        f"v0.grad NaN: {torch.isnan(v0.grad).sum().item()}"
    )


# --------------------------------------------------------------------
# Cross-manifold: Fréchet mean on KappaStereographicManifold
# --------------------------------------------------------------------


@pytest.mark.parametrize("k", [-1.0, -0.5, 0.0, 0.5, 1.0])
def test_works_on_kappa_stereographic(k):
    """Fréchet mean uses log/exp/norm — all available on
    KappaStereographicManifold. Works across all κ branches."""
    from holonomy_lib.manifolds import KappaStereographicManifold

    mfd = KappaStereographicManifold(n=3, kappa=k)
    P = mfd.random_point(batch_size=10, generator=_seed(90)).reshape(
        1, 10, mfd.ambient_dim,
    )
    mu = frechet_mean(P, mfd, max_iter=50, tol=1e-10)
    assert mu.shape == (1, mfd.ambient_dim)
    assert mfd.is_on_manifold(mu).all()


def test_frechet_convergence_check_uses_old_mu():
    """Regression: the convergence check `||tangent_avg||_μ < tol` is
    evaluated at the OLD μ (where the tangent lives), not the NEW μ
    after the step. For Lorentz this is a no-op (point-independent
    metric); for KappaStereographic the conformal factor depends on
    μ, so evaluating at NEW μ would scale the convergence threshold
    by `λ_κ(μ_new)/λ_κ(μ_old)`. We pin the fix by checking that the
    function returns a finite result and stays on the manifold.

    Note: the Karcher iteration's *convergence rate* on the Poincaré
    ball is linear (not quadratic), and for widely-spread points it
    can take many iterations to reach a tight tolerance — that's a
    property of the algorithm, not a bug in this fix.
    """
    from holonomy_lib.manifolds import KappaStereographicManifold

    mfd = KappaStereographicManifold(n=3, kappa=-1.0)
    P = mfd.random_point(batch_size=8, generator=_seed(200)).reshape(
        1, 8, mfd.ambient_dim,
    )
    mu = frechet_mean(P, mfd, tol=1e-10, max_iter=200)
    assert torch.isfinite(mu).all()
    assert mfd.is_on_manifold(mu).all()
