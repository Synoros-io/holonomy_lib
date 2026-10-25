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


def test_rejects_empty():
    mfd = LorentzManifold(n=3)
    P = mfd.random_point(batch_size=0, generator=_seed(1)).reshape(
        1, 0, mfd.n + 1,
    )
    with pytest.raises(ValueError, match="empty"):
        frechet_mean(P, mfd)
    valid = mfd.random_point(batch_size=2, generator=_seed(2)).reshape(1, 2, 4)
    for weights in ([[-1., 2.]], [[0., 0.]], [[float("nan"), 1.]],
                    [[float("inf"), 1.]]):
        with pytest.raises(ValueError, match="weights"):
            frechet_mean(valid, mfd, weights=torch.tensor(weights))


@pytest.mark.parametrize("batch,copies,seed", [
    (0, 1, 4), (1, 1, 4), (3, 1, 4),
    (0, 2, 5), (1, 2, 5), (2, 2, 5),
])
def test_single_point_is_itself(batch, copies, seed):
    """Singletons and repeated points have the known mean, including B=0."""
    mfd = LorentzManifold(n=3)
    point = mfd.random_point(batch_size=batch, generator=_seed(seed))
    points = point.unsqueeze(1).expand(batch, copies, 4)
    mean = frechet_mean(points, mfd, tol=1e-12)
    assert mean.shape == (batch, 4)
    torch.testing.assert_close(mean, point, atol=1e-10, rtol=0)


def test_mean_minimizes_sum_of_squared_distances():
    """Check a sampled perturbation and the symmetric triangle's exact mean."""
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
    assert torch.isfinite(mu).all()
    assert mfd.is_on_manifold(mu).all()

    # Symmetry forces the mean of this equilateral hyperbolic triangle
    # to the origin. Unit Karcher steps oscillate for this radius.
    plane = LorentzManifold(n=2)
    angles = torch.arange(3, dtype=torch.float64) * (2 * torch.pi / 3)
    triangle = plane.exp_0(4 * torch.stack((angles.cos(), angles.sin()), -1))
    mean = frechet_mean(triangle.unsqueeze(0), plane, max_iter=200)
    torch.testing.assert_close(mean, plane.origin(1), atol=1e-9, rtol=0)
    with pytest.warns(RuntimeWarning, match="did not converge"):
        frechet_mean(triangle.unsqueeze(0), plane, max_iter=1)

    # Preserve the manifold-domain cases, including the former convergence
    # regression's inputs; finiteness alone does not pin an old/new-base bug.
    from holonomy_lib.manifolds import KappaStereographicManifold

    for curvature, seed, count in [(-1., 200, 8), (-1., 90, 10),
                                    (-.5, 90, 10), (0., 90, 10),
                                    (.5, 90, 10), (1., 90, 10)]:
        curved = KappaStereographicManifold(n=3, kappa=curvature)
        points = curved.random_point(batch_size=count, generator=_seed(seed))
        mean = frechet_mean(points.unsqueeze(0), curved, max_iter=200, tol=1e-10)
        assert mean.shape == (1, curved.ambient_dim)
        assert torch.isfinite(mean).all()
        assert curved.is_on_manifold(mean).all()
    wide = LorentzManifold(n=4)
    points = wide.random_point(batch_size=20, generator=_seed(3)).unsqueeze(0)
    assert wide.is_on_manifold(frechet_mean(points, wide)).all()


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

def test_frechet_mean_backward_finite():
    """Implicit derivatives match reference values and are batch independent."""
    from holonomy_lib.manifolds import LorentzManifold

    mfd = LorentzManifold(n=4, k=-1.0)
    # 8 points parameterized by Euclidean tangent at origin
    v = (torch.randn(8, 4, dtype=torch.float64,
                     generator=_seed(20)) * 0.3)
    v.requires_grad_(True)
    points = mfd.exp_0(v).unsqueeze(0)            # (1, 8, 5)
    with pytest.warns(RuntimeWarning, match="did not converge"):
        mu = frechet_mean(points, mfd, max_iter=20, tol=1e-12)
    loss = mu.sum()
    loss.backward()
    assert v.grad is not None
    assert torch.isfinite(v.grad).all(), (
        f"v.grad NaN: {torch.isnan(v.grad).sum().item()}"
    )

    mfd = LorentzManifold(n=3, k=-1.0)
    # All 5 points = exp_0(v0); the mean is exactly that point.
    v0 = torch.tensor([0.2, 0.1, -0.3], dtype=torch.float64,
                      requires_grad=True)
    point = mfd.exp_0(v0.unsqueeze(0))             # (1, 4)
    P = point.expand(1, 5, 4)                       # (1, 5, 4)
    mu = frechet_mean(P, mfd, max_iter=5, tol=1e-12)
    mu.sum().backward()
    assert v0.grad is not None
    assert torch.isfinite(v0.grad).all(), (
        f"v0.grad NaN: {torch.isnan(v0.grad).sum().item()}"
    )

    # The flat mean's derivative averages both inputs even when the initial
    # point is already stationary; a zero accepted step would lose that.
    from holonomy_lib.manifolds import KappaStereographicManifold

    flat = KappaStereographicManifold(n=1, kappa=0.)
    empty_points = torch.empty(0, 2, 1, dtype=torch.float64, requires_grad=True)
    empty_weights = torch.empty(0, 2, dtype=torch.float64, requires_grad=True)
    empty_mean = frechet_mean(empty_points, flat, empty_weights)
    empty_grads = torch.autograd.grad(empty_mean.sum(), (empty_points, empty_weights))
    assert empty_grads[0].shape == empty_points.shape
    assert empty_grads[1].shape == empty_weights.shape
    coincident = torch.zeros(1, 2, 1, dtype=torch.float64, requires_grad=True)
    frechet_mean(coincident, flat).sum().backward()
    assert coincident.grad is not None
    torch.testing.assert_close(
        coincident.grad, torch.full_like(coincident, .5), atol=1e-12, rtol=0,
    )


    curved = KappaStereographicManifold(n=2, kappa=-1.)
    symmetric = torch.tensor(
        [[[0., 0.], [2., 0.], [-2., 0.], [0., 2.], [0., -2.]]],
        dtype=torch.float64, requires_grad=True,
    )

    def symmetric_mean(tangents):
        return frechet_mean(curved.exp_0(tangents), curved, tol=1e-12)

    mean = symmetric_mean(symmetric)
    gradient = torch.autograd.grad(mean[0, 0], symmetric)[0]
    # At the origin the radial/transverse Hessian eigenvalues of half
    # squared hyperbolic distance are 1 and r*coth(r). Here r=4 and
    # the five-point Hessian is (3 + 8*coth(4))/5 times the identity.
    radius = symmetric.new_tensor(4.)
    reference = 1 / (3 + 2 * radius / radius.tanh())
    torch.testing.assert_close(gradient[0, 0, 0], reference, atol=1e-12, rtol=0)
    perturbation = torch.zeros_like(symmetric)
    perturbation[0, 0, 0] = 1e-6
    finite_difference = (
        symmetric_mean(symmetric.detach() + perturbation)
        - symmetric_mean(symmetric.detach() - perturbation)
    ) / 2e-6
    torch.testing.assert_close(
        gradient[0, 0, 0], finite_difference[0, 0], atol=1e-8, rtol=0,
    )
    combined = symmetric.detach().repeat(2, 1, 1)
    combined[1, 0] = symmetric.new_tensor([.1, .05])
    combined.requires_grad_()
    combined_gradient = torch.autograd.grad(
        symmetric_mean(combined)[0, 0], combined,
    )[0]
    torch.testing.assert_close(combined_gradient[:1], gradient, atol=1e-12, rtol=0)
    torch.testing.assert_close(
        combined_gradient[1], torch.zeros_like(combined_gradient[1]),
        atol=0, rtol=0,
    )

    for model in (curved, LorentzManifold(n=2)):
        tangents = torch.tensor(
            [[[.1, .2], [-.2, .1], [.05, -.1]]],
            dtype=torch.float64, requires_grad=True,
        )
        weights = torch.tensor([[1., 2., 3.]], dtype=torch.float64,
                               requires_grad=True)

        def weighted_mean(tangents, weights, model=model):
            return frechet_mean(
                model.exp_0(tangents), model, weights, max_iter=200, tol=1e-12,
            )

        assert torch.autograd.gradcheck(weighted_mean, (tangents, weights))

    # Antipodal sphere points have a circle of means. A singular member
    # must report NaNs without corrupting its nonsingular batch neighbor.
    sphere = KappaStereographicManifold(n=2, kappa=1.)
    points = torch.tensor(
        [[[0., 0.], [1., 0.], [-1., 0.]],
         [[0., 0.], [.1, 0.], [-.1, 0.]]],
        dtype=torch.float64, requires_grad=True,
    )
    weights = torch.tensor([[0., .5, .5], [0., .5, .5]],
                           dtype=torch.float64, requires_grad=True)
    mean = frechet_mean(points, sphere, weights)
    point_grad, weight_grad = torch.autograd.grad(mean.sum(), (points, weights))
    assert torch.isnan(point_grad[0]).all() and torch.isnan(weight_grad[0]).all()
    assert torch.isfinite(point_grad[1]).all() and torch.isfinite(weight_grad[1]).all()
