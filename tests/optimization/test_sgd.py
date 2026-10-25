# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Synoros

"""Tests for holonomy_lib.optimization.RiemannianSGD.

Four layers:
  1. Validation (lr > 0, manifold API contract).
  2. Convergence to the true minimum on closed-form problems.
  3. Manifold-constraint preservation across iterates.
  4. Comparison against pymanopt's SteepestDescent (importorskip).
"""

from __future__ import annotations

import pytest
import torch

from holonomy_lib.manifolds import FixedRankManifold, SPDManifold
from holonomy_lib.optimization import (
    RiemannianSGD,
)


def _seeded(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


# --------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------


class TestValidation:
    def test_rejects_nonpositive_lr(self):
        mfd = SPDManifold(n=3)
        with pytest.raises(ValueError, match="lr"):
            RiemannianSGD(mfd, lr=0.0)
        with pytest.raises(ValueError, match="lr"):
            RiemannianSGD(mfd, lr=-0.1)


class TestBatchZero:
    """`B = 0` must work; required by CONVENTIONS.md §1.1 for every
    primitive that accepts a batched input."""

    def test_spd_batch_zero(self):
        mfd = SPDManifold(n=3, dtype=torch.float64)
        S = mfd.random_point(batch_size=0, generator=_seeded(80))
        assert S.shape == (0, 3, 3)
        g = torch.zeros(0, 3, 3, dtype=torch.float64)
        opt = RiemannianSGD(mfd, lr=0.1)
        S_next = opt.step(S, g)
        assert S_next.shape == (0, 3, 3)

        mfd = FixedRankManifold(m=4, n=5, r=2, dtype=torch.float64)
        pt = mfd.random_point(batch_size=0, generator=_seeded(81))
        assert pt[0].shape == (0, 4, 2)
        g = torch.zeros(0, 4, 5, dtype=torch.float64)
        opt = RiemannianSGD(mfd, lr=0.1)
        pt_next = opt.step(pt, g)
        assert pt_next[0].shape == (0, 4, 2)
        assert pt_next[1].shape == (0, 2)
        assert pt_next[2].shape == (0, 2, 5)


# --------------------------------------------------------------------
# Convergence: rank-r best-approximation on FixedRankManifold
# --------------------------------------------------------------------


class TestFixedRankConvergence:
    """Optimize ‖M_target − dense(point)‖² over `FixedRankManifold(m, n, r)`.
    The Euclidean argmin is SVD-truncation of `M_target` to rank `r`,
    so the optimizer should converge to that fixed point."""

    def test_converges_to_truncated_svd(self):
        torch.manual_seed(0)
        m, n, r = 10, 12, 3
        # Target: a rank-(2r) matrix; the best rank-r approximation is
        # the top-r SVD of M_target.
        U_t = torch.randn(1, m, 2 * r, generator=_seeded(1), dtype=torch.float64)
        V_t = torch.randn(1, n, 2 * r, generator=_seeded(2), dtype=torch.float64)
        S_t = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0, 0.5], dtype=torch.float64)
        # Build M_target = U_t diag(S_t) V_t^T (low-rank, well-conditioned)
        M_target = U_t @ torch.diag(S_t) @ V_t.mT  # (1, m, n)

        # Ground truth: best rank-r approximation via direct SVD
        U_full, S_full, Vt_full = torch.linalg.svd(M_target, full_matrices=False)
        best_recon = (
            U_full[..., :r] @ torch.diag_embed(S_full[..., :r]) @ Vt_full[..., :r, :]
        )

        mfd = FixedRankManifold(m=m, n=n, r=r, dtype=torch.float64)
        point = mfd.random_point(batch_size=1, generator=_seeded(3))
        opt = RiemannianSGD(mfd, lr=0.05)

        for _ in range(300):
            current = mfd.dense(point)
            # Gradient of ½‖dense(point) − M_target‖² w.r.t. dense form
            ambient_grad = current - M_target
            point = opt.step(point, ambient_grad)

        final = mfd.dense(point)
        residual = (final - best_recon).norm() / best_recon.norm()
        assert residual < 1e-3, (
            f"converged residual {residual.item():.3e} too large; "
            f"expected sub-1e-3"
        )


# --------------------------------------------------------------------
# Convergence: geodesic gradient descent on SPDManifold
# --------------------------------------------------------------------


class TestSPDConvergence:
    """Optimize ½ · d(S, T)² over `SPDManifold(n)` for a fixed target T.
    The Riemannian gradient of this objective is `−log_S(T)`; one step
    of unit-rate Riemannian SGD lands exactly at T (the geodesic from
    S in the direction log_S(T) reaches T at time 1). With lr=1 we
    should converge in one step modulo numerical roundoff."""

    def test_one_step_lr_one_lands_on_target(self):
        mfd = SPDManifold(n=4, dtype=torch.float64)
        S = mfd.random_point(batch_size=1, generator=_seeded(20))
        T = mfd.random_point(batch_size=1, generator=_seeded(21))
        # Convert -log_S(T), the Riemannian gradient, to the ambient
        # gradient S^{-1} (-log_S(T)) S^{-1} required by the optimizer.
        opt = RiemannianSGD(mfd, lr=1.0)
        factor = torch.linalg.cholesky(S)
        left = torch.cholesky_solve(-mfd.log(S, T), factor)
        ambient_grad = torch.cholesky_solve(left.mT, factor).mT
        S_next = opt.step(S, ambient_grad)
        # After metric conversion and exp_S(log_S(T))
        # = exp_S(log_S(T)) = T.
        torch.testing.assert_close(S_next, T, atol=1e-8, rtol=0)

        mfd = SPDManifold(n=4, dtype=torch.float64)
        S = mfd.random_point(batch_size=2, generator=_seeded(30))
        T = mfd.random_point(batch_size=2, generator=_seeded(31))
        opt = RiemannianSGD(mfd, lr=0.3)
        for _ in range(20):
            factor = torch.linalg.cholesky(S)
            left = torch.cholesky_solve(-mfd.log(S, T), factor)
            grad = torch.cholesky_solve(left.mT, factor).mT
            S = opt.step(S, grad)
            assert mfd.is_spd(S).all()

        mfd = SPDManifold(n=4, dtype=torch.float64)
        S = mfd.random_point(batch_size=1, generator=_seeded(40))
        T = mfd.random_point(batch_size=1, generator=_seeded(41))
        opt = RiemannianSGD(mfd, lr=0.5)
        for _ in range(50):
            factor = torch.linalg.cholesky(S)
            left = torch.cholesky_solve(-mfd.log(S, T), factor)
            grad = torch.cholesky_solve(left.mT, factor).mT
            S = opt.step(S, grad)
        # After enough steps, S should be very close to T
        dist = mfd.distance(S, T)
        assert dist[0].item() < 1e-9, (
            f"failed to converge to target; final distance {dist[0].item():.3e}"
        )

        identity = torch.eye(2, dtype=torch.float64)[None]
        mfd = SPDManifold(2)
        actual = RiemannianSGD(mfd, lr=0.1).step(2 * identity, identity)
        expected = 2 * torch.exp(torch.tensor(-0.2, dtype=identity.dtype)) * identity
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=0)


# --------------------------------------------------------------------
# Comparison: pymanopt SteepestDescent (importorskip)
# --------------------------------------------------------------------


try:
    import pymanopt  # noqa: F401
    _HAVE_PYMANOPT = True
except ImportError:
    _HAVE_PYMANOPT = False


@pytest.mark.skipif(not _HAVE_PYMANOPT, reason="pymanopt not installed")
class TestAgainstPymanopt:
    """Both should converge to the same SPD geodesic minimum from the
    same starting point. We use the closed-form problem `min ½ d(S, T)²`
    where both implementations should reach T in O(log) steps."""

    def test_spd_geodesic_descent_matches(self):
        import numpy as np
        import pymanopt
        from pymanopt.manifolds import SymmetricPositiveDefinite
        from pymanopt.optimizers import SteepestDescent

        n = 4
        torch.manual_seed(70)

        # Build problem in pymanopt
        manifold = SymmetricPositiveDefinite(n)
        S0 = manifold.random_point()
        T = manifold.random_point()

        @pymanopt.function.numpy(manifold)
        def cost(S):
            # ½ d(S, T)² via the affine-invariant metric
            return 0.5 * (manifold.dist(S, T)) ** 2

        @pymanopt.function.numpy(manifold)
        def riemannian_gradient(S):
            return -manifold.log(S, T)

        problem = pymanopt.Problem(
            manifold, cost, riemannian_gradient=riemannian_gradient,
        )
        solver = SteepestDescent(max_iterations=100, verbosity=0)
        S_pymanopt = solver.run(problem, initial_point=S0).point

        # Same problem in holonomy_lib
        S0_torch = torch.tensor(
            S0, dtype=torch.float64,
        ).unsqueeze(0)
        T_torch = torch.tensor(
            T, dtype=torch.float64,
        ).unsqueeze(0)
        mfd = SPDManifold(n=n, dtype=torch.float64)
        opt = RiemannianSGD(mfd, lr=0.5)
        S = S0_torch
        for _ in range(100):
            factor = torch.linalg.cholesky(S)
            left = torch.cholesky_solve(-mfd.log(S, T_torch), factor)
            grad = torch.cholesky_solve(left.mT, factor).mT
            S = opt.step(S, grad)
        S_holonomy = S[0].numpy()

        # Both should be at (or very close to) T
        diff_pymanopt = np.linalg.norm(S_pymanopt - T)
        diff_holonomy = np.linalg.norm(S_holonomy - T)
        assert diff_pymanopt < 1e-6
        assert diff_holonomy < 1e-6
        # And they should be close to each other
        diff_between = np.linalg.norm(S_pymanopt - S_holonomy)
        assert diff_between < 1e-4
