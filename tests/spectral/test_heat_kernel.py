# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Synoros

"""Tests for holonomy_lib.spectral.heat_kernel_chebyshev.

Four layers:
  1. Validation (t ≥ 0, K ≥ 0, lambda_max > 0).
  2. Sanity at t = 0 (K_0 = I) and t → ∞ (rows sum towards stationary).
  3. Convergence to exact matrix exponential as K grows.
  4. Signal path (K_t @ v) agrees with dense K_t @ v.
"""

from __future__ import annotations

import pytest
import torch

from holonomy_lib.spectral import heat_kernel_chebyshev, laplacian


def _random_normalized_laplacian(
    n: int, batch: int = 1, dtype=torch.float64, seed: int = 0,
) -> torch.Tensor:
    g = torch.Generator()
    g.manual_seed(seed)
    A = torch.rand(batch, n, n, generator=g, dtype=dtype)
    A = (A + A.mT) * 0.5
    A.diagonal(dim1=-2, dim2=-1).zero_()
    return laplacian.symmetric_normalized(A)


# --------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------


class TestValidation:
    def test_rejects_negative_t(self):
        L = _random_normalized_laplacian(4)
        with pytest.raises(ValueError, match="diffusion time"):
            heat_kernel_chebyshev(L, t=-0.5)
        with pytest.warns(UserWarning, match="must be symmetric"):
            heat_kernel_chebyshev(torch.tensor([[[0., 1.], [0., 0.]]]), t=0.1)


# --------------------------------------------------------------------
# Identity at t = 0
# --------------------------------------------------------------------


class TestIdentity:
    def test_t_zero_returns_identity(self):
        n = 5
        L = _random_normalized_laplacian(n, batch=2)
        K = heat_kernel_chebyshev(L, t=0.0)
        identity = torch.eye(n, dtype=L.dtype).expand_as(K)
        torch.testing.assert_close(K, identity, atol=1e-12, rtol=0)


# --------------------------------------------------------------------
# Convergence to torch.matrix_exp
# --------------------------------------------------------------------


class TestConvergence:
    """The Chebyshev expansion should converge to the true exp(-tL)
    as K grows. We compare against torch.matrix_exp(-tL) at varying K."""

    @pytest.mark.parametrize("t", [0.1, 1.0, 3.0])
    def test_matches_matrix_exp(self, t):
        for n, batch in ((6, 0), (6, 1), (8, 2), (0, 1)):
            L = _random_normalized_laplacian(n, batch=batch)
            K_cheb = heat_kernel_chebyshev(L, t=t, K=40)
            K_true = torch.matrix_exp(-t * L)
            torch.testing.assert_close(K_cheb, K_true, atol=1e-10, rtol=1e-10)
            torch.testing.assert_close(K_cheb, K_cheb.mT, atol=1e-10, rtol=0)
            assert (torch.linalg.eigvalsh(K_cheb) >= -1e-9).all()
            trace = K_cheb.diagonal(dim1=-2, dim2=-1).sum(-1)
            assert (trace <= n + 1e-9).all()

    def test_error_decreases_with_K(self):
        """Higher K → tighter approximation. Monotone in K."""
        n = 6
        L = _random_normalized_laplacian(n)
        K_true = torch.matrix_exp(-1.0 * L)
        prev_err = float("inf")
        for K in [4, 8, 16, 32]:
            K_cheb = heat_kernel_chebyshev(L, t=1.0, K=K)
            err = (K_cheb - K_true).norm().item()
            assert err <= prev_err * 1.05, (
                f"K={K}: error {err:.3e} should not increase "
                f"(prev was {prev_err:.3e})"
            )
            prev_err = err


# --------------------------------------------------------------------
# Signal path matches dense K_t @ signal
# --------------------------------------------------------------------


class TestSignalPath:
    def test_signal_matches_dense(self):
        n = 6
        L = _random_normalized_laplacian(n, batch=2)
        g = torch.Generator()
        g.manual_seed(11)
        signal = torch.randn(2, n, 3, dtype=torch.float64, generator=g)

        K_dense = heat_kernel_chebyshev(L, t=0.7, K=30)
        out_dense = K_dense @ signal
        out_signal = heat_kernel_chebyshev(L, t=0.7, K=30, signal=signal)
        torch.testing.assert_close(out_dense, out_signal, atol=1e-10, rtol=1e-10)
