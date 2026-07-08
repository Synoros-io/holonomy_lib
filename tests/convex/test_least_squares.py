# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Synoros

"""Tests for holonomy_lib.convex.regularized_lstsq.

Three layers:
  1. Unit — shapes across B ∈ {0, 1, >1}; validation rejects.
  2. Property — λ=0 reduces to ordinary lstsq; the three methods agree;
     monotone shrinkage ‖x(λ₁)‖ ≥ ‖x(λ₂)‖ for λ₁ < λ₂; the closed-form
     normal-equations solution is reproduced.
  3. Comparison — scipy lsqr with damp=√λ on the sparse path.
"""

from __future__ import annotations

import math

import pytest
import torch

from holonomy_lib.convex import regularized_lstsq

DT = torch.float64


def _problem(B, m, n, seed=0):
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(B, m, n, generator=g, dtype=DT)
    b = torch.randn(B, m, generator=g, dtype=DT)
    return A, b


def _closed_form(A, b, lam):
    n = A.shape[-1]
    S = A.mT @ A + lam * torch.eye(n, dtype=A.dtype)
    return torch.linalg.solve(S, (A.mT @ b.unsqueeze(-1))).squeeze(-1)


@pytest.mark.parametrize("method", ["cholesky", "cg", "lsqr", "auto"])
@pytest.mark.parametrize("B", [0, 1, 3])
def test_methods_match_closed_form(method, B):
    m, n, lam = 30, 10, 0.3
    A, b = _problem(max(B, 1), m, n)
    A, b = A[:B], b[:B]
    x = regularized_lstsq(A, b, lam, method=method)
    assert x.shape == (B, n)
    if B:
        ref = _closed_form(A, b, lam)
        torch.testing.assert_close(x, ref, atol=1e-7, rtol=0)


def test_lam_zero_reduces_to_lstsq():
    A, b = _problem(1, 40, 12, seed=1)
    A, b = A.squeeze(0), b.squeeze(0)
    x = regularized_lstsq(A, b, 0.0, method="lsqr")
    ref = torch.linalg.lstsq(A, b.unsqueeze(-1)).solution.squeeze(-1)
    torch.testing.assert_close(x, ref, atol=1e-7, rtol=0)


def test_monotone_shrinkage():
    A, b = _problem(1, 30, 10, seed=2)
    A, b = A.squeeze(0), b.squeeze(0)
    norms = [
        regularized_lstsq(A, b, lam).norm()
        for lam in (0.0, 0.1, 1.0, 10.0)
    ]
    for a, c in zip(norms, norms[1:]):
        assert a >= c - 1e-12


def test_sparse_auto_dispatches_to_lsqr():
    # Sparse incidence + ridge — the hierarchy-potential pattern.
    scipy_spla = pytest.importorskip("scipy.sparse.linalg")
    import scipy.sparse as sp

    n, lam = 20, 0.5
    rows, cols, vals = [], [], []
    for e in range(n - 1):
        rows += [e, e]
        cols += [e, e + 1]
        vals += [-1.0, 1.0]
    A = torch.sparse_coo_tensor(
        torch.tensor([rows, cols]), torch.tensor(vals, dtype=DT),
        (n - 1, n),
    ).coalesce()
    g = torch.Generator().manual_seed(3)
    b = torch.randn(n - 1, generator=g, dtype=DT)
    x = regularized_lstsq(A, b, lam)  # auto → lsqr
    A_sp = sp.coo_matrix(
        (A.values().numpy(), A.indices().numpy()), shape=(n - 1, n),
    ).tocsr()
    x_sp = scipy_spla.lsqr(
        A_sp, b.numpy(), damp=math.sqrt(lam), atol=1e-12, btol=1e-12,
    )[0]
    torch.testing.assert_close(x, torch.from_numpy(x_sp), atol=1e-6, rtol=0)


def test_sparse_cg_agrees_with_lsqr():
    n, lam = 20, 0.5
    rows, cols, vals = [], [], []
    for e in range(n - 1):
        rows += [e, e]
        cols += [e, e + 1]
        vals += [-1.0, 1.0]
    A = torch.sparse_coo_tensor(
        torch.tensor([rows, cols]), torch.tensor(vals, dtype=DT),
        (n - 1, n),
    ).coalesce()
    g = torch.Generator().manual_seed(4)
    b = torch.randn(n - 1, generator=g, dtype=DT)
    x_cg = regularized_lstsq(A, b, lam, method="cg")
    x_ls = regularized_lstsq(A, b, lam, method="lsqr")
    torch.testing.assert_close(x_cg, x_ls, atol=1e-6, rtol=0)


def test_validation():
    A, b = _problem(1, 10, 4, seed=5)
    A, b = A.squeeze(0), b.squeeze(0)
    with pytest.raises(ValueError, match="method"):
        regularized_lstsq(A, b, 0.1, method="qr")
    with pytest.raises(ValueError, match="lam"):
        regularized_lstsq(A, b, -0.1)
    with pytest.raises(ValueError, match="cholesky"):
        regularized_lstsq(A.to_sparse_coo(), b, 0.1, method="cholesky")
    with pytest.raises(ValueError, match="torch.Tensor"):
        regularized_lstsq([[1.0]], b, 0.1)
