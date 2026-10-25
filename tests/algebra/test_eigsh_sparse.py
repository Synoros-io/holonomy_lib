# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Synoros

"""Tests for holonomy_lib.algebra.eigsh_sparse (sparse smallest-k eigenpairs).

Three layers:
  1. Unit — validation (dense input rejected, k range, bad solver).
  2. Property — eigenpair residual `A v ≈ λ v`; ascending eigenvalues;
     orthonormal eigenvectors.
  3. Comparison — every backend against dense `torch.linalg.eigvalsh`
     on the same operator, and the two SciPy branches against each
     other (the substrate-era "same modes" claim, re-verified here).
"""

from __future__ import annotations

import pytest
import torch

from holonomy_lib.algebra import eigsh_sparse_sa, lanczos_eigsh

DT = torch.float64


def _sparse_psd(n: int, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Sparse PSD test operator with a 1-dim kernel: the combinatorial
    Laplacian of a random connected graph (cycle + chords). Returns
    (sparse COO, dense)."""
    g = torch.Generator().manual_seed(seed)
    edges = [(i, (i + 1) % n) for i in range(n)]  # cycle: connected
    extra = torch.randint(0, n, (n // 2, 2), generator=g)
    for u, v in extra.tolist():
        if u != v and (u, v) not in edges and (v, u) not in edges:
            edges.append((u, v))
    A = torch.zeros(n, n, dtype=DT)
    for u, v in edges:
        w = float(torch.rand(1, generator=g)) + 0.5
        A[u, v] += w
        A[v, u] += w
    L = torch.diag(A.sum(dim=1)) - A
    return L.to_sparse_coo().coalesce(), L


class TestValidation:
    def test_rejects_dense_input(self):
        with pytest.raises(ValueError, match="sparse"):
            eigsh_sparse_sa(torch.eye(8, dtype=DT), k=2)

    def test_rejects_bad_k(self):
        A, _ = _sparse_psd(10)
        with pytest.raises(ValueError, match="k must satisfy"):
            eigsh_sparse_sa(A, k=0)
        with pytest.raises(ValueError, match="k must satisfy"):
            eigsh_sparse_sa(A, k=10)  # k < n strict

    def test_rejects_bad_solver(self):
        A, _ = _sparse_psd(10)
        with pytest.raises(ValueError, match="solver"):
            eigsh_sparse_sa(A, k=2, solver="qr")


@pytest.mark.parametrize("solver", ["eigsh", "lobpcg", "torch_lobpcg", "auto"])
def test_bottom_k_matches_dense_eigvalsh(solver):
    A_sp, A_dn = _sparse_psd(60, seed=1)
    k = 5
    g = torch.Generator().manual_seed(0)
    vals, vecs = eigsh_sparse_sa(A_sp, k=k, solver=solver, generator=g)
    ref = torch.linalg.eigvalsh(A_dn)[:k]
    # eigsh (direct factorization) is exact to float64 round-off; the
    # LOBPCG branches stop at LOBPCG_TOL=1e-6 on the residual, which
    # for these well-separated spectra gives eigenvalues well below
    # 1e-8 absolute error.
    atol = 1e-10 if solver in ("eigsh", "auto") else 1e-7
    torch.testing.assert_close(vals, ref, atol=atol, rtol=0)
    # Property: eigenpair residual + orthonormality + ascending order.
    # The LOBPCG branches stop on a residual tolerance of LOBPCG_TOL=1e-6
    # RELATIVE to the operator scale (‖L‖ ≈ max degree ≈ 10 for these
    # graphs), so absolute residuals land a small factor above 1e-6.
    resid = (A_dn @ vecs - vecs * vals.unsqueeze(0)).norm(dim=0)
    assert (resid < 1e-6 * torch.linalg.matrix_norm(A_dn, ord=2)).all()
    torch.testing.assert_close(
        vecs.mT @ vecs, torch.eye(k, dtype=DT), atol=1e-8, rtol=0,
    )
    assert (vals.diff() >= -1e-12).all()


def test_scipy_branches_agree_with_each_other():
    A_sp, _ = _sparse_psd(80, seed=2)
    k = 6
    g = torch.Generator().manual_seed(0)
    vals_e, _ = eigsh_sparse_sa(A_sp, k=k, solver="eigsh")
    vals_l, _ = eigsh_sparse_sa(A_sp, k=k, solver="lobpcg", generator=g)
    # The substrate-era claim: same modes from both branches. Values
    # agree to the LOBPCG convergence tolerance or better.
    torch.testing.assert_close(vals_e, vals_l, atol=1e-7, rtol=0)


def test_lanczos_eigsh_dispatches_sparse_sa():
    A_sp, A_dn = _sparse_psd(50, seed=3)
    k = 4
    vals, vecs = lanczos_eigsh(A_sp, k, which="SA")
    ref = torch.linalg.eigvalsh(A_dn)[:k]
    torch.testing.assert_close(vals, ref, atol=1e-10, rtol=0)
    assert vecs.shape == (50, k)


def test_dense_and_sparse_sa_agree():
    # The dense in-house shift-invert and the sparse backend answer the
    # same question; cross-check on one operator.
    A_sp, A_dn = _sparse_psd(40, seed=4)
    k = 3
    vals_sp, _ = lanczos_eigsh(A_sp, k, which="SA")
    vals_dn, _ = lanczos_eigsh(
        A_dn.unsqueeze(0), k, which="SA",
        sigma=-1e-6, n_iter=40,
    )
    torch.testing.assert_close(vals_sp, vals_dn[0], atol=1e-8, rtol=0)


def test_float32_input_round_trips_dtype():
    A_sp, A_dn = _sparse_psd(30, seed=5)
    A32 = A_sp.to(torch.float32).coalesce()
    vals, vecs = eigsh_sparse_sa(A32, k=3, solver="eigsh")
    assert vals.dtype == torch.float32
    assert vecs.dtype == torch.float32
    ref = torch.linalg.eigvalsh(A_dn)[:3].to(torch.float32)
    torch.testing.assert_close(vals, ref, atol=1e-5, rtol=0)
