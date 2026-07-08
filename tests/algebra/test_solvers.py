# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Synoros

"""Tests for holonomy_lib.algebra.solvers (cg, lsqr).

Three layers:
  1. Unit — shapes across B ∈ {0, 1, >1}; validation rejects.
  2. Property — cg on SPD matches `torch.linalg.solve`; lsqr matches
     `torch.linalg.lstsq`; damped lsqr matches the augmented system.
  3. Comparison — scipy.sparse.linalg.cg / lsqr on the same operators.
"""

from __future__ import annotations

import pytest
import torch

from holonomy_lib.algebra import cg, lsqr

DT = torch.float64


def _spd(B, n, seed=0):
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(B, n, n, generator=g, dtype=DT)
    return A @ A.mT + n * torch.eye(n, dtype=DT)


# --------------------------------------------------------------------
# cg
# --------------------------------------------------------------------


class TestCG:
    @pytest.mark.parametrize("B", [0, 1, 4])
    def test_matches_dense_solve_batched(self, B):
        n = 24
        A = _spd(max(B, 1), n)[:B] if B else _spd(1, n)[:0]
        g = torch.Generator().manual_seed(1)
        b = torch.randn(B, n, generator=g, dtype=DT)
        x = cg(A, b)
        assert x.shape == (B, n)
        if B:
            ref = torch.linalg.solve(A, b.unsqueeze(-1)).squeeze(-1)
            torch.testing.assert_close(x, ref, atol=1e-6, rtol=0)

    def test_unbatched_b_round_trips_shape(self):
        n = 16
        A = _spd(1, n).squeeze(0)
        g = torch.Generator().manual_seed(2)
        b = torch.randn(n, generator=g, dtype=DT)
        x = cg(A, b)
        assert x.shape == (n,)
        ref = torch.linalg.solve(A, b)
        torch.testing.assert_close(x, ref, atol=1e-6, rtol=0)

    def test_unbatched_A_broadcasts_over_batched_b(self):
        n, B = 16, 3
        A = _spd(1, n).squeeze(0)
        g = torch.Generator().manual_seed(3)
        b = torch.randn(B, n, generator=g, dtype=DT)
        x = cg(A, b)
        ref = torch.linalg.solve(A, b.mT).mT
        torch.testing.assert_close(x, ref, atol=1e-6, rtol=0)

    def test_sparse_operator(self):
        # Sparse SPD (graph Laplacian + I) applied to a batch of rhs.
        n, B = 30, 3
        edges = [(i, (i + 1) % n) for i in range(n)]
        L = torch.zeros(n, n, dtype=DT)
        for u, v in edges:
            L[u, u] += 1.0
            L[v, v] += 1.0
            L[u, v] -= 1.0
            L[v, u] -= 1.0
        S = L + torch.eye(n, dtype=DT)
        g = torch.Generator().manual_seed(4)
        b = torch.randn(B, n, generator=g, dtype=DT)
        x = cg(S.to_sparse_coo(), b)
        ref = torch.linalg.solve(S, b.mT).mT
        torch.testing.assert_close(x, ref, atol=1e-6, rtol=0)

    def test_callable_operator(self):
        n = 20
        A = _spd(1, n).squeeze(0)
        g = torch.Generator().manual_seed(5)
        b = torch.randn(2, n, generator=g, dtype=DT)
        x = cg(lambda v: (A @ v.mT).mT, b)
        ref = torch.linalg.solve(A, b.mT).mT
        torch.testing.assert_close(x, ref, atol=1e-6, rtol=0)

    def test_jacobi_preconditioner(self):
        n = 24
        A = _spd(1, n).squeeze(0)
        g = torch.Generator().manual_seed(6)
        b = torch.randn(1, n, generator=g, dtype=DT)
        inv_diag = 1.0 / A.diagonal()
        x = cg(A, b, M=lambda r: r * inv_diag)
        ref = torch.linalg.solve(A, b.mT).mT
        torch.testing.assert_close(x, ref, atol=1e-6, rtol=0)

    def test_warns_when_not_converged(self):
        n = 32
        A = _spd(1, n)
        g = torch.Generator().manual_seed(7)
        b = torch.randn(1, n, generator=g, dtype=DT)
        with pytest.warns(UserWarning, match="not converged"):
            cg(A, b, maxiter=1)

    def test_validation(self):
        A = _spd(1, 8)
        b = torch.randn(1, 8, dtype=DT)
        with pytest.raises(ValueError, match="sync_every"):
            cg(A, b, sync_every=0)
        with pytest.raises(ValueError, match="callable"):
            cg("not an operator", b)
        with pytest.raises(ValueError, match="b must be"):
            cg(A, torch.randn(2, 3, 8, dtype=DT))

    def test_comparison_scipy(self):
        scipy_spla = pytest.importorskip("scipy.sparse.linalg")
        n = 40
        A = _spd(1, n, seed=8).squeeze(0)
        g = torch.Generator().manual_seed(8)
        b = torch.randn(n, generator=g, dtype=DT)
        x = cg(A, b)
        x_sp, info = scipy_spla.cg(A.numpy(), b.numpy(), rtol=1e-10)
        assert info == 0
        torch.testing.assert_close(
            x, torch.from_numpy(x_sp), atol=1e-6, rtol=0,
        )


# --------------------------------------------------------------------
# lsqr
# --------------------------------------------------------------------


def _tall(B, m, n, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, m, n, generator=g, dtype=DT)


class TestLSQR:
    @pytest.mark.parametrize("B", [0, 1, 4])
    def test_overdetermined_matches_lstsq(self, B):
        m, n = 40, 12
        A = _tall(max(B, 1), m, n)[:B] if B else _tall(1, m, n)[:0]
        g = torch.Generator().manual_seed(1)
        b = torch.randn(B, m, generator=g, dtype=DT)
        x = lsqr(A, b)
        assert x.shape == (B, n)
        if B:
            ref = torch.linalg.lstsq(A, b.unsqueeze(-1)).solution.squeeze(-1)
            torch.testing.assert_close(x, ref, atol=1e-7, rtol=0)

    def test_unbatched_shapes(self):
        m, n = 30, 10
        A = _tall(1, m, n, seed=2).squeeze(0)
        g = torch.Generator().manual_seed(2)
        b = torch.randn(m, generator=g, dtype=DT)
        x = lsqr(A, b)
        assert x.shape == (n,)
        ref = torch.linalg.lstsq(A, b.unsqueeze(-1)).solution.squeeze(-1)
        torch.testing.assert_close(x, ref, atol=1e-7, rtol=0)

    def test_damped_matches_augmented_system(self):
        # min ‖Ax−b‖² + damp²‖x‖²  ==  lstsq on [[A], [damp·I]] vs [b; 0].
        m, n, damp = 30, 10, 0.7
        A = _tall(1, m, n, seed=3).squeeze(0)
        g = torch.Generator().manual_seed(3)
        b = torch.randn(m, generator=g, dtype=DT)
        x = lsqr(A, b, damp=damp)
        A_aug = torch.cat((A, damp * torch.eye(n, dtype=DT)), dim=0)
        b_aug = torch.cat((b, torch.zeros(n, dtype=DT)))
        ref = torch.linalg.lstsq(A_aug, b_aug.unsqueeze(-1)).solution.squeeze(-1)
        torch.testing.assert_close(x, ref, atol=1e-7, rtol=0)

    def test_rank_deficient_minimum_norm(self):
        # Column 2 = column 0 + column 1: rank-deficient. LSQR converges
        # to the minimum-norm least-squares solution (= pinv solution).
        m = 20
        g = torch.Generator().manual_seed(4)
        C = torch.randn(m, 2, generator=g, dtype=DT)
        A = torch.cat((C, (C[:, :1] + C[:, 1:2])), dim=1)  # (m, 3), rank 2
        b = torch.randn(m, generator=g, dtype=DT)
        x = lsqr(A, b)
        ref = torch.linalg.pinv(A) @ b
        torch.testing.assert_close(x, ref, atol=1e-7, rtol=0)

    def test_sparse_matches_scipy(self):
        scipy_spla = pytest.importorskip("scipy.sparse.linalg")
        import scipy.sparse as sp

        # Sparse oriented incidence of a path graph (the
        # hierarchy-potential shape from substrate).
        n = 25
        rows, cols, vals = [], [], []
        for e in range(n - 1):
            rows += [e, e]
            cols += [e, e + 1]
            vals += [-1.0, 1.0]
        A = torch.sparse_coo_tensor(
            torch.tensor([rows, cols]), torch.tensor(vals, dtype=DT),
            (n - 1, n),
        ).coalesce()
        g = torch.Generator().manual_seed(5)
        b = torch.randn(n - 1, generator=g, dtype=DT)
        x = lsqr(A, b)
        A_sp = sp.coo_matrix(
            (A.values().numpy(), A.indices().numpy()), shape=(n - 1, n),
        ).tocsr()
        x_sp = scipy_spla.lsqr(A_sp, b.numpy(), atol=1e-12, btol=1e-12)[0]
        torch.testing.assert_close(
            x, torch.from_numpy(x_sp), atol=1e-6, rtol=0,
        )

    def test_zero_rhs_gives_zero(self):
        A = _tall(1, 12, 5, seed=6).squeeze(0)
        x = lsqr(A, torch.zeros(12, dtype=DT))
        torch.testing.assert_close(
            x, torch.zeros(5, dtype=DT), atol=0.0, rtol=0.0,
        )

    def test_validation(self):
        A = _tall(1, 12, 5, seed=7).squeeze(0)
        b = torch.randn(12, dtype=DT)
        with pytest.raises(ValueError, match="damp"):
            lsqr(A, b, damp=-1.0)
        with pytest.raises(ValueError, match="row count"):
            lsqr(A, torch.randn(9, dtype=DT))
        with pytest.raises(ValueError, match="callable|torch.Tensor"):
            lsqr(lambda v: v, b)
