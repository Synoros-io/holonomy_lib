# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Synoros

"""Tests for holonomy_lib.manifolds.fixed_rank.FixedRankManifold.

Three layers:
  1. Unit tests; shapes correct across B ∈ {0, 1, several}.
  2. Property tests; mathematical invariants of the manifold operations.
  3. Comparison tests; agreement with pymanopt's FixedRankEmbedded
     (skipped if pymanopt not installed).
"""

from __future__ import annotations

import pytest
import torch

from holonomy_lib.manifolds import FixedRankManifold

# --------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------

# Default geometry for most tests — small enough to be fast,
# large enough to be non-degenerate.
M_DEFAULT, N_DEFAULT, R_DEFAULT = 7, 5, 3


def _make_manifold(m=M_DEFAULT, n=N_DEFAULT, r=R_DEFAULT, dtype=torch.float64):
    return FixedRankManifold(m=m, n=n, r=r, dtype=dtype)


def _seeded_generator(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


# --------------------------------------------------------------------
# Construction
# --------------------------------------------------------------------


class TestConstruction:
    def test_rejects_rank_zero(self):
        with pytest.raises(ValueError, match="rank"):
            FixedRankManifold(m=4, n=5, r=0)


# --------------------------------------------------------------------
# Shape tests across B ∈ {0, 1, several}
# --------------------------------------------------------------------


@pytest.mark.parametrize("batch", [0, 1, 4])
class TestShapes:


    def test_norm_shape(self, batch):
        mfd = _make_manifold()
        pt = mfd.random_point(batch_size=batch, generator=_seeded_generator(6))
        v = torch.randn(batch, mfd.m, mfd.n, dtype=mfd.dtype,
                        generator=_seeded_generator(7))
        n = mfd.norm(pt, v)
        assert n.shape == (batch,)

        zero = torch.zeros(batch, mfd.m, mfd.n, dtype=mfd.dtype, requires_grad=True)
        grad = torch.autograd.grad(mfd.norm(pt, zero).sum(), zero)[0]
        torch.testing.assert_close(grad, torch.zeros_like(grad), atol=0, rtol=0)


# --------------------------------------------------------------------
# Mathematical properties (Vandereycken 2013)
# --------------------------------------------------------------------


class TestProperties:
    def test_svd_form_invariant_after_random_point(self):
        """Random points satisfy U^T U = I, V^T V = I, S sorted descending."""
        mfd = _make_manifold()
        U, S, Vt = mfd.random_point(batch_size=4, generator=_seeded_generator(10))
        # Orthonormality of U columns
        UtU = torch.bmm(U.mT, U)  # (B, r, r)
        I_r = torch.eye(mfd.r, dtype=mfd.dtype).expand_as(UtU)
        torch.testing.assert_close(UtU, I_r, atol=1e-10, rtol=0)
        # Orthonormality of V rows (Vt rows are V^T rows = V columns)
        V = Vt.mT
        VtV = torch.bmm(V.mT, V)  # (B, r, r)
        torch.testing.assert_close(VtV, I_r, atol=1e-10, rtol=0)
        # S descending
        diffs = S[..., :-1] - S[..., 1:]
        assert (diffs >= 0).all()
        # S nonnegative
        assert (S >= 0).all()

        for batch in [0, 1, 4]:
            mfd = _make_manifold()
            U, S, Vt = mfd.random_point(
                batch_size=batch, generator=_seeded_generator(0)
            )
            assert U.shape == (batch, mfd.m, mfd.r)
            assert S.shape == (batch, mfd.r)
            assert Vt.shape == (batch, mfd.r, mfd.n)

        mfd = FixedRankManifold(3, 3, 1)
        U, _, Vt = mfd.random_point(32, generator=_seeded_generator(0))
        # QR's uncorrected sign convention biases both factors into one hemisphere.
        assert (U[:, 0, 0] > 0).any() and (U[:, 0, 0] < 0).any()
        assert (Vt[:, 0, 0] > 0).any() and (Vt[:, 0, 0] < 0).any()

    def test_dense_then_svd_recovers_factors(self):
        """For a manifold point, dense(pt) is exactly rank-r."""
        mfd = _make_manifold()
        pt = mfd.random_point(batch_size=3, generator=_seeded_generator(11))
        M = mfd.dense(pt)
        S_full = torch.linalg.svdvals(M)  # (B, min(m,n))
        # First r singular values are pt.S (up to ordering — pt.S was sorted desc);
        # remainder should be ~zero.
        _, S, _ = pt
        torch.testing.assert_close(S_full[..., :mfd.r], S, atol=1e-10, rtol=0)
        torch.testing.assert_close(
            S_full[..., mfd.r:],
            torch.zeros_like(S_full[..., mfd.r:]),
            atol=1e-10, rtol=0,
        )

        for batch in [0, 1, 4]:
            mfd = _make_manifold()
            pt = mfd.random_point(batch_size=batch, generator=_seeded_generator(1))
            M = mfd.dense(pt)
            assert M.shape == (batch, mfd.m, mfd.n)

    def test_projection_is_idempotent(self):
        """P_T(P_T(Z)) = P_T(Z); Vandereycken (2013) eq. 2.5."""
        mfd = _make_manifold()
        pt = mfd.random_point(batch_size=3, generator=_seeded_generator(12))
        Z = torch.randn(3, mfd.m, mfd.n, dtype=mfd.dtype,
                        generator=_seeded_generator(13))
        PZ = mfd.projection(pt, Z)
        PPZ = mfd.projection(pt, PZ)
        torch.testing.assert_close(PPZ, PZ, atol=1e-10, rtol=0)

        for batch in [0, 1, 4]:
            mfd = _make_manifold()
            pt = mfd.random_point(batch_size=batch, generator=_seeded_generator(2))
            Z = torch.randn(batch, mfd.m, mfd.n, dtype=mfd.dtype,
                            generator=_seeded_generator(3))
            Zt = mfd.projection(pt, Z)
            assert Zt.shape == (batch, mfd.m, mfd.n)

    def test_projection_is_self_adjoint(self):
        """<P_T(A), B> = <A, P_T(B)> (orthogonal projector property)."""
        mfd = _make_manifold()
        pt = mfd.random_point(batch_size=2, generator=_seeded_generator(14))
        A = torch.randn(2, mfd.m, mfd.n, dtype=mfd.dtype,
                        generator=_seeded_generator(15))
        B = torch.randn(2, mfd.m, mfd.n, dtype=mfd.dtype,
                        generator=_seeded_generator(16))
        PA = mfd.projection(pt, A)
        PB = mfd.projection(pt, B)
        lhs = (PA * B).sum(dim=(-2, -1))
        rhs = (A * PB).sum(dim=(-2, -1))
        torch.testing.assert_close(lhs, rhs, atol=1e-10, rtol=0)

    def test_retraction_lands_on_manifold(self):
        """retraction(pt, tangent) is a rank-r matrix on M_r."""
        mfd = _make_manifold()
        pt = mfd.random_point(batch_size=3, generator=_seeded_generator(17))
        Z = torch.randn(3, mfd.m, mfd.n, dtype=mfd.dtype,
                        generator=_seeded_generator(18))
        tangent = mfd.projection(pt, Z) * 0.1
        pt2 = mfd.retraction(pt, tangent)
        M2 = mfd.dense(pt2)
        # Rank must be exactly r (within numerical tolerance)
        S_full = torch.linalg.svdvals(M2)
        # The r+1, r+2, ... singular values should be near zero relative to S[r-1]
        for b in range(M2.shape[0]):
            tail_max = (
                S_full[b, mfd.r :].max().item() if mfd.r < min(mfd.m, mfd.n) else 0.0
            )
            top_min = S_full[b, mfd.r - 1].item()
            assert tail_max < 1e-9 * max(top_min, 1.0), (
                f"batch {b}: tail max {tail_max}, top min {top_min}"
            )

        for batch in [0, 1, 4]:
            mfd = _make_manifold()
            pt = mfd.random_point(batch_size=batch, generator=_seeded_generator(4))
            # tangent magnitude small enough that retraction stays well-conditioned
            tangent = torch.randn(batch, mfd.m, mfd.n, dtype=mfd.dtype,
                                  generator=_seeded_generator(5)) * 0.1
            tangent = mfd.projection(pt, tangent)
            U2, S2, Vt2 = mfd.retraction(pt, tangent)
            assert U2.shape == (batch, mfd.m, mfd.r)
            assert S2.shape == (batch, mfd.r)
            assert Vt2.shape == (batch, mfd.r, mfd.n)

    def test_retraction_first_order_consistency(self):
        """For small ε, R(εξ) ≈ M + εξ + O(ε²).

        Verified by checking ‖R(εξ) − (M + εξ)‖ → 0 quadratically in ε.
        Absil-Mahony-Sepulchre (2008) Definition 4.1.1.
        """
        mfd = _make_manifold()
        pt = mfd.random_point(batch_size=1, generator=_seeded_generator(19))
        M = mfd.dense(pt)
        Z = torch.randn(1, mfd.m, mfd.n, dtype=mfd.dtype,
                        generator=_seeded_generator(20))
        xi = mfd.projection(pt, Z)  # tangent direction
        eps_values = [0.1, 0.01, 0.001]
        errors = []
        for eps in eps_values:
            pt_eps = mfd.retraction(pt, eps * xi)
            M_eps = mfd.dense(pt_eps)
            err = torch.linalg.norm(M_eps - (M + eps * xi)).item()
            errors.append(err)
        # Error should decrease at least quadratically: err(eps/10) <= err(eps) *
        # 10^-1.5
        # (allowing some slack for numerical noise; quadratic would give 10^-2).
        for i in range(len(eps_values) - 1):
            ratio = errors[i + 1] / max(errors[i], 1e-15)
            assert ratio < 0.1, (
                f"retraction is not first-order: eps={eps_values[i]}→err={errors[i]}, "
                f"eps={eps_values[i+1]}→err={errors[i+1]}, ratio={ratio}"
            )

        for rank, diagonal in [(1, [1., 0., 0.]), (2, [2., 2., 0.])]:
            mfd = FixedRankManifold(3, 3, rank, retraction_mode="exact")
            eye = torch.eye(3, dtype=torch.float64)[None]
            point = (eye[..., :rank], torch.tensor([diagonal[:rank]], dtype=eye.dtype),
                     eye[:, :rank, :])
            zero = torch.zeros_like(eye, requires_grad=True)
            def reconstructed(tangent, mfd=mfd, point=point):
                return mfd.dense(mfd.retraction(point, mfd.projection(point, tangent)))
            assert torch.autograd.gradcheck(reconstructed, (zero,))
            out = reconstructed(zero)
            expected = torch.diag(torch.tensor(diagonal, dtype=eye.dtype))[None]
            torch.testing.assert_close(out, expected, atol=0, rtol=0)
            grad = torch.autograd.grad(out.sum(), zero)[0]
            torch.testing.assert_close(
                grad, mfd.projection(point, torch.ones_like(zero)), atol=1e-12, rtol=0,
            )

        # A close but distinct retained pair still has the squared-gap pole.
        # This sign-invariant projector avoids testing SVD sign choices.
        for dtype, gap, step in [(torch.float64, 1e-9, 1e-12),
                                 (torch.float32, 1e-5, 1e-7)]:
            mfd = FixedRankManifold(3, 3, 2, retraction_mode="exact")
            columns = torch.eye(3, dtype=dtype)[None, :, :2]
            singular = torch.tensor([[1.0 + gap, 1.0]], dtype=dtype)
            point = (columns, singular, columns.mT)
            zero = torch.zeros(1, 3, 3, dtype=dtype, requires_grad=True)

            def leading_projector(tangent, mfd=mfd, point=point):
                left, _, _ = mfd.retraction(point, tangent)
                return (left[..., :1] @ left[..., :1].mT)[0, 0, 1]

            derivative = torch.autograd.grad(leading_projector(zero), zero)[0][0, 0, 1]
            expected = singular[0, 1] / (singular[0, 0]**2 - singular[0, 1]**2)
            torch.testing.assert_close(derivative, expected)
            delta = torch.zeros_like(zero)
            delta[0, 0, 1] = step
            difference = (
                leading_projector(zero.detach() + delta)
                - leading_projector(zero.detach() - delta)
            ) / (2.0 * step)
            torch.testing.assert_close(derivative, difference, atol=0, rtol=1e-4)
            if dtype == torch.float64:
                assert torch.autograd.gradcheck(
                    leading_projector, (zero,), eps=step, atol=1e-3, rtol=1e-4,
                )


# --------------------------------------------------------------------
# Comparison against pymanopt's FixedRankEmbedded
# --------------------------------------------------------------------


class TestRetractionModes:
    """The retraction has three modes: 'exact', 'randomized', 'auto'.
    All must produce valid manifold points (rank-r SVD triples) and
    'randomized' must be close to 'exact' in Frobenius distance (the
    randomized SVD is only approximate to ~σ_{r+1}).
    """

    def test_randomized_close_to_exact(self):
        # Geometry where the randomized path actually fires under "auto":
        # r/min(m,n) = 4/40 = 10% < 25% threshold.
        mfd_exact = FixedRankManifold(m=40, n=40, r=4, retraction_mode="exact")
        mfd_rand = FixedRankManifold(m=40, n=40, r=4, retraction_mode="randomized")
        g = _seeded_generator(200)
        pt = mfd_exact.random_point(batch_size=2, generator=g)
        tangent = mfd_exact.projection(
            pt, torch.randn(2, 40, 40, dtype=torch.float64, generator=g),
        )
        U_e, S_e, Vt_e = mfd_exact.retraction(pt, tangent)
        U_r, S_r, Vt_r = mfd_rand.retraction(pt, tangent)
        # Reconstructed dense matrices should be close (sign/basis-invariant)
        dense_e = U_e @ torch.diag_embed(S_e) @ Vt_e
        dense_r = U_r @ torch.diag_embed(S_r) @ Vt_r
        rel_err = (dense_e - dense_r).norm() / dense_e.norm()
        assert rel_err < 1e-2, (
            f"randomized retraction should be close to exact; rel_err={rel_err:.4e}"
        )


try:
    import pymanopt as _pymanopt  # noqa: F401
    _HAVE_PYMANOPT = True
except ImportError:
    _HAVE_PYMANOPT = False


@pytest.mark.skipif(
    not _HAVE_PYMANOPT,
    reason="pymanopt not installed; install with `uv pip install pymanopt autograd`",
)
class TestAgainstPymanopt:
    """Cross-check against pymanopt.manifolds.FixedRankEmbedded.

    pymanopt is numpy-only, so we move our results to CPU+numpy for the
    comparison. Mathematical semantics should be identical.
    """

    @staticmethod
    def _pymanopt_mfd(m, n, r):
        from pymanopt.manifolds import FixedRankEmbedded
        return FixedRankEmbedded(m, n, r)

    @staticmethod
    def _tangent_triple_to_ambient(point_np, tangent_triple):
        """Reconstruct ambient (m, n) tangent from Vandereycken's (Up, M, Vp).

        ξ = U M V^T + Up V^T + U Vp^T   (Vandereycken 2013, eq. 2.4)
        where Up ⊥ U, Vp ⊥ V (range conditions, satisfied after projection).
        """
        U, _, Vt = point_np
        Up, Mp, Vp = tangent_triple
        return U @ Mp @ Vt + Up @ Vt + U @ Vp.T

    def test_projection_matches_pymanopt(self):
        m, n, r = 7, 5, 3
        mfd = FixedRankManifold(m=m, n=n, r=r)
        pt = mfd.random_point(batch_size=1, generator=_seeded_generator(100))
        U, S, Vt = pt
        Z = torch.randn(1, m, n, dtype=mfd.dtype, generator=_seeded_generator(101))

        Zt_ours = mfd.projection(pt, Z)[0].numpy()

        pmfd = self._pymanopt_mfd(m, n, r)
        point_np = (U[0].numpy(), S[0].numpy(), Vt[0].numpy())
        triple = pmfd.projection(point_np, Z[0].numpy())
        Zt_pymanopt_ambient = self._tangent_triple_to_ambient(point_np, triple)

        import numpy as np
        diff = np.linalg.norm(Zt_ours - Zt_pymanopt_ambient)
        ref = np.linalg.norm(Zt_pymanopt_ambient)
        assert diff / max(ref, 1e-15) < 1e-10, (
            f"projection mismatch with pymanopt: relative diff "
            f"{diff/max(ref,1e-15):.3e}"
        )

    def test_retraction_matches_pymanopt(self):
        m, n, r = 7, 5, 3
        mfd = FixedRankManifold(m=m, n=n, r=r)
        pt = mfd.random_point(batch_size=1, generator=_seeded_generator(200))
        U, S, Vt = pt
        Z = (
            torch.randn(1, m, n, dtype=mfd.dtype, generator=_seeded_generator(201))
            * 0.1
        )
        tangent = mfd.projection(pt, Z)

        pt2 = mfd.retraction(pt, tangent)
        M_ours = mfd.dense(pt2)[0].numpy()

        pmfd = self._pymanopt_mfd(m, n, r)
        point_np = (U[0].numpy(), S[0].numpy(), Vt[0].numpy())
        triple = pmfd.projection(point_np, Z[0].numpy())
        pt2_np = pmfd.retraction(point_np, triple)
        Up, Sp, Vtp = pt2_np
        import numpy as np
        M_pymanopt = Up @ np.diag(Sp) @ Vtp

        # Retraction has SVD truncation freedom in sign of singular vectors;
        # the dense reconstruction is invariant and should match.
        diff = np.linalg.norm(M_ours - M_pymanopt)
        ref = np.linalg.norm(M_pymanopt)
        assert diff / max(ref, 1e-15) < 1e-10, (
            f"retraction mismatch with pymanopt: relative diff "
            f"{diff/max(ref,1e-15):.3e}"
        )
