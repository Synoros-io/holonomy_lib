# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Synoros

"""Tests for holonomy_lib.manifolds.spd.SPDManifold.

Three layers:
  1. Unit tests — shapes correct across B ∈ {0, 1, several}.
  2. Property tests — mathematical invariants (exp/log inverse, symmetry
     and positivity of distance, isometry under congruence).
  3. Comparison tests — agreement with geoopt's `SymmetricPositiveDefinite`
     (skipped if geoopt not installed).
"""

from __future__ import annotations

import pytest
import torch

from holonomy_lib.manifolds import SPDManifold

N_DEFAULT = 4


def _make_manifold(n=N_DEFAULT, dtype=torch.float64):
    return SPDManifold(n=n, dtype=dtype)


def _seeded_generator(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


# --------------------------------------------------------------------
# Construction
# --------------------------------------------------------------------


class TestConstruction:
    def test_rejects_n_zero(self):
        with pytest.raises(ValueError, match="n"):
            SPDManifold(n=0)

    def test_rejects_n_negative(self):
        with pytest.raises(ValueError, match="n"):
            SPDManifold(n=-3)

    def test_dimension_formula(self):
        # P(n) has dimension n(n+1)/2
        assert SPDManifold(n=1).dim == 1
        assert SPDManifold(n=4).dim == 10
        assert SPDManifold(n=7).dim == 28


# --------------------------------------------------------------------
# Shapes across B ∈ {0, 1, several}
# --------------------------------------------------------------------


@pytest.mark.parametrize("batch", [0, 1, 4])
class TestShapes:
    def test_random_point_shape(self, batch):
        mfd = _make_manifold()
        S = mfd.random_point(batch_size=batch, generator=_seeded_generator(0))
        assert S.shape == (batch, mfd.n, mfd.n)

    def test_projection_shape(self, batch):
        mfd = _make_manifold()
        S = mfd.random_point(batch_size=batch, generator=_seeded_generator(1))
        Z = torch.randn(batch, mfd.n, mfd.n, dtype=mfd.dtype,
                        generator=_seeded_generator(2))
        V = mfd.projection(S, Z)
        assert V.shape == (batch, mfd.n, mfd.n)

    def test_inner_shape(self, batch):
        mfd = _make_manifold()
        S = mfd.random_point(batch_size=max(batch, 1),
                              generator=_seeded_generator(3))
        # Use batch=1 for "B=0" inner-product test — skip B=0 entirely
        # since solve on (0, n, n) is fine but adds no value.
        if batch == 0:
            pytest.skip("inner product on B=0 is trivially empty")
        U = mfd.projection(
            S, torch.randn(batch, mfd.n, mfd.n, dtype=mfd.dtype,
                            generator=_seeded_generator(4)),
        )
        V = mfd.projection(
            S, torch.randn(batch, mfd.n, mfd.n, dtype=mfd.dtype,
                            generator=_seeded_generator(5)),
        )
        ip = mfd.inner(S, U, V)
        assert ip.shape == (batch,)

    def test_exp_shape(self, batch):
        mfd = _make_manifold()
        S = mfd.random_point(batch_size=batch, generator=_seeded_generator(6))
        V = mfd.projection(
            S, torch.randn(batch, mfd.n, mfd.n, dtype=mfd.dtype,
                            generator=_seeded_generator(7)),
        ) * 0.1
        out = mfd.exp(S, V)
        assert out.shape == (batch, mfd.n, mfd.n)

    def test_distance_shape(self, batch):
        mfd = _make_manifold()
        S = mfd.random_point(batch_size=batch, generator=_seeded_generator(8))
        T = mfd.random_point(batch_size=batch, generator=_seeded_generator(9))
        d = mfd.distance(S, T)
        assert d.shape == (batch,)


# --------------------------------------------------------------------
# Properties — affine-invariant SPD
# --------------------------------------------------------------------


class TestProperties:
    def test_random_point_is_spd(self):
        mfd = _make_manifold()
        S = mfd.random_point(batch_size=8, generator=_seeded_generator(10))
        assert mfd.is_spd(S).all()

    def test_projection_symmetrizes(self):
        mfd = _make_manifold()
        S = mfd.random_point(batch_size=3, generator=_seeded_generator(11))
        Z = torch.randn(3, mfd.n, mfd.n, dtype=mfd.dtype,
                        generator=_seeded_generator(12))
        V = mfd.projection(S, Z)
        torch.testing.assert_close(V, V.mT, atol=1e-12, rtol=0)

    def test_projection_idempotent(self):
        mfd = _make_manifold()
        S = mfd.random_point(batch_size=3, generator=_seeded_generator(13))
        Z = torch.randn(3, mfd.n, mfd.n, dtype=mfd.dtype,
                        generator=_seeded_generator(14))
        V = mfd.projection(S, Z)
        VV = mfd.projection(S, V)
        torch.testing.assert_close(VV, V, atol=1e-12, rtol=0)

    def test_inner_is_symmetric_and_positive(self):
        mfd = _make_manifold()
        S = mfd.random_point(batch_size=4, generator=_seeded_generator(15))
        U = mfd.projection(
            S, torch.randn(4, mfd.n, mfd.n, dtype=mfd.dtype,
                            generator=_seeded_generator(16)),
        )
        V = mfd.projection(
            S, torch.randn(4, mfd.n, mfd.n, dtype=mfd.dtype,
                            generator=_seeded_generator(17)),
        )
        # <U, V>_S == <V, U>_S
        torch.testing.assert_close(
            mfd.inner(S, U, V), mfd.inner(S, V, U), atol=1e-10, rtol=0
        )
        # <U, U>_S > 0 for nonzero U
        self_ip = mfd.inner(S, U, U)
        assert (self_ip > 0).all()

    def test_exp_lands_on_spd(self):
        mfd = _make_manifold()
        S = mfd.random_point(batch_size=4, generator=_seeded_generator(18))
        V = mfd.projection(
            S, torch.randn(4, mfd.n, mfd.n, dtype=mfd.dtype,
                            generator=_seeded_generator(19)),
        ) * 0.1
        T = mfd.exp(S, V)
        assert mfd.is_spd(T).all()

    def test_exp_log_inverse(self):
        """log_S(exp_S(V)) = V for V in the tangent space."""
        mfd = _make_manifold()
        S = mfd.random_point(batch_size=3, generator=_seeded_generator(20))
        V = mfd.projection(
            S, torch.randn(3, mfd.n, mfd.n, dtype=mfd.dtype,
                            generator=_seeded_generator(21)),
        ) * 0.1
        T = mfd.exp(S, V)
        V_recovered = mfd.log(S, T)
        torch.testing.assert_close(V_recovered, V, atol=1e-10, rtol=1e-10)

    def test_log_exp_inverse(self):
        """exp_S(log_S(T)) = T for T on the manifold."""
        mfd = _make_manifold()
        S = mfd.random_point(batch_size=3, generator=_seeded_generator(22))
        T = mfd.random_point(batch_size=3, generator=_seeded_generator(23))
        V = mfd.log(S, T)
        T_recovered = mfd.exp(S, V)
        torch.testing.assert_close(T_recovered, T, atol=1e-10, rtol=1e-10)

    def test_distance_symmetric(self):
        """d(S, T) = d(T, S)."""
        mfd = _make_manifold()
        S = mfd.random_point(batch_size=4, generator=_seeded_generator(24))
        T = mfd.random_point(batch_size=4, generator=_seeded_generator(25))
        d_st = mfd.distance(S, T)
        d_ts = mfd.distance(T, S)
        torch.testing.assert_close(d_st, d_ts, atol=1e-10, rtol=0)

    def test_distance_zero_at_same_point(self):
        """d(S, S) = 0."""
        mfd = _make_manifold()
        S = mfd.random_point(batch_size=3, generator=_seeded_generator(26))
        d = mfd.distance(S, S)
        torch.testing.assert_close(
            d, torch.zeros_like(d), atol=1e-9, rtol=0,
        )

    def test_distance_matches_norm_of_log(self):
        """d(S, T) = ‖log_S(T)‖_S (geodesic length)."""
        mfd = _make_manifold()
        S = mfd.random_point(batch_size=3, generator=_seeded_generator(27))
        T = mfd.random_point(batch_size=3, generator=_seeded_generator(28))
        d = mfd.distance(S, T)
        V = mfd.log(S, T)
        n_v = mfd.norm(S, V)
        torch.testing.assert_close(d, n_v, atol=1e-10, rtol=1e-10)

    def test_affine_invariance(self):
        """d(A S Aᵀ, A T Aᵀ) = d(S, T) for any invertible A.

        This is the defining property of the affine-invariant metric.
        References: Pennec et al. (2006), §3.
        """
        mfd = _make_manifold()
        B = 3
        S = mfd.random_point(batch_size=B, generator=_seeded_generator(29))
        T = mfd.random_point(batch_size=B, generator=_seeded_generator(30))
        # Random invertible A: A = exp of random matrix (always invertible)
        M = torch.randn(B, mfd.n, mfd.n, dtype=mfd.dtype,
                        generator=_seeded_generator(31)) * 0.1
        A = torch.matrix_exp(M)
        A_T = A.mT
        S_cong = A @ S @ A_T
        T_cong = A @ T @ A_T
        d_orig = mfd.distance(S, T)
        d_cong = mfd.distance(S_cong, T_cong)
        torch.testing.assert_close(d_orig, d_cong, atol=1e-9, rtol=1e-9)


# --------------------------------------------------------------------
# Comparison against geoopt's SymmetricPositiveDefinite
# --------------------------------------------------------------------


class TestNumericalRobustness:
    """SPD operations must survive near-singular inputs (low eigenvalue
    near zero) without producing NaN/inf, and exp/log outputs must
    stay symmetric to machine precision so chained calls don't drift
    off the SPD cone."""

    def test_n_equal_one_is_spd(self):
        """Trivial n=1 case — output is a scalar SPD matrix."""
        mfd = SPDManifold(n=1)
        S = mfd.random_point(batch_size=4, generator=_seeded_generator(110))
        assert S.shape == (4, 1, 1)
        assert mfd.is_spd(S).all()

    def test_near_singular_does_not_explode(self):
        """For an SPD matrix with one tiny but well-above-tiny eigenvalue,
        exp/log/distance produce finite (if large) values. The clamp
        protects against exact-zero or float-noise-negative eigenvalues
        produced by eigh on numerically singular SPD; it does NOT
        promise finite results for inputs at the cone boundary, which
        are mathematically undefined under the affine-invariant metric.
        """
        mfd = SPDManifold(n=4)
        # eigvals = [1e-12, 1, 1, 1] — tiny but well-conditioned for
        # the clamp's guarantee. Pre-clamp this would still be finite,
        # but the test pins down that the clamp doesn't *hurt* the
        # well-conditioned case.
        eigs = torch.tensor([[1e-12, 1.0, 1.0, 1.0]], dtype=torch.float64)
        Q, _ = torch.linalg.qr(
            torch.randn(1, 4, 4, dtype=torch.float64,
                        generator=_seeded_generator(111))
        )
        S = Q @ torch.diag_embed(eigs) @ Q.mT
        S = 0.5 * (S + S.mT)
        T = torch.eye(4, dtype=torch.float64).unsqueeze(0)
        d = mfd.distance(S, T)
        assert torch.isfinite(d).all(), f"distance produced non-finite: {d}"
        log_ST = mfd.log(S, T)
        assert torch.isfinite(log_ST).all(), "log produced non-finite"

    def test_zero_eigenvalue_does_not_nan(self):
        """The critical case: an eigenvalue at EXACTLY 0 (e.g. an SPD
        matrix that has been projected onto the cone boundary). Before
        the clamp, `torch.rsqrt(0)` produced inf → NaN propagation.
        After the clamp we get a finite (huge) result for inv_sqrt, and
        downstream values stay finite-or-overflow rather than NaN.
        """
        mfd = SPDManifold(n=3)
        # Force a zero eigenvalue
        eigs = torch.tensor([[0.0, 1.0, 1.0]], dtype=torch.float64)
        Q, _ = torch.linalg.qr(
            torch.randn(1, 3, 3, dtype=torch.float64,
                        generator=_seeded_generator(116))
        )
        S = Q @ torch.diag_embed(eigs) @ Q.mT
        S = 0.5 * (S + S.mT)
        # We only require: no NaN. inf is mathematically expected at
        # the cone boundary; the test guards against the silent-NaN
        # regression.
        S_sqrt, S_inv_sqrt = mfd._sqrt_and_inv_sqrt(S)
        assert not torch.isnan(S_sqrt).any()
        assert not torch.isnan(S_inv_sqrt).any()

    def test_exp_output_is_symmetric_to_machine_precision(self):
        """After the symmetrize-output fix, exp_S(V) is symmetric to
        roundoff, not just the inner argument."""
        mfd = SPDManifold(n=5)
        S = mfd.random_point(batch_size=3, generator=_seeded_generator(112))
        V = torch.randn(3, 5, 5, dtype=torch.float64,
                        generator=_seeded_generator(113))
        V = 0.5 * (V + V.mT)  # symmetric tangent
        out = mfd.exp(S, V)
        torch.testing.assert_close(out, out.mT, atol=1e-12, rtol=0)

    def test_log_output_is_symmetric_to_machine_precision(self):
        mfd = SPDManifold(n=5)
        S = mfd.random_point(batch_size=3, generator=_seeded_generator(114))
        T = mfd.random_point(batch_size=3, generator=_seeded_generator(115))
        out = mfd.log(S, T)
        torch.testing.assert_close(out, out.mT, atol=1e-12, rtol=0)


try:
    import geoopt as _geoopt  # noqa: F401
    _HAVE_GEOOPT = True
except ImportError:
    _HAVE_GEOOPT = False


@pytest.mark.skipif(
    not _HAVE_GEOOPT,
    reason="geoopt not installed; install with `uv pip install geoopt`",
)
class TestAgainstGeoopt:
    """Cross-check exp / log / distance against geoopt's SPD.

    geoopt uses the affine-invariant metric by default (its
    `SymmetricPositiveDefinite` manifold with default `default_metric="AIM"`).
    """

    @staticmethod
    def _geoopt_mfd():
        from geoopt import SymmetricPositiveDefinite
        return SymmetricPositiveDefinite()

    def test_distance_matches_geoopt(self):
        mfd = _make_manifold()
        S = mfd.random_point(batch_size=4, generator=_seeded_generator(100))
        T = mfd.random_point(batch_size=4, generator=_seeded_generator(101))
        d_ours = mfd.distance(S, T)
        gmfd = self._geoopt_mfd()
        d_geoopt = gmfd.dist(S, T)
        torch.testing.assert_close(d_ours, d_geoopt, atol=1e-10, rtol=1e-10)

    def test_exp_matches_geoopt(self):
        mfd = _make_manifold()
        S = mfd.random_point(batch_size=3, generator=_seeded_generator(102))
        V = mfd.projection(
            S, torch.randn(3, mfd.n, mfd.n, dtype=mfd.dtype,
                            generator=_seeded_generator(103)),
        ) * 0.1
        T_ours = mfd.exp(S, V)
        gmfd = self._geoopt_mfd()
        T_geoopt = gmfd.expmap(S, V)
        torch.testing.assert_close(T_ours, T_geoopt, atol=1e-10, rtol=1e-10)

    def test_log_matches_geoopt(self):
        mfd = _make_manifold()
        S = mfd.random_point(batch_size=3, generator=_seeded_generator(104))
        T = mfd.random_point(batch_size=3, generator=_seeded_generator(105))
        V_ours = mfd.log(S, T)
        gmfd = self._geoopt_mfd()
        V_geoopt = gmfd.logmap(S, T)
        torch.testing.assert_close(V_ours, V_geoopt, atol=1e-10, rtol=1e-10)
