# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Synoros

"""Tests for holonomy_lib.manifolds.lorentz.LorentzManifold.

Five layers:
  1. Construction validation.
  2. Unit tests; shapes correct across B ∈ {0, 1, several}.
  3. Property tests; point invariants, exp/log inverse, distance
     symmetry & triangle inequality, parallel-transport isometry.
  4. Numerical robustness; near-origin tangents, large tangents.
  5. Comparison tests; agreement with geoopt's Lorentz model
     (skipped if geoopt not installed).
  6. RiemannianSGD integration.
"""

from __future__ import annotations

import pytest
import torch

from holonomy_lib.manifolds import LorentzManifold
from holonomy_lib.optimization import RiemannianSGD

N_DEFAULT = 4


def _make_manifold(n=N_DEFAULT, k=-1.0, dtype=torch.float64):
    return LorentzManifold(n=n, k=k, dtype=dtype)


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
            LorentzManifold(n=0)


    def test_origin_satisfies_constraint(self):
        for n in (1, 3, 7):
            for k in (-1.0, -0.5, -2.0):
                mfd = LorentzManifold(n=n, k=k)
                o = mfd.origin(batch_size=2)
                assert o.shape == (2, n + 1)
                # ⟨o, o⟩_M = 1/k
                ip = mfd.minkowski_inner(o, o)
                torch.testing.assert_close(
                    ip, torch.full_like(ip, 1.0 / k), atol=1e-12, rtol=0,
                )
                assert (o[..., 0] > 0).all()

        for batch in [0, 1, 4]:
            mfd = _make_manifold()
            o = mfd.origin(batch_size=batch)
            assert o.shape == (batch, mfd.n + 1)

        mfd = _make_manifold(k=-0.7)
        o = mfd.origin(batch_size=3)
        assert mfd.is_on_manifold(o).all()


# --------------------------------------------------------------------
# Properties — hyperboloid invariants
# --------------------------------------------------------------------


class TestProperties:
    def test_random_point_is_on_manifold(self):
        mfd = _make_manifold()
        x = mfd.random_point(batch_size=8, generator=_seeded_generator(20))
        assert mfd.is_on_manifold(x).all()

        for batch in [0, 1, 4]:
            mfd = _make_manifold()
            x = mfd.random_point(batch_size=batch, generator=_seeded_generator(0))
            assert x.shape == (batch, mfd.n + 1)

        for k in [-0.5, -2.0, -1.7]:
            mfd = _make_manifold(k=k)
            x = mfd.random_point(batch_size=4, generator=_seeded_generator(60))
            assert mfd.is_on_manifold(x).all()

        mfd = _make_manifold(k=-1.0)
        # The local manifold exposes inner via .inner; on the hyperboloid all
        # points satisfy ⟨x, x⟩_M = -1, so this is a fixed check.
        x = mfd.random_point(batch_size=2, generator=_seeded_generator(80))
        assert torch.allclose(
            mfd.minkowski_inner(x, x),
            torch.full((2,), -1.0, dtype=torch.float64),
            atol=1e-10,
        )


    def test_projection_lands_in_tangent_space(self):
        """proj(x, w) must be orthogonal to x in the Minkowski form."""
        mfd = _make_manifold()
        x = mfd.random_point(batch_size=4, generator=_seeded_generator(21))
        w = torch.randn(4, mfd.n + 1, dtype=mfd.dtype,
                        generator=_seeded_generator(22))
        v = mfd.projection(x, w)
        ip = mfd.minkowski_inner(x, v)
        torch.testing.assert_close(
            ip, torch.zeros_like(ip), atol=1e-12, rtol=0,
        )

        for batch in [0, 1, 4]:
            mfd = _make_manifold()
            x = mfd.random_point(batch_size=batch, generator=_seeded_generator(1))
            w = torch.randn(batch, mfd.n + 1, dtype=mfd.dtype,
                            generator=_seeded_generator(2))
            v = mfd.projection(x, w)
            assert v.shape == (batch, mfd.n + 1)

    def test_projection_idempotent(self):
        mfd = _make_manifold()
        x = mfd.random_point(batch_size=4, generator=_seeded_generator(23))
        w = torch.randn(4, mfd.n + 1, dtype=mfd.dtype,
                        generator=_seeded_generator(24))
        v = mfd.projection(x, w)
        vv = mfd.projection(x, v)
        torch.testing.assert_close(vv, v, atol=1e-12, rtol=0)

    def test_inner_is_symmetric_and_positive(self):
        """Riemannian inner is symmetric and positive on tangent vectors."""
        mfd = _make_manifold()
        x = mfd.random_point(batch_size=4, generator=_seeded_generator(25))
        u = mfd.projection(
            x, torch.randn(4, mfd.n + 1, dtype=mfd.dtype,
                            generator=_seeded_generator(26)),
        )
        v = mfd.projection(
            x, torch.randn(4, mfd.n + 1, dtype=mfd.dtype,
                            generator=_seeded_generator(27)),
        )
        torch.testing.assert_close(
            mfd.inner(x, u, v), mfd.inner(x, v, u), atol=1e-12, rtol=0,
        )
        self_ip = mfd.inner(x, u, u)
        assert (self_ip > 0).all()

        for batch in [0, 1, 4]:
            mfd = _make_manifold()
            x = mfd.random_point(batch_size=batch, generator=_seeded_generator(3))
            u = mfd.projection(
                x, torch.randn(batch, mfd.n + 1, dtype=mfd.dtype,
                                generator=_seeded_generator(4)),
            )
            v = mfd.projection(
                x, torch.randn(batch, mfd.n + 1, dtype=mfd.dtype,
                                generator=_seeded_generator(5)),
            )
            ip = mfd.inner(x, u, v)
            assert ip.shape == (batch,)

        x = mfd.random_point(3, generator=_seeded_generator(506))
        grad = torch.ones_like(x)
        tangent = mfd.projection(x, grad)
        torch.testing.assert_close(
            mfd.inner(x, mfd.egrad_to_rgrad(x, grad), tangent),
            (grad * tangent).sum(-1),
            atol=1e-10,
            rtol=1e-10,
        )


    def test_exp_at_zero_returns_base(self):
        """exp_x(0) = x exactly (no drift)."""
        mfd = _make_manifold()
        x = mfd.random_point(batch_size=3, generator=_seeded_generator(30))
        zero = torch.zeros_like(x)
        out = mfd.exp(x, zero)
        torch.testing.assert_close(out, x, atol=1e-12, rtol=0)

    def test_log_at_same_point_is_zero(self):
        """log_x(x) = 0 exactly."""
        mfd = _make_manifold()
        x = mfd.random_point(batch_size=3, generator=_seeded_generator(31))
        v = mfd.log(x, x)
        torch.testing.assert_close(
            v, torch.zeros_like(v), atol=1e-9, rtol=0,
        )

    def test_exp_log_inverse(self):
        """log_x(exp_x(v)) = v for v ∈ T_x H."""
        mfd = _make_manifold()
        x = mfd.random_point(batch_size=3, generator=_seeded_generator(32))
        v = mfd.projection(
            x, torch.randn(3, mfd.n + 1, dtype=mfd.dtype,
                            generator=_seeded_generator(33)),
        ) * 0.1
        y = mfd.exp(x, v)
        v_recovered = mfd.log(x, y)
        torch.testing.assert_close(v_recovered, v, atol=1e-10, rtol=1e-10)

        for k in [-0.5, -2.0, -1.7]:
            mfd = _make_manifold(k=k)
            x = mfd.random_point(batch_size=3, generator=_seeded_generator(61))
            v = mfd.projection(
                x, torch.randn(3, mfd.n + 1, dtype=mfd.dtype,
                                generator=_seeded_generator(62)),
            ) * 0.1
            y = mfd.exp(x, v)
            torch.testing.assert_close(
                mfd.log(x, y), v, atol=1e-10, rtol=1e-10,
            )

    def test_log_exp_inverse(self):
        """exp_x(log_x(y)) = y for y on the manifold."""
        mfd = _make_manifold()
        x = mfd.random_point(batch_size=3, generator=_seeded_generator(34))
        y = mfd.random_point(batch_size=3, generator=_seeded_generator(35))
        v = mfd.log(x, y)
        y_recovered = mfd.exp(x, v)
        torch.testing.assert_close(y_recovered, y, atol=1e-10, rtol=1e-10)

        for batch in [0, 1, 4]:
            mfd = _make_manifold()
            x = mfd.random_point(batch_size=batch, generator=_seeded_generator(8))
            y = mfd.random_point(batch_size=batch, generator=_seeded_generator(9))
            v = mfd.log(x, y)
            assert v.shape == (batch, mfd.n + 1)

    def test_distance_symmetric(self):
        mfd = _make_manifold()
        x = mfd.random_point(batch_size=4, generator=_seeded_generator(36))
        y = mfd.random_point(batch_size=4, generator=_seeded_generator(37))
        torch.testing.assert_close(
            mfd.distance(x, y), mfd.distance(y, x), atol=1e-12, rtol=0,
        )

    def test_distance_zero_at_same_point(self):
        mfd = _make_manifold()
        x = mfd.random_point(batch_size=3, generator=_seeded_generator(38))
        d = mfd.distance(x, x)
        torch.testing.assert_close(
            d, torch.zeros_like(d), atol=1e-9, rtol=0,
        )

        mfd = _make_manifold()
        x = mfd.random_point(batch_size=3, generator=_seeded_generator(74))
        d = mfd.distance(x, x)
        # Roundoff in k·⟨x,x⟩_M may push z below 1; the clamp pulls it
        # to 1 and acosh(1) = 0. Asserting < 1e-9 keeps this honest.
        assert d.abs().max().item() < 1e-9

    def test_distance_matches_norm_of_log(self):
        """d(x, y) = ‖log_x(y)‖_x (geodesic length)."""
        mfd = _make_manifold()
        x = mfd.random_point(batch_size=3, generator=_seeded_generator(39))
        y = mfd.random_point(batch_size=3, generator=_seeded_generator(40))
        d = mfd.distance(x, y)
        v = mfd.log(x, y)
        n_v = mfd.norm(x, v)
        torch.testing.assert_close(d, n_v, atol=1e-10, rtol=1e-10)

        for batch in [0, 1, 4]:
            mfd = _make_manifold()
            x = mfd.random_point(batch_size=batch, generator=_seeded_generator(10))
            y = mfd.random_point(batch_size=batch, generator=_seeded_generator(11))
            d = mfd.distance(x, y)
            assert d.shape == (batch,)

        for k in [-0.5, -2.0, -1.7]:
            mfd = _make_manifold(k=k)
            x = mfd.random_point(batch_size=3, generator=_seeded_generator(63))
            y = mfd.random_point(batch_size=3, generator=_seeded_generator(64))
            d = mfd.distance(x, y)
            n_v = mfd.norm(x, mfd.log(x, y))
            torch.testing.assert_close(d, n_v, atol=1e-10, rtol=1e-10)

        mfd = LorentzManifold(3)
        a = torch.tensor([[.6, 0., .8]], dtype=torch.float64, requires_grad=True)
        b = torch.tensor([[.8, 0., .6]], dtype=torch.float64, requires_grad=True)
        def spatial_distance(x, y):
            return mfd.distance(
                torch.cat((torch.sqrt(1 + x.square().sum(-1, keepdim=True)), x), -1),
                torch.cat((torch.sqrt(1 + y.square().sum(-1, keepdim=True)), y), -1),
            )

        assert torch.autograd.gradcheck(spatial_distance, (a, b))
        large = (
            torch.randn(100, 3, dtype=torch.float32, generator=_seeded_generator(42))
            * 1e8
        )
        actual = spatial_distance(large, 2 * large)
        radius = torch.linalg.vector_norm(large.double(), dim=-1)
        expected = torch.asinh(2 * radius) - torch.asinh(radius)
        torch.testing.assert_close(actual.double(), expected, atol=2e-6, rtol=0)

    def test_distance_triangle_inequality(self):
        mfd = _make_manifold()
        x = mfd.random_point(batch_size=8, generator=_seeded_generator(41))
        y = mfd.random_point(batch_size=8, generator=_seeded_generator(42))
        z = mfd.random_point(batch_size=8, generator=_seeded_generator(43))
        d_xz = mfd.distance(x, z)
        d_xy = mfd.distance(x, y)
        d_yz = mfd.distance(y, z)
        # d(x, z) ≤ d(x, y) + d(y, z). Allow a tiny float-noise slack.
        assert (d_xz <= d_xy + d_yz + 1e-9).all()

    def test_parallel_transport_lands_in_tangent_at_y(self):
        mfd = _make_manifold()
        x = mfd.random_point(batch_size=4, generator=_seeded_generator(44))
        y = mfd.random_point(batch_size=4, generator=_seeded_generator(45))
        v = mfd.projection(
            x, torch.randn(4, mfd.n + 1, dtype=mfd.dtype,
                            generator=_seeded_generator(46)),
        )
        pt = mfd.parallel_transport(x, y, v)
        # ⟨y, pt⟩_M = 0
        ip = mfd.minkowski_inner(y, pt)
        torch.testing.assert_close(
            ip, torch.zeros_like(ip), atol=1e-10, rtol=0,
        )

        for batch in [0, 1, 4]:
            mfd = _make_manifold()
            x = mfd.random_point(batch_size=batch, generator=_seeded_generator(12))
            y = mfd.random_point(batch_size=batch, generator=_seeded_generator(13))
            v = mfd.projection(
                x, torch.randn(batch, mfd.n + 1, dtype=mfd.dtype,
                                generator=_seeded_generator(14)),
            )
            pt = mfd.parallel_transport(x, y, v)
            assert pt.shape == (batch, mfd.n + 1)

    def test_parallel_transport_preserves_inner_product(self):
        """⟨PT(u), PT(v)⟩_y = ⟨u, v⟩_x (parallel transport is an isometry)."""
        mfd = _make_manifold()
        x = mfd.random_point(batch_size=4, generator=_seeded_generator(47))
        y = mfd.random_point(batch_size=4, generator=_seeded_generator(48))
        u = mfd.projection(
            x, torch.randn(4, mfd.n + 1, dtype=mfd.dtype,
                            generator=_seeded_generator(49)),
        )
        v = mfd.projection(
            x, torch.randn(4, mfd.n + 1, dtype=mfd.dtype,
                            generator=_seeded_generator(50)),
        )
        ip_x = mfd.inner(x, u, v)
        pt_u = mfd.parallel_transport(x, y, u)
        pt_v = mfd.parallel_transport(x, y, v)
        ip_y = mfd.inner(y, pt_u, pt_v)
        torch.testing.assert_close(ip_y, ip_x, atol=1e-10, rtol=1e-10)

    def test_parallel_transport_identity_at_same_point(self):
        """PT_{x→x}(v) = v."""
        mfd = _make_manifold()
        x = mfd.random_point(batch_size=3, generator=_seeded_generator(51))
        v = mfd.projection(
            x, torch.randn(3, mfd.n + 1, dtype=mfd.dtype,
                            generator=_seeded_generator(52)),
        )
        pt = mfd.parallel_transport(x, x, v)
        torch.testing.assert_close(pt, v, atol=1e-10, rtol=0)

    def test_exp_0_matches_exp_at_origin(self):
        """exp_0(v) ≡ exp(origin, (0, v))."""
        mfd = _make_manifold()
        v_spatial = torch.randn(4, mfd.n, dtype=mfd.dtype,
                                generator=_seeded_generator(53)) * 0.2
        out_short = mfd.exp_0(v_spatial)
        v_ambient = torch.zeros(4, mfd.n + 1, dtype=mfd.dtype)
        v_ambient[..., 1:] = v_spatial
        out_long = mfd.exp(mfd.origin(batch_size=4), v_ambient)
        torch.testing.assert_close(
            out_short, out_long, atol=1e-12, rtol=1e-12,
        )

        for batch in [0, 1, 4]:
            mfd = _make_manifold()
            v = torch.randn(batch, mfd.n, dtype=mfd.dtype,
                            generator=_seeded_generator(15)) * 0.1
            x = mfd.exp_0(v)
            assert x.shape == (batch, mfd.n + 1)

        mfd = _make_manifold(n=3)
        x = mfd.origin(batch_size=2)
        v_spatial = torch.tensor([[5.0, 0.0, 0.0], [0.0, 3.0, 4.0]],
                                  dtype=torch.float64)
        out = mfd.exp_0(v_spatial)
        assert torch.isfinite(out).all()
        assert mfd.is_on_manifold(out).all()

    def test_log_0_matches_log_at_origin(self):
        """log_0(y) gives the spatial part of log(origin, y)."""
        mfd = _make_manifold()
        y = mfd.random_point(batch_size=4, generator=_seeded_generator(54))
        out_short = mfd.log_0(y)
        out_long = mfd.log(mfd.origin(batch_size=4), y)
        torch.testing.assert_close(
            out_short, out_long[..., 1:], atol=1e-12, rtol=1e-12,
        )

        for batch in [0, 1, 4]:
            mfd = _make_manifold()
            x = mfd.random_point(batch_size=batch, generator=_seeded_generator(16))
            v = mfd.log_0(x)
            assert v.shape == (batch, mfd.n)

        mfd = LorentzManifold(2, dtype=torch.float32)
        y = mfd.exp_0(torch.tensor([[20., 0.]], dtype=torch.float32))
        expected = torch.tensor([[0., 20., 0.]], dtype=torch.float32)
        torch.testing.assert_close(
            mfd.log(mfd.origin(), y), expected, atol=1e-5, rtol=0,
        )

    def test_exp_0_log_0_inverse(self):
        mfd = _make_manifold()
        v = torch.randn(3, mfd.n, dtype=mfd.dtype,
                        generator=_seeded_generator(55)) * 0.3
        y = mfd.exp_0(v)
        v_recovered = mfd.log_0(y)
        torch.testing.assert_close(v_recovered, v, atol=1e-10, rtol=1e-10)

        mfd = LorentzManifold(2, dtype=torch.float32)
        v = torch.tensor([[20., 0.]], dtype=torch.float32)
        y = mfd.exp_0(v)
        torch.testing.assert_close(mfd.log_0(y), v, atol=1e-5, rtol=0)
        torch.testing.assert_close(
            mfd.distance(mfd.origin(), y), torch.tensor([20.]), atol=1e-5, rtol=0,
        )


# --------------------------------------------------------------------
# Numerical robustness
# --------------------------------------------------------------------


class TestNumericalRobustness:
    """Operations must survive near-origin tangents and large tangents
    without producing NaN; chained exp/log must stay on the hyperboloid
    to machine precision."""


    def test_exp_output_on_hyperboloid_to_machine_precision(self):
        """The re-projection step should give post-exp constraint to ~eps."""
        mfd = _make_manifold(n=5)
        x = mfd.random_point(batch_size=4, generator=_seeded_generator(72))
        v = mfd.projection(
            x, torch.randn(4, mfd.n + 1, dtype=mfd.dtype,
                            generator=_seeded_generator(73)),
        ) * 0.5
        out = mfd.exp(x, v)
        constraint = mfd.minkowski_inner(out, out) - (1.0 / mfd.k)
        assert constraint.abs().max().item() < 1e-12

        for batch in [0, 1, 4]:
            mfd = _make_manifold()
            x = mfd.random_point(batch_size=batch, generator=_seeded_generator(6))
            v = mfd.projection(
                x, torch.randn(batch, mfd.n + 1, dtype=mfd.dtype,
                                generator=_seeded_generator(7)),
            ) * 0.1
            out = mfd.exp(x, v)
            assert out.shape == (batch, mfd.n + 1)

        mfd = _make_manifold()
        x = mfd.random_point(batch_size=4, generator=_seeded_generator(28))
        v = mfd.projection(
            x, torch.randn(4, mfd.n + 1, dtype=mfd.dtype,
                            generator=_seeded_generator(29)),
        ) * 0.1
        out = mfd.exp(x, v)
        assert mfd.is_on_manifold(out).all()

        mfd = _make_manifold()
        x = mfd.random_point(batch_size=2, generator=_seeded_generator(70))
        v_seed = torch.randn(2, mfd.n + 1, dtype=mfd.dtype,
                              generator=_seeded_generator(71))
        v = mfd.projection(x, v_seed) * 1e-12
        out = mfd.exp(x, v)
        assert torch.isfinite(out).all()
        assert mfd.is_on_manifold(out).all()


    def test_chained_exp_stays_on_manifold(self):
        """20 chained exp/proj cycles must not drift off the hyperboloid.

        We walk at exactly 0.1 *Riemannian* geodesic per step (not 0.1
        in ambient norm; that's unbounded since the ambient norm of a
        projected vector grows with the base point's magnitude). After
        20 steps we've traversed ~ 2 units of intrinsic distance, well
        within float64's range.
        """
        mfd = _make_manifold(n=5)
        x = mfd.random_point(batch_size=2, generator=_seeded_generator(75))
        for step in range(20):
            v_seed = torch.randn(2, mfd.n + 1, dtype=mfd.dtype,
                                  generator=_seeded_generator(76 + step))
            v_raw = mfd.projection(x, v_seed)
            v_norm = mfd.norm(x, v_raw).unsqueeze(-1).clamp(
                min=torch.finfo(v_raw.dtype).tiny,
            )
            v = v_raw / v_norm * 0.1
            x = mfd.exp(x, v)
        assert mfd.is_on_manifold(x).all()


# --------------------------------------------------------------------
# RiemannianSGD integration
# --------------------------------------------------------------------


class TestRiemannianSGD:
    """The optimizer module's RSGD should work transparently on
    LorentzManifold via the projection + retraction API. We use the
    same closed-form problem as the SPD test: `min ½ d(x, target)²`,
    where one step of unit-rate RSGD lands exactly on the target."""

    def test_one_step_lr_one_lands_on_target(self):
        mfd = _make_manifold()
        x = mfd.random_point(batch_size=2, generator=_seeded_generator(100))
        target = mfd.random_point(batch_size=2, generator=_seeded_generator(101))
        # The metric dual of -log_x(target) flips its temporal component.
        opt = RiemannianSGD(mfd, lr=1.0)
        rgrad = -mfd.log(x, target)
        grad = torch.cat((-rgrad[..., :1], rgrad[..., 1:]), dim=-1)
        x_next = opt.step(x, grad)
        torch.testing.assert_close(x_next, target, atol=1e-8, rtol=0)


    def test_converges_to_target(self):
        mfd = _make_manifold()
        x = mfd.random_point(batch_size=1, generator=_seeded_generator(104))
        target = mfd.random_point(batch_size=1, generator=_seeded_generator(105))
        opt = RiemannianSGD(mfd, lr=0.5)
        for _ in range(50):
            rgrad = -mfd.log(x, target)
            grad = torch.cat((-rgrad[..., :1], rgrad[..., 1:]), dim=-1)
            x = opt.step(x, grad)
        dist = mfd.distance(x, target)
        assert dist[0].item() < 1e-9, (
            f"failed to converge to target; final distance {dist[0].item():.3e}"
        )

        mfd = _make_manifold()
        x = mfd.random_point(batch_size=3, generator=_seeded_generator(102))
        target = mfd.random_point(batch_size=3, generator=_seeded_generator(103))
        opt = RiemannianSGD(mfd, lr=0.3)
        for _ in range(20):
            rgrad = -mfd.log(x, target)
            grad = torch.cat((-rgrad[..., :1], rgrad[..., 1:]), dim=-1)
            x = opt.step(x, grad)
            assert mfd.is_on_manifold(x).all()

    def test_batch_zero(self):
        mfd = _make_manifold(n=3)
        x = mfd.random_point(batch_size=0, generator=_seeded_generator(106))
        assert x.shape == (0, 4)
        g = torch.zeros(0, 4, dtype=torch.float64)
        opt = RiemannianSGD(mfd, lr=0.1)
        x_next = opt.step(x, g)
        assert x_next.shape == (0, 4)


# --------------------------------------------------------------------
# Comparison against geoopt's Lorentz
# --------------------------------------------------------------------


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
    """Cross-check distance / exp / log against geoopt's `Lorentz`.

    Curvature-convention mapping. Both libraries use the mostly-plus
    signature ⟨x, y⟩_M = −x_0 y_0 + Σ x_i y_i. The difference is the
    parameter:

      - **geoopt** `Lorentz(k=K)` (with `K > 0`) describes the
        hyperboloid `⟨x, x⟩_M = −K`, i.e. radius √K, sectional
        curvature `−1/K`.
      - **Ours** `LorentzManifold(k=k_ours)` (with `k_ours < 0`)
        describes the hyperboloid `⟨x, x⟩_M = 1/k_ours`, i.e. radius
        √(−1/k_ours), sectional curvature `k_ours` directly.

    Mapping: `K = −1/k_ours` (so `k_ours = −1` ↔ `K = 1`,
    `k_ours = −2` ↔ `K = 0.5`, `k_ours = −0.5` ↔ `K = 2`).
    """

    @staticmethod
    def _geoopt_mfd(k_ours: float):
        from geoopt import Lorentz
        return Lorentz(k=-1.0 / k_ours)

    def _seed_pair_aligned(self, mfd, n_batch, seeds):
        """Both libraries take the same input tensors so the cross-check is
        on the operation, not the random-draw distribution."""
        x = mfd.random_point(batch_size=n_batch,
                              generator=_seeded_generator(seeds[0]))
        y = mfd.random_point(batch_size=n_batch,
                              generator=_seeded_generator(seeds[1]))
        return x, y


    def test_distance_matches_geoopt(self):
        mfd = _make_manifold(k=-1.0)
        gmfd = self._geoopt_mfd(-1.0)
        x, y = self._seed_pair_aligned(mfd, 4, seeds=(81, 82))
        d_ours = mfd.distance(x, y)
        d_geoopt = gmfd.dist(x, y)
        torch.testing.assert_close(d_ours, d_geoopt, atol=1e-10, rtol=1e-10)

        mfd = _make_manifold(k=-2.0)
        gmfd = self._geoopt_mfd(-2.0)
        x, y = self._seed_pair_aligned(mfd, 4, seeds=(87, 88))
        d_ours = mfd.distance(x, y)
        d_geoopt = gmfd.dist(x, y)
        torch.testing.assert_close(d_ours, d_geoopt, atol=1e-6, rtol=1e-6)

    def test_exp_matches_geoopt(self):
        mfd = _make_manifold(k=-1.0)
        gmfd = self._geoopt_mfd(-1.0)
        x = mfd.random_point(batch_size=3, generator=_seeded_generator(83))
        v = mfd.projection(
            x, torch.randn(3, mfd.n + 1, dtype=mfd.dtype,
                            generator=_seeded_generator(84)),
        ) * 0.1
        y_ours = mfd.exp(x, v)
        y_geoopt = gmfd.expmap(x, v)
        torch.testing.assert_close(y_ours, y_geoopt, atol=1e-10, rtol=1e-10)

    def test_log_matches_geoopt(self):
        mfd = _make_manifold(k=-1.0)
        gmfd = self._geoopt_mfd(-1.0)
        x, y = self._seed_pair_aligned(mfd, 3, seeds=(85, 86))
        v_ours = mfd.log(x, y)
        v_geoopt = gmfd.logmap(x, y)
        torch.testing.assert_close(v_ours, v_geoopt, atol=1e-10, rtol=1e-10)


# --------------------------------------------------------------------
# Autograd-finite: gradients must be finite at boundary inputs.
# --------------------------------------------------------------------
#
# The classic PyTorch gotcha: `torch.where(cond, formula, default)`
# evaluates BOTH branches in backward, even when one is masked. If the
# masked-out branch contains an op that produces NaN at the boundary
# (e.g. sqrt(0) → grad ∞, then 0·∞ = NaN through a clamp), the NaN
# propagates into v.grad even when forward is finite.
#
# These tests pin down that exp / log / exp_0 / log_0 / distance /
# norm / parallel_transport all produce finite gradients at the
# boundary inputs that arise in real training loops: tangent-at-origin
# embeddings where some v rows are tiny or zero, target points equal
# to the source, etc.


class TestAutogradFinite:
    """Backward through every manifold primitive must produce finite
    gradients at the boundary inputs (v=0 tangent, x=y endpoints,
    distance(x,x)=0 etc.) that arise in tangent-at-origin training."""


    def test_exp_0_backward_with_exact_zero_row(self):
        """exp_0(v) backward is finite even when one row of v is
        exactly zero (origin in tangent space)."""
        mfd = LorentzManifold(n=4, k=-1.0)
        v = torch.zeros(3, 4, dtype=torch.float64, requires_grad=True)
        with torch.no_grad():
            v[1] = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64)
            v[2] = torch.tensor([0.5, -0.5, 0.5, -0.5], dtype=torch.float64)
        v.requires_grad_(True)
        T = mfd.exp_0(v)
        T.sum().backward()
        assert torch.isfinite(v.grad).all(), (
            f"v.grad NaN count: {torch.isnan(v.grad).sum().item()}"
        )

        mfd = LorentzManifold(n=5, k=-1.0)
        v = (torch.randn(10, 5, dtype=torch.float64,
                         generator=_seeded_generator(200)) * 0.1)
        v.requires_grad_(True)
        T = mfd.exp_0(v)
        T.sum().backward()
        assert torch.isfinite(v.grad).all(), (
            f"v.grad NaN count: {torch.isnan(v.grad).sum().item()}, "
            f"inf count: {torch.isinf(v.grad).sum().item()}"
        )


    def test_log_backward_at_same_point(self):
        """log(x, x) backward is finite; the zero-tangent boundary."""
        mfd = LorentzManifold(n=5, k=-1.0)
        v = (torch.randn(5, 5, dtype=torch.float64,
                         generator=_seeded_generator(204)) * 0.3)
        v.requires_grad_(True)
        T = mfd.exp_0(v)
        out = mfd.log(T, T)
        out.sum().backward()
        assert torch.isfinite(v.grad).all(), (
            f"v.grad NaN count: {torch.isnan(v.grad).sum().item()}"
        )


    def test_exp_backward_at_zero_tangent(self):
        """exp_x(0) backward is finite; sinhc analytic-limit branch."""
        mfd = LorentzManifold(n=5, k=-1.0)
        v_init = (torch.randn(5, 5, dtype=torch.float64,
                              generator=_seeded_generator(205)) * 0.3)
        v_init.requires_grad_(True)
        x = mfd.exp_0(v_init)
        # Tangent vector that's exactly zero
        zero_tangent = torch.zeros_like(x)
        out = mfd.exp(x, zero_tangent)
        out.sum().backward()
        assert torch.isfinite(v_init.grad).all(), (
            f"v_init.grad NaN count: {torch.isnan(v_init.grad).sum().item()}"
        )

    def test_norm_backward_at_zero(self):
        """norm(x, 0) backward is finite; clamp(min=0)+sqrt boundary.

        The Lorentz induced metric is point-independent in ambient
        form, so we put `requires_grad` on the tangent (not the base
        point) to trace the chain that reaches `sqrt(<v,v>_M)`; the
        path that would NaN without the `_safe_sqrt` fix.
        """
        mfd = LorentzManifold(n=5, k=-1.0)
        x = mfd.origin(batch_size=5)
        v = torch.zeros(5, mfd.n + 1, dtype=torch.float64,
                        requires_grad=True)
        n = mfd.norm(x, v)
        n.sum().backward()
        assert torch.isfinite(v.grad).all(), (
            f"v.grad NaN count: {torch.isnan(v.grad).sum().item()}"
        )

    def test_parallel_transport_backward_at_same_point(self):
        """parallel_transport(x, x, v) = v; backward must be finite."""
        mfd = LorentzManifold(n=5, k=-1.0)
        v_init = (torch.randn(4, 5, dtype=torch.float64,
                              generator=_seeded_generator(207)) * 0.3)
        v_init.requires_grad_(True)
        x = mfd.exp_0(v_init)
        # Use a finite tangent (projected ambient noise)
        v_seed = torch.randn(4, mfd.n + 1, dtype=torch.float64,
                              generator=_seeded_generator(208)) * 0.1
        v = mfd.projection(x, v_seed)
        pt = mfd.parallel_transport(x, x, v)
        pt.sum().backward()
        assert torch.isfinite(v_init.grad).all()

    def test_full_substrate_chain(self):
        """End-to-end: tangent-at-origin parameterization → exp_0 →
        all-pairs distance with NLL-like loss → backward. This is the
        chain reported in the substrate-training bug repro."""
        mfd = LorentzManifold(n=8, k=-1.0)
        v = (torch.randn(16, 8, dtype=torch.float64,
                         generator=_seeded_generator(209)) * 0.4)
        v.requires_grad_(True)
        T = mfd.exp_0(v)
        # NLL-style loss: for each anchor, target ~ random other index,
        # noise samples ~ all others. The partition function over
        # ALL distances (including d(x, x) = 0) is the standard form.
        N = T.shape[0]
        Ti = T.unsqueeze(1).expand(N, N, -1).reshape(N * N, -1)
        Tj = T.unsqueeze(0).expand(N, N, -1).reshape(N * N, -1)
        d_all = mfd.distance(Ti, Tj).reshape(N, N)
        log_partition = torch.logsumexp(-d_all, dim=-1)  # (N,)
        # Pretend target is offset-1 cyclic permutation
        target_idx = (torch.arange(N) + 1) % N
        target_d = d_all[torch.arange(N), target_idx]
        nll = (target_d + log_partition).sum()
        nll.backward()
        assert torch.isfinite(v.grad).all(), (
            f"v.grad NaN count: {torch.isnan(v.grad).sum().item()}/"
            f"{v.grad.numel()}; inf count: "
            f"{torch.isinf(v.grad).sum().item()}"
        )

        mfd = LorentzManifold(n=5, k=-1.0)
        v = (torch.randn(10, 5, dtype=torch.float64,
                         generator=_seeded_generator(201)) * 0.3)
        v.requires_grad_(True)
        T = mfd.exp_0(v)
        d = mfd.distance(T[:5], T[5:])
        d.sum().backward()
        assert torch.isfinite(v.grad).all(), (
            f"v.grad NaN count: {torch.isnan(v.grad).sum().item()}"
        )

        mfd = LorentzManifold(n=5, k=-1.0)
        v = (torch.randn(5, 5, dtype=torch.float64,
                         generator=_seeded_generator(202)) * 0.3)
        v.requires_grad_(True)
        T = mfd.exp_0(v)
        # distance(T, T) — every distance is identically zero
        d = mfd.distance(T, T)
        d.sum().backward()
        assert torch.isfinite(v.grad).all(), (
            f"v.grad NaN count: {torch.isnan(v.grad).sum().item()}/"
            f"{v.grad.numel()}"
        )

        mfd = LorentzManifold(n=5, k=-1.0)
        v = (torch.randn(8, 5, dtype=torch.float64,
                         generator=_seeded_generator(203)) * 0.3)
        v.requires_grad_(True)
        T = mfd.exp_0(v)
        # All-pairs distance including diagonal — the diagonal is 0.
        N = T.shape[0]
        Ti = T.unsqueeze(1).expand(N, N, T.shape[-1]).reshape(N * N, -1)
        Tj = T.unsqueeze(0).expand(N, N, T.shape[-1]).reshape(N * N, -1)
        d = mfd.distance(Ti, Tj).reshape(N, N)
        # Mock NLL-style aggregation: softmax over distances per row
        weights = torch.softmax(-d, dim=-1)
        loss = (weights * d).sum()
        loss.backward()
        assert torch.isfinite(v.grad).all(), (
            f"v.grad NaN count: {torch.isnan(v.grad).sum().item()}/"
            f"{v.grad.numel()}"
        )


# --------------------------------------------------------------------
# Autograd stress tests — aggressive chains beyond the boundary
# tests, modeling real training-loop patterns. Added after the
# substrate team reported that the boundary fixes weren't enough for
# their use case — these confirm the manifold primitives are also
# clean under deeper composition.
# --------------------------------------------------------------------


@pytest.mark.parametrize("k", [-1.0, -0.5, -2.0])
class TestAutogradStress:
    def test_long_chained_exp_backward(self, k):
        """v → exp_0 → [exp, exp, ...] 10× → distance.sum.backward must
        produce finite gradient regardless of k."""
        mfd = LorentzManifold(n=5, k=k)
        v = (torch.randn(3, 5, dtype=torch.float64,
                          generator=_seeded_generator(300 + int(-k * 10))) * 0.3)
        v.requires_grad_(True)
        x = mfd.exp_0(v)
        for step in range(10):
            v_step_seed = torch.zeros(3, mfd.n + 1, dtype=torch.float64)
            v_step_seed[..., 1:] = (torch.randn(
                3, 5, dtype=torch.float64,
                generator=_seeded_generator(310 + step),
            ) * 0.01)
            v_step = mfd.projection(x, v_step_seed)
            x = mfd.exp(x, v_step)
        loss = mfd.distance(x, x[0:1].expand_as(x)).sum()
        loss.backward()
        assert torch.isfinite(v.grad).all()

    def test_parallel_transport_near_identity_backward(self, k):
        """parallel_transport at x ≈ y (distance ≈ 1e-8); autograd
        must not blow up despite the near-singular log."""
        mfd = LorentzManifold(n=5, k=k)
        v = (torch.randn(3, 5, dtype=torch.float64,
                          generator=_seeded_generator(320)) * 0.3)
        v.requires_grad_(True)
        x = mfd.exp_0(v)
        eps_tan = torch.zeros(3, mfd.n + 1, dtype=torch.float64)
        eps_tan[..., 1:] = (torch.randn(3, 5, dtype=torch.float64,
                                          generator=_seeded_generator(321))
                            * 1e-8)
        eps_tan = mfd.projection(x, eps_tan)
        y_close = mfd.exp(x, eps_tan)
        v_test = mfd.projection(
            x, torch.randn(3, mfd.n + 1, dtype=torch.float64,
                            generator=_seeded_generator(322)),
        )
        pt = mfd.parallel_transport(x, y_close, v_test)
        pt.sum().backward()
        assert torch.isfinite(v.grad).all()

    def test_log_0_exp_0_round_trip_backward(self, k):
        """log_0(exp_0(0)) = 0 chain; both ends at origin, gradient
        must remain finite (no 0/0 collapse)."""
        mfd = LorentzManifold(n=5, k=k)
        v = torch.zeros(3, 5, dtype=torch.float64, requires_grad=True)
        out = mfd.log_0(mfd.exp_0(v))
        out.sum().backward()
        assert torch.isfinite(v.grad).all()

        mfd = LorentzManifold(n=4, k=-1.0)
        # T starts at origin via exp_0 of zero tangent
        v = torch.zeros(3, 4, dtype=torch.float64, requires_grad=True)
        T = mfd.exp_0(v)
        out = mfd.log_0(T)
        out.sum().backward()
        assert torch.isfinite(v.grad).all(), (
            f"v.grad NaN count: {torch.isnan(v.grad).sum().item()}"
        )
