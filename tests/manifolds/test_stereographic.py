# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Synoros

"""Tests for holonomy_lib.manifolds.stereographic.KappaStereographicManifold.

Six layers, parametrized across the three κ branches (spherical,
Euclidean, hyperbolic):
  1. Construction validation.
  2. Shapes across B ∈ {0, 1, several}.
  3. Property tests — exp/log inverse, distance symmetry, Möbius
     additive inverse, Euclidean recovery at κ = 0.
  4. Autograd-finite tests at boundary inputs.
  6. Cross-comparison against geoopt's Stereographic (when installed).
"""

from __future__ import annotations

import pytest
import torch

from holonomy_lib.manifolds import KappaStereographicManifold

# Test the three κ branches plus a couple of magnitudes
KAPPA_VALUES = [-1.0, -0.5, 0.0, 0.5, 1.0]


def _make_manifold(n=3, kappa=-1.0, dtype=torch.float64):
    return KappaStereographicManifold(n=n, kappa=kappa, dtype=dtype)


def _seed(s: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(s)
    return g


# --------------------------------------------------------------------
# Construction
# --------------------------------------------------------------------


class TestConstruction:
    def test_rejects_n_zero(self):
        with pytest.raises(ValueError, match="n"):
            KappaStereographicManifold(n=0)

    def test_rejects_n_negative(self):
        with pytest.raises(ValueError, match="n"):
            KappaStereographicManifold(n=-2)

    def test_rejects_non_scalar_kappa(self):
        """Multi-element Tensor kappa rejected (must be 0-dim scalar)."""
        with pytest.raises(TypeError, match="0-dim"):
            KappaStereographicManifold(
                n=3, kappa=torch.tensor([0.5, 1.0]),
            )

    def test_accepts_scalar_tensor_kappa(self):
        """0-dim Tensor kappa is accepted for learnable-κ workflows."""
        mfd = KappaStereographicManifold(
            n=3, kappa=torch.tensor(-1.0, dtype=torch.float64),
        )
        # Basic ops should still work
        x = mfd.random_point(batch_size=2, generator=_seed(80))
        y = mfd.random_point(batch_size=2, generator=_seed(81))
        d = mfd.distance(x, y)
        assert torch.isfinite(d).all()

    @pytest.mark.parametrize("k", KAPPA_VALUES)
    def test_branch_dispatch(self, k):
        mfd = _make_manifold(kappa=k)
        if k > 0:
            assert mfd._branch == "spherical"
        elif k < 0:
            assert mfd._branch == "hyperbolic"
        else:
            assert mfd._branch == "euclidean"

    def test_dim_and_ambient_dim_equal(self):
        mfd = _make_manifold(n=5)
        assert mfd.dim == 5
        assert mfd.ambient_dim == 5


# --------------------------------------------------------------------
# Shapes across batch sizes
# --------------------------------------------------------------------


@pytest.mark.parametrize("batch", [0, 1, 4])
@pytest.mark.parametrize("k", KAPPA_VALUES)
class TestShapes:
    def test_random_point(self, batch, k):
        mfd = _make_manifold(kappa=k)
        x = mfd.random_point(batch_size=batch, generator=_seed(0))
        assert x.shape == (batch, mfd.n)

    def test_origin(self, batch, k):
        mfd = _make_manifold(kappa=k)
        o = mfd.origin(batch_size=batch)
        assert o.shape == (batch, mfd.n)

    def test_exp_log(self, batch, k):
        mfd = _make_manifold(kappa=k)
        x = mfd.random_point(batch_size=batch, generator=_seed(1))
        v = torch.randn(batch, mfd.n, dtype=mfd.dtype,
                        generator=_seed(2)) * 0.05
        y = mfd.exp(x, v)
        out = mfd.log(x, y)
        assert y.shape == (batch, mfd.n)
        assert out.shape == (batch, mfd.n)

    def test_distance(self, batch, k):
        mfd = _make_manifold(kappa=k)
        x = mfd.random_point(batch_size=batch, generator=_seed(3))
        y = mfd.random_point(batch_size=batch, generator=_seed(4))
        d = mfd.distance(x, y)
        assert d.shape == (batch,)


# --------------------------------------------------------------------
# Properties — per-κ-branch invariants
# --------------------------------------------------------------------


@pytest.mark.parametrize("k", KAPPA_VALUES)
class TestProperties:
    def test_random_point_in_domain(self, k):
        mfd = _make_manifold(kappa=k)
        x = mfd.random_point(batch_size=8, generator=_seed(10))
        assert mfd.is_on_manifold(x).all()

    def test_origin_on_manifold(self, k):
        mfd = _make_manifold(kappa=k)
        o = mfd.origin(batch_size=3)
        assert mfd.is_on_manifold(o).all()

    def test_mobius_add_neutral_at_origin(self, k):
        """0 ⊕_κ y = y and x ⊕_κ 0 = x."""
        mfd = _make_manifold(kappa=k)
        o = mfd.origin(batch_size=3)
        y = mfd.random_point(batch_size=3, generator=_seed(11))
        torch.testing.assert_close(
            mfd.mobius_add(o, y), y, atol=1e-12, rtol=1e-12,
        )
        torch.testing.assert_close(
            mfd.mobius_add(y, o), y, atol=1e-12, rtol=1e-12,
        )

    def test_mobius_add_inverse(self, k):
        """(-x) ⊕_κ x = 0 (Möbius additive inverse)."""
        mfd = _make_manifold(kappa=k)
        x = mfd.random_point(batch_size=3, generator=_seed(12))
        result = mfd.mobius_add(-x, x)
        torch.testing.assert_close(
            result, torch.zeros_like(result), atol=1e-10, rtol=0,
        )

    def test_exp_log_inverse(self, k):
        """log_x(exp_x(v)) = v for small v."""
        mfd = _make_manifold(kappa=k)
        x = mfd.random_point(batch_size=3, generator=_seed(13))
        v = torch.randn(3, mfd.n, dtype=mfd.dtype,
                        generator=_seed(14)) * 0.05
        v_back = mfd.log(x, mfd.exp(x, v))
        torch.testing.assert_close(v_back, v, atol=1e-9, rtol=1e-9)

    def test_log_exp_inverse(self, k):
        """exp_x(log_x(y)) = y."""
        mfd = _make_manifold(kappa=k)
        x = mfd.random_point(batch_size=3, generator=_seed(15))
        y = mfd.random_point(batch_size=3, generator=_seed(16))
        y_back = mfd.exp(x, mfd.log(x, y))
        torch.testing.assert_close(y_back, y, atol=1e-9, rtol=1e-9)

    def test_exp_0_log_0_inverse(self, k):
        """log_0(exp_0(v)) = v."""
        mfd = _make_manifold(kappa=k)
        v = torch.randn(4, mfd.n, dtype=mfd.dtype,
                        generator=_seed(17)) * 0.1
        v_back = mfd.log_0(mfd.exp_0(v))
        torch.testing.assert_close(v_back, v, atol=1e-9, rtol=1e-9)

    def test_distance_symmetric(self, k):
        mfd = _make_manifold(kappa=k)
        x = mfd.random_point(batch_size=4, generator=_seed(18))
        y = mfd.random_point(batch_size=4, generator=_seed(19))
        torch.testing.assert_close(
            mfd.distance(x, y), mfd.distance(y, x),
            atol=1e-10, rtol=0,
        )

    def test_distance_at_same_point(self, k):
        """d(x, x) = 0 to machine precision."""
        mfd = _make_manifold(kappa=k)
        x = mfd.random_point(batch_size=3, generator=_seed(20))
        d = mfd.distance(x, x)
        assert d.abs().max().item() < 1e-9


# --------------------------------------------------------------------
# κ = 0 Euclidean recovery (separately verified — the limit is the
# Bachmann-normalization Euclidean form, where d_κ=0 = 2·‖y - x‖).
# --------------------------------------------------------------------


class TestEuclideanRecovery:
    """At κ = 0 the manifold IS Euclidean R^n — but with the conformal-
    factor-2 normalization Bachmann uses, distances are 2× the standard."""

    def test_exp_is_identity(self):
        mfd = _make_manifold(kappa=0.0)
        x = mfd.random_point(batch_size=3, generator=_seed(30))
        v = torch.randn(3, mfd.n, dtype=mfd.dtype,
                        generator=_seed(31)) * 0.5
        out = mfd.exp(x, v)
        torch.testing.assert_close(out, x + v, atol=1e-12, rtol=0)

    def test_log_is_difference_over_lambda(self):
        """log_x(y) = (2/λ_κ(x)) · (y - x). At κ=0 λ_κ ≡ 2, so log_x(y)
        = y - x."""
        mfd = _make_manifold(kappa=0.0)
        x = mfd.random_point(batch_size=3, generator=_seed(32))
        y = mfd.random_point(batch_size=3, generator=_seed(33))
        out = mfd.log(x, y)
        torch.testing.assert_close(out, y - x, atol=1e-12, rtol=0)

    def test_distance_is_twice_euclidean(self):
        """d_{κ=0}(x, y) = 2 · ‖y - x‖_2 (Bachmann normalization)."""
        mfd = _make_manifold(kappa=0.0)
        x = mfd.random_point(batch_size=4, generator=_seed(34))
        y = mfd.random_point(batch_size=4, generator=_seed(35))
        d = mfd.distance(x, y)
        expected = 2.0 * torch.linalg.vector_norm(y - x, dim=-1)
        torch.testing.assert_close(d, expected, atol=1e-10, rtol=1e-10)


# --------------------------------------------------------------------
# Autograd-finite at boundary inputs
# --------------------------------------------------------------------


class TestAutogradFinite:
    """Same boundary cases as Lorentz: x = y, v = 0, y = origin must
    all produce finite gradients in backward."""

    @pytest.mark.parametrize("k", KAPPA_VALUES)
    def test_distance_at_same_point_backward(self, k):
        mfd = _make_manifold(kappa=k)
        v = (torch.randn(5, mfd.n, dtype=torch.float64,
                          generator=_seed(40)) * 0.1)
        v.requires_grad_(True)
        x = mfd.exp_0(v)
        d = mfd.distance(x, x)
        d.sum().backward()
        assert torch.isfinite(v.grad).all(), (
            f"κ={k}: v.grad NaN: {torch.isnan(v.grad).sum().item()}/"
            f"{v.grad.numel()}"
        )

    @pytest.mark.parametrize("k", KAPPA_VALUES)
    def test_exp_0_backward_at_zero(self, k):
        mfd = _make_manifold(kappa=k)
        v = torch.zeros(3, mfd.n, dtype=torch.float64, requires_grad=True)
        x = mfd.exp_0(v)
        x.sum().backward()
        assert torch.isfinite(v.grad).all()

    @pytest.mark.parametrize("k", KAPPA_VALUES)
    def test_log_0_backward_at_origin(self, k):
        mfd = _make_manifold(kappa=k)
        v = torch.zeros(3, mfd.n, dtype=torch.float64, requires_grad=True)
        y = mfd.exp_0(v)  # all rows = origin
        out = mfd.log_0(y)
        out.sum().backward()
        assert torch.isfinite(v.grad).all()

    @pytest.mark.parametrize("k", KAPPA_VALUES)
    def test_full_nll_chain_backward(self, k):
        """Substrate-style chain: exp_0 → all-pairs distance → NLL.
        Includes d(x_i, x_i) = 0 self-pairs in the partition function."""
        mfd = _make_manifold(kappa=k, n=4)
        v = (torch.randn(8, 4, dtype=torch.float64,
                          generator=_seed(41)) * 0.2)
        v.requires_grad_(True)
        x = mfd.exp_0(v)
        N = x.shape[0]
        Xi = x.unsqueeze(1).expand(N, N, mfd.n).reshape(-1, mfd.n)
        Xj = x.unsqueeze(0).expand(N, N, mfd.n).reshape(-1, mfd.n)
        d = mfd.distance(Xi, Xj).reshape(N, N)
        log_partition = torch.logsumexp(-d, dim=-1)
        loss = log_partition.sum()
        loss.backward()
        assert torch.isfinite(v.grad).all(), (
            f"κ={k}: NaN {torch.isnan(v.grad).sum().item()}"
        )


# --------------------------------------------------------------------
# Comparison against geoopt's Stereographic (when installed)
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
    """Cross-check distance / exp_0 / log_0 against geoopt's
    `Stereographic`. Convention mapping is direct: both libraries
    use the same κ-stereographic formulas (Bachmann et al. 2020) and
    the same parameter sign (κ > 0 spherical, κ < 0 hyperbolic).
    """

    @staticmethod
    def _geoopt_mfd(k: float):
        from geoopt import Stereographic
        return Stereographic(k=k)

    @pytest.mark.parametrize("k", [-1.0, -0.5, 0.5, 1.0])
    def test_distance_matches_geoopt(self, k):
        mfd = _make_manifold(kappa=k)
        gmfd = self._geoopt_mfd(k)
        x = mfd.random_point(batch_size=4, generator=_seed(70))
        y = mfd.random_point(batch_size=4, generator=_seed(71))
        d_ours = mfd.distance(x, y)
        d_geoopt = gmfd.dist(x, y)
        torch.testing.assert_close(d_ours, d_geoopt, atol=1e-7, rtol=1e-7)

    @pytest.mark.parametrize("k", [-1.0, -0.5, 0.5, 1.0])
    def test_exp_0_matches_geoopt(self, k):
        mfd = _make_manifold(kappa=k)
        gmfd = self._geoopt_mfd(k)
        v = torch.randn(3, mfd.n, dtype=mfd.dtype,
                        generator=_seed(72)) * 0.1
        out_ours = mfd.exp_0(v)
        out_geoopt = gmfd.expmap0(v)
        torch.testing.assert_close(out_ours, out_geoopt,
                                    atol=1e-7, rtol=1e-7)

    @pytest.mark.parametrize("k", [-1.0, -0.5, 0.5, 1.0])
    def test_log_0_matches_geoopt(self, k):
        mfd = _make_manifold(kappa=k)
        gmfd = self._geoopt_mfd(k)
        # Generate y inside the domain via geoopt's own exp
        v = torch.randn(3, mfd.n, dtype=mfd.dtype,
                        generator=_seed(73)) * 0.1
        y = gmfd.expmap0(v)
        out_ours = mfd.log_0(y)
        out_geoopt = gmfd.logmap0(y)
        torch.testing.assert_close(out_ours, out_geoopt,
                                    atol=1e-7, rtol=1e-7)


# --------------------------------------------------------------------
# Autograd stress — chained operations beyond simple boundary cases.
# Same suite as `LorentzManifold` to confirm the κ-stereographic
# branches are equally autograd-clean.
# --------------------------------------------------------------------


@pytest.mark.parametrize("k", KAPPA_VALUES)
class TestAutogradStress:
    def test_long_chained_exp_backward(self, k):
        mfd = _make_manifold(kappa=k, n=4)
        v = (torch.randn(3, 4, dtype=torch.float64,
                          generator=_seed(100)) * 0.2)
        v.requires_grad_(True)
        x = mfd.exp_0(v)
        for step in range(5):
            v_step = mfd.projection(
                x, torch.randn(3, 4, dtype=torch.float64,
                                generator=_seed(110 + step)) * 0.01,
            )
            x = mfd.exp(x, v_step)
        loss = mfd.distance(x, x[0:1].expand_as(x)).sum()
        loss.backward()
        assert torch.isfinite(v.grad).all()

    def test_mobius_inverse_backward(self, k):
        """mobius_add(x, -x) = 0; backward must be finite."""
        mfd = _make_manifold(kappa=k, n=4)
        v = (torch.randn(3, 4, dtype=torch.float64,
                          generator=_seed(120)) * 0.2)
        v.requires_grad_(True)
        x = mfd.exp_0(v)
        out = mfd.mobius_add(x, -x)
        out.sum().backward()
        assert torch.isfinite(v.grad).all()

    def test_exp_log_round_trip_backward(self, k):
        """exp_x(log_x(y)) = y; backward of the round-trip identity."""
        mfd = _make_manifold(kappa=k, n=4)
        v = (torch.randn(3, 4, dtype=torch.float64,
                          generator=_seed(121)) * 0.2)
        v.requires_grad_(True)
        x = mfd.exp_0(v)
        y = mfd.exp_0(torch.randn(3, 4, dtype=torch.float64,
                                    generator=_seed(122)) * 0.2)
        out = mfd.exp(x, mfd.log(x, y))
        loss = (out - y).pow(2).sum()
        loss.backward()
        assert torch.isfinite(v.grad).all()


# --------------------------------------------------------------------
# Learnable κ — gradient flows back to a κ Parameter via distance
# --------------------------------------------------------------------


class TestLearnableKappa:
    """Allow κ to be a 0-dim torch.Tensor (e.g. `nn.Parameter`); the
    gradient of distance-based losses should flow back to κ so SGD
    can learn the curvature magnitude from data."""

    def test_kappa_tensor_gradient_through_distance(self):
        """With κ as a Parameter, `distance(x, y).sum().backward()`
        should give a non-trivial gradient on κ."""
        kappa = torch.nn.Parameter(torch.tensor(-1.0, dtype=torch.float64))
        mfd = KappaStereographicManifold(n=3, kappa=kappa,
                                          dtype=torch.float64)
        x = mfd.random_point(batch_size=4, generator=_seed(100))
        y = mfd.random_point(batch_size=4, generator=_seed(101))
        d = mfd.distance(x, y)
        d.sum().backward()
        assert kappa.grad is not None
        assert torch.isfinite(kappa.grad).all()
        # The gradient should be non-zero (distance genuinely depends
        # on κ for hyperbolic Poincaré ball).
        assert kappa.grad.abs().item() > 0

    def test_kappa_sgd_step_reduces_loss(self):
        """End-to-end: an SGD step on κ should reduce a distance-based
        loss (specifically: pull κ toward a curvature that minimizes
        the loss). Smoke test for the learnability of κ."""
        # Initialize at κ=-0.5, target distance behavior at κ=-1.0
        kappa = torch.nn.Parameter(torch.tensor(-0.5, dtype=torch.float64))
        mfd_ref = KappaStereographicManifold(
            n=3, kappa=-1.0, dtype=torch.float64,
        )
        # Generate a target set of distances at κ=-1
        x = mfd_ref.random_point(batch_size=10, generator=_seed(110))
        y = mfd_ref.random_point(batch_size=10, generator=_seed(111))
        target_d = mfd_ref.distance(x, y).detach()
        # Same x, y interpreted at the *learnable* κ
        mfd_learn = KappaStereographicManifold(
            n=3, kappa=kappa, dtype=torch.float64,
        )
        optimizer = torch.optim.SGD([kappa], lr=0.1)
        loss_history = []
        for _ in range(30):
            optimizer.zero_grad()
            d = mfd_learn.distance(x, y)
            loss = ((d - target_d) ** 2).mean()
            loss.backward()
            optimizer.step()
            loss_history.append(loss.item())
        # Loss should decrease over training.
        assert loss_history[-1] < loss_history[0] * 0.9, (
            f"Loss did not decrease: start={loss_history[0]:.4e}, "
            f"end={loss_history[-1]:.4e}"
        )


# --------------------------------------------------------------------
# Full κ-differentiability — every κ-dependent op contributes a
# finite, non-zero gradient on κ when κ is a learnable Tensor.
# --------------------------------------------------------------------


class TestKappaGradientFlowPerOp:
    """For each operation that depends on κ, a downstream
    `.sum().backward()` must produce a finite, non-zero `κ.grad`.

    These tests pin the live-κ plumbing through `_conformal_factor`,
    `_tan_kappa_c`, `_atan_kappa_c`, `mobius_add`, and the higher-
    level methods that compose them.
    """

    @staticmethod
    def _make(kappa_init: float = -1.0):
        kappa = torch.nn.Parameter(
            torch.tensor(kappa_init, dtype=torch.float64),
        )
        mfd = KappaStereographicManifold(
            n=3, kappa=kappa, dtype=torch.float64,
        )
        return mfd, kappa

    def test_distance(self):
        mfd, kappa = self._make()
        x = mfd.random_point(batch_size=4, generator=_seed(200))
        y = mfd.random_point(batch_size=4, generator=_seed(201))
        d = mfd.distance(x, y)
        d.sum().backward()
        assert torch.isfinite(kappa.grad)
        assert kappa.grad.abs() > 0

    def test_inner(self):
        """inner uses _conformal_factor which uses live κ."""
        mfd, kappa = self._make()
        x = mfd.random_point(batch_size=4, generator=_seed(202))
        u = torch.randn(4, mfd.n, dtype=torch.float64, generator=_seed(203))
        v = torch.randn(4, mfd.n, dtype=torch.float64, generator=_seed(204))
        out = mfd.inner(x, u, v)
        out.sum().backward()
        assert torch.isfinite(kappa.grad)
        assert kappa.grad.abs() > 0

    def test_norm(self):
        """norm uses _conformal_factor which uses live κ."""
        mfd, kappa = self._make()
        x = mfd.random_point(batch_size=4, generator=_seed(205))
        v = torch.randn(4, mfd.n, dtype=torch.float64, generator=_seed(206))
        out = mfd.norm(x, v)
        out.sum().backward()
        assert torch.isfinite(kappa.grad)
        assert kappa.grad.abs() > 0

    def test_exp_0(self):
        """exp_0 uses _tan_kappa_c which uses live κ."""
        mfd, kappa = self._make()
        v = (torch.randn(4, mfd.n, dtype=torch.float64,
                          generator=_seed(207)) * 0.2)
        out = mfd.exp_0(v)
        out.sum().backward()
        assert torch.isfinite(kappa.grad)
        assert kappa.grad.abs() > 0

    def test_log_0(self):
        """log_0 uses _atan_kappa_c which uses live κ."""
        mfd, kappa = self._make()
        v = (torch.randn(4, mfd.n, dtype=torch.float64,
                          generator=_seed(208)) * 0.2)
        y = mfd.exp_0(v).detach()  # detach so only κ-grad accrues
        out = mfd.log_0(y)
        out.sum().backward()
        assert torch.isfinite(kappa.grad)
        assert kappa.grad.abs() > 0

    def test_exp(self):
        """exp uses both _conformal_factor and _tan_kappa_c + mobius_add."""
        mfd, kappa = self._make()
        x = mfd.random_point(batch_size=3, generator=_seed(209))
        v_tan = (torch.randn(3, mfd.n, dtype=torch.float64,
                              generator=_seed(210)) * 0.1)
        out = mfd.exp(x, v_tan)
        out.sum().backward()
        assert torch.isfinite(kappa.grad)
        assert kappa.grad.abs() > 0

    def test_log(self):
        """log uses _conformal_factor + _atan_kappa_c + mobius_add."""
        mfd, kappa = self._make()
        x = mfd.random_point(batch_size=3, generator=_seed(211))
        y = mfd.random_point(batch_size=3, generator=_seed(212))
        out = mfd.log(x, y)
        out.sum().backward()
        assert torch.isfinite(kappa.grad)
        assert kappa.grad.abs() > 0

    def test_parallel_transport(self):
        """parallel_transport uses _conformal_factor + mobius_add."""
        mfd, kappa = self._make()
        x = mfd.random_point(batch_size=3, generator=_seed(213))
        y = mfd.random_point(batch_size=3, generator=_seed(214))
        v_tan = (torch.randn(3, mfd.n, dtype=torch.float64,
                              generator=_seed(215)) * 0.1)
        out = mfd.parallel_transport(x, y, v_tan)
        out.sum().backward()
        assert torch.isfinite(kappa.grad)
        assert kappa.grad.abs() > 0

    def test_mobius_add(self):
        """Direct test on mobius_add — used by exp / log / pt."""
        mfd, kappa = self._make()
        x = mfd.random_point(batch_size=3, generator=_seed(216))
        y = mfd.random_point(batch_size=3, generator=_seed(217))
        out = mfd.mobius_add(x, y)
        out.sum().backward()
        assert torch.isfinite(kappa.grad)
        assert kappa.grad.abs() > 0

    def test_kappa_grad_through_substrate_chain(self):
        """End-to-end pattern: v -> exp_0(v) -> all-pairs distance ->
        NLL. Confirms κ is differentiable through the realistic
        substrate-training chain."""
        mfd, kappa = self._make()
        v = (torch.randn(6, mfd.n, dtype=torch.float64,
                          generator=_seed(218)) * 0.3)
        v.requires_grad_(True)
        T = mfd.exp_0(v)
        N = T.shape[0]
        Ti = T.unsqueeze(1).expand(N, N, mfd.n).reshape(-1, mfd.n)
        Tj = T.unsqueeze(0).expand(N, N, mfd.n).reshape(-1, mfd.n)
        d_all = mfd.distance(Ti, Tj).reshape(N, N)
        # NLL with cyclic target
        log_partition = torch.logsumexp(-d_all, dim=-1)
        target_d = d_all[torch.arange(N), (torch.arange(N) + 1) % N]
        loss = (target_d + log_partition).sum()
        loss.backward()
        # Both v AND κ get finite gradients
        assert torch.isfinite(v.grad).all()
        assert torch.isfinite(kappa.grad)
        # κ's gradient is non-trivial — the loss depends on curvature
        assert kappa.grad.abs() > 0


# --------------------------------------------------------------------
# κ-sign crossing during training (dynamic dispatch)
# --------------------------------------------------------------------


class TestKappaSignCrossing:
    """Tensor-κ with dynamic dispatch — the manifold's branch is no
    longer locked at construction time; SGD can push κ from
    hyperbolic (κ<0) to spherical (κ>0) or vice versa without
    breakdown. This was the previous "undefined behavior" case."""

    def test_kappa_crosses_zero_negative_to_positive(self):
        """κ_init = -0.5, regularized toward +0.5. After SGD, κ ends
        positive; throughout, distance is finite and points stay on
        the manifold."""
        kappa = torch.nn.Parameter(
            torch.tensor(-0.5, dtype=torch.float64),
        )
        mfd = KappaStereographicManifold(
            n=3, kappa=kappa, dtype=torch.float64,
        )
        v = (torch.randn(5, 3, dtype=torch.float64,
                          generator=_seed(300)) * 0.2)
        optimizer = torch.optim.SGD([kappa], lr=0.01)

        target = torch.tensor(0.5, dtype=torch.float64)
        for step in range(50):
            optimizer.zero_grad()
            T = mfd.exp_0(v)
            d = mfd.distance(T[:2], T[2:4])
            loss = d.sum() + 5.0 * (kappa - target) ** 2
            loss.backward()
            optimizer.step()
            # Check finiteness every step
            assert torch.isfinite(d).all().item(), (
                f"step {step}: distance not finite, κ={kappa.item()}"
            )
            assert torch.isfinite(T).all().item()

        # κ went from -0.5 to near +0.5 — branch must have flipped
        assert kappa.item() > 0, (
            f"κ should be positive after regularization, got {kappa.item()}"
        )
        # And points remain on manifold under the new sign
        T_final = mfd.exp_0(v).detach()
        assert mfd.is_on_manifold(T_final).all().item()

    def test_kappa_crosses_zero_positive_to_negative(self):
        """Reverse direction: κ_init = +0.5, regularized toward -0.5."""
        kappa = torch.nn.Parameter(
            torch.tensor(0.5, dtype=torch.float64),
        )
        mfd = KappaStereographicManifold(
            n=3, kappa=kappa, dtype=torch.float64,
        )
        # Sample directly via origin (avoid initialization in the
        # not-yet-correct branch) — points at origin work for any κ.
        T_origin = mfd.origin(batch_size=4)
        # Small Euclidean offsets, evaluated AT each step
        v = (torch.randn(4, 3, dtype=torch.float64,
                          generator=_seed(301)) * 0.1)
        optimizer = torch.optim.SGD([kappa], lr=0.01)
        target = torch.tensor(-0.5, dtype=torch.float64)
        for step in range(50):
            optimizer.zero_grad()
            T = mfd.exp_0(v)
            d = mfd.distance(T[:2], T[2:4])
            loss = d.sum() + 5.0 * (kappa - target) ** 2
            loss.backward()
            optimizer.step()
            assert torch.isfinite(d).all().item()
        assert kappa.item() < 0
