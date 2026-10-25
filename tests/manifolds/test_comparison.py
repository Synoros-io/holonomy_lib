# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Synoros

"""Tests for holonomy_lib.manifolds.comparison (model-space volumes).

Three layers:
  1. Unit tests; shapes across B ∈ {0, 1, several}; input validation.
  2. Property tests; closed-form known values (Euclidean / sphere /
     hyperbolic), κ→0 continuity, and the area = dV/dr identity.
  3. Comparison; the closed forms above are themselves the ground truth, so no
     external library is needed.
"""

from __future__ import annotations

import math

import pytest
import torch

from holonomy_lib.manifolds import (
    model_anisotropic_flux,
    model_ball_volume,
    model_sphere_area,
)
from holonomy_lib.manifolds.comparison import _unit_sphere_surface_area

DT = torch.float64


def _t(x):
    return torch.tensor(x, dtype=DT)


# --------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------


class TestValidation:
    def test_rejects_N_below_1(self):
        with pytest.raises(ValueError, match="N"):
            model_ball_volume(_t([0.0]), _t([0.5]), _t([1.0]))

    def test_rejects_negative_r(self):
        with pytest.raises(ValueError, match="r"):
            model_sphere_area(_t([0.0]), _t([3.0]), _t([-1.0]))


# --------------------------------------------------------------------
# Closed-form known values — flat (κ = 0); quadrature is exact here
# --------------------------------------------------------------------


class TestEuclidean:
    def test_sphere_area(self):
        r = _t([2.0])
        torch.testing.assert_close(
            model_sphere_area(_t([0.0]), _t([2.0]), r),
            _t([2 * math.pi * 2.0]), atol=1e-12, rtol=0)   # 2π r
        torch.testing.assert_close(
            model_sphere_area(_t([0.0]), _t([3.0]), r),
            _t([4 * math.pi * 4.0]), atol=1e-12, rtol=0)   # 4π r²

        for batch in [0, 1, 4]:
            out = model_sphere_area(
                torch.zeros(batch, dtype=DT),
                torch.full((batch,), 3.0, dtype=DT),
                torch.ones(batch, dtype=DT),
            )
            assert out.shape == (batch,)

    def test_ball_volume(self):
        r = _t([2.0])
        torch.testing.assert_close(
            model_ball_volume(_t([0.0]), _t([1.0]), r),
            _t([2 * 2.0]), atol=1e-12, rtol=0)             # 2r
        torch.testing.assert_close(
            model_ball_volume(_t([0.0]), _t([2.0]), r),
            _t([math.pi * 4.0]), atol=1e-12, rtol=0)       # π r²
        torch.testing.assert_close(
            model_ball_volume(_t([0.0]), _t([3.0]), r),
            _t([4 / 3 * math.pi * 8.0]), atol=1e-12, rtol=0)  # 4/3 π r³


# --------------------------------------------------------------------
# Closed-form known values — sphere (κ = 1) and hyperbolic (κ = -1)
# atol loosened on volumes: Gauss-Legendre quadrature vs analytic (64 nodes
# give ~1e-13 for these analytic integrands; bound set conservatively).
# --------------------------------------------------------------------


class TestSphere:
    def test_full_2sphere(self):
        # κ=1, N=2: the "ball" at r=π is the whole unit 2-sphere, area 4π.
        torch.testing.assert_close(
            model_ball_volume(_t([1.0]), _t([2.0]), _t([math.pi])),
            _t([4 * math.pi]), atol=1e-10, rtol=0)
        torch.testing.assert_close(
            model_sphere_area(_t([1.0]), _t([2.0]), _t([math.pi])),
            _t([0.0]), atol=1e-12, rtol=0)                 # pole: area 0

        r = torch.tensor([math.pi], dtype=torch.float32)
        area = model_sphere_area(torch.ones_like(r), torch.full_like(r, 1.5), r)
        torch.testing.assert_close(area, torch.zeros_like(area), atol=0, rtol=0)

    def test_full_3sphere_volume(self):
        # κ=1, N=3: full unit 3-sphere volume 2π².
        torch.testing.assert_close(
            model_ball_volume(_t([1.0]), _t([3.0]), _t([math.pi])),
            _t([2 * math.pi ** 2]), atol=1e-10, rtol=0)


class TestHyperbolic:
    def test_n2_area_and_volume(self):
        r = _t([1.3])
        torch.testing.assert_close(
            model_sphere_area(_t([-1.0]), _t([2.0]), r),
            _t([2 * math.pi * math.sinh(1.3)]), atol=1e-12, rtol=0)
        torch.testing.assert_close(
            model_ball_volume(_t([-1.0]), _t([2.0]), r),
            _t([2 * math.pi * (math.cosh(1.3) - 1)]), atol=1e-10, rtol=0)

        for batch in [0, 1, 4]:
            out = model_ball_volume(
                torch.full((batch,), -1.0, dtype=DT),
                torch.full((batch,), 2.5, dtype=DT),
                torch.ones(batch, dtype=DT),
            )
            assert out.shape == (batch,)


# --------------------------------------------------------------------
# Property tests
# --------------------------------------------------------------------


class TestProperties:
    def test_kappa_to_zero_continuity(self):
        # No discontinuity at κ=0: V(κ→0) → V(0). The residual at κ=1e-6 is
        # O(κ) (~8e-7), set atol well above that yet far below an O(1) jump.
        N, r = _t([3.0]), _t([1.0])
        v_flat = model_ball_volume(_t([0.0]), N, r)
        v_eps = model_ball_volume(_t([1e-6]), N, r)
        torch.testing.assert_close(v_flat, v_eps, atol=1e-5, rtol=0)

        from holonomy_lib.manifolds.comparison import sn_kappa
        k = torch.zeros(1, dtype=DT, requires_grad=True)
        t = torch.ones_like(k)
        assert torch.autograd.gradcheck(sn_kappa, (k, t))
        grad = torch.autograd.grad(sn_kappa(k, t).sum(), k)[0]
        torch.testing.assert_close(grad, torch.full_like(k, -1 / 6), atol=1e-12, rtol=0)

    def test_area_is_derivative_of_volume(self):
        # dV/dr = S(r); central difference, non-integer N, hyperbolic.
        kappa, N, r, h = _t([-1.0]), _t([2.5]), _t([0.7]), _t([1e-5])
        dV = (model_ball_volume(kappa, N, r + h)
              - model_ball_volume(kappa, N, r - h)) / (2 * h)
        torch.testing.assert_close(dV, model_sphere_area(kappa, N, r),
                                   atol=1e-6, rtol=0)

    def test_noninteger_N_positive(self):
        v = model_ball_volume(_t([0.3]), _t([2.5]), _t([0.9]))
        assert (v > 0).all()

        r = torch.zeros(1, dtype=DT, requires_grad=True)
        volume = model_ball_volume(torch.zeros_like(r), torch.full_like(r, 1.5), r)
        grad = torch.autograd.grad(volume.sum(), r)[0]
        torch.testing.assert_close(grad, torch.zeros_like(grad), atol=0, rtol=0)

    def test_volume_monotone_in_radius(self):
        kappa, N = _t([-1.0]), _t([3.0])
        v1 = model_ball_volume(kappa, N, _t([0.5]))
        v2 = model_ball_volume(kappa, N, _t([1.5]))
        assert (v2 > v1).all()


class TestAnisotropicFlux:
    def test_reduces_to_isotropic_sphere_area(self):
        # all kappa equal over K = N-1 directions -> Π sn_κ = sn_κ^{N-1}
        # = model_sphere_area / ω_{N-1} (the isotropic factor recovered).
        kappa, r, K = -1.0, 0.7, 4
        flux = model_anisotropic_flux(_t([[kappa] * K]), _t([r]))
        N = _t([K + 1.0])
        torch.testing.assert_close(
            flux * _unit_sphere_surface_area(N),
            model_sphere_area(_t([kappa]), N, _t([r])), atol=1e-9, rtol=0)

    def test_flat_is_r_to_the_K(self):
        # κ_i = 0 -> sn_0(r) = r, so the product is r^K.
        torch.testing.assert_close(
            model_anisotropic_flux(_t([[0.0, 0.0, 0.0]]), _t([2.0])),
            _t([8.0]), atol=1e-9, rtol=0)

    def test_mixed_signature_product(self):
        # one spherical, one flat, one hyperbolic: sn_1 * sn_0 * sn_{-1}.
        expected = math.sin(0.5) * 0.5 * math.sinh(0.5)
        torch.testing.assert_close(
            model_anisotropic_flux(_t([[1.0, 0.0, -1.0]]), _t([0.5])),
            _t([expected]), atol=1e-9, rtol=0)

        for B in (0, 1, 3):
            out = model_anisotropic_flux(torch.zeros(B, 4, dtype=DT),
                                         torch.ones(B, dtype=DT))
            assert out.shape == (B,)


    def test_grad_flows_to_kappas(self):
        kappas = _t([[-0.5, -0.2, 0.1]]).requires_grad_(True)
        model_anisotropic_flux(kappas, _t([0.8])).sum().backward()
        assert kappas.grad is not None and torch.isfinite(kappas.grad).all()

    def test_bad_shapes_raise(self):
        with pytest.raises(ValueError):
            model_anisotropic_flux(_t([1.0, 2.0]), _t([0.5]))        # 1-D kappas
        with pytest.raises(ValueError):
            model_anisotropic_flux(_t([[1.0]]), _t([0.5, 0.6]))      # r batch mismatch
