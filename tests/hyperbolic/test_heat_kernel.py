# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Synoros

"""Closed forms, independent quadrature, dimensional recurrence, and radial PDE."""

from __future__ import annotations

import math

import pytest
import torch

from holonomy_lib.hyperbolic import hyperbolic_heat_kernel
from holonomy_lib.manifolds import LorentzManifold


def test_n1_is_gaussian() -> None:
    """H^1 heat kernel is the standard Gaussian on R."""
    t = torch.tensor(0.5, dtype=torch.float64)
    d = torch.linspace(0, 3.0, 7, dtype=torch.float64)
    actual = hyperbolic_heat_kernel(t, d, LorentzManifold(n=1))
    expected = (4 * math.pi * t) ** -0.5 * torch.exp(-d.square() / (4 * t))
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("n", [3, 5, 7])
def test_n3_at_zero_distance(n: int) -> None:
    """Closed-form diagonal values include the continuous H^5/H^7 limits."""
    t = torch.tensor([0.1, 0.25, 0.5, 1.0, 2.0], dtype=torch.float64)
    factor = torch.ones_like(t)
    if n == 5:
        factor = 1 + 2 * t / 3
    elif n == 7:
        factor = 1 + 2 * t + 16 * t.square() / 15
    expected = (
        (4 * math.pi * t) ** (-n / 2) * torch.exp(-(((n - 1) / 2) ** 2) * t) * factor
    )
    actual = hyperbolic_heat_kernel(t, torch.zeros_like(t), LorentzManifold(n=n))
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    assert torch.isfinite(actual).all()


def test_n3_davies_mandouvalos_closed_form() -> None:
    """Nonzero H^3 values match the published closed form."""
    t = torch.tensor(0.7, dtype=torch.float64)
    d = torch.tensor([0.1, 0.5, 1.0, 2.0], dtype=torch.float64)
    actual = hyperbolic_heat_kernel(t, d, LorentzManifold(n=3))
    expected = (
        (4 * math.pi * t) ** -1.5 * torch.exp(-t - d.square() / (4 * t)) * d / d.sinh()
    )
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6, 7, 9])
def test_kernel_decays_with_distance(n: int) -> None:
    """The kernel is finite, positive, and decreases with radius."""
    t = torch.tensor(0.5, dtype=torch.float64)
    d = torch.tensor([0.1, 0.5, 1.0, 2.0, 3.0], dtype=torch.float64)
    k = hyperbolic_heat_kernel(t, d, LorentzManifold(n=n))
    assert torch.isfinite(k).all()
    assert (k > 0).all()
    assert (k[1:] < k[:-1]).all()


def test_curvature_scaling_n3() -> None:
    """k_K(t,d) = K^(n/2) k_1(Kt,sqrt(K)d)."""
    curvature = 2.0
    t = torch.tensor(0.5, dtype=torch.float64)
    d = torch.tensor([0.5, 1.0, 1.5], dtype=torch.float64)
    actual = hyperbolic_heat_kernel(t, d, LorentzManifold(n=3, k=-curvature))
    expected = curvature**1.5 * hyperbolic_heat_kernel(
        curvature * t,
        math.sqrt(curvature) * d,
        LorentzManifold(n=3),
    )
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("n", [3, 5])
def test_recursion_identity_n3_to_n5(n: int) -> None:
    """Finite differences verify k_(n+2)=-exp(-nt) k_n'/(2 pi sinh r)."""
    t = torch.tensor(0.5, dtype=torch.float64)
    d = torch.tensor([0.5, 1.0, 1.5], dtype=torch.float64)
    step = 1e-5
    previous = LorentzManifold(n=n)
    derivative = (
        hyperbolic_heat_kernel(t, d + step, previous)
        - hyperbolic_heat_kernel(t, d - step, previous)
    ) / (2 * step)
    expected = -torch.exp(-n * t) * derivative / (2 * math.pi * d.sinh())
    actual = hyperbolic_heat_kernel(t, d, LorentzManifold(n=n + 2))
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("n", [5, 7, 9])
def test_heat_equation_residual_n5(n: int) -> None:
    """Closed forms and recursion satisfy the radial heat equation."""
    manifold = LorentzManifold(n=n)
    t = torch.tensor(0.5, dtype=torch.float64)
    r = torch.tensor(1.0, dtype=torch.float64)
    step = 1e-5
    center = hyperbolic_heat_kernel(t, r, manifold)
    radial_plus = hyperbolic_heat_kernel(t, r + step, manifold)
    radial_minus = hyperbolic_heat_kernel(t, r - step, manifold)
    dt = (
        hyperbolic_heat_kernel(t + step, r, manifold)
        - hyperbolic_heat_kernel(t - step, r, manifold)
    ) / (2 * step)
    dr = (radial_plus - radial_minus) / (2 * step)
    drr = (radial_plus - 2 * center + radial_minus) / step**2
    laplacian = drr + (n - 1) * r.cosh() / r.sinh() * dr
    residual = (dt - laplacian).abs() / torch.maximum(dt.abs(), laplacian.abs())
    # Second finite differences amplify double-precision rounding.
    assert residual < 1e-4


@pytest.mark.parametrize("n", [2, 3, 4, 5, 6, 7, 9])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_n7_closed_form_backward_finite(n: int, dtype: torch.dtype) -> None:
    """Closed forms, small-radius expansions, and recursion retain gradients."""
    manifold = LorentzManifold(n=n)
    t = torch.tensor(1.0, dtype=dtype, requires_grad=True)
    d = torch.tensor(
        [0.0, 1e-7, 1e-4, 0.1, 0.5, 1.0, 1.5, 2.0], dtype=dtype, requires_grad=True
    )
    k = hyperbolic_heat_kernel(t, d, manifold)
    assert torch.isfinite(k).all()
    assert (k > 0).all()
    k.sum().backward()
    assert d.grad is not None and torch.isfinite(d.grad).all()
    assert t.grad is not None and torch.isfinite(t.grad).all()
    assert (d.grad[1:] != 0).any()
    # 60-digit mpmath evaluation of the operator-chain formula, not a
    # rounded evaluation of the library's cancellation-prone quotient.
    radial_references = {
        4: 8.1600935866099337668e-4,
        6: 2.4296242798357668138e-6,
        7: 7.1343384984359798949e-8,
    }
    if n in radial_references:
        expected = torch.tensor(radial_references[n], dtype=dtype)
        torch.testing.assert_close(k[2], expected, atol=0, rtol=5e-7)

    # The final even-dimensional value is representable in float32 even
    # when an unscaled intermediate odd-dimensional kernel underflows.
    if n in (2, 6):
        large_time, diagonal_reference = {
            2: (100.0, 2.93832691120845e-15),
            6: (10.0, 1.0569925138280928e-32),
        }[n]
        time = torch.tensor(large_time, dtype=dtype, requires_grad=True)
        radius = torch.tensor(0.0, dtype=dtype, requires_grad=True)
        value = hyperbolic_heat_kernel(time, radius, manifold, n_quad=64)
        torch.testing.assert_close(
            value,
            torch.tensor(diagonal_reference, dtype=dtype),
            atol=0,
            rtol=1e-6,
        )
        value.backward()
        assert time.grad is not None and torch.isfinite(time.grad)
        assert radius.grad is not None and torch.isfinite(radius.grad)
    for batch in (0, 1):
        distances = torch.zeros(batch, dtype=dtype, requires_grad=True)
        empty_or_single = hyperbolic_heat_kernel(t, distances, manifold)
        assert empty_or_single.shape == (batch,)
        empty_or_single.sum().backward()
        assert distances.grad is not None and torch.isfinite(distances.grad).all()

    if n in (4, 6, 9):
        times = torch.tensor([0.3, 0.7], dtype=torch.float64)
        distance = torch.tensor(0.8, dtype=torch.float64)
        with torch.no_grad():
            batched = hyperbolic_heat_kernel(times, distance, manifold)
            independent = torch.stack(
                [hyperbolic_heat_kernel(time, distance, manifold) for time in times]
            )
        torch.testing.assert_close(batched, independent, atol=1e-12, rtol=1e-12)
        assert not batched.requires_grad
        grad_times = times.requires_grad_()
        assert torch.autograd.gradcheck(
            lambda time: hyperbolic_heat_kernel(time, distance, manifold),
            (grad_times,),
        )


@pytest.mark.parametrize("n", [5, 7])
def test_n7_closed_form_matches_recursion(n: int) -> None:
    """Closed forms agree with a differentiated lower-dimensional formula."""
    from holonomy_lib.hyperbolic.heat_kernel import (
        _apply_one_recursion,
        _heat_kernel_unit_n3,
        _heat_kernel_unit_n5,
        _heat_kernel_unit_n7,
    )

    previous, closed = (
        (_heat_kernel_unit_n3, _heat_kernel_unit_n5)
        if n == 5
        else (
            _heat_kernel_unit_n5,
            _heat_kernel_unit_n7,
        )
    )
    t = torch.tensor(0.5, dtype=torch.float64)
    d = torch.linspace(0.1, 3.0, 7, dtype=torch.float64)
    torch.testing.assert_close(
        closed(t, d),
        _apply_one_recursion(previous, n - 2, t, d),
        atol=1e-10,
        rtol=1e-10,
    )


def test_n2_quadrature_node_count_convergence() -> None:
    """Node refinement and independently normalized high-precision H^2 values."""
    manifold = LorentzManifold(n=2)
    t = torch.tensor(0.5, dtype=torch.float64)
    d = torch.tensor(1.0, dtype=torch.float64)
    k32 = hyperbolic_heat_kernel(t, d, manifold, n_quad=32)
    k64 = hyperbolic_heat_kernel(t, d, manifold, n_quad=64)
    torch.testing.assert_close(k32, k64, atol=1e-9, rtol=0)
    # mpmath at 60 digits, integrating the published normalized H^2
    # integral on [0,infinity), independently of the finite GL rule.
    reference = [0.057535755205721974619, 2.908062405149898886681]
    for dtype, tolerance in ((torch.float32, 5e-7), (torch.float64, 1e-9)):
        times = torch.tensor([1.0, 0.01], dtype=dtype)
        distances = torch.tensor([0.0, 0.2], dtype=dtype)
        expected = torch.tensor(reference, dtype=dtype)
        actual = hyperbolic_heat_kernel(times, distances, manifold)
        torch.testing.assert_close(actual, expected, atol=0, rtol=tolerance)
