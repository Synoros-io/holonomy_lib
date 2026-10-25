# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Synoros

"""Tests for holonomy_lib.hyperbolic.hyperbolic_heat_kernel.

Layers:
  1. Closed-form sanity for n=1, 3.
  2. Integral form for n=2 (probability mass, symmetry).
  3. Recursion identity: k^{n+2} = -(2π sinh d)^{-1} ∂_d k^n —
     verified numerically by finite differences.
  4. Curvature scaling: k_K(t, d) = K^{n/2} · k_1(K·t, √K·d).
"""

from __future__ import annotations

import math

import pytest
import torch

from holonomy_lib.hyperbolic import hyperbolic_heat_kernel
from holonomy_lib.manifolds import LorentzManifold

# --------------------------------------------------------------------
# Sanity at boundary / closed-form values
# --------------------------------------------------------------------


def test_n1_is_gaussian():
    """H^1 heat kernel is the standard Gaussian on R."""
    mfd = LorentzManifold(n=1)
    t = torch.tensor(0.5, dtype=torch.float64)
    d = torch.linspace(0, 3.0, 7, dtype=torch.float64)
    k = hyperbolic_heat_kernel(t, d, mfd)
    expected = (4.0 * math.pi * 0.5) ** -0.5 \
        * torch.exp(-d * d / (4.0 * 0.5))
    torch.testing.assert_close(k, expected, atol=1e-12, rtol=1e-12)


def test_n3_at_zero_distance():
    """k^3_t(0) = (4πt)^{-3/2} · exp(-t) · 1  (the d/sinh d → 1 limit)."""
    mfd = LorentzManifold(n=3)
    t_vals = torch.tensor([0.1, 0.5, 1.0, 2.0], dtype=torch.float64)
    d = torch.zeros_like(t_vals)
    k = hyperbolic_heat_kernel(t_vals, d, mfd)
    expected = (4.0 * math.pi * t_vals) ** -1.5 * torch.exp(-t_vals)
    torch.testing.assert_close(k, expected, atol=1e-12, rtol=1e-12)


def test_n3_davies_mandouvalos_closed_form():
    """k^3_t(d) = (4πt)^{-3/2} · exp(-t - d²/4t) · d/sinh d."""
    mfd = LorentzManifold(n=3)
    t = torch.tensor(0.7, dtype=torch.float64)
    d = torch.tensor([0.1, 0.5, 1.0, 2.0], dtype=torch.float64)
    k = hyperbolic_heat_kernel(t, d, mfd)
    expected = (
        (4.0 * math.pi * t) ** -1.5
        * torch.exp(-t - d * d / (4.0 * t))
        * d / torch.sinh(d)
    )
    torch.testing.assert_close(k, expected, atol=1e-12, rtol=1e-12)


# --------------------------------------------------------------------
# Symmetry and positivity
# --------------------------------------------------------------------


def test_kernel_positive():
    """Heat kernel is a probability density — positive everywhere."""
    mfd = LorentzManifold(n=2)
    t = torch.tensor(1.0, dtype=torch.float64)
    d = torch.linspace(0.1, 3.0, 8, dtype=torch.float64)
    k = hyperbolic_heat_kernel(t, d, mfd)
    assert (k >= 0).all()
    assert torch.isfinite(k).all()


@pytest.mark.parametrize("n", [1, 2, 3, 5])
def test_kernel_decays_with_distance(n):
    """At fixed t, the kernel decreases as d grows (for d >> √t)."""
    mfd = LorentzManifold(n=n)
    t = torch.tensor(0.5, dtype=torch.float64)
    d = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    k = hyperbolic_heat_kernel(t, d, mfd)
    # Monotonically decreasing
    assert (k[1] < k[0]) and (k[2] < k[1])


# --------------------------------------------------------------------
# Curvature scaling
# --------------------------------------------------------------------


def test_curvature_scaling_n3():
    """k^n_{−K, t}(d) = K^{n/2} · k^n_{−1, K·t}(√K · d)."""
    K = 2.0
    mfd_unit = LorentzManifold(n=3, k=-1.0)
    mfd_k = LorentzManifold(n=3, k=-K)
    t = torch.tensor(0.5, dtype=torch.float64)
    d = torch.tensor([0.5, 1.0, 1.5], dtype=torch.float64)
    k_curved = hyperbolic_heat_kernel(t, d, mfd_k)
    k_unit_scaled = (
        K ** 1.5
        * hyperbolic_heat_kernel(K * t, math.sqrt(K) * d, mfd_unit)
    )
    torch.testing.assert_close(
        k_curved, k_unit_scaled, atol=1e-12, rtol=1e-12,
    )


# --------------------------------------------------------------------
# Recursion identity
# --------------------------------------------------------------------


def test_recursion_identity_n3_to_n5():
    """Numerical check of the correct spectral-shifted recursion:

        k^5_t(d) = -exp(-3·t) / (2π sinh d) · ∂_d k^3_t(d)

    where the `exp(-3·t)` factor is the spectral-shift correction
    (the heat-kernel spectral bottom on `H^n` is `((n-1)/2)²`; going
    from n=3 to n=5 shifts by 4-1 = 3, so the recursion picks up
    `exp(-3·t)`).

    The earlier version of this test verified the *uncorrected*
    `k^5 = -1/(2π sinh r) · ∂_r k^3` identity — which is what our
    code computed — but that quantity is NOT the actual H^5 heat
    kernel. The omission was caught by the heat-equation residual
    validation (`notes/validation/heat_kernel_results.md`).
    """
    mfd_3 = LorentzManifold(n=3)
    mfd_5 = LorentzManifold(n=5)
    t = torch.tensor(0.5, dtype=torch.float64)
    d = torch.tensor([0.5, 1.0, 1.5], dtype=torch.float64)
    eps = 1e-5
    k_plus = hyperbolic_heat_kernel(t, d + eps, mfd_3)
    k_minus = hyperbolic_heat_kernel(t, d - eps, mfd_3)
    fd = (k_plus - k_minus) / (2.0 * eps)
    expected_n5 = (
        -torch.exp(-3.0 * t) * fd / (2.0 * math.pi * torch.sinh(d))
    )
    actual_n5 = hyperbolic_heat_kernel(t, d, mfd_5)
    torch.testing.assert_close(actual_n5, expected_n5, atol=1e-6, rtol=1e-6)


def test_recursion_identity_n5_to_n7():
    """Same correct recursion at the next step: `k^7 = -exp(-5·t) ·
    (2π sinh d)^{-1} · ∂_d k^5`. Catches any further off-by-shift
    errors in the iteration."""
    mfd_5 = LorentzManifold(n=5)
    mfd_7 = LorentzManifold(n=7)
    t = torch.tensor(0.5, dtype=torch.float64)
    d = torch.tensor([0.5, 1.0, 1.5], dtype=torch.float64)
    eps = 1e-5
    k_plus = hyperbolic_heat_kernel(t, d + eps, mfd_5)
    k_minus = hyperbolic_heat_kernel(t, d - eps, mfd_5)
    fd = (k_plus - k_minus) / (2.0 * eps)
    expected_n7 = (
        -torch.exp(-5.0 * t) * fd / (2.0 * math.pi * torch.sinh(d))
    )
    actual_n7 = hyperbolic_heat_kernel(t, d, mfd_7)
    torch.testing.assert_close(actual_n7, expected_n7, atol=1e-6, rtol=1e-6)


def test_heat_equation_residual_n5():
    """Strong validation: k^5_t(d) satisfies the radial heat equation
    on H^5 to finite-difference noise floor (~1e-5). This is the
    independent check that caught the earlier missing spectral-shift
    factor — see `notes/validation/heat_kernel_results.md`."""
    mfd = LorentzManifold(n=5)
    t = torch.tensor(0.5, dtype=torch.float64)
    r = torch.tensor(1.0, dtype=torch.float64)
    dt = 1e-5
    dr = 1e-5
    k_plus_t = hyperbolic_heat_kernel(t + dt, r, mfd).item()
    k_minus_t = hyperbolic_heat_kernel(t - dt, r, mfd).item()
    k_0 = hyperbolic_heat_kernel(t, r, mfd).item()
    k_plus_r = hyperbolic_heat_kernel(t, r + dr, mfd).item()
    k_minus_r = hyperbolic_heat_kernel(t, r - dr, mfd).item()
    dk_dt = (k_plus_t - k_minus_t) / (2.0 * dt)
    dk_dr = (k_plus_r - k_minus_r) / (2.0 * dr)
    d2k_dr2 = (k_plus_r - 2.0 * k_0 + k_minus_r) / (dr * dr)
    coth_r = math.cosh(r.item()) / math.sinh(r.item())
    lap_k = d2k_dr2 + 4.0 * coth_r * dk_dr   # n - 1 = 4 for n=5
    residual = abs(dk_dt - lap_k) / max(abs(dk_dt), abs(lap_k), 1e-300)
    assert residual < 1e-4, (
        f"heat-equation residual at (n=5, t=0.5, r=1.0): {residual:.4e}"
    )


def test_heat_equation_residual_n7():
    """k^7_t(d) (hand-derived closed form) satisfies the radial heat
    equation on H^7 to finite-difference noise floor — an independent
    check on the closed form, separate from the recursion-identity test.
    See notes/verification/heat_kernel_n7_sympy.py."""
    mfd = LorentzManifold(n=7)
    t = torch.tensor(0.5, dtype=torch.float64)
    r = torch.tensor(1.0, dtype=torch.float64)
    dt = 1e-5
    dr = 1e-5
    k_plus_t = hyperbolic_heat_kernel(t + dt, r, mfd).item()
    k_minus_t = hyperbolic_heat_kernel(t - dt, r, mfd).item()
    k_0 = hyperbolic_heat_kernel(t, r, mfd).item()
    k_plus_r = hyperbolic_heat_kernel(t, r + dr, mfd).item()
    k_minus_r = hyperbolic_heat_kernel(t, r - dr, mfd).item()
    dk_dt = (k_plus_t - k_minus_t) / (2.0 * dt)
    dk_dr = (k_plus_r - k_minus_r) / (2.0 * dr)
    d2k_dr2 = (k_plus_r - 2.0 * k_0 + k_minus_r) / (dr * dr)
    coth_r = math.cosh(r.item()) / math.sinh(r.item())
    lap_k = d2k_dr2 + 6.0 * coth_r * dk_dr   # n - 1 = 6 for n=7
    residual = abs(dk_dt - lap_k) / max(abs(dk_dt), abs(lap_k), 1e-300)
    assert residual < 1e-4, (
        f"heat-equation residual at (n=7, t=0.5, r=1.0): {residual:.4e}"
    )


def test_n7_closed_form_zero_distance_limit():
    """The n=7 closed form's r → 0 analytic limit is
    (4πt)^{-7/2} · exp(-9t) · (1 + 2t + 16t²/15) (verified to 16 digits
    in notes/verification/heat_kernel_n7_sympy.py)."""
    mfd = LorentzManifold(n=7)
    t = torch.tensor([0.25, 0.5, 1.0], dtype=torch.float64)
    d = torch.zeros(3, dtype=torch.float64)
    k = hyperbolic_heat_kernel(t, d, mfd)
    expected = (
        (4.0 * math.pi * t) ** (-3.5)
        * torch.exp(-9.0 * t)
        * (1.0 + 2.0 * t + 16.0 * t * t / 15.0)
    )
    torch.testing.assert_close(k, expected, atol=1e-10, rtol=1e-10)
    assert torch.isfinite(k).all()


def test_n7_closed_form_backward_finite():
    """Closed-form n=7 has finite forward + backward at boundary inputs,
    including the d = 0 analytic-limit branch."""
    mfd = LorentzManifold(n=7)
    t = torch.tensor(0.5, dtype=torch.float64)
    d = torch.tensor([0.0, 0.5, 2.0], dtype=torch.float64, requires_grad=True)
    k = hyperbolic_heat_kernel(t, d, mfd)
    assert torch.isfinite(k).all()
    assert (k > 0).all()
    k.sum().backward()
    assert torch.isfinite(d.grad).all()


def test_n7_closed_form_matches_recursion():
    """The n=7 closed form equals one corrected recursion step from the n=5
    closed form (mirrors test_recursion_identity_n5_to_n7 at the unit-function
    level, to machine precision rather than finite-difference)."""
    from holonomy_lib.hyperbolic.heat_kernel import (
        _apply_one_recursion,
        _heat_kernel_unit_n5,
        _heat_kernel_unit_n7,
    )
    t = torch.tensor(0.5, dtype=torch.float64)
    d = torch.linspace(0.1, 3.0, 7, dtype=torch.float64)
    k_closed = _heat_kernel_unit_n7(t, d)
    k_recursion = _apply_one_recursion(_heat_kernel_unit_n5, 5, t, d)
    torch.testing.assert_close(k_closed, k_recursion, atol=1e-10, rtol=1e-10)


def test_n9_via_recursion_from_n7():
    """n=9 now seeds the recursion from the n=7 closed form (was n=5); verify
    the cascaded result still satisfies the H^9 radial heat equation (n-1=8)."""
    mfd = LorentzManifold(n=9)
    t = torch.tensor(0.5, dtype=torch.float64)
    r = torch.tensor(1.0, dtype=torch.float64)
    dt = dr = 1e-5
    k_plus_t = hyperbolic_heat_kernel(t + dt, r, mfd).item()
    k_minus_t = hyperbolic_heat_kernel(t - dt, r, mfd).item()
    k_0 = hyperbolic_heat_kernel(t, r, mfd).item()
    k_plus_r = hyperbolic_heat_kernel(t, r + dr, mfd).item()
    k_minus_r = hyperbolic_heat_kernel(t, r - dr, mfd).item()
    dk_dt = (k_plus_t - k_minus_t) / (2.0 * dt)
    dk_dr = (k_plus_r - k_minus_r) / (2.0 * dr)
    d2k_dr2 = (k_plus_r - 2.0 * k_0 + k_minus_r) / (dr * dr)
    coth_r = math.cosh(r.item()) / math.sinh(r.item())
    lap_k = d2k_dr2 + 8.0 * coth_r * dk_dr   # n - 1 = 8 for n=9
    residual = abs(dk_dt - lap_k) / max(abs(dk_dt), abs(lap_k), 1e-300)
    assert residual < 1e-4, (
        f"heat-equation residual at (n=9, t=0.5, r=1.0): {residual:.4e}"
    )


def test_even_n_via_recursion():
    """Even n ≥ 4 now uses the same spectral-shift-corrected recursion
    as odd n, seeded from the n=2 Davies–Mandouvalos integral form.
    Validated by heat-equation residual + probability-mass normalization
    in `notes/validation/heat_kernel_results.md`. Here we just verify
    forward + backward are finite and the result is a positive density."""
    mfd_4 = LorentzManifold(n=4)
    t = torch.tensor(0.5, dtype=torch.float64)
    d = torch.tensor([0.5, 1.0, 2.0], dtype=torch.float64,
                      requires_grad=True)
    k = hyperbolic_heat_kernel(t, d, mfd_4)
    assert torch.isfinite(k).all()
    assert (k > 0).all()  # heat kernel is a probability density
    # Backward through the recursion path must produce finite gradient
    k.sum().backward()
    assert torch.isfinite(d.grad).all()


def test_n6_via_two_recursion_steps():
    """n=6 applies the recursion twice (n=2 → n=4 → n=6). The
    compounded autograd chain must stay finite and produce a
    positive-valued kernel."""
    mfd_6 = LorentzManifold(n=6)
    t = torch.tensor(0.5, dtype=torch.float64)
    d = torch.tensor([0.5, 1.0, 2.0], dtype=torch.float64)
    k = hyperbolic_heat_kernel(t, d, mfd_6)
    assert torch.isfinite(k).all()
    assert (k > 0).all()


# --------------------------------------------------------------------
# Integral path (n=2) self-consistency
# --------------------------------------------------------------------


def test_n2_quadrature_node_count_convergence():
    """Doubling quadrature nodes shouldn't change the result much
    once the integrand is well-resolved."""
    mfd = LorentzManifold(n=2)
    t = torch.tensor(0.5, dtype=torch.float64)
    d = torch.tensor(1.0, dtype=torch.float64)
    k_32 = hyperbolic_heat_kernel(t, d, mfd, n_quad=32)
    k_64 = hyperbolic_heat_kernel(t, d, mfd, n_quad=64)
    # Should agree to integration-error tolerance
    assert (k_32 - k_64).abs().max().item() < 1e-9


# --------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------


def test_provenance():
    from holonomy_lib import provenance

    mfd = LorentzManifold(n=3)
    t = torch.tensor(1.0, dtype=torch.float64)
    d = torch.tensor([0.5, 1.0], dtype=torch.float64)
    with provenance.record() as reg:
        _ = hyperbolic_heat_kernel(t, d, mfd)
    ops = {n.op_id for n in reg}
    assert "holonomy_lib.hyperbolic.hyperbolic_heat_kernel" in ops


# --------------------------------------------------------------------
# Autograd-finite — backward through the recursion at n ≥ 5
# --------------------------------------------------------------------


def test_heat_kernel_backward_n3():
    """Closed-form path (n=3): backward w.r.t. d must work."""
    mfd = LorentzManifold(n=3)
    t = torch.tensor(0.5, dtype=torch.float64)
    d = torch.tensor([0.5, 1.0, 1.5], dtype=torch.float64,
                      requires_grad=True)
    k = hyperbolic_heat_kernel(t, d, mfd)
    k.sum().backward()
    assert torch.isfinite(d.grad).all()


def test_heat_kernel_backward_n5_recursion():
    """Recursion path (n=5): the previous create_graph=False + detach()
    broke autograd through `distances`. With create_graph=True, the
    gradient should flow."""
    mfd = LorentzManifold(n=5)
    t = torch.tensor(0.5, dtype=torch.float64)
    d = torch.tensor([0.5, 1.0, 1.5], dtype=torch.float64,
                      requires_grad=True)
    k = hyperbolic_heat_kernel(t, d, mfd)
    k.sum().backward()
    assert d.grad is not None, "no gradient — recursion detached the graph"
    assert torch.isfinite(d.grad).all()
    # Gradient should be non-trivial (k decreases as d grows away
    # from 0, so ∂k/∂d < 0 for d > peak).
    assert (d.grad != 0).any()


# --------------------------------------------------------------------
# Closed-form n=5 precision push
# --------------------------------------------------------------------


def test_n5_closed_form_matches_recursion():
    """The closed-form n=5 path (operator chain expanded analytically)
    should agree with the autograd-based recursion path to a tight
    tolerance — they compute the SAME mathematical function via
    different numerical chains."""
    from holonomy_lib.hyperbolic.heat_kernel import (
        _apply_one_recursion,
        _heat_kernel_unit_n3,
        _heat_kernel_unit_n5,
    )

    t = torch.tensor(0.5, dtype=torch.float64)
    d = torch.linspace(0.1, 3.0, 7, dtype=torch.float64)

    k_closed = _heat_kernel_unit_n5(t, d)
    # Apply the recursion path manually
    k_recursion = _apply_one_recursion(
        _heat_kernel_unit_n3, 3, t, d,
    )

    torch.testing.assert_close(
        k_closed, k_recursion, atol=1e-10, rtol=1e-10,
    )


def test_n5_closed_form_at_zero():
    """Limit at r=0: k^5(t, 0) = (4πt)^{-5/2} exp(-4t) (1 + 2t/3)."""
    mfd = LorentzManifold(n=5)
    t_vals = torch.tensor([0.1, 0.5, 1.0, 2.0], dtype=torch.float64)
    d = torch.zeros_like(t_vals)
    k = hyperbolic_heat_kernel(t_vals, d, mfd)
    expected = (
        (4.0 * math.pi * t_vals) ** -2.5
        * torch.exp(-4.0 * t_vals)
        * (1.0 + 2.0 * t_vals / 3.0)
    )
    torch.testing.assert_close(k, expected, atol=1e-12, rtol=1e-12)


def test_n5_closed_form_backward_finite():
    """Backward through the closed form path stays finite."""
    mfd = LorentzManifold(n=5)
    t = torch.tensor(0.5, dtype=torch.float64)
    d = torch.tensor([0.1, 0.5, 1.0, 2.0], dtype=torch.float64,
                      requires_grad=True)
    k = hyperbolic_heat_kernel(t, d, mfd)
    k.sum().backward()
    assert torch.isfinite(d.grad).all()
    assert (d.grad != 0).any()
