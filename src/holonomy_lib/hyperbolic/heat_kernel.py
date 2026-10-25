# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Synoros

"""Differentiable radial heat kernels on hyperbolic space.

For unit curvature, Grigor'yan-Noguchi (1998), equations (1.6)-(1.8), give

    k_3(t,r) = (4 pi t)^(-3/2) exp(-t-r^2/(4t)) r/sinh(r),
    k_2(t,r) = sqrt(2) exp(-t/4) (4 pi t)^(-3/2)
               integral_r^infinity s exp(-s^2/(4t))/sqrt(cosh(s)-cosh(r)) ds,
    k_(n+2)(t,r) = -exp(-nt)/(2 pi sinh(r)) d_r k_n(t,r).

H^1 is the Euclidean Gaussian. H^3, H^5 and H^7 use closed forms;
higher odd dimensions and even dimensions above two use the recurrence.
Odd kernels have a dtype-aware small-radius series, and even kernels
use a smooth dimension-descent integral in that neighborhood. Recursion
is sequential in dimension because each step differentiates the preceding
kernel. All tensor elements and quadrature nodes are evaluated together.

For curvature -K, k_(K,n)(t,r) = K^(n/2) k_n(Kt,sqrt(K)r).
Quadrature is approximate: the tail budget controls Gaussian decay at
the endpoint, not a uniform bound on the full integral's relative error.

References:
  Grigor'yan, A., Noguchi, M. (1998). The heat kernel on hyperbolic
    space. Bull. London Math. Soc. 30(6):643-650, equations (1.4)-(1.8).
  Davies, E. B., Mandouvalos, N. (1988). Heat kernel bounds on
    hyperbolic space and Kleinian groups. Proc. London Math. Soc.
    57(1):182-208.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from fractions import Fraction
from functools import cache
from typing import Protocol

import torch

from holonomy_lib._quadrature import gauss_legendre_nodes

# Dimensions with dedicated closed-form or quadrature routines. Above
# these we fall back to the Grigor'yan–Noguchi recursion. The values are
# fixed by the geometry, not tuned: 3 is the lowest dimension with a
# Davies–Mandouvalos closed form, and 2 is the lowest that needs the
# integral representation. `notes/magic_numbers.md` records them.
_N_GRIGORYAN_INTEGRAL: int = 2
_N_DAVIES_MANDOUVALOS: int = 3

# n = 5 has a polynomial closed form, obtained by applying the operator
# chain (1/sinh r · ∂_r)² analytically. Using it directly skips the
# autograd recursion, which adds float noise through two
# `torch.autograd.grad` calls.
_N_DAVIES_MANDOUVALOS_N5_CLOSED: int = 5

# n = 7 has a polynomial closed form, one operator-chain step beyond n = 5
# (derived and checked symbolically in notes/verification/heat_kernel_n7_sympy.py).
# It is the same argument as n = 5, and it seeds the odd-n recursion at n >= 9.
_N_OPERATOR_CHAIN_N7_CLOSED: int = 7

# Default quadrature resolution, checked by node refinement and independent
# normalized mpmath integrals. Increase n_quad for unusually narrow or broad
# time/radius regimes. Cataloged in notes/magic_numbers.md.
HEAT_KERNEL_QUADRATURE_NODES: int = 32

# This is an exponent budget: S=sqrt(4*budget*t) gives S^2/(4t)=budget.
# The endpoint of the H^2 substitution s=r+u^2 is r+S. The smooth
# descent substitution uses sqrt(r^2+S^2). Neither endpoint promises
# a uniform relative tail bound after the geometric factors are included.
HEAT_KERNEL_QUADRATURE_TAIL_BUDGET: float = 20.0


# The 4 in the Gaussian normalization (4 pi t)^(-n/2) and in the exponent
# d^2 / (4t). It is a geometric constant, not a fitted scale.
_GAUSSIAN_TIME_SCALE = 2 * 2

# Sixteen powers of r^2 retain terms through r^30. The switching radius
# balances the omitted r^32 term against eps / r^(n-3) cancellation in
# the odd-dimensional quotient. Checked against 60-digit operator chains.
_RADIAL_SERIES_TERMS = 16


class _HyperbolicManifold(Protocol):
    n: int
    k: float


@cache
def _inverse_sinhc_coefficients(order: int) -> tuple[Fraction, ...]:
    """Taylor coefficients of sqrt(q)/sinh(sqrt(q)), through q^order."""
    # Invert sum(q^j / (2j+1)!) as a formal power series. Fractions keep
    # these geometry-derived coefficients exact until tensor evaluation.
    coefficients = [Fraction(1)]
    for degree in range(1, order + 1):
        coefficients.append(
            -sum(
                (
                    coefficients[degree - j] / math.factorial(2 * j + 1)
                    for j in range(1, degree + 1)
                ),
                start=Fraction(0),
            )
        )
    return tuple(coefficients)


def _odd_radial_series(n: int, t: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Polynomial factor in k_n=(4 pi t)^(-n/2) exp(-rho^2 t-q/4t) P_n.

    The Grigor'yan-Noguchi (1998), equation (1.8), recurrence gives
    P_(n+2) = (r/sinh r) (P_n - 4t dP_n/dq), q=r^2, P_1=1.
    Each derivative consumes a coefficient, so begin with enough terms
    for every dimension-raising step. Loops are over series coefficients
    and dimensions, never over tensor elements.
    """
    steps = (n - 1) // 2
    order = _RADIAL_SERIES_TERMS + steps
    inverse_sinhc = _inverse_sinhc_coefficients(order)
    coefficients = [torch.ones_like(t)] + [torch.zeros_like(t)] * order
    for _ in range(steps):
        differentiated = [
            coefficients[j] - _GAUSSIAN_TIME_SCALE * t * (j + 1) * coefficients[j + 1]
            for j in range(len(coefficients) - 1)
        ]
        coefficients = [
            sum(
                (float(inverse_sinhc[j - k]) * differentiated[k] for k in range(j + 1)),
                start=torch.zeros_like(t),
            )
            for j in range(len(differentiated))
        ]
    result = torch.zeros_like(t + q)
    for coefficient in reversed(coefficients[:_RADIAL_SERIES_TERMS]):
        result = result * q + coefficient
    return result


def _small_radius(n: int, d: torch.Tensor) -> torch.Tensor:
    """Balance series truncation r^(2L) with quotient cancellation eps/r^(n-3)."""
    radius = torch.finfo(d.dtype).eps ** (
        1 / (2 * _RADIAL_SERIES_TERMS + n - _N_DAVIES_MANDOUVALOS)
    )
    return d.abs() <= radius


def _heat_kernel_unit_n1(t: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """Heat kernel on H^1, which is R: the standard Gaussian.

        k^1_t(d) = (4πt)^{-1/2} · exp(-d²/4t)

    Boundary case for the recursion (H^1 is Euclidean; curvature 0 is
    a degenerate limit of sectional curvature < 0). Included so the
    recursion has a clean odd-n seed.
    """
    return (_GAUSSIAN_TIME_SCALE * math.pi * t) ** -0.5 * torch.exp(
        -d * d / (2.0 * 2.0 * t)
    )


def _csch_and_coth(r: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Hyperbolic reciprocals for positive r without sinh/cosh overflow."""
    # Everything is written in e^(-r), so large r does not overflow. expm1
    # keeps 1 - e^(-2r) accurate for small r.
    exp_minus_r = torch.exp(-r)
    denominator = -torch.expm1(-2 * r)
    return 2 * exp_minus_r / denominator, (1 + exp_minus_r.square()) / denominator


def _odd_closed_form(
    n: int,
    t: torch.Tensor,
    d: torch.Tensor,
    spectral_decay: float | None,
) -> torch.Tensor:
    """Evaluate the H^3/H^5/H^7 operator-chain polynomials.

    Grigor'yan-Noguchi (1998), equation (1.4), gives the coefficients.
    Factoring out powers of csch(r) avoids overflowing hyperbolic powers;
    the small-radius series avoids cancellation in the remaining bracket.
    A spectral decay override goes into the same exponential as the Gaussian
    term, so the two are evaluated together.
    """
    small = _small_radius(n, d)
    # The closed form is 0/0 near r = 0, and torch.where still evaluates it.
    # Those lanes get r = 1 so the discarded branch stays finite. The series
    # below replaces them.
    r = torch.where(small, torch.ones_like(d), d)
    csch, coth = _csch_and_coth(r)
    if n == _N_DAVIES_MANDOUVALOS:
        ratio = r * csch
    elif n == _N_DAVIES_MANDOUVALOS_N5_CLOSED:
        ratio = csch.square() * (r.square() + 2 * t * (r * coth - 1))
    else:
        # n = 7 bracket: the operator-chain step after n = 5, derived from (1.4).
        three = _N_DAVIES_MANDOUVALOS
        six = 2 * three
        eight = 2 * _GAUSSIAN_TIME_SCALE
        twelve = _GAUSSIAN_TIME_SCALE * three
        ratio = csch**three * (
            r**three
            + six * r.square() * t * coth
            + (eight * t.square() - six * t) * r
            + twelve * t.square() * (r * csch.square() - coth)
        )
    # The series is in q = r^2. Outside the small band q is zero, and the
    # series output there is discarded.
    q = torch.where(small, d.square(), torch.zeros_like(d))
    ratio = torch.where(small, _odd_radial_series(n, t, q), ratio)
    # Default decay rho^2 with rho = (n-1)/2 is the bottom of the L^2 spectrum of H^n.
    decay = ((n - 1) / 2) ** 2 if spectral_decay is None else spectral_decay
    return (
        (_GAUSSIAN_TIME_SCALE * math.pi * t) ** (-n / 2)
        * torch.exp(
            -decay * t - d.square() / (_GAUSSIAN_TIME_SCALE * t),
        )
        * ratio
    )


def _heat_kernel_unit_n3(
    t: torch.Tensor,
    d: torch.Tensor,
    spectral_decay: float | None = None,
) -> torch.Tensor:
    """H^3: (4 pi t)^(-3/2) exp(-t-r^2/4t) r/sinh(r)."""
    return _odd_closed_form(_N_DAVIES_MANDOUVALOS, t, d, spectral_decay)


def _heat_kernel_unit_n5(
    t: torch.Tensor,
    d: torch.Tensor,
    spectral_decay: float | None = None,
) -> torch.Tensor:
    """H^5 closed form; its radial polynomial tends to 1+2t/3 at zero."""
    return _odd_closed_form(_N_DAVIES_MANDOUVALOS_N5_CLOSED, t, d, spectral_decay)


def _heat_kernel_unit_n7(
    t: torch.Tensor,
    d: torch.Tensor,
    spectral_decay: float | None = None,
) -> torch.Tensor:
    """H^7 closed form; its radial polynomial tends to 1+2t+16t^2/15."""
    return _odd_closed_form(_N_OPERATOR_CHAIN_N7_CLOSED, t, d, spectral_decay)


def _heat_kernel_unit_n2(
    t: torch.Tensor,
    d: torch.Tensor,
    n_quad: int = HEAT_KERNEL_QUADRATURE_NODES,
    tail_budget: float = HEAT_KERNEL_QUADRATURE_TAIL_BUDGET,
) -> torch.Tensor:
    """H^2 via Gauss-Legendre quadrature of equation (1.7).

    In s=r+u^2 coordinates, the upper u endpoint is (4*budget*t)^(1/4).
    Factor cosh(r+u^2)-cosh(r) as 2 sinh(r+u^2/2) sinh(u^2/2) to
    avoid cancellation. Close to zero, use the smooth descent integral
    so differentiating the quadrature does not create a spurious cusp.
    """
    original_d = d
    small = _small_radius(_N_DAVIES_MANDOUVALOS, d)
    # The square-root endpoint substitution has a nonuniform radial
    # derivative at zero. Use the smooth descent integral there instead.
    d = torch.where(small, torch.ones_like(d), d)
    # Gauss–Legendre nodes and weights come from the library-wide cache
    # (`_quadrature.gauss_legendre_nodes`), computed once per n_quad and cast
    # here. Nodes lie in (-1, 1). Since t may be batched, each element is
    # rescaled to u in (0, sqrt_S), and s = r + u^2 then runs up to
    # r + sqrt(4 * tail_budget * t).
    nodes, weights = gauss_legendre_nodes(
        n_quad,
        dtype=d.dtype,
        device=d.device,
    )

    # Broadcast t and d to a common shape and add a quadrature-node axis.
    t_b, d_b = torch.broadcast_tensors(t, d)
    sqrt_S = (_GAUSSIAN_TIME_SCALE * tail_budget * t_b).sqrt().sqrt()  # (...,)
    # Affine map from nodes in (-1, 1) onto u in (0, sqrt_S):
    #   u = (sqrt_S / 2) · (node + 1)
    #   du = (sqrt_S / 2) · d(node)
    u = 0.5 * sqrt_S.unsqueeze(-1) * (nodes + 1.0)  # (..., n_quad)
    jacobian = 0.5 * sqrt_S.unsqueeze(-1)  # (..., n_quad)

    # s = d + u²
    s = d_b.unsqueeze(-1) + u * u
    # f(s) = s · exp(-s²/4t)
    f_s = s * torch.exp(-s * s / (2.0 * 2.0 * t_b.unsqueeze(-1)))  # 4t
    # Denominator: √(cosh s − cosh d). At u = 0 (i.e. s = d) this is
    # zero; the change of variable absorbs the singularity in the
    # `2u` from the Jacobian. After the substitution the *transformed*
    # integrand `f(d+u²) · 2u / √(cosh(d+u²) − cosh d)` is bounded.
    half_u_squared = u.square() / 2
    sqrt_diff = (
        math.sqrt(2)
        * torch.sinh(d_b.unsqueeze(-1) + half_u_squared).sqrt()
        * torch.sinh(half_u_squared).sqrt()
    )
    # Transformed integrand: f(d+u²) · 2u / √(cosh(d+u²) − cosh d)
    # (the `2u du` from `ds = 2u du`)
    integrand = f_s * 2.0 * u / sqrt_diff
    # Numerical integration: Σ weights · jacobian · integrand
    integral = (weights * jacobian * integrand).sum(dim=-1)

    # Prefactor of equation (1.7): sqrt(2) exp(-t/4) (4 pi t)^(-3/2).
    prefactor = math.sqrt(2.0) * torch.exp(-t_b / (2.0 * 2.0))
    prefactor = prefactor / (_GAUSSIAN_TIME_SCALE * math.pi * t_b) ** (
        _N_DAVIES_MANDOUVALOS / 2
    )
    regular = prefactor * integral
    # The descent branch needs real radii only where small is true. Elsewhere it
    # gets zero, which keeps it finite. The where below discards its output there.
    small_d = torch.where(small, original_d, torch.zeros_like(original_d))
    smooth = _even_descent(_N_GRIGORYAN_INTEGRAL, t, small_d, n_quad, tail_budget)
    return torch.where(small, smooth, regular)


def _even_descent(
    n: int,
    t: torch.Tensor,
    d: torch.Tensor,
    n_quad: int,
    tail_budget: float,
) -> torch.Tensor:
    """Smooth radial evaluation from the next odd-dimensional kernel.

    Grigor'yan-Noguchi (1998), equations (1.6)-(1.8), imply
    k_n(r) = sqrt(2) exp((2n-1)t/4) integral_r^infinity
             k_(n+1)(s) sinh(s) / sqrt(cosh(s)-cosh(r)) ds.
    The Abel integral in x=cosh(r) commutes with d/dx, so its H^2/H^3
    identity extends to each even/odd pair by dimension raising.

    Substitute s=sqrt(r^2+u^2). Factoring the cosh difference gives
    the weight below, which is smooth in r^2 even at r=0. The odd
    kernel already has a small-radius series. Quadrature nodes avoid
    u=0, so all three sinhc ratios have strictly positive arguments.
    """
    nodes, weights = gauss_legendre_nodes(n_quad, dtype=d.dtype, device=d.device)
    t_b, d_b = torch.broadcast_tensors(t, d)
    upper = (_GAUSSIAN_TIME_SCALE * tail_budget * t_b).sqrt().unsqueeze(-1)
    u = upper * (nodes + 1) / 2
    r = d_b.unsqueeze(-1)
    s = (r.square() + u.square()).sqrt()
    plus = (s + r) / 2
    # s-r = u^2/(s+r) avoids subtraction when r dominates u.
    minus = u.square() / (2 * (s + r))

    # log(sinh(x)/x) = x + log(1 - e^(-2x)) - log(2x). expm1 keeps the middle
    # term accurate for small x, and nothing overflows for large x.
    def log_sinhc(x: torch.Tensor) -> torch.Tensor:
        return x + torch.log(-torch.expm1(-2 * x)) - torch.log(2 * x)

    # This equals sinh(s)/s · u / (2 sqrt(sinh(plus) sinh(minus))), because
    # plus · minus = u^2 / 4.
    weight = torch.exp(log_sinhc(s) - (log_sinhc(plus) + log_sinhc(minus)) / 2)
    # The Abel prefactor exp((2n-1)t/4) and the odd kernel's exp(-n^2 t/4)
    # combine into one exponential with decay (n-1)^2/4. Two separate ones
    # could underflow where the even kernel is still representable.
    odd = _heat_kernel_unit(
        n + 1,
        t_b.unsqueeze(-1),
        s,
        n_quad,
        tail_budget,
        spectral_decay=((n - 1) / 2) ** 2,
    )
    return (weights * upper * odd * weight).sum(dim=-1)


def _apply_one_recursion(
    prev_kernel_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    n_prev: int,
    t: torch.Tensor,
    d: torch.Tensor,
    *,
    apply_spectral_shift: bool = True,
) -> torch.Tensor:
    """Apply one step of the Grigor'yan recursion.

    The recursion, derived from the operator chain on the
    spectrally-shifted Gaussian (validation results in
    `notes/validation/heat_kernel_results.md`), is

        k^{n_prev + 2}(t, r) = -exp(-n_prev · t) / (2π · sinh r)
                                · ∂_r k^{n_prev}(t, r).

    The `exp(-n_prev · t)` factor is required. Heat kernels on
    `H^{n}` carry a spectral-bottom factor `exp(-((n-1)/2)²·t)`; going
    from `n_prev` to `n_prev+2` shifts the spectral bottom by
    `((n_prev+1)/2)² − ((n_prev-1)/2)² = n_prev`, so we multiply by
    `exp(-n_prev · t)`. The heat-equation-residual check in
    `notes/validation/heat_kernel_validation.py` depends on it: without
    the factor, the residual is O(1) for n ≥ 5 instead of float noise.

    Args:
      prev_kernel_fn: function `(t, d) -> k^{n_prev}(t, d)`.
      n_prev: the dimension `n_prev` of `prev_kernel_fn`'s output.
      t, d: time and radial distance tensors.
      apply_spectral_shift: false when the seed already includes the
        final dimension's spectral exponential, so only radial raising remains.

    Returns:
      `k^{n_prev + 2}(t, d)`, heat kernel two dimensions higher.
    """
    # Read the caller's grad mode before enable_grad() turns it on.
    keep_graph = torch.is_grad_enabled() and (t.requires_grad or d.requires_grad)
    with torch.enable_grad():
        # Differentiate each broadcast pair independently. Differentiating the
        # original scalar d instead would sum derivatives across the time batch.
        t_b, d_b = torch.broadcast_tensors(t, d)
        # A fresh leaf when the caller gave no grad-tracked d. A d already in a
        # graph is kept, so the outer graph survives.
        if not d_b.requires_grad:
            d_b = d_b.detach().requires_grad_(True)
        kn = prev_kernel_fn(t_b, d_b)
        # kn.sum() is safe because each element depends only on its own d, so
        # the gradient is elementwise.
        (dk_dd,) = torch.autograd.grad(kn.sum(), d_b, create_graph=True)
        # At d = 0 the csch factor is singular. Those lanes take a finite
        # placeholder and then get the limit below.
        nonzero = d_b != 0
        safe_d = torch.where(nonzero, d_b, torch.ones_like(d_b))
        csch, _ = _csch_and_coth(safe_d)
        radial_derivative = dk_dd * csch
        # At d = 0, k'(d)/sinh(d) tends to k''(0) by L'Hopital's rule.
        (d2k_dd2,) = torch.autograd.grad(dk_dd.sum(), d_b, create_graph=True)
        radial_derivative = torch.where(nonzero, radial_derivative, d2k_dd2)
        shift = torch.exp(-n_prev * t_b) if apply_spectral_shift else 1
        result = -shift * radial_derivative / (2 * math.pi)
    # Detach unless the inputs needed a graph, so the recursion does not keep
    # graphs nobody asked for.
    return result if keep_graph else result.detach()


def _heat_kernel_unit(
    n: int,
    t: torch.Tensor,
    d: torch.Tensor,
    n_quad: int,
    tail_budget: float,
    spectral_decay: float | None = None,
) -> torch.Tensor:
    """Dispatch closed forms, quadrature, and dimension-raising recurrence.

    H^5/H^7 and all higher odd dimensions use the even-in-r series
    near zero. Even dimensions use dimension descent there because
    repeated radial differentiation amplifies the quadrature error.
    """
    decay = ((n - 1) / 2) ** 2 if spectral_decay is None else spectral_decay
    if n == 1:
        return _heat_kernel_unit_n1(t, d)
    if n == _N_GRIGORYAN_INTEGRAL:
        return _heat_kernel_unit_n2(t, d, n_quad, tail_budget)
    if n == _N_DAVIES_MANDOUVALOS:
        return _heat_kernel_unit_n3(t, d, decay)
    if n == _N_DAVIES_MANDOUVALOS_N5_CLOSED:
        return _heat_kernel_unit_n5(t, d, decay)
    if n == _N_OPERATOR_CHAIN_N7_CLOSED:
        return _heat_kernel_unit_n7(t, d, decay)

    # Equation (1.8) applies to both parities. Closure construction is
    # sequential in dimension; each evaluation stays fully batched.
    # The recursion starts from the n = 7 closed form (odd n) or the n = 2
    # integral (even n), then raises the dimension by two until it reaches n.
    current_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    if n % 2 == 1:
        current_n = _N_OPERATOR_CHAIN_N7_CLOSED

        def odd_seed(_t: torch.Tensor, _d: torch.Tensor) -> torch.Tensor:
            return _heat_kernel_unit_n7(_t, _d, decay)

        current_fn = odd_seed
    else:
        current_n = _N_GRIGORYAN_INTEGRAL

        def integral_seed(_t: torch.Tensor, _d: torch.Tensor) -> torch.Tensor:
            return _heat_kernel_unit_n2(_t, _d, n_quad, tail_budget)

        current_fn = integral_seed

    def raise_dimension(
        previous: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        dimension: int,
    ) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        def raised(_t: torch.Tensor, _d: torch.Tensor) -> torch.Tensor:
            return _apply_one_recursion(
                previous,
                dimension,
                _t,
                _d,
                # Odd targets: the seed already carries the final decay, so each
                # step is radial only. Even targets: each step adds exp(-n_prev t).
                apply_spectral_shift=(n % 2 == 0),
            )

        return raised

    while current_n < n:
        current_fn = raise_dimension(current_fn, current_n)
        current_n += 2

    # Even n: the recursion divides by sinh near zero, so the small band uses
    # the smooth descent integral and the rest uses the recursion.
    if n % 2 == 0:
        small = _small_radius(n, d)
        safe_d = torch.where(small, torch.ones_like(d), d)
        regular = current_fn(t, safe_d)
        small_d = torch.where(small, d, torch.zeros_like(d))
        smooth = _even_descent(n, t, small_d, n_quad, tail_budget)
        return torch.where(small, smooth, regular)
    small = _small_radius(n, d)
    safe_d = torch.where(small, torch.ones_like(d), d)
    regular = current_fn(t, safe_d)
    # Odd n: the small band uses the even-in-r series, which is smooth at zero.
    q = torch.where(small, d.square(), torch.zeros_like(d))
    series = (
        (_GAUSSIAN_TIME_SCALE * math.pi * t) ** (-n / 2)
        * torch.exp(
            -decay * t - q / (_GAUSSIAN_TIME_SCALE * t),
        )
        * _odd_radial_series(n, t, q)
    )
    return torch.where(small, series, regular)


def hyperbolic_heat_kernel(
    t: torch.Tensor,
    distances: torch.Tensor,
    manifold: _HyperbolicManifold,
    n_quad: int = HEAT_KERNEL_QUADRATURE_NODES,
    tail_budget: float = HEAT_KERNEL_QUADRATURE_TAIL_BUDGET,
) -> torch.Tensor:
    """Heat kernel `k^n_t(d)` on the hyperbolic manifold `manifold`.

    Computes the probability density of a Brownian-motion particle at
    geodesic distance `d` after time `t`, starting from a delta source
    at the manifold origin.

    The kernel depends only on `t`, the geodesic distance `d`, and the
    intrinsic dimension `n` of the manifold (rotational symmetry
    around the source). Curvature scales out: for curvature `k = -|k|`,

        k^n_{−|k|, t}(d)  =  |k|^{n/2} · k^n_{−1, |k|·t}(√|k| · d).

    Args:
      t: positive time(s). Broadcastable with `distances`. Typically
        a scalar, but `(B,)` or matching `distances.shape` are also
        supported.
      distances: non-negative geodesic distance(s). Any shape, must
        be broadcastable with `t`.
      manifold: a manifold object exposing `.n` (intrinsic dim) and
        `.k` (sectional curvature, expected `k < 0`).
      n_quad: number of Gauss–Legendre nodes for the quadrature: the n=2
        integral and the even-dimension descent. Default 32; cataloged as
        `HEAT_KERNEL_QUADRATURE_NODES`.
      tail_budget: integration-upper-bound budget factor (truncate at
        `d + sqrt(4 * tail_budget * t)`). Default 20. Cataloged.
        The small-radius descent integral ends at
        `sqrt(d**2 + 4 * tail_budget * t)` instead.

    Returns:
      Tensor of heat-kernel values, same shape as
      `broadcast(t, distances)`.

    Dimension support:
      n=1 is Gaussian; n=2 uses quadrature; n=3,5,7 have closed forms.
      Remaining dimensions use the spectral-shifted recurrence. Near
      zero, odd dimensions use a series and even dimensions use a smooth
      descent integral. Gradients flow through `t` and `distances`. Under
      `torch.no_grad()` the same values come back with no graph kept.
      Quadrature accuracy depends on n_quad and tail_budget; very high
      dimensions, extreme times and large radii can still exceed
      floating-point range.

    References:
      Davies–Mandouvalos (1988); Grigor'yan (2009), Theorem 8.21;
      Grigor'yan–Noguchi (1998); Anker–Ostellari (2003).
    """
    n = manifold.n
    abs_k = abs(manifold.k)

    # Curvature scaling: convert to unit-curvature arguments, evaluate,
    # then rescale the output by |k|^{n/2}. abs(k) only takes the magnitude,
    # so k must be negative here; a k >= 0 is not checked.
    t_unit = abs_k * t
    sqrt_abs_k = math.sqrt(abs_k)
    d_unit = sqrt_abs_k * distances

    # A Python float t makes t_unit a float. Turn it into a tensor with the
    # distances' dtype and device before the kernel runs.
    if not isinstance(t_unit, torch.Tensor):
        t_unit = torch.as_tensor(
            t_unit,
            dtype=distances.dtype,
            device=distances.device,
        )
    k_unit = _heat_kernel_unit(n, t_unit, d_unit, n_quad, tail_budget)
    scale = abs_k ** (n * 0.5)
    return scale * k_unit
