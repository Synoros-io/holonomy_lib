# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Synoros

"""Model-space (constant-curvature) geodesic ball and sphere volumes.

The simply-connected space form `M_κ^N` of constant sectional curvature `κ` and
real (possibly non-integer) dimension `N` is the comparison reference for
Bishop-Gromov volume comparison. Its geodesic sphere of radius `r` has surface
area

    S(κ, N, r) = ω_{N-1} · sn_κ(r)^{N-1},

and its geodesic ball has volume

    V(κ, N, r) = ω_{N-1} · ∫_0^r sn_κ(t)^{N-1} dt,

where `ω_{N-1} = 2 π^{N/2} / Γ(N/2)` is the surface area of the unit
`(N-1)`-sphere (well-defined for real `N` via the Gamma function), and `sn_κ`
is the generalized sine

    sn_κ(t) = sin(√κ · t) / √κ          (κ > 0, spherical)
            = t                          (κ = 0, flat)
            = sinh(√(-κ) · t) / √(-κ)    (κ < 0, hyperbolic).

The volume integral is closed-form for integer `N-1` but has no elementary form
for general real `N`; it is evaluated by Gauss-Legendre quadrature. The
integrand can have endpoint singular derivatives for noninteger `N`, so
quadrature then converges algebraically. For `κ = 0` and integer `N`, the
integrand is a polynomial, exact when `N <= 2 * n_quad`.

These are the model-space (comparison) quantities: on a general space with a
Ricci lower bound `Ric ≥ (N-1) κ` they bound the true volume from above
(Bishop-Gromov). Useful for curvature- and dimension-dependent flux / decay
laws.

References:
  Bishop, R. L., Crittenden, R. J. (1964). Geometry of Manifolds. Academic
    Press, Ch. 11 (volume comparison).
  Petersen, P. (2016). Riemannian Geometry, 3rd ed. Springer, §7.1
    (Bishop-Gromov; the `sn_κ` model-space volume element).
  Dai, X., Wei, G. (2019). Comparison Geometry for Ricci Curvature. Lecture
    notes, §1.2, eq. (1.2.5) (model volume element using `sn_H^{n-1}`).
"""

from __future__ import annotations

import math

import torch

from holonomy_lib._quadrature import gauss_legendre_nodes

# Existing quadrature default, cataloged in notes/magic_numbers.md.
# For flat integer N the degree-(N-1) polynomial is integrated exactly when
# N <= 2 * nodes. Fractional N can have endpoint singular derivatives and
# requires a convergence check at the caller's desired accuracy.
COMPARISON_BALL_VOLUME_QUADRATURE_NODES: int = 64


def _prepare(
    kappa: torch.Tensor, N: torch.Tensor, r: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Coerce `(kappa, N, r)` to a common float dtype/device and broadcast.

    dtype/device are inferred from the first floating-point tensor argument,
    defaulting to `float64` / `cpu` (CONVENTIONS §1.2).
    """
    floats = [v for v in (kappa, N, r)
              if isinstance(v, torch.Tensor) and v.is_floating_point()]
    dtype = floats[0].dtype if floats else torch.float64
    device = floats[0].device if floats else torch.device("cpu")
    kappa = torch.as_tensor(kappa, dtype=dtype, device=device)
    N = torch.as_tensor(N, dtype=dtype, device=device)
    r = torch.as_tensor(r, dtype=dtype, device=device)
    try:
        shape = torch.broadcast_shapes(kappa.shape, N.shape, r.shape)
    except RuntimeError as exc:
        raise ValueError(
            f"kappa {tuple(kappa.shape)}, N {tuple(N.shape)}, r "
            f"{tuple(r.shape)} are not broadcastable"
        ) from exc
    return kappa.broadcast_to(shape), N.broadcast_to(shape), r.broadcast_to(shape)


def _validate(kappa: torch.Tensor, N: torch.Tensor, r: torch.Tensor) -> None:
    """Value-range checks (CONVENTIONS §1.3). `.any()` is False on an empty
    batch, so `B = 0` passes."""
    if not torch.isfinite(kappa).all() or not torch.isfinite(N).all():
        raise ValueError("kappa and N must be finite")
    if (N < 1).any():
        raise ValueError("N must be >= 1 (effective dimension)")
    if (r < 0).any():
        raise ValueError("r must be >= 0 (geodesic radius)")
    if not torch.isfinite(r).all():
        raise ValueError("r must be finite")
    positive = kappa > 0
    if positive.any():
        tiny = torch.finfo(kappa.dtype).tiny
        sqrt_k = torch.sqrt(kappa.clamp(min=tiny))
        if (positive & (r > math.pi / sqrt_k)).any():
            raise ValueError(
                "for kappa > 0, r must be <= pi / sqrt(kappa) "
                "(the model sphere's diameter)"
            )


# From sin(z)/z = 1 - z²/3! + z⁴/5! + O(z⁶), continued in signed κ.
SN_LINEAR = 1.0 / 6.0
SN_QUADRATIC = 1.0 / 120.0


def _sn_ratio(kappa: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """sn_κ(t)/t, with its analytic signed-κ expansion at zero."""
    # Scaling before the second multiplication preserves κ=0 at radii
    # whose square overflows, and tiny curvatures with moderate κt².
    q = (kappa * t) * t
    small = torch.isfinite(q) & (q.abs() <= torch.finfo(t.dtype).eps**0.5)
    # Mask before multiplying in the differentiable series path as well:
    # a zero upstream gradient times an overflowing κt would otherwise be NaN.
    series_q = (torch.where(small, kappa, torch.zeros_like(kappa)) * t) * t
    # The curvature supplies the branch even if κt² overflows. Taking its
    # square root first also lets the angle remain representable in that case.
    positive = (kappa > 0) & ~small
    negative = (kappa < 0) & ~small
    spherical_arg = (
        torch.sqrt(torch.where(positive, kappa, torch.ones_like(kappa)))
        * torch.where(positive, t.abs(), torch.ones_like(t))
    )
    hyperbolic_arg = (
        torch.sqrt(torch.where(negative, -kappa, torch.ones_like(kappa)))
        * torch.where(negative, t.abs(), torch.ones_like(t))
    )
    spherical = torch.sin(spherical_arg) / spherical_arg
    hyperbolic = torch.sinh(hyperbolic_arg) / hyperbolic_arg
    series = 1.0 - SN_LINEAR * series_q + SN_QUADRATIC * series_q.square()
    return torch.where(small, series, torch.where(positive, spherical, hyperbolic))


def _log_sn_ratio(kappa: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Log of the positive model sine ratio at interior quadrature nodes."""
    q = (kappa * t) * t
    small = torch.isfinite(q) & (q.abs() <= torch.finfo(t.dtype).eps**0.5)
    hyperbolic = (kappa < 0) & ~small
    # log(sinh(z)/z) = z + log(1-exp(-2z)) - log(2) - log(z).
    # This avoids forming sinh(z), which can overflow before the volume's
    # radius and sphere-area factors bring the answer back into range.
    angle = (
        torch.sqrt(torch.where(hyperbolic, -kappa, torch.ones_like(kappa)))
        * torch.where(hyperbolic, t, torch.ones_like(t))
    )
    hyperbolic_log = (
        angle + torch.log(-torch.expm1(-2.0 * angle))
        - math.log(2.0) - torch.log(angle)
    )
    other_log = torch.log(_sn_ratio(
        torch.where(hyperbolic, torch.zeros_like(kappa), kappa), t,
    ))
    return torch.where(hyperbolic, hyperbolic_log, other_log)


def sn_kappa(kappa: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Generalized sine, differentiable in signed curvature through zero.

    sn_κ(t) = sin(√κ t)/√κ for κ > 0, t for κ = 0, and
    sinh(√(-κ) t)/√(-κ) for κ < 0. Near zero the signed-curvature
    series starts t - κt³/6; in particular ∂κ sn_0(t) = -t³/6.

    Args:
      kappa: floating curvature tensor, any sign, broadcastable with t.
      t: floating radial coordinate tensor; shape may include leading batches.
        Both tensors must share a device and compatible floating dtypes.
    Returns:
      Tensor with the broadcast shape of kappa and t, preserving their dtype.

    References:
      Petersen (2016), Riemannian Geometry, §7.1, model-space Jacobi fields.
    """
    kappa, t = torch.broadcast_tensors(kappa, t)
    return t * _sn_ratio(kappa, t)


def _model_sn(kappa: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """Nonnegative model sine on the validated closed radial interval."""
    sqrt_positive = torch.sqrt(torch.where(kappa > 0, kappa, torch.ones_like(kappa)))
    endpoint = (kappa > 0) & (r == math.pi / sqrt_positive)
    return torch.where(endpoint, torch.zeros_like(r), sn_kappa(kappa, r)).clamp_min(0)


def _unit_sphere_surface_area(N: torch.Tensor) -> torch.Tensor:
    """`ω_{N-1} = 2 π^{N/2} / Γ(N/2)`, surface area of the unit `(N-1)`-sphere
    in `R^N`. Computed in log space via `lgamma` for real `N`."""
    return 2.0 * torch.exp((N / 2.0) * math.log(math.pi) - torch.lgamma(N / 2.0))


def model_sphere_area(
    kappa: torch.Tensor, N: torch.Tensor, r: torch.Tensor,
) -> torch.Tensor:
    """Surface area of the geodesic sphere of radius `r` in the space form
    `M_κ^N`: `S = ω_{N-1} · sn_κ(r)^{N-1}`.

    Args:
      kappa: `(B,)` constant sectional curvature per element (any sign).
      N: `(B,)` or scalar real effective dimension, `N >= 1`.
      r: `(B,)` geodesic radius, `r >= 0` (and `r <= π/√κ` where `κ > 0`).
    Returns:
      `(B,)` geodesic-sphere surface area.

    References:
      Dai-Wei (2019), §1.2 eq. (1.2.5).
    """
    kappa, N, r = _prepare(kappa, N, r)
    _validate(kappa, N, r)
    return _unit_sphere_surface_area(N) * _model_sn(kappa, r) ** (N - 1.0)


def model_ball_volume(
    kappa: torch.Tensor, N: torch.Tensor, r: torch.Tensor,
    n_quad: int = COMPARISON_BALL_VOLUME_QUADRATURE_NODES,
) -> torch.Tensor:
    """Volume of the geodesic ball of radius `r` in the space form `M_κ^N`:
    `V = ω_{N-1} · ∫_0^r sn_κ(t)^{N-1} dt`.

    The integral is evaluated in log space by `n_quad`-point Gauss-Legendre quadrature
    (exact for flat integer dimensions with `N <= 2 * n_quad`).
    Noninteger dimensions can have endpoint singular derivatives, so their
    quadrature convergence is algebraic rather than geometric.

    Args:
      kappa: `(B,)` constant sectional curvature per element (any sign).
      N: `(B,)` or scalar real effective dimension, `N >= 1`.
      r: `(B,)` geodesic radius, `r >= 0` (and `r <= π/√κ` where `κ > 0`).
      n_quad: Gauss-Legendre node count (cataloged default).
    Returns:
      `(B,)` geodesic-ball volume.

    References:
      Sturm (2006), Acta Math 196, model-space volume; Petersen (2016) §7.1.
    """
    kappa, N, r = _prepare(kappa, N, r)
    _validate(kappa, N, r)

    nodes, weights = gauss_legendre_nodes(
        n_quad, dtype=kappa.dtype, device=kappa.device,
    )

    # Keep every factor in log space: r^N and the curvature-dependent
    # integrand can each overflow or underflow while their product is finite.
    # The exact origin uses its analytic first derivative, including N=1.
    nonzero = r > 0
    safe_r = torch.where(nonzero, r, torch.ones_like(r))
    safe_kappa = torch.where(nonzero, kappa, torch.zeros_like(kappa))
    unit_nodes = 0.5 * (nodes + 1.0)
    t = safe_r.unsqueeze(-1) * unit_nodes
    log_ratio = torch.log(unit_nodes) + _log_sn_ratio(safe_kappa.unsqueeze(-1), t)
    log_integral = torch.logsumexp(
        torch.log(weights) + (N.unsqueeze(-1) - 1.0) * log_ratio, dim=-1,
    ) - math.log(2.0)
    log_area = math.log(2.0) + (N / 2.0) * math.log(math.pi) - torch.lgamma(N / 2.0)
    volume = torch.exp(log_area + N * torch.log(safe_r) + log_integral)
    origin = torch.where(N == 1.0, 2.0 * r, torch.zeros_like(r))
    return torch.where(nonzero, volume, origin)


def model_anisotropic_flux(
    kappas: torch.Tensor, r: torch.Tensor,
) -> torch.Tensor:
    """Anisotropic geodesic flux: the volume element at radius `r` of a space whose
    `K` principal sectional curvatures are `kappas`,

        flux = Π_i sn_{κ_i}(r),

    the product of the per-direction generalized sines. This is the anisotropic
    generalization of the isotropic geodesic-sphere factor `sn_κ(r)^{N-1}`; which
    it recovers when all `kappas` are equal (the same factor repeated); but here
    each principal direction contributes its OWN `sn_{κ_i}(r)`. Use as a
    curvature-matched flux for a force-decay driven by a per-direction curvature
    TENSOR, where a single scalar κ would collapse the directional structure.

    Args:
      kappas: `(B, K)` principal sectional curvatures (any sign), `K` directions.
      r: `(B,)` geodesic radius, `r >= 0` (and `r <= π/√κ_i` per spherical direction).
    Returns:
      `(B,)` anisotropic flux `Π_i sn_{κ_i}(r)`.

    References:
      Petersen, P. (2016). Riemannian Geometry, 3rd ed. Springer, §7.1; the
      `sn_κ` volume element, taken here per principal direction (product over the
      anisotropic frame).
    """
    kappas = torch.as_tensor(kappas)
    dtype = kappas.dtype if kappas.is_floating_point() else torch.float64
    kappas = kappas.to(dtype)
    r = torch.as_tensor(r, dtype=dtype, device=kappas.device)
    if kappas.ndim != 2:
        raise ValueError(f"kappas must be (B, K); got {tuple(kappas.shape)}")
    if r.ndim != 1 or r.shape[0] != kappas.shape[0]:
        raise ValueError(
            f"r must be (B,) matching the kappas batch B={kappas.shape[0]}; "
            f"got {tuple(r.shape)}"
        )
    if (r < 0).any() or not torch.isfinite(r).all():
        raise ValueError("r must be finite and >= 0")
    positive = kappas > 0
    if positive.any():
        tiny = torch.finfo(dtype).tiny
        sqrt_k = torch.sqrt(kappas.clamp(min=tiny))
        if (positive & (r.unsqueeze(-1) > math.pi / sqrt_k)).any():
            raise ValueError(
                "for kappa > 0, r must be <= pi / sqrt(kappa) per direction "
                "(that direction's model-sphere diameter)"
            )
    sn = _model_sn(kappas, r.unsqueeze(-1))            # (B, K)
    return sn.prod(dim=-1)                              # (B,)
