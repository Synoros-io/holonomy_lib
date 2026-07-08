# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Synoros

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
integrand is smooth and analytic on `[0, r]`, so the quadrature converges
geometrically; for `κ = 0` the integrand `t^{N-1}` is a polynomial and the rule
is exact.

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
from holonomy_lib.provenance import with_provenance

# Default Gauss-Legendre node count for the model ball-volume integral
# `∫_0^r sn_κ(t)^{N-1} dt`. The integrand is smooth/analytic on a finite
# interval, so Gauss-Legendre converges geometrically; for `κ = 0` (integrand a
# degree-`(N-1)` polynomial) the rule is exact for `N ≤ 2 · nodes`. 64 nodes
# give agreement below 1e-12 with the closed forms (Euclidean, sphere,
# hyperbolic) for `N` up to ~100 and `√κ · r` up to the model diameter
# (validated by `tests/manifolds/test_comparison.py`). Catalog:
# `COMPARISON_BALL_VOLUME_QUADRATURE_NODES` in `notes/magic_numbers.md`.
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


def _sn_kappa(kappa: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Generalized sine `sn_κ(t)`: `sin(√κ t)/√κ` (κ>0), `t` (κ=0),
    `sinh(√-κ t)/√-κ` (κ<0). One `torch.where` branch on `sign(κ)`; the abs is
    clamped at `finfo.tiny` so the `κ = 0` branch never divides by zero."""
    sqrt_abs_k = torch.sqrt(kappa.abs().clamp(min=torch.finfo(kappa.dtype).tiny))
    arg = sqrt_abs_k * t
    spherical = torch.sin(arg) / sqrt_abs_k
    hyperbolic = torch.sinh(arg) / sqrt_abs_k
    return torch.where(
        kappa > 0, spherical, torch.where(kappa < 0, hyperbolic, t),
    )


# Public alias: the generalized sine is a reusable primitive. A caller may need the
# PER-DIRECTION `sn_{κ_i}(r)` (not just the product) to build sign-split / masked
# anisotropic fluxes -- e.g. routing spherical (κ>0) vs hyperbolic (κ<0) directions to
# different force terms. Batched `(B, K)` kappa, `(B, 1)` r -> `(B, K)`.
sn_kappa = _sn_kappa


def _unit_sphere_surface_area(N: torch.Tensor) -> torch.Tensor:
    """`ω_{N-1} = 2 π^{N/2} / Γ(N/2)`, surface area of the unit `(N-1)`-sphere
    in `R^N`. Computed in log space via `lgamma` for real `N`."""
    return 2.0 * torch.exp((N / 2.0) * math.log(math.pi) - torch.lgamma(N / 2.0))


@with_provenance(
    "holonomy_lib.manifolds.comparison.model_sphere_area", op_version="0.1",
)
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
    return _unit_sphere_surface_area(N) * _sn_kappa(kappa, r) ** (N - 1.0)


@with_provenance(
    "holonomy_lib.manifolds.comparison.model_ball_volume", op_version="0.1",
)
def model_ball_volume(
    kappa: torch.Tensor, N: torch.Tensor, r: torch.Tensor,
    n_quad: int = COMPARISON_BALL_VOLUME_QUADRATURE_NODES,
) -> torch.Tensor:
    """Volume of the geodesic ball of radius `r` in the space form `M_κ^N`:
    `V = ω_{N-1} · ∫_0^r sn_κ(t)^{N-1} dt`.

    The integral is evaluated by `n_quad`-point Gauss-Legendre quadrature
    (exact for `κ = 0`; geometrically convergent otherwise).

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

    # Map nodes ∈ (-1, 1) → t ∈ (0, r): t = ½·r·(node + 1), Jacobian = ½·r.
    half_r = 0.5 * r.unsqueeze(-1)                          # (..., 1)
    t = half_r * (nodes + 1.0)                              # (..., n_quad)
    integrand = _sn_kappa(kappa.unsqueeze(-1), t) ** (N.unsqueeze(-1) - 1.0)
    integral = (weights * integrand).sum(dim=-1) * (0.5 * r)
    return _unit_sphere_surface_area(N) * integral


@with_provenance(
    "holonomy_lib.manifolds.comparison.model_anisotropic_flux", op_version="0.1",
)
def model_anisotropic_flux(
    kappas: torch.Tensor, r: torch.Tensor,
) -> torch.Tensor:
    """Anisotropic geodesic flux: the volume element at radius `r` of a space whose
    `K` principal sectional curvatures are `kappas`,

        flux = Π_i sn_{κ_i}(r),

    the product of the per-direction generalized sines. This is the anisotropic
    generalization of the isotropic geodesic-sphere factor `sn_κ(r)^{N-1}` — which
    it recovers when all `kappas` are equal (the same factor repeated) — but here
    each principal direction contributes its OWN `sn_{κ_i}(r)`. Use as a
    curvature-matched flux for a force-decay driven by a per-direction curvature
    TENSOR, where a single scalar κ would collapse the directional structure.

    Args:
      kappas: `(B, K)` principal sectional curvatures (any sign), `K` directions.
      r: `(B,)` geodesic radius, `r >= 0` (and `r <= π/√κ_i` per spherical direction).
    Returns:
      `(B,)` anisotropic flux `Π_i sn_{κ_i}(r)`.

    References:
      Petersen, P. (2016). Riemannian Geometry, 3rd ed. Springer, §7.1 — the
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
    sn = _sn_kappa(kappas, r.unsqueeze(-1))            # (B, K)
    return sn.prod(dim=-1)                              # (B,)
