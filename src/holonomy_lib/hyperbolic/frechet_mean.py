# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Synoros

"""Fréchet (Karcher) mean on a Riemannian manifold.

The Fréchet mean of a finite set of points `{p_1, …, p_N}` on a
manifold `M` with weights `w_i ≥ 0` is the point that minimizes the
weighted sum of squared geodesic distances:

    μ*  =  argmin_{μ ∈ M}  Σ_i w_i · d_M(μ, p_i)².

On a Hadamard manifold (simply connected, non-positive sectional
curvature — includes hyperbolic spaces of any negative curvature),
the objective is geodesically convex and the minimizer is unique.
Karcher (1977) showed that the gradient descent

    μ_{t+1}  =  exp_{μ_t}( Σ_i w_i · log_{μ_t}(p_i) / Σ_i w_i )

converges to `μ*` at a linear rate. The Riemannian gradient of
`d_M(μ, p)²` w.r.t. `μ` is `−2 · log_μ(p)`, so the inner sum is
exactly the gradient step direction (up to a factor of `2/Σ w_i`,
absorbed in the implicit step size `1`).

References:
  Karcher, H. (1977). Riemannian center of mass and mollifier
    smoothing. Comm. Pure Appl. Math. 30(5):509–541.
  Afsari, B. (2011). Riemannian Lᵖ center of mass: existence,
    uniqueness, and convexity. Proc. AMS 139(2):655–673.
  Pennec, X. (2006). Intrinsic statistics on Riemannian manifolds.
    J. Math. Imaging Vision 25(1):127–154, §4 (intrinsic mean).
"""

from __future__ import annotations

import torch

from holonomy_lib.provenance import with_provenance


@with_provenance(
    "holonomy_lib.hyperbolic.frechet_mean", op_version="0.1",
)
def frechet_mean(
    points: torch.Tensor,
    manifold,
    weights: torch.Tensor | None = None,
    max_iter: int = 100,
    tol: float = 1e-9,
) -> torch.Tensor:
    """Karcher-iteration Fréchet mean of `points` on `manifold`.

    Args:
      points: `(B, N, ambient_dim)` — `N` points per batch on the
        manifold. For `LorentzManifold(n)`, `ambient_dim = n + 1`.
      manifold: object exposing `.log(x, y)`, `.exp(x, v)`,
        `.norm(x, v)` (e.g. `LorentzManifold`, `SPDManifold`).
      weights: optional `(B, N)` non-negative weights. If `None`,
        uniform weights `1/N` are used. Weights are normalized to sum
        to 1 along the `N` axis before use.
      max_iter: hard cap on Karcher iterations. Default 100 — a
        comfortable upper bound for the linear convergence rate on
        Hadamard manifolds; typical inputs reach `tol` in 5–30 steps.
      tol: stop when the Riemannian norm of the update tangent is
        below this floor across all batches. Default 1e-9 (the
        library's `numerical_floor_convention`).

    Returns:
      `(B, ambient_dim)` Fréchet mean per batch.

    Example:
      >>> from holonomy_lib.manifolds import LorentzManifold
      >>> mfd = LorentzManifold(n=3)
      >>> P = mfd.random_point(batch_size=10).unsqueeze(0)  # (1, 10, 4)
      >>> mu = frechet_mean(P, mfd)
      >>> mu.shape
      torch.Size([1, 4])

    Notes:
      Existence and uniqueness of the minimizer hold on Hadamard
      manifolds (Karcher 1977; Afsari 2011); hyperbolic spaces of any
      negative curvature are Hadamard, so this primitive is well-
      defined on `LorentzManifold` for all `k < 0` and on
      `KappaStereographicManifold(κ < 0)`.

      **Limitation on positive curvature.** For
      `KappaStereographicManifold(κ > 0)` (spherical) the manifold is
      NOT Hadamard — Karcher's convergence guarantee requires inputs
      lie within the injectivity radius `π/√κ` from each other. For
      well-spread spherical inputs the iteration may converge to a
      local optimum or fail to converge. The implementation runs
      unconditionally; the caller is responsible for checking the
      input spread.

      Per-batch convergence is checked against the **max** update norm
      across the batch, not each batch element separately — so a slow-
      converging element delays the loop for the whole batch. This is
      the standard cost of vectorized iterative algorithms.

    References:
      Karcher (1977); Pennec (2006), §4.
    """
    if points.ndim != 3:
        raise ValueError(
            f"points must have shape (B, N, ambient_dim), got {tuple(points.shape)}"
        )
    B, N, D = points.shape
    if N == 0:
        raise ValueError("Fréchet mean of an empty set is undefined")

    # Resolve / normalize weights to sum to 1 along the N axis.
    if weights is None:
        w = torch.full((B, N), 1.0 / N,
                       device=points.device, dtype=points.dtype)
    else:
        if weights.shape != (B, N):
            raise ValueError(
                f"weights must have shape (B={B}, N={N}), "
                f"got {tuple(weights.shape)}"
            )
        # Normalize. The clamp guards a degenerate all-zero-weights row.
        w_sum = weights.sum(dim=-1, keepdim=True).clamp(
            min=torch.finfo(weights.dtype).tiny,
        )
        w = weights / w_sum

    # Initialize the mean as the weighted-first point — a generic
    # heuristic that always lies on the manifold (it IS one of the
    # input points). The Karcher iteration is locally-monotone on
    # Hadamard manifolds, so the starting point only affects the
    # iteration count, not the limit.
    mu = points[:, 0].clone()  # (B, D)

    for _ in range(max_iter):
        # Broadcast μ across the N points so we can call log batched.
        mu_expanded = mu.unsqueeze(1).expand(B, N, D)
        # Manifold.log expects (·, ambient_dim) leading-batch form; we
        # flatten the (B, N) pair to a single batch axis, call, reshape.
        log_p = manifold.log(
            mu_expanded.reshape(B * N, D),
            points.reshape(B * N, D),
        ).reshape(B, N, D)
        # Weighted tangent average — the Riemannian gradient step.
        tangent_avg = (w.unsqueeze(-1) * log_p).sum(dim=1)  # (B, D)
        # Convergence: Riemannian norm of the update tangent at the
        # CURRENT base point. Must be computed BEFORE the `exp` step:
        # `tangent_avg ∈ T_{μ_t}` is a tangent at the OLD μ, and for
        # point-dependent metrics (e.g. KappaStereographicManifold's
        # conformal factor) `‖tangent_avg‖_{μ_new}` differs from
        # `‖tangent_avg‖_{μ_old}` by the metric ratio. Use the OLD μ.
        update_norm = manifold.norm(mu, tangent_avg)         # (B,)
        # Step.
        mu = manifold.exp(mu, tangent_avg)
        if update_norm.max().item() < tol:
            break

    # Fail-loud: detect convergence to a point outside the manifold.
    # On positively-curved branches (spherical κ > 0) the Karcher
    # iteration can escape the domain mid-iteration and converge to a
    # stationary point of the *ambient* objective rather than the
    # intrinsic Fréchet mean — caught by
    # `notes/validation/frechet_spherical_results.md`. Without this
    # check, the function silently returns a point with kappa‖μ‖² ≥ 1.
    # Manifolds that always pass `is_on_manifold` (Lorentz, Euclidean)
    # are unaffected.
    if hasattr(manifold, "is_on_manifold") and not torch.isfinite(mu).all():
        raise RuntimeError(
            "frechet_mean produced non-finite output. Likely cause: "
            "the Karcher iteration overshot the domain (common on "
            "positively-curved manifolds with input spread approaching "
            "the injectivity radius). Try reducing the input spread or "
            "warm-starting from a closer initial point."
        )
    if hasattr(manifold, "is_on_manifold"):
        on_mfd = manifold.is_on_manifold(mu)
        if not on_mfd.all():
            raise RuntimeError(
                f"frechet_mean converged to a point outside the manifold "
                f"({(~on_mfd).sum().item()} of {on_mfd.numel()} batch "
                f"elements). This happens on positively-curved branches "
                f"(`KappaStereographicManifold(kappa > 0)`) when the "
                f"input spread approaches the injectivity radius "
                f"`pi/sqrt(kappa)`. Caller is responsible for keeping "
                f"inputs well within the injectivity radius — see "
                f"`notes/validation/frechet_spherical_results.md`."
            )

    return mu
