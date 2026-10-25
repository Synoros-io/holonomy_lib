# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Synoros

"""Fréchet means of vector-coordinate points on Riemannian manifolds.

The objective is the weighted sum of squared geodesic distances. On a
Hadamard manifold it has a unique minimizer, but that fact alone does not
make unit Karcher steps converge. This implementation controls each batch's
step with backtracking on the objective.

References:
  Karcher, H. (1977). Riemannian center of mass and mollifier smoothing.
    Comm. Pure Appl. Math. 30(5):509–541.
  Afsari, B., Tron, R., Vidal, R. (2013). On the convergence of gradient
    descent for finding the Riemannian center of mass. SIAM J. Control
    Optim. 51(3):2230–2260.
"""

from __future__ import annotations

import math
import warnings
from typing import Any

import torch
from torch.autograd.function import once_differentiable

# Retain the existing work budget; convergence is checked rather than assumed.
FRECHET_MAX_ITER_DEFAULT = 100
# Match the host-check cadence in CONVENTIONS §4.1.
FRECHET_SYNC_EVERY_DEFAULT = 8
# A bounded line search tries unit steps through 2**-15. If these fail,
# the batch stays put and the final residual warning reports the failure.
FRECHET_MAX_BACKTRACKS_DEFAULT = 16


def frechet_mean(
    points: torch.Tensor,
    manifold: Any,
    weights: torch.Tensor | None = None,
    max_iter: int = FRECHET_MAX_ITER_DEFAULT,
    tol: float = 1e-9,
    *,
    sync_every: int = FRECHET_SYNC_EVERY_DEFAULT,
    max_backtracks: int = FRECHET_MAX_BACKTRACKS_DEFAULT,
) -> torch.Tensor:
    """Compute a Fréchet mean with backtracking Karcher descent.

    Args:
      points: `(B, N, ambient_dim)` vector-coordinate points. Matrix-valued
        manifolds such as SPD are not supported by this shape contract.
      manifold: object exposing `log`, `exp`, `norm`, and `distance` on
        vector-coordinate batches, for example `LorentzManifold`. Backward
        also requires `dim`, `projection`, and `inner`.
      weights: optional `(B, N)` finite non-negative weights, with positive
        total in every batch. Normalized internally; omitted means uniform.
      max_iter: maximum number of descent steps, default 100.
      tol: requested residual norm of the weighted average logarithm.
      sync_every: check all-batch convergence on the host every this many
        iterations. At most `sync_every - 1` extra small updates occur
        after all residuals first meet the tolerance.
      max_backtracks: number of trial step sizes, starting at the last
        accepted step and halving (initially one). A batch with no
        acceptable trial stays at its current point.

    Returns:
      `(B, ambient_dim)` mean estimate. `B=0` returns an empty tensor.
      Warns if its residual exceeds `tol` after the iteration budget.

    Notes:
      Iterations and backtracking trials are sequential because each step
      depends on the preceding objective value. Points and batch elements
      are processed together by tensor operations. The line search is
      evaluated without autograd. First derivatives with respect to points
      and weights use the implicit function theorem at the returned mean,
      with the manifold held fixed. They do not differentiate the iterations.
      Accuracy therefore requires convergence of the stationarity residual;
      a warning about nonconvergence also applies to the gradients.

      The backward system uses an orthonormal tangent frame and has size
      `dim x dim`, including for embedded manifolds such as the Lorentz
      hyperboloid. A singular stationarity Jacobian (to working precision)
      returns NaN gradients for that batch member only. Higher derivatives
      and derivatives with respect to manifold parameters are unsupported.

      On positive-curvature manifolds, spread-out inputs may have several
      local minima or cross a logarithm's cut locus. Backtracking does not
      establish global optimality there. The caller must choose inputs in
      a convex normal neighborhood to use local convergence guarantees.

    References:
      Karcher (1977), Riemannian center of mass and mollifier smoothing.
      Afsari, Tron, Vidal (2013), On the convergence of gradient descent
        for finding the Riemannian center of mass, §§2–3.
      Absil, Mahony, Sepulchre (2008), Optimization Algorithms on Matrix
        Manifolds, §4.2, line-search methods.
      Lou et al. (2020), Differentiating through the Frechet Mean,
        ICML, PMLR 119:6393-6403, implicit differentiation.
    """
    if points.ndim != 3:
        raise ValueError(
            f"points must have shape (B, N, ambient_dim), got {tuple(points.shape)}"
        )
    batch, count, dimension = points.shape
    if count == 0:
        raise ValueError("Fréchet mean of an empty set is undefined")
    if weights is not None and weights.shape != (batch, count):
        raise ValueError(
            f"weights must have shape ({batch}, {count}), got {tuple(weights.shape)}"
        )
    if max_iter < 0 or sync_every < 1 or max_backtracks < 1:
        raise ValueError("max_iter must be >= 0; sync_every and max_backtracks >= 1")
    if not torch.is_floating_point(points):
        raise ValueError("points must have floating dtype")
    if not math.isfinite(tol) or tol < 0:
        raise ValueError("tol must be finite and non-negative")
    if weights is None:
        normalized_weights = points.new_full((batch, count), 1.0 / count)
    else:
        if not torch.isfinite(weights).all() or (weights < 0).any():
            raise ValueError("weights must be finite and non-negative")
        weights = weights.to(device=points.device, dtype=points.dtype)
        total = weights.sum(dim=-1, keepdim=True)
        if not torch.isfinite(total).all() or (total <= 0).any():
            raise ValueError(
                "weights must be finite, non-negative, and have positive total"
            )
        normalized_weights = weights / total

    if batch == 0:
        # Keep both differentiable inputs connected even for an empty batch.
        return (normalized_weights.unsqueeze(-1) * points).sum(dim=1)
    return _ImplicitFrechetMean.apply(
        points, normalized_weights, manifold, max_iter, tol,
        sync_every, max_backtracks,
    )


def _frechet_mean_forward(
    points: torch.Tensor, normalized_weights: torch.Tensor, manifold: Any,
    max_iter: int, tol: float, sync_every: int, max_backtracks: int,
) -> torch.Tensor:
    """Karcher descent for validated inputs; called without an autograd graph."""
    batch, count, dimension = points.shape
    mean = points[:, 0].clone()
    flat_points = points.reshape(batch * count, dimension)

    def average_log(base: torch.Tensor) -> torch.Tensor:
        expanded = base.unsqueeze(1).expand(batch, count, dimension)
        logarithms = manifold.log(
            expanded.reshape(batch * count, dimension), flat_points,
        ).reshape(batch, count, dimension)
        return (normalized_weights.unsqueeze(-1) * logarithms).sum(dim=1)

    def objective(base: torch.Tensor) -> torch.Tensor:
        expanded = base.unsqueeze(1).expand(batch, count, dimension)
        distances = manifold.distance(
            expanded.reshape(batch * count, dimension), flat_points,
        ).reshape(batch, count)
        return (normalized_weights * distances.square()).sum(dim=-1)

    previous_step = points.new_ones(batch)
    for iteration in range(max_iter):
        tangent = average_log(mean)
        # The tangent belongs to the old mean; its norm must use that base.
        residual = manifold.norm(mean, tangent)
        active = residual > tol
        with torch.no_grad():
            cost = objective(mean)
            # Keeping the accepted step avoids restarting at a large step
            # when objective differences are below floating-point resolution.
            step = previous_step
            # Preserve the solver's small accepted updates until the next
            # convergence check. Backward is independent of their count.
            accepted_step = torch.where(active, torch.zeros_like(step), step)
            accepted = ~active
            # The derivative of the normalized squared-distance objective
            # along average_log is -2*residual**2. Armijo with coefficient
            # one half therefore asks for a decrease of step*residual**2.
            # Allow one relative eps of objective rounding near a minimum.
            rounding = torch.finfo(points.dtype).eps * cost.abs()
            for _ in range(max_backtracks):
                trial = manifold.exp(mean, step.unsqueeze(-1) * tangent)
                trial_cost = objective(trial)
                decrease = step * residual.square()
                # Near the minimizer, subtraction of two squared-distance
                # objectives loses the decrease. The gradient residual still
                # resolves it, so demand residual reduction in that regime.
                trial_residual = manifold.norm(trial, average_log(trial))
                improved = torch.where(
                    decrease <= rounding,
                    trial_residual < residual,
                    trial_cost <= cost - decrease + rounding,
                )
                take = ~accepted & torch.isfinite(trial_cost) & improved
                accepted_step = torch.where(take, step, accepted_step)
                accepted = accepted | take
                step = step * 0.5
        previous_step = torch.where(active, accepted_step, previous_step)
        mean = manifold.exp(mean, accepted_step.unsqueeze(-1) * tangent)
        if (iteration + 1) % sync_every == 0 and not active.any().item():
            break

    if not torch.isfinite(mean).all():
        raise RuntimeError("frechet_mean produced non-finite output")
    if hasattr(manifold, "is_on_manifold") and not manifold.is_on_manifold(mean).all():
        raise RuntimeError("frechet_mean produced a point outside the manifold")
    with torch.no_grad():
        final_residual = manifold.norm(mean, average_log(mean)).max().item()
    if not math.isfinite(final_residual) or final_residual > tol:
        warnings.warn(
            f"frechet_mean did not converge: residual {final_residual:.6g} "
            f"exceeds tol={tol:.6g} after at most {max_iter} iterations",
            RuntimeWarning, stacklevel=2,
        )
    return mean


def _tangent_frame(mean: torch.Tensor, manifold: Any) -> torch.Tensor:
    """Return metric-orthonormal tangent columns, with the base held fixed."""
    batch, ambient = mean.shape
    dimension = manifold.dim
    axes = torch.eye(ambient, dtype=mean.dtype, device=mean.device)
    axes = axes.expand(batch, ambient, ambient)
    if dimension == ambient:
        frame = axes
    else:
        # Projection may be oblique in Euclidean coordinates (Lorentz), but
        # its column space is still the tangent space. SVD extracts just the
        # intrinsic number of directions; no normal direction enters the solve.
        bases = mean[:, None, :].expand_as(axes).reshape(-1, ambient)
        projected = manifold.projection(bases, axes.reshape(-1, ambient))
        projected = projected.reshape(batch, ambient, ambient).mT
        frame = torch.linalg.svd(projected, full_matrices=False).U[..., :dimension]

    rows = frame.mT
    bases = mean[:, None, None, :].expand(batch, dimension, dimension, ambient)
    left = rows[:, :, None, :].expand_as(bases)
    right = rows[:, None, :, :].expand_as(bases)
    gram = manifold.inner(
        bases.reshape(-1, ambient), left.reshape(-1, ambient),
        right.reshape(-1, ambient),
    ).reshape(batch, dimension, dimension)
    factor = torch.linalg.cholesky(gram)
    return torch.linalg.solve_triangular(factor, rows, upper=False).mT


class _ImplicitFrechetMean(torch.autograd.Function):
    """Differentiate the converged tangent stationarity equation, per batch."""

    @staticmethod
    def forward(
        ctx: Any, points: torch.Tensor, weights: torch.Tensor, manifold: Any,
        max_iter: int, tol: float, sync_every: int, max_backtracks: int,
    ) -> torch.Tensor:
        mean = _frechet_mean_forward(
            points, weights, manifold, max_iter, tol, sync_every, max_backtracks,
        )
        ctx.manifold = manifold
        ctx.save_for_backward(points, weights, mean)
        return mean

    @staticmethod
    @once_differentiable
    def backward(
        ctx: Any, grad_mean: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, None, None, None, None, None]:
        points, weights, mean = ctx.saved_tensors
        manifold = ctx.manifold
        mean = mean.detach()
        batch, count, ambient = points.shape
        frame = _tangent_frame(mean, manifold)
        dimension = frame.shape[-1]

        with torch.enable_grad():
            points = points.detach().requires_grad_(True)
            weights = weights.detach().requires_grad_(True)
            coordinates = mean.new_zeros(batch, dimension, requires_grad=True)
            tangent = (frame @ coordinates.unsqueeze(-1)).squeeze(-1)
            base = manifold.exp(mean, tangent)
            expanded = base[:, None, :].expand(batch, count, ambient)
            logarithms = manifold.log(
                expanded.reshape(-1, ambient), points.reshape(-1, ambient),
            ).reshape(batch, count, ambient)
            average = (weights.unsqueeze(-1) * logarithms).sum(dim=1)
            projected = manifold.projection(mean, average)
            bases = mean[:, None, :].expand(batch, dimension, ambient)
            vectors = projected[:, None, :].expand_as(bases)
            stationarity = manifold.inner(
                bases.reshape(-1, ambient), frame.mT.reshape(-1, ambient),
                vectors.reshape(-1, ambient),
            ).reshape(batch, dimension)

            # Summing one coordinate across the batch gives every member's
            # Jacobian row together: manifold operations never mix members.
            jacobian = torch.stack([
                torch.autograd.grad(
                    stationarity[:, direction].sum(), coordinates,
                    retain_graph=True,
                )[0]
                for direction in range(dimension)
            ], dim=-2)
            cotangent = torch.autograd.grad(
                base, coordinates, grad_mean, retain_graph=True,
            )[0]
            finite = torch.isfinite(jacobian).all(dim=(-2, -1))
            safe_jacobian = torch.where(
                finite[:, None, None], jacobian, torch.zeros_like(jacobian),
            )
            singular_values = torch.linalg.svdvals(safe_jacobian)
            # The usual numerical-rank criterion scales with dimension and
            # machine precision, not an absolute Hessian cutoff.
            rank_tolerance = (
                dimension * torch.finfo(mean.dtype).eps * singular_values[:, 0]
            )
            nonsingular = finite & (singular_values[:, -1] > rank_tolerance)
            identity = torch.eye(dimension, dtype=mean.dtype, device=mean.device)
            system = torch.where(
                nonsingular[:, None, None], safe_jacobian.mT, identity,
            )
            multiplier, info = torch.linalg.solve_ex(system, cotangent)
            valid = nonsingular & (info == 0)
            multiplier = torch.where(valid[:, None], multiplier, 0.0)
            grad_points, grad_weights = torch.autograd.grad(
                stationarity, (points, weights), -multiplier,
            )
        grad_points = torch.where(valid[:, None, None], grad_points, float("nan"))
        grad_weights = torch.where(valid[:, None], grad_weights, float("nan"))
        return grad_points, grad_weights, None, None, None, None, None
