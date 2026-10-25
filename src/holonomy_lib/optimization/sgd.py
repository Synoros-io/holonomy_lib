# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Synoros

"""Riemannian steepest descent (RSGD).

For a Riemannian manifold M with `egrad_to_rgrad` (ambient gradient to
Riemannian gradient, metric included) and `retraction` (move along a
tangent vector to a new point), Riemannian SGD takes the ambient
gradient `g` at the current point, converts it with the metric, scales
by `-lr`, and retracts back onto M.

  ξ = egrad_to_rgrad(point, g)
  point' = retraction(point, -lr · ξ)

References:
  Absil, P.-A., Mahony, R., Sepulchre, R. (2008). Optimization
    Algorithms on Matrix Manifolds. Princeton University Press,
    §4.1-§4.2.
  Bonnabel, S. (2013). Stochastic gradient descent on Riemannian
    manifolds. IEEE TAC 58(9):2217-2229.
"""

from __future__ import annotations

from typing import Any

import torch

from holonomy_lib.optimization.base import RiemannianOptimizer


def riemannian_sgd_step(
    manifold: Any,
    point: Any,
    ambient_grad: torch.Tensor,
    lr: float,
) -> Any:
    """Functional one-step Riemannian SGD.

    Args:
      manifold: object exposing `egrad_to_rgrad(point, ambient)` and
        `retraction(point, tangent)` methods (e.g.
        `FixedRankManifold`, `SPDManifold`).
      point: current point on the manifold (tensor for SPD; triple
        of tensors for FixedRank).
      ambient_grad: ambient-space gradient of the objective. Always
        a single `torch.Tensor`; for `FixedRankManifold` this is the
        `(B, m, n)` ambient gradient per Vandereycken (2013)'s
        embedded-tangent representation.
      lr: learning rate, a positive scalar. Not checked here;
        `RiemannianSGD` rejects lr <= 0.

    Returns:
      new_point: the manifold-constrained next iterate.

    References:
      Absil, P.-A., Mahony, R., Sepulchre, R. (2008). Optimization
        Algorithms on Matrix Manifolds. Princeton University Press,
        §4.1; retraction-based steepest descent.
      Bonnabel, S. (2013). Stochastic gradient descent on Riemannian
        manifolds. IEEE TAC 58(9):2217-2229, §III.
    """
    # A plain tangent projection is the right conversion only when the metric
    # is the embedded Euclidean one, as for FixedRankManifold. SPD's
    # affine-invariant metric needs S sym(G) S, so each manifold does its own
    # conversion through egrad_to_rgrad. The step then scales that tangent by
    # -lr and retracts it back onto the manifold.
    tangent = manifold.egrad_to_rgrad(point, ambient_grad)
    return manifold.retraction(point, -lr * tangent)


class RiemannianSGD(RiemannianOptimizer):
    """Optimizer that holds a manifold and a learning rate.

    Each `step` call forwards to `riemannian_sgd_step`. The object keeps no
    moment buffers or step count, so nothing carries over between steps.

    Args:
      manifold: the underlying manifold object (see `riemannian_sgd_step`).
      lr: learning rate, must be > 0 (ValueError otherwise). The default
        `1e-2` is a starting point, not a tuned value; tune it per problem.

    Example:
      >>> from holonomy_lib.manifolds import SPDManifold
      >>> mfd = SPDManifold(n=4)
      >>> opt = RiemannianSGD(mfd, lr=0.05)
      >>> point = mfd.random_point(batch_size=1)
      >>> # in a training loop:
      >>> ambient_grad = compute_ambient_grad(point)
      >>> point = opt.step(point, ambient_grad)
    """

    def __init__(self, manifold: Any, lr: float = 1e-2) -> None:
        super().__init__(manifold)
        if lr <= 0:
            raise ValueError(f"lr must be > 0, got {lr}")
        self.lr = lr

    def step(self, point: Any, ambient_grad: Any) -> Any:
        """Convert the ambient gradient and retract one descent step.

        Args:
          point: current point on the manifold.
          ambient_grad: ambient-space gradient of the objective at `point`.
        Returns:
          new_point: the point after the step.

        References:
          Bonnabel (2013), Stochastic Gradient Descent on Riemannian
          Manifolds, §III.
        """
        return riemannian_sgd_step(self.manifold, point, ambient_grad, self.lr)
