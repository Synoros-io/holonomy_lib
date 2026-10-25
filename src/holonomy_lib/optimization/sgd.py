# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Synoros

"""Riemannian steepest descent (RSGD).

For a Riemannian manifold M with `projection` (ambient → tangent) and
`retraction` (move along tangent → new point), Riemannian SGD takes
the ambient gradient `g`, projects it to the tangent space at the
current point, scales by `-lr`, and retracts back onto M.

  ξ = projection(point, -lr · g)
  point' = retraction(point, ξ)

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
      manifold: object exposing `projection(point, ambient)` and
        `retraction(point, tangent)` methods (e.g.
        `FixedRankManifold`, `SPDManifold`).
      point: current point on the manifold (tensor for SPD; triple
        of tensors for FixedRank).
      ambient_grad: ambient-space gradient of the objective. Always
        a single `torch.Tensor`; for `FixedRankManifold` this is the
        `(B, m, n)` ambient gradient per Vandereycken (2013)'s
        embedded-tangent representation.
      lr: learning rate (positive scalar).

    Returns:
      new_point: the manifold-constrained next iterate.

    References:
      Absil, P.-A., Mahony, R., Sepulchre, R. (2008). Optimization
        Algorithms on Matrix Manifolds. Princeton University Press,
        §4.1 — retraction-based steepest descent.
      Bonnabel, S. (2013). Stochastic gradient descent on Riemannian
        manifolds. IEEE TAC 58(9):2217-2229, §III.

    Note: this function is intentionally NOT `@with_provenance`-
    decorated. Optimizer steps are inner-loop calls that fire
    hundreds or thousands of times per training run, and the hex
    computation would dominate the actual math. If you need to
    audit an optimization trajectory, record the manifold primitives
    (`projection`, `retraction`) it calls instead.
    """
    # Project the negative ambient gradient onto the tangent space at
    # `point`, then retract. The projection is linear, so scaling by
    # `-lr` before the projection is equivalent to scaling after.
    tangent = manifold.projection(point, -lr * ambient_grad)
    return manifold.retraction(point, tangent)


class RiemannianSGD(RiemannianOptimizer):
    """Stateful wrapper around `riemannian_sgd_step`.

    Args:
      manifold: the underlying manifold object (see `riemannian_sgd_step`).
      lr: learning rate. Default `1e-2` (the torch.optim.SGD default).

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
        return riemannian_sgd_step(self.manifold, point, ambient_grad, self.lr)
