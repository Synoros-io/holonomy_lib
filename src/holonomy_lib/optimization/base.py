# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Synoros

"""Base class for Riemannian optimizers.

Subclasses implement `step(point, ambient_grad) -> new_point`. The base
class holds the manifold reference and per-instance hyperparameters.

The "manifold" parameter must expose `projection(point, ambient)` and
`retraction(point, tangent)` methods with the same signatures as
`FixedRankManifold` and `SPDManifold`. The optimizer is otherwise
agnostic to which manifold it's operating on.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class RiemannianOptimizer(ABC):
    """Abstract base for Riemannian optimizers.

    A subclass owns:
      - the manifold (provides projection + retraction),
      - hyperparameters (learning rate, momentum coefficients, etc.),
      - per-instance state (moment buffers, step count) if stateful.

    The subclass implements `step(point, ambient_grad)`, returning the
    new point on the manifold. Caller is responsible for storing the
    new point and reading its gradient on the next iteration.

    Why this shape: the existing `FixedRankManifold`/`SPDManifold`
    methods return new points rather than mutating in-place, so the
    optimizer follows the same convention. Future v2 work can wrap a
    `torch.optim.Optimizer` around this for compatibility with
    `loss.backward()` flows; for v1 the explicit point-in / point-out
    contract is the simpler primitive.
    """

    def __init__(self, manifold: Any) -> None:
        self.manifold = manifold

    @abstractmethod
    def step(self, point: Any, ambient_grad: Any) -> Any:
        """Apply one optimizer step and return the new point.

        Args:
          point: current point on the manifold (tensor for SPD, triple
            of tensors for FixedRank).
          ambient_grad: gradient of the objective with respect to the
            ambient (embedding-space) coordinates. Same structure as
            an ambient tangent: a single tensor for both SPD and
            FixedRank (FixedRank tangents are stored in ambient
            `(B, m, n)` form per Vandereycken 2013).

        Returns:
          new_point: the point after the step.
        """
        ...
