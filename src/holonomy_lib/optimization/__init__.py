# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Synoros

"""holonomy_lib.optimization: Riemannian optimizers on the library's manifolds.

Currently implemented:
  RiemannianSGD(manifold, lr)  : steepest descent on a manifold.

The optimizer only calls `egrad_to_rgrad` and `retraction` on its manifold,
so it works with any manifold that has both (`FixedRankManifold`,
`SPDManifold`, `ProductManifold`). The flow is:

  1. Caller computes the ambient gradient `g` of the objective.
  2. `optimizer.step(point, g)` converts `g` using the metric at
     `point`, scales by `-lr`, and retracts back onto the manifold.
  3. Returns the new point. Caller drives the loop.

The functional helper `riemannian_sgd_step` exposes the same logic
without state, useful for tests and for integrating into custom
training loops.

Why no Adam (or RMSProp, AdamW, ...): adaptive step-size schemes are
user-side ergonomics, not part of the math of optimization on a
manifold. SGD uses metric gradient conversion followed by retraction.
Users who want Adam-style rates can compute their adaptive step in
ambient space and pass the rescaled gradient into `step()`.

Planned:
  - `torch.optim.Optimizer` integration via a `ManifoldParameter`
    wrapper so manifold params can be mixed with ambient params in a
    normal PyTorch training loop. Out of scope for v1.
  - Trust-region and conjugate-gradient solvers.
"""

from holonomy_lib.optimization.base import RiemannianOptimizer
from holonomy_lib.optimization.sgd import RiemannianSGD, riemannian_sgd_step

__all__ = [
    "RiemannianOptimizer",
    "RiemannianSGD",
    "riemannian_sgd_step",
]
