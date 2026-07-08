# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Synoros

"""Cached Gauss-Legendre quadrature nodes (internal).

`scipy.special.roots_legendre` recomputes the rule on every call (the
Golub-Welsch tridiagonal eigensolve plus Newton refinement); it does NOT
cache (only `scipy.integrate.fixed_quad`'s private wrapper does). The
library's quadrature call sites (`manifolds.comparison.model_ball_volume`,
`hyperbolic.heat_kernel`) sit inside batched paths where the same `n_quad`
recurs on every invocation, so the recompute plus the numpy-to-torch
round-trip is pure per-call overhead. This module keeps one float64 CPU
master copy per `n_quad` and hands out dtype/device casts.

Callers must treat the returned tensors as read-only: when the requested
dtype/device match the master copy, `Tensor.to` returns the cached tensor
itself, not a copy.

References:
  Golub, G. H., Welsch, J. H. (1969). Calculation of Gauss quadrature
    rules. Mathematics of Computation 23(106):221-230. (The algorithm
    underlying `scipy.special.roots_legendre`.)
"""

from __future__ import annotations

import functools

import torch
from scipy.special import roots_legendre


@functools.lru_cache(maxsize=None)
def _gauss_legendre_float64_cpu(n_quad: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Master copy: float64 CPU `(nodes, weights)` of the `n_quad`-point rule."""
    nodes_np, weights_np = roots_legendre(n_quad)
    return (
        torch.as_tensor(nodes_np, dtype=torch.float64),
        torch.as_tensor(weights_np, dtype=torch.float64),
    )


def gauss_legendre_nodes(
    n_quad: int, *, dtype: torch.dtype, device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """`(nodes, weights)` of the `n_quad`-point Gauss-Legendre rule on (-1, 1).

    The float64 CPU master is computed once per `n_quad` (`lru_cache`); the
    per-call `.to(dtype, device)` is a cheap cast/copy (no-op when they match).

    Args:
      n_quad: node count of the rule.
      dtype: floating dtype of the returned tensors.
      device: device of the returned tensors.

    Returns:
      `(nodes, weights)`, each `(n_quad,)`, read-only by convention.

    References:
      Golub-Welsch (1969); see module docstring.
    """
    nodes, weights = _gauss_legendre_float64_cpu(n_quad)
    return (
        nodes.to(dtype=dtype, device=device),
        weights.to(dtype=dtype, device=device),
    )
