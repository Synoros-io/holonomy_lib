# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Synoros

"""Manifold-aware inner product, cosine-analog on a Riemannian manifold.

For points x, y on a Riemannian manifold M, the natural "inner
product" is not the ambient dot product (which ignores the geometry)
but the Riemannian inner product of their tangent representations at
a common base point. The standard construction uses the origin (or
any user-supplied base point) and pulls each point back via `log`:

    ⟨x, y⟩_M  :=  ⟨log_o(x), log_o(y)⟩_o,

where `log_o(·)` is the inverse exponential at the origin and the
right-hand inner product is the Riemannian metric `g_o` on `T_o M`.
The quantity is symmetric, bilinear in its tangent representatives,
and reduces to the standard Euclidean inner when M is Euclidean.

Manifold-agnostic: this primitive works on any manifold that exposes
`log(x, y)`, `inner(x, u, v)`, and `origin(batch_size)`. For
`LorentzManifold` the Riemannian inner at the north-pole tangent
agrees with the Euclidean inner of the spatial coordinates (the time
coordinate of the tangent is 0 at the origin). For
`KappaStereographicManifold` it carries the conformal factor
`λ_κ(o)² = 4`, so `manifold_aware_inner(x, x) = d(o, x)²` on both
manifolds, a model-invariant similarity.

References:
  Pennec, X. (2006). Intrinsic statistics on Riemannian manifolds.
    J. Math. Imaging Vision 25(1):127–154 (§3 tangent statistics).
  Said, S., Bombrun, L., Berthoumieu, Y. (2015). Riemannian inner
    products in covariance estimation on the SPD manifold. SIIMS.
"""

from __future__ import annotations

from typing import Any

import torch


def manifold_aware_inner(
    x: torch.Tensor,
    y: torch.Tensor,
    manifold: Any,
) -> torch.Tensor:
    """Riemannian inner product `⟨log_o(x), log_o(y)⟩_o` at the
    manifold origin.

    Manifold-agnostic, works on any manifold exposing `log(x, y)`,
    `inner(x, u, v)`, and `origin(batch_size)`. The result is the
    metric-consistent "similarity at origin" and satisfies
    `manifold_aware_inner(x, x) = d_M(o, x)²` on all manifolds.

    Args:
      x, y: points on the manifold, shape `(B, ambient_dim)`.
      manifold: a manifold object exposing `log`, `inner`, `origin`.

    Returns:
      `(B,)` tensor of Riemannian inner products.

    Example:
      >>> from holonomy_lib.manifolds import LorentzManifold
      >>> mfd = LorentzManifold(n=3)
      >>> x = mfd.random_point(batch_size=4)
      >>> sim = manifold_aware_inner(x, x, mfd)  # = d(o, x)² per batch
      >>> sim.shape
      torch.Size([4])

    References:
      Pennec (2006), §3, tangent-space statistics on Riemannian manifolds.
    """
    B = x.shape[0]
    origin = manifold.origin(batch_size=B).to(device=x.device, dtype=x.dtype)
    log_x = manifold.log(origin, x)       # ambient tangent at o
    log_y = manifold.log(origin, y)
    return manifold.inner(origin, log_x, log_y)
