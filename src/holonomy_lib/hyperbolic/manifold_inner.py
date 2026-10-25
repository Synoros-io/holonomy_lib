# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Synoros

"""Manifold-aware inner product, cosine-analog on a Riemannian manifold.

For points x, y on a Riemannian manifold M, the natural "inner
product" is not the ambient dot product (which ignores the geometry)
but the Riemannian inner product of their tangent representations at
a common base point. This module always uses the origin as that base
point and pulls each point back to the tangent space there with `log`:

    ⟨x, y⟩_M  :=  ⟨log_o(x), log_o(y)⟩_o,

where `log_o(·)` is the inverse exponential at the origin and the
right-hand inner product is the Riemannian metric `g_o` on `T_o M`.
The quantity is symmetric, bilinear in its tangent representatives,
and reduces to the standard Euclidean inner when M is Euclidean.

Manifold-agnostic: this works on any manifold that exposes `log(x, y)`,
`inner(x, u, v)`, and `origin(batch_size)`. Two cases worth checking by hand.
On `LorentzManifold` the origin is the north pole, whose tangent vectors
have zero time coordinate, so the Lorentz inner there is the Euclidean
inner of the spatial coordinates. On `KappaStereographicManifold` the
conformal factor at the origin is `λ_κ(o) = 2`, so the metric there is
4 times the Euclidean one. In both cases `manifold_aware_inner(x, x)`
equals `d(o, x)²`, the same value in either model.

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

    Works on any manifold exposing `log(x, y)`, `inner(x, u, v)`, and
    `origin(batch_size)`. With `x = y` the result is `d_M(o, x)²`, as long
    as `inner` is the Riemannian metric and `log` inverts `exp`.
    Assumes `x` and `y` have the same batch size B, since the origin is
    built with `x`'s batch size.

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
    # The manifold builds its origin on its own device and dtype. Move it onto
    # x's, or a GPU or float64 input would meet a CPU or float32 base point.
    origin = manifold.origin(batch_size=B).to(device=x.device, dtype=x.dtype)
    log_x = manifold.log(origin, x)       # (B, ambient_dim) tangent vectors at o
    log_y = manifold.log(origin, y)
    # The metric at the origin, applied to the two tangent vectors.
    return manifold.inner(origin, log_x, log_y)
