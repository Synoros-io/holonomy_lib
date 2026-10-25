# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Synoros

"""Mixed-curvature product manifold M_1 × M_2 × … × M_k.

A point is a tuple of per-submanifold coordinates, stored as a
**flat concatenated tensor** for ergonomic batched ops:

    `(B, sum(m.ambient_dim for m in manifolds))`

The combined Riemannian metric is the (optionally weighted) direct
sum of the submanifold metrics:

    g((u_1, …, u_k), (v_1, …, v_k))_x = Σ_i w_i · g_i(u_i, v_i)_{x_i}

so geodesic distance is Pythagorean:

    d²((x_1, …, x_k), (y_1, …, y_k)) = Σ_i w_i · d_i²(x_i, y_i).

Each manifold operation (exp, log, projection, retraction, …)
delegates to the corresponding submanifold method, slice by slice.
A point `(x_E, x_H, x_S)` with one Euclidean, one hyperbolic, and one
spherical component therefore works by calling the submanifold classes
(`KappaStereographicManifold`, `LorentzManifold`) on each slice.

References:
  Skopek, O., Ganea, O.-E., Bécigneul, G. (2019). Mixed-curvature
    Variational Autoencoders. ICLR. (mixed-curvature embedding for
    deep generative models)
  Gu, A., Sala, F., Gunel, B., Ré, C. (2019). Learning mixed-
    curvature representations in product spaces. ICLR.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from holonomy_lib.manifolds.lorentz import _safe_sqrt


class ProductManifold:
    """Riemannian product of one or more manifolds.

    Args:
      manifolds: ordered list of submanifolds. Each must expose
        `.dim`, `.ambient_dim`, `random_point`, `origin`,
        `is_on_manifold`, `distance`, `exp`, `log`, `inner`,
        `projection`, `retraction`, `egrad_to_rgrad`. `exp_0` and `log_0`
        are needed only by the tangent-at-origin methods. Heterogeneous
        mixes are fine, e.g.
        `[KappaStereographicManifold(kappa=-1.0, n=4), LorentzManifold(n=4)]`.
      weights: optional per-submanifold positive finite scalars `w_i` on
        the squared-distance contribution. Default uniform (`w_i = 1`).
        Use `weights` to up-weight one geometry relative to others in
        the combined metric.
      device, dtype: tensor placement / precision for `random_point`
        / `origin`. (Sub-manifold operations use the input tensor's
        own device.)

    Example:
      >>> from holonomy_lib.manifolds import (
      ...     KappaStereographicManifold, LorentzManifold, ProductManifold,
      ... )
      >>> mfd = ProductManifold([
      ...     KappaStereographicManifold(n=4, kappa=0.0),   # Euclidean part
      ...     LorentzManifold(n=4, k=-1.0),                  # Hyperbolic part
      ... ])
      >>> x = mfd.random_point(batch_size=3)
      >>> x.shape  # 4 (Euclidean ambient) + 5 (Lorentz ambient) = 9
      torch.Size([3, 9])
      >>> mfd.distance(x, mfd.random_point(batch_size=3)).shape
      torch.Size([3])
    """

    def __init__(
        self,
        manifolds: Sequence,
        weights: Sequence[float] | None = None,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float64,
    ):
        if not manifolds:
            raise ValueError("manifolds must contain at least one submanifold")
        self.manifolds = list(manifolds)
        if weights is None:
            self.weights = [1.0] * len(self.manifolds)
        else:
            if len(weights) != len(self.manifolds):
                raise ValueError(
                    f"weights length {len(weights)} != "
                    f"manifolds length {len(self.manifolds)}"
                )
            if any(w <= 0 or not math.isfinite(w) for w in weights):
                raise ValueError("weights must be positive and finite")
            self.weights = [float(w) for w in weights]
        self.device = torch.device(device)
        self.dtype = dtype

        # Each submanifold owns a contiguous block of the flat tensor. The
        # running sum of block sizes gives where each block starts, and pairing
        # adjacent entries gives (start, end) for every slice.
        self._ambient_dims = [m.ambient_dim for m in self.manifolds]
        offsets = [0]
        for d in self._ambient_dims:
            offsets.append(offsets[-1] + d)
        self._ambient_slices = list(zip(offsets[:-1], offsets[1:], strict=True))

        # Same bookkeeping for intrinsic coordinates, which exp_0 and log_0
        # use. They are shorter than the ambient points for components that
        # embed, like Lorentz (n intrinsic, n + 1 ambient).
        self._intrinsic_dims = [m.dim for m in self.manifolds]
        i_offsets = [0]
        for d in self._intrinsic_dims:
            i_offsets.append(i_offsets[-1] + d)
        self._intrinsic_slices = list(zip(i_offsets[:-1], i_offsets[1:], strict=True))

    @property
    def dim(self) -> int:
        """Intrinsic manifold dimension Σ dim_i."""
        return sum(self._intrinsic_dims)

    @property
    def ambient_dim(self) -> int:
        """Ambient dimension Σ ambient_dim_i."""
        return sum(self._ambient_dims)

    # ----------------------------------------------------------------
    # Slicing helpers (used by every op)
    # ----------------------------------------------------------------

    def _split_ambient(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Split a flat ambient tensor into one view per submanifold.

        Input: `x` of shape `(..., ambient_dim)`. Output: list of views in
        submanifold order. Raises ValueError on the wrong shape.
        """
        if x.ndim < 2 or x.shape[-1] != self.ambient_dim:
            raise ValueError(
                f"x must have shape (..., {self.ambient_dim}); got {tuple(x.shape)}"
            )
        return [x[..., s:e] for s, e in self._ambient_slices]

    def _split_intrinsic(self, v: torch.Tensor) -> list[torch.Tensor]:
        """Split a flat intrinsic tangent into one view per submanifold.

        Input: `v` of shape `(..., dim)`. Output: list of views in submanifold
        order. Raises ValueError on the wrong shape.
        """
        if v.ndim < 2 or v.shape[-1] != self.dim:
            raise ValueError(
                f"v must have shape (..., {self.dim}); got {tuple(v.shape)}"
            )
        return [v[..., s:e] for s, e in self._intrinsic_slices]

    def _join(self, pieces: list[torch.Tensor]) -> torch.Tensor:
        """Concatenate per-submanifold tensors back into flat form along the last
        axis. The pieces must share their leading (batch) dims.
        """
        return torch.cat(pieces, dim=-1)

    def component(self, x: torch.Tensor, index: int) -> torch.Tensor:
        """Extract the `index`-th submanifold's coordinates from a
        flat point tensor `x`. Public convenience for inspection /
        diagnostics."""
        return self._split_ambient(x)[index]


    # ----------------------------------------------------------------
    # Construction
    # ----------------------------------------------------------------

    def origin(self, batch_size: int = 1) -> torch.Tensor:
        """Concatenated origins of every submanifold.

        References:
          Lee (2018), Introduction to Riemannian Manifolds, product metrics.
          The component formulas are cited in their manifold methods.
        """
        if batch_size < 0:
            raise ValueError(f"batch_size must be >= 0, got {batch_size}")
        return self._join([
            m.origin(batch_size=batch_size).to(device=self.device, dtype=self.dtype)
            for m in self.manifolds
        ])

    def random_point(
        self,
        batch_size: int = 1,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Sample each submanifold and concatenate the results. Each
        submanifold's `random_point` decides its own distribution (uniform,
        Gaussian tangent, ...). One `generator` is passed to all of them in
        order, so a single seed fixes the whole draw.

        References:
          Lee (2018), Introduction to Riemannian Manifolds, product metrics.
          The component formulas are cited in their manifold methods.
        """
        if batch_size < 0:
            raise ValueError(f"batch_size must be >= 0, got {batch_size}")
        return self._join([
            m.random_point(batch_size=batch_size, generator=generator).to(
                device=self.device, dtype=self.dtype,
            )
            for m in self.manifolds
        ])

    def is_on_manifold(
        self, x: torch.Tensor, atol: float = 1e-9,
    ) -> torch.Tensor:
        """A product point is on the manifold iff every component is
        on its submanifold.

        References:
          Lee (2018), Introduction to Riemannian Manifolds, product metrics.
          The component formulas are cited in their manifold methods.
        """
        parts = self._split_ambient(x)
        on_each = torch.stack(
            [
                m.is_on_manifold(p, atol=atol)
                for m, p in zip(self.manifolds, parts, strict=True)
            ],
            dim=0,
        )
        return on_each.all(dim=0)

    # ----------------------------------------------------------------
    # Tangent operations
    # ----------------------------------------------------------------

    def projection(
        self, x: torch.Tensor, w: torch.Tensor,
    ) -> torch.Tensor:
        """Project the ambient direction `w` (flat) onto the tangent
        space at `x`, per-submanifold.

        References:
          Lee (2018), Introduction to Riemannian Manifolds, product metrics.
          The component formulas are cited in their manifold methods.
        """
        x_parts = self._split_ambient(x)
        w_parts = self._split_ambient(w)
        return self._join([
            m.projection(x_i, w_i)
            for m, x_i, w_i in zip(self.manifolds, x_parts, w_parts, strict=True)
        ])

    def egrad_to_rgrad(self, point: torch.Tensor, egrad: torch.Tensor) -> torch.Tensor:
        """Apply each component's metric inverse, including its weight.

        Args:
          point: (B, ambient_dim) base point.
          egrad: (B, ambient_dim) Euclidean gradient.
        Returns:
          (B, ambient_dim) metric gradient in tangent coordinates.

        References:
          Absil, Mahony, Sepulchre (2008), §3.6, direct-sum metric duality.
        """
        points = self._split_ambient(point)
        gradients = self._split_ambient(egrad)
        # The weighted metric on component i is w_i · g_i, whose inverse is
        # g_i^{-1} / w_i. Dividing each slice by its weight gives the gradient.
        return self._join([
            m.egrad_to_rgrad(p, g) / weight
            for m, p, g, weight in zip(
                self.manifolds, points, gradients, self.weights, strict=True,
            )
        ])

    def inner(
        self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor,
    ) -> torch.Tensor:
        """Weighted direct-sum inner product:
        `⟨u, v⟩_x = Σ_i w_i · ⟨u_i, v_i⟩_{x_i}`.

        References:
          Lee (2018), Introduction to Riemannian Manifolds, product metrics.
          The component formulas are cited in their manifold methods.
        """
        x_parts = self._split_ambient(x)
        u_parts = self._split_ambient(u)
        v_parts = self._split_ambient(v)
        terms = [
            weight * m.inner(x_i, u_i, v_i)
            for m, x_i, u_i, v_i, weight in zip(
                self.manifolds, x_parts, u_parts, v_parts, self.weights, strict=True,
            )
        ]
        return torch.stack(terms).sum(dim=0)

    def norm(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """`sqrt(⟨v, v⟩_x)`. Autograd-safe at v=0.

        References:
          Lee (2018), Introduction to Riemannian Manifolds, product metrics.
          The component formulas are cited in their manifold methods.
        """
        return _safe_sqrt(self.inner(x, v, v))

    # ----------------------------------------------------------------
    # Distance, exp, log
    # ----------------------------------------------------------------

    def distance(
        self, x: torch.Tensor, y: torch.Tensor,
    ) -> torch.Tensor:
        """Pythagorean: `d² = Σ_i w_i · d_i²(x_i, y_i)`.

        References:
          Lee (2018), Introduction to Riemannian Manifolds, product metrics.
          The component formulas are cited in their manifold methods.
        """
        x_parts = self._split_ambient(x)
        y_parts = self._split_ambient(y)
        # Scaling each distance by sqrt(w_i) makes its square w_i * d_i^2, so
        # the vector norm over components is the weighted Pythagorean sum.
        distances = [
            math.sqrt(weight) * m.distance(x_i, y_i)
            for m, x_i, y_i, weight in zip(
                self.manifolds, x_parts, y_parts, self.weights, strict=True,
            )
        ]
        return torch.linalg.vector_norm(torch.stack(distances, dim=-1), dim=-1)

    def exp(
        self, x: torch.Tensor, v: torch.Tensor,
    ) -> torch.Tensor:
        """Per-component exponential map.

        References:
          Lee (2018), Introduction to Riemannian Manifolds, product metrics.
          The component formulas are cited in their manifold methods.
        """
        x_parts = self._split_ambient(x)
        v_parts = self._split_ambient(v)
        return self._join([
            m.exp(x_i, v_i)
            for m, x_i, v_i in zip(self.manifolds, x_parts, v_parts, strict=True)
        ])

    def log(
        self, x: torch.Tensor, y: torch.Tensor,
    ) -> torch.Tensor:
        """Per-component logarithmic map.

        References:
          Lee (2018), Introduction to Riemannian Manifolds, product metrics.
          The component formulas are cited in their manifold methods.
        """
        x_parts = self._split_ambient(x)
        y_parts = self._split_ambient(y)
        return self._join([
            m.log(x_i, y_i)
            for m, x_i, y_i in zip(self.manifolds, x_parts, y_parts, strict=True)
        ])

    def retraction(
        self, x: torch.Tensor, v: torch.Tensor,
    ) -> torch.Tensor:
        """Per-component retraction. A component whose retraction is its
        exponential map gets exp here.

        References:
          Lee (2018), Introduction to Riemannian Manifolds, product metrics.
          The component formulas are cited in their manifold methods.
        """
        x_parts = self._split_ambient(x)
        v_parts = self._split_ambient(v)
        return self._join([
            m.retraction(x_i, v_i)
            for m, x_i, v_i in zip(self.manifolds, x_parts, v_parts, strict=True)
        ])

    # ----------------------------------------------------------------
    # Tangent-at-origin convenience (exp_0 / log_0)
    # ----------------------------------------------------------------

    def exp_0(self, v: torch.Tensor) -> torch.Tensor:
        """Embed a flat Euclidean tangent `(B, Σ dim_i)` onto the
        product manifold via per-submanifold `exp_0`. Result is the
        flat ambient form `(B, Σ ambient_dim_i)`.

        A submanifold without `exp_0` raises AttributeError, which is the
        intended failure. Typical products (Euclidean, hyperbolic,
        κ-stereographic) all provide it.

        References:
          Lee (2018), Introduction to Riemannian Manifolds, product metrics.
          The component formulas are cited in their manifold methods.
        """
        if v.ndim < 2 or v.shape[-1] != self.dim:
            raise ValueError(
                f"v last dim must be Σ intrinsic dims = {self.dim}, "
                f"got {tuple(v.shape)}"
            )
        v_parts = self._split_intrinsic(v)
        return self._join([
            m.exp_0(v_i)
            for m, v_i in zip(self.manifolds, v_parts, strict=True)
        ])

    def log_0(self, y: torch.Tensor) -> torch.Tensor:
        """Inverse of `exp_0`: flat-ambient point → flat-intrinsic
        tangent. Per-submanifold `log_0` concatenated.

        References:
          Lee (2018), Introduction to Riemannian Manifolds, product metrics.
          The component formulas are cited in their manifold methods.
        """
        if y.ndim < 2 or y.shape[-1] != self.ambient_dim:
            raise ValueError(
                f"y last dim must be Σ ambient dims = {self.ambient_dim}, "
                f"got {tuple(y.shape)}"
            )
        y_parts = self._split_ambient(y)
        return self._join([
            m.log_0(y_i)
            for m, y_i in zip(self.manifolds, y_parts, strict=True)
        ])
