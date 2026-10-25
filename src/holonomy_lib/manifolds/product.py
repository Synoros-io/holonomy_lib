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
delegates to the corresponding submanifold method per slice, so a
mixed-curvature embedding `concept_c = (x_E, x_H, x_S)` with one
Euclidean, one hyperbolic, and one spherical component just works
through the existing `holonomy_lib.hyperbolic.*` primitives.

References:
  Skopek, O., Ganea, O.-E., Bécigneul, G. (2019). Mixed-curvature
    Variational Autoencoders. ICLR. (mixed-curvature embedding for
    deep generative models)
  Gu, A., Sala, F., Gunel, B., Ré, C. (2019). Learning mixed-
    curvature representations in product spaces. ICLR.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from holonomy_lib.manifolds.lorentz import _safe_sqrt
from holonomy_lib.provenance import register_provenance_class, with_provenance


@register_provenance_class("ProductManifold")
class ProductManifold:
    """Riemannian product of one or more manifolds.

    Args:
      manifolds: ordered list of submanifolds. Each must expose
        `.dim`, `.ambient_dim`, `random_point`, `origin`,
        `is_on_manifold`, `distance`, `exp`, `log`, `inner`, `norm`,
        `projection`, `retraction`. Heterogeneous mixes are fine —
        e.g. `[KappaStereographicManifold(κ=-1, n=4), LorentzManifold(n=4)]`.
      weights: optional per-submanifold non-negative scalars `w_i` on
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
            if any(w < 0 for w in weights):
                raise ValueError("weights must be non-negative")
            self.weights = [float(w) for w in weights]
        self.device = torch.device(device)
        self.dtype = dtype

        # Precompute ambient-slice boundaries: offsets[i] is the
        # starting flat-index of submanifold i.
        self._ambient_dims = [m.ambient_dim for m in self.manifolds]
        offsets = [0]
        for d in self._ambient_dims:
            offsets.append(offsets[-1] + d)
        self._ambient_slices = list(zip(offsets[:-1], offsets[1:]))

        # Intrinsic-slice boundaries (for exp_0 / log_0 which use
        # the intrinsic dim).
        self._intrinsic_dims = [m.dim for m in self.manifolds]
        i_offsets = [0]
        for d in self._intrinsic_dims:
            i_offsets.append(i_offsets[-1] + d)
        self._intrinsic_slices = list(zip(i_offsets[:-1], i_offsets[1:]))

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
        """Split a flat ambient tensor into per-submanifold slices."""
        return [x[..., s:e] for s, e in self._ambient_slices]

    def _split_intrinsic(self, v: torch.Tensor) -> list[torch.Tensor]:
        """Split a flat intrinsic tangent into per-submanifold slices."""
        return [v[..., s:e] for s, e in self._intrinsic_slices]

    def _join(self, pieces: list[torch.Tensor]) -> torch.Tensor:
        """Concatenate per-submanifold tensors back into flat form."""
        return torch.cat(pieces, dim=-1)

    def component(self, x: torch.Tensor, index: int) -> torch.Tensor:
        """Extract the `index`-th submanifold's coordinates from a
        flat point tensor `x`. Public convenience for inspection /
        diagnostics."""
        s, e = self._ambient_slices[index]
        return x[..., s:e]

    # ----------------------------------------------------------------
    # Provenance
    # ----------------------------------------------------------------

    def _provenance_signature(self) -> dict:
        return {
            "class": "ProductManifold",
            "submanifolds": [m._provenance_signature() for m in self.manifolds],
            "weights": list(self.weights),
            "device": str(self.device),
            "dtype": str(self.dtype),
        }

    @classmethod
    def _from_signature(cls, sig: dict) -> ProductManifold:
        # Reconstruct each submanifold via its own _from_signature. The
        # registry stores the bound `_from_signature` callable, so we
        # apply it directly (not on the class).
        from holonomy_lib.provenance.protocol import _CLASS_REGISTRY
        submanifolds = []
        for sub_sig in sig["submanifolds"]:
            from_sig = _CLASS_REGISTRY[sub_sig["class"]]
            submanifolds.append(from_sig(sub_sig))
        dtype_name = sig["dtype"].split(".")[-1]
        return cls(
            manifolds=submanifolds,
            weights=sig["weights"],
            device=sig["device"],
            dtype=getattr(torch, dtype_name),
        )

    # ----------------------------------------------------------------
    # Construction
    # ----------------------------------------------------------------

    def origin(self, batch_size: int = 1) -> torch.Tensor:
        """Concatenated origins of every submanifold."""
        if batch_size < 0:
            raise ValueError(f"batch_size must be >= 0, got {batch_size}")
        return self._join([
            m.origin(batch_size=batch_size) for m in self.manifolds
        ])

    def random_point(
        self,
        batch_size: int = 1,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Sample by drawing independently on each submanifold and
        concatenating. Each submanifold's `random_point` decides its
        own distribution (uniform / Gaussian tangent / etc.)."""
        if batch_size < 0:
            raise ValueError(f"batch_size must be >= 0, got {batch_size}")
        return self._join([
            m.random_point(batch_size=batch_size, generator=generator)
            for m in self.manifolds
        ])

    def is_on_manifold(
        self, x: torch.Tensor, atol: float = 1e-9,
    ) -> torch.Tensor:
        """A product point is on the manifold iff every component is
        on its submanifold."""
        parts = self._split_ambient(x)
        on_each = torch.stack(
            [m.is_on_manifold(p, atol=atol) for m, p in zip(self.manifolds, parts)],
            dim=0,
        )
        return on_each.all(dim=0)

    # ----------------------------------------------------------------
    # Tangent operations
    # ----------------------------------------------------------------

    @with_provenance(
        "holonomy_lib.manifolds.ProductManifold.projection",
        op_version="0.1",
    )
    def projection(
        self, x: torch.Tensor, w: torch.Tensor,
    ) -> torch.Tensor:
        """Project the ambient direction `w` (flat) onto the tangent
        space at `x`, per-submanifold."""
        x_parts = self._split_ambient(x)
        w_parts = self._split_ambient(w)
        return self._join([
            m.projection(x_i, w_i)
            for m, x_i, w_i in zip(self.manifolds, x_parts, w_parts)
        ])

    @with_provenance(
        "holonomy_lib.manifolds.ProductManifold.inner",
        op_version="0.1",
    )
    def inner(
        self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor,
    ) -> torch.Tensor:
        """Weighted direct-sum inner product:
        `⟨u, v⟩_x = Σ_i w_i · ⟨u_i, v_i⟩_{x_i}`."""
        x_parts = self._split_ambient(x)
        u_parts = self._split_ambient(u)
        v_parts = self._split_ambient(v)
        total = None
        for m, x_i, u_i, v_i, w_i in zip(
            self.manifolds, x_parts, u_parts, v_parts, self.weights,
        ):
            ip = m.inner(x_i, u_i, v_i)
            term = w_i * ip
            total = term if total is None else total + term
        return total

    @with_provenance(
        "holonomy_lib.manifolds.ProductManifold.norm",
        op_version="0.1",
    )
    def norm(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """`sqrt(⟨v, v⟩_x)`. Autograd-safe at v=0."""
        return _safe_sqrt(self.inner(x, v, v))

    # ----------------------------------------------------------------
    # Distance, exp, log
    # ----------------------------------------------------------------

    @with_provenance(
        "holonomy_lib.manifolds.ProductManifold.distance",
        op_version="0.1",
    )
    def distance(
        self, x: torch.Tensor, y: torch.Tensor,
    ) -> torch.Tensor:
        """Pythagorean: `d² = Σ_i w_i · d_i²(x_i, y_i)`."""
        x_parts = self._split_ambient(x)
        y_parts = self._split_ambient(y)
        d_sq = None
        for m, x_i, y_i, w_i in zip(
            self.manifolds, x_parts, y_parts, self.weights,
        ):
            d_i = m.distance(x_i, y_i)
            term = w_i * (d_i * d_i)
            d_sq = term if d_sq is None else d_sq + term
        return _safe_sqrt(d_sq)

    @with_provenance(
        "holonomy_lib.manifolds.ProductManifold.exp",
        op_version="0.1",
    )
    def exp(
        self, x: torch.Tensor, v: torch.Tensor,
    ) -> torch.Tensor:
        """Per-component exponential map."""
        x_parts = self._split_ambient(x)
        v_parts = self._split_ambient(v)
        return self._join([
            m.exp(x_i, v_i)
            for m, x_i, v_i in zip(self.manifolds, x_parts, v_parts)
        ])

    @with_provenance(
        "holonomy_lib.manifolds.ProductManifold.log",
        op_version="0.1",
    )
    def log(
        self, x: torch.Tensor, y: torch.Tensor,
    ) -> torch.Tensor:
        """Per-component logarithmic map."""
        x_parts = self._split_ambient(x)
        y_parts = self._split_ambient(y)
        return self._join([
            m.log(x_i, y_i)
            for m, x_i, y_i in zip(self.manifolds, x_parts, y_parts)
        ])

    @with_provenance(
        "holonomy_lib.manifolds.ProductManifold.retraction",
        op_version="0.1",
    )
    def retraction(
        self, x: torch.Tensor, v: torch.Tensor,
    ) -> torch.Tensor:
        """Per-component retraction. For submanifolds where
        retraction = exp (Hadamard etc.), this collapses to the
        per-component exp."""
        x_parts = self._split_ambient(x)
        v_parts = self._split_ambient(v)
        return self._join([
            m.retraction(x_i, v_i)
            for m, x_i, v_i in zip(self.manifolds, x_parts, v_parts)
        ])

    # ----------------------------------------------------------------
    # Tangent-at-origin convenience (exp_0 / log_0)
    # ----------------------------------------------------------------

    @with_provenance(
        "holonomy_lib.manifolds.ProductManifold.exp_0",
        op_version="0.1",
    )
    def exp_0(self, v: torch.Tensor) -> torch.Tensor:
        """Embed a flat Euclidean tangent `(B, Σ dim_i)` onto the
        product manifold via per-submanifold `exp_0`. Result is the
        flat ambient form `(B, Σ ambient_dim_i)`.

        Submanifolds without `exp_0` raise AttributeError; that's
        expected — the typical use of ProductManifold has all
        submanifolds supporting the tangent-at-origin embedding
        (Euclidean / hyperbolic / κ-stereographic all do).
        """
        if v.shape[-1] != self.dim:
            raise ValueError(
                f"v last dim must be Σ intrinsic dims = {self.dim}, "
                f"got {v.shape[-1]}"
            )
        v_parts = self._split_intrinsic(v)
        return self._join([
            m.exp_0(v_i)
            for m, v_i in zip(self.manifolds, v_parts)
        ])

    @with_provenance(
        "holonomy_lib.manifolds.ProductManifold.log_0",
        op_version="0.1",
    )
    def log_0(self, y: torch.Tensor) -> torch.Tensor:
        """Inverse of `exp_0`: flat-ambient point → flat-intrinsic
        tangent. Per-submanifold `log_0` concatenated."""
        if y.shape[-1] != self.ambient_dim:
            raise ValueError(
                f"y last dim must be Σ ambient dims = {self.ambient_dim}, "
                f"got {y.shape[-1]}"
            )
        y_parts = self._split_ambient(y)
        return self._join([
            m.log_0(y_i)
            for m, y_i in zip(self.manifolds, y_parts)
        ])
