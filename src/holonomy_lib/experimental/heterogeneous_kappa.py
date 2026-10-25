# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Synoros

"""Experimental per-point curvature operations in stereographic coordinates.

Status: experimental, in progress. `dissimilarity` evaluates a constant-
curvature distance under a combined pair curvature. It can violate the
triangle inequality and is not a variable-curvature geodesic metric.
Both endpoints must lie in the chart domain under the combined curvature.
`exp_0` and `log_0` apply the ordinary constant-curvature maps independently
at each point's supplied curvature; they do not integrate a curvature field.

References:
  Bachmann, Bécigneul, Ganea (2020). Constant Curvature Graph Convolutional
    Networks, ICML, §2 (the homogeneous formulas used per point/pair).
  The pair-curvature combination is an experimental modeling choice,
    not a formula established by that reference.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch

from holonomy_lib.manifolds.stereographic import (
    SPHERICAL_SAMPLE_RADIUS_FRACTION,
    _kappa_scale,
    _project_ball,
    _stereographic_distance,
)


# Built-in combiners. Each takes two κ tensors of the same shape and
# returns a single pair-effective κ.
#
# The arithmetic mean is the default. It is linear in each κ, and when
# κ_a = κ_b it returns that κ, so the pair formula reduces to the standard
# one. With opposite signs the effective κ can pass through 0, where the
# sign dispatch in `_kappa_scale` switches to the Euclidean series.
#
# The harmonic mean also returns the shared κ when κ_a = κ_b, for either
# sign. It is singular when the two sum to zero, and the call raises then.
def _combiner_arithmetic_mean(
    kappa_a: torch.Tensor, kappa_b: torch.Tensor,
) -> torch.Tensor:
    """`(κ_a + κ_b) / 2`; the default per-pair κ combiner."""
    return 0.5 * (kappa_a + kappa_b)


def _combiner_harmonic_mean(
    kappa_a: torch.Tensor, kappa_b: torch.Tensor,
) -> torch.Tensor:
    """Signed harmonic mean; zero sum is singular and raises ValueError.

    One zero operand with a nonzero other operand has the continuous
    extension zero. Opposite operands, including (0, 0), have no such
    unique extension.
    """
    denominator = kappa_a + kappa_b
    if (denominator == 0).any():
        raise ValueError(
            "harmonic curvature mean is undefined when kappa_a + kappa_b = 0"
        )
    return 2.0 * kappa_a * (kappa_b / denominator)



_BUILTIN_COMBINERS = {
    "arithmetic_mean": _combiner_arithmetic_mean,
    "harmonic_mean": _combiner_harmonic_mean,
}


class HeterogeneousKappaManifold:
    """κ-stereographic geometry with per-point curvature.

    Args:
      n: intrinsic / ambient dimension (same; open subset of `R^n`).
      combiner: how to combine two per-point κ's into a single pair-κ
        for pairwise dissimilarity calculations. Either a built-in name
        (`"arithmetic_mean"` default, `"harmonic_mean"`) or a callable
        `(κ_a, κ_b) → κ_eff` that's commutative and well-defined for
        the supplied curvatures (the harmonic mean excludes zero sums).
      device, dtype: tensor placement / precision for `random_point`
        / `origin`.

    Every method that involves curvature takes an explicit κ tensor (or a
    pair of them). The class stores no κ. The caller owns it, which keeps
    the manifold a pure math primitive that any parameterization (smooth
    field, per-point residual, gated mixture) can sit on top of.

    Example:
      >>> mfd = HeterogeneousKappaManifold(n=4)
      >>> # 10 concepts, each with its own learnable κ
      >>> kappas = torch.nn.Parameter(torch.randn(10) * 0.5)
      >>> x = torch.randn(10, 4) * 0.1   # coordinates
      >>> # Pairwise distance: take coord + κ for each side
      >>> d_01 = mfd.dissimilarity(x[0:1], kappas[0:1], x[1:2], kappas[1:2])
    """

    def __init__(
        self,
        n: int,
        combiner: str | Callable = "arithmetic_mean",
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float64,
    ):
        if n <= 0:
            raise ValueError(f"n must be > 0, got n={n}")
        self.n = n
        if isinstance(combiner, str):
            if combiner not in _BUILTIN_COMBINERS:
                raise ValueError(
                    f"unknown combiner {combiner!r}; "
                    f"built-ins: {sorted(_BUILTIN_COMBINERS)}"
                )
            self.combiner = _BUILTIN_COMBINERS[combiner]
        elif callable(combiner):
            self.combiner = combiner
        else:
            raise TypeError(
                f"combiner must be str or callable, got {type(combiner).__name__}"
            )
        self.device = torch.device(device)
        self.dtype = dtype

    @property
    def dim(self) -> int:
        return self.n

    @property
    def ambient_dim(self) -> int:
        return self.n


    # ----------------------------------------------------------------
    # κ-trig helpers: per-point κ, dynamic sign dispatch
    # ----------------------------------------------------------------
    #
    # Each helper takes alpha (a tangent-like scalar per point) and
    # kappa (per-point curvature, same shape as alpha). Computes the
    # branch-aware κ-trig using torch.where on sign(κ). Works
    # uniformly for κ ∈ R, no static branch lock. Same dynamic
    # dispatch as the sign-flip-during-training case in
    # `KappaStereographicManifold`.

    def _tan_kappa_c(
        self, alpha: torch.Tensor, kappa: torch.Tensor,
    ) -> torch.Tensor:
        """`tan(√κ·α) / (√κ·α)` for κ > 0, `tanh(√|κ|·α) / (√|κ|·α)` for κ < 0,
        and the series at zero. Smooth across κ ∈ R."""
        return _kappa_scale(alpha, kappa)

    def _atan_kappa_c(
        self, alpha: torch.Tensor, kappa: torch.Tensor,
    ) -> torch.Tensor:
        """Signed-curvature inverse scale, including its derivative at zero."""
        return _kappa_scale(alpha, kappa, inverse=True)

    # ----------------------------------------------------------------
    # Construction
    # ----------------------------------------------------------------

    def origin(self, batch_size: int = 1) -> torch.Tensor:
        """Origin `(0, …, 0) ∈ R^n`. It is the same point in every curvature's
        κ-stereographic chart. Curvature changes how directions behave, not
        where the origin sits.

        References:
          Bachmann, Bécigneul, Ganea (2020), Constant Curvature Graph
          Convolutional Networks, §2, homogeneous formula applied per point.
        """
        if batch_size < 0:
            raise ValueError(f"batch_size must be >= 0, got {batch_size}")
        return torch.zeros(batch_size, self.n,
                           device=self.device, dtype=self.dtype)

    def random_point(
        self,
        batch_size: int = 1,
        kappa: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Sample `batch_size` random points on the manifold.

        For a heterogeneous-κ manifold the per-point curvature is
        the user's modeling choice, supplied as a
        `kappa` tensor of shape `(batch_size,)` (or `(batch_size, 1)`
        broadcastable). If `kappa` is omitted, we default to a
        small standard-normal κ per point; useful for tests but
        not a substantively-motivated prior.

        The standard sampling pattern: draw a Euclidean tangent
        `v ~ N(0, σ² I)` and apply `exp_0(v, κ)`. This stays
        inside the manifold's domain when `‖v‖` is small enough
        (the typical regime for tangent-at-origin training init).

        References:
          Bachmann, Bécigneul, Ganea (2020), Constant Curvature Graph
          Convolutional Networks, §2, homogeneous formula applied per point.
        """
        if batch_size < 0:
            raise ValueError(f"batch_size must be >= 0, got {batch_size}")
        draw_device = generator.device if generator is not None else self.device
        # Default κ: per-point standard normal scaled by 0.5
        # (gives a mix of mild spherical / hyperbolic for tests).
        if kappa is None:
            kappa = 0.5 * torch.randn(
                batch_size, generator=generator,
                device=draw_device, dtype=self.dtype,
            ).to(self.device)
        elif kappa.shape == (batch_size, 1):
            kappa = kappa.squeeze(-1)
        elif kappa.shape != (batch_size,):
            raise ValueError(
                f"kappa shape {tuple(kappa.shape)} must be ({batch_size},)"
            )
        # Small tangent scale so spherical samples stay in-domain.
        v = 0.5 * 0.5 * torch.randn(
            batch_size, self.n, generator=generator,
            device=draw_device, dtype=self.dtype,
        ).to(self.device)
        kappa = kappa.to(device=v.device, dtype=v.dtype)
        # For κ > 0 the sampled tangent norm is capped at 1/8 of the model
        # sphere's diameter π/√κ, a quarter of the tan pole radius π/(2√κ).
        # The where() keeps the root finite for κ ≤ 0, where the cap is unused.
        positive_root = torch.sqrt(
            torch.where(kappa > 0, kappa, torch.ones_like(kappa))
        )
        cap = SPHERICAL_SAMPLE_RADIUS_FRACTION * math.pi / positive_root
        # Draws past the cap are pulled back to it, and only where κ > 0.
        # safe_norm keeps the division finite for a zero draw.
        norm = torch.linalg.vector_norm(v, dim=-1)
        safe_norm = torch.where(norm > 0, norm, torch.ones_like(norm))
        scale = torch.where((kappa > 0) & (norm > cap), cap / safe_norm,
                            torch.ones_like(norm))
        return self.exp_0(v * scale.unsqueeze(-1), kappa)

    def is_on_manifold(
        self, x: torch.Tensor, kappa: torch.Tensor,
        atol: float = 1e-9,
    ) -> torch.Tensor:
        """Check finite coordinates and -κ||x||² < 1 - atol for κ < 0.

        Nonnegative curvature admits every finite stereographic coordinate.

        Args:
          x: `(..., n)` coordinates.
          kappa: `(...)` per-point curvature, matching the batch shape of `x`.
          atol: margin kept inside the open ball for κ < 0.
        Returns:
          Boolean `(...)` mask, true where the point is finite and in the chart.

        References:
          Bachmann et al. (2020), §2, homogeneous chart domains.
        """
        if x.ndim < 2 or x.shape[-1] != self.n:
            raise ValueError(f"x must have shape (..., {self.n}); got {tuple(x.shape)}")
        if kappa.shape != x.shape[:-1]:
            raise ValueError("kappa must match the point batch shape")
        domain = (kappa >= 0) | (-kappa * x.square().sum(dim=-1) < 1.0 - atol)
        return torch.isfinite(x).all(dim=-1) & torch.isfinite(kappa) & domain

    # ----------------------------------------------------------------
    # Tangent-at-origin operations (most common substrate use case)
    # ----------------------------------------------------------------

    def exp_0(self, v: torch.Tensor, kappa: torch.Tensor) -> torch.Tensor:
        """Per-point exp at the origin: `exp_0(v_i; κ_i)`.

        Args:
          v: `(B, n)` Euclidean tangents at the origin (one per point).
          kappa: `(B,)` per-point curvatures.
        Returns:
          `(B, n)` embedded points, each at its own curvature. For κ < 0,
          large tangents saturate tanh, and `_project_ball` clips them just
          inside the ball of radius about 1/√|κ|.
          Spherical exp/log are inverses only for ||v|| < π/(2√κ).

        References:
          Bachmann, Bécigneul, Ganea (2020), Constant Curvature Graph
          Convolutional Networks, §2, homogeneous formula applied per point.
        """
        if v.ndim < 2 or v.shape[-1] != self.n:
            raise ValueError(
                f"v last dim must be n={self.n}, got {tuple(v.shape)}"
            )
        if kappa.shape != v.shape[:-1]:
            raise ValueError(
                f"kappa shape {tuple(kappa.shape)} must match v.shape[:-1] = "
                f"{tuple(v.shape[:-1])}"
            )
        # The scale tan(√κ|v|)/(√κ|v|) multiplies v, so the image sits on v's
        # ray at distance tan(√κ|v|)/√κ (tanh/√|κ| when κ < 0).
        v_norm = torch.linalg.vector_norm(v, dim=-1)
        scaling = self._tan_kappa_c(v_norm, kappa)
        return _project_ball(scaling.unsqueeze(-1) * v, kappa)

    def log_0(self, y: torch.Tensor, kappa: torch.Tensor) -> torch.Tensor:
        """Per-point log at the origin: `log_0(y_i; κ_i)`. Inverse of
        `exp_0` at the same κ within its injective, unsaturated range.
        Positive curvature uses the principal log; hyperbolic saturation
        cannot be inverted.

        References:
          Bachmann, Bécigneul, Ganea (2020), Constant Curvature Graph
          Convolutional Networks, §2, homogeneous formula applied per point.
        """
        if y.ndim < 2 or y.shape[-1] != self.n:
            raise ValueError(
                f"y last dim must be n={self.n}, got {tuple(y.shape)}"
            )
        if kappa.shape != y.shape[:-1]:
            raise ValueError(
                f"kappa shape {tuple(kappa.shape)} must match y.shape[:-1]"
            )
        y_norm = torch.linalg.vector_norm(y, dim=-1)
        scaling = self._atan_kappa_c(y_norm, kappa)
        return scaling.unsqueeze(-1) * y

    # ----------------------------------------------------------------
    # Pair operations: combine two per-point κ's
    # ----------------------------------------------------------------

    def dissimilarity(
        self,
        x: torch.Tensor, kappa_x: torch.Tensor,
        y: torch.Tensor, kappa_y: torch.Tensor,
    ) -> torch.Tensor:
        """Pair-conditioned dissimilarity under the combined curvature.

        This agrees with homogeneous distance when both curvatures agree,
        but need not obey the triangle inequality. A custom combiner must
        be commutative for symmetry. Invalid combined domains raise ValueError.

        Args:
          x, y: (B, n) coordinates.
          kappa_x, kappa_y: (B,) endpoint curvatures.
        Returns:
          (B,) nonnegative pair-conditioned values, not metric distances.

        References:
          Bachmann et al. (2020), §2, constant-curvature distance only.
          Combining endpoint curvatures is an experimental modeling choice.
        """
        if x.shape != y.shape or kappa_x.shape != kappa_y.shape:
            raise ValueError(
                "endpoint coordinates and curvatures must have matching shapes"
            )
        kappa_pair = self.combiner(kappa_x, kappa_y)
        # Both endpoints are checked under the pair curvature, since that is the
        # chart the distance is computed in. atol=0 means no margin: the strict
        # inequality rejects points on the boundary itself.
        if not (self.is_on_manifold(x, kappa_pair, atol=0.0)
                & self.is_on_manifold(y, kappa_pair, atol=0.0)).all():
            raise ValueError(
                "both endpoints must be valid under the combined curvature"
            )
        return _stereographic_distance(x, y, kappa_pair)
