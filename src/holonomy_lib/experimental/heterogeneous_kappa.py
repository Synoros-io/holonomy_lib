# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Synoros

"""Heterogeneous-curvature κ-stereographic manifold.

Status: experimental, in progress. `distance` combines the two endpoints'
curvatures and evaluates a constant-curvature distance at the combined value.
That is a pair-conditioned dissimilarity, not a metric: it can violate the
triangle inequality. A true variable-curvature distance needs a different
construction.

A κ-stereographic geometry where the curvature `κ` is **not a single
global scalar but varies per point**. This is the natural ML
primitive when concepts in an embedding space have different local
curvatures — e.g. some concepts naturally live in hyperbolic
neighborhoods (hierarchical), others in spherical neighborhoods
(cyclical), and the global geometry should adapt.

This class is the **per-point-κ math primitive**, not an opinionated
"how do you parameterize κ" component. The user owns:

  - the κ values per concept (typically `nn.Parameter` of shape `(N,)`)
  - any structure on those κ's (smooth field, residuals, gating, …)

and passes effective-κ tensors into the manifold methods. The
manifold provides the κ-aware math: distance, exp_0, log_0, etc.,
each with explicit per-point κ arguments. Pair operations combine
two per-point κ's into an effective pair-κ via a configurable
`combiner` (default: arithmetic mean).

Concrete usage patterns the user can build on top:

  - **Pure per-point κ.** `κ = nn.Parameter(torch.randn(N))`,
    `mfd.distance(x[i], κ[i], x[j], κ[j])`.

  - **Smooth κ-field plus per-point residual** (substrate-team
    suggested design):
        effective_κ(c) = κ_field(T[c]) + δ[c]
    where `κ_field` is a callable (small NN / polynomial / radial
    basis) producing a smooth global curvature profile, and
    `δ[c]` is a per-concept residual. Both are learnable; either
    can be disabled (set to 0). The field captures broad structure,
    the residual captures concept-specific deviations.

  - **Gated mixture-of-curvatures.** Per concept, pick from a
    discrete set of κ values via a learned gating network. (Cf.
    GraphMoRE.)

The manifold class implements the math; the parameterization layer
is the user's modeling choice.

Citations (positioning):

  - **Standard prior art** (well-published, freely citable):
    * Bachmann–Bécigneul–Ganea (2020) *Constant Curvature Graph
      Convolutional Networks* (ICML) — the κ-stereographic model
      with a single learnable κ. We extend to per-point κ.
    * Skopek et al. (2019) *Mixed-curvature VAEs* (ICLR) — multiple
      fixed curvatures across components.

  - **Related research on heterogeneous / per-node curvature:**
    * Di Giovanni, Luise, Bronstein (2022) *Heterogeneous manifolds
      for curvature-aware graph embedding* (arXiv 2202.01185) —
      product of a homogeneous factor and a spherically-symmetric
      factor; allows pointwise curvature variation. Closest prior
      art in the "manifold-construction" direction.
    * Fu et al. (2021) *ACE-HGNN* — adaptive global curvature via
      RL. Single curvature, not per-point.
    * Yang et al. (2022) *kHGCN* (arXiv 2212.01793) — per-node
      discrete Ollivier–Ricci curvature as message-passing
      weights inside a single-curvature hyperbolic embedding.
      Different mechanism.
    * Guo et al. (2024) *GraphMoRE* (arXiv 2412.11085, AAAI-2025) —
      mixture-of-experts gating selects per-node Riemannian
      expert from a discrete set of constant-curvature spaces.
      Closest prior art in the "per-node κ" direction; uses
      discrete gating rather than a continuous κ.

  - **Our contribution** (less-established, our research within an
    active direction):
    * **Continuous per-point κ as a real number** (vs. discrete
      gating in GraphMoRE).
    * **Pair-κ combiner abstraction** — distance between two
      points with different κ's reduces to a single effective
      pair-κ via a configurable rule (arithmetic mean default;
      caller can pass any commutative `Callable[[κ_x, κ_y], κ_eff]`).
      This rule isn't standardized in the literature and is a
      design knob; we ship the most defensible default and let
      the user override.
    * **κ-field + per-point residual decomposition** is the
      substrate-team's suggested pattern; we provide the math
      primitive that supports it, not the parameterization
      itself.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from holonomy_lib.manifolds.lorentz import _safe_sqrt
from holonomy_lib.manifolds.stereographic import (
    _safe_atanc,
    _safe_atanhc,
    _safe_tanc,
    _safe_tanhc,
)


# Built-in combiners. Each takes two κ tensors of the same shape and
# returns a single pair-effective κ. The arithmetic mean is the
# simplest defensible default — preserves linearity, has a
# well-defined limit at κ_a = κ_b (recovers the standard formula),
# and degrades gracefully when κ_a and κ_b have opposite signs
# (effective κ can pass through 0, dispatching to the Euclidean
# limit via the dynamic-sign-dispatch helpers).
#
# The harmonic mean is the alternative — better-behaved when both
# κ's are non-zero and same-sign (preserves their geometric mean),
# but blows up when one κ is 0. Tried but not made default.
def _combiner_arithmetic_mean(
    kappa_a: torch.Tensor, kappa_b: torch.Tensor,
) -> torch.Tensor:
    """`(κ_a + κ_b) / 2` — the default per-pair κ combiner."""
    return 0.5 * (kappa_a + kappa_b)


def _combiner_harmonic_mean(
    kappa_a: torch.Tensor, kappa_b: torch.Tensor,
) -> torch.Tensor:
    """`2 / (1/κ_a + 1/κ_b)` — preserves same-sign behavior; ill-defined
    when either κ = 0. Caller is responsible for keeping κ's away
    from 0 when using this combiner."""
    tiny = torch.finfo(kappa_a.dtype).tiny
    return 2.0 / (1.0 / kappa_a.clamp(min=tiny) + 1.0 / kappa_b.clamp(min=tiny))


_BUILTIN_COMBINERS = {
    "arithmetic_mean": _combiner_arithmetic_mean,
    "harmonic_mean": _combiner_harmonic_mean,
}


class HeterogeneousKappaManifold:
    """κ-stereographic geometry with per-point curvature.

    Args:
      n: intrinsic / ambient dimension (same — open subset of `R^n`).
      combiner: how to combine two per-point κ's into a single pair-κ
        for distance / log calculations. Either a built-in name
        (`"arithmetic_mean"` default, `"harmonic_mean"`) or a callable
        `(κ_a, κ_b) → κ_eff` that's commutative and well-defined for
        any real κ's.
      device, dtype: tensor placement / precision for `random_point`
        / `origin`.

    All methods that involve curvature take an explicit κ tensor (or
    pair of κ tensors). The class does NOT store κ — that's the
    user's responsibility. This keeps the manifold a pure math
    primitive and lets the user attach any parameterization on top
    (smooth field, per-point residual, gated mixture, …).

    Example:
      >>> mfd = HeterogeneousKappaManifold(n=4)
      >>> # 10 concepts, each with its own learnable κ
      >>> kappas = torch.nn.Parameter(torch.randn(10) * 0.5)
      >>> x = torch.randn(10, 4) * 0.1   # coordinates
      >>> # Pairwise distance: take coord + κ for each side
      >>> d_01 = mfd.distance(x[0:1], kappas[0:1], x[1:2], kappas[1:2])
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
    # κ-trig helpers — per-point κ, dynamic sign dispatch
    # ----------------------------------------------------------------
    #
    # Each helper takes alpha (a tangent-like scalar per point) and
    # kappa (per-point curvature, same shape as alpha). Computes the
    # branch-aware κ-trig using torch.where on sign(κ). Works
    # uniformly for κ ∈ R, no static branch lock — same dynamic
    # dispatch as the sign-flip-during-training case in
    # `KappaStereographicManifold`.

    def _tan_kappa_c(
        self, alpha: torch.Tensor, kappa: torch.Tensor,
    ) -> torch.Tensor:
        """`tan_κ(√|κ|·α) / (√|κ|·α)`, smooth across κ ∈ R."""
        abs_k = torch.abs(kappa).clamp(min=torch.finfo(alpha.dtype).tiny)
        scaled = torch.sqrt(abs_k) * alpha
        return torch.where(
            kappa > 0, _safe_tanc(scaled), _safe_tanhc(scaled),
        )

    def _atan_kappa_c(
        self, alpha: torch.Tensor, kappa: torch.Tensor,
    ) -> torch.Tensor:
        """`arctan_κ(√|κ|·α) / (√|κ|·α)`, smooth across κ ∈ R."""
        abs_k = torch.abs(kappa).clamp(min=torch.finfo(alpha.dtype).tiny)
        scaled = torch.sqrt(abs_k) * alpha
        return torch.where(
            kappa > 0, _safe_atanc(scaled), _safe_atanhc(scaled),
        )

    # ----------------------------------------------------------------
    # Construction
    # ----------------------------------------------------------------

    def origin(self, batch_size: int = 1) -> torch.Tensor:
        """Origin `(0, …, 0) ∈ R^n`. The origin is the SAME point on
        every curvature's κ-stereographic — curvature varies what
        directions look like, not where the origin is."""
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
        the user's modeling choice, so this method requires a
        `kappa` tensor of shape `(batch_size,)` (or `(batch_size, 1)`
        broadcastable). If `kappa` is omitted, we default to a
        small standard-normal κ per point — useful for tests but
        not a substantively-motivated prior.

        The standard sampling pattern: draw a Euclidean tangent
        `v ~ N(0, σ² I)` and apply `exp_0(v, κ)`. This stays
        inside the manifold's domain when `‖v‖` is small enough
        (the typical regime for tangent-at-origin training init).
        """
        if batch_size < 0:
            raise ValueError(f"batch_size must be >= 0, got {batch_size}")
        # Default κ: per-point standard normal scaled by 0.5
        # (gives a mix of mild spherical / hyperbolic for tests).
        if kappa is None:
            kappa = 0.5 * torch.randn(
                batch_size, generator=generator,
                device=self.device, dtype=self.dtype,
            )
        elif kappa.shape != (batch_size,):
            raise ValueError(
                f"kappa shape {tuple(kappa.shape)} must be ({batch_size},)"
            )
        # Small tangent scale so spherical samples stay in-domain.
        v = 0.5 * 0.5 * torch.randn(
            batch_size, self.n, generator=generator,
            device=self.device, dtype=self.dtype,
        )
        return self.exp_0(v, kappa)

    def is_on_manifold(
        self, x: torch.Tensor, kappa: torch.Tensor,
        atol: float = 1e-9,
    ) -> torch.Tensor:
        """Per-element membership test: `κ · ‖x‖² < 1 − atol` for
        κ > 0 (spherical domain bound); always True for κ ≤ 0."""
        norm_sq = (x * x).sum(dim=-1)
        return (kappa * norm_sq) < (1.0 - atol)

    # ----------------------------------------------------------------
    # Tangent-at-origin operations (most common substrate use case)
    # ----------------------------------------------------------------

    def exp_0(self, v: torch.Tensor, kappa: torch.Tensor) -> torch.Tensor:
        """Per-point exp at the origin: `exp_0(v_i; κ_i)`.

        Args:
          v: `(B, n)` Euclidean tangents at the origin (one per point).
          kappa: `(B,)` per-point curvatures.
        Returns:
          `(B, n)` embedded points, each at its own curvature.
        """
        if v.shape[-1] != self.n:
            raise ValueError(
                f"v last dim must be n={self.n}, got {v.shape[-1]}"
            )
        if kappa.shape != v.shape[:-1]:
            raise ValueError(
                f"kappa shape {tuple(kappa.shape)} must match v.shape[:-1] = "
                f"{tuple(v.shape[:-1])}"
            )
        v_sq = (v * v).sum(dim=-1)
        v_norm = _safe_sqrt(v_sq)
        scaling = self._tan_kappa_c(v_norm, kappa)
        return scaling.unsqueeze(-1) * v

    def log_0(self, y: torch.Tensor, kappa: torch.Tensor) -> torch.Tensor:
        """Per-point log at the origin: `log_0(y_i; κ_i)`. Inverse of
        `exp_0` at the same κ."""
        if y.shape[-1] != self.n:
            raise ValueError(
                f"y last dim must be n={self.n}, got {y.shape[-1]}"
            )
        if kappa.shape != y.shape[:-1]:
            raise ValueError(
                f"kappa shape {tuple(kappa.shape)} must match y.shape[:-1]"
            )
        y_sq = (y * y).sum(dim=-1)
        y_norm = _safe_sqrt(y_sq)
        scaling = self._atan_kappa_c(y_norm, kappa)
        return scaling.unsqueeze(-1) * y

    # ----------------------------------------------------------------
    # Pair operations: combine two per-point κ's
    # ----------------------------------------------------------------

    def distance(
        self,
        x: torch.Tensor, kappa_x: torch.Tensor,
        y: torch.Tensor, kappa_y: torch.Tensor,
    ) -> torch.Tensor:
        """Geodesic distance between points `x` and `y` with per-point
        curvatures `κ_x` and `κ_y`.

        The two curvatures are combined into a single pair-effective
        κ via `self.combiner(κ_x, κ_y)`. The standard
        constant-curvature distance formula at the combined κ is then
        evaluated, using the unified arcsinh / arctan / arctanh
        dispatch.

        Args:
          x, y: `(B, n)` coordinate tensors.
          kappa_x, kappa_y: `(B,)` per-point curvatures.

        Returns:
          `(B,)` non-negative distances.

        **Limit semantics:**
          - `κ_x = κ_y` (homogeneous case): reduces to the standard
            constant-curvature distance — matches
            `KappaStereographicManifold(κ).distance` exactly.
          - `κ_x = -κ_y` (opposite-sign mix): arithmetic-mean
            combiner gives `κ_eff = 0`, Euclidean distance. Harmonic
            combiner gives `κ_eff = 0/0` (blows up, hence the
            clamp). The arithmetic default is more defensible at
            sign mixes.
        """
        kappa_pair = self.combiner(kappa_x, kappa_y)
        # Now compute the distance under the (per-element) pair-κ
        # using the unified formula `2·d·_atan_kappa_c(d; κ_pair)`.
        diff = self._mobius_add(-x, y, kappa_pair)
        diff_sq = (diff * diff).sum(dim=-1)
        diff_norm = _safe_sqrt(diff_sq)
        return 2.0 * diff_norm * self._atan_kappa_c(diff_norm, kappa_pair)

    def _mobius_add(
        self,
        x: torch.Tensor, y: torch.Tensor, kappa: torch.Tensor,
    ) -> torch.Tensor:
        """Möbius `x ⊕_κ y` with per-point κ. Same algebra as
        `KappaStereographicManifold.mobius_add` but κ is broadcast
        per-element (each pair has its own κ_pair from the combiner).
        """
        xy = (x * y).sum(dim=-1, keepdim=True)
        xx = (x * x).sum(dim=-1, keepdim=True)
        yy = (y * y).sum(dim=-1, keepdim=True)
        k = kappa.unsqueeze(-1)
        num = (1.0 - 2.0 * k * xy - k * yy) * x + (1.0 + k * xx) * y
        denom = 1.0 - 2.0 * k * xy + k * k * xx * yy
        return num / denom
