# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Synoros

"""κ-stereographic projection model — constant curvature manifold
that interpolates spherical (κ > 0), Euclidean (κ = 0), and
hyperbolic (κ < 0) geometry via a single parameter.

The κ-stereographic model represents:

  - **κ > 0** (spherical): the stereographic projection of the
    `n`-sphere of radius `1/√κ` onto the open Euclidean ball
    `{x ∈ R^n : κ‖x‖² < 1}`.
  - **κ = 0** (Euclidean): the standard `R^n`.
  - **κ < 0** (hyperbolic): the Poincaré ball of radius `1/√|κ|`,
    `{x ∈ R^n : ‖x‖² < 1/|κ|}`.

Crucially, all three cases share the SAME formulas with κ as the
only parameter — the Möbius gyro-addition, κ-tangent, conformal
metric, and exp/log maps unify the three geometries. This makes the
manifold a natural choice for "let the curvature be learned from
data" architectures.

Coordinate convention: points live in `R^n` (no extra ambient
dimension, unlike the Lorentz model). The intrinsic and ambient
dimensions both equal `n`. Tangent vectors at any base point are
also `R^n`.

Core formulas (Bachmann–Bécigneul–Ganea 2020, §3):

    λ_κ(x) = 2 / (1 + κ‖x‖²)                          (conformal factor)
    ⟨u, v⟩_x = λ_κ(x)² · ⟨u, v⟩_Eucl                  (Riemannian metric)
    x ⊕_κ y = ((1 - 2κ⟨x,y⟩ - κ‖y‖²) x + (1 + κ‖x‖²) y)
              / (1 - 2κ⟨x,y⟩ + κ²‖x‖² ‖y‖²)           (Möbius addition)
    d_κ(x, y) = (2/√|κ|) · tan_κ⁻¹(√|κ| · ‖(-x) ⊕_κ y‖)
    exp_x(v) = x ⊕_κ (tan_κ(λ_κ(x)·√|κ|·‖v‖/2)
                       · v / (√|κ|·‖v‖))
    log_x(y) = (2/(λ_κ(x)·√|κ|)) · tan_κ⁻¹(√|κ|·‖(-x) ⊕_κ y‖)
                · ((-x) ⊕_κ y) / ‖(-x) ⊕_κ y‖

where `tan_κ` is `tan` for κ > 0, `tanh` for κ < 0, identity for
κ = 0; correspondingly `tan_κ⁻¹` is `arctan` / `arctanh` / identity.

v1 takes κ as a Python float (set at construction time, dispatched
on sign at every call). Learnable / scalar-tensor κ is a planned
extension once the static-κ math is settled.

References:
  Bachmann, G., Bécigneul, G., Ganea, O.-E. (2020). Constant Curvature
    Graph Convolutional Networks. ICML. (the κ-stereographic formulation)
  Skopek, O., Ganea, O.-E., Bécigneul, G. (2019). Mixed-curvature
    Variational Autoencoders. ICLR.
  Ungar, A. A. (2008). A Gyrovector Space Approach to Hyperbolic
    Geometry. (gyro-addition / κ-trig algebra)
  Ganea, O.-E., Bécigneul, G., Hofmann, T. (2018). Hyperbolic Neural
    Networks. NeurIPS. (Poincaré ball special case)
"""

from __future__ import annotations

import math
from typing import Literal

import torch

from holonomy_lib.manifolds.lorentz import _safe_sqrt


def _safe_tanhc(t: torch.Tensor) -> torch.Tensor:
    """tanh(t)/t with autograd-safe handling at t = 0 (limit = 1).

    Same idiom as `_safe_sinhc` in lorentz.py: compute the formula on
    a where-substituted input that's never 0, then mask the output.
    Used in `exp_0` for κ < 0 (hyperbolic / Poincaré ball).
    """
    is_positive = t > 0
    t_safe = torch.where(is_positive, t, torch.ones_like(t))
    return torch.where(
        is_positive,
        torch.tanh(t_safe) / t_safe,
        torch.ones_like(t),
    )


def _safe_tanc(t: torch.Tensor) -> torch.Tensor:
    """tan(t)/t with autograd-safe handling at t = 0 (limit = 1).

    Domain: `t ∈ (-π/2, π/2)` (tan singular at ±π/2). Used in
    `exp_0` for κ > 0 (spherical). Caller is responsible for keeping
    `t < π/2`.
    """
    is_positive = t > 0
    t_safe = torch.where(is_positive, t, torch.ones_like(t))
    return torch.where(
        is_positive,
        torch.tan(t_safe) / t_safe,
        torch.ones_like(t),
    )


def _safe_atanhc(t: torch.Tensor) -> torch.Tensor:
    """arctanh(t)/t with autograd-safe handling at t = 0 (limit = 1).

    Domain: `|t| < 1` (arctanh's natural domain). Used in `log_0` for
    κ < 0. Caller is responsible for keeping points strictly inside
    the Poincaré ball so `√|κ|·‖y‖ < 1`.

    **Substitute-value note** (deliberate inconsistency with the
    other `_safe_*c` helpers in this module): the masked-out
    `t_safe` is `0.5`, NOT `1.0` as in `_safe_tanc` / `_safe_tanhc` /
    `_safe_atanc` / `_safe_sinhc` etc. The reason: `arctanh(1.0) = ∞`
    (the function diverges at the domain boundary), so substituting
    `1.0` would produce a non-finite *value* in the formula branch.
    `torch.where`'s gradient-masking by the boolean condition
    multiplies the formula-branch gradient by `(1 − cond)`; an
    infinite gradient times the boolean-as-int mask gives `∞ · 0 =
    NaN` and propagates. `0.5` is the smallest "round" value
    strictly inside `(-1, 1)` that keeps both the value and
    derivative of `arctanh(0.5)/0.5 ≈ 1.099 / 1.099`-derivative
    finite, so the masking works cleanly. Any value in
    `(-1, 1) \\ {0}` would do; we pick `0.5` for definiteness.
    """
    is_positive = t > 0
    t_safe = torch.where(is_positive, t, torch.ones_like(t) * 0.5)
    return torch.where(
        is_positive,
        torch.atanh(t_safe) / t_safe,
        torch.ones_like(t),
    )


def _safe_atanc(t: torch.Tensor) -> torch.Tensor:
    """arctan(t)/t with autograd-safe handling at t = 0 (limit = 1).

    No domain restriction (arctan is entire). Used in `log_0` for
    κ > 0.
    """
    is_positive = t > 0
    t_safe = torch.where(is_positive, t, torch.ones_like(t))
    return torch.where(
        is_positive,
        torch.atan(t_safe) / t_safe,
        torch.ones_like(t),
    )


# Tag for the κ-sign branch chosen at __init__. Static — set once
# from the Python-float κ and reused on every call.
Branch = Literal["spherical", "euclidean", "hyperbolic"]


class KappaStereographicManifold:
    """Constant-curvature κ-stereographic manifold.

    Args:
      n: intrinsic / ambient dimension. Points are stored as `(B, n)`.
      kappa: constant sectional curvature. v1 accepts a Python float
        only — set the sign at construction, dispatch on it
        thereafter. Default `-1.0` matches `LorentzManifold`'s unit
        hyperbolic case (Poincaré ball of radius 1).
      device, dtype: tensor placement and precision.

    Example:
      >>> mfd = KappaStereographicManifold(n=3, kappa=-1.0)
      >>> x = mfd.random_point(batch_size=4)
      >>> x.shape, mfd.is_on_manifold(x).all().item()
      (torch.Size([4, 3]), True)
    """

    def __init__(self, n: int, kappa: float | torch.Tensor = -1.0,
                 device: str | torch.device = "cpu",
                 dtype: torch.dtype = torch.float64):
        if n <= 0:
            raise ValueError(f"n must be > 0, got n={n}")
        # Accept Python float OR 0-d torch.Tensor (for learnable κ via
        # SGD). The branch ("spherical" / "hyperbolic" / "euclidean")
        # is fixed at construction from the sign of κ's *initial* value;
        # subsequent updates that push κ across 0 produce undefined
        # behavior (the branch dispatch stays fixed, so a positive κ
        # value running through the "hyperbolic" branch's `tanh` will
        # not give meaningful spherical geometry). Keep κ in one
        # sign-half during learning.
        if isinstance(kappa, torch.Tensor):
            if kappa.ndim != 0:
                raise TypeError(
                    "kappa as Tensor must be 0-dim (scalar); "
                    f"got shape {tuple(kappa.shape)}"
                )
            self._kappa_init = float(kappa.item())
            self.kappa: float | torch.Tensor = kappa
            self._kappa_is_tensor = True
        elif isinstance(kappa, (int, float)):
            self._kappa_init = float(kappa)
            self.kappa = float(kappa)
            self._kappa_is_tensor = False
        else:
            raise TypeError(
                "kappa must be float or 0-dim torch.Tensor, got "
                f"{type(kappa).__name__}"
            )
        self.n = n
        self.device = torch.device(device)
        self.dtype = dtype

        # Branch selection: sign of κ at construction time. Once fixed,
        # operations use closed forms for that branch — pushing the
        # learnable κ across 0 won't switch branches at runtime.
        if self._kappa_init > 0:
            self._branch: Branch = "spherical"
        elif self._kappa_init < 0:
            self._branch = "hyperbolic"
        else:
            self._branch = "euclidean"

        # Cached scalar magnitudes for the static-float fast path.
        # For Tensor κ we use the live `|kappa|` instead (computed
        # on demand to keep autograd graph attached).
        if not self._kappa_is_tensor:
            self._abs_kappa = abs(self.kappa)
            self._sqrt_abs_kappa = (
                math.sqrt(self._abs_kappa)
                if self._abs_kappa > 0 else 0.0
            )
        else:
            # Will be computed from `self.kappa` on each call via
            # `_get_sqrt_abs_kappa()` to preserve autograd.
            self._abs_kappa = abs(self._kappa_init)
            self._sqrt_abs_kappa = math.sqrt(self._abs_kappa)

    def _get_sqrt_abs_kappa(self):
        """Return `sqrt(|κ|)` — as a tensor if κ is a tensor (for
        autograd), as a float otherwise.
        """
        if self._kappa_is_tensor:
            return torch.sqrt(torch.abs(self.kappa))
        return self._sqrt_abs_kappa

    def _get_kappa(self):
        """Return κ — as a tensor if κ is a tensor, as a float otherwise."""
        return self.kappa

    @property
    def dim(self) -> int:
        """Intrinsic manifold dimension `n`. Same as ambient for this
        model — no extra dimension (unlike `LorentzManifold(n)` which
        embeds in `R^{n+1}`).
        """
        return self.n

    @property
    def ambient_dim(self) -> int:
        """Ambient dimension `n`. Matches `.dim` for stereographic;
        `LorentzManifold` overrides this to `n + 1`. Used by
        manifold-generic primitives in `holonomy_lib.hyperbolic`.
        """
        return self.n


    # ----------------------------------------------------------------
    # Helpers — κ-trig functions, branch-dispatched
    # ----------------------------------------------------------------

    def _tan_kappa_c(self, alpha: torch.Tensor) -> torch.Tensor:
        """`tan_κ(√|κ|·α) / (√|κ|·α)` with the analytic limit `1` at
        α = 0 OR κ = 0.

        Dual-mode dispatch:
          - **Static-float κ**: fast path using the construction-time
            branch (`spherical` / `hyperbolic` / `Euclidean`).
          - **Tensor κ**: dynamic dispatch on `sign(κ)` at each call,
            so κ can cross 0 during training. Both spherical
            (`tan/scaled`) and hyperbolic (`tanh/scaled`) formulas are
            computed and `torch.where` picks per element. At κ ≈ 0
            the clamped `sqrt(|κ|)` keeps `scaled` near 0, where both
            branches' analytic limits are 1 — so the κ-sign flip is
            continuous.
        """
        if not self._kappa_is_tensor:
            # Static κ — locked branch.
            if self._branch == "euclidean":
                return torch.ones_like(alpha)
            scaled = self._sqrt_abs_kappa * alpha
            if self._branch == "spherical":
                return _safe_tanc(scaled)
            return _safe_tanhc(scaled)
        # Tensor κ — dynamic dispatch so sign(κ) can change during SGD.
        kappa = self._get_kappa()
        abs_k = torch.abs(kappa).clamp(min=torch.finfo(alpha.dtype).tiny)
        scaled = torch.sqrt(abs_k) * alpha
        return torch.where(
            kappa > 0, _safe_tanc(scaled), _safe_tanhc(scaled),
        )

    def _atan_kappa_c(self, alpha: torch.Tensor) -> torch.Tensor:
        """`tan_κ⁻¹(√|κ|·α) / (√|κ|·α)` with the analytic limit `1` at
        α = 0 OR κ = 0.

        Same dual-mode dispatch as `_tan_kappa_c`. Note: the hyperbolic
        branch (`arctanh`) requires the scaled input < 1, i.e. caller
        keeps points inside the Poincaré-ball domain when κ < 0.
        """
        if not self._kappa_is_tensor:
            if self._branch == "euclidean":
                return torch.ones_like(alpha)
            scaled = self._sqrt_abs_kappa * alpha
            if self._branch == "spherical":
                return _safe_atanc(scaled)
            return _safe_atanhc(scaled)
        # Tensor κ — dynamic dispatch
        kappa = self._get_kappa()
        abs_k = torch.abs(kappa).clamp(min=torch.finfo(alpha.dtype).tiny)
        scaled = torch.sqrt(abs_k) * alpha
        return torch.where(
            kappa > 0, _safe_atanc(scaled), _safe_atanhc(scaled),
        )

    def _conformal_factor(self, x: torch.Tensor) -> torch.Tensor:
        """λ_κ(x) = 2 / (1 + κ‖x‖²). At origin = 2.

        Uses `_get_kappa()` so the conformal factor's derivative w.r.t.
        a learnable-κ tensor is captured — affects `inner`, `norm`,
        `exp`, `log`, `parallel_transport`.
        """
        norm_sq = (x * x).sum(dim=-1)
        return 2.0 / (1.0 + self._get_kappa() * norm_sq)

    # ----------------------------------------------------------------
    # Construction
    # ----------------------------------------------------------------

    def origin(self, batch_size: int = 1) -> torch.Tensor:
        """The origin `(0, …, 0) ∈ R^n` — common base point for
        tangent-space embeddings. Same shape as a regular point.
        """
        if batch_size < 0:
            raise ValueError(f"batch_size must be >= 0, got {batch_size}")
        return torch.zeros(batch_size, self.n,
                            device=self.device, dtype=self.dtype)

    def random_point(
        self,
        batch_size: int = 1,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Sample B random points on the manifold via `exp_0` of a
        Gaussian tangent.

        For the **spherical** branch (κ > 0) the tangent norm has a
        hard upper bound `‖v‖ < π/(2√κ)` (the `tan` singularity in
        `exp_0`). Random Gaussian draws can exceed this; we clip the
        per-element tangent norm at `π/(8√κ)` (one-quarter of the
        singular radius — gives `‖exp_0(v)‖ ≤ tan(π/8)/√κ ≈ 0.414/√κ`,
        so `κ‖x‖² ≤ 0.172` — comfortably inside the κ‖x‖² < 1 domain
        with slack for `is_on_manifold` to pass at `atol = 1e-9`).

        For hyperbolic and Euclidean branches there is no upper bound
        on ‖v‖, so the raw Gaussian draw is used as-is.
        """
        if batch_size < 0:
            raise ValueError(f"batch_size must be >= 0, got {batch_size}")
        v = torch.randn(batch_size, self.n, generator=generator,
                         device=self.device, dtype=self.dtype) * 0.5
        if self._branch == "spherical":
            # Cap ‖v‖ at π/(4√κ) (half the singular `exp_0` radius).
            # Use _safe_sqrt for the norm + a `where` to apply the cap
            # only when the draw exceeds the budget.
            v_sq = (v * v).sum(dim=-1, keepdim=True)
            v_norm = _safe_sqrt(v_sq.squeeze(-1)).unsqueeze(-1)
            # Cap ‖v‖ at π/(8√κ). At that bound, √κ·‖v‖ = π/8 and
            # ‖exp_0(v)‖ = tan(π/8)/√κ ≈ 0.414/√κ, so κ‖exp_0(v)‖² ≈
            # 0.172 — comfortably inside the κ‖x‖² < 1 domain.
            # `π/(4√κ)` would land EXACTLY on the boundary
            # (tan(π/4) = 1), so we shrink by another factor of 2.
            # Re-evaluate against the CURRENT √|κ| (for learnable κ
            # that may have changed magnitude during training). The
            # cap is not part of any gradient chain — `random_point` is
            # a non-differentiable sampling op — so we extract a
            # detached scalar.
            if self._kappa_is_tensor:
                sqrt_abs_k_now = float(
                    torch.sqrt(torch.abs(self.kappa)).detach().item()
                )
            else:
                sqrt_abs_k_now = self._sqrt_abs_kappa
            safe_v_norm = (
                0.5 * math.pi / (2.0 * 2.0 * sqrt_abs_k_now)
            )
            v_norm_safe = torch.where(
                v_norm > 0, v_norm, torch.ones_like(v_norm),
            )
            scale_factor = torch.where(
                v_norm > safe_v_norm,
                safe_v_norm / v_norm_safe,
                torch.ones_like(v_norm),
            )
            v = v * scale_factor
        return self.exp_0(v)

    def is_on_manifold(
        self, x: torch.Tensor, atol: float = 1e-9,
    ) -> torch.Tensor:
        """Per-batch test for manifold membership.

        Domain:
          - κ > 0 (spherical): `κ‖x‖² < 1`.
          - κ = 0 (Euclidean): always True.
          - κ < 0 (hyperbolic): `‖x‖² < 1/|κ|`.

        Uses the **current** value of κ — important when κ is a
        learnable Tensor that has drifted from its initial value
        during training (the membership domain shrinks/grows with |κ|).
        The `atol` parameter absorbs float drift at the boundary.
        """
        if self._branch == "euclidean":
            return torch.ones(x.shape[:-1], dtype=torch.bool,
                               device=x.device)
        norm_sq = (x * x).sum(dim=-1)
        # Extract a scalar value for the comparison; detach so this
        # method (used in validation checks) doesn't insert spurious
        # autograd connections to κ.
        kappa_value = (
            self.kappa.detach().item()
            if self._kappa_is_tensor else self.kappa
        )
        return kappa_value * norm_sq < 1.0 - atol

    # ----------------------------------------------------------------
    # Möbius addition (gyro-addition)
    # ----------------------------------------------------------------

    def mobius_add(
        self, x: torch.Tensor, y: torch.Tensor,
    ) -> torch.Tensor:
        """Möbius (gyro-) addition `x ⊕_κ y` on the κ-stereographic
        manifold.

        Formula (Bachmann et al. 2020, eq. 4):

            x ⊕_κ y = ((1 - 2κ⟨x,y⟩ - κ‖y‖²)·x + (1 + κ‖x‖²)·y)
                      / (1 - 2κ⟨x,y⟩ + κ²‖x‖²‖y‖²)

        Reduces to ordinary vector addition at κ = 0.

        Args:
          x, y: `(..., n)` points on the manifold.
        Returns:
          `(..., n)` Möbius sum.
        """
        if self._branch == "euclidean":
            return x + y
        xy = (x * y).sum(dim=-1, keepdim=True)        # (..., 1)
        xx = (x * x).sum(dim=-1, keepdim=True)
        yy = (y * y).sum(dim=-1, keepdim=True)
        k = self._get_kappa()
        num = (1.0 - 2.0 * k * xy - k * yy) * x + (1.0 + k * xx) * y
        # The denominator is strictly positive for x, y in the domain
        # (1 - 2k⟨x,y⟩ + k²‖x‖²‖y‖² = (1 + k‖x‖²)(1 + k‖y‖²)/(...)
        # ≥ small positive); no zero-denominator guard needed for
        # well-conditioned inputs.
        denom = 1.0 - 2.0 * k * xy + k * k * xx * yy
        return num / denom

    # ----------------------------------------------------------------
    # Tangent operations
    # ----------------------------------------------------------------

    def projection(
        self, x: torch.Tensor, w: torch.Tensor,
    ) -> torch.Tensor:
        """Tangent-space projection.

        The κ-stereographic manifold is an open subset of `R^n`, so
        the tangent space at every point is all of `R^n`. Projection
        is the identity (returns `w` unchanged); included for API
        parity with `LorentzManifold`, where projection enforces
        `⟨x, v⟩_M = 0`.

        Args:
          x: base point (unused; included for API parity).
          w: ambient direction.
        Returns:
          `w` (no-op).
        """
        del x
        return w

    def inner(
        self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor,
    ) -> torch.Tensor:
        """Riemannian inner product `λ_κ(x)² · ⟨u, v⟩_Eucl`.

        The metric is conformal to Euclidean with conformal factor
        `λ_κ(x) = 2/(1 + κ‖x‖²)`. Point-dependent.

        References:
          Bachmann et al. (2020), §3.
        """
        lam = self._conformal_factor(x)
        return (lam * lam) * (u * v).sum(dim=-1)

    def norm(
        self, x: torch.Tensor, v: torch.Tensor,
    ) -> torch.Tensor:
        """Riemannian norm `λ_κ(x) · ‖v‖_Eucl`.

        Always non-negative; the Euclidean norm is squared and
        square-rooted via `_safe_sqrt` for autograd-finite behavior
        at v = 0.
        """
        v_sq = (v * v).sum(dim=-1)
        lam = self._conformal_factor(x)
        return lam * _safe_sqrt(v_sq)

    # ----------------------------------------------------------------
    # Exponential / logarithmic maps, geodesic distance
    # ----------------------------------------------------------------

    def exp_0(self, v: torch.Tensor) -> torch.Tensor:
        """Exponential map at the origin: `exp_0(v) = tan_κ(√|κ|·‖v‖)
        · v / (√|κ|·‖v‖)`.

        At κ = 0: `exp_0(v) = v` (Euclidean identity).
        At κ < 0: maps Euclidean coords to the Poincaré ball.
        At κ > 0: maps to the open spherical-cap projection.

        Args:
          v: `(..., n)` Euclidean tangent vector at the origin.
        Returns:
          `(..., n)` point on the manifold.
        """
        if v.shape[-1] != self.n:
            raise ValueError(
                f"v last dim must be n={self.n}, got {v.shape[-1]}"
            )
        if self._branch == "euclidean":
            return v
        v_sq = (v * v).sum(dim=-1)
        v_norm = _safe_sqrt(v_sq)
        # tan_κ(√|κ|·‖v‖) / (√|κ|·‖v‖) · v.
        # Factor (1/√|κ|·‖v‖) absorbed via _tan_kappa_c which returns
        # tan_κ(scaled)/scaled where scaled = √|κ|·v_norm.
        scaling = self._tan_kappa_c(v_norm)
        return scaling.unsqueeze(-1) * v

    def log_0(self, y: torch.Tensor) -> torch.Tensor:
        """Logarithmic map at the origin: `log_0(y) =
        (1/√|κ|) · tan_κ⁻¹(√|κ|·‖y‖) · y / ‖y‖`.

        At κ = 0: `log_0(y) = y` (Euclidean identity).

        Args:
          y: `(..., n)` point on the manifold.
        Returns:
          `(..., n)` Euclidean tangent at the origin.
        """
        if y.shape[-1] != self.n:
            raise ValueError(
                f"y last dim must be n={self.n}, got {y.shape[-1]}"
            )
        if self._branch == "euclidean":
            return y
        y_sq = (y * y).sum(dim=-1)
        y_norm = _safe_sqrt(y_sq)
        # arctan_κ(√|κ|·‖y‖) / (√|κ|·‖y‖) · y. Same prefactor structure
        # as exp_0 — _atan_kappa_c returns the (atan_κ(α)/α) factor.
        scaling = self._atan_kappa_c(y_norm)
        return scaling.unsqueeze(-1) * y

    def distance(
        self, x: torch.Tensor, y: torch.Tensor,
    ) -> torch.Tensor:
        """Geodesic distance `d_κ(x, y) = (2/√|κ|) · tan_κ⁻¹(√|κ|
        · ‖(-x) ⊕_κ y‖)`.

        At κ = 0: `d(x, y) = 2 · ‖y - x‖_Eucl` (Bachmann normalization;
        the conformal factor at the origin is 2).

        Args:
          x, y: `(..., n)` points on the manifold.
        Returns:
          `(...,)` non-negative distances.
        """
        diff = self.mobius_add(-x, y)              # (..., n)
        diff_sq = (diff * diff).sum(dim=-1)
        diff_norm = _safe_sqrt(diff_sq)
        # Unified κ-trig form: distance = 2·d·(arctan_κ(√|κ|·d)/(√|κ|·d))
        # = 2·d · _atan_kappa_c(d). Works for any κ ∈ R:
        #   - κ > 0: 2·d · arctan(√κ·d)/(√κ·d)  = (2/√κ)·arctan(√κ·d)
        #   - κ < 0: 2·d · arctanh(√|κ|·d)/(√|κ|·d) = (2/√|κ|)·arctanh(√|κ|·d)
        #   - κ = 0: 2·d · 1 = 2d
        # The `_atan_kappa_c` helper does the sign-dispatch (dynamic
        # for Tensor κ, static fast-path for float κ), so κ can flip
        # signs during training without breakdown.
        return 2.0 * diff_norm * self._atan_kappa_c(diff_norm)

    def exp(
        self, x: torch.Tensor, v: torch.Tensor,
    ) -> torch.Tensor:
        """Exponential map at a general base point:

            exp_x(v) = x ⊕_κ (tan_κ(λ_κ(x) · √|κ| · ‖v‖/2)
                              · v / (√|κ|·‖v‖))

        References:
          Bachmann et al. (2020), eq. 5.
        """
        if self._branch == "euclidean":
            return x + v
        v_sq = (v * v).sum(dim=-1)
        v_norm = _safe_sqrt(v_sq)
        lam = self._conformal_factor(x)
        # tan_κ(arg) / (√|κ| · ‖v‖) · v, with arg = λ · √|κ| · ‖v‖ / 2.
        # Rewrite as: (lam / 2) · (tan_κ(arg)/arg) · v   (factor √|κ|·‖v‖
        # cancels with the implicit factor inside arg).
        arg = 0.5 * lam * v_norm
        scaling = 0.5 * lam * self._tan_kappa_c(arg)
        rhs = scaling.unsqueeze(-1) * v
        return self.mobius_add(x, rhs)

    def log(
        self, x: torch.Tensor, y: torch.Tensor,
    ) -> torch.Tensor:
        """Logarithmic map at a general base point:

            log_x(y) = (2/(λ_κ(x)·√|κ|)) · tan_κ⁻¹(√|κ| · ‖(-x) ⊕_κ y‖)
                       · ((-x) ⊕_κ y) / ‖(-x) ⊕_κ y‖

        References:
          Bachmann et al. (2020), eq. 5.
        """
        diff = self.mobius_add(-x, y)
        diff_sq = (diff * diff).sum(dim=-1)
        diff_norm = _safe_sqrt(diff_sq)
        lam = self._conformal_factor(x)
        if self._branch == "euclidean":
            # (2/λ) · ‖diff‖ · diff/‖diff‖ = 2/λ · diff. At origin
            # λ = 2 so this is just `diff = y - x`. ✓
            return (2.0 / lam).unsqueeze(-1) * diff
        # atan_κ(√|κ|·‖diff‖) / (√|κ|·‖diff‖) · diff factor:
        scaling = (2.0 / lam) * self._atan_kappa_c(diff_norm)
        return scaling.unsqueeze(-1) * diff

    def retraction(
        self, x: torch.Tensor, v: torch.Tensor,
    ) -> torch.Tensor:
        """Retraction = exponential map (second-order, the geodesic
        itself satisfies the retraction axioms on constant-curvature
        spaces).
        """
        return self.exp(x, v)

    def parallel_transport(
        self, x: torch.Tensor, y: torch.Tensor, v: torch.Tensor,
    ) -> torch.Tensor:
        """Parallel transport on the κ-stereographic manifold:

            P_{x→y}(v) = (λ_κ(x) / λ_κ(y)) · gyr[y, -x] · v

        where `gyr[a, b]·v = -(a ⊕_κ b) ⊕_κ (a ⊕_κ (b ⊕_κ v))` is the
        Möbius gyrator. For κ = 0 this collapses to the identity
        (Euclidean parallel transport is trivial).

        References:
          Bachmann et al. (2020), eq. 7; Ungar (2008), Theorem 8.4.
        """
        if self._branch == "euclidean":
            return v
        # Gyrator: gyr[y, -x] · v = -(y ⊕ (-x)) ⊕ (y ⊕ ((-x) ⊕ v))
        ngx = self.mobius_add(y, -x)
        inner_add = self.mobius_add(-x, v)
        outer_add = self.mobius_add(y, inner_add)
        gyrated = self.mobius_add(-ngx, outer_add)
        # Conformal factor ratio
        lam_x = self._conformal_factor(x)
        lam_y = self._conformal_factor(y)
        return (lam_x / lam_y).unsqueeze(-1) * gyrated
