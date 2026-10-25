# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Synoros

"""Lorentz (hyperboloid) model of hyperbolic space, GPU-native, batched-first.

The Lorentz model of `n`-dimensional hyperbolic space of constant
sectional curvature `k < 0` is the upper sheet of the two-sheeted
hyperboloid in (n+1)-dimensional Minkowski space:

    H^n_k = { x ∈ R^{n+1} : ⟨x, x⟩_M = 1/k,  x_0 > 0 },
    ⟨x, y⟩_M = −x_0·y_0 + Σ_{i ≥ 1} x_i·y_i

With `k = −1` the constraint collapses to ⟨x, x⟩_M = −1 — the
canonical "unit hyperboloid." The Riemannian metric on H^n_k is the
restriction of ⟨·, ·⟩_M to tangent vectors

    T_x H^n_k = { v ∈ R^{n+1} : ⟨x, v⟩_M = 0 }.

Restricted to T_x H^n_k, ⟨·, ·⟩_M is positive-definite, making the
hyperboloid a Riemannian manifold of constant negative curvature.

All seven core operations admit closed forms:

    proj_T(x, w) = w − k · ⟨x, w⟩_M · x       (orthogonal in ⟨·,·⟩_M)
    ⟨u, v⟩_x    = ⟨u, v⟩_M                    (induced metric)
    d_k(x, y)   = (1/√|k|) · arccosh(k · ⟨x, y⟩_M)
    exp_x(v)    = cosh(α) · x + sinh(α) · v / α,   α = √|k| · ‖v‖_x
    log_x(y)    = β · u / ‖u‖_x,
                  u = y − k·⟨x,y⟩_M · x,   β = d_k(x, y)
    PT_{x→y}(v) = v − ⟨log_x(y), v⟩_x / d_k(x,y)² · (log_x(y) + log_y(x))
    retraction  = exp_x   (geodesic; second-order)

Points are stored as `(B, n+1)` ambient tensors; the leading batch
dimension is required for every operation. Tangent vectors are
stored in the same ambient form, with the understanding that they
satisfy ⟨x, v⟩_M = 0 at the corresponding base point.

The manifold dimension reported by `.dim` is the **intrinsic** `n`,
not the ambient `n+1`.

References:
  Nickel, M., Kiela, D. (2018). Learning continuous hierarchies in the
    Lorentz model of hyperbolic geometry. ICML.
  Chen, W., et al. (2022). Fully hyperbolic neural networks. ACL.
  Lee, J. M. (2018). Introduction to Riemannian Manifolds, 2nd ed.,
    Springer, §5–§6 (model spaces, hyperbolic geometry).
  Cannon, J. W., Floyd, W. J., Kenyon, R., Parry, W. R. (1997).
    Hyperbolic geometry. In Flavors of Geometry, MSRI Publications 31.
  Pennec, X. (2006). Intrinsic statistics on Riemannian manifolds.
    Journal of Mathematical Imaging and Vision 25(1):127–154.
    (parallel transport in terms of log/distance, eq. 4).
"""

from __future__ import annotations

import math

import torch

from holonomy_lib.provenance import register_provenance_class, with_provenance


def _safe_sqrt(x: torch.Tensor) -> torch.Tensor:
    """sqrt(x) with autograd-safe handling at x = 0.

    The classic PyTorch trap: `sqrt(clamp(x, min=0))` produces NaN in
    backward at the boundary (`clamp`'s gradient is 0 there, `sqrt`'s
    is ∞, product is `0·∞ = NaN`). This helper computes sqrt only on a
    where-substituted input and masks the output back, so the masked
    branch never touches the singular operation:

      - forward: `sqrt(x)` for `x > 0`, `0` for `x ≤ 0`
      - backward: `1/(2·sqrt(x))` for `x > 0`, `0` for `x ≤ 0`
        (the 0-subgradient choice at the boundary; the analytic
        derivative is undefined there).

    Used pervasively in `norm`, `distance`, `log_0` etc. wherever the
    inner quantity vanishes at a natural limit (x = y, v = 0, etc.).
    """
    is_positive = x > 0
    x_safe = torch.where(is_positive, x, torch.ones_like(x))
    return torch.where(is_positive, torch.sqrt(x_safe), torch.zeros_like(x))


def _safe_sinhc(alpha: torch.Tensor) -> torch.Tensor:
    """sinh(α)/α with autograd-safe handling at α = 0.

    Without care, `sinh(α)/α_safe` (where α_safe = where(α > 0, α, 1))
    produces NaN in backward because the FORMULA branch still
    references the raw α (which may be 0), and the gradient flow
    couples through both `where`s. Using `sinh(α_safe)/α_safe`
    confines unsafe values to the masked-out branch:

      - forward: `sinh(α)/α` for `α > 0`, `1` for `α = 0`
      - backward: `(α·cosh α − sinh α)/α²` for `α > 0`, `0` for α = 0
    """
    is_positive = alpha > 0
    alpha_safe = torch.where(is_positive, alpha, torch.ones_like(alpha))
    return torch.where(
        is_positive,
        torch.sinh(alpha_safe) / alpha_safe,
        torch.ones_like(alpha),
    )


def _safe_arcsinhc(arg: torch.Tensor) -> torch.Tensor:
    """arcsinh(x)/x with autograd-safe handling at x = 0.

    Analytic limit at 0 is 1. Same idiom as `_safe_sinhc`: the formula
    branch evaluates at a where-substituted input that's never 0.
    """
    is_positive = arg > 0
    arg_safe = torch.where(is_positive, arg, torch.ones_like(arg))
    return torch.where(
        is_positive,
        torch.asinh(arg_safe) / arg_safe,
        torch.ones_like(arg),
    )


@register_provenance_class("LorentzManifold")
class LorentzManifold:
    """Lorentz (hyperboloid) model of hyperbolic space H^n_k.

    Args:
      n: intrinsic manifold dimension. Points live in ambient R^{n+1}.
      k: sectional curvature; must be strictly negative. Default `−1.0`
        is the unit hyperboloid (the canonical literature choice).
      device, dtype: tensor placement and precision.

    Example:
      >>> mfd = LorentzManifold(n=3)
      >>> x = mfd.random_point(batch_size=4)
      >>> x.shape, mfd.is_on_manifold(x).all().item()
      (torch.Size([4, 4]), True)
    """

    def __init__(self, n: int, k: float = -1.0,
                 device: str | torch.device = "cpu",
                 dtype: torch.dtype = torch.float64):
        if n <= 0:
            raise ValueError(f"n must be > 0, got n={n}")
        if k >= 0:
            raise ValueError(
                f"k must be strictly negative (hyperbolic), got k={k}"
            )
        self.n = n
        self.k = float(k)
        self.device = torch.device(device)
        self.dtype = dtype
        # Precompute √|k| as a Python float; it's used in every closed-form
        # call but does not depend on input device/dtype.
        self._abs_k = abs(self.k)
        self._sqrt_abs_k = math.sqrt(self._abs_k)
        self._inv_sqrt_abs_k = 1.0 / self._sqrt_abs_k

    @property
    def dim(self) -> int:
        """Intrinsic manifold dimension `n`. Ambient dimension is `n+1`.

        References:
          Lee (2018), §3.
        """
        return self.n

    @property
    def ambient_dim(self) -> int:
        """Ambient dimension `n + 1`. The Lorentz model embeds H^n_k in
        Minkowski R^{n+1}. Used by manifold-generic primitives in
        `holonomy_lib.hyperbolic`; `KappaStereographicManifold`
        overrides this to `n` (no extra dimension).
        """
        return self.n + 1

    def _provenance_signature(self) -> dict:
        """Deterministic canonical form used by `@with_provenance` to
        hash the bound `self` of decorated methods. See SPDManifold for
        the rationale (registered classes need this + `_from_signature`
        so `ProvenanceRegistry.replay` can rebuild the manifold instance).
        """
        return {
            "class": "LorentzManifold",
            "n": self.n,
            "k": self.k,
            "device": str(self.device),
            "dtype": str(self.dtype),
        }

    @classmethod
    def _from_signature(cls, sig: dict) -> LorentzManifold:
        dtype_name = sig["dtype"].split(".")[-1]
        return cls(
            n=sig["n"],
            k=sig["k"],
            device=sig["device"],
            dtype=getattr(torch, dtype_name),
        )

    # ----------------------------------------------------------------
    # Minkowski form (helper)
    # ----------------------------------------------------------------

    def minkowski_inner(
        self, x: torch.Tensor, y: torch.Tensor,
    ) -> torch.Tensor:
        """Minkowski inner product ⟨x, y⟩_M = −x_0 y_0 + Σ_{i≥1} x_i y_i.

        This is **not** the Riemannian inner product (which equals this
        only when restricted to tangent vectors at a common base point).
        Public so that downstream callers can express their own
        constructions without re-implementing the sign convention.

        Args:
          x, y: ambient vectors of shape (B, n+1).
        Returns:
          (B,) tensor of Minkowski inner products.
        """
        # x_0 · y_0 with the Minkowski sign flip; remaining components
        # contribute Euclidean dot product.
        return (x[..., 1:] * y[..., 1:]).sum(dim=-1) - x[..., 0] * y[..., 0]

    def is_on_manifold(
        self, x: torch.Tensor, atol: float = 1e-9,
    ) -> torch.Tensor:
        """Per-batch test for membership in H^n_k.

        Checks ⟨x, x⟩_M ≈ 1/k (within `atol`) and x_0 > 0 (upper sheet).

        Args:
          x: (B, n+1) ambient candidate points.
          atol: absolute tolerance for the constraint. Default 1e-9 is
            the library's `numerical_floor_convention`; exp/log re-project
            onto the hyperboloid so the post-call constraint holds to
            machine precision and 1e-9 is comfortably above any drift.
        Returns:
          (B,) boolean tensor.
        """
        constraint = self.minkowski_inner(x, x) - (1.0 / self.k)
        on_sheet = constraint.abs() <= atol
        upper = x[..., 0] > 0
        return on_sheet & upper

    # ----------------------------------------------------------------
    # Construction
    # ----------------------------------------------------------------

    def origin(self, batch_size: int = 1) -> torch.Tensor:
        """The canonical "north pole" `(1/√|k|, 0, …, 0)` of H^n_k.

        Useful as a base point for `exp_0` / `log_0` and for tangent-space
        embeddings of Euclidean data.

        Args:
          batch_size: leading batch dim. Pass 0/1/many.
        Returns:
          (B, n+1) tensor with the north-pole repeated.
        """
        if batch_size < 0:
            raise ValueError(f"batch_size must be >= 0, got {batch_size}")
        out = torch.zeros(batch_size, self.n + 1,
                          device=self.device, dtype=self.dtype)
        # x_0 = 1/√|k| ⇒ ⟨x, x⟩_M = -1/|k| = 1/k for k<0. ✓
        out[..., 0] = self._inv_sqrt_abs_k
        return out

    def random_point(
        self,
        batch_size: int = 1,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Sample B random points on H^n_k.

        Construction: draw a Euclidean tangent at the origin from
        `N(0, I_n)`, embed it as an ambient tangent `(0, v_spatial)`,
        then apply `exp_origin`. The resulting distribution has full
        support on the upper sheet of the hyperboloid.

        This is a **convenience** draw for tests/initialization; it is
        not the wrapped-normal or any other statistically-motivated
        prior on H^n_k.

        Args:
          batch_size: leading batch dim.
          generator: optional torch.Generator for reproducibility.
        Returns:
          (B, n+1) tensor of points on H^n_k.
        """
        if batch_size < 0:
            raise ValueError(f"batch_size must be >= 0, got {batch_size}")
        v_spatial = torch.randn(batch_size, self.n, generator=generator,
                                device=self.device, dtype=self.dtype)
        return self.exp_0(v_spatial)

    # ----------------------------------------------------------------
    # Tangent operations
    # ----------------------------------------------------------------

    @with_provenance(
        "holonomy_lib.manifolds.LorentzManifold.projection", op_version="0.1",
    )
    def projection(
        self, x: torch.Tensor, w: torch.Tensor,
    ) -> torch.Tensor:
        """Project ambient direction `w` onto T_x H^n_k.

        T_x H^n_k = { v : ⟨x, v⟩_M = 0 }. The orthogonal projection in
        the Minkowski form is

            proj_T(x, w) = w − ⟨x, w⟩_M / ⟨x, x⟩_M · x
                        = w − k · ⟨x, w⟩_M · x         (since ⟨x,x⟩_M = 1/k)

        Idempotent: proj(proj(w)) = proj(w) (a vector already in T_x
        satisfies ⟨x, v⟩_M = 0, so the subtracted term is zero).

        References:
          Nickel & Kiela (2018), eq. 6.

        Args:
          x: base point (B, n+1) on H^n_k.
          w: ambient direction (B, n+1).
        Returns:
          Tangent (B, n+1).
        """
        coeff = self.k * self.minkowski_inner(x, w)  # (B,)
        return w - coeff.unsqueeze(-1) * x

    @with_provenance(
        "holonomy_lib.manifolds.LorentzManifold.inner", op_version="0.1",
    )
    def inner(
        self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor,
    ) -> torch.Tensor:
        """Riemannian inner product ⟨u, v⟩_x = ⟨u, v⟩_M on T_x H^n_k.

        The induced metric: positive-definite when both inputs are
        tangent vectors at `x` (i.e. ⟨x, u⟩_M = ⟨x, v⟩_M = 0). The base
        point `x` is unused arithmetically (the induced metric is
        point-independent in ambient form) but is included for API
        symmetry with point-dependent metrics like SPD's affine-invariant.

        References:
          Lee (2018), §5.

        Args:
          x: base point (B, n+1) — unused; included for API parity.
          u, v: tangent vectors (B, n+1).
        Returns:
          (B,) Riemannian inner products.
        """
        del x  # induced metric is point-independent in ambient form
        return self.minkowski_inner(u, v)

    @with_provenance(
        "holonomy_lib.manifolds.LorentzManifold.norm", op_version="0.1",
    )
    def norm(
        self, x: torch.Tensor, v: torch.Tensor,
    ) -> torch.Tensor:
        """Riemannian norm √⟨v, v⟩_x.

        For a true tangent at `x`, ⟨v, v⟩_M ≥ 0. Float drift can push
        this slightly negative (~ machine eps × ‖v‖²); we use
        `_safe_sqrt` which both clamps non-positive entries to 0 in
        the forward pass AND produces a finite (zero) gradient at the
        boundary — the naive `sqrt(clamp(x, min=0))` pattern would
        propagate `0·∞ = NaN` through backward at v = 0.

        Returns:
          (B,) non-negative norms.
        """
        return _safe_sqrt(self.inner(x, v, v))

    # ----------------------------------------------------------------
    # Exponential / logarithmic maps, geodesic distance
    # ----------------------------------------------------------------

    @with_provenance(
        "holonomy_lib.manifolds.LorentzManifold.exp", op_version="0.1",
    )
    def exp(
        self, x: torch.Tensor, v: torch.Tensor,
    ) -> torch.Tensor:
        """Exponential map exp_x(v) = cosh(α)·x + sinh(α)·v/α,
        α = √|k| · ‖v‖_x.

        Closed-form geodesic on a constant-curvature space. At v = 0,
        the analytic limit is `x`; the implementation enforces this via
        a `torch.where` guard so that gradients (if taken) and forward
        values are both finite and correct at the origin of T_x.

        Result is re-normalized onto the hyperboloid to remove the small
        Minkowski-form drift introduced by float arithmetic (cosh²−sinh²
        identity holds in exact arithmetic; finite precision violates it
        at the eps · |x_0| level). Without this, chained exps would
        accumulate drift and trip `is_on_manifold`.

        References:
          Nickel & Kiela (2018), eq. 8; Lee (2018), §5.

        Args:
          x: base point (B, n+1) on H^n_k.
          v: tangent (B, n+1) at x (⟨x, v⟩_M = 0).
        Returns:
          (B, n+1) point on H^n_k.
        """
        norm_v = self.norm(x, v)                  # (B,)
        alpha = self._sqrt_abs_k * norm_v         # (B,)

        # sinh(α)/α with the analytic limit 1 at α = 0. Use the
        # autograd-safe helper which keeps unsafe (0) values inside
        # the masked-out branch only — the formula branch evaluates at
        # `alpha_safe`, never the raw 0.
        sinhc = _safe_sinhc(alpha)
        cosh_alpha = torch.cosh(alpha)             # cosh(0) = 1, safe

        out = cosh_alpha.unsqueeze(-1) * x + sinhc.unsqueeze(-1) * v
        return self._reproject_to_hyperboloid(out)

    @with_provenance(
        "holonomy_lib.manifolds.LorentzManifold.log", op_version="0.1",
    )
    def log(
        self, x: torch.Tensor, y: torch.Tensor,
    ) -> torch.Tensor:
        """Logarithmic map log_x(y) = (α / sinh α) · u,
        u = y − k·⟨x,y⟩_M · x, α = arccosh(k·⟨x,y⟩_M).

        Inverse of `exp` on the domain where it's a diffeomorphism (all
        of H^n_k, a Hadamard manifold). At y = x, both numerator (α) and
        the natural denominator (‖u‖) vanish.

        **Numerical note.** The textbook form is `β·u/‖u‖_x` with
        `β = (1/√|k|)·arccosh(z)`. For x ≈ y, `arccosh(z)` near `z = 1`
        and `‖u‖` near 0 both round to small positive values
        independently, and their ratio is unstable (we've seen
        `β/‖u‖ ≈ 10⁸` for `β, ‖u‖ ~ 1e-8 / 1e-16` from float roundoff,
        amplifying a noise-level `u` into a 1e-7 spurious log). The
        equivalent form

            log_x(y) = (α / sinh α) · u

        with `α = arccosh(z)` uses the smooth `sinhc⁻¹` factor (bounded
        by 1, equal to 1 at α=0), which keeps the output well-scaled by
        the small `u` directly. Derivation: `‖u‖_M = sinh(α)/√|k|`
        (from `⟨u,u⟩_M = (z² − 1)/|k|` for x, y on H^n_k), and
        `β = α/√|k|`, so `β/‖u‖ = α/sinh α` — the `√|k|` factors cancel.

        References:
          Nickel & Kiela (2018), eq. 9; Lee (2018), §5.

        Args:
          x, y: points on H^n_k, both (B, n+1).
        Returns:
          (B, n+1) tangent at x.
        """
        ip = self.minkowski_inner(x, y)
        u = y - (self.k * ip).unsqueeze(-1) * x        # (B, n+1)
        # We need α/sinh(α) where α = arccosh(z), z = k·⟨x,y⟩_M.
        # Both arccosh (derivative ∞ at z=1) and sinh(arccosh(z)) =
        # sqrt(z²-1) have boundary singularities that NaN backward at
        # x ≈ y. Reparameterize via `arg = sqrt((z-1)/2)`:
        #
        #   α = 2·arcsinh(arg)       (half-angle identity)
        #   sinh(α) = 2·arg·sqrt(arg²+1)
        #   ⇒  α/sinh(α) = arcsinh(arg) / (arg · sqrt(arg²+1))
        #
        # Both `arcsinh(arg)/arg` and `1/sqrt(arg²+1)` are smooth at
        # arg = 0 (limits 1 and 1 respectively).
        #
        # Compute `arg` via the Minkowski norm of `y − x`, which we
        # showed in `distance` is exact for x, y on H^n_k:
        #     arg² = (z-1)/2 = |k|·‖y-x‖_M² / 4.
        diff = y - x
        diff_sq = ((diff[..., 1:] ** 2).sum(dim=-1) - diff[..., 0] ** 2)
        arg = self._sqrt_abs_k * _safe_sqrt(diff_sq) * 0.5
        # arcsinh(arg)/arg — autograd-safe at arg = 0
        inv_sinhc = _safe_arcsinhc(arg)
        # 1/sqrt(arg² + 1) — always finite, no boundary
        inv_cosh_factor = torch.rsqrt(arg * arg + 1.0)
        scale = inv_sinhc * inv_cosh_factor
        return scale.unsqueeze(-1) * u

    @with_provenance(
        "holonomy_lib.manifolds.LorentzManifold.distance", op_version="0.1",
    )
    def distance(
        self, x: torch.Tensor, y: torch.Tensor,
    ) -> torch.Tensor:
        """Geodesic distance d_k(x, y).

        Computed in the numerically stable `arcsinh` form rather than the
        textbook `(1/√|k|) · arccosh(k·⟨x,y⟩_M)`. At x ≈ y, arccosh hits
        its derivative singularity at 1 and float noise in `k·⟨x,y⟩_M`
        propagates as a spurious O(√eps) ≈ 1e-8 distance. The arcsinh
        identity

            cosh(α) − 1 = 2·sinh²(α/2)

        rearranges arccosh into

            d_k(x, y) = (2/√|k|) · arcsinh(√|k| · ‖y − x‖_M / 2),

        where `‖y − x‖_M² = -2(⟨x,y⟩_M − 1/k)`. arcsinh has no
        derivative singularity, and `‖y − x‖_M` is computed from
        coordinate differences without the `<x,y>_M` near-1 cancellation.
        For x = y bit-identical, `y − x = 0` and `d = 0` exactly.

        References:
          Nickel & Kiela (2018), eq. 7; Cannon et al. (1997), §3;
          Olver (2010) NIST Handbook of Mathematical Functions §4.37
          (the arcsinh-arccosh identity used here).

        Args:
          x, y: (B, n+1) points on H^n_k.
        Returns:
          (B,) non-negative distances.
        """
        diff = y - x
        # ‖y - x‖_M² = -(diff_0)² + Σ diff_i². For x, y on H^n_k this
        # is ≥ 0 in exact arithmetic; `_safe_sqrt` clamps tiny-negative
        # float noise to 0 in forward AND keeps backward finite at the
        # boundary diff_sq = 0 (the d(x,x) = 0 case). The naive
        # `sqrt(clamp(diff_sq, min=0))` pattern propagates `0 · ∞ =
        # NaN` through backward at diff_sq = 0.
        diff_sq = (diff[..., 1:] ** 2).sum(dim=-1) - diff[..., 0] ** 2
        arg = self._sqrt_abs_k * _safe_sqrt(diff_sq) * 0.5
        return (2.0 * self._inv_sqrt_abs_k) * torch.asinh(arg)

    @with_provenance(
        "holonomy_lib.manifolds.LorentzManifold.parallel_transport",
        op_version="0.1",
    )
    def parallel_transport(
        self, x: torch.Tensor, y: torch.Tensor, v: torch.Tensor,
    ) -> torch.Tensor:
        """Parallel transport v ∈ T_x H^n_k → T_y H^n_k along the
        geodesic from x to y.

        Pennec (2006) closed form for Hadamard manifolds:

            PT_{x→y}(v) = v − ⟨log_x(y), v⟩_x / d_k(x,y)²
                              · (log_x(y) + log_y(x))

        At x = y this evaluates to `v` (both log terms vanish).
        Isometric: ⟨PT(u), PT(v)⟩_y = ⟨u, v⟩_x.

        References:
          Pennec (2006), eq. 4 (parallel transport on Hadamard manifolds).
          Nickel & Kiela (2018), eq. 11 (Lorentz-specific form).
          Lee (2018), §4 (Levi-Civita connection, parallel transport).

        Args:
          x, y: (B, n+1) base and target points on H^n_k.
          v: (B, n+1) tangent at x.
        Returns:
          (B, n+1) tangent at y.
        """
        log_xy = self.log(x, y)
        log_yx = self.log(y, x)
        d = self.distance(x, y)
        # ⟨log_x(y), v⟩_x — uses the induced metric at x
        ip = self.inner(x, log_xy, v)
        # 1/d² with the 0/0 guard via where-on-both. At d = 0 both
        # log terms are 0 and the correction vanishes, so PT(v) = v.
        d_sq = d * d
        d_sq_safe = torch.where(d > 0, d_sq, torch.ones_like(d_sq))
        coeff = torch.where(
            d > 0,
            ip / d_sq_safe,
            torch.zeros_like(d_sq),
        )
        return v - coeff.unsqueeze(-1) * (log_xy + log_yx)

    # ----------------------------------------------------------------
    # Retraction (= exp for the hyperboloid; second-order)
    # ----------------------------------------------------------------

    @with_provenance(
        "holonomy_lib.manifolds.LorentzManifold.retraction", op_version="0.1",
    )
    def retraction(
        self, x: torch.Tensor, v: torch.Tensor,
    ) -> torch.Tensor:
        """Retraction = exponential map.

        On constant-curvature spaces, exp is the canonical second-order
        retraction (the geodesic itself satisfies the retraction axioms
        trivially). The optimizer module's `RiemannianSGD` calls this
        method through the standard `projection → retraction` pipeline.

        References:
          Absil, Mahony, Sepulchre (2008), §4.1.
        """
        return self.exp(x, v)

    # ----------------------------------------------------------------
    # Convenience: exp / log at the origin
    # ----------------------------------------------------------------

    @with_provenance(
        "holonomy_lib.manifolds.LorentzManifold.exp_0", op_version="0.1",
    )
    def exp_0(self, v_spatial: torch.Tensor) -> torch.Tensor:
        """Exponential map at the origin from an (n,)-dimensional
        Euclidean coordinate.

        Specialized form of `exp` with the radial component fixed: at the
        origin `o = (1/√|k|, 0, …, 0)`, a tangent has the shape
        `(0, v_spatial)` with `‖(0, v_spatial)‖_o = ‖v_spatial‖_2`. So:

            exp_o((0, v)) = (cosh(α)/√|k|,  sinh(α)·v/α),
            α = √|k| · ‖v‖_2.

        Useful for embedding a Euclidean point cloud onto H^n_k. Strictly
        equivalent to `exp(origin(B), zeros-prefixed v)` but avoids the
        radial-projection cost.

        Args:
          v_spatial: (B, n) Euclidean coordinates.
        Returns:
          (B, n+1) points on H^n_k.
        """
        if v_spatial.shape[-1] != self.n:
            raise ValueError(
                f"v_spatial last dim must be n={self.n}, "
                f"got {v_spatial.shape[-1]}"
            )
        # `_safe_sqrt(Σ v²)` instead of `torch.linalg.vector_norm` so
        # the gradient is finite at v_spatial = 0 (vector_norm's
        # backward there is 0/0 = NaN). Forward is identical when
        # v_spatial ≠ 0.
        norm_v_sq = (v_spatial * v_spatial).sum(dim=-1)
        norm_v = _safe_sqrt(norm_v_sq)
        alpha = self._sqrt_abs_k * norm_v
        cosh_a = torch.cosh(alpha)
        sinhc = _safe_sinhc(alpha)

        # `torch.cat` rather than in-place `out[..., 0] = …` so that
        # autograd doesn't break (in-place assignment into a freshly-
        # allocated `torch.empty` would not be a problem in principle,
        # but `torch.cat` is the more idiomatic differentiable form).
        time = (cosh_a * self._inv_sqrt_abs_k).unsqueeze(-1)
        space = sinhc.unsqueeze(-1) * v_spatial
        out = torch.cat([time, space], dim=-1)
        return self._reproject_to_hyperboloid(out)

    @with_provenance(
        "holonomy_lib.manifolds.LorentzManifold.log_0", op_version="0.1",
    )
    def log_0(self, y: torch.Tensor) -> torch.Tensor:
        """Logarithmic map at the origin to an (n,)-dimensional Euclidean
        coordinate.

        Specialized form of `log` from `o = (1/√|k|, 0, …, 0)`. For y on
        H^n_k, the projection `u = y − k·⟨o,y⟩_M·o` has zero temporal
        component and spatial part `y_{1:}`. Hence

            log_o(y)_spatial = β · y_{1:} / ‖y_{1:}‖_2,
            β = (1/√|k|) · arccosh(√|k| · y_0).

        Returns:
          (B, n) Euclidean coordinates in T_o H^n_k.
        """
        if y.shape[-1] != self.n + 1:
            raise ValueError(
                f"y last dim must be n+1={self.n + 1}, got {y.shape[-1]}"
            )
        # Reuse the arcsinh reparameterization from `distance` / `log`:
        # both arccosh's derivative at z=1 and `vector_norm`'s 0/0
        # gradient at y_spatial=0 would NaN the backward at y = origin.
        # log_0(y) = (α/sinh α) · y_spatial, derived via:
        #   diff_sq = ‖y − origin‖_M² = -(y_0 − 1/√|k|)² + Σ y_i²
        #   arg² = |k|·diff_sq/4 = (z-1)/2   with z = √|k|·y_0
        #   α = 2·arcsinh(arg), sinh(α) = 2·arg·sqrt(arg²+1)
        #   ⇒  α/sinh(α) = arcsinh(arg)/(arg · sqrt(arg² + 1))
        y_spatial = y[..., 1:]
        diff_temporal = y[..., 0] - self._inv_sqrt_abs_k
        diff_sq = (y_spatial * y_spatial).sum(dim=-1) - diff_temporal ** 2
        arg = self._sqrt_abs_k * _safe_sqrt(diff_sq) * 0.5
        inv_sinhc = _safe_arcsinhc(arg)
        inv_cosh_factor = torch.rsqrt(arg * arg + 1.0)
        scale = inv_sinhc * inv_cosh_factor
        return scale.unsqueeze(-1) * y_spatial

    # ----------------------------------------------------------------
    # Internal: re-project a near-manifold ambient point onto H^n_k
    # ----------------------------------------------------------------

    def _reproject_to_hyperboloid(self, x: torch.Tensor) -> torch.Tensor:
        """Rescale x_0 so ⟨x, x⟩_M = 1/k exactly (modulo float).

        For a point that satisfies the constraint in exact arithmetic
        but drifts by ε from float operations, recovering exactness is

            x_0_new = √(‖x_{1:}‖² + |1/k|)

        which preserves the spatial part and the upper-sheet sign. This
        is the standard "renormalize after exp" trick (Nickel-Kiela
        2018 §3) and follows the same drift-correction doctrine as
        `SPDManifold.exp`'s symmetrization step.

        Implementation: `torch.cat` rather than in-place subscript
        assignment on a `clone`, so autograd treats the operation as
        an out-of-place tensor construction. The argument of sqrt is
        always strictly positive (`spatial_sq + |1/k| ≥ |1/k| > 0`),
        so sqrt's backward is finite without a `_safe_sqrt` wrapper.
        """
        spatial = x[..., 1:]
        spatial_sq = (spatial * spatial).sum(dim=-1)
        # For k<0, ⟨x,x⟩_M = -x_0² + ‖x_{1:}‖² = 1/k ⇒ x_0² = ‖x_{1:}‖² - 1/k
        # = ‖x_{1:}‖² + |1/k|. Always ≥ |1/k| > 0; sqrt is safe.
        x0_new = torch.sqrt(spatial_sq + (-1.0 / self.k))
        return torch.cat([x0_new.unsqueeze(-1), spatial], dim=-1)
