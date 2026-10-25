# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Synoros

"""Lorentz (hyperboloid) model of hyperbolic space, GPU-native, batched-first.

The Lorentz model of `n`-dimensional hyperbolic space of constant
sectional curvature `k < 0` is the upper sheet of the two-sheeted
hyperboloid in (n+1)-dimensional Minkowski space:

    H^n_k = { x ∈ R^{n+1} : ⟨x, x⟩_M = 1/k,  x_0 > 0 },
    ⟨x, y⟩_M = −x_0·y_0 + Σ_{i ≥ 1} x_i·y_i

With `k = −1` the constraint collapses to ⟨x, x⟩_M = −1; the
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

Points are stored as `(B, n+1)` ambient tensors, batch first. Tangent
vectors use the same ambient form and satisfy ⟨x, v⟩_M = 0 at their
base point.

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


def _safe_sqrt(x: torch.Tensor) -> torch.Tensor:
    """sqrt(x), returning 0 with zero gradient where x ≤ 0.

    The derivative of sqrt is infinite at 0, and a squared norm can come
    out slightly negative from float drift. This helper takes the sqrt only
    of a where-substituted input that is always positive, then masks the
    result back:

      - forward: `sqrt(x)` for `x > 0`, `0` for `x ≤ 0`
      - backward: `1/(2·sqrt(x))` for `x > 0`, `0` for `x ≤ 0`
        (the 0-subgradient choice at the boundary; the analytic
        derivative is undefined there).

    Used by `LorentzManifold.norm` and by `ProductManifold`
    (`manifolds/product.py`).
    """
    is_positive = x > 0
    x_safe = torch.where(is_positive, x, torch.ones_like(x))
    return torch.where(is_positive, torch.sqrt(x_safe), torch.zeros_like(x))


def _safe_sinhc(alpha: torch.Tensor) -> torch.Tensor:
    """sinh(α)/α, with the limit 1 at α = 0.

    The formula is 0/0 at α = 0. A `where` keeps that NaN out of the forward
    value, but it does not stop a NaN in the unselected branch from reaching
    the backward pass. Evaluating `sinh(α_safe)/α_safe`, with α_safe = 1
    wherever α = 0, keeps both branches finite:

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
        `holonomy_lib.hyperbolic`. `KappaStereographicManifold` needs no
        extra coordinate, so its `ambient_dim` is `n`.
        """
        return self.n + 1


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

        References:
          Nickel and Kiela (2018), Learning Continuous Hierarchies in the
          Lorentz Model of Hyperbolic Geometry, §3.
        """
        # The time coordinate x_0 enters with a minus sign. The spatial
        # coordinates 1: give the ordinary Euclidean dot product.
        return (x[..., 1:] * y[..., 1:]).sum(dim=-1) - x[..., 0] * y[..., 0]

    def is_on_manifold(
        self, x: torch.Tensor, atol: float = 1e-9,
    ) -> torch.Tensor:
        """Per-batch test for membership in H^n_k.

        Checks ⟨x, x⟩_M ≈ 1/k (within `atol`) and x_0 > 0 (upper sheet).

        Args:
          x: (B, n+1) ambient candidate points.
          atol: absolute tolerance for the constraint. Default 1e-9 is
            the library's `numerical_floor_convention`. `exp` re-projects
            onto the hyperboloid, so its output meets the constraint to
            machine precision, and 1e-9 sits well above that drift.
        Returns:
          (B,) boolean tensor.

        References:
          Nickel and Kiela (2018), Learning Continuous Hierarchies in the
          Lorentz Model of Hyperbolic Geometry, §3.
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

        References:
          Nickel and Kiela (2018), Learning Continuous Hierarchies in the
          Lorentz Model of Hyperbolic Geometry, §3.
        """
        if batch_size < 0:
            raise ValueError(f"batch_size must be >= 0, got {batch_size}")
        out = torch.zeros(batch_size, self.n + 1,
                          device=self.device, dtype=self.dtype)
        # x_0 = 1/√|k| gives ⟨x, x⟩_M = −1/|k| = 1/k, so the origin is on the sheet.
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
        then apply `exp_0`. The resulting distribution has full
        support on the upper sheet of the hyperboloid.

        This is a **convenience** draw for tests/initialization; it is
        not the wrapped-normal or any other statistically-motivated
        prior on H^n_k.

        Args:
          batch_size: leading batch dim.
          generator: optional torch.Generator for reproducibility.
        Returns:
          (B, n+1) tensor of points on H^n_k.

        References:
          Nickel and Kiela (2018), Learning Continuous Hierarchies in the
          Lorentz Model of Hyperbolic Geometry, §3.
        """
        if batch_size < 0:
            raise ValueError(f"batch_size must be >= 0, got {batch_size}")
        # A torch.Generator belongs to one device, so the draw runs on the
        # generator's device and is moved to the manifold's device afterward.
        v_spatial = torch.randn(
            batch_size,
            self.n,
            generator=generator,
            device=generator.device if generator is not None else self.device,
            dtype=self.dtype,
        ).to(self.device)
        return self.exp_0(v_spatial)

    # ----------------------------------------------------------------
    # Tangent operations
    # ----------------------------------------------------------------

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

    def egrad_to_rgrad(self, point: torch.Tensor, egrad: torch.Tensor) -> torch.Tensor:
        """Apply the Minkowski metric inverse, then project to the tangent.

        Args:
          point: (B, ambient_dim) base point.
          egrad: (B, ambient_dim) Euclidean gradient.
        Returns:
          (B, ambient_dim) metric gradient in tangent coordinates.

        References:
          Nickel and Kiela (2018), Learning Continuous Hierarchies in the
          Lorentz Model of Hyperbolic Geometry, §3, Riemannian optimization.
          Absil, Mahony, Sepulchre (2008), §3.6.
        """
        # The Minkowski metric is diag(-1, 1, ..., 1), which is its own
        # inverse, so the Riemannian gradient flips the sign of the time
        # component. The projection then removes the normal part.
        metric_gradient = torch.cat((-egrad[..., :1], egrad[..., 1:]), dim=-1)
        return self.projection(point, metric_gradient)

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
          x: base point (B, n+1); unused; included for API parity.
          u, v: tangent vectors (B, n+1).
        Returns:
          (B,) Riemannian inner products.
        """
        del x  # induced metric is point-independent in ambient form
        return self.minkowski_inner(u, v)

    def norm(
        self, x: torch.Tensor, v: torch.Tensor,
    ) -> torch.Tensor:
        """Riemannian norm √⟨v, v⟩_x.

        For a true tangent at `x`, ⟨v, v⟩_M ≥ 0. Float drift can push
        this slightly negative (about machine eps × ‖v‖²), so the root is
        taken with `_safe_sqrt`. That helper returns 0 for non-positive input
        and has a zero gradient there, so v = 0 gives 0 and not a NaN.

        Returns:
          (B,) non-negative norms.

        References:
          Nickel and Kiela (2018), Learning Continuous Hierarchies in the
          Lorentz Model of Hyperbolic Geometry, §3.
        """
        return _safe_sqrt(self.inner(x, v, v))

    # ----------------------------------------------------------------
    # Exponential / logarithmic maps, geodesic distance
    # ----------------------------------------------------------------

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

        # sinh(α)/α is 0/0 at α = 0, where its limit is 1. `_safe_sinhc`
        # handles that point, so the forward value and the gradient stay
        # finite when v = 0.
        sinhc = _safe_sinhc(alpha)
        cosh_alpha = torch.cosh(alpha)             # cosh(0) = 1, safe

        out = cosh_alpha.unsqueeze(-1) * x + sinhc.unsqueeze(-1) * v
        return self._reproject_to_hyperboloid(out)

    def log(
        self, x: torch.Tensor, y: torch.Tensor,
    ) -> torch.Tensor:
        """Logarithmic map log_x(y) = (α / sinh α) · u,
        u = y − k·⟨x,y⟩_M · x, α = arccosh(k·⟨x,y⟩_M).

        Inverse of `exp` on the domain where it's a diffeomorphism (all
        of H^n_k, a Hadamard manifold). At y = x, both numerator (α) and
        the natural denominator (‖u‖) vanish.

        Coordinates and tangent components must remain representable. Very
        large radii can still lose angular/tangent precision, especially in
        float32; distance and log_0 have a wider useful numerical range.

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
        `β = α/√|k|`, so `β/‖u‖ = α/sinh α`; the `√|k|` factors cancel.

        References:
          Nickel & Kiela (2018), eq. 9; Lee (2018), §5.

        Args:
          x, y: points on H^n_k, both (B, n+1).
        Returns:
          (B, n+1) tangent at x.
        """
        alpha = self._sqrt_abs_k * self.distance(x, y)
        # The distance comes from the spatial coordinates, which avoids the
        # cancellation in the ambient Minkowski product at large radius. The
        # temporal coordinate of the tangent is rebuilt from tangency
        # (⟨x, u⟩_M = 0) rather than subtracted, since that subtraction would
        # cancel two large, nearly equal temporal components.
        scale = _safe_sinhc(alpha).reciprocal()
        spatial = scale.unsqueeze(-1) * (
            y[..., 1:] - torch.cosh(alpha).unsqueeze(-1) * x[..., 1:]
        )
        time_at_x = torch.sqrt(x[..., 1:].square().sum(-1) - 1.0 / self.k)
        temporal = (x[..., 1:] * spatial).sum(-1) / time_at_x
        return torch.cat((temporal.unsqueeze(-1), spatial), dim=-1)

    def distance(
        self, x: torch.Tensor, y: torch.Tensor,
    ) -> torch.Tensor:
        """Geodesic distance, computed from the spatial coordinates alone.

        A point's temporal coordinate is fixed by its spatial part, so only the
        spatial parts enter. Resolve the smaller
        spatial vector into components parallel/perpendicular to the larger
        one. If m = sqrt(1 + ||perp||²) and δ is their axial rapidity
        difference, cosh(√|k| d) - 1 = (m - 1) + 2m sinh²(δ/2).
        Rationalizing m - 1 avoids subtracting nearly equal squares.

        Unlike an ambient Minkowski subtraction, this retains large radial
        distances in float32. Spatial squared norms and hyperbolic functions
        must remain representable; angles below coordinate precision cannot
        be recovered. Off-manifold temporal coordinates are ignored.

        Args:
          x, y: (B, n+1) points on the positive hyperboloid sheet.
        Returns:
          (B,) distances, with PyTorch's zero subgradient at coincident points.

        References:
          Nickel and Kiela (2018), eq. 7, Lorentz distance; Olver (2010),
          NIST Handbook §4.37, hyperbolic addition and half-angle identities.
        """
        a = self._sqrt_abs_k * x[..., 1:]
        b = self._sqrt_abs_k * y[..., 1:]
        a_norm = torch.linalg.vector_norm(a, dim=-1, keepdim=True)
        b_norm = torch.linalg.vector_norm(b, dim=-1, keepdim=True)
        a_larger = a_norm >= b_norm
        axis = torch.where(a_larger, a, b)
        other = torch.where(a_larger, b, a)
        radius = torch.where(a_larger, a_norm, b_norm)
        unit = axis / torch.where(radius > 0, radius, torch.ones_like(radius))
        # Take the residual against the pivot coordinate instead of against a
        # rounded unit vector. Subtracting the unit vector directly invents a
        # large transverse component for large collinear spatial vectors. The
        # pivot is the largest coordinate of `axis`, so dividing by it is
        # well conditioned.
        pivot = axis.abs().argmax(dim=-1, keepdim=True)
        axis_pivot = axis.gather(-1, pivot)
        safe_pivot = torch.where(axis_pivot != 0, axis_pivot, 1.0)
        ratio = other.gather(-1, pivot) / safe_pivot
        residual = other - ratio * axis
        along_residual = (residual * unit).sum(dim=-1, keepdim=True)
        parallel = ratio * radius + along_residual
        perpendicular = residual - along_residual * unit
        transverse_scale = torch.sqrt(
            1.0 + perpendicular.square().sum(-1, keepdim=True)
        )
        delta = torch.asinh(parallel / transverse_scale) - torch.asinh(radius)
        transverse = perpendicular / torch.sqrt(2.0 * (transverse_scale + 1.0))
        axial = torch.sqrt(transverse_scale) * torch.sinh(0.5 * delta)
        half_sinh = torch.linalg.vector_norm(
            torch.cat((transverse, axial), dim=-1), dim=-1
        )
        distance = 2.0 * self._inv_sqrt_abs_k * torch.asinh(half_sinh)
        # Identical spatial parts give exactly 0. The check is explicit so the
        # zero does not depend on how the decomposition rounds.
        same = (x[..., 1:] == y[..., 1:]).all(dim=-1)
        return torch.where(same, torch.zeros_like(distance), distance)

    def parallel_transport(
        self, x: torch.Tensor, y: torch.Tensor, v: torch.Tensor,
    ) -> torch.Tensor:
        """Parallel transport v ∈ T_x H^n_k → T_y H^n_k along the
        geodesic from x to y.

        Closed form on the constant-curvature hyperboloid:

            PT_{x→y}(v) = v − ⟨log_x(y), v⟩_x / d_k(x,y)²
                              · (log_x(y) + log_y(x))

        At x = y this evaluates to `v` (both log terms vanish).
        Isometric: ⟨PT(u), PT(v)⟩_y = ⟨u, v⟩_x.

        References:
          Nickel & Kiela (2018), §3 (Lorentz geometry).
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
        # ⟨log_x(y), v⟩_x, using the induced metric at x
        ip = self.inner(x, log_xy, v)
        # d² is 0 at d = 0, where both log terms vanish and PT(v) = v. Divide
        # by 1 there so no 0/0 is formed, then zero the coefficient.
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

    def retraction(
        self, x: torch.Tensor, v: torch.Tensor,
    ) -> torch.Tensor:
        """Retraction = exponential map.

        On constant-curvature spaces, exp is a retraction, and a second-order
        one: the geodesic it traces has zero acceleration at the base point.
        `RiemannianSGD` calls this method through the
        `egrad_to_rgrad → retraction` pipeline.

        References:
          Absil, Mahony, Sepulchre (2008), §4.1.
        """
        return self.exp(x, v)

    # ----------------------------------------------------------------
    # Convenience: exp / log at the origin
    # ----------------------------------------------------------------

    def exp_0(self, v_spatial: torch.Tensor) -> torch.Tensor:
        """Exponential map at the origin from an (n,)-dimensional
        Euclidean coordinate.

        Specialized form of `exp` with the radial component fixed: at the
        origin `o = (1/√|k|, 0, …, 0)`, a tangent has the shape
        `(0, v_spatial)` with `‖(0, v_spatial)‖_o = ‖v_spatial‖_2`. So:

            exp_o((0, v)) = (cosh(α)/√|k|,  sinh(α)·v/α),
            α = √|k| · ‖v‖_2.

        Useful for embedding a Euclidean point cloud onto H^n_k. It equals
        `exp(origin(B), (0, v))`, but it takes the (n,) coordinates directly,
        so no ambient tangent is built and no Minkowski product is taken.

        Args:
          v_spatial: (B, n) Euclidean coordinates.
        Returns:
          (B, n+1) points on H^n_k.

        References:
          Nickel and Kiela (2018), Learning Continuous Hierarchies in the
          Lorentz Model of Hyperbolic Geometry, §3.
        """
        if v_spatial.shape[-1] != self.n:
            raise ValueError(
                f"v_spatial last dim must be n={self.n}, "
                f"got {v_spatial.shape[-1]}"
            )
        norm_v = torch.linalg.vector_norm(v_spatial, dim=-1)
        alpha = self._sqrt_abs_k * norm_v
        cosh_a = torch.cosh(alpha)
        sinhc = _safe_sinhc(alpha)

        # Assemble the ambient point with torch.cat, out of place. Writing into
        # a preallocated tensor is an in-place op that autograd has to track.
        time = (cosh_a * self._inv_sqrt_abs_k).unsqueeze(-1)
        space = sinhc.unsqueeze(-1) * v_spatial
        out = torch.cat([time, space], dim=-1)
        return self._reproject_to_hyperboloid(out)

    def log_0(self, y: torch.Tensor) -> torch.Tensor:
        """Logarithmic map at the origin to an (n,)-dimensional Euclidean
        coordinate.

        Specialized form of `log` from `o = (1/√|k|, 0, …, 0)`. For y on
        H^n_k, the projection `u = y − k·⟨o,y⟩_M·o` has zero temporal
        component and spatial part `y_{1:}`. Hence

            log_o(y)_spatial = β · y_{1:} / ‖y_{1:}‖_2,
            β = asinh(√|k| · ||y_spatial||) / √|k|.

        Working from the spatial part alone avoids the temporal/spatial
        cancellation at large radius. At y_spatial = 0 the factor asinh(a)/a
        takes its limit 1 (see `_safe_arcsinhc`), so log_0 is defined there.

        Returns:
          (B, n) Euclidean coordinates in T_o H^n_k.

        References:
          Nickel and Kiela (2018), Learning Continuous Hierarchies in the
          Lorentz Model of Hyperbolic Geometry, §3.
        """
        if y.shape[-1] != self.n + 1:
            raise ValueError(
                f"y last dim must be n+1={self.n + 1}, got {y.shape[-1]}"
            )
        y_spatial = y[..., 1:]
        arg = self._sqrt_abs_k * torch.linalg.vector_norm(y_spatial, dim=-1)
        return _safe_arcsinhc(arg).unsqueeze(-1) * y_spatial

    # ----------------------------------------------------------------
    # Internal: re-project a near-manifold ambient point onto H^n_k
    # ----------------------------------------------------------------

    def _reproject_to_hyperboloid(self, x: torch.Tensor) -> torch.Tensor:
        """Recompute x_0 so that ⟨x, x⟩_M = 1/k holds again, up to float error.

        For a point that satisfies the constraint in exact arithmetic
        but drifts by ε from float operations, recovering exactness is

            x_0_new = √(‖x_{1:}‖² + |1/k|)

        which preserves the spatial part and the upper-sheet sign. This
        is the standard "renormalize after exp" trick (Nickel-Kiela
        2018 §3) and it does the same job as the symmetrization in
        `SPDManifold.exp`.

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
