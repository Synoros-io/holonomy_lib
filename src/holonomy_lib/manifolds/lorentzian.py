# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Synoros

"""Lorentzian (pseudo-Riemannian, signature (1, n-1)) manifold ,
flat Minkowski spacetime `R^{1, n-1}`.

Distinct from `LorentzManifold` (which is the *unit hyperboloid*, a
Riemannian submanifold sitting inside this space). `LorentzianManifold`
**is** the ambient Minkowski space:

  - The manifold itself is `R^n` (no constraint surface).
  - The metric is `g(u, v) = ⟨u, v⟩_M = -u_0·v_0 + Σ_{i ≥ 1} u_i·v_i` ,
    **indefinite signature (1, n-1)**: one timelike direction, `n - 1`
    spacelike directions.
  - Tangent space at any point is the full `R^n`; tangent vectors carry
    a signed norm-squared `⟨v, v⟩_M ∈ R` (negative = timelike,
    zero = null/lightlike, positive = spacelike).
  - Geodesics are straight lines (the manifold is flat); `exp_x(v) = x + v`
    and `log_x(y) = y - x`, regardless of causal type.

The interesting structure isn't the geodesics (trivial) but the **causal
classification**: which point pairs are timelike-separated, lightlike-
separated, or spacelike-separated. This is the substrate-as-spacetime
primitive; Lorentzian-geometry-aware operations that respect the
sign-indefinite metric.

API summary (compared to a Riemannian manifold):

  - `minkowski_inner(u, v)`: SIGNED inner product; can be any real
    number (vs `LorentzManifold.minkowski_inner` which is the same
    function used purely as a helper, and `LorentzManifold.inner`
    which is the *induced* positive-definite tangent-space metric).
  - `norm_sq(v)`: signed `⟨v, v⟩_M`. **There is no `norm`**; the
    square root would be complex for timelike vectors.
  - `interval_sq(x, y) := ⟨y - x, y - x⟩_M`: signed squared spacetime
    interval. Negative for timelike-separated pairs, zero for null,
    positive for spacelike.
  - `causal_type(x, y) -> int`: integer label per pair:
      0 = spacelike, 1 = future-timelike, -1 = past-timelike,
      2 = future-null, -2 = past-null.
  - `proper_time(x, y)`: `sqrt(-interval_sq)` for timelike pairs only;
    raises (or returns NaN element-wise) for non-timelike.
  - `proper_distance(x, y)`: `sqrt(interval_sq)` for spacelike pairs
    only.
  - `exp`, `log`, `retraction`: the trivial flat-space forms.

References:
  Misner, C. W., Thorne, K. S., Wheeler, J. A. (1973). *Gravitation*.
    W. H. Freeman. Chapters 1–5 (Lorentzian-geometry foundations).
  Hawking, S. W., Ellis, G. F. R. (1973). *The Large Scale Structure of
    Space-Time*. Cambridge University Press, Ch. 4 (causal structure).
  O'Neill, B. (1983). *Semi-Riemannian Geometry With Applications to
    Relativity*. Academic Press, Ch. 5 (formal pseudo-Riemannian
    geometry).
"""

from __future__ import annotations

import torch


class LorentzianManifold:
    """Flat pseudo-Riemannian manifold `R^{1, n-1}`.

    Args:
      n: ambient dimension. The signature is `(1, n-1)`; one timelike
        component (index 0) and `n - 1` spacelike components.
      device, dtype: tensor placement / precision.

    Example:
      >>> mfd = LorentzianManifold(n=4)               # 4D spacetime
      >>> x = mfd.random_point(batch_size=3)
      >>> x.shape
      torch.Size([3, 4])
      >>> # Future-timelike pair (light cone of x_0)
      >>> y = x + torch.tensor([2.0, 0.5, 0.0, 0.0])
      >>> mfd.causal_type(x, y).tolist()              # 1 = future-timelike
      [1, 1, 1]
    """

    # Causal-type integer codes (returned by `causal_type`). The
    # mathematical sign agrees with the temporal-component sign of
    # `y - x` for non-spacelike pairs.
    SPACELIKE: int = 0
    FUTURE_TIMELIKE: int = 1
    PAST_TIMELIKE: int = -1
    FUTURE_NULL: int = 2
    PAST_NULL: int = -2

    def __init__(self, n: int,
                 device: str | torch.device = "cpu",
                 dtype: torch.dtype = torch.float64):
        if n < 2:
            raise ValueError(
                f"n must be >= 2 for a meaningful Lorentzian signature, "
                f"got n={n}"
            )
        self.n = n
        self.device = torch.device(device)
        self.dtype = dtype

    @property
    def dim(self) -> int:
        """Manifold dimension `n`. Same as ambient; flat space, no
        constraint surface.
        """
        return self.n

    @property
    def ambient_dim(self) -> int:
        """Ambient dimension `n`. Same as `.dim` for flat space."""
        return self.n


    # ----------------------------------------------------------------
    # Minkowski form
    # ----------------------------------------------------------------

    def minkowski_inner(
        self, u: torch.Tensor, v: torch.Tensor,
    ) -> torch.Tensor:
        """Signed Minkowski inner product
        `⟨u, v⟩_M = -u_0·v_0 + Σ_{i≥1} u_i·v_i`.

        Can be any real number; negative on timelike, zero on null,
        positive on spacelike directions.

        Args:
          u, v: `(B, n)` vectors.
        Returns:
          `(B,)` signed scalars.

        References:
          O'Neill (1983), Semi-Riemannian Geometry, Ch. 3, flat metric
          and its Levi-Civita connection.
        """
        return (u[..., 1:] * v[..., 1:]).sum(dim=-1) - u[..., 0] * v[..., 0]

    def inner(
        self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor,
    ) -> torch.Tensor:
        """Pseudo-Riemannian inner product `⟨u, v⟩_x = ⟨u, v⟩_M`.

        **Signed**: the Minkowski form restricted to ambient vectors,
        with the (1, n-1) signature. NOT positive-definite; `inner(x,
        v, v)` can be any sign depending on `v`'s causal type. This
        differs from the conventional Riemannian `inner` (which is
        always positive); for `LorentzianManifold` the indefinite
        metric IS the geometry.

        Provided primarily for API uniformity with the other manifold
        classes; so `manifold_aware_inner`, `ProductManifold.inner`
        and similar manifold-generic primitives don't crash when
        passed a `LorentzianManifold`. Callers that compute Riemannian
        norms via `sqrt(inner(...))` should be aware that the result
        is imaginary on timelike inputs; prefer `proper_time` /
        `proper_distance` for type-appropriate magnitudes.

        Args:
          x: base point (unused; flat-space metric is point-independent).
          u, v: ambient `(B, n)` vectors.
        Returns:
          `(B,)` signed inner products.

        References:
          O'Neill (1983), Semi-Riemannian Geometry, Ch. 3, flat metric
          and its Levi-Civita connection.
        """
        del x
        return self.minkowski_inner(u, v)

    def norm_sq(self, v: torch.Tensor) -> torch.Tensor:
        """Signed `⟨v, v⟩_M`. Negative = timelike, zero = null,
        positive = spacelike.

        **No `norm` method exists**; `√⟨v, v⟩_M` is imaginary for
        timelike `v`. Use `proper_time` or `proper_distance` for the
        type-appropriate magnitude.

        References:
          MTW (1973) §2.3 (Minkowski metric); O'Neill (1983) §3.1.
        """
        return self.minkowski_inner(v, v)

    def interval_sq(
        self, x: torch.Tensor, y: torch.Tensor,
    ) -> torch.Tensor:
        """Signed squared spacetime interval `⟨y - x, y - x⟩_M`.

        - Negative: `x` and `y` are timelike-separated (one is in the
          past or future light cone of the other).
        - Zero: `x` and `y` are null-separated (on each other's light
          cone; photon path).
        - Positive: `x` and `y` are spacelike-separated (no signal
          can connect them).

        References:
          MTW (1973) §1.5; Hawking-Ellis (1973) §4.1.
        """
        return self.norm_sq(y - x)

    # ----------------------------------------------------------------
    # Causal classification
    # ----------------------------------------------------------------

    def causal_type(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        null_atol: float = 1e-9,
    ) -> torch.Tensor:
        """Causal classification of point pairs `(x_i, y_i)`.

        Returns an integer label per batch element:
          - `SPACELIKE` (= 0): `(y - x)` is spacelike
          - `FUTURE_TIMELIKE` (= 1): timelike with `(y - x)_0 > 0`
          - `PAST_TIMELIKE` (= -1): timelike with `(y - x)_0 < 0`
          - `FUTURE_NULL` (= 2): null/lightlike with `(y - x)_0 > 0`
          - `PAST_NULL` (= -2): null/lightlike with `(y - x)_0 <= 0`

        Coincident events have no time orientation; this API assigns them
        PAST_NULL, also used for tolerance-null pairs with zero time difference.

        The `null_atol` tolerance pulls "exactly null" inwards from
        the |interval_sq| < null_atol band. Reuses the library's
        `numerical_floor_convention`.

        References:
          Hawking-Ellis (1973) §4.1 (causal structure).
          O'Neill (1983) §5.1 (timelike / null / spacelike vectors).

        Args:
          x, y: `(B, n)` points.
          null_atol: |interval_sq| floor below which the pair is
            classified as null. Default 1e-9.
        Returns:
          `(B,)` int64 tensor of causal-type codes.
        """
        diff = y - x
        i_sq = self.norm_sq(diff)
        dt = diff[..., 0]
        # Classify by sign of interval_sq and direction in time
        is_null = i_sq.abs() <= null_atol
        is_timelike = (i_sq < -null_atol)  # negative beyond floor
        is_future = dt > 0
        # Start with SPACELIKE everywhere
        out = torch.zeros(diff.shape[:-1], dtype=torch.int64,
                           device=diff.device)
        out = torch.where(
            is_timelike & is_future,
            torch.full_like(out, self.FUTURE_TIMELIKE), out,
        )
        out = torch.where(
            is_timelike & ~is_future,
            torch.full_like(out, self.PAST_TIMELIKE), out,
        )
        out = torch.where(
            is_null & is_future,
            torch.full_like(out, self.FUTURE_NULL), out,
        )
        out = torch.where(
            is_null & ~is_future,
            torch.full_like(out, self.PAST_NULL), out,
        )
        return out

    # ----------------------------------------------------------------
    # Proper time / distance
    # ----------------------------------------------------------------

    def proper_time(
        self, x: torch.Tensor, y: torch.Tensor,
    ) -> torch.Tensor:
        """Proper time `τ = √(-⟨y - x, y - x⟩_M)` for timelike-
        separated pairs. Returns NaN element-wise for non-timelike
        pairs (where the square root would be of a non-negative
        quantity, making the operation type-inappropriate).

        Physically: proper time along the straight-line world-line
        from `x` to `y`, the time elapsed for an observer travelling
        directly between the events.

        References:
          MTW (1973) §1.4 (proper time, "wristwatch time").
        """
        i_sq = self.interval_sq(x, y)
        # Timelike ⇔ i_sq < 0. Mask out non-timelike to NaN to make
        # type-error obvious to callers rather than silently returning
        # a meaningless value (e.g. for spacelike, sqrt of positive ≠
        # proper time but proper distance — different physics).
        is_timelike = i_sq < 0
        # Compute √(-i_sq) on the safe (timelike) values; substitute
        # NaN elsewhere via where-mask.
        safe = torch.where(is_timelike, -i_sq, torch.ones_like(i_sq))
        tau = torch.where(
            is_timelike,
            torch.sqrt(safe),
            torch.full_like(i_sq, float("nan")),
        )
        return tau

    def proper_distance(
        self, x: torch.Tensor, y: torch.Tensor,
    ) -> torch.Tensor:
        """Proper distance `s = √⟨y - x, y - x⟩_M` for spacelike-
        separated pairs. Returns NaN element-wise for non-spacelike
        pairs.

        Physically: spatial distance in the reference frame where the
        events are simultaneous.

        References:
          MTW (1973) §1.4 (proper distance / spatial interval).
        """
        i_sq = self.interval_sq(x, y)
        is_spacelike = i_sq > 0
        safe = torch.where(is_spacelike, i_sq, torch.ones_like(i_sq))
        return torch.where(
            is_spacelike,
            torch.sqrt(safe),
            torch.full_like(i_sq, float("nan")),
        )

    # ----------------------------------------------------------------
    # Curvature-tensor primitives
    # ----------------------------------------------------------------
    #
    # Flat Minkowski spacetime is identically Ricci-flat: the metric is
    # constant, all Christoffel symbols vanish, and the full Riemann
    # tensor is zero. These methods exist for API parity with curved
    # Lorentzian backgrounds (Schwarzschild, FLRW, etc.) that may be
    # added later as subclasses or sibling manifolds.
    #
    # Shape convention follows physics literature:
    #   g_μν          (n, n)    metric tensor with one upper one lower
    #                            index irrelevant; we store as (n, n)
    #   Γ^σ_μν        (n, n, n) Christoffel symbol of the 2nd kind,
    #                            with the upper index σ first
    #   R^σ_ρμν       (n, n, n, n) Riemann curvature tensor
    #   R_μν          (n, n)    Ricci tensor
    #   R             ( )       Ricci scalar (scalar curvature)

    def metric_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """Minkowski metric `g_μν = diag(-1, +1, +1, …, +1)`.

        Constant on flat spacetime; the `x` argument is unused but
        included for API parity with curved Lorentzian backgrounds.

        Args:
          x: `(..., n)` base point (unused).
        Returns:
          `(..., n, n)` metric tensor.

        References:
          O'Neill (1983), Semi-Riemannian Geometry, Ch. 3, flat metric
          and its Levi-Civita connection.
        """
        eta = torch.ones(self.n, device=x.device, dtype=x.dtype)
        eta[0] = -1.0
        # Broadcast diagonal to (..., n, n)
        out_shape = x.shape[:-1] + (self.n, self.n)
        return torch.diag_embed(eta).expand(out_shape).contiguous()

    def christoffel_symbols(self, x: torch.Tensor) -> torch.Tensor:
        """Christoffel symbols `Γ^σ_μν` of the Levi-Civita connection.

        Identically zero on flat Minkowski spacetime (the metric has
        zero derivatives). Returned as a `(..., n, n, n)` tensor with
        the upper index first.

        Subclasses representing curved Lorentzian backgrounds should
        override this with the appropriate non-zero expression.

        References:
          MTW (1973) §8.5 (Christoffel symbols in flat spacetime);
          O'Neill (1983) §3.5 (Levi-Civita connection on
          semi-Riemannian manifolds).
        """
        return torch.zeros(
            *x.shape[:-1], self.n, self.n, self.n,
            device=x.device, dtype=x.dtype,
        )

    def riemann_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """Riemann curvature tensor `R^σ_ρμν`.

        Identically zero on flat Minkowski (the spacetime is flat:
        parallel transport around any closed loop returns the original
        vector). Shape `(..., n, n, n, n)`.

        References:
          MTW (1973) §11.3 (Riemann tensor in flat spacetime);
          O'Neill (1983) §3.5 (curvature tensor formulas).
        """
        return torch.zeros(
            *x.shape[:-1], self.n, self.n, self.n, self.n,
            device=x.device, dtype=x.dtype,
        )

    def ricci_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """Ricci tensor `R_μν = R^σ_μσν`.

        Identically zero on flat Minkowski. Shape `(..., n, n)`.

        Physics relevance: Einstein's equations equate `R_μν − (1/2) R
        g_μν + Λ g_μν` to the stress-energy tensor (up to constants);
        in vacuum without cosmological constant, `R_μν = 0`; flat
        Minkowski is the simplest such solution.

        References:
          MTW (1973) §17.2.
        """
        return torch.zeros(
            *x.shape[:-1], self.n, self.n,
            device=x.device, dtype=x.dtype,
        )

    def scalar_curvature(self, x: torch.Tensor) -> torch.Tensor:
        """Ricci scalar `R = g^μν R_μν`.

        Identically zero on flat Minkowski. Shape `(...,)` (a scalar
        per batch element).

        References:
          MTW (1973) §17.4 (scalar curvature).
        """
        return torch.zeros(
            *x.shape[:-1], device=x.device, dtype=x.dtype,
        )

    # ----------------------------------------------------------------
    # Construction
    # ----------------------------------------------------------------

    def origin(self, batch_size: int = 1) -> torch.Tensor:
        """Origin `(0, 0, …, 0) ∈ R^{1, n-1}`; convention for the
        spacetime origin event.

        References:
          O'Neill (1983), Semi-Riemannian Geometry, Ch. 3, flat metric
          and its Levi-Civita connection.
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
        """Sample B random points in spacetime; Gaussian in each of
        the `n` coordinates. The manifold is `R^n` so no rejection
        needed.

        References:
          O'Neill (1983), Semi-Riemannian Geometry, Ch. 3, flat metric
          and its Levi-Civita connection.
        """
        if batch_size < 0:
            raise ValueError(f"batch_size must be >= 0, got {batch_size}")
        return torch.randn(
            batch_size,
            self.n,
            generator=generator,
            device=generator.device if generator is not None else self.device,
            dtype=self.dtype,
        ).to(self.device)

    # ----------------------------------------------------------------
    # Trivial flat-space geodesic primitives (included for API parity
    # with the Riemannian manifolds)
    # ----------------------------------------------------------------

    def projection(
        self, x: torch.Tensor, w: torch.Tensor,
    ) -> torch.Tensor:
        """Tangent-space projection. For flat space the tangent at
        every point is the full `R^n`, so projection is the identity.
        Included for API parity with curved Riemannian manifolds.

        References:
          O'Neill (1983), Semi-Riemannian Geometry, Ch. 3, flat metric
          and its Levi-Civita connection.
        """
        del x
        return w

    def egrad_to_rgrad(self, point: torch.Tensor, egrad: torch.Tensor) -> torch.Tensor:
        """Metric dual of an ambient covector in flat spacetime.

        The Lorentzian metric is indefinite: this is not generally a descent
        direction, unlike a Riemannian metric gradient.

        Args:
          point: (B, ambient_dim) base point.
          egrad: (B, ambient_dim) Euclidean gradient.
        Returns:
          (B, ambient_dim) metric gradient in tangent coordinates.

        References:
          O'Neill (1983), Semi-Riemannian Geometry, Ch. 3, metric duality.
        """
        return torch.cat((-egrad[..., :1], egrad[..., 1:]), dim=-1)

    def exp(
        self, x: torch.Tensor, v: torch.Tensor,
    ) -> torch.Tensor:
        """Exponential map. In flat space all geodesics are straight
        lines, so `exp_x(v) = x + v` regardless of `v`'s causal type.

        References:
          O'Neill (1983) §3.5 (flat semi-Riemannian geodesics).
        """
        return x + v

    def log(
        self, x: torch.Tensor, y: torch.Tensor,
    ) -> torch.Tensor:
        """Logarithmic map = `y - x` (the geodesic from `x` to `y` is
        a straight line; its tangent at `x` is the difference vector).

        References:
          O'Neill (1983), Semi-Riemannian Geometry, Ch. 3, flat metric
          and its Levi-Civita connection.
        """
        return y - x

    def retraction(
        self, x: torch.Tensor, v: torch.Tensor,
    ) -> torch.Tensor:
        """Retraction = exponential map for flat space.

        References:
          O'Neill (1983), Semi-Riemannian Geometry, Ch. 3, flat metric
          and its Levi-Civita connection.
        """
        return self.exp(x, v)

    # ----------------------------------------------------------------
    # is_on_manifold — trivially True (flat space has no constraint)
    # ----------------------------------------------------------------

    def is_on_manifold(
        self, x: torch.Tensor, atol: float = 1e-9,
    ) -> torch.Tensor:
        """Always True per-batch element; `R^n` has no constraint
        surface. Included for API parity. `atol` is accepted for
        signature compatibility but unused.

        References:
          O'Neill (1983), Semi-Riemannian Geometry, Ch. 3, flat metric
          and its Levi-Civita connection.
        """
        del atol
        return torch.ones(x.shape[:-1], dtype=torch.bool,
                           device=x.device)
