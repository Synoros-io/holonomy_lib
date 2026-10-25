# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Synoros

"""κ-stereographic projection model; constant curvature manifold
that interpolates spherical (κ > 0), Euclidean (κ = 0), and
hyperbolic (κ < 0) geometry via a single parameter.

The κ-stereographic model represents:

  - **κ > 0** (spherical): the stereographic projection of the
    `n`-sphere of radius `1/√κ`, excluding one pole, onto all of `R^n`.
  - **κ = 0** (Euclidean): the standard `R^n`.
  - **κ < 0** (hyperbolic): the Poincaré ball of radius `1/√|κ|`,
    `{x ∈ R^n : ‖x‖² < 1/|κ|}`.

The models share curvature-dependent conformal metrics and exp/log maps.
The spherical formulas use a stereographic chart, so operations that hit
its omitted pole are undefined even though distance itself remains defined.

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

κ may be a Python float or a learnable scalar tensor. Tensor curvature
is evaluated on every call, including at and across zero.

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

import torch

# Taylor coefficients of tan(sqrt(q))/sqrt(q) and atan(sqrt(q))/sqrt(q).
# These analytic coefficients apply to signed q = kappa * alpha² near zero.
KAPPA_SCALE_LINEAR = 1.0 / 3.0
KAPPA_TAN_QUADRATIC = 2.0 / 15.0
KAPPA_ATAN_QUADRATIC = 1.0 / 5.0
# Preserve the existing sampling cap: one quarter of the tan pole radius.
SPHERICAL_SAMPLE_RADIUS_FRACTION = 1.0 / 8.0
# asin(sqrt(q))/sqrt(q) = 1 + q/6 + 3q²/40 + O(q³).
DISTANCE_SCALE_LINEAR = 1.0 / 6.0
DISTANCE_SCALE_QUADRATIC = 3.0 / 40.0


def _kappa_scale(
    alpha: torch.Tensor, kappa: float | torch.Tensor, *, inverse: bool = False,
) -> torch.Tensor:
    """Signed-curvature tan/artan scale with finite inactive branches.

    The quadratic series has O(q³) remainder. At |q| <= sqrt(eps),
    its value error is below roundoff and its derivative error is O(eps).
    """
    q = kappa * alpha.square()
    small = q.abs() <= torch.finfo(alpha.dtype).eps**0.5
    series_q = torch.where(small, q, torch.zeros_like(q))
    positive = (q > 0) & ~small
    negative = (q < 0) & ~small
    positive_arg = torch.sqrt(torch.where(positive, q, torch.ones_like(q)))
    # atanh's substitute must be strictly inside its domain.
    substitute = torch.full_like(q, 0.5) if inverse else torch.ones_like(q)
    negative_arg = torch.sqrt(torch.where(negative, -q, substitute.square()))
    if inverse:
        series = 1.0 - KAPPA_SCALE_LINEAR * series_q
        series = series + KAPPA_ATAN_QUADRATIC * series_q.square()
        spherical = torch.atan(positive_arg) / positive_arg
        hyperbolic = torch.atanh(negative_arg) / negative_arg
    else:
        series = 1.0 + KAPPA_SCALE_LINEAR * series_q
        series = series + KAPPA_TAN_QUADRATIC * series_q.square()
        spherical = torch.tan(positive_arg) / positive_arg
        hyperbolic = torch.tanh(negative_arg) / negative_arg
    return torch.where(small, series, torch.where(positive, spherical, hyperbolic))


def _project_ball(x: torch.Tensor, kappa: float | torch.Tensor) -> torch.Tensor:
    """Keep rounded hyperbolic coordinates strictly inside their open ball.

    A dimension-scaled eps margin covers accumulation error in the norm.
    Beyond this representable radius, exp saturates and log cannot recover
    the original large tangent. This is a coordinate precision limit.
    """
    k = torch.as_tensor(kappa, dtype=x.dtype, device=x.device)
    sqrt_negative = torch.sqrt(torch.where(k < 0, -k, torch.ones_like(k)))
    norm = torch.linalg.vector_norm(x, dim=-1)
    radius = (1.0 - x.shape[-1] * torch.finfo(x.dtype).eps) / sqrt_negative
    safe_norm = torch.where(norm > 0, norm, torch.ones_like(norm))
    scale = torch.where((k < 0) & (norm > radius), radius / safe_norm,
                        torch.ones_like(norm))
    return x * scale.unsqueeze(-1)


def _stereographic_distance(
    x: torch.Tensor, y: torch.Tensor, kappa: float | torch.Tensor,
) -> torch.Tensor:
    """Distance using sphere chords or a hyperbolic asinh chord formula.

    The signed series supplies the curvature derivative at zero. Sphere
    embedding chords also cover antipodes where Mobius subtraction is
    undefined; hyperbolic asinh avoids an atanh argument rounding to one.

    References:
      Bachmann et al. (2020), §2, stereographic metric; Cannon et al.
      (1997), Hyperbolic Geometry, §3, hyperbolic distance models.
    """
    k = torch.as_tensor(kappa, dtype=x.dtype, device=x.device)
    x_sq, y_sq = x.square().sum(-1), y.square().sum(-1)
    separation = torch.linalg.vector_norm(x - y, dim=-1)
    product = (1.0 + k * x_sq) * (1.0 + k * y_sq)
    chord = separation / torch.sqrt(product)
    q = k * chord.square()
    small = q.abs() <= torch.finfo(x.dtype).eps**0.5
    series_q = torch.where(small, q, torch.zeros_like(q))
    scale = 1.0 + DISTANCE_SCALE_LINEAR * series_q
    scale = scale + DISTANCE_SCALE_QUADRATIC * series_q.square()
    series = 2.0 * chord * scale

    spherical = (k > 0) & ~small
    root = torch.sqrt(torch.where(spherical, k, torch.ones_like(k)))
    # Mask inputs before lifting to the sphere so inactive denominators and
    # transcendental branches cannot inject NaNs into backward.
    sx = torch.where(spherical.unsqueeze(-1), root.unsqueeze(-1) * x, 0.0)
    sy = torch.where(spherical.unsqueeze(-1), root.unsqueeze(-1) * y, 0.0)
    sx_sq, sy_sq = sx.square().sum(-1, keepdim=True), sy.square().sum(-1, keepdim=True)
    lifted_x = torch.cat((2.0 * sx, 1.0 - sx_sq), dim=-1) / (1.0 + sx_sq)
    lifted_y = torch.cat((2.0 * sy, 1.0 - sy_sq), dim=-1) / (1.0 + sy_sq)
    sphere_chord = torch.linalg.vector_norm(lifted_x - lifted_y, dim=-1)
    sphere_sum = torch.linalg.vector_norm(lifted_x + lifted_y, dim=-1)
    sphere_distance = 2.0 * torch.atan2(sphere_chord, sphere_sum) / root

    hyperbolic = (k < 0) & ~small
    negative_root = torch.sqrt(torch.where(hyperbolic, -k, torch.ones_like(k)))
    hyperbolic_chord = torch.where(hyperbolic, chord, torch.zeros_like(chord))
    hyperbolic_distance = (
        2.0 * torch.asinh(negative_root * hyperbolic_chord) / negative_root
    )
    return torch.where(small, series, torch.where(spherical, sphere_distance,
                                                 hyperbolic_distance))


class KappaStereographicManifold:
    """Constant-curvature κ-stereographic manifold.

    Args:
      n: intrinsic / ambient dimension. Points are stored as `(B, n)`.
      kappa: sectional curvature, a Python float or learnable scalar tensor.
        Tensor curvature is evaluated on every call, including across zero.
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
        if isinstance(kappa, torch.Tensor):
            if kappa.ndim != 0 or not kappa.is_floating_point():
                raise TypeError("kappa must be a floating-point scalar tensor")
            self.kappa: float | torch.Tensor = kappa
        elif isinstance(kappa, (int, float)):
            self.kappa = float(kappa)
        else:
            raise TypeError("kappa must be a float or scalar tensor")
        self.n = n
        self.device = torch.device(device)
        self.dtype = dtype

    def _get_kappa(self) -> float | torch.Tensor:
        """Return the current curvature without detaching its gradient."""
        return self.kappa

    @property
    def dim(self) -> int:
        """Intrinsic manifold dimension `n`. Same as ambient for this
        model; no extra dimension (unlike `LorentzManifold(n)` which
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
        return _kappa_scale(alpha, self.kappa)

    def _atan_kappa_c(self, alpha: torch.Tensor) -> torch.Tensor:
        return _kappa_scale(alpha, self.kappa, inverse=True)

    def _conformal_factor(self, x: torch.Tensor) -> torch.Tensor:
        """λ_κ(x) = 2 / (1 + κ‖x‖²). At origin = 2.

        Uses `_get_kappa()` so the conformal factor's derivative w.r.t.
        a learnable-κ tensor is captured; affects `inner`, `norm`,
        `exp`, `log`, `parallel_transport`.
        """
        norm_sq = (x * x).sum(dim=-1)
        return 2.0 / (1.0 + self._get_kappa() * norm_sq)

    # ----------------------------------------------------------------
    # Construction
    # ----------------------------------------------------------------

    def origin(self, batch_size: int = 1) -> torch.Tensor:
        """The origin `(0, …, 0) ∈ R^n`; common base point for
        tangent-space embeddings. Same shape as a regular point.

        References:
          Bachmann, Bécigneul, Ganea (2020), Constant Curvature Graph
          Convolutional Networks, §2, stereographic metric and maps.
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
        """Draw Gaussian origin tangents and map them with the current κ.

        Positive-curvature tangents are capped at π/(8√κ), away from
        the stereographic chart's pole. Hyperbolic outputs saturate at
        a dtype-dependent interior radius. Draws use the generator's
        device before transfer to the manifold's device.

        References:
          Bachmann et al. (2020), §2, origin exponential.
        """
        if batch_size < 0:
            raise ValueError(f"batch_size must be >= 0, got {batch_size}")
        draw_device = generator.device if generator is not None else self.device
        v = torch.randn(batch_size, self.n, generator=generator,
                        device=draw_device, dtype=self.dtype).to(self.device) * 0.5
        k = torch.as_tensor(self.kappa, device=v.device, dtype=v.dtype)
        sqrt_positive = torch.sqrt(torch.where(k > 0, k, torch.ones_like(k)))
        cap = SPHERICAL_SAMPLE_RADIUS_FRACTION * math.pi / sqrt_positive
        norm = torch.linalg.vector_norm(v, dim=-1, keepdim=True)
        safe_norm = torch.where(norm > 0, norm, torch.ones_like(norm))
        v = v * torch.where((k > 0) & (norm > cap), cap / safe_norm,
                            torch.ones_like(norm))
        return self.exp_0(v)

    def is_on_manifold(
        self, x: torch.Tensor, atol: float = 1e-9,
    ) -> torch.Tensor:
        """Check finite coordinates and the hyperbolic open-ball boundary.

        For κ < 0, -κ||x||² < 1 - atol; for κ >= 0 every finite chart
        coordinate is valid. The sphere's omitted pole is at infinity.

        References:
          Bachmann et al. (2020), §2, stereographic domain.
        """
        if x.ndim < 2 or x.shape[-1] != self.n:
            raise ValueError(f"x must have shape (..., {self.n}); got {tuple(x.shape)}")
        k = torch.as_tensor(self.kappa, device=x.device, dtype=x.dtype)
        domain = (k >= 0) | (-k * x.square().sum(dim=-1) < 1.0 - atol)
        return torch.isfinite(x).all(dim=-1) & torch.isfinite(k) & domain

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

        References:
          Bachmann, Bécigneul, Ganea (2020), Constant Curvature Graph
          Convolutional Networks, §2, stereographic metric and maps.
        """
        xy = (x * y).sum(dim=-1, keepdim=True)        # (..., 1)
        xx = (x * x).sum(dim=-1, keepdim=True)
        yy = (y * y).sum(dim=-1, keepdim=True)
        k = self._get_kappa()
        num = (1.0 - 2.0 * k * xy - k * yy) * x + (1.0 + k * xx) * y
        # Hyperbolic inputs have a positive denominator. In the spherical
        # chart the sum can be the omitted pole; that operation is undefined.
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

        References:
          Bachmann, Bécigneul, Ganea (2020), Constant Curvature Graph
          Convolutional Networks, §2, stereographic metric and maps.
        """
        del x
        return w

    def egrad_to_rgrad(self, point: torch.Tensor, egrad: torch.Tensor) -> torch.Tensor:
        """Divide an ambient gradient by the squared conformal factor.

        Args:
          point: (B, ambient_dim) base point.
          egrad: (B, ambient_dim) Euclidean gradient.
        Returns:
          (B, ambient_dim) metric gradient in tangent coordinates.

        References:
          Absil, Mahony, Sepulchre (2008), §3.6; Bachmann et al. (2020), §2.
        """
        return egrad / self._conformal_factor(point).square().unsqueeze(-1)

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

        Uses PyTorch's finite zero subgradient at v = 0.

        References:
          Bachmann, Bécigneul, Ganea (2020), Constant Curvature Graph
          Convolutional Networks, §2, stereographic metric and maps.
        """
        lam = self._conformal_factor(x)
        return lam * torch.linalg.vector_norm(v, dim=-1)

    # ----------------------------------------------------------------
    # Exponential / logarithmic maps, geodesic distance
    # ----------------------------------------------------------------

    def exp_0(self, v: torch.Tensor) -> torch.Tensor:
        """Exponential map at the origin: `exp_0(v) = tan_κ(√|κ|·‖v‖)
        · v / (√|κ|·‖v‖)`.

        At κ = 0: `exp_0(v) = v` (Euclidean identity).
        At κ < 0: maps Euclidean coords to the Poincaré ball.
        At κ > 0: the chart excludes the pole at infinity.
        At κ < 0: large tangents saturate at a dimension-scaled eps
        margin inside the ball; log_0 cannot invert that saturation.

        Args:
          v: `(..., n)` Euclidean tangent vector at the origin.
        Returns:
          `(..., n)` point on the manifold.

        References:
          Bachmann, Bécigneul, Ganea (2020), Constant Curvature Graph
          Convolutional Networks, §2, stereographic metric and maps.
        """
        if v.ndim < 2 or v.shape[-1] != self.n:
            raise ValueError(
                f"v last dim must be n={self.n}, got {tuple(v.shape)}"
            )
        v_norm = torch.linalg.vector_norm(v, dim=-1)
        # tan_κ(√|κ|·‖v‖) / (√|κ|·‖v‖) · v.
        # Factor (1/√|κ|·‖v‖) absorbed via _tan_kappa_c which returns
        # tan_κ(scaled)/scaled where scaled = √|κ|·v_norm.
        scaling = self._tan_kappa_c(v_norm)
        return _project_ball(scaling.unsqueeze(-1) * v, self.kappa)

    def log_0(self, y: torch.Tensor) -> torch.Tensor:
        """Logarithmic map at the origin: `log_0(y) =
        (1/√|κ|) · tan_κ⁻¹(√|κ|·‖y‖) · y / ‖y‖`.

        At κ = 0: `log_0(y) = y` (Euclidean identity).
        For κ > 0 this selects the principal logarithm; exp_0 is invertible
        only for ||v|| < π/(2√κ). Hyperbolic saturation is not invertible.

        Args:
          y: `(..., n)` point on the manifold.
        Returns:
          `(..., n)` Euclidean tangent at the origin.

        References:
          Bachmann, Bécigneul, Ganea (2020), Constant Curvature Graph
          Convolutional Networks, §2, stereographic metric and maps.
        """
        if y.ndim < 2 or y.shape[-1] != self.n:
            raise ValueError(
                f"y last dim must be n={self.n}, got {tuple(y.shape)}"
            )
        y_norm = torch.linalg.vector_norm(y, dim=-1)
        # arctan_κ(√|κ|·‖y‖) / (√|κ|·‖y‖) · y. Same prefactor structure
        # as exp_0 — _atan_kappa_c returns the (atan_κ(α)/α) factor.
        scaling = self._atan_kappa_c(y_norm)
        return scaling.unsqueeze(-1) * y

    def distance(
        self, x: torch.Tensor, y: torch.Tensor,
    ) -> torch.Tensor:
        """Geodesic distance `d_κ(x, y) = (2/√|κ|) · tan_κ⁻¹(√|κ|
        · ‖(-x) ⊕_κ y‖)`.

        Evaluated through sphere chords or a hyperbolic asinh chord formula
        to cover spherical antipodes and near-boundary hyperbolic pairs.
        The distance is nonsmooth at coincident points and spherical antipodes;
        the implementation uses finite zero-subgradient conventions there.

        At κ = 0: `d(x, y) = 2 · ‖y - x‖_Eucl` (Bachmann normalization;
        the conformal factor at the origin is 2).

        Args:
          x, y: `(..., n)` points on the manifold.
        Returns:
          `(...,)` non-negative distances.

        References:
          Bachmann, Bécigneul, Ganea (2020), Constant Curvature Graph
          Convolutional Networks, §2, stereographic metric and maps.
        """
        return _stereographic_distance(x, y, self.kappa)

    def exp(
        self, x: torch.Tensor, v: torch.Tensor,
    ) -> torch.Tensor:
        """Exponential map at a general base point:

            exp_x(v) = x ⊕_κ (tan_κ(λ_κ(x) · √|κ| · ‖v‖/2)
                              · v / (√|κ|·‖v‖))

        References:
          Bachmann et al. (2020), eq. 5.
        """
        v_norm = torch.linalg.vector_norm(v, dim=-1)
        lam = self._conformal_factor(x)
        # tan_κ(arg) / (√|κ| · ‖v‖) · v, with arg = λ · √|κ| · ‖v‖ / 2.
        # Rewrite as: (lam / 2) · (tan_κ(arg)/arg) · v   (factor √|κ|·‖v‖
        # cancels with the implicit factor inside arg).
        arg = 0.5 * lam * v_norm
        scaling = 0.5 * lam * self._tan_kappa_c(arg)
        rhs = scaling.unsqueeze(-1) * v
        return _project_ball(self.mobius_add(x, rhs), self.kappa)

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
        diff_norm = torch.linalg.vector_norm(diff, dim=-1)
        lam = self._conformal_factor(x)
        # atan_κ(√|κ|·‖diff‖) / (√|κ|·‖diff‖) · diff factor:
        scaling = (2.0 / lam) * self._atan_kappa_c(diff_norm)
        return scaling.unsqueeze(-1) * diff

    def retraction(
        self, x: torch.Tensor, v: torch.Tensor,
    ) -> torch.Tensor:
        """Retraction = exponential map (second-order, the geodesic
        itself satisfies the retraction axioms on constant-curvature
        spaces).

        References:
          Bachmann, Bécigneul, Ganea (2020), Constant Curvature Graph
          Convolutional Networks, §2, stereographic metric and maps.
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
          Bachmann et al. (2020), §2, κ-stereographic geometry;
          Ungar (2008), A Gyrovector Space Approach to Hyperbolic Geometry,
          gyration. The linear formula agrees with Geoopt stereographic.math.
        """
        # Gyration is linear on unrestricted tangent vectors. Evaluating
        # nested Mobius additions would incorrectly treat v as a ball point.
        a, b = y, -x
        aa = a.square().sum(dim=-1, keepdim=True)
        bb = b.square().sum(dim=-1, keepdim=True)
        ab = (a * b).sum(dim=-1, keepdim=True)
        av = (a * v).sum(dim=-1, keepdim=True)
        bv = (b * v).sum(dim=-1, keepdim=True)
        k = self.kappa
        coef_a = -k * k * av * bb - k * bv + 2.0 * k * k * ab * bv
        coef_b = -k * k * bv * aa + k * av
        denominator = 1.0 - 2.0 * k * ab + k * k * aa * bb
        gyrated = v + 2.0 * (coef_a * a + coef_b * b) / denominator
        ratio = self._conformal_factor(x) / self._conformal_factor(y)
        return ratio.unsqueeze(-1) * gyrated
