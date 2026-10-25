# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Synoros

"""SPD (symmetric positive definite) matrix manifold, GPU-native, batched-first.

The SPD manifold

    P(n) = { S ∈ R^{n × n} : S = Sᵀ, x ⊺ S x > 0 ∀ x ≠ 0 }

is an open submanifold of the symmetric matrices Sym(n) ⊂ R^{n × n} of
dimension n(n+1)/2. The tangent space at any S ∈ P(n) is canonically
identified with Sym(n) (it is the open cone, so the tangent at any
interior point is the ambient symmetric matrices).

This implementation uses the **affine-invariant metric** (Pennec et al.
2006):

    ⟨U, V⟩_S = tr(S⁻¹ U S⁻¹ V),   U, V ∈ Sym(n).

Under this metric the geodesics, exponential and logarithm maps admit
the closed forms

    exp_S(V) = S^{1/2} expm( S^{−1/2} V S^{−1/2} ) S^{1/2},
    log_S(T) = S^{1/2} logm( S^{−1/2} T S^{−1/2} ) S^{1/2},
    d(S, T)² = ‖ logm( S^{−1/2} T S^{−1/2} ) ‖_F² = Σ_i log²(λ_i),

where λ_i are the generalized eigenvalues solving T v = λ S v. The
affine-invariant metric is invariant under congruence (S ↦ Aᵀ S A for
invertible A), which is the property that makes it the canonical choice
for covariance-matrix and information-geometric work.

Tangent vectors are stored as `(B, n, n)` symmetric tensors.

References:
  Pennec, X., Fillard, P., Ayache, N. (2006). A Riemannian framework for
    tensor computing. International Journal of Computer Vision, 66(1):41–66.
  Bhatia, R. (2007). Positive Definite Matrices. Princeton University Press,
    chapters 4–6.
  Sra, S., Hosseini, R. (2015). Conic geometric optimization on the manifold
    of positive definite matrices. SIAM J. Optim. 25(1):713–739.
"""

from __future__ import annotations

from typing import Any, Literal

import torch
from torch.autograd.function import once_differentiable

_MatrixFunction = Literal["sqrt", "inv_sqrt", "exp", "log"]


class _SymmetricMatrixFunction(torch.autograd.Function):
    """Spectral functions with the continuous Daleckii-Krein derivative.

    Eigenvectors are only coordinates for the derivative, so repeated
    eigenvalues do not introduce the singular eigenvector derivatives of eigh.
    This backward implements first derivatives, not higher-order derivatives.

    References:
      Higham (2008), Functions of Matrices, Theorem 3.11.
    """

    @staticmethod
    def forward(ctx: Any, matrix: torch.Tensor, kind: _MatrixFunction) -> torch.Tensor:
        values, vectors = torch.linalg.eigh(0.5 * (matrix + matrix.mT))
        if kind != "exp":
            values = values.clamp_min(torch.finfo(matrix.dtype).tiny)
        if kind == "sqrt":
            transformed = values.sqrt()
        elif kind == "inv_sqrt":
            transformed = values.rsqrt()
        elif kind == "exp":
            transformed = values.exp()
        else:
            transformed = values.log()
        ctx.kind = kind
        ctx.save_for_backward(values, vectors, transformed)
        return (vectors * transformed.unsqueeze(-2)) @ vectors.mT

    @staticmethod
    @once_differentiable
    def backward(ctx: Any, grad: torch.Tensor) -> tuple[torch.Tensor, None]:
        values, vectors, transformed = ctx.saved_tensors
        left, right = values.unsqueeze(-1), values.unsqueeze(-2)
        gap = left - right
        # The midpoint derivative has O(gap²) error. Switching at sqrt(eps)
        # times the local eigenvalue scale keeps that error at roundoff.
        scale = torch.maximum(left.abs(), right.abs())
        close = gap.abs() <= torch.finfo(values.dtype).eps**0.5 * scale
        safe_gap = torch.where(close, torch.ones_like(gap), gap)
        midpoint = 0.5 * (left + right)
        if ctx.kind == "sqrt":
            divided = (left.sqrt() + right.sqrt()).reciprocal()
            derivative = 0.5 * midpoint.rsqrt()
        elif ctx.kind == "inv_sqrt":
            roots = left.sqrt() * right.sqrt()
            divided = -1.0 / (roots * (left.sqrt() + right.sqrt()))
            derivative = -0.5 * midpoint.rsqrt() / midpoint
        elif ctx.kind == "log":
            # log1p retains the small difference between nearby eigenvalues.
            low = torch.minimum(left, right)
            nearby = gap.abs() <= low
            relative_gap = torch.where(nearby, gap.abs(), 0.0) / low
            local = torch.log1p(relative_gap) / safe_gap.abs()
            log_difference = transformed.unsqueeze(-1) - transformed.unsqueeze(-2)
            divided = torch.where(nearby, local, log_difference / safe_gap)
            derivative = midpoint.reciprocal()
        else:
            # Factor out the larger exponential to avoid expm1 overflow.
            high = torch.maximum(transformed.unsqueeze(-1), transformed.unsqueeze(-2))
            divided = high * (-torch.expm1(-gap.abs())) / safe_gap.abs()
            derivative = midpoint.exp()
        loewner = torch.where(close, derivative, divided)
        local_grad = vectors.mT @ (0.5 * (grad + grad.mT)) @ vectors
        return vectors @ (loewner * local_grad) @ vectors.mT, None


def _symmetric_function(matrix: torch.Tensor, kind: _MatrixFunction) -> torch.Tensor:
    return _SymmetricMatrixFunction.apply(matrix, kind)


class SPDManifold:
    """Affine-invariant SPD(n) manifold, GPU-native + batched-first.

    Spectral exp/log/distance and whitening support first derivatives at
    repeated eigenvalues through a Daleckii-Krein backward. Higher-order
    derivatives of these operations are not implemented.


    Args:
      n: matrix size (operations work on (B, n, n) symmetric tensors).
      device, dtype: tensor placement and precision.

    Example:
      >>> mfd = SPDManifold(n=4)
      >>> S = mfd.random_point(batch_size=3)
      >>> S.shape, mfd.is_spd(S).all().item()
      (torch.Size([3, 4, 4]), True)
    """

    def __init__(self, n: int,
                 device: str | torch.device = "cpu",
                 dtype: torch.dtype = torch.float64):
        if n <= 0:
            raise ValueError(f"n must be > 0, got n={n}")
        self.n = n
        self.device = torch.device(device)
        self.dtype = dtype

    @property
    def dim(self) -> int:
        """Manifold dimension n(n+1)/2.

        References:
          Pennec et al. (2006), §3.1.
        """
        # n*(n+1) is a derived even integer; integer division by 2 is exact.
        return self.n * (self.n + 1) // 2


    # ----------------------------------------------------------------
    # Construction
    # ----------------------------------------------------------------

    def random_point(
        self,
        batch_size: int = 1,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Sample B SPD matrices via the Wishart-like construction A Aᵀ.

        Construction: A ~ N(0, I)^{B × n × n}; return A Aᵀ. When A is
        full-rank (with probability one for Gaussian A), A Aᵀ is SPD.
        Distribution-wise this is a (rescaled) Wishart W_n(I, n); we do not
        normalize since callers typically only require *some* SPD draw for
        initialization / testing.

        References:
          Anderson, T. W. (2003). An Introduction to Multivariate Statistical
            Analysis, 3rd ed., chapter 7; Wishart distribution.

        Returns:
          Tensor of shape (B, n, n).
        """
        if batch_size < 0:
            raise ValueError(f"batch_size must be >= 0, got {batch_size}")
        A = torch.randn(
            batch_size,
            self.n,
            self.n,
            generator=generator,
            device=generator.device if generator is not None else self.device,
            dtype=self.dtype,
        ).to(self.device)
        return torch.bmm(A, A.mT)

    def is_spd(self, S: torch.Tensor) -> torch.Tensor:
        """Per-batch test for SPD: symmetric and all eigenvalues > 0.

        Returns:
          (B,) boolean tensor.

        References:
          Pennec, Fillard, Ayache (2006), A Riemannian Framework for Tensor
          Computing, §3, positive-definite matrices and affine-invariant geometry.
        """
        sym_err = torch.linalg.matrix_norm(S - S.mT, dim=(-2, -1))
        sym_ref = torch.linalg.matrix_norm(S, dim=(-2, -1))
        # Treat as symmetric if asymmetric part is at numerical-noise level.
        is_sym = sym_err <= 1e-9 * torch.clamp(sym_ref, min=1.0)
        eigvals = torch.linalg.eigvalsh(0.5 * (S + S.mT))  # (B, n)
        is_pos = (eigvals > 0).all(dim=-1)
        return is_sym & is_pos

    # ----------------------------------------------------------------
    # Tangent operations
    # ----------------------------------------------------------------

    def projection(self, S: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
        """Project ambient direction Z onto the tangent space at S.

        T_S P(n) = Sym(n), so the projection is just symmetrization:

            P_T(Z) = ½ (Z + Zᵀ).

        Independent of the base point S (T_S P(n) does not depend on S in
        the embedded ambient form).

        References:
          Bhatia (2007), §6.1.

        Args:
          S: base point (B, n, n), unused (included for API consistency).
          Z: ambient direction (B, n, n).
        Returns:
          Symmetric tangent (B, n, n).
        """
        del S  # tangent space is point-independent in ambient form
        return 0.5 * (Z + Z.mT)

    def inner(
        self, S: torch.Tensor, U: torch.Tensor, V: torch.Tensor
    ) -> torch.Tensor:
        """Affine-invariant inner product ⟨U, V⟩_S = tr(S⁻¹ U S⁻¹ V).

        A shared Cholesky factor computes both solves. The trace contracts
        transposed indices; the solved matrices need not be symmetric.

        References:
          Pennec et al. (2006), eq. 4.

        Args:
          S: base point (B, n, n).
          U, V: tangent vectors (B, n, n).
        Returns:
          (B,) inner products.
        """
        factor = torch.linalg.cholesky(S)
        S_inv_U = torch.cholesky_solve(U, factor)
        S_inv_V = torch.cholesky_solve(V, factor)
        # tr(X Y) = sum_{i,j} X_ij Y_ji = sum_{i,j} X_ij (Yᵀ)_ij
        return (S_inv_U * S_inv_V.mT).sum(dim=(-2, -1))

    def norm(self, S: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """Frobenius norm of the Cholesky-whitened tangent, shape (B,).

        Uses PyTorch's zero subgradient at a zero tangent.

        References:
          Pennec et al. (2006), eq. 4.
        """
        factor = torch.linalg.cholesky(S)
        left = torch.linalg.solve_triangular(factor, V, upper=False)
        whitened = torch.linalg.solve_triangular(factor, left.mT, upper=False).mT
        return torch.linalg.vector_norm(whitened, dim=(-2, -1))

    def egrad_to_rgrad(self, point: torch.Tensor, egrad: torch.Tensor) -> torch.Tensor:
        """Convert an ambient gradient G to S sym(G) S, shape (B, n, n).

        Args:
          point: (B, n, n) base point.
          egrad: (B, n, n) Euclidean gradient.
        Returns:
          (B, n, n) metric gradient in tangent coordinates.

        References:
          Absil, Mahony, Sepulchre (2008), §3.6; Pennec et al. (2006), eq. 4.
        """
        return point @ self.projection(point, egrad) @ point

    # ----------------------------------------------------------------
    # Exponential and logarithmic maps, geodesic distance
    # ----------------------------------------------------------------

    def _sqrt_and_inv_sqrt(
        self, S: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (S^{1/2}, S^{−1/2}) via eigh.

        Both are symmetric. S must be SPD; we clamp eigenvalues from
        below by the dtype's smallest-representable-positive to guard
        against floating-point eigh returning a slightly negative value
        for nearly-singular SPD matrices (a common artifact near the
        SPD-cone boundary, e.g. from low-sample-count Wishart draws or
        from accumulated drift in iterative pipelines). Without this
        clamp, `torch.rsqrt(0)` would propagate `inf` through all
        downstream exp/log/distance computations.
        """
        return _symmetric_function(S, "sqrt"), _symmetric_function(S, "inv_sqrt")

    def precompute_whitening(
        self, S: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute (S^{1/2}, S^{-1/2}) once for reuse across geodesic ops.

        The affine-invariant exp/log/distance all need S^{1/2} and S^{-1/2}
        at the base point. Computing them once and passing to multiple
        geodesic queries avoids redundant eighs in code paths like
        "compare distances from S to many T_i" or "step along several
        tangent directions at S in one iteration". Each eigh is
        O(n³) and dominates the operations on large n.

        Returns:
          (S_sqrt, S_inv_sqrt); both (..., n, n), symmetric.

        References:
          Pennec, Fillard, Ayache (2006), A Riemannian Framework for Tensor
          Computing, §3, positive-definite matrices and affine-invariant geometry.
        """
        return self._sqrt_and_inv_sqrt(S)

    def exp(
        self, S: torch.Tensor, V: torch.Tensor,
        whitening: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Exponential map exp_S(V) = S^{1/2} expm(S^{−1/2} V S^{−1/2}) S^{1/2}.

        References:
          Pennec et al. (2006), eq. 5.

        Args:
          S: base point (B, n, n), SPD.
          V: tangent (B, n, n), symmetric.
          whitening: optional (S^{1/2}, S^{-1/2}) precomputed via
            `precompute_whitening(S)`. Pass when calling exp/log
            multiple times at the same S to skip the redundant eigh.
        Returns:
          (B, n, n) SPD result.
        """
        S_sqrt, S_inv_sqrt = whitening if whitening is not None \
                                       else self._sqrt_and_inv_sqrt(S)
        inner = S_inv_sqrt @ V @ S_inv_sqrt
        # Symmetrize against numerical drift before the spectral exponential; result of
        # symmetric @ symmetric @ symmetric is symmetric in exact arithmetic.
        inner = 0.5 * (inner + inner.mT)
        # The outer S_sqrt @ ... @ S_sqrt is symmetric in exact arithmetic
        # but accumulates float drift; symmetrize so callers can chain
        # exp() into is_spd / further geodesics without false negatives.
        out = S_sqrt @ _symmetric_function(inner, "exp") @ S_sqrt
        return 0.5 * (out + out.mT)

    def log(
        self, S: torch.Tensor, T: torch.Tensor,
        whitening: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Logarithm map log_S(T) = S^{1/2} logm(S^{−1/2} T S^{−1/2}) S^{1/2}.

        Matrix log via eigh: logm(M) = U diag(log λ) Uᵀ for M = U diag(λ) Uᵀ
        symmetric. The inner argument is symmetric SPD (congruence preserves
        positivity), so this is well-defined.

        References:
          Pennec et al. (2006), eq. 6.

        Args:
          S, T: base point and target, both (B, n, n) SPD.
          whitening: optional (S^{1/2}, S^{-1/2}) precomputed via
            `precompute_whitening(S)`. See `exp` for the use case.
        Returns:
          (B, n, n) symmetric tangent.
        """
        S_sqrt, S_inv_sqrt = whitening if whitening is not None \
                                       else self._sqrt_and_inv_sqrt(S)
        inner = S_inv_sqrt @ T @ S_inv_sqrt
        inner = 0.5 * (inner + inner.mT)  # symmetrize
        log_inner = _symmetric_function(inner, "log")
        # log_S(T) lives in Sym(n); symmetrize the outer product to
        # absorb float drift, consistent with exp() above.
        out = S_sqrt @ log_inner @ S_sqrt
        return 0.5 * (out + out.mT)

    def distance(
        self, S: torch.Tensor, T: torch.Tensor,
        whitening: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Geodesic distance d(S, T) = ‖log(S^{−1/2} T S^{−1/2})‖_F.

        Equivalently d(S, T)² = Σ_i log²(λ_i) where λ_i are generalized
        eigenvalues (T v = λ S v). We compute via the whitened form which
        is well-conditioned for SPD inputs.

        References:
          Pennec et al. (2006), eq. 7.

        Args:
          S, T: base point and target, both (B, n, n) SPD.
          whitening: optional (S^{1/2}, S^{-1/2}) precomputed via
            `precompute_whitening(S)` to skip the S-eigh.
        Returns:
          (B,) geodesic distances.
        """
        if whitening is not None:
            _, S_inv_sqrt = whitening
        else:
            _, S_inv_sqrt = self._sqrt_and_inv_sqrt(S)
        whitened = S_inv_sqrt @ T @ S_inv_sqrt
        whitened = 0.5 * (whitened + whitened.mT)
        # Eigenvalues of S^{-1/2} T S^{-1/2} are positive in exact
        # arithmetic (congruent to the SPD T). Clamp by the dtype tiny
        # to guard against float error driving an eigenvalue slightly
        # below zero and producing NaN through log().
        log_matrix = _symmetric_function(whitened, "log")
        return torch.linalg.vector_norm(log_matrix, dim=(-2, -1))

    # ----------------------------------------------------------------
    # Retraction (exponential map is the canonical choice on SPD)
    # ----------------------------------------------------------------

    def retraction(
        self, S: torch.Tensor, V: torch.Tensor,
        whitening: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Retraction = exponential map.

        On SPD with the affine-invariant metric, exp is the canonical
        retraction (it is the geodesic, hence trivially a second-order
        retraction). For a cheaper first-order alternative add a separate
        method (e.g. project S + V onto SPD via eigh + positive-clamp)
        when benchmarks demand it.

        References:
          Absil-Mahony-Sepulchre (2008), §4.1; exponential as retraction.
        """
        return self.exp(S, V, whitening=whitening)
