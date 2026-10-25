# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Synoros

"""Fixed-rank matrix manifold, GPU-native, batched-first.

The fixed-rank matrix manifold

    M_r(m, n) = { M ∈ R^{m × n} : rank(M) = r }

is a smooth ( r·(m + n − r) )-dimensional embedded submanifold of R^{m×n}.
Points are represented in SVD form (U, S, Vt) where

    U  ∈ R^{B × m × r}   columns orthonormal
    S  ∈ R^{B × r}       positive, sorted descending
    Vt ∈ R^{B × r × n}   rows orthonormal (i.e. V^T)

and the ambient dense matrix is `M = U @ diag(S) @ Vt`. The leading batch
dimension B is required for all operations; pass B = 1 for single-point use.

Tangent vectors are stored in *ambient form*; i.e. as (B, m, n) matrices
in the embedding space, with the understanding that they satisfy the
tangent-space constraint at the corresponding point. This matches
Vandereycken's exposition and matches the calling convention used by most
Riemannian optimizers (the gradient comes in as an ambient matrix; we
project it to a valid tangent).

References:
  Absil, P.-A., Mahony, R., Sepulchre, R. (2008). Optimization Algorithms
    on Matrix Manifolds. Princeton University Press.
  Vandereycken, B. (2013). Low-rank matrix completion by Riemannian
    optimization. SIAM Journal on Optimization, 23(2):1214–1236.
"""

from __future__ import annotations

from typing import Any, Literal

import torch
from torch.autograd.function import once_differentiable

from holonomy_lib.algebra.linear import truncated_svd

# A batched manifold point is the triple (U, S, Vt) of stacked tensors.
# Shapes: U (B, m, r), S (B, r), Vt (B, r, n).
FixedRankPoint = tuple[torch.Tensor, torch.Tensor, torch.Tensor]

# Rank ratio r / min(m, n) where the retraction's top-r truncation stops
# using randomized SVD. Measured on CPU in the 0.7.0 benchmark baseline
# (notes/benchmark_baseline.md, in git history at d26eb84): above
# 0.25 the exact SVD's constant factor wins back the gap, and its accuracy
# and determinism matter more for research workloads. At or below it,
# randomized SVD (HMT 2011) is strictly faster in our CPU baseline, 37× at
# 1024×1024 with r = 32. Cataloged in notes/magic_numbers.md under
# `RETRACTION_RANDOMIZED_THRESHOLD`.
RETRACTION_RANDOMIZED_THRESHOLD: float = 0.25


class _TruncatedSVD(torch.autograd.Function):
    """Top-r SVD factors of a matrix, with a first-derivative backward.

    Takes a (..., m, n) matrix and a rank r and returns (U, S, Vt) with
    shapes (..., m, r), (..., r), (..., r, n). The forward runs a full thin
    SVD and slices, so the backward can see the discarded singular values.

    The backward is Townsend's SVD derivative restricted to the retained
    factors. Pairs of discarded values carry no gradient, so a run of zero
    discarded singular values (a rank-deficient input) does not create the
    1/0 poles that differentiating the full SVD would. At repeated retained
    values the derivative is defined for losses on the reconstructed matrix,
    not arbitrary losses on nonunique factors. A tie across the
    retained/discarded boundary is truly nondifferentiable and comes back
    as inf or NaN.

    References:
      Townsend (2016), Differentiating the Singular Value Decomposition.
      Vandereycken (2013), §2.3, metric projection retraction.
    """

    @staticmethod
    def forward(ctx: Any, matrix: torch.Tensor, rank: int) -> FixedRankPoint:
        u, singular, vt = torch.linalg.svd(matrix, full_matrices=False)
        ctx.rank = rank
        ctx.save_for_backward(u, singular, vt)
        return u[..., :rank], singular[..., :rank], vt[..., :rank, :]

    @staticmethod
    @once_differentiable
    def backward(
        ctx: Any, grad_u: torch.Tensor, grad_s: torch.Tensor, grad_vt: torch.Tensor,
    ) -> tuple[torch.Tensor, None]:
        u, singular, vt = ctx.saved_tensors
        rank = ctx.rank
        v = vt.mT
        # Pad the incoming gradients out to the full thin SVD with zeros in
        # the discarded slots, so Townsend's formulas can run on k = min(m, n)
        # columns and the discarded directions simply contribute nothing.
        gu, gs, gv = (
            torch.zeros_like(u),
            torch.zeros_like(singular),
            torch.zeros_like(v),
        )
        gu[..., :rank], gs[..., :rank], gv[..., :rank] = grad_u, grad_s, grad_vt.mT
        a, b = u.mT @ gu, vt @ gv  # Uᵀ Ū and Vᵀ V̄, both (..., k, k)
        left, right = singular.unsqueeze(-1), singular.unsqueeze(-2)
        # Masks over (i, j) pairs: `active` when at least one of the two is
        # retained, `both_kept` when both are.
        kept = torch.arange(singular.shape[-1], device=singular.device) < rank
        active = kept.unsqueeze(-1) | kept.unsqueeze(-2)
        both_kept = kept.unsqueeze(-1) & kept.unsqueeze(-2)
        gap = right.square() - left.square()  # s_j² − s_i², Townsend's 1/F
        # Every nonzero gap has Townsend's reciprocal, however small it is.
        # The reconstruction limit applies only to exact ties in this dtype.
        repeated = both_kept & (gap == 0)
        # Only discarded/discarded and repeated-retained pairs are removable.
        # A boundary tie remains a zero gap and signals a nonunique projection.
        safe_gap = torch.where(~active | repeated, torch.ones_like(gap), gap)
        # Townsend's off-diagonal block, (F ∘ (a − aᵀ)) S + S (F ∘ (b − bᵀ)).
        core = ((a - a.mT) * right + left * (b - b.mT)) / safe_gap
        # 1/s on retained values and 0 on discarded ones. The first where
        # swaps discarded values for 1 before the reciprocal, so a zero
        # discarded singular value never produces inf to be masked later.
        inverse = torch.where(kept, singular, torch.ones_like(singular)).reciprocal()
        inverse = torch.where(kept, inverse, torch.zeros_like(inverse))
        # Exact ties among retained values replace the 0/0 entry by its finite
        # limit, which is valid for losses on the reconstructed matrix.
        limit = 0.5 * (a * inverse.unsqueeze(-2) + b.mT * inverse.unsqueeze(-1))
        core = torch.where(repeated, limit, torch.where(active, core, 0.0))
        # The diagonal of the core is the singular-value gradient itself.
        diagonal = torch.eye(
            singular.shape[-1], dtype=torch.bool, device=singular.device
        )
        core = torch.where(diagonal, torch.diag_embed(gs), core)
        grad = u @ core @ vt
        # The last two terms carry the parts of Ū and V̄ that leave the
        # spans of U and V, (I − U Uᵀ) Ū S⁻¹ Vᵀ and its V counterpart. They
        # matter whenever m or n exceeds k.
        grad = grad + ((gu - u @ a) * inverse.unsqueeze(-2)) @ vt
        grad = grad + (u * inverse.unsqueeze(-2)) @ (gv - v @ b).mT
        return grad, None


class FixedRankManifold:
    """Fixed-rank (m, n) matrix manifold of rank r, on batched SVD factors.

    All operations expect a leading batch dim B and return tensors with a
    leading batch dim B. Single-point use: pass B = 1 (the manifold does not
    silently broadcast scalars to batches; the caller owns the batch axis).

    Args:
      m, n: ambient dimensions
      r:    target rank, with 0 < r <= min(m, n)
      device, dtype: tensor placement and precision

    References:
      Vandereycken (2013), §2.

    Example:
      >>> mfd = FixedRankManifold(m=5, n=7, r=3)
      >>> pt = mfd.random_point(batch_size=4)
      >>> mfd.dense(pt).shape
      torch.Size([4, 5, 7])
    """

    def __init__(self, m: int, n: int, r: int,
                 device: str | torch.device = "cpu",
                 dtype: torch.dtype = torch.float64,
                 retraction_mode: Literal["auto", "exact", "randomized"] = "auto"):
        """Args (in addition to m/n/r/device/dtype):

          retraction_mode: how to compute the SVD inside `retraction`.
            - "auto" (default): pick "randomized" when `r / min(m, n)` is
              at or below `RETRACTION_RANDOMIZED_THRESHOLD`, else "exact".
              On the CPU baseline that is an 8-37× speedup for small-rank
              truncations (notes/magic_numbers.md).
            - "exact": always full SVD then slice (Eckart-Young exact).
              Slower but deterministic; pick this when downstream code
              depends on bit-for-bit reproducibility.
            - "randomized": always Halko-Martinsson-Tropp via the
              library's `truncated_svd(mode="randomized")`, which
              oversamples and runs subspace iterations. Non-deterministic,
              since the random projection is drawn from the global RNG on
              every call, and approximate to ~σ_{r+1}-precision.
        """
        if r <= 0 or r > min(m, n):
            raise ValueError(
                f"rank r={r} must satisfy 0 < r <= min(m={m}, n={n})"
            )
        if retraction_mode not in {"auto", "exact", "randomized"}:
            raise ValueError(
                f"retraction_mode must be 'auto'/'exact'/'randomized', "
                f"got {retraction_mode!r}"
            )
        self.m = m
        self.n = n
        self.r = r
        self.device = torch.device(device)
        self.dtype = dtype
        self.retraction_mode = retraction_mode

    # ----------------------------------------------------------------
    # Manifold dimension (informational)
    # ----------------------------------------------------------------

    @property
    def dim(self) -> int:
        """Manifold dimension, r·(m + n − r). Vandereycken (2013), §2.1."""
        return self.r * (self.m + self.n - self.r)


    # ----------------------------------------------------------------
    # Construction
    # ----------------------------------------------------------------

    def random_point(
        self,
        batch_size: int = 1,
        generator: torch.Generator | None = None,
    ) -> FixedRankPoint:
        """Sample B random points on the manifold.

        Construction: U from QR of a Gaussian random matrix (Haar-distributed
        on the Stiefel manifold St(m, r) by the Mezzadri (2007) construction);
        singular values uniform in [0, 1) from `torch.rand`, sorted descending
        so the SVD-form invariant (S decreasing) holds; V analogous.

        Note: the singular-value distribution here is uniform, NOT the
        Marchenko–Pastur or any "natural" prior on M_r; it's a convenience
        for tests/initialization, not a statistically motivated draw.

        References:
          Mezzadri, F. (2007). How to generate random matrices from the
            classical compact groups. Notices of the AMS, 54(5):592–604.

        Returns:
          (U, S, Vt) with shapes (B, m, r), (B, r), (B, r, n).
        """
        if batch_size < 0:
            raise ValueError(f"batch_size must be >= 0, got {batch_size}")
        B, m, n, r = batch_size, self.m, self.n, self.r
        A = torch.randn(
            B,
            m,
            r,
            generator=generator,
            device=generator.device if generator is not None else self.device,
            dtype=self.dtype,
        ).to(self.device)
        # QR alone is not Haar: LAPACK's sign convention on diag(R) biases Q.
        # Flipping each column of Q by the sign of the matching R diagonal
        # entry is Mezzadri's fix and makes the draw uniform on the Stiefel
        # manifold.
        U, R = torch.linalg.qr(A)
        U = U * torch.sign(R.diagonal(dim1=-2, dim2=-1)).unsqueeze(-2)
        C = torch.randn(
            B,
            n,
            r,
            generator=generator,
            device=generator.device if generator is not None else self.device,
            dtype=self.dtype,
        ).to(self.device)
        V, R = torch.linalg.qr(C)
        V = V * torch.sign(R.diagonal(dim1=-2, dim2=-1)).unsqueeze(-2)
        Vt = V.mT.contiguous()
        S_raw = torch.rand(
            B,
            r,
            generator=generator,
            device=generator.device if generator is not None else self.device,
            dtype=self.dtype,
        ).to(self.device)
        S, _ = torch.sort(S_raw, dim=-1, descending=True)
        return U, S, Vt

    # ----------------------------------------------------------------
    # Embedding
    # ----------------------------------------------------------------

    def dense(self, point: FixedRankPoint) -> torch.Tensor:
        """Reconstruct ambient dense matrices: M = U @ diag(S) @ Vt.

        Args:
          point: (U, S, Vt) with shapes (B, m, r), (B, r), (B, r, n).
        Returns:
          M of shape (B, m, n).

        References:
          Vandereycken (2013), Low-Rank Matrix Completion by Riemannian
          Optimization, §2, embedded matrix representation and metric.
        """
        U, S, Vt = point
        # Broadcast scale: U (B, m, r) times diag(S) along last dim
        US = U * S.unsqueeze(dim=-2)
        return torch.bmm(US, Vt)

    # ----------------------------------------------------------------
    # Tangent operations
    # ----------------------------------------------------------------

    def projection(
        self, point: FixedRankPoint, Z: torch.Tensor
    ) -> torch.Tensor:
        """Project ambient direction Z onto the tangent space at point.

        The tangent space at M = U S Vt is

            T_M M_r = { U Mp Vt + Up Vt + U Vpt :
                        Mp ∈ R^{r×r}, Up ⊥ U, Vpt ⊥ V^T }

        and the orthogonal projector onto it (in Frobenius inner product)
        admits the closed form

            P_T(Z) = U U^T Z + Z V V^T − U U^T Z V V^T.

        Idempotent: P_T(P_T(Z)) = P_T(Z) by construction.

        References:
          Vandereycken (2013), eq. 2.5.

        Args:
          point: (U, S, Vt); S is unused for projection, included for API
            consistency.
          Z: ambient direction, shape (B, m, n).
        Returns:
          Projected tangent in ambient form, shape (B, m, n).
        """
        U, _, Vt = point
        V = Vt.mT  # (B, n, r)
        Ut_Z = torch.bmm(U.mT, Z)              # (B, r, n)
        UUt_Z = torch.bmm(U, Ut_Z)              # (B, m, n)
        Z_V = torch.bmm(Z, V)                   # (B, m, r)
        Z_VVt = torch.bmm(Z_V, Vt)              # (B, m, n)
        Ut_Z_V = torch.bmm(Ut_Z, V)             # (B, r, r)
        UUt_Z_VVt = torch.bmm(U, torch.bmm(Ut_Z_V, Vt))  # (B, m, n)
        return UUt_Z + Z_VVt - UUt_Z_VVt

    def inner(
        self,
        point: FixedRankPoint,
        tangent_a: torch.Tensor,
        tangent_b: torch.Tensor,
    ) -> torch.Tensor:
        """Riemannian inner product of two tangents at point.

        For the fixed-rank manifold embedded in R^{m×n}, the induced metric
        is just the ambient Frobenius inner product on tangent vectors
        represented in ambient form.

        References:
          Vandereycken (2013), §2.2; induced metric.

        Args:
          point: unused (induced metric is point-independent in ambient form;
            included for API consistency with manifolds whose metric depends
            on the point).
          tangent_a, tangent_b: tangent vectors in ambient form, (B, m, n).
        Returns:
          (B,) tensor of inner products.
        """
        del point  # induced ambient metric does not depend on the base point
        return (tangent_a * tangent_b).sum(dim=(-2, -1))

    def norm(
        self, point: FixedRankPoint, tangent: torch.Tensor
    ) -> torch.Tensor:
        """Riemannian norm sqrt(<v,v>) of a tangent vector. (B,)-shaped output.

        References:
          Vandereycken (2013), Low-Rank Matrix Completion by Riemannian
          Optimization, §2, embedded matrix representation and metric.
        """
        return torch.linalg.vector_norm(tangent, dim=(-2, -1))

    def egrad_to_rgrad(
        self, point: FixedRankPoint, egrad: torch.Tensor,
    ) -> torch.Tensor:
        """Project an ambient gradient under the embedded Frobenius metric.

        Args:
          point: factor triple (U, S, Vt).
          egrad: (B, m, n) Euclidean gradient.
        Returns:
          (B, m, n) metric gradient in tangent coordinates.

        References:
          Absil, Mahony, Sepulchre (2008), §3.6; Vandereycken (2013), eq. 2.5.
        """
        return self.projection(point, egrad)

    # ----------------------------------------------------------------
    # Retraction
    # ----------------------------------------------------------------

    def retraction(
        self, point: FixedRankPoint, tangent: torch.Tensor
    ) -> FixedRankPoint:
        """Move along tangent and project back onto the manifold via SVD.

        The standard projection-based retraction:

            R_M(ξ) = argmin_{Y ∈ M_r} ‖ (M + ξ) − Y ‖_F  =  SVD_r(M + ξ)

        where SVD_r truncates to the top r singular triples. This is a
        first-order retraction (Absil et al. 2008, §4.1).

        Performance: full SVD scales O(m·n·min(m, n)). For r ≪ min(m, n)
        we switch to the library's randomized `truncated_svd`
        (Halko-Martinsson-Tropp), which scales O(m·n·r); on our CPU
        baseline the speedup is 8-37× at common ranks. The threshold is governed by
        `retraction_mode` (default "auto"); see `__init__` docstring.

        Differentiated calls use exact SVD, regardless of mode. The backward
        differentiates the reconstructed rank-r matrix through repeated
        retained or discarded values. Individual factors have no unique
        derivative at repeated retained values. A tie across the truncation
        boundary makes the rank-r projection itself nondifferentiable.
        Only first derivatives are implemented by the custom backward.

        References:
          Absil, Mahony, Sepulchre (2008), §4.1; retractions.
          Vandereycken (2013), eq. 2.7.
          Halko, Martinsson, Tropp (2011); randomized SVD.

        Args:
          point: (U, S, Vt); used only to call dense().
          tangent: ambient (B, m, n).
        Returns:
          (U', S', Vt') with shapes (B, m, r), (B, r), (B, r, n).
        """
        M_plus = self.dense(point) + tangent  # (B, m, n)
        r = self.r
        # Anything autograd will differentiate goes through the exact SVD
        # with the custom backward. The randomized path has no derivative of
        # its own and the plain svd backward has poles at repeated values.
        if M_plus.requires_grad and torch.is_grad_enabled():
            return _TruncatedSVD.apply(M_plus, r)
        mode = self.retraction_mode
        if mode == "auto":
            min_dim = min(M_plus.shape[-2], M_plus.shape[-1])
            mode = "randomized" if r <= RETRACTION_RANDOMIZED_THRESHOLD * min_dim \
                                 else "exact"
        if mode == "randomized":
            # Use our `truncated_svd` (HMT with oversample=5, n_iter=2 by
            # its cataloged defaults) rather than `torch.svd_lowrank`
            # directly. svd_lowrank with q=r has no oversampling and gives
            # ~40% relative error on retraction inputs, which is too much
            # for an optimizer step.
            return truncated_svd(M_plus, r=r, mode="randomized")
        U_full, S_full, Vt_full = torch.linalg.svd(M_plus, full_matrices=False)
        U_r = U_full[..., :, :r].contiguous()
        S_r = S_full[..., :r].contiguous()
        Vt_r = Vt_full[..., :r, :].contiguous()
        return U_r, S_r, Vt_r
