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
dimension B is required for all operations — pass B = 1 for single-point use.

Tangent vectors are stored in *ambient form* — i.e. as (B, m, n) matrices
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

from typing import Literal

import torch

from holonomy_lib.algebra.linear import truncated_svd
from holonomy_lib.provenance import register_provenance_class, with_provenance

# A batched manifold point is the triple (U, S, Vt) of stacked tensors.
# Shapes: U (B, m, r), S (B, r), Vt (B, r, n).
FixedRankPoint = tuple[torch.Tensor, torch.Tensor, torch.Tensor]

# Threshold above which exact SVD is competitive with randomized for
# the retraction's top-r truncation. Empirical (see notes/benchmark_
# baseline.md): at r/min(m,n) ≥ 0.25 the exact-SVD constant factor
# wins back the gap, and accuracy / determinism matter more for
# research workloads. Below this ratio, randomized SVD (HMT 2011) is
# strictly faster — 37× at 1024×1024, r=32 in our CPU baseline.
# Cataloged as `fixed_rank_randomized_threshold`.
RETRACTION_RANDOMIZED_THRESHOLD: float = 0.25


@register_provenance_class("FixedRankManifold")
class FixedRankManifold:
    """Fixed-rank (m, n) matrix manifold of rank r, GPU-native + batched-first.

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
              below `RETRACTION_RANDOMIZED_THRESHOLD`, else "exact".
              Empirically a 5-37× speedup on tall-and-thin truncations.
            - "exact": always full SVD then slice (Eckart-Young exact).
              Slower but deterministic; pick this when downstream code
              depends on bit-for-bit reproducibility.
            - "randomized": always Halko-Martinsson-Tropp via
              `torch.svd_lowrank`. Non-deterministic — the random
              projection differs across calls — and approximate to
              ~σ_{r+1}-precision.
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

    def _provenance_signature(self) -> dict:
        """Deterministic canonical form used by `@with_provenance` to
        hash the bound `self` of decorated methods. Without this, the
        provenance hex would embed the manifold's Python `id()`, making
        the same call across processes produce different hexes.
        """
        return {
            "class": "FixedRankManifold",
            "m": self.m,
            "n": self.n,
            "r": self.r,
            "device": str(self.device),
            "dtype": str(self.dtype),
            "retraction_mode": self.retraction_mode,
        }

    @classmethod
    def _from_signature(cls, sig: dict) -> FixedRankManifold:
        """Inverse of `_provenance_signature` — reconstruct an instance
        with the same parameters. Used by `ProvenanceRegistry.replay`
        to rebuild the bound `self` of recorded class-method calls.
        """
        # dtype was stored as e.g. "torch.float64"; map back via attr lookup.
        dtype_name = sig["dtype"].split(".")[-1]
        return cls(
            m=sig["m"],
            n=sig["n"],
            r=sig["r"],
            device=sig["device"],
            dtype=getattr(torch, dtype_name),
            retraction_mode=sig["retraction_mode"],
        )

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
        singular values uniform in (0, 1), sorted descending so the SVD-form
        invariant (S decreasing) holds; V analogous.

        Note: the singular-value distribution here is uniform, NOT the
        Marchenko–Pastur or any "natural" prior on M_r — it's a convenience
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
        A = torch.randn(B, m, r, generator=generator,
                        device=self.device, dtype=self.dtype)
        U, _ = torch.linalg.qr(A)
        C = torch.randn(B, n, r, generator=generator,
                        device=self.device, dtype=self.dtype)
        V, _ = torch.linalg.qr(C)
        Vt = V.mT.contiguous()
        S_raw = torch.rand(B, r, generator=generator,
                           device=self.device, dtype=self.dtype)
        S, _ = torch.sort(S_raw, dim=-1, descending=True)
        return U, S, Vt

    # ----------------------------------------------------------------
    # Embedding
    # ----------------------------------------------------------------

    @with_provenance(
        "holonomy_lib.manifolds.FixedRankManifold.dense", op_version="0.1",
    )
    def dense(self, point: FixedRankPoint) -> torch.Tensor:
        """Reconstruct ambient dense matrices: M = U @ diag(S) @ Vt.

        Args:
          point: (U, S, Vt) with shapes (B, m, r), (B, r), (B, r, n).
        Returns:
          M of shape (B, m, n).
        """
        U, S, Vt = point
        # Broadcast scale: U (B, m, r) times diag(S) along last dim
        US = U * S.unsqueeze(dim=-2)
        return torch.bmm(US, Vt)

    # ----------------------------------------------------------------
    # Tangent operations
    # ----------------------------------------------------------------

    @with_provenance(
        "holonomy_lib.manifolds.FixedRankManifold.projection", op_version="0.1",
    )
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
          point: (U, S, Vt) — S is unused for projection, included for API
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

    @with_provenance(
        "holonomy_lib.manifolds.FixedRankManifold.inner", op_version="0.1",
    )
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
          Vandereycken (2013), §2.2 — induced metric.

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

    @with_provenance(
        "holonomy_lib.manifolds.FixedRankManifold.norm", op_version="0.1",
    )
    def norm(
        self, point: FixedRankPoint, tangent: torch.Tensor
    ) -> torch.Tensor:
        """Riemannian norm sqrt(<v,v>) of a tangent vector. (B,)-shaped output."""
        return torch.sqrt(self.inner(point, tangent, tangent))

    # ----------------------------------------------------------------
    # Retraction
    # ----------------------------------------------------------------

    @with_provenance(
        "holonomy_lib.manifolds.FixedRankManifold.retraction", op_version="0.1",
    )
    def retraction(
        self, point: FixedRankPoint, tangent: torch.Tensor
    ) -> FixedRankPoint:
        """Move along tangent and project back onto the manifold via SVD.

        The standard projection-based retraction:

            R_M(ξ) = argmin_{Y ∈ M_r} ‖ (M + ξ) − Y ‖_F  =  SVD_r(M + ξ)

        where SVD_r truncates to the top r singular triples. This is a
        first-order retraction (Absil et al. 2008, §4.1).

        Performance: full SVD scales O(m·n·min(m, n)). For r ≪ min(m, n)
        we switch to `torch.svd_lowrank` (Halko-Martinsson-Tropp) which
        scales O(m·n·r); on our CPU baseline the speedup is 8-37× at
        common ranks. The threshold is governed by
        `retraction_mode` (default "auto"); see `__init__` docstring.

        Numerically may produce degenerate singular values when the
        truncation rank coincides with a spectral gap — caller can detect
        S[r-1] ≈ S[r] and halve the step.

        References:
          Absil, Mahony, Sepulchre (2008), §4.1 — retractions.
          Vandereycken (2013), eq. 2.7.
          Halko, Martinsson, Tropp (2011) — randomized SVD.

        Args:
          point: (U, S, Vt) — used only to call dense().
          tangent: ambient (B, m, n).
        Returns:
          (U', S', Vt') with shapes (B, m, r), (B, r), (B, r, n).
        """
        M_plus = self.dense(point) + tangent  # (B, m, n)
        r = self.r
        mode = self.retraction_mode
        if mode == "auto":
            min_dim = min(M_plus.shape[-2], M_plus.shape[-1])
            mode = "randomized" if r <= RETRACTION_RANDOMIZED_THRESHOLD * min_dim \
                                 else "exact"
        if mode == "randomized":
            # Use our `truncated_svd` (HMT with oversample=5, n_iter=2)
            # rather than `torch.svd_lowrank` directly. svd_lowrank with
            # q=r has NO oversampling and produces ~40% relative error
            # on retraction inputs — unacceptable for an optimizer step.
            # Our randomized path oversamples per Halko-Martinsson-Tropp.
            return truncated_svd(M_plus, r=r, mode="randomized")
        U_full, S_full, Vt_full = torch.linalg.svd(M_plus, full_matrices=False)
        U_r = U_full[..., :, :r].contiguous()
        S_r = S_full[..., :r].contiguous()
        Vt_r = Vt_full[..., :r, :].contiguous()
        return U_r, S_r, Vt_r
