# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Synoros

"""Tensor decompositions — HOSVD and the n-mode product machinery.

This module assumes the leading dimension of every tensor is a batch
dimension. So a tensor of order d is stored as a tensor of shape
`(B, n_1, n_2, ..., n_d)` where B is the batch size. The "tensor modes"
are axes 1 through d (axis 0 is batch).

Operations implemented:

  mode_product(T, A, axis)
      n-mode product T ×_axis A. Contracts `T`'s `axis`-th dimension with
      the last dimension of `A` (Kolda-Bader 2009 convention).

  mode_unfolding(T, axis)
      Matricize T by moving `axis` to position 1 and flattening the rest.
      Output shape `(B, n_axis, prod_of_other_modes)`.

  hosvd(T, ranks, mode)
      Truncated Higher-Order SVD (De Lathauwer-De Moor-Vandewalle 2000).
      Returns (core, factors). Each factor is the top-r_k left singular
      vectors of T's mode-k unfolding.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import torch

from holonomy_lib.algebra.linear import truncated_svd
from holonomy_lib.provenance import with_provenance


@with_provenance("holonomy_lib.tensor_calculus.mode_product", op_version="0.1")
def mode_product(
    T: torch.Tensor, A: torch.Tensor, axis: int
) -> torch.Tensor:
    """n-mode product T ×_axis A.

    Contracts axis `axis` of T (size n_axis) with the **last** axis of A
    (also size n_axis), producing a tensor where axis `axis` has size
    A.shape[-2] instead. Kolda-Bader (2009) §2.5 convention.

    Args:
      T: (B, n_1, ..., n_d).
      A: (B, j, n_axis) — last axis must match T's `axis`-th axis.
      axis: which axis of T to contract, in [1, T.ndim − 1] (axis 0 is batch).

    Returns:
      Tensor of shape (B, ..., j, ...), where size j appears at position `axis`.

    References:
      Kolda, T. G., Bader, B. W. (2009). Tensor decompositions and
        applications. SIAM Review 51(3):455–500, §2.5.
    """
    if axis <= 0 or axis >= T.ndim:
        raise ValueError(
            f"axis must be in [1, {T.ndim - 1}] (axis 0 is batch), got {axis}"
        )
    if A.ndim != 3:
        raise ValueError(
            f"A must be 3-D (B, j, n_axis); got A.shape={tuple(A.shape)}"
        )
    if A.shape[0] != T.shape[0]:
        raise ValueError(
            f"batch sizes disagree: T has B={T.shape[0]}, A has B={A.shape[0]}"
        )
    if A.shape[-1] != T.shape[axis]:
        raise ValueError(
            f"size mismatch: A.shape[-1]={A.shape[-1]} must equal "
            f"T.shape[{axis}]={T.shape[axis]}"
        )

    B = T.shape[0]
    n_axis = T.shape[axis]
    j = A.shape[-2]

    # Move the contracted axis of T to the last position, then reshape
    # so that the contracted axis is the inner dim of a 2-D matmul.
    T_perm = T.movedim(axis, -1)              # (B, ..., n_axis)
    # Flatten the middle (non-batch non-contracted) dims:
    T_flat = T_perm.reshape(B, -1, n_axis)    # (B, K, n_axis), K = prod others
    # Multiply: (B, K, n_axis) @ (B, n_axis, j) → (B, K, j)
    out_flat = torch.bmm(T_flat, A.mT)        # uses A.mT = (B, n_axis, j)
    # Restore the middle dims:
    middle = list(T_perm.shape[1:-1])
    out_perm = out_flat.reshape(B, *middle, j)
    # Move the new (size-j) axis from last back to position `axis`:
    return out_perm.movedim(-1, axis)


@with_provenance("holonomy_lib.tensor_calculus.mode_unfolding", op_version="0.1")
def mode_unfolding(T: torch.Tensor, axis: int) -> torch.Tensor:
    """Matricize T along `axis`: move `axis` to position 1, flatten the rest.

    Args:
      T: (B, n_1, ..., n_d).
      axis: in [1, T.ndim − 1].

    Returns:
      (B, n_axis, prod of other tensor modes).

    References:
      Kolda, T. G., Bader, B. W. (2009), §2.4 — mode-n matricization.
    """
    if axis <= 0 or axis >= T.ndim:
        raise ValueError(
            f"axis must be in [1, {T.ndim - 1}] (axis 0 is batch), got {axis}"
        )
    B = T.shape[0]
    n_axis = T.shape[axis]
    T_perm = T.movedim(axis, 1)  # (B, n_axis, ...)
    return T_perm.reshape(B, n_axis, -1)


@with_provenance("holonomy_lib.tensor_calculus.hosvd", op_version="0.1")
def hosvd(
    T: torch.Tensor,
    ranks: Sequence[int],
    mode: Literal["exact", "randomized"] = "exact",
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Truncated Higher-Order SVD (HOSVD).

    For a tensor T ∈ R^{B × n_1 × ... × n_d}, returns factors
    {U_k}_{k=1..d} and a core G such that

        T ≈ G ×_1 U_1 ×_2 U_2 ×_3 ... ×_d U_d

    where U_k has shape (B, n_k, r_k) with orthonormal columns and G
    has shape (B, r_1, ..., r_d). The factors are taken as the top-r_k
    left singular vectors of T's mode-k unfolding; the core is then

        G = T ×_1 U_1ᵀ ×_2 U_2ᵀ ×_3 ... ×_d U_dᵀ.

    Truncated HOSVD is in general **not** the best rank-(r_1,...,r_d)
    Tucker approximation (that requires HOOI / ALS), but it is a
    quasi-optimal initializer and is itself a useful approximation in
    practice; see Vannieuwenhoven et al. (2012) for analysis.

    Args:
      T: (B, n_1, ..., n_d). Must have ndim ≥ 2 (one batch dim + ≥1 mode).
      ranks: tuple of length d giving the truncation rank per tensor mode.
      mode: "exact" or "randomized" — passed through to `truncated_svd`.
      generator: PyTorch RNG, forwarded to randomized SVDs.

    Returns:
      core: (B, r_1, ..., r_d).
      factors: list of d tensors, factors[k-1] of shape (B, n_k, r_k).

    References:
      De Lathauwer, L., De Moor, B., Vandewalle, J. (2000). A multilinear
        singular value decomposition. SIAM J. Matrix Anal. Appl.
        21(4):1253–1278.
      Vannieuwenhoven, N., Vandebril, R., Meerbergen, K. (2012). A new
        truncation strategy for the higher-order singular value
        decomposition. SIAM J. Sci. Comput. 34(2):A1027–A1052.
      Kolda, T. G., Bader, B. W. (2009), §4.2 — Tucker decomposition.
    """
    if T.ndim < 2:
        raise ValueError(
            f"T must have ≥1 batch dim and ≥1 tensor mode (ndim ≥ 2); "
            f"got T.shape={tuple(T.shape)}"
        )
    d = T.ndim - 1  # number of tensor modes
    if len(ranks) != d:
        raise ValueError(
            f"len(ranks)={len(ranks)} must equal tensor order d={d}"
        )
    for k, r_k in enumerate(ranks):
        n_k = T.shape[k + 1]
        if r_k <= 0 or r_k > n_k:
            raise ValueError(
                f"ranks[{k}]={r_k} must be in (0, n_{k + 1}={n_k}]"
            )

    # Step 1: factors. For each mode k, take top-r_k left singular vectors
    # of the mode-k unfolding.
    factors: list[torch.Tensor] = []
    for k in range(1, d + 1):
        T_k = mode_unfolding(T, axis=k)               # (B, n_k, prod_others)
        U_k, _, _ = truncated_svd(
            T_k, r=ranks[k - 1], mode=mode, generator=generator,
        )
        factors.append(U_k)

    # Step 2: core G = T ×_1 U_1ᵀ ×_2 U_2ᵀ × ... ×_d U_dᵀ.
    # Our `mode_product(T, A, axis)` contracts T's axis with A's last dim.
    # We want to project: T's axis_k of size n_k → r_k. Pass A = factors[k-1].mT
    # so A has shape (B, r_k, n_k) and the result has r_k at axis_k.
    core = T
    for k in range(1, d + 1):
        core = mode_product(core, factors[k - 1].mT, axis=k)

    return core, factors
