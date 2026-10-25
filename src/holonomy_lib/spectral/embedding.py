# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Synoros

"""Spectral embedding via Laplacian eigenmaps, batched-first, GPU-native.

The Laplacian eigenmap (Belkin-Niyogi 2003) embeds a graph into R^k by
taking the bottom-k eigenvectors of a graph Laplacian. Each row of the
resulting (n, k) matrix is the k-dimensional coordinate of one node.

This module provides a single primitive

    laplacian_eigenmaps(A, k, laplacian_type)

that returns the bottom-k eigenvalue/eigenvector pairs of the chosen
Laplacian, with the eigenvectors sorted by eigenvalue ascending. We
do **not** automatically drop the trivial null eigenvector; that
choice depends on the use case (it should be dropped for connected
unsigned graphs but kept for signed Laplacians, where the null
eigenvector encodes the balance partition; Kunegis 2010). The caller
decides.

This routine uses dense `torch.linalg.eigh`, with O(n³) cost. For sparse
operators, use `holonomy_lib.algebra.lanczos_eigsh` or `eigsh_sparse_sa`.

References:
  Belkin, M., Niyogi, P. (2003). Laplacian eigenmaps for dimensionality
    reduction and data representation. Neural Computation 15(6):1373–1396.
  von Luxburg, U. (2007). A tutorial on spectral clustering. Statistics
    and Computing 17(4):395–416, discussion of which Laplacian to use.
  Coifman, R. R., Lafon, S. (2006). Diffusion maps. Applied and
    Computational Harmonic Analysis 21(1):5–30, related embedding.
"""

from __future__ import annotations

from typing import Literal

import torch

from holonomy_lib.spectral import laplacian as _L

LaplacianType = Literal[
    "combinatorial", "symmetric_normalized", "random_walk", "signed",
]


def laplacian_eigenmaps(
    A: torch.Tensor,
    k: int,
    laplacian_type: LaplacianType = "symmetric_normalized",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Bottom-k Laplacian eigenmap embedding of a (batched) graph.

    Args:
      A: (B, n, n) weighted adjacency. Symmetric for the supported
        Laplacian variants (combinatorial, symmetric_normalized,
        random_walk, signed).
      k: number of eigenvectors to return. Must satisfy 1 ≤ k ≤ n.
      laplacian_type: which Laplacian to diagonalize.
        - "symmetric_normalized" (default). Belkin-Niyogi (2003) solve
          the generalized problem L f = λ D f, which is the
          "random_walk" option below; L_sym has the same eigenvalues.
        - "combinatorial" (unnormalized)
        - "random_walk" (non-symmetric; the returned eigenvectors are
          the L_rw eigenvectors, obtained by D^{−1/2}-scaling the
          L_sym eigenvectors so they diagonalize L_rw on the right;
          the scaling is identity on isolated-node coordinates)
        - "signed" (Kunegis 2010)

    Returns:
      eigenvalues: (B, k), sorted ascending.
      eigenvectors: (B, n, k), column j is the eigenvector for
        eigenvalues[..., j].

    Notes:
      The trivial null eigenvector (eigenvalue ≈ 0 for connected
      unsigned graphs) is NOT automatically dropped. Drop it explicitly
      with `eigvecs[..., 1:]` when you want the embedding to exclude
      it. For signed Laplacians, do NOT drop it; it encodes the
      balance partition (Kunegis 2010, Thm 3.4).

    References:
      Belkin-Niyogi (2003), §2, Laplacian eigenmap.
      von Luxburg (2007), §5, practical considerations.
    """
    if A.ndim < 2 or A.shape[-1] != A.shape[-2]:
        raise ValueError(
            f"A must be (..., n, n); got A.shape={tuple(A.shape)}"
        )
    n = A.shape[-1]
    if k <= 0 or k > n:
        raise ValueError(f"k must satisfy 1 <= k <= n={n}, got k={k}")

    if laplacian_type == "combinatorial":
        L = _L.combinatorial(A)
        eigvals, eigvecs = torch.linalg.eigh(L)
    elif laplacian_type == "symmetric_normalized":
        L = _L.symmetric_normalized(A)
        eigvals, eigvecs = torch.linalg.eigh(L)
    elif laplacian_type == "signed":
        L = _L.signed(A)
        eigvals, eigvecs = torch.linalg.eigh(L)
    elif laplacian_type == "random_walk":
        # L_rw is similar (in the linear-algebra sense) to L_sym via
        # L_rw = D^{−1/2} L_sym D^{1/2}. So if u_i is the L_sym
        # eigenvector for λ_i, then v_i = D^{−1/2} u_i is the L_rw
        # right-eigenvector for the same λ_i. (Equivalently, v_i is the
        # solution to the generalized eigenproblem L v = λ D v.)
        # L_rw is not symmetric, so eigh cannot take it directly. Solving
        # the symmetric L_sym and mapping its eigenvectors over keeps the
        # real, sorted eigh output
        L_sym = _L.symmetric_normalized(A)
        eigvals, u = torch.linalg.eigh(L_sym)  # (B, n), (B, n, n)
        d = _L.degree(A, signed=False)
        # an isolated node has a diagonal entry of one in both L_rw and
        # L_sym. Using degree 1 there makes the similarity transform the
        # identity on that coordinate, so its eigenvector survives instead
        # of being multiplied by a zero (pseudoinverse) degree factor
        safe_degree = torch.where(d > 0, d, 1.0)
        d_inv_sqrt = safe_degree.rsqrt()
        eigvecs = u * d_inv_sqrt.unsqueeze(dim=-1)  # broadcast over columns
    else:
        raise ValueError(
            f"laplacian_type must be one of "
            f"'combinatorial', 'symmetric_normalized', 'random_walk', 'signed'; "
            f"got {laplacian_type!r}"
        )

    # eigh already returns ascending order, so the bottom k are the first k.
    # contiguous() so the caller gets owning tensors, not strided slices
    return eigvals[..., :k].contiguous(), eigvecs[..., :, :k].contiguous()
