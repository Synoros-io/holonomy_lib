# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Synoros

"""Hodge Laplacians + Betti numbers on simplicial complexes.

The k-th Hodge Laplacian on a simplicial complex `K` is

    L_k = ∂_{k+1} ∂_{k+1}^T + ∂_k^T ∂_k,

acting on the space of k-chains `C_k(K) ≅ ℝ^{n_k}`. It generalizes
the graph Laplacian (which is `L_0` for a 1-complex) to higher
dimensions and connects spectral analysis to topology via the Hodge
decomposition:

    ker(L_k) ≅ H_k(K),   so   β_k = dim(ker(L_k)) = n_k − rank(L_k).

For a topological invariant of `K`, `β_k` is the k-th Betti number:
`β_0` counts connected components, `β_1` counts independent 1-cycles
(holes), `β_2` counts independent voids, and so on.

Two paths share the same public function `hodge_laplacian(complex, k)`:

  Dense complex (`DenseSimplicialComplex`)
    Boundary operators returned as dense `(B, n_{k-1}_max, n_k_max)`
    tensors. The Hodge Laplacian is built via batched dense matmuls
    and returned as `(B, n_k_max, n_k_max)`. Boundary and Laplacian
    storage both scale with the padded batch dimensions.

  Sparse complex (`SparseSimplicialComplex`)
    Boundary operators returned as `torch.sparse_csc_tensor
    (n_{k-1}, n_k)`. The Hodge Laplacian is built via sparse matmul
    and returned as a dense `(n_k, n_k)` tensor. This avoids dense
    boundary storage but still requires space for the final dense L_k.

`betti_numbers(complex, max_dim)` returns the Betti numbers for dims
0..max_dim from float64 boundary ranks. Rank tolerances use each
member's real simplex counts, independently of batch padding.

References:
  Eckmann, B. (1944). Harmonische Funktionen und Randwertaufgaben in
    einem Komplex. Commentarii Mathematici Helvetici 17:240–255.
    The original Hodge-theoretic interpretation on simplicial
    complexes.
  Lim, L.-H. (2020). Hodge Laplacians on graphs. SIAM Review
    62(3):685–715. Modern survey.
  Schaub, M. T., Benson, A. R., Horn, P., Lippner, G., Jadbabaie, A.
    (2020). Random walks on simplicial complexes and the normalized
    Hodge 1-Laplacian. SIAM Review 62(2):353–391.
  Ribando-Gros, E., Wang, R., Chen, J., Tong, Y., Wei, G.-W. (2024).
    Combinatorial and Hodge Laplacians: Similarities and differences.
    SIAM Review 66(3):575–601.
"""

from __future__ import annotations

import torch

from holonomy_lib.simplicial import (
    DenseSimplicialComplex,
    SparseSimplicialComplex,
)


def hodge_laplacian(
    complex: DenseSimplicialComplex | SparseSimplicialComplex,
    k: int,
) -> torch.Tensor:
    """Hodge Laplacian `L_k = ∂_{k+1} ∂_{k+1}^T + ∂_k^T ∂_k`.

    Args:
      complex: a `DenseSimplicialComplex` (batched) or
        `SparseSimplicialComplex` (single-instance).
      k: simplex dimension. Must be in `[0, complex.max_dim]`.

    Returns:
      Dense complex  → `(B, n_k_max, n_k_max)` Tensor.
      Sparse complex → `(n_k, n_k)` dense Tensor. Boundary products
        are computed sparsely, then materialized for this return contract.

    References:
      Eckmann (1944). Harmonische Funktionen und Randwertaufgaben in
        einem Komplex. Comment. Math. Helv. 17:240–255.
      Lim (2020). Hodge Laplacians on graphs. SIAM Review 62:685–715.
    """
    if k < 0:
        raise ValueError(f"k must be >= 0, got {k}")
    if isinstance(complex, DenseSimplicialComplex):
        return _hodge_dense(complex, k)
    if isinstance(complex, SparseSimplicialComplex):
        return _hodge_sparse(complex, k)
    raise TypeError(
        f"complex must be a Dense or Sparse SimplicialComplex; "
        f"got {type(complex).__name__}"
    )


def betti_numbers(
    complex: DenseSimplicialComplex | SparseSimplicialComplex,
    max_dim: int,
    threshold: float | None = None,
) -> torch.Tensor:
    """Betti numbers `(β_0, β_1, …, β_max_dim)` for a simplicial complex.

    `β_k = n_k - rank(∂_k) - rank(∂_{k+1})`. Boundary entries are
    exactly 0 or ±1, so ranks are computed in float64 regardless of
    the complex's storage dtype. This avoids squaring the conditioning
    by forming a Hodge Laplacian.

    Args:
      complex: a `DenseSimplicialComplex` or `SparseSimplicialComplex`.
      max_dim: highest dim to return Betti numbers for. Must satisfy
        `0 ≤ max_dim ≤ complex.max_dim`.
      threshold: optional positive relative rank tolerance. With `None`,
        each boundary uses `max(n_{k-1}, n_k) * eps(float64)`, with
        real simplex counts for that batch member. An explicit value
        applies to boundary singular values, with zero absolute tolerance.

    Returns:
      Dense complex  → `(B, max_dim + 1)` Tensor of int64 Betti numbers.
      Sparse complex → `(max_dim + 1,)` Tensor.

    Notes:
      Dense boundaries retain zero padding for batched SVD; it adds only
      zero singular values and does not enter the per-member tolerance.
      Sparse boundaries are materialized for the rank computation.

    References:
      Eckmann (1944); kernel of L_k equals harmonic k-forms.
      Lim (2020), §4.
    """
    if max_dim < 0:
        raise ValueError(f"max_dim must be >= 0, got {max_dim}")
    if max_dim > complex.max_dim:
        raise ValueError(
            f"max_dim={max_dim} exceeds complex.max_dim={complex.max_dim}"
        )
    if threshold is not None and threshold <= 0:
        raise ValueError(f"threshold must be > 0, got {threshold}")

    if isinstance(complex, DenseSimplicialComplex):
        B = complex.batch_size
        ranks = torch.zeros(B, max_dim + 2, dtype=torch.int64,
                            device=complex.device)
        for k in range(1, min(max_dim + 1, complex.max_dim) + 1):
            boundary = complex.boundary(k).to(torch.float64)
            relative_tol: float | torch.Tensor | None = threshold
            if relative_tol is None:
                real_size = torch.maximum(
                    complex.n_simplices(k - 1), complex.n_simplices(k),
                )
                relative_tol = real_size.to(torch.float64) * torch.finfo(
                    torch.float64,
                ).eps
            ranks[:, k] = torch.linalg.matrix_rank(boundary, rtol=relative_tol)
        counts = torch.stack([
            complex.n_simplices(k) for k in range(max_dim + 1)
        ], dim=-1)
        return counts - ranks[:, :-1] - ranks[:, 1:]

    if isinstance(complex, SparseSimplicialComplex):
        ranks = torch.zeros(max_dim + 2, dtype=torch.int64, device=complex.device)
        for k in range(1, min(max_dim + 1, complex.max_dim) + 1):
            boundary = complex.boundary(k, dtype=torch.float64).to_dense()
            ranks[k] = torch.linalg.matrix_rank(boundary, rtol=threshold)
        counts = torch.tensor([
            complex.n_simplices(k) for k in range(max_dim + 1)
        ], dtype=torch.int64, device=complex.device)
        return counts - ranks[:-1] - ranks[1:]

    raise TypeError(
        f"complex must be a Dense or Sparse SimplicialComplex; "
        f"got {type(complex).__name__}"
    )


# ============================================================
# Internal — dispatched paths
# ============================================================


def _hodge_dense(
    complex: DenseSimplicialComplex, k: int,
) -> torch.Tensor:
    """L_k = ∂_{k+1} ∂_{k+1}^T + ∂_k^T ∂_k on the dense path."""
    if k > complex.max_dim:
        raise ValueError(
            f"k={k} exceeds complex.max_dim={complex.max_dim}"
        )

    B = complex.batch_size
    n_k_max = (
        complex.simplices_by_dim[k].shape[1]
        if k in complex.simplices_by_dim else 0
    )
    L = torch.zeros(B, n_k_max, n_k_max,
                     device=complex.device, dtype=complex.dtype)

    # ∂_k^T ∂_k term — zero for k=0 (no (-1)-simplices).
    if k > 0:
        d_k = complex.boundary(k)                    # (B, n_{k-1}, n_k)
        L = L + torch.matmul(d_k.mT, d_k)

    # ∂_{k+1} ∂_{k+1}^T term — zero if (k+1)-simplices don't exist.
    if (k + 1) in complex.simplices_by_dim:
        d_kp1 = complex.boundary(k + 1)              # (B, n_k, n_{k+1})
        L = L + torch.matmul(d_kp1, d_kp1.mT)

    return L


def _hodge_sparse(
    complex: SparseSimplicialComplex, k: int,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """L_k on the sparse path; returns dense (n_k, n_k).

    Boundaries and their products stay sparse until the final product
    is materialized, because the public return contract is dense.

    `SparseSimplicialComplex` doesn't carry a float dtype attribute,
    so we accept one explicitly (default `torch.float64`). Boundary
    operators are constructed at this dtype to keep the L_k matmul
    type-consistent.
    """
    if k > complex.max_dim:
        raise ValueError(
            f"k={k} exceeds complex.max_dim={complex.max_dim}"
        )

    n_k = complex.n_simplices(k)
    device = complex.device
    L = torch.zeros(n_k, n_k, device=device, dtype=dtype)

    if k > 0:
        d_k = complex.boundary(k, dtype=dtype)       # sparse (n_{k-1}, n_k)
        d_k_coo = d_k.to_sparse_coo().coalesce()
        L = L + torch.sparse.mm(d_k_coo.mT, d_k_coo).to_dense()

    if (k + 1) in complex.simplices_by_dim:
        d_kp1 = complex.boundary(k + 1, dtype=dtype).to_sparse_coo().coalesce()
        L = L + torch.sparse.mm(d_kp1, d_kp1.mT).to_dense()

    return L
