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
    and returned as `(B, n_k_max, n_k_max)`. Use this path for small
    complexes (n_k ≤ 10⁴) where `eigvalsh` on the dense matrix is
    fast and batching helps amortize overhead.

  Sparse complex (`SparseSimplicialComplex`)
    Boundary operators returned as `torch.sparse_csc_tensor
    (n_{k-1}, n_k)`. The Hodge Laplacian is built via sparse matmul
    and returned as a dense `(n_k, n_k)` tensor (the product of two
    sparse boundary matrices is generally dense). Use this path when
    `n_k` is too large for the dense batched path but the product
    `L_k` still fits in memory.

`betti_numbers(complex, max_dim, threshold=1e-9)` returns the Betti
numbers for dims 0..max_dim by counting eigenvalues of each `L_k`
that fall below a relative `threshold` (cataloged as the library's
`numerical_floor_convention`).

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

from holonomy_lib.provenance import with_provenance
from holonomy_lib.simplicial import (
    DenseSimplicialComplex,
    SparseSimplicialComplex,
)


@with_provenance(
    "holonomy_lib.topology.hodge_laplacian", op_version="0.1",
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
      Sparse complex → `(n_k, n_k)` Tensor (dense, since the matmul
        of two sparse boundary matrices is generally dense).
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


@with_provenance(
    "holonomy_lib.topology.betti_numbers", op_version="0.1",
)
def betti_numbers(
    complex: DenseSimplicialComplex | SparseSimplicialComplex,
    max_dim: int,
    threshold: float = 1e-9,
) -> torch.Tensor:
    """Betti numbers `(β_0, β_1, …, β_max_dim)` for a simplicial complex.

    `β_k = n_k - rank(L_k)`; equivalently, the multiplicity of zero
    in `spec(L_k)`. We count eigenvalues of `L_k` that fall below a
    relative threshold (default 1e-9, the library's
    `numerical_floor_convention`).

    Args:
      complex: a `DenseSimplicialComplex` or `SparseSimplicialComplex`.
      max_dim: highest dim to return Betti numbers for. Must satisfy
        `0 ≤ max_dim ≤ complex.max_dim`.
      threshold: relative tolerance for "eigenvalue is zero". An
        eigenvalue `λ` is counted as zero iff
        `|λ| ≤ threshold · max(|spec(L_k)|, 1)`.

    Returns:
      Dense complex  → `(B, max_dim + 1)` Tensor of int64 Betti numbers.
      Sparse complex → `(max_dim + 1,)` Tensor.

    Notes:
      For each k, we compute the full eigenspectrum via
      `torch.linalg.eigvalsh` and count near-zero entries. For very
      large complexes you can compute Betti numbers more efficiently
      via `lanczos_eigsh` for the bottom-k smallest eigenvalues with
      a known spectrum bound, but the dense path is simpler and fast
      at v1 sizes (n_k ≤ a few thousand).

    References:
      Eckmann (1944) — kernel of L_k equals harmonic k-forms.
      Lim (2020), §4.
    """
    if max_dim < 0:
        raise ValueError(f"max_dim must be >= 0, got {max_dim}")
    if max_dim > complex.max_dim:
        raise ValueError(
            f"max_dim={max_dim} exceeds complex.max_dim={complex.max_dim}"
        )
    if threshold <= 0:
        raise ValueError(f"threshold must be > 0, got {threshold}")

    if isinstance(complex, DenseSimplicialComplex):
        B = complex.batch_size
        out = torch.zeros(B, max_dim + 1, dtype=torch.int64,
                           device=complex.device)
        for k in range(max_dim + 1):
            L_k = hodge_laplacian(complex, k)            # (B, n_k_max, n_k_max)
            eigvals = torch.linalg.eigvalsh(L_k)         # (B, n_k_max)
            abs_eigvals = eigvals.abs()
            # Spectral norm per-batch as reference; clamp to 1 so that
            # all-zero spectra (e.g., empty dim) use absolute threshold.
            ref = abs_eigvals.max(dim=-1, keepdim=True).values.clamp(min=1.0)
            zero_mask = abs_eigvals <= threshold * ref
            # Also mask off the padding rows of L_k — they contribute
            # eigenvalue 0 spuriously. The valid_mask for dim k tells
            # us which positions are real; we need to subtract the
            # padding count from the zero-eigenvalue count.
            valid_count = complex.valid_mask[k].sum(dim=-1)
            zero_count = zero_mask.sum(dim=-1)
            n_k_max = complex.simplices_by_dim[k].shape[1]
            pad_count = n_k_max - valid_count
            out[:, k] = (zero_count - pad_count).clamp(min=0).to(torch.int64)
        return out

    if isinstance(complex, SparseSimplicialComplex):
        out = torch.zeros(max_dim + 1, dtype=torch.int64,
                           device=complex.device)
        for k in range(max_dim + 1):
            L_k = hodge_laplacian(complex, k)            # (n_k, n_k)
            eigvals = torch.linalg.eigvalsh(L_k)
            abs_eigvals = eigvals.abs()
            ref = abs_eigvals.max().clamp(min=1.0)
            # No padding subtraction here: the sparse complex has no
            # padded rows/cols (every row of L_k corresponds to a real
            # simplex), so every near-zero eigenvalue counts. Contrast
            # with the dense path above, which must subtract the
            # padding count from the raw zero count.
            zero_count = (abs_eigvals <= threshold * ref).sum()
            out[k] = zero_count.to(torch.int64)
        return out

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

    The matmul of two sparse boundary matrices is generally dense,
    so we materialize the dense result. For very large complexes
    where this would exceed memory, the user can call Lanczos
    directly with a sparse `LinearOperator`-style implicit Hodge
    operator (planned for v2).

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
        # sparse @ dense → dense via to_dense().T @ to_dense().
        # PyTorch sparse-CSC @ sparse-CSC requires both to be in the
        # same layout and may produce a sparse-COO result. For v1
        # simplicity, densify and use the standard matmul.
        d_k_dense = d_k.to_dense()
        L = L + d_k_dense.mT @ d_k_dense

    if (k + 1) in complex.simplices_by_dim:
        d_kp1 = complex.boundary(k + 1, dtype=dtype)
        d_kp1_dense = d_kp1.to_dense()
        L = L + d_kp1_dense @ d_kp1_dense.mT

    return L
