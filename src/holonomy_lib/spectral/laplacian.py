# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Synoros

"""Graph Laplacians, batched-first, GPU-native.

Given a weighted adjacency matrix `A` of shape `(B, n, n)`, this module
constructs the standard real-valued Laplacian variants. All variants
assume `A` is symmetric — non-symmetric (directed) graphs need the
magnetic Laplacian.

Sparse inputs (CSR/CSC/COO, 2-D `(n, n)`, no batch) are supported on
every variant via a layout-dispatching internal path. The output's
layout matches the input's: sparse-in → sparse-COO-out, dense-in →
dense-out. Combined with `algebra.lanczos_eigsh`'s sparse path you get
end-to-end large-graph spectral chains without ever materializing the
dense `(n, n)` Laplacian.

Variants:
  combinatorial(A)          — L = D − A.            Always PSD.
  symmetric_normalized(A)   — L_sym = I − D^{−1/2} A D^{−1/2}.
                              PSD, eigenvalues in [0, 2].
  random_walk(A)            — L_rw = I − D^{−1} A.
                              Eigenvalues real and in [0, 2] (similar to L_sym).
  signed(A)                 — L^σ = D^{|σ|} − A,
                              D^{|σ|}_{ii} = Σ_j |A_{ij}|.
                              PSD even with negative edge weights (Kunegis 2010).

Isolated nodes (zero degree) are handled by the Moore-Penrose convention:
when D_{ii} = 0, the corresponding row/column of D^{−1/2} and D^{−1} is
set to zero. This makes L_sym and L_rw well-defined for arbitrary graphs;
the isolated node then sits at eigenvalue 1 (for L_rw and the
combinatorial form) or 0 (for L_sym). Convention follows Cheng-Wu (2024)
SIAM J. Math. Data Sci.

References:
  Chung, F. R. K. (1997). Spectral Graph Theory. CBMS Regional Conference
    Series in Mathematics, 92. American Mathematical Society.
  von Luxburg, U. (2007). A tutorial on spectral clustering. Statistics
    and Computing 17(4):395–416.
  Kunegis, J., Schmidt, S., Lommatzsch, A., Lerner, J., De Luca, E. W.,
    Albayrak, S. (2010). Spectral analysis of signed graphs for
    clustering, prediction and visualization. Proceedings of the 2010
    SIAM International Conference on Data Mining, 559–570.
  Ribando-Gros, E., Wang, R., Chen, J., Tong, Y., Wei, G.-W. (2024).
    Combinatorial and Hodge Laplacians: Similarities and differences.
    SIAM Review 66(3):575–601.
  Cheng, X., Wu, N. (2024). Bi-stochastically normalized graph
    Laplacian: convergence to manifold Laplacian and robustness to
    outlier noise. Information and Inference: A Journal of the IMA.
"""

from __future__ import annotations

import torch

from holonomy_lib._graph_utils import drop_self_loops

_SPARSE_LAYOUTS = (
    torch.sparse_coo, torch.sparse_csr, torch.sparse_csc,
)


def degree(A: torch.Tensor, signed: bool = False) -> torch.Tensor:
    """Degree vector of a (batched) weighted adjacency matrix.

    Per the library's simple-graph convention (`CONVENTIONS.md`), the
    diagonal of `A` is dropped before summing: self-loops do not
    contribute to a node's degree. Idempotent on inputs that already
    have a zero diagonal.

    Args:
      A: (B, n, n) dense or 2-D sparse `(n, n)` weighted adjacency.
        Symmetry is assumed.
      signed: if True, use absolute weights for the degree (∑_j |A_ij|),
        the Kunegis (2010) convention used in the signed Laplacian.
        Otherwise use the raw weighted degree (∑_j A_ij), which is the
        standard definition for non-negative weights.

    Returns:
      Dense `(B, n)` for dense input, dense `(n,)` for sparse input.

    References:
      Chung (1997), §1.2 — weighted degree.
      Kunegis et al. (2010), eq. 1 — signed degree.
    """
    if A.layout in _SPARSE_LAYOUTS:
        return _degree_sparse(A, signed=signed)
    _check_square_with_batch(A)
    A = drop_self_loops(A)
    return A.abs().sum(dim=-1) if signed else A.sum(dim=-1)


def combinatorial(A: torch.Tensor) -> torch.Tensor:
    """Combinatorial Laplacian L = D − A.

    For symmetric `A` with non-negative weights, L is PSD with eigenvalue
    0 of multiplicity equal to the number of connected components.

    Args:
      A: (B, n, n) dense or 2-D sparse `(n, n)` symmetric weighted
        adjacency.
    Returns:
      Dense input → `(B, n, n)`. Sparse input → sparse-COO `(n, n)`.

    References:
      Chung (1997), §1.2.
      von Luxburg (2007), §3 — unnormalized Laplacian.
    """
    if A.layout in _SPARSE_LAYOUTS:
        return _combinatorial_sparse(A)
    _check_square_with_batch(A)
    A = drop_self_loops(A)
    d = degree(A, signed=False)              # (B, n)
    return torch.diag_embed(d) - A


def symmetric_normalized(A: torch.Tensor) -> torch.Tensor:
    """Symmetric normalized Laplacian L_sym = I − D^{−1/2} A D^{−1/2}.

    PSD with spectrum in [0, 2]. The factor D^{−1/2} uses the
    Moore-Penrose convention for isolated nodes (set to 0 where D = 0).

    Args:
      A: (B, n, n) dense or 2-D sparse `(n, n)` symmetric weighted
        adjacency, non-negative weights.
    Returns:
      Dense input → `(B, n, n)`. Sparse input → sparse-COO `(n, n)`.

    References:
      Chung (1997), §1.2.
      von Luxburg (2007), §3 — L_sym.
      Cheng-Wu (2024) — pseudoinverse handling of isolated nodes.
    """
    if A.layout in _SPARSE_LAYOUTS:
        return _symmetric_normalized_sparse(A)
    _check_square_with_batch(A)
    A = drop_self_loops(A)
    d = degree(A, signed=False)               # (B, n)
    d_inv_sqrt = _safe_inv_sqrt(d)            # (B, n) — zeros where d=0
    n = A.shape[-1]
    eye = _batched_eye(n, batch=A.shape[:-2], device=A.device, dtype=A.dtype)
    # D^{-1/2} A D^{-1/2}  via broadcast multiplication
    A_norm = A * d_inv_sqrt.unsqueeze(dim=-1) * d_inv_sqrt.unsqueeze(dim=-2)
    return eye - A_norm


def random_walk(A: torch.Tensor) -> torch.Tensor:
    """Random-walk Laplacian L_rw = I − D^{−1} A.

    Has the same eigenvalues as L_sym (it is similar to L_sym via D^{1/2}
    conjugation). Eigenvectors of L_rw are the random-walk transition
    eigenvectors, useful for clustering and diffusion. Spectrum in [0, 2].

    Args:
      A: (B, n, n) dense or 2-D sparse `(n, n)` symmetric weighted
        adjacency, non-negative weights.
    Returns:
      Dense input → `(B, n, n)`. Sparse input → sparse-COO `(n, n)`.

    References:
      Chung (1997), §1.5.
      von Luxburg (2007), §3 — L_rw.
    """
    if A.layout in _SPARSE_LAYOUTS:
        return _random_walk_sparse(A)
    _check_square_with_batch(A)
    A = drop_self_loops(A)
    d = degree(A, signed=False)               # (B, n)
    d_inv = _safe_inv(d)                       # (B, n) — zeros where d=0
    n = A.shape[-1]
    eye = _batched_eye(n, batch=A.shape[:-2], device=A.device, dtype=A.dtype)
    A_rw = A * d_inv.unsqueeze(dim=-1)         # broadcast: (B, n, n) * (B, n, 1)
    return eye - A_rw


def signed(A: torch.Tensor) -> torch.Tensor:
    """Signed Laplacian L^σ = D^{|σ|} − A,  D^{|σ|}_{ii} = Σ_j |A_{ij}|.

    For symmetric `A` with arbitrary real weights (positive AND negative),
    L^σ is PSD. The eigenvalue 0 corresponds to *balanced* connected
    components of the signed graph (Kunegis 2010, Theorem 3.4): a
    signed graph is balanced iff zero is in the spectrum of L^σ.

    Args:
      A: (B, n, n) dense or 2-D sparse `(n, n)` symmetric weighted
        adjacency, weights may be negative.
    Returns:
      Dense input → `(B, n, n)`. Sparse input → sparse-COO `(n, n)`.

    References:
      Kunegis et al. (2010), eq. 5 — definition of the signed Laplacian.
      Kunegis et al. (2010), Theorem 3.4 — PSD property and balance.
      Mercado-Tudisco-Hein (2019), NeurIPS — Signed Power Mean Laplacians
        generalize this to a one-parameter family.
    """
    if A.layout in _SPARSE_LAYOUTS:
        return _signed_sparse(A)
    _check_square_with_batch(A)
    A = drop_self_loops(A)
    d_abs = degree(A, signed=True)             # (B, n) — ∑_j |A_ij|
    return torch.diag_embed(d_abs) - A


# ----------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------


def _check_square_with_batch(A: torch.Tensor) -> None:
    """A must be ≥3-D with the last two dims square: shape (..., n, n)."""
    if A.ndim < 2:
        raise ValueError(
            f"A must have at least one batch dim and be square (..., n, n); "
            f"got A.shape={tuple(A.shape)}"
        )
    if A.shape[-1] != A.shape[-2]:
        raise ValueError(
            f"A must be square in the last two dims; got A.shape={tuple(A.shape)}"
        )


def _safe_inv_sqrt(d: torch.Tensor) -> torch.Tensor:
    """Pseudoinverse of D^{1/2}: 1/√d where d > 0, zero where d = 0.

    Moore-Penrose convention for isolated nodes (Cheng-Wu 2024).
    Uses `torch.where` rather than boolean-mask assignment so the
    operation stays fully vectorized — important for `torch.vmap`,
    second-order autograd, and `torch.compile`. We pre-clamp the
    rsqrt input so the unselected branch never evaluates to `inf`
    (which would still appear in the autograd graph even though
    `where` masks it out at the value level).
    """
    safe_d = d.clamp(min=torch.finfo(d.dtype).tiny)
    return torch.where(d > 0, torch.rsqrt(safe_d), torch.zeros_like(d))


def _safe_inv(d: torch.Tensor) -> torch.Tensor:
    """Pseudoinverse of D: 1/d where d > 0, zero where d = 0.

    Moore-Penrose convention for isolated nodes (Cheng-Wu 2024).
    See `_safe_inv_sqrt` for the `torch.where` rationale.
    """
    safe_d = d.clamp(min=torch.finfo(d.dtype).tiny)
    return torch.where(d > 0, torch.reciprocal(safe_d), torch.zeros_like(d))


def _batched_eye(
    n: int,
    batch: tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Identity matrix broadcast to the given batch shape: (*batch, n, n)."""
    eye = torch.eye(n, device=device, dtype=dtype)
    return eye.expand(*batch, n, n)


# ----------------------------------------------------------------
# Sparse helpers — 2-D `(n, n)` paths for the Laplacian variants.
#
# Sparse-batched semantics are too thin in PyTorch (no batched
# sparse-COO that handles broadcasting), so we restrict sparse inputs
# to 2-D and return sparse-COO. The Lanczos sparse path already
# handles that layout end-to-end.
# ----------------------------------------------------------------


def _check_2d_square_sparse(A: torch.Tensor) -> None:
    if A.ndim != 2 or A.shape[-1] != A.shape[-2]:
        raise ValueError(
            f"sparse A must be 2-D (n, n); got A.shape={tuple(A.shape)}"
        )


def _coalesced_no_self_loops(A: torch.Tensor) -> torch.Tensor:
    """Coalesce A to sparse-COO and drop diagonal entries."""
    A_coo = A.to_sparse_coo().coalesce()
    indices = A_coo.indices()
    values = A_coo.values()
    keep = indices[0] != indices[1]
    return torch.sparse_coo_tensor(
        indices[:, keep], values[keep], A.shape,
    ).coalesce()


def _degree_sparse(A: torch.Tensor, signed: bool) -> torch.Tensor:
    """Dense `(n,)` row-sum of a 2-D sparse `A` after dropping self-loops."""
    _check_2d_square_sparse(A)
    A = _coalesced_no_self_loops(A)
    if signed:
        # abs() on coo values is a value-only op; coalesced layout
        # survives because abs is monotone in the index ordering.
        vals = A.values().abs()
        A = torch.sparse_coo_tensor(A.indices(), vals, A.shape)
    return torch.sparse.sum(A, dim=-1).to_dense()


def _sparse_DminusA(
    A_noloop: torch.Tensor,
    d_diag: torch.Tensor,
    A_value_scale: torch.Tensor | None = None,
    diag_sign: float = 1.0,
) -> torch.Tensor:
    """Build a sparse-COO Laplacian of the form `diag_sign · diag(d) − S · A`.

    Args:
      A_noloop: coalesced sparse-COO `(n, n)` with the diagonal already
        zero. Caller passes the (already-self-loop-stripped) input.
      d_diag: dense `(n,)` values for the diagonal of the diag-term.
      A_value_scale: optional dense `(nnz,)` vector multiplied
        elementwise into `A_noloop.values()` before the `−` step.
        Used by the normalized variants to apply `D^{−1/2} · A · D^{−1/2}`
        or `D^{−1} · A` row/col scalings.
      diag_sign: ±1. `+1` for the unnormalized forms (where the
        diagonal contribution is the raw degree `d`); `+1` and
        d_diag = ones for the normalized forms (where the diagonal is
        the identity).

    Returns:
      Coalesced sparse-COO `(n, n)`.
    """
    n = A_noloop.shape[-1]
    device, dtype = A_noloop.device, A_noloop.dtype

    diag_indices = torch.arange(n, device=device).unsqueeze(0).expand(2, n)
    diag_values = diag_sign * d_diag

    A_indices = A_noloop.indices()
    A_values = A_noloop.values()
    if A_value_scale is not None:
        A_values = A_value_scale * A_values

    combined_indices = torch.cat([diag_indices, A_indices], dim=1)
    combined_values = torch.cat([diag_values, -A_values])
    L = torch.sparse_coo_tensor(combined_indices, combined_values, (n, n))
    return L.coalesce()


def _combinatorial_sparse(A: torch.Tensor) -> torch.Tensor:
    """Sparse-COO combinatorial Laplacian L = D − A."""
    _check_2d_square_sparse(A)
    A_noloop = _coalesced_no_self_loops(A)
    d = torch.sparse.sum(A_noloop, dim=-1).to_dense()       # (n,)
    return _sparse_DminusA(A_noloop, d)


def _symmetric_normalized_sparse(A: torch.Tensor) -> torch.Tensor:
    """Sparse-COO normalized Laplacian L_sym = I − D^{−1/2} A D^{−1/2}."""
    _check_2d_square_sparse(A)
    A_noloop = _coalesced_no_self_loops(A)
    d = torch.sparse.sum(A_noloop, dim=-1).to_dense()       # (n,)
    d_inv_sqrt = _safe_inv_sqrt(d)
    # Scale each off-diagonal value by d_inv_sqrt[row] · d_inv_sqrt[col].
    row = A_noloop.indices()[0]
    col = A_noloop.indices()[1]
    scale = d_inv_sqrt[row] * d_inv_sqrt[col]
    ones = torch.ones_like(d_inv_sqrt)
    return _sparse_DminusA(A_noloop, ones, A_value_scale=scale)


def _random_walk_sparse(A: torch.Tensor) -> torch.Tensor:
    """Sparse-COO random-walk Laplacian L_rw = I − D^{−1} A."""
    _check_2d_square_sparse(A)
    A_noloop = _coalesced_no_self_loops(A)
    d = torch.sparse.sum(A_noloop, dim=-1).to_dense()       # (n,)
    d_inv = _safe_inv(d)
    row = A_noloop.indices()[0]
    scale = d_inv[row]                                        # (nnz,) — row scaling only
    ones = torch.ones_like(d_inv)
    return _sparse_DminusA(A_noloop, ones, A_value_scale=scale)


def _signed_sparse(A: torch.Tensor) -> torch.Tensor:
    """Sparse-COO signed Laplacian L^σ = D^{|.|} − A."""
    _check_2d_square_sparse(A)
    A_noloop = _coalesced_no_self_loops(A)
    abs_A = torch.sparse_coo_tensor(
        A_noloop.indices(), A_noloop.values().abs(), A_noloop.shape,
    )
    d_abs = torch.sparse.sum(abs_A, dim=-1).to_dense()      # (n,)
    return _sparse_DminusA(A_noloop, d_abs)
