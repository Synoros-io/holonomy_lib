# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Synoros

"""Symmetric Lanczos iteration for top-k eigenpairs of a batched matrix.

For a symmetric matrix `A ∈ ℝ^{n×n}`, the Lanczos process builds an
orthonormal Krylov basis `V ∈ ℝ^{n × m}` (with `m ≪ n` typically) such
that the projected matrix `T = Vᵀ A V` is tridiagonal. Diagonalizing
`T` (a tiny `m × m` problem) gives Ritz values and Ritz vectors that
approximate the extreme eigenpairs of `A`. After `m` iterations,
typically `m / 2` of those approximate the `m / 2` most-extreme
eigenvalues (largest by magnitude) to machine precision; the others
converge more slowly.

Algorithm (Lanczos 1950, Paige 1972):

  V[:, 0] = v_0 / ‖v_0‖                                 # random unit vector
  β_{-1} = 0
  for j in 0, …, m − 1:
      w   = A · V[:, j]                                 # 1 matmul
      α_j = ⟨w, V[:, j]⟩
      w   = w − α_j · V[:, j] − β_{j-1} · V[:, j-1]    # 3-term recurrence
      w   = orthogonalize(w, V[:, :j+1])                # full reorth
      β_j = ‖w‖
      V[:, j+1] = w / β_j
  T = tridiag(α, β)
  ritz_vals, ritz_vecs = eigh(T)                        # small
  eigenvectors of A ≈ V[:, :m] @ ritz_vecs

For numerical robustness on finite-precision GPUs we use **full
reorthogonalization** at every step: the cheaper two-step
Gram-Schmidt and the partial reorthogonalization variants are
known to lose orthogonality of `V` once Ritz values converge
(Paige's "ghost eigenvalues"). Full reorth costs `O(m² · n)` extra
flops on top of the `O(m · n²)` baseline — fine when `m` is small.

The default convention is `which = "largest_algebraic"` (top-`k` by
value). For bottom-`k` of a known-bounded-spectrum operator (e.g.
the symmetric-normalized Laplacian with spectrum in `[0, 2]`), call
`lanczos_eigsh(2.0 * I − L_sym, k=k)` and recover smallest eigenvalues
of `L_sym` as `2.0 − ritz_vals`. Shift-and-invert for general
"smallest" mode is planned.

References:
  Lanczos, C. (1950). An iteration method for the solution of the
    eigenvalue problem of linear differential and integral
    operators. Journal of Research of the National Bureau of
    Standards 45:255–282. Original algorithm.
  Paige, C. C. (1972). Computational variants of the Lanczos method
    for the eigenproblem. Journal of the Institute of Mathematics
    and Its Applications 10:373–381. Identified loss of orthogonality
    and prescribed reorthogonalization.
  Saad, Y. (2011). Numerical Methods for Large Eigenvalue Problems,
    2nd ed. SIAM. §6.5 covers full reorthogonalization.
  Golub, G. H., Van Loan, C. F. (2013). Matrix Computations, 4th ed.
    Johns Hopkins University Press. §10.1.
"""

from __future__ import annotations

from typing import Literal

import torch

from holonomy_lib.provenance import with_provenance

# Default oversampling: extra Lanczos iterations beyond `k` to improve
# accuracy of the top-k Ritz values. Saad (2011), §6.5 recommends
# ~5–10 extra for reliable top-k convergence. Halko-Martinsson-Tropp
# (2011), §1.2 use the same value (5) for randomized SVD, which is
# a closely related Krylov-projection scheme. Cataloged as
# `lanczos_oversample_default`.
LANCZOS_OVERSAMPLE_DEFAULT: int = 10


_SPARSE_LAYOUTS = (
    torch.sparse_coo, torch.sparse_csr, torch.sparse_csc,
)


@with_provenance(
    "holonomy_lib.algebra.lanczos_eigsh", op_version="0.4",
)
def lanczos_eigsh(
    A: torch.Tensor,
    k: int,
    n_iter: int | None = None,
    oversample: int = LANCZOS_OVERSAMPLE_DEFAULT,
    generator: torch.Generator | None = None,
    which: Literal["LA", "SA"] = "LA",
    sigma: float | None = None,
    solver: Literal["auto", "eigsh", "lobpcg", "torch_lobpcg"] = "auto",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-k symmetric eigenpairs via Lanczos (LA) or shift-invert (SA).

    Dispatches on layout: dense batched `(B, n, n)` runs the standard
    path; sparse `(n, n)` (CSR/CSC/COO) runs a single-instance path
    with sparse matmul on each iteration. Sparse input must be 2-D
    (no batch); the output shape is `(k,)` / `(n, k)` without a batch
    dim. For batched sparse you can either iterate yourself or convert
    to a sparse batched COO and call repeatedly.

    Args:
      A: dense `(B, n, n)` or sparse-CSC/CSR/COO `(n, n)` symmetric.
        We assume but do not check `A == A.T`; violation produces
        meaningless results.
      k: number of eigenpairs to return. `1 ≤ k ≤ n`.
      n_iter: total Lanczos iterations. Default `k + oversample`,
        clamped to `n`. More iterations → better convergence to interior
        eigenvalues but more compute.
      oversample: extra iterations beyond `k`. Default 10.
      generator: torch.Generator for the random starting vector.
      which: `"LA"` for largest-algebraic (default), or `"SA"` for
        smallest-algebraic via shift-and-invert. SA mode runs Lanczos
        on `(A − σ·I)^{−1}` whose dominant Ritz values converge to
        the eigenvalues of `A` closest to σ. Dense input runs the
        in-house shift-invert Lanczos; sparse input dispatches to
        `algebra.eigsh_sparse.eigsh_sparse_sa` (hybrid SciPy
        shift-invert / SciPy LOBPCG+spilu / `torch.lobpcg` backends,
        selected by `solver`).
      sigma: shift used in SA mode. Default `0.0` for dense input
        (finds smallest |λ|, which is exactly the smallest λ on
        positive-semidefinite `A`); sparse input defaults to the
        cataloged `SPARSE_SA_SHIFT_INVERT_POLE` (just below a PSD
        spectrum, so a Laplacian's λ=0 kernel stays factorable).
        For indefinite `A` set σ below the smallest eigenvalue, or set
        it close to whichever interior eigenvalues you want.
      solver: sparse-SA backend selection; `"auto"` picks by device
        and size. Ignored for dense input and for `which="LA"`. See
        `algebra.eigsh_sparse` for the dispatch rule and the
        scale-of-validity notes.

    Returns:
      Dense input  → `(eigvals: (B, k), eigvecs: (B, n, k))`.
      Sparse input → `(eigvals: (k,), eigvecs: (n, k))`.

      LA mode: eigenvalues descending in value.
      SA mode: eigenvalues ascending in value (smallest first).
      In both cases, eigenvectors are orthonormal with columns aligned
      to the eigenvalues.

    Notes:
      Cost is `O(B · n_iter · n²)` for the matmuls plus
      `O(B · n_iter² · n)` for full reorthogonalization. For `n_iter ≪ n`
      this is much cheaper than the `O(B · n³)` dense `torch.linalg.eigh`.

      Convergence: the top-`oversample` Ritz values are the most
      accurate; the worst-converged of the returned `k` may have
      error `O(λ_{k+1} / λ_k)^{2·oversample}` in the typical case
      (Saad 2011, §6.7).

    References:
      Lanczos (1950); Paige (1972); Saad (2011), §6.5.
      Ericsson & Ruhe (1980). The spectral transformation Lanczos
        method for the numerical solution of large sparse generalized
        symmetric eigenvalue problems. Math. Comp. 35(152):1251–1268.
        The shift-and-invert formulation.
    """
    if which not in ("LA", "SA"):
        raise ValueError(
            f"which must be 'LA' (largest algebraic) or 'SA' "
            f"(smallest algebraic); got which={which!r}"
        )
    # Shift-invert path (SA mode).
    if which == "SA":
        if A.layout in _SPARSE_LAYOUTS:
            from holonomy_lib.algebra.eigsh_sparse import eigsh_sparse_sa

            return eigsh_sparse_sa(
                A, k, sigma=sigma, solver=solver, generator=generator,
            )
        return _lanczos_shift_invert_dense(
            A, k,
            sigma=0.0 if sigma is None else sigma,
            n_iter=n_iter, oversample=oversample, generator=generator,
        )
    # LA mode below — original code path.
    if sigma is not None:
        raise ValueError(
            "sigma is only meaningful in shift-invert mode (which='SA'); "
            "with which='LA' the spectrum is read directly off A."
        )
    # Dispatch on layout. Sparse inputs go through a 2-D no-batch path.
    if A.layout in _SPARSE_LAYOUTS:
        return _lanczos_sparse(A, k, n_iter, oversample, generator)
    if A.ndim < 2 or A.shape[-1] != A.shape[-2]:
        raise ValueError(
            f"A must be (..., n, n); got A.shape={tuple(A.shape)}"
        )
    n = A.shape[-1]
    if k <= 0 or k > n:
        raise ValueError(f"k must satisfy 1 <= k <= n={n}, got k={k}")

    # Total Lanczos iterations. Cap at n (the Krylov subspace can't
    # exceed n dimensions in exact arithmetic).
    if n_iter is None:
        n_iter = k + oversample
    n_iter = min(n_iter, n)
    if n_iter < k:
        raise ValueError(
            f"n_iter={n_iter} must be >= k={k} to return k eigenpairs"
        )

    *batch, _ = A.shape[:-1]
    device, dtype = A.device, A.dtype

    # Initial random unit vector. Same per-batch shape conventions.
    v0 = torch.randn(*batch, n, generator=generator, device=device, dtype=dtype)
    v0 = v0 / torch.linalg.norm(v0, dim=-1, keepdim=True)

    # Pre-allocate the Lanczos basis tensor and accumulators. Writing
    # in-place to `V[..., j]` avoids the O(m² · n) of re-stacking
    # `V_cols` every iteration. The (n_iter)-th column is the boundary
    # vector v_{n_iter}; it's computed inside the loop for the
    # 3-term recurrence but not used in the projected problem.
    V = torch.empty(*batch, n, n_iter + 1, device=device, dtype=dtype)
    V[..., 0] = v0
    alphas: list[torch.Tensor] = []                     # each (B,)
    betas: list[torch.Tensor] = []                      # each (B,)

    # Beta_{-1} = 0, v_{-1} = 0 (3-term recurrence boundary condition).
    v_prev = torch.zeros_like(v0)
    beta_prev = torch.zeros(*batch, device=device, dtype=dtype)

    for j in range(n_iter):
        v_curr = V[..., j]
        # Matmul: A · v_curr.
        w = torch.matmul(A, v_curr.unsqueeze(dim=-1)).squeeze(dim=-1)
        alpha_j = (w * v_curr).sum(dim=-1)
        alphas.append(alpha_j)

        # 3-term recurrence: w = w − α_j v_curr − β_{j−1} v_prev.
        w = (
            w
            - alpha_j.unsqueeze(dim=-1) * v_curr
            - beta_prev.unsqueeze(dim=-1) * v_prev
        )

        # Full reorthogonalization against all V[:, :j+1]. Slicing
        # the pre-allocated tensor is O(1) — no copy.
        V_used = V[..., :j + 1]                          # (B, n, j+1)
        coeffs = torch.matmul(
            V_used.mT, w.unsqueeze(dim=-1),
        ).squeeze(dim=-1)                                 # (B, j+1)
        w = w - torch.matmul(V_used, coeffs.unsqueeze(dim=-1)).squeeze(dim=-1)

        beta_j = torch.linalg.norm(w, dim=-1)             # (B,)
        # Avoid division by zero on Lanczos breakdown. We clamp to the
        # dtype's tiny positive; the resulting basis vector is
        # numerical noise but doesn't propagate NaN. In batched code we
        # can't selectively stop per-batch, so this is the safest
        # behaviour. (Production-grade: replace with a fresh random
        # vector and continue — left as future work.)
        floor = torch.finfo(dtype).tiny
        safe_beta = beta_j.clamp(min=floor)
        V[..., j + 1] = w / safe_beta.unsqueeze(dim=-1)

        betas.append(beta_j)
        v_prev = v_curr
        beta_prev = beta_j

    # The Krylov basis we project onto is the first `n_iter` columns
    # of V; the (n_iter)-th column is the boundary vector for the
    # 3-term recurrence and is not part of the projected problem.
    V_basis = V[..., :n_iter]                             # (B, n, n_iter)

    # Build tridiagonal T = diag(alpha) + offdiag(beta).
    # alphas: list of (B,) of length n_iter
    # betas:  list of (B,) of length n_iter (we only need the first
    # n_iter − 1 for the tridiagonal off-diagonal).
    T = _build_tridiagonal(alphas, betas[:-1] if len(betas) >= 1 else [])

    # Solve the small (B, n_iter, n_iter) symmetric eigenproblem.
    ritz_vals, ritz_vecs = torch.linalg.eigh(T)            # ascending

    # Approximate eigenvectors of A.
    approx_eigvecs = torch.matmul(V_basis, ritz_vecs)      # (B, n, n_iter)

    # Take top-k by VALUE (descending). eigh returns ascending; flip.
    top_vals = ritz_vals.flip(dims=(-1,))[..., :k]
    top_vecs = approx_eigvecs.flip(dims=(-1,))[..., :k]
    return top_vals, top_vecs


# ============================================================
# Internal helpers
# ============================================================


def _lanczos_sparse(
    A: torch.Tensor,
    k: int,
    n_iter: int | None,
    oversample: int,
    generator: torch.Generator | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """2-D no-batch Lanczos for sparse `A`.

    Mirrors the dense path's algorithm — the only difference is that
    `A @ v` goes through sparse matmul, and there's no leading batch
    dimension. Used by `betti_numbers` on sparse Hodge Laplacians and
    by callers who pass sparse `A` directly.

    Limitations:
      - 2-D only (no batch). Caller wraps for batched sparse cases.
      - Provenance recording of sparse `A` reads its dense form via
        `to_dense()` before hashing, which negates the memory win of
        sparse inputs inside `record()`. Outside `record()` the
        decorator is transparent and this doesn't matter.
    """
    if A.ndim != 2 or A.shape[-1] != A.shape[-2]:
        raise ValueError(
            f"sparse A must be (n, n); got A.shape={tuple(A.shape)}"
        )
    n = A.shape[-1]
    if k <= 0 or k > n:
        raise ValueError(f"k must satisfy 1 <= k <= n={n}, got k={k}")

    if n_iter is None:
        n_iter = k + oversample
    n_iter = min(n_iter, n)
    if n_iter < k:
        raise ValueError(
            f"n_iter={n_iter} must be >= k={k} to return k eigenpairs"
        )

    device, dtype = A.device, A.dtype

    v0 = torch.randn(n, generator=generator, device=device, dtype=dtype)
    v0 = v0 / torch.linalg.norm(v0)

    V = torch.empty(n, n_iter + 1, device=device, dtype=dtype)
    V[:, 0] = v0
    alphas: list[torch.Tensor] = []
    betas: list[torch.Tensor] = []

    v_prev = torch.zeros_like(v0)
    beta_prev = torch.zeros((), device=device, dtype=dtype)

    for j in range(n_iter):
        v_curr = V[:, j]
        # Sparse matmul: A @ v. For sparse-CSC + dense vector this
        # uses the appropriate spmv kernel. The (n, 1) shape is needed
        # because torch.matmul on sparse expects a 2-D RHS.
        w = torch.matmul(A, v_curr.unsqueeze(-1)).squeeze(-1)
        alpha_j = (w * v_curr).sum()
        alphas.append(alpha_j)

        w = w - alpha_j * v_curr - beta_prev * v_prev

        # Full reorthogonalization against V[:, :j+1] (all dense).
        V_used = V[:, :j + 1]                            # (n, j+1)
        coeffs = V_used.mT @ w                            # (j+1,)
        w = w - V_used @ coeffs

        beta_j = torch.linalg.norm(w)
        floor = torch.finfo(dtype).tiny
        safe_beta = beta_j.clamp(min=floor)
        V[:, j + 1] = w / safe_beta

        betas.append(beta_j)
        v_prev = v_curr
        beta_prev = beta_j

    V_basis = V[:, :n_iter]                              # (n, n_iter)

    # Build tridiagonal T = diag(alpha) + offdiag(beta).
    # alphas/betas here are 0-D tensors; stack and use the dense path.
    diag = torch.stack(alphas)                            # (n_iter,)
    T = torch.diag(diag)
    if len(betas) > 1:
        off = torch.stack(betas[:-1])                     # (n_iter - 1,)
        T = T + torch.diag(off, diagonal=1) + torch.diag(off, diagonal=-1)

    ritz_vals, ritz_vecs = torch.linalg.eigh(T)
    approx_eigvecs = V_basis @ ritz_vecs

    top_vals = ritz_vals.flip(dims=(0,))[:k]
    top_vecs = approx_eigvecs.flip(dims=(-1,))[:, :k]
    return top_vals, top_vecs


def _lanczos_shift_invert_dense(
    A: torch.Tensor,
    k: int,
    sigma: float,
    n_iter: int | None,
    oversample: int,
    generator: torch.Generator | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Shift-and-invert Lanczos: top-k eigenpairs of `A` closest to σ.

    Algorithm (Ericsson-Ruhe 1980): replace the action `A · v` with
    `(A − σ·I)^{−1} · v` inside the Lanczos loop. The Ritz values μ_i
    of the inverted operator satisfy `μ_i = 1/(λ_i − σ)`, so the
    largest |μ_i| (which Lanczos converges to fastest) correspond to
    the λ_i closest to σ. After the loop we recover
    `λ_i = σ + 1/μ_i`.

    Implementation:
      - Pre-factor `A − σ·I` once with LU; each iteration is one
        `lu_solve` instead of a fresh `solve`. n_iter solves total
        plus one factorization → O(n³) for factor + O(n_iter · n²)
        for solves, vs O(n_iter · n³) without caching.
      - Returns smallest-first ordering: with σ ≤ smallest_λ, larger
        μ_i means smaller (λ_i − σ) means smaller λ_i.
    """
    if A.ndim < 2 or A.shape[-1] != A.shape[-2]:
        raise ValueError(
            f"A must be (..., n, n); got A.shape={tuple(A.shape)}"
        )
    n = A.shape[-1]
    if k <= 0 or k > n:
        raise ValueError(f"k must satisfy 1 <= k <= n={n}, got k={k}")

    if n_iter is None:
        n_iter = k + oversample
    n_iter = min(n_iter, n)
    if n_iter < k:
        raise ValueError(
            f"n_iter={n_iter} must be >= k={k} to return k eigenpairs"
        )

    *batch, _ = A.shape[:-1]
    device, dtype = A.device, A.dtype

    # Pre-factor (A − σ·I). Done once outside the Lanczos loop. We
    # build the shifted matrix without an explicit `torch.eye` expand
    # by subtracting σ from the diagonal in-place on a clone. A
    # `LinAlgError`-style failure from `lu_factor` means (A − σ·I)
    # has an exact zero on its diagonal after pivoting — i.e. σ is
    # an exact eigenvalue of A. Re-raise as our breakdown error so
    # the caller sees a uniform message regardless of where the
    # singularity surfaces.
    A_shifted = A.clone()
    diag_indices = torch.arange(n, device=device)
    A_shifted[..., diag_indices, diag_indices] -= sigma
    try:
        LU, piv = torch.linalg.lu_factor(A_shifted)
    except RuntimeError as e:
        if "zero" in str(e).lower() or "singular" in str(e).lower():
            raise RuntimeError(
                f"lanczos_eigsh shift-invert breakdown: "
                f"(A - σ·I) appears singular at σ={sigma}. "
                f"Pick σ outside the spectrum of A. For graph "
                f"Laplacians (which have 0 in spectrum) use a small "
                f"negative shift, e.g. σ=-1e-3."
            ) from e
        raise

    def solve(v: torch.Tensor) -> torch.Tensor:
        """One application of (A − σ·I)^{−1} to a batched vector."""
        return torch.linalg.lu_solve(
            LU, piv, v.unsqueeze(dim=-1),
        ).squeeze(dim=-1)

    v0 = torch.randn(*batch, n, generator=generator, device=device, dtype=dtype)
    v0 = v0 / torch.linalg.norm(v0, dim=-1, keepdim=True)

    V = torch.empty(*batch, n, n_iter + 1, device=device, dtype=dtype)
    V[..., 0] = v0
    alphas: list[torch.Tensor] = []
    betas: list[torch.Tensor] = []

    v_prev = torch.zeros_like(v0)
    beta_prev = torch.zeros(*batch, device=device, dtype=dtype)

    for j in range(n_iter):
        v_curr = V[..., j]
        # Apply (A − σ·I)^{−1} in place of A.
        w = solve(v_curr)
        alpha_j = (w * v_curr).sum(dim=-1)
        alphas.append(alpha_j)

        w = (
            w
            - alpha_j.unsqueeze(dim=-1) * v_curr
            - beta_prev.unsqueeze(dim=-1) * v_prev
        )

        V_used = V[..., :j + 1]
        coeffs = torch.matmul(
            V_used.mT, w.unsqueeze(dim=-1),
        ).squeeze(dim=-1)
        w = w - torch.matmul(V_used, coeffs.unsqueeze(dim=-1)).squeeze(dim=-1)

        beta_j = torch.linalg.norm(w, dim=-1)
        floor = torch.finfo(dtype).tiny
        safe_beta = beta_j.clamp(min=floor)
        V[..., j + 1] = w / safe_beta.unsqueeze(dim=-1)

        betas.append(beta_j)
        v_prev = v_curr
        beta_prev = beta_j

    V_basis = V[..., :n_iter]
    T = _build_tridiagonal(alphas, betas[:-1] if len(betas) >= 1 else [])

    # Ritz values of the inverted operator μ_i.
    mu_vals, ritz_vecs = torch.linalg.eigh(T)               # ascending
    approx_eigvecs = torch.matmul(V_basis, ritz_vecs)

    # Pick top-k by |μ| (largest magnitude first). These are the Ritz
    # values most converged in shift-invert, corresponding to the λ_i
    # closest to σ. NB: picking by value (`mu_vals.flip`) only works
    # when all μ_i have the same sign — e.g. σ ≤ smallest_λ on PSD —
    # and silently picks the wrong end for interior σ.
    abs_sort_idx = mu_vals.abs().argsort(dim=-1, descending=True)
    top_idx = abs_sort_idx[..., :k]
    top_mu = torch.gather(mu_vals, dim=-1, index=top_idx)
    top_idx_expand = top_idx.unsqueeze(dim=-2).expand(
        *approx_eigvecs.shape[:-1], k,
    )
    top_vecs = torch.gather(approx_eigvecs, dim=-1, index=top_idx_expand)

    # Recover original eigenvalues. A breakdown is `|μ| ≈ 0`, which
    # means σ is very close to one of A's eigenvalues — the shifted
    # operator `(A − σI)` is nearly singular and the recovered
    # `λ = σ + 1/μ` blows up. The `finfo(dtype).tiny` floor (≈2e-308)
    # is too small to catch this in practice: real breakdowns
    # propagate as `inf`/`nan` through the `lu_solve`, not as tiny
    # finite μ. Use a more practical threshold (`sqrt(eps)`, scaled by
    # the typical μ magnitude) that fires on the actual failure mode.
    eps = torch.finfo(dtype).eps
    breakdown_threshold = eps ** 0.5 * top_mu.abs().max().clamp(min=1.0)
    if (top_mu.abs() < breakdown_threshold).any() or not torch.isfinite(top_mu).all():
        raise RuntimeError(
            f"lanczos_eigsh shift-invert breakdown: "
            f"(A - σ·I) appears singular at σ={sigma}. "
            f"Pick σ outside the spectrum of A. For graph Laplacians "
            f"(which have 0 in spectrum) use a small negative shift, "
            f"e.g. σ=-1e-3."
        )
    lambdas = sigma + 1.0 / top_mu

    # SA convention: smallest first. Sort ascending across the k axis,
    # carrying the eigenvectors along.
    sort_idx = lambdas.argsort(dim=-1)
    lambdas_sorted = torch.gather(lambdas, dim=-1, index=sort_idx)
    # Reorder the eigenvector columns. `gather` doesn't easily handle
    # the n-axis so we use index_select-style via expand+gather along
    # the last axis of `top_vecs`.
    idx_expand = sort_idx.unsqueeze(dim=-2).expand_as(top_vecs)
    vecs_sorted = torch.gather(top_vecs, dim=-1, index=idx_expand)
    return lambdas_sorted, vecs_sorted


def _build_tridiagonal(
    alphas: list[torch.Tensor],
    betas: list[torch.Tensor],
) -> torch.Tensor:
    """Build a batched tridiagonal matrix from diagonal + sub/super-diagonal.

    Args:
      alphas: list of length n, each (B,). The diagonal entries.
      betas: list of length n − 1, each (B,). The sub/super-diagonal entries
        (the matrix is symmetric).

    Returns:
      T: (B, n, n) tridiagonal.
    """
    n = len(alphas)
    # Explicit ValueError (rather than assert) so this guard survives
    # `python -O`: an internal indexing mismatch would silently
    # produce a wrong-sized T and corrupt the Ritz pairs.
    if len(betas) != n - 1:
        raise ValueError(
            f"need {n - 1} off-diagonal entries for {n}x{n} tridiagonal, "
            f"got {len(betas)}"
        )
    diag = torch.stack(alphas, dim=-1)                    # (B, n)
    T = torch.diag_embed(diag)
    if n > 1:
        off = torch.stack(betas, dim=-1)                  # (B, n-1)
        # Place off on both sub- and super-diagonal.
        T = T + torch.diag_embed(off, offset=1) + torch.diag_embed(off, offset=-1)
    return T
