# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Synoros

"""Regularized (Tikhonov / ridge) least squares with solver dispatch.

synoros-substrate hand-rolls this one convex problem four inconsistent
ways (normal-equation `solve`, `lstsq`, sparse LSQR, dense pseudoinverse),
so it lives here once:

    min_x ‖A x − b‖² + λ ‖x‖²

Three solution paths, one interface:

  - `"cholesky"`: normal equations `(AᵀA + λI) x = Aᵀb` via Cholesky
    factorization + triangular solve (CONVENTIONS §3.3). Exact, one
    factorization, the default for dense problems. It squares
    the condition number, so for ill-conditioned `A` with tiny `λ`
    prefer `"lsqr"`.
  - `"cg"`: conjugate gradient on the normal operator `v ↦ Aᵀ(Av) + λv`.
    `AᵀA` is never materialized. For very large dense problems where
    the `(n, n)` Gram matrix itself is the memory bottleneck.
  - `"lsqr"`: `algebra.solvers.lsqr` with `damp = √λ` (the damped
    least-squares equivalence, Paige-Saunders 1982 §1). Works directly
    on `A` (condition number not squared) and on sparse `A`.

`method="auto"` dispatch rule (benchmark-derived, see
`notes/benchmark_2026-07-08_solvers.md`): sparse `A` → `"lsqr"`
(Cholesky needs a dense Gram matrix); dense `A` → `"cholesky"`. In the
sweep, Cholesky dominated the iterative methods by 3-24x up to n=256
and stayed within noise of them (±25%) at n=1024-2048; its runtime is
also condition-number-independent (fixed flops), while CG/LSQR
iteration counts grow with `√cond`. No crossing justifies a size
threshold, so there is none to tune.

References:
  Tikhonov, A. N. (1963). Solution of incorrectly formulated problems
    and the regularization method. Soviet Math. Dokl. 4:1035-1038.
  Björck, Å. (1996). Numerical Methods for Least Squares Problems.
    SIAM, §2.7 (regularized LS, normal-equations vs damped-LSQR
    trade-offs).
  Paige-Saunders (1982). ACM TOMS 8(1):43-71, §1 (LSQR's damp
    parameter solves the ridge problem).
  Boyd, S., Vandenberghe, L. (2004). Convex Optimization. Cambridge
    UP, §6.3.2 (ridge/Tikhonov as a convex regularized approximation
    problem).
"""

from __future__ import annotations

import math
from typing import Literal

import torch

from holonomy_lib.algebra.solvers import cg, lsqr

_SPARSE_LAYOUTS = (
    torch.sparse_coo, torch.sparse_csr, torch.sparse_csc,
)

Method = Literal["auto", "cholesky", "cg", "lsqr"]


def regularized_lstsq(
    A: torch.Tensor,
    b: torch.Tensor,
    lam: float,
    method: Method = "auto",
    rtol: float | None = None,
    maxiter: int | None = None,
) -> torch.Tensor:
    """Solve the ridge problem `min_x ‖A x − b‖² + λ ‖x‖²`.

    Strictly convex for `λ > 0` (unique solution regardless of `A`'s
    rank); for `λ = 0` reduces to ordinary least squares and requires
    full column rank on the `"cholesky"` path (`"lsqr"` handles the
    rank-deficient case, converging to the minimum-norm solution).

    Args:
      A: dense `(m, n)` / `(B, m, n)`, or sparse 2-D `(m, n)`
        (COO/CSR/CSC; dispatched to `"lsqr"`/`"cg"`; `"cholesky"`
        rejects sparse input).
      b: `(B, m)` or `(m,)` observations.
      lam: ridge weight `λ ≥ 0`.
      method: `"auto"` (see module docstring for the dispatch rule),
        `"cholesky"`, `"cg"`, or `"lsqr"`.
      rtol: convergence tolerance for the iterative paths (forwarded to
        `cg` as `rtol`, to `lsqr` as `atol`/`btol`); ignored by
        `"cholesky"`. Default `sqrt(eps(dtype))` (derived).
      maxiter: iteration cap for the iterative paths; ignored by
        `"cholesky"`.

    Returns:
      `x`: `(B, n)` (or `(n,)` for unbatched `b`), minimizing the ridge
      objective.

    References:
      Tikhonov (1963); Björck (1996) §2.7; Paige-Saunders (1982) §1.
      See the module docstring.
    """
    if method not in ("auto", "cholesky", "cg", "lsqr"):
        raise ValueError(
            f"method must be one of 'auto', 'cholesky', 'cg', 'lsqr'; "
            f"got method={method!r}"
        )
    if not isinstance(A, torch.Tensor):
        raise ValueError(f"A must be a torch.Tensor; got {type(A)}")
    if lam < 0:
        raise ValueError(f"lam must be >= 0; got {lam}")

    sparse = A.layout in _SPARSE_LAYOUTS
    # Cholesky forms a dense Gram matrix, so under "auto" sparse A never
    # takes that route.
    if method == "auto":
        method = "lsqr" if sparse else "cholesky"
    if sparse and method == "cholesky":
        raise ValueError(
            "method='cholesky' requires dense A (it factorizes the "
            "dense Gram matrix); use 'lsqr' or 'cg' for sparse input"
        )

    if method == "cholesky":
        if A.ndim < 2:
            raise ValueError(
                f"dense A must be (..., m, n); got A.shape={tuple(A.shape)}"
            )
        n = A.shape[-1]
        # A lone right-hand side (m,) is lifted to (1, m) so the batched code
        # below handles it. The batch axis is dropped again at the return.
        squeeze = b.ndim == 1
        b2 = b.unsqueeze(0) if squeeze else b
        gram = A.mT @ A                                   # (..., n, n)
        eye = torch.eye(n, dtype=A.dtype, device=A.device)
        # λI makes S positive definite even when A is rank deficient. Cholesky
        # can therefore only fail for λ = 0 with a singular AᵀA, which is the
        # full-column-rank requirement in the docstring.
        S = gram + lam * eye
        Atb = (A.mT @ b2.unsqueeze(-1)).squeeze(-1)       # (..., n) or (B, n)
        L = torch.linalg.cholesky(S)
        x = torch.cholesky_solve(Atb.unsqueeze(-1), L).squeeze(-1)
        return x.squeeze(0) if squeeze and x.ndim > 1 else x

    if method == "lsqr":
        # Damped LSQR minimizes ‖Ax − b‖² + damp²‖x‖², so damp = √λ gives the
        # ridge objective exactly.
        return lsqr(
            A, b, damp=math.sqrt(lam),
            atol=rtol, btol=rtol, maxiter=maxiter,
        )

    # method == "cg": CG on the normal operator v ↦ Aᵀ(Av) + λv. The
    # Gram matrix is never materialized; `A` may be dense or sparse.
    if sparse:
        # One COO copy of A and one of Aᵀ, built once here, so every matvec
        # inside cg is a sparse mm rather than a fresh conversion.
        A_coo = (A if A.layout == torch.sparse_coo
                 else A.to_sparse_coo()).coalesce()
        At = A_coo.t().coalesce()

        def normal_op(v: torch.Tensor) -> torch.Tensor:
            Av = torch.sparse.mm(A_coo, v.mT).mT
            return torch.sparse.mm(At, Av.mT).mT + lam * v

        def rproject(u: torch.Tensor) -> torch.Tensor:
            return torch.sparse.mm(At, u.mT).mT
    else:

        def normal_op(v: torch.Tensor) -> torch.Tensor:
            Av = (A @ v.unsqueeze(-1)).squeeze(-1)
            return (A.mT @ Av.unsqueeze(-1)).squeeze(-1) + lam * v

        def rproject(u: torch.Tensor) -> torch.Tensor:
            return (A.mT @ u.unsqueeze(-1)).squeeze(-1)

    # A lone b is lifted to (1, m) and squeezed back at the end, as in the
    # cholesky path. The right-hand side of the normal equations is Aᵀb.
    squeeze = b.ndim == 1
    b2 = b.unsqueeze(0) if squeeze else b
    x = cg(normal_op, rproject(b2), rtol=rtol, maxiter=maxiter)
    return x.squeeze(0) if squeeze else x
