# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Synoros

"""Iterative Krylov solvers: conjugate gradient (CG) and LSQR.

The two workhorse iterative solvers for the library's linear systems:

  - `cg`: symmetric positive-definite `A x = b` (Hestenes-Stiefel 1952).
    Batched-first over dense operators, single-instance over sparse,
    and operator-form via a callable matvec — the same operator can be
    "A is never materialized" (e.g. normal equations `AᵀA + λI` applied
    as two matvecs, or a sheaf Laplacian applied as `δᵀ(δx)`).
  - `lsqr`: least squares `min ‖Ax − b‖` for rectangular `A`, with
    optional Tikhonov damping `min ‖Ax − b‖² + damp²·‖x‖²`
    (Paige-Saunders 1982, via Golub-Kahan bidiagonalization). More
    numerically reliable than solving the normal equations when `A` is
    ill-conditioned: it works with `A` directly, never squaring the
    condition number.

Defaults are derived, not tuned: convergence tolerances default to
`sqrt(eps(dtype))` (the point past which normal-equation-quality error
stops improving; ≈1.5e-8 for float64, ≈3.5e-4 for float32), CG's
iteration cap defaults to `n` (finite-termination bound of the Krylov
space), LSQR's to `2·min(m, n)` (SciPy's convention).

Convergence checks force a device sync (`.item()`), so they run every
`sync_every` iterations (CONVENTIONS §4.1; same cadence pattern as
Sinkhorn's `SINKHORN_SYNC_EVERY_DEFAULT`).

References:
  Hestenes, M. R., Stiefel, E. (1952). Methods of conjugate gradients
    for solving linear systems. J. Res. Natl. Bur. Stand. 49(6):409-436.
  Paige, C. C., Saunders, M. A. (1982). LSQR: an algorithm for sparse
    linear equations and sparse least squares. ACM TOMS 8(1):43-71.
  Saad, Y. (2003). Iterative Methods for Sparse Linear Systems, 2nd ed.
    SIAM, §6.7 (CG convergence theory).
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Union

import torch

# Cadence of the host-syncing convergence check inside the solver
# loops. Same value and rationale as the established
# `SINKHORN_SYNC_EVERY_DEFAULT` (CONVENTIONS §4.1): checking every
# iteration serializes GPU work on a `.item()` sync; every 8 amortizes
# the sync to ~12% of iterations while wasting at most 7 cheap
# iterations past convergence. Catalog: `SOLVER_SYNC_EVERY_DEFAULT`
# in `notes/magic_numbers.md`.
SOLVER_SYNC_EVERY_DEFAULT: int = 8

_SPARSE_LAYOUTS = (
    torch.sparse_coo, torch.sparse_csr, torch.sparse_csc,
)

Operator = Union[torch.Tensor, Callable[[torch.Tensor], torch.Tensor]]


def _default_tol(dtype: torch.dtype) -> float:
    """`sqrt(eps(dtype))` — derived default tolerance (module docstring)."""
    return float(torch.finfo(dtype).eps) ** 0.5


def _sym_matvec(A: Operator) -> Callable[[torch.Tensor], torch.Tensor]:
    """Normalize a symmetric operator to a `(B, n) → (B, n)` matvec.

    Accepts a callable (used as-is), a sparse 2-D `(n, n)` tensor
    (applied to the batch via one sparse×dense matmul), or a dense
    `(n, n)` / `(B, n, n)` tensor (batched matmul, broadcasting an
    unbatched A over the batch of right-hand sides).
    """
    if callable(A):
        return A
    if not isinstance(A, torch.Tensor):
        raise ValueError(
            f"A must be a torch.Tensor or a callable matvec; got {type(A)}"
        )
    if A.layout in _SPARSE_LAYOUTS:
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError(
                f"sparse A must be (n, n); got A.shape={tuple(A.shape)}"
            )
        A_coo = (
            A if A.layout != torch.sparse_csc else A.to_sparse_csr()
        )
        return lambda v: torch.sparse.mm(A_coo, v.mT).mT
    if A.ndim < 2 or A.shape[-1] != A.shape[-2]:
        raise ValueError(
            f"dense A must be (..., n, n); got A.shape={tuple(A.shape)}"
        )
    return lambda v: (A @ v.unsqueeze(-1)).squeeze(-1)


def _normalize_rhs(b: torch.Tensor) -> tuple[torch.Tensor, bool]:
    """`(n,) → (1, n)` with a flag to squeeze the output back."""
    if b.ndim == 1:
        return b.unsqueeze(0), True
    if b.ndim != 2:
        raise ValueError(
            f"b must be (n,) or (B, n); got b.shape={tuple(b.shape)}"
        )
    return b, False


def cg(
    A: Operator,
    b: torch.Tensor,
    M: Callable[[torch.Tensor], torch.Tensor] | None = None,
    x0: torch.Tensor | None = None,
    rtol: float | None = None,
    atol: float = 0.0,
    maxiter: int | None = None,
    sync_every: int = SOLVER_SYNC_EVERY_DEFAULT,
) -> torch.Tensor:
    """Conjugate gradient solve of symmetric positive-definite `A x = b`.

    Math: minimizes the quadratic `½ xᵀAx − bᵀx` over expanding Krylov
    subspaces; in exact arithmetic terminates in at most `n` steps, in
    floating point converges at rate `((√κ − 1)/(√κ + 1))^k` with
    `κ = cond(A)` (Saad 2003, §6.11).

    Args:
      A: the operator. One of: dense `(n, n)` or `(B, n, n)` tensor;
        sparse `(n, n)` (COO/CSR/CSC; applied to the whole batch in one
        sparse matmul); or a callable `(B, n) → (B, n)` matvec (for
        never-materialized operators). Symmetric PD assumed,
        not checked.
      b: `(B, n)` or `(n,)` right-hand sides.
      M: optional preconditioner as a callable `(B, n) → (B, n)`
        applying `M⁻¹` (e.g. a Jacobi inverse-diagonal multiply).
      x0: optional `(B, n)` / `(n,)` initial guess (default zeros).
      rtol: relative tolerance on `‖r‖ ≤ max(rtol·‖b‖, atol)`; default
        `sqrt(eps(dtype))` (derived — see module docstring).
      atol: absolute tolerance floor (default 0).
      maxiter: iteration cap; default `n` (finite-termination bound).
      sync_every: convergence-check cadence (CONVENTIONS §4.1).

    Returns:
      `x` with `b`'s shape. Warns (`UserWarning`) if any batch element
      has not met the tolerance at `maxiter`.

    References:
      Hestenes-Stiefel (1952); Saad (2003) §6.7, §6.11.
    """
    b2, squeeze = _normalize_rhs(b)
    B, n = b2.shape
    if not b2.is_floating_point():
        raise ValueError(f"b must be floating point; got dtype={b2.dtype}")
    if sync_every < 1:
        raise ValueError(f"sync_every must be >= 1; got {sync_every}")
    if B == 0:
        return b.clone()

    matvec = _sym_matvec(A)
    if rtol is None:
        rtol = _default_tol(b2.dtype)
    if maxiter is None:
        maxiter = n
    tiny = torch.finfo(b2.dtype).tiny

    x = torch.zeros_like(b2) if x0 is None else x0.reshape(B, n).clone()
    r = b2 - matvec(x)
    z = M(r) if M is not None else r
    p = z.clone()
    rz = (r * z).sum(dim=-1)                          # (B,)
    # Per-element residual target ‖r‖ ≤ max(rtol·‖b‖, atol).
    target = torch.linalg.norm(b2, dim=-1) * rtol
    target = torch.clamp(target, min=atol)

    n_done = 0
    for it in range(1, maxiter + 1):
        Ap = matvec(p)
        pAp = (p * Ap).sum(dim=-1)
        alpha = rz / pAp.clamp(min=tiny)              # (B,)
        x = x + alpha.unsqueeze(-1) * p
        r = r - alpha.unsqueeze(-1) * Ap
        z = M(r) if M is not None else r
        rz_new = (r * z).sum(dim=-1)
        beta = rz_new / rz.clamp(min=tiny)
        p = z + beta.unsqueeze(-1) * p
        rz = rz_new
        if it % sync_every == 0 or it == maxiter:
            n_done = it
            if bool(
                (torch.linalg.norm(r, dim=-1) <= target).all()
            ):
                break

    resid = torch.linalg.norm(r, dim=-1)
    if bool((resid > target).any()):
        worst = float((resid / target.clamp(min=tiny)).max())
        warnings.warn(
            f"cg: {int((resid > target).sum())}/{B} batch elements not "
            f"converged after {n_done} iterations (worst residual "
            f"{worst:.2e}x its target). Result returned as-is; raise "
            f"maxiter, loosen rtol, or precondition.",
            stacklevel=2,
        )
    return x.squeeze(0) if squeeze else x


def _lsqr_matvecs(
    A: torch.Tensor,
) -> tuple[
    Callable[[torch.Tensor], torch.Tensor],
    Callable[[torch.Tensor], torch.Tensor],
    int,
    int,
]:
    """`(matvec, rmatvec, m, n)` for dense batched or sparse 2-D `A`."""
    if not isinstance(A, torch.Tensor):
        raise ValueError(
            f"A must be a torch.Tensor for lsqr; got {type(A)}"
        )
    if A.layout in _SPARSE_LAYOUTS:
        if A.ndim != 2:
            raise ValueError(
                f"sparse A must be 2-D (m, n); got A.shape={tuple(A.shape)}"
            )
        m, n = A.shape
        A_coo = (A if A.layout == torch.sparse_coo
                 else A.to_sparse_coo()).coalesce()
        At = A_coo.t().coalesce()
        return (
            lambda v: torch.sparse.mm(A_coo, v.mT).mT,
            lambda u: torch.sparse.mm(At, u.mT).mT,
            m, n,
        )
    if A.ndim < 2:
        raise ValueError(
            f"dense A must be (..., m, n); got A.shape={tuple(A.shape)}"
        )
    m, n = A.shape[-2], A.shape[-1]
    return (
        lambda v: (A @ v.unsqueeze(-1)).squeeze(-1),
        lambda u: (A.mT @ u.unsqueeze(-1)).squeeze(-1),
        m, n,
    )


def lsqr(
    A: torch.Tensor,
    b: torch.Tensor,
    damp: float = 0.0,
    atol: float | None = None,
    btol: float | None = None,
    maxiter: int | None = None,
    sync_every: int = SOLVER_SYNC_EVERY_DEFAULT,
) -> torch.Tensor:
    """LSQR: iterative least squares `min ‖Ax − b‖²  (+ damp²·‖x‖²)`.

    Golub-Kahan bidiagonalization of `A` with QR-by-Givens updates
    (Paige-Saunders 1982, Alg. LSQR). Analytically equivalent to CG on
    the normal equations `(AᵀA + damp²·I) x = Aᵀb` but numerically
    preferable: it never forms `AᵀA`, so the effective condition number
    is `cond(A)`, not `cond(A)²`. Handles rank-deficient and
    inconsistent systems (converges to the minimum-norm least-squares
    solution).

    Stopping follows Paige-Saunders §6, tests 1-2 (SciPy's `atol`/`btol`
    semantics, without the condition-limit test):
      1. compatible systems: `‖r‖ ≤ btol·‖b‖`
      2. least-squares:      `‖Aᵀr‖ ≤ atol·‖A‖·‖r‖`
    with `‖A‖` the running Frobenius estimate from the bidiagonalization.

    Args:
      A: dense `(m, n)` / `(B, m, n)`, or sparse 2-D `(m, n)` (applied
        to the whole batch via sparse matmuls). Callables are not
        accepted here (LSQR needs the adjoint too; pass a tensor).
      b: `(B, m)` or `(m,)`.
      damp: Tikhonov damping `≥ 0`; `damp = √λ` solves the ridge
        problem `min ‖Ax − b‖² + λ‖x‖²`.
      atol, btol: stopping tolerances; default `sqrt(eps(dtype))`
        (derived — see module docstring).
      maxiter: iteration cap; default `2·min(m, n)` (SciPy convention).
      sync_every: convergence-check cadence (CONVENTIONS §4.1).

    Returns:
      `x`: `(B, n)` (or `(n,)` for unbatched `b`). Warns
      (`UserWarning`) if any batch element met neither stopping test
      at `maxiter`.

    References:
      Paige-Saunders (1982), ACM TOMS 8(1):43-71, Alg. LSQR + §6
        (stopping criteria). Reference implementation compared against:
        `scipy.sparse.linalg.lsqr` (tests).
    """
    b2, squeeze = _normalize_rhs(b)
    B = b2.shape[0]
    if not b2.is_floating_point():
        raise ValueError(f"b must be floating point; got dtype={b2.dtype}")
    if damp < 0:
        raise ValueError(f"damp must be >= 0; got {damp}")
    if sync_every < 1:
        raise ValueError(f"sync_every must be >= 1; got {sync_every}")

    matvec, rmatvec, m, n = _lsqr_matvecs(A)
    if b2.shape[-1] != m:
        raise ValueError(
            f"b last dim must equal A's row count m={m}; got "
            f"b.shape={tuple(b.shape)}"
        )
    if B == 0:
        out = b2.new_zeros(B, n)
        return out.squeeze(0) if squeeze else out

    dtype = b2.dtype
    if atol is None:
        atol = _default_tol(dtype)
    if btol is None:
        btol = _default_tol(dtype)
    if maxiter is None:
        maxiter = 2 * min(m, n)
    tiny = torch.finfo(dtype).tiny

    def _safe_normalize(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        nrm = torch.linalg.norm(t, dim=-1)            # (B,)
        out = torch.where(
            (nrm > tiny).unsqueeze(-1), t / nrm.clamp(min=tiny).unsqueeze(-1),
            torch.zeros_like(t),
        )
        return out, nrm

    # --- Golub-Kahan initialization (Paige-Saunders §3) ---
    u, beta = _safe_normalize(b2)                     # (B, m), (B,)
    bnorm = beta.clone()
    v, alfa = _safe_normalize(rmatvec(u))             # (B, n), (B,)
    w = v.clone()
    x = b2.new_zeros(B, n)
    phibar = beta.clone()
    rhobar = alfa.clone()
    anorm2 = torch.zeros_like(beta)                   # running ‖A‖_F² estimate
    res2 = torch.zeros_like(beta)                     # Σ ψ² (damped residual part)

    # Per-element convergence mask, maintained ON DEVICE every iteration
    # (cheap (B,) ops, no host sync): once an element passes a stopping
    # test its x is frozen. Iterating a converged element past Krylov
    # breakdown is not benign here — on inconsistent systems `phibar`
    # stays at ‖r‖ > 0 while `rho` collapses to rounding noise, so an
    # unfrozen `phi/rho` update explodes (SciPy avoids this by testing
    # every iteration; we test every iteration too, but only pay the
    # `.item()` host sync on the `sync_every` cadence).
    done = torch.zeros(B, dtype=torch.bool, device=b2.device)
    n_done = 0
    for itn in range(1, maxiter + 1):
        # Bidiagonalization step: next left/right vectors.
        u = matvec(v) - alfa.unsqueeze(-1) * u
        u, beta = _safe_normalize(u)
        anorm2 = anorm2 + alfa * alfa + beta * beta + damp * damp
        v = rmatvec(u) - beta.unsqueeze(-1) * v
        v, alfa = _safe_normalize(v)

        # Eliminate the damping row (extra Givens rotation).
        rhobar1 = torch.sqrt(rhobar * rhobar + damp * damp)
        cs1 = rhobar / rhobar1.clamp(min=tiny)
        psi = (damp / rhobar1.clamp(min=tiny)) * phibar
        phibar = cs1 * phibar

        # QR step on the bidiagonal matrix (plane rotation).
        rho = torch.sqrt(rhobar1 * rhobar1 + beta * beta)
        cs = rhobar1 / rho.clamp(min=tiny)
        sn = beta / rho.clamp(min=tiny)
        theta = sn * alfa
        rhobar = -cs * alfa
        phi = cs * phibar
        phibar = sn * phibar

        step = torch.where(done, torch.zeros_like(phi), phi / rho.clamp(min=tiny))
        x = x + step.unsqueeze(-1) * w
        w = v - (theta / rho.clamp(min=tiny)).unsqueeze(-1) * w

        # Residual estimates (Paige-Saunders §6): ‖r‖² = φ̄² + Σψ²,
        # ‖Aᵀr‖ = φ̄·α·|cos| of the current rotation.
        res2 = res2 + psi * psi
        rnorm_it = torch.sqrt(phibar * phibar + res2)
        arnorm_it = alfa * (sn * phi).abs()
        anorm = torch.sqrt(anorm2)
        # `done` is monotonic: estimates of already-done elements keep
        # evolving as garbage past breakdown, but cannot un-set the flag.
        done = done | (rnorm_it <= btol * bnorm) | (
            arnorm_it <= atol * anorm * rnorm_it.clamp(min=tiny)
        )

        if itn % sync_every == 0 or itn == maxiter:
            n_done = itn
            if bool(done.all()):
                break

    if bool((~done).any()):
        warnings.warn(
            f"lsqr: {int((~done).sum())}/{B} batch elements met neither "
            f"stopping test after {n_done} iterations. Result returned "
            f"as-is; raise maxiter or loosen atol/btol.",
            stacklevel=2,
        )
    return x.squeeze(0) if squeeze else x
