# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Synoros

"""Iterative Krylov solvers: conjugate gradient (CG) and LSQR.

The two workhorse iterative solvers for the library's linear systems:

  - `cg`: symmetric positive-definite `A x = b` (Hestenes-Stiefel 1952).
    Batched-first over dense operators, single-instance over sparse,
    and operator-form via a callable matvec; the same operator can be
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
space), LSQR's to `2·n` (SciPy's convention).

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

Operator = torch.Tensor | Callable[[torch.Tensor], torch.Tensor]


def _default_tol(dtype: torch.dtype) -> float:
    """`sqrt(eps(dtype))`; derived default tolerance (module docstring)."""
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


def _safe_norm(v: torch.Tensor) -> torch.Tensor:
    """Euclidean norm without unscaled squares; empty vectors have norm zero."""
    if v.shape[-1] == 0:
        return v.new_zeros(v.shape[:-1])
    scale = v.abs().amax(dim=-1)
    denominator = torch.where(scale > 0, scale, torch.ones_like(scale))
    return scale * torch.linalg.vector_norm(v / denominator.unsqueeze(-1), dim=-1)


def _rhs_scale(b: torch.Tensor) -> torch.Tensor:
    """Per-member maximum magnitude, with unit scale for a zero right-hand side."""
    scale = b.abs().amax(dim=-1)
    return torch.where(scale > 0, scale, torch.ones_like(scale))


def _scale_matrix(
    A: torch.Tensor, damp: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scale an operator and its damping together without densifying sparse inputs."""
    if A.layout in _SPARSE_LAYOUTS:
        coo = A.to_sparse_coo().coalesce()
        values = coo.values()
        scale = values.abs().amax() if values.numel() else values.new_zeros(())
        scale = scale.clamp(min=damp)
        scale = torch.where(scale > 0, scale, torch.ones_like(scale))
        scaled = torch.sparse_coo_tensor(
            coo.indices(), values / scale, coo.shape, device=A.device, dtype=A.dtype,
            check_invariants=False,
        ).coalesce()
        return scaled, scale
    scale = A.abs().amax(dim=(-2, -1)).clamp(min=damp)
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    return A / scale[..., None, None], scale


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
        `sqrt(eps(dtype))` (derived; see module docstring).
      atol: absolute tolerance floor (default 0).
      maxiter: iteration cap; default `n` (finite-termination bound).
      sync_every: convergence-check cadence (CONVENTIONS §4.1).

    Iterations are sequential Krylov recurrences; all batch members advance
    together, with converged members frozen. Tensor operators and right-hand
    sides are scaled before iteration to avoid overflow in inner products.

    Returns:
      `x` with `b`'s shape. Warns (`UserWarning`) if any batch element
      has not met the tolerance at `maxiter`, or a nonfinite iterate occurs.

    References:
      Hestenes-Stiefel (1952); Saad (2003) §6.7, §6.11.
    """
    b2, squeeze = _normalize_rhs(b)
    B, n = b2.shape
    matvec = _sym_matvec(A)
    if isinstance(A, torch.Tensor) and A.shape[-1] != n:
        raise ValueError(f"A last dimension must match b: {A.shape}, {b.shape}")
    if not b2.is_floating_point():
        raise ValueError(f"b must be floating point; got dtype={b2.dtype}")
    if sync_every < 1:
        raise ValueError(f"sync_every must be >= 1; got {sync_every}")
    if rtol is None:
        rtol = _default_tol(b2.dtype)
    if maxiter is None:
        maxiter = n
    if rtol < 0 or atol < 0 or maxiter < 0:
        raise ValueError("rtol, atol and maxiter must be nonnegative")
    if B == 0 or n == 0:
        return b.clone()

    # Solve in relative units so large finite right-hand sides do not overflow
    # when their entries are multiplied in the CG recurrence.
    bscale = _rhs_scale(b2)
    solution_scale = bscale
    if isinstance(A, torch.Tensor):
        scaled_A, ascale = _scale_matrix(A)
        matvec = _sym_matvec(scaled_A)
        solution_scale = bscale / ascale
    rhs = b2 / bscale.unsqueeze(-1)

    def _precondition(residual: torch.Tensor) -> torch.Tensor:
        z = M(residual) if M is not None else residual
        # A scalar rescaling of z also rescales r·z and the next search
        # direction, so the CG step is unchanged. Keeping z in unit range
        # avoids overflow for either a large or a small preconditioner.
        return z / _rhs_scale(z).unsqueeze(-1)

    x = (torch.zeros_like(rhs) if x0 is None
         else x0.reshape(B, n) / solution_scale.unsqueeze(-1))
    r = rhs - matvec(x)
    z = _precondition(r)
    p = z.clone()
    rz = (r * z).sum(dim=-1)
    target = torch.maximum(_safe_norm(rhs) * rtol, atol / bscale)
    finite = torch.isfinite(x).all(dim=-1) & torch.isfinite(r).all(dim=-1)
    done = finite & (_safe_norm(r) <= target)
    failed = ~finite
    n_done = 0
    for it in range(1, maxiter + 1):
        active = ~(done | failed)
        Ap = matvec(p)
        pAp = (p * Ap).sum(dim=-1)
        denominator = torch.where(active & (pAp != 0), pAp, torch.ones_like(pAp))
        alpha = torch.where(active, rz / denominator, torch.zeros_like(rz))
        candidate = x + alpha.unsqueeze(-1) * p
        r_new = r - alpha.unsqueeze(-1) * Ap
        finite = (torch.isfinite(candidate).all(dim=-1)
                  & torch.isfinite(r_new).all(dim=-1) & torch.isfinite(pAp))
        failed = failed | (active & (~finite | (pAp <= 0)))
        x = torch.where(active.unsqueeze(-1), candidate, x)
        r = torch.where(active.unsqueeze(-1), r_new, r)
        done = done | (active & ~failed & (_safe_norm(r) <= target))
        z = _precondition(r)
        rz_new = (r * z).sum(dim=-1)
        denominator = torch.where(rz != 0, rz, torch.ones_like(rz))
        beta = rz_new / denominator
        p = torch.where((~(done | failed)).unsqueeze(-1),
                        z + beta.unsqueeze(-1) * p, torch.zeros_like(p))
        rz = rz_new
        n_done = it
        if it % sync_every == 0 or it == maxiter:
            if bool((done | failed).all()):
                break

    result = x * solution_scale.unsqueeze(-1)
    finite_result = torch.isfinite(result).all(dim=-1)
    failed = failed | ~finite_result
    unsuccessful = ~done | failed
    if bool(unsuccessful.any()):
        warnings.warn(
            f"cg: {int(unsuccessful.sum())}/{B} batch elements not converged "
            f"after {n_done} iterations (nonfinite or breakdown failures: "
            f"{int(failed.sum())}). Result returned as-is; raise maxiter, "
            "loosen rtol, or precondition.",
            stacklevel=2,
        )
    return result.squeeze(0) if squeeze else result


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

    Stopping follows Paige-Saunders §6, tests 1-2 (without SciPy's
    condition-limit test):
      1. compatible systems: `‖r‖ ≤ btol·‖b‖ + atol·‖A‖·‖x‖`
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
        (derived; see module docstring).
      maxiter: iteration cap; default `2·n` (SciPy convention).
      sync_every: convergence-check cadence (CONVENTIONS §4.1).

    Iterations are sequential Golub-Kahan recurrences. Operators and right-hand
    sides are scaled before iteration; rotations use overflow-safe hypotenuses.
    Batch members that converge stop updating their solution.

    Returns:
      `x`: `(B, n)` (or `(n,)` for unbatched `b`). Warns
      (`UserWarning`) if any batch element met neither stopping test
      at `maxiter`, or a nonfinite iterate occurs.

    References:
      Paige-Saunders (1982), ACM TOMS 8(1):43-71, Alg. LSQR + §6
        (stopping criteria). Reference implementation compared against:
        `scipy.sparse.linalg.lsqr` (tests).
    """
    b2, squeeze = _normalize_rhs(b)
    B = b2.shape[0]
    _, _, m, n = _lsqr_matvecs(A)
    if b2.shape[-1] != m:
        raise ValueError(
            f"b last dim must equal A's row count m={m}; got "
            f"b.shape={tuple(b.shape)}"
        )
    if not b2.is_floating_point() or not A.is_floating_point():
        raise ValueError("A and b must be floating point")
    if damp < 0:
        raise ValueError(f"damp must be >= 0; got {damp}")
    if sync_every < 1:
        raise ValueError(f"sync_every must be >= 1; got {sync_every}")
    if atol is None:
        atol = _default_tol(b2.dtype)
    if btol is None:
        btol = _default_tol(b2.dtype)
    if maxiter is None:
        maxiter = 2 * n
    if atol < 0 or btol < 0 or maxiter < 0:
        raise ValueError("atol, btol and maxiter must be nonnegative")
    if B == 0 or m == 0 or n == 0:
        out = b2.new_zeros(B, n)
        return out.squeeze(0) if squeeze else out

    scaled_A, ascale = _scale_matrix(A, damp=damp)
    matvec, rmatvec, _, _ = _lsqr_matvecs(scaled_A)
    bscale = _rhs_scale(b2)
    rhs = b2 / bscale.unsqueeze(-1)
    damping = damp / ascale

    def _safe_normalize(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        nrm = _safe_norm(t)
        denominator = torch.where(nrm > 0, nrm, torch.ones_like(nrm))
        return t / denominator.unsqueeze(-1), nrm

    def _denominator(t: torch.Tensor) -> torch.Tensor:
        return torch.where(t != 0, t, torch.ones_like(t))

    def _hypot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # hypot(0, 0) has undefined derivatives even when a later mask
        # discards its value. Supply a safe pair before calling it.
        nonzero = (a != 0) | (b != 0)
        safe_a = torch.where(nonzero, a, torch.ones_like(a))
        safe_b = torch.where(nonzero, b, torch.zeros_like(b))
        return torch.where(nonzero, torch.hypot(safe_a, safe_b), 0.0)

    u, beta = _safe_normalize(rhs)
    bnorm = beta.clone()
    v, alfa = _safe_normalize(rmatvec(u))
    w = v.clone()
    x = b2.new_zeros(B, n)
    phibar = beta.clone()
    rhobar = alfa.clone()
    anorm = torch.zeros_like(beta)
    damped_residual = torch.zeros_like(beta)
    finite = torch.isfinite(beta) & torch.isfinite(alfa)
    done = finite & ((beta == 0) | (alfa == 0))
    failed = ~finite
    n_done = 0
    for itn in range(1, maxiter + 1):
        active = ~(done | failed)
        u, beta = _safe_normalize(matvec(v) - alfa.unsqueeze(-1) * u)
        anorm = _hypot(_hypot(anorm, alfa), _hypot(beta, damping))
        v, alfa = _safe_normalize(rmatvec(u) - beta.unsqueeze(-1) * v)

        # Finished members still participate until the next host sync.
        # Give them an identity rotation before any hypot or division.
        rhobar = torch.where(active, rhobar, torch.ones_like(rhobar))
        beta = torch.where(active, beta, 0.0)
        alfa = torch.where(active, alfa, 0.0)
        active_damping = torch.where(active, damping, 0.0)
        phibar = torch.where(active, phibar, 0.0)
        # Both rotations use hypot: squaring a finite coefficient need not
        # produce a finite intermediate, even when the rotation is well-defined.
        rhobar1 = _hypot(rhobar, active_damping)
        cs1 = rhobar / _denominator(rhobar1)
        psi = (active_damping / _denominator(rhobar1)) * phibar
        phibar = cs1 * phibar
        rho = _hypot(rhobar1, beta)
        cs = rhobar1 / _denominator(rho)
        sn = beta / _denominator(rho)
        theta = sn * alfa
        rhobar = -cs * alfa
        phi = cs * phibar
        phibar = sn * phibar

        candidate = x + (phi / _denominator(rho)).unsqueeze(-1) * w
        x = torch.where(active.unsqueeze(-1), candidate, x)
        w = v - (theta / _denominator(rho)).unsqueeze(-1) * w
        damped_residual = _hypot(damped_residual, psi)
        rnorm = _hypot(phibar, damped_residual)
        arnorm = alfa * (sn * phi).abs()
        finite = (torch.isfinite(x).all(dim=-1) & torch.isfinite(rnorm)
                  & torch.isfinite(arnorm) & torch.isfinite(anorm)
                  & torch.isfinite(u).all(dim=-1) & torch.isfinite(v).all(dim=-1))
        failed = failed | (active & ~finite)
        compatible = rnorm <= btol * bnorm + atol * anorm * _safe_norm(x)
        least_squares = arnorm <= atol * anorm * rnorm
        done = done | (active & finite & (compatible | least_squares))
        n_done = itn
        if itn % sync_every == 0 or itn == maxiter:
            if bool((done | failed).all()):
                break

    result = x * (bscale / ascale).unsqueeze(-1)
    failed = failed | ~torch.isfinite(result).all(dim=-1)
    unsuccessful = ~done | failed
    if bool(unsuccessful.any()):
        warnings.warn(
            f"lsqr: {int(unsuccessful.sum())}/{B} batch elements met neither "
            f"stopping test after {n_done} iterations (nonfinite failures: "
            f"{int(failed.sum())}). Result returned as-is; raise maxiter "
            "or loosen atol/btol.",
            stacklevel=2,
        )
    return result.squeeze(0) if squeeze else result
