# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Synoros

"""Batched symmetric eigenpairs with fully reorthogonalized Lanczos.

The Krylov recurrence is sequential in the subspace dimension; each step
applies the operator and orthogonalizes all batch members together. A
member whose Krylov space closes restarts in its orthogonal complement,
with zero coupling between the two Lanczos blocks. Full-subspace requests
use ``torch.linalg.eigh`` directly.

Dense LA and unshifted SA select the largest and smallest algebraic modes.
An explicit dense SA shift selects modes nearest that shift. Sparse SA
uses the backends in :mod:`holonomy_lib.algebra.eigsh_sparse`.

References:
  Lanczos, C. (1950). An iteration method for the solution of the
    eigenvalue problem of linear differential and integral operators.
    Journal of Research of the National Bureau of Standards 45:255-282.
  Paige, C. C. (1972). Computational variants of the Lanczos method
    for the eigenproblem. J. Inst. Math. Appl. 10:373-381.
  Saad, Y. (2011). Numerical Methods for Large Eigenvalue Problems,
    2nd ed. SIAM, §6.5 (full reorthogonalization).
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable
from typing import Literal

import torch

# Extra Krylov dimensions beyond k help the top-k Ritz pairs converge
# (Saad 2011, §6.5). Cataloged as LANCZOS_OVERSAMPLE_DEFAULT.
LANCZOS_OVERSAMPLE_DEFAULT: int = 10

_SPARSE_LAYOUTS = (torch.sparse_coo, torch.sparse_csr, torch.sparse_csc)


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
    """Return extreme symmetric eigenpairs, or dense modes nearest ``sigma``.

    Args:
      A: real symmetric dense ``(..., n, n)`` or sparse ``(n, n)``.
        Asymmetry emits a warning; callers must supply a symmetric operator.
      k: number of eigenpairs, ``1 <= k <= n``.
      n_iter: Krylov dimension, default ``min(k + oversample, n)``.
        A full-dimensional request uses dense ``eigh``.
      oversample: nonnegative extra dimensions when ``n_iter`` is omitted.
      generator: random-start generator compatible with ``A.device``.
      which: ``LA`` selects largest algebraic eigenvalues; ``SA`` selects
        smallest algebraic eigenvalues unless a dense ``sigma`` is supplied.
      sigma: for dense SA, select eigenvalues nearest this finite shift and
        return them in ascending order. Partial-subspace requests factor
        ``A - sigma I`` and reject a singular factorization. For sparse SA,
        an explicit pole must lie below the Gershgorin lower bound, so the
        smallest-algebraic contract is preserved.
      solver: sparse-SA backend; see ``eigsh_sparse_sa``.

    Returns:
      Eigenvalues ``(..., k)`` and eigenvectors ``(..., n, k)``; sparse
      input has no batch dimension. LA is descending, SA ascending.

    Notes:
      Partial-subspace results are Ritz approximations. Increasing
      ``n_iter`` improves convergence; no fixed iteration count guarantees
      accuracy for every spectrum. Shift-invert warns if the original
      operator's relative Ritz residual exceeds ``n * sqrt(eps(dtype))``.
      Dense autograd follows the computed subspace, with the usual
      eigenvector-gradient limitations at repeated eigenvalues. Sparse
      eigensolvers are explicitly nondifferentiable.

    References:
      Lanczos (1950); Paige (1972); Saad (2011), §6.5, cited above.
      Ericsson, T., Ruhe, A. (1980). The spectral transformation Lanczos
        method for the numerical solution of large sparse generalized
        symmetric eigenvalue problems. Math. Comp. 35(152):1251-1268.
    """
    sparse = A.layout in _SPARSE_LAYOUTS
    if A.ndim < 2 or A.shape[-1] != A.shape[-2] or (sparse and A.ndim != 2):
        raise ValueError(f"A must be (..., n, n), sparse (n, n); got {A.shape}")
    if not A.is_floating_point():
        raise ValueError("A must have a real floating dtype")
    n = A.shape[-1]
    if not isinstance(k, int) or isinstance(k, bool) or not 1 <= k <= n:
        raise ValueError(f"k must satisfy 1 <= k <= n={n}; got k={k}")
    if (not isinstance(oversample, int) or isinstance(oversample, bool)
            or oversample < 0):
        raise ValueError("oversample must be a nonnegative integer")
    if which not in ("LA", "SA"):
        raise ValueError("which must be 'LA' or 'SA'")
    if sigma is not None and (which != "SA" or not math.isfinite(sigma)):
        raise ValueError("sigma must be finite and is only meaningful with which='SA'")
    if n_iter is None:
        n_iter = k + oversample
    if not isinstance(n_iter, int) or isinstance(n_iter, bool) or n_iter < k:
        raise ValueError(f"n_iter must be an integer >= k={k}")
    # The default, or an explicit n_iter, can exceed n. Clamping after the
    # k check keeps n_iter >= k, since k <= n. A clamped n_iter == n routes
    # to dense eigh below.
    n_iter = min(n_iter, n)
    # Sparse input can't be differentiated here, so requires_grad is rejected
    # outright instead of being detached silently. SA goes to the dedicated
    # backend, which handles its own shift. LA falls through to the Lanczos
    # loop below, where the matvec is a sparse product.
    if sparse:
        if A.requires_grad:
            raise ValueError(
                "sparse eigensolvers do not support autograd; detach A explicitly"
            )
        if which == "SA":
            from holonomy_lib.algebra.eigsh_sparse import eigsh_sparse_sa

            return eigsh_sparse_sa(
                A, k, sigma=sigma, solver=solver, generator=generator
            )
    _warn_asymmetric(A)
    # A Krylov space of full dimension n already spans everything, so dense
    # eigh gives the same answer with no iteration. An empty batch has nothing
    # to iterate over and takes this path too.
    if n_iter == n or A.numel() == 0:
        dense = A.to_dense() if sparse else A
        values, vectors = torch.linalg.eigh(dense)
        return _select(values, vectors, k, which, sigma)

    # With a shift, Lanczos runs on (A - sigma I)^-1 instead. That turns the
    # modes nearest sigma into the largest inverse eigenvalues. _shift_invert
    # does the factorization and checks the result against the original A.
    if sigma is not None:
        return _shift_invert(A, k, n_iter, sigma, generator)

    def matvec(vector: torch.Tensor) -> torch.Tensor:
        return (A @ vector.unsqueeze(-1)).squeeze(-1)

    values, vectors = _iterate(matvec, A, n_iter, generator)
    return _select(values, vectors, k, which, None)


def _warn_asymmetric(A: torch.Tensor) -> None:
    """Emit a UserWarning when A is not symmetric. Called once per solver call.

    Input is a dense or sparse A. Output is None; the only effect is the
    warning. The check is advisory, so an asymmetric A still reaches the solver.
    """
    if A.layout in _SPARSE_LAYOUTS:
        # Sparse path: compare the stored values of A - A^T without densifying.
        # The coalesce keeps entries that exist in only one of A or A^T, so a
        # missing mirror entry still shows up as a difference.
        coo = A.to_sparse_coo().coalesce()
        difference = (coo - coo.mT).coalesce().values()
        # amax fails on an empty value tensor, so an all-zero matrix gets scale 0.
        scale = coo.values().abs().amax() if coo._nnz() else coo.values().new_zeros(())
        # Tolerance is n * eps * max entry, the size of roundoff in a row sum
        # of n terms.
        tolerance = A.shape[-1] * torch.finfo(A.dtype).eps * scale
        symmetric = bool((difference.abs() <= tolerance).all())
    else:
        # Same rule as the sparse path, per batch member: roundoff in a row of
        # n terms is about n * eps * (largest entry), so anything bigger than
        # that is real asymmetry. Each member uses its own scale so one large
        # matrix doesn't hide asymmetry in a small one.
        scale = A.abs().amax(dim=(-2, -1), keepdim=True)
        tolerance = A.shape[-1] * torch.finfo(A.dtype).eps * scale
        symmetric = bool(((A - A.mT).abs() <= tolerance).all())
    if not symmetric:
        warnings.warn(
            "symmetric eigensolver received asymmetric A", UserWarning, stacklevel=2
        )


def _select(
    values: torch.Tensor,
    vectors: torch.Tensor,
    k: int,
    which: str,
    sigma: float | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select ordered columns from a symmetric eigendecomposition."""
    if sigma is not None:
        # Take the k values nearest sigma, then put those k back in ascending
        # order so every mode returns in the same convention.
        indices = (values - sigma).abs().argsort(dim=-1)[..., :k]
        chosen = values.gather(-1, indices)
        indices = indices.gather(-1, chosen.argsort(dim=-1))
    else:
        indices = values.argsort(dim=-1, descending=which == "LA")[..., :k]
    # Eigenvectors are columns, so the index is broadcast down the n rows
    # before gathering along the column axis.
    return (
        values.gather(-1, indices),
        vectors.gather(-1, indices.unsqueeze(-2).expand(*vectors.shape[:-1], k)),
    )


def _orthogonalize(vector: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """Project `vector` off `basis` with two Gram-Schmidt passes.

    Input: vector (..., n); basis (..., n, j) with orthonormal columns.
    Output: the residual, orthogonal to the basis up to roundoff. The second
    pass removes the roundoff the first one leaves behind.
    """
    for _ in range(2):
        vector = vector - (basis @ (basis.mT @ vector.unsqueeze(-1))).squeeze(-1)
    return vector


def _iterate(
    matvec: Callable[[torch.Tensor], torch.Tensor],
    A: torch.Tensor,
    n_iter: int,
    generator: torch.Generator | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run one Lanczos recurrence per batch member, then diagonalize the projection.

    Input: matvec applies the symmetric operator to (..., n) vectors; A gives the
    shape and dtype; n_iter is the Krylov dimension, at most n; generator seeds
    the random start. Output: Ritz values (..., n_iter), ascending, and Ritz
    vectors (..., n, n_iter). Each member has its own start and its own basis,
    so members never share a Krylov space.
    """
    n = A.shape[-1]
    # One random unit start per batch member, drawn in a single call.
    vector = torch.randn(
        *A.shape[:-1],
        dtype=A.dtype,
        device=A.device,
        generator=generator,
    )
    vector = vector / torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
    columns: list[torch.Tensor] = []
    diagonal: list[torch.Tensor] = []
    off_diagonal: list[torch.Tensor] = []
    # No previous vector before the first step. beta_0 = 0 makes the
    # subtraction below a no-op.
    previous = torch.zeros_like(vector)
    previous_beta = vector.new_zeros(vector.shape[:-1])
    for step in range(n_iter):
        # Every update below builds a new tensor and nothing is written in
        # place, so the vectors kept in `columns` keep their values for autograd.
        columns.append(vector)
        basis = torch.stack(columns, dim=-1)
        applied = matvec(vector)
        alpha = (applied * vector).sum(dim=-1)
        diagonal.append(alpha)
        residual = applied - alpha.unsqueeze(-1) * vector
        residual = residual - previous_beta.unsqueeze(-1) * previous
        residual = _orthogonalize(residual, basis)
        if step + 1 == n_iter:
            break
        beta = torch.linalg.vector_norm(residual, dim=-1)
        scale = torch.linalg.vector_norm(applied, dim=-1)
        # Breakdown: the residual has shrunk to roundoff relative to the size of
        # A v, so this member's Krylov space has closed. Its next vector is a
        # restart, and its beta is zeroed below so the two blocks stay uncoupled.
        breakdown = beta <= n * torch.finfo(A.dtype).eps * scale
        safe_beta = torch.where(breakdown, torch.ones_like(beta), beta)

        # The diagonal of the basis projector sums to the number of basis
        # vectors, which is below n here. So the coordinate with the least mass
        # has mass under 1, and e_i is not entirely inside the span. Projecting
        # e_i off the basis gives a nonzero restart vector.
        coordinate = basis.square().sum(dim=-1).argmin(dim=-1)
        restart = torch.nn.functional.one_hot(coordinate, num_classes=n).to(A.dtype)
        restart = _orthogonalize(restart, basis)
        restart = restart / torch.linalg.vector_norm(restart, dim=-1, keepdim=True)
        # On breakdown the next vector is the restart. Otherwise it is the
        # normalized residual.
        next_vector = torch.where(
            breakdown.unsqueeze(-1),
            restart,
            residual / safe_beta.unsqueeze(-1),
        )
        previous_beta = torch.where(breakdown, torch.zeros_like(beta), beta)
        off_diagonal.append(previous_beta)
        previous, vector = vector, next_vector
    # T is tridiagonal: alpha on the diagonal and beta one step off it. The
    # last step breaks before computing a beta, so there is one fewer beta
    # than alpha. With n_iter = 1 there are no betas at all.
    projected = torch.diag_embed(torch.stack(diagonal, dim=-1))
    if off_diagonal:
        off = torch.stack(off_diagonal, dim=-1)
        projected = projected + torch.diag_embed(off, offset=1)
        projected = projected + torch.diag_embed(off, offset=-1)
    # Ritz pairs: eigenvalues of T, with vectors equal to the Lanczos basis
    # times the eigenvectors of T.
    values, small_vectors = torch.linalg.eigh(projected)
    return values, torch.stack(columns, dim=-1) @ small_vectors


def _shift_invert(
    A: torch.Tensor,
    k: int,
    n_iter: int,
    sigma: float,
    generator: torch.Generator | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Factor once and validate each member against the original operator."""
    identity = torch.eye(A.shape[-1], dtype=A.dtype, device=A.device)
    # One LU factorization of A - sigma I serves every Lanczos step. lu_factor_ex
    # reports a zero pivot in `info` instead of raising, so the check below can
    # name the failed batch members.
    factors, pivots, info = torch.linalg.lu_factor_ex(A - sigma * identity)
    failed = (info != 0) | ~torch.isfinite(factors).all(dim=(-2, -1))
    if failed.any():
        members = failed.nonzero().tolist()
        raise RuntimeError(
            f"shift-invert breakdown: factorization failed for members {members}"
        )

    # Each step applies (A - sigma I)^-1 as a pair of triangular solves on the
    # stored factors, so no step refactors.
    def solve(vector: torch.Tensor) -> torch.Tensor:
        return torch.linalg.lu_solve(factors, pivots, vector.unsqueeze(-1)).squeeze(-1)

    inverse_values, vectors = _iterate(solve, A, n_iter, generator)
    # The largest inverse eigenvalues in magnitude belong to the eigenvalues of
    # A nearest sigma, so keep the k largest.
    indices = inverse_values.abs().argsort(dim=-1, descending=True)[..., :k]
    inverse_values = inverse_values.gather(-1, indices)
    vectors = vectors.gather(-1, indices.unsqueeze(-2).expand(*vectors.shape[:-1], k))
    # Undo the transform: lambda = sigma + 1/mu.
    values = sigma + inverse_values.reciprocal()
    # The Lanczos residual was taken for the inverse operator, so each Ritz
    # pair is checked against the original A before it is returned.
    residuals = torch.linalg.vector_norm(
        A @ vectors - vectors * values.unsqueeze(-2), dim=-2
    )
    # scale is ||A||_inf (the largest absolute row sum) plus |lambda|, so the
    # residual test is relative to both the operator and the eigenvalue.
    scale = A.abs().sum(dim=-1).amax(dim=-1, keepdim=True) + values.abs()
    finite = torch.isfinite(values).all(dim=-1) & torch.isfinite(residuals).all(dim=-1)
    if not finite.all():
        members = (~finite).nonzero().tolist()
        raise RuntimeError(
            f"shift-invert breakdown: nonfinite Ritz pairs for members {members}"
        )
    # A pair passes when its residual is within n * sqrt(eps) of scale, the
    # same sqrt(eps) the solvers use for their default tolerances. Pairs that
    # fail are still returned, with a warning.
    inaccurate = (
        residuals > A.shape[-1] * torch.finfo(A.dtype).eps ** 0.5 * scale
    ).any(-1)
    if inaccurate.any():
        warnings.warn(
            "shift-invert Ritz pairs have not converged for batch members "
            f"{inaccurate.nonzero().tolist()}; increase n_iter",
            UserWarning,
            stacklevel=2,
        )
    return _select(values, vectors, k, "SA", None)
