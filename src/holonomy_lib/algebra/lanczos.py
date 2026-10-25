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

# Saad (2011), §6.5 motivates extra Krylov dimensions for convergence.
# Existing catalog entry: LANCZOS_OVERSAMPLE_DEFAULT.
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
    if not isinstance(oversample, int) or oversample < 0:
        raise ValueError("oversample must be a nonnegative integer")
    if which not in ("LA", "SA"):
        raise ValueError("which must be 'LA' or 'SA'")
    if sigma is not None and (which != "SA" or not math.isfinite(sigma)):
        raise ValueError("sigma must be finite and is only meaningful with which='SA'")
    if n_iter is None:
        n_iter = k + oversample
    if not isinstance(n_iter, int) or n_iter < k:
        raise ValueError(f"n_iter must be an integer >= k={k}")
    n_iter = min(n_iter, n)
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
    if n_iter == n or A.numel() == 0:
        dense = A.to_dense() if sparse else A
        values, vectors = torch.linalg.eigh(dense)
        return _select(values, vectors, k, which, sigma)

    if sigma is not None:
        return _shift_invert(A, k, n_iter, sigma, generator)

    def matvec(vector: torch.Tensor) -> torch.Tensor:
        return (A @ vector.unsqueeze(-1)).squeeze(-1)

    values, vectors = _iterate(matvec, A, n_iter, generator)
    return _select(values, vectors, k, which, None)


def _warn_asymmetric(A: torch.Tensor) -> None:
    """Warn once at the eigensolver entry boundary for nonsymmetric input."""
    if A.layout in _SPARSE_LAYOUTS:
        coo = A.to_sparse_coo().coalesce()
        difference = (coo - coo.mT).coalesce().values()
        scale = coo.values().abs().amax() if coo._nnz() else coo.values().new_zeros(())
        tolerance = A.shape[-1] * torch.finfo(A.dtype).eps * scale
        symmetric = bool((difference.abs() <= tolerance).all())
    else:
        symmetric = torch.allclose(A, A.mT)
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
        indices = (values - sigma).abs().argsort(dim=-1)[..., :k]
        chosen = values.gather(-1, indices)
        indices = indices.gather(-1, chosen.argsort(dim=-1))
    else:
        indices = values.argsort(dim=-1, descending=which == "LA")[..., :k]
    return (
        values.gather(-1, indices),
        vectors.gather(-1, indices.unsqueeze(-2).expand(*vectors.shape[:-1], k)),
    )


def _orthogonalize(vector: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """Two Gram-Schmidt passes remove roundoff left by the first projection."""
    for _ in range(2):
        vector = vector - (basis @ (basis.mT @ vector.unsqueeze(-1))).squeeze(-1)
    return vector


def _iterate(
    matvec: Callable[[torch.Tensor], torch.Tensor],
    A: torch.Tensor,
    n_iter: int,
    generator: torch.Generator | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build independent Lanczos blocks per batch member, then diagonalize."""
    n = A.shape[-1]
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
    previous = torch.zeros_like(vector)
    previous_beta = vector.new_zeros(vector.shape[:-1])
    for step in range(n_iter):
        # Immutable columns keep tensors saved for backward valid.
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
        breakdown = beta <= n * torch.finfo(A.dtype).eps * scale
        safe_beta = torch.where(breakdown, torch.ones_like(beta), beta)

        # A coordinate with the least basis mass has a nonzero projection
        # onto the complement while fewer than n basis vectors are present.
        coordinate = basis.square().sum(dim=-1).argmin(dim=-1)
        restart = torch.nn.functional.one_hot(coordinate, num_classes=n).to(A.dtype)
        restart = _orthogonalize(restart, basis)
        restart = restart / torch.linalg.vector_norm(restart, dim=-1, keepdim=True)
        next_vector = torch.where(
            breakdown.unsqueeze(-1),
            restart,
            residual / safe_beta.unsqueeze(-1),
        )
        previous_beta = torch.where(breakdown, torch.zeros_like(beta), beta)
        off_diagonal.append(previous_beta)
        previous, vector = vector, next_vector
    projected = torch.diag_embed(torch.stack(diagonal, dim=-1))
    if off_diagonal:
        off = torch.stack(off_diagonal, dim=-1)
        projected = projected + torch.diag_embed(off, offset=1)
        projected = projected + torch.diag_embed(off, offset=-1)
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
    factors, pivots, info = torch.linalg.lu_factor_ex(A - sigma * identity)
    failed = (info != 0) | ~torch.isfinite(factors).all(dim=(-2, -1))
    if failed.any():
        members = failed.nonzero().tolist()
        raise RuntimeError(
            f"shift-invert breakdown: factorization failed for members {members}"
        )

    def solve(vector: torch.Tensor) -> torch.Tensor:
        return torch.linalg.lu_solve(factors, pivots, vector.unsqueeze(-1)).squeeze(-1)

    inverse_values, vectors = _iterate(solve, A, n_iter, generator)
    indices = inverse_values.abs().argsort(dim=-1, descending=True)[..., :k]
    inverse_values = inverse_values.gather(-1, indices)
    vectors = vectors.gather(-1, indices.unsqueeze(-2).expand(*vectors.shape[:-1], k))
    values = sigma + inverse_values.reciprocal()
    residuals = torch.linalg.vector_norm(
        A @ vectors - vectors * values.unsqueeze(-2), dim=-2
    )
    scale = A.abs().sum(dim=-1).amax(dim=-1, keepdim=True) + values.abs()
    finite = torch.isfinite(values).all(dim=-1) & torch.isfinite(residuals).all(dim=-1)
    if not finite.all():
        members = (~finite).nonzero().tolist()
        raise RuntimeError(
            f"shift-invert breakdown: nonfinite Ritz pairs for members {members}"
        )
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
