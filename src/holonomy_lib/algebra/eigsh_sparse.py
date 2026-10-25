# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Synoros

"""Smallest-algebraic eigenpairs of sparse real symmetric matrices.

CPU inputs use SciPy shift-invert eigsh or preconditioned LOBPCG; other
devices use torch.lobpcg. The SciPy boundary explicitly computes in CPU
float64 and casts the result back. All backends are nondifferentiable.
A Gershgorin lower bound puts every shift-invert pole below the spectrum,
including for indefinite inputs. Full/near-full requests use dense eigh.

References:
  Ericsson, T., Ruhe, A. (1980). The spectral transformation Lanczos
    method for the numerical solution of large sparse generalized
    symmetric eigenvalue problems. Math. Comp. 35(152):1251-1268.
  Knyazev, A. V. (2001). Toward the optimal preconditioned eigensolver:
    locally optimal block preconditioned conjugate gradient method.
    SIAM J. Sci. Comput. 23(2):517-541.
  Saad, Y. (2011). Numerical Methods for Large Eigenvalue Problems,
    2nd ed. SIAM, §3.2 (spectral transformation).
  Golub, G. H., Van Loan, C. F. (2013). Matrix Computations, 4th ed.
    Johns Hopkins University Press, §7.2 (Gershgorin bounds).
"""

from __future__ import annotations

import math
from typing import Literal

import torch

# Existing throughput threshold from substrate's sheaf_scale.py (2026-06).
# Its measurements concern Laplacian-like CPU sparsity, not every operator.
SPARSE_SA_ITER_SOLVER_N: int = 8000

# Existing cataloged convergence/preconditioner knobs from sheaf_scale.py.
SPILU_EPS_FRAC: float = 1e-3
SPILU_DROP_TOL: float = 1e-3
SPILU_FILL_FACTOR: int = 10
LOBPCG_TOL: float = 1e-6
LOBPCG_MAXITER: int = 100

# LOBPCG uses current vectors, residuals and previous search directions.
# torch.lobpcg requires n >= 3*k for this three-block Ritz subspace.
LOBPCG_SUBSPACE_BLOCKS: int = len(("iterate", "residual", "search_direction"))

_SPARSE_LAYOUTS = (torch.sparse_coo, torch.sparse_csr, torch.sparse_csc)
Solver = Literal["auto", "eigsh", "lobpcg", "torch_lobpcg"]


def eigsh_sparse_sa(
    A: torch.Tensor,
    k: int,
    sigma: float | None = None,
    solver: Solver = "auto",
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the smallest algebraic eigenpairs of sparse symmetric ``A``.

    Args:
      A: real sparse ``(n, n)`` COO/CSR/CSC. Asymmetry emits a warning.
        Grad-requiring inputs raise an error; detach explicitly to use
        this nondifferentiable solver.
      k: number of eigenpairs, ``1 <= k <= n``. Full/near-full requests
        use dense eigh because iterative backend subspaces do not fit.
      sigma: optional eigsh pole, strictly below the Gershgorin lower
        bound. The default is that bound minus a dtype-scaled margin.
        LOBPCG does not use a pole; explicit sigma is rejected there.
      solver: ``auto`` selects torch.lobpcg off CPU, SciPy eigsh on CPU
        up to ``SPARSE_SA_ITER_SOLVER_N``, then SciPy LOBPCG with shifted
        incomplete-LU preconditioning. The three explicit names are
        ``eigsh``, ``lobpcg`` and ``torch_lobpcg``.
      generator: random-start generator compatible with ``A.device``.
        Used by every iterative backend, including native LOBPCG.

    Returns:
      Ascending eigenvalues ``(k,)`` and orthonormal eigenvector columns
      ``(n, k)``, on ``A.device`` with ``A.dtype``. SciPy computes in
      float64 on CPU; its transfer is an explicit backend boundary.

    Notes:
      Native LOBPCG is unpreconditioned; clustered eigenvalues can need
      more iterations than the fixed cataloged limit. SciPy's shifted
      ILU preconditioner is intended for Laplacian-like sparse problems.

    References:
      Ericsson and Ruhe (1980); Knyazev (2001); Saad (2011), §3.2;
      Golub and Van Loan (2013), §7.2. Full citations above.
    """
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"sparse A must be (n, n); got {A.shape}")
    if A.layout not in _SPARSE_LAYOUTS:
        raise ValueError("A must be a sparse tensor (COO/CSR/CSC)")
    if not A.is_floating_point():
        raise ValueError("A must have a real floating dtype")
    n = A.shape[0]
    if not isinstance(k, int) or isinstance(k, bool) or not 1 <= k <= n:
        raise ValueError(f"k must satisfy 1 <= k <= n={n}; got k={k}")
    if solver not in ("auto", "eigsh", "lobpcg", "torch_lobpcg"):
        raise ValueError("solver must be auto, eigsh, lobpcg or torch_lobpcg")
    if A.requires_grad:
        raise ValueError(
            "sparse eigensolvers do not support autograd; detach A explicitly"
        )
    if sigma is not None and not math.isfinite(sigma):
        raise ValueError("sigma must be finite")
    from holonomy_lib.algebra.lanczos import _warn_asymmetric

    _warn_asymmetric(A)
    coo = A.to_sparse_coo().coalesce()
    if not torch.isfinite(coo.values()).all():
        raise ValueError("A must have finite entries")
    if solver == "auto":
        if A.device.type != "cpu":
            solver = "torch_lobpcg"
        elif n <= SPARSE_SA_ITER_SOLVER_N:
            solver = "eigsh"
        else:
            solver = "lobpcg"
    if sigma is not None and solver != "eigsh":
        raise ValueError("sigma is only used by the eigsh backend")

    row, column = coo.indices()
    entries = coo.values()
    diagonal = entries.new_zeros(n).scatter_add(0, row, entries * (row == column))
    radius = entries.new_zeros(n).scatter_add(0, row, entries.abs() * (row != column))
    lower = (diagonal - radius).amin()
    scale = (diagonal.abs() + radius).amax()
    if sigma is not None and not bool(sigma < lower):
        raise ValueError("sigma must be strictly below the Gershgorin lower bound")
    # Zero operators have an arbitrary orthonormal eigenspace and need
    # neither an iterative solve nor a singular sparse factorization.
    if not bool(scale):
        return entries.new_zeros(k), torch.eye(n, k, dtype=A.dtype, device=A.device)
    if k == n or (solver != "eigsh" and n < LOBPCG_SUBSPACE_BLOCKS * k):
        values, vectors = torch.linalg.eigh(A.to_dense())
        return values[:k], vectors[:, :k]

    if solver == "torch_lobpcg":
        initial = torch.randn(n, k, dtype=A.dtype, device=A.device, generator=generator)
        values, vectors = torch.lobpcg(
            coo,
            k=k,
            X=initial,
            largest=False,
            tol=LOBPCG_TOL,
            niter=LOBPCG_MAXITER,
        )
        order = values.argsort()
        return values[order], vectors[:, order]

    import numpy as np
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    indices_np = coo.indices().cpu().numpy()
    A_sp = sp.coo_matrix(
        (entries.to(torch.float64).cpu().numpy(), (indices_np[0], indices_np[1])),
        shape=(n, n),
    ).tocsr()
    if solver == "eigsh":
        if sigma is None:
            # The nextafter step keeps the pole strictly below the bound
            # even if subtraction rounds back to the bound at this scale.
            pole_tensor = lower - torch.finfo(A.dtype).eps ** 0.5 * scale
            pole = float(
                torch.nextafter(pole_tensor, torch.full_like(lower, -torch.inf))
            )
        else:
            pole = sigma
        initial_vector = (
            torch.randn(
                n,
                device=A.device,
                dtype=A.dtype,
                generator=generator,
            )
            .to(torch.float64)
            .cpu()
            .numpy()
        )
        values_np, vectors_np = spla.eigsh(
            A_sp,
            k=k,
            sigma=pole,
            which="LM",
            v0=initial_vector,
        )
    else:
        # Shift below a proven lower bound, so the preconditioner's
        # target is positive definite even when A itself is indefinite.
        shift = float(lower - SPILU_EPS_FRAC * scale)
        ilu = spla.spilu(
            (A_sp - shift * sp.identity(n, format="csc")).tocsc(),
            drop_tol=SPILU_DROP_TOL,
            fill_factor=SPILU_FILL_FACTOR,
        )
        preconditioner = spla.LinearOperator((n, n), matvec=ilu.solve)
        initial_block = (
            torch.randn(
                n,
                k,
                device=A.device,
                dtype=A.dtype,
                generator=generator,
            )
            .to(torch.float64)
            .cpu()
            .numpy()
        )
        values_np, vectors_np = spla.lobpcg(
            A_sp,
            initial_block,
            M=preconditioner,
            largest=False,
            tol=LOBPCG_TOL,
            maxiter=LOBPCG_MAXITER,
        )
    order_np = np.argsort(values_np)
    return (
        torch.from_numpy(values_np[order_np].copy()).to(device=A.device, dtype=A.dtype),
        torch.from_numpy(vectors_np[:, order_np].copy()).to(
            device=A.device, dtype=A.dtype
        ),
    )
