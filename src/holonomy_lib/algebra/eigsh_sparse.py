# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Synoros

"""Smallest-algebraic eigenpairs of a sparse symmetric matrix (hybrid backends).

The bottom of the spectrum of a large sparse PSD operator (graph/sheaf
Laplacian) is the regime plain Lanczos handles worst: the smallest
eigenvalues are clustered near 0 while Krylov iteration converges to the
dominant end first. The two standard remedies, both used here:

  - **Shift-and-invert** (Ericsson-Ruhe 1980): run the iteration on
    `(A − σI)^{-1}` with σ just below the spectrum; the smallest λ of A
    become the dominant eigenvalues of the inverse. Needs a sparse
    factorization (LU), whose fill-in grows superlinearly with n — the
    right tool up to medium n.
  - **LOBPCG** (Knyazev 2001) with an incomplete-LU preconditioner
    approximating `A^{-1}`: factorization-free iteration whose
    convergence on the clustered-near-zero modes is restored by the
    preconditioner — the right tool for large n.

Backend selection (`solver="auto"`):
  - CUDA sparse input → `torch.lobpcg(largest=False)` (native, stays on
    device; unpreconditioned — see scale-of-validity note below).
  - CPU sparse, `n ≤ SPARSE_SA_ITER_SOLVER_N` → SciPy `eigsh` shift-invert.
  - CPU sparse, larger → SciPy `lobpcg` + `spilu` preconditioner.

The two SciPy branches compute the SAME modes (validated to ~1e-14
against each other and against dense `eigvalsh` in
`tests/algebra/test_eigsh_sparse.py`; originally validated in
synoros-substrate's `sheaf_scale.py`, 2026-06-06). The threshold between
them is a throughput choice only.

Scale-of-validity, CUDA path: `torch.lobpcg` has no ILU-class
preconditioner, so its convergence on Laplacians with modes clustered
near 0 degrades with n; validated here on test-scale problems only.
For large ill-conditioned spectra prefer the CPU SciPy path until a
torch-native preconditioner lands.

References:
  Ericsson, T., Ruhe, A. (1980). The spectral transformation Lanczos
    method for the numerical solution of large sparse generalized
    symmetric eigenvalue problems. Math. Comp. 35(152):1251-1268.
  Knyazev, A. V. (2001). Toward the optimal preconditioned eigensolver:
    locally optimal block preconditioned conjugate gradient method.
    SIAM J. Sci. Comput. 23(2):517-541.
  Saad, Y. (2003). Iterative Methods for Sparse Linear Systems, 2nd ed.
    SIAM, §10.3-10.4 (incomplete-LU preconditioning).
  Saad, Y. (2011). Numerical Methods for Large Eigenvalue Problems,
    2nd ed. SIAM, §3.2 (shift-and-invert rationale).
"""

from __future__ import annotations

from typing import Literal, Optional

import torch

# Above this n, the SciPy CPU path switches from eigsh shift-invert
# (full sparse LU — fill-in grows too fast beyond this) to iterative
# LOBPCG + incomplete-LU. THROUGHPUT threshold only: both branches
# solve the same bottom-k modes (verified to ~1e-14 in tests). Set in
# synoros-substrate `sheaf_scale.py` on sheaf Laplacians up to ~150k
# stalk-dims (2026-06). **Scale of validity**: CPU, Laplacian-like
# sparsity (bounded fill per row); re-derive if either changes.
# Catalog: `SPARSE_SA_ITER_SOLVER_N` in `notes/magic_numbers.md`.
SPARSE_SA_ITER_SOLVER_N: int = 8000

# Shift-invert pole: a small NEGATIVE shift so `(A − σI)` is
# non-singular despite a λ=0 kernel (Laplacians), placing the pole just
# below the spectrum so the solve targets the SMALLEST eigenvalues.
# A pole offset in the spirit of the library's floor convention, not
# architectural. Catalog: `SPARSE_SA_SHIFT_INVERT_POLE`.
SPARSE_SA_SHIFT_INVERT_POLE: float = -1e-6

# spilu/LOBPCG numerical knobs for the large-n CPU path. These affect
# CONVERGENCE RATE and achieved precision of the modes, not WHICH modes
# (verified against the eigsh branch). eps-shift `SPILU_EPS_FRAC ·
# mean(diag A)` makes a singular A factorable by the incomplete LU;
# drop_tol/fill_factor trade preconditioner quality against memory;
# tol/maxiter are the LOBPCG stopping criteria. Values from
# synoros-substrate `sheaf_scale.py` (validated 2026-06-06 on sheaf
# Laplacians, n up to ~150k). Catalog: `SPARSE_SA_SPILU_LOBPCG_KNOBS`
# (one entry for the set).
SPILU_EPS_FRAC: float = 1e-3
SPILU_DROP_TOL: float = 1e-3
SPILU_FILL_FACTOR: int = 10
LOBPCG_TOL: float = 1e-6
LOBPCG_MAXITER: int = 100

_SPARSE_LAYOUTS = (
    torch.sparse_coo, torch.sparse_csr, torch.sparse_csc,
)

Solver = Literal["auto", "eigsh", "lobpcg", "torch_lobpcg"]


def _to_scipy_csr(A: torch.Tensor):
    """Torch sparse (any layout) → SciPy CSR, float64 CPU."""
    import scipy.sparse as sp

    coo = (A if A.layout == torch.sparse_coo else A.to_sparse_coo()).coalesce()
    idx = coo.indices().cpu().numpy()
    vals = coo.values().to(torch.float64).cpu().numpy()
    return sp.coo_matrix(
        (vals, (idx[0], idx[1])), shape=tuple(coo.shape),
    ).tocsr()


def eigsh_sparse_sa(
    A: torch.Tensor,
    k: int,
    sigma: Optional[float] = None,
    solver: Solver = "auto",
    generator: Optional[torch.Generator] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Bottom-k eigenpairs `(eigvals (k,), eigvecs (n, k))` of sparse symmetric `A`.

    Eigenvalues ascend (smallest first); eigenvectors are orthonormal,
    columns aligned to the eigenvalues. Single-instance 2-D only (same
    contract as the sparse LA path of `lanczos_eigsh`). Output lands on
    `A.device` / `A.dtype`; the SciPy branches compute in float64 on CPU
    and cast back.

    Args:
      A: sparse `(n, n)` symmetric (COO/CSR/CSC). Symmetry is assumed,
        not checked.
      k: number of eigenpairs, `1 ≤ k < n` (strict: the iterative
        backends need at least one leftover dimension).
      sigma: shift-invert pole for the `eigsh` branch. Default
        `SPARSE_SA_SHIFT_INVERT_POLE` (just below a PSD spectrum). Only
        meaningful for `solver="eigsh"`/auto-small; ignored by the
        LOBPCG branches.
      solver: `"auto"` (dispatch by device and size), `"eigsh"` (SciPy
        shift-invert), `"lobpcg"` (SciPy LOBPCG + spilu), or
        `"torch_lobpcg"` (native, device-preserving, unpreconditioned).
      generator: torch generator for the LOBPCG starting block (both
        LOBPCG branches); ignored by `eigsh`.

    Returns:
      `(eigvals (k,), eigvecs (n, k))` on `A`'s device/dtype.

    References:
      Ericsson-Ruhe (1980); Knyazev (2001); Saad (2011) §3.2. See the
      module docstring for the full citations and backend rationale.
    """
    if A.layout not in _SPARSE_LAYOUTS:
        raise ValueError(
            f"A must be a sparse tensor (COO/CSR/CSC); got layout={A.layout}"
        )
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(
            f"sparse A must be (n, n); got A.shape={tuple(A.shape)}"
        )
    n = A.shape[0]
    if not 1 <= k < n:
        raise ValueError(f"k must satisfy 1 <= k < n={n}; got k={k}")
    if solver not in ("auto", "eigsh", "lobpcg", "torch_lobpcg"):
        raise ValueError(
            f"solver must be one of 'auto', 'eigsh', 'lobpcg', "
            f"'torch_lobpcg'; got solver={solver!r}"
        )

    if solver == "auto":
        if A.is_cuda:
            solver = "torch_lobpcg"
        elif n <= SPARSE_SA_ITER_SOLVER_N:
            solver = "eigsh"
        else:
            solver = "lobpcg"

    if solver == "torch_lobpcg":
        coo = (
            A if A.layout == torch.sparse_coo else A.to_sparse_coo()
        ).coalesce()
        vals, vecs = torch.lobpcg(
            coo, k=k, largest=False, tol=LOBPCG_TOL, niter=LOBPCG_MAXITER,
        )
        order = torch.argsort(vals)
        return vals[order], vecs[:, order]

    import scipy.sparse.linalg as spla

    A_sp = _to_scipy_csr(A)
    if solver == "eigsh":
        pole = SPARSE_SA_SHIFT_INVERT_POLE if sigma is None else sigma
        # which="LM" on the INVERTED operator == closest to the pole ==
        # smallest of a PSD spectrum with the pole just below it.
        vals_np, vecs_np = spla.eigsh(A_sp, k=k, sigma=pole, which="LM")
    else:  # "lobpcg"
        import numpy as np
        import scipy.sparse as sp

        eps = SPILU_EPS_FRAC * float(A_sp.diagonal().mean())
        ilu = spla.spilu(
            (A_sp + eps * sp.identity(n, format="csc")).tocsc(),
            drop_tol=SPILU_DROP_TOL,
            fill_factor=SPILU_FILL_FACTOR,
        )
        M = spla.LinearOperator((n, n), matvec=ilu.solve)
        X = torch.randn(
            n, k, generator=generator, dtype=torch.float64,
        ).numpy()
        vals_np, vecs_np = spla.lobpcg(
            A_sp, X, M=M, largest=False,
            tol=LOBPCG_TOL, maxiter=LOBPCG_MAXITER,
        )
        order = np.argsort(vals_np)
        vals_np, vecs_np = vals_np[order], vecs_np[:, order]

    vals = torch.from_numpy(vals_np).to(device=A.device, dtype=A.dtype)
    vecs = torch.from_numpy(vecs_np).to(device=A.device, dtype=A.dtype)
    order = torch.argsort(vals)
    return vals[order], vecs[:, order]
