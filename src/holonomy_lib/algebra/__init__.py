# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Synoros

"""holonomy_lib.algebra: audit-clean wrappers around torch.linalg + extras.

Currently implemented:
  linear.truncated_svd:  batched truncated SVD (exact + randomized).
  lanczos_eigsh:         largest/smallest algebraic dense eigenpairs;
                         explicit dense shifts target nearby modes.
                         Full subspaces use torch.linalg.eigh.
                         Sparse SA uses the backends in eigsh_sparse
                         (bounded-pole SciPy eigsh / preconditioned
                         SciPy LOBPCG / torch.lobpcg).
  eigsh_sparse.eigsh_sparse_sa: the sparse smallest-k backend, callable
                         directly for backend control.
  solvers.cg:            conjugate gradient for SPD systems : batched
                         dense, sparse, or callable-operator form.
  solvers.lsqr:          iterative (damped) least squares via
                         Golub-Kahan bidiagonalization.

Planned:
  Einstein notation helpers, group representations (cyclic, dihedral),
  Lie algebras (su(n), so(n)), exterior algebra (wedge products).
"""

from holonomy_lib.algebra.eigsh_sparse import eigsh_sparse_sa
from holonomy_lib.algebra.lanczos import lanczos_eigsh
from holonomy_lib.algebra.linear import truncated_svd
from holonomy_lib.algebra.solvers import cg, lsqr

__all__ = ["cg", "eigsh_sparse_sa", "lanczos_eigsh", "lsqr", "truncated_svd"]
