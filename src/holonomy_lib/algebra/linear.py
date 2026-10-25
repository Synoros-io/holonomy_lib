# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Synoros

"""Batched exact and randomized singular value decompositions.

This module wraps `torch.linalg` operations with documented defaults
(no hidden tolerances, no silently chosen ranks) and adds primitives
that PyTorch does not expose at the granularity we want.

Currently implemented:
  truncated_svd(M, r, mode, oversample)
      Batched truncated SVD. `mode="exact"` runs full SVD then truncates
      (cost O(min(m,n)² · max(m,n))). `mode="randomized"` uses the
      Halko-Martinsson-Tropp randomized projection (cost roughly
      O(m·n·r), much faster when r ≪ min(m,n)).
"""

from __future__ import annotations

from typing import Literal

import torch

# Default oversampling parameter for randomized SVD, per
# Halko-Martinsson-Tropp (2011) §1.2. The catalog entry is
# `randomized_svd_oversample`. Five additional projection columns
# typically give relative spectral error below 1e-3 for matrices whose
# singular values decay smoothly past index r. Scale of validity:
# adequate when the (r+1)-th singular value is small relative to the
# r-th; increase to 10 when the spectrum is flat near the truncation.
RANDOMIZED_SVD_OVERSAMPLE_DEFAULT: int = 5

# Default number of subspace iterations for randomized SVD power method.
# Halko-Martinsson-Tropp (2011) §4.5; q=2 is the standard recommendation
# for matrices with slowly decaying spectra. Two iterations multiply the
# singular-value gap by (σ_{r+1}/σ_r)^(2q+1), which is enough for typical
# low-rank truncation. Cataloged as `randomized_svd_n_iter`.
RANDOMIZED_SVD_N_ITER_DEFAULT: int = 2


def truncated_svd(
    M: torch.Tensor,
    r: int,
    mode: Literal["exact", "randomized"] = "exact",
    oversample: int = RANDOMIZED_SVD_OVERSAMPLE_DEFAULT,
    n_iter: int = RANDOMIZED_SVD_N_ITER_DEFAULT,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batched truncated SVD: top-r singular triples of M.

    Returns (U, S, Vh) such that

        U @ diag(S) @ Vh   ≈   best rank-r approximation of M.

    For `mode="exact"`, the approximation is the best (Eckart-Young) rank-r
    approximation up to floating-point error. For `mode="randomized"`, it
    is approximate with accuracy controlled by `oversample` and `n_iter`.

    Args:
      M: input matrix, shape (..., m, n). Leading dims are treated as batch.
      r: target rank, 1 <= r <= min(m, n).
      mode: "exact" (full SVD + truncate) or "randomized" (HMT projection).
      oversample: extra projection columns for randomized mode (default
        derived from `RANDOMIZED_SVD_OVERSAMPLE_DEFAULT`, cataloged).
      n_iter: subspace-iteration power method steps for randomized mode
        (default derived from `RANDOMIZED_SVD_N_ITER_DEFAULT`, cataloged).
        Ignored in exact mode.
      generator: PyTorch RNG for randomized mode's projection matrix.

    Returns:
      U:  (..., m, r)  orthonormal columns
      S:  (..., r)     singular values, sorted descending, ≥ 0
      Vh: (..., r, n)  orthonormal rows (V conjugate-transposed)

    References:
      Eckart, C., Young, G. (1936). The approximation of one matrix by
        another of lower rank. Psychometrika, 1(3):211–218.
      Halko, N., Martinsson, P. G., Tropp, J. A. (2011). Finding structure
        with randomness: probabilistic algorithms for constructing
        approximate matrix decompositions. SIAM Review, 53(2):217–288.
    """
    if M.ndim < 2:
        raise ValueError(f"M must have shape (..., m, n), got {tuple(M.shape)}")
    m, n = M.shape[-2:]
    if not (M.is_floating_point() or M.is_complex()):
        raise ValueError(f"M must be floating point or complex, got {M.dtype}")
    if isinstance(r, bool) or not isinstance(r, int) or r <= 0:
        raise ValueError(f"r must be > 0 and an integer, got r={r}")
    for name, value in (("oversample", oversample), ("n_iter", n_iter)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer, got {value}")
    if r > min(m, n):
        raise ValueError(
            f"r={r} exceeds min(m={m}, n={n}); cannot truncate higher than full rank"
        )

    if mode == "exact":
        U_full, S_full, Vh_full = torch.linalg.svd(M, full_matrices=False)
        return (
            U_full[..., :, :r].contiguous(),
            S_full[..., :r].contiguous(),
            Vh_full[..., :r, :].contiguous(),
        )

    if mode == "randomized":
        return _randomized_svd(M, r, oversample=oversample,
                                n_iter=n_iter, generator=generator)

    raise ValueError(f"mode must be 'exact' or 'randomized', got {mode!r}")


def _randomized_svd(
    M: torch.Tensor,
    r: int,
    oversample: int,
    n_iter: int,
    generator: torch.Generator | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Halko-Martinsson-Tropp randomized SVD with subspace iteration.

    Algorithm (Halko-Martinsson-Tropp 2011, Algorithm 5.1 + 4.4):

      1. Draw Ω ∈ R^{n × ℓ}, ℓ = r + oversample, Gaussian.
      2. Form Y = M Ω.   # range capture
      3. Power iteration (Algorithm 4.4): for q steps,
            Q, _ = qr(Y)
            Z = Mᴴ Q
            Q, _ = qr(Z)
            Y = M Q.
      4. Q, _ = qr(Y).   # orthonormal basis for the captured range
      5. B = Qᴴ M.       # (ℓ × n)
      6. SVD on the small B: B = Û Ŝ Vh.
      7. U = Q Û.
      8. Truncate to top-r triples.

    The power iterations are sequential because each uses the preceding subspace.
    Each subspace iteration costs O(m·n·ℓ); total O(m·n·ℓ·(n_iter+1)).
    """
    *batch, m, n = M.shape
    ell = r + oversample
    if ell > min(m, n):
        # Can't oversample past full rank; fall back to exact (cheaper anyway).
        return truncated_svd(M, r, mode="exact")

    Omega = torch.randn(*batch, n, ell, generator=generator,
                        device=M.device, dtype=M.dtype)
    Y = torch.matmul(M, Omega)  # (..., m, ℓ)

    # Subspace iteration: alternately apply M and Mᴴ to amplify dominant
    # singular directions. Re-orthonormalize each step to prevent
    # numerical collapse onto the leading singular vector.
    for _ in range(n_iter):
        Q, _ = torch.linalg.qr(Y)
        Z = torch.matmul(M.mH, Q)
        Q, _ = torch.linalg.qr(Z)
        Y = torch.matmul(M, Q)

    Q, _ = torch.linalg.qr(Y)            # (..., m, ℓ)
    B = torch.matmul(Q.mH, M)            # (..., ℓ, n)
    U_small, S, Vh = torch.linalg.svd(B, full_matrices=False)
    U = torch.matmul(Q, U_small)         # (..., m, ℓ)

    return (
        U[..., :, :r].contiguous(),
        S[..., :r].contiguous(),
        Vh[..., :r, :].contiguous(),
    )
