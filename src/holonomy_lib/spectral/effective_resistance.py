# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Synoros

"""Effective resistance and commute-time distances on graphs.

For an undirected weighted graph, the **effective resistance**
`R(u, v)` between nodes u and v is the resistance the graph presents
when each edge is treated as a resistor of conductance `A_{ij}`
(weight) and we measure between nodes u and v. The classical formula
(Klein-Randić 1993, Doyle-Snell 1984) is

    R(u, v) = (e_u − e_v)ᵀ L⁺ (e_u − e_v)
            = L⁺[u, u] − 2 L⁺[u, v] + L⁺[v, v]

where `L⁺` is the Moore-Penrose pseudoinverse of the combinatorial
graph Laplacian `L = D − A`. The **commute time** is

    C(u, v) = 2 · ‖A‖₁ / 2 · R(u, v) = vol(A) · R(u, v)

with `vol(A) = Σ_ij A_{ij}` the total edge weight (Chandra et al. 1996,
Lovász 1993). Both are metric distances on connected graphs.

These primitives are useful for:
  - Graph kernels and similarity (effective resistance is the squared
    diffusion distance at infinite time, modulo normalization).
  - Spectral sparsification (Spielman-Srivastava 2008 sample edges by
    effective resistance).
  - Random walk analysis (commute time = expected round-trip).

For disconnected graphs the formulas give finite values on the
connected components (via the Moore-Penrose convention on the
pseudoinverse), but the cross-component result is not physically
meaningful — those pairs have "infinite resistance" in the electric-
network picture. Mask by component membership before interpreting.

References:
  Doyle, P. G., Snell, J. L. (1984). Random Walks and Electric
    Networks. Mathematical Association of America (Carus Math.
    Monographs 22). The foundational electrical-network treatment.
  Klein, D. J., Randić, M. (1993). Resistance distance. Journal of
    Mathematical Chemistry 12(1):81–95. Coined the term "resistance
    distance" and gave the closed-form in terms of L⁺.
  Chandra, A. K., Raghavan, P., Ruzzo, W. L., Smolensky, R., Tiwari, P.
    (1996). The electrical resistance of a graph captures its commute
    and cover times. Computational Complexity 6(4):312–340. The
    `C(u, v) = vol(A) · R(u, v)` identity.
  Spielman, D. A., Srivastava, N. (2011). Graph sparsification by
    effective resistances. SIAM Journal on Computing 40(6):1913–1926.
"""

from __future__ import annotations

import torch

from holonomy_lib.provenance import with_provenance
from holonomy_lib.spectral import laplacian as _L


@with_provenance(
    "holonomy_lib.spectral.effective_resistance", op_version="0.1",
)
def effective_resistance(A: torch.Tensor) -> torch.Tensor:
    """Pairwise effective resistance R(u, v) for a batched weighted graph.

    Args:
      A: (B, n, n) symmetric non-negative weighted adjacency.

    Returns:
      R: (B, n, n) effective resistance. R[..., u, v] is the resistance
        between nodes u and v under the electric-network interpretation
        of A. Diagonal R[..., u, u] = 0 by definition.

    Notes:
      Computed in O(B · n³) via dense eigendecomposition of L. For
      large graphs prefer a Lanczos-based variant (planned).

      Pseudoinverse handling: L has a one-dimensional null space (the
      constant vector) on a connected graph, so we drop the
      corresponding eigenvalue from the inversion. The threshold
      `max_eig · 1e-9` is the library's `numerical_floor_convention`.

    References:
      Klein-Randić (1993), Eq. (10) for the L⁺ expression.
      Doyle-Snell (1984), §1.
    """
    if A.ndim < 2 or A.shape[-1] != A.shape[-2]:
        raise ValueError(
            f"A must be (..., n, n); got A.shape={tuple(A.shape)}"
        )

    L = _L.combinatorial(A)
    L_plus = _moore_penrose_symmetric(L)
    # R[u, v] = L⁺[u, u] - 2 L⁺[u, v] + L⁺[v, v]
    diag = torch.diagonal(L_plus, dim1=-2, dim2=-1)  # (B, n)
    return diag.unsqueeze(dim=-1) + diag.unsqueeze(dim=-2) - 2.0 * L_plus


@with_provenance(
    "holonomy_lib.spectral.commute_time", op_version="0.1",
)
def commute_time(A: torch.Tensor) -> torch.Tensor:
    """Pairwise commute time `C(u, v) = vol(A) · R(u, v)`.

    Commute time is the expected number of steps a random walk takes
    to go from u to v and back. The Chandra-Raghavan-Ruzzo-Smolensky-
    Tiwari (1996) identity expresses it as the total edge weight
    `vol(A) = Σ_{ij} A_{ij}` times the effective resistance.

    Args:
      A: (B, n, n) symmetric non-negative weighted adjacency.

    Returns:
      C: (B, n, n) commute times.

    References:
      Chandra et al. (1996), Theorem 2.1.
      Lovász (1993), §5.
    """
    vol = A.sum(dim=(-2, -1))                        # (B,)
    R = effective_resistance(A)
    return vol.unsqueeze(dim=-1).unsqueeze(dim=-1) * R


# ============================================================
# Internal helpers
# ============================================================


def _moore_penrose_symmetric(M: torch.Tensor) -> torch.Tensor:
    """Moore-Penrose pseudoinverse of a symmetric (batched) matrix.

    Computed via eigendecomposition: M = V diag(λ) Vᵀ, so
    M⁺ = V diag(1/λ_i if λ_i > τ else 0) Vᵀ.

    The threshold τ = max|λ_i| · 1e-9 follows the library's
    numerical_floor_convention (`1e-9`, an ALLOWED literal).
    `torch.linalg.pinv` does essentially the same internally; we
    spell it out so the audit catalog records the convention.
    """
    eigvals, eigvecs = torch.linalg.eigh(M)
    max_abs = eigvals.abs().max(dim=-1, keepdim=True).values
    threshold = max_abs * 1e-9
    safe_eig = eigvals.clamp(min=torch.finfo(M.dtype).tiny)
    inv_eig = torch.where(
        eigvals > threshold,
        torch.reciprocal(safe_eig),
        torch.zeros_like(eigvals),
    )
    return torch.matmul(
        eigvecs * inv_eig.unsqueeze(dim=-2), eigvecs.mT,
    )
