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
  - Graph kernels and similarity through the Laplacian pseudoinverse.
  - Spectral sparsification (Spielman-Srivastava 2008 sample edges by
    effective resistance).
  - Random walk analysis (commute time = expected round-trip).

Different connected components have infinite resistance and commute
time, since no current or random walk can pass between them. Within a
component, commute time uses that component's volume. Isolated nodes
have zero distance to themselves and infinite distance to every other node.

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

from holonomy_lib._graph_utils import drop_self_loops
from holonomy_lib.spectral import laplacian as _L


def effective_resistance(A: torch.Tensor) -> torch.Tensor:
    """Pairwise effective resistance R(u, v) for a batched weighted graph.

    Args:
      A: (B, n, n) symmetric non-negative weighted adjacency.

    Returns:
      R: (B, n, n) effective resistance. R[..., u, v] is the resistance
        between nodes u and v under the electric-network interpretation
        of A. Diagonal R[..., u, u] = 0 by definition. Pairs in different
        connected components have infinite resistance.

    Notes:
      Each component is scaled by its maximum weighted degree. Adding the
      projector onto component constants makes the scaled Laplacian positive
      definite; Cholesky inversion and projector subtraction give its
      pseudoinverse without a spectral rank cutoff. Connectivity uses
      Boolean reachability by repeated matrix squaring, with O(log n)
      sequential rounds and O(B · n³ log n) dense work. Each round doubles
      the path lengths represented, so there are no node or batch loops.

    References:
      Klein-Randić (1993), Eq. (10) for the L⁺ expression.
      Doyle-Snell (1984), §1.
    """
    A = _checked_adjacency(A)
    return _resistance_and_reachability(A)[0]


def commute_time(A: torch.Tensor) -> torch.Tensor:
    """Pairwise commute time `C(u, v) = vol(component) · R(u, v)`.

    Commute time is the expected number of steps a random walk takes
    to go from u to v and back. The Chandra-Raghavan-Ruzzo-Smolensky-
    Tiwari (1996) identity expresses it as the total edge weight
    `vol(A) = Σ_{ij} A_{ij}` times the effective resistance on a connected
    graph. On disconnected graphs the volume is that of the component
    containing u and v. Self-loops are dropped before computing this volume.

    Args:
      A: (B, n, n) symmetric non-negative weighted adjacency.

    Returns:
      C: (B, n, n) commute times. Cross-component pairs are infinite,
        and the diagonal is zero, including isolated nodes.

    References:
      Chandra et al. (1996), Theorem 2.1.
      Lovász (1993), §5.
    """
    A = _checked_adjacency(A)
    resistance, connected = _resistance_and_reachability(A)
    degree = A.sum(dim=-1)
    # Row u of `connected` marks the component of u, so this matmul sums the
    # degrees over that component, which is its volume.
    volume = (connected.to(A.dtype) @ degree.unsqueeze(-1)).squeeze(-1)
    # Cross-component entries of `resistance` are inf. An isolated node has
    # volume 0, and 0 * inf is NaN, so those entries are zeroed before the
    # multiply. The where below puts inf back on the cross-component pairs.
    finite_resistance = torch.where(connected, resistance, 0.0)
    return torch.where(
        connected, volume.unsqueeze(-1) * finite_resistance, float("inf"),
    )


# ============================================================
# Internal helpers
# ============================================================


def _checked_adjacency(A: torch.Tensor) -> torch.Tensor:
    """Validate a dense adjacency and zero its diagonal.

    Takes a dense `(..., n, n)` real floating-point tensor and returns it with
    the diagonal zeroed. Raises ValueError if it is not square, not floating
    point, or has a negative off-diagonal entry. Self-loops are dropped before
    the sign check, so a negative diagonal entry is ignored rather than
    rejected. Assumes A is symmetric. This function does not check that; the
    Laplacian it feeds warns when it is not.
    """
    if A.ndim < 2 or A.shape[-1] != A.shape[-2]:
        raise ValueError(f"A must be (..., n, n); got A.shape={tuple(A.shape)}")
    if not A.is_floating_point():
        raise ValueError("A must have a real floating-point dtype")
    A = drop_self_loops(A)
    if (A < 0).any():
        raise ValueError("A must have nonnegative conductances")
    return A


def _resistance_and_reachability(
    A: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return resistance and connectivity for a validated, loop-free adjacency.

    Takes the output of `_checked_adjacency`: a `(B, n, n)` symmetric,
    non-negative adjacency with zero diagonal. Returns `(resistance,
    connected)`, both `(B, n, n)`. `connected` is a Boolean mask of the pairs
    in the same component, diagonal included. `resistance` is inf wherever
    `connected` is False. Assumes symmetry; the Laplacian it builds warns
    otherwise.
    """
    n = A.shape[-1]
    # A node always reaches itself, so the identity goes into the pattern.
    connected = (A > 0) | torch.eye(n, device=A.device, dtype=torch.bool)
    # An empty batch or a zero-node graph has nothing to invert. A.clone() is
    # already the right, possibly empty, shape for the result.
    if n == 0 or A.numel() == 0:
        return A.clone(), connected
    # A simple path has at most n - 1 edges. Each squaring doubles the path
    # length the mask covers, and all graphs in the batch square together, so
    # bit_length(n - 1) rounds cover every pair.
    for _ in range((n - 1).bit_length()):
        paths = connected.to(A.dtype)
        connected = connected | (paths @ paths > 0)
    laplacian = _L.combinatorial(A)
    degree = A.sum(dim=-1)
    # Each node takes the largest degree in its own component. Entries outside
    # the component are zeroed before the max, so they cannot win it.
    component_scale = torch.where(connected, degree.unsqueeze(-2), 0.0).amax(-1)
    # Every node in a component gets the same scale, so dividing rows by it
    # keeps the matrix symmetric. An isolate has degree 0, and dividing its
    # zero Laplacian row by 0 would give NaN, so its scale is swapped for 1.
    component_scale = torch.where(
        component_scale > 0, component_scale, torch.ones_like(component_scale),
    ).unsqueeze(-1)
    # Projector onto the constant vectors of each component, P = 1 1^T / |c|.
    # Adding it turns the zero eigenvalue of each component into 1, which makes
    # the scaled Laplacian positive definite.
    projector = connected.to(A.dtype) / connected.sum(dim=-1, keepdim=True)
    grounded = laplacian / component_scale + projector
    # grounded^{-1} - P is the pseudoinverse of L / s. Dividing by s undoes the
    # scaling, since (L / s)^+ = s L^+.
    inverse = (
        torch.cholesky_inverse(torch.linalg.cholesky(grounded)) - projector
    ) / component_scale
    # Expanding (e_u - e_v)^T L^+ (e_u - e_v) gives the two diagonal entries
    # minus twice the off-diagonal one.
    diagonal = inverse.diagonal(dim1=-2, dim2=-1)
    resistance = diagonal.unsqueeze(-1) + diagonal.unsqueeze(-2) - 2.0 * inverse
    # The inverse is block diagonal, so a cross-component entry is 0 and its
    # resistance comes out finite. Those pairs have no path, so they get inf.
    return torch.where(connected, resistance, float("inf")), connected
