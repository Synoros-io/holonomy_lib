# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Synoros

"""Forman-Ricci curvature on batched graphs.

Forman (2003) defined a combinatorial Ricci curvature for general CW
complexes that specializes to a very simple per-edge formula on graphs.
Unlike Ollivier-Ricci (`ricci.ollivier_ricci_curvature`), no optimal-
transport solve is needed: each edge's curvature is a closed-form
function of the local degrees and, for the augmented form, the
triangles on that edge. Both forms are cheaper than a Sinkhorn solve
per edge.

Forman-Ricci is less geometrically faithful than Ollivier on weighted
metric measure spaces, but it tracks the same qualitative bottleneck
structure: edges bridging two clusters get negative curvature, edges
embedded in dense sub-communities get positive curvature. For networks
too large for Ollivier, Forman-Ricci is a common substitute (Sreejith
et al. 2016).

Cost note: the implementations here are dense. The simple form is
O(B · n²) and the augmented form is O(B · n³), the cost of the
triangle-count matmul A @ A. A sparse edge-list path would scale with
|E| instead of n², but it is not implemented. Memory is O(B · n²) for
the dense tensors.

Two forms are exposed:

  forman_ricci_simple(A)
      κ_F(u, v) = w_e · ( w_u/w_e + w_v/w_e
                          - Σ_{e_u ≠ e} w_u/√(w_e · w_{e_u})
                          - Σ_{e_v ≠ e} w_v/√(w_e · w_{e_v}) )
      The original CW-complex specialization without 2-faces (Sreejith
      et al. 2016, eq. 1). On unweighted simple graphs with node
      weights = 1 this collapses to `4 - deg(u) - deg(v)`.

  forman_ricci_augmented(A)
      For binary adjacency only:
      κ_F^aug(u, v) = κ_F_simple(u, v) + 3 · #triangles_through(u, v)
      Adds the contribution from 2-faces (triangles incident to the
      edge), restoring the "augmented" form that more closely tracks
      Ollivier-Ricci on dense substructures (Samal et al. 2018,
      §"Augmented Forman curvature").

References:
  Forman, R. (2003). Bochner's method for cell complexes and
    combinatorial Ricci curvature. Discrete & Computational Geometry
    29:323–374.
  Sreejith, R. P., Mohanraj, K., Jost, J., Saucan, E., Samal, A.
    (2016). Forman curvature for complex networks. Journal of
    Statistical Mechanics: Theory and Experiment 2016:063206. Eq. 1
    (no 2-face contribution).
  Samal, A., Sreejith, R. P., Gu, J., Liu, S., Saucan, E., Jost, J.
    (2018). Comparative analysis of two discretizations of Ricci
    curvature for complex networks. Scientific Reports 8:8650.
    Equations 9–10 give the weighted and unit-weight specializations.
  Sandhu, R., Georgiou, T., Tannenbaum, A. (2015). Ricci curvature: An
    economic indicator for market fragility and systemic risk.
    Science Advances 2(5):e1501495. Application demonstrating Forman's
    qualitative agreement with Ollivier at lower compute cost.
"""

from __future__ import annotations

import warnings

import torch

from holonomy_lib._graph_utils import drop_self_loops

# Unit vertex weights in the graph specialization of Sreejith et al.
# (2016), equation 1. Edge weights alone do not encode general vertex
# weights, so the public functions fix every vertex weight to one.
FORMAN_NODE_WEIGHT_DEFAULT: float = 1.0

# Samal et al. (2018), equations 9–10: with unit vertex, edge, and face
# weights, an incident triangular face contributes one face term and
# removes two negative parallel-edge contributions, giving 1 + 2 = 3.
FORMAN_TRIANGLE_CONTRIBUTION: float = 3.0


def forman_ricci_simple(A: torch.Tensor) -> torch.Tensor:
    """Forman-Ricci curvature on every edge of a batched weighted graph.

    Implements the Sreejith et al. (2016) form without 2-face
    contributions:

      κ_F(u, v) = w_e · ( w_u/w_e + w_v/w_e
                          - Σ_{e' ∋ u, e' ≠ e} w_u / √(w_e · w_{e'})
                          - Σ_{e' ∋ v, e' ≠ e} w_v / √(w_e · w_{e'}) )

    For unweighted simple graphs (w_u = w_v = 1, w_e ∈ {0, 1}) this
    collapses to `4 - deg(u) - deg(v)`: positive on edges in sparse
    contexts (low degree), negative on edges anchored in dense hubs.

    Args:
      A: (B, n, n) symmetric weighted adjacency. Non-negative weights.
        Zero off-diagonal means no edge. Diagonal is ignored.

    Returns:
      κ_F: (B, n, n) curvature tensor. κ_F[..., u, v] is the Forman-
        Ricci curvature of edge (u, v) when A[..., u, v] > 0, and 0
        otherwise. The diagonal is set to 0 by convention, since Forman
        curvature is defined per edge and a self-loop is not an edge here.

    Notes:
      Cost is O(B · n²). An edge-list variant would cost O(B · |E|) on
      sparse graphs, but it is not implemented.

    References:
      Sreejith et al. (2016), eq. 1.
      Forman (2003), §"Combinatorial Ricci curvature for graphs".
    """
    _validate_adjacency(A)
    A = drop_self_loops(A)
    *batch, n, _ = A.shape

    # Edge positions as a 0/1 tensor. Staying in tensor form keeps Python
    # from branching on tensor values, which would force a device-to-host
    # sync. Non-edges contribute zero to every sum below.
    edge_mask = (A != 0).to(A.dtype)  # (B, n, n)

    # Node weights are fixed at 1 (Sreejith et al. 2016). Weighted nodes
    # would need these passed in; for now they are constants.
    w_u = FORMAN_NODE_WEIGHT_DEFAULT
    w_v = FORMAN_NODE_WEIGHT_DEFAULT

    # Replace non-edge zeros with ones before any division. rsqrt(0) is inf,
    # and inf times the zero mask is NaN. The mask zeroes the non-edge
    # entries of kappa afterward.
    safe_we = torch.where(edge_mask > 0, A, torch.ones_like(A))

    # The two diagonal terms of the formula, w_u / w_e + w_v / w_e.
    diagonal_term = (w_u + w_v) / safe_we  # (B, n, n)

    # Neighbor sums. The u-side term for edge (u, v) is
    # w_u / √w_e · Σ_{u' ≠ v} 1/√A[u, u'], so one row sum of inv_sqrt_A
    # gives every neighbor sum. Non-edges are zero in inv_sqrt_A, and the
    # (u, v) entry itself is subtracted out below.
    inv_sqrt_A = torch.where(
        edge_mask > 0, torch.rsqrt(safe_we), torch.zeros_like(A),
    )  # (B, n, n)

    # row_sum is (B, n, 1) and holds each node's neighbor sum; col_sum is
    # (B, 1, n) and holds the same sums laid out by column. Broadcast, they
    # give position (u, v) both u's sum and v's sum. Subtracting
    # inv_sqrt_A[u, v] removes the edge from its own sums.
    row_sum = inv_sqrt_A.sum(dim=-1, keepdim=True)         # (B, n, 1)
    col_sum = inv_sqrt_A.sum(dim=-2, keepdim=True)         # (B, 1, n)
    u_neighbor_sum = row_sum - inv_sqrt_A                  # (B, n, n)
    v_neighbor_sum = col_sum - inv_sqrt_A                  # (B, n, n)
    u_side = w_u * inv_sqrt_A * u_neighbor_sum             # (B, n, n)
    v_side = w_v * inv_sqrt_A * v_neighbor_sum             # (B, n, n)

    # Non-edges come out as zero through the edge_mask factor.
    kappa = safe_we * (diagonal_term - u_side - v_side) * edge_mask

    # The diagonal is already zero, since edge_mask is zero there after
    # drop_self_loops. The explicit where keeps that true without relying
    # on the mask.
    eye = torch.eye(n, device=A.device, dtype=A.dtype).expand_as(A)
    kappa = torch.where(eye > 0, torch.zeros_like(kappa), kappa)
    return kappa


def forman_ricci_augmented(A: torch.Tensor) -> torch.Tensor:
    """Augmented Forman-Ricci curvature on an unweighted graph.

    With unit vertex, edge, and triangular-face weights, Samal et al.
    (2018), equations 9–10, give

      kappa_aug(u, v) = kappa_simple(u, v) + 3 * #triangles(u, v).

    Each face contributes one positive term and removes two negative
    parallel-edge terms. Weighted edges require a face-weight convention
    and the full weighted expression, which this function does not expose.

    Args:
      A: (B, n, n) symmetric real floating binary adjacency. Zero means
        no edge and one means an edge. The diagonal is ignored.

    Returns:
      (B, n, n) curvature, zero on the diagonal and nonedges.

    Raises:
      ValueError: if any off-diagonal weight is neither zero nor one.
        The shape, dtype, and sign checks in `_validate_adjacency` also raise.

    References:
      Samal, A., et al. (2018). Comparative analysis of two discretizations
        of Ricci curvature for complex networks. Scientific Reports
        8:8650, equations 9–10.
    """
    _validate_adjacency(A)
    A = drop_self_loops(A)
    if ((A != 0) & (A != 1)).any():
        raise ValueError(
            "forman_ricci_augmented requires binary (unweighted) adjacency; "
            "weighted curvature needs an explicit face-weight convention "
            "(Samal et al. 2018, equations 9–10)."
        )
    # After drop_self_loops, (A @ A)[u, v] counts the nodes adjacent to both
    # u and v, which is the number of triangles on edge (u, v) when A is
    # binary. Multiplying by A keeps only real edges, since non-edges get no
    # curvature.
    triangle_count = (A @ A) * A
    return forman_ricci_simple(A) + FORMAN_TRIANGLE_CONTRIBUTION * triangle_count


# ============================================================
# Internal helpers
# ============================================================


def _validate_adjacency(A: torch.Tensor) -> None:
    """Check that A is a real (..., n, n) tensor of non-negative edge weights.

    Input: A, with its diagonal ignored.
    Output: None. Raises ValueError on a bad shape, a non-floating dtype,
    or a nonfinite or negative off-diagonal entry. Warns, without raising,
    when A is not symmetric within a small absolute tolerance.
    """
    if A.ndim < 2 or A.shape[-1] != A.shape[-2]:
        raise ValueError(
            f"A must be (..., n, n); got A.shape={tuple(A.shape)}"
        )
    if not A.is_floating_point():
        raise ValueError("A must have a real floating dtype")
    A = drop_self_loops(A)
    if not torch.isfinite(A).all() or (A < 0).any():
        raise ValueError(
            "Forman-Ricci is defined for non-negative edge weights; "
            "nonfinite or negative entries detected."
        )
    # Symmetry check matches the convention in `ollivier_ricci_curvature`.
    if not torch.allclose(A, A.mT, atol=1e-9, rtol=0):
        warnings.warn(
            "forman_ricci received an asymmetric adjacency; Forman-Ricci is "
            "only well-defined for symmetric (undirected) graphs. Symmetrize "
            "A first (e.g. `0.5 * (A + A.mT)`) for meaningful results.",
            stacklevel=2,
        )
