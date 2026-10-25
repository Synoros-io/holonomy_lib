# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Synoros

"""Forman-Ricci curvature on graphs, batched-first, GPU-native.

Forman (2003) defined a combinatorial Ricci curvature for general CW
complexes that specializes to a very simple per-edge formula on graphs.
Unlike Ollivier-Ricci (`ricci.ollivier_ricci_curvature`), no optimal-
transport solve is needed: each edge's curvature is a closed-form
function of the local degrees and (optionally) the triangles incident
to it. The cost is therefore O(B · |E|) for the simple form or
O(B · |E| · max_deg²) for the triangle-aware form, dramatically cheaper
than Sinkhorn-based Ollivier.

Forman-Ricci is less geometrically faithful than Ollivier on weighted
metric measure spaces, but it tracks the same qualitative bottleneck
structure: edges bridging two clusters get negative curvature, edges
embedded in dense sub-communities get positive curvature. For large
networks where Ollivier is infeasible, Forman-Ricci is the standard
substitute (Sreejith et al. 2016).

Cost note: the implementations here are O(B · n²) for the simple
form and O(B · n³) for the augmented form (the triangle-count
matmul). A future sparse path would be O(B · |E| · max_deg²) for
the augmented form; the dense path is fine up to n ~ 10⁴.

Two forms are exposed:

  forman_ricci_simple(A)
      κ_F(u, v) = w_e · ( w_u/w_e + w_v/w_e
                          - Σ_{e_u ≠ e} w_u/√(w_e · w_{e_u})
                          - Σ_{e_v ≠ e} w_v/√(w_e · w_{e_v}) )
      The original CW-complex specialization without 2-faces (Sreejith
      et al. 2016, eq. 1). On unweighted simple graphs with node
      weights = 1 this collapses to `4 - deg(u) - deg(v)`.

  forman_ricci_augmented(A)
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
    §"Augmented Forman curvature" gives the +3·#triangles term.
  Sandhu, R., Georgiou, T., Tannenbaum, A. (2015). Ricci curvature: An
    economic indicator for market fragility and systemic risk.
    Science Advances 2(5):e1501495. Application demonstrating Forman's
    qualitative agreement with Ollivier at lower compute cost.
"""

from __future__ import annotations

import warnings

import torch

from holonomy_lib._graph_utils import drop_self_loops

# Forman's "node weights" — Samal et al. (2018), §2.1: the standard
# choice for unweighted simple graphs is w_u = 1 for every node, so the
# curvature reduces to a pure function of degrees. Weighted-node Forman
# is an extension we don't expose here (caller can pre-scale the
# adjacency to fold node weights into the edge weights if needed).
# Cataloged as `forman_node_weight_default` (universal-1 identity, no
# tuning needed).
FORMAN_NODE_WEIGHT_DEFAULT: float = 1.0

# Augmented-Forman contribution per incident 2-face (triangle).
# Samal et al. (2018), §"Augmented Forman curvature" — each triangle
# contributes `+1` per face times the standard cell-complex Forman
# formula's prefactor. For graphs the constant works out to 3 because a
# triangle's three edges each see one triangular 2-face. This is a
# DERIVED structural identity, not a tuned constant.
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
        otherwise. The diagonal is set to 0 by convention (Forman is
        defined per edge, not per node-self-loop).

    Notes:
      Cost is O(B · n²) for the per-pair sums; for sparse graphs an
      edge-list variant (planned) would be O(B · |E|).

    References:
      Sreejith et al. (2016), eq. 1.
      Forman (2003), §"Combinatorial Ricci curvature for graphs".
    """
    _validate_adjacency(A)
    A = drop_self_loops(A)
    *batch, n, _ = A.shape

    # Mask of which positions are edges. We avoid `A > 0` host syncs by
    # working in tensor form throughout. Non-edges contribute zero to
    # the sums by construction.
    edge_mask = (A != 0).to(A.dtype)  # (B, n, n)

    # w_u = node weight = 1 (Sreejith et al. convention). When the
    # library grows weighted-node Forman we'd pass w as an argument; for
    # now the constant is folded in at the value level.
    w_u = FORMAN_NODE_WEIGHT_DEFAULT
    w_v = FORMAN_NODE_WEIGHT_DEFAULT

    # Safe edge weights — replace zeros with ones in the denominator
    # paths so we never divide by zero. The edge_mask multiplier zeroes
    # the contribution at non-edges.
    safe_we = torch.where(edge_mask > 0, A, torch.ones_like(A))

    # First two diagonal-term contributions per edge: w_u/w_e + w_v/w_e
    # = (w_u + w_v) / w_e.
    diagonal_term = (w_u + w_v) / safe_we  # (B, n, n)

    # Sum-over-neighbors term: for each edge (u, v), we need
    #   Σ_{u' ≠ v, A[u, u']>0} w_u / √(w_e · A[u, u'])
    # Build a "1/√A" tensor with zeros at non-edges; the row-sum minus
    # the diagonal (i.e., minus the edge (u, v) itself) gives the
    # Σ_{u' ≠ v} term.
    inv_sqrt_A = torch.where(
        edge_mask > 0, torch.rsqrt(safe_we), torch.zeros_like(A),
    )  # (B, n, n)

    # For each edge (u, v):
    #   row_sum_u  = Σ_{u'} 1/√A[u, u']   over all neighbors u' of u
    #   contribution from u-side (excluding (u, v) itself) =
    #     (row_sum_u - 1/√A[u, v]) · w_u / √A[u, v]
    #   = w_u · (row_sum_u - inv_sqrt_A[u, v]) · inv_sqrt_A[u, v]
    # Equivalently using broadcasting:
    row_sum = inv_sqrt_A.sum(dim=-1, keepdim=True)         # (B, n, 1)
    col_sum = inv_sqrt_A.sum(dim=-2, keepdim=True)         # (B, 1, n)
    # row_sum broadcast over columns gives, at position (u, v), the
    # sum over u's neighbors of 1/√A[u, ·]; subtract the (u, v) entry
    # to exclude the edge itself.
    u_neighbor_sum = row_sum - inv_sqrt_A                  # (B, n, n)
    v_neighbor_sum = col_sum - inv_sqrt_A                  # (B, n, n)
    u_side = w_u * inv_sqrt_A * u_neighbor_sum             # (B, n, n)
    v_side = w_v * inv_sqrt_A * v_neighbor_sum             # (B, n, n)

    # Assemble curvature; non-edges multiplied by edge_mask → 0.
    kappa = safe_we * (diagonal_term - u_side - v_side) * edge_mask

    # Force diagonal to zero (Forman per-edge; no self-loop contribution).
    eye = torch.eye(n, device=A.device, dtype=A.dtype).expand_as(A)
    kappa = torch.where(eye > 0, torch.zeros_like(kappa), kappa)
    return kappa


def forman_ricci_augmented(A: torch.Tensor) -> torch.Tensor:
    """Augmented Forman-Ricci curvature: simple Forman + 2-face term.

    Adds the triangle (2-face) contribution from Samal et al. (2018):

      κ_F^aug(u, v) = κ_F_simple(u, v) + 3 · #triangles(u, v)

    where #triangles(u, v) counts the common neighbors of u and v
    (each common neighbor w forms a triangle u-v-w). The "+3" factor
    arises because each triangular 2-face contributes one unit of
    curvature to each of its three edges, and Forman's prefactor in
    the cell-complex formula carries through as 1 here.

    Augmented Forman tracks Ollivier-Ricci more closely than simple
    Forman in graphs with dense substructures (Samal et al. 2018,
    Fig. 3).

    Args:
      A: (B, n, n) symmetric weighted adjacency. Non-negative weights;
        zero off-diagonal means no edge.

    Returns:
      κ_F^aug: (B, n, n) curvature tensor.

    References:
      Samal et al. (2018), §"Augmented Forman curvature".
    """
    _validate_adjacency(A)
    A = drop_self_loops(A)

    # Common-neighbor count, masked to actual edges.
    # For unweighted graphs (A binary), (A @ A)[u, v] counts the number
    # of length-2 paths u → w → v, which equals the number of common
    # neighbors. For weighted graphs we follow the standard convention
    # of using the edge presence (A > 0), not the weights themselves —
    # the augmented form's 2-face contribution is combinatorial in
    # nature. (A weighted 2-face theory exists but is not implemented.)
    edge_mask = (A != 0).to(A.dtype)
    # IMPORTANT: zero the diagonal of edge_mask BEFORE the matmul.
    # Otherwise a self-loop at node i contributes spurious length-2
    # walks `j → i → i → k` to the triangle count for every edge
    # incident to i, double-counting the augmented term.
    *batch, n, _ = A.shape
    eye = torch.eye(n, device=A.device, dtype=A.dtype).expand_as(A)
    edge_mask_no_diag = torch.where(
        eye > 0, torch.zeros_like(edge_mask), edge_mask,
    )
    triangle_count = torch.matmul(edge_mask_no_diag, edge_mask_no_diag)
    # Only count triangles on actual edges; mask out non-edges. We use
    # `edge_mask_no_diag` here too — a self-loop "edge" shouldn't get
    # an augmented Forman contribution.
    triangle_count = triangle_count * edge_mask_no_diag

    return (
        forman_ricci_simple(A)
        + FORMAN_TRIANGLE_CONTRIBUTION * triangle_count
    )


# ============================================================
# Internal helpers
# ============================================================


def _validate_adjacency(A: torch.Tensor) -> None:
    if A.ndim < 2 or A.shape[-1] != A.shape[-2]:
        raise ValueError(
            f"A must be (..., n, n); got A.shape={tuple(A.shape)}"
        )
    if (A < 0).any():
        raise ValueError(
            "Forman-Ricci is defined for non-negative edge weights; "
            "negative entries detected."
        )
    # Symmetry check matches the convention in `ollivier_ricci_curvature`.
    if not torch.allclose(A, A.mT, atol=1e-9, rtol=0):
        warnings.warn(
            "forman_ricci received an asymmetric adjacency; Forman-Ricci is "
            "only well-defined for symmetric (undirected) graphs. Symmetrize "
            "A first (e.g. `0.5 * (A + A.mT)`) for meaningful results.",
            stacklevel=2,
        )
