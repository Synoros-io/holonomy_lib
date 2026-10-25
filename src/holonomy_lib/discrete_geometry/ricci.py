# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Synoros

"""Ollivier-Ricci curvature on graphs, GPU-batched, Sinkhorn-based.

The Ollivier (2009) discrete Ricci curvature on a metric measure space
is defined for any two points x, y as

    κ(x, y)  =  1  −  W_1( μ_x , μ_y ) / d(x, y)

where d is the metric, μ_x and μ_y are probability measures attached
to x and y (typically lazy-random-walk distributions), and W_1 is the
Wasserstein-1 distance between them. For graphs we follow the standard
choices:

    d   = shortest-path distance (hop count for unweighted graphs;
          weight-respecting for weighted graphs assuming edge weights
          are interpreted as lengths)
    μ_x = (1−α) · u_x  +  α · δ_x
          where u_x is the (weight-normalized) distribution over x's
          neighbors and δ_x is the unit mass at x.

The parameter α ∈ [0, 1] is the "laziness" parameter. α = 0 gives the
original Ollivier (2009) curvature; α = 1 makes the walk completely
lazy (returns trivial 0 curvature); the limit α → 1 with appropriate
rescaling gives the Liu-Lin-Yau (2011) curvature, which is the most
common convention in network analysis.

The W_1 computation uses entropic Sinkhorn (Cuturi 2013) with cost
matrix equal to the shortest-path distance, which is well-suited for
GPU batching since each iteration is a pair of matrix-vector products.
Entropic regularization introduces a small bias; we expose `reg` so
callers can trade accuracy for stability.

Why this approximates Perelman's Ricci flow on networks:

  Perelman (2003) studied ∂g/∂t = −2 Ric(g) on Riemannian 3-manifolds.
  In the discrete setting, Sia-Jonckheere-Bogdan (2019) and Ni et al.
  (2019) show that iterating w_ij(t+1) = (1 − δt · κ_ij(t)) · w_ij(t)
  on edge weights with κ = Ollivier-Ricci leads to a network analog of
  Ricci flow, with negatively-curved edges getting elongated (analog
  of forming neckpinches) until "surgery" — edge removal — separates
  the graph into geometric pieces (the network analog of Thurston's
  geometric decomposition). This module provides the κ primitive; the
  flow and surgery operations are planned follow-ons.

References:
  Ollivier, Y. (2009). Ricci curvature of Markov chains on metric
    spaces. Journal of Functional Analysis 256(3):810–864.
  Liu, F., Lin, Y., Yau, S.-T. (2011). Ricci curvature of graphs.
    Tohoku Math. Journal 63(4):605–627.  [Often cited as "Lin-Lu-Yau"
    after the related Lu coauthor on the broader programme.]
  Cuturi, M. (2013). Sinkhorn distances: lightspeed computation of
    optimal transport. Advances in Neural Information Processing
    Systems 26 (NIPS 2013):2292–2300.
  Sia, J., Jonckheere, E., Bogdan, P. (2019). Ollivier-Ricci
    curvature-based method to community detection in complex networks.
    Scientific Reports 9:9800.
  Ni, C.-C., Lin, Y.-Y., Luo, F., Gao, J. (2019). Community detection
    on networks with Ricci flow. Scientific Reports 9:9984.
"""

from __future__ import annotations

import warnings

import torch

from holonomy_lib._graph_utils import drop_self_loops

# Default regularization for entropic Sinkhorn, per Cuturi (2013) §4.
# A regularization of 0.01 is a common middle-ground: small enough that
# the entropic bias on W_1 is < 1% for typical edge-distance scales,
# large enough that the Sinkhorn iteration converges in O(100) steps
# without numerical underflow. Catalog name: `sinkhorn_reg_default`.
# Scale-of-validity: shortest-path costs in [1, ~diameter]; if the
# graph has a much wider distance scale, scale `reg` proportionally.
SINKHORN_REG_DEFAULT: float = 0.01

# Default Sinkhorn iteration count, per Cuturi (2013) §4 — 100 iterations
# achieves convergence (relative change < 1e-4) for graph-metric costs
# at the default regularization. Catalog: `sinkhorn_n_iter_default`.
SINKHORN_N_ITER_DEFAULT: int = 100

# Default convergence tolerance for Sinkhorn early stopping: stop when
# the max-abs change of the dual variable log_u between successive
# iterations falls below this value. Reuses the library's
# `numerical_floor_convention` (1e-9) — small enough that the
# transport plan is symmetric to well below typical use cases' noise
# floor, large enough that we don't chase machine-precision residuals.
# When `tol` is not exceeded after `n_iter` iterations, iteration
# stops at `n_iter`; we do not warn (caller chose n_iter).
SINKHORN_TOL_DEFAULT: float = 1e-9

# Sync cadence for Sinkhorn convergence checks. The `.item()` call
# on the delta tensor forces a host sync on GPU; doing it every iter
# adds tens of µs per iter to wait for the kernel queue. Checking
# every 8 iters keeps the asymptotic work the same (within 8 iters
# of true convergence) while cutting host sync count by 8×.
# **Scale of validity**: dimensionless cadence — does not need
# re-derivation across input sizes or precisions. Cataloged as
# `sinkhorn_sync_every_default`.
SINKHORN_SYNC_EVERY_DEFAULT: int = 8

# Default tile size (in pairs) for the batched Sinkhorn computation.
# The unrolled `(*batch, n², n)` source/target plus the inner broadcast
# `(*batch, n², n, n)` is the dominant memory cost of Ollivier curvature.
# Tiling caps the pairs processed simultaneously at this value, trading
# a small Python-loop overhead for an n²/tile_size reduction in peak
# memory. **Scale of validity**: 256 keeps the inner `n·n` broadcast
# under ~16 MB for n up to ~256 in float64, comfortable on most GPUs.
# Bump higher when memory headroom is generous and lower when n is
# big. Cataloged as `sinkhorn_tile_default`.
SINKHORN_TILE_DEFAULT: int = 256

# Convention: distance from a node to itself is 0; "no path" is +inf.
# We represent +inf with a large finite value derived from the matrix
# to avoid NaN propagation through Sinkhorn while still strongly
# disincentivizing transport across disconnected components.
DISCONNECTED_DISTANCE_MULTIPLIER: float = 1000.0

# Default surgery period for `ricci_flow_with_surgery` — perform surgery
# every N Ricci-flow steps. Ni-Lin-Luo-Gao (2019) §3.2 use values in
# [10, 15] for community detection; we default to 10 as the more
# aggressive choice. Catalog: `ricci_flow_surgery_period_default`.
RICCI_FLOW_SURGERY_PERIOD_DEFAULT: int = 10

# Default surgery threshold — edges whose weight grows past this value
# are removed as "neckpinches forming". Sia-Jonckheere-Bogdan (2019)
# and Ni et al. (2019) both use thresholds around 2-3× initial weight.
# Conservative default at 3.0 lets only clearly-stretched edges go.
# Catalog: `ricci_flow_surgery_threshold_default`.
RICCI_FLOW_SURGERY_THRESHOLD_DEFAULT: float = 3.0


def ollivier_ricci_curvature(
    A: torch.Tensor,
    alpha: float = 0.0,
    reg: float = SINKHORN_REG_DEFAULT,
    n_iter: int = SINKHORN_N_ITER_DEFAULT,
    tol: float = SINKHORN_TOL_DEFAULT,
    tile_size: int = SINKHORN_TILE_DEFAULT,
) -> torch.Tensor:
    """Pairwise Ollivier-Ricci curvature on a (batched) weighted graph.

    Args:
      A: (B, n, n) symmetric weighted adjacency. Non-negative entries
        interpreted as edge weights (larger = stronger connection,
        shorter as a length in the shortest-path metric). Zero off-
        diagonal means no edge.
      alpha: laziness in [0, 1]. 0 is Ollivier's original convention;
        0.5 is a common "half lazy" choice in network analysis.
      reg: entropic-Sinkhorn regularization ε > 0. See module docstring.
      n_iter: maximum Sinkhorn iteration count.
      tol: Sinkhorn convergence tolerance — iteration stops early once
        the max-abs change in log_u across all pairs is below `tol`.
        With log-domain Sinkhorn at small `reg`, partial convergence
        can plateau for hundreds of iterations and only flip into the
        symmetric basin past some threshold; tol-based stopping lets
        easy cases finish fast while still allowing hard cases to run
        the full `n_iter` budget.
      tile_size: chunk the n² pairs into tiles of this size before
        running Sinkhorn. The peak memory cost of one Sinkhorn iter
        is O(B · tile_size · n²) bytes; the wall-clock cost is
        roughly constant in `tile_size` once the GPU is saturated.
        Default 256 keeps the inner broadcast under ~16 MB at n=256
        in float64. Smaller tiles → less memory + slightly more
        Python overhead; larger tiles → more memory + slightly less
        loop overhead.

    Returns:
      κ: (B, n, n) curvature tensor. κ[..., i, j] is the curvature
        between nodes i and j using the shortest-path metric. The
        diagonal κ[..., i, i] = 1 by convention (W_1(δ, δ) = 0 and
        d(x, x) = 0 — we set the diagonal to 1 to avoid 0/0).
        Pairs from different connected components have undefined
        geometric meaning; we return 0 there to keep the tensor finite.

    Shape note:
      The result is dense across all pairs, not just edges. To get
      edge-only curvatures, mask: `κ * (A > 0)`. Computing all pairs
      costs O(B · n³ · n_iter) for the Sinkhorn step plus O(B · n³)
      for shortest paths; for large n use a sparse/edge-only follow-on
      (planned).

    Disconnected components:
      Ollivier curvature has no standard definition between nodes in
      different components. We replace the +inf shortest-path distance
      with a large finite sentinel (DISCONNECTED_DISTANCE_MULTIPLIER ×
      max finite distance) so Sinkhorn stays numerically stable; the
      resulting κ for cross-component pairs is close to 1 (since the
      transport cost is dominated by the within-component support and
      thus small relative to the inflated d_G). These cross-component
      values are NOT geometrically meaningful — mask by `A > 0` (or by
      a connectivity mask) before interpreting κ. The Ricci-flow
      primitives in this module already do this internally, so the
      flow is unaffected; only direct consumers of the dense κ tensor
      need to filter.

    References:
      Ollivier (2009), Definition 1.
      Cuturi (2013), Algorithm 1 — Sinkhorn iteration.
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    if reg <= 0:
        raise ValueError(f"reg must be > 0, got {reg}")
    if tol <= 0:
        raise ValueError(f"tol must be > 0, got {tol}")
    if A.ndim < 2 or A.shape[-1] != A.shape[-2]:
        raise ValueError(
            f"A must be (..., n, n) symmetric; got A.shape={tuple(A.shape)}"
        )
    # Ollivier curvature is defined for symmetric (undirected) graphs.
    # The Floyd-Warshall step happily returns directed shortest paths
    # if A is asymmetric; the resulting κ would be silently meaningless.
    # We warn rather than raise so callers can intentionally feed a
    # symmetrized form of an asymmetric matrix without first cloning
    # (`0.5*(A+A.mT)` is one common preprocess); the warning serves as
    # documentation that we noticed the asymmetry.
    # 1e-9 is the library's numerical_floor_convention (audit ALLOWED).
    # Roundoff on any reasonable symmetric construction is far below
    # 1e-9 even in float32, so this only fires when A is meaningfully
    # asymmetric (e.g. a directed graph passed by mistake).
    if not torch.allclose(A, A.mT, atol=1e-9, rtol=0):
        warnings.warn(
            "ollivier_ricci_curvature received an asymmetric adjacency; "
            "the curvature is only well-defined for symmetric (undirected) "
            "graphs. Symmetrize A first (e.g. `0.5 * (A + A.mT)`) for "
            "meaningful results.",
            stacklevel=2,
        )
    # Library convention (`CONVENTIONS.md`): graph primitives treat
    # `A` as a simple graph. Self-loops would change the row sums in
    # `_lazy_walk_distributions` and propagate through Sinkhorn.
    A = drop_self_loops(A)

    *batch, n, _ = A.shape

    # 1. Shortest-path distances d_G(i, j). Floyd-Warshall on the
    # weighted graph, batched. Treat A[i,j]=0 (off-diagonal) as no edge.
    d_G = _shortest_path_distances(A)  # (B, n, n)

    # 2. Lazy-random-walk distributions μ_x for each node x.
    #    μ_x(y) = (1−α) · (A[x,y] / Σ_z A[x,z])   for y in N(x)
    #    μ_x(x) = α (+ 0 if x is isolated, in which case μ_x = δ_x)
    mu = _lazy_walk_distributions(A, alpha)  # (B, n, n) — μ_x is row x

    # 3. Sinkhorn-based W_1 between μ_x and μ_y for every pair (x, y).
    #    All pairs of rows of `mu` — the cost matrix is d_G.
    W1 = _batched_sinkhorn_w1(
        mu, d_G, reg=reg, n_iter=n_iter, tol=tol, tile_size=tile_size,
    )  # (B, n, n)

    # 4. Curvature κ(x, y) = 1 - W_1 / d_G.
    # Diagonal: d_G is 0 there; set κ to 1 by convention (vacuous).
    safe_dG = torch.where(d_G > 1e-9, d_G, torch.full_like(d_G, 1e-9))
    kappa = 1.0 - W1 / safe_dG
    # Diagonal entries: 1 by convention.
    eye = torch.eye(n, device=A.device, dtype=A.dtype).expand(*batch, n, n)
    kappa = torch.where(eye > 0, torch.ones_like(kappa), kappa)
    return kappa


# ============================================================
# Discrete Ricci flow (Perelman-on-networks)
# ============================================================


def discrete_ricci_flow(
    A: torch.Tensor,
    n_steps: int,
    dt: float = 1.0,
    alpha: float = 0.0,
    normalize: bool = True,
    reg: float = SINKHORN_REG_DEFAULT,
    n_sinkhorn_iters: int = SINKHORN_N_ITER_DEFAULT,
    sinkhorn_tol: float = SINKHORN_TOL_DEFAULT,
    sinkhorn_tile_size: int = SINKHORN_TILE_DEFAULT,
) -> torch.Tensor:
    """Discrete Ricci flow on edge weights (Sia 2019, Ni 2019).

    Iterates the edge-weight update

        w_ij(t+1) = (1 − dt · κ_ij(t)) · w_ij(t)

    where κ_ij is the Ollivier-Ricci curvature at edge (i, j). Edges
    with negative curvature (κ < 0) get stretched (multiplier > 1);
    edges with positive curvature (κ > 0) get shortened. Over many
    steps this is the discrete analog of Perelman's smooth Ricci flow
    on networks: bottleneck edges between communities elongate while
    intra-community edges contract.

    Args:
      A: (B, n, n) symmetric weighted adjacency. Non-negative weights.
      n_steps: number of flow iterations.
      dt: step size. Smaller → more numerically stable, slower convergence.
      alpha: laziness parameter forwarded to `ollivier_ricci_curvature`.
      normalize: if True, rescale edge weights after each step so the
        Frobenius norm of A stays constant. Prevents global scaling
        from running away, isolating the *structural* effect of the flow.
      reg, n_sinkhorn_iters: Sinkhorn parameters for curvature
        computation.

    Returns:
      (B, n, n) adjacency after `n_steps` of flow.

    References:
      Sia, J., Jonckheere, E., Bogdan, P. (2019). Ollivier-Ricci
        curvature-based method to community detection in complex
        networks. Scientific Reports 9:9800, eq. 3.
      Ni, C.-C., Lin, Y.-Y., Luo, F., Gao, J. (2019). Community
        detection on networks with Ricci flow. Scientific Reports
        9:9984.
    """
    if n_steps < 0:
        raise ValueError(f"n_steps must be >= 0, got {n_steps}")
    if dt <= 0:
        raise ValueError(f"dt must be > 0, got {dt}")

    W = A.clone()
    if normalize:
        initial_norm = torch.linalg.matrix_norm(W, dim=(-2, -1), keepdim=True)
        initial_norm = initial_norm.clamp(min=1e-9)

    for _ in range(n_steps):
        # Compute curvature on current edge weights
        kappa = ollivier_ricci_curvature(
            W, alpha=alpha, reg=reg, n_iter=n_sinkhorn_iters,
            tol=sinkhorn_tol, tile_size=sinkhorn_tile_size,
        )
        # Update: w *= (1 - dt * kappa). Mask to existing edges only;
        # non-edges (W == 0) stay zero — surgery is separate.
        edge_mask = (W > 1e-9).to(W.dtype)
        update = (1.0 - dt * kappa) * edge_mask
        W = W * update + W * (1.0 - edge_mask)  # leave non-edges alone
        # Clamp negative weights to zero (edge would have flipped sign)
        W = W.clamp(min=0)
        # Maintain symmetry (curvature is symmetric, but float drift)
        W = 0.5 * (W + W.mT)
        if normalize:
            current_norm = torch.linalg.matrix_norm(
                W, dim=(-2, -1), keepdim=True,
            ).clamp(min=1e-9)
            W = W * (initial_norm / current_norm)
    return W


def ricci_flow_with_surgery(
    A: torch.Tensor,
    n_steps: int,
    surgery_period: int = RICCI_FLOW_SURGERY_PERIOD_DEFAULT,
    surgery_threshold: float = RICCI_FLOW_SURGERY_THRESHOLD_DEFAULT,
    dt: float = 1.0,
    alpha: float = 0.0,
    normalize: bool = True,
    reg: float = SINKHORN_REG_DEFAULT,
    n_sinkhorn_iters: int = SINKHORN_N_ITER_DEFAULT,
    sinkhorn_tol: float = SINKHORN_TOL_DEFAULT,
    sinkhorn_tile_size: int = SINKHORN_TILE_DEFAULT,
) -> torch.Tensor:
    """Discrete Ricci flow with surgery — Perelman-on-networks.

    Alternates Ricci-flow steps (edges evolve by their curvature) with
    surgery passes (heavily-stretched edges are removed). The standard
    use case is community detection: after enough flow steps with
    surgery, the bottleneck edges between communities are gone and the
    graph splits into geometric pieces — the discrete analog of
    Thurston's geometrization that Perelman (2003) proved smoothly
    for 3-manifolds.

    Algorithm (Ni-Lin-Luo-Gao 2019, §3.2):
      Repeat `n_steps` times:
        1. Take one Ricci-flow step: w ← (1 − dt · κ) · w.
        2. Renormalize if requested.
        3. Every `surgery_period` steps, surgery: set w_ij = 0 for any
           edge whose current weight exceeds `surgery_threshold` ×
           initial mean edge weight. These are the "necks" forming
           around forming singularities.

    Args:
      A: (B, n, n) symmetric weighted adjacency.
      n_steps: total flow iterations.
      surgery_period: perform surgery every N steps. Catalog default.
      surgery_threshold: edge-removal threshold, as a multiplier of
        initial mean edge weight. Catalog default. Note that when
        `normalize=True`, the cutoff uses the **pre-normalization**
        initial mean as a fixed reference; the per-step normalization
        rescales the weights but the threshold does not move with it.
        This matches Ni-Lin-Luo-Gao (2019)'s intent of "remove edges
        that have stretched far past their initial relative scale," but
        the practical sensitivity of the surgery is then conditioned on
        the interaction between `dt`, `normalize`, and the curvature
        magnitudes — tune `surgery_threshold` empirically per dataset.
      Other args: as in `discrete_ricci_flow`.

    Returns:
      (B, n, n) adjacency after flow + surgery. Disconnected components
      indicate detected communities.

    References:
      Perelman, G. (2002, 2003). The entropy formula for the Ricci flow
        and its geometric applications; Ricci flow with surgery on
        three-manifolds; Finite extinction time. arXiv:math/0211159,
        math/0303109, math/0307245. (Smooth original.)
      Sia-Jonckheere-Bogdan (2019), Ni-Lin-Luo-Gao (2019). (Discrete
        analog for network community detection.)
      Liu, F., Wang, X., Yau, S.-T., Zeng, W. (2017). A realization of
        Thurston's geometrization: discrete Ricci flow with surgery.
        arXiv:1709.08494. (Discrete on 3D simplicial complexes — closest
        in spirit to Perelman.)
    """
    if surgery_period <= 0:
        raise ValueError(f"surgery_period must be > 0, got {surgery_period}")
    if surgery_threshold <= 0:
        raise ValueError(
            f"surgery_threshold must be > 0, got {surgery_threshold}"
        )

    W = A.clone()
    # Initial mean edge weight — the scale against which we threshold.
    edge_mask_init = (W > 1e-9).to(W.dtype)
    n_edges = edge_mask_init.sum(dim=(-2, -1), keepdim=True).clamp(min=1.0)
    initial_mean = (W * edge_mask_init).sum(
        dim=(-2, -1), keepdim=True,
    ) / n_edges
    cutoff = surgery_threshold * initial_mean

    if normalize:
        initial_norm = torch.linalg.matrix_norm(
            W, dim=(-2, -1), keepdim=True,
        ).clamp(min=1e-9)

    for step in range(n_steps):
        # One Ricci-flow step
        kappa = ollivier_ricci_curvature(
            W, alpha=alpha, reg=reg, n_iter=n_sinkhorn_iters,
            tol=sinkhorn_tol, tile_size=sinkhorn_tile_size,
        )
        edge_mask = (W > 1e-9).to(W.dtype)
        update = (1.0 - dt * kappa) * edge_mask
        W = W * update + W * (1.0 - edge_mask)
        W = W.clamp(min=0)
        W = 0.5 * (W + W.mT)
        if normalize:
            current_norm = torch.linalg.matrix_norm(
                W, dim=(-2, -1), keepdim=True,
            ).clamp(min=1e-9)
            W = W * (initial_norm / current_norm)

        # Surgery: remove edges that have stretched past the cutoff.
        # `step + 1` because we want surgery after the FIRST flow step
        # for surgery_period = 1 (i.e. surgery every step).
        if (step + 1) % surgery_period == 0:
            keep = W <= cutoff
            W = W * keep.to(W.dtype)
            W = 0.5 * (W + W.mT)

    return W


# ============================================================
# Internal helpers
# ============================================================


def _shortest_path_distances(A: torch.Tensor) -> torch.Tensor:
    """Batched Floyd-Warshall on a weighted adjacency.

    Treats A[i,j] = 0 (off-diagonal, i ≠ j) as "no direct edge" — distance
    infinity. Diagonal is forced to 0. Symmetric (we don't enforce, but
    if A is non-symmetric this returns the directed shortest path).

    For disconnected pairs we return a large finite value rather than
    +inf, so downstream Sinkhorn computations don't underflow exp(−d/ε).
    """
    *batch, n, _ = A.shape
    # Build initial distance matrix: A[i,j] where i≠j and A[i,j]>0, else inf.
    eye = torch.eye(n, device=A.device, dtype=A.dtype).expand(*batch, n, n)
    off_diag = (eye == 0)
    no_edge = off_diag & (A == 0)
    # Use a finite "infinity" for numerical stability; will be relaxed
    # to a large multiple of the diameter at the end if any pair is
    # still unreachable.
    inf_sentinel = torch.full_like(A, float("inf"))
    D = torch.where(no_edge, inf_sentinel, A.clone())
    # Force diagonal to 0
    D = torch.where(eye > 0, torch.zeros_like(D), D)

    # Floyd-Warshall: for k in 0..n-1: D[i,j] = min(D[i,j], D[i,k] + D[k,j]).
    # Batched: vectorize over (i, j); k is the loop axis.
    for k in range(n):
        # via_k[..., i, j] = D[..., i, k] + D[..., k, j]
        via_k = D[..., :, k : k + 1] + D[..., k : k + 1, :]
        D = torch.minimum(D, via_k)

    # Replace any remaining +inf with a large finite value derived from
    # the existing finite distances, to keep Sinkhorn numerics sane.
    # Always run the replacement (no `.all()` host sync on the happy
    # path) — `torch.where` is a near-noop when no infs are present,
    # and the explicit branch would force a GPU→CPU sync every call.
    finite_mask = torch.isfinite(D)
    D_finite = torch.where(finite_mask, D, torch.zeros_like(D))
    per_batch_max = D_finite.flatten(start_dim=-2).max(dim=-1).values  # (..., )
    big = (per_batch_max * DISCONNECTED_DISTANCE_MULTIPLIER).clamp(min=1.0)
    big_full = big.view(*per_batch_max.shape, 1, 1).expand_as(D)
    return torch.where(finite_mask, D, big_full)


def _lazy_walk_distributions(A: torch.Tensor, alpha: float) -> torch.Tensor:
    """Lazy-random-walk distributions for every node.

    Row x of the output is μ_x, where
      μ_x(x) = α   (plus 1−α concentrated on x if x is isolated)
      μ_x(y) = (1−α) · A[x,y] / Σ_z A[x,z]   for y ≠ x with A[x,z]>0.

    Returns:
      (..., n, n) tensor; row x is μ_x as a probability vector.
    """
    n = A.shape[-1]
    row_sums = A.sum(dim=-1, keepdim=True)  # (..., n, 1)
    is_isolated = row_sums.squeeze(dim=-1) <= 0  # (..., n)

    # Avoid div-by-zero: use a safe denominator
    safe_row_sums = torch.where(
        row_sums > 0, row_sums, torch.ones_like(row_sums),
    )
    walk = A / safe_row_sums  # (..., n, n) — rows are normalized

    # Lazy mixture: (1−α) * walk + α * δ_x (identity rows)
    eye = torch.eye(n, device=A.device, dtype=A.dtype).expand_as(A)
    mu = (1.0 - alpha) * walk + alpha * eye

    # Isolated nodes default to δ_x (mass on themselves)
    delta_rows = eye
    isolated_mask = is_isolated.unsqueeze(dim=-1).expand_as(mu)
    return torch.where(isolated_mask, delta_rows, mu)


def _batched_sinkhorn_w1(
    mu: torch.Tensor,
    cost: torch.Tensor,
    reg: float,
    n_iter: int,
    tol: float,
    tile_size: int,
) -> torch.Tensor:
    """W_1(μ_i, μ_j) for every pair (i, j) of rows of `mu`, Sinkhorn-based.

    Args:
      mu:   (..., n, n) — row i is the source distribution μ_i.
      cost: (..., n, n) — pairwise cost matrix (used for all pairs).
      reg:  entropic regularization ε > 0.
      n_iter: maximum Sinkhorn iteration count.
      tol: stop early when max|Δ log_u| across all pairs is < tol.
      tile_size: max pairs processed simultaneously inside the loop.

    Returns:
      (..., n, n) — entry [i, j] is the Sinkhorn-approximated
      W_1(μ_i, μ_j) using the given cost.

    Algorithm (vectorized across all source/target pairs):
      Stack pairs as a batch of n² OT problems with a shared cost.
      Iterate Sinkhorn updates in log-domain for stability.

    Notes:
      Log-domain Sinkhorn (Cuturi 2013 §4 + Schmitzer 2019) avoids
      underflow when reg is small. We use it unconditionally for
      robustness; the cost is one extra logsumexp per iteration.
    """
    # Shapes:
    #   *batch, n, n — for both mu and cost
    *batch, n, _ = mu.shape
    n_pairs = n * n

    # Cost is shared across all pairs. The log-kernel `-cost / reg` is
    # also shared; compute once. cost_b has the (..., 1, n, n) shape so
    # broadcasts cleanly against (..., tile, n, ·) duals.
    cost_b = cost.unsqueeze(dim=-3)            # (*batch, 1, n, n)
    log_K = -cost_b / reg                       # (*batch, 1, n, n)
    # Library numerical_floor_convention (1e-9 = `ALLOWED_LITERALS`).
    log_mu = torch.log(mu.clamp(min=1e-9))      # (*batch, n, n)

    # Output buffer for all pair W_1 values.
    sink_w1 = torch.empty(*batch, n_pairs, dtype=mu.dtype, device=mu.device)

    # Tile over pairs to keep the inner broadcast tensor bounded.
    # Pair p ∈ [0, n²) decomposes as p = i·n + j where i is the source
    # row index and j the target row index in `mu`. We compute the
    # (i, j) indices via the arange→divmod trick so we never have to
    # materialize a full (n², n) source or target tensor — each tile
    # gathers only its own slice.
    pair_idx = torch.arange(n_pairs, device=mu.device)
    src_idx_all = pair_idx // n   # (n_pairs,)
    tgt_idx_all = pair_idx % n    # (n_pairs,)

    chunk = max(1, int(tile_size))
    for start in range(0, n_pairs, chunk):
        end = min(start + chunk, n_pairs)
        src_idx = src_idx_all[start:end]       # (tile,)
        tgt_idx = tgt_idx_all[start:end]       # (tile,)
        # Gather per-tile sources and targets via advanced indexing.
        # Result shapes: (*batch, tile, n).
        log_src = log_mu[..., src_idx, :]
        log_tgt = log_mu[..., tgt_idx, :]

        log_u = torch.zeros_like(log_src)
        log_v = torch.zeros_like(log_tgt)
        sync_cadence = SINKHORN_SYNC_EVERY_DEFAULT
        prev_log_u = log_u
        for it in range(n_iter):
            # Sum over target support — `dim=-1`. log_K broadcasts from
            # (*batch, 1, n, n) to (*batch, tile, n, n).
            log_u = log_src - torch.logsumexp(
                log_K + log_v.unsqueeze(dim=-2), dim=-1,
            )
            # Sum over source support — `dim=-2`.
            log_v = log_tgt - torch.logsumexp(
                log_K + log_u.unsqueeze(dim=-1), dim=-2,
            )
            # Early stop. The `.item()` host sync amortizes when checked
            # every `sync_cadence` iters: typically a 8× reduction in
            # GPU→CPU syncs versus per-iter checking, at the cost of
            # at most `sync_cadence - 1` extra iters past true
            # convergence. Bracket the check by `it >= sync_cadence` so
            # the loop never exits on iter 0 (warmup).
            if (it + 1) % sync_cadence == 0:
                delta = (log_u - prev_log_u).abs().max().item()
                if delta < tol:
                    break
                prev_log_u = log_u

        # Transport plan + transport cost for this tile.
        log_pi = log_u.unsqueeze(dim=-1) + log_K + log_v.unsqueeze(dim=-2)
        pi = torch.exp(log_pi)
        sink_w1[..., start:end] = (pi * cost_b).sum(dim=(-2, -1))

    return sink_w1.reshape(*batch, n, n)
