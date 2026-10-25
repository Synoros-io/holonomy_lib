# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Synoros

"""Batched Ollivier-Ricci curvature and discrete Ricci flow on graphs.

The metric is shortest-path distance with adjacency weights interpreted as
edge lengths. Those same weights determine the neighbor probabilities of
mu_x = (1-alpha) A[x, :] / sum(A[x, :]) + alpha delta_x. Curvature is
kappa(x, y) = 1 - W_1(mu_x, mu_y) / d(x, y).

Stabilized Sinkhorn on neighborhood supports approximates transport with
entropic regularization and warm-started epsilon scaling. A small
regularization does not guarantee convergence:
callers control the iteration budget and marginal tolerance, and receive
a warning when the returned transport plan fails that tolerance.

Flow updates edge lengths by w <- (1 - dt*kappa) w. The surgery variant
removes edges above a fixed multiple of their initial mean weight.
Floyd-Warshall, Sinkhorn, and flow steps are sequential recurrences;
the arithmetic within each step is batched on PyTorch.

References:
  Ollivier, Y. (2009). Ricci curvature of Markov chains on metric
    spaces. Journal of Functional Analysis 256(3):810–864.
  Lin, Y., Lu, L., Yau, S.-T. (2011). Ricci curvature of graphs.
    Tohoku Mathematical Journal 63(4):605–627. The rescaled alpha -> 1
    limit is the Lin-Lu-Yau curvature, not computed by this module.
  Cuturi, M. (2013). Sinkhorn distances: lightspeed computation of
    optimal transport. Advances in Neural Information Processing
    Systems 26:2292–2300.
  Schmitzer, B. (2019). Stabilized sparse scaling algorithms for entropy
    regularized transport problems. SIAM Journal on Scientific Computing
    41(3):A1443–A1481, Algorithms 2 and 3.
  Sia, J., Jonckheere, E., Bogdan, P. (2019). Ollivier-Ricci
    curvature-based method to community detection in complex networks.
    Scientific Reports 9:9800.
  Ni, C.-C., Lin, Y.-Y., Luo, F., Gao, J. (2019). Community detection
    on networks with Ricci flow. Scientific Reports 9:9984.
"""

from __future__ import annotations

import math
import warnings

import torch

from holonomy_lib._graph_utils import drop_self_loops

# Chosen with the iteration budget and marginal tolerance below against exact
# LP transport on karate, paths, cycles, cliques, a clique ring, a weighted
# random graph and a stochastic block graph (2026-10-25). Maximum edge-curvature
# error was below 1e-2. This is in edge-length units: the validation used unit
# lengths and weights drawn in (0, 1) or [0.5, 1.5], not arbitrary rescalings.
SINKHORN_REG_DEFAULT: float = 0.003

# Total scaling sweeps, including epsilon warmup. The validation graphs above
# converged within this budget; tiles stop as soon as all their pairs finish.
# This is an empirical budget, not a convergence guarantee for other inputs.
SINKHORN_N_ITER_DEFAULT: int = 65536

# Maximum absolute error in either marginal, chosen alongside reg and n_iter
# in the exact-W1 validation above. It permits float32 roundoff while keeping
# the observed curvature error below 1e-2; it does not bound entropic bias.
SINKHORN_TOL_DEFAULT: float = 1e-4

# Schmitzer (2019), Algorithm 3, allows a geometric epsilon schedule. We halve
# epsilon only once a stage meets the stage tolerance (the caller's tol, or
# SINKHORN_TOL_DEFAULT if the caller's is tighter). That way a small mass
# deficit is not carried into a colder problem. A stricter caller tolerance is
# held back for the target epsilon, since oversolving the warmup stages can use
# up the budget before we reach the problem the caller asked for.
SINKHORN_EPSILON_DECAY: float = 0.5

# Schmitzer (2019), section 5, uses tau=100 for absorption. We compare the
# accumulated log scaling against log(tau) to decide when to rebuild a
# normalized kernel from the dual potentials.
SINKHORN_ABSORPTION_THRESHOLD: float = 100.0

# Sync cadence for Sinkhorn convergence checks. The `.item()` call on the
# marginal-error tensor forces a host sync on GPU, and doing it every iteration
# adds tens of µs per iteration waiting on the kernel queue. Checking every 8
# iterations overshoots true convergence by at most 8 sweeps while cutting the
# host sync count by 8×.
# **Scale of validity**: a dimensionless cadence, so it does not need
# re-derivation across input sizes or precisions. Cataloged as
# `sinkhorn_sync_every_default`.
SINKHORN_SYNC_EVERY_DEFAULT: int = 8

# Default tile size (in pairs) for the batched Sinkhorn computation.
# Tiling caps simultaneous support-cost matrices at B*tile_size*s*s, where
# s <= max_degree+1. At s=256 one float64 temporary is 128 MiB per batch
# member; sparse graphs use less. Cataloged as `sinkhorn_tile_default`.
SINKHORN_TILE_DEFAULT: int = 256

# Default surgery period for `ricci_flow_with_surgery`: surgery runs once
# every N Ricci-flow steps. Ni-Lin-Luo-Gao (2019) §3.2 use values in
# [10, 15] for community detection; we default to 10 as the more
# aggressive choice. Catalog: `ricci_flow_surgery_period_default`.
RICCI_FLOW_SURGERY_PERIOD_DEFAULT: int = 10

# Default surgery threshold, as a multiple of the initial mean edge weight.
# Edges whose weight grows past this multiple are removed as forming
# neckpinches. Sia-Jonckheere-Bogdan (2019) and Ni et al. (2019) both use
# thresholds around 2-3× initial weight. The conservative end, 3.0, lets
# only clearly stretched edges go.
# Catalog: `ricci_flow_surgery_threshold_default`.
RICCI_FLOW_SURGERY_THRESHOLD_DEFAULT: float = 3.0


def ollivier_ricci_curvature(
    A: torch.Tensor,
    alpha: float = 0.0,
    reg: float = SINKHORN_REG_DEFAULT,
    n_iter: int = SINKHORN_N_ITER_DEFAULT,
    tol: float = SINKHORN_TOL_DEFAULT,
    tile_size: int = SINKHORN_TILE_DEFAULT,
    sync_every: int = SINKHORN_SYNC_EVERY_DEFAULT,
) -> torch.Tensor:
    """Pairwise Ollivier-Ricci curvature on a batched weighted graph.

    Args:
      A: (B, n, n) real floating, symmetric non-negative adjacency.
        Positive entries are edge lengths, used directly in shortest
        paths and also as weights in the neighbor walk probabilities.
        Zero off-diagonal entries mean no edge; self-loops are ignored.
      alpha: laziness in [0, 1].
      reg: positive entropic regularization, in the units of edge length.
      n_iter: positive maximum scaling sweeps, including epsilon warmup.
      tol: positive absolute tolerance for both transport marginals.
        This checks feasibility, not the entropic approximation bias.
      tile_size: positive number of node pairs per Sinkhorn tile.
      sync_every: positive interval between marginal checks that sync
        with the host. The final iteration is always checked.

    Returns:
      (B, n, n) curvature tensor, with diagonal 1 by convention and
      cross-component entries 0. Cross-component curvature is undefined;
      zero is an output convention, not a transport result.

    Notes:
      A warning reports the achieved marginal error if the iteration
      budget is exhausted without meeting tol. The returned approximation
      can then be infeasible. Zero-probability entries (self when alpha is 0)
      carry no transport mass.
      Padded supports contain at most s=max_degree+1 entries. Symmetry
      permits solving each unordered pair once. Transport costs
      O(B*n**2*s**2*n_iter), with O(B*tile_size*s**2) temporary storage;
      Floyd-Warshall costs O(B*n**3). Scaling sweeps and shortest-path
      recurrence steps are sequential; batches and pairs are vectorized.
      Epsilon decreases from each pair's cost scale to reg, preserving
      dimensional dual potentials. Convergence is checked only at reg.
      The defaults were checked on unit-length and order-one weighted
      graphs; rescale reg with edge lengths to retain comparable bias.
      Very small reg on weighted graphs can still stall despite epsilon
      scaling. Large iteration budgets alone do not ensure practical
      convergence; heed the marginal-error warning in that regime.

    References:
      Ollivier (2009), Definition 1.
      Cuturi (2013), Algorithm 1, for entropic Sinkhorn transport.
      Schmitzer (2019), Algorithms 2 and 3, for stabilized scaling and
        warm starts across decreasing regularization parameters.
    """
    return _curvature(A, alpha, reg, n_iter, tol, tile_size, sync_every)


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
    sinkhorn_sync_every: int = SINKHORN_SYNC_EVERY_DEFAULT,
) -> torch.Tensor:
    """Discrete Ricci flow on edge weights (Sia 2019, Ni 2019).

    Iterates the edge-weight update

        w_ij(t+1) = (1 − dt · κ_ij(t)) · w_ij(t)

    where κ_ij is the Ollivier-Ricci curvature at edge (i, j). Edges
    with negative curvature (κ < 0) get stretched (multiplier > 1);
    edges with positive curvature (κ > 0) get shortened. Over many
    steps this is the discrete analog of Perelman's smooth Ricci flow
    on networks: bottleneck edges between communities elongate while
    intra-community edges contract. Steps are sequential because each
    curvature computation uses the preceding step's edge lengths.

    Args:
      A: (B, n, n) symmetric weighted adjacency. Non-negative weights.
      n_steps: number of flow iterations.
      dt: step size. Each step multiplies an edge by (1 - dt * kappa).
      alpha: laziness parameter, as in `ollivier_ricci_curvature`.
      normalize: if True, rescale edge weights after each step so their
        Frobenius norm stays at its initial value. Without this, global
        scaling can drift, and normalizing isolates the structural effect
        of the flow.
      reg, n_sinkhorn_iters, sinkhorn_tol, sinkhorn_tile_size,
        sinkhorn_sync_every: forwarded to the curvature computation.
        The sync cadence controls how often marginal checks reach the host.

    Returns:
      (B, n, n) adjacency after `n_steps` of flow.

    Notes:
      Only existing edge pairs need transport solves. As lengths evolve,
      even the default budget can be exhausted; a convergence warning means
      that step used an infeasible transport approximation. Increase the
      budget or reconsider reg in the evolved graph's edge-length units.

    References:
      Sia, J., Jonckheere, E., Bogdan, P. (2019). Ollivier-Ricci
        curvature-based method to community detection in complex
        networks. Scientific Reports 9:9800, eq. 3.
      Ni, C.-C., Lin, Y.-Y., Luo, F., Gao, J. (2019). Community
        detection on networks with Ricci flow. Scientific Reports
        9:9984.
    """
    W = _validate_flow_inputs(
        A, n_steps, dt, alpha, reg, n_sinkhorn_iters, sinkhorn_tol,
        sinkhorn_tile_size, sinkhorn_sync_every,
    )
    if normalize:
        initial_norm = torch.linalg.matrix_norm(W, dim=(-2, -1), keepdim=True)

    for _ in range(n_steps):
        # Only the existing edges get updated, so only their transports are
        # solved. Non-edge entries of kappa come back as placeholders that
        # the edge mask below throws away.
        kappa = _curvature(
            W, alpha, reg, n_sinkhorn_iters, sinkhorn_tol,
            sinkhorn_tile_size, sinkhorn_sync_every, pair_mask=W > 0,
        )
        # w *= (1 - dt * kappa) on existing edges only. Non-edges are exactly
        # zero (weights are non-negative), so they stay zero and the flow never
        # creates an edge. An edge whose weight reaches zero stays gone.
        edge_mask = (W > 0).to(W.dtype)
        update = (1.0 - dt * kappa) * edge_mask
        W = W * update
        # kappa <= 1 always (W_1 is non-negative), so the multiplier
        # 1 - dt * kappa can only go negative when dt > 1. A negative length has
        # no meaning, so the clamp sets that edge to zero.
        W = W.clamp(min=0)
        # Curvature is symmetric in exact arithmetic; averaging with the
        # transpose stops float drift from making the graph directed.
        W = 0.5 * (W + W.mT)
        if normalize:
            current_norm = torch.linalg.matrix_norm(
                W, dim=(-2, -1), keepdim=True,
            )
            # If the step removed every edge, the norm is zero and the division
            # would give NaN. Dividing by one keeps the all-zero graph at zero.
            # Any positive norm, however small, is used as it is.
            current_norm = torch.where(
                current_norm == 0, torch.ones_like(current_norm), current_norm,
            )
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
    sinkhorn_sync_every: int = SINKHORN_SYNC_EVERY_DEFAULT,
) -> torch.Tensor:
    """Discrete Ricci flow with threshold-based edge surgery.

    Each step updates w <- (1 - dt*kappa) w, clamps negative weights to
    zero, and optionally restores the initial Frobenius norm. Every
    surgery_period steps, edges above surgery_threshold times the initial
    mean edge weight are removed. The reference mean excludes self-loops.
    Steps are sequential because curvature depends on the current graph.
    Only existing edge pairs need transport solves. Evolving edge lengths
    can exhaust the transport budget even at defaults; a convergence warning
    means that step used an infeasible approximation, as in discrete_ricci_flow.

    Args:
      A: (B, n, n) symmetric weighted adjacency.
      n_steps: total flow iterations.
      surgery_period: perform surgery every N steps. Catalog default.
      surgery_threshold: edge-removal threshold, as a multiplier of
        initial mean edge weight. Catalog default. Note that when
        `normalize=True`, the cutoff uses the **pre-normalization**
        initial mean as a fixed reference; the per-step normalization
        rescales the weights but the threshold does not move with it.
        Its effect therefore depends on `dt`, `normalize`, and the
        curvature magnitudes; choose the threshold for the input scale.
      Other args: as in `discrete_ricci_flow`.

    Returns:
      (B, n, n) adjacency after flow and surgery. Components that surgery
      disconnects are the candidate communities; nothing here checks that
      they are.

    References:
      Sia, J., Jonckheere, E., Bogdan, P. (2019). Ollivier-Ricci
        curvature-based method to community detection in complex networks.
        Scientific Reports 9:9800.
      Ni, C.-C., Lin, Y.-Y., Luo, F., Gao, J. (2019). Community detection
        on networks with Ricci flow. Scientific Reports 9:9984, §3.2.
    """
    W = _validate_flow_inputs(
        A, n_steps, dt, alpha, reg, n_sinkhorn_iters, sinkhorn_tol,
        sinkhorn_tile_size, sinkhorn_sync_every,
    )
    if not isinstance(surgery_period, int) or surgery_period <= 0:
        raise ValueError(
            f"surgery_period must be a positive integer, got {surgery_period}"
        )
    if not math.isfinite(surgery_threshold) or surgery_threshold <= 0:
        raise ValueError(
            f"surgery_threshold must be finite and > 0, got {surgery_threshold}"
        )

    # Initial mean edge weight is the fixed scale we threshold against. Both
    # (i, j) and (j, i) are counted, which gives the same mean on a symmetric
    # graph. The clamp keeps an edgeless graph from dividing by zero.
    edge_mask_init = (W > 0).to(W.dtype)
    n_edges = edge_mask_init.sum(dim=(-2, -1), keepdim=True).clamp(min=1.0)
    initial_mean = (W * edge_mask_init).sum(
        dim=(-2, -1), keepdim=True,
    ) / n_edges
    cutoff = surgery_threshold * initial_mean

    if normalize:
        initial_norm = torch.linalg.matrix_norm(
            W, dim=(-2, -1), keepdim=True,
        )

    for step in range(n_steps):
        # Same flow step as discrete_ricci_flow; see the comments there.
        kappa = _curvature(
            W, alpha, reg, n_sinkhorn_iters, sinkhorn_tol,
            sinkhorn_tile_size, sinkhorn_sync_every, pair_mask=W > 0,
        )
        edge_mask = (W > 0).to(W.dtype)
        update = (1.0 - dt * kappa) * edge_mask
        W = W * update
        W = W.clamp(min=0)
        W = 0.5 * (W + W.mT)
        if normalize:
            current_norm = torch.linalg.matrix_norm(
                W, dim=(-2, -1), keepdim=True,
            )
            current_norm = torch.where(
                current_norm == 0, torch.ones_like(current_norm), current_norm,
            )
            W = W * (initial_norm / current_norm)

        # Surgery: remove edges that have stretched past the cutoff. `step + 1`
        # counts flow steps from one, so surgery runs after step number
        # surgery_period, 2 * surgery_period, and so on. With surgery_period = 1
        # it runs after every step, the first one included.
        if (step + 1) % surgery_period == 0:
            keep = W <= cutoff
            W = W * keep.to(W.dtype)
            W = 0.5 * (W + W.mT)

    return W


# ============================================================
# Internal helpers
# ============================================================


def _curvature(
    A: torch.Tensor,
    alpha: float,
    reg: float,
    n_iter: int,
    tol: float,
    tile_size: int,
    sync_every: int,
    pair_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute Ollivier-Ricci curvature, solving transport only where asked.

    Takes the public arguments plus an optional (..., n, n) boolean pair_mask
    and returns the (..., n, n) curvature. Public curvature needs every pair,
    so it passes no mask. Flow only uses existing edges, so it passes W > 0
    and skips transports that cannot affect its update. A reachable pair
    outside the mask gets W1 = 0 and so kappa = 1, a placeholder the caller
    must mask out.
    """
    A = _validate_curvature_inputs(
        A, alpha, reg, n_iter, tol, tile_size, sync_every,
    )
    # An empty batch or a zero-node graph has nothing to solve.
    if A.numel() == 0:
        return A.clone()

    *batch, n, _ = A.shape

    # 1. Shortest-path distances d_G(i, j). Floyd-Warshall on the
    # weighted graph, batched. Treat A[i,j]=0 (off-diagonal) as no edge.
    d_G, reachable = _shortest_path_distances(A)  # (B, n, n)

    # 2. Lazy-random-walk distributions μ_x for each node x.
    #    μ_x(y) = (1−α) · (A[x,y] / Σ_z A[x,z])   for y in N(x)
    #    μ_x(x) = α, or 1 if x is isolated (then μ_x = δ_x)
    mu = _lazy_walk_distributions(A, alpha)  # (B, n, n), μ_x is row x

    # 3. Sinkhorn-based W_1 between μ_x and μ_y for each pair (x, y) that
    #    `reachable` (and `pair_mask`, if given) selects. The solver works on
    #    each node's walk support and reads the costs between supports from d_G
    #    (see _batched_sinkhorn_w1).
    W1 = _batched_sinkhorn_w1(
        mu, d_G, reg=reg, n_iter=n_iter, tol=tol, tile_size=tile_size,
        sync_every=sync_every,
        reachable=reachable if pair_mask is None else reachable & pair_mask,
    )  # (B, n, n)

    # 4. Curvature κ(x, y) = 1 - W_1 / d_G.
    # d_G is 0 on the diagonal and holds a 0 placeholder between components.
    # Dividing by 1 there keeps the arithmetic finite; both cases are
    # overwritten below, the diagonal with 1 by convention and cross-component
    # pairs with 0.
    safe_dG = torch.where(d_G > 0, d_G, torch.ones_like(d_G))
    kappa = 1.0 - W1 / safe_dG
    eye = torch.eye(n, device=A.device, dtype=A.dtype).expand(*batch, n, n)
    kappa = torch.where(eye > 0, torch.ones_like(kappa), kappa)
    return torch.where(reachable, kappa, torch.zeros_like(kappa))



def _validate_curvature_inputs(
    A: torch.Tensor,
    alpha: float,
    reg: float,
    n_iter: int,
    tol: float,
    tile_size: int,
    sync_every: int,
) -> torch.Tensor:
    """Check the curvature arguments and return A with its self-loops removed.

    Raises ValueError on a bad shape, dtype, negative or non-finite length,
    or out-of-range parameter. An asymmetric A only warns, since the caller
    may have a near-symmetric input; curvature here assumes undirected graphs.
    """
    if A.ndim < 2 or A.shape[-1] != A.shape[-2]:
        raise ValueError(f"A must be (..., n, n); got A.shape={tuple(A.shape)}")
    if not A.is_floating_point():
        raise ValueError("A must have a real floating dtype")
    A = drop_self_loops(A)
    if not torch.isfinite(A).all() or (A < 0).any():
        raise ValueError("A must have finite non-negative edge lengths")
    if not 0 <= alpha <= 1:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    for name, value in (("reg", reg), ("tol", tol)):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and > 0, got {value}")
    for name, count in (
        ("n_iter", n_iter), ("tile_size", tile_size), ("sync_every", sync_every),
    ):
        if not isinstance(count, int) or count <= 0:
            raise ValueError(f"{name} must be a positive integer, got {count}")
    if not torch.allclose(A, A.mT, atol=1e-9, rtol=0):
        warnings.warn(
            "ollivier_ricci_curvature received an asymmetric adjacency; "
            "curvature is only defined here for undirected graphs. "
            "Symmetrize A first for meaningful results.",
            stacklevel=2,
        )
    return A


def _validate_flow_inputs(
    A: torch.Tensor,
    n_steps: int,
    dt: float,
    alpha: float,
    reg: float,
    n_iter: int,
    tol: float,
    tile_size: int,
    sync_every: int,
) -> torch.Tensor:
    """Check the flow arguments and return A with its self-loops removed.

    Runs the curvature checks too, so a zero-step call still rejects bad
    input instead of passing it straight back.
    """
    A = _validate_curvature_inputs(A, alpha, reg, n_iter, tol, tile_size, sync_every)
    if not isinstance(n_steps, int) or n_steps < 0:
        raise ValueError(f"n_steps must be a non-negative integer, got {n_steps}")
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError(f"dt must be finite and > 0, got {dt}")
    return A


def _shortest_path_distances(A: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return Floyd-Warshall distances and a reachability mask.

    Input: A, (..., n, n) non-negative adjacency; its diagonal is ignored.
    Output: (distances, reachable). Unreachable distances are set to zero, so
    the mask is the only record of which pairs are unreachable. The vertex
    recurrence is sequential, with batches and endpoint pairs handled by
    tensor operations. Assumes an undirected graph, so reachability is
    symmetric.
    """
    *batch, n, _ = A.shape
    # Starting distances: the edge length A[i,j] where there is an edge, inf
    # where there is none, and 0 on the diagonal.
    eye = torch.eye(n, device=A.device, dtype=A.dtype).expand(*batch, n, n)
    off_diag = (eye == 0)
    no_edge = off_diag & (A == 0)
    # Infinity distinguishes missing paths while the recurrence runs.
    inf_sentinel = torch.full_like(A, float("inf"))
    D = torch.where(no_edge, inf_sentinel, A.clone())
    D = torch.where(eye > 0, torch.zeros_like(D), D)

    # Floyd-Warshall: for k in 0..n-1: D[i,j] = min(D[i,j], D[i,k] + D[k,j]).
    # Batched: vectorize over (i, j); k is the loop axis.
    for k in range(n):
        # via_k[..., i, j] = D[..., i, k] + D[..., k, j]
        via_k = D[..., :, k : k + 1] + D[..., k : k + 1, :]
        D = torch.minimum(D, via_k)

    # Unreachable entries become zero, so no inf is left for the transport
    # arithmetic to multiply by a zero plan entry. Only supported entries of
    # reachable pairs carry real costs; padded slots are masked in
    # _sinkhorn_cost.
    finite_mask = torch.isfinite(D)
    return torch.where(finite_mask, D, torch.zeros_like(D)), finite_mask


def _lazy_walk_distributions(A: torch.Tensor, alpha: float) -> torch.Tensor:
    """Lazy-random-walk distributions for every node.

    Takes a (..., n, n) non-negative adjacency with its diagonal already
    zeroed. Row x of the output is μ_x, where
      μ_x(x) = α   (or 1 if x is isolated, so μ_x = δ_x)
      μ_x(y) = (1−α) · A[x,y] / Σ_z A[x,z]   for y ≠ x with A[x,y]>0.
    The same weights are edge lengths and walk weights, so a longer edge also
    gets more walk probability.

    Returns:
      (..., n, n) tensor; row x is μ_x as a probability vector.
    """
    n = A.shape[-1]
    row_sums = A.sum(dim=-1, keepdim=True)  # (..., n, 1)
    is_isolated = row_sums.squeeze(dim=-1) <= 0  # (..., n)

    # Isolated rows have a zero sum. Dividing them by 1 leaves a zero row,
    # which the isolated-node branch below replaces anyway.
    safe_row_sums = torch.where(
        row_sums > 0, row_sums, torch.ones_like(row_sums),
    )
    walk = A / safe_row_sums  # (..., n, n), rows are normalized

    eye = torch.eye(n, device=A.device, dtype=A.dtype).expand_as(A)
    mu = (1.0 - alpha) * walk + alpha * eye

    # An isolated node has nowhere to walk, so all its mass stays on itself.
    # Without this its row would only carry alpha and not sum to one.
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
    sync_every: int,
    reachable: torch.Tensor,
) -> torch.Tensor:
    """Gather support costs and solve unordered pairs in memory-bounded tiles.

    Takes the (..., n, n) walk distributions (row x is μ_x), the (..., n, n)
    shortest-path cost, and the boolean pairs to solve. Returns the
    (..., n, n) transport costs. Assumes the measures are normalized rows on
    an undirected graph, so W1 is symmetric and each unordered pair is solved
    once. The diagonal, unselected pairs, and cross-component pairs hold zero
    placeholders. Padding carries exactly zero mass, including in log space.
    Warns once if any solved pair missed the marginal tolerance.
    """
    *batch, n, _ = mu.shape
    support = (mu > 0) | torch.eye(n, device=mu.device, dtype=torch.bool)
    # Pad every node's support to the widest neighborhood. That width comes
    # from one shape sync, and it bounds storage by the max degree rather than
    # by n. Self stays in the support even when alpha=0, with zero probability.
    # topk sorts by value, so the 1s (support) come first and the 0s (padding,
    # where mu is zero) come last. A tile can then slice the first tile_width
    # columns.
    node_width = support.sum(dim=-1)
    width = int(node_width.amax().item())
    indices = support.to(torch.int).topk(width, dim=-1).indices
    mu = mu.gather(-1, indices)
    sink_w1 = cost.new_zeros(*batch, n, n)
    # Strict upper triangle: each unordered pair once, diagonal excluded.
    src_indices, tgt_indices = torch.triu_indices(n, n, offset=1, device=mu.device)
    if src_indices.numel() == 0:
        return sink_w1
    # Keep a pair if any batch member needs it; members that do not are
    # masked again inside the solve by checked_pairs.
    selected = reachable[..., src_indices, tgt_indices].reshape(
        -1, src_indices.numel(),
    ).any(dim=0)
    src_indices, tgt_indices = src_indices[selected], tgt_indices[selected]
    # Sort pairs by their widest support, so each tile pads only to its own
    # widest pair rather than to the graph's largest degree. A pair's width is
    # the max over the batch, so every batch member uses the same pair order.
    pair_width = torch.maximum(
        node_width[..., src_indices], node_width[..., tgt_indices],
    ).reshape(math.prod(batch), -1).amax(dim=0)
    order = pair_width.argsort(descending=True)
    src_indices, tgt_indices = src_indices[order], tgt_indices[order]
    pair_width = pair_width[order]
    worst_error = 0.0

    for start in range(0, src_indices.numel(), tile_size):
        src_idx = src_indices[start:start + tile_size]
        tgt_idx = tgt_indices[start:start + tile_size]
        # Pairs are sorted widest first, so the first pair's width covers the
        # whole tile.
        tile_width = int(pair_width[start].item())
        # Flat index i * n + j into the (n, n) cost matrix, for every pair of
        # support entries: i from the source node's support, j from the target
        # node's support.
        cost_indices = (
            indices[..., src_idx, :tile_width, None] * n
            + indices[..., tgt_idx, None, :tile_width]
        )
        support_cost = cost.flatten(-2).gather(
            -1, cost_indices.flatten(len(batch)),
        ).reshape(*batch, src_idx.numel(), tile_width, tile_width)
        value, error = _sinkhorn_cost(
            mu[..., src_idx, :tile_width], mu[..., tgt_idx, :tile_width], support_cost,
            reachable[..., src_idx, tgt_idx], reg, n_iter, tol, sync_every,
        )
        sink_w1[..., src_idx, tgt_idx] = value
        sink_w1[..., tgt_idx, src_idx] = value
        worst_error = max(worst_error, error)

    if worst_error > tol:
        warnings.warn(
            f"Sinkhorn did not converge after {n_iter} iterations: achieved "
            f"maximum marginal error {worst_error:.6g} exceeds tol={tol:.6g}. "
            "Increase n_iter or reg, or choose an appropriate tolerance.",
            stacklevel=2,
        )
    return sink_w1


def _sinkhorn_cost(
    src: torch.Tensor,
    tgt: torch.Tensor,
    cost: torch.Tensor,
    checked_pairs: torch.Tensor,
    reg: float,
    n_iter: int,
    tol: float,
    sync_every: int,
) -> tuple[torch.Tensor, float]:
    """Solve one tile of entropic transports and return their costs.

    Takes (..., P, s) source and target masses on padded supports, the
    (..., P, s, s) support costs, and a (..., P) mask of pairs to solve.
    Returns the (..., P) transport costs (zero for unchecked pairs) and the
    largest marginal error among pairs that did not meet tol, or 0.0 if all
    of them did.

    Schmitzer (2019), Algorithms 2 and 3: alternate matrix scaling with
    log-domain absorption, and carry dimensional potentials between epsilon
    stages. Each rebuild starts with a log-domain sweep using the original
    costs, so an underflowed kernel entry can become positive again.
    """
    output_shape = src.shape[:-1]
    width = src.shape[-1]
    # Flatten the batch into one list of problems. `active` holds the flat
    # indices of the pairs to solve; the rest keep a zero cost.
    values = src.new_zeros(checked_pairs.numel())
    active = torch.arange(values.numel(), device=src.device)[checked_pairs.flatten()]
    src = src.reshape(-1, width)[active]
    tgt = tgt.reshape(-1, width)[active]
    cost = cost.reshape(-1, width, width)[active]
    # Nothing to solve in this tile.
    if active.numel() == 0:
        return values.reshape(output_shape), 0.0
    # Support is where the mass is positive. Padding and zero-probability
    # entries fall outside it.
    src_support, tgt_support = src > 0, tgt > 0
    # Log masses, with -inf on zero-mass entries. The inner where swaps those
    # zeros for 1 before the log, so log(0) is never evaluated.
    log_src = torch.where(src_support, torch.where(src_support, src, 1.0).log(),
                          -torch.inf)
    log_tgt = torch.where(tgt_support, torch.where(tgt_support, tgt, 1.0).log(),
                          -torch.inf)
    # Padded indices can name unrelated vertices. Their costs must not affect
    # the schedule, or batching graphs of different degrees changes the solve.
    # Each pair's warmup starts at its own largest support cost, floored at reg.
    active_cost = torch.where(
        src_support.unsqueeze(-1) & tgt_support.unsqueeze(-2), cost, 0.0,
    )
    epsilon = active_cost.amax(dim=(-2, -1)).clamp_min(reg).unsqueeze(-1)
    # A log sweep recomputes the source potential from the target potential,
    # so only the latter needs to be carried between reconstructions.
    g = torch.zeros_like(tgt)
    kernel = torch.zeros_like(cost)
    rebuild = torch.ones_like(active, dtype=torch.bool)
    # Summed block norms bound the cumulative log scaling since the last
    # reconstruction. This triggers absorption before either factor can
    # drift by more than the threshold, even if later updates cancel.
    absorbed_change = torch.zeros_like(active, dtype=src.dtype)
    # Smallest positive normal float. A row or column sum at or below it counts
    # as underflow. Dividing by it would blow up, so the division is clamped and
    # the pair is marked unstable.
    tiny = torch.finfo(src.dtype).tiny
    # Tolerance for moving a pair to the next epsilon. It is the caller's tol,
    # but never tighter than SINKHORN_TOL_DEFAULT.
    stage_tol = max(tol, SINKHORN_TOL_DEFAULT)
    marginal_error = float("inf")

    for it in range(0, n_iter, sync_every):
        steps = min(sync_every, n_iter - it)
        # On the last block every pair moves to the requested reg, so the plan
        # returned is at reg even if warmup used up the budget.
        if it + steps == n_iter:
            rebuild = rebuild | (epsilon.squeeze(-1) != reg)
            epsilon = torch.full_like(epsilon, reg)
        # Rebuild the kernel from the carried potential g at the
        # current epsilon. One log-domain row sweep then column sweep, and the
        # kernel becomes that sweep's plan, so its columns match tgt exactly.
        rebuild_idx = torch.nonzero(rebuild).flatten()
        if rebuild_idx.numel() > 0:
            eps = epsilon[rebuild_idx]
            log_kernel = -cost[rebuild_idx] / eps.unsqueeze(-1)
            log_v = torch.where(tgt_support[rebuild_idx],
                                g[rebuild_idx] / eps, -torch.inf)
            log_u = log_src[rebuild_idx] - torch.logsumexp(
                log_kernel + log_v.unsqueeze(-2), dim=-1,
            )
            log_v = log_tgt[rebuild_idx] - torch.logsumexp(
                log_kernel + log_u.unsqueeze(-1), dim=-2,
            )
            g = g.index_copy(
                0, rebuild_idx,
                torch.where(tgt_support[rebuild_idx], log_v, 0.0) * eps,
            )
            rebuilt = (
                log_u.unsqueeze(-1) + log_kernel + log_v.unsqueeze(-2)
            ).exp()
            kernel = kernel.index_copy(0, rebuild_idx, rebuilt)
        # A rebuilt kernel carries its own scaling, so the count starts over.
        absorbed_change = torch.where(rebuild, 0.0, absorbed_change)

        # Between rebuilds the kernel is only rescaled. Each block multiplies it
        # by new row and column factors u and v, and the scaled plan becomes the
        # next kernel. Scaling cannot bring back an entry that underflowed to
        # zero, so a row or column sum at or below tiny marks the pair unstable,
        # and the next log-domain rebuild recomputes the kernel from the costs.
        # Divisions by such sums are clamped to stay finite.
        u, v = torch.ones_like(src), torch.ones_like(tgt)
        stable = torch.ones_like(active, dtype=torch.bool)
        for step in range(steps):
            row_sum = (kernel @ v.unsqueeze(-1)).squeeze(-1)
            stable = stable & ((row_sum > tiny) | ~src_support).all(-1)
            u = src / row_sum.clamp_min(tiny)
            # A rebuilt kernel already had this block's first sweep done in
            # log space, so its first u and v stay at one.
            if step == 0:
                u = torch.where(rebuild.unsqueeze(-1), 1.0, u)
            col_sum = (kernel.mT @ u.unsqueeze(-1)).squeeze(-1)
            stable = stable & ((col_sum > tiny) | ~tgt_support).all(-1)
            v = tgt / col_sum.clamp_min(tiny)
            if step == 0:
                v = torch.where(rebuild.unsqueeze(-1), 1.0, v)
        # Absorb the block's scaling. v goes into g in potential units; u is
        # not stored, since the next rebuild recomputes it from g. A pair
        # whose scaling went unstable keeps its old kernel and g, and is
        # rebuilt in log space at the next block.
        stable = stable & ((u > 0) | ~src_support).all(-1)
        stable = stable & ((v > 0) | ~tgt_support).all(-1)
        # Zero-mass entries have u or v equal to 0. Their log is taken as 0, so
        # they add no scaling to the absorbed total.
        log_u = torch.where(u > 0, u, 1.0).log()
        log_v = torch.where(v > 0, v, 1.0).log()
        g = g + epsilon * torch.where(stable.unsqueeze(-1), log_v, 0.0)
        # An unstable pair falls back to its kernel, not to a plan built from
        # clamped divisions.
        plan = kernel * u.unsqueeze(-1) * v.unsqueeze(-2)
        plan = torch.where(stable.unsqueeze(-1).unsqueeze(-1), plan, kernel)
        absorbed_change = absorbed_change + torch.maximum(
            log_u.abs().amax(-1), log_v.abs().amax(-1),
        )

        # Check each pair's plan against both marginals. The cost written below
        # comes from that same plan. A pair finishes only at target reg, and
        # finished pairs are dropped at the end of the block, so a slower pair
        # in the tile cannot change their answer.
        row_error = (plan.sum(-1) - src).abs().amax(-1)
        col_error = (plan.sum(-2) - tgt).abs().amax(-1)
        error = torch.maximum(row_error, col_error)
        error = torch.where(torch.isfinite(error), error, torch.inf)
        values = values.index_copy(0, active, (plan * cost).sum((-2, -1)))
        # A pair is finished only at target epsilon. One still in warmup
        # counts as an infinite error, so the loop cannot stop early on it.
        finished = (error <= tol) & (epsilon.squeeze(-1) == reg)
        done_error = torch.where(finished, 0.0, error)
        done_error = torch.where(
            (epsilon.squeeze(-1) > reg) & ~finished, torch.inf, done_error,
        )
        marginal_error = done_error.amax().item()
        if marginal_error <= tol:
            break
        # Pairs that met the stage tolerance move to the next, colder
        # epsilon. A changed epsilon, an unstable block, or too much
        # accumulated scaling all force a log-domain rebuild next block.
        ready = (error <= stage_tol).unsqueeze(-1)
        next_epsilon = torch.where(
            ready, (epsilon * SINKHORN_EPSILON_DECAY).clamp_min(reg), epsilon,
        )
        rebuild = (
            (next_epsilon.squeeze(-1) != epsilon.squeeze(-1)) | ~stable
            | (absorbed_change > math.log(SINKHORN_ABSORPTION_THRESHOLD))
        )
        epsilon = next_epsilon
        kernel = plan
        # Drop finished pairs at the same sync boundary as the convergence
        # check, so the hard pairs keep iterating without re-solving the
        # accepted ones.
        keep = ~finished
        active = active[keep]
        src, tgt, cost = src[keep], tgt[keep], cost[keep]
        src_support, tgt_support = src_support[keep], tgt_support[keep]
        log_src, log_tgt = log_src[keep], log_tgt[keep]
        g, epsilon = g[keep], epsilon[keep]
        kernel, rebuild = kernel[keep], rebuild[keep]
        absorbed_change = absorbed_change[keep]
    return values.reshape(output_shape), marginal_error
