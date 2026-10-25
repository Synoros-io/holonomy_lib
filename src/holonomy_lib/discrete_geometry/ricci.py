# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Synoros

"""Batched Ollivier-Ricci curvature and discrete Ricci flow on graphs.

The metric is shortest-path distance with adjacency weights interpreted as
edge lengths. Those same weights determine the neighbor probabilities of
mu_x = (1-alpha) A[x, :] / sum(A[x, :]) + alpha delta_x. Curvature is
kappa(x, y) = 1 - W_1(mu_x, mu_y) / d(x, y).

Log-domain Sinkhorn approximates the transport problem with entropic
regularization. A small regularization does not guarantee convergence:
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

# Existing regularization default in edge-length units. Its accuracy depends on
# the graph's length scale; Cuturi (2013) defines the regularized problem,
# not a universal error bound or an iteration guarantee for this value.
SINKHORN_REG_DEFAULT: float = 0.01

# Existing runtime budget, not a convergence guarantee. Hard transport
# problems may need a larger value or a larger regularization.
SINKHORN_N_ITER_DEFAULT: int = 100

# Absolute probability error for both transport marginals. This retains
# the library's numerical-floor default, but float32 roundoff may exceed
# it; callers can choose a tolerance appropriate to their input dtype.
SINKHORN_TOL_DEFAULT: float = 1e-9

# Sync cadence for Sinkhorn convergence checks. The `.item()` call
# on the marginal-error tensor forces a host sync on GPU; doing it every iter
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
# at 128 MiB for n=256 in float64, per batch member and temporary tensor.
# Bump higher when memory headroom is generous and lower when n is
# big. Cataloged as `sinkhorn_tile_default`.
SINKHORN_TILE_DEFAULT: int = 256

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
      n_iter: positive maximum Sinkhorn iteration count.
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
      can then be infeasible. Exact zeros in the walk support stay zero.
      All-pairs Sinkhorn costs O(B*n**4*n_iter), with
      O(B*tile_size*n**2) temporary storage. A 256-by-256-by-256
      float64 tensor alone occupies 128 MiB. Floyd-Warshall costs
      O(B*n**3). Both algorithms have sequential recurrence steps.

    References:
      Ollivier (2009), Definition 1.
      Cuturi (2013), Algorithm 1, for entropic Sinkhorn transport.
    """
    A = _validate_curvature_inputs(
        A, alpha, reg, n_iter, tol, tile_size, sync_every,
    )
    if A.numel() == 0:
        return A.clone()

    *batch, n, _ = A.shape

    # 1. Shortest-path distances d_G(i, j). Floyd-Warshall on the
    # weighted graph, batched. Treat A[i,j]=0 (off-diagonal) as no edge.
    d_G, reachable = _shortest_path_distances(A)  # (B, n, n)

    # 2. Lazy-random-walk distributions μ_x for each node x.
    #    μ_x(y) = (1−α) · (A[x,y] / Σ_z A[x,z])   for y in N(x)
    #    μ_x(x) = α (+ 0 if x is isolated, in which case μ_x = δ_x)
    mu = _lazy_walk_distributions(A, alpha)  # (B, n, n) — μ_x is row x

    # 3. Sinkhorn-based W_1 between μ_x and μ_y for every pair (x, y).
    #    All pairs of rows of `mu` — the cost matrix is d_G.
    W1 = _batched_sinkhorn_w1(
        mu, d_G, reg=reg, n_iter=n_iter, tol=tol, tile_size=tile_size,
        sync_every=sync_every, reachable=reachable,
    )  # (B, n, n)

    # 4. Curvature κ(x, y) = 1 - W_1 / d_G.
    # Diagonal: d_G is 0 there; set κ to 1 by convention (vacuous).
    safe_dG = torch.where(d_G > 0, d_G, torch.ones_like(d_G))
    kappa = 1.0 - W1 / safe_dG
    # Diagonal entries: 1 by convention.
    eye = torch.eye(n, device=A.device, dtype=A.dtype).expand(*batch, n, n)
    kappa = torch.where(eye > 0, torch.ones_like(kappa), kappa)
    return torch.where(reachable, kappa, torch.zeros_like(kappa))


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
      dt: step size. Smaller → more numerically stable, slower convergence.
      alpha: laziness parameter forwarded to `ollivier_ricci_curvature`.
      normalize: if True, rescale edge weights after each step so the
        Frobenius norm of A stays constant. Prevents global scaling
        from running away, isolating the *structural* effect of the flow.
      reg, n_sinkhorn_iters, sinkhorn_tol, sinkhorn_tile_size,
        sinkhorn_sync_every: forwarded to the curvature computation.
        The sync cadence controls how often marginal checks reach the host.

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
    W = _validate_flow_inputs(
        A, n_steps, dt, alpha, reg, n_sinkhorn_iters, sinkhorn_tol,
        sinkhorn_tile_size, sinkhorn_sync_every,
    )
    if normalize:
        initial_norm = torch.linalg.matrix_norm(W, dim=(-2, -1), keepdim=True)

    for _ in range(n_steps):
        # Compute curvature on current edge weights
        kappa = ollivier_ricci_curvature(
            W, alpha=alpha, reg=reg, n_iter=n_sinkhorn_iters,
            tol=sinkhorn_tol, tile_size=sinkhorn_tile_size,
            sync_every=sinkhorn_sync_every,
        )
        # Update: w *= (1 - dt * kappa). Mask to existing edges only;
        # non-edges (W == 0) stay zero — surgery is separate.
        edge_mask = (W > 0).to(W.dtype)
        update = (1.0 - dt * kappa) * edge_mask
        W = W * update + W * (1.0 - edge_mask)  # leave non-edges alone
        # Clamp negative weights to zero (edge would have flipped sign)
        W = W.clamp(min=0)
        # Maintain symmetry (curvature is symmetric, but float drift)
        W = 0.5 * (W + W.mT)
        if normalize:
            current_norm = torch.linalg.matrix_norm(
                W, dim=(-2, -1), keepdim=True,
            )
            # A positive norm keeps its own scale, however small. If the
            # step removes every edge, zero weights must remain zero.
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
      (B, n, n) adjacency after flow + surgery. Disconnected components
      indicate detected communities.

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

    # Initial mean edge weight — the scale against which we threshold.
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
        # One Ricci-flow step
        kappa = ollivier_ricci_curvature(
            W, alpha=alpha, reg=reg, n_iter=n_sinkhorn_iters,
            tol=sinkhorn_tol, tile_size=sinkhorn_tile_size,
            sync_every=sinkhorn_sync_every,
        )
        edge_mask = (W > 0).to(W.dtype)
        update = (1.0 - dt * kappa) * edge_mask
        W = W * update + W * (1.0 - edge_mask)
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


def _validate_curvature_inputs(
    A: torch.Tensor,
    alpha: float,
    reg: float,
    n_iter: int,
    tol: float,
    tile_size: int,
    sync_every: int,
) -> torch.Tensor:
    """Validate shared inputs and return adjacency with its loops removed."""
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
    """Share flow validation, including zero-step calls."""
    A = _validate_curvature_inputs(A, alpha, reg, n_iter, tol, tile_size, sync_every)
    if not isinstance(n_steps, int) or n_steps < 0:
        raise ValueError(f"n_steps must be a non-negative integer, got {n_steps}")
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError(f"dt must be finite and > 0, got {dt}")
    return A


def _shortest_path_distances(A: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return Floyd-Warshall distances and the original reachability mask.

    The vertex recurrence is sequential, with all batches and endpoint
    pairs handled by tensor operations. Unreachable distances receive a
    zero placeholder for transport arithmetic; callers retain the mask.
    Exact-zero probability support excludes those costs in all transport
    problems within an undirected component.
    """
    *batch, n, _ = A.shape
    # Build initial distance matrix: A[i,j] where i≠j and A[i,j]>0, else inf.
    eye = torch.eye(n, device=A.device, dtype=A.dtype).expand(*batch, n, n)
    off_diag = (eye == 0)
    no_edge = off_diag & (A == 0)
    # Infinity distinguishes missing paths while the recurrence runs.
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

    # Cross-component outputs are masked by the public function. For
    # every within-component solve, exact zero marginal support already
    # excludes these costs. A zero placeholder avoids sentinel/reg
    # overflow in low-precision dtypes without changing those solves.
    finite_mask = torch.isfinite(D)
    return torch.where(finite_mask, D, torch.zeros_like(D)), finite_mask


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
    sync_every: int,
    reachable: torch.Tensor,
) -> torch.Tensor:
    """Return tiled entropic transport costs, checking both marginals.

    Each pair is an independent transport problem. Sinkhorn iterations
    are sequential scaling recurrences, but each update covers every
    batch member and pair in the current tile. Convergence checks ignore
    unreachable pairs and diagonal pairs, whose output is set directly
    by the public curvature convention.
    """
    *batch, n, _ = mu.shape
    n_pairs = n * n
    cost_b = cost[..., None, :, :]
    log_kernel = -cost_b / reg
    support = mu > 0
    # The safe argument also keeps log's backward finite at absent
    # support. The -inf mask gives those entries exactly zero mass.
    safe_mu = torch.where(support, mu, torch.ones_like(mu))
    log_mu = torch.where(support, safe_mu.log(), -torch.inf)
    sink_w1 = torch.empty(*batch, n_pairs, dtype=mu.dtype, device=mu.device)
    pair_idx = torch.arange(n_pairs, device=mu.device)
    src_idx_all = pair_idx // n
    tgt_idx_all = pair_idx % n
    worst_error = 0.0

    for start in range(0, n_pairs, tile_size):
        end = min(start + tile_size, n_pairs)
        src_idx = src_idx_all[start:end]
        tgt_idx = tgt_idx_all[start:end]
        src = mu[..., src_idx, :]
        tgt = mu[..., tgt_idx, :]
        log_src = log_mu[..., src_idx, :]
        log_tgt = log_mu[..., tgt_idx, :]
        checked_pairs = reachable[..., src_idx, tgt_idx] & (src_idx != tgt_idx)
        log_u = torch.zeros_like(log_src)
        log_v = torch.zeros_like(log_tgt)
        marginal_error = float("inf")
        for it in range(n_iter):
            log_u = log_src - torch.logsumexp(
                log_kernel + log_v.unsqueeze(dim=-2), dim=-1,
            )
            log_v = log_tgt - torch.logsumexp(
                log_kernel + log_u.unsqueeze(dim=-1), dim=-2,
            )
            if (it + 1) % sync_every == 0 or it + 1 == n_iter:
                # A small dual update alone can conceal an infeasible
                # plan. Check its row and column sums against the two
                # probability distributions before declaring convergence.
                row_mass = torch.exp(log_u + torch.logsumexp(
                    log_kernel + log_v.unsqueeze(dim=-2), dim=-1,
                ))
                col_mass = torch.exp(log_v + torch.logsumexp(
                    log_kernel + log_u.unsqueeze(dim=-1), dim=-2,
                ))
                row_error = (row_mass - src).abs().amax(dim=-1)
                col_error = (col_mass - tgt).abs().amax(dim=-1)
                error = torch.maximum(row_error, col_error)
                error = torch.where(checked_pairs, error, torch.zeros_like(error))
                # Nonfinite arithmetic must never be reported as convergence.
                error = torch.where(torch.isfinite(error), error, torch.inf)
                marginal_error = error.amax().item()
                if marginal_error <= tol:
                    break
        worst_error = max(worst_error, marginal_error)
        log_pi = log_u.unsqueeze(dim=-1) + log_kernel + log_v.unsqueeze(dim=-2)
        sink_w1[..., start:end] = (log_pi.exp() * cost_b).sum(dim=(-2, -1))

    if worst_error > tol:
        warnings.warn(
            f"Sinkhorn did not converge after {n_iter} iterations: achieved "
            f"maximum marginal error {worst_error:.6g} exceeds tol={tol:.6g}. "
            "Increase n_iter or reg, or choose an appropriate tolerance.",
            stacklevel=2,
        )
    return sink_w1.reshape(*batch, n, n)
