# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Synoros

"""Manifold-valued Laplacian eigenmaps via Riemannian SGD.

Status: experimental, in progress. The objective is attraction-only:
it pulls neighbours together and admits collapsed connected components as
minimizers. No constraint keeps nodes apart, and the iteration can diverge.
A correct version needs a non-collapse constraint (the Belkin-Niyogi
normalization, or negative sampling as in Nickel-Kiela 2017).

The implemented energy is Σ_ij A_ij d(Y_i, Y_j)². Updates use the
per-node average logarithm to neighbors, followed by projection and
retraction. Degree normalization rescales the update direction, so its
scale does not grow directly with a node's degree. There is no guarantee
that a single learning rate is suitable for all input geometries.

Belkin and Niyogi's spectral method additionally constrains the embedding
to exclude constant solutions. That constraint is absent here.

References:
  Belkin, M., Niyogi, P. (2003). Laplacian eigenmaps for
    dimensionality reduction and data representation. Neural Comp.
    15(6):1373–1396.
  Nickel, M., Kiela, D. (2017). Poincaré Embeddings for Learning
    Hierarchical Representations. NIPS.
  Sarkar, R. (2011). Low distortion Delaunay embedding of trees in
    hyperbolic plane. International Symposium on Graph Drawing.
  Liu, Q., Nickel, M., Kiela, D. (2019). Hyperbolic graph neural
    networks. NeurIPS.
"""

from __future__ import annotations

import warnings
from typing import Any

import torch

from holonomy_lib._graph_utils import drop_self_loops

# Preserve the existing finite work budget for this experimental optimizer.
EIGENMAPS_MAX_STEPS_DEFAULT = 200
# Existing exploratory step size, with no claim of universal convergence.
EIGENMAPS_LR_DEFAULT = 0.05


def hyperbolic_laplacian_eigenmaps(
    adjacency: torch.Tensor,
    manifold: Any,
    max_steps: int = EIGENMAPS_MAX_STEPS_DEFAULT,
    lr: float = EIGENMAPS_LR_DEFAULT,
    generator: torch.Generator | None = None,
    init: torch.Tensor | None = None,
) -> torch.Tensor:
    """Embed the nodes of `adjacency` on `manifold` by minimizing

        Σ_{i,j} A_{ij} · d_M(Y_i, Y_j)²

    via Riemannian SGD.

    Status: experimental. This attraction-only objective collapses each
    connected component; it does not compute constrained spectral eigenmaps.

    Args:
      adjacency: `(B, N, N)` symmetric non-negative graph adjacency
        per batch. Self-loops are dropped before computing degrees
        or updates. Asymmetric inputs produce a warning.
      manifold: a manifold object exposing `random_point`, `log`,
        `projection`, `retraction`, `norm` (see e.g.
        `LorentzManifold`). The embedding dimension is taken from
        `manifold.ambient_dim`, falling back to `manifold.n + 1`
        for manifolds without that property.
      max_steps: number of sequential RSGD steps. Default 200 is a
        work budget, not a convergence guarantee.
      lr: Riemannian SGD step size. Default 0.05 is an exploratory
        setting; tune for the input geometry.
      generator: optional `torch.Generator` for reproducible init, compatible
        with the manifold's configured sampling device.
      init: optional `(B, N, ambient_dim)` initial embedding. If
        provided, must lie on the manifold. Default: random points
        via `manifold.random_point`.

    Returns:
      `(B, N, ambient_dim)` embedding on `manifold`. For
      `LorentzManifold(n)`, `ambient_dim = n + 1`.

    Example:
      >>> from holonomy_lib.manifolds import LorentzManifold
      >>> mfd = LorentzManifold(n=2)
      >>> A = torch.eye(5).unsqueeze(0)  # trivial graph, B=1, N=5
      >>> Y = hyperbolic_laplacian_eigenmaps(A, mfd, max_steps=10)
      >>> Y.shape
      torch.Size([1, 5, 3])

    Notes:
      Per-step cost is `O(B · N² · D)` from the pairwise `log` call
      (`D = ambient_dim`). Acceptable for `N ≤ a few hundred`; larger
      graphs should chunk along the `N²` dimension or use a sparse
      adjacency-based loss (not implemented here, Stage 2 scope is
      the dense case).

      The result is **not** rotation-canonicalized (the optimal
      embedding is determined only up to isometries of the manifold);
      caller is responsible for any canonical alignment.

    References:
      Belkin–Niyogi (2003); Nickel–Kiela (2017); Liu et al. (2019).
    """
    if adjacency.ndim != 3 or adjacency.shape[-1] != adjacency.shape[-2]:
        raise ValueError(
            f"adjacency must have shape (B, N, N), got {tuple(adjacency.shape)}"
        )
    if max_steps < 0:
        raise ValueError(f"max_steps must be >= 0, got {max_steps}")
    if lr <= 0:
        raise ValueError(f"lr must be > 0, got {lr}")

    adjacency = drop_self_loops(adjacency)
    if not torch.allclose(adjacency, adjacency.mT):
        warnings.warn("adjacency is not symmetric", UserWarning, stacklevel=2)
    B, N, _ = adjacency.shape
    # Ambient dim is manifold-specific (n+1 for Lorentz, n for
    # κ-stereographic). Use the `ambient_dim` property when available
    # so the primitive is model-agnostic.
    D = getattr(manifold, "ambient_dim", None)
    if D is None:
        D = manifold.n + 1

    if init is None:
        # `random_point` returns (BN, D); reshape into (B, N, D).
        embedding = manifold.random_point(
            batch_size=B * N, generator=generator,
        ).reshape(B, N, D)
    else:
        if init.shape != (B, N, D):
            raise ValueError(
                f"init must have shape ({B}, {N}, {D}), "
                f"got {tuple(init.shape)}"
            )
        embedding = init.clone()

    embedding = embedding.to(device=adjacency.device, dtype=adjacency.dtype)

    # Average each node's logarithms over its neighbors. Dropping loops
    # first keeps them from diluting the update through the degree.
    # Isolated rows have a zero numerator; the floor only makes 0/0 safe.
    degree = adjacency.sum(dim=-1, keepdim=True)              # (B, N, 1)
    degree_safe = degree.clamp(min=torch.finfo(adjacency.dtype).tiny)

    for _ in range(max_steps):
        # Pairwise log_Y_i(Y_j), broadcast `embedding` along the i and
        # j axes. Shapes:
        #   emb_i (B, N, 1, D) → expand → (B, N, N, D)
        #   emb_j (B, 1, N, D) → expand → (B, N, N, D)
        # Then flatten (B, N, N) to a single batch axis so `manifold.log`
        # can be called on a leading-batch tensor.
        emb_i = embedding.unsqueeze(2).expand(B, N, N, D)
        emb_j = embedding.unsqueeze(1).expand(B, N, N, D)
        log_ij = manifold.log(
            emb_i.reshape(B * N * N, D),
            emb_j.reshape(B * N * N, D),
        ).reshape(B, N, N, D)

        # Gradient on each node i: −2 Σ_j A_{ij} · log_{Y_i}(Y_j), normalized
        # by deg(i) so the per-step magnitude is the *average* log-direction
        # toward neighbors (degree-invariant). Log maps are tangent vectors,
        # so this is already a Riemannian gradient: it must not go through
        # the optimizer's Euclidean-to-Riemannian conversion.
        riemannian_grad = -2.0 * (
            adjacency.unsqueeze(-1) * log_ij
        ).sum(dim=2) / degree_safe   # (B, N, D)

        # Flatten (B, N) → BN so each node is an independent point (the
        # manifold projection and retraction are point-wise), then take
        # the step: project −lr · grad onto the tangent space and retract.
        points = embedding.reshape(B * N, D)
        step = manifold.projection(points, -lr * riemannian_grad.reshape(B * N, D))
        embedding = manifold.retraction(points, step).reshape(B, N, D)

    # Non-finite iterates are unusable. Include the work budget and
    # graph size so the caller can identify the divergent call.
    if not torch.isfinite(embedding).all():
        max_deg = adjacency.sum(dim=-1).max().item()
        n_nan = torch.isnan(embedding).sum().item()
        n_inf = torch.isinf(embedding).sum().item()
        raise RuntimeError(
            f"hyperbolic_laplacian_eigenmaps diverged after {max_steps} "
            f"steps: {n_nan} NaN, {n_inf} inf out of {embedding.numel()} "
            f"output elements. Graph stats: N={N}, max degree={max_deg:.0f}. "
            f"This usually means `lr={lr}` is too large for the graph "
            f"structure, retry with a smaller `lr` (an order of magnitude "
            f"smaller is usually enough). Inputs are normalized by "
            f"per-node degree already, but extreme degree variance or "
            f"heavy adjacency weights can still trigger overflow."
        )

    return embedding
