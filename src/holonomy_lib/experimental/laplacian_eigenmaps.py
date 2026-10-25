# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Synoros

"""Manifold-valued Laplacian eigenmaps via Riemannian SGD.

Status: experimental, in progress. The objective is attraction-only:
it pulls neighbours together and has no term that keeps the embedding from
collapsing, so with enough steps every connected component contracts to a point.
A correct version needs a non-collapse constraint (the Belkin-Niyogi
normalization, or negative sampling as in Nickel-Kiela 2017).

The Euclidean Laplacian-eigenmap embedding (Belkin–Niyogi 2003)
positions graph nodes in `R^k` by solving

    min_{Y ∈ R^{N×k}}  Σ_{i,j} A_{ij} · ‖Y_i − Y_j‖²,

a quadratic objective whose minimizer is the bottom-k eigenvectors of
the graph Laplacian (modulo a constant constraint). The natural
generalization to a Riemannian manifold replaces the Euclidean
squared distance with the manifold geodesic distance squared:

    min_{Y ∈ M^N}  Σ_{i,j} A_{ij} · d_M(Y_i, Y_j)²,

which no longer has a closed-form spectral solution but admits
Riemannian gradient descent: the Riemannian gradient of the
per-edge term `d_M(Y_i, Y_j)²` with respect to `Y_i` is
`−2 · log_{Y_i}(Y_j)` (the steepest direction to move `Y_i` toward
`Y_j`). For symmetric adjacency `A`, the total gradient on `Y_i` is

    grad_{Y_i}  =  −2 · (1 / deg(i)) · Σ_j A_{ij} · log_{Y_i}(Y_j).

The `1/deg(i)` factor is the random-walk normalization (the
gradient of the *normalized* Laplacian energy, equivalent to using
`L_rw = I − D⁻¹A`). Without it, hub nodes receive steps ~ deg(i)
times larger than leaf nodes, and dense graphs (n_edges ≳ 500)
overflow the manifold's `cosh`/`tan` factors well before
convergence. The normalized form is degree-invariant, so the same
`lr` works for graphs of any size and density.

We push this through `RiemannianSGD`, which handles the
`projection → retraction` cycle.

When `M = LorentzManifold`, the output embedding inherits hyperbolic
geometry's exponential capacity — the standard cure for high-tree-
distortion graphs that Euclidean embeddings saturate on (Sarkar
2011 distortion bound; Nickel–Kiela 2017 *Poincaré Embeddings*).

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

import torch

from holonomy_lib.optimization import RiemannianSGD


def hyperbolic_laplacian_eigenmaps(
    adjacency: torch.Tensor,
    manifold,
    max_steps: int = 200,
    lr: float = 0.05,
    generator: torch.Generator | None = None,
    init: torch.Tensor | None = None,
) -> torch.Tensor:
    """Embed the nodes of `adjacency` on `manifold` by minimizing

        Σ_{i,j} A_{ij} · d_M(Y_i, Y_j)²

    via Riemannian SGD.

    Args:
      adjacency: `(B, N, N)` symmetric non-negative graph adjacency
        per batch. Self-loops `A_{ii}` are tolerated (their gradient
        contribution is zero by `log_{Y_i}(Y_i) = 0`) but discouraged
        for numerical hygiene; pre-zero the diagonal if you want.
      manifold: a manifold object exposing `random_point`, `log`,
        `projection`, `retraction`, `norm` (see e.g.
        `LorentzManifold`). The embedding dimension is taken from
        `manifold.n`; output has shape `(B, N, manifold.n + 1)` for
        the Lorentz case (ambient form).
      max_steps: number of RSGD steps. Default 200 — enough for
        moderate-size graphs (N ≤ a few hundred) to reach a stable
        configuration on a Hadamard manifold; raise for harder
        objectives.
      lr: Riemannian SGD step size. Default 0.05 follows
        `torch.optim.SGD`-style empirical defaults for embedding
        problems; tune if the loss diverges (smaller) or stalls
        (larger).
      generator: optional `torch.Generator` for reproducible init.
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
      adjacency-based loss (not implemented here — Stage 2 scope is
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

    B, N, _ = adjacency.shape
    # Ambient dim is manifold-specific (n+1 for Lorentz, n for
    # κ-stereographic). Use the `ambient_dim` property when available
    # so the primitive is model-agnostic.
    D = getattr(manifold, "ambient_dim", manifold.n + 1)

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

    opt = RiemannianSGD(manifold, lr=lr)

    # Per-node degree, used to make the gradient scale-invariant w.r.t.
    # graph size. Without this, a node of degree 100 receives a step
    # ~50× larger than a node of degree 2 at the same lr, and dense
    # graphs (n_edges ≳ 500) blow up the manifold operations
    # (cosh(α) overflows at α ≈ 700 on float64). Equivalent to using
    # the random-walk normalized Laplacian L_rw = I − D⁻¹A — the
    # canonical normalization for spectral embedding (von Luxburg 2007).
    #
    # For isolated nodes (deg = 0), the gradient is zero anyway (no
    # edges contribute), so the clamp's exact floor doesn't affect
    # correctness — only prevents a divide-by-zero on the kept-zero row.
    degree = adjacency.sum(dim=-1, keepdim=True)              # (B, N, 1)
    degree_safe = degree.clamp(min=torch.finfo(adjacency.dtype).tiny)

    for step in range(max_steps):
        # Pairwise log_Y_i(Y_j) — broadcast `embedding` along the i and
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

        # Ambient gradient on each node i: −2 Σ_j A_{ij} · log_{Y_i}(Y_j),
        # normalized by deg(i) so the per-step magnitude is the
        # *average* log-direction toward neighbors (degree-invariant).
        ambient_grad = -2.0 * (
            adjacency.unsqueeze(-1) * log_ij
        ).sum(dim=2) / degree_safe   # (B, N, D)

        # RSGD treats the leading axis as the batch axis; we flatten
        # (B, N) → BN so each node is an independent optimization
        # point (the manifold projection/retraction is point-wise).
        embedding = opt.step(
            embedding.reshape(B * N, D),
            ambient_grad.reshape(B * N, D),
        ).reshape(B, N, D)

    # Fail-loud: silent NaN is the failure mode we just fixed; refuse
    # to return a degenerate result. Diagnostic includes the hyper-
    # parameters and graph stats so the caller can adjust.
    if not torch.isfinite(embedding).all():
        max_deg = adjacency.sum(dim=-1).max().item()
        n_nan = torch.isnan(embedding).sum().item()
        n_inf = torch.isinf(embedding).sum().item()
        raise RuntimeError(
            f"hyperbolic_laplacian_eigenmaps diverged after {max_steps} "
            f"steps: {n_nan} NaN, {n_inf} inf out of {embedding.numel()} "
            f"output elements. Graph stats: N={N}, max degree={max_deg:.0f}. "
            f"This usually means `lr={lr}` is too large for the graph "
            f"structure — retry with a smaller `lr` (an order of magnitude "
            f"smaller is usually enough). Inputs are normalized by "
            f"per-node degree already, but extreme degree variance or "
            f"heavy adjacency weights can still trigger overflow."
        )

    return embedding
