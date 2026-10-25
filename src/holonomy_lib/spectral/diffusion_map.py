# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Synoros

"""Diffusion-map embedding (Coifman-Lafon 2006).

Given an undirected weighted graph `A`, the random walk on its nodes
has a transition matrix `P = D⁻¹ A`. The eigendecomposition of `P`
gives the long-time behavior of the random walk: for an eigenpair
(μ_j, φ_j), starting at node `x_i` and taking `t` random-walk steps,
the probability of arriving at `x_l` is

    P^t(x_i, x_l) = Σ_j μ_j^t · φ_j(x_i) · φ_j(x_l).

The **diffusion map** at time `t` lifts each node to a `k`-dimensional
Euclidean point whose pairwise distances mirror the diffusion-time `t`
random walk distance:

    Ψ_t(x_i) = ( μ_1^t · φ_1(x_i), …, μ_k^t · φ_k(x_i) )

    D_t(x_i, x_l)² = ‖ Ψ_t(x_i) − Ψ_t(x_l) ‖²
                   = Σ_j μ_j^{2t} · ( φ_j(x_i) − φ_j(x_l) )²

The trivial top eigenpair (μ_1 = 1, φ_1 = constant) is dropped — it
encodes only "we're somewhere on the graph", carrying no metric
information.

Implementation: we compute the bottom-(k+1) eigenpairs of the random-
walk Laplacian `L_rw = I − D⁻¹ A` (via `laplacian_eigenmaps`), then
convert eigenvalues to transition-matrix eigenvalues by `μ_j = 1 − λ_j`
and scale the eigenvectors by `μ_j^t`. The dropped null eigenvector
is at index 0 (eigenvalue 0 of L_rw ↔ eigenvalue 1 of P).

Disconnected components have multiple zero L-eigenvalues; we drop only
one. Callers on disconnected graphs should mask by component
membership before interpreting cross-component distances.

References:
  Coifman, R. R., Lafon, S. (2006). Diffusion maps. Applied and
    Computational Harmonic Analysis 21(1):5–30. The defining paper.
  Lafon, S., Lee, A. B. (2006). Diffusion maps and coarse-graining:
    A unified framework for dimensionality reduction, graph
    partitioning, and data set parameterization. IEEE PAMI 28(9):
    1393–1403.
  Nadler, B., Lafon, S., Coifman, R. R., Kevrekidis, I. G. (2006).
    Diffusion maps, spectral clustering and reaction coordinates of
    dynamical systems. Applied and Computational Harmonic Analysis
    21(1):113–127.
"""

from __future__ import annotations

import warnings

import torch

from holonomy_lib.spectral.embedding import laplacian_eigenmaps


def diffusion_map(
    A: torch.Tensor,
    k: int,
    t: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Diffusion-map embedding at time `t`.

    Args:
      A: (B, n, n) symmetric non-negative weighted adjacency.
      k: embedding dimension. Must satisfy `1 ≤ k ≤ n − 1` (we drop the
        trivial null eigenvector, leaving `n − 1` non-trivial modes).
      t: diffusion time. `t ≥ 0`; larger `t` emphasizes longer-range
        random-walk structure.

    Returns:
      transition_eigvals: (B, k) eigenvalues μ_j = 1 − λ_j of the
        transition matrix P (descending in magnitude).
      embedding: (B, n, k) diffusion coordinates `Ψ_t(x_i)`.

    Notes:
      Cost is dominated by the eigendecomposition of L_sym, O(B · n³).
      For large `n` use a Lanczos-based variant (planned).

    References:
      Coifman-Lafon (2006), §3.
    """
    if t < 0:
        raise ValueError(f"diffusion time t must be >= 0, got {t}")
    if A.ndim < 2 or A.shape[-1] != A.shape[-2]:
        raise ValueError(
            f"A must be (..., n, n); got A.shape={tuple(A.shape)}"
        )
    n = A.shape[-1]
    if k <= 0 or k > n - 1:
        raise ValueError(
            f"k must satisfy 1 <= k <= n - 1 = {n - 1}, got k={k}"
        )

    # Bottom-(k+1) eigenpairs of L_rw — the +1 is for the trivial null
    # eigenvector we'll drop. `random_walk` eigenvectors are the
    # P-eigenvectors needed by Coifman-Lafon.
    eigvals, eigvecs = laplacian_eigenmaps(
        A, k=k + 1, laplacian_type="random_walk",
    )
    # A connected graph has exactly one zero L_rw eigenvalue (the
    # stationary distribution). A disconnected graph with `c` components
    # has `c` zero eigenvalues; we still drop only one, leaving `c − 1`
    # degenerate null modes embedded in the output as constant-on-
    # component vectors with `μ_j^t ≈ 1`. These coordinates are NOT
    # geometrically meaningful — they're stationary modes per component.
    # Warn so the caller can mask by component membership.
    # 1e-9 is the library's `numerical_floor_convention` (ALLOWED).
    n_near_zero = (eigvals.abs() < 1e-9).sum(dim=-1).max().item()
    if n_near_zero > 1:
        warnings.warn(
            f"diffusion_map: input graph has at least {n_near_zero} near-zero "
            f"L_rw eigenvalues, suggesting disconnected components. The "
            f"embedding will contain {n_near_zero - 1} additional null "
            f"modes that are not geometrically meaningful — mask by "
            f"component membership before interpreting cross-component "
            f"distances.",
            stacklevel=2,
        )
    # Drop index 0 (the smallest eigenvalue, ≈ 0; eigenvector is the
    # stationary mode).
    eigvals_non_null = eigvals[..., 1:]              # (B, k)
    eigvecs_non_null = eigvecs[..., 1:]              # (B, n, k)

    # Convert L_rw eigenvalues to P eigenvalues: μ = 1 − λ.
    transition_eigvals = 1.0 - eigvals_non_null      # (B, k)

    # μ_j^t scaling. Clamp transition eigenvalues to non-negative to
    # guard against tiny float drift (P eigenvalues are guaranteed in
    # [-1, 1] but mostly in [0, 1] for bipartite-free graphs; for
    # bipartite-containing graphs the negative tail is meaningful but
    # produces complex μ^t when raised to non-integer t. We clamp to
    # 0 in that boundary case, which is the standard convention).
    mu_clamped = transition_eigvals.clamp(min=0.0)
    scale = (mu_clamped ** t).unsqueeze(dim=-2)      # (B, 1, k)

    embedding = scale * eigvecs_non_null
    return transition_eigvals, embedding
