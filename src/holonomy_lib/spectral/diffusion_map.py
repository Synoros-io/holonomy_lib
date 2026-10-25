# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Synoros

"""Diffusion-map embedding (Coifman-Lafon 2006).

Given an undirected weighted graph `A`, the random walk on its nodes
has a transition matrix `P = D⁻¹ A`. The eigendecomposition of `P`
gives the long-time behavior of the random walk: for an eigenpair
(μ_j, φ_j), starting at node `x_i` and taking `t` random-walk steps,
the probability of arriving at `x_l` is

    P^t(x_i, x_l) = π(x_l) Σ_j μ_j^t · φ_j(x_i) · φ_j(x_l),

where π(x_l) = d_l / vol(A), and the right eigenvectors φ_j are
orthonormal under the π-weighted inner product.

The **diffusion map** at time `t` lifts each node to a `k`-dimensional
Euclidean point whose pairwise distances mirror the diffusion-time `t`
random walk distance:

    Ψ_t(x_i) = ( μ_1^t · φ_1(x_i), …, μ_k^t · φ_k(x_i) )

    D_t(x_i, x_l)² = ‖ Ψ_t(x_i) − Ψ_t(x_l) ‖²
                   = Σ_j μ_j^{2t} · ( φ_j(x_i) − φ_j(x_l) )²

The trivial top eigenpair (μ_1 = 1, φ_1 = constant) is dropped; it
encodes only "we're somewhere on the graph", carrying no metric
information.

Implementation: diagonalize the full random-walk Laplacian, convert
its eigenvalues by `μ_j = 1 − λ_j`, and select the k largest absolute
nontrivial transition eigenvalues. Multiply its right eigenvectors by
sqrt(vol(A)) to normalize under π, then by the signed integer power
`μ_j^t`. The first Laplacian eigenvector is dropped.

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
from numbers import Integral

import torch

from holonomy_lib._graph_utils import drop_self_loops
from holonomy_lib.spectral import laplacian as _L
from holonomy_lib.spectral.embedding import laplacian_eigenmaps

# The normalized Laplacian of a nonnegative undirected graph has spectrum
# in [0, 2] (Chung, Spectral Graph Theory, 1997, Lemma 1.7).
NORMALIZED_LAPLACIAN_SPECTRAL_RADIUS = 2.0


def diffusion_map(
    A: torch.Tensor,
    k: int,
    t: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Diffusion-map embedding at time `t`.

    Args:
      A: (B, n, n) symmetric non-negative weighted adjacency.
      k: embedding dimension. Must satisfy `1 ≤ k ≤ n − 1` (we drop the
        trivial null eigenvector, leaving `n − 1` non-trivial modes).
      t: nonnegative integer number of random-walk steps. Larger `t`
        emphasizes modes whose transition eigenvalues have magnitude
        close to one. Negative eigenvalues retain their signed powers.

    Returns:
      transition_eigvals: (B, k) eigenvalues μ_j = 1 − λ_j of the
        transition matrix P (descending in magnitude).
      embedding: (B, n, k) diffusion coordinates `Ψ_t(x_i)`.

    Notes:
      Cost is dominated by the eigendecomposition of L_sym, O(B · n³).
      Disconnected graphs emit a warning and retain all but one stationary
      mode. Isolated nodes have no positive stationary mass; their coordinates
      are set to zero, and diffusion distances involving them are undefined.

    References:
      Coifman, R. R., Lafon, S. (2006). Diffusion maps. Applied and
        Computational Harmonic Analysis 21(1):5-30, §2.3-2.4.
    """
    if A.ndim < 2 or A.shape[-1] != A.shape[-2]:
        raise ValueError(
            f"A must be (..., n, n); got A.shape={tuple(A.shape)}"
        )
    if not A.is_floating_point():
        raise ValueError("A must have a real floating-point dtype")
    A = drop_self_loops(A)
    n = A.shape[-1]
    if k <= 0 or k > n - 1:
        raise ValueError(
            f"k must satisfy 1 <= k <= n - 1 = {n - 1}, got k={k}"
        )
    if isinstance(t, bool) or not isinstance(t, Integral) or t < 0:
        raise ValueError(f"diffusion time t must be a nonnegative integer, got {t}")
    if (A < 0).any():
        raise ValueError("A must have nonnegative weights")

    # Small Laplacian eigenvalues alone miss negative transition modes.
    # We need the full spectrum before sorting by transition magnitude.
    eigvals, eigvecs = laplacian_eigenmaps(
        A, k=n, laplacian_type="random_walk",
    )
    # A disconnected graph has a stationary mode for each nontrivial
    # component. Removing any one of these preserves within-component
    # differences, but it need not remove the global constant mode.
    degree = _L.degree(A)
    # L_sym has spectrum in [0, 2]; the eigensolver's roundoff grows
    # with the dimension, rather than with the adjacency's units.
    tolerance = torch.finfo(A.dtype).eps * n * NORMALIZED_LAPLACIAN_SPECTRAL_RADIUS
    counts = (eigvals.abs() <= tolerance).sum(dim=-1)
    n_near_zero = int(counts.max().item()) if counts.numel() else 0
    has_isolates = bool((degree == 0).any())
    if n_near_zero > 1 or has_isolates:
        warnings.warn(
            "diffusion_map: input graph is disconnected or has isolated nodes. "
            "Only one stationary mode is dropped; interpret distances within "
            "components with positive stationary mass. Isolated coordinates "
            "are set to zero.",
            stacklevel=2,
        )
    # Drop index 0 (the smallest eigenvalue, ≈ 0; eigenvector is the
    # stationary mode).
    eigvals_non_null = eigvals[..., 1:]
    eigvecs_non_null = eigvecs[..., 1:]

    # Convert L_rw eigenvalues to P eigenvalues: μ = 1 − λ.
    transition_eigvals = 1.0 - eigvals_non_null
    order = transition_eigvals.abs().argsort(dim=-1, descending=True)[..., :k]
    transition_eigvals = transition_eigvals.gather(-1, order)
    eigvecs_non_null = eigvecs_non_null.gather(
        -1, order.unsqueeze(-2).expand(*A.shape[:-2], n, k),
    )
    volume_scale = degree.sum(dim=-1).sqrt().unsqueeze(-1).unsqueeze(-1)
    embedding = eigvecs_non_null * volume_scale
    embedding = torch.where((degree > 0).unsqueeze(-1), embedding, 0.0)
    embedding = embedding * transition_eigvals.pow(int(t)).unsqueeze(-2)
    return transition_eigvals, embedding
