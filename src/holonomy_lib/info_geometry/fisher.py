# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Synoros

"""Fisher information metric + natural-gradient primitives.

The Fisher information matrix is the canonical Riemannian metric on a
parametric statistical manifold:

    g_{ij}(θ) = E_{p_θ}[ (∂_i log p_θ)(∂_j log p_θ) ]
              = − E_{p_θ}[ ∂_i ∂_j log p_θ ].

It is the second-order Taylor coefficient of KL near the diagonal:

    KL(p_θ ‖ p_{θ + dθ}) = ½ dθᵀ F(θ) dθ + O(‖dθ‖³).

Information geometry studies the geodesics of this metric; in
optimization the natural gradient `F(θ)^{−1} ∇L` follows the
steepest-descent direction with respect to the same metric, which is
parameterization-invariant and a much better step for the loss
landscapes that overparameterized models induce (Amari 1998).

Two closed-form Fisher matrices are exposed for the most common
families; users with a custom likelihood can compute their own Fisher
either analytically or via `torch.autograd` over `log p`.

  fisher_information_categorical(p)
    Diagonal Fisher on the simplex, g_ii(p) = 1/p_i. Recovers the
    Hellinger-Bhattacharyya geometry whose geodesic distance is twice
    the Hellinger angle.

  fisher_information_gaussian_mean(Sigma)
    The Fisher information for the mean of N(μ, Σ) at fixed Σ. The
    closed form is g(μ) = Σ^{−1}, which makes the Fisher-metric
    geodesics between two means coincide with the Mahalanobis-distance
    straight lines for that fixed Σ.

  natural_gradient(grad, fisher_matrix)
    Apply the Fisher inverse to a Euclidean gradient via a stable
    linear solve (no explicit inverse). The output is the direction
    that minimizes the loss most efficiently in the Fisher geometry.

References:
  Amari, S.-I. (1998). Natural gradient works efficiently in learning.
    Neural Computation 10(2):251–276. The natural-gradient paper.
  Amari, S.-I. (2016). Information Geometry and Its Applications.
    Applied Mathematical Sciences 194, Springer. §2.5, §4.
  Martens, J. (2020). New insights and perspectives on the natural
    gradient method. JMLR 21(146):1–76. Modern revisit.
  Petersen, K. B., Pedersen, M. S. (2012). The Matrix Cookbook,
    §6 (Statistics) — Fisher information for the Gaussian family.
"""

from __future__ import annotations

import torch


def fisher_information_categorical(p: torch.Tensor) -> torch.Tensor:
    """Fisher information metric on the simplex Δ^{k−1}.

    For p in the open simplex, the Fisher information metric is the
    diagonal matrix

        g_ij(p) = δ_ij / p_i.

    This is the metric induced by the KL divergence's second-order
    Taylor expansion: KL(p ‖ p + δ) ≈ ½ Σ_i δ_i² / p_i.

    Args:
      p: (B, k) probability vectors on the open simplex. Entries must
        be positive — boundary cases (p_i = 0) make the Fisher
        information singular (geodesics blow up to infinite length).
        We clamp entries from below by the numerical-floor convention
        `1e-9` to keep the computation finite; pass cleaner inputs if
        you need bit-stable results.

    Returns:
      Fisher matrix: (B, k, k) diagonal SPD.

    References:
      Amari (2016), §2.4 example 4.
      Nielsen, F. (2020). An elementary introduction to information
        geometry. Entropy 22(10):1100.
    """
    if p.ndim < 1:
        raise ValueError(f"p must have at least 1 dim; got p.shape={tuple(p.shape)}")
    # Library-wide numerical floor (`1e-9`) lives in the audit catalog
    # as `numerical_floor_convention`. We never let 1/p go to infinity
    # silently — caller's job to either keep p in the interior, or to
    # interpret the clamped output (a large but finite Fisher).
    p_safe = p.clamp(min=1e-9)
    inv_p = torch.reciprocal(p_safe)
    return torch.diag_embed(inv_p)


def fisher_information_gaussian_mean(Sigma: torch.Tensor) -> torch.Tensor:
    """Fisher information for the mean μ of N(μ, Σ), at fixed Σ.

    Standard result (Amari 2016, eq. 2.43): for the Gaussian family
    parameterized by the mean with covariance held fixed,

        g(μ) = Σ^{−1}.

    The Fisher-geodesic distance between two means μ_1, μ_2 is then
    `sqrt((μ_1 − μ_2)ᵀ Σ^{−1} (μ_1 − μ_2))`, i.e. the Mahalanobis
    distance for the given Σ.

    Computed via Cholesky-based inverse rather than `linalg.inv` for
    numerical stability on near-singular Σ.

    Args:
      Sigma: (B, d, d) SPD covariance matrices.

    Returns:
      Fisher matrix: (B, d, d) SPD.

    References:
      Amari (2016), §2.4 example 5.
      Petersen-Pedersen (2012), §6.
    """
    if Sigma.ndim < 2 or Sigma.shape[-1] != Sigma.shape[-2]:
        raise ValueError(
            f"Sigma must be (..., d, d); got Sigma.shape={tuple(Sigma.shape)}"
        )
    # Cholesky-based inverse: stable on the SPD cone, faster than the
    # general LU inverse, and matches the convention used in
    # `kl_divergence_gaussian` for consistency.
    L = torch.linalg.cholesky(Sigma)
    eye = torch.eye(
        Sigma.shape[-1], dtype=Sigma.dtype, device=Sigma.device,
    ).expand_as(Sigma)
    L_inv = torch.linalg.solve_triangular(L, eye, upper=False)
    # Σ^{−1} = L^{−T} L^{−1}.
    return L_inv.mT @ L_inv


def natural_gradient(
    grad: torch.Tensor,
    fisher_matrix: torch.Tensor,
) -> torch.Tensor:
    """Natural gradient: F(θ)^{−1} · ∇L.

    The natural gradient (Amari 1998) is the steepest-descent direction
    on a statistical manifold under the Fisher metric. Unlike the
    Euclidean gradient, it is invariant under reparameterization, which
    eliminates the parameterization-dependent ill-conditioning that
    standard gradients suffer from on overparameterized models.

    Implementation: solve `F · d = grad` rather than `inv(F) @ grad`,
    so the linear system inherits whatever conditioning F has without
    the extra rounding from an explicit inverse.

    Args:
      grad: (B, d) Euclidean gradient at θ.
      fisher_matrix: (B, d, d) Fisher information matrix at θ, SPD.

    Returns:
      (B, d) natural gradient.

    References:
      Amari (1998).
      Martens (2020), §2 — modern derivation.
    """
    if fisher_matrix.ndim < 2:
        raise ValueError(
            f"fisher_matrix must be (..., d, d) with at least 2 dims; "
            f"got fisher_matrix.shape={tuple(fisher_matrix.shape)}"
        )
    if grad.shape[-1] != fisher_matrix.shape[-1]:
        raise ValueError(
            f"grad last dim {grad.shape[-1]} must match fisher_matrix "
            f"last dim {fisher_matrix.shape[-1]}"
        )
    if fisher_matrix.shape[-1] != fisher_matrix.shape[-2]:
        raise ValueError(
            f"fisher_matrix must be (..., d, d); got "
            f"fisher_matrix.shape={tuple(fisher_matrix.shape)}"
        )
    return torch.linalg.solve(
        fisher_matrix, grad.unsqueeze(dim=-1),
    ).squeeze(dim=-1)
