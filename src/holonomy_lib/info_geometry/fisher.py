# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Synoros

"""Fisher information metric + natural-gradient primitives.

The Fisher information matrix is the canonical Riemannian metric on a
parametric statistical manifold:

    g_{ij}(θ) = E_{p_θ}[ (∂_i log p_θ)(∂_j log p_θ) ]
              = − E_{p_θ}[ ∂_i ∂_j log p_θ ].

It is the second-order Taylor coefficient of KL near the diagonal:

    KL(p_θ ‖ p_{θ + dθ}) = ½ dθᵀ F(θ) dθ + O(‖dθ‖³).

Information geometry studies the geodesics of this metric. In
optimization, the natural gradient `F(θ)^{−1} ∇L` is the steepest-descent
direction under this metric. The metric does not depend on how θ is
parameterized, so the direction does not either (Amari 1998).

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
    Apply the Fisher inverse to a Euclidean gradient via a Cholesky
    solve (no explicit inverse). The output is the steepest-descent
    direction under the Fisher metric.

References:
  Amari, S.-I. (1998). Natural gradient works efficiently in learning.
    Neural Computation 10(2):251–276. The natural-gradient paper.
  Amari, S.-I. (2016). Information Geometry and Its Applications.
    Applied Mathematical Sciences 194, Springer. §2.5, §4.
  Martens, J. (2020). New insights and perspectives on the natural
    gradient method. JMLR 21(146):1–76. Modern revisit.
  Petersen, K. B., Pedersen, M. S. (2012). The Matrix Cookbook,
    §6 (Statistics); Fisher information for the Gaussian family.
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
      p: (B, k) probability vectors on the open simplex. Entries should
        be positive. Metric coefficients diverge as p_i approaches zero,
        but the boundary is at finite Fisher distance. For example, the
        Bernoulli distance from p=1/2 to p=0 is pi/2.
        Entries below `1e-9` are clamped up to `1e-9` in the input dtype
        (float16 is a special case, see the body).

    Returns:
      Fisher matrix: (B, k, k) diagonal SPD.

    References:
      Amari (2016), §2.4 example 4.
      Nielsen, F. (2020). An elementary introduction to information
        geometry. Entropy 22(10):1100.
    """
    if p.ndim < 1:
        raise ValueError(f"p must have at least 1 dim; got p.shape={tuple(p.shape)}")
    # The clamp is what keeps 1/p finite at the boundary. The floor is the
    # library-wide convention `numerical_floor_convention` in
    # notes/magic_numbers.md, not a per-call knob. Keeping p in the interior
    # is the caller's job; a clamped entry gives a large but finite value.
    # float16 cannot represent 1e-9 (it rounds to 0), so the floor is raised
    # to the dtype's smallest normal number when that is larger. Every other
    # float dtype keeps the 1e-9 convention.
    p_safe = p.clamp(min=max(1e-9, torch.finfo(p.dtype).tiny))
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

    Computed from a Cholesky factor with `cholesky_inverse`, not
    `linalg.inv`. A matrix that is not SPD makes `cholesky` raise.

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
    # Σ is SPD, so factor it once and invert from the factor. This skips
    # the pivoting an LU inverse does, and kl_divergence_gaussian uses the
    # same factorization.
    L = torch.linalg.cholesky(Sigma)
    return torch.cholesky_inverse(L)


def natural_gradient(
    grad: torch.Tensor,
    fisher_matrix: torch.Tensor,
) -> torch.Tensor:
    """Natural gradient: F(θ)^{−1} · ∇L.

    The natural gradient (Amari 1998) is the steepest-descent direction
    on a statistical manifold under the Fisher metric. Unlike the
    Euclidean gradient, the direction does not change under a smooth
    reparameterization of θ, since the metric is the same geometric
    object in every coordinate system. That removes one source of
    parameterization-dependent behavior. It does not make an
    ill-conditioned F any better conditioned.

    Implementation: solve `F · d = grad` through a Cholesky factor of F,
    rather than forming `inv(F) @ grad`. The solve inherits whatever
    conditioning F has, without the extra rounding from an explicit inverse.

    Args:
      grad: (B, d) Euclidean gradient at θ.
      fisher_matrix: (B, d, d) Fisher information matrix at θ, SPD.

    Returns:
      (B, d) natural gradient.

    References:
      Amari (1998).
      Martens (2020), §2; modern derivation.
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
    L = torch.linalg.cholesky(fisher_matrix)
    return torch.cholesky_solve(grad.unsqueeze(dim=-1), L).squeeze(dim=-1)
