# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Synoros

"""Bregman and KL divergences, batched over leading dimensions on any device.

A Bregman divergence is the asymmetric "distance" induced by a strictly
convex generator F : Ω ⊂ ℝᵈ → ℝ:

    D_F(p ‖ q) = F(p) − F(q) − ⟨∇F(q), p − q⟩.

It is the gap between the function value at `p` and its first-order
Taylor approximation at `q`. Choices of `F` recover familiar
divergences:

  F(x) = ½ ⟨x, x⟩                        → squared Euclidean ½‖p−q‖²
  F(x) = Σ x_i log x_i − x_i             → generalized KL (Csiszár 1991)
                                            (= KL on the simplex)
  F(x) = − Σ log x_i                     → Itakura-Saito
  F(X) = −log det X (for SPD matrices)  → LogDet / Burg matrix divergence

This module provides:

  bregman_divergence(p, q, potential)
      The fully general form, parameterized by a callable `potential`
      that returns (F(x), ∇F(x)) given x. Caller supplies the convex
      function; we evaluate the Bregman formula and respect the leading
      batch dimensions of `p`/`q`.

  kl_divergence_categorical(p, q)
      KL(p ‖ q) for discrete distributions on the simplex, with the
      0·log 0 = 0 convention. On the simplex this equals `bregman_divergence`
      with `potential = (x · log x − x).sum()`, and it is cheaper. The
      Bregman form also adds Σq − Σp, which vanishes only when both sum to one.

      Autograd note: `xlogy` preserves zero-mass terms and returns
      `+inf` when `p_i > 0` and `q_i = 0` (support mismatch).
      Gradients through near-zero bins are not numerically reliable;
      use only for inference unless you're sure about the support.

  kl_divergence_gaussian(mu_p, Sigma_p, mu_q, Sigma_q)
      KL between two multivariate Gaussians:

          KL = ½ ( tr(Σ_q⁻¹ Σ_p) + (μ_q − μ_p)ᵀ Σ_q⁻¹ (μ_q − μ_p)
                   − d + log det Σ_q − log det Σ_p ).

      Uses Cholesky-based solves for stability.

References:
  Bregman, L. M. (1967). The relaxation method of finding the common
    point of convex sets ... USSR Comp. Math. Math. Phys. 7(3):200-217.
    The original definition.
  Banerjee, A., Merugu, S., Dhillon, I. S., Ghosh, J. (2005).
    Clustering with Bregman divergences. JMLR 6:1705-1749, §2.
  Csiszár, I. (1991). Why least squares and maximum entropy?
    An axiomatic approach to inference for linear inverse problems.
    Ann. Stat. 19(4):2032-2066. Bregman ↔ generalized KL connection.
  Amari, S.-I. (2016). Information Geometry and Its Applications, §1.5.
"""

from __future__ import annotations

from collections.abc import Callable

import torch


def bregman_divergence(
    p: torch.Tensor,
    q: torch.Tensor,
    potential: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
) -> torch.Tensor:
    """Bregman divergence `D_F(p ‖ q) = F(p) − F(q) − ⟨∇F(q), p − q⟩`.

    Args:
      p, q: (B, d) tensors. Must broadcast against each other; output
        has the broadcast batch shape.
      potential: callable taking `x: (..., d)` and returning
        `(F(x): (...,), grad_F(x): (..., d))`. Both outputs must
        preserve the leading dims of the input. The caller is
        responsible for choosing a strictly convex `F` and confirming
        `p`, `q` live in its domain.

    Returns:
      D_F(p ‖ q): (B,) divergence values. Non-negative when F is
      strictly convex (Bregman 1967); equals zero iff p = q.

    Example (squared Euclidean):
      >>> def half_norm_sq(x):
      ...     return 0.5 * (x * x).sum(dim=-1), x
      >>> bregman_divergence(p, q, half_norm_sq)
      # = ½ ‖p − q‖²

    References:
      Bregman (1967).
      Banerjee et al. (2005), §2.
    """
    if p.shape[-1] != q.shape[-1]:
        raise ValueError(
            f"p and q must have matching last dim; got p.shape[-1]="
            f"{p.shape[-1]} vs q.shape[-1]={q.shape[-1]}"
        )
    # Only the gradient at q enters the formula, so the potential's gradient
    # at p is discarded.
    F_p, _ = potential(p)
    F_q, grad_F_q = potential(q)
    # ⟨∇F(q), p − q⟩, summed over the feature dim so it has the batch shape.
    inner = (grad_F_q * (p - q)).sum(dim=-1)
    return F_p - F_q - inner


def kl_divergence_categorical(
    p: torch.Tensor,
    q: torch.Tensor,
) -> torch.Tensor:
    """KL divergence between two discrete distributions on the simplex.

        KL(p ‖ q) = Σ_i p_i · (log p_i − log q_i).

    Uses the convention 0 · log(0/x) = 0 (continuous limit). Both `p`
    and `q` must be on the simplex (non-negative, sum to 1 along the
    last dim); we do not enforce this, so it is a precondition.
    Gradients at zero probabilities need not be finite; differentiate
    only where both distributions have strictly positive entries.

    Args:
      p, q: (B, k) probability tensors. Last dim is the support size.

    Returns:
      KL: (B,) divergence values, non-negative (Gibbs inequality).

    References:
      Cover, T. M., Thomas, J. A. (2006). Elements of Information
        Theory, 2nd ed., §2.3.
      Amari (2016), §2.4.
    """
    if p.shape[-1] != q.shape[-1]:
        raise ValueError(
            f"p and q must have matching last dim; got {p.shape[-1]} vs "
            f"{q.shape[-1]}"
        )
    # p log(p / q) is split into two xlogy terms. xlogy(p, x) is 0 wherever
    # p is 0, so zero-mass bins drop out without a 0 * log 0 nan. No clamp
    # is applied, so tiny positive p keep their exact p log p. A bin with
    # p_i > 0 and q_i = 0 gives +inf, as the module note says.
    return (torch.special.xlogy(p, p) - torch.special.xlogy(p, q)).sum(dim=-1)


def kl_divergence_gaussian(
    mu_p: torch.Tensor,
    Sigma_p: torch.Tensor,
    mu_q: torch.Tensor,
    Sigma_q: torch.Tensor,
) -> torch.Tensor:
    """KL divergence between two multivariate Gaussians.

        KL(N(μ_p, Σ_p) ‖ N(μ_q, Σ_q))
          = ½ ( tr(Σ_q⁻¹ Σ_p)
                + (μ_q − μ_p)ᵀ Σ_q⁻¹ (μ_q − μ_p)
                − d
                + log det Σ_q − log det Σ_p ).

    Stability: we factorize Σ_q via Cholesky once and reuse the
    factorization for both the trace term `tr(Σ_q⁻¹ Σ_p)` and the
    Mahalanobis term `(μ_q − μ_p)ᵀ Σ_q⁻¹ (μ_q − μ_p)`. `log det Σ_q`
    comes for free from the Cholesky diagonal. For `Σ_p` we just need
    `log det Σ_p`, also via Cholesky. A covariance that is not SPD makes
    `torch.linalg.cholesky` raise, so bad input fails instead of returning
    a number.

    Args:
      mu_p, mu_q: (..., d) means.
      Sigma_p, Sigma_q: (..., d, d) covariances. Must be SPD.

    Returns:
      KL: (...,) divergence values, non-negative.

    References:
      Amari (2016), §2.4; exponential families.
      Petersen, K. B., Pedersen, M. S. (2012). The Matrix Cookbook,
        eq. 380; Gaussian-Gaussian KL.
    """
    d = mu_p.shape[-1]
    if mu_q.shape[-1] != d:
        raise ValueError(
            f"mu_q's last dim must match mu_p's d={d}; "
            f"got mu_q.shape={tuple(mu_q.shape)}"
        )
    if Sigma_p.shape[-1] != d or Sigma_p.shape[-2] != d:
        raise ValueError(
            f"Sigma_p must be (..., d, d) with d={d}; "
            f"got {tuple(Sigma_p.shape)}"
        )
    if Sigma_q.shape[-1] != d or Sigma_q.shape[-2] != d:
        raise ValueError(
            f"Sigma_q must be (..., d, d) with d={d}; "
            f"got {tuple(Sigma_q.shape)}"
        )

    L_q = torch.linalg.cholesky(Sigma_q)
    # Σ_q⁻¹ Σ_p through the Cholesky factor of Σ_q: two triangular solves,
    # so no explicit inverse of Σ_q is formed.
    Sigma_q_inv_Sigma_p = torch.cholesky_solve(Sigma_p, L_q)
    trace_term = torch.diagonal(
        Sigma_q_inv_Sigma_p, dim1=-2, dim2=-1,
    ).sum(dim=-1)

    diff = (mu_q - mu_p).unsqueeze(dim=-1)  # (..., d, 1)
    # Σ_q⁻¹ Δμ via cholesky_solve; then ⟨Δμ, ·⟩ is a dot product.
    Sigma_q_inv_diff = torch.cholesky_solve(diff, L_q).squeeze(dim=-1)
    mahalanobis = ((mu_q - mu_p) * Sigma_q_inv_diff).sum(dim=-1)

    # log det Σ = 2 Σ log diag(L) for the Cholesky factor.
    log_det_Sigma_q = 2.0 * torch.log(
        torch.diagonal(L_q, dim1=-2, dim2=-1)
    ).sum(dim=-1)
    L_p = torch.linalg.cholesky(Sigma_p)
    log_det_Sigma_p = 2.0 * torch.log(
        torch.diagonal(L_p, dim1=-2, dim2=-1)
    ).sum(dim=-1)

    return 0.5 * (
        trace_term + mahalanobis - float(d)
        + log_det_Sigma_q - log_det_Sigma_p
    )
