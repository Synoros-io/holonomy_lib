# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Synoros

"""Spectral dimension from a graph / operator Laplacian spectrum.

The spectral dimension `d_s` is the exponent governing the small-time decay of
the heat-kernel return probability:

    p(t) = (1/n) Σ_i exp(-t·λ_i)  ~  t^(-d_s/2)   as t → 0+,

equivalently `d_s = -2 · d log p / d log t`. It need not be an integer (fractals,
multifractal media) and on a finite graph it is read off as the slope of
`log p` against `log t` over a scaling window: outside that window finite-size
effects flatten it (`p → (#zero modes)/n` as `t → ∞`, biasing `d_s` toward 0).

This estimates `d_s` as the least-squares slope of `log p(t)` against `log t`
over a caller-supplied set of times `t`; choose them inside the power-law
window. The input is the Laplacian spectrum (e.g. from
`holonomy_lib.spectral.laplacian` + an eigensolver), so the same routine serves
graphs, fractal approximants, and discretized continua.

References:
  Rammal, R., Toulouse, G. (1983). Random walks on fractal structures and
    percolation clusters. J. Physique Lett. 44(1):L13-L22 (spectral dimension).
  Durhuus, B., Jonsson, T., Wheater, J. F. (2007). The spectral dimension of
    generic trees. J. Stat. Phys. 128:1237-1260 (return-probability form).
  Hambly, B. M., Kigami, J., Kumagai, T. (2002). Multifractal formalisms for
    the local spectral and walk dimensions. Math. Proc. Camb. Phil. Soc.
    132:555-571 (local / pointwise version).
"""

from __future__ import annotations

import math

import torch


def spectral_dimension(
    eigenvalues: torch.Tensor, t: torch.Tensor,
) -> torch.Tensor:
    """Spectral dimension `d_s = -2 · slope of log p(t) vs log t`, where
    `p(t) = mean_i exp(-t·λ_i)` is the heat-kernel return probability.

    Args:
      eigenvalues: `(B, n)` Laplacian eigenvalues, `>= 0` (float-noise negatives
        are clamped to 0). A `(n,)` input is treated as `B = 1`.
      t: `(T,)` strictly positive sample times, `T >= 2`. Choose them inside the
        power-law window: small enough that many modes contribute, large enough
        that the spectral gap has not yet flattened `p`.
    Returns:
      `(B,)` estimated spectral dimension (a `(n,)` input returns shape `()`).

    Notes:
      `d_s` is a fitted asymptotic exponent, not exact on a finite spectrum: as
      `t → ∞`, `p → (#zero modes)/n`, so an over-large `t` biases `d_s` toward 0.
      See the module docstring for the window caveat.

    References:
      Rammal-Toulouse (1983); Durhuus-Jonsson-Wheater (2007).
    """
    if t.ndim != 1 or t.shape[0] < 2:
        raise ValueError(
            f"t must be a 1-D tensor of at least 2 sample times, got shape "
            f"{tuple(t.shape)}"
        )
    if eigenvalues.ndim < 1:
        raise ValueError(
            f"eigenvalues must have a mode axis (ndim >= 1), got shape "
            f"{tuple(eigenvalues.shape)}"
        )
    if eigenvalues.shape[-1] == 0:
        raise ValueError("eigenvalues must contain at least one mode")
    if not eigenvalues.is_floating_point() or not t.is_floating_point():
        raise ValueError("eigenvalues and t must have real floating-point dtypes")
    if not torch.isfinite(t).all() or (t <= 0).any():
        raise ValueError("t must be finite and strictly positive")
    if not torch.isfinite(eigenvalues).all():
        raise ValueError("eigenvalues must be finite")

    squeeze = eigenvalues.ndim == 1
    lam = (eigenvalues.unsqueeze(0) if squeeze else eigenvalues).clamp(min=0.0)

    # least-squares slope of log p against log t, per batch element
    log_t = torch.log(t)                                    # (T,)
    lt_centered = log_t - log_t.mean()                      # (T,)
    variance = lt_centered.square().sum()
    if variance == 0:
        raise ValueError("t must have nonzero log-time variance")
    # Keep the sum in log space, since valid late times can underflow
    # every exp(-t * lambda) even though the fitted slope is finite.
    log_p = torch.logsumexp(-lam.unsqueeze(-2) * t.unsqueeze(-1), dim=-1)
    log_p = log_p - math.log(lam.shape[-1])
    lp_centered = log_p - log_p.mean(dim=-1, keepdim=True)  # (B, T)
    slope = (lp_centered * lt_centered).sum(dim=-1) / variance
    d_s = -2.0 * slope

    return d_s.squeeze(0) if squeeze else d_s
