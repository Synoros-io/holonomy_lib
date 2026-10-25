# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Synoros

"""holonomy_lib.info_geometry: information-geometric primitives.

Information geometry studies probability distributions as points on a
Riemannian manifold whose metric is the Fisher information. Three
families of primitives appear here:

  Bregman divergences: `bregman_divergence(p, q, potential)`.
    The asymmetric "distance" induced by a convex potential function.
    Generalizes squared-Euclidean (potential = ½‖·‖²) and Itakura-
    Saito (potential = −log(·)) and Kullback-Leibler (potential =
    Σ p log p, the negative entropy). Banerjee et al. (2005) is the
    canonical reference.

  KL divergences: closed forms for the common exponential
    families:
      `kl_divergence_categorical(p, q)`: discrete distributions.
      `kl_divergence_gaussian(mu_p, Sigma_p, mu_q, Sigma_q)`: multi-
        variate Gaussians.

  Fisher metric: Riemannian inner product on the statistical
    manifold; the second-order Taylor expansion of KL near the
    diagonal. Plus `natural_gradient`, the steepest-descent direction
    under this metric. See Amari (1998).
      `fisher_information_categorical(p)`: diagonal Fisher on the simplex.
      `fisher_information_gaussian_mean(Sigma)`: Fisher for the mean parameter.
      `natural_gradient(grad, fisher_matrix)`: F^{-1} · grad.

All primitives are batched-first, GPU-native, with citations.

References:
  Amari, S.-I. (1998). Natural gradient works efficiently in learning.
    Neural Computation 10(2):251–276.
  Amari, S.-I. (2016). Information Geometry and Its Applications.
    Applied Mathematical Sciences 194, Springer. The textbook.
  Banerjee, A., Merugu, S., Dhillon, I. S., Ghosh, J. (2005).
    Clustering with Bregman divergences. JMLR 6:1705–1749.
  Martens, J. (2020). New insights and perspectives on the natural
    gradient method. JMLR 21(146):1–76.
  Nielsen, F. (2020). An elementary introduction to information
    geometry. Entropy 22(10):1100.
"""

from holonomy_lib.info_geometry.divergences import (
    bregman_divergence,
    kl_divergence_categorical,
    kl_divergence_gaussian,
)
from holonomy_lib.info_geometry.fisher import (
    fisher_information_categorical,
    fisher_information_gaussian_mean,
    natural_gradient,
)

__all__ = [
    "bregman_divergence",
    "fisher_information_categorical",
    "fisher_information_gaussian_mean",
    "kl_divergence_categorical",
    "kl_divergence_gaussian",
    "natural_gradient",
]
