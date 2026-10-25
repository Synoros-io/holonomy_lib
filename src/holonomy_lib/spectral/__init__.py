# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Synoros

"""holonomy_lib.spectral: graph Laplacians and spectral utilities.

Currently implemented:
  laplacian.combinatorial(A)          : L = D - A
  laplacian.symmetric_normalized(A)   : L_sym = I - D^{-1/2} A D^{-1/2}
  laplacian.random_walk(A)            : L_rw = I - D^{-1} A
  laplacian.signed(A)                 : L^sigma = D^{|sigma|} - A
                                          (Kunegis 2010)
  laplacian.degree(A)                 : degree vector (signed-aware)
  magnetic.combinatorial(A, q)        : Hermitian Laplacian on directed
                                          graphs (Furutani 2020, Fanuel 2017)
  magnetic.symmetric_normalized(A, q) : normalized magnetic Laplacian,
                                          spectrum in [0, 2]
  magnetic.sign_magnetic_combinatorial(A, q)
                                      : signed-directed Hermitian
                                          Laplacian; unifies Kunegis
                                          signed and magnetic
                                          (He et al. 2022, MSGNN)
  magnetic.sign_magnetic_symmetric_normalized(A, q)
                                      : normalized signed-directed form
  laplacian_eigenmaps(A, k, ...)      : bottom-k spectral embedding
  spectral_dimension(eigenvalues, t)  : d_s = -2 d log p / d log t from the
                                          heat-kernel return decay
                                          (Rammal-Toulouse 1983)
"""

from holonomy_lib.spectral import laplacian, magnetic
from holonomy_lib.spectral.diffusion_map import diffusion_map
from holonomy_lib.spectral.effective_resistance import (
    commute_time,
    effective_resistance,
)
from holonomy_lib.spectral.embedding import laplacian_eigenmaps
from holonomy_lib.spectral.heat_kernel import heat_kernel_chebyshev
from holonomy_lib.spectral.spectral_dimension import spectral_dimension

__all__ = [
    "commute_time",
    "diffusion_map",
    "effective_resistance",
    "heat_kernel_chebyshev",
    "laplacian",
    "laplacian_eigenmaps",
    "magnetic",
    "spectral_dimension",
]
