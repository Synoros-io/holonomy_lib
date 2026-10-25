# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Synoros

"""holonomy_lib.lie — Lie group primitives for SO(3) and the 2-sphere.

The Lie group `SO(3)` is the group of orientation-preserving
rotations of R³; its Lie algebra `so(3)` is the 3-dim space of
skew-symmetric matrices (equivalently, angular-velocity vectors).
This module ships the standard exp/log/random/compose primitives
plus real spherical harmonics, which are the natural basis for any
SO(3)-equivariant function on the unit sphere.

Currently exposed:
  so3.axis_angle_to_matrix(axis, angle)  — Rodrigues formula.
  so3.matrix_to_axis_angle(R)            — inverse log map.
  so3.so3_exp(omega)                     — matrix exp on so(3).
  so3.so3_log(R)                         — matrix log on SO(3).
  so3.random_so3(batch_size, generator)  — Haar-uniform via
                                            quaternion sampling.
  so3.compose(R1, R2)                    — group product (matmul).
  real_spherical_harmonics(directions, l_max)
                                          — closed-form Y_lm for
                                            l ≤ 4 evaluated at
                                            (B, 3) unit direction
                                            vectors.

Planned (v0.2 follow-ups):
  Wigner-D matrices (real basis) for higher-l rotation actions on
    spherical-harmonic features.
  Clebsch-Gordan coefficients for SO(3) tensor products (l₁ ⊗ l₂ → L).
  SE(3), SU(2), SL(n) primitives.

References:
  Hall, B. C. (2015). Lie Groups, Lie Algebras, and Representations,
    2nd ed. Springer GTM 222. The canonical reference for matrix Lie
    groups; §3.1 covers SO(3) and its Lie algebra.
  Edmonds, A. R. (1957). Angular Momentum in Quantum Mechanics.
    Princeton University Press. Real and complex spherical harmonics
    + Wigner D-matrices.
  Shoemake, K. (1992). Uniform random rotations. In Graphics Gems
    III, pp. 124–132. Academic Press. The quaternion-from-3-uniforms
    construction used by `random_so3`.
  Cohen, T. S., Geiger, M., Köhler, J., Welling, M. (2018).
    Spherical CNNs. ICLR. Modern application of these primitives in
    SO(3)-equivariant deep learning.
"""

from holonomy_lib.lie import so3
from holonomy_lib.lie.spherical_harmonics import real_spherical_harmonics

__all__ = [
    "real_spherical_harmonics",
    "so3",
]
