# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Synoros

"""Real spherical harmonics Y_lm on the unit 2-sphere, closed form
for l ≤ 4.

Real spherical harmonics are the orthonormal basis for square-
integrable functions on `S² ⊂ R³` that diagonalizes the Laplace-
Beltrami operator on the sphere. Each `Y_lm` is a homogeneous-degree-l
polynomial in `(x, y, z)` restricted to `‖r‖ = 1`; there are `2l + 1`
of them at each `l`, indexed by `m ∈ {−l, …, l}`. The total at
truncation `l_max` is `(l_max + 1)²` components.

In equivariant deep learning these are the natural basis for
SO(3)-equivariant features on the sphere (or any function of a
direction vector). They are the steerable filters of e3nn and of
SE(3)-Transformer; their tensor products via Clebsch-Gordan
coefficients give the building blocks of SO(3)-equivariant
networks.

This module returns the orthonormal Condon-Shortley-free convention
(positive prefactors, no `(−1)^m` phase). Inputs are direction
vectors `(..., 3)`; the function normalizes to unit norm internally
so callers can pass unnormalized vectors.

References:
  Wikipedia, "Real spherical harmonics" — the explicit formulas
    transcribed here are the standard tesseral/sectoral basis.
  Edmonds (1957), §2.5.
  Geiger, M., Smidt, T. (2022). e3nn: Euclidean neural networks.
    arXiv:2207.09453. The modern implementation in geometric
    deep learning; conventions match.
"""

from __future__ import annotations

import math

import torch

from holonomy_lib.provenance import with_provenance

_SUPPORTED_L_MAX: int = 4


@with_provenance(
    "holonomy_lib.lie.real_spherical_harmonics", op_version="0.1",
)
def real_spherical_harmonics(
    directions: torch.Tensor, l_max: int = 2,
) -> torch.Tensor:
    """Real spherical harmonics `Y_lm` at the given direction vectors.

    For each input direction `(x, y, z)` (normalized internally),
    returns the `(l_max + 1)²` real spherical harmonics in the
    standard ordering:

        [Y_{0,0},
         Y_{1,-1}, Y_{1,0}, Y_{1,1},
         Y_{2,-2}, Y_{2,-1}, Y_{2,0}, Y_{2,1}, Y_{2,2},
         …,
         Y_{l_max, -l_max}, …, Y_{l_max, l_max}]

    Args:
      directions: `(..., 3)` direction vectors. Need not be unit; the
        function normalizes internally.
      l_max: highest degree to compute. Must be in `[0, 4]` for v1;
        higher degrees require a recurrence relation that is not
        implemented yet.

    Returns:
      `(..., (l_max + 1)²)` real spherical harmonic values.

    References:
      Wikipedia, "Table of spherical harmonics" — real form.
      Edmonds (1957), §2.5.
    """
    if directions.shape[-1] != 3:
        raise ValueError(
            f"directions must end with dim 3; got shape="
            f"{tuple(directions.shape)}"
        )
    if not (0 <= l_max <= _SUPPORTED_L_MAX):
        raise ValueError(
            f"l_max must be in [0, {_SUPPORTED_L_MAX}]; got l_max={l_max}. "
            f"Higher degrees need the associated-Legendre recurrence "
            f"(planned for v0.2)."
        )

    floor = torch.finfo(directions.dtype).tiny
    norm = torch.linalg.norm(directions, dim=-1, keepdim=True).clamp(min=floor)
    unit = directions / norm
    x, y, z = unit[..., 0], unit[..., 1], unit[..., 2]

    components: list[torch.Tensor] = []

    # ----- l = 0 -----
    # Y_{0,0} = 0.5 · sqrt(1/π)
    y00 = torch.full_like(x, 0.5 * math.sqrt(1.0 / math.pi))
    components.append(y00)

    if l_max < 1:
        return torch.stack(components, dim=-1)

    # ----- l = 1 -----
    # Y_{1,m} = sqrt(3/(4π)) · {y, z, x} for m = -1, 0, 1.
    c1 = math.sqrt(3.0 / (4.0 * math.pi))
    components.extend([c1 * y, c1 * z, c1 * x])

    if l_max < 2:
        return torch.stack(components, dim=-1)

    # ----- l = 2 -----
    # Y_{2,-2} = 0.5 · sqrt(15/π) · xy
    # Y_{2,-1} = 0.5 · sqrt(15/π) · yz
    # Y_{2,0}  = 0.25 · sqrt(5/π) · (3z² − 1)
    # Y_{2,1}  = 0.5 · sqrt(15/π) · xz
    # Y_{2,2}  = 0.25 · sqrt(15/π) · (x² − y²)
    c2a = 0.5 * math.sqrt(15.0 / math.pi)
    c20 = 0.25 * math.sqrt(5.0 / math.pi)
    c22 = 0.25 * math.sqrt(15.0 / math.pi)
    components.extend([
        c2a * x * y,
        c2a * y * z,
        c20 * (3.0 * z * z - 1.0),
        c2a * x * z,
        c22 * (x * x - y * y),
    ])

    if l_max < 3:
        return torch.stack(components, dim=-1)

    # ----- l = 3 -----
    # Y_{3,-3} = 0.25 · sqrt(35/(2π)) · y(3x² − y²)
    # Y_{3,-2} = 0.5  · sqrt(105/π) · xyz
    # Y_{3,-1} = 0.25 · sqrt(21/(2π)) · y(5z² − 1)
    # Y_{3,0}  = 0.25 · sqrt(7/π) · (5z³ − 3z)
    # Y_{3,1}  = 0.25 · sqrt(21/(2π)) · x(5z² − 1)
    # Y_{3,2}  = 0.25 · sqrt(105/π) · z(x² − y²)
    # Y_{3,3}  = 0.25 · sqrt(35/(2π)) · x(x² − 3y²)
    c3_a = 0.25 * math.sqrt(35.0 / (2.0 * math.pi))
    c3_b = 0.5 * math.sqrt(105.0 / math.pi)
    c3_c = 0.25 * math.sqrt(21.0 / (2.0 * math.pi))
    c3_0 = 0.25 * math.sqrt(7.0 / math.pi)
    c3_d = 0.25 * math.sqrt(105.0 / math.pi)
    components.extend([
        c3_a * y * (3.0 * x * x - y * y),
        c3_b * x * y * z,
        c3_c * y * (5.0 * z * z - 1.0),
        c3_0 * (5.0 * z ** 3 - 3.0 * z),
        c3_c * x * (5.0 * z * z - 1.0),
        c3_d * z * (x * x - y * y),
        c3_a * x * (x * x - 3.0 * y * y),
    ])

    if l_max < 4:
        return torch.stack(components, dim=-1)

    # ----- l = 4 -----
    # Y_{4,-4} = 0.75 · sqrt(35/π) · xy(x² − y²)
    # Y_{4,-3} = 0.75 · sqrt(35/(2π)) · yz(3x² − y²)
    # Y_{4,-2} = 0.75 · sqrt(5/π) · xy(7z² − 1)
    # Y_{4,-1} = 0.75 · sqrt(5/(2π)) · yz(7z² − 3)
    # Y_{4,0}  = 3/16 · sqrt(1/π) · (35z⁴ − 30z² + 3)
    # Y_{4,1}  = 0.75 · sqrt(5/(2π)) · xz(7z² − 3)
    # Y_{4,2}  = 0.375 · sqrt(5/π) · (x² − y²)(7z² − 1)
    # Y_{4,3}  = 0.75 · sqrt(35/(2π)) · xz(x² − 3y²)
    # Y_{4,4}  = 3/16 · sqrt(35/π) · (x⁴ − 6x²y² + y⁴)
    c4_a = 0.75 * math.sqrt(35.0 / math.pi)
    c4_b = 0.75 * math.sqrt(35.0 / (2.0 * math.pi))
    c4_c = 0.75 * math.sqrt(5.0 / math.pi)
    c4_d = 0.75 * math.sqrt(5.0 / (2.0 * math.pi))
    c4_0 = (3.0 / 16.0) * math.sqrt(1.0 / math.pi)
    c4_e = 0.375 * math.sqrt(5.0 / math.pi)
    c4_f = (3.0 / 16.0) * math.sqrt(35.0 / math.pi)
    components.extend([
        c4_a * x * y * (x * x - y * y),
        c4_b * y * z * (3.0 * x * x - y * y),
        c4_c * x * y * (7.0 * z * z - 1.0),
        c4_d * y * z * (7.0 * z * z - 3.0),
        c4_0 * (35.0 * z ** 4 - 30.0 * z * z + 3.0),
        c4_d * x * z * (7.0 * z * z - 3.0),
        c4_e * (x * x - y * y) * (7.0 * z * z - 1.0),
        c4_b * x * z * (x * x - 3.0 * y * y),
        c4_f * (x ** 4 - 6.0 * x * x * y * y + y ** 4),
    ])
    return torch.stack(components, dim=-1)
