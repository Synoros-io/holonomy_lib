# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Synoros

"""SO(3) rotation-group primitives: batched, GPU-native.

The 3D rotation group SO(3) has a 3-dim Lie algebra so(3) ≅ R³ via the
hat map  ω ↦ [ω]_×, where [ω]_× is the 3×3 skew-symmetric matrix

    [[ 0,    −ω_z,  ω_y],
     [ ω_z,  0,    −ω_x],
     [−ω_y,  ω_x,   0  ]].

The Lie-exp map `so(3) → SO(3)` is the matrix exponential, which has
the closed-form Rodrigues formula:

    exp([ω]_×) = I + (sin θ / θ) [ω]_× + ((1 − cos θ) / θ²) [ω]_×²,
    θ = ‖ω‖.

Its inverse (the Lie log) is the matrix log on SO(3), well-defined
locally smooth below angle π. At π the axis has an unavoidable sign
ambiguity; see `so3_log` for the selected representative.

References:
  Hall (2015), §3.1: SO(3) and the Rodrigues formula.
  Shoemake (1992): uniform-random rotations via quaternions.
  Murray, Li, Sastry (1994). A Mathematical Introduction to Robotic
    Manipulation, §2.3: the hat/vee maps and the SO(3) exp/log
    closed forms used here.
"""

from __future__ import annotations

import math

import torch

# Mathematical dimension of SO(3): the group lives in R^{3×3} and its
# Lie algebra is R^3. **Scale of validity**: hard-coded by the
# definition of "3D rotation"; not tunable. Cataloged as `SO3_DIM`.
SO3_DIM: int = 3

# Quaternion dimension. **Scale of validity**: hard-coded by the
# definition of "unit quaternion"; not tunable. Cataloged as
# `QUATERNION_DIM`.
QUATERNION_DIM: int = 4

# The near-pi crossover was calibrated in float64 against Rodrigues round
# trips; the dtype floor below widens it for lower precision inputs.
# Cataloged as SO3_LOG_NEAR_PI_RAD; this is an angle in radians.
SO3_LOG_NEAR_PI_RAD: float = 1e-7

# arcsin(s) / s = 1 + s² / 3! + O(s⁴), where s = sin(theta).
# The small-angle neighborhood is set from dtype epsilon so the omitted
# term stays below roundoff, rather than introducing an angular tolerance.
SO3_LOG_SERIES_SECOND_ORDER: float = 1.0 / math.factorial(SO3_DIM)

# Taylor coefficients of sin(theta)/theta and (1-cos(theta))/theta².
# They are reciprocal factorials from Rodrigues' formula, not fitted values.
SO3_EXP_SINC_LINEAR: float = 1.0 / 6.0
SO3_EXP_SINC_QUADRATIC: float = 1.0 / 120.0
SO3_EXP_COSC_LINEAR: float = 1.0 / 24.0
SO3_EXP_COSC_QUADRATIC: float = 1.0 / 720.0


def axis_angle_to_matrix(
    axis: torch.Tensor, angle: torch.Tensor,
) -> torch.Tensor:
    """Rodrigues formula: convert (unit axis, angle) → rotation matrix.

        R = I + sin(θ)·K + (1 − cos(θ))·K²,   K = [axis]_×.

    The axis is normalized to unit length internally so callers do
    not need to pre-normalize.

    Args:
      axis:  `(..., 3)` axis vectors; will be normalized.
      angle: `(...,)` rotation angle in radians.

    Returns:
      `(..., 3, 3)` rotation matrices in SO(3).

    References:
      Hall (2015), Theorem 3.3.
    """
    if axis.ndim < 1 or axis.shape[-1] != SO3_DIM:
        raise ValueError(
            f"axis must end with dim {SO3_DIM}; got axis.shape={tuple(axis.shape)}"
        )
    if axis.shape[:-1] != angle.shape:
        raise ValueError(
            f"axis batch shape {tuple(axis.shape[:-1])} must match "
            f"angle shape {tuple(angle.shape)}"
        )
    floor = torch.finfo(axis.dtype).tiny
    norm = torch.linalg.norm(axis, dim=-1, keepdim=True).clamp(min=floor)
    axis_unit = axis / norm                                # (..., 3)

    K = _hat(axis_unit)                                     # (..., 3, 3)
    s = torch.sin(angle).unsqueeze(dim=-1).unsqueeze(dim=-1)  # (..., 1, 1)
    c = torch.cos(angle).unsqueeze(dim=-1).unsqueeze(dim=-1)
    eye = torch.eye(SO3_DIM, dtype=axis.dtype, device=axis.device)
    eye = eye.expand(*axis.shape[:-1], SO3_DIM, SO3_DIM)
    return eye + s * K + (1.0 - c) * (K @ K)


def matrix_to_axis_angle(R: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Inverse Rodrigues: convert SO(3) matrix to (unit axis, angle).

    Uses the Lie logarithm, then normalizes its rotation vector. At the
    identity the angle is zero and the axis is chosen as (1, 0, 0).
    The axis-angle representation is not differentiable at identity.

    Args:
      R: `(..., 3, 3)` rotation matrix.

    Returns:
      `(axis, angle)`:
        axis  `(..., 3)` unit vector.
        angle `(...,)` in `[0, π]`.

    References:
      Murray-Li-Sastry (1994), §2.3.3: the dual-branch logarithm.
    """
    if R.shape[-2:] != (SO3_DIM, SO3_DIM):
        raise ValueError(
            f"R must end with ({SO3_DIM}, {SO3_DIM}); got R.shape={tuple(R.shape)}"
        )
    omega = so3_log(R)
    angle = torch.linalg.vector_norm(omega, dim=-1)
    # An axis has no meaning at identity. Only this axis-angle API makes
    # the canonical choice; the smooth Lie logarithm keeps the skew vector.
    nonzero = angle > 0
    divisor = torch.where(nonzero, angle, torch.ones_like(angle))
    axis = omega / divisor.unsqueeze(-1)
    canonical = torch.zeros_like(axis)
    canonical[..., 0] = 1.0
    return torch.where(nonzero.unsqueeze(-1), axis, canonical), angle


def so3_exp(omega: torch.Tensor) -> torch.Tensor:
    """Matrix exponential on so(3): ω ∈ R³ ↦ R ∈ SO(3).

    Evaluates Rodrigues' formula with analytic Taylor coefficients near
    zero, including the skew-generator derivative at zero angular velocity.

    Args:
      omega: `(..., 3)` Lie algebra elements.

    Returns:
      `(..., 3, 3)` rotation matrices.

    References:
      Hall (2015). Lie Groups, Lie Algebras, and Representations, §3.1.
      Murray, Li, Sastry (1994). A Mathematical Introduction to Robotic
        Manipulation, equation 2.14.
    """
    if omega.ndim < 1 or omega.shape[-1] != SO3_DIM:
        raise ValueError(
            f"omega must end with dim {SO3_DIM}; got omega.shape={tuple(omega.shape)}"
        )
    theta_squared = omega.square().sum(dim=-1)
    small = theta_squared <= math.sqrt(torch.finfo(omega.dtype).eps)
    # Mask before sqrt and division so inactive branches have finite
    # derivatives too. The polynomial uses theta² directly at the origin.
    safe_squared = torch.where(small, torch.ones_like(theta_squared), theta_squared)
    theta = torch.sqrt(safe_squared)
    series_squared = torch.where(small, theta_squared, torch.zeros_like(theta_squared))
    sinc = torch.where(
        small,
        1.0 - SO3_EXP_SINC_LINEAR * series_squared
        + SO3_EXP_SINC_QUADRATIC * series_squared.square(),
        torch.sin(theta) / theta,
    )
    cosc = torch.where(
        small,
        0.5 - SO3_EXP_COSC_LINEAR * series_squared
        + SO3_EXP_COSC_QUADRATIC * series_squared.square(),
        (1.0 - torch.cos(theta)) / safe_squared,
    )
    generator = _hat(omega)
    eye = torch.eye(SO3_DIM, dtype=omega.dtype, device=omega.device)
    return (
        eye + sinc[..., None, None] * generator
        + cosc[..., None, None] * (generator @ generator)
    )


def so3_log(R: torch.Tensor) -> torch.Tensor:
    """Principal rotation vector of an SO(3) matrix, with norm in [0, π].

    Away from π, the skew part gives sin(theta) times the rotation axis.
    A series for theta / sin(theta) preserves its direction at small angles.
    Near π the symmetric part determines the axis; its sign at exactly π
    is intrinsically ambiguous, so the logarithm is not smooth there.

    Args:
      R: `(..., 3, 3)` rotation matrices.

    Returns:
      `(..., 3)` rotation vectors.

    References:
      Murray, Li, Sastry (1994). A Mathematical Introduction to Robotic
        Manipulation, §2.3.3, the logarithm of a rotation matrix.
    """
    if R.shape[-2:] != (SO3_DIM, SO3_DIM):
        raise ValueError(
            f"R must end with ({SO3_DIM}, {SO3_DIM}); got R.shape={tuple(R.shape)}"
        )
    skew_vector = _vee(0.5 * (R - R.mT))
    sin_angle = torch.linalg.vector_norm(skew_vector, dim=-1)
    sin_squared = (skew_vector * skew_vector).sum(dim=-1)
    cos_angle = ((R.diagonal(dim1=-2, dim2=-1).sum(dim=-1) - 1.0) * 0.5)
    angle = torch.atan2(sin_angle, cos_angle)
    small = (cos_angle > 0) & (sin_squared < math.sqrt(torch.finfo(R.dtype).eps))
    near_pi = angle > math.pi - max(
        SO3_LOG_NEAR_PI_RAD, math.sqrt(torch.finfo(R.dtype).eps),
    )
    # Neither inactive quotient divides by a vanishing sine. This matters
    # in backward too: where masks a value, not an undefined derivative.
    safe_sine = torch.where(small | near_pi, torch.ones_like(sin_angle), sin_angle)
    scale = torch.where(
        small, 1.0 + SO3_LOG_SERIES_SECOND_ORDER * sin_squared, angle / safe_sine,
    )
    omega = scale.unsqueeze(-1) * skew_vector

    # Sym(R) - cos(theta) I = (1 - cos(theta)) axis axis^T. Its largest
    # diagonal selects a row with a nonzero axis component even at π.
    eye = torch.eye(SO3_DIM, dtype=R.dtype, device=R.device)
    symmetric = 0.5 * (R + R.mT) - cos_angle[..., None, None] * eye
    symmetric = torch.where(near_pi[..., None, None], symmetric, eye)
    pivot = symmetric.diagonal(dim1=-2, dim2=-1).argmax(dim=-1)
    row = symmetric.gather(
        -2, pivot[..., None, None].expand(*R.shape[:-2], 1, SO3_DIM),
    ).squeeze(-2)
    axis = row / torch.linalg.vector_norm(row, dim=-1, keepdim=True)
    alignment = (axis * skew_vector).sum(dim=-1, keepdim=True)
    axis = torch.where(alignment < 0, -axis, axis)
    return torch.where(near_pi.unsqueeze(-1), angle.unsqueeze(-1) * axis, omega)


def compose(R1: torch.Tensor, R2: torch.Tensor) -> torch.Tensor:
    """SO(3) group product: `R1 · R2`. Just `matmul`, exposed for API
    consistency with the rest of the module.

    Args:
      R1, R2: broadcastable `(..., 3, 3)` rotation matrices.

    Returns:
      `(..., 3, 3)` rotation matrices, applying R2 before R1.

    References:
      Hall (2015). Lie Groups, Lie Algebras, and Representations, §1.2.
    """
    if R1.shape[-2:] != (SO3_DIM, SO3_DIM) or R2.shape[-2:] != (SO3_DIM, SO3_DIM):
        raise ValueError(
            f"both inputs must end with ({SO3_DIM}, {SO3_DIM}); got "
            f"R1.shape={tuple(R1.shape)}, R2.shape={tuple(R2.shape)}"
        )
    return R1 @ R2


def random_so3(
    batch_size: int,
    generator: torch.Generator | None = None,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Haar-uniform random rotation matrices.

    Construction: sample a unit quaternion uniformly on S³ (the
    Shoemake 1992 method), then convert to a rotation matrix. The
    quaternion-Haar measure pushes through to the SO(3)-Haar measure
    under the standard 2-to-1 cover.

    Args:
      batch_size: number of independent rotations.
      generator: torch.Generator for reproducibility.
      device, dtype: tensor placement.

    Returns:
      `(batch_size, 3, 3)` SO(3) matrices.

    References:
      Shoemake (1992).
    """
    if batch_size < 0:
        raise ValueError(f"batch_size must be >= 0, got {batch_size}")
    # Shoemake: three uniform [0, 1] samples u1, u2, u3 → unit
    # quaternion (w, x, y, z) with q on S³ uniformly. The sampling
    # and trig run on `device` so a `device="cuda"` call doesn't
    # silently fall back to CPU for the per-sample scalar work. The
    # generator must be on the matching device: if the caller wants
    # GPU sampling, they must pass a CUDA generator.
    target_device = torch.device(device)
    if generator is not None and generator.device != target_device:
        raise ValueError(
            f"generator is on {generator.device} but device={target_device} "
            f"was requested; pass a generator on the target device or omit "
            f"`generator` to use the default"
        )
    u = torch.rand(
        batch_size, SO3_DIM,
        generator=generator, dtype=dtype, device=target_device,
    )
    u1, u2, u3 = u[..., 0], u[..., 1], u[..., 2]
    sqrt_u1 = torch.sqrt(u1)
    sqrt_1_minus_u1 = torch.sqrt(1.0 - u1)
    two_pi_u2 = 2.0 * math.pi * u2
    two_pi_u3 = 2.0 * math.pi * u3
    w = sqrt_1_minus_u1 * torch.sin(two_pi_u2)
    x = sqrt_1_minus_u1 * torch.cos(two_pi_u2)
    y = sqrt_u1 * torch.sin(two_pi_u3)
    z = sqrt_u1 * torch.cos(two_pi_u3)
    q = torch.stack([w, x, y, z], dim=-1)                   # (B, 4)
    return _quaternion_to_matrix(q)


# ============================================================
# Internal helpers
# ============================================================


def _hat(v: torch.Tensor) -> torch.Tensor:
    """Hat map: R³ → so(3). v ↦ [v]_× as a (..., 3, 3) skew-symmetric."""
    zero = torch.zeros_like(v[..., 0])
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    row0 = torch.stack([zero, -z, y], dim=-1)
    row1 = torch.stack([z, zero, -x], dim=-1)
    row2 = torch.stack([-y, x, zero], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def _vee(K: torch.Tensor) -> torch.Tensor:
    """Vee map: skew(3) → R³. The inverse of `_hat`."""
    return torch.stack(
        [K[..., 2, 1], K[..., 0, 2], K[..., 1, 0]], dim=-1,
    )


def _quaternion_to_matrix(q: torch.Tensor) -> torch.Tensor:
    """Convert unit quaternions `(w, x, y, z)` to rotation matrices.

    Standard formula; works for arbitrary batch shape ending in 4.
    """
    # Quaternion unpacking: (w, x, y, z). The literal 3 is the index
    # of the last quaternion component (QUATERNION_DIM - 1); the
    # algorithm requires accessing every component by name.
    last = QUATERNION_DIM - 1
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., last]
    # Off-diagonal pairs reuse the same products: factor for clarity.
    wx, wy, wz = w * x, w * y, w * z
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    row0 = torch.stack([1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)], dim=-1)
    row1 = torch.stack([2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)], dim=-1)
    row2 = torch.stack([2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)
