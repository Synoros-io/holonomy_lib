# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Synoros

"""SO(3) rotation-group primitives — batched, GPU-native.

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
for any R ∉ {I rotated by π through a non-canonical axis} — see
`so3_log` for handling the θ = π edge case.

References:
  Hall (2015), §3.1 — SO(3) and the Rodrigues formula.
  Shoemake (1992) — uniform-random rotations via quaternions.
  Murray, Li, Sastry (1994). A Mathematical Introduction to Robotic
    Manipulation, §2.3 — the hat/vee maps and the SO(3) exp/log
    closed forms used here.
"""

from __future__ import annotations

import math

import torch

from holonomy_lib.provenance import with_provenance

# Mathematical dimension of SO(3): the group lives in R^{3×3} and its
# Lie algebra is R^3. **Scale of validity**: hard-coded by the
# definition of "3D rotation"; not tunable. Cataloged as `SO3_DIM`.
SO3_DIM: int = 3

# Quaternion dimension. **Scale of validity**: hard-coded by the
# definition of "unit quaternion"; not tunable. Cataloged as
# `QUATERNION_DIM`.
QUATERNION_DIM: int = 4

# Angle threshold (in radians) below which `matrix_to_axis_angle`
# switches to the "near-zero rotation" branch and returns a canonical
# axis (1, 0, 0). The default sin(θ) / sin(θ_clamped) division
# becomes ill-conditioned within ~sqrt(eps) of zero; `1e-6` is far
# above that and well below any rotation that meaningfully transports
# information. **Scale of validity**: dimensionless angle in radians;
# does not depend on input size. Cataloged as `SO3_LOG_NEAR_ZERO_RAD`.
SO3_LOG_NEAR_ZERO_RAD: float = 1e-6

# Angle threshold (in radians, measured below π) above which
# `matrix_to_axis_angle` switches to the "near-π rotation" branch
# that recovers the axis from the symmetric `(R + I)/2` rather than
# from the antisymmetric `vee((R − R^T)/2)`.
#
# Empirical comparison (float64, axis=[0.6,-0.4,0.7]/‖·‖) of the
# two branches' round-trip max-error per gap = π - θ:
#
#     gap        general err     near-pi err
#     1e-2       1.1e-14         2.6e-5
#     1e-4       1.5e-12         2.6e-9
#     1e-6       1.1e-10         2.6e-13
#     1e-7       2.0e-9          1.6e-9
#     1e-9       7.0e-10         7.0e-10
#     1e-12      8.0e-5          7.0e-13
#
# The general branch wins by 4-9 orders of magnitude in the wide
# range gap ∈ [1e-6, 1e-2]; the near-π branch wins only at
# gap < 1e-7 where `arccos(trace - 1) / 2` itself becomes the
# precision bottleneck for the general branch. `1e-7` is the
# crossover for float64. **Scale of validity**: dimensionless
# angle in radians, calibrated for float64. Float32 callers may
# want a larger threshold (~1e-3) since their amplification
# tolerance is ~6 orders of magnitude weaker. Cataloged as
# `SO3_LOG_NEAR_PI_RAD`.
SO3_LOG_NEAR_PI_RAD: float = 1e-7


@with_provenance(
    "holonomy_lib.lie.so3.axis_angle_to_matrix", op_version="0.1",
)
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
    if axis.shape[-1] != SO3_DIM:
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


@with_provenance(
    "holonomy_lib.lie.so3.matrix_to_axis_angle", op_version="0.1",
)
def matrix_to_axis_angle(R: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Inverse Rodrigues: convert SO(3) matrix to (unit axis, angle).

    Algorithm (standard, robust against θ near 0 and near π):
      cos(θ) = (trace(R) − 1) / 2;
      for θ small, axis from `vee((R − R^T) / 2) / sin(θ)`;
      for θ near π, recover axis from the symmetric `(R + I)/2` whose
      eigenvector for eigenvalue 1 is exactly the rotation axis.

    Args:
      R: `(..., 3, 3)` rotation matrix.

    Returns:
      `(axis, angle)`:
        axis  `(..., 3)` unit vector.
        angle `(...,)` in `[0, π]`.

    References:
      Murray-Li-Sastry (1994), §2.3.3 — the dual-branch logarithm.
    """
    if R.shape[-2:] != (SO3_DIM, SO3_DIM):
        raise ValueError(
            f"R must end with ({SO3_DIM}, {SO3_DIM}); got R.shape={tuple(R.shape)}"
        )
    trace = R.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    # cos θ ∈ [-1, 1]; clamp against rounding overshoot before arccos.
    cos_t = ((trace - 1.0) * 0.5).clamp(min=-1.0, max=1.0)
    angle = torch.arccos(cos_t)                            # (...,)

    # General-position axis recovery: 2 sin(θ) · axis = vee(R − R^T).
    # For θ near 0 or π, sin(θ) ~ 0 and we use a different formula
    # (eigenvector branch). We blend via `torch.where` to keep the
    # graph differentiable.
    skew = 0.5 * (R - R.mT)                                 # (..., 3, 3)

    # Branch 1: general position. axis = vee(skew) / sin(θ).
    sin_t = torch.sin(angle).unsqueeze(dim=-1)
    floor = torch.finfo(R.dtype).tiny
    axis_general = _vee(skew) / sin_t.clamp(min=floor)

    # Branch 2: θ near π → R is symmetric, (R + I) / 2 = u uᵀ where u
    # is the axis. Extract u from the diagonal: u_i = ± sqrt((1 + R_ii) / 2).
    # Sign: pick u_i > 0 for the largest diagonal entry, then fix the
    # other two from R_ij = u_i u_j (when u_i ≠ 0).
    near_pi = angle > (math.pi - SO3_LOG_NEAR_PI_RAD)
    eye = torch.eye(SO3_DIM, dtype=R.dtype, device=R.device).expand_as(R)
    sym = 0.5 * (R + eye)                                   # (..., 3, 3)
    diag = sym.diagonal(dim1=-2, dim2=-1).clamp(min=0.0)
    axis_near_pi = torch.sqrt(diag)
    # Argmax along the last dim of the diagonal, per batch element.
    i_max = diag.argmax(dim=-1)                              # (...,)
    # Use sym[..., i_max, j] to recover signs of axis_near_pi[..., j].
    # Gather row sym[..., i_max, :] for each batch element.
    i_max_expand = i_max.unsqueeze(dim=-1).unsqueeze(dim=-1).expand(
        *R.shape[:-2], 1, SO3_DIM,
    )
    row_at_imax = torch.gather(sym, dim=-2, index=i_max_expand).squeeze(dim=-2)
    # axis_near_pi[..., i_max] is positive by construction; assign signs
    # via sign(row_at_imax) so all entries are consistent.
    signs = torch.sign(row_at_imax).clamp(min=-1.0, max=1.0)
    # Where signs == 0 (entries that happen to be exactly 0), pick +.
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    axis_near_pi = axis_near_pi * signs

    # Branch 3: θ near zero — axis is ill-defined; pick (1, 0, 0)
    # canonically. The angle is ~0 so any axis gives R ≈ I.
    near_zero = angle < SO3_LOG_NEAR_ZERO_RAD
    canonical = torch.zeros_like(axis_general)
    canonical[..., 0] = 1.0

    # Select between branches.
    axis = torch.where(
        near_pi.unsqueeze(dim=-1), axis_near_pi, axis_general,
    )
    axis = torch.where(
        near_zero.unsqueeze(dim=-1), canonical, axis,
    )
    # Renormalize against accumulated drift.
    norm = torch.linalg.norm(axis, dim=-1, keepdim=True).clamp(min=floor)
    axis = axis / norm
    return axis, angle


@with_provenance(
    "holonomy_lib.lie.so3.so3_exp", op_version="0.1",
)
def so3_exp(omega: torch.Tensor) -> torch.Tensor:
    """Matrix exponential on so(3): ω ∈ R³ ↦ R ∈ SO(3).

    Internally calls `axis_angle_to_matrix(ω / ‖ω‖, ‖ω‖)`.

    Args:
      omega: `(..., 3)` Lie algebra elements.

    Returns:
      `(..., 3, 3)` rotation matrices.
    """
    if omega.shape[-1] != SO3_DIM:
        raise ValueError(
            f"omega must end with dim {SO3_DIM}; got omega.shape={tuple(omega.shape)}"
        )
    angle = torch.linalg.norm(omega, dim=-1)
    # `axis_angle_to_matrix` normalizes axis internally and uses
    # sin/cos of the angle directly, so a zero ω gives a zero
    # angle whose Rodrigues output is exactly the identity.
    return axis_angle_to_matrix(omega, angle)


@with_provenance(
    "holonomy_lib.lie.so3.so3_log", op_version="0.1",
)
def so3_log(R: torch.Tensor) -> torch.Tensor:
    """Matrix logarithm on SO(3): R ∈ SO(3) ↦ ω ∈ R³.

    Returns the so(3) vector whose hat is `log(R)`, i.e.,
    `ω = axis · angle` for the dual-branch axis/angle recovery in
    `matrix_to_axis_angle`.
    """
    axis, angle = matrix_to_axis_angle(R)
    return axis * angle.unsqueeze(dim=-1)


@with_provenance(
    "holonomy_lib.lie.so3.compose", op_version="0.1",
)
def compose(R1: torch.Tensor, R2: torch.Tensor) -> torch.Tensor:
    """SO(3) group product: `R1 · R2`. Just `matmul`, exposed for API
    consistency with the rest of the module."""
    if R1.shape[-2:] != (SO3_DIM, SO3_DIM) or R2.shape[-2:] != (SO3_DIM, SO3_DIM):
        raise ValueError(
            f"both inputs must end with ({SO3_DIM}, {SO3_DIM}); got "
            f"R1.shape={tuple(R1.shape)}, R2.shape={tuple(R2.shape)}"
        )
    return R1 @ R2


@with_provenance(
    "holonomy_lib.lie.so3.random_so3", op_version="0.1",
)
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
    # generator must be on the matching device — if the caller wants
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
    # Off-diagonal pairs reuse the same products — factor for clarity.
    wx, wy, wz = w * x, w * y, w * z
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    row0 = torch.stack([1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)], dim=-1)
    row1 = torch.stack([2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)], dim=-1)
    row2 = torch.stack([2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)
