# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Synoros

"""Tests for holonomy_lib.lie.so3.

Cover:
  1. Validation (shape mismatches).
  2. Rodrigues formula correctness:
     - R^T R = identity, det R = +1.
     - Angle 0 → identity.
     - Angle π around the canonical axes → known matrices.
  3. axis_angle ↔ matrix round-trip (mod sign ambiguity at θ=π).
  4. so3_exp / so3_log round-trip.
  5. random_so3:
     - Output is in SO(3).
     - Mean bias and squared-entry moment against Haar expectations
       (the trivial-rep projection on SO(3) under Haar is zero
       for the standard 3D representation).
"""

from __future__ import annotations

import math

import pytest
import torch

from holonomy_lib.lie import so3


def _seeded(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


# --------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------


class TestValidation:
    def test_axis_angle_rejects_bad_axis_dim(self):
        with pytest.raises(ValueError, match="dim 3"):
            so3.axis_angle_to_matrix(
                torch.zeros(5, 4, dtype=torch.float64),
                torch.zeros(5, dtype=torch.float64),
            )


    def test_matrix_to_axis_angle_rejects_bad_shape(self):
        with pytest.raises(ValueError, match="\\(3, 3\\)"):
            so3.matrix_to_axis_angle(torch.zeros(2, 3, dtype=torch.float64))

    def test_compose_rejects_bad_shape(self):
        with pytest.raises(ValueError, match="\\(3, 3\\)"):
            so3.compose(
                torch.zeros(3, 3, dtype=torch.float64),
                torch.zeros(2, 2, dtype=torch.float64),
            )

    def test_random_so3_rejects_negative_batch(self):
        with pytest.raises(ValueError, match="batch_size"):
            so3.random_so3(batch_size=-1)


# --------------------------------------------------------------------
# Rodrigues formula correctness
# --------------------------------------------------------------------


class TestRodriguesFormula:
    def test_output_is_in_SO3(self):
        """Random axis-angle → R^T R = identity, det R = +1."""
        g = _seeded(0)
        axis = torch.randn(8, 3, dtype=torch.float64, generator=g)
        angle = torch.rand(8, dtype=torch.float64, generator=g) * math.pi
        R = so3.axis_angle_to_matrix(axis, angle)
        identity = torch.eye(3, dtype=torch.float64).expand(8, 3, 3)
        torch.testing.assert_close(R @ R.mT, identity, atol=1e-12, rtol=0)
        det = torch.linalg.det(R)
        torch.testing.assert_close(
            det, torch.ones(8, dtype=torch.float64), atol=1e-12, rtol=0,
        )



    @pytest.mark.parametrize(
        "axis,angle,expected",
        [
            ([1., 2., 3.], 0., [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]),
            ([0., 0., 1.], math.pi, [[-1., 0., 0.], [0., -1., 0.], [0., 0., 1.]]),
            ([1., 0., 0.], math.pi / 2., [[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]]),
        ],
    )
    def test_pi_over_2_around_x_axis(self, axis, angle, expected):
        """Known zero, half-turn, and quarter-turn Rodrigues values."""
        rotation = so3.axis_angle_to_matrix(
            torch.tensor([axis], dtype=torch.float64),
            torch.tensor([angle], dtype=torch.float64),
        )
        torch.testing.assert_close(
            rotation[0], torch.tensor(expected, dtype=torch.float64),
            atol=1e-12, rtol=0,
        )

    def test_unnormalized_axis_works(self):
        """Caller can pass non-unit axis; rotation is the same."""
        unit = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64)
        scaled = torch.tensor([[0.0, 0.0, 5.0]], dtype=torch.float64)
        angle = torch.tensor([math.pi / 3.0], dtype=torch.float64)
        R_unit = so3.axis_angle_to_matrix(unit, angle)
        R_scaled = so3.axis_angle_to_matrix(scaled, angle)
        torch.testing.assert_close(R_unit, R_scaled, atol=1e-12, rtol=0)


# --------------------------------------------------------------------
# axis_angle ↔ matrix round-trip
# --------------------------------------------------------------------


class TestRoundTrip:
    def test_axis_angle_roundtrip_general(self):
        """Random axis/angle → matrix → axis/angle recovers the same
        rotation (axis may be sign-flipped together with angle)."""
        g = _seeded(2)
        n = 32
        axis = torch.randn(n, 3, dtype=torch.float64, generator=g)
        axis = axis / torch.linalg.norm(axis, dim=-1, keepdim=True)
        # Angles strictly inside (0, π) to avoid edge cases.
        angle = (
            torch.rand(n, dtype=torch.float64, generator=g) * (math.pi - 0.1)
            + 0.05
        )
        R = so3.axis_angle_to_matrix(axis, angle)
        axis_back, angle_back = so3.matrix_to_axis_angle(R)
        # Reconstruct R from the recovered axis/angle; it must match.
        R_back = so3.axis_angle_to_matrix(axis_back, angle_back)
        torch.testing.assert_close(R, R_back, atol=1e-10, rtol=0)

    def test_so3_exp_log_roundtrip(self):
        """log(exp(ω)) = ω for ‖ω‖ < π."""
        g = _seeded(3)
        # Random ω with magnitude < π (the injectivity radius of so(3)).
        omega = torch.randn(16, 3, dtype=torch.float64, generator=g)
        omega_norm = torch.linalg.norm(omega, dim=-1, keepdim=True)
        scale = torch.clamp(omega_norm, max=math.pi - 0.1) / omega_norm
        omega = omega * scale
        R = so3.so3_exp(omega)
        omega_back = so3.so3_log(R)
        torch.testing.assert_close(omega, omega_back, atol=1e-9, rtol=0)
        small = torch.tensor([[0., 1e-7, 0.]], dtype=torch.float64)
        torch.testing.assert_close(
            so3.so3_log(so3.so3_exp(small)), small, atol=1e-20, rtol=0,
        )
        axis, angle = so3.matrix_to_axis_angle(so3.so3_exp(small))
        torch.testing.assert_close(axis, small / 1e-7, atol=1e-12, rtol=0)
        torch.testing.assert_close(angle, small[:, 1], atol=1e-20, rtol=0)
        for batch in (0, 1):
            values = small.expand(batch, -1)
            recovered = so3.so3_log(so3.so3_exp(values))
            assert recovered.shape == (batch, 3)
            torch.testing.assert_close(recovered, values, atol=1e-20, rtol=0)

    def test_so3_exp_zero_is_identity(self):
        omega = torch.zeros(3, 3, dtype=torch.float64)
        R = so3.so3_exp(omega)
        identity = torch.eye(3, dtype=torch.float64).expand(3, 3, 3)
        torch.testing.assert_close(R, identity, atol=1e-12, rtol=0)
        zero = omega[:1].requires_grad_()
        assert torch.autograd.gradcheck(so3.so3_exp, (zero,))
        assert torch.autograd.gradcheck(
            lambda w: so3.so3_log(so3.so3_exp(w)), (zero,),
        )
        identity = identity[:1].clone().requires_grad_()
        log_gradient, = torch.autograd.grad(so3.so3_log(identity).sum(), identity)
        expected_gradient = torch.tensor(
            [[[0., -0.5, 0.5], [0.5, 0., -0.5], [-0.5, 0.5, 0.]]],
            dtype=torch.float64,
        )
        torch.testing.assert_close(log_gradient, expected_gradient, atol=1e-12, rtol=0)
        for values in ([[1e-7, 2e-7, -1e-7]], [[0.3, -0.2, 0.4]]):
            tangent = torch.tensor(values, dtype=torch.float64, requires_grad=True)
            assert torch.autograd.gradcheck(so3.so3_exp, (tangent,))

        # Large float32 angles still describe rotations; scaling and squaring
        # of a general matrix exponential accumulates error in this regime.
        large = torch.tensor([[1000., 0., 0.], [1e6, 0., 0.]])
        rotations = so3.so3_exp(large)
        reference = so3.axis_angle_to_matrix(large, large.norm(dim=-1))
        torch.testing.assert_close(rotations, reference)
        torch.testing.assert_close(
            rotations.mT @ rotations, torch.eye(3).expand(2, 3, 3),
        )
        torch.testing.assert_close(torch.linalg.det(rotations), torch.ones(2))


# --------------------------------------------------------------------
# random_so3
# --------------------------------------------------------------------


class TestRandomSO3:
    def test_output_is_in_SO3(self):
        n = 128
        R = so3.random_so3(batch_size=n, generator=_seeded(0))
        identity = torch.eye(3, dtype=torch.float64).expand(n, 3, 3)
        torch.testing.assert_close(R @ R.mT, identity, atol=1e-12, rtol=0)
        det = torch.linalg.det(R)
        torch.testing.assert_close(
            det, torch.ones(n, dtype=torch.float64), atol=1e-12, rtol=0,
        )

    @pytest.mark.parametrize("seed", [7, 11])
    def test_haar_uniform_mean_approaches_zero(self, seed):
        """Check mean bias and the Haar second moment, not full uniformity.

        Haar entries have mean zero and second moment 1/3. A cycle of
        diagonal rotations has zero mean too, but fails the second moment.
        """
        n = 10000
        rotations = so3.random_so3(batch_size=n, generator=_seeded(seed))
        mean_rotation = rotations.mean(dim=0)
        fro = torch.linalg.matrix_norm(mean_rotation, ord="fro").item()
        # chi-squared(9) upper quantile gives a conservative mean-bias check.
        chi2_p1e_6 = 41.0
        threshold = (chi2_p1e_6 / (3 * n)) ** 0.5
        assert fro < threshold
        assert mean_rotation.abs().max().item() < 0.04
        # Var(R_ij²) = 1/5 - 1/9; 0.03 is ten standard errors at this n.
        second_moment = rotations.square().mean(dim=0)
        torch.testing.assert_close(
            second_moment, torch.full_like(second_moment, 1. / 3.),
            atol=0.03, rtol=0,
        )

    def test_deterministic_with_seed(self):
        R1 = so3.random_so3(batch_size=4, generator=_seeded(42))
        R2 = so3.random_so3(batch_size=4, generator=_seeded(42))
        torch.testing.assert_close(R1, R2, atol=1e-12, rtol=0)

    def test_zero_batch_size(self):
        R = so3.random_so3(batch_size=0, generator=_seeded(0))
        assert R.shape == (0, 3, 3)


        # The mismatch raises only when devices actually differ, which
        # we can't construct without a CUDA build. The validation logic
        # is exercised by the device-equality check on entry.


# --------------------------------------------------------------------
# matrix_to_axis_angle near θ = π : the numerical danger zone where
# `sin(θ) → 0` makes the general branch blow up. The eigenvector
# branch must take over BEFORE precision degrades.
# --------------------------------------------------------------------


class TestNearPiAxisRecovery:
    """Round-trip accuracy in the awkward angular range `θ → π`. The
    general branch divides by `sin(θ)`, amplifying noise; the near-π
    branch recovers axis from `(R+identity)/2 = u u^T`, exact at θ = π but
    with `O(gap²)` model error away from there. The threshold is
    calibrated for float64 to use the better branch at each gap; see
    the empirical table in `so3.SO3_LOG_NEAR_PI_RAD`.

    Per-gap tolerances reflect the actual achievable accuracy of the
    selected branch : generous enough to pass at all gaps, tight
    enough that an incorrectly-routed branch (e.g. near-π used where
    general should be) would fail."""

    @pytest.mark.parametrize(
        "gap,atol",
        [
            (1e-2, 1e-12),    # general branch
            (1e-3, 1e-12),    # general branch
            (1e-5, 1e-9),     # general branch (close to crossover)
            (1e-7, 1e-7),     # crossover region
            (1e-9, 1e-8),     # near-π branch
        ],
    )
    def test_axis_recovery_in_danger_zone(self, gap, atol):
        """For each `θ = π − gap`, round-trip recovery reproduces the
        rotation matrix to the calibrated tolerance for that gap."""
        axis = torch.tensor(
            [[0.6, -0.4, 0.7]], dtype=torch.float64,
        )
        axis = axis / torch.linalg.norm(axis, dim=-1, keepdim=True)
        angle = torch.tensor([math.pi - gap], dtype=torch.float64)
        R = so3.axis_angle_to_matrix(axis, angle)
        axis_back, angle_back = so3.matrix_to_axis_angle(R)
        R_back = so3.axis_angle_to_matrix(axis_back, angle_back)
        torch.testing.assert_close(R, R_back, atol=atol, rtol=0)

    def test_pi_around_canonical_axis(self):
        """Exact-π rotation around the z-axis. R = diag(-1, -1, 1).
        Round-trip must recover the axis up to sign and the angle = π."""
        z_axis = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64)
        angle = torch.tensor([math.pi], dtype=torch.float64)
        R = so3.axis_angle_to_matrix(z_axis, angle)
        axis_back, angle_back = so3.matrix_to_axis_angle(R)
        # Axis can be ±z. Check colinearity.
        cos_align = (axis_back * z_axis).sum(dim=-1).abs()
        torch.testing.assert_close(
            cos_align, torch.ones(1, dtype=torch.float64), atol=1e-9, rtol=0,
        )
        torch.testing.assert_close(
            angle_back, angle, atol=1e-9, rtol=0,
        )


# --------------------------------------------------------------------
# Group law
# --------------------------------------------------------------------


class TestCompose:
    def test_identity_left_and_right(self):
        n = 4
        R = so3.random_so3(batch_size=n, generator=_seeded(0))
        identity = torch.eye(3, dtype=torch.float64).expand(n, 3, 3)
        torch.testing.assert_close(so3.compose(identity, R), R, atol=1e-12, rtol=0)
        torch.testing.assert_close(so3.compose(R, identity), R, atol=1e-12, rtol=0)
        # Quarter-turns around different axes do not commute. Rx Ry sends
        # e_x to e_y; the reversed multiplication sends it to -e_z.
        quarter_turns = so3.so3_exp(torch.tensor(
            [[math.pi / 2., 0., 0.], [0., math.pi / 2., 0.]], dtype=torch.float64,
        ))
        product = so3.compose(quarter_turns[:1], quarter_turns[1:])
        expected = torch.tensor(
            [[[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]]], dtype=torch.float64,
        )
        torch.testing.assert_close(product, expected, atol=1e-12, rtol=0)

    def test_inverse_via_transpose(self):
        """For R ∈ SO(3), R · R^T = identity."""
        n = 4
        R = so3.random_so3(batch_size=n, generator=_seeded(1))
        identity = torch.eye(3, dtype=torch.float64).expand(n, 3, 3)
        torch.testing.assert_close(
            so3.compose(R, R.mT), identity, atol=1e-12, rtol=0,
        )
