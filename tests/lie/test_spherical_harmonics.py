# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Synoros

"""Tests for holonomy_lib.lie.real_spherical_harmonics.

Cover:
  1. Shape contract: output is (..., (l_max+1)²).
  2. Validation: rejects bad shapes + out-of-range l_max.
  3. Y_{0,0} is the constant 1/(2√π).
  4. Y_{1,m} match the closed forms (linear in the direction components).
  5. Direction normalization works (caller can pass non-unit vectors).
  6. Orthonormality: Monte-Carlo integral over uniform-on-S² samples
     gives `δ_{lm,l'm'}` to ~1/√N.
  7. SO(3) equivariance: rotating the input direction permutes the
     same-l components by an orthogonal real Wigner matrix — concretely,
     the l-block stays in the l-block and its norm is preserved.
"""

from __future__ import annotations

import math

import pytest
import torch

from holonomy_lib.lie import real_spherical_harmonics, so3


def _seeded(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def _uniform_sphere(n: int, seed: int) -> torch.Tensor:
    """N points uniformly on the unit 2-sphere via Gaussian-then-normalize."""
    g = _seeded(seed)
    v = torch.randn(n, 3, generator=g, dtype=torch.float64)
    return v / torch.linalg.norm(v, dim=-1, keepdim=True)


# --------------------------------------------------------------------
# Validation + shape
# --------------------------------------------------------------------


class TestValidationAndShape:
    def test_rejects_bad_input_dim(self):
        with pytest.raises(ValueError, match="dim 3"):
            real_spherical_harmonics(
                torch.zeros(5, 2, dtype=torch.float64), l_max=1,
            )

    def test_rejects_l_max_too_high(self):
        with pytest.raises(ValueError, match="l_max"):
            real_spherical_harmonics(
                torch.zeros(5, 3, dtype=torch.float64), l_max=5,
            )

    def test_rejects_negative_l_max(self):
        with pytest.raises(ValueError, match="l_max"):
            real_spherical_harmonics(
                torch.zeros(5, 3, dtype=torch.float64), l_max=-1,
            )

    @pytest.mark.parametrize("l_max", [0, 1, 2, 3, 4])
    def test_output_shape(self, l_max):
        v = _uniform_sphere(7, seed=0)
        y = real_spherical_harmonics(v, l_max=l_max)
        assert y.shape == (7, (l_max + 1) ** 2)


# --------------------------------------------------------------------
# Closed-form correctness for l = 0, 1
# --------------------------------------------------------------------


class TestClosedForms:
    def test_y00_is_constant(self):
        """Y_{0,0} = 0.5 · sqrt(1/π) at every direction."""
        v = _uniform_sphere(20, seed=1)
        y = real_spherical_harmonics(v, l_max=0)
        expected = 0.5 * math.sqrt(1.0 / math.pi)
        torch.testing.assert_close(
            y[..., 0], torch.full((20,), expected, dtype=torch.float64),
            atol=1e-12, rtol=0,
        )

    def test_l1_matches_yzx_components(self):
        """For unit direction (x, y, z):
          Y_{1,-1} = sqrt(3/(4π)) · y
          Y_{1, 0} = sqrt(3/(4π)) · z
          Y_{1, 1} = sqrt(3/(4π)) · x.
        """
        v = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=torch.float64,
        )
        y = real_spherical_harmonics(v, l_max=1)
        c = math.sqrt(3.0 / (4.0 * math.pi))
        # Indices 1, 2, 3 are Y_{1,-1}, Y_{1,0}, Y_{1,1}.
        # First direction (x=1): Y_{1,1} = c, others 0.
        torch.testing.assert_close(
            y[0, 1:], torch.tensor([0.0, 0.0, c], dtype=torch.float64),
            atol=1e-12, rtol=0,
        )
        # Second (y=1): Y_{1,-1} = c.
        torch.testing.assert_close(
            y[1, 1:], torch.tensor([c, 0.0, 0.0], dtype=torch.float64),
            atol=1e-12, rtol=0,
        )
        # Third (z=1): Y_{1,0} = c.
        torch.testing.assert_close(
            y[2, 1:], torch.tensor([0.0, c, 0.0], dtype=torch.float64),
            atol=1e-12, rtol=0,
        )

    def test_unnormalized_input_gives_same_result(self):
        """Caller does not need to pre-normalize directions."""
        unit = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64)
        scaled = torch.tensor([[0.0, 0.0, 5.0]], dtype=torch.float64)
        y1 = real_spherical_harmonics(unit, l_max=4)
        y2 = real_spherical_harmonics(scaled, l_max=4)
        torch.testing.assert_close(y1, y2, atol=1e-12, rtol=0)


# --------------------------------------------------------------------
# Orthonormality via Monte Carlo over uniform-on-S² samples
# --------------------------------------------------------------------


class TestOrthonormality:
    def test_monte_carlo_orthonormal(self):
        """For r̂ uniform on S² (area = 4π), the random variable
        4π · Y_lm(r̂) · Y_l'm'(r̂) has expectation δ_lm,l'm'. Average
        a large sample and check the matrix is close to identity."""
        n = 80000
        l_max = 3                                       # 16 components
        D = (l_max + 1) ** 2
        r = _uniform_sphere(n, seed=42)
        y = real_spherical_harmonics(r, l_max=l_max)    # (n, D)
        # Empirical Gram: 4π · (y^T y) / n ≈ I_D.
        gram = (4.0 * math.pi) * (y.mT @ y) / n
        I = torch.eye(D, dtype=torch.float64)
        # MC error scales as 1/√n; with n = 80000, expect ~3-4e-3.
        max_err = (gram - I).abs().max().item()
        assert max_err < 0.02, (
            f"empirical Gram should be ~I; max |G − I| = {max_err:.4e}"
        )


# --------------------------------------------------------------------
# SO(3) equivariance: same-l block stays in same-l block
# --------------------------------------------------------------------


class TestRotationEquivariance:
    """For an SO(3) rotation R, Y_lm(R⁻¹ r̂) lies in the span of
    {Y_lm'(r̂) : m' ∈ {-l, ..., l}}. We verify the weaker but easily-
    checked consequence: the l²-norm of the l-block is preserved
    under rotation of the input direction. The full Wigner-D
    equivariance (mixing matrix is orthogonal) is a v0.2 test once
    Wigner-D matrices ship."""

    @pytest.mark.parametrize("l_max", [1, 2, 3, 4])
    def test_l_block_norm_invariant_under_rotation(self, l_max):
        n = 50
        r = _uniform_sphere(n, seed=11)
        R = so3.random_so3(batch_size=n, generator=_seeded(13))
        # Rotate each direction by its own random rotation.
        # r' = R @ r where r is treated as a column vector.
        r_rot = torch.einsum("nij,nj->ni", R, r)
        y0 = real_spherical_harmonics(r, l_max=l_max)
        y1 = real_spherical_harmonics(r_rot, l_max=l_max)
        # For each l, slice the (2l+1) components and compare l²-norms.
        offset = 0
        for l in range(l_max + 1):
            width = 2 * l + 1
            block0 = y0[..., offset:offset + width]
            block1 = y1[..., offset:offset + width]
            norm0 = (block0 * block0).sum(dim=-1)
            norm1 = (block1 * block1).sum(dim=-1)
            torch.testing.assert_close(
                norm0, norm1, atol=1e-10, rtol=0,
            )
            offset += width
