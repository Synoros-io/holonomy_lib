# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Synoros

"""Tests for holonomy_lib.spectral.diffusion_map.

Layers:
  1. Integer-time validation and empty-batch handling.
  2. Closed forms on complete graphs and transition powers on paths.
  3. Diffusion-time decay for modes with `|μ_j| ≤ 1`.
"""

from __future__ import annotations

import pytest
import torch

from holonomy_lib.spectral.diffusion_map import diffusion_map


def _complete_graph(n: int, batch: int = 1, dtype=torch.float64) -> torch.Tensor:
    A = torch.ones(batch, n, n, dtype=dtype) - torch.eye(n, dtype=dtype).unsqueeze(0)
    return A


def _path_graph(n: int, batch: int = 1, dtype=torch.float64) -> torch.Tensor:
    A = torch.zeros(batch, n, n, dtype=dtype)
    for i in range(n - 1):
        A[:, i, i + 1] = 1.0
        A[:, i + 1, i] = 1.0
    return A


# --------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------


class TestValidation:
    def test_rejects_negative_t(self):
        A = _complete_graph(5)
        for t in (-1, 0.5, 1.0, float("nan"), True):
            with pytest.raises(ValueError, match="nonnegative integer"):
                diffusion_map(A, k=2, t=t)  # type: ignore[arg-type]


# --------------------------------------------------------------------
# Closed-form sanity
# --------------------------------------------------------------------


class TestClosedForms:
    def test_complete_graph_transition_eigenvalues(self):
        """On K_n the random-walk Laplacian L_rw = I − D⁻¹ A has
        eigenvalue 0 (multiplicity 1) and eigenvalue n/(n−1)
        (multiplicity n − 1). Therefore the P-eigenvalues
        μ_j = 1 − λ_j are 1 (dropped) and −1/(n − 1) (the rest).
        """
        n = 5
        A = _complete_graph(n)
        eigvals, _ = diffusion_map(A, k=n - 1, t=1)
        expected = -1.0 / (n - 1)
        torch.testing.assert_close(
            eigvals[0],
            torch.full_like(eigvals[0], expected),
            atol=1e-9, rtol=0,
        )
        triangle = _complete_graph(3)
        _, coordinates = diffusion_map(triangle, k=2, t=1)
        assert (coordinates[0, 0] - coordinates[0, 1]).square().sum() == (
            pytest.approx(1.5)
        )
        _, scaled = diffusion_map(triangle * 4, k=2, t=1)
        torch.testing.assert_close(scaled, coordinates)
        empty_values, empty_vectors = diffusion_map(triangle[:0], k=2, t=1)
        assert empty_values.shape == (0, 2)
        assert empty_vectors.shape == (0, 3, 2)

    def test_t_zero_returns_raw_eigenvectors(self):
        """Full coordinates recover the transition-row distance weighted by π."""
        A = torch.cat((_path_graph(5), _complete_graph(5)))
        degree = A.sum(-1)
        stationary = degree / degree.sum(-1, keepdim=True)
        transition = A / degree.unsqueeze(-1)
        for t in (0, 1, 2):
            _, coordinates = diffusion_map(A, k=4, t=t)
            walk = torch.linalg.matrix_power(transition, t)
            row_difference = walk.unsqueeze(-2) - walk.unsqueeze(-3)
            expected = (row_difference.square() / stationary[:, None, None]).sum(-1)
            actual = torch.cdist(coordinates, coordinates).square()
            torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)
        eigenvalues, _ = diffusion_map(A[:1], k=1, t=1)
        torch.testing.assert_close(eigenvalues, -torch.ones_like(eigenvalues))

    def test_diffusion_time_shrinks_subdominant_modes(self):
        """Every transition eigenvalue has |μ| ≤ 1, including negative modes."""
        A = _path_graph(6)
        _, emb_t1 = diffusion_map(A, k=2, t=1)
        _, emb_t5 = diffusion_map(A, k=2, t=5)
        # The Frobenius norm of the t=5 embedding must be ≤ t=1's,
        # since each component shrank by |μ_j|^(5-1) ≤ 1.
        assert emb_t5.norm() <= emb_t1.norm() + 1e-9
