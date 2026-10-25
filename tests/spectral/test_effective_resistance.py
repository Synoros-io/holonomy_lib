# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Synoros

"""Tests for holonomy_lib.spectral.effective_resistance + commute_time.

Layers:
  1. Validation (non-square rejected).
  2. Identity properties (R(u, u) = 0; R symmetric).
  3. Closed forms on K_n (R = 2/n) and P_n (R(0, n-1) = n - 1).
  4. Commute-time identity `C = vol(A) · R`.
"""

from __future__ import annotations

import pytest
import torch

from holonomy_lib.spectral.effective_resistance import (
    commute_time,
    effective_resistance,
)


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
    def test_rejects_non_square(self):
        with pytest.raises(ValueError, match="must be"):
            effective_resistance(torch.zeros(1, 4, 5))


# --------------------------------------------------------------------
# Identity properties
# --------------------------------------------------------------------


class TestIdentities:
    def test_self_resistance_zero(self):
        A = _complete_graph(5)
        R = effective_resistance(A)
        diag = torch.diagonal(R, dim1=-2, dim2=-1)
        torch.testing.assert_close(
            diag, torch.zeros_like(diag), atol=1e-10, rtol=0,
        )

    def test_symmetric(self):
        A = _path_graph(6)
        R = effective_resistance(A)
        torch.testing.assert_close(R, R.mT, atol=1e-10, rtol=0)

    def test_non_negative(self):
        """R(u, v) ≥ 0 always (it's a metric distance)."""
        A = _complete_graph(7)
        R = effective_resistance(A)
        assert (R >= -1e-10).all()


# --------------------------------------------------------------------
# Closed-form sanity
# --------------------------------------------------------------------


class TestClosedForms:
    def test_complete_graph_edge_resistance(self):
        """On K_n (unweighted), R(u, v) = 2/n for any pair u ≠ v.

        Derivation: the symmetry of K_n forces R to be constant across
        all pairs. By the sum rule, Σ_{u, v} R(u, v) = 2(n-1)·trace(L⁺),
        and a direct computation of L⁺ on K_n gives the 2/n value.
        Reference: Klein-Randić (1993), §3.
        """
        for n in [3, 4, 5, 6, 7]:
            A = _complete_graph(n)
            R = effective_resistance(A)
            i, j = torch.triu_indices(n, n, offset=1)
            off_diag = R[0, i, j]
            expected = 2.0 / n
            torch.testing.assert_close(
                off_diag,
                torch.full_like(off_diag, expected),
                atol=1e-10, rtol=0,
            ), f"K_{n}: expected R = 2/n = {expected}, got {off_diag}"

    def test_path_graph_endpoint_resistance(self):
        """On P_n (unweighted path 0—1—...—(n−1)), the resistance
        between the two endpoints is n − 1: each unit-resistance edge
        adds 1 in series. Series-resistance identity (Doyle-Snell 1984
        §1.3).
        """
        for n in [3, 4, 5, 6]:
            A = _path_graph(n)
            R = effective_resistance(A)
            r_ends = R[0, 0, n - 1].item()
            assert r_ends == pytest.approx(float(n - 1), abs=1e-9), (
                f"P_{n}: expected R(0, {n-1}) = {n-1}, got {r_ends}"
            )

    def test_two_node_unit_edge(self):
        """Simplest case: a single edge of unit weight. R(0, 1) = 1."""
        A = torch.tensor([[[0.0, 1.0], [1.0, 0.0]]], dtype=torch.float64)
        R = effective_resistance(A)
        assert R[0, 0, 1].item() == pytest.approx(1.0, abs=1e-12)


# --------------------------------------------------------------------
# Commute-time identity
# --------------------------------------------------------------------


class TestCommuteTime:
    def test_commute_time_equals_vol_times_resistance(self):
        """C(u, v) = vol(A) · R(u, v) exactly (Chandra et al. 1996,
        Theorem 2.1)."""
        A = _complete_graph(5)
        R = effective_resistance(A)
        C = commute_time(A)
        vol = A.sum(dim=(-2, -1))
        expected = vol.unsqueeze(-1).unsqueeze(-1) * R
        torch.testing.assert_close(C, expected, atol=1e-12, rtol=0)

    def test_self_commute_zero(self):
        A = _complete_graph(5)
        C = commute_time(A)
        diag = torch.diagonal(C, dim1=-2, dim2=-1)
        torch.testing.assert_close(diag, torch.zeros_like(diag), atol=1e-10, rtol=0)


# --------------------------------------------------------------------
# Shapes / batching
# --------------------------------------------------------------------


@pytest.mark.parametrize("batch", [1, 2, 4])
class TestShapes:
    def test_resistance_shape(self, batch):
        A = _complete_graph(5, batch=batch)
        R = effective_resistance(A)
        assert R.shape == (batch, 5, 5)

    def test_commute_time_shape(self, batch):
        A = _complete_graph(5, batch=batch)
        C = commute_time(A)
        assert C.shape == (batch, 5, 5)


class TestEdgeCases:
    def test_batch_zero(self):
        """B=0 must work — project convention is `B ∈ {0, 1, >1}`."""
        A = torch.zeros(0, 4, 4, dtype=torch.float64)
        R = effective_resistance(A)
        assert R.shape == (0, 4, 4)
        C = commute_time(A)
        assert C.shape == (0, 4, 4)

    def test_n_one(self):
        """Single-node "graph": R(0, 0) = 0 trivially."""
        A = torch.zeros(1, 1, 1, dtype=torch.float64)
        R = effective_resistance(A)
        assert R.shape == (1, 1, 1)
        assert R[0, 0, 0].item() == pytest.approx(0.0, abs=1e-12)

    def test_all_zeros_graph(self):
        """An empty adjacency (no edges anywhere) is a valid degenerate
        input. R must be all zeros: every node is isolated, the
        Laplacian is identically zero, the pseudoinverse is zero, and
        the formula R = L⁺[u,u] + L⁺[v,v] − 2 L⁺[u,v] evaluates to 0."""
        A = torch.zeros(1, 5, 5, dtype=torch.float64)
        R = effective_resistance(A)
        torch.testing.assert_close(R, torch.zeros_like(R), atol=1e-12, rtol=0)
