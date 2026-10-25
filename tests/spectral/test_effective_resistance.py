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

        The pseudoinverse is L⁺ = (I - 11ᵀ/n)/n, so the voltage drop
        for a unit current between any two distinct nodes is 2/n.
        Reference: Klein-Randić (1993), §3.
        """
        for dtype in (torch.float32, torch.float64):
            for n in [3, 4, 5, 6, 7]:
                weights = torch.tensor([1.0, 4.0, 0.5], dtype=dtype)
                A = _complete_graph(n, batch=3, dtype=dtype) * weights[:, None, None]
                R = effective_resistance(A)
                i, j = torch.triu_indices(n, n, offset=1)
                expected = (2.0 / (n * weights))[:, None].expand(-1, i.numel())
                torch.testing.assert_close(R[:, i, j], expected)
                for batch_index in range(A.shape[0]):
                    singleton = A[batch_index:batch_index + 1]
                    torch.testing.assert_close(
                        R[batch_index], effective_resistance(singleton)[0],
                    )

    def test_path_graph_endpoint_resistance(self):
        """On P_n (unweighted path 0,1,...,(n−1)), the resistance
        between the two endpoints is n − 1: each unit-resistance edge
        adds 1 in series. Series-resistance identity (Doyle-Snell 1984
        §1.3).
        """
        for n in [2, 3, 4, 5, 6]:
            A = _path_graph(n)
            R = effective_resistance(A)
            r_ends = R[0, 0, n - 1].item()
            assert r_ends == pytest.approx(float(n - 1), abs=1e-9), (
                f"P_{n}: expected R(0, {n-1}) = {n-1}, got {r_ends}"
            )


# --------------------------------------------------------------------
# Commute-time identity
# --------------------------------------------------------------------


class TestCommuteTime:

    def test_self_commute_zero(self):
        A = _complete_graph(5)
        C = commute_time(A)
        diag = torch.diagonal(C, dim1=-2, dim2=-1)
        torch.testing.assert_close(diag, torch.zeros_like(diag), atol=1e-10, rtol=0)
        # On K_n the expected round trip between distinct nodes is 2(n - 1).
        off_diagonal = ~torch.eye(5, dtype=torch.bool)
        torch.testing.assert_close(C[0, off_diagonal], torch.full((20,), 8.0).double())
        distinct = torch.cat((_complete_graph(5), _path_graph(5), 4 * _path_graph(5)))
        batched = commute_time(distinct)
        assert batched[1, 0, 4] == pytest.approx(32.0)
        torch.testing.assert_close(batched[1], batched[2])
        for batch_index in range(distinct.shape[0]):
            singleton = distinct[batch_index:batch_index + 1]
            torch.testing.assert_close(
                batched[batch_index], commute_time(singleton)[0],
            )
        edge = _path_graph(2)
        looped = edge + 10 * torch.eye(2, dtype=edge.dtype)
        assert commute_time(looped)[0, 0, 1] == pytest.approx(2.0)


# --------------------------------------------------------------------
# Shapes / batching
# --------------------------------------------------------------------


class TestEdgeCases:
    def test_batch_zero(self):
        """B=0 preserves empty shapes, including zero-node graphs."""
        A = torch.zeros(0, 4, 4, dtype=torch.float64)
        R = effective_resistance(A)
        assert R.shape == (0, 4, 4)
        C = commute_time(A)
        assert C.shape == (0, 4, 4)
        for function in (effective_resistance, commute_time):
            empty = torch.empty(1, 0, 0, dtype=torch.float64)
            assert function(empty).shape == (1, 0, 0)


    def test_all_zeros_graph(self):
        """No path means infinite resistance and commute time between distinct nodes."""
        for n in (1, 5):
            A = torch.zeros(1, n, n, dtype=torch.float64)
            expected = torch.full_like(A, float("inf"))
            expected.diagonal(dim1=-2, dim2=-1).zero_()
            torch.testing.assert_close(effective_resistance(A), expected)
            torch.testing.assert_close(commute_time(A), expected)
        disjoint_edges = torch.block_diag(_path_graph(2)[0], _path_graph(2)[0])[None]
        resistance = effective_resistance(disjoint_edges)
        commute = commute_time(disjoint_edges)
        assert resistance[0, 0, 1] == pytest.approx(1.0)
        assert commute[0, 0, 1] == pytest.approx(2.0)
        assert torch.isinf(resistance[0, 0, 2])
        assert torch.isinf(commute[0, 0, 2])
