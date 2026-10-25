# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Synoros

"""Tests for holonomy_lib.discrete_geometry.forman.

Three layers:
  1. Closed-form properties on simple graphs (paths, cycles, K_n).
  2. Qualitative agreement with Ollivier-Ricci on the bridge graph
     (bridge edge << intra-clique edges for both curvatures).
  3. Edge cases: B=0, disconnected, isolated nodes, asymmetric input.
"""

from __future__ import annotations

import pytest
import torch

from holonomy_lib.discrete_geometry import (
    forman_ricci_augmented,
    forman_ricci_simple,
    ollivier_ricci_curvature,
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


def _cycle_graph(n: int, batch: int = 1, dtype=torch.float64) -> torch.Tensor:
    A = _path_graph(n, batch=batch, dtype=dtype)
    A[:, 0, n - 1] = 1.0
    A[:, n - 1, 0] = 1.0
    return A


# --------------------------------------------------------------------
# Input validation
# --------------------------------------------------------------------


class TestValidation:
    def test_rejects_non_square(self):
        with pytest.raises(ValueError, match="must be"):
            forman_ricci_simple(torch.zeros(1, 4, 5))


# --------------------------------------------------------------------
# Closed-form properties on simple graphs
# --------------------------------------------------------------------


class TestClosedForms:
    """Sreejith et al. (2016), eq. 1 reduces on unweighted simple graphs to
    `κ_F(u, v) = 4 - deg(u) - deg(v)`. The augmented form adds
    3·#triangles_through(u,v).
    """


    def test_path_terminal_edge(self):
        """Edge (0, 1) in P_4: deg(0) = 1, deg(1) = 2 → κ_F = 4 - 3 = 1."""
        A = _path_graph(4)
        kappa = forman_ricci_simple(A)
        assert kappa[0, 0, 1].item() == pytest.approx(1.0, abs=1e-12)
        assert kappa[0, 1, 2].item() == pytest.approx(0.0, abs=1e-12)
        # Sreejith's weighted path formula gives 2-sqrt(1/4) and
        # 2-sqrt(4/1), so ignoring the supplied weights cannot pass.
        weighted = torch.tensor(
            [[[0., 1., 0.], [1., 0., 4.], [0., 4., 0.]]],
            dtype=torch.float64,
        )
        values = forman_ricci_simple(weighted)[0, [0, 1], [1, 2]]
        torch.testing.assert_close(
            values, torch.tensor([1.5, 0.], dtype=values.dtype), atol=1e-12, rtol=0,
        )

    def test_cycle_edges_zero(self):
        """Every edge in C_n has deg(u) = deg(v) = 2 → κ_F = 0."""
        for n in [4, 5, 6, 8]:
            A = _cycle_graph(n)
            kappa = forman_ricci_simple(A)
            for i in range(n):
                j = (i + 1) % n
                assert kappa[0, i, j].item() == pytest.approx(0.0, abs=1e-12), (
                    f"C_{n}: edge ({i},{j}) should have κ_F=0, "
                    f"got {kappa[0, i, j].item()}"
                )

    def test_complete_graph_uniform_curvature(self):
        """On K_n every edge has deg(u) = deg(v) = n-1 → κ_F = 4 - 2(n-1) = 6 - 2n."""
        for n in [3, 4, 5, 6]:
            A = _complete_graph(n)
            kappa = forman_ricci_simple(A)
            torch.testing.assert_close(kappa, kappa.mT, atol=1e-12, rtol=0)
            expected = 6 - 2 * n
            i, j = torch.triu_indices(n, n, offset=1)
            edge_kappa = kappa[0, i, j]
            torch.testing.assert_close(
                edge_kappa,
                torch.full_like(edge_kappa, float(expected)),
                atol=1e-12, rtol=0,
            )

    def test_diagonal_is_zero(self):
        """Forman per-edge; self-loops contribute nothing."""
        A = _complete_graph(5)
        kappa = forman_ricci_simple(A)
        diag = torch.diagonal(kappa, dim1=-2, dim2=-1)
        torch.testing.assert_close(diag, torch.zeros_like(diag), atol=0, rtol=0)

    def test_non_edges_are_zero(self):
        """κ_F is meaningful per-edge; non-edges are zeroed."""
        A = _path_graph(5)  # 0—1—2—3—4
        kappa = forman_ricci_simple(A)
        # (0, 2) is not an edge → kappa = 0
        assert kappa[0, 0, 2].item() == 0.0
        # (0, 4) is not an edge → kappa = 0
        assert kappa[0, 0, 4].item() == 0.0


# --------------------------------------------------------------------
# Augmented form: simple + 3·#triangles
# --------------------------------------------------------------------


class TestAugmented:
    def test_path_no_triangles_matches_simple(self):
        """P_n has no triangles → augmented = simple."""
        A = _path_graph(6)
        torch.testing.assert_close(
            forman_ricci_augmented(A),
            forman_ricci_simple(A),
            atol=1e-12, rtol=0,
        )

    def test_complete_graph_augmented_formula(self):
        """On K_n, every edge has (n - 2) common neighbors (triangles).
        κ_F^aug = (6 - 2n) + 3(n - 2) = n.
        """
        for n in [3, 4, 5, 6, 7]:
            A = _complete_graph(n)
            kappa = forman_ricci_augmented(A)
            i, j = torch.triu_indices(n, n, offset=1)
            edge_kappa = kappa[0, i, j]
            expected = float(n)
            torch.testing.assert_close(
                edge_kappa,
                torch.full_like(edge_kappa, expected),
                atol=1e-12, rtol=0,
            ), f"K_{n}: augmented κ should equal n={n}, got {edge_kappa}"
        # The weighted formula requires face weights; the unweighted
        # +3-per-triangle expression is not a weighted extension.
        with pytest.raises(ValueError, match="binary.*face-weight"):
            forman_ricci_augmented(2 * _complete_graph(3))


# --------------------------------------------------------------------
# Qualitative agreement with Ollivier on the bridge graph
# --------------------------------------------------------------------


class TestAgainstOllivier:
    """Both Ollivier-Ricci and Forman-Ricci should make the bridge
    between two cliques much more negative than the intra-clique edges.
    This is the qualitative property that justifies Forman as a cheap
    substitute for Ollivier on large graphs.
    """

    def test_bridge_more_negative_than_intra_clique(self):
        # Two K_4 cliques (nodes 0-3 and 4-7), one bridge edge 3-4.
        n = 8
        A = torch.zeros(1, n, n, dtype=torch.float64)
        for i in range(4):
            for j in range(4):
                if i != j:
                    A[0, i, j] = 1.0
        for i in range(4, 8):
            for j in range(4, 8):
                if i != j:
                    A[0, i, j] = 1.0
        A[0, 3, 4] = A[0, 4, 3] = 1.0

        kappa_F = forman_ricci_augmented(A)
        kappa_O = ollivier_ricci_curvature(A, alpha=0.0, reg=0.005, n_iter=200)

        # Bridge vs intra-clique for Forman
        bridge_F = kappa_F[0, 3, 4].item()
        intra_F = kappa_F[0, 0, 1].item()
        # Bridge vs intra-clique for Ollivier
        bridge_O = kappa_O[0, 3, 4].item()
        intra_O = kappa_O[0, 0, 1].item()

        # Both must rank the bridge below the intra-clique edge.
        assert bridge_F < intra_F, (
            f"Forman: bridge ({bridge_F:.3f}) should be < "
            f"intra-clique ({intra_F:.3f})"
        )
        assert bridge_O < intra_O, (
            f"Ollivier: bridge ({bridge_O:.3f}) should be < "
            f"intra-clique ({intra_O:.3f})"
        )


# --------------------------------------------------------------------
# Shapes / batching
# --------------------------------------------------------------------


# --------------------------------------------------------------------
# Edge cases
# --------------------------------------------------------------------


class TestEdgeCases:
    def test_isolated_node_contributes_zero(self):
        """An isolated node (no edges) participates in no curvature
        entries; all of its row/column should be zero."""
        n = 5
        A = torch.zeros(1, n, n, dtype=torch.float64)
        # Triangle on 0-1-2, node 3 and 4 isolated
        for i in range(3):
            for j in range(3):
                if i != j:
                    A[0, i, j] = 1.0
        kappa = forman_ricci_simple(A)
        # Rows 3, 4 are all zero
        assert kappa[0, 3, :].abs().max().item() == 0.0
        assert kappa[0, 4, :].abs().max().item() == 0.0

    def test_disconnected_components(self):
        """On disconnected components Forman is computed independently
        per component; no cross-component non-edges activated."""
        n = 6
        A = torch.zeros(1, n, n, dtype=torch.float64)
        # Two disjoint triangles
        for i in range(3):
            for j in range(3):
                if i != j:
                    A[0, i, j] = 1.0
        for i in range(3, 6):
            for j in range(3, 6):
                if i != j:
                    A[0, i, j] = 1.0
        kappa = forman_ricci_simple(A)
        # Each triangle edge: K_3 has κ_F_simple = 6 - 2·3 = 0
        assert kappa[0, 0, 1].item() == pytest.approx(0.0, abs=1e-12)
        assert kappa[0, 3, 4].item() == pytest.approx(0.0, abs=1e-12)
        # Cross-component non-edges are zero
        assert kappa[0, 0, 3].item() == 0.0

    def test_batch_zero(self):
        A = torch.zeros(0, 4, 4, dtype=torch.float64)
        for curvature in (forman_ricci_simple, forman_ricci_augmented):
            kappa = curvature(A)
            assert kappa.shape == (0, 4, 4)
