# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Synoros

"""Tests for holonomy_lib.discrete_geometry.ricci.ollivier_ricci_curvature.

Closed forms check transport values; distinct batches check support gathering.
Finite budgets must warn when either transport marginal is still infeasible.
"""

from __future__ import annotations

import warnings

import pytest
import torch

from holonomy_lib.discrete_geometry import ollivier_ricci_curvature


def _complete_graph(n: int, batch: int = 1, dtype=torch.float64) -> torch.Tensor:
    """K_n complete graph adjacency, batched."""
    A = torch.ones(batch, n, n, dtype=dtype) - torch.eye(n, dtype=dtype)
    return A


def _path_graph(n: int, batch: int = 1, dtype=torch.float64) -> torch.Tensor:
    """P_n path graph: 0; 1; 2; ...; n−1."""
    A = torch.zeros(batch, n, n, dtype=dtype)
    for i in range(n - 1):
        A[:, i, i + 1] = 1.0
        A[:, i + 1, i] = 1.0
    return A


# --------------------------------------------------------------------
# Input validation
# --------------------------------------------------------------------


class TestValidation:
    def test_rejects_alpha_out_of_range(self):
        with pytest.raises(ValueError, match="alpha"):
            ollivier_ricci_curvature(_complete_graph(3), alpha=-0.1)
        with pytest.raises(ValueError, match="alpha"):
            ollivier_ricci_curvature(_complete_graph(3), alpha=1.5)
        for kwargs in ({"n_iter": 0}, {"tile_size": 0}, {"sync_every": 0}):
            with pytest.raises(ValueError, match="positive integer"):
                ollivier_ricci_curvature(_complete_graph(3), **kwargs)


# --------------------------------------------------------------------
# Shapes
# --------------------------------------------------------------------


# --------------------------------------------------------------------
# Closed-form properties
# --------------------------------------------------------------------


class TestClosedForms:

    def test_complete_graph_K_n_formula(self):
        """On K_n, κ(any edge) = (n−2)/(n−1).

        Derivation: μ_x, μ_y are uniform on each other's neighborhoods,
        differing only by swapping mass at x and y, each of weight 1/(n−1).
        Optimal transport moves 1/(n−1) from y → x at unit cost.
        """
        for n in [3, 4, 5, 6]:
            A = _complete_graph(n)
            kappa = ollivier_ricci_curvature(A)
            torch.testing.assert_close(kappa, kappa.mT, atol=1e-6, rtol=0)
            # At alpha=1 each measure is an exact Dirac mass. Introducing
            # small probabilities at absent support would perturb this.
            lazy = ollivier_ricci_curvature(A, alpha=1.0)
            torch.testing.assert_close(
                lazy, torch.eye(n, dtype=A.dtype).unsqueeze(0), atol=1e-12, rtol=0,
            )
            expected = (n - 2) / (n - 1)
            i, j = torch.triu_indices(n, n, offset=1)
            off_diag = kappa[0, i, j]
            torch.testing.assert_close(
                off_diag, torch.full_like(off_diag, expected),
                atol=1e-8, rtol=0,
            ), f"K_{n}: expected κ={expected}, got {off_diag}"

    def test_path_interior_edge_is_flat(self):
        """On a path P_n, the curvature of an interior edge is ≈ 0
        (paths are "flat"; neither positively nor negatively curved).
        """
        A = _path_graph(7)  # 0—1—2—3—4—5—6; edge (3, 4) is interior
        kappa = ollivier_ricci_curvature(A)
        # Edge (3, 4) is the most-interior, fully symmetric
        assert abs(kappa[0, 3, 4].item()) < 1e-8

    def test_diagonal_is_one(self):
        """κ(x, x) = 1 by convention (W_1(δ_x, δ_x) = 0, d=0 vacuous)."""
        isolated = torch.zeros(1, 4, 4, dtype=torch.float64)
        isolated[:, :3, :3] = _complete_graph(3)
        for A in (_complete_graph(5), isolated, torch.empty(0, 5, 5),
                  torch.empty(1, 0, 0)):
            kappa = ollivier_ricci_curvature(A, alpha=0.0)
            diag = torch.diagonal(kappa, dim1=-2, dim2=-1)
            assert kappa.shape == A.shape
            torch.testing.assert_close(diag, torch.ones_like(diag), atol=0, rtol=0)


    def test_symmetric_weighted_with_generous_budget(self):
        """Weighted transport is feasible, exact on the triangle, and batched."""
        n = 5
        g = torch.Generator()
        g.manual_seed(7)
        U = torch.rand(1, n, n, generator=g, dtype=torch.float64)
        A = torch.triu(U, diagonal=1)
        A = A + A.mT
        kappa = ollivier_ricci_curvature(
            A, alpha=0.0, reg=0.01, n_iter=5000, tol=1e-12,
        )
        torch.testing.assert_close(kappa, kappa.mT, atol=1e-8, rtol=0)
        # Distinct neighborhoods exercise padding and gathered costs. Repeating
        # the same graph would miss a batch mixup or a padded-cost scale leak.
        batch = torch.cat((A, _path_graph(n), _complete_graph(n)))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            batched = ollivier_ricci_curvature(batch)
            singletons = torch.cat([
                ollivier_ricci_curvature(graph.unsqueeze(0)) for graph in batch
            ])
            low_precision = ollivier_ricci_curvature(batch.float())
        torch.testing.assert_close(batched, singletons, atol=1e-10, rtol=0)
        torch.testing.assert_close(
            batched, low_precision.double(), atol=1e-5, rtol=0,
        )

        # Exact transport for this weighted triangle gives kappa(1,2)=1/6.
        # Defaults must solve it; a deliberately exhausted budget must warn.
        triangle = torch.tensor(
            [[[0., 1., 4.], [1., 0., 2.], [4., 2., 0.]]], dtype=torch.float64,
        )
        with pytest.warns(UserWarning, match="marginal error"):
            ollivier_ricci_curvature(triangle, n_iter=1)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            default = ollivier_ricci_curvature(triangle)
        torch.testing.assert_close(
            default[0, [1, 2], [2, 1]],
            torch.full((2,), 1 / 6, dtype=default.dtype), atol=1e-8, rtol=0,
        )
        for cadence in (1, 8):
            result = ollivier_ricci_curvature(
                triangle, n_iter=1000, sync_every=cadence, tol=1e-10,
            )
            torch.testing.assert_close(
                result[0, [1, 2], [2, 1]],
                torch.full((2,), 1 / 6, dtype=result.dtype), atol=1e-8, rtol=0,
            )


# --------------------------------------------------------------------
# Bottleneck / community structure: surgery-relevant properties
# --------------------------------------------------------------------


class TestDisconnectedComponents:
    """Disconnected pairs are zero by the public output convention."""

    def test_within_component_unaffected_by_disconnection(self):
        """A K_3 alongside a disjoint K_3 has κ=1/2 on its edges,
        same as a standalone K_3 (the closed-form result).
        """
        n = 6
        A = torch.zeros(1, n, n, dtype=torch.float64)
        for i in range(3):
            for j in range(3):
                if i != j:
                    A[0, i, j] = 1.0
        for i in range(3, 6):
            for j in range(3, 6):
                if i != j:
                    A[0, i, j] = 1.0
        kappa = ollivier_ricci_curvature(A, alpha=0.0, reg=0.005, n_iter=300)
        assert torch.isfinite(kappa).all()
        torch.testing.assert_close(
            kappa[0, :3, 3:], torch.zeros(3, 3, dtype=kappa.dtype), atol=0, rtol=0,
        )
        torch.testing.assert_close(
            kappa[0, 3:, :3], torch.zeros(3, 3, dtype=kappa.dtype), atol=0, rtol=0,
        )
        within_edge = kappa[0, 0, 1].item()
        assert abs(within_edge - 0.5) < 0.02, (
            f"within-component K_3 edge κ should be 0.5, got {within_edge}"
        )
        low_precision = torch.zeros(1, 4, 4, dtype=torch.float16)
        low_precision[0, 0, 1] = low_precision[0, 1, 0] = 1
        low_precision[0, 2, 3] = low_precision[0, 3, 2] = 1
        # Dirac-to-Dirac transport on two disjoint edges has zero edge
        # curvature. A large disconnected sentinel overflows float16.
        torch.testing.assert_close(
            ollivier_ricci_curvature(low_precision),
            torch.eye(4, dtype=low_precision.dtype).unsqueeze(0), atol=0, rtol=0,
        )


    def test_flow_ignores_cross_component_curvature(self):
        """Ricci flow only updates edges (A > 0). Cross-component pairs
        stay zero through the flow, regardless of their (possibly
        large) curvature value.
        """
        from holonomy_lib.discrete_geometry import discrete_ricci_flow
        n = 6
        A = torch.zeros(1, n, n, dtype=torch.float64)
        for i in range(3):
            for j in range(3):
                if i != j:
                    A[0, i, j] = 1.0
        for i in range(3, 6):
            for j in range(3, 6):
                if i != j:
                    A[0, i, j] = 1.0
        W = discrete_ricci_flow(
            A, n_steps=3, dt=0.5, alpha=0.0, normalize=False,
            reg=0.01, n_sinkhorn_iters=50,
        )
        # Cross-component pairs remain exactly zero.
        cross_block = W[0, :3, 3:]
        torch.testing.assert_close(
            cross_block, torch.zeros_like(cross_block), atol=0, rtol=0,
        )
        # Within-component edges have evolved (non-zero, but ≠ original 1.0).
        assert (W[0, :3, :3].diagonal() == 0).all()


class TestBottleneckCurvature:
    def test_bridge_edge_more_negative_than_intra_cluster(self):
        """For a two-clique graph K_4 ⊔_e K_4 (joined by a single bridge
        edge), the bridge edge has curvature much lower than intra-clique
        edges. This is the signature surgery exploits (Sia 2019, Ni 2019).
        """
        n = 8  # nodes 0..3 in clique 0, 4..7 in clique 1; bridge 3-4
        A = torch.zeros(1, n, n, dtype=torch.float64)
        # Intra-clique 0..3 (K_4)
        for i in range(4):
            for j in range(4):
                if i != j:
                    A[0, i, j] = 1.0
        # Intra-clique 4..7 (K_4)
        for i in range(4, 8):
            for j in range(4, 8):
                if i != j:
                    A[0, i, j] = 1.0
        # Single bridge
        A[0, 3, 4] = A[0, 4, 3] = 1.0
        kappa = ollivier_ricci_curvature(A, alpha=0.0, reg=0.005, n_iter=300)
        # Bridge edge curvature
        bridge = kappa[0, 3, 4].item()
        # An intra-clique edge (say 0–1) curvature
        intra = kappa[0, 0, 1].item()
        # We expect bridge ≪ intra (much more negative)
        assert bridge < intra - 0.3, (
            f"bridge κ={bridge:.3f} should be much less than intra κ={intra:.3f}"
        )


# --------------------------------------------------------------------
# Comparison against GraphRicciCurvature library
# --------------------------------------------------------------------


class TestAgainstGraphRicciCurvature:
    """Cross-check against the canonical CPU NetworkX implementation.

    GraphRicciCurvature uses POT (exact LP-based W_1) under the hood
    while we use entropic Sinkhorn, so we allow ~2% tolerance per edge.
    """

    def test_K4_matches_grc(self):
        nx = pytest.importorskip("networkx")
        OllivierRicci = pytest.importorskip(
            "GraphRicciCurvature.OllivierRicci",
        ).OllivierRicci

        # K_4
        A = _complete_graph(4)
        kappa_ours = ollivier_ricci_curvature(
            A, alpha=0.0, reg=0.005, n_iter=400,
        )

        # Reference — note that `compute_ricci_curvature` stores results
        # on `orc.G`, not the input graph.
        G = nx.complete_graph(4)
        orc = OllivierRicci(G, alpha=0.0, verbose="ERROR")
        orc.compute_ricci_curvature()
        for u, v, data in orc.G.edges(data=True):
            ref = data["ricciCurvature"]
            ours = kappa_ours[0, u, v].item()
            assert abs(ours - ref) < 0.03, (
                f"K_4 edge ({u},{v}): ours={ours:.4f}, ref={ref:.4f}"
            )
