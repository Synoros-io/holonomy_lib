# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Synoros

"""Tests for discrete_ricci_flow + ricci_flow_with_surgery.

These exercise the Perelman-on-networks pipeline: Ollivier-Ricci
curvature drives an edge-weight update; surgery removes edges that
have stretched past a threshold. Test graphs are small for speed
because each Ricci-flow step computes O(n²) Sinkhorn problems.
"""

from __future__ import annotations

import pytest
import torch

from holonomy_lib.discrete_geometry import (
    discrete_ricci_flow,
    ricci_flow_with_surgery,
)


def _two_cliques_with_bridge(
    cluster_size: int = 4, dtype=torch.float64,
) -> torch.Tensor:
    """Two K_{cluster_size} cliques joined by a single bridge edge."""
    n = 2 * cluster_size
    A = torch.zeros(1, n, n, dtype=dtype)
    for i in range(cluster_size):
        for j in range(cluster_size):
            if i != j:
                A[0, i, j] = 1.0
                A[0, i + cluster_size, j + cluster_size] = 1.0
    # Bridge
    A[0, cluster_size - 1, cluster_size] = 1.0
    A[0, cluster_size, cluster_size - 1] = 1.0
    return A


def _complete_graph(n: int, dtype=torch.float64) -> torch.Tensor:
    A = torch.ones(1, n, n, dtype=dtype) - torch.eye(n, dtype=dtype).unsqueeze(dim=0)
    return A


# --------------------------------------------------------------------
# Input validation
# --------------------------------------------------------------------


class TestValidation:
    def test_flow_rejects_negative_steps(self):
        A = _complete_graph(4)
        with pytest.raises(ValueError, match="n_steps"):
            discrete_ricci_flow(A, n_steps=-1)

    def test_flow_rejects_nonpositive_dt(self):
        A = _complete_graph(4)
        with pytest.raises(ValueError, match="dt"):
            discrete_ricci_flow(A, n_steps=1, dt=0.0)

    def test_surgery_rejects_nonpositive_period(self):
        A = _complete_graph(4)
        with pytest.raises(ValueError, match="surgery_period"):
            ricci_flow_with_surgery(A, n_steps=1, surgery_period=0)

    def test_surgery_rejects_nonpositive_threshold(self):
        A = _complete_graph(4)
        with pytest.raises(ValueError, match="surgery_threshold"):
            ricci_flow_with_surgery(A, n_steps=1, surgery_threshold=-1.0)


# --------------------------------------------------------------------
# Flow — basic properties
# --------------------------------------------------------------------


class TestFlow:
    def test_zero_steps_returns_input(self):
        A = _complete_graph(4)
        out = discrete_ricci_flow(A, n_steps=0)
        torch.testing.assert_close(out, A, atol=0, rtol=0)

    def test_flow_preserves_symmetry(self):
        A = _two_cliques_with_bridge(cluster_size=3)
        out = discrete_ricci_flow(A, n_steps=3, dt=0.5)
        torch.testing.assert_close(out, out.mT, atol=1e-9, rtol=0)

    def test_flow_preserves_nonneg_weights(self):
        A = _two_cliques_with_bridge(cluster_size=3)
        out = discrete_ricci_flow(A, n_steps=3, dt=0.5)
        assert (out >= 0).all()

    def test_flow_normalize_keeps_norm_constant(self):
        A = _two_cliques_with_bridge(cluster_size=3)
        n0 = torch.linalg.matrix_norm(A).item()
        out = discrete_ricci_flow(A, n_steps=5, dt=0.3, normalize=True)
        n1 = torch.linalg.matrix_norm(out).item()
        assert abs(n0 - n1) / max(n0, 1e-9) < 1e-6


class TestFlowDynamics:
    def test_bridge_grows_relative_to_clique_under_flow(self):
        """The bottleneck edge has negative curvature ⇒ under the flow
        it elongates (gets larger relative to intra-clique edges).
        """
        A = _two_cliques_with_bridge(cluster_size=4)
        bridge = (3, 4)  # the bridge edge
        intra = (0, 1)   # an intra-clique edge

        out = discrete_ricci_flow(A, n_steps=2, dt=0.5, normalize=True)

        # Compute ratios
        initial_ratio = A[0, bridge[0], bridge[1]] / A[0, intra[0], intra[1]]
        final_ratio = out[0, bridge[0], bridge[1]] / out[0, intra[0], intra[1]]
        assert final_ratio > initial_ratio, (
            f"bridge/intra ratio should grow under flow; "
            f"initial={initial_ratio:.3f}, final={final_ratio:.3f}"
        )


# --------------------------------------------------------------------
# Surgery — the Perelman-spirit primitive
# --------------------------------------------------------------------


class TestSurgery:
    def test_zero_steps_returns_input(self):
        A = _two_cliques_with_bridge(cluster_size=4)
        out = ricci_flow_with_surgery(A, n_steps=0)
        torch.testing.assert_close(out, A, atol=0, rtol=0)

    def test_complete_clique_no_surgery_with_high_threshold(self):
        """On K_n every edge has positive curvature — under flow they
        shrink, not grow. Surgery shouldn't remove anything when
        threshold is well above the initial mean.
        """
        A = _complete_graph(5)
        out = ricci_flow_with_surgery(
            A, n_steps=2, surgery_period=1, surgery_threshold=5.0,
            dt=0.3, normalize=True,
        )
        # Number of nonzero edges preserved
        n_edges_before = (A > 1e-9).sum().item()
        n_edges_after = (out > 1e-9).sum().item()
        assert n_edges_after == n_edges_before, (
            f"K_n should not lose edges under flow+surgery with high threshold; "
            f"before={n_edges_before}, after={n_edges_after}"
        )

    def test_bridge_removed_by_surgery(self):
        """Two cliques joined by a bridge edge: after enough flow + surgery,
        the bridge is removed. This is the network analog of Perelman's
        surgery cutting a forming neck.
        """
        A = _two_cliques_with_bridge(cluster_size=4)
        cluster_size = 4
        # Aggressive surgery with low threshold
        out = ricci_flow_with_surgery(
            A,
            n_steps=4,
            surgery_period=1,
            surgery_threshold=1.5,  # remove anything 1.5x mean
            dt=0.5,
            normalize=True,
        )
        # Check the bridge is gone
        bridge_weight = out[0, cluster_size - 1, cluster_size].item()
        assert bridge_weight < 1e-9, (
            f"bridge should be surgered away; weight={bridge_weight:.3e}"
        )
        # Check intra-clique edges survive
        intra_weight = out[0, 0, 1].item()
        assert intra_weight > 1e-9, (
            f"intra-clique edges should survive; weight={intra_weight:.3e}"
        )

    def test_n_components_increases_with_surgery(self):
        """After surgery removes the bridge, the graph has 2 connected
        components instead of 1.
        """
        A = _two_cliques_with_bridge(cluster_size=4)
        out = ricci_flow_with_surgery(
            A, n_steps=4, surgery_period=1, surgery_threshold=1.5,
            dt=0.5, normalize=True,
        )

        # Count connected components via Laplacian rank deficiency
        from holonomy_lib.spectral import laplacian
        L = laplacian.combinatorial(out)
        eigvals = torch.linalg.eigvalsh(L)
        # Components = number of near-zero eigenvalues
        n_components = (eigvals[0] < 1e-6).sum().item()
        assert n_components >= 2, (
            f"expected ≥ 2 components after surgery; got {n_components}"
        )


# --------------------------------------------------------------------
# Provenance integration — flow + surgery participate in mech-interp
# --------------------------------------------------------------------


class TestProvenanceIntegration:
    def test_flow_emits_provenance_node(self):
        from holonomy_lib import provenance
        A = _complete_graph(4)
        with provenance.record() as reg:
            discrete_ricci_flow(A, n_steps=1, dt=0.5)
        # Should have the outer flow node + the inner curvature node(s)
        op_ids = {n.op_id for n in reg}
        assert "holonomy_lib.discrete_geometry.discrete_ricci_flow" in op_ids
        assert "holonomy_lib.discrete_geometry.ollivier_ricci_curvature" in op_ids

    def test_surgery_emits_provenance_node(self):
        from holonomy_lib import provenance
        A = _two_cliques_with_bridge(cluster_size=3)
        with provenance.record() as reg:
            ricci_flow_with_surgery(
                A, n_steps=1, surgery_period=1, surgery_threshold=2.0, dt=0.3,
            )
        op_ids = {n.op_id for n in reg}
        assert "holonomy_lib.discrete_geometry.ricci_flow_with_surgery" in op_ids
