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


    def test_surgery_rejects_nonpositive_period(self):
        A = _complete_graph(4)
        with pytest.raises(ValueError, match="surgery_period"):
            ricci_flow_with_surgery(A, n_steps=1, surgery_period=0)
        with pytest.raises(ValueError, match="n_steps"):
            ricci_flow_with_surgery(A, n_steps=-1)


# --------------------------------------------------------------------
# Flow — basic properties
# --------------------------------------------------------------------


class TestFlow:
    def test_zero_steps_returns_input(self):
        A = _complete_graph(4)
        out = discrete_ricci_flow(A + 5 * torch.eye(4), n_steps=0)
        torch.testing.assert_close(out, A, atol=0, rtol=0)
        empty = torch.empty(0, 4, 4, dtype=A.dtype)
        torch.testing.assert_close(
            discrete_ricci_flow(empty, n_steps=1), empty, atol=0, rtol=0,
        )


    def test_flow_normalize_keeps_norm_constant(self):
        A = _two_cliques_with_bridge(cluster_size=3)
        n0 = torch.linalg.matrix_norm(A).item()
        out = discrete_ricci_flow(A, n_steps=5, dt=0.3, normalize=True)
        n1 = torch.linalg.matrix_norm(out).item()
        assert abs(n0 - n1) / max(n0, 1e-9) < 1e-6
        path = torch.tensor(
            [[[0., 1., 0.], [1., 0., 1.], [0., 1., 0.]]], dtype=A.dtype,
        )
        looped = discrete_ricci_flow(path + 5 * torch.eye(3), n_steps=1)
        torch.testing.assert_close(looped, path, atol=1e-12, rtol=0)
        # Normalization is relative to the graph's own scale, including
        # lengths below the old absolute norm floor.
        for dtype in (torch.float64, torch.float32):
            for scale in (1e-12, 1.0):
                tiny = _complete_graph(3, dtype=dtype) * scale
                for flow in (discrete_ricci_flow, ricci_flow_with_surgery):
                    result = flow(
                        tiny, n_steps=1, dt=.2, normalize=True,
                        reg=.4 * scale, n_sinkhorn_iters=1000, sinkhorn_tol=1e-6,
                    )
                    torch.testing.assert_close(result.norm() / tiny.norm(),
                                               tiny.new_tensor(1.))
                    zero = torch.zeros_like(tiny)
                    torch.testing.assert_close(flow(zero, n_steps=1), zero,
                                               atol=0, rtol=0)


class TestFlowDynamics:
    def test_bridge_grows_relative_to_clique_under_flow(self):
        """The bottleneck edge has negative curvature ⇒ under the flow
        it elongates (gets larger relative to intra-clique edges).
        """
        A = _two_cliques_with_bridge(cluster_size=4)
        bridge = (3, 4)  # the bridge edge
        intra = (0, 1)   # an intra-clique edge

        out = discrete_ricci_flow(A, n_steps=2, dt=0.5, normalize=True)
        torch.testing.assert_close(out, out.mT, atol=1e-9, rtol=0)
        assert (out >= 0).all()
        small = discrete_ricci_flow(
            _two_cliques_with_bridge(cluster_size=3), n_steps=3, dt=0.5,
        )
        torch.testing.assert_close(small, small.mT, atol=1e-9, rtol=0)
        assert (small >= 0).all()
        # K3 curvature is 1/2, so one unnormalized dt=.2 step is .9.
        clique = _complete_graph(3)
        absolute = discrete_ricci_flow(clique, n_steps=1, dt=0.2, normalize=False)
        torch.testing.assert_close(absolute, 0.9 * clique, atol=1e-9, rtol=0)

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
        out = ricci_flow_with_surgery(A + 5 * torch.eye(A.shape[-1]), n_steps=0)
        torch.testing.assert_close(out, A, atol=0, rtol=0)
        empty = torch.empty(0, 4, 4, dtype=A.dtype)
        torch.testing.assert_close(
            ricci_flow_with_surgery(empty, n_steps=1), empty, atol=0, rtol=0,
        )

    def test_complete_clique_no_surgery_with_high_threshold(self):
        """On K_n every edge has positive curvature; under flow they
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
        # Self-loops cannot alter either normalization or surgery's cutoff.
        looped = ricci_flow_with_surgery(
            A + 5 * torch.eye(A.shape[-1]), n_steps=4, surgery_period=1,
            surgery_threshold=1.5, dt=0.5, normalize=True,
        )
        torch.testing.assert_close(looped, out, atol=1e-12, rtol=0)
        # The surviving cliques form exactly two connected components.
        from holonomy_lib.spectral import laplacian
        L = laplacian.combinatorial(out)
        n_components = L.shape[-1] - torch.linalg.matrix_rank(L, hermitian=True)
        assert n_components.item() == 2
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
