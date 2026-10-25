# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Synoros

"""Tests for holonomy_lib.sheaf.

Cover layers:
  1. Validation (shape/dtype/device/index-range mismatches).
  2. `GraphSheaf.trivial` helper produces a usable sheaf.
  3. The trivial 1-D sheaf's Laplacian equals the standard graph
     Laplacian for several topologies.
  4. The trivial d-D sheaf's Laplacian equals `d × L` block-diagonally
     embedded.
  5. The sheaf Laplacian is symmetric and PSD on arbitrary sheaves.
  6. Coboundary and Laplacian agree: `L_F = δ^T δ`.
  7. Dirichlet energy `E(x) = x^T L_F x = ‖δx‖²` on batched signals.
  8. A non-trivial restriction map gives a different kernel than the
     trivial sheaf — the actual content of the sheaf-theoretic
     framework.
"""

from __future__ import annotations

import pytest
import torch

from holonomy_lib.sheaf import (
    GraphSheaf,
    sheaf_coboundary,
    sheaf_dirichlet_energy,
    sheaf_laplacian,
)
from holonomy_lib.spectral import laplacian as graph_lap


def _seeded(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def _cycle_edges(n: int) -> torch.Tensor:
    """Directed edges of a cycle: (0→1), (1→2), …, (n−1→0)."""
    return torch.tensor(
        [(i, (i + 1) % n) for i in range(n)], dtype=torch.int64,
    )


def _path_edges(n: int) -> torch.Tensor:
    return torch.tensor(
        [(i, i + 1) for i in range(n - 1)], dtype=torch.int64,
    )


def _adjacency_from_undirected_edges(
    n: int, edges: torch.Tensor, dtype=torch.float64,
) -> torch.Tensor:
    """Symmetric (n, n) 0/1 adjacency matrix from a directed edge list."""
    A = torch.zeros(n, n, dtype=dtype)
    for u, v in edges.tolist():
        A[u, v] = 1.0
        A[v, u] = 1.0
    return A


# --------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------


class TestValidation:
    def test_rejects_bad_n_nodes(self):
        with pytest.raises(ValueError, match="n_nodes"):
            GraphSheaf(
                n_nodes=0, edges=torch.zeros(0, 2, dtype=torch.int64),
                node_stalk_dim=1, edge_stalk_dim=1,
                F_left=torch.zeros(0, 1, 1, dtype=torch.float64),
                F_right=torch.zeros(0, 1, 1, dtype=torch.float64),
            )

    def test_rejects_F_shape_mismatch(self):
        edges = torch.tensor([[0, 1], [1, 2]], dtype=torch.int64)
        with pytest.raises(ValueError, match="F_left must have shape"):
            GraphSheaf(
                n_nodes=3, edges=edges,
                node_stalk_dim=2, edge_stalk_dim=3,
                # Wrong shape: should be (2, 3, 2)
                F_left=torch.zeros(2, 2, 2, dtype=torch.float64),
                F_right=torch.zeros(2, 3, 2, dtype=torch.float64),
            )

    def test_rejects_edge_out_of_range(self):
        edges = torch.tensor([[0, 5]], dtype=torch.int64)
        with pytest.raises(ValueError, match="\\[0, n_nodes"):
            GraphSheaf(
                n_nodes=3, edges=edges,
                node_stalk_dim=1, edge_stalk_dim=1,
                F_left=torch.zeros(1, 1, 1, dtype=torch.float64),
                F_right=torch.zeros(1, 1, 1, dtype=torch.float64),
            )

    def test_trivial_helper_rejects_bad_edges(self):
        with pytest.raises(ValueError, match="edges must be"):
            GraphSheaf.trivial(n_nodes=3, edges=torch.zeros(5))

    def test_dirichlet_energy_dim_check(self):
        sheaf = GraphSheaf.trivial(n_nodes=4, edges=_cycle_edges(4))
        x_wrong = torch.zeros(3, dtype=torch.float64)
        with pytest.raises(ValueError, match="n_nodes"):
            sheaf_dirichlet_energy(sheaf, x_wrong)

    def test_rejects_self_loops(self):
        """A self-loop edge `(u, u)` makes the sheaf Laplacian disagree
        silently with the standard graph Laplacian (which drops
        self-loops by convention). Reject upfront."""
        edges = torch.tensor([[0, 1], [1, 1], [2, 0]], dtype=torch.int64)
        with pytest.raises(ValueError, match="self-loops"):
            GraphSheaf.trivial(n_nodes=3, edges=edges)

    def test_rejects_duplicate_edges(self):
        """Duplicate `(u, v)` entries would double the off-diagonal
        block of the sheaf Laplacian, breaking the trivial-sheaf
        reduction. Reject upfront."""
        edges = torch.tensor([[0, 1], [1, 2], [0, 1]], dtype=torch.int64)
        with pytest.raises(ValueError, match="duplicate"):
            GraphSheaf.trivial(n_nodes=3, edges=edges)

    def test_dense_size_cap_raises(self):
        """A sheaf large enough to overflow the dense-δ byte cap should
        fail with a useful pre-flight message, not an OOM later."""
        from holonomy_lib.sheaf import sheaf_coboundary
        # Construct a sheaf that would need > 2 GiB of dense allocation
        # without actually allocating it (the byte check runs before the
        # zeros() call).
        n_nodes = 10000
        d_v = 32
        # Build a non-trivial number of edges that crosses 2 GiB:
        # bytes ≈ 2 · n_e · d_e · n_v · d_v · 8. For d_e = d_v = 32,
        # n_v = 10000, 8-byte float, need n_e ≳ 130 for >2GiB.
        n_e = 200
        edges = torch.stack(
            [torch.arange(n_e) % n_nodes,
             (torch.arange(n_e) + 1) % n_nodes],
            dim=-1,
        )
        eye = torch.eye(d_v, dtype=torch.float64)
        F = eye.unsqueeze(0).expand(n_e, d_v, d_v).contiguous()
        sheaf = GraphSheaf(
            n_nodes=n_nodes, edges=edges,
            node_stalk_dim=d_v, edge_stalk_dim=d_v,
            F_left=F.clone(), F_right=F.clone(),
        )
        with pytest.raises(RuntimeError, match="dense path"):
            sheaf_coboundary(sheaf)


class TestTrivialHelperReturnsIndependentTensors:
    """`F_left` and `F_right` returned by `GraphSheaf.trivial` must be
    independent storage — mutating one must not affect the other.
    Otherwise the frozen-dataclass contract leaks mutability
    asymmetrically."""

    def test_F_left_F_right_are_independent(self):
        sheaf = GraphSheaf.trivial(n_nodes=4, edges=_cycle_edges(4), stalk_dim=2)
        # Mutate F_left in place; F_right should be unaffected.
        sheaf.F_left[0, 0, 0] = 99.0
        assert sheaf.F_right[0, 0, 0].item() == 1.0, (
            "F_right is aliasing F_left; trivial() must return clones"
        )


# --------------------------------------------------------------------
# Trivial 1-D sheaf reduces to standard graph Laplacian
# --------------------------------------------------------------------


class TestTrivialSheafReducesToGraphLaplacian:
    """For stalk_dim = 1 with identity restriction maps, the sheaf
    Laplacian must equal the combinatorial graph Laplacian of the
    same edge set."""

    @pytest.mark.parametrize("n", [4, 5, 8])
    def test_cycle(self, n):
        edges = _cycle_edges(n)
        sheaf = GraphSheaf.trivial(n_nodes=n, edges=edges)
        L_sheaf = sheaf_laplacian(sheaf)
        A = _adjacency_from_undirected_edges(n, edges)
        L_graph = graph_lap.combinatorial(A.unsqueeze(0)).squeeze(0)
        torch.testing.assert_close(L_sheaf, L_graph, atol=1e-12, rtol=0)

    @pytest.mark.parametrize("n", [4, 6])
    def test_path(self, n):
        edges = _path_edges(n)
        sheaf = GraphSheaf.trivial(n_nodes=n, edges=edges)
        L_sheaf = sheaf_laplacian(sheaf)
        A = _adjacency_from_undirected_edges(n, edges)
        L_graph = graph_lap.combinatorial(A.unsqueeze(0)).squeeze(0)
        torch.testing.assert_close(L_sheaf, L_graph, atol=1e-12, rtol=0)

    def test_complete_graph_k4(self):
        n = 4
        edges = torch.tensor(
            [(i, j) for i in range(n) for j in range(i + 1, n)],
            dtype=torch.int64,
        )
        sheaf = GraphSheaf.trivial(n_nodes=n, edges=edges)
        L_sheaf = sheaf_laplacian(sheaf)
        A = _adjacency_from_undirected_edges(n, edges)
        L_graph = graph_lap.combinatorial(A.unsqueeze(0)).squeeze(0)
        torch.testing.assert_close(L_sheaf, L_graph, atol=1e-12, rtol=0)


# --------------------------------------------------------------------
# Trivial d-dim sheaf is block-diagonal d copies of the scalar L
# --------------------------------------------------------------------


class TestTrivialHigherDimSheaf:
    """For stalk_dim = d > 1 with identity restriction maps, the sheaf
    Laplacian is `I_d ⊗ L_graph` in node-major / dim-minor ordering:
    each node contributes a `d × d` diagonal block of the scalar
    Laplacian's value at that diagonal entry, and each edge contributes
    a `d × d` off-diagonal block.
    """

    def test_dim_2_cycle_4(self):
        n = 4
        d = 2
        edges = _cycle_edges(n)
        sheaf = GraphSheaf.trivial(n_nodes=n, edges=edges, stalk_dim=d)
        L_sheaf = sheaf_laplacian(sheaf)

        A = _adjacency_from_undirected_edges(n, edges)
        L_graph = graph_lap.combinatorial(A.unsqueeze(0)).squeeze(0)

        # Build the reference: L_graph entries become d×d blocks. Using
        # the "node-major" layout we picked (x_node_i has its d entries
        # at positions i*d : (i+1)*d), the reference is L_graph ⊗ I_d.
        L_ref = torch.kron(L_graph, torch.eye(d, dtype=torch.float64))
        torch.testing.assert_close(L_sheaf, L_ref, atol=1e-12, rtol=0)


# --------------------------------------------------------------------
# Symmetry + PSD for arbitrary sheaves
# --------------------------------------------------------------------


class TestSymmetryAndPSD:
    """The sheaf Laplacian is `δ^T δ` so it must be symmetric PSD by
    construction. Test on randomly initialized non-trivial sheaves."""

    @pytest.mark.parametrize("d_v, d_e", [(1, 1), (3, 2), (4, 4)])
    def test_random_sheaf_symmetric_and_psd(self, d_v, d_e):
        n = 6
        edges = _cycle_edges(n)
        g = _seeded(42)
        F_left = torch.randn(n, d_e, d_v, dtype=torch.float64, generator=g)
        F_right = torch.randn(n, d_e, d_v, dtype=torch.float64, generator=g)
        sheaf = GraphSheaf(
            n_nodes=n, edges=edges,
            node_stalk_dim=d_v, edge_stalk_dim=d_e,
            F_left=F_left, F_right=F_right,
        )
        L = sheaf_laplacian(sheaf)
        torch.testing.assert_close(L, L.mT, atol=1e-12, rtol=0)
        eigvals = torch.linalg.eigvalsh(L)
        # PSD: all eigenvalues ≥ 0 up to numerical noise. eigvalsh on a
        # near-zero eigenvalue can return ~ -dtype_eps; tolerate that.
        assert (eigvals >= -1e-9).all(), (
            f"sheaf Laplacian must be PSD; got min eigenvalue "
            f"{eigvals.min().item():.4e}"
        )


# --------------------------------------------------------------------
# L = δ^T δ identity
# --------------------------------------------------------------------


class TestLaplacianEqualsDeltaTDelta:
    def test_explicit_construction_matches(self):
        n = 5
        edges = _cycle_edges(n)
        d_v, d_e = 2, 3
        g = _seeded(7)
        F_left = torch.randn(n, d_e, d_v, dtype=torch.float64, generator=g)
        F_right = torch.randn(n, d_e, d_v, dtype=torch.float64, generator=g)
        sheaf = GraphSheaf(
            n_nodes=n, edges=edges,
            node_stalk_dim=d_v, edge_stalk_dim=d_e,
            F_left=F_left, F_right=F_right,
        )
        delta = sheaf_coboundary(sheaf)
        L_explicit = delta.mT @ delta
        L_explicit = 0.5 * (L_explicit + L_explicit.mT)  # match symmetrize
        L = sheaf_laplacian(sheaf)
        torch.testing.assert_close(L, L_explicit, atol=1e-12, rtol=0)


# --------------------------------------------------------------------
# Dirichlet energy
# --------------------------------------------------------------------


class TestDirichletEnergy:
    def test_unbatched_matches_xTLx(self):
        n = 4
        sheaf = GraphSheaf.trivial(n_nodes=n, edges=_cycle_edges(n))
        x = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
        E = sheaf_dirichlet_energy(sheaf, x)
        L = sheaf_laplacian(sheaf)
        ref = x @ L @ x
        torch.testing.assert_close(E, ref, atol=1e-12, rtol=0)

    def test_batched_matches_xTLx(self):
        n = 5
        d_v = 2
        sheaf = GraphSheaf.trivial(n_nodes=n, edges=_cycle_edges(n), stalk_dim=d_v)
        B = 3
        x = torch.randn(B, n * d_v, dtype=torch.float64, generator=_seeded(0))
        E = sheaf_dirichlet_energy(sheaf, x)
        L = sheaf_laplacian(sheaf)
        ref = torch.einsum("bi,ij,bj->b", x, L, x)
        torch.testing.assert_close(E, ref, atol=1e-10, rtol=0)

    def test_constant_signal_has_zero_energy_for_trivial_sheaf(self):
        """The kernel of the trivial-sheaf Laplacian = constant signals
        per connected component. Dirichlet energy should be zero."""
        n = 6
        sheaf = GraphSheaf.trivial(n_nodes=n, edges=_cycle_edges(n))
        x = torch.full((n,), 3.14, dtype=torch.float64)
        E = sheaf_dirichlet_energy(sheaf, x)
        assert E.item() == pytest.approx(0.0, abs=1e-12)


# --------------------------------------------------------------------
# Non-trivial sheaves see something the graph Laplacian cannot
# --------------------------------------------------------------------


class TestNonTrivialSheafSeparatesFromGraphLaplacian:
    """The whole point of sheaves: they capture structure invisible to
    the scalar Laplacian. Take a graph where two sheaves with the same
    edge set but different restriction maps have different kernel
    dimensions — that's a property the scalar graph Laplacian cannot
    distinguish."""

    def test_orientation_sheaf_on_triangle_has_smaller_kernel(self):
        """Triangle (3-cycle) with all stalks dim 1. Trivial sheaf has
        kernel = constants (dim 1). The "orientation sheaf" where one
        edge has its right map flipped to -1 represents a
        non-orientable line bundle on the cycle and has kernel dim 0
        (no global section survives the sign flip going around the
        loop)."""
        n = 3
        edges = _cycle_edges(n)
        # Trivial sheaf.
        sheaf_triv = GraphSheaf.trivial(n_nodes=n, edges=edges)
        L_triv = sheaf_laplacian(sheaf_triv)
        eig_triv = torch.linalg.eigvalsh(L_triv)
        # Number of zero eigenvalues = kernel dim.
        kdim_triv = int((eig_triv < 1e-9).sum().item())
        assert kdim_triv == 1, f"trivial cycle should have kernel dim 1; got {kdim_triv}"

        # Orientation-flip sheaf: same edges, but F_right on the last
        # edge is [-1] instead of [+1]. The "monodromy around the loop"
        # is now -1 ≠ +1, so no nonzero constant signal can satisfy
        # `F_left x_u = F_right x_v` simultaneously on every edge.
        F_left = torch.ones(n, 1, 1, dtype=torch.float64)
        F_right = torch.ones(n, 1, 1, dtype=torch.float64)
        F_right[n - 1, 0, 0] = -1.0
        sheaf_flip = GraphSheaf(
            n_nodes=n, edges=edges,
            node_stalk_dim=1, edge_stalk_dim=1,
            F_left=F_left, F_right=F_right,
        )
        L_flip = sheaf_laplacian(sheaf_flip)
        eig_flip = torch.linalg.eigvalsh(L_flip)
        kdim_flip = int((eig_flip < 1e-9).sum().item())
        assert kdim_flip == 0, (
            f"orientation-flip cycle should have NO global sections; "
            f"got kernel dim {kdim_flip}"
        )


# --------------------------------------------------------------------
# Provenance integration
# --------------------------------------------------------------------


class TestProvenanceIntegration:
    def test_sheaf_laplacian_emits_node(self):
        from holonomy_lib import provenance

        sheaf = GraphSheaf.trivial(n_nodes=4, edges=_cycle_edges(4))
        with provenance.record() as reg:
            sheaf_laplacian(sheaf)
        op_ids = sorted(n.op_id for n in reg)
        assert "holonomy_lib.sheaf.sheaf_laplacian" in op_ids

    def test_sheaf_provenance_signature_stable_across_instances(self):
        """Two GraphSheaf instances with identical topology + stalk
        dims must produce the same provenance signature — the dict
        must NOT embed the Python `id()`."""
        s1 = GraphSheaf.trivial(n_nodes=4, edges=_cycle_edges(4))
        s2 = GraphSheaf.trivial(n_nodes=4, edges=_cycle_edges(4))
        assert s1._provenance_signature() == s2._provenance_signature()
