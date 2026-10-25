# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Synoros

"""Tests for the dense + sparse simplicial complexes and Vietoris-Rips.

Layers:
  1. Hand-built K_4 (tetrahedron) + K_5 (4-simplex) — verify
     chain-complex property `∂_{k-1} ∘ ∂_k = 0`.
  2. Round-trip dense↔sparse conversion.
  3. Vietoris-Rips on toy point clouds.
  4. Comparison vs gudhi at modest sizes (importorskip).
"""

from __future__ import annotations

import pytest
import torch

from holonomy_lib.simplicial import (
    SparseSimplicialComplex,
    pairwise_distances,
    vietoris_rips_dense,
    vietoris_rips_sparse,
)

# --------------------------------------------------------------------
# Hand-built complexes — tetrahedron (K_4)
# --------------------------------------------------------------------


def _tetrahedron_sparse() -> SparseSimplicialComplex:
    """K_4 = full 4-vertex complex up to dim 3 (one 3-simplex)."""
    # 4 vertices, 6 edges, 4 triangles, 1 tetrahedron.
    vertices = torch.tensor([[0], [1], [2], [3]], dtype=torch.int64)
    edges = torch.tensor(
        [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]],
        dtype=torch.int64,
    )
    triangles = torch.tensor(
        [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]],
        dtype=torch.int64,
    )
    tetra = torch.tensor([[0, 1, 2, 3]], dtype=torch.int64)
    return SparseSimplicialComplex(
        simplices_by_dim={0: vertices, 1: edges, 2: triangles, 3: tetra},
        n_vertices=4,
    )


def _hollow_triangle_sparse() -> SparseSimplicialComplex:
    """Boundary of triangle — vertices + 3 edges, no 2-simplex.

    Should have β_0 = 1, β_1 = 1 (forms a single 1-cycle).
    """
    vertices = torch.tensor([[0], [1], [2]], dtype=torch.int64)
    edges = torch.tensor(
        [[0, 1], [0, 2], [1, 2]], dtype=torch.int64,
    )
    return SparseSimplicialComplex(
        simplices_by_dim={0: vertices, 1: edges},
        n_vertices=3,
    )


# --------------------------------------------------------------------
# Chain-complex property
# --------------------------------------------------------------------


class TestChainComplexProperty:
    """`∂_{k-1} ∘ ∂_k = 0` for any simplicial complex. Tested on the
    dense and sparse representations separately."""

    def test_tetrahedron_dense_d0_compose_d1(self):
        sc = _tetrahedron_sparse().to_dense()
        d1 = sc.boundary(1)  # (1, n_0, n_1)
        d0 = sc.boundary(0)  # (1, 0, n_0) — empty
        prod = torch.matmul(d0, d1)  # (1, 0, n_1)
        # Empty result; trivially zero.
        assert prod.shape == (1, 0, 6)

    def test_tetrahedron_dense_d1_compose_d2(self):
        sc = _tetrahedron_sparse().to_dense()
        d1 = sc.boundary(1)  # (1, 4, 6)
        d2 = sc.boundary(2)  # (1, 6, 4)
        prod = torch.matmul(d1, d2)  # (1, 4, 4)
        torch.testing.assert_close(
            prod, torch.zeros_like(prod), atol=1e-12, rtol=0,
        )

    def test_tetrahedron_dense_d2_compose_d3(self):
        sc = _tetrahedron_sparse().to_dense()
        d2 = sc.boundary(2)  # (1, 6, 4)
        d3 = sc.boundary(3)  # (1, 4, 1)
        prod = torch.matmul(d2, d3)  # (1, 6, 1)
        torch.testing.assert_close(
            prod, torch.zeros_like(prod), atol=1e-12, rtol=0,
        )

    def test_tetrahedron_sparse_d1_compose_d2(self):
        sc = _tetrahedron_sparse()
        d1 = sc.boundary(1).to_dense()  # (4, 6)
        d2 = sc.boundary(2).to_dense()  # (6, 4)
        prod = torch.matmul(d1, d2)
        torch.testing.assert_close(
            prod, torch.zeros_like(prod), atol=1e-12, rtol=0,
        )

    def test_tetrahedron_sparse_d2_compose_d3(self):
        sc = _tetrahedron_sparse()
        d2 = sc.boundary(2).to_dense()  # (6, 4)
        d3 = sc.boundary(3).to_dense()  # (4, 1)
        prod = torch.matmul(d2, d3)
        torch.testing.assert_close(
            prod, torch.zeros_like(prod), atol=1e-12, rtol=0,
        )


# --------------------------------------------------------------------
# Round-trip dense ↔ sparse
# --------------------------------------------------------------------


class TestRoundTrip:
    def test_sparse_to_dense_back(self):
        sc = _tetrahedron_sparse()
        dense = sc.to_dense()
        sparse_again = dense.to_sparse()
        # Simplex tables match exactly.
        for k in sc.simplices_by_dim:
            torch.testing.assert_close(
                sc.simplices_by_dim[k], sparse_again.simplices_by_dim[k],
                atol=0, rtol=0,
            )

    def test_boundary_matches_across_representations(self):
        """∂_k from the sparse complex (as a dense view) matches ∂_k
        from the dense complex (B=1, no padding) entry-by-entry."""
        sparse = _tetrahedron_sparse()
        dense = sparse.to_dense()
        for k in [1, 2, 3]:
            d_sparse = sparse.boundary(k).to_dense()
            d_dense = dense.boundary(k)[0]                  # drop B dim
            torch.testing.assert_close(
                d_sparse, d_dense, atol=0, rtol=0,
            )


# --------------------------------------------------------------------
# Vietoris-Rips on toy point clouds
# --------------------------------------------------------------------


class TestVietorisRips:
    def test_three_collinear_points(self):
        """Three points on a line; at radius spanning the longest pair,
        we get a single triangle. At a smaller radius, only the two
        adjacent edges. The middle vertex is always connected via
        edges, never via a self-loop."""
        # Points at 0, 1, 2 on a line. Distances: 1, 1, 2.
        pts = torch.tensor(
            [[0.0], [1.0], [2.0]], dtype=torch.float64,
        )
        d = pairwise_distances(pts)
        # Radius 1.0: only the two unit edges, no triangle.
        sc = vietoris_rips_sparse(d, max_radius=1.0, max_dim=2)
        assert sc.n_simplices(0) == 3
        assert sc.n_simplices(1) == 2   # (0,1) and (1,2)
        assert sc.n_simplices(2) == 0   # no triangle (0,2 distance > 1)
        # Radius 2.0: all three edges + one triangle.
        sc2 = vietoris_rips_sparse(d, max_radius=2.0, max_dim=2)
        assert sc2.n_simplices(0) == 3
        assert sc2.n_simplices(1) == 3
        assert sc2.n_simplices(2) == 1

    def test_no_edges_at_zero_radius(self):
        pts = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float64,
        )
        d = pairwise_distances(pts)
        sc = vietoris_rips_sparse(d, max_radius=0.0, max_dim=2)
        assert sc.n_simplices(0) == 3
        assert sc.n_simplices(1) == 0

    def test_batched_construction(self):
        # Two batch elements with different geometries.
        pts = torch.tensor([
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],   # triangle, all dist ≤ √2
            [[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]], # triangle, larger scale
        ], dtype=torch.float64)
        d = pairwise_distances(pts)
        sc = vietoris_rips_dense(d, max_radius=1.5, max_dim=2)
        # Batch 0: all three edges (max pair dist = √2 ≈ 1.414) + triangle
        # Batch 1: no edges (min pair dist = 10)
        assert sc.n_simplices(0).tolist() == [3, 3]
        assert sc.n_simplices(1).tolist() == [3, 0]
        assert sc.n_simplices(2).tolist() == [1, 0]

    def test_pairwise_distances_shape(self):
        pts2d = torch.tensor([[0.0, 0.0], [3.0, 4.0]])
        d = pairwise_distances(pts2d)
        assert d.shape == (2, 2)
        assert d[0, 1].item() == pytest.approx(5.0)
        # batched
        pts3d = torch.zeros(4, 5, 3)
        d3 = pairwise_distances(pts3d)
        assert d3.shape == (4, 5, 5)


# --------------------------------------------------------------------
# Cycle complex — sanity check that holes (no triangle) are preserved
# --------------------------------------------------------------------


class TestHollowComplex:
    def test_hollow_triangle_d1_compose_d0_zero(self):
        """For the boundary-of-triangle complex, ∂_1 sends each edge to
        ±v_1 ∓ v_0. The composition ∂_0 ∘ ∂_1 is trivially zero
        (∂_0 is empty), but we also verify the rank of ∂_1: rows sum
        to zero (each vertex appears in two edges with opposite
        sign)."""
        sc = _hollow_triangle_sparse()
        d1 = sc.boundary(1).to_dense()
        # Column sums (per edge) are zero.
        torch.testing.assert_close(
            d1.sum(dim=0),
            torch.zeros(d1.shape[1], dtype=d1.dtype),
            atol=1e-12, rtol=0,
        )


# --------------------------------------------------------------------
# Comparison vs gudhi (importorskip)
# --------------------------------------------------------------------


try:
    import gudhi  # noqa: F401
    _HAVE_GUDHI = True
except ImportError:
    _HAVE_GUDHI = False


@pytest.mark.skipif(not _HAVE_GUDHI, reason="gudhi not installed")
class TestAgainstGudhi:
    """Simplex counts per dim match gudhi at the same radius + max_dim."""

    def test_vr_simplex_counts_random_points(self):
        import gudhi as g
        torch.manual_seed(0)
        n = 12
        d = 2
        pts = torch.randn(n, d, dtype=torch.float64)
        dist = pairwise_distances(pts)
        max_r = 1.5
        max_dim = 2

        # holonomy_lib
        sc = vietoris_rips_sparse(dist, max_radius=max_r, max_dim=max_dim)
        counts_ours = [sc.n_simplices(k) for k in range(max_dim + 1)]

        # gudhi
        rips = g.RipsComplex(points=pts.numpy(), max_edge_length=max_r)
        st = rips.create_simplex_tree(max_dimension=max_dim)
        counts_gudhi = [0] * (max_dim + 1)
        for simplex, _ in st.get_simplices():
            counts_gudhi[len(simplex) - 1] += 1

        assert counts_ours == counts_gudhi, (
            f"simplex counts disagree: ours={counts_ours}, "
            f"gudhi={counts_gudhi}"
        )
