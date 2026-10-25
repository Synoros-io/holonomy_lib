# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Synoros

"""Tests for holonomy_lib.topology.hodge.

Closed-form Betti numbers verified against hand-built triangulations
of known spaces:

  - Triangulated S¹ (cycle, 1-complex):           β = (1, 1)
  - Triangulated S² (octahedron):                 β = (1, 0, 1)
  - Triangulated T² (Möbius's 7-vertex torus):    β = (1, 2, 1)
  - Two disjoint triangles:                       β_0 = 2
  - Tetrahedron K_4 (3-simplex boundary disk):    β = (1, 0, 0, 0)
"""

from __future__ import annotations

import pytest
import torch

from holonomy_lib.simplicial import DenseSimplicialComplex, SparseSimplicialComplex
from holonomy_lib.topology import betti_numbers, hodge_laplacian

# --------------------------------------------------------------------
# Hand-built triangulations
# --------------------------------------------------------------------


def _circle_S1(n: int = 6) -> SparseSimplicialComplex:
    """Triangulated S¹ as a cycle graph: n vertices, n edges in a loop."""
    vertices = torch.arange(n, dtype=torch.int64).unsqueeze(-1)
    edge_list = [[i, (i + 1) % n] for i in range(n)]
    # Sort each edge so the table is canonical (smaller vertex first).
    edges = torch.tensor(
        [sorted(e) for e in edge_list], dtype=torch.int64,
    )
    return SparseSimplicialComplex(
        simplices_by_dim={0: vertices, 1: edges},
        n_vertices=n,
    )


def _octahedron_S2() -> SparseSimplicialComplex:
    """Triangulated S² as an octahedron: 6 vertices, 12 edges, 8 triangles.

    Vertex 0 and 1 are the north/south poles; 2, 3, 4, 5 form the equator
    in cyclic order. Each pole connects to every equatorial vertex.
    Triangles are the 8 faces.
    """
    vertices = torch.arange(6, dtype=torch.int64).unsqueeze(-1)
    # Edges
    edge_pairs = []
    # Pole 0 to every equatorial vertex (4 edges)
    for v in [2, 3, 4, 5]:
        edge_pairs.append((0, v))
    # Pole 1 to every equatorial vertex (4 edges)
    for v in [2, 3, 4, 5]:
        edge_pairs.append((1, v))
    # Equator cycle (4 edges)
    equator = [2, 3, 4, 5]
    for i in range(4):
        a = equator[i]
        b = equator[(i + 1) % 4]
        edge_pairs.append((min(a, b), max(a, b)))
    edges = torch.tensor(edge_pairs, dtype=torch.int64)
    # Triangles: each pole + each consecutive pair of equatorial vertices.
    triangle_list = []
    for pole in [0, 1]:
        for i in range(4):
            a = equator[i]
            b = equator[(i + 1) % 4]
            triangle_list.append(sorted([pole, a, b]))
    triangles = torch.tensor(triangle_list, dtype=torch.int64)
    return SparseSimplicialComplex(
        simplices_by_dim={0: vertices, 1: edges, 2: triangles},
        n_vertices=6,
    )


def _torus_T2() -> SparseSimplicialComplex:
    """Möbius's 7-vertex triangulation of T² (1861). 7 vertices, 21 edges,
    14 triangles. β = (1, 2, 1)."""
    # Vertices 0..6
    vertices = torch.arange(7, dtype=torch.int64).unsqueeze(-1)
    # Triangles per Möbius (numbered 0..6 cyclically).
    triangle_list = [
        [0, 1, 3], [1, 2, 4], [2, 3, 5], [3, 4, 6], [4, 5, 0],
        [5, 6, 1], [6, 0, 2],
        [0, 1, 5], [1, 2, 6], [2, 3, 0], [3, 4, 1], [4, 5, 2],
        [5, 6, 3], [6, 0, 4],
    ]
    triangles = torch.tensor(
        [sorted(t) for t in triangle_list], dtype=torch.int64,
    )
    # Derive all edges from the triangles.
    edge_set: set[tuple[int, int]] = set()
    for t in triangle_list:
        t_sorted = sorted(t)
        for i, j in [(0, 1), (0, 2), (1, 2)]:
            edge_set.add((t_sorted[i], t_sorted[j]))
    edges = torch.tensor(sorted(edge_set), dtype=torch.int64)
    return SparseSimplicialComplex(
        simplices_by_dim={0: vertices, 1: edges, 2: triangles},
        n_vertices=7,
    )


def _two_disjoint_triangles() -> SparseSimplicialComplex:
    """Two disjoint solid triangles. β_0 = 2 (two components),
    β_1 = β_2 = 0."""
    vertices = torch.arange(6, dtype=torch.int64).unsqueeze(-1)
    edges = torch.tensor([
        [0, 1], [0, 2], [1, 2],
        [3, 4], [3, 5], [4, 5],
    ], dtype=torch.int64)
    triangles = torch.tensor([
        [0, 1, 2], [3, 4, 5],
    ], dtype=torch.int64)
    return SparseSimplicialComplex(
        simplices_by_dim={0: vertices, 1: edges, 2: triangles},
        n_vertices=6,
    )


# --------------------------------------------------------------------
# Betti numbers (sparse path)
# --------------------------------------------------------------------


class TestBettiNumbersSparse:

    @pytest.mark.parametrize("dense", [False, True])
    def test_S2_octahedron(self, dense):
        complex = _octahedron_S2()
        result = betti_numbers(complex.to_dense() if dense else complex, 2)
        assert result.reshape(-1).tolist() == [1, 0, 1]
        for dim in range(3):
            torch.testing.assert_close(
                hodge_laplacian(complex, dim),
                hodge_laplacian(complex.to_dense(), dim)[0], atol=1e-12, rtol=0,
            )

    def test_T2_torus(self):
        sc = _torus_T2()
        b = betti_numbers(sc, max_dim=2)
        assert b.tolist() == [1, 2, 1]

    def test_two_disjoint_triangles(self):
        sc = _two_disjoint_triangles()
        b = betti_numbers(sc, max_dim=2)
        assert b.tolist() == [2, 0, 0]

    @pytest.mark.parametrize("representation,dtype", [
        ("sparse", torch.float64), ("dense", torch.float64),
        ("dense", torch.float32), ("dense", torch.float16),
        ("dense", torch.bfloat16),
    ])
    def test_circle_at_various_sizes(self, representation, dtype):
        for n in [4, 5, 6, 7, 8, 10]:
            sparse = _circle_S1(n)
            complex: DenseSimplicialComplex | SparseSimplicialComplex = (
                sparse.to_dense(dtype=dtype) if representation == "dense" else sparse
            )
            assert betti_numbers(complex, 1).reshape(-1).tolist() == [1, 1]
        # Empty chain dimensions have rank zero, including an empty batch.
        from holonomy_lib.simplicial import vietoris_rips_dense
        for batch in (0, 1, 2):
            distances = (torch.ones(batch, 3, 3, dtype=dtype)
                         - torch.eye(3, dtype=dtype))
            complex = vietoris_rips_dense(distances, max_radius=0., max_dim=1)
            result = betti_numbers(complex, 1)
            assert result.shape == (batch, 2)
            assert result.tolist() == [[3, 0]] * batch
        empty_edges = SparseSimplicialComplex(
            {0: torch.arange(3).reshape(-1, 1),
             1: torch.empty(0, 2, dtype=torch.int64)}, n_vertices=3,
        )
        assert betti_numbers(empty_edges, 1).tolist() == [3, 0]
        if representation == "dense" and dtype == torch.float32:
            # A path has one component and no cycles. Long paths expose
            # the loss of rank from squaring boundary conditioning.
            for n in (300, 512):
                points = torch.arange(n, dtype=dtype)
                distances = (points[:, None] - points[None, :]).abs()[None]
                path = vietoris_rips_dense(distances, 1.0, 1)
                assert betti_numbers(path, 1).tolist() == [[1, 0]]
                assert betti_numbers(path, 0).tolist() == [[1]]

            # A larger edge table in another member cannot change this
            # path's rank tolerance or contribute padding as topology.
            n = 256
            points = torch.arange(n, dtype=dtype)
            distances = (points[:, None] - points[None, :]).abs()[None]
            other = torch.full((n, n), 2.0, dtype=dtype)
            other.fill_diagonal_(0)
            rows, cols = torch.triu_indices(n, n, offset=1)[:, :600]
            other[rows, cols] = other[cols, rows] = 1.0
            alone = vietoris_rips_dense(distances, 1.0, 1)
            padded = vietoris_rips_dense(
                torch.cat((distances, other[None])), 1.0, 1,
            )
            assert betti_numbers(alone, 1).tolist() == [[1, 0]]
            assert betti_numbers(padded, 1)[0].tolist() == [1, 0]


# --------------------------------------------------------------------
# Hodge Laplacian spectrum
# --------------------------------------------------------------------


class TestHodgeSpectrum:


    def test_L0_matches_graph_laplacian_for_S1(self):
        """For a 1-complex (no triangles), L_0 = ∂_1 ∂_1^T = graph
        Laplacian. The cycle graph C_n has eigenvalues
        `{4·sin²(πk/n) : k = 0, …, n−1}`. We compute the closed form
        in float64 to compare at machine precision."""
        import math
        n = 6
        sc = _circle_S1(n=n)
        L0 = hodge_laplacian(sc, 0)
        eigvals = torch.linalg.eigvalsh(L0).sort().values
        expected = torch.tensor(
            sorted(
                4.0 * (math.sin(math.pi * k / n) ** 2) for k in range(n)
            ),
            dtype=torch.float64,
        )
        torch.testing.assert_close(
            eigvals, expected, atol=1e-9, rtol=0,
        )


# --------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------


class TestValidation:
    def test_rejects_negative_k(self):
        sc = _circle_S1(n=4)
        with pytest.raises(ValueError, match="k"):
            hodge_laplacian(sc, -1)

    def test_rejects_negative_max_dim(self):
        sc = _circle_S1(n=4)
        with pytest.raises(ValueError, match="max_dim"):
            betti_numbers(sc, -1)


# --------------------------------------------------------------------
# Sparse Lanczos sanity (Phase 3 extension)
# --------------------------------------------------------------------


class TestSparseLanczos:
    """The extended lanczos_eigsh accepts sparse-CSC input. Result must
    agree with dense Lanczos / eigh on the same matrix."""

    def test_sparse_lanczos_matches_dense(self):
        from holonomy_lib.algebra import lanczos_eigsh

        n = 12
        torch.manual_seed(0)
        # Random symmetric matrix.
        A = torch.randn(n, n, dtype=torch.float64)
        A = 0.5 * (A + A.mT)

        # Sparse CSC version.
        A_sparse = A.to_sparse_csc()

        g = torch.Generator().manual_seed(1)
        vals_sparse, _ = lanczos_eigsh(
            A_sparse, k=1, n_iter=n, oversample=0, generator=g,
        )
        ref = torch.linalg.eigvalsh(A).flip(0)[:1]
        torch.testing.assert_close(
            vals_sparse, ref, atol=1e-9, rtol=0,
        )

    def test_sparse_rejects_3d(self):
        from holonomy_lib.algebra import lanczos_eigsh
        # 3-D sparse — not supported.
        A = torch.randn(2, 4, 4, dtype=torch.float64)
        A = 0.5 * (A + A.mT)
        A_sparse = A.to_sparse_coo()
        with pytest.raises(ValueError, match="sparse"):
            lanczos_eigsh(A_sparse, k=1)
