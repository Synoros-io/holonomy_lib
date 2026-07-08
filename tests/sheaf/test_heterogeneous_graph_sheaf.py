# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Synoros

"""Tests for holonomy_lib.sheaf.HeterogeneousGraphSheaf (per-node stalk dims).

Three layers:
  1. Unit — construction + validation (shape/self-loop/duplicate/range).
  2. Property — reduction: a heterogeneous sheaf with UNIFORM dims + identity
     maps reproduces `GraphSheaf.trivial`'s Laplacian exactly (cross-checks the
     ragged coboundary against the tested uniform path); genuinely-variable dims
     give a symmetric PSD Laplacian of size Σ d_v; Dirichlet energy = xᵀ L x.
  3. Edge cases — zero edges, batched Dirichlet energy.
"""

from __future__ import annotations

import pytest
import torch

from holonomy_lib.sheaf import (
    GraphSheaf,
    HeterogeneousGraphSheaf,
    sheaf_coboundary,
    sheaf_dirichlet_energy,
    sheaf_laplacian,
)

DT = torch.float64


def _het(node_dims, edges_list, edge_dims=None, seed=0):
    g = torch.Generator().manual_seed(seed)
    n = len(node_dims)
    edges = torch.tensor(edges_list, dtype=torch.int64).reshape(-1, 2)
    node_sd = torch.tensor(node_dims, dtype=torch.int64)
    if edge_dims is None:
        edge_dims = [1] * len(edges_list)
    edge_sd = torch.tensor(edge_dims, dtype=torch.int64)
    F_left, F_right = [], []
    for i, (u, v) in enumerate(edges_list):
        de = edge_dims[i]
        F_left.append(torch.randn(de, node_dims[u], generator=g, dtype=DT))
        F_right.append(torch.randn(de, node_dims[v], generator=g, dtype=DT))
    return HeterogeneousGraphSheaf(n, edges, node_sd, edge_sd,
                                   tuple(F_left), tuple(F_right))


# --------------------------------------------------------------------
# Property: reduction to the tested uniform path
# --------------------------------------------------------------------


def test_reduces_to_uniform_when_homogeneous():
    d, n = 3, 4
    edges_list = [[0, 1], [1, 2], [2, 3], [0, 3]]
    edges = torch.tensor(edges_list, dtype=torch.int64)
    eye = torch.eye(d, dtype=DT)
    het = HeterogeneousGraphSheaf(
        n, edges,
        torch.full((n,), d, dtype=torch.int64),
        torch.full((len(edges_list),), d, dtype=torch.int64),
        tuple(eye.clone() for _ in edges_list),
        tuple(eye.clone() for _ in edges_list),
    )
    uni = GraphSheaf.trivial(n, edges, stalk_dim=d)
    torch.testing.assert_close(sheaf_laplacian(het), sheaf_laplacian(uni),
                               atol=1e-12, rtol=0)


def test_variable_dims_give_psd_laplacian_of_total_size():
    node_dims = [2, 3, 1, 4]
    het = _het(node_dims, [[0, 1], [1, 2], [2, 3], [0, 3]], edge_dims=[2, 1, 1, 3])
    L = sheaf_laplacian(het)
    total = sum(node_dims)                         # 10
    assert L.shape == (total, total)
    assert torch.allclose(L, L.mT, atol=1e-10)     # symmetric
    assert (torch.linalg.eigvalsh(L) >= -1e-9).all()   # PSD


def test_dirichlet_energy_matches_quadratic_form():
    het = _het([2, 3, 1], [[0, 1], [1, 2]], edge_dims=[2, 1])
    total = 6
    x = torch.randn(5, total, dtype=DT)
    e = sheaf_dirichlet_energy(het, x)
    assert e.shape == (5,)
    assert (e >= -1e-12).all()
    L = sheaf_laplacian(het)
    torch.testing.assert_close(e, torch.einsum("bi,ij,bj->b", x, L, x),
                               atol=1e-9, rtol=0)


# --------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------


class TestValidation:
    def test_rejects_F_left_shape_mismatch(self):
        edges = torch.tensor([[0, 1]], dtype=torch.int64)
        with pytest.raises(ValueError, match="F_left"):
            HeterogeneousGraphSheaf(
                2, edges,
                torch.tensor([2, 3], dtype=torch.int64),
                torch.tensor([2], dtype=torch.int64),
                (torch.randn(2, 99, dtype=DT),),     # wrong d_v[u]=2, got 99
                (torch.randn(2, 3, dtype=DT),),
            )

    def test_rejects_self_loop(self):
        with pytest.raises(ValueError, match="self-loop"):
            _het([2, 2], [[0, 0]])

    def test_rejects_duplicate_edges(self):
        with pytest.raises(ValueError, match="duplicate"):
            _het([2, 2, 2], [[0, 1], [0, 1]])

    def test_rejects_out_of_range_edge(self):
        # construct directly: the range check must fire before the F-shape loop
        # (which would otherwise index a nonexistent node's dim)
        with pytest.raises(ValueError, match="n_nodes"):
            HeterogeneousGraphSheaf(
                2, torch.tensor([[0, 5]], dtype=torch.int64),
                torch.tensor([2, 2], dtype=torch.int64),
                torch.tensor([1], dtype=torch.int64),
                (torch.randn(1, 2, dtype=DT),),
                (torch.randn(1, 2, dtype=DT),),
            )

    def test_rejects_bad_node_dims_shape(self):
        edges = torch.tensor([[0, 1]], dtype=torch.int64)
        with pytest.raises(ValueError, match="node_stalk_dims"):
            HeterogeneousGraphSheaf(
                2, edges,
                torch.tensor([2, 3, 4], dtype=torch.int64),   # 3 != n_nodes=2
                torch.tensor([2], dtype=torch.int64),
                (torch.randn(2, 2, dtype=DT),),
                (torch.randn(2, 3, dtype=DT),),
            )


# --------------------------------------------------------------------
# Edge cases
# --------------------------------------------------------------------


def test_zero_edges_gives_zero_laplacian():
    het = HeterogeneousGraphSheaf(
        2, torch.zeros(0, 2, dtype=torch.int64),
        torch.tensor([2, 3], dtype=torch.int64),
        torch.zeros(0, dtype=torch.int64),
        (), (),
    )
    L = sheaf_laplacian(het)
    assert L.shape == (5, 5)
    assert torch.allclose(L, torch.zeros(5, 5, dtype=DT))


# --------------------------------------------------------------------
# Regression: vectorized COO coboundary == reference per-edge loop
# --------------------------------------------------------------------


def _reference_coboundary(sheaf: HeterogeneousGraphSheaf) -> torch.Tensor:
    """The pre-vectorization per-edge slice-write loop, kept as the
    regression oracle for the COO-based assembly in `_coboundary_coo`."""
    node_dims = sheaf.node_stalk_dims.tolist()
    edge_dims = sheaf.edge_stalk_dims.tolist()
    uv = sheaf.edges.tolist()
    delta = torch.zeros(
        sum(edge_dims), sum(node_dims), dtype=sheaf.dtype, device=sheaf.device,
    )
    node_off, acc = [], 0
    for d in node_dims:
        node_off.append(acc)
        acc += d
    row = 0
    for e in range(sheaf.n_edges):
        u, v = uv[e]
        d_e = edge_dims[e]
        delta[row:row + d_e, node_off[u]:node_off[u] + node_dims[u]] = sheaf.F_left[e]
        delta[row:row + d_e, node_off[v]:node_off[v] + node_dims[v]] = -sheaf.F_right[e]
        row += d_e
    return delta


def test_vectorized_coboundary_matches_reference_loop():
    het = _het(
        [2, 3, 1, 4, 2],
        [[0, 1], [1, 2], [2, 3], [0, 3], [3, 4], [1, 4]],
        edge_dims=[2, 1, 3, 2, 1, 4],
        seed=7,
    )
    # atol=0: the assembly places values, no arithmetic happens.
    torch.testing.assert_close(
        sheaf_coboundary(het), _reference_coboundary(het), atol=0.0, rtol=0.0,
    )


def test_vectorized_coboundary_single_edge():
    het = _het([3, 2], [[0, 1]], edge_dims=[2], seed=1)
    torch.testing.assert_close(
        sheaf_coboundary(het), _reference_coboundary(het), atol=0.0, rtol=0.0,
    )


def test_vectorized_coboundary_reversed_edge_pair():
    # (u, v) and (v, u) are distinct edges (both allowed); their row blocks
    # stay disjoint so plain index assignment is exact.
    het = _het([2, 3], [[0, 1], [1, 0]], edge_dims=[2, 1], seed=3)
    torch.testing.assert_close(
        sheaf_coboundary(het), _reference_coboundary(het), atol=0.0, rtol=0.0,
    )
