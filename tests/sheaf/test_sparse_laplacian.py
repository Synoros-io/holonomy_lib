# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Synoros

"""Tests for the sparse sheaf path: sheaf_coboundary_sparse,
sheaf_laplacian_sparse, sheaf_bottom_spectrum.

Three layers:
  1. Unit — shapes, layouts, zero-edge sheaves, validation raises.
  2. Property — sparse ≡ dense on the same sheaf (the dense path is the
     tested oracle); L_F symmetric PSD; kernel dim = #components on the
     trivial sheaf; eigenpair residuals.
  3. Comparison — bottom-k nonzero modes against dense
     `torch.linalg.eigvalsh` with the kernel stripped.
"""

from __future__ import annotations

import pytest
import torch

from holonomy_lib.sheaf import (
    GraphSheaf,
    HeterogeneousGraphSheaf,
    sheaf_bottom_spectrum,
    sheaf_coboundary,
    sheaf_coboundary_sparse,
    sheaf_laplacian,
    sheaf_laplacian_sparse,
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


def _cycle_edges(n, offset=0):
    return [[offset + i, offset + (i + 1) % n] for i in range(n)]


# --------------------------------------------------------------------
# Property: sparse ≡ dense (the dense path is the tested oracle)
# --------------------------------------------------------------------


def test_sparse_coboundary_matches_dense_heterogeneous():
    het = _het([2, 3, 1, 4], [[0, 1], [1, 2], [2, 3], [0, 3]],
               edge_dims=[2, 1, 3, 2], seed=11)
    d_sp = sheaf_coboundary_sparse(het)
    assert d_sp.layout == torch.sparse_coo
    torch.testing.assert_close(
        d_sp.to_dense(), sheaf_coboundary(het), atol=0.0, rtol=0.0,
    )


def test_sparse_coboundary_matches_dense_uniform():
    edges = torch.tensor(_cycle_edges(6), dtype=torch.int64)
    uni = GraphSheaf.trivial(6, edges, stalk_dim=3)
    d_sp = sheaf_coboundary_sparse(uni)
    torch.testing.assert_close(
        d_sp.to_dense(), sheaf_coboundary(uni), atol=0.0, rtol=0.0,
    )


def test_sparse_laplacian_matches_dense():
    het = _het([2, 3, 1, 4, 2], [[0, 1], [1, 2], [2, 3], [3, 4], [0, 4]],
               edge_dims=[2, 1, 3, 1, 2], seed=12)
    L_sp = sheaf_laplacian_sparse(het)
    torch.testing.assert_close(
        L_sp.to_dense(), sheaf_laplacian(het), atol=1e-12, rtol=0,
    )


def test_sparse_laplacian_symmetric_psd():
    het = _het([2, 2, 3], [[0, 1], [1, 2], [0, 2]], edge_dims=[2, 2, 1],
               seed=13)
    L = sheaf_laplacian_sparse(het).to_dense()
    torch.testing.assert_close(L, L.mT, atol=1e-12, rtol=0)
    assert (torch.linalg.eigvalsh(L) >= -1e-9).all()


def test_zero_edges_sparse():
    het = HeterogeneousGraphSheaf(
        2, torch.zeros(0, 2, dtype=torch.int64),
        torch.tensor([2, 3], dtype=torch.int64),
        torch.zeros(0, dtype=torch.int64),
        (), (),
    )
    d_sp = sheaf_coboundary_sparse(het)
    assert d_sp.shape == (0, 5)
    L_sp = sheaf_laplacian_sparse(het)
    assert L_sp.shape == (5, 5)
    assert L_sp._nnz() == 0


# --------------------------------------------------------------------
# Bottom spectrum: kernel handling + comparison to dense eigh
# --------------------------------------------------------------------


def _trivial_two_components(n1=20, n2=15):
    """Trivial (stalk-dim-1) sheaf on two disjoint cycles: L_F is the
    combinatorial Laplacian with a 2-dim kernel (one constant vector
    per component)."""
    edges = torch.tensor(
        _cycle_edges(n1) + _cycle_edges(n2, offset=n1), dtype=torch.int64,
    )
    return GraphSheaf.trivial(n1 + n2, edges, stalk_dim=1)


def test_bottom_spectrum_drops_kernel_and_matches_dense():
    sheaf = _trivial_two_components()
    k = 4
    vals, vecs, offsets = sheaf_bottom_spectrum(
        sheaf, k, kernel_headroom=6,
    )
    L = sheaf_laplacian(sheaf)
    dense_vals = torch.linalg.eigvalsh(L)
    # Two connected components → 2-dim kernel; the returned modes are
    # the smallest NONZERO ones.
    torch.testing.assert_close(vals, dense_vals[2:2 + k], atol=1e-8, rtol=0)
    # Eigenpair residual against the dense operator.
    resid = (L @ vecs - vecs * vals.unsqueeze(0)).norm(dim=0)
    assert (resid < 1e-6).all()
    # Trivial sheaf, stalk dim 1: offsets are 0..n_nodes.
    assert offsets.tolist() == list(range(sheaf.n_nodes + 1))


def test_bottom_spectrum_keep_kernel():
    sheaf = _trivial_two_components()
    vals, vecs, _ = sheaf_bottom_spectrum(
        sheaf, 3, drop_kernel=False,
    )
    # Without dropping, the two kernel modes lead (λ ≈ 0).
    assert (vals[:2].abs() < 1e-8).all()
    assert vals[2] > 1e-6


def test_bottom_spectrum_raises_when_kernel_exceeds_headroom():
    # Ten disjoint cycles → 10-dim kernel; headroom 2 cannot clear it.
    edges_list = []
    for c in range(10):
        edges_list += _cycle_edges(5, offset=5 * c)
    sheaf = GraphSheaf.trivial(
        50, torch.tensor(edges_list, dtype=torch.int64), stalk_dim=1,
    )
    with pytest.raises(ValueError, match="kernel"):
        sheaf_bottom_spectrum(sheaf, 4, kernel_headroom=2)


def test_bottom_spectrum_heterogeneous_offsets():
    node_dims = [2, 3, 1, 4, 2, 2, 3, 1]
    edges_list = _cycle_edges(8) + [[0, 4], [2, 6]]
    het = _het(node_dims, edges_list,
               edge_dims=[2] * len(edges_list), seed=21)
    k = 3
    vals, vecs, offsets = sheaf_bottom_spectrum(het, k, kernel_headroom=8)
    total = sum(node_dims)
    assert vecs.shape == (total, k)
    assert offsets[0] == 0 and offsets[-1] == total
    assert offsets.diff().tolist() == node_dims
    # Comparison against the dense spectrum with the kernel stripped.
    L = sheaf_laplacian(het)
    dense_vals = torch.linalg.eigvalsh(L)
    thresh = L.diagonal().max() * 1e-9
    dense_nonzero = dense_vals[dense_vals > thresh]
    torch.testing.assert_close(vals, dense_nonzero[:k], atol=1e-8, rtol=0)


def test_bottom_spectrum_validation():
    sheaf = _trivial_two_components(6, 5)
    with pytest.raises(ValueError, match="k must be"):
        sheaf_bottom_spectrum(sheaf, 0)
    with pytest.raises(ValueError, match="kernel_headroom"):
        sheaf_bottom_spectrum(sheaf, 2, kernel_headroom=-1)
