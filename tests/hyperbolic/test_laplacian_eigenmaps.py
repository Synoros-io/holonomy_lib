# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Synoros

"""Tests for holonomy_lib.hyperbolic.hyperbolic_laplacian_eigenmaps."""

from __future__ import annotations

import pytest
import torch

from holonomy_lib.hyperbolic import hyperbolic_laplacian_eigenmaps
from holonomy_lib.manifolds import LorentzManifold


def _seed(s: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(s)
    return g


def _binary_tree_adjacency(depth: int) -> torch.Tensor:
    """Build a complete binary tree of given depth as a (1, N, N) unweighted
    symmetric adjacency tensor. Total nodes N = 2^{depth+1} − 1."""
    N = 2 ** (depth + 1) - 1
    A = torch.zeros(1, N, N, dtype=torch.float64)
    for parent in range(N):
        left = 2 * parent + 1
        right = 2 * parent + 2
        if left < N:
            A[0, parent, left] = 1.0
            A[0, left, parent] = 1.0
        if right < N:
            A[0, parent, right] = 1.0
            A[0, right, parent] = 1.0
    return A


def test_output_shape():
    mfd = LorentzManifold(n=3)
    A = _binary_tree_adjacency(depth=2)  # N = 7
    Y = hyperbolic_laplacian_eigenmaps(A, mfd, max_steps=20,
                                         generator=_seed(0))
    assert Y.shape == (1, 7, mfd.n + 1)


def test_output_on_manifold():
    mfd = LorentzManifold(n=3)
    A = _binary_tree_adjacency(depth=2)
    Y = hyperbolic_laplacian_eigenmaps(A, mfd, max_steps=30,
                                         generator=_seed(1))
    assert mfd.is_on_manifold(Y.reshape(-1, mfd.n + 1)).all()


def test_loss_decreases():
    """Random init → optimized embedding should have strictly lower
    Laplacian energy."""
    mfd = LorentzManifold(n=3)
    A = _binary_tree_adjacency(depth=2)
    N = A.shape[-1]

    def lap_energy(Y):
        # Σ A_ij · d_M²(Y_i, Y_j)
        Yi = Y.unsqueeze(2).expand(1, N, N, mfd.n + 1)
        Yj = Y.unsqueeze(1).expand(1, N, N, mfd.n + 1)
        d = mfd.distance(
            Yi.reshape(-1, mfd.n + 1), Yj.reshape(-1, mfd.n + 1),
        ).reshape(1, N, N)
        return (A * d * d).sum(dim=(-2, -1))

    Y0 = mfd.random_point(batch_size=N, generator=_seed(2)).reshape(
        1, N, mfd.n + 1,
    )
    Yf = hyperbolic_laplacian_eigenmaps(
        A, mfd, max_steps=100, lr=0.05, init=Y0.clone(),
    )
    assert (lap_energy(Yf) < lap_energy(Y0)).all()


def test_rejects_bad_adjacency_shape():
    mfd = LorentzManifold(n=3)
    A_bad = torch.zeros(5, 4)  # not (B, N, N)
    with pytest.raises(ValueError, match="adjacency"):
        hyperbolic_laplacian_eigenmaps(A_bad, mfd)


def test_rejects_bad_max_steps_lr():
    mfd = LorentzManifold(n=3)
    A = _binary_tree_adjacency(depth=1)
    with pytest.raises(ValueError, match="max_steps"):
        hyperbolic_laplacian_eigenmaps(A, mfd, max_steps=-1)
    with pytest.raises(ValueError, match="lr"):
        hyperbolic_laplacian_eigenmaps(A, mfd, lr=0.0)


def test_zero_steps_returns_init():
    """max_steps=0 should leave the init untouched."""
    mfd = LorentzManifold(n=2)
    A = _binary_tree_adjacency(depth=1)  # N = 3
    init = mfd.random_point(batch_size=3, generator=_seed(3)).reshape(
        1, 3, mfd.n + 1,
    )
    Y = hyperbolic_laplacian_eigenmaps(A, mfd, max_steps=0, init=init.clone())
    torch.testing.assert_close(Y, init, atol=0, rtol=0)


# --------------------------------------------------------------------
# Cross-manifold: laplacian_eigenmaps on KappaStereographicManifold
# --------------------------------------------------------------------


@pytest.mark.parametrize("k", [-1.0, 0.0, 1.0])
def test_works_on_kappa_stereographic(k):
    """hyperbolic_laplacian_eigenmaps now uses `manifold.ambient_dim`
    so it works on κ-stereographic without modification."""
    from holonomy_lib.manifolds import KappaStereographicManifold

    mfd = KappaStereographicManifold(n=3, kappa=k)
    A = _binary_tree_adjacency(depth=2)  # N = 7
    Y = hyperbolic_laplacian_eigenmaps(A, mfd, max_steps=20,
                                         lr=0.05, generator=_seed(40))
    assert Y.shape == (1, 7, mfd.ambient_dim)
    assert mfd.is_on_manifold(Y.reshape(-1, mfd.ambient_dim)).all()


# --------------------------------------------------------------------
# Scale invariance — the bug that motivated the degree normalization
# --------------------------------------------------------------------


def test_dense_graph_does_not_nan():
    """Dense random graph (N=50, n_edges ≳ 700) used to silently
    produce all-NaN output with default lr=0.05 + max_steps=200. The
    degree normalization makes lr scale-invariant so this works."""
    from holonomy_lib.manifolds import LorentzManifold

    torch.manual_seed(0)
    N = 50
    A = (torch.rand(1, N, N) > 0.6).double()
    A = (A + A.mT) * 0.5
    A.diagonal(dim1=-2, dim2=-1).zero_()
    n_edges = ((A > 0).sum().item() // 2)
    assert n_edges > 500, f"test fixture has only {n_edges} edges"

    mfd = LorentzManifold(n=4, k=-1.0)
    Y = hyperbolic_laplacian_eigenmaps(A, mfd, max_steps=200, lr=0.05,
                                         generator=_seed(50))
    assert torch.isfinite(Y).all(), (
        f"NaN={torch.isnan(Y).sum().item()}, "
        f"Inf={torch.isinf(Y).sum().item()}"
    )
    assert mfd.is_on_manifold(Y.reshape(-1, mfd.ambient_dim)).all()


def test_diverging_lr_fails_loud():
    """If lr is set absurdly high so divergence is unavoidable, the
    function should RAISE a RuntimeError rather than silently return
    NaN output."""
    import pytest

    from holonomy_lib.manifolds import LorentzManifold

    torch.manual_seed(1)
    N = 30
    A = (torch.rand(1, N, N) > 0.5).double()
    A = (A + A.mT) * 0.5
    A.diagonal(dim1=-2, dim2=-1).zero_()
    mfd = LorentzManifold(n=4, k=-1.0)
    # lr = 1e3 will overflow cosh/sinh in exp regardless of normalization
    with pytest.raises(RuntimeError, match="diverged"):
        hyperbolic_laplacian_eigenmaps(A, mfd, max_steps=20, lr=1e3,
                                          generator=_seed(51))


def test_degree_variance_does_not_break_convergence():
    """Star graph (one hub of degree N-1, leaves of degree 1) — the
    classic worst case for non-normalized gradient SGD. Normalization
    should keep the hub from overshooting."""
    from holonomy_lib.manifolds import LorentzManifold

    N = 20
    A = torch.zeros(1, N, N, dtype=torch.float64)
    # Hub is node 0; all others connect only to it
    A[0, 0, 1:] = 1.0
    A[0, 1:, 0] = 1.0
    mfd = LorentzManifold(n=3, k=-1.0)
    Y = hyperbolic_laplacian_eigenmaps(A, mfd, max_steps=100, lr=0.05,
                                         generator=_seed(52))
    assert torch.isfinite(Y).all()
    assert mfd.is_on_manifold(Y.reshape(-1, mfd.ambient_dim)).all()
