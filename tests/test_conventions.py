# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Synoros

"""Tests that pin down the library-wide conventions from CONVENTIONS.md.

The self-loop convention especially: every graph primitive must give
**identical** output regardless of whether the input adjacency
contains self-loops, because the diagonal is dropped at entry. This
is what makes the simple-graph contract universal across the library.
"""

from __future__ import annotations

import torch

from holonomy_lib.discrete_geometry import (
    forman_ricci_augmented,
    forman_ricci_simple,
    ollivier_ricci_curvature,
)
from holonomy_lib.spectral import (
    effective_resistance,
    magnetic,
)
from holonomy_lib.spectral import (
    laplacian as _L,
)


def _make_undirected(n: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator(); g.manual_seed(seed)
    A = torch.rand(1, n, n, generator=g, dtype=torch.float64)
    A = (A + A.mT) * 0.5
    A.diagonal(dim1=-2, dim2=-1).zero_()
    return A


def _make_directed(n: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator(); g.manual_seed(seed)
    A = torch.rand(1, n, n, generator=g, dtype=torch.float64)
    A.diagonal(dim1=-2, dim2=-1).zero_()
    return A


def _add_self_loops(A: torch.Tensor) -> torch.Tensor:
    """Add random self-loop weights to every node."""
    g = torch.Generator(); g.manual_seed(42)
    out = A.clone()
    diag = torch.rand(A.shape[0], A.shape[-1], generator=g, dtype=A.dtype)
    out.diagonal(dim1=-2, dim2=-1).copy_(diag)
    return out


# --------------------------------------------------------------------
# Self-loop invariance across all graph primitives
# --------------------------------------------------------------------


class TestSelfLoopInvariance:
    """Every graph primitive must produce the same output on
    `A` and on `A + diag(arbitrary)`. The diagonal is dropped at
    entry, so any self-loop weight is irrelevant."""

    def test_laplacian_combinatorial(self):
        A = _make_undirected(8, seed=1)
        A_with = _add_self_loops(A)
        torch.testing.assert_close(
            _L.combinatorial(A), _L.combinatorial(A_with),
            atol=1e-12, rtol=0,
        )

    def test_laplacian_symmetric_normalized(self):
        A = _make_undirected(8, seed=2)
        A_with = _add_self_loops(A)
        torch.testing.assert_close(
            _L.symmetric_normalized(A), _L.symmetric_normalized(A_with),
            atol=1e-12, rtol=0,
        )

    def test_laplacian_random_walk(self):
        A = _make_undirected(8, seed=3)
        A_with = _add_self_loops(A)
        torch.testing.assert_close(
            _L.random_walk(A), _L.random_walk(A_with),
            atol=1e-12, rtol=0,
        )

    def test_laplacian_signed(self):
        # Use signed weights for a non-trivial test.
        g = torch.Generator(); g.manual_seed(4)
        A = torch.randn(1, 6, 6, generator=g, dtype=torch.float64)
        A = (A + A.mT) * 0.5
        A.diagonal(dim1=-2, dim2=-1).zero_()
        A_with = _add_self_loops(A)
        torch.testing.assert_close(
            _L.signed(A), _L.signed(A_with),
            atol=1e-12, rtol=0,
        )

    def test_laplacian_degree(self):
        A = _make_undirected(6, seed=5)
        A_with = _add_self_loops(A)
        torch.testing.assert_close(
            _L.degree(A), _L.degree(A_with),
            atol=1e-12, rtol=0,
        )

    def test_magnetic_combinatorial(self):
        A = _make_directed(6, seed=6)
        A_with = _add_self_loops(A)
        torch.testing.assert_close(
            magnetic.combinatorial(A, q=0.25),
            magnetic.combinatorial(A_with, q=0.25),
            atol=1e-12, rtol=0,
        )

    def test_magnetic_symmetric_normalized(self):
        A = _make_directed(6, seed=7)
        A_with = _add_self_loops(A)
        torch.testing.assert_close(
            magnetic.symmetric_normalized(A, q=0.25),
            magnetic.symmetric_normalized(A_with, q=0.25),
            atol=1e-12, rtol=0,
        )

    def test_ollivier_ricci_curvature(self):
        # Small graph + few iters; we only need to verify the entry-
        # point handling, not numerical convergence here.
        A = _make_undirected(5, seed=8)
        A_with = _add_self_loops(A)
        k1 = ollivier_ricci_curvature(A, alpha=0.0, reg=0.05, n_iter=50)
        k2 = ollivier_ricci_curvature(A_with, alpha=0.0, reg=0.05, n_iter=50)
        torch.testing.assert_close(k1, k2, atol=1e-10, rtol=0)

    def test_forman_ricci_simple(self):
        A = _make_undirected(7, seed=9)
        A_with = _add_self_loops(A)
        torch.testing.assert_close(
            forman_ricci_simple(A), forman_ricci_simple(A_with),
            atol=1e-12, rtol=0,
        )

    def test_forman_ricci_augmented(self):
        A = _make_undirected(7, seed=10)
        A_with = _add_self_loops(A)
        torch.testing.assert_close(
            forman_ricci_augmented(A), forman_ricci_augmented(A_with),
            atol=1e-12, rtol=0,
        )

    def test_effective_resistance(self):
        A = _make_undirected(6, seed=11)
        A_with = _add_self_loops(A)
        torch.testing.assert_close(
            effective_resistance(A), effective_resistance(A_with),
            atol=1e-10, rtol=0,
        )
