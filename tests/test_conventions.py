# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Synoros

"""Tests that pin down the library-wide conventions from CONVENTIONS.md.

The self-loop convention especially: every graph primitive must give
**identical** output regardless of whether the input adjacency
contains self-loops, because the diagonal is dropped at entry. This
is what makes the simple-graph contract universal across the library.
"""

from __future__ import annotations

import pytest
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
    g = torch.Generator().manual_seed(seed)
    A = torch.rand(1, n, n, generator=g, dtype=torch.float64)
    A = (A + A.mT) * 0.5
    A.diagonal(dim1=-2, dim2=-1).zero_()
    return A


def _make_directed(n: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    A = torch.rand(1, n, n, generator=g, dtype=torch.float64)
    A.diagonal(dim1=-2, dim2=-1).zero_()
    return A


def _add_self_loops(A: torch.Tensor) -> torch.Tensor:
    """Add random self-loop weights to every node."""
    g = torch.Generator().manual_seed(42)
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

    @pytest.mark.parametrize("function,kind,n,seed,kwargs,atol", [
        (_L.combinatorial, "undirected", 8, 1, {}, 1e-12),
        (_L.symmetric_normalized, "undirected", 8, 2, {}, 1e-12),
        (_L.random_walk, "undirected", 8, 3, {}, 1e-12),
        (_L.signed, "signed", 6, 4, {}, 1e-12),
        (_L.degree, "undirected", 6, 5, {}, 1e-12),
        (magnetic.combinatorial, "directed", 6, 6, {"q": 0.25}, 1e-12),
        (magnetic.symmetric_normalized, "directed", 6, 7, {"q": 0.25}, 1e-12),
        (ollivier_ricci_curvature, "undirected", 5, 8,
         {"alpha": 0., "reg": 0.05, "n_iter": 50}, 1e-10),
        (forman_ricci_simple, "undirected", 7, 9, {}, 1e-12),
        (forman_ricci_augmented, "binary", 7, 10, {}, 1e-12),
        (effective_resistance, "undirected", 6, 11, {}, 1e-10),
    ])
    def test_laplacian_combinatorial(self, function, kind, n, seed, kwargs, atol):
        if kind == "directed":
            adjacency = _make_directed(n, seed=seed)
        elif kind == "signed":
            generator = torch.Generator().manual_seed(seed)
            adjacency = torch.randn(1, n, n, generator=generator,
                                    dtype=torch.float64)
            adjacency = (adjacency + adjacency.mT) * 0.5
            adjacency.diagonal(dim1=-2, dim2=-1).zero_()
        else:
            adjacency = _make_undirected(n, seed=seed)
        if kind == "binary":
            adjacency = (adjacency > 0).to(adjacency.dtype)
        fixtures = [adjacency]
        if kind == "binary":
            # Preserve the augmented-Forman triangle-plus-isolate fixture.
            triangle = torch.zeros(1, 4, 4, dtype=torch.float64)
            triangle[:, :3, :3] = 1. - torch.eye(3, dtype=torch.float64)
            fixtures.append(triangle)
        for adjacency in fixtures:
            torch.testing.assert_close(
                function(adjacency, **kwargs),
                function(_add_self_loops(adjacency), **kwargs),
                atol=atol, rtol=0,
            )
