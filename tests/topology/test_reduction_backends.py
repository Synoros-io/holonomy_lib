# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Synoros

"""Compare complete persistence-pair multisets across reduction backends.

Torch keeps columns on the filtration device while metadata and pivot
bookkeeping use the host. These checks concern numerical equivalence,
not performance or asynchronous GPU execution.
"""

from __future__ import annotations

import math

import pytest
import torch

from holonomy_lib.simplicial import vietoris_rips_sparse
from holonomy_lib.topology._filtration import build_filtration
from holonomy_lib.topology._reduction import reduce_filtration


def _seeded(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def _circle(n: int, radius: float = 1.0) -> torch.Tensor:
    theta = torch.linspace(0, 2 * math.pi, n + 1, dtype=torch.float64)[:-1]
    return torch.stack(
        [radius * torch.cos(theta), radius * torch.sin(theta)], dim=-1,
    )


def _pairs_sorted(pairs_by_dim, k):
    """Compare complete pairs without discarding repeated bars."""
    return torch.tensor(sorted(pairs_by_dim.get(k, [])),
                        dtype=torch.float64).reshape(-1, 2)


class TestReductionBackendsAgree:
    @pytest.mark.parametrize("fixture,n", [
        ("circle", 10), ("circle", 20), ("circle", 30),
        ("collinear", 3), ("random", 12),
    ])
    def test_circle_diagrams_match(self, fixture, n):
        if fixture == "circle":
            points, radius = _circle(n), 1.5
        elif fixture == "collinear":
            points = torch.tensor([[0., 0.], [1., 0.], [2., 0.]],
                                  dtype=torch.float64)
            radius = 3.
        else:
            points = torch.randn(n, 3, dtype=torch.float64, generator=_seeded(7))
            radius = 3.
        distances = torch.cdist(points, points)
        complex = vietoris_rips_sparse(distances, max_radius=radius, max_dim=2)
        filtration = build_filtration(distances, complex)
        python_pairs = reduce_filtration(filtration, backend="python")
        torch_pairs = reduce_filtration(filtration, backend="torch")
        for dim in (0, 1):
            torch.testing.assert_close(
                _pairs_sorted(python_pairs, dim), _pairs_sorted(torch_pairs, dim),
                atol=1e-12, rtol=0,
            )
        assert len(python_pairs[0]) == n
        if fixture == "circle":
            torch.testing.assert_close(
                _pairs_sorted(python_pairs, 1),
                torch.tensor([[2 * math.sin(math.pi / n), math.inf]],
                             dtype=torch.float64), atol=1e-12, rtol=0,
            )
        elif fixture == "collinear":
            assert python_pairs.get(1, []) == []
            assert python_pairs[0].count((0., 1.)) == 2

        # No 2-simplices means no H_1 bars killable; expect both
        # backends to agree (likely empty at dim 1).


class TestReductionBackendValidation:
    def test_unknown_backend_raises(self):
        pts = torch.zeros(3, 2, dtype=torch.float64)
        d = torch.cdist(pts, pts)
        complex = vietoris_rips_sparse(d, max_radius=1.0, max_dim=1)
        filt = build_filtration(d, complex)
        with pytest.raises(ValueError, match="backend"):
            reduce_filtration(filt, backend="cuda_kernel")  # type: ignore[arg-type]
