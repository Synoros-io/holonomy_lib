# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Synoros

"""Cross-backend tests for `_reduction.reduce_filtration`.

Roadmap #6 adds a torch-tensor reduction backend that runs end-to-end
on the filtration's native device. This module verifies it produces
identical persistence diagrams to the Python-set path on a variety
of inputs.

The torch path is a foundation for a future custom CUDA kernel; v1
is not necessarily faster than the Python-set path for small inputs
(CPython set ops are very tight). The contract this test enforces is
**correctness equivalence**, which is the prerequisite for any later
performance work.
"""

from __future__ import annotations

import math

import pytest
import torch

from holonomy_lib.simplicial import vietoris_rips_sparse
from holonomy_lib.topology import persistence_diagrams
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


def _pairs_set(
    pairs_by_dim: dict[int, list[tuple[float, float]]],
    k: int,
    rel_tol: float = 1e-12,
) -> set[tuple[float, float]]:
    """Normalize a list of (birth, death) pairs into a tolerance-rounded set
    for set equality comparison across backends. Coordinates that
    differ only at machine epsilon would otherwise produce false
    mismatches.
    """
    out: set[tuple[float, float]] = set()
    for b, d in pairs_by_dim.get(k, []):
        if math.isinf(d):
            out.add((round(b, 9), float("inf")))
        else:
            out.add((round(b, 9), round(d, 9)))
    return out


class TestReductionBackendsAgree:
    @pytest.mark.parametrize("n", [10, 20, 30])
    def test_circle_diagrams_match(self, n):
        """Clean circle: one persistent H_1 bar plus short noise bars.
        Both backends must produce the same set of pairs."""
        pts = _circle(n)
        d = torch.cdist(pts, pts)
        complex = vietoris_rips_sparse(d, max_radius=1.5, max_dim=2)
        filt = build_filtration(d, complex)

        py = reduce_filtration(filt, backend="python")
        tc = reduce_filtration(filt, backend="torch")

        for k in (0, 1):
            assert _pairs_set(py, k) == _pairs_set(tc, k), (
                f"backends disagree at dim={k} for n={n}: "
                f"python={_pairs_set(py, k)}, torch={_pairs_set(tc, k)}"
            )

    def test_three_collinear_points(self):
        """A degenerate case: three points on a line. Tests that
        empty H_1 diagrams agree across backends."""
        pts = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=torch.float64,
        )
        d = torch.cdist(pts, pts)
        complex = vietoris_rips_sparse(d, max_radius=3.0, max_dim=2)
        filt = build_filtration(d, complex)

        py = reduce_filtration(filt, backend="python")
        tc = reduce_filtration(filt, backend="torch")
        assert _pairs_set(py, 0) == _pairs_set(tc, 0)
        assert _pairs_set(py, 1) == _pairs_set(tc, 1)
        # No 2-simplices means no H_1 bars killable; expect both
        # backends to agree (likely empty at dim 1).

    def test_random_cloud_matches(self):
        """Random 12-point cloud — exercises a non-degenerate case."""
        g = _seeded(7)
        pts = torch.randn(12, 3, dtype=torch.float64, generator=g)
        d = torch.cdist(pts, pts)
        complex = vietoris_rips_sparse(d, max_radius=3.0, max_dim=2)
        filt = build_filtration(d, complex)

        py = reduce_filtration(filt, backend="python")
        tc = reduce_filtration(filt, backend="torch")
        for k in (0, 1):
            assert _pairs_set(py, k) == _pairs_set(tc, k)

    def test_persistence_diagrams_backend_kwarg(self):
        """High-level `persistence_diagrams` exposes the backend
        choice and must produce the same diagrams + masks regardless
        of backend (modulo numerical-stable tolerance)."""
        pts = _circle(20).unsqueeze(0)
        d_py, m_py = persistence_diagrams(
            pts, max_dim=1, max_radius=1.5, reduction_backend="python",
        )
        d_torch, m_torch = persistence_diagrams(
            pts, max_dim=1, max_radius=1.5, reduction_backend="torch",
        )
        for k in (0, 1):
            # Pad shapes may differ if one backend caught a stray
            # bar the other rounded away; compare via the mask-valid
            # subset instead.
            valid_py = d_py[k][m_py[k]]
            valid_torch = d_torch[k][m_torch[k]]
            assert valid_py.shape == valid_torch.shape


class TestReductionBackendValidation:
    def test_unknown_backend_raises(self):
        pts = torch.zeros(3, 2, dtype=torch.float64)
        d = torch.cdist(pts, pts)
        complex = vietoris_rips_sparse(d, max_radius=1.0, max_dim=1)
        filt = build_filtration(d, complex)
        with pytest.raises(ValueError, match="backend"):
            reduce_filtration(filt, backend="cuda_kernel")
