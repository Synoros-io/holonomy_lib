# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Synoros

"""Tests for holonomy_lib.topology.persistent_homology.

Layers:
  1. H₀ closed forms (collinear points, equilateral triangle, etc.).
  2. H₀ stability under input perturbation.
  3. H₁ on noisy circles (one persistent loop, others die at small scale).
  4. Algebraic invariants (sum of bar lengths = total filtration mass).
  5. Comparison vs ripser/gudhi (importorskip).
"""

from __future__ import annotations

import importlib
import math

import pytest
import torch

from holonomy_lib.topology import persistence_diagrams


def _seeded(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def _sorted_pairs(pairs):
    """Order whole pairs, preserving both their pairing and multiplicity."""
    order = torch.argsort(pairs[:, 1], stable=True)
    order = order[torch.argsort(pairs[order, 0], stable=True)]
    return pairs[order]


# --------------------------------------------------------------------
# H_0 closed forms
# --------------------------------------------------------------------


class TestH0ClosedForms:
    @pytest.mark.parametrize("points", [
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
        [[0.0, 0.0], [1.0, 0.0], [0.5, math.sqrt(3) / 2]],
    ])
    @pytest.mark.parametrize("backend", ["python", "torch"])
    @pytest.mark.parametrize("dtype", [
        torch.float16, torch.bfloat16, torch.float32, torch.float64,
    ])
    def test_three_collinear_points(self, points, backend, dtype):
        """The line and filled equilateral triangle have no persistent H_1."""
        diagrams, masks = persistence_diagrams(
            torch.tensor([points], dtype=dtype), max_dim=1,
            reduction_backend=backend,
        )
        expected = torch.tensor([[0., 1.], [0., 1.], [0., math.inf]],
                                dtype=dtype)
        torch.testing.assert_close(
            _sorted_pairs(diagrams[0][masks[0]]), expected,
            atol=2 * torch.finfo(dtype).eps, rtol=0,
        )
        assert masks[1].sum() == 0


    def test_two_clusters_separated(self):
        """Two tight clusters far apart. H_0: 3 components die early
        (within each cluster) + 1 dies at the cluster gap + 1 ∞."""
        pts = torch.tensor([[
            [0.0, 0.0], [0.1, 0.0], [0.0, 0.1],   # cluster 1
            [10.0, 0.0], [10.1, 0.0], [10.0, 0.1],  # cluster 2
        ]], dtype=torch.float64)
        diagrams, _ = persistence_diagrams(pts, max_dim=0)
        diag = diagrams[0][0]
        deaths = diag[:, 1]
        finite_deaths = deaths[torch.isfinite(deaths)].sort().values
        # Within-cluster deaths: ~0.1 (4 of them, one per merge inside clusters);
        # cluster-bridge death: ~10.
        # Total finite deaths = n - 1 = 5.
        assert len(finite_deaths) == 5
        # The largest finite death must be the minimum cross-cluster
        # distance: from (0.1, 0) in cluster 1 to (10.0, 0) in cluster
        # 2, which is exactly 9.9.
        assert finite_deaths[-1].item() == pytest.approx(9.9, abs=1e-9)
        # All earlier deaths are ≤ √(0.01) ≈ 0.1 + a bit.
        assert finite_deaths[-2].item() < 1.0


# --------------------------------------------------------------------
# H_0 stability
# --------------------------------------------------------------------


class TestH0Stability:
    """The bottleneck distance between persistence diagrams is
    bounded by the L_inf distance between the input metrics
    (Cohen-Steiner-Edelsbrunner-Harer 2007). For a small Gaussian
    perturbation of a point cloud, each finite-death bar moves by at
    most ε."""

    def test_small_perturbation_keeps_deaths_close(self):
        torch.manual_seed(0)
        n = 8
        pts_base = torch.randn(1, n, 2, generator=_seeded(0), dtype=torch.float64)
        eps = 1e-3
        noise = eps * torch.randn(1, n, 2, generator=_seeded(1), dtype=torch.float64)
        pts_pert = pts_base + noise

        d_base, _ = persistence_diagrams(pts_base, max_dim=0)
        d_pert, _ = persistence_diagrams(pts_pert, max_dim=0)

        deaths_base = d_base[0][0, :, 1]
        deaths_pert = d_pert[0][0, :, 1]
        # Sort both finite-death sets and pair element-wise.
        fb = deaths_base[torch.isfinite(deaths_base)].sort().values
        fp = deaths_pert[torch.isfinite(deaths_pert)].sort().values
        assert fb.numel() == fp.numel()
        # Each paired bar moved by at most 2·eps (triangle inequality
        # on the Lipschitz bound).
        diffs = (fb - fp).abs()
        assert diffs.max().item() <= 4.0 * eps, (
            f"max bar shift {diffs.max().item():.3e} exceeds 4·eps={4 * eps}"
        )


# --------------------------------------------------------------------
# H_1 on a noisy circle
# --------------------------------------------------------------------


class TestH1Circle:
    """A point cloud sampled densely from a unit circle has one
    persistent H_1 bar (the loop survives until simplices fill the
    interior) and short bars from noise."""

    @pytest.mark.parametrize("backend", ["python", "torch"])
    def test_clean_circle_has_one_persistent_loop(self, backend):
        n = 30
        theta = torch.linspace(0, 2 * math.pi, n + 1, dtype=torch.float64)[:-1]
        pts = torch.stack([theta.cos(), theta.sin()], dim=-1).unsqueeze(0)
        diagrams, masks = persistence_diagrams(
            pts, max_dim=1, max_radius=1.5, reduction_backend=backend,
        )
        expected = torch.tensor([[2 * math.sin(math.pi / n), math.inf]],
                                dtype=torch.float64)
        torch.testing.assert_close(
            diagrams[1][masks[1]], expected, atol=1e-12, rtol=0,
        )
        # In a square the loop dies at the diagonal, an independent finite bar.
        square = torch.tensor([[[0., 0.], [1., 0.], [1., 1.], [0., 1.]]],
                              dtype=torch.float64)
        diagrams, masks = persistence_diagrams(
            square, max_dim=1, reduction_backend=backend,
        )
        torch.testing.assert_close(
            diagrams[1][masks[1]],
            torch.tensor([[1., math.sqrt(2)]], dtype=torch.float64),
            atol=1e-12, rtol=0,
        )


# --------------------------------------------------------------------
# Algebraic invariants
# --------------------------------------------------------------------


class TestAlgebraicInvariants:
    def test_h0_bar_count_equals_n(self):
        """H_0 has exactly `n` bars (n-1 finite + 1 infinite) for any
        connected-enough point cloud."""
        for n in [3, 5, 8, 12]:
            pts = torch.randn(1, n, 2, generator=_seeded(n), dtype=torch.float64)
            diagrams, masks = persistence_diagrams(pts, max_dim=0)
            count = masks[0][0].sum().item()
            assert count == n, (
                f"n={n}: expected {n} H_0 bars, got {count}"
            )

    @pytest.mark.parametrize("backend", ["python", "torch"])
    def test_batched_consistency(self, backend):
        """Distinct square scales must stay attached to their batch members."""
        square = torch.tensor([[0., 0.], [1., 0.], [1., 1.], [0., 1.]],
                              dtype=torch.float64)
        scales = torch.tensor([1., 1.2, 0.8], dtype=torch.float64)
        points = scales[:, None, None] * square
        diagrams, masks = persistence_diagrams(
            points, max_dim=1, max_radius=2., reduction_backend=backend,
        )
        for batch, scale in enumerate(scales.tolist()):
            single, single_masks = persistence_diagrams(
                points[batch:batch + 1], max_dim=1, max_radius=2.,
                reduction_backend=backend,
            )
            expected = [
                [[0., scale]] * 3 + [[0., math.inf]],
                [[scale, scale * math.sqrt(2)]],
            ]
            for dim in (0, 1):
                actual = _sorted_pairs(diagrams[dim][batch][masks[dim][batch]])
                torch.testing.assert_close(
                    actual, torch.tensor(expected[dim], dtype=torch.float64),
                    atol=1e-12, rtol=0,
                )
                torch.testing.assert_close(
                    actual, _sorted_pairs(single[dim][single_masks[dim]]),
                    atol=1e-12, rtol=0,
                )


# --------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------


class TestEdgeCases:
    def test_max_radius_zero(self):
        """No edges → every H_0 component is essential (infinite bar)."""
        pts = torch.randn(1, 5, 2, dtype=torch.float64, generator=_seeded(50))
        diagrams, masks = persistence_diagrams(pts, max_dim=0, max_radius=0.0)
        diag0 = diagrams[0][0][masks[0][0]]
        # All 5 bars are infinite
        deaths = diag0[:, 1]
        assert torch.isinf(deaths).all()
        # Cardinality equals n
        assert diag0.shape[0] == 5

    def test_batch_zero(self):
        """B=0 must work; required by CONVENTIONS.md §1.1."""
        pts = torch.zeros(0, 5, 2, dtype=torch.float64)
        diagrams, masks = persistence_diagrams(pts, max_dim=1)
        for k in range(2):
            assert diagrams[k].shape[0] == 0
            assert masks[k].shape[0] == 0


class TestValidation:
    def test_rejects_negative_max_dim(self):
        pts = torch.randn(1, 4, 2, dtype=torch.float64)
        with pytest.raises(ValueError, match="max_dim"):
            persistence_diagrams(pts, max_dim=-1)


    def test_distance_input(self):
        n = 4
        pts = torch.randn(1, n, 2, generator=_seeded(99), dtype=torch.float64)
        from holonomy_lib.simplicial import pairwise_distances
        d = pairwise_distances(pts)
        diag_from_pts, _ = persistence_diagrams(pts, max_dim=0)
        diag_from_d, _ = persistence_diagrams(
            d, max_dim=0, input_is_distance=True,
        )
        torch.testing.assert_close(
            diag_from_pts[0], diag_from_d[0], atol=1e-12, rtol=0,
        )


# --------------------------------------------------------------------
# Comparison vs ripser / gudhi (importorskip)
# --------------------------------------------------------------------


try:
    importlib.import_module("ripser")
    _HAVE_RIPSER = True
except ImportError:
    _HAVE_RIPSER = False


@pytest.mark.skipif(not _HAVE_RIPSER, reason="ripser not installed")
class TestAgainstRipser:
    def test_h0_h1_match_ripser_on_random_2d(self):
        ripser_compute = importlib.import_module("ripser").ripser
        torch.manual_seed(42)
        n = 12
        pts = torch.randn(n, 2, generator=_seeded(42), dtype=torch.float64)

        # holonomy_lib
        pts_batch = pts.unsqueeze(0)
        diagrams, masks = persistence_diagrams(
            pts_batch, max_dim=1, max_radius=3.0,
        )
        result = ripser_compute(pts.numpy(), maxdim=1, thresh=3.0)
        for dim in (0, 1):
            ours = _sorted_pairs(diagrams[dim][0][masks[dim][0]])
            reference = _sorted_pairs(torch.as_tensor(
                result["dgms"][dim], dtype=torch.float64,
            ))
            # Ripser computes in float32. Include births of essential bars.
            torch.testing.assert_close(ours, reference, atol=1e-6, rtol=1e-6)
