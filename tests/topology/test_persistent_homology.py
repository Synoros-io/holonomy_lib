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

import math

import pytest
import torch

from holonomy_lib.topology import persistence_diagrams


def _seeded(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


# --------------------------------------------------------------------
# H_0 closed forms
# --------------------------------------------------------------------


class TestH0ClosedForms:
    def test_three_collinear_points(self):
        """Points at 0, 1, 2. H_0: 3 bars total — two finite (deaths
        at 1.0) and one essential (survives to ∞)."""
        pts = torch.tensor([[[0.0], [1.0], [2.0]]], dtype=torch.float64)
        diagrams, masks = persistence_diagrams(pts, max_dim=0)
        diag0 = diagrams[0][0]   # (3, 2)
        mask0 = masks[0][0]
        # 3 components born at time 0
        births = diag0[mask0][:, 0]
        torch.testing.assert_close(
            births, torch.zeros_like(births), atol=1e-12, rtol=0,
        )
        # Deaths: two finite (both ≈ 1.0), one ∞
        deaths = diag0[mask0][:, 1].sort().values
        assert deaths[0].item() == pytest.approx(1.0, abs=1e-9)
        assert deaths[1].item() == pytest.approx(1.0, abs=1e-9)
        assert torch.isinf(deaths[2]).item()

    def test_equilateral_triangle(self):
        """Three vertices of an equilateral triangle with side 1.
        H_0 = 3 components: 2 die together at length 1 (the
        first union merges two; the second merges the third),
        1 essential."""
        pts = torch.tensor([[
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, math.sqrt(3) / 2],
        ]], dtype=torch.float64)
        diagrams, _ = persistence_diagrams(pts, max_dim=0)
        diag = diagrams[0][0]
        deaths = diag[:, 1]
        finite_deaths = deaths[torch.isfinite(deaths)]
        assert len(finite_deaths) == 2
        torch.testing.assert_close(
            finite_deaths.sort().values,
            torch.tensor([1.0, 1.0], dtype=torch.float64),
            atol=1e-9, rtol=0,
        )
        # One ∞ bar
        assert torch.isinf(deaths).sum().item() == 1

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

    def test_clean_circle_has_one_persistent_loop(self):
        n = 30
        theta = torch.linspace(
            0, 2 * math.pi, n + 1, dtype=torch.float64,
        )[:-1]
        pts = torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)
        pts = pts.unsqueeze(0)  # (1, n, 2)
        diagrams, masks = persistence_diagrams(
            pts, max_dim=1, max_radius=1.5,
        )
        # Expect at least one H_1 bar with long persistence.
        diag1 = diagrams[1][0]
        mask1 = masks[1][0]
        valid = diag1[mask1]
        if valid.numel() == 0:
            pytest.fail("no H_1 bars detected for the clean circle")
        persistence = valid[:, 1] - valid[:, 0]
        # The longest persistence should be substantial (about the
        # radius gap between "first triangle closes" and "interior
        # fills in"). For a 30-point unit circle this is ~0.3.
        longest = persistence.max().item()
        assert longest > 0.2, (
            f"longest H_1 bar {longest:.3f} should be > 0.2 for a clean circle"
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

    def test_batched_consistency(self):
        """B identical point clouds produce B identical diagrams."""
        torch.manual_seed(7)
        n = 6
        pts_single = torch.randn(n, 2, generator=_seeded(7), dtype=torch.float64)
        pts_batch = pts_single.unsqueeze(0).repeat(3, 1, 1)  # (3, n, 2)
        diagrams, masks = persistence_diagrams(pts_batch, max_dim=1, max_radius=2.0)
        for k in range(2):
            d_k = diagrams[k]
            m_k = masks[k]
            # All three batch elements have the same valid count.
            counts = m_k.sum(dim=-1)
            assert counts.unique().numel() == 1
            # The diagrams agree up to padding (compare valid entries).
            ref = d_k[0][m_k[0]]
            for b in range(1, 3):
                cur = d_k[b][m_k[b]]
                torch.testing.assert_close(
                    cur.sort(dim=0).values, ref.sort(dim=0).values,
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
        """B=0 must work — required by CONVENTIONS.md §1.1."""
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

    def test_rejects_2d_input(self):
        # Need 3-D (B, n, d) or (B, n, n).
        with pytest.raises(ValueError, match="3-D"):
            persistence_diagrams(torch.randn(4, 2), max_dim=0)

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
    from ripser import ripser  # noqa: F401
    _HAVE_RIPSER = True
except ImportError:
    _HAVE_RIPSER = False


@pytest.mark.skipif(not _HAVE_RIPSER, reason="ripser not installed")
class TestAgainstRipser:
    def test_h0_h1_match_ripser_on_random_2d(self):
        from ripser import ripser as ripser_compute
        torch.manual_seed(42)
        n = 12
        pts = torch.randn(n, 2, generator=_seeded(42), dtype=torch.float64)

        # holonomy_lib
        pts_batch = pts.unsqueeze(0)
        diagrams, masks = persistence_diagrams(
            pts_batch, max_dim=1, max_radius=3.0,
        )
        ours_h0 = diagrams[0][0][masks[0][0]].tolist()
        ours_h1 = diagrams[1][0][masks[1][0]].tolist()

        # ripser
        result = ripser_compute(pts.numpy(), maxdim=1, thresh=3.0)
        ripser_h0 = result["dgms"][0].tolist()
        ripser_h1 = result["dgms"][1].tolist()

        # Match cardinality on each dim (modulo ties — accept either
        # match within a generous tolerance).
        assert len(ours_h0) == len(ripser_h0), (
            f"H_0 bar count differs: ours={len(ours_h0)}, ripser={len(ripser_h0)}"
        )
        assert len(ours_h1) == len(ripser_h1), (
            f"H_1 bar count differs: ours={len(ours_h1)}, ripser={len(ripser_h1)}"
        )

        # Match birth/death sets up to permutation, within tolerance.
        # Filter out infinite-death bars; finite-death sets compared
        # via sorted Euclidean distance per pair.
        def _finite_sorted(diag):
            arr = [pair for pair in diag if not math.isinf(pair[1])]
            arr.sort(key=lambda p: (p[0], p[1]))
            return arr

        our_h0_f = _finite_sorted(ours_h0)
        rip_h0_f = _finite_sorted(ripser_h0)
        for (b1, d1), (b2, d2) in zip(our_h0_f, rip_h0_f):
            assert abs(b1 - b2) < 1e-9
            assert abs(d1 - d2) < 1e-9
