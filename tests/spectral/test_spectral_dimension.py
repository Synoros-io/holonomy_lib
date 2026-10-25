# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Synoros

"""Tests for holonomy_lib.spectral.spectral_dimension.

Three layers:
  1. Empty batches and input validation.
  2. The log-log slope mechanics exactly (two-point closed form);
     physical `d_s ≈ 1` (1-D, λ ~ k²) and `d_s ≈ 2` (2-D, λ ~ k²+l²) on
     constructed continuum-proxy spectra. The latter are asymptotic estimates,
     so the tolerance is loose enough to separate d_s ≈ 1 from 2 / 0.
"""

from __future__ import annotations

import math

import pytest
import torch

from holonomy_lib.spectral import spectral_dimension

DT = torch.float64


# --------------------------------------------------------------------
# Shapes across B ∈ {0, 1, several}
# --------------------------------------------------------------------


@pytest.mark.parametrize("batch", [0])
class TestShapes:
    def test_shape(self, batch):
        eig = torch.rand(batch, 50, dtype=DT)
        t = torch.logspace(-3, -1, 6, dtype=DT)
        assert spectral_dimension(eig, t).shape == (batch,)


# --------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------


class TestValidation:
    def test_rejects_single_t(self):
        with pytest.raises(ValueError, match="2 sample"):
            spectral_dimension(torch.rand(1, 10, dtype=DT),
                               torch.tensor([0.1], dtype=DT))
        with pytest.raises(ValueError, match="log-time variance"):
            spectral_dimension(torch.tensor([[1.0, 2.0]], dtype=DT),
                               torch.tensor([1.0, 1.0], dtype=DT))
        with pytest.raises(ValueError, match="at least one mode"):
            spectral_dimension(torch.empty(1, 0, dtype=DT),
                               torch.tensor([1.0, 2.0], dtype=DT))


# --------------------------------------------------------------------
# Mechanics (exact) and shape of 1-D input
# --------------------------------------------------------------------


class TestMechanics:
    def test_loglog_slope_exact(self):
        # eigenvalues {0, 1}; two times => d_s = -2·(Δlog p / Δlog t), by hand.
        eig = torch.tensor([[0.0, 1.0], [1.0, 2.0]], dtype=DT)
        t = torch.tensor([0.5, 1.0], dtype=DT)
        p0 = 0.5 * (1.0 + math.exp(-0.5))
        p1 = 0.5 * (1.0 + math.exp(-1.0))
        slope = (math.log(p1) - math.log(p0)) / (math.log(1.0) - math.log(0.5))
        shifted_slope = slope - 0.5 / math.log(2.0)
        expected = torch.tensor([-2.0 * slope, -2.0 * shifted_slope], dtype=DT)
        torch.testing.assert_close(spectral_dimension(eig, t), expected,
                                   atol=1e-12, rtol=0)
        # At these times the second mode is negligible, so log p drops
        # by exactly 1000 to float64 precision despite exp(-1000) underflow.
        eig = torch.tensor([[1.0, 2.0]], dtype=DT)
        t = torch.tensor([1000.0, 2000.0], dtype=DT)
        expected = torch.tensor([2000.0 / math.log(2.0)], dtype=DT)
        torch.testing.assert_close(spectral_dimension(eig, t), expected)


# --------------------------------------------------------------------
# Physical: continuum-proxy spectra (asymptotic; loose tolerance)
# --------------------------------------------------------------------


class TestPhysical:
    def test_d_s_approx_1_for_1d_spectrum(self):
        # λ_k = k², k = 1..K mimics a 1-D continuum: p(t) ~ t^(-1/2) => d_s ≈ 1.
        k = torch.arange(1, 3001, dtype=DT)
        eig = (k * k).unsqueeze(0)                       # (1, 3000)
        t = torch.logspace(-5, -3, 8, dtype=DT)          # power-law window
        d_s = float(spectral_dimension(eig, t))
        assert abs(d_s - 1.0) < 0.1, d_s

    def test_d_s_approx_2_for_2d_spectrum(self):
        # λ = k²+l², k,l = 1..K mimics a 2-D continuum: p(t) ~ t^(-1) => d_s ≈ 2.
        k = torch.arange(1, 201, dtype=DT)
        kk = k * k
        eig = (kk[:, None] + kk[None, :]).reshape(1, -1)  # (1, 200²)
        t = torch.logspace(-4.3, -3.0, 8, dtype=DT)       # window: t·λ_max ≳ 1
        d_s = float(spectral_dimension(eig, t))
        assert abs(d_s - 2.0) < 0.2, d_s


# --------------------------------------------------------------------
# Comparison: recover known d_s (integer lattice + Sierpinski gasket)
# --------------------------------------------------------------------


def _sierpinski_gasket_eigs(level: int) -> torch.Tensor:
    """Combinatorial-Laplacian spectrum of the level-`level` Sierpinski
    gasket (right-triangle embedding, only adjacency matters)."""
    edges: set = set()

    def mid(p, q):
        return ((p[0] + q[0]) // 2, (p[1] + q[1]) // 2)

    def sub(a, b, c, lvl):
        if lvl == 0:
            for u, v in ((a, b), (b, c), (a, c)):
                edges.add((min(u, v), max(u, v)))
            return
        mab, mbc, mac = mid(a, b), mid(b, c), mid(a, c)
        sub(a, mab, mac, lvl - 1)
        sub(mab, b, mbc, lvl - 1)
        sub(mac, mbc, c, lvl - 1)

    s = 1 << level
    sub((0, 0), (s, 0), (0, s), level)
    verts = sorted({p for e in edges for p in e})
    idx = {p: i for i, p in enumerate(verts)}
    adj = torch.zeros((len(verts), len(verts)), dtype=DT)
    for u, v in edges:
        adj[idx[u], idx[v]] = 1.0
        adj[idx[v], idx[u]] = 1.0
    return torch.linalg.eigvalsh(torch.diag(adj.sum(1)) - adj)


def _tail_window(eigs: torch.Tensor) -> torch.Tensor:
    """Asymptotic-tail power-law window (rule from
    notes/validation/spectral_dimension_validation.py)."""
    eigs = eigs.clamp(min=0.0)
    n = eigs.numel()
    lam_max = float(eigs.max())
    lam_gap = float(eigs[eigs > 1e-9].min())
    floor = (eigs <= 1e-9).sum().item() / n
    grid = torch.logspace(math.log10(0.1 / lam_max), math.log10(2.0 / lam_gap),
                          1000, dtype=DT)
    p = torch.exp(-eigs[None, :] * grid[:, None]).mean(dim=1)
    above = grid[p > 4.0 * floor]
    t_hi = float(above.max()) if above.numel() else 1.0 / lam_gap
    t_lo = max(2.0, t_hi / 100.0)
    if t_lo >= t_hi:
        t_lo = t_hi / 30.0
    return torch.logspace(math.log10(t_lo), math.log10(t_hi), 80, dtype=DT)


class TestComparison:
    def test_sierpinski_gasket_recovers_fractional_dimension(self):
        # Known d_s = 2·ln3/ln5 ≈ 1.3652 (Rammal-Toulouse 1983), a genuinely
        # non-integer spectral dimension, the canonical cross-check.
        eigs = _sierpinski_gasket_eigs(6)            # ~1095 nodes, sub-second
        d_s = float(spectral_dimension(eigs, _tail_window(eigs)))
        known = 2.0 * math.log(3.0) / math.log(5.0)
        assert abs(d_s - known) < 0.1, (d_s, known)

    def test_ring_recovers_dimension_one(self):
        k = torch.arange(4096, dtype=DT)
        ring = 2.0 - 2.0 * torch.cos(2.0 * math.pi * k / 4096.0)
        d_s = float(spectral_dimension(ring, _tail_window(ring)))
        assert abs(d_s - 1.0) < 0.05, d_s
