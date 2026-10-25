# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Synoros

"""Benchmarks for holonomy_lib.spectral.

Laplacian construction is O(n²); eigh-based ops (`laplacian_eigenmaps`,
`effective_resistance`, `diffusion_map`) are O(n³). The Chebyshev
heat kernel is O(K · n²) per matmul (dense form). The magnetic
Laplacian adds the cost of a single complex outer product on top of
the real construction. The benchmarks here cover all those regimes
so optimization claims become measurable.
"""

from __future__ import annotations

import torch

from holonomy_lib.spectral import (
    diffusion_map,
    effective_resistance,
    heat_kernel_chebyshev,
    laplacian_eigenmaps,
    magnetic,
)
from holonomy_lib.spectral import (
    laplacian as _L,
)
from tests.benchmarks.harness import Bench

bench = Bench("spectral")

# Preserve the original signed-graph workload: negate 30% of undirected edges.
SIGNED_EDGE_PROBABILITY = 0.3


def _make_adj(size, device, dtype, signed=False):
    g = torch.Generator(device="cpu")
    g.manual_seed(0)
    A = torch.rand(size["B"], size["n"], size["n"], generator=g, dtype=dtype)
    A = (A + A.mT) * 0.5
    if signed:
        # Add some negative weights for signed Laplacian benchmarks.
        # Both directions of an edge need the same sign for a symmetric input.
        mask = torch.rand(A.shape, generator=g, device=g.device, dtype=dtype)
        mask = (mask < SIGNED_EDGE_PROBABILITY).triu(diagonal=1)
        mask = mask | mask.mT
        A = torch.where(mask, -A, A)
    A.diagonal(dim1=-2, dim2=-1).zero_()
    return A.to(device)


def _setup_combinatorial(size, device, dtype):
    A = _make_adj(size, device, dtype)
    def fn():
        return _L.combinatorial(A)
    return fn


def _setup_sym_norm(size, device, dtype):
    A = _make_adj(size, device, dtype)
    def fn():
        return _L.symmetric_normalized(A)
    return fn


def _setup_random_walk(size, device, dtype):
    A = _make_adj(size, device, dtype)
    def fn():
        return _L.random_walk(A)
    return fn


def _setup_signed(size, device, dtype):
    A = _make_adj(size, device, dtype, signed=True)
    def fn():
        return _L.signed(A)
    return fn


def _setup_eigenmaps(size, device, dtype):
    A = _make_adj(size, device, dtype)
    k = size["k"]
    def fn():
        return laplacian_eigenmaps(A, k=k, laplacian_type="symmetric_normalized")
    return fn


_laplacian_sizes = [
    {"B": 1,  "n": 16},
    {"B": 1,  "n": 64},
    {"B": 1,  "n": 256},
    {"B": 16, "n": 64},
    {"B": 1,  "n": 1024},
]
_eigenmap_sizes = [
    {"B": 1,  "n": 64,   "k": 8},
    {"B": 1,  "n": 256,  "k": 16},
    {"B": 16, "n": 64,   "k": 8},
    {"B": 1,  "n": 1024, "k": 32},
]


bench.case("laplacian.combinatorial", _setup_combinatorial, _laplacian_sizes,
            notes="L = D - A; diag_embed + subtract.")
bench.case("laplacian.symmetric_normalized", _setup_sym_norm, _laplacian_sizes,
            notes="L_sym = I - D^{-1/2} A D^{-1/2}; two broadcasts.")
bench.case("laplacian.random_walk", _setup_random_walk, _laplacian_sizes,
            notes="L_rw = I - D^{-1} A; one broadcast.")
bench.case("laplacian.signed", _setup_signed, _laplacian_sizes,
            notes="L^σ = D^{|σ|} - A; |.|-sum + diag_embed.")
bench.case("laplacian_eigenmaps", _setup_eigenmaps, _eigenmap_sizes,
            notes="Bottom-k eigenpairs via dense eigh of L_sym.")


# ----------------- magnetic Laplacian (directed graphs) -----------------

def _make_directed_adj(size, device, dtype):
    """Asymmetric weighted adjacency for the magnetic-Laplacian
    benchmarks; the asymmetry is what the phase factor actually
    interacts with."""
    g = torch.Generator(device="cpu")
    g.manual_seed(0)
    A = torch.rand(size["B"], size["n"], size["n"], generator=g, dtype=dtype)
    A.diagonal(dim1=-2, dim2=-1).zero_()
    return A.to(device)


def _setup_magnetic_combinatorial(size, device, dtype):
    A = _make_directed_adj(size, device, dtype)
    q = size.get("q", 0.25)
    def fn():
        return magnetic.combinatorial(A, q=q)
    return fn


def _setup_magnetic_sym_norm(size, device, dtype):
    A = _make_directed_adj(size, device, dtype)
    q = size.get("q", 0.25)
    def fn():
        return magnetic.symmetric_normalized(A, q=q)
    return fn


_magnetic_sizes = [
    {"B": 1,  "n": 16,  "q": 0.25},
    {"B": 1,  "n": 64,  "q": 0.25},
    {"B": 1,  "n": 256, "q": 0.25},
    {"B": 16, "n": 64,  "q": 0.25},
    # q = 0 hits the real-only short-circuit
    {"B": 1,  "n": 256, "q": 0.0},
]

bench.case("magnetic.combinatorial", _setup_magnetic_combinatorial,
            _magnetic_sizes,
            notes="Hermitian magnetic Laplacian for directed graphs; complex output.")
bench.case("magnetic.symmetric_normalized", _setup_magnetic_sym_norm,
            _magnetic_sizes,
            notes="Normalized magnetic Laplacian; spectrum in [0, 2].")


# ----------------- heat kernel via Chebyshev -----------------

def _setup_heat_kernel_dense(size, device, dtype):
    A = _make_adj(size, device, dtype)
    L = _L.symmetric_normalized(A)
    t = size.get("t", 1.0)
    K = size.get("K", 30)
    def fn():
        return heat_kernel_chebyshev(L, t=t, K=K)
    return fn


def _setup_heat_kernel_signal(size, device, dtype):
    A = _make_adj(size, device, dtype)
    L = _L.symmetric_normalized(A)
    g = torch.Generator(device="cpu")
    g.manual_seed(1)
    signal = torch.randn(
        size["B"], size["n"], size["k_signal"], generator=g, dtype=dtype,
    ).to(device)
    t = size.get("t", 1.0)
    K = size.get("K", 30)
    def fn():
        return heat_kernel_chebyshev(L, t=t, K=K, signal=signal)
    return fn


_heat_dense_sizes = [
    {"B": 1, "n": 32,  "t": 1.0, "K": 30},
    {"B": 1, "n": 128, "t": 1.0, "K": 30},
    {"B": 1, "n": 512, "t": 1.0, "K": 30},
    # Compare against ground truth via larger K
    {"B": 1, "n": 128, "t": 1.0, "K": 60},
]
_heat_signal_sizes = [
    {"B": 1, "n": 128, "k_signal": 4,  "t": 1.0, "K": 30},
    {"B": 1, "n": 512, "k_signal": 4,  "t": 1.0, "K": 30},
    {"B": 1, "n": 512, "k_signal": 64, "t": 1.0, "K": 30},
]

bench.case("heat_kernel_chebyshev_dense", _setup_heat_kernel_dense,
            _heat_dense_sizes,
            notes="exp(-t·L) as a dense (B, n, n) tensor via Chebyshev.")
bench.case("heat_kernel_chebyshev_signal", _setup_heat_kernel_signal,
            _heat_signal_sizes,
            notes="exp(-t·L) @ signal via the same Chebyshev recurrence; "
                  "should beat the dense path when k_signal << n.")


# ----------------- effective resistance -----------------

def _setup_effective_resistance(size, device, dtype):
    A = _make_adj(size, device, dtype)
    def fn():
        return effective_resistance(A)
    return fn


_resistance_sizes = [
    {"B": 1, "n": 32},
    {"B": 1, "n": 128},
    {"B": 1, "n": 512},
    {"B": 8, "n": 64},
]

bench.case("effective_resistance", _setup_effective_resistance,
            _resistance_sizes,
            notes="Pairwise R via Moore-Penrose pseudoinverse; "
                  "dense eigh + outer product.")


# ----------------- diffusion map -----------------

def _setup_diffusion_map(size, device, dtype):
    A = _make_adj(size, device, dtype)
    k = size["k"]
    t = size.get("t", 1)
    def fn():
        return diffusion_map(A, k=k, t=t)
    return fn


_diffusion_sizes = [
    {"B": 1, "n": 64,   "k": 8,  "t": 1},
    {"B": 1, "n": 256,  "k": 16, "t": 1},
    {"B": 1, "n": 1024, "k": 32, "t": 1},
]

bench.case("diffusion_map", _setup_diffusion_map, _diffusion_sizes,
            notes="Coifman-Lafon embedding with integer diffusion time; "
                  "dense symmetric eigendecomposition.")


# ----------------- sign-magnetic Laplacian (roadmap #1) -----------------

def _make_signed_directed_adj(size, device, dtype):
    """Signed-directed adjacency. Half the edges get sign-flipped to
    break the gauge-equivalence with the plain magnetic Laplacian."""
    g = torch.Generator(device="cpu")
    g.manual_seed(0)
    A = torch.rand(size["B"], size["n"], size["n"], generator=g, dtype=dtype)
    signs = torch.where(
        torch.rand(A.shape, generator=g, device=g.device, dtype=dtype) < 0.5,
        torch.ones_like(A), -torch.ones_like(A),
    )
    A = A * signs
    A.diagonal(dim1=-2, dim2=-1).zero_()
    return A.to(device)


def _setup_sign_magnetic_combinatorial(size, device, dtype):
    A = _make_signed_directed_adj(size, device, dtype)
    q = size.get("q", 0.25)
    def fn():
        return magnetic.sign_magnetic_combinatorial(A, q=q)
    return fn


def _setup_sign_magnetic_sym_norm(size, device, dtype):
    A = _make_signed_directed_adj(size, device, dtype)
    q = size.get("q", 0.25)
    def fn():
        return magnetic.sign_magnetic_symmetric_normalized(A, q=q)
    return fn


bench.case("magnetic.sign_magnetic_combinatorial",
            _setup_sign_magnetic_combinatorial, _magnetic_sizes,
            notes="Signed-directed Hermitian Laplacian; "
                  "one extra abs() vs plain magnetic.")
bench.case("magnetic.sign_magnetic_symmetric_normalized",
            _setup_sign_magnetic_sym_norm, _magnetic_sizes,
            notes="Normalized signed-directed form.")


# ----------------- sparse Laplacian backend (roadmap #5) -----------------

def _make_sparse_adj(size, device, dtype):
    """Sparse symmetric adjacency on `device` with target nnz ~ size['nnz']."""
    n = size["n"]
    g = torch.Generator(device="cpu")
    g.manual_seed(0)
    dense_full = torch.rand(n, n, generator=g, dtype=dtype)
    dense_full = 0.5 * (dense_full + dense_full.mT)
    # Threshold to control density.
    density = size.get("density", 0.01)
    dense_full = dense_full * (dense_full > 1.0 - density)
    dense_full.diagonal().zero_()
    return dense_full.to(device).to_sparse_coo()


def _setup_sparse_combinatorial(size, device, dtype):
    A_sparse = _make_sparse_adj(size, device, dtype)
    def fn():
        return _L.combinatorial(A_sparse)
    return fn


def _setup_sparse_sym_norm(size, device, dtype):
    A_sparse = _make_sparse_adj(size, device, dtype)
    def fn():
        return _L.symmetric_normalized(A_sparse)
    return fn


# Same densities for a dense baseline at matching size — measures the
# crossover. Sparse wins as n grows at fixed density.
_sparse_sizes = [
    {"n": 256,  "density": 0.05},
    {"n": 1024, "density": 0.01},
    {"n": 4096, "density": 0.003},
]


bench.case("laplacian.combinatorial_sparse",
            _setup_sparse_combinatorial, _sparse_sizes,
            notes="Sparse-COO combinatorial Laplacian.")
bench.case("laplacian.symmetric_normalized_sparse",
            _setup_sparse_sym_norm, _sparse_sizes,
            notes="Sparse-COO L_sym; same crossover pattern.")
