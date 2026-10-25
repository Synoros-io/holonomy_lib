# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Synoros

"""Benchmarks for holonomy_lib.algebra.

Cases:
  truncated_svd (exact mode)        — sizes m × n, target rank r.
  truncated_svd (randomized mode)   — sizes m × n, target rank r,
                                       oversample=5, n_iter=2.

Sizes span "small enough to be a hot inner-loop primitive" (50×50) to
"big enough that fancy algorithms matter" (1024×1024). All on a single
batch — see provenance benchmarks for batched-overhead cases.
"""

from __future__ import annotations

import torch

from holonomy_lib.algebra import truncated_svd
from tests.benchmarks.harness import Bench

bench = Bench("algebra")


def _make_M(size, device, dtype):
    """Build (B, m, n) random matrix on the requested device/dtype."""
    g = torch.Generator(device="cpu")
    g.manual_seed(0)
    # Generators are device-bound in PyTorch; for cpu-generated values we
    # then move to the target. This is fine for benchmarks because the
    # setup is not what we're timing.
    M = torch.randn(
        size["B"], size["m"], size["n"], generator=g, dtype=dtype,
    ).to(device)
    return M


def _setup_truncated_svd_exact(size, device, dtype):
    M = _make_M(size, device, dtype)
    r = size["r"]
    def fn():
        return truncated_svd(M, r=r, mode="exact")
    return fn


def _setup_truncated_svd_randomized(size, device, dtype):
    M = _make_M(size, device, dtype)
    r = size["r"]
    def fn():
        return truncated_svd(M, r=r, mode="randomized")
    return fn


_svd_sizes = [
    {"B": 1, "m": 64,   "n": 64,   "r": 8},
    {"B": 1, "m": 256,  "n": 256,  "r": 16},
    {"B": 1, "m": 1024, "n": 1024, "r": 32},
    # Asymmetric (tall) matrices — common in low-rank approximation.
    {"B": 1, "m": 2048, "n": 64,   "r": 8},
    # Batched — exposes whether the kernel handles batching well.
    {"B": 16, "m": 128, "n": 128, "r": 16},
]

bench.case(
    "truncated_svd_exact", _setup_truncated_svd_exact, _svd_sizes,
    notes="Full SVD then top-r slice (Eckart-Young exact).",
)
bench.case(
    "truncated_svd_randomized", _setup_truncated_svd_randomized, _svd_sizes,
    notes="Halko-Martinsson-Tropp randomized SVD with oversample=5, n_iter=2.",
)


# ----------------- Lanczos top-k -----------------

from holonomy_lib.algebra import lanczos_eigsh


def _setup_lanczos(size, device, dtype):
    """Symmetric A for Lanczos benchmarking."""
    g = torch.Generator(device="cpu"); g.manual_seed(0)
    A = torch.randn(size["B"], size["n"], size["n"], generator=g, dtype=dtype)
    A = 0.5 * (A + A.mT)
    A = A.to(device)
    k = size["k"]
    n_iter = size.get("n_iter", None)
    gen = torch.Generator(device="cpu"); gen.manual_seed(1)
    def fn():
        return lanczos_eigsh(A, k=k, n_iter=n_iter, generator=gen)
    return fn


def _setup_dense_eigh(size, device, dtype):
    """Reference: full eigendecomposition. Useful for comparison."""
    g = torch.Generator(device="cpu"); g.manual_seed(0)
    A = torch.randn(size["B"], size["n"], size["n"], generator=g, dtype=dtype)
    A = 0.5 * (A + A.mT)
    A = A.to(device)
    def fn():
        return torch.linalg.eigh(A)
    return fn


_lanczos_sizes = [
    # Small n, top-1 — Lanczos overhead vs eigh.
    {"B": 1, "n": 64,   "k": 1, "n_iter": 30},
    # Medium n, k=8 — typical eigenmap-style query.
    {"B": 1, "n": 256,  "k": 8, "n_iter": 40},
    # Large n, k=16 — where Lanczos should win.
    {"B": 1, "n": 1024, "k": 16, "n_iter": 60},
    # Batched
    {"B": 8, "n": 256,  "k": 8, "n_iter": 40},
]
_dense_eigh_sizes = [
    {"B": 1, "n": 64},
    {"B": 1, "n": 256},
    {"B": 1, "n": 1024},
    {"B": 8, "n": 256},
]

bench.case("lanczos_eigsh", _setup_lanczos, _lanczos_sizes,
            notes="Symmetric Lanczos with full reorthogonalization; "
                  "winning regime is k << n with modest n_iter.")
bench.case("dense_eigh_reference", _setup_dense_eigh, _dense_eigh_sizes,
            notes="torch.linalg.eigh on the SAME symmetric inputs; "
                  "the cost Lanczos should beat at large n.")


# ----------------- shift-and-invert Lanczos (SA mode, roadmap #2) -----------------


def _setup_lanczos_sa(size, device, dtype):
    """Symmetric Positive Definite A so σ=0 is a clean negative shift
    below the smallest eigenvalue."""
    g = torch.Generator(device="cpu"); g.manual_seed(0)
    n = size["n"]
    X = torch.randn(size["B"], n, n, generator=g, dtype=dtype)
    A = torch.matmul(X, X.mT) + n * torch.eye(n, dtype=dtype)
    A = A.to(device)
    k = size["k"]
    n_iter = size.get("n_iter", None)
    sigma = size.get("sigma", 0.0)
    gen = torch.Generator(device="cpu"); gen.manual_seed(1)
    def fn():
        return lanczos_eigsh(
            A, k=k, n_iter=n_iter, which="SA", sigma=sigma, generator=gen,
        )
    return fn


_lanczos_sa_sizes = [
    {"B": 1, "n": 64,   "k": 1, "n_iter": 20, "sigma": 0.0},
    {"B": 1, "n": 256,  "k": 8, "n_iter": 30, "sigma": 0.0},
    {"B": 1, "n": 1024, "k": 16, "n_iter": 40, "sigma": 0.0},
]

bench.case(
    "lanczos_eigsh_SA", _setup_lanczos_sa, _lanczos_sa_sizes,
    notes="Shift-and-invert Lanczos for smallest eigenvalues. "
          "LU-factor (A-σI) once, then per-iter lu_solve. "
          "Compare cost vs `lanczos_eigsh` (LA) and dense `eigvalsh`.",
)
