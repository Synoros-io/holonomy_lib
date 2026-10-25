# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Synoros

"""Benchmarks for holonomy_lib.topology.

Cases:
  - persistence_diagrams with reduction_backend="python" (CPython sets)
  - persistence_diagrams with reduction_backend="torch" (LongTensor cols)

The torch backend is NOT a custom CUDA kernel; it's a same-algorithm
port to torch ops so the reduction can stay on whatever device the
filtration lives on. For typical small inputs CPython sets are
competitive or faster (tight C ops). The crossover, if any, is at
sizes large enough that vectorized `unique` beats hashing; these
benchmarks measure that.
"""

from __future__ import annotations

import math

import torch

from holonomy_lib.topology import persistence_diagrams
from tests.benchmarks.harness import Bench

bench = Bench("topology")


def _seeded(seed: int) -> torch.Generator:
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    return g


def _circle(n: int, dtype, seed: int = 0) -> torch.Tensor:
    """n-point unit circle in 2D with mild Gaussian jitter."""
    theta = torch.linspace(0, 2 * math.pi, n + 1, dtype=dtype)[:-1]
    pts = torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)
    jitter = 0.05 * torch.randn(
        n, 2, generator=_seeded(seed), dtype=dtype,
    )
    return (pts + jitter).unsqueeze(0)


def _setup_persistence_python(size, device, dtype):
    pts = _circle(size["n"], dtype, seed=0).to(device)
    max_dim = size.get("max_dim", 1)
    max_radius = size.get("max_radius", 1.5)
    def fn():
        return persistence_diagrams(
            pts, max_dim=max_dim, max_radius=max_radius,
            reduction_backend="python",
        )
    return fn


def _setup_persistence_torch(size, device, dtype):
    pts = _circle(size["n"], dtype, seed=0).to(device)
    max_dim = size.get("max_dim", 1)
    max_radius = size.get("max_radius", 1.5)
    def fn():
        return persistence_diagrams(
            pts, max_dim=max_dim, max_radius=max_radius,
            reduction_backend="torch",
        )
    return fn


# Noisy circles span the regime where the algorithm's column count
# grows roughly with n^2 (max_radius controls VR cutoff).
_persistence_sizes = [
    {"n": 20,  "max_dim": 1, "max_radius": 1.5},
    {"n": 40,  "max_dim": 1, "max_radius": 1.0},
    {"n": 80,  "max_dim": 1, "max_radius": 0.7},
]

bench.case(
    "persistence_diagrams_python",
    _setup_persistence_python, _persistence_sizes,
    notes="VR + Z/2 reduction with CPython set columns. "
          "Reference for the torch-backend crossover.",
)
bench.case(
    "persistence_diagrams_torch",
    _setup_persistence_torch, _persistence_sizes,
    notes="VR + Z/2 reduction with torch.LongTensor columns. "
          "Device-agnostic; not yet a CUDA kernel — same algorithm, "
          "expect slower than CPython sets for small inputs.",
)
