# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Synoros

"""Benchmarks for hyperbolic means and experimental graph embeddings.

The eigenmaps case calls holonomy_lib.experimental. Its attraction-only
objective admits collapsed components, so runtime is not evidence of
embedding quality. It sweeps dense graphs with N in {50, 200, 500}.
The frechet_mean case sweeps N in {50, 500, 5000} points.
"""

from __future__ import annotations

import torch

from holonomy_lib.experimental import hyperbolic_laplacian_eigenmaps
from holonomy_lib.hyperbolic import frechet_mean
from holonomy_lib.manifolds import LorentzManifold
from tests.benchmarks.harness import Bench

bench = Bench("hyperbolic")


def _seeded(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def _setup_laplacian_eigenmaps(size, device, dtype):
    mfd = LorentzManifold(n=size["n"], device=device, dtype=dtype)
    g = _seeded(0)
    # Random symmetric adjacency, edge probability `p`
    A = (torch.rand(1, size["N"], size["N"], generator=g, dtype=dtype)
         < size["p"]).to(dtype)
    A = (A + A.mT) * 0.5
    A.diagonal(dim1=-2, dim2=-1).zero_()
    A = A.to(device)

    def fn():
        return hyperbolic_laplacian_eigenmaps(
            A, mfd, max_steps=size["max_steps"], lr=0.05,
            generator=_seeded(1),
        )
    return fn


def _setup_frechet_mean(size, device, dtype):
    mfd = LorentzManifold(n=size["n"], device=device, dtype=dtype)
    g = _seeded(0)
    P = mfd.random_point(batch_size=size["N"], generator=g).reshape(
        1, size["N"], mfd.ambient_dim,
    ).to(device)

    def fn():
        return frechet_mean(P, mfd, max_iter=size["max_iter"], tol=1e-9)
    return fn


_lap_sizes = [
    {"B": 1, "N": 50, "n": 4, "p": 0.3, "max_steps": 50},
    {"B": 1, "N": 200, "n": 4, "p": 0.3, "max_steps": 50},
    {"B": 1, "N": 500, "n": 4, "p": 0.3, "max_steps": 50},
]

_frechet_sizes = [
    {"B": 1, "N": 50, "n": 4, "max_iter": 50},
    {"B": 1, "N": 500, "n": 4, "max_iter": 50},
    {"B": 1, "N": 5000, "n": 4, "max_iter": 50},
]


bench.case(
    "hyperbolic_laplacian_eigenmaps", _setup_laplacian_eigenmaps, _lap_sizes,
    notes=("Experimental attraction-only RSGD with degree normalization; "
           "collapsed components minimize the objective. "
           "Timing only, not embedding quality; O(B*N²*D) per step."),
)
bench.case(
    "frechet_mean", _setup_frechet_mean, _frechet_sizes,
    notes=("Karcher iteration: weighted average of log_μ(p_i) per step. "
           "Pure batched log/exp on the manifold, O(N · D) per step."),
)
