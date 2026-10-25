# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Synoros

"""Benchmarks for holonomy_lib.discrete_geometry (Ollivier + flow + surgery).

The Ollivier curvature is THE bottleneck: O(B·n³·n_iter) for Sinkhorn
with a materialized (B, n², n) pair structure. The flow primitives
call Ollivier repeatedly, multiplying the cost. We sweep small n only
on first pass; once memory tiling lands we'll expand.
"""

from __future__ import annotations

import torch

from holonomy_lib.discrete_geometry import (
    discrete_ricci_flow,
    forman_ricci_augmented,
    forman_ricci_simple,
    ollivier_ricci_curvature,
    ricci_flow_with_surgery,
)
from tests.benchmarks.harness import Bench

bench = Bench("discrete_geometry")


def _make_adj(size, device, dtype):
    n = size["n"]
    B = size["B"]
    g = torch.Generator(device="cpu"); g.manual_seed(0)
    # Erdős–Rényi-like random weighted graph
    A = (torch.rand(B, n, n, generator=g, dtype=dtype) * (
        torch.rand(B, n, n, generator=g, dtype=dtype) < 0.4
    ).to(dtype))
    A = (A + A.mT) * 0.5
    A.diagonal(dim1=-2, dim2=-1).zero_()
    return A.to(device)


def _setup_ollivier(size, device, dtype):
    A = _make_adj(size, device, dtype)
    n_iter = size.get("n_iter", 100)
    def fn():
        return ollivier_ricci_curvature(A, alpha=0.0, reg=0.01, n_iter=n_iter)
    return fn


def _setup_flow(size, device, dtype):
    A = _make_adj(size, device, dtype)
    n_steps = size.get("n_steps", 5)
    def fn():
        return discrete_ricci_flow(
            A, n_steps=n_steps, dt=0.5, alpha=0.0, normalize=True,
            reg=0.01, n_sinkhorn_iters=size.get("n_iter", 50),
        )
    return fn


def _setup_surgery(size, device, dtype):
    A = _make_adj(size, device, dtype)
    def fn():
        return ricci_flow_with_surgery(
            A, n_steps=size.get("n_steps", 5),
            surgery_period=2, surgery_threshold=3.0,
            dt=0.5, alpha=0.0, normalize=True,
            reg=0.01, n_sinkhorn_iters=size.get("n_iter", 50),
        )
    return fn


_ollivier_sizes = [
    {"B": 1, "n": 16,  "n_iter": 100},
    {"B": 1, "n": 32,  "n_iter": 100},
    {"B": 1, "n": 64,  "n_iter": 100},
    {"B": 4, "n": 16,  "n_iter": 100},
]
_flow_sizes = [
    {"B": 1, "n": 16, "n_steps": 5, "n_iter": 50},
    {"B": 1, "n": 32, "n_steps": 5, "n_iter": 50},
]
_surgery_sizes = [
    {"B": 1, "n": 16, "n_steps": 10, "n_iter": 50},
    {"B": 1, "n": 32, "n_steps": 10, "n_iter": 50},
]


bench.case("ollivier_ricci_curvature", _setup_ollivier, _ollivier_sizes,
            notes="Sinkhorn-based pairwise W_1 with shortest-path metric.")
bench.case("discrete_ricci_flow", _setup_flow, _flow_sizes,
            notes="Edge-weight evolution: w *= (1 - dt*κ) per step.")
bench.case("ricci_flow_with_surgery", _setup_surgery, _surgery_sizes,
            notes="Flow + periodic threshold-based edge removal.")


# ----------------- Forman-Ricci (combinatorial) -----------------

def _setup_forman_simple(size, device, dtype):
    A = _make_adj(size, device, dtype)
    def fn():
        return forman_ricci_simple(A)
    return fn


def _setup_forman_augmented(size, device, dtype):
    A = _make_adj(size, device, dtype)
    def fn():
        return forman_ricci_augmented(A)
    return fn


_forman_sizes = [
    {"B": 1,  "n": 16},
    {"B": 1,  "n": 64},
    {"B": 1,  "n": 256},
    {"B": 1,  "n": 1024},
    {"B": 16, "n": 64},
]

bench.case("forman_ricci_simple", _setup_forman_simple, _forman_sizes,
            notes="Combinatorial Forman-Ricci; O(B*n^2). No OT solve.")
bench.case("forman_ricci_augmented", _setup_forman_augmented, _forman_sizes,
            notes="Augmented form adds the +3*#triangles term; "
                  "triangle-count matmul makes this O(B*n^3).")
