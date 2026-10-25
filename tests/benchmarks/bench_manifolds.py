# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Synoros

"""Benchmarks for holonomy_lib.manifolds (FixedRank, SPD).

The manifold methods are the inner-loop ops of any Riemannian
optimizer, so per-call latency matters. Cases:

  FixedRankManifold:
    projection      — given a point and an ambient direction, project.
    retraction      — currently full SVD; we want this faster.
    inner / norm    — should be near-zero overhead.

  SPDManifold:
    exp / log / distance  — all do eigh; cache opportunities.
    inner                  — two solves; possibly Cholesky-based.
    is_spd                 — eigh + reductions.
"""

from __future__ import annotations

import torch

from holonomy_lib.manifolds import FixedRankManifold, SPDManifold
from holonomy_lib.manifolds.comparison import model_ball_volume
from tests.benchmarks.harness import Bench

bench = Bench("manifolds")


# ----------------- FixedRank -----------------

def _setup_fr_projection(size, device, dtype):
    mfd = FixedRankManifold(
        m=size["m"], n=size["n"], r=size["r"], device=device, dtype=dtype,
    )
    g = torch.Generator(device="cpu"); g.manual_seed(0)
    pt = mfd.random_point(batch_size=size["B"], generator=g)
    # Move to device
    pt = tuple(t.to(device) for t in pt)
    Z = torch.randn(size["B"], size["m"], size["n"], generator=g,
                     dtype=dtype).to(device)
    def fn():
        return mfd.projection(pt, Z)
    return fn


def _setup_fr_retraction(size, device, dtype):
    mfd = FixedRankManifold(
        m=size["m"], n=size["n"], r=size["r"], device=device, dtype=dtype,
    )
    g = torch.Generator(device="cpu"); g.manual_seed(0)
    pt = mfd.random_point(batch_size=size["B"], generator=g)
    pt = tuple(t.to(device) for t in pt)
    tangent = mfd.projection(
        pt, torch.randn(size["B"], size["m"], size["n"], generator=g,
                          dtype=dtype).to(device),
    )
    def fn():
        return mfd.retraction(pt, tangent)
    return fn


def _setup_fr_inner(size, device, dtype):
    mfd = FixedRankManifold(
        m=size["m"], n=size["n"], r=size["r"], device=device, dtype=dtype,
    )
    g = torch.Generator(device="cpu"); g.manual_seed(0)
    pt = mfd.random_point(batch_size=size["B"], generator=g)
    pt = tuple(t.to(device) for t in pt)
    A = torch.randn(size["B"], size["m"], size["n"], generator=g,
                     dtype=dtype).to(device)
    B = torch.randn(size["B"], size["m"], size["n"], generator=g,
                     dtype=dtype).to(device)
    def fn():
        return mfd.inner(pt, A, B)
    return fn


_fr_sizes = [
    {"B": 1,  "m": 64,   "n": 64,   "r": 8},
    {"B": 1,  "m": 256,  "n": 256,  "r": 16},
    {"B": 16, "m": 64,   "n": 64,   "r": 8},
    {"B": 1,  "m": 1024, "n": 1024, "r": 32},
]

bench.case(
    "FixedRank.projection", _setup_fr_projection, _fr_sizes,
    notes="Ambient → tangent projection at a point (5 bmms).",
)
bench.case(
    "FixedRank.retraction", _setup_fr_retraction, _fr_sizes,
    notes="Move-and-project-back retraction via full SVD truncation.",
)
bench.case(
    "FixedRank.inner", _setup_fr_inner, _fr_sizes,
    notes="Frobenius inner product over ambient tangents.",
)


# ----------------- SPD -----------------

def _well_conditioned_spd(n, B, device, dtype, generator):
    """Wishart + ε·I — keeps eigh stable across float32/64 for the
    benchmark inputs. Identity scale is `n` (≈ trace of A Aᵀ for unit
    Wishart) so the condition number stays bounded.
    """
    A = torch.randn(B, n, n, generator=generator, dtype=dtype)
    S = A @ A.mT
    eye = torch.eye(n, dtype=dtype).unsqueeze(0) * float(n)
    return (S + eye).to(device)


def _setup_spd_exp(size, device, dtype):
    mfd = SPDManifold(n=size["n"], device=device, dtype=dtype)
    g = torch.Generator(device="cpu"); g.manual_seed(0)
    S = _well_conditioned_spd(size["n"], size["B"], device, dtype, g)
    V = 0.5 * (
        torch.randn(size["B"], size["n"], size["n"], generator=g,
                     dtype=dtype).to(device)
    )
    V = 0.5 * (V + V.mT)
    def fn():
        return mfd.exp(S, V)
    return fn


def _setup_spd_log(size, device, dtype):
    mfd = SPDManifold(n=size["n"], device=device, dtype=dtype)
    g = torch.Generator(device="cpu"); g.manual_seed(0)
    S = _well_conditioned_spd(size["n"], size["B"], device, dtype, g)
    T = _well_conditioned_spd(size["n"], size["B"], device, dtype, g)
    def fn():
        return mfd.log(S, T)
    return fn


def _setup_spd_distance(size, device, dtype):
    mfd = SPDManifold(n=size["n"], device=device, dtype=dtype)
    g = torch.Generator(device="cpu"); g.manual_seed(0)
    S = _well_conditioned_spd(size["n"], size["B"], device, dtype, g)
    T = _well_conditioned_spd(size["n"], size["B"], device, dtype, g)
    def fn():
        return mfd.distance(S, T)
    return fn


def _setup_spd_inner(size, device, dtype):
    mfd = SPDManifold(n=size["n"], device=device, dtype=dtype)
    g = torch.Generator(device="cpu"); g.manual_seed(0)
    S = _well_conditioned_spd(size["n"], size["B"], device, dtype, g)
    U = 0.5 * (
        torch.randn(size["B"], size["n"], size["n"], generator=g,
                     dtype=dtype).to(device)
    )
    U = 0.5 * (U + U.mT)
    V = 0.5 * (
        torch.randn(size["B"], size["n"], size["n"], generator=g,
                     dtype=dtype).to(device)
    )
    V = 0.5 * (V + V.mT)
    def fn():
        return mfd.inner(S, U, V)
    return fn


_spd_sizes = [
    {"B": 1,  "n": 16},
    {"B": 1,  "n": 64},
    {"B": 16, "n": 16},
    {"B": 16, "n": 64},
    {"B": 1,  "n": 256},
]

bench.case(
    "SPD.exp", _setup_spd_exp, _spd_sizes,
    notes="exp_S(V) = S^{1/2} expm(S^{-1/2} V S^{-1/2}) S^{1/2}; one eigh + matrix_exp.",
)
bench.case(
    "SPD.log", _setup_spd_log, _spd_sizes,
    notes="log_S(T); one eigh of S + one eigh of the inner whitened matrix.",
)
bench.case(
    "SPD.distance", _setup_spd_distance, _spd_sizes,
    notes="Affine-invariant geodesic distance; two eighs.",
)
bench.case(
    "SPD.inner", _setup_spd_inner, _spd_sizes,
    notes="Affine-invariant inner; two linear solves.",
)


# ----------------- comparison (model-space volumes) -----------------

def _setup_ball_volume(size, device, dtype):
    g = torch.Generator(device="cpu"); g.manual_seed(0)
    # kappa in [-1, 1] keeps the spherical diameter bound pi/sqrt(kappa)
    # above the r < 1 samples for every element.
    kappa = (2.0 * torch.rand(size["B"], generator=g, dtype=dtype) - 1.0).to(device)
    N = torch.full((size["B"],), 5.0, dtype=dtype, device=device)
    r = torch.rand(size["B"], generator=g, dtype=dtype).to(device)
    def fn():
        return model_ball_volume(kappa, N, r)
    return fn


_ball_sizes = [
    {"B": 1},
    {"B": 1024},
    {"B": 65536},
]

bench.case(
    "comparison.model_ball_volume", _setup_ball_volume, _ball_sizes,
    notes="Gauss-Legendre quadrature; nodes served from the _quadrature cache "
          "(previously recomputed via scipy on every call).",
)
