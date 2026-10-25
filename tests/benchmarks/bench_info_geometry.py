# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Synoros

"""Benchmarks for holonomy_lib.info_geometry.

Bregman + KL primitives are cheap per-sample; the interesting cost
scaling is in `kl_divergence_gaussian` where the Cholesky drives an
O(B · d³) per call. Categorical KL is O(B · k) cheap. Bregman cost
is dominated by the caller's `potential` function.
"""

from __future__ import annotations

import torch

from holonomy_lib.info_geometry import (
    bregman_divergence,
    kl_divergence_categorical,
    kl_divergence_gaussian,
)
from tests.benchmarks.harness import Bench

bench = Bench("info_geometry")


def _seeded(seed: int) -> torch.Generator:
    g = torch.Generator(device="cpu"); g.manual_seed(seed)
    return g


def _half_norm_sq(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return 0.5 * (x * x).sum(dim=-1), x


def _setup_bregman_squared_euclidean(size, device, dtype):
    p = torch.randn(size["B"], size["d"], generator=_seeded(0),
                     dtype=dtype).to(device)
    q = torch.randn(size["B"], size["d"], generator=_seeded(1),
                     dtype=dtype).to(device)
    def fn():
        return bregman_divergence(p, q, _half_norm_sq)
    return fn


def _setup_kl_categorical(size, device, dtype):
    p = torch.rand(size["B"], size["k"], generator=_seeded(0), dtype=dtype)
    p = p / p.sum(dim=-1, keepdim=True)
    q = torch.rand(size["B"], size["k"], generator=_seeded(1), dtype=dtype)
    q = q / q.sum(dim=-1, keepdim=True)
    p = p.to(device); q = q.to(device)
    def fn():
        return kl_divergence_categorical(p, q)
    return fn


def _make_spd(B, d, seed, dtype):
    g = _seeded(seed)
    A = torch.randn(B, d, d, generator=g, dtype=dtype)
    return A @ A.mT + torch.eye(d, dtype=dtype).unsqueeze(0) * d


def _setup_kl_gaussian(size, device, dtype):
    B, d = size["B"], size["d"]
    mu_p = torch.randn(B, d, generator=_seeded(2), dtype=dtype).to(device)
    mu_q = torch.randn(B, d, generator=_seeded(3), dtype=dtype).to(device)
    Sigma_p = _make_spd(B, d, seed=4, dtype=dtype).to(device)
    Sigma_q = _make_spd(B, d, seed=5, dtype=dtype).to(device)
    def fn():
        return kl_divergence_gaussian(mu_p, Sigma_p, mu_q, Sigma_q)
    return fn


_bregman_sizes = [
    {"B": 1,   "d": 16},
    {"B": 1,   "d": 256},
    {"B": 1,   "d": 4096},
    {"B": 256, "d": 64},
]
_kl_categorical_sizes = [
    {"B": 1,    "k": 16},
    {"B": 1,    "k": 256},
    {"B": 1024, "k": 64},
]
_kl_gaussian_sizes = [
    {"B": 1,  "d": 4},
    {"B": 1,  "d": 16},
    {"B": 1,  "d": 64},
    {"B": 16, "d": 16},
    {"B": 1,  "d": 256},
]


bench.case("bregman_divergence_squared_euclidean",
            _setup_bregman_squared_euclidean, _bregman_sizes,
            notes="Bregman with F(x) = (1/2)||x||^2; sum-of-squares cost.")
bench.case("kl_divergence_categorical", _setup_kl_categorical,
            _kl_categorical_sizes,
            notes="Discrete KL with the torch.where guard for the q=0 case.")
bench.case("kl_divergence_gaussian", _setup_kl_gaussian, _kl_gaussian_sizes,
            notes="Gaussian KL with Cholesky-based solves; dominated by O(B*d^3).")


# ----------------- Fisher metric + natural gradient (roadmap #3) -----------------

from holonomy_lib.info_geometry import (
    fisher_information_categorical,
    fisher_information_gaussian_mean,
    natural_gradient,
)


def _setup_fisher_categorical(size, device, dtype):
    p = torch.rand(size["B"], size["k"], generator=_seeded(0), dtype=dtype) + 0.1
    p = p / p.sum(dim=-1, keepdim=True)
    p = p.to(device)
    def fn():
        return fisher_information_categorical(p)
    return fn


def _setup_fisher_gaussian_mean(size, device, dtype):
    Sigma = _make_spd(size["B"], size["d"], seed=6, dtype=dtype).to(device)
    def fn():
        return fisher_information_gaussian_mean(Sigma)
    return fn


def _setup_natural_gradient(size, device, dtype):
    Sigma = _make_spd(size["B"], size["d"], seed=7, dtype=dtype).to(device)
    F = fisher_information_gaussian_mean(Sigma)
    grad = torch.randn(size["B"], size["d"], generator=_seeded(8),
                        dtype=dtype).to(device)
    def fn():
        return natural_gradient(grad, F)
    return fn


_fisher_categorical_sizes = [
    {"B": 1,    "k": 16},
    {"B": 1,    "k": 256},
    {"B": 1024, "k": 64},
]
_fisher_gaussian_sizes = [
    {"B": 1,  "d": 16},
    {"B": 1,  "d": 64},
    {"B": 1,  "d": 256},
    {"B": 16, "d": 64},
]


bench.case("fisher_information_categorical",
            _setup_fisher_categorical, _fisher_categorical_sizes,
            notes="Diagonal Fisher on the simplex; clamp + reciprocal + diag_embed.")
bench.case("fisher_information_gaussian_mean",
            _setup_fisher_gaussian_mean, _fisher_gaussian_sizes,
            notes="Sigma^{-1} via Cholesky; same path as kl_divergence_gaussian.")
bench.case("natural_gradient",
            _setup_natural_gradient, _fisher_gaussian_sizes,
            notes="F^{-1} grad via torch.linalg.solve (no explicit inverse).")
