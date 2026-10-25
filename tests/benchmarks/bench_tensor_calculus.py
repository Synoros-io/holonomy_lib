# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Synoros

"""Benchmarks for holonomy_lib.tensor_calculus (HOSVD + mode ops).

HOSVD's d serial SVDs dominate for large tensors; mode_product is
batched bmm under the hood. Cases sweep tensor order (3 vs 4) and
the inner dims.
"""

from __future__ import annotations

import torch

from holonomy_lib.tensor_calculus import hosvd, mode_product, mode_unfolding
from tests.benchmarks.harness import Bench

bench = Bench("tensor_calculus")


def _make_T(size, device, dtype):
    dims = size["dims"]            # tuple of tensor-mode dims
    B = size["B"]
    g = torch.Generator(device="cpu")
    g.manual_seed(0)
    return torch.randn(B, *dims, generator=g, dtype=dtype).to(device)


def _setup_mode_unfolding(size, device, dtype):
    T = _make_T(size, device, dtype)
    axis = size["axis"]
    def fn():
        return mode_unfolding(T, axis=axis)
    return fn


def _setup_mode_product(size, device, dtype):
    T = _make_T(size, device, dtype)
    axis = size["axis"]
    j = size["j"]
    n_axis = T.shape[axis]
    g = torch.Generator(device="cpu")
    g.manual_seed(1)
    A = torch.randn(T.shape[0], j, n_axis, generator=g, dtype=dtype).to(device)
    def fn():
        return mode_product(T, A, axis=axis)
    return fn


def _setup_hosvd(size, device, dtype):
    T = _make_T(size, device, dtype)
    ranks = size["ranks"]
    def fn():
        return hosvd(T, ranks=ranks, mode="exact")
    return fn


_unfold_sizes = [
    {"B": 1,  "dims": (16, 16, 16), "axis": 1},
    {"B": 1,  "dims": (32, 32, 32), "axis": 2},
    {"B": 1,  "dims": (64, 64, 64), "axis": 3},
    {"B": 8,  "dims": (32, 32, 32), "axis": 1},
]
_product_sizes = [
    {"B": 1, "dims": (16, 16, 16), "axis": 1, "j": 8},
    {"B": 1, "dims": (64, 64, 64), "axis": 2, "j": 16},
    {"B": 1, "dims": (32, 32, 32, 32), "axis": 2, "j": 16},
]
_hosvd_sizes = [
    {"B": 1, "dims": (16, 16, 16),  "ranks": (4, 4, 4)},
    {"B": 1, "dims": (32, 32, 32),  "ranks": (8, 8, 8)},
    {"B": 1, "dims": (64, 64, 64),  "ranks": (16, 16, 16)},
    {"B": 1, "dims": (16, 16, 16, 16), "ranks": (4, 4, 4, 4)},
    {"B": 4, "dims": (32, 32, 32),  "ranks": (8, 8, 8)},
]

bench.case("mode_unfolding", _setup_mode_unfolding, _unfold_sizes,
            notes="movedim + reshape; ideally zero copies.")
bench.case("mode_product", _setup_mode_product, _product_sizes,
            notes="n-mode product = movedim + bmm + reshape.")
bench.case("hosvd", _setup_hosvd, _hosvd_sizes,
            notes="Truncated higher-order SVD; d serial SVDs.")
