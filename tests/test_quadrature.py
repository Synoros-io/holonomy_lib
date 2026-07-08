# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Synoros

"""Tests for holonomy_lib._quadrature (cached Gauss-Legendre nodes).

Layers: unit (values match scipy exactly; dtype/device casts) and the
caching property (master copy computed once, returned by identity).
"""

from __future__ import annotations

import torch
from scipy.special import roots_legendre

from holonomy_lib._quadrature import (
    _gauss_legendre_float64_cpu,
    gauss_legendre_nodes,
)

CPU = torch.device("cpu")


def test_matches_scipy_exactly():
    for n_quad in (8, 32, 64):
        nodes, weights = gauss_legendre_nodes(
            n_quad, dtype=torch.float64, device=CPU,
        )
        nodes_np, weights_np = roots_legendre(n_quad)
        torch.testing.assert_close(
            nodes, torch.as_tensor(nodes_np), atol=0.0, rtol=0.0,
        )
        torch.testing.assert_close(
            weights, torch.as_tensor(weights_np), atol=0.0, rtol=0.0,
        )


def test_master_copy_is_cached_by_identity():
    a = _gauss_legendre_float64_cpu(16)
    b = _gauss_legendre_float64_cpu(16)
    assert a[0] is b[0] and a[1] is b[1]


def test_dtype_cast_equals_downcast_of_master():
    nodes32, weights32 = gauss_legendre_nodes(
        16, dtype=torch.float32, device=CPU,
    )
    assert nodes32.dtype == torch.float32
    assert weights32.dtype == torch.float32
    nodes64, weights64 = gauss_legendre_nodes(
        16, dtype=torch.float64, device=CPU,
    )
    torch.testing.assert_close(
        nodes32, nodes64.to(torch.float32), atol=0.0, rtol=0.0,
    )
    torch.testing.assert_close(
        weights32, weights64.to(torch.float32), atol=0.0, rtol=0.0,
    )
