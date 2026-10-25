# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Synoros

"""Tests for holonomy_lib.spectral.embedding.laplacian_eigenmaps."""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch

from holonomy_lib.spectral import laplacian, laplacian_eigenmaps
from holonomy_lib.spectral.embedding import LaplacianType


def _seeded_generator(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def _random_connected_graph(
    n: int, batch: int, seed: int, dtype=torch.float64
) -> torch.Tensor:
    """Random graph that's likely connected: dense Erdős-Rényi with p=0.6
    plus a guaranteed Hamilton path 0–1–2–...–n−1 so connectivity is
    forced.
    """
    g = _seeded_generator(seed)
    U = (torch.rand(batch, n, n, generator=g, dtype=dtype) < 0.6).to(dtype)
    triu = torch.triu(U, diagonal=1)
    A = triu + triu.mT
    # Force a Hamilton path: ensure A[i, i+1] = A[i+1, i] = 1
    for i in range(n - 1):
        A[:, i, i + 1] = 1.0
        A[:, i + 1, i] = 1.0
    return A


# --------------------------------------------------------------------
# Input validation
# --------------------------------------------------------------------


class TestValidation:
    def test_rejects_non_square(self):
        with pytest.raises(ValueError, match="must be"):
            laplacian_eigenmaps(torch.zeros(2, 4, 5), k=2)


# --------------------------------------------------------------------
# Eigenvalues, sorted ascending; in expected ranges
# --------------------------------------------------------------------


class TestEigenvalues:
    def test_sorted_ascending(self):
        A = _random_connected_graph(n=10, batch=2, seed=10)
        variants: tuple[LaplacianType, ...] = (
            "combinatorial", "symmetric_normalized", "random_walk", "signed",
        )
        for ltype in variants:
            eigvals, _ = laplacian_eigenmaps(A, k=10, laplacian_type=ltype)
            diffs = eigvals[..., 1:] - eigvals[..., :-1]
            assert (diffs >= -1e-10).all(), (
                f"{ltype}: eigenvalues not sorted; "
                f"min diff {diffs.min().item()}"
            )


    def test_sym_norm_spectrum_in_zero_two(self):
        """L_sym spectrum ⊂ [0, 2]."""
        A = _random_connected_graph(n=10, batch=2, seed=12)
        eigvals, _ = laplacian_eigenmaps(
            A, k=10, laplacian_type="symmetric_normalized",
        )
        assert (eigvals >= -1e-10).all()
        assert (eigvals <= 2 + 1e-10).all()


# --------------------------------------------------------------------
# Eigenvectors
# --------------------------------------------------------------------


class TestEigenvectors:
    def test_orthonormal_for_symmetric_laplacians(self):
        """For symmetric L (combinatorial, sym-norm, signed), the
        eigh-returned eigenvectors are orthonormal.
        """
        A = _random_connected_graph(n=10, batch=2, seed=20)
        variants: tuple[LaplacianType, ...] = (
            "combinatorial", "symmetric_normalized", "signed",
        )
        for ltype in variants:
            _, U = laplacian_eigenmaps(A, k=5, laplacian_type=ltype)
            UtU = U.mT @ U
            identity = torch.eye(5, dtype=U.dtype).expand_as(UtU)
            torch.testing.assert_close(UtU, identity, atol=1e-10, rtol=0)


    def test_residual_random_walk(self):
        """L_rw v ≈ λ v for each returned eigenpair.

        L_rw is non-symmetric; eigenvectors are right eigenvectors.
        """
        A = _random_connected_graph(n=10, batch=2, seed=22)
        L_rw = laplacian.random_walk(A)
        eigvals, V = laplacian_eigenmaps(
            A, k=5, laplacian_type="random_walk",
        )
        residual = L_rw @ V - V * eigvals.unsqueeze(dim=-2)
        max_res = torch.linalg.norm(residual, dim=(-2, -1)).max().item()
        assert max_res < 1e-8, f"residual {max_res} too large"
        isolated = torch.tensor(
            [[[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]],
            dtype=torch.float64,
        )
        values, vectors = laplacian_eigenmaps(isolated, 3, "random_walk")
        torch.testing.assert_close(values, torch.tensor([[0.0, 1.0, 2.0]]).double())
        torch.testing.assert_close(vectors.norm(dim=-2), torch.ones(1, 3).double())
        torch.testing.assert_close(
            laplacian.random_walk(isolated) @ vectors,
            vectors * values.unsqueeze(-2),
        )
        values, vectors = laplacian_eigenmaps(isolated[:0], 3, "random_walk")
        assert values.shape == (0, 3)
        assert vectors.shape == (0, 3, 3)


# --------------------------------------------------------------------
# Direct comparison against torch.linalg.eigh on hand-built Laplacians
# --------------------------------------------------------------------


class TestDirectComparison:
    """Independent re-computation: build L via the laplacian module,
    take eigh, compare bottom-k against laplacian_eigenmaps output.

    For symmetric Laplacians (combinatorial, sym-norm, signed), the
    output is fully determined up to sign/orthogonal-rotation freedom
    in degenerate subspaces. We check via the bottom-k projector
    P = U U^T, which is invariant to those freedoms.
    """

    def _check_projector_match(self, A, k, ltype, builder):
        eigvals_ours, U_ours = laplacian_eigenmaps(A, k=k, laplacian_type=ltype)
        L = builder(A)
        eigvals_ref, U_ref = torch.linalg.eigh(L)
        # Compare bottom-k projectors
        P_ours = U_ours @ U_ours.mT
        P_ref = (U_ref[..., :k]) @ (U_ref[..., :k]).mT
        torch.testing.assert_close(P_ours, P_ref, atol=1e-9, rtol=1e-9)
        # And eigenvalues match elementwise
        torch.testing.assert_close(
            eigvals_ours, eigvals_ref[..., :k], atol=1e-10, rtol=1e-10,
        )
        torch.testing.assert_close(
            L @ U_ours, U_ours * eigvals_ours.unsqueeze(-2), atol=1e-9, rtol=1e-9,
        )

    def test_combinatorial(self):
        cases: tuple[
            tuple[LaplacianType, Callable[[torch.Tensor], torch.Tensor], int], ...,
        ] = (
            ("combinatorial", laplacian.combinatorial, 30),
            ("symmetric_normalized", laplacian.symmetric_normalized, 31),
            ("signed", laplacian.signed, 32),
        )
        for ltype, builder, seed in cases:
            A = _random_connected_graph(n=8, batch=3, seed=seed)
            for k in (1, 4):
                self._check_projector_match(A, k, ltype, builder)
            values, _ = laplacian_eigenmaps(A, 1, ltype)
            torch.testing.assert_close(
                values, torch.zeros_like(values), atol=1e-9, rtol=0,
            )
            values, vectors = laplacian_eigenmaps(A[:0], 4, ltype)
            assert values.shape == (0, 4)
            assert vectors.shape == (0, 8, 4)
        g = _seeded_generator(32)
        upper = torch.triu(
            torch.rand(2, 8, 8, generator=g, dtype=torch.float64) * 2 - 1, 1,
        )
        self._check_projector_match(upper + upper.mT, 4, "signed", laplacian.signed)
        A = _random_connected_graph(n=12, batch=2, seed=21)
        self._check_projector_match(
            A, 6, "symmetric_normalized", laplacian.symmetric_normalized,
        )
