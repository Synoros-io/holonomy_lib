# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Synoros

"""Tests for holonomy_lib.spectral.embedding.laplacian_eigenmaps."""

from __future__ import annotations

import pytest
import torch

from holonomy_lib.spectral import laplacian, laplacian_eigenmaps


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

    def test_rejects_k_zero(self):
        with pytest.raises(ValueError, match="k must"):
            laplacian_eigenmaps(torch.zeros(1, 4, 4), k=0)

    def test_rejects_k_too_large(self):
        with pytest.raises(ValueError, match="k must"):
            laplacian_eigenmaps(torch.zeros(1, 4, 4), k=5)

    def test_rejects_unknown_laplacian_type(self):
        with pytest.raises(ValueError, match="laplacian_type"):
            laplacian_eigenmaps(
                torch.zeros(1, 4, 4), k=2, laplacian_type="nonsense",  # type: ignore[arg-type]
            )


# --------------------------------------------------------------------
# Shapes
# --------------------------------------------------------------------


@pytest.mark.parametrize("batch", [1, 3])
@pytest.mark.parametrize("laplacian_type", [
    "combinatorial", "symmetric_normalized", "random_walk", "signed",
])
class TestShapes:
    def test_shapes(self, batch, laplacian_type):
        n, k = 8, 3
        A = _random_connected_graph(n=n, batch=batch, seed=0)
        eigvals, eigvecs = laplacian_eigenmaps(A, k=k, laplacian_type=laplacian_type)
        assert eigvals.shape == (batch, k)
        assert eigvecs.shape == (batch, n, k)


# --------------------------------------------------------------------
# Eigenvalues — sorted ascending; in expected ranges
# --------------------------------------------------------------------


class TestEigenvalues:
    def test_sorted_ascending(self):
        A = _random_connected_graph(n=10, batch=2, seed=10)
        for ltype in ["combinatorial", "symmetric_normalized",
                       "random_walk", "signed"]:
            eigvals, _ = laplacian_eigenmaps(A, k=10, laplacian_type=ltype)
            diffs = eigvals[..., 1:] - eigvals[..., :-1]
            assert (diffs >= -1e-10).all(), (
                f"{ltype}: eigenvalues not sorted; "
                f"min diff {diffs.min().item()}"
            )

    def test_smallest_is_zero_for_connected(self):
        """For a connected graph, the combinatorial and symmetric-norm
        Laplacians have smallest eigenvalue 0.
        """
        A = _random_connected_graph(n=8, batch=3, seed=11)
        for ltype in ["combinatorial", "symmetric_normalized"]:
            eigvals, _ = laplacian_eigenmaps(A, k=1, laplacian_type=ltype)
            assert eigvals.abs().max() < 1e-9, (
                f"{ltype}: smallest eigenvalue should be 0 for connected; "
                f"got {eigvals.abs().max().item()}"
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
        for ltype in ["combinatorial", "symmetric_normalized", "signed"]:
            _, U = laplacian_eigenmaps(A, k=5, laplacian_type=ltype)
            UtU = U.mT @ U
            I = torch.eye(5, dtype=U.dtype).expand_as(UtU)
            torch.testing.assert_close(UtU, I, atol=1e-10, rtol=0)

    def test_residual_sym_norm(self):
        """L_sym u ≈ λ u for each returned eigenpair."""
        A = _random_connected_graph(n=12, batch=2, seed=21)
        L = laplacian.symmetric_normalized(A)
        eigvals, U = laplacian_eigenmaps(
            A, k=6, laplacian_type="symmetric_normalized",
        )
        # L @ U should equal U * eigvals (broadcast across columns)
        residual = L @ U - U * eigvals.unsqueeze(dim=-2)
        max_res = torch.linalg.norm(residual, dim=(-2, -1)).max().item()
        assert max_res < 1e-9, f"residual {max_res} too large"

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

    def test_combinatorial(self):
        A = _random_connected_graph(n=8, batch=2, seed=30)
        self._check_projector_match(A, k=4, ltype="combinatorial",
                                       builder=laplacian.combinatorial)

    def test_symmetric_normalized(self):
        A = _random_connected_graph(n=8, batch=2, seed=31)
        self._check_projector_match(A, k=4, ltype="symmetric_normalized",
                                       builder=laplacian.symmetric_normalized)

    def test_signed(self):
        # Random signed graph; reuse signed-graph helper logic
        g = _seeded_generator(32)
        U = torch.rand(2, 8, 8, generator=g, dtype=torch.float64) * 2 - 1
        triu = torch.triu(U, diagonal=1)
        A = triu + triu.mT
        self._check_projector_match(A, k=4, ltype="signed",
                                       builder=laplacian.signed)
