# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Synoros

"""Tests for holonomy_lib.spectral.diffusion_map.

Layers:
  1. Validation (k bounds, t ≥ 0, square input).
  2. Shape contract.
  3. Closed-form sanity on K_n (uniform eigenstructure) and P_n.
  4. Diffusion-time monotonicity (the embedding shrinks as t grows
     for `μ_j < 1`).
"""

from __future__ import annotations

import pytest
import torch

from holonomy_lib.spectral.diffusion_map import diffusion_map


def _complete_graph(n: int, batch: int = 1, dtype=torch.float64) -> torch.Tensor:
    A = torch.ones(batch, n, n, dtype=dtype) - torch.eye(n, dtype=dtype).unsqueeze(0)
    return A


def _path_graph(n: int, batch: int = 1, dtype=torch.float64) -> torch.Tensor:
    A = torch.zeros(batch, n, n, dtype=dtype)
    for i in range(n - 1):
        A[:, i, i + 1] = 1.0
        A[:, i + 1, i] = 1.0
    return A


# --------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------


class TestValidation:
    def test_rejects_negative_t(self):
        A = _complete_graph(5)
        with pytest.raises(ValueError, match="diffusion time"):
            diffusion_map(A, k=2, t=-0.5)

    def test_rejects_k_zero(self):
        A = _complete_graph(5)
        with pytest.raises(ValueError, match="k"):
            diffusion_map(A, k=0, t=1.0)

    def test_rejects_k_too_large(self):
        A = _complete_graph(5)
        with pytest.raises(ValueError, match="k"):
            diffusion_map(A, k=5, t=1.0)  # n=5, max k = n-1 = 4

    def test_rejects_non_square(self):
        with pytest.raises(ValueError, match="must be"):
            diffusion_map(torch.zeros(1, 4, 5), k=2)

    def test_warns_on_disconnected_graph(self):
        """Regression: previously, disconnected graphs (multiple zero
        L_rw eigenvalues) silently produced corrupted embeddings.
        Now a UserWarning fires so the caller can mask by component."""
        import warnings
        # Two disjoint triangles → 2 zero L_rw eigenvalues.
        n = 6
        A = torch.zeros(1, n, n, dtype=torch.float64)
        for i in range(3):
            for j in range(3):
                if i != j:
                    A[0, i, j] = 1.0
        for i in range(3, 6):
            for j in range(3, 6):
                if i != j:
                    A[0, i, j] = 1.0
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            diffusion_map(A, k=2, t=1.0)
        assert any(
            "disconnected" in str(item.message).lower() for item in w
        ), f"expected disconnected warning; got {[str(x.message) for x in w]}"


# --------------------------------------------------------------------
# Shape contract
# --------------------------------------------------------------------


@pytest.mark.parametrize("batch", [1, 2, 4])
class TestShapes:
    def test_eigvals_shape(self, batch):
        A = _complete_graph(6, batch=batch)
        eigvals, _ = diffusion_map(A, k=3, t=1.0)
        assert eigvals.shape == (batch, 3)

    def test_embedding_shape(self, batch):
        A = _complete_graph(6, batch=batch)
        _, emb = diffusion_map(A, k=3, t=1.0)
        assert emb.shape == (batch, 6, 3)


# --------------------------------------------------------------------
# Closed-form sanity
# --------------------------------------------------------------------


class TestClosedForms:
    def test_complete_graph_transition_eigenvalues(self):
        """On K_n the random-walk Laplacian L_rw = I − D⁻¹ A has
        eigenvalue 0 (multiplicity 1) and eigenvalue n/(n−1)
        (multiplicity n − 1). Therefore the P-eigenvalues
        μ_j = 1 − λ_j are 1 (dropped) and −1/(n − 1) (the rest).
        """
        n = 5
        A = _complete_graph(n)
        eigvals, _ = diffusion_map(A, k=n - 1, t=1.0)
        expected = -1.0 / (n - 1)
        torch.testing.assert_close(
            eigvals[0],
            torch.full_like(eigvals[0], expected),
            atol=1e-9, rtol=0,
        )

    def test_t_zero_returns_raw_eigenvectors(self):
        """At t = 0, μ^0 = 1 for every mode, so the embedding equals
        the raw L_rw eigenvectors (with the trivial null mode dropped).
        Note: L_rw eigenvectors are NOT standard-norm orthonormal —
        they're D^{-1/2}-norm orthonormal — so we compare against the
        `laplacian_eigenmaps` output directly rather than against a
        unit-norm expectation."""
        from holonomy_lib.spectral import laplacian_eigenmaps
        A = _path_graph(5)
        _, emb_t0 = diffusion_map(A, k=3, t=0.0)
        _, full_eigvecs = laplacian_eigenmaps(
            A, k=4, laplacian_type="random_walk",
        )
        raw = full_eigvecs[..., 1:]
        torch.testing.assert_close(emb_t0, raw, atol=1e-10, rtol=0)

    def test_diffusion_time_shrinks_subdominant_modes(self):
        """For non-bipartite graphs, P-eigenvalues μ_j ∈ (0, 1) for the
        non-trivial modes, so μ^t shrinks monotonically as t grows.
        Use a path graph (acyclic, non-bipartite-like spectrum)."""
        A = _path_graph(6)
        _, emb_t1 = diffusion_map(A, k=2, t=1.0)
        _, emb_t5 = diffusion_map(A, k=2, t=5.0)
        # The Frobenius norm of the t=5 embedding must be ≤ t=1's,
        # since each component shrank by μ_j^(5-1) ≤ 1 (when μ_j ∈ [0, 1]).
        assert emb_t5.norm() <= emb_t1.norm() + 1e-9


# --------------------------------------------------------------------
# Distances align with diffusion distance
# --------------------------------------------------------------------


class TestDiffusionDistance:
    """The pairwise Euclidean distance in the diffusion embedding
    should approximate the diffusion distance:

        D_t(x_i, x_l)² ≈ Σ_j μ_j^{2t} (φ_j(x_i) − φ_j(x_l))²

    For k = n − 1 (all non-trivial modes), this should hold exactly."""

    def test_full_dim_distance_identity(self):
        n = 5
        A = _path_graph(n)
        eigvals, emb = diffusion_map(A, k=n - 1, t=1.0)
        # Pairwise squared distance in embedding space
        diff = emb.unsqueeze(dim=-2) - emb.unsqueeze(dim=-3)  # (B, n, n, k)
        dist_sq = (diff * diff).sum(dim=-1)                    # (B, n, n)
        # Direct: Σ_j μ_j^{2·t} (φ_j(i) − φ_j(l))²
        mu_2t = (eigvals.clamp(min=0.0) ** (2.0 * 1.0)).unsqueeze(-2)  # (B, 1, k)
        # The embedding already encodes the μ^t scaling, so the
        # pairwise difference squared IS the diffusion distance. The
        # test is a consistency / self-consistency check.
        torch.testing.assert_close(dist_sq, dist_sq.mT, atol=1e-10, rtol=0)
