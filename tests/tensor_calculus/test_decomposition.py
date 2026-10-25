# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Synoros

"""Tests for holonomy_lib.tensor_calculus.decomposition."""

from __future__ import annotations

import pytest
import torch

from holonomy_lib.tensor_calculus import hosvd, mode_product, mode_unfolding


def _seeded_generator(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


# --------------------------------------------------------------------
# mode_product
# --------------------------------------------------------------------


class TestModeProduct:
    def test_shapes(self):
        # T: (B=2, n1=4, n2=5, n3=6), A: (B=2, j=3, n2=5), axis=2
        # Result: (2, 4, 3, 6)
        T = torch.randn(2, 4, 5, 6, dtype=torch.float64,
                        generator=_seeded_generator(0))
        A = torch.randn(2, 3, 5, dtype=torch.float64,
                        generator=_seeded_generator(1))
        out = mode_product(T, A, axis=2)
        assert out.shape == (2, 4, 3, 6)

    def test_rejects_axis_zero(self):
        T = torch.randn(2, 4, 5)
        A = torch.randn(2, 3, 4)
        with pytest.raises(ValueError, match="axis"):
            mode_product(T, A, axis=0)

    def test_rejects_axis_out_of_range(self):
        T = torch.randn(2, 4, 5)
        A = torch.randn(2, 3, 4)
        with pytest.raises(ValueError, match="axis"):
            mode_product(T, A, axis=3)

    def test_rejects_size_mismatch(self):
        T = torch.randn(2, 4, 5, 6)
        A = torch.randn(2, 3, 7)  # last dim 7 ≠ T.shape[2] = 5
        with pytest.raises(ValueError, match="size mismatch"):
            mode_product(T, A, axis=2)

    def test_associativity_axes_commute_when_different(self):
        """T ×_i A ×_j B  ==  T ×_j B ×_i A  for i ≠ j.

        Standard fact: n-mode products on different modes commute.
        Kolda-Bader (2009), property in §2.5.
        """
        T = torch.randn(2, 4, 5, 6, dtype=torch.float64,
                        generator=_seeded_generator(2))
        A = torch.randn(2, 3, 5, dtype=torch.float64,
                        generator=_seeded_generator(3))  # contract axis=2
        B = torch.randn(2, 7, 6, dtype=torch.float64,
                        generator=_seeded_generator(4))  # contract axis=3
        left = mode_product(mode_product(T, A, axis=2), B, axis=3)
        right = mode_product(mode_product(T, B, axis=3), A, axis=2)
        torch.testing.assert_close(left, right, atol=1e-12, rtol=0)

    def test_matrix_special_case(self):
        """For a 2-tensor (matrix), mode_product(T, A, axis=1) = A @ T
        (when A is (B, j, n_1)) and mode_product(T, A, axis=2) = T @ A.mT.
        """
        T = torch.randn(2, 4, 5, dtype=torch.float64,
                        generator=_seeded_generator(5))
        A = torch.randn(2, 3, 4, dtype=torch.float64,
                        generator=_seeded_generator(6))
        ours = mode_product(T, A, axis=1)        # (2, 3, 5)
        ref = torch.bmm(A, T)                     # (2, 3, 5)
        torch.testing.assert_close(ours, ref, atol=1e-12, rtol=0)


# --------------------------------------------------------------------
# mode_unfolding
# --------------------------------------------------------------------


class TestModeUnfolding:
    def test_shapes(self):
        T = torch.randn(2, 4, 5, 6, dtype=torch.float64,
                        generator=_seeded_generator(10))
        u1 = mode_unfolding(T, axis=1)
        u2 = mode_unfolding(T, axis=2)
        u3 = mode_unfolding(T, axis=3)
        assert u1.shape == (2, 4, 5 * 6)
        assert u2.shape == (2, 5, 4 * 6)
        assert u3.shape == (2, 6, 4 * 5)

    def test_rejects_axis_zero(self):
        T = torch.randn(2, 4, 5)
        with pytest.raises(ValueError, match="axis"):
            mode_unfolding(T, axis=0)

    def test_matrix_special_case(self):
        """mode_unfolding(T, axis=1) for a matrix = T itself (B, m, n) → (B, m, n)."""
        T = torch.randn(3, 4, 5, dtype=torch.float64,
                        generator=_seeded_generator(11))
        out = mode_unfolding(T, axis=1)
        torch.testing.assert_close(out, T, atol=0, rtol=0)


# --------------------------------------------------------------------
# HOSVD — construction validation
# --------------------------------------------------------------------


class TestHosvdValidation:
    def test_rejects_low_ndim(self):
        with pytest.raises(ValueError, match="ndim"):
            hosvd(torch.randn(5), ranks=(2,))

    def test_rejects_rank_count_mismatch(self):
        T = torch.randn(2, 4, 5, 6)
        with pytest.raises(ValueError, match="ranks"):
            hosvd(T, ranks=(2, 3))  # expects 3 ranks for 3-mode tensor

    def test_rejects_rank_zero(self):
        T = torch.randn(2, 4, 5, 6)
        with pytest.raises(ValueError, match="ranks\\["):
            hosvd(T, ranks=(0, 3, 4))

    def test_rejects_rank_too_large(self):
        T = torch.randn(2, 4, 5, 6)
        with pytest.raises(ValueError, match="ranks\\["):
            hosvd(T, ranks=(5, 3, 4))  # 5 > n_1=4


# --------------------------------------------------------------------
# HOSVD — shapes
# --------------------------------------------------------------------


@pytest.mark.parametrize("batch", [1, 3])
@pytest.mark.parametrize("svd_mode", ["exact", "randomized"])
class TestHosvdShapes:
    def test_3d_shapes(self, batch, svd_mode):
        T = torch.randn(batch, 4, 5, 6, dtype=torch.float64,
                        generator=_seeded_generator(20))
        ranks = (2, 3, 4)
        core, factors = hosvd(T, ranks=ranks, mode=svd_mode,
                               generator=_seeded_generator(21))
        assert core.shape == (batch, *ranks)
        assert len(factors) == 3
        assert factors[0].shape == (batch, 4, 2)
        assert factors[1].shape == (batch, 5, 3)
        assert factors[2].shape == (batch, 6, 4)

    def test_4d_shapes(self, batch, svd_mode):
        T = torch.randn(batch, 3, 4, 5, 6, dtype=torch.float64,
                        generator=_seeded_generator(22))
        ranks = (2, 3, 4, 5)
        core, factors = hosvd(T, ranks=ranks, mode=svd_mode,
                               generator=_seeded_generator(23))
        assert core.shape == (batch, *ranks)
        assert len(factors) == 4


# --------------------------------------------------------------------
# HOSVD — mathematical properties
# --------------------------------------------------------------------


class TestHosvdProperties:
    def test_factor_orthonormality(self):
        """Each factor U_k has orthonormal columns: U_kᵀ U_k = I."""
        T = torch.randn(2, 5, 6, 7, dtype=torch.float64,
                        generator=_seeded_generator(30))
        ranks = (3, 4, 5)
        _, factors = hosvd(T, ranks=ranks, mode="exact")
        for k, (U_k, r_k) in enumerate(zip(factors, ranks)):
            UtU = U_k.mT @ U_k
            I = torch.eye(r_k, dtype=torch.float64).expand_as(UtU)
            torch.testing.assert_close(
                UtU, I, atol=1e-12, rtol=0,
            ), f"factor {k} not orthonormal"

    def test_recovery_of_low_multilinear_rank_tensor(self):
        """If T has true multilinear rank ≤ ranks, HOSVD reconstructs T exactly.

        Construct T = G_true ×_1 U_1 ×_2 U_2 ×_3 U_3 with orthonormal U_k
        of shape (n_k, r_k_true), so T has multilinear rank (r_k_true).
        Then HOSVD at ranks ≥ r_k_true must reproduce T exactly.
        """
        B = 2
        n1, n2, n3 = 6, 7, 8
        r1, r2, r3 = 3, 4, 5
        # Random orthonormal factors
        U1 = torch.linalg.qr(
            torch.randn(B, n1, r1, dtype=torch.float64,
                        generator=_seeded_generator(40))
        )[0]
        U2 = torch.linalg.qr(
            torch.randn(B, n2, r2, dtype=torch.float64,
                        generator=_seeded_generator(41))
        )[0]
        U3 = torch.linalg.qr(
            torch.randn(B, n3, r3, dtype=torch.float64,
                        generator=_seeded_generator(42))
        )[0]
        G_true = torch.randn(B, r1, r2, r3, dtype=torch.float64,
                             generator=_seeded_generator(43))
        # T = G ×_1 U_1 ×_2 U_2 ×_3 U_3 — use mode_product with axis (1, 2, 3)
        T = mode_product(G_true, U1, axis=1)
        T = mode_product(T, U2, axis=2)
        T = mode_product(T, U3, axis=3)
        # Now HOSVD at the true ranks
        core, factors = hosvd(T, ranks=(r1, r2, r3), mode="exact")
        # Reconstruct
        T_recon = mode_product(core, factors[0], axis=1)
        T_recon = mode_product(T_recon, factors[1], axis=2)
        T_recon = mode_product(T_recon, factors[2], axis=3)
        torch.testing.assert_close(T_recon, T, atol=1e-10, rtol=1e-10)

    def test_reconstruction_at_full_rank_is_exact(self):
        """HOSVD at full ranks (n_1, ..., n_d) reconstructs T exactly.

        At full ranks the truncation is no truncation; the factors are
        orthonormal bases for the unfolded matrices and the core is
        T expressed in those bases.
        """
        T = torch.randn(2, 4, 5, 6, dtype=torch.float64,
                        generator=_seeded_generator(50))
        ranks = (4, 5, 6)  # full multilinear rank
        core, factors = hosvd(T, ranks=ranks, mode="exact")
        T_recon = mode_product(core, factors[0], axis=1)
        T_recon = mode_product(T_recon, factors[1], axis=2)
        T_recon = mode_product(T_recon, factors[2], axis=3)
        torch.testing.assert_close(T_recon, T, atol=1e-10, rtol=1e-10)

    def test_truncation_error_smaller_for_larger_ranks(self):
        """Increasing any rank cannot make reconstruction worse."""
        T = torch.randn(1, 6, 7, 8, dtype=torch.float64,
                        generator=_seeded_generator(60))

        def reconstruct(ranks):
            core, factors = hosvd(T, ranks=ranks, mode="exact")
            R = mode_product(core, factors[0], axis=1)
            R = mode_product(R, factors[1], axis=2)
            R = mode_product(R, factors[2], axis=3)
            return torch.linalg.norm(R - T)

        e_small = reconstruct((2, 2, 2))
        e_large = reconstruct((4, 4, 4))
        e_full = reconstruct((6, 7, 8))
        # Each step should reduce reconstruction error monotonically.
        assert e_small >= e_large > 0 or torch.isclose(e_large, torch.zeros_like(e_large))
        # Full-rank gives essentially zero error.
        assert e_full < 1e-10


# --------------------------------------------------------------------
# HOSVD — comparison against tensorly
# --------------------------------------------------------------------


try:
    import tensorly as _tensorly  # noqa: F401
    _HAVE_TENSORLY = True
except ImportError:
    _HAVE_TENSORLY = False


@pytest.mark.skipif(
    not _HAVE_TENSORLY,
    reason="tensorly not installed",
)
class TestAgainstTensorly:
    """Cross-check the truncated HOSVD reconstruction against tensorly's
    Tucker decomposition initialized with HOSVD (`init='svd',
    n_iter_max=0`) — pure HOSVD with no ALS refinement, which is
    apples-to-apples with our `hosvd`.

    Tensorly returns slightly different factor/core conventions; we
    compare the dense reconstruction, which is invariant to sign /
    basis-rotation freedom inside the truncation.
    """

    def test_reconstruction_matches_tensorly_hosvd(self):
        import numpy as np
        import tensorly as tl

        tl.set_backend("numpy")

        B = 1
        n1, n2, n3 = 6, 7, 8
        ranks = (3, 4, 5)
        T = torch.randn(B, n1, n2, n3, dtype=torch.float64,
                        generator=_seeded_generator(70))

        # Our HOSVD reconstruction
        core, factors = hosvd(T, ranks=ranks, mode="exact")
        R = mode_product(core, factors[0], axis=1)
        R = mode_product(R, factors[1], axis=2)
        R = mode_product(R, factors[2], axis=3)
        recon_ours = R[0].numpy()

        # Tensorly pure HOSVD: init="svd" with n_iter_max=0 skips the
        # ALS refinement, returning the truncated HOSVD as-is.
        from tensorly.decomposition import tucker
        core_tl, factors_tl = tucker(
            T[0].numpy(), rank=list(ranks), init="svd", n_iter_max=0,
            tol=1.0,
        )
        recon_tl = tl.tucker_to_tensor((core_tl, factors_tl))

        # Pure-HOSVD vs pure-HOSVD: identical reconstruction up to
        # floating-point rounding.
        T_np = T[0].numpy()
        err_ours = np.linalg.norm(recon_ours - T_np)
        err_tl = np.linalg.norm(recon_tl - T_np)
        assert abs(err_ours - err_tl) / max(err_tl, 1e-15) < 1e-10, (
            f"pure-HOSVD reconstruction errors should match tensorly's: "
            f"ours={err_ours:.10e}, tl={err_tl:.10e}"
        )
