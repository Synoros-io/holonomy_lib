# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Synoros

"""Tests for holonomy_lib.tensor_calculus.decomposition."""

from __future__ import annotations

from importlib.util import find_spec

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
    def test_rejects_axis_zero(self):
        T = torch.randn(2, 4, 5)
        A = torch.randn(2, 3, 4)
        with pytest.raises(ValueError, match="axis"):
            mode_product(T, A, axis=0)

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

    @pytest.mark.parametrize("batch", [0, 1, 2])
    def test_matrix_special_case(self, batch):
        """For a 2-tensor (matrix), mode_product(T, A, axis=1) = A @ T
        (when A is (B, j, n_1)) and mode_product(T, A, axis=2) = T @ A.mT.
        """
        T = torch.randn(batch, 4, 5, dtype=torch.float64,
                        generator=_seeded_generator(5))
        A = torch.randn(batch, 3, 4, dtype=torch.float64,
                        generator=_seeded_generator(6))
        ours = mode_product(T, A, axis=1)        # (2, 3, 5)
        ref = torch.bmm(A, T)                     # (2, 3, 5)
        torch.testing.assert_close(ours, ref, atol=1e-12, rtol=0)
        assert ours.shape == (batch, 3, 5)

        A = torch.randn(batch, 3, 5, dtype=torch.float64,
                        generator=_seeded_generator(7))
        ours = mode_product(T, A, axis=2)
        torch.testing.assert_close(ours, torch.bmm(T, A.mT), atol=1e-12, rtol=0)
        assert ours.shape == (batch, 4, 3)


# --------------------------------------------------------------------
# mode_unfolding
# --------------------------------------------------------------------


class TestModeUnfolding:
    def test_rejects_axis_zero(self):
        T = torch.randn(2, 4, 5)
        with pytest.raises(ValueError, match="axis"):
            mode_unfolding(T, axis=0)

    @pytest.mark.parametrize("batch", [0, 1, 3])
    def test_matrix_special_case(self, batch):
        """Unfolding preserves matrix values and orders higher-order columns."""
        T = torch.randn(batch, 4, 5, dtype=torch.float64,
                        generator=_seeded_generator(11))
        out = mode_unfolding(T, axis=1)
        torch.testing.assert_close(out, T, atol=0, rtol=0)
        assert out.shape == (batch, 4, 5)

        T = torch.arange(batch * 12, dtype=torch.float64).reshape(batch, 2, 2, 3)
        # Each row fixes the selected mode. The other modes remain in
        # their original order, with the last one varying fastest.
        references = (
            [[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]],
            [[0, 1, 2, 6, 7, 8], [3, 4, 5, 9, 10, 11]],
            [[0, 3, 6, 9], [1, 4, 7, 10], [2, 5, 8, 11]],
        )
        batch_offset = torch.arange(batch, dtype=T.dtype).reshape(batch, 1, 1) * 12
        for axis, entries in enumerate(references, start=1):
            expected = torch.tensor(entries, dtype=T.dtype).unsqueeze(0) + batch_offset
            torch.testing.assert_close(
                mode_unfolding(T, axis), expected, atol=0, rtol=0,
            )


# --------------------------------------------------------------------
# HOSVD — construction validation
# --------------------------------------------------------------------


class TestHosvdValidation:
    def test_rejects_low_ndim(self):
        with pytest.raises(ValueError, match="ndim"):
            hosvd(torch.randn(5), ranks=(2,))
        with pytest.raises(ValueError, match=r"ranks\[0\].*unfolding"):
            hosvd(torch.ones(1, 3, 2), ranks=(3, 2))


# --------------------------------------------------------------------
# HOSVD — mathematical properties
# --------------------------------------------------------------------


class TestHosvdProperties:
    @pytest.mark.parametrize("batch", [0, 1, 2])
    @pytest.mark.parametrize("svd_mode", ["exact", "randomized"])
    def test_factor_orthonormality(self, batch, svd_mode):
        """Each factor U_k has orthonormal columns: U_kᵀ U_k = I."""
        T = torch.randn(batch, 5, 6, 7, dtype=torch.float64,
                        generator=_seeded_generator(30))
        ranks = (3, 4, 5)
        core, factors = hosvd(T, ranks=ranks, mode=svd_mode)
        assert core.shape == (batch, *ranks)
        for k, (U_k, r_k) in enumerate(zip(factors, ranks, strict=True)):
            assert U_k.shape == (batch, T.shape[k + 1], r_k)
            UtU = U_k.mT @ U_k
            identity = torch.eye(r_k, dtype=torch.float64).expand_as(UtU)
            torch.testing.assert_close(
                UtU, identity, atol=1e-12, rtol=0,
            )

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
        assert e_small >= e_large > 0 or torch.isclose(
            e_large, torch.zeros_like(e_large), atol=1e-12, rtol=0,
        )
        # Full-rank gives essentially zero error.
        assert e_full < 1e-10

        T = torch.randn(2, 4, 5, 6, dtype=torch.float64,
                        generator=_seeded_generator(50))
        ranks = (4, 5, 6)  # full multilinear rank
        core, factors = hosvd(T, ranks=ranks, mode="exact")
        T_recon = mode_product(core, factors[0], axis=1)
        T_recon = mode_product(T_recon, factors[1], axis=2)
        T_recon = mode_product(T_recon, factors[2], axis=3)
        torch.testing.assert_close(T_recon, T, atol=1e-10, rtol=1e-10)


# --------------------------------------------------------------------
# HOSVD — comparison against tensorly
# --------------------------------------------------------------------


@pytest.mark.skipif(
    find_spec("tensorly") is None,
    reason="tensorly not installed",
)
class TestAgainstTensorly:
    """Cross-check the truncated HOSVD error against tensorly's
    Tucker decomposition initialized with HOSVD (`init='svd',
    n_iter_max=0`): pure HOSVD with no ALS refinement, which is
    apples-to-apples with our `hosvd`.

    Tensorly returns slightly different factor/core conventions; we
    compare the reconstruction error, which is invariant to sign /
    basis-rotation freedom inside the truncation.
    """

    def test_reconstruction_error_matches_tensorly_hosvd(self):
        import numpy as np
        tl = pytest.importorskip("tensorly")

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
        tucker = pytest.importorskip("tensorly.decomposition").tucker
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
