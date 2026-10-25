# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Synoros

"""Tests for holonomy_lib.spectral.magnetic.

Four layers:
  1. Validation (square, q in [0, 1)).
  2. q = 0 collapses to the symmetric real Laplacian of A_s.
  3. Hermitian output + real-eigenvalue spectrum (eigh works).
  4. Symmetric input independence of q + bounded spectrum for the
     normalized form.
"""

from __future__ import annotations

import pytest
import torch

from holonomy_lib.spectral import laplacian, magnetic


def _directed_cycle(n: int, batch: int = 1, dtype=torch.float64) -> torch.Tensor:
    A = torch.zeros(batch, n, n, dtype=dtype)
    for i in range(n):
        A[:, i, (i + 1) % n] = 1.0
    return A


def _undirected_cycle(n: int, batch: int = 1, dtype=torch.float64) -> torch.Tensor:
    A = _directed_cycle(n, batch=batch, dtype=dtype)
    return (A + A.mT) * 0.5


# --------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------


class TestValidation:
    def test_rejects_non_square(self):
        with pytest.raises(ValueError, match="must be"):
            magnetic.combinatorial(torch.zeros(1, 4, 5))

    def test_rejects_q_out_of_range(self):
        A = torch.zeros(1, 3, 3, dtype=torch.float64)
        with pytest.raises(ValueError, match="q"):
            magnetic.combinatorial(A, q=-0.1)
        with pytest.raises(ValueError, match="q"):
            magnetic.combinatorial(A, q=1.0)
        with pytest.raises(ValueError, match="q"):
            magnetic.symmetric_normalized(A, q=1.5)

    def test_rejects_complex_input(self):
        """Regression: previously, a complex `A` would crash deep inside
        `torch.complex(cos, sin)` because cos/sin of complex inputs are
        complex. Now we reject up front with a clear ValueError."""
        A = torch.zeros(1, 3, 3, dtype=torch.complex128)
        with pytest.raises(ValueError, match="real"):
            magnetic.combinatorial(A, q=0.25)
        with pytest.raises(ValueError, match="real"):
            magnetic.symmetric_normalized(A, q=0.25)


# --------------------------------------------------------------------
# q = 0 reduces to symmetric Laplacian of A_s
# --------------------------------------------------------------------


class TestZeroCharge:
    """At q = 0 the phase factor is identically 1; the magnetic Laplacian
    reduces to the real-valued Laplacian of the symmetrized adjacency
    A_s = (A + A.T)/2."""

    def test_combinatorial_q_zero_matches_symmetric(self):
        # Directed cycle — has non-trivial asymmetric part.
        A = _directed_cycle(6)
        A_s = (A + A.mT) * 0.5
        L_mag = magnetic.combinatorial(A, q=0.0)
        L_sym = laplacian.combinatorial(A_s).to(L_mag.dtype)
        torch.testing.assert_close(L_mag, L_sym, atol=1e-12, rtol=0)

    def test_symmetric_normalized_q_zero_matches_real(self):
        A = _directed_cycle(7)
        A_s = (A + A.mT) * 0.5
        L_mag = magnetic.symmetric_normalized(A, q=0.0)
        L_real = laplacian.symmetric_normalized(A_s).to(L_mag.dtype)
        torch.testing.assert_close(L_mag, L_real, atol=1e-12, rtol=0)


# --------------------------------------------------------------------
# Hermitian output, real eigenvalues
# --------------------------------------------------------------------


class TestHermitianProperty:
    """The magnetic Laplacian is Hermitian by construction: L = L.conj().T.
    Its eigenvalues are therefore real, and torch.linalg.eigh succeeds."""

    def test_combinatorial_is_hermitian(self):
        A = _directed_cycle(8)
        L = magnetic.combinatorial(A, q=0.25)
        torch.testing.assert_close(
            L, L.conj().mT, atol=1e-12, rtol=0,
        )

    def test_symmetric_normalized_is_hermitian(self):
        A = _directed_cycle(8)
        L = magnetic.symmetric_normalized(A, q=0.25)
        torch.testing.assert_close(
            L, L.conj().mT, atol=1e-12, rtol=0,
        )

    def test_eigenvalues_are_real(self):
        A = _directed_cycle(8)
        L = magnetic.symmetric_normalized(A, q=0.25)
        eigvals = torch.linalg.eigvalsh(L)
        # eigvalsh returns real-valued tensor; assert dtype.
        assert eigvals.dtype in (torch.float32, torch.float64)
        # Spectrum bounded in [0, 2] for normalized form
        # (Furutani 2020, Prop. 1).
        assert (eigvals >= -1e-10).all()
        assert (eigvals <= 2 + 1e-10).all()


# --------------------------------------------------------------------
# Symmetric input: q has no effect
# --------------------------------------------------------------------


class TestSymmetricInput:
    """For symmetric A, A - A.T = 0 → phase is identically 1
    regardless of q. The magnetic Laplacian must agree across q
    values and match the real Laplacian of A."""

    def test_combinatorial_independent_of_q(self):
        A = _undirected_cycle(8)
        L_q0 = magnetic.combinatorial(A, q=0.0)
        L_q25 = magnetic.combinatorial(A, q=0.25)
        L_q5 = magnetic.combinatorial(A, q=0.5)
        torch.testing.assert_close(L_q0, L_q25, atol=1e-12, rtol=0)
        torch.testing.assert_close(L_q0, L_q5, atol=1e-12, rtol=0)

    def test_symmetric_normalized_independent_of_q(self):
        A = _undirected_cycle(7)
        L_q0 = magnetic.symmetric_normalized(A, q=0.0)
        L_q25 = magnetic.symmetric_normalized(A, q=0.25)
        torch.testing.assert_close(L_q0, L_q25, atol=1e-12, rtol=0)


# --------------------------------------------------------------------
# Directed graph: phase actually moves the spectrum
# --------------------------------------------------------------------


class TestDirectedSensitivity:
    """For a non-trivially directed graph, varying q changes the
    Laplacian non-trivially. This pins down that the phase factor
    is actually being applied."""

    def test_q_changes_spectrum_on_directed_cycle(self):
        A = _directed_cycle(6)
        L0 = magnetic.symmetric_normalized(A, q=0.0)
        L25 = magnetic.symmetric_normalized(A, q=0.25)
        # Eigenvalues should differ once q is non-zero on a directed graph.
        eig0 = torch.linalg.eigvalsh(L0).sort().values
        eig25 = torch.linalg.eigvalsh(L25).sort().values
        diff = (eig0 - eig25).abs().max().item()
        assert diff > 1e-3, (
            f"directed cycle's spectrum should shift between q=0 and q=0.25; "
            f"max|Δλ|={diff:.4e}"
        )


# --------------------------------------------------------------------
# Isolated nodes use Moore-Penrose convention
# --------------------------------------------------------------------


class TestIsolatedNodes:
    def test_isolated_node_normalized_does_not_explode(self):
        """An isolated node should not produce NaN/inf in the normalized
        magnetic Laplacian (consistent with the real-Laplacian
        Moore-Penrose convention)."""
        n = 4
        A = torch.zeros(1, n, n, dtype=torch.float64)
        # Triangle 0-1-2 (undirected), node 3 isolated.
        A[0, 0, 1] = A[0, 1, 0] = 1.0
        A[0, 1, 2] = A[0, 2, 1] = 1.0
        A[0, 0, 2] = A[0, 2, 0] = 1.0
        L = magnetic.symmetric_normalized(A, q=0.25)
        assert torch.isfinite(L.real).all()
        assert torch.isfinite(L.imag).all()


# --------------------------------------------------------------------
# Batching shapes
# --------------------------------------------------------------------


@pytest.mark.parametrize("batch", [1, 2, 4])
class TestShapes:
    def test_combinatorial_shape(self, batch):
        A = _directed_cycle(5, batch=batch)
        L = magnetic.combinatorial(A, q=0.25)
        assert L.shape == (batch, 5, 5)
        assert L.dtype.is_complex

    def test_normalized_shape(self, batch):
        A = _directed_cycle(5, batch=batch)
        L = magnetic.symmetric_normalized(A, q=0.25)
        assert L.shape == (batch, 5, 5)
        assert L.dtype.is_complex


# ====================================================================
# Sign-magnetic Laplacian: extends magnetic to signed-directed graphs.
# Must reduce correctly to magnetic / signed / standard in their
# respective limits, and stay Hermitian for arbitrary real input.
# ====================================================================


def _signed_directed(n: int, batch: int = 1, dtype=torch.float64) -> torch.Tensor:
    """Cycle with alternating-sign edges traversed forward: a signed
    directed graph with non-trivial sign AND non-trivial direction.
    """
    A = torch.zeros(batch, n, n, dtype=dtype)
    for i in range(n):
        sign = 1.0 if i % 2 == 0 else -1.0
        A[:, i, (i + 1) % n] = sign
    return A


def _signed_undirected(n: int, batch: int = 1, dtype=torch.float64) -> torch.Tensor:
    """Symmetric matrix with mixed signs."""
    A = _signed_directed(n, batch=batch, dtype=dtype)
    return 0.5 * (A + A.mT)


class TestSignMagneticValidation:
    def test_rejects_non_square(self):
        with pytest.raises(ValueError, match="must be"):
            magnetic.sign_magnetic_combinatorial(torch.zeros(1, 4, 5))

    def test_rejects_q_out_of_range(self):
        A = torch.zeros(1, 3, 3, dtype=torch.float64)
        with pytest.raises(ValueError, match="q"):
            magnetic.sign_magnetic_combinatorial(A, q=1.0)
        with pytest.raises(ValueError, match="q"):
            magnetic.sign_magnetic_symmetric_normalized(A, q=-0.5)

    def test_rejects_complex_input(self):
        A = torch.zeros(1, 3, 3, dtype=torch.complex128)
        with pytest.raises(ValueError, match="real"):
            magnetic.sign_magnetic_combinatorial(A, q=0.25)


class TestSignMagneticReductions:
    """The sign-magnetic Laplacian must collapse to existing library
    primitives in each special case."""

    def test_unsigned_reduces_to_magnetic_combinatorial(self):
        """A ≥ 0 → sign-magnetic == magnetic (signs are absent)."""
        A = _directed_cycle(6)
        L_sign = magnetic.sign_magnetic_combinatorial(A, q=0.25)
        L_mag = magnetic.combinatorial(A, q=0.25)
        torch.testing.assert_close(L_sign, L_mag, atol=1e-12, rtol=0)

    def test_unsigned_reduces_to_magnetic_normalized(self):
        A = _directed_cycle(7)
        L_sign = magnetic.sign_magnetic_symmetric_normalized(A, q=0.25)
        L_mag = magnetic.symmetric_normalized(A, q=0.25)
        torch.testing.assert_close(L_sign, L_mag, atol=1e-12, rtol=0)

    def test_signed_symmetric_reduces_to_kunegis_signed(self):
        """A = A^T → phase ≡ 1 → sign-magnetic ≡ Kunegis signed Laplacian
        (cast to complex). This is the q-doesn't-matter regime for
        signed undirected graphs."""
        A = _signed_undirected(8)
        for q in (0.0, 0.25, 0.5):
            L_sign = magnetic.sign_magnetic_combinatorial(A, q=q)
            L_kun = laplacian.signed(A).to(L_sign.dtype)
            torch.testing.assert_close(L_sign, L_kun, atol=1e-12, rtol=0)

    def test_unsigned_symmetric_reduces_to_standard(self):
        """A ≥ 0 ∧ A = A^T → sign-magnetic ≡ standard combinatorial L."""
        A = _undirected_cycle(6)
        L_sign = magnetic.sign_magnetic_combinatorial(A, q=0.25)
        L_std = laplacian.combinatorial(A).to(L_sign.dtype)
        torch.testing.assert_close(L_sign, L_std, atol=1e-12, rtol=0)

    def test_q_zero_combinatorial_is_real_signed(self):
        """q = 0 collapses to (D_|.| − A_s^signed) as complex."""
        A = _signed_directed(6)
        A_s_signed = 0.5 * (A + A.mT)
        d_abs = 0.5 * (A.abs() + A.abs().mT).sum(dim=-1)
        L_expected_real = torch.diag_embed(d_abs) - A_s_signed
        L = magnetic.sign_magnetic_combinatorial(A, q=0.0)
        L_expected = L_expected_real.to(L.dtype)
        torch.testing.assert_close(L, L_expected, atol=1e-12, rtol=0)


class TestSignMagneticHermitian:
    """L = L^H for any real A, any q."""

    def test_combinatorial_hermitian(self):
        A = _signed_directed(8)
        L = magnetic.sign_magnetic_combinatorial(A, q=0.25)
        torch.testing.assert_close(L, L.conj().mT, atol=1e-12, rtol=0)

    def test_normalized_hermitian(self):
        A = _signed_directed(8)
        L = magnetic.sign_magnetic_symmetric_normalized(A, q=0.25)
        torch.testing.assert_close(L, L.conj().mT, atol=1e-12, rtol=0)

    def test_eigenvalues_real(self):
        A = _signed_directed(7)
        L = magnetic.sign_magnetic_combinatorial(A, q=0.25)
        eigvals = torch.linalg.eigvalsh(L)
        assert eigvals.dtype in (torch.float32, torch.float64)


def _unbalanced_signed_directed(dtype=torch.float64) -> torch.Tensor:
    """A signed-directed graph that is NOT balanced (no 2-coloring
    absorbs the signs into vertex gauge). The 6-cycle with
    alternating signs IS balanced and gauge-equivalent to the
    unsigned cycle — its spectrum is invariant under sign flips
    and under q, so it's useless for sensitivity tests.

    Construction: directed cycle (0→1, 1→2, ..., 5→0) with all edges
    positive EXCEPT 0→1 negative. This breaks the even-cycle parity
    that lets gauges absorb signs.
    """
    n = 6
    A = torch.zeros(1, n, n, dtype=dtype)
    for i in range(n):
        A[:, i, (i + 1) % n] = 1.0
    A[:, 0, 1] = -1.0
    return A


class TestSignMagneticDirectedSensitivity:
    """For a signed-directed graph, both q AND sign matter — flipping
    edge signs or changing q must move the spectrum. Uses an
    UNBALANCED signed graph: balanced graphs are gauge-equivalent to
    their unsigned counterparts and have spectrum invariant under
    sign flips, which would defeat the test."""

    def test_q_shifts_spectrum_on_signed_directed(self):
        A = _unbalanced_signed_directed()
        L0 = magnetic.sign_magnetic_symmetric_normalized(A, q=0.0)
        L25 = magnetic.sign_magnetic_symmetric_normalized(A, q=0.25)
        eig0 = torch.linalg.eigvalsh(L0).sort().values
        eig25 = torch.linalg.eigvalsh(L25).sort().values
        diff = (eig0 - eig25).abs().max().item()
        assert diff > 1e-3, f"q should change spectrum; got max|Δλ|={diff:.4e}"

    def test_signs_change_spectrum(self):
        """Flipping a single sign on an otherwise-positive directed
        cycle moves the spectrum (the resulting signed cycle is
        unbalanced, so sign info is not absorbable by gauge)."""
        A_signed = _unbalanced_signed_directed()
        A_unsigned = A_signed.abs()
        L_signed = magnetic.sign_magnetic_combinatorial(A_signed, q=0.25)
        L_unsigned = magnetic.sign_magnetic_combinatorial(A_unsigned, q=0.25)
        diff = (
            torch.linalg.eigvalsh(L_signed).sort().values
            - torch.linalg.eigvalsh(L_unsigned).sort().values
        ).abs().max().item()
        assert diff > 1e-3


class TestSignMagneticIsolatedNodes:
    def test_isolated_node_normalized_finite(self):
        n = 4
        A = torch.zeros(1, n, n, dtype=torch.float64)
        A[0, 0, 1] = 1.0;  A[0, 1, 0] = -1.0          # signed directed edge
        A[0, 1, 2] = -1.0; A[0, 2, 1] = 1.0
        # node 3 isolated
        L = magnetic.sign_magnetic_symmetric_normalized(A, q=0.25)
        assert torch.isfinite(L.real).all()
        assert torch.isfinite(L.imag).all()


@pytest.mark.parametrize("batch", [1, 2, 4])
class TestSignMagneticShapes:
    def test_combinatorial_shape(self, batch):
        A = _signed_directed(5, batch=batch)
        L = magnetic.sign_magnetic_combinatorial(A, q=0.25)
        assert L.shape == (batch, 5, 5)
        assert L.dtype.is_complex

    def test_normalized_shape(self, batch):
        A = _signed_directed(5, batch=batch)
        L = magnetic.sign_magnetic_symmetric_normalized(A, q=0.25)
        assert L.shape == (batch, 5, 5)
        assert L.dtype.is_complex
