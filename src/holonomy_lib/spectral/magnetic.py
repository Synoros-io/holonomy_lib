# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Synoros

"""Magnetic Laplacian for directed and signed-directed graphs.

The standard real-valued graph Laplacians (`spectral.laplacian.*`)
assume a symmetric adjacency. For a directed graph the asymmetry
encodes information that any symmetrization throws away. The
**magnetic Laplacian** (Lieb-Loss 1993; Fanuel et al. 2017; Furutani
et al. 2020) recovers that information by attaching a complex phase
factor `e^{i · 2π · q · (A_{ij} - A_{ji})}` to each edge, while keeping
the magnitude structure symmetric. The result is a Hermitian operator
whose eigenvalues are real and whose eigenvectors are complex; the
phase of those eigenvectors carries the directional information.

Definition (combinatorial form):

  A_s   = (A + A^T) / 2                                (symmetrized)
  D_s   = diag(Σ_j A_s[i, j])                          (symmetric degree)
  H     = exp(i · 2π · q · (A - A^T))                  (Hermitian phase factor)
  L^(q) = D_s - H ⊙ A_s

For symmetric A this reduces to L = D - A regardless of q. For purely
directed A (entries in {0, 1}, A ≠ A^T) the phase encodes "flow
direction": q = 1/4 separates directed eigenmodes most clearly
(Furutani 2020, §5).

Four forms exposed:
  combinatorial(A, q)                  — L^(q) = D_s − H ⊙ A_s
  symmetric_normalized(A, q)           — L_sym^(q) = I − D_s^{−1/2} (H ⊙ A_s) D_s^{−1/2}
  sign_magnetic_combinatorial(A, q)    — signed-directed analog,
                                          unifies Kunegis signed + magnetic
  sign_magnetic_symmetric_normalized(A, q)  — normalized signed-directed

The signed-magnetic forms accept arbitrary real `A` (entries may be
negative AND the matrix may be non-symmetric). They keep the sign of
each edge in the off-diagonal `A_s^signed = (A + A^T)/2` but use the
signless degree `D^|.| = Σ_j |A_s|_{ij}` for the diagonal, exactly as
the Kunegis (2010) signed Laplacian does. The phase argument uses the
signed antisymmetric part `A − A^T`, so opposite-sign edges contribute
opposing phase shifts. The result is Hermitian and reduces correctly
to every existing primitive in this library:

  unsigned A ≥ 0                → magnetic.combinatorial(A, q)
  signed symmetric A            → laplacian.signed(A) (cast to complex)
  unsigned symmetric A          → standard combinatorial Laplacian
  zero-charge q = 0             → real Kunegis signed Laplacian of A_s^signed

The phase argument in the signed-magnetic forms uses |A| (magnitudes),
not signed A. That keeps direction and sign independent: flipping a
single directed edge's sign genuinely changes the spectrum, instead of
collapsing into the gauge-equivalent unsigned magnetic spectrum.

All four functions return a complex Hermitian tensor of shape
(B, n, n). Use `torch.linalg.eigh` on the result — its real-eigenvalue
output is the spectrum and its complex eigenvectors are the
directed-graph modes.

References:
  Lieb, E. H., Loss, M. (1993). Fluxes, Laplacians, and Kasteleyn's
    theorem. Duke Mathematical Journal 71(2):337–363. The original
    magnetic / flux-discrete Laplacian on graphs.
  Fanuel, M., Alaiz, C. M., Suykens, J. A. K. (2017). Magnetic
    eigenmaps for the visualization of directed networks. Applied and
    Computational Harmonic Analysis 44(1):189–199. The modern spectral-
    embedding application.
  Furutani, S., Shibahara, T., Akiyama, M., Hato, K., Aida, M. (2020).
    Graph Signal Processing for Directed Graphs based on the Hermitian
    Laplacian. Journal of Machine Learning Research 21(122):1–37.
    Gives the q = 1/4 default and the normalization conventions used
    here. Section 5 covers the spectral interpretation.
  Kunegis, J., Schmidt, S., Lommatzsch, A., Lerner, J., De Luca, E. W.,
    Albayrak, S. (2010). Spectral analysis of signed graphs for
    clustering, prediction and visualization. SDM 2010. The signed
    Laplacian that the q = 0 signed-magnetic case reduces to.
  Fiorini, S., Coniglio, S., Ciavotta, M., Messina, E. (2023). SigMaNet:
    One Laplacian to Rule Them All. AAAI 2023. The sign-magnetic
    Laplacian recipe: signless degree on the diagonal, signed
    symmetrized adjacency off-diagonal, directional phase factor.
  He, Y., Reinert, G., Wang, S., Cucuringu, M. (2023). MSGNN: A
    Spectral Graph Neural Network Based on a Novel Magnetic Signed
    Laplacian. LoG 2022 / TMLR 2023. Closely related formulation.
"""

from __future__ import annotations

import math

import torch

from holonomy_lib._graph_utils import drop_self_loops

# Default magnetic charge q ∈ [0, 1) per Furutani et al. (2020), §5.
# `q = 1/4` is the value that maximally separates the directed
# eigenmodes for a cycle and is therefore the standard literature
# default. **Scale of validity**: dimensionless ratio of magnetic flux
# units; does not depend on n. Cataloged as `magnetic_charge_default`.
MAGNETIC_CHARGE_DEFAULT: float = 0.25


def combinatorial(
    A: torch.Tensor, q: float = MAGNETIC_CHARGE_DEFAULT,
) -> torch.Tensor:
    """Combinatorial magnetic Laplacian L^(q) = D_s − H ⊙ A_s.

    Args:
      A: (B, n, n) weighted adjacency. May be non-symmetric. Entries
        may be negative (signed-directed graphs); the symmetrized
        adjacency `A_s = (A + A^T)/2` keeps the "edge presence",
        and the antisymmetric part `A − A^T` enters via the phase.
      q: magnetic charge in [0, 1). Default 1/4 per Furutani 2020 §5.

    Returns:
      L: (B, n, n) complex Hermitian Laplacian. Use `torch.linalg.eigh`
        to get real eigenvalues + complex eigenvectors.

    References:
      Furutani et al. (2020), eq. 8.
      Lieb-Loss (1993).
    """
    _validate_real_adjacency(A, q)
    A = drop_self_loops(A)
    A_s = 0.5 * (A + A.mT)                       # symmetrized magnitude
    D_s = A_s.sum(dim=-1)                         # (B, n) symmetric degree

    H_times_As = _magnetic_phase_times_adj(A, A_s, q)

    # D_s on the diagonal, embedded in the complex dtype.
    D_diag = torch.diag_embed(D_s).to(H_times_As.dtype)
    return D_diag - H_times_As


def symmetric_normalized(
    A: torch.Tensor, q: float = MAGNETIC_CHARGE_DEFAULT,
) -> torch.Tensor:
    """Symmetric-normalized magnetic Laplacian.

    L_sym^(q) = I − D_s^{−1/2} (H ⊙ A_s) D_s^{−1/2}

    Spectrum ⊂ [0, 2] regardless of q (Furutani 2020, Prop. 1). Use
    this form for spectral embeddings of directed graphs; the
    eigenvectors at the bottom-k eigenvalues are the directed analog
    of Laplacian eigenmaps.

    Args:
      A: (B, n, n) weighted adjacency. May be non-symmetric.
      q: magnetic charge in [0, 1).

    Returns:
      L_sym^(q): (B, n, n) complex Hermitian, spectrum in [0, 2].

    References:
      Furutani et al. (2020), Definition 4 and Prop. 1.
    """
    _validate_real_adjacency(A, q)
    A = drop_self_loops(A)
    A_s = 0.5 * (A + A.mT)
    D_s = A_s.sum(dim=-1)                         # (B, n)

    # Moore-Penrose pseudoinverse handling for isolated nodes — same
    # convention as `holonomy_lib.spectral.laplacian.symmetric_normalized`.
    # The library-wide `1e-9` (numerical_floor_convention) doubles as
    # the dtype-tiny floor below: for any reasonable dtype, `tiny` is
    # well below 1e-9, so `where(D_s > 0, ...)` selects the same set.
    floor = torch.finfo(A.dtype).tiny
    safe_D = D_s.clamp(min=floor)
    d_inv_sqrt = torch.where(
        D_s > 0, torch.rsqrt(safe_D), torch.zeros_like(D_s),
    )                                              # (B, n)

    H_times_As = _magnetic_phase_times_adj(A, A_s, q)

    # D^{-1/2} H ⊙ A_s D^{-1/2} via broadcasting.
    scale = (
        d_inv_sqrt.unsqueeze(dim=-1) * d_inv_sqrt.unsqueeze(dim=-2)
    ).to(H_times_As.dtype)
    normalized = H_times_As * scale

    n = A.shape[-1]
    eye = torch.eye(n, device=A.device, dtype=H_times_As.dtype).expand_as(
        normalized,
    )
    return eye - normalized


# ============================================================
# Sign-magnetic Laplacian (signed-directed graphs)
# ============================================================


def sign_magnetic_combinatorial(
    A: torch.Tensor, q: float = MAGNETIC_CHARGE_DEFAULT,
) -> torch.Tensor:
    """Sign-magnetic combinatorial Laplacian for signed-directed graphs.

    L^(q,σ) = D_|.| − exp(i · 2π · q · (|A| − |A|^T)) ⊙ A_s^signed

    where:
      A_s^signed = (A + A^T) / 2                   (signed symmetric part)
      A_s^|.|    = (|A| + |A|^T) / 2               (magnitude symmetric part)
      D_|.|      = diag(Σ_j A_s^|.|[i, j])         (signless degree, à la
                                                    Kunegis 2010)

    Phase uses |A| (magnitudes) so direction and sign stay independent.

    Reduces to:
      magnetic.combinatorial(A, q)   if A ≥ 0      (signs irrelevant)
      laplacian.signed(A)            if A = A^T    (Θ ≡ 0, signs visible)
      laplacian.combinatorial(A)     if A ≥ 0 ∧ A = A^T

    Args:
      A: (B, n, n) real-valued adjacency. May be non-symmetric and may
        have negative entries.
      q: magnetic charge in [0, 1). Default 1/4.

    Returns:
      L: (B, n, n) complex Hermitian Laplacian, PSD when q = 0 and the
        underlying signed graph is balanced (Kunegis 2010, Theorem 3.4).

    References:
      Fiorini et al. (2023) SigMaNet, eq. 6.
      Kunegis et al. (2010) for the q = 0 (signed-undirected) reduction.
    """
    _validate_real_adjacency(A, q)
    A = drop_self_loops(A)
    abs_A = A.abs()
    A_s_signed = 0.5 * (A + A.mT)
    A_s_abs = 0.5 * (abs_A + abs_A.mT)
    D_abs = A_s_abs.sum(dim=-1)                       # (B, n)

    H_times_As = _sign_magnetic_phase_times_adj(A, abs_A, A_s_signed, q)
    D_diag = torch.diag_embed(D_abs).to(H_times_As.dtype)
    return D_diag - H_times_As


def sign_magnetic_symmetric_normalized(
    A: torch.Tensor, q: float = MAGNETIC_CHARGE_DEFAULT,
) -> torch.Tensor:
    """Sign-magnetic symmetric-normalized Laplacian.

    L_sym^(q,σ) = I − D_|.|^{−1/2} (exp(iΘ) ⊙ A_s^signed) D_|.|^{−1/2}

    where Θ_{ij} = 2π · q · (|A|_{ij} − |A|_{ji}). Use this form for
    spectral embeddings of signed-directed graphs; the eigenvectors at
    the bottom-k eigenvalues are the signed-directed analog of
    Laplacian eigenmaps.

    Args:
      A: (B, n, n) real-valued adjacency, may be non-symmetric and signed.
      q: magnetic charge in [0, 1).

    Returns:
      L_sym^(q,σ): (B, n, n) complex Hermitian.

    References:
      Fiorini et al. (2023) SigMaNet, normalized form.
    """
    _validate_real_adjacency(A, q)
    A = drop_self_loops(A)
    abs_A = A.abs()
    A_s_signed = 0.5 * (A + A.mT)
    A_s_abs = 0.5 * (abs_A + abs_A.mT)
    D_abs = A_s_abs.sum(dim=-1)                       # (B, n)

    # Same pseudoinverse handling as `laplacian.symmetric_normalized` and
    # `magnetic.symmetric_normalized`: isolated nodes get d^{-1/2} = 0.
    floor = torch.finfo(A.dtype).tiny
    safe_D = D_abs.clamp(min=floor)
    d_inv_sqrt = torch.where(
        D_abs > 0, torch.rsqrt(safe_D), torch.zeros_like(D_abs),
    )                                                  # (B, n)

    H_times_As = _sign_magnetic_phase_times_adj(A, abs_A, A_s_signed, q)

    scale = (
        d_inv_sqrt.unsqueeze(dim=-1) * d_inv_sqrt.unsqueeze(dim=-2)
    ).to(H_times_As.dtype)
    normalized = H_times_As * scale

    n = A.shape[-1]
    eye = torch.eye(n, device=A.device, dtype=H_times_As.dtype).expand_as(
        normalized,
    )
    return eye - normalized


# ============================================================
# Internal helpers
# ============================================================


def _validate_real_adjacency(A: torch.Tensor, q: float) -> None:
    """Shared input validation for the magnetic Laplacian primitives.

    Rejects complex-valued `A` outright: the phase factor
    `H = exp(i · 2π · q · (A − A^T))` is only defined for real
    antisymmetric arguments. Passing a complex `A` would let
    `torch.cos`/`torch.sin` produce complex outputs and
    `torch.complex(cos, sin)` would raise an opaque "expected real
    tensors" error deep inside the helper.
    """
    if A.ndim < 2 or A.shape[-1] != A.shape[-2]:
        raise ValueError(
            f"A must be (..., n, n); got A.shape={tuple(A.shape)}"
        )
    if not A.is_floating_point():
        raise ValueError(
            f"A must be a real floating-point tensor; got dtype={A.dtype}. "
            f"The magnetic Laplacian's phase factor is defined for real "
            f"adjacencies only."
        )
    if not (0.0 <= q < 1.0):
        raise ValueError(f"magnetic charge q must be in [0, 1), got {q}")


def _magnetic_phase_times_adj(
    A: torch.Tensor, A_s: torch.Tensor, q: float,
) -> torch.Tensor:
    """Return H ⊙ A_s where H_{ij} = exp(i · 2π · q · (A_{ij} - A_{ji})).

    Computed as a single complex tensor of shape (*A_s.shape).
    For symmetric A the phase is identically 1 and the result reduces
    to A_s + 0j — but we still pay the complex allocation. For q = 0
    the phase is also identically 1; we short-circuit that case.
    """
    if q == 0.0:
        # Phase is identically 1; result is A_s as a complex tensor.
        # Cast to the appropriate complex dtype matching A.dtype.
        complex_dtype = _matching_complex_dtype(A.dtype)
        return A_s.to(complex_dtype)

    # Antisymmetric part of A; for purely symmetric inputs this is 0
    # and we land in the q-doesn't-matter regime anyway.
    asym = A - A.mT                                # (B, n, n)
    angle = (2.0 * math.pi * q) * asym             # (B, n, n) real
    phase = torch.complex(torch.cos(angle), torch.sin(angle))
    complex_dtype = phase.dtype
    return phase * A_s.to(complex_dtype)


def _sign_magnetic_phase_times_adj(
    A: torch.Tensor, abs_A: torch.Tensor,
    A_s_signed: torch.Tensor, q: float,
) -> torch.Tensor:
    """Return exp(i · 2π · q · (|A| − |A|^T)) ⊙ A_s_signed.

    Callers pass `abs_A = A.abs()` so we don't recompute it (each
    sign-magnetic entry point already needs the magnitudes for the
    signless degree, so hoisting saves one extra `.abs()` kernel
    launch per call).

    The phase argument uses **magnitudes** |A| (not signed A) so that
    direction and sign are independent: flipping a single directed
    edge's sign genuinely changes the result. Using signed A in the
    phase would let sign and direction collapse into each other for
    purely directed edges (sign-flip = direction-flip + a phase-
    conjugate cancellation), making the operator insensitive to signs.

    Reductions still hold:
      - For A ≥ 0: |A| = A → phase matches magnetic.combinatorial,
        A_s_signed = A_s, total reduces to the standard magnetic
        Laplacian.
      - For A = A^T: |A| is symmetric → phase ≡ 1, A_s_signed = A,
        total reduces to D_|.| − A which is the Kunegis signed
        Laplacian after taking the signless degree on the diagonal.
    """
    if q == 0.0:
        complex_dtype = _matching_complex_dtype(A.dtype)
        return A_s_signed.to(complex_dtype)

    asym = abs_A - abs_A.mT                            # (B, n, n)
    angle = (2.0 * math.pi * q) * asym                 # (B, n, n) real
    phase = torch.complex(torch.cos(angle), torch.sin(angle))
    complex_dtype = phase.dtype
    return phase * A_s_signed.to(complex_dtype)


def _matching_complex_dtype(real_dtype: torch.dtype) -> torch.dtype:
    """Pick a complex dtype matching the precision of a real dtype."""
    if real_dtype == torch.float64:
        return torch.complex128
    if real_dtype == torch.float32:
        return torch.complex64
    # Fallback for half-precision and others — go up to complex64.
    return torch.complex64
