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
  combinatorial(A, q)                  : L^(q) = D_s − H ⊙ A_s
  symmetric_normalized(A, q)           : L_sym^(q) = I − D_s^{−1/2} (H ⊙ A_s) D_s^{−1/2}
  sign_magnetic_combinatorial(A, q)    : signed-directed analog,
                                          unifies Kunegis signed + magnetic
  sign_magnetic_symmetric_normalized(A, q)  : normalized signed-directed

The signed forms implement MSGNN (He, Perlmutter, Reinert, Cucuringu
2022, equations 1 and 2). They use the signed symmetric adjacency
A_s = (A + A^T)/2, absolute degree D_abs[i,i] = sum_j
(|A[i,j]| + |A[j,i]|)/2, and phase exp(i 2π q (A - A^T)).
Both operators are positive semidefinite; the normalized spectrum
lies in [0, 2]. Nonnegative input gives the ordinary magnetic
Laplacian. Symmetric signed input gives the signed Laplacian for
any charge. At q = 0 the phase disappears, leaving D_abs - A_s.
For opposite-sign reciprocal edges, D_abs differs from sum_j |A_s|;
this distinction is part of MSGNN's definition.

All four functions return a complex Hermitian tensor of shape
(B, n, n). Use `torch.linalg.eigh` on the result : its real-eigenvalue
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
  He, Y., Perlmutter, M., Reinert, G., Cucuringu, M. (2022). MSGNN:
    A Spectral Graph Neural Network Based on a Novel Magnetic Signed
    Laplacian. LoG 2022, PMLR 198, §3.2, equations 1 and 2.
"""

from __future__ import annotations

import math

import torch

from holonomy_lib._graph_utils import drop_self_loops

# Default magnetic charge q ∈ [0, 1) per Furutani et al. (2020), §5.
# `q = 1/4` is the value that maximally separates the directed
# eigenmodes for a cycle and is therefore the standard literature
# default. **Scale of validity**: dimensionless ratio of magnetic flux
# units; does not depend on n. Cataloged as `MAGNETIC_CHARGE_DEFAULT`.
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

    For nonnegative A, spectrum ⊂ [0, 2] regardless of q (Furutani 2020, Prop. 1). Use
    this form for spectral embeddings of directed graphs; the
    eigenvectors at the bottom-k eigenvalues are the directed analog
    of Laplacian eigenmaps.

    Args:
      A: (B, n, n) nonnegative weighted adjacency. May be non-symmetric.
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

    # Moore-Penrose pseudoinverse handling for isolated nodes, the same
    # convention as `holonomy_lib.spectral.laplacian.symmetric_normalized`.
    # The clamp keeps rsqrt finite on the branch `where` throws away, so
    # an isolated node never puts an inf into the autograd graph
    floor = torch.finfo(A.dtype).tiny
    safe_D = D_s.clamp(min=floor)
    d_inv_sqrt = torch.where(
        D_s > 0, torch.rsqrt(safe_D), torch.zeros_like(D_s),
    )                                              # (B, n)

    H_times_As = _magnetic_phase_times_adj(A, A_s, q)

    # D^{-1/2} (H ⊙ A_s) D^{-1/2} as an outer product of the two scalings.
    # The real scale has to be cast to complex before it multiplies the
    # complex phase matrix
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

    L^(q,σ) = D_|.| − exp(i · 2π · q · (A − A^T)) ⊙ A_s^signed

    where:
      A_s^signed = (A + A^T) / 2                   (signed symmetric part)
      A_s^|.|    = (|A| + |A|^T) / 2               (magnitude symmetric part)
      D_|.|      = diag(Σ_j A_s^|.|[i, j])         (signless degree, à la
                                                    Kunegis 2010)

    Phase uses signed A, as in MSGNN equation 1. At q = 0 the phase
    disappears. The absolute degree is computed before symmetrization,
    so opposite-sign reciprocal edges still contribute to the degree.

    Reduces to:
      magnetic.combinatorial(A, q)   if A ≥ 0      (signs irrelevant)
      laplacian.signed(A)            if A = A^T    (Θ ≡ 0, signs visible)
      laplacian.combinatorial(A)     if A ≥ 0 ∧ A = A^T

    Args:
      A: (B, n, n) real-valued adjacency. May be non-symmetric and may
        have negative entries.
      q: magnetic charge in [0, 1). Default 1/4.

    Returns:
      L: (B, n, n) complex Hermitian positive-semidefinite Laplacian.

    References:
      He, Perlmutter, Reinert, Cucuringu (2022). MSGNN. LoG 2022,
        PMLR 198, §3.2, equation 1 and Theorem 1.
      Kunegis et al. (2010) for the q = 0 (signed-undirected) reduction.
    """
    _validate_real_adjacency(A, q)
    A = drop_self_loops(A)
    abs_A = A.abs()
    A_s_signed = 0.5 * (A + A.mT)
    A_s_abs = 0.5 * (abs_A + abs_A.mT)
    D_abs = A_s_abs.sum(dim=-1)                       # (B, n)

    H_times_As = _magnetic_phase_times_adj(A, A_s_signed, q)
    D_diag = torch.diag_embed(D_abs).to(H_times_As.dtype)
    return D_diag - H_times_As


def sign_magnetic_symmetric_normalized(
    A: torch.Tensor, q: float = MAGNETIC_CHARGE_DEFAULT,
) -> torch.Tensor:
    """Sign-magnetic symmetric-normalized Laplacian.

    L_sym^(q,σ) = I − D_|.|^{−1/2} (exp(iΘ) ⊙ A_s^signed) D_|.|^{−1/2}

    where Θ_{ij} = 2π · q · (A_{ij} − A_{ji}) and
    D_|.|[i,i] = sum_j (|A[i,j]| + |A[j,i]|)/2. Use this form for
    spectral embeddings of signed-directed graphs; the eigenvectors at
    the bottom-k eigenvalues are the signed-directed analog of
    Laplacian eigenmaps.

    Args:
      A: (B, n, n) real-valued adjacency, may be non-symmetric and signed.
      q: magnetic charge in [0, 1).

    Returns:
      L_sym^(q,σ): (B, n, n) complex Hermitian, spectrum in [0, 2].
        Isolated nodes have diagonal 1 under the pseudoinverse convention.

    References:
      He, Perlmutter, Reinert, Cucuringu (2022). MSGNN. LoG 2022,
        PMLR 198, §3.2, equation 2 and Theorems 1 and 2.
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

    H_times_As = _magnetic_phase_times_adj(A, A_s_signed, q)

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

    Takes the adjacency and charge, returns nothing, raises `ValueError`
    if `A` is not square in its last two dims, is not a real floating
    dtype (complex and integer tensors both fail), or `q` is outside
    [0, 1). Complex `A` is rejected outright: the phase factor
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

    Takes the real adjacency `A` (for the phase) and the already
    symmetrized `A_s` (signed or unsigned, the caller picks), both
    `(..., n, n)`, and returns one complex tensor of the same shape. For
    symmetric A the phase is identically 1 and the result reduces to
    A_s + 0j, but we still pay for building the phase. For q = 0 the
    phase is also identically 1, and that case skips the trig entirely.
    """
    if q == 0.0:
        # no phase at all, so the answer is just A_s cast to complex
        complex_dtype = _matching_complex_dtype(A.dtype)
        return A_s.to(complex_dtype)

    # the antisymmetric part A - A^T carries the direction of each edge.
    # For a symmetric input it is all zeros, the phase is 1 everywhere,
    # and q has no effect, which is the reduction the module promises
    asym = A - A.mT                                # (B, n, n)
    angle = (2.0 * math.pi * q) * asym             # (B, n, n) real
    phase = torch.complex(torch.cos(angle), torch.sin(angle))
    complex_dtype = phase.dtype
    return phase * A_s.to(complex_dtype)


def _matching_complex_dtype(real_dtype: torch.dtype) -> torch.dtype:
    """Pick a complex dtype matching the precision of a real dtype.

    float64 maps to complex128 and float32 to complex64. Anything else
    (half, bfloat16) goes up to complex64, since half-precision complex
    support in torch is thin.
    """
    if real_dtype == torch.float64:
        return torch.complex128
    if real_dtype == torch.float32:
        return torch.complex64
    return torch.complex64
