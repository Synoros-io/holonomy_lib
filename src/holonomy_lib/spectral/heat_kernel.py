# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Synoros

"""Heat kernel exp(-t·L) on graphs via Chebyshev-polynomial expansion.

The graph heat kernel `K_t = exp(-t · L)` is the diffusion operator
underlying spectral graph wavelets, diffusion maps, label propagation,
graph neural net normalizers, and a dozen other ML primitives. The
direct route through eigendecomposition is O(n³); for medium-to-large
graphs and small-to-medium diffusion times we instead expand `exp(-tL)`
in Chebyshev polynomials of a rescaled Laplacian (Hammond, Vandergheynst,
Gribonval 2011). Each polynomial term costs one matmul with L, so the
total cost drops to O(K · n³) dense (or O(K · |E|) for sparse — but
this implementation is dense first; sparse-Lanczos variants are
planned).

The chief advantage is that **K only needs to be modest** (tens, not
hundreds) for tight accuracy: the Chebyshev coefficients of `exp(-τy)`
on [-1, 1] are the modified Bessel functions I_k(τ), which decay
super-exponentially in k once k > τ. So for diffusion time t = O(1) on
the symmetric-normalized Laplacian (spectrum ⊂ [0, 2], so τ = t · 1),
K ≈ 20 gives ~1e-15 relative error.

Math (Hammond-Vandergheynst-Gribonval 2011, eq. 3.3):

  Given L with spectrum in [0, λ_max], rescale
    L̃ = (2 / λ_max) · L − I            (spectrum in [−1, 1])
    τ  = t · λ_max / 2
  Then
    exp(−t · L) ≈ Σ_{k=0}^{K} c_k(τ) · T_k(L̃)
    c_k(τ) = (2 − δ_{k,0}) · (−1)^k · exp(−τ) · I_k(τ)
    T_0(L̃) = I,   T_1(L̃) = L̃,   T_{k+1}(L̃) = 2 L̃ T_k(L̃) − T_{k−1}(L̃)

  We use `scipy.special.ive(k, τ) = exp(−τ) · I_k(τ)` for the
  coefficients, which is numerically stable for large τ.

For `L = laplacian.symmetric_normalized(A)` the spectrum is guaranteed
in `[0, 2]`, so `lambda_max = 2.0` is a safe default. For other
Laplacians (combinatorial, signed) pass `lambda_max` explicitly or
estimate via `torch.linalg.eigvalsh(L)[-1]` (still O(n³); cheaper power
iteration is planned).

References:
  Hammond, D. K., Vandergheynst, P., Gribonval, R. (2011). Wavelets on
    graphs via spectral graph theory. Applied and Computational
    Harmonic Analysis 30(2):129–150. §3 gives the Chebyshev expansion
    of an arbitrary spectral filter; §6 gives wall-clock benchmarks.
  Shuman, D. I., Vandergheynst, P., Frossard, P. (2011). Chebyshev
    polynomial approximation for distributed signal processing.
    DCOSS 2011. Distributed-implementation companion paper.
  Abramowitz, M., Stegun, I. A. (1972). Handbook of Mathematical
    Functions, §9.6 — modified Bessel functions and the Jacobi-Anger
    identity used to derive `c_k(τ) = (2 − δ_{k,0})(−1)^k I_k(τ)`.
"""

from __future__ import annotations

import numpy as np
import scipy.special
import torch

from holonomy_lib.provenance import with_provenance

# Default Chebyshev expansion order. Empirically (Hammond-Vandergheynst-
# Gribonval 2011 §6 Fig. 4, and our own audit), `K = 30` gives sub-1e-12
# relative error for diffusion times `t ≤ 5` on the symmetric-normalized
# Laplacian (τ ≤ 5). Larger t calls for K ≈ 2τ + 10 to stay tight; the
# function accepts `K` as a parameter for that case. **Scale of validity**:
# K should grow roughly linearly with `t · λ_max / 2`. Cataloged as
# `heat_kernel_chebyshev_k_default`.
CHEBYSHEV_ORDER_DEFAULT: int = 30

# Default upper bound on the Laplacian spectrum. The symmetric-normalized
# Laplacian L_sym has spectrum ⊂ [0, 2] (Chung 1997, Thm. 1.7) regardless
# of n, so 2.0 is the universal-invariant upper bound. For other
# Laplacians (combinatorial, signed) the caller MUST pass `lambda_max`
# matching their input. **Scale of validity**: this constant is the
# canonical L_sym bound; not a tuned parameter. Cataloged as
# `lambda_max_l_sym`.
LAMBDA_MAX_L_SYM: float = 2.0


@with_provenance(
    "holonomy_lib.spectral.heat_kernel_chebyshev", op_version="0.1",
)
def heat_kernel_chebyshev(
    L: torch.Tensor,
    t: float,
    signal: torch.Tensor | None = None,
    K: int = CHEBYSHEV_ORDER_DEFAULT,
    lambda_max: float = LAMBDA_MAX_L_SYM,
) -> torch.Tensor:
    """Heat kernel `exp(−t · L)` (or `exp(−t · L) @ signal`) via
    Chebyshev expansion.

    Args:
      L: (B, n, n) Laplacian. Must be symmetric (real eigenvalues).
        Use `spectral.laplacian.symmetric_normalized(A)` for the
        default `lambda_max = 2.0`; for other Laplacians pass
        `lambda_max` matching the input's spectral radius.
      t: diffusion time (non-negative). Larger t diffuses farther.
      signal: optional (B, n, k) signal to multiply. If given, returns
        `K_t @ signal` directly using the same Chebyshev recurrence
        applied to vectors. Big memory + time savings when k ≪ n.
        If None, returns the dense `K_t = exp(−t · L)` of shape (B, n, n).
      K: Chebyshev expansion order. Default 30. Increase to ~2 · t + 10
        for very large t.
      lambda_max: upper bound on the spectrum of L. Default 2.0 (the
        L_sym bound). For other Laplacians, pass the actual upper bound
        (an over-estimate is safe but reduces accuracy; an under-
        estimate violates the Chebyshev domain assumption).

    Returns:
      (B, n, n) if `signal is None`, else (B, n, k).

    References:
      Hammond-Vandergheynst-Gribonval (2011), eq. 3.3.
    """
    if t < 0:
        raise ValueError(f"diffusion time t must be >= 0, got {t}")
    if K < 0:
        raise ValueError(f"Chebyshev order K must be >= 0, got {K}")
    if lambda_max <= 0:
        raise ValueError(f"lambda_max must be > 0, got {lambda_max}")
    if L.ndim < 2 or L.shape[-1] != L.shape[-2]:
        raise ValueError(
            f"L must be (..., n, n); got L.shape={tuple(L.shape)}"
        )
    if signal is not None:
        # `signal` is (..., n, k_signal); the check compares
        # (..., n) against L's (..., n) — this covers both the leading
        # batch dims and the node count in a single equality test.
        if signal.shape[:-1] != L.shape[:-1]:
            raise ValueError(
                f"signal's batch + node dims {tuple(signal.shape[:-1])} must "
                f"match L's {tuple(L.shape[:-1])}"
            )

    n = L.shape[-1]
    # Rescale L to L̃ with spectrum in [-1, 1] for the Chebyshev domain.
    eye = torch.eye(n, device=L.device, dtype=L.dtype).expand_as(L)
    L_scaled = (2.0 / lambda_max) * L - eye
    tau = t * lambda_max / 2.0

    # Chebyshev coefficients: c_k(τ) = (2 − δ_{k,0}) · (−1)^k · ive(k, τ).
    # ive(k, τ) = exp(−τ) · I_k(τ) is numerically stable for any τ.
    # We compute the coeffs in float64 on CPU then move to L's device.
    # `torch.from_numpy().to(device=...)` is robust across CPU/CUDA/ROCm;
    # `torch.as_tensor(numpy, device='cuda')` historically raises on
    # some backends because numpy is CPU-bound.
    coeffs_np = scipy.special.ive(np.arange(K + 1), tau)
    coeffs_np = coeffs_np * np.where(
        np.arange(K + 1) == 0, 1.0, 2.0
    ) * np.where(np.arange(K + 1) % 2 == 0, 1.0, -1.0)
    coeffs = torch.from_numpy(coeffs_np.astype(
        np.float64 if L.dtype == torch.float64 else np.float32
    )).to(device=L.device, dtype=L.dtype)

    if signal is not None:
        return _chebyshev_apply_to_signal(L_scaled, signal, coeffs)
    return _chebyshev_dense(L_scaled, coeffs, eye)


# ============================================================
# Internal helpers
# ============================================================


def _chebyshev_dense(
    L_scaled: torch.Tensor,
    coeffs: torch.Tensor,
    eye: torch.Tensor,
) -> torch.Tensor:
    """Build K_t = Σ c_k · T_k(L_scaled) as a dense (B, n, n) tensor.

    Three-term recurrence: T_0 = I, T_1 = L̃, T_{k+1} = 2·L̃·T_k − T_{k−1}.
    """
    K = coeffs.shape[0] - 1
    if K < 0:
        # Degenerate: K = −1 makes no sense; K = 0 should fall through
        # to the regular path. We're guarded against K < 0 in the
        # public entry.
        raise RuntimeError("internal: K < 0 reached _chebyshev_dense")

    # k = 0 contribution
    result = coeffs[0] * eye
    if K == 0:
        return result

    # k = 1 contribution
    T_prev = eye
    T_curr = L_scaled
    result = result + coeffs[1] * T_curr
    for k in range(2, K + 1):
        T_next = 2.0 * (L_scaled @ T_curr) - T_prev
        result = result + coeffs[k] * T_next
        T_prev, T_curr = T_curr, T_next
    return result


def _chebyshev_apply_to_signal(
    L_scaled: torch.Tensor,
    signal: torch.Tensor,
    coeffs: torch.Tensor,
) -> torch.Tensor:
    """Compute K_t @ signal without building dense K_t.

    Same recurrence but on the signal vectors: T_k(L̃) · v computed
    one matmul at a time. Memory O(B · n · k) vs O(B · n²) for the
    dense form, and the per-iter matmul is O(B · n² · k) vs O(B · n³).
    """
    K = coeffs.shape[0] - 1

    v_prev = signal                                # T_0(L̃) · v = v
    result = coeffs[0] * v_prev
    if K == 0:
        return result

    v_curr = L_scaled @ signal                     # T_1(L̃) · v = L̃ v
    result = result + coeffs[1] * v_curr
    for k in range(2, K + 1):
        v_next = 2.0 * (L_scaled @ v_curr) - v_prev
        result = result + coeffs[k] * v_next
        v_prev, v_curr = v_curr, v_next
    return result
