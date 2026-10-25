# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Synoros

"""Internal graph-primitive utilities shared across modules.

Conventions enforced here (see `CONVENTIONS.md`):

  Self-loop policy
  ----------------
  Graph primitives in this library treat the input adjacency `A` as a
  simple graph: the diagonal `A[i, i]` is **dropped at entry**, not
  treated as an edge from node `i` to itself.

  Rationale: most of the math (Forman-Ricci, Ollivier-Ricci, the
  random-walk and symmetric-normalized Laplacians, magnetic Laplacian)
  is defined on simple graphs in the source literature. Including a
  self-loop changes `d_i = Σ_j A[i, j]` and every quantity built from
  it (normalizers, walk distributions, phase factors). If each
  primitive left the diagonal to the caller, the same input could give
  different answers depending on whether that primitive happened to
  cancel it, so the policy is applied once, at entry.

  The combinatorial Laplacian `L = D − A` happens to be invariant to
  self-loops (the diagonal subtraction zeroes them), but we still drop
  them at entry so callers see consistent behavior across the library.
"""

from __future__ import annotations

import torch


def drop_self_loops(A: torch.Tensor) -> torch.Tensor:
    """Return a tensor with the diagonal of `A` zeroed.

    Args:
      A: (..., n, n) adjacency-like tensor.

    Returns:
      Tensor with the same shape as `A`, identical to `A` everywhere
      except `result[..., i, i] = 0` for all i.

    Notes:
      Uses `torch.where`, so the zeroing is one vectorized op over the
      batch. The boolean eye mask leaves the value dtype of `A` unchanged.

    References:
      Chung (1997). Spectral Graph Theory. AMS, Chapter 1, simple graphs.
    """
    n = A.shape[-1]
    eye = torch.eye(n, device=A.device, dtype=torch.bool).expand_as(A)
    return torch.where(eye, torch.zeros_like(A), A)
