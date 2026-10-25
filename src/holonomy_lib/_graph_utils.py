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
  is defined on simple graphs in the source literature. Implicitly
  including self-loops changes `d_i = Σ_j A[i, j]` and propagates into
  every downstream operation (normalizers, walk distributions, phase
  factors). Letting the caller decide silently produced
  inconsistent semantics across primitives — different ops gave
  different answers on the same input depending on whether they
  happened to cancel the diagonal.

  The combinatorial Laplacian `L = D − A` happens to be invariant to
  self-loops (the diagonal subtraction zeroes them), but we still drop
  them at entry so callers see consistent behavior across the library.
"""

from __future__ import annotations

import torch


def drop_self_loops(A: torch.Tensor) -> torch.Tensor:
    """Return a view of `A` with the diagonal zeroed.

    Args:
      A: (..., n, n) adjacency-like tensor.

    Returns:
      Tensor with the same shape as `A`, identical to `A` everywhere
      except `result[..., i, i] = 0` for all i.

    Notes:
      Uses `torch.where` so the operation stays vectorized and is
      compatible with `torch.vmap` / `torch.compile`. The eye tensor
      is built with `A`'s dtype to avoid implicit dtype promotion.
    """
    n = A.shape[-1]
    eye = torch.eye(n, device=A.device, dtype=A.dtype).expand_as(A)
    return torch.where(eye > 0, torch.zeros_like(A), A)
