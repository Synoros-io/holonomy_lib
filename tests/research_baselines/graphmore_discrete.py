# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Synoros

"""GraphMoRE-style discrete-gating per-node curvature baseline.

A minimal implementation of the "pick one κ from a discrete set via
learned gating per node" pattern, used as the comparison baseline for
`HeterogeneousKappaManifold`'s continuous per-point κ in the C5 + C6
strengthening study.

This is **NOT a faithful reimplementation of GraphMoRE** (Guo et al.,
AAAI 2025, arXiv:2412.11085); that paper has architectural choices
around mixture-of-experts attention, Riemannian routing, etc., that
are out of scope here. What we extract is the **interpretable
mechanism** we want to compare against: a discrete bank of K
curvature-experts, and a per-node soft gating that mixes their
distances.

The discrete-gating mechanism we implement:

  - Bank of K curvatures `κ_1, …, κ_K` (e.g. `{-1.0, 0.0, +0.5}`)
  - Per-node gating logits `g_i ∈ R^K` (learnable)
  - Soft assignment via softmax: `w_i = softmax(g_i)` ∈ Δ^{K-1}
  - For a pair (i, j), pair gating is `w_pair = (w_i + w_j) / 2`
    (the natural per-pair generalization of per-node soft gating)
  - Pair distance is the gating-weighted sum of constant-κ distances:
        d(x_i, x_j) = Σ_k w_pair[k] · d_{κ_k}(x_i, x_j)
    using each expert's `KappaStereographicManifold.distance`.

Why this mechanism is the right comparison:

  - **Mass per κ is a probability**: a node "is in expert k" with
    weight `w_i[k]`. Compared to our continuous κ, this is a
    coarse-grained representation; the model can only assign a
    probability over K fixed κ's, not a real number.
  - **Hard gating** (`argmax` instead of softmax) makes the
    assignment fully discrete; the K-many expert distance is just
    `d_{κ_k}(x_i, x_j)` where k is the chosen expert. We support
    both modes; soft is differentiable, hard is the literal
    "expert assignment" semantics.

References:
  Guo, S., et al. (2024). GraphMoRE: Mitigating the Topology Heterogeneity
    Problem with Mixture-of-Riemannian-Experts. arXiv:2412.11085, AAAI-2025.
  (We use the "discrete gating over fixed-κ experts" idea, not the
  full architecture.)
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from holonomy_lib.manifolds.stereographic import KappaStereographicManifold


class DiscreteGatingKappa(nn.Module):
    """Per-node discrete-gating curvature, GraphMoRE-style.

    Args:
      n_nodes: number of nodes (each gets its own gating vector).
      kappa_bank: fixed κ values of the experts. Each is a float that
        defines a `KappaStereographicManifold` at that κ.
      dim: embedding dimension (every expert shares this).
      mode: `"soft"` (softmax gating, differentiable) or `"hard"`
        (argmax + straight-through). Soft is the default for
        gradient-based training; hard is the literal
        "node-is-in-expert-k" semantics.
      device, dtype: as usual.

    The module owns:
      - `self.gates`: `nn.Parameter` of shape `(n_nodes, K)`. Per-node
        gating logits.
      - `self.experts`: list of `KappaStereographicManifold` at each
        bank curvature. Constructed at init; the bank κ's are fixed
        (not learned).
    """

    def __init__(
        self,
        n_nodes: int,
        kappa_bank: Sequence[float],
        dim: int,
        mode: str = "soft",
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float64,
    ):
        super().__init__()
        if mode not in ("soft", "hard"):
            raise ValueError(f"mode must be 'soft' or 'hard', got {mode!r}")
        self.n_nodes = n_nodes
        self.kappa_bank = tuple(float(k) for k in kappa_bank)
        self.K = len(self.kappa_bank)
        self.dim = dim
        self.mode = mode
        # Initialize gates with small noise — softmax then ≈ uniform.
        self.gates = nn.Parameter(
            0.01 * torch.randn(n_nodes, self.K, device=device, dtype=dtype)
        )
        # One manifold per bank κ. KappaStereographicManifold's static-
        # float-κ fast path is fine: each expert has a fixed κ.
        self.experts = [
            KappaStereographicManifold(
                n=dim, kappa=k, device=device, dtype=dtype,
            )
            for k in self.kappa_bank
        ]

    def gate_weights(self, indices: torch.Tensor) -> torch.Tensor:
        """Per-node soft (or hard) gate weights, shape `(B, K)`."""
        logits = self.gates[indices]
        if self.mode == "soft":
            return torch.softmax(logits, dim=-1)
        # hard: argmax with straight-through estimator for gradient
        soft = torch.softmax(logits, dim=-1)
        hard = torch.zeros_like(soft)
        hard.scatter_(-1, soft.argmax(dim=-1, keepdim=True), 1.0)
        # straight-through: forward = hard, backward = soft
        return hard + (soft - soft.detach())

    def pairwise_distance(
        self,
        x_i: torch.Tensor, idx_i: torch.Tensor,
        x_j: torch.Tensor, idx_j: torch.Tensor,
    ) -> torch.Tensor:
        """Distance between pairs `(x_i, x_j)` with per-node gating.

        Pair gating is the average of the two endpoint gates. Distance
        is the gate-weighted sum of constant-κ expert distances.

        Args:
          x_i, x_j: `(B, dim)` coordinates.
          idx_i, idx_j: `(B,)` node indices (for fetching gates).
        Returns:
          `(B,)` distances.
        """
        w_i = self.gate_weights(idx_i)       # (B, K)
        w_j = self.gate_weights(idx_j)
        w_pair = 0.5 * (w_i + w_j)           # (B, K)
        # Compute expert-distance per k:
        d_expert = torch.stack(
            [exp.distance(x_i, x_j) for exp in self.experts], dim=-1,
        )                                     # (B, K)
        return (w_pair * d_expert).sum(dim=-1)

    def recovered_kappa(self) -> torch.Tensor:
        """Per-node expectation `E_k[κ_k] = Σ_k w_i[k] · κ_k`; the
        "soft" recovered curvature for each node. Used to compare
        recovery against the true per-node κ in the ablation."""
        soft = torch.softmax(self.gates, dim=-1)             # (N, K)
        bank = torch.tensor(
            self.kappa_bank, device=self.gates.device, dtype=self.gates.dtype,
        )                                                     # (K,)
        return (soft * bank).sum(dim=-1)                     # (N,)

    def hard_assignment(self) -> torch.Tensor:
        """Per-node argmax expert index; the "this node lives in
        expert k" interpretation. Shape `(N,)`."""
        return self.gates.argmax(dim=-1)
