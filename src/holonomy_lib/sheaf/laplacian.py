# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Synoros

"""Sheaf coboundary, sheaf Laplacian, and Dirichlet energy on graphs.

The coboundary operator `δ: C^0 → C^1` sends a node-stalk vector
`x ∈ R^{n_v · d_v}` to an edge-stalk vector whose value at each edge
`e = (u, v)` is `F_left[e] · x_u − F_right[e] · x_v`. Stacked across
all edges, `δ` is an `(n_e · d_e) × (n_v · d_v)` matrix.

The sheaf Laplacian is `L_F = δ^T δ`, which is symmetric PSD by
construction. Its kernel is the space of **global sections**: tuples
of node-stalk values that all incident edges agree on under the
restriction maps. For the trivial sheaf this kernel is the constant
vectors (one per connected component) — exactly the kernel of the
combinatorial Laplacian.

Implementation: builds `δ` densely in v1. For a graph with `n_e`
edges, node-stalk dim `d_v`, edge-stalk dim `d_e`, this is a `(n_e
· d_e) × (n_v · d_v)` matrix — fine up to thousands of nodes; a
sparse path is a natural extension for large graphs.

References:
  Hansen-Ghrist (2019), §3-4 — sheaf Laplacian + spectral theory.
  Bodnar et al. (2022), eq. 6 — the cellular-sheaf Laplacian as a
    block matrix.
"""

from __future__ import annotations

import torch

from holonomy_lib.provenance import with_provenance
from holonomy_lib.sheaf.graph_sheaf import GraphSheaf, HeterogeneousGraphSheaf


# Dense-coboundary allocation guard: refuse to allocate beyond this
# many bytes for the staging tensor + δ. 2 GiB picks a generous
# threshold for a research-grade primitive on commodity hardware; a
# sparse path is planned for larger graphs. **Scale of validity**:
# byte budget for the v1 dense implementation. Cataloged as
# `sheaf_dense_bytes_cap`.
SHEAF_DENSE_BYTES_CAP: int = 2 * 2**30


def _coboundary_coo(
    sheaf: GraphSheaf | HeterogeneousGraphSheaf,
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
    """COO triplets of the coboundary `δ`: `(indices (2, nnz), values (nnz,),
    shape (Σd_e, Σd_v))`.

    Vectorized ragged assembly, no Python loop over edges (one flattening
    pass over the ragged restriction-map tuples aside): edge `e = (u, v)`
    contributes the block `+F_left[e]` at rows `[eoff[e], eoff[e]+d_e)` ×
    cols `[voff[u], voff[u]+d_v[u])` and `−F_right[e]` at the same rows ×
    `v`'s columns. Row/col indices come from `repeat_interleave` + exclusive
    prefix-sum arithmetic over the per-edge block sizes. Handles both the
    uniform `GraphSheaf` (constant dims) and the heterogeneous sheaf with
    the same code path.

    The index sets of distinct blocks are disjoint: different edges occupy
    different row blocks, and within an edge `u != v` (self-loops rejected
    at construction) separates the two column blocks.
    """
    device, dtype = sheaf.device, sheaf.dtype
    hetero = isinstance(sheaf, HeterogeneousGraphSheaf)
    if hetero:
        dv = sheaf.node_stalk_dims.to(device=device, dtype=torch.long)
        de = sheaf.edge_stalk_dims.to(device=device, dtype=torch.long)
    else:
        dv = torch.full(
            (sheaf.n_nodes,), sheaf.node_stalk_dim,
            device=device, dtype=torch.long,
        )
        de = torch.full(
            (sheaf.n_edges,), sheaf.edge_stalk_dim,
            device=device, dtype=torch.long,
        )
    shape = (int(de.sum()), int(dv.sum()))

    n_e = sheaf.n_edges
    if n_e == 0:
        return (
            torch.zeros(2, 0, device=device, dtype=torch.long),
            torch.zeros(0, device=device, dtype=dtype),
            shape,
        )

    if hetero:
        left_vals = torch.cat([t.reshape(-1) for t in sheaf.F_left])
        right_vals = torch.cat([t.reshape(-1) for t in sheaf.F_right])
    else:
        left_vals = sheaf.F_left.reshape(-1)
        right_vals = sheaf.F_right.reshape(-1)

    edges = sheaf.edges.to(device=device, dtype=torch.long)
    eoff = torch.cumsum(de, dim=0) - de              # exclusive row offsets
    voff = torch.cumsum(dv, dim=0) - dv              # exclusive col offsets

    def _side(node_idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Block e is (d_e[e], d_v[node]) laid out row-major in the flattened
        # values; recover (row-in-block, col-in-block) from the flat position.
        w = dv[node_idx]                             # (n_e,) block widths
        m = de * w                                   # (n_e,) block numels
        starts = torch.cumsum(m, dim=0) - m          # (n_e,) exclusive
        eid = torch.repeat_interleave(
            torch.arange(n_e, device=device), m,
        )                                            # (nnz_side,)
        p = torch.arange(int(m.sum()), device=device) - starts[eid]
        w_el = w[eid]
        rows = eoff[eid] + p // w_el
        cols = voff[node_idx][eid] + p % w_el
        return rows, cols

    rows_l, cols_l = _side(edges[:, 0])
    rows_r, cols_r = _side(edges[:, 1])
    indices = torch.stack(
        (torch.cat((rows_l, rows_r)), torch.cat((cols_l, cols_r))),
    )
    values = torch.cat((left_vals, -right_vals))
    return indices, values, shape


def _heterogeneous_coboundary(sheaf: HeterogeneousGraphSheaf) -> torch.Tensor:
    """Dense coboundary `δ` for a heterogeneous (per-node-stalk) sheaf.

    Built from the vectorized COO triplets of `_coboundary_coo`: block
    positions are pairwise disjoint (no self-loops, no duplicate edges,
    enforced at construction), so a single advanced-indexing assignment
    places every entry exactly once — same semantics as a per-edge slice
    write, without the Python loop over edges.
    """
    device, dtype = sheaf.device, sheaf.dtype
    total_v = int(sheaf.node_stalk_dims.sum())
    total_e = int(sheaf.edge_stalk_dims.sum())

    bytes_per_elem = torch.empty((), dtype=dtype).element_size()
    dense_bytes = 2 * total_e * total_v * bytes_per_elem
    if dense_bytes > SHEAF_DENSE_BYTES_CAP:
        raise RuntimeError(
            f"sheaf_coboundary would allocate {dense_bytes:,} bytes for the dense "
            f"δ tensor (Σd_e={total_e}, Σd_v={total_v}); the v1 dense path is "
            f"capped at {SHEAF_DENSE_BYTES_CAP:,} bytes. Restrict to smaller "
            f"sheaves or contribute a sparse implementation."
        )

    delta = torch.zeros(total_e, total_v, device=device, dtype=dtype)
    if sheaf.n_edges == 0:
        return delta

    indices, values, _ = _coboundary_coo(sheaf)
    delta[indices[0], indices[1]] = values
    return delta


@with_provenance(
    "holonomy_lib.sheaf.sheaf_coboundary", op_version="0.1",
)
def sheaf_coboundary(sheaf: GraphSheaf | HeterogeneousGraphSheaf) -> torch.Tensor:
    """Build the sheaf coboundary `δ` as a dense `(n_e·d_e, n_v·d_v)`
    tensor.

    For edge index `e = (u, v)`, the corresponding `d_e`-row block of
    `δ` has `+F_left[e]` at columns `u·d_v : (u+1)·d_v` and
    `−F_right[e]` at columns `v·d_v : (v+1)·d_v`.

    Args:
      sheaf: a `GraphSheaf`.

    Returns:
      `(n_e · d_e, n_v · d_v)` coboundary on the sheaf's device/dtype.

    References:
      Hansen-Ghrist (2019), eq. 2.
    """
    if isinstance(sheaf, HeterogeneousGraphSheaf):
        return _heterogeneous_coboundary(sheaf)
    n_e = sheaf.n_edges
    n_v = sheaf.n_nodes
    d_e = sheaf.edge_stalk_dim
    d_v = sheaf.node_stalk_dim
    device, dtype = sheaf.device, sheaf.dtype

    # Dense δ size guard: the staging tensor `block` is `(n_e, d_e,
    # n_v · d_v)` and `delta` is `(n_e · d_e, n_v · d_v)`. For graphs
    # with thousands of nodes × edges × non-trivial stalk dims, the
    # raw dense allocation crosses many GB silently. Pre-flight check
    # against a configurable byte cap and tell the user explicitly
    # that they're past the dense regime.
    bytes_per_elem = torch.empty((), dtype=dtype).element_size()
    dense_bytes = 2 * n_e * d_e * n_v * d_v * bytes_per_elem
    if dense_bytes > SHEAF_DENSE_BYTES_CAP:
        raise RuntimeError(
            f"sheaf_coboundary would allocate {dense_bytes:,} bytes "
            f"for the dense δ tensor (n_e={n_e}, d_e={d_e}, n_v={n_v}, "
            f"d_v={d_v}); the v1 dense path is capped at "
            f"{SHEAF_DENSE_BYTES_CAP:,} bytes. A sparse path is on the "
            f"v0.2 roadmap; for now, restrict to smaller sheaves or "
            f"contribute a sparse implementation."
        )

    delta = torch.zeros(n_e * d_e, n_v * d_v, device=device, dtype=dtype)

    if n_e == 0:
        return delta

    # Vectorized assembly: build a (n_e, d_e, n_v * d_v) staging tensor
    # then reshape to (n_e * d_e, n_v * d_v). Allocating one big
    # intermediate beats a Python loop over edges for any non-trivial n_e.
    block = torch.zeros(n_e, d_e, n_v * d_v, device=device, dtype=dtype)
    edge_idx = torch.arange(n_e, device=device)
    u_idx = sheaf.edges[:, 0]                                     # (n_e,)
    v_idx = sheaf.edges[:, 1]                                     # (n_e,)
    # For each edge e, place F_left[e] into columns [u·d_v : (u+1)·d_v].
    # Use advanced indexing with an explicit column-range tensor.
    col_off = torch.arange(d_v, device=device)
    u_cols = u_idx.unsqueeze(dim=-1) * d_v + col_off              # (n_e, d_v)
    v_cols = v_idx.unsqueeze(dim=-1) * d_v + col_off              # (n_e, d_v)
    # Broadcast assignment: block[e, :, u_cols[e, k]] = +F_left[e, :, k].
    # scatter is cleanest with the (n_e, d_e, d_v) source.
    block.scatter_(
        dim=-1,
        index=u_cols.unsqueeze(dim=-2).expand(n_e, d_e, d_v),
        src=sheaf.F_left,
    )
    # `scatter_` overwrites; for v_cols we want to subtract F_right.
    # Pull current values at v_cols (which are 0, since we haven't
    # touched those entries — but defensively use `scatter_add` with
    # -F_right against zeros to compose with any prior content).
    block.scatter_add_(
        dim=-1,
        index=v_cols.unsqueeze(dim=-2).expand(n_e, d_e, d_v),
        src=-sheaf.F_right,
    )

    return block.reshape(n_e * d_e, n_v * d_v)


@with_provenance(
    "holonomy_lib.sheaf.sheaf_laplacian", op_version="0.1",
)
def sheaf_laplacian(sheaf: GraphSheaf | HeterogeneousGraphSheaf) -> torch.Tensor:
    """Sheaf Laplacian `L_F = δ^T δ`, a symmetric PSD operator of
    shape `(n_v · d_v, n_v · d_v)`.

    For the *trivial* sheaf (every restriction map is identity,
    uniform stalk dim `d`):
      d = 1   → L_F equals the combinatorial graph Laplacian.
      d > 1   → L_F is the block-diagonal `d × L` (each channel is a
                copy of the scalar Laplacian, no cross-channel mixing).

    Args:
      sheaf: a `GraphSheaf`.

    Returns:
      `(n_v · d_v, n_v · d_v)` symmetric PSD Laplacian.

    References:
      Hansen-Ghrist (2019), §3.1.
      Bodnar et al. (2022), eq. 6.
    """
    delta = sheaf_coboundary(sheaf)
    # `δ^T @ δ` is the canonical form; matmul of (n_v·d_v, n_e·d_e) by
    # its transpose. Symmetrize against any drift introduced by the
    # blocked BLAS path (rounding can leave |L − L^T|_∞ ~ 1e-15 even
    # though mathematically `δ^T δ` is exactly symmetric — symmetrize
    # so callers can chain straight into `torch.linalg.eigh`).
    L = delta.mT @ delta
    return 0.5 * (L + L.mT)


@with_provenance(
    "holonomy_lib.sheaf.sheaf_dirichlet_energy", op_version="0.1",
)
def sheaf_dirichlet_energy(
    sheaf: GraphSheaf | HeterogeneousGraphSheaf, x: torch.Tensor,
) -> torch.Tensor:
    """Per-batch Dirichlet energy `E(x) = x^T L_F x` for sheaf-valued
    signals.

    The Dirichlet energy is the squared coboundary norm — it measures
    how much each edge's two restriction-mapped values disagree.
    Minimized at the global sections (kernel of `L_F`).

    Args:
      sheaf: a `GraphSheaf`.
      x: `(B, n_v · d_v)` or `(n_v · d_v,)` node-stalk signals.

    Returns:
      Scalar `()` for unbatched input or `(B,)` for batched input.

    References:
      Hansen-Ghrist (2019), §3.2.
    """
    total = sheaf.total_node_dim
    if x.shape[-1] != total:
        raise ValueError(
            f"x last dim must equal the total node-stalk dim (sum of d_v over "
            f"n_nodes) = {total}; got x.shape={tuple(x.shape)}"
        )
    delta = sheaf_coboundary(sheaf)                  # (n_e·d_e, n_v·d_v)
    # x: (..., n_v·d_v); δ x: (..., n_e·d_e); energy: (..., )
    delta_x = torch.einsum("ij,...j->...i", delta, x)
    return (delta_x * delta_x).sum(dim=-1)
