# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Synoros

"""Sheaf coboundary, sheaf Laplacian, and Dirichlet energy on graphs.

The coboundary operator `δ: C^0 → C^1` sends a node-stalk vector
`x ∈ R^{n_v · d_v}` to an edge-stalk vector whose value at each edge
`e = (u, v)` is `F_left[e] · x_u − F_right[e] · x_v`. Stacked across
all edges, `δ` is an `(n_e · d_e) × (n_v · d_v)` matrix.

The sheaf Laplacian is `L_F = δ^T δ`, which is symmetric PSD by
construction. Its kernel is the space of **global sections**: tuples
of node-stalk values that all incident edges agree on under the
restriction maps. For the trivial sheaf this kernel is the constant
vectors in each stalk channel on each connected component, as for
the scalar graph Laplacian.

Dense and sparse COO assembly are available through `sheaf_coboundary`
and `sheaf_coboundary_sparse`; the Laplacian has matching dense and sparse
entry points. `sheaf_bottom_spectrum` solves for its lowest eigenmodes.

References:
  Hansen-Ghrist (2019), §3-4; sheaf Laplacian + spectral theory.
  Bodnar et al. (2022), eq. 6; the cellular-sheaf Laplacian as a
    block matrix.
"""

from __future__ import annotations

from typing import Literal

import torch

from holonomy_lib.sheaf.graph_sheaf import GraphSheaf, HeterogeneousGraphSheaf

# Dense-coboundary allocation guard: refuse to allocate beyond this
# many bytes for the staging tensor + δ. 2 GiB is a chosen threshold for
# machines with 24 to 32 GB of host RAM; the sparse path
# (`sheaf_coboundary_sparse`) covers larger graphs.
# **Scale of validity**: byte budget for the dense implementation.
# Cataloged as `SHEAF_DENSE_BYTES_CAP`.
SHEAF_DENSE_BYTES_CAP: int = 2 * 2**30

# Extra bottom modes requested beyond `k` in `sheaf_bottom_spectrum` so
# the kernel (global sections, λ≈0) can be dropped and still leave `k`
# nonzero modes. The sheaf kernel of a (near-)connected corpus is small;
# 16 covers it, and the solve raises with instructions when a sheaf has
# more global sections than the headroom. Value from synoros-substrate
# `sheaf_scale.py` (validated 2026-06-06). **Scale of validity**:
# corpora whose global-section space is ≤ 16-dimensional; raise the
# parameter for heavily disconnected sheaves. Catalog:
# `SHEAF_KERNEL_HEADROOM` in `notes/magic_numbers.md`.
SHEAF_KERNEL_HEADROOM: int = 16

# LOBPCG searches over three blocks (Knyazev 2001): the current Ritz block,
# the residuals, and the previous search directions. The 1 is the Ritz
# block and the 2 is the residual and previous-direction pair. torch.lobpcg
# needs room for all three.
_LOBPCG_SUBSPACE_BLOCKS: int = 1 + 2


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
    if isinstance(sheaf, HeterogeneousGraphSheaf):
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
    if isinstance(sheaf, HeterogeneousGraphSheaf):
        shape = (sum(t.shape[0] for t in sheaf.F_left), sheaf.total_node_dim)
    else:
        shape = (sheaf.n_edges * sheaf.edge_stalk_dim, sheaf.total_node_dim)

    n_e = sheaf.n_edges
    if n_e == 0:
        return (
            torch.zeros(2, 0, device=device, dtype=torch.long),
            torch.zeros(0, device=device, dtype=dtype),
            shape,
        )

    if isinstance(sheaf, HeterogeneousGraphSheaf):
        left_vals = torch.cat([t.reshape(-1) for t in sheaf.F_left])
        right_vals = torch.cat([t.reshape(-1) for t in sheaf.F_right])
    else:
        left_vals = sheaf.F_left.reshape(-1)
        right_vals = sheaf.F_right.reshape(-1)

    edges = sheaf.edges.to(device=device, dtype=torch.long)
    eoff = torch.cumsum(de, dim=0) - de              # exclusive row offsets
    voff = torch.cumsum(dv, dim=0) - dv              # exclusive col offsets

    def _side(node_idx: torch.Tensor, size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Row and column COO indices for one side (left or right) of all edges.

        Input: `node_idx` is the node on this side of each edge, `size` is the
        total number of values on this side. Output: `(rows, cols)`, one entry
        per value. Assumes the values are laid out edge by edge, row-major.
        """
        # Block e is (d_e[e], d_v[node]) laid out row-major in the flattened
        # values; recover (row-in-block, col-in-block) from the flat position.
        w = dv[node_idx]                             # (n_e,) block widths
        m = de * w                                   # (n_e,) block numels
        starts = torch.cumsum(m, dim=0) - m          # (n_e,) exclusive
        eid = torch.repeat_interleave(
            torch.arange(n_e, device=device), m,
        )                                            # (nnz_side,)
        # p is the position inside edge eid's block. Blocks are row-major with
        # width w, so p // w is the row in the block and p % w is the column.
        p = torch.arange(size, device=device) - starts[eid]
        w_el = w[eid]
        rows = eoff[eid] + p // w_el
        cols = voff[node_idx][eid] + p % w_el
        return rows, cols

    rows_l, cols_l = _side(edges[:, 0], left_vals.numel())
    rows_r, cols_r = _side(edges[:, 1], right_vals.numel())
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
    places every entry exactly once; same semantics as a per-edge slice
    write, without the Python loop over edges.
    """
    device, dtype = sheaf.device, sheaf.dtype
    total_v = sheaf.total_node_dim
    total_e = sum(t.shape[0] for t in sheaf.F_left)

    bytes_per_elem = torch.empty((), dtype=dtype).element_size()
    dense_bytes = 2 * total_e * total_v * bytes_per_elem
    if dense_bytes > SHEAF_DENSE_BYTES_CAP:
        raise RuntimeError(
            f"sheaf_coboundary would allocate {dense_bytes:,} bytes for the dense "
            f"δ tensor (Σd_e={total_e}, Σd_v={total_v}); the v1 dense path is "
            f"capped at {SHEAF_DENSE_BYTES_CAP:,} bytes. Restrict to smaller "
            f"sheaves or use sheaf_coboundary_sparse."
        )

    delta = torch.zeros(total_e, total_v, device=device, dtype=dtype)
    if sheaf.n_edges == 0:
        return delta

    indices, values, _ = _coboundary_coo(sheaf)
    delta[indices[0], indices[1]] = values
    return delta


def sheaf_coboundary(sheaf: GraphSheaf | HeterogeneousGraphSheaf) -> torch.Tensor:
    """Build the sheaf coboundary `δ` as a dense `(n_e·d_e, n_v·d_v)`
    tensor.

    For edge index `e = (u, v)`, the corresponding `d_e`-row block of
    `δ` has `+F_left[e]` at columns `u·d_v : (u+1)·d_v` and
    `−F_right[e]` at columns `v·d_v : (v+1)·d_v`.

    Args:
      sheaf: a `GraphSheaf` or `HeterogeneousGraphSheaf`.

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

    # Dense δ size guard. The staging tensor `block` is `(n_e, d_e, n_v · d_v)`
    # and `delta` is `(n_e · d_e, n_v · d_v)`, the same size, so the factor 2
    # counts both. Graphs with many nodes and edges, or large stalk dims, reach
    # many GB without warning, so the size is checked against
    # SHEAF_DENSE_BYTES_CAP before anything is allocated.
    bytes_per_elem = torch.empty((), dtype=dtype).element_size()
    dense_bytes = 2 * n_e * d_e * n_v * d_v * bytes_per_elem
    if dense_bytes > SHEAF_DENSE_BYTES_CAP:
        raise RuntimeError(
            f"sheaf_coboundary would allocate {dense_bytes:,} bytes "
            f"for the dense δ tensor (n_e={n_e}, d_e={d_e}, n_v={n_v}, "
            f"d_v={d_v}); the v1 dense path is capped at "
            f"{SHEAF_DENSE_BYTES_CAP:,} bytes. Use "
            f"sheaf_coboundary_sparse for larger sheaves."
        )

    if n_e == 0:
        return torch.zeros(0, n_v * d_v, device=device, dtype=dtype)

    # Vectorized assembly: build a (n_e, d_e, n_v * d_v) staging tensor
    # then reshape to (n_e * d_e, n_v * d_v). Allocating one big
    # intermediate beats a Python loop over edges for any non-trivial n_e.
    block = torch.zeros(n_e, d_e, n_v * d_v, device=device, dtype=dtype)
    u_idx = sheaf.edges[:, 0]                                     # (n_e,)
    v_idx = sheaf.edges[:, 1]                                     # (n_e,)
    # For each edge e, F_left[e] goes into the columns of node u, which are
    # [u·d_v : (u+1)·d_v]. u_cols and v_cols hold the column index of each
    # node-stalk coordinate for every edge.
    col_off = torch.arange(d_v, device=device)
    u_cols = u_idx.unsqueeze(dim=-1) * d_v + col_off              # (n_e, d_v)
    v_cols = v_idx.unsqueeze(dim=-1) * d_v + col_off              # (n_e, d_v)
    # scatter_ writes block[e, :, u_cols[e, k]] = F_left[e, :, k]. The index is
    # expanded over the d_e rows so it matches the (n_e, d_e, d_v) source.
    block.scatter_(
        dim=-1,
        index=u_cols.unsqueeze(dim=-2).expand(n_e, d_e, d_v),
        src=sheaf.F_left,
    )
    # scatter_ only wrote the u-columns. The v-columns are still zero for this
    # edge because self-loops are rejected (u != v), so adding -F_right here
    # is the same as writing it. The add also gives F_left - F_right if u == v,
    # where both blocks land in the same columns.
    block.scatter_add_(
        dim=-1,
        index=v_cols.unsqueeze(dim=-2).expand(n_e, d_e, d_v),
        src=-sheaf.F_right,
    )

    return block.reshape(n_e * d_e, n_v * d_v)


def sheaf_laplacian(sheaf: GraphSheaf | HeterogeneousGraphSheaf) -> torch.Tensor:
    """Sheaf Laplacian `L_F = δ^T δ`, a symmetric PSD operator of
    shape `(n_v · d_v, n_v · d_v)`.

    For the *trivial* sheaf (every restriction map is identity,
    uniform stalk dim `d`):
      d = 1   → L_F equals the combinatorial graph Laplacian.
      d > 1   → L_F = L ⊗ I_d in node-major order, permutation-similar
                to a block diagonal matrix of d scalar Laplacians.

    Args:
      sheaf: a `GraphSheaf` or `HeterogeneousGraphSheaf`.

    Returns:
      `(n_v · d_v, n_v · d_v)` symmetric PSD Laplacian.

    References:
      Hansen-Ghrist (2019), §3.1.
      Bodnar et al. (2022), eq. 6.
    """
    delta = sheaf_coboundary(sheaf)
    # δ^T δ is the Gram matrix of the columns of δ, so it is
    # (n_v·d_v, n_v·d_v). Rounding in the matmul can leave a tiny asymmetry
    # even though the exact product is symmetric. Symmetrizing lets callers
    # pass the result straight to `torch.linalg.eigh`.
    L = delta.mT @ delta
    return 0.5 * (L + L.mT)


def sheaf_coboundary_sparse(
    sheaf: GraphSheaf | HeterogeneousGraphSheaf,
) -> torch.Tensor:
    """Sheaf coboundary `δ` as a coalesced sparse-COO `(Σd_e, Σd_v)` tensor.

    Same operator as `sheaf_coboundary`, without materializing the dense
    matrix: `δ` has at most `Σ_e d_e[e]·(d_v[u_e] + d_v[v_e])` nonzeros
    (two blocks per edge), so its sparse form scales with the edge count
    rather than with `Σd_e · Σd_v`. Use for large sheaves past the dense
    byte cap; convert with `.to_sparse_csr()` for repeated matvecs.

    Args:
      sheaf: a `GraphSheaf` or `HeterogeneousGraphSheaf`.

    Returns:
      Coalesced sparse-COO `(Σ d_e, Σ d_v)` coboundary on the sheaf's
      device/dtype.

    References:
      Hansen-Ghrist (2019), eq. 2.
    """
    indices, values, shape = _coboundary_coo(sheaf)
    return torch.sparse_coo_tensor(
        indices, values, shape,
        device=sheaf.device, dtype=sheaf.dtype,
    ).coalesce()


def sheaf_laplacian_sparse(
    sheaf: GraphSheaf | HeterogeneousGraphSheaf,
) -> torch.Tensor:
    """Sheaf Laplacian `L_F = δ^T δ` as a coalesced sparse-COO tensor.

    Same operator as `sheaf_laplacian`, assembled sparsely: `L_F`'s
    block-sparsity pattern is the graph adjacency's (one `(d_v[u],
    d_v[v])` off-diagonal block per edge plus the node-diagonal blocks),
    so nnz scales with edges, not with `(Σd_v)²`. Symmetrized as in the
    dense path so callers can chain into symmetric eigensolvers.

    Args:
      sheaf: a `GraphSheaf` or `HeterogeneousGraphSheaf`.

    Returns:
      Coalesced sparse-COO `(Σ d_v, Σ d_v)` symmetric PSD Laplacian.

    References:
      Hansen-Ghrist (2019), §3.1.
      Bodnar et al. (2022), eq. 6.
    """
    delta = sheaf_coboundary_sparse(sheaf)
    L = torch.sparse.mm(delta.t(), delta)
    # Same symmetrization rationale as the dense `sheaf_laplacian`.
    return (0.5 * (L + L.t())).coalesce()


def _node_offsets(sheaf: GraphSheaf | HeterogeneousGraphSheaf) -> torch.Tensor:
    """`(n_nodes + 1,)` int64 exclusive prefix sums of the node-stalk dims:
    node `i`'s block of the stacked node-stalk vector is
    `[offsets[i], offsets[i+1])`."""
    if isinstance(sheaf, HeterogeneousGraphSheaf):
        dims = sheaf.node_stalk_dims.to(device=sheaf.device, dtype=torch.long)
        return torch.cat(
            (torch.zeros(1, dtype=torch.long, device=dims.device),
             torch.cumsum(dims, dim=0)),
        )
    return torch.arange(
        sheaf.n_nodes + 1, dtype=torch.long, device=sheaf.device,
    ) * sheaf.node_stalk_dim


def sheaf_bottom_spectrum(
    sheaf: GraphSheaf | HeterogeneousGraphSheaf,
    k: int,
    drop_kernel: bool = True,
    kernel_headroom: int = SHEAF_KERNEL_HEADROOM,
    solver: Literal["auto", "eigsh", "lobpcg", "torch_lobpcg"] = "auto",
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Bottom-k spectral form of the sheaf Laplacian via a sparse solve.

    Returns the `k` smallest **nonzero** eigenpairs of `L_F = δ^T δ`
    (with `drop_kernel=True`, the default): the kernel of `L_F` is the
    space of global sections. For a trivial sheaf these are constant
    in each stalk channel on each connected component. Dropping them
    starts the returned spectrum at the first resolved positive mode. A mode
    is positive only above both its residual bound `||Lv - λv|| / ||v||`
    and a matrix-scale tolerance. That tolerance is at least
    `n · eps(dtype) · scale`, and at least `LOBPCG_TOL · scale` for
    LOBPCG, where `scale` is the full spectral norm on the dense path
    and its upper bound `max_i Σ_j |L[i,j]|` on sparse paths.

    The eigensolve runs on the SPARSE Laplacian via
    `algebra.lanczos_eigsh(which="SA")`; see `algebra.eigsh_sparse` for
    the backend dispatch and its scale-of-validity notes. Full-spectrum
    requests and requests too wide for the LOBPCG search subspace use
    dense `torch.linalg.eigh`. This spectral entry point is deliberately
    nondifferentiable: maps requiring gradients are rejected on every
    backend. Use `sheaf_laplacian` and a dense torch eigensolver for autograd.

    Args:
      sheaf: a `GraphSheaf` or `HeterogeneousGraphSheaf`.
      k: number of nonzero eigenpairs to return.
      drop_kernel: drop λ≈0 modes (default). With `False`, returns the
        raw bottom-k modes including global sections.
      kernel_headroom: extra modes requested beyond `k` so the kernel
        can be dropped and still leave `k` nonzero modes (cataloged
        default `SHEAF_KERNEL_HEADROOM`). Raise it for sheaves with
        many global sections (e.g. many connected components).
      solver: sparse eigensolver backend, forwarded to `lanczos_eigsh`.
      generator: torch generator for the LOBPCG starting block.

    Returns:
      `(eigvals (k,), eigvecs (Σd_v, k), node_offsets (n_nodes + 1,))`.
      Eigenvalues ascend. Node `i`'s rows of the eigenvector block are
      `eigvecs[node_offsets[i]:node_offsets[i+1]]`.

    References:
      Hansen-Ghrist (2019), §3.1 (kernel = global sections), §4
        (harmonic/spectral reads of the sheaf Laplacian).
      Ericsson-Ruhe (1980); Knyazev (2001), for the sparse bottom-k
        backends, via `algebra.eigsh_sparse`.
    """
    from holonomy_lib.algebra.eigsh_sparse import LOBPCG_TOL, SPARSE_SA_ITER_SOLVER_N
    from holonomy_lib.algebra.lanczos import lanczos_eigsh

    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError(f"k must be >= 1; got k={k}")
    if (not isinstance(kernel_headroom, int)
            or isinstance(kernel_headroom, bool) or kernel_headroom < 0):
        raise ValueError(
            f"kernel_headroom must be >= 0; got {kernel_headroom}"
        )

    if solver not in ("auto", "eigsh", "lobpcg", "torch_lobpcg"):
        raise ValueError(f"unknown sparse eigensolver {solver!r}")
    n = sheaf.total_node_dim
    if k > n:
        raise ValueError(f"k must not exceed total node dim {n}; got k={k}")
    L = sheaf_laplacian_sparse(sheaf)
    if L.requires_grad:
        raise ValueError(
            "sheaf_bottom_spectrum is nondifferentiable; detach restriction "
            "maps or use torch.linalg.eigh(sheaf_laplacian(sheaf))"
        )
    k_req = min(k + kernel_headroom, n) if drop_kernel else k
    # Machine-epsilon floor for the matrix-scale tolerance, scaled by n. A
    # LOBPCG run raises it below.
    relative_tol = n * torch.finfo(sheaf.dtype).eps
    # A zero Laplacian has no positive modes, so the solver is skipped: it
    # would have to factor a singular matrix. The identity columns only give
    # the residual check something of the right shape. Every mode is zero,
    # so the cutoff rejects all of them.
    if L._nnz() == 0 or not torch.any(L.values()):
        vals = L.values().new_zeros(k_req)
        kernel_scale = vals.new_zeros(())
        vecs = torch.eye(n, k_req, dtype=sheaf.dtype, device=sheaf.device)
    elif k_req == n or (
        solver != "eigsh" and n < _LOBPCG_SUBSPACE_BLOCKS * k_req
    ):
        vals, vecs = torch.linalg.eigh(L.to_dense())
        kernel_scale = vals.abs().max()
        vals, vecs = vals[:k_req], vecs[:, :k_req]
    else:
        vals, vecs = lanczos_eigsh(
            L, k_req, which="SA", solver=solver, generator=generator,
        )
        row_magnitudes = L.values().new_zeros(n).scatter_add_(
            0, L.indices()[0], L.values().abs(),
        )
        kernel_scale = row_magnitudes.max()
        # This repeats the "auto" choice in eigsh_sparse: off the CPU, or past
        # SPARSE_SA_ITER_SOLVER_N nodes, the solver is LOBPCG and the tolerance
        # gets the LOBPCG floor.
        uses_lobpcg = solver in ("lobpcg", "torch_lobpcg") or (
            solver == "auto" and (
                L.device.type != "cpu" or n > SPARSE_SA_ITER_SOLVER_N
            )
        )
        if uses_lobpcg:
            relative_tol = max(relative_tol, LOBPCG_TOL)

    if not drop_kernel:
        return vals[:k], vecs[:, :k], _node_offsets(sheaf)

    # On the sparse path the computed values are only the bottom modes, so
    # their maximum says nothing about the top of the spectrum. For symmetric
    # L the largest absolute row sum bounds the spectral norm, so that bound
    # sets the scale without another eigensolve. The dense path has the full
    # spectrum and used its own maximum above.
    residual_bounds = torch.linalg.vector_norm(
        torch.sparse.mm(L, vecs) - vecs * vals, dim=0,
    ) / torch.linalg.vector_norm(vecs, dim=0)
    cutoff = torch.maximum(relative_tol * kernel_scale, residual_bounds)
    nonzero = vals > cutoff
    n_nonzero = int(nonzero.sum())
    if n_nonzero < k:
        reason = (
            "the full spectrum contains too few resolved positive modes"
            if k_req == n else
            f"increase kernel_headroom={kernel_headroom} to inspect more modes"
        )
        raise ValueError(
            f"only {n_nonzero} resolved positive modes among the {k_req} computed "
            f"for k={k}; {reason} (remaining modes are in the sheaf kernel "
            f"or unresolved at the solver accuracy; scale cutoff="
            f"{float(relative_tol * kernel_scale):.6g}, "
            f"max residual={float(residual_bounds.max()):.6g})"
        )
    vals = vals[nonzero][:k]
    vecs = vecs[:, nonzero][:, :k]
    return vals, vecs, _node_offsets(sheaf)


def sheaf_dirichlet_energy(
    sheaf: GraphSheaf | HeterogeneousGraphSheaf, x: torch.Tensor,
) -> torch.Tensor:
    """Per-batch Dirichlet energy `E(x) = x^T L_F x` for sheaf-valued
    signals.

    The Dirichlet energy is the squared coboundary norm; it measures
    how much each edge's two restriction-mapped values disagree.
    Minimized at the global sections (kernel of `L_F`).

    Args:
      sheaf: a `GraphSheaf` or `HeterogeneousGraphSheaf`.
      x: `(B, n_v · d_v)` or `(n_v · d_v,)` node-stalk signals.

    Returns:
      Scalar `()` for unbatched input or `(B,)` for batched input.

    References:
      Hansen-Ghrist (2019), §3.2.
    """
    total = sheaf.total_node_dim
    if x.ndim not in (1, 2) or x.shape[-1] != total:
        raise ValueError(
            f"x last dim must equal the total node-stalk dim (sum of d_v over "
            f"n_nodes) = {total}; got x.shape={tuple(x.shape)}"
        )
    delta = sheaf_coboundary(sheaf)                  # (n_e·d_e, n_v·d_v)
    # x: (..., n_v·d_v); δ x: (..., n_e·d_e); energy: (..., )
    delta_x = torch.einsum("ij,...j->...i", delta, x)
    return (delta_x * delta_x).sum(dim=-1)
