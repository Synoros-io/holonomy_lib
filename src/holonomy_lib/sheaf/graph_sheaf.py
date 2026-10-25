# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Synoros

"""GraphSheaf dataclass — the data of a cellular sheaf on a graph.

The sheaf attaches a `d_v`-dim stalk to each node and a `d_e`-dim
stalk to each edge. For every edge `e = (u, v)` it carries two
restriction maps:

  F_left[e]  : R^{d_v} → R^{d_e}   (the map from node u's stalk)
  F_right[e] : R^{d_v} → R^{d_e}   (the map from node v's stalk)

The coboundary on a node-stalk vector `x ∈ R^{n_v · d_v}` reads each
edge's contribution as `F_left[e] · x_u − F_right[e] · x_v`. The
sheaf Laplacian is `δ^T δ`.

v1 restrictions:
  - Uniform stalk dims: every node has the same `d_v`, every edge the
    same `d_e`. (Heterogeneous stalks are mathematically fine and a
    natural v2 extension; v1 keeps the storage as dense tensors.)
  - Undirected graphs: edges are unordered, but our (u, v) tuple
    convention encodes the orientation that defines `δ`'s sign
    convention. Two sheaves with the same edge set but swapped (u, v)
    orientations represent the same underlying sheaf with reversed
    coboundary signs; the Laplacian `L_F = δ^T δ` is identical in
    both since it is quadratic in δ.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch

from holonomy_lib.provenance.protocol import HEX_PREFIX_LEN


@dataclass(frozen=True)
class GraphSheaf:
    """A cellular sheaf on a graph.

    Attributes:
      n_nodes:      number of nodes (each carries a d_v-dim stalk).
      edges:        `(n_edges, 2)` int64 tensor of `(u, v)` pairs.
      node_stalk_dim:  `d_v`.
      edge_stalk_dim:  `d_e`.
      F_left:       `(n_edges, d_e, d_v)` restriction maps for the
                    `u`-side of each edge.
      F_right:      `(n_edges, d_e, d_v)` restriction maps for the
                    `v`-side of each edge.

    Validation:
      - `edges` must be `(n_edges, 2)` int64.
      - `F_left`, `F_right` shapes match `(n_edges, d_e, d_v)`.
      - All tensors must share `device` and a single float dtype.
    """

    n_nodes: int
    edges: torch.Tensor
    node_stalk_dim: int
    edge_stalk_dim: int
    F_left: torch.Tensor
    F_right: torch.Tensor

    def __post_init__(self) -> None:
        if self.n_nodes <= 0:
            raise ValueError(f"n_nodes must be > 0; got {self.n_nodes}")
        if self.node_stalk_dim <= 0:
            raise ValueError(
                f"node_stalk_dim must be > 0; got {self.node_stalk_dim}"
            )
        if self.edge_stalk_dim <= 0:
            raise ValueError(
                f"edge_stalk_dim must be > 0; got {self.edge_stalk_dim}"
            )
        if self.edges.ndim != 2 or self.edges.shape[-1] != 2:
            raise ValueError(
                f"edges must be (n_edges, 2); got "
                f"shape={tuple(self.edges.shape)}"
            )
        n_e = self.edges.shape[0]
        d_e, d_v = self.edge_stalk_dim, self.node_stalk_dim
        for name, t in (("F_left", self.F_left), ("F_right", self.F_right)):
            if t.shape != (n_e, d_e, d_v):
                raise ValueError(
                    f"{name} must have shape (n_edges={n_e}, "
                    f"edge_stalk_dim={d_e}, node_stalk_dim={d_v}); got "
                    f"shape={tuple(t.shape)}"
                )
        if self.F_left.device != self.F_right.device:
            raise ValueError(
                f"F_left and F_right must be on the same device; "
                f"got {self.F_left.device} vs {self.F_right.device}"
            )
        if self.F_left.dtype != self.F_right.dtype:
            raise ValueError(
                f"F_left and F_right must share dtype; got "
                f"{self.F_left.dtype} vs {self.F_right.dtype}"
            )
        # Edge indices must lie in [0, n_nodes).
        if (self.edges < 0).any() or (self.edges >= self.n_nodes).any():
            raise ValueError(
                f"edges entries must be in [0, n_nodes={self.n_nodes})"
            )
        # Self-loops (u == v) silently make the sheaf Laplacian
        # differ from the standard combinatorial Laplacian, which the
        # library-wide CONVENTIONS.md drops via `_graph_utils.
        # drop_self_loops`. Sheaf v1 follows the same convention by
        # rejecting them outright. Pre-process: `mask = u != v; edges
        # = edges[mask]; F_left = F_left[mask]; F_right = F_right[mask]`.
        if (self.edges[:, 0] == self.edges[:, 1]).any():
            raise ValueError(
                "self-loops (u == v in edges) are not supported in v1; "
                "drop them before constructing the GraphSheaf"
            )
        # Duplicate (u, v) entries would silently double the
        # off-diagonal mass of the sheaf Laplacian, breaking the
        # "trivial sheaf → graph Laplacian" reduction. Reject them.
        unique_edges = torch.unique(self.edges, dim=0)
        if unique_edges.shape[0] != self.edges.shape[0]:
            raise ValueError(
                "duplicate edges are not supported in v1; "
                "the resulting sheaf Laplacian would double-count "
                "the duplicated pair's off-diagonal block"
            )

    @property
    def n_edges(self) -> int:
        return int(self.edges.shape[0])

    @property
    def total_node_dim(self) -> int:
        """Length of the stacked node-stalk vector, `n_nodes · d_v`."""
        return self.n_nodes * self.node_stalk_dim

    @property
    def device(self) -> torch.device:
        return self.F_left.device

    @property
    def dtype(self) -> torch.dtype:
        return self.F_left.dtype

    def _provenance_signature(self) -> dict:
        """Deterministic canonical form for `@with_provenance` hashing.

        Sheaf identity = topology (n_nodes + edges) + stalk dims +
        device + dtype. The restriction maps are themselves tensors,
        so they pass through `_resolve_tensor_hex` separately and
        carry their content into the input_hexes of any decorated
        sheaf op; we do NOT put them in the signature dict.

        We summarize the edge list as a content hash (sha256 over the
        contiguous int64 bytes) rather than as a Python tuple. The
        provenance hot path canonicalizes this dict to JSON on every
        decorated call; an O(n_edges) tuple would dominate runtime
        for sheaves with tens of thousands of edges.
        """
        edges_bytes = self.edges.cpu().contiguous().numpy().tobytes()
        # Same prefix length as the library-wide content-hash convention
        # (HEX_PREFIX_LEN cataloged in `magic_numbers.md`).
        edges_hash = hashlib.sha256(edges_bytes).hexdigest()[:HEX_PREFIX_LEN]
        return {
            "class": "GraphSheaf",
            "n_nodes": self.n_nodes,
            "node_stalk_dim": self.node_stalk_dim,
            "edge_stalk_dim": self.edge_stalk_dim,
            "n_edges": self.n_edges,
            "edges_sha256_prefix": edges_hash,
            "device": str(self.device),
            "dtype": str(self.dtype),
        }

    @staticmethod
    def trivial(
        n_nodes: int,
        edges: torch.Tensor,
        stalk_dim: int = 1,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float64,
    ) -> GraphSheaf:
        """Construct the *trivial* sheaf: identity restriction maps on
        every edge, uniform stalk dimension on nodes and edges.

        With `stalk_dim = 1` this is the sheaf whose Laplacian equals
        the standard combinatorial graph Laplacian; with `stalk_dim
        = d > 1` it's the block-diagonal `d × L` "d copies of the
        scalar Laplacian" sheaf. Useful baseline + smoke test.
        """
        if edges.ndim != 2 or edges.shape[-1] != 2:
            raise ValueError(
                f"edges must be (n_edges, 2); got shape={tuple(edges.shape)}"
            )
        n_e = edges.shape[0]
        eye = torch.eye(stalk_dim, dtype=dtype, device=device)
        F = eye.unsqueeze(dim=0).expand(n_e, stalk_dim, stalk_dim).contiguous()
        edges = edges.to(device=device, dtype=torch.int64)
        # Independent clones for F_left and F_right: external mutation
        # of one must not affect the other (the frozen-dataclass
        # contract implies value semantics for callers).
        return GraphSheaf(
            n_nodes=n_nodes,
            edges=edges,
            node_stalk_dim=stalk_dim,
            edge_stalk_dim=stalk_dim,
            F_left=F.clone(),
            F_right=F.clone(),
        )


@dataclass(frozen=True)
class HeterogeneousGraphSheaf:
    """A cellular sheaf with PER-NODE (heterogeneous) stalk dimensions.

    The v2 generalization of `GraphSheaf`: node `i` carries a `d_v[i]`-dim stalk
    and edge `e` a `d_e[e]`-dim stalk, all free to differ. For edge `e = (u, v)`:

      F_left[e]  : R^{d_v[u]} -> R^{d_e[e]}   shape (d_e[e], d_v[u])
      F_right[e] : R^{d_v[v]} -> R^{d_e[e]}   shape (d_e[e], d_v[v])

    Coboundary, Laplacian, and Dirichlet energy mean exactly what they do in the
    uniform case (each edge block of `δ` is `F_left[e]·x_u − F_right[e]·x_v`,
    `L_F = δ^T δ`); only the per-node/per-edge dims and the ragged restriction
    maps differ. The maps are stored as a tuple of tensors (ragged), since the
    dims vary, rather than one dense `(n_e, d_e, d_v)` block.

    This is the structure needed when a node's stalk dimension IS a per-node
    quantity (e.g. each concept's relational rank) rather than a single global K.

    Attributes:
      n_nodes:         number of nodes.
      edges:           `(n_edges, 2)` int64 `(u, v)` pairs.
      node_stalk_dims: `(n_nodes,)` int64, per-node stalk dim `d_v[i] > 0`.
      edge_stalk_dims: `(n_edges,)` int64, per-edge stalk dim `d_e[e] > 0`.
      F_left:          length-`n_edges` tuple; `F_left[e]` is `(d_e[e], d_v[u])`.
      F_right:         length-`n_edges` tuple; `F_right[e]` is `(d_e[e], d_v[v])`.

    References:
      Hansen-Ghrist (2019): cellular sheaves carry per-cell stalks of arbitrary
        finite dimension; the uniform `GraphSheaf` is the special case.
    """

    n_nodes: int
    edges: torch.Tensor
    node_stalk_dims: torch.Tensor
    edge_stalk_dims: torch.Tensor
    F_left: tuple[torch.Tensor, ...]
    F_right: tuple[torch.Tensor, ...]

    def __post_init__(self) -> None:
        if self.n_nodes <= 0:
            raise ValueError(f"n_nodes must be > 0; got {self.n_nodes}")
        if self.edges.ndim != 2 or self.edges.shape[-1] != 2:
            raise ValueError(
                f"edges must be (n_edges, 2); got shape={tuple(self.edges.shape)}"
            )
        n_e = int(self.edges.shape[0])
        if tuple(self.node_stalk_dims.shape) != (self.n_nodes,):
            raise ValueError(
                f"node_stalk_dims must be (n_nodes={self.n_nodes},); got "
                f"shape={tuple(self.node_stalk_dims.shape)}"
            )
        if tuple(self.edge_stalk_dims.shape) != (n_e,):
            raise ValueError(
                f"edge_stalk_dims must be (n_edges={n_e},); got "
                f"shape={tuple(self.edge_stalk_dims.shape)}"
            )
        if (self.node_stalk_dims <= 0).any() or (self.edge_stalk_dims <= 0).any():
            raise ValueError("all stalk dims must be > 0")
        if len(self.F_left) != n_e or len(self.F_right) != n_e:
            raise ValueError(
                f"F_left and F_right must each have n_edges={n_e} maps; got "
                f"{len(self.F_left)} and {len(self.F_right)}"
            )
        if (self.edges < 0).any() or (self.edges >= self.n_nodes).any():
            raise ValueError(
                f"edges entries must be in [0, n_nodes={self.n_nodes})"
            )
        if (self.edges[:, 0] == self.edges[:, 1]).any():
            raise ValueError(
                "self-loops (u == v in edges) are not supported; drop them first"
            )
        if torch.unique(self.edges, dim=0).shape[0] != n_e:
            raise ValueError("duplicate edges are not supported")
        # Per-edge restriction-map shapes must match the incident stalk dims.
        node_dims = self.node_stalk_dims.tolist()
        edge_dims = self.edge_stalk_dims.tolist()
        uv = self.edges.tolist()
        for e in range(n_e):
            u, v = uv[e]
            want_left = (edge_dims[e], node_dims[u])
            want_right = (edge_dims[e], node_dims[v])
            if tuple(self.F_left[e].shape) != want_left:
                raise ValueError(
                    f"F_left[{e}] must be (d_e[e], d_v[u])={want_left}; got "
                    f"{tuple(self.F_left[e].shape)}"
                )
            if tuple(self.F_right[e].shape) != want_right:
                raise ValueError(
                    f"F_right[{e}] must be (d_e[e], d_v[v])={want_right}; got "
                    f"{tuple(self.F_right[e].shape)}"
                )
        if n_e > 0:
            dev, dt = self.F_left[0].device, self.F_left[0].dtype
            for t in (*self.F_left, *self.F_right):
                if t.device != dev or t.dtype != dt:
                    raise ValueError(
                        "all restriction maps must share device and dtype"
                    )

    @property
    def n_edges(self) -> int:
        return int(self.edges.shape[0])

    @property
    def total_node_dim(self) -> int:
        """Length of the stacked node-stalk vector, `Σ_i d_v[i]`."""
        return int(self.node_stalk_dims.sum())

    @property
    def device(self) -> torch.device:
        return (self.F_left[0].device if self.n_edges
                else self.node_stalk_dims.device)

    @property
    def dtype(self) -> torch.dtype:
        return self.F_left[0].dtype if self.n_edges else torch.float64

    def _provenance_signature(self) -> dict:
        """Deterministic canonical form for `@with_provenance` hashing: topology
        (n_nodes + edges) + the per-node/per-edge stalk dims + device/dtype. As
        in `GraphSheaf`, the restriction-map tensors are not put in the dict."""
        edges_bytes = self.edges.cpu().contiguous().numpy().tobytes()
        dims_bytes = (
            self.node_stalk_dims.cpu().contiguous().numpy().tobytes()
            + self.edge_stalk_dims.cpu().contiguous().numpy().tobytes()
        )
        return {
            "class": "HeterogeneousGraphSheaf",
            "n_nodes": self.n_nodes,
            "n_edges": self.n_edges,
            "edges_sha256_prefix": hashlib.sha256(edges_bytes).hexdigest()[:HEX_PREFIX_LEN],
            "stalk_dims_sha256_prefix": hashlib.sha256(dims_bytes).hexdigest()[:HEX_PREFIX_LEN],
            "device": str(self.device),
            "dtype": str(self.dtype),
        }
