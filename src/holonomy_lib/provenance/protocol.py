# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Synoros

"""Content-addressable provenance protocol for math primitives.

Core types:
  ProvenanceNode    — one operation in the Merkle DAG.
  ProvenanceRegistry — content-addressed store of nodes + tensor cache.
  with_provenance   — decorator that wraps a primitive so it emits a node
                      when called inside a `record()` context.
  record            — context manager that activates recording.

Hex computation (see `_op_hex` and `_tensor_content_hex`):
  - For a user-supplied tensor with no upstream op:
      hex = sha256(shape + dtype + tensor.bytes)[:HEX_PREFIX_LEN]
  - For the output of an op:
      hex = sha256(op_id + op_version + canonical(params)
                   + ":".join(input_hexes))[:HEX_PREFIX_LEN]
  - Multi-output ops: each output i carries the hex `op_hex:i`.

The protocol is opt-in: outside a `record()` context the decorator is
transparent. Performance impact when not recording is one Python
attribute lookup per call.

Thread-safety: the recording context is thread-local. Sub-threads do
not inherit the parent's recording state.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import threading
import warnings
import weakref
from collections import OrderedDict, deque
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from functools import wraps
from pathlib import Path
from typing import Any, Literal

import torch


class ProvenanceVersionWarning(UserWarning):
    """Emitted when loading a registry whose recorded op_versions
    differ from the currently-installed implementations.

    A version drift means the math primitive has been bumped since
    the registry was recorded — replay results may legitimately differ
    from the cached values. Loading proceeds (the metadata is still
    useful for inspection); replay() against drifted ops is the user's
    call to make.
    """

# Pluggable hash algorithm. blake3 is ~5-10× faster than sha256 and
# still cryptographic; falls back to sha256 if blake3 isn't installed.
# The hex output format is identical so downstream code is unaffected,
# but a registry's hexes are tied to whichever hash was used to record
# it — recordings are not portable across hash algorithms.
try:
    import blake3 as _blake3

    def _make_hasher_blake3():
        return _blake3.blake3()

    _DEFAULT_HASHER_NAME = "blake3"
    _HASHER_FACTORIES: dict[str, Callable] = {
        "blake3": _make_hasher_blake3,
        "sha256": hashlib.sha256,
    }
except ImportError:
    _DEFAULT_HASHER_NAME = "sha256"
    _HASHER_FACTORIES = {"sha256": hashlib.sha256}


# How many hex chars to keep from the sha256 prefix. 16 = 64 bits ≈ 2^32
# expected ops before collision, more than enough for interactive
# sessions. Bump to 32 (= 128 bits) if you're recording huge runs.
HEX_PREFIX_LEN: int = 16

# Sketch-mode hashing: how many evenly-spaced samples to draw from a
# tensor's flattened content. Combined with shape, dtype, sum, and std
# this gives an O(1)-bytes-hashed digest regardless of tensor size,
# trading negligible collision risk for ~100× speedup on multi-MB
# tensors. Empirical collision rate on 10⁴ random (256, 256) float64
# tensors is 0 (no collisions); structural-collision risk (two tensors
# that share strided samples + sum + std but differ in the unsampled
# elements) is bounded but non-zero — full mode remains the default.
SKETCH_SAMPLES: int = 64

# Display-cap constants for human-readable text output. All cataloged
# as 🔬 experimentally-set in notes/magic_numbers.md.
#
# Default cap on the number of per-op-id rows that to_llm_context()
# lists before collapsing the rest into "...and N more". 20 keeps
# the output bounded for huge registries while staying useful for
# typical research-scale chains.
LLM_CONTEXT_MAX_OPS_DEFAULT: int = 20

# Preview limit for roots / leaves in to_llm_context() and for the
# per-op-id hex lists in diff_summary(). Five is enough to identify
# pattern; more is noise.
DISPLAY_PREVIEW_COUNT: int = 5

# Detail-line cap in load()'s ProvenanceVersionWarning. Caps both
# the drifted-nodes list and the unknown-op_ids list, so a huge
# registry doesn't produce an unreadable warning. Larger than the
# preview cap because drift is more actionable per entry.
LOAD_DRIFT_DETAIL_LIMIT: int = 10

# Hash mode literal — sketch trades crypto-grade for speed.
HashMode = Literal["full", "sketch"]

# Global registry mapping op_id → (callable, op_version). Used for replay
# / lookup. Populated by the @with_provenance decorator at import time.
OP_REGISTRY: dict[str, tuple[Callable, str]] = {}

# Global registry mapping class name → factory(sig: dict) -> instance.
# Used by `_restore_value` during replay to reconstruct the bound `self`
# of class methods from their provenance signature. Populated at import
# time by `@register_provenance_class`.
_CLASS_REGISTRY: dict[str, Callable[[dict], Any]] = {}


def register_provenance_class(class_name: str | None = None) -> Callable:
    """Decorator: opt a class into class-method replay.

    The class must define a classmethod `_from_signature(cls, sig: dict)`
    that reconstructs an instance from the dict produced by the
    instance's `_provenance_signature(self) -> dict` method. Together
    these two are an inverse pair: `_from_signature(self._provenance_signature())`
    should return an equivalent instance.

    Without this registration, `ProvenanceRegistry.replay()` raises a
    clear `NotImplementedError` when it encounters a method node whose
    `self` was canonicalized into a signature dict.

    Args:
      class_name: registry key. Defaults to the class's `__name__`;
        override when two classes share a name (e.g., across modules
        with the same simple name).
    """
    def decorator(cls):
        if not hasattr(cls, "_from_signature"):
            raise TypeError(
                f"@register_provenance_class: {cls.__name__} must "
                f"implement a classmethod `_from_signature(cls, sig: dict)` "
                f"that reverses `_provenance_signature`."
            )
        name = class_name if class_name is not None else cls.__name__
        _CLASS_REGISTRY[name] = cls._from_signature
        return cls
    return decorator

# Thread-local active recording context.
_local = threading.local()


# ============================================================
# Data model
# ============================================================


@dataclass(frozen=True)
class ProvenanceNode:
    """One operation in the provenance DAG.

    Frozen so nodes are hashable and immutable once registered.

    Attributes:
      hex: content-addressable identifier for this op call.
      op_id: dotted-name identifier, e.g. "holonomy_lib.algebra.linear.truncated_svd".
      op_version: version string; bump when the implementation changes.
      params: canonicalized non-tensor kwargs (sorted JSON-serializable).
      input_hexes: hex codes of input tensors, in argument order.
      output_shape: tuple of int tuples — one shape per output tensor.
        Length 1 for single-output ops; len > 1 for tuple-returning ops.
      output_dtype: tuple of dtype names, one per output.
    """

    hex: str
    op_id: str
    op_version: str
    params: str  # canonical JSON; dict would make node unhashable
    input_hexes: tuple[str, ...]
    output_shape: tuple[tuple[int, ...], ...]
    output_dtype: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def parsed_params(self) -> dict[str, Any]:
        """JSON-decode `params` back into a dict."""
        return json.loads(self.params) if self.params else {}


# ============================================================
# Registry
# ============================================================


class ProvenanceRegistry:
    """Content-addressed registry of provenance nodes.

    Lookup by hex, query by op_id, export to networkx / dataframe / dict,
    and substitute (activation patching) on subsequent calls.
    """

    def __init__(
        self,
        cache_tensors: bool = False,
        hash_algorithm: str = _DEFAULT_HASHER_NAME,
        max_cache_size: int | None = None,
        cache_ops: Iterable[str] | None = None,
        hash_mode: HashMode = "full",
        cache_to_disk: str | Path | None = None,
    ):
        self._nodes: dict[str, ProvenanceNode] = {}
        # OrderedDict so we can evict in insertion order ("oldest first")
        # when `max_cache_size` is enforced. Mech-interp users typically
        # want recent intermediates around; if a workload needs LRU-by-
        # access, swap to `move_to_end` on read here.
        self._tensor_cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        # Setting cache_to_disk implies caching is on — there's no point
        # configuring a disk path if nothing is being cached. Make this
        # implicit so users only need the one kwarg.
        if cache_to_disk is not None:
            cache_tensors = True
        self._cache_tensors: bool = cache_tensors
        self._max_cache_size: int | None = max_cache_size
        # Selective caching: if set, only outputs from these op_ids are
        # cached. Implies caching is on for those ops regardless of
        # `cache_tensors`. None means "cache all when cache_tensors=True".
        self._cache_ops: set[str] | None = (
            set(cache_ops) if cache_ops is not None else None
        )
        if max_cache_size is not None and max_cache_size <= 0:
            raise ValueError(
                f"max_cache_size must be > 0 or None, got {max_cache_size}"
            )
        # tensor_id (Python id) → (weakref, hex), for chaining inputs to
        # upstream outputs. The weakref guards against id reuse: when a
        # tensor is GC'd, Python may hand out its memory address to a
        # new allocation; without the weakref check, the new tensor
        # would inherit the dead tensor's hex (silent corruption).
        self._tensor_id_to_hex: dict[
            int, tuple[weakref.ReferenceType, str]
        ] = {}
        # User-set substitutions: hex → tensor (or tuple of tensors).
        self._substitutions: dict[str, Any] = {}
        # Hook callbacks: op_id → list[Callable[[ProvenanceNode, Any], None]]
        self._hooks: dict[str, list[Callable[[ProvenanceNode, Any], None]]] = {}
        # Hash algorithm — frozen at construction so all hexes in this
        # registry are computed consistently.
        if hash_algorithm not in _HASHER_FACTORIES:
            raise ValueError(
                f"hash_algorithm must be one of {sorted(_HASHER_FACTORIES)}, "
                f"got {hash_algorithm!r}"
            )
        self.hash_algorithm: str = hash_algorithm
        self._hasher_factory = _HASHER_FACTORIES[hash_algorithm]
        if hash_mode not in ("full", "sketch"):
            raise ValueError(
                f"hash_mode must be 'full' or 'sketch', got {hash_mode!r}"
            )
        self.hash_mode: HashMode = hash_mode
        # Disk cache: a directory that mirrors the in-memory cache.
        # When set, _cache_put writes the tensor both to memory and to
        # disk; memory eviction (via max_cache_size) drops the in-memory
        # copy but the disk copy persists and is reloaded on demand.
        # Use an absolute path so a registry survives a chdir.
        self._disk_cache_dir: Path | None = (
            Path(cache_to_disk).expanduser().resolve()
            if cache_to_disk is not None else None
        )
        self._disk_cache_keys: set[str] = set()
        if self._disk_cache_dir is not None:
            self._disk_cache_dir.mkdir(parents=True, exist_ok=True)
        # User-input cache: tensors that were passed AS INPUTS to a
        # recorded op (not produced by one). Kept separate from
        # `_tensor_cache` so `max_cache_size` continues to bound only
        # op outputs — user inputs are typically few and replay needs
        # them all to reconstruct a chain.
        self._user_input_cache: dict[str, torch.Tensor] = {}

    # ----- low-level registration -----

    def _register(
        self,
        node: ProvenanceNode,
        output: Any,
    ) -> None:
        self._nodes[node.hex] = node
        # Decide whether to cache this output:
        #   - if `cache_ops` was set, only cache when op_id is in the set
        #     (this implicitly enables caching for those ops, regardless
        #      of `cache_tensors`)
        #   - else, fall through to the boolean `cache_tensors`
        if self._cache_ops is not None:
            should_cache = node.op_id in self._cache_ops
        else:
            should_cache = self._cache_tensors
        if not should_cache:
            return
        if isinstance(output, torch.Tensor):
            self._cache_put(node.hex, output)
        elif isinstance(output, tuple):
            for i, o in enumerate(output):
                if isinstance(o, torch.Tensor):
                    self._cache_put(f"{node.hex}:{i}", o)

    def _cache_put(self, key: str, value: torch.Tensor) -> None:
        """Insert into the tensor cache, evicting oldest if size-bounded.

        When `cache_to_disk` is set, the tensor is also persisted via
        `torch.save` to a file named for the (filesystem-sanitized)
        hex key. Memory eviction (from `max_cache_size`) drops the
        in-memory copy but the disk copy persists; `get_tensor` will
        reload it on demand.
        """
        if key in self._tensor_cache:
            # Refresh to the end so it's not the next eviction victim.
            self._tensor_cache.move_to_end(key)
        self._tensor_cache[key] = value
        if self._disk_cache_dir is not None:
            torch.save(
                value.detach().cpu(),
                self._disk_path_for(key),
            )
            self._disk_cache_keys.add(key)
        if (
            self._max_cache_size is not None
            and len(self._tensor_cache) > self._max_cache_size
        ):
            # popitem(last=False) pops the oldest entry from memory.
            # Disk copy (if any) is retained — get_tensor will reload it.
            self._tensor_cache.popitem(last=False)

    def _disk_path_for(self, key: str) -> Path:
        """Filesystem path for a cache key. Replaces ':' (multi-output
        index separator) with '__' for Windows compatibility.
        """
        assert self._disk_cache_dir is not None
        safe = key.replace(":", "__")
        return self._disk_cache_dir / f"{safe}.pt"

    def clear(self, delete_disk: bool = False) -> None:
        """Drop the in-memory caches. Optionally also delete disk artifacts.

        Args:
          delete_disk: if True and a disk cache is configured, remove
            every persisted `.pt` file (and the directory if it becomes
            empty). Default False so that a `clear()` doesn't silently
            destroy persisted state.
        """
        self._tensor_cache.clear()
        self._user_input_cache.clear()
        if delete_disk and self._disk_cache_dir is not None:
            for key in list(self._disk_cache_keys):
                p = self._disk_path_for(key)
                if p.exists():
                    p.unlink()
            self._disk_cache_keys.clear()

    def _register_tensor_hex(self, t: torch.Tensor, hex_id: str) -> None:
        """Tag a tensor with its hex so downstream ops can chain."""
        self._tensor_id_to_hex[id(t)] = (weakref.ref(t), hex_id)

    def _lookup_tensor_hex(self, t: torch.Tensor) -> str | None:
        """Return the stored hex for `t`, or None if no live entry exists.

        The stored weakref must still resolve to the same tensor object;
        if it has been GC'd and `id(t)` was reused for a different tensor,
        we treat the entry as stale and let the caller fall back to a
        fresh content hash.
        """
        entry = self._tensor_id_to_hex.get(id(t))
        if entry is None:
            return None
        ref, hex_id = entry
        if ref() is not t:
            # id reuse: drop the stale entry so we don't keep checking.
            self._tensor_id_to_hex.pop(id(t), None)
            return None
        return hex_id

    # ----- public query API -----

    def __getitem__(self, hex_id: str) -> ProvenanceNode:
        return self._nodes[hex_id]

    def __contains__(self, hex_id: str) -> bool:
        return hex_id in self._nodes

    def __iter__(self) -> Iterator[ProvenanceNode]:
        return iter(self._nodes.values())

    def __len__(self) -> int:
        return len(self._nodes)

    def where(
        self, op_id: str | None = None, op_version: str | None = None,
    ) -> list[ProvenanceNode]:
        """Filter nodes by op_id and/or op_version."""
        out = list(self)
        if op_id is not None:
            out = [n for n in out if n.op_id == op_id]
        if op_version is not None:
            out = [n for n in out if n.op_version == op_version]
        return out

    def get_tensor(self, hex_id: str) -> torch.Tensor | None:
        """Retrieve a cached tensor by hex (requires cache_tensors=True).

        Looks in three places:
          1. In-memory op-output cache (`_tensor_cache`).
          2. In-memory user-input cache (`_user_input_cache`).
          3. Disk cache (if configured), which mirrors both.

        Returns None if the key isn't cached anywhere.
        """
        if hex_id in self._tensor_cache:
            return self._tensor_cache[hex_id]
        if hex_id in self._user_input_cache:
            return self._user_input_cache[hex_id]
        if (
            self._disk_cache_dir is not None
            and hex_id in self._disk_cache_keys
        ):
            t = torch.load(
                self._disk_path_for(hex_id),
                map_location="cpu",
                weights_only=True,
            )
            # Re-promote to memory; respects max_cache_size eviction.
            self._cache_put_memory_only(hex_id, t)
            return t
        return None

    def _cache_put_memory_only(self, key: str, value: torch.Tensor) -> None:
        """Insert into the in-memory cache only (no disk write).

        Used by `get_tensor` to re-promote a disk-loaded tensor without
        re-persisting it — the disk copy already exists.
        """
        if key in self._tensor_cache:
            self._tensor_cache.move_to_end(key)
        self._tensor_cache[key] = value
        if (
            self._max_cache_size is not None
            and len(self._tensor_cache) > self._max_cache_size
        ):
            self._tensor_cache.popitem(last=False)

    def parents(self, hex_id: str) -> list[ProvenanceNode]:
        """Direct upstream nodes (input_hexes of `hex_id`)."""
        node = self._nodes[hex_id]
        parents = []
        for input_hex in node.input_hexes:
            # input_hex is "name=hex" or "name=hex:i" — extract the base hex
            _, _, hex_part = input_hex.partition("=")
            base = hex_part.split(":")[0]
            if base in self._nodes:
                parents.append(self._nodes[base])
        return parents

    def ancestors(self, hex_id: str) -> set[str]:
        """All transitive upstream hexes — the Merkle DAG above `hex_id`."""
        seen: set[str] = set()
        stack = [hex_id]
        while stack:
            h = stack.pop()
            base = h.split(":")[0]
            if base in seen or base not in self._nodes:
                continue
            seen.add(base)
            for input_hex in self._nodes[base].input_hexes:
                _, _, hex_part = input_hex.partition("=")
                stack.append(hex_part)
        return seen

    def ancestors_with_tensors(
        self, hex_id: str,
    ) -> dict[str, tuple[ProvenanceNode, torch.Tensor | None]]:
        """Ancestor subgraph paired with cached output tensors.

        One call instead of `ancestors()` plus N `get_tensor()` calls —
        useful in interactive sessions where you want to inspect the
        full upstream chain after recording.

        Returns:
          A dict mapping each ancestor hex to `(node, tensor_or_None)`.
          The tensor is None for ancestors that weren't cached (e.g.
          the recording ran without `cache_tensors=True`, or the
          tensor was evicted under `max_cache_size`).
        """
        result: dict[str, tuple[ProvenanceNode, torch.Tensor | None]] = {}
        for h in self.ancestors(hex_id):
            node = self._nodes[h]
            result[h] = (node, self.get_tensor(h))
        return result

    # ----- hooks / callbacks (observation without mutation) -----

    def on_op(
        self, op_id: str,
        callback: Callable[[ProvenanceNode, Any], None],
    ) -> None:
        """Register a callback that fires every time `op_id` is called
        inside this registry's recording context.

        The callback receives `(node, output)` where `output` is whatever
        the decorated primitive returned (a tensor, tuple of tensors, etc.).
        Multiple callbacks for the same op_id are called in registration order.

        TransformerLens-equivalent: forward-hook on a module, but indexed
        by op_id instead of module path. Doesn't change behavior (unlike
        substitute()); just lets you observe / log / accumulate.

        Example:
          activations = []
          reg.on_op("holonomy_lib.spectral.laplacian.combinatorial",
                      lambda node, out: activations.append((node.hex, out)))
          # ... run the pipeline ...
          # activations now lists every Laplacian computed
        """
        self._hooks.setdefault(op_id, []).append(callback)

    def clear_hooks(self, op_id: str | None = None) -> None:
        """Remove all hooks (or just for one op_id)."""
        if op_id is None:
            self._hooks.clear()
        else:
            self._hooks.pop(op_id, None)

    # ----- substitution / activation patching -----

    @contextmanager
    def substitute(self, mapping: dict[str, Any]) -> Iterator[ProvenanceRegistry]:
        """Within this context, decorated primitives that *would* produce a
        hex in `mapping` instead return the mapped value.

        Use case: activation patching à la TransformerLens, applied to
        math primitives. The caller re-runs the chain; downstream ops
        compute on the substituted tensor.

        Example:
          with provenance.record() as reg:
              result = some_pipeline(x)
              target = reg.where(op_id="my.op")[0].hex
              with reg.substitute({target: zeros}):
                  ablated = some_pipeline(x)  # `target` returns zeros
        """
        prev = dict(self._substitutions)
        self._substitutions.update(mapping)
        try:
            yield self
        finally:
            self._substitutions = prev

    # ----- export / interop -----

    def to_networkx(self):
        """Export the DAG as a NetworkX DiGraph (requires networkx)."""
        import networkx as nx  # optional dep — imported lazily

        G = nx.DiGraph()
        for node in self:
            G.add_node(node.hex, **{
                "op_id": node.op_id,
                "op_version": node.op_version,
                "params": node.params,
                "output_shape": node.output_shape,
                "output_dtype": node.output_dtype,
            })
            for input_hex in node.input_hexes:
                _, _, hex_part = input_hex.partition("=")
                base = hex_part.split(":")[0]
                G.add_edge(base, node.hex)
        return G

    def to_dataframe(self):
        """Export node table as a pandas DataFrame (requires pandas)."""
        import pandas as pd  # optional dep
        return pd.DataFrame([n.to_dict() for n in self])

    def to_dict(self) -> dict[str, Any]:
        """Plain-Python JSON-friendly export."""
        return {
            "schema_version": "0.3",
            "hash_algorithm": self.hash_algorithm,
            "hash_mode": self.hash_mode,
            "cache_to_disk": (
                str(self._disk_cache_dir)
                if self._disk_cache_dir is not None else None
            ),
            "disk_cache_keys": sorted(self._disk_cache_keys),
            "nodes": [n.to_dict() for n in self],
        }

    def to_llm_context(
        self,
        max_ops: int = LLM_CONTEXT_MAX_OPS_DEFAULT,
        show_shapes: bool = True,
        show_params: bool = False,
    ) -> str:
        """Compact text summary suitable for an LLM agent prompt.

        Format:
          Provenance registry: N ops, M cached tensors, hash_mode=...
          Ops by op_id:
            <op_id> x <count>
            ...
          Notable shapes:
            <short_op> @ <hex>: <shape> <dtype>
            ...
          Roots:  <hex>, <hex>, ...
          Leaves: <hex>, <hex>, ...

        Args:
          max_ops: cap on the number of per-op-id rows and per-shape
            rows. Keeps the output bounded for large registries.
          show_shapes: include a "Notable shapes" block listing each
            distinct output shape that appears in the registry.
          show_params: include each node's params dict in the
            ops-by-op_id block (verbose; off by default).
        """
        n_cached = len(self._tensor_cache) + len(self._user_input_cache)
        lines = [
            f"Provenance registry: {len(self._nodes)} ops, "
            f"{n_cached} cached tensors, hash_mode={self.hash_mode}"
        ]

        # Count nodes by op_id.
        by_op_id: dict[str, list[ProvenanceNode]] = {}
        for n in self:
            by_op_id.setdefault(n.op_id, []).append(n)
        if by_op_id:
            lines.append("Ops by op_id:")
            sorted_ops = sorted(by_op_id.items(), key=lambda kv: -len(kv[1]))
            for op_id, nodes in sorted_ops[:max_ops]:
                lines.append(f"  {op_id} x {len(nodes)}")
                if show_params:
                    # Show the first node's params; subsequent calls
                    # often share them.
                    lines.append(f"    params: {nodes[0].params}")
            if len(sorted_ops) > max_ops:
                lines.append(
                    f"  ...and {len(sorted_ops) - max_ops} more op_id(s)"
                )

        # Notable shapes: unique (op_id_basename, hex_prefix, shape, dtype) rows.
        if show_shapes and self._nodes:
            lines.append("Notable shapes:")
            shown = 0
            for n in self:
                if shown >= max_ops:
                    break
                short_op = n.op_id.rsplit(".", 1)[-1]
                shape_str = " ".join(str(s) for s in n.output_shape)
                dtype_str = " ".join(n.output_dtype)
                lines.append(
                    f"  {short_op} @ {n.hex}: {shape_str} {dtype_str}"
                )
                shown += 1
            if len(self._nodes) > shown:
                lines.append(f"  ...and {len(self._nodes) - shown} more")

        # Roots: nodes with no provenance-internal parents (inputs are
        # either empty or all reference hexes not in _nodes — i.e.,
        # user input tensors).
        # Leaves: nodes that no other node consumes.
        consumed: set[str] = set()
        for n in self:
            for input_hex in n.input_hexes:
                _, _, hex_part = input_hex.partition("=")
                base = hex_part.split(":")[0]
                if base in self._nodes:
                    consumed.add(base)
        roots: list[str] = []
        leaves: list[str] = []
        for n in self:
            has_op_parent = any(
                hp.partition("=")[2].split(":")[0] in self._nodes
                for hp in n.input_hexes
            )
            if not has_op_parent:
                roots.append(n.hex)
            if n.hex not in consumed:
                leaves.append(n.hex)
        if roots:
            shown_roots = roots[:DISPLAY_PREVIEW_COUNT]
            extra = (
                "" if len(roots) <= DISPLAY_PREVIEW_COUNT
                else f", ...({len(roots) - DISPLAY_PREVIEW_COUNT} more)"
            )
            lines.append(f"Roots:  {', '.join(shown_roots)}{extra}")
        if leaves:
            shown_leaves = leaves[:DISPLAY_PREVIEW_COUNT]
            extra = (
                "" if len(leaves) <= DISPLAY_PREVIEW_COUNT
                else f", ...({len(leaves) - DISPLAY_PREVIEW_COUNT} more)"
            )
            lines.append(f"Leaves: {', '.join(shown_leaves)}{extra}")

        return "\n".join(lines)

    def to_mermaid(self) -> str:
        """Render the DAG as a Mermaid flowchart string.

        Output is suitable for inline GitHub Markdown or any Mermaid
        renderer (mermaid.live, IPython display, etc.). Each node
        carries its hex + the last segment of its op_id; edges point
        from parents to children.

        Hexes are already 16 hex chars (no special characters) so they
        serve as Mermaid node IDs directly with no escaping.
        """
        lines = ["flowchart TD"]
        for n in self:
            short_op = n.op_id.rsplit(".", 1)[-1]
            # Escape any quote chars in case op_id ever contains one.
            label = f"{short_op}<br/>{n.hex}".replace('"', '&quot;')
            lines.append(f'    {n.hex}["{label}"]')
        for n in self:
            for input_hex in n.input_hexes:
                _, _, hex_part = input_hex.partition("=")
                base = hex_part.split(":")[0]
                if base in self._nodes:
                    lines.append(f"    {base} --> {n.hex}")
        return "\n".join(lines)

    def to_graphviz(self) -> str:
        """Render the DAG as Graphviz DOT source.

        Returns a string suitable for `graphviz`/`dot` rendering. We
        don't import the optional `graphviz` Python package; users
        pipe the string to `dot -Tpng` (or similar) themselves. This
        keeps the dependency surface zero for inspection-only use.
        """
        lines = ["digraph provenance {"]
        lines.append("    rankdir=TB;")
        lines.append('    node [shape=box, style=rounded];')
        for n in self:
            short_op = n.op_id.rsplit(".", 1)[-1]
            label = f"{short_op}\\n{n.hex}".replace('"', r'\"')
            lines.append(f'    "{n.hex}" [label="{label}"];')
        for n in self:
            for input_hex in n.input_hexes:
                _, _, hex_part = input_hex.partition("=")
                base = hex_part.split(":")[0]
                if base in self._nodes:
                    lines.append(f'    "{base}" -> "{n.hex}";')
        lines.append("}")
        return "\n".join(lines)

    def to_sae_dataset(
        self, op_id: str | None = None,
    ) -> Iterator[tuple[torch.Tensor, dict[str, Any]]]:
        """Yield `(tensor, metadata)` pairs from cached outputs, in registration
        order. Useful as input to SAELens-style training where you want a
        stream of activations + labels.

        Args:
          op_id: if given, only yield records for this op.

        Yields:
          (output_tensor, dict) — the dict has {"hex", "op_id", "op_version",
          "params", "input_hexes"} for downstream filtering / labeling.

        Requires `cache_tensors=True` on the recording context.
        """
        for node in self.where(op_id=op_id):
            tensor = self.get_tensor(node.hex)
            if tensor is not None:
                yield tensor, {
                    "hex": node.hex,
                    "op_id": node.op_id,
                    "op_version": node.op_version,
                    "params": node.params,
                    "input_hexes": list(node.input_hexes),
                }
            else:
                # Multi-output op: yield each cached output with its hex:i
                for i in range(len(node.output_shape)):
                    sub_hex = f"{node.hex}:{i}"
                    t = self.get_tensor(sub_hex)
                    if t is not None:
                        yield t, {
                            "hex": sub_hex,
                            "op_id": node.op_id,
                            "op_version": node.op_version,
                            "params": node.params,
                            "input_hexes": list(node.input_hexes),
                            "output_index": i,
                        }

    # ----- diff -----

    def diff(self, other: ProvenanceRegistry) -> dict[str, Any]:
        """Compare two recordings and return a structured diff.

        Useful for "did my refactor preserve semantics?" or "what changed
        between these two experimental conditions?". The diff groups by
        op_id so you see, per operation, which hexes are unique to each
        registry vs. shared.

        Returns:
          dict with keys:
            "only_in_self":   {op_id: [hex, ...]}  — ops in self, not other
            "only_in_other":  {op_id: [hex, ...]}
            "shared":         {op_id: [hex, ...]}
            "op_ids_only_in_self":   list[str]
            "op_ids_only_in_other":  list[str]
        """
        self_hexes_by_op: dict[str, set[str]] = {}
        for n in self:
            self_hexes_by_op.setdefault(n.op_id, set()).add(n.hex)
        other_hexes_by_op: dict[str, set[str]] = {}
        for n in other:
            other_hexes_by_op.setdefault(n.op_id, set()).add(n.hex)

        all_ops = set(self_hexes_by_op) | set(other_hexes_by_op)
        only_in_self: dict[str, list[str]] = {}
        only_in_other: dict[str, list[str]] = {}
        shared: dict[str, list[str]] = {}
        for op in all_ops:
            s = self_hexes_by_op.get(op, set())
            o = other_hexes_by_op.get(op, set())
            if s - o:
                only_in_self[op] = sorted(s - o)
            if o - s:
                only_in_other[op] = sorted(o - s)
            if s & o:
                shared[op] = sorted(s & o)

        return {
            "only_in_self": only_in_self,
            "only_in_other": only_in_other,
            "shared": shared,
            "op_ids_only_in_self": sorted(
                set(self_hexes_by_op) - set(other_hexes_by_op),
            ),
            "op_ids_only_in_other": sorted(
                set(other_hexes_by_op) - set(self_hexes_by_op),
            ),
        }

    def diff_summary(self, other: ProvenanceRegistry) -> str:
        """Human-readable rendering of `diff()`.

        Buckets the result into four categories:
          - **Cache hits**: nodes with matching hexes (same op + same
            params + same input chain produced the same hex).
          - **Drift**: same op_id appears in both, but with different
            hexes — same operation called with different parameters
            or upstream tensors.
          - **Only in self / Only in other**: ops that ran only on one
            side.

        Returns a multi-line string; if the registries are identical,
        returns a single-line "(No differences.)" message.
        """
        d = self.diff(other)
        in_self_ops = set(d["only_in_self"])
        in_other_ops = set(d["only_in_other"])
        drifted = in_self_ops & in_other_ops
        new_ops = in_self_ops - in_other_ops
        gone_ops = in_other_ops - in_self_ops

        lines: list[str] = ["Diff: self vs other"]
        if d["shared"]:
            total = sum(len(v) for v in d["shared"].values())
            lines.append(
                f"\nCache hits ({total} call(s) with matching hexes):"
            )
            for op_id, hexes in sorted(d["shared"].items()):
                lines.append(f"  {op_id} x {len(hexes)}")

        if drifted:
            drift_total = sum(
                len(d["only_in_self"][op]) + len(d["only_in_other"][op])
                for op in drifted
            )
            lines.append(
                f"\nDrift ({len(drifted)} op(s), {drift_total} divergent call(s)):"
            )
            for op_id in sorted(drifted):
                self_h = d["only_in_self"][op_id]
                other_h = d["only_in_other"][op_id]
                lines.append(
                    f"  {op_id}: self has "
                    f"{self_h[:DISPLAY_PREVIEW_COUNT]}"
                    f"{'...' if len(self_h) > DISPLAY_PREVIEW_COUNT else ''}, "
                    f"other has "
                    f"{other_h[:DISPLAY_PREVIEW_COUNT]}"
                    f"{'...' if len(other_h) > DISPLAY_PREVIEW_COUNT else ''}"
                )

        if new_ops:
            new_total = sum(len(d["only_in_self"][op]) for op in new_ops)
            lines.append(
                f"\nOnly in self ({len(new_ops)} op(s), {new_total} call(s)):"
            )
            for op_id in sorted(new_ops):
                lines.append(f"  {op_id} x {len(d['only_in_self'][op_id])}")

        if gone_ops:
            gone_total = sum(len(d["only_in_other"][op]) for op in gone_ops)
            lines.append(
                f"\nOnly in other ({len(gone_ops)} op(s), {gone_total} call(s)):"
            )
            for op_id in sorted(gone_ops):
                lines.append(f"  {op_id} x {len(d['only_in_other'][op_id])}")

        if not d["shared"] and not drifted and not new_ops and not gone_ops:
            return "(No differences.)"
        return "\n".join(lines)

    # ----- causal replay: downstream DAG re-execution -----

    def replay(
        self,
        substitutions: dict[str, torch.Tensor],
        final_hex: str | None = None,
    ) -> dict[str, torch.Tensor]:
        """Re-execute the downstream DAG with substitutions propagating.

        Unlike `substitute()` (which only affects ops whose hex is in
        the substitution map at *call* time, requiring you to re-run
        your pipeline), this method walks the recorded DAG and
        re-executes only the affected nodes — the upstream computation
        is reused from the cache, so this is much cheaper than re-running
        a full pipeline when you only want to intervene at one node.

        Args:
          substitutions: hex → replacement tensor. Each substituted hex
            must already exist in the registry's tensor cache.
          final_hex: if given, stop early once this node has been
            re-executed and return only its value (under that key).
            Otherwise return all re-executed nodes' new outputs.

        Returns:
          dict mapping hex → new output tensor for every re-executed
          node. Hexes match the *original* recording's hex (we use the
          original DAG topology; only the *values* change).

        Requires `cache_tensors=True` on the original recording.

        Caveats:
          - Random ops (those that take a Generator and call torch.randn
            etc.) replay with a fresh generator — the result will differ
            from the original. Avoid replaying through stochastic ops.
          - The substituted tensor must be shape-compatible with the
            original at that node, or downstream ops will raise.
        """
        if not self._cache_tensors:
            raise ValueError("replay requires cache_tensors=True on record()")
        for h in substitutions:
            base = h.split(":")[0]
            if base not in self._nodes:
                raise KeyError(
                    f"substitution target {h!r} not in registry"
                )

        # Reverse adjacency: parent_hex → list[child_hex]
        children_of: dict[str, list[str]] = {h: [] for h in self._nodes}
        for child_hex, child_node in self._nodes.items():
            for parent_input in child_node.input_hexes:
                # input_hex is "name=hex"; extract the hex part
                _, _, parent_hex = parent_input.partition("=")
                parent_base = parent_hex.split(":")[0]
                if parent_base in children_of:
                    children_of[parent_base].append(child_hex)

        # Collect ALL hexes downstream of any substituted hex.
        affected: set[str] = set()
        queue = deque(h.split(":")[0] for h in substitutions)
        while queue:
            h = queue.popleft()
            for child in children_of.get(h, ()):
                if child not in affected:
                    affected.add(child)
                    queue.append(child)

        # Topological sort (Kahn's algorithm) restricted to `affected`.
        indegree: dict[str, int] = {}
        for h in affected:
            count = 0
            for parent_input in self._nodes[h].input_hexes:
                _, _, parent_hex = parent_input.partition("=")
                parent_base = parent_hex.split(":")[0]
                if parent_base in affected:
                    count += 1
            indegree[h] = count
        ready = deque(h for h, d in indegree.items() if d == 0)
        order: list[str] = []
        while ready:
            h = ready.popleft()
            order.append(h)
            for child in children_of.get(h, ()):
                if child in indegree:
                    indegree[child] -= 1
                    if indegree[child] == 0:
                        ready.append(child)

        # Shadow cache: starts from baseline (user inputs + op outputs),
        # gets overwritten as we go. Substitutions are merged in
        # immediately so any node that reads them sees the substituted
        # value. User inputs come first so output-cache entries win on
        # any hex collision (shouldn't happen in practice — outputs and
        # inputs use the same content hex, so if the same tensor was
        # used both ways, the entries are equal anyway).
        shadow: dict[str, torch.Tensor] = dict(self._user_input_cache)
        shadow.update(self._tensor_cache)
        for h, val in substitutions.items():
            shadow[h] = val

        new_outputs: dict[str, torch.Tensor] = {}
        for h in order:
            node = self._nodes[h]
            if node.op_id not in OP_REGISTRY:
                raise RuntimeError(
                    f"cannot replay: op_id {node.op_id!r} not in OP_REGISTRY"
                )
            fn, _ver = OP_REGISTRY[node.op_id]
            # Reconstruct call kwargs from input_hexes + params.
            # Restore canonicalized values (e.g. torch.Generator) before
            # invoking the op — params serialize to JSON-stable forms
            # but the op expects real objects.
            call_kwargs: dict[str, Any] = {
                k: _restore_value(v) for k, v in node.parsed_params().items()
            }
            # `_restore_value` reconstructs class instances for any class
            # registered via `@register_provenance_class`. If a param
            # still carries the signature tag here, the class wasn't
            # registered — emit a clear, actionable error.
            for k, v in call_kwargs.items():
                if isinstance(v, dict) and v.get(_PROVENANCE_SIGNATURE_TAG):
                    cls_name = v.get("sig", {}).get("class", "?")
                    raise NotImplementedError(
                        f"replay of class-method calls requires the class "
                        f"to be registered via @register_provenance_class. "
                        f"Class {cls_name!r} (op_id={node.op_id!r}, "
                        f"param {k!r}) is not registered; either register "
                        f"it or use `substitute()` to intervene without "
                        f"replaying through this op."
                    )
            # Walk input_hexes. Scalar tensor inputs go straight to
            # call_kwargs[name]; tuple/list inputs were unpacked into
            # per-element hex keys at record time (e.g. `point[0]`,
            # `point[1]`, `point[2]` for a FixedRankPoint = (U, S, Vt))
            # and need to be reassembled into a positional tuple before
            # calling fn(**call_kwargs).
            tuple_inputs: dict[str, dict[int, torch.Tensor]] = {}
            for input_hex in node.input_hexes:
                name, _, hex_part = input_hex.partition("=")
                if hex_part not in shadow:
                    raise RuntimeError(
                        f"replay: missing tensor for {hex_part!r} (parent of {h!r})"
                    )
                if "[" in name and name.endswith("]"):
                    # name is e.g. "point[2]"; split into base + index.
                    bracket = name.index("[")
                    base_name = name[:bracket]
                    try:
                        index = int(name[bracket + 1:-1])
                    except ValueError:
                        raise RuntimeError(
                            f"replay: malformed tuple-input name {name!r}"
                        )
                    tuple_inputs.setdefault(base_name, {})[index] = shadow[hex_part]
                else:
                    call_kwargs[name] = shadow[hex_part]
            # Reassemble tuple inputs in index order; require contiguous
            # 0..n-1 indices so we don't silently drop a missing element.
            for base_name, by_index in tuple_inputs.items():
                sorted_indices = sorted(by_index)
                if sorted_indices != list(range(len(sorted_indices))):
                    raise RuntimeError(
                        f"replay: tuple input {base_name!r} has non-contiguous "
                        f"indices {sorted_indices!r}; can't reassemble"
                    )
                call_kwargs[base_name] = tuple(by_index[i] for i in sorted_indices)
            output = fn(**call_kwargs)
            # Store in shadow so downstream ops can find it.
            if isinstance(output, torch.Tensor):
                shadow[h] = output
                new_outputs[h] = output
            elif isinstance(output, tuple):
                for i, o in enumerate(output):
                    if isinstance(o, torch.Tensor):
                        shadow[f"{h}:{i}"] = o
                        new_outputs[f"{h}:{i}"] = o
            if final_hex is not None and h.split(":")[0] == final_hex.split(":")[0]:
                # Hit the requested terminus. For multi-output nodes,
                # `shadow` holds the outputs under `h:0`, `h:1`, ... —
                # so a `final_hex` matching only the base (no :i) cannot
                # be looked up directly. Return every shadow entry that
                # belongs to this node, plus the exact-match key if the
                # caller passed `h:i` for a specific output.
                base = h.split(":")[0]
                if final_hex in shadow:
                    return {final_hex: shadow[final_hex]}
                return {
                    k: v for k, v in new_outputs.items()
                    if k.split(":")[0] == base
                }

        if final_hex is not None:
            return {final_hex: shadow[final_hex]} if final_hex in shadow else {}
        return new_outputs

    # ----- persistence -----

    def save(self, path: str | Path) -> None:
        """Persist the registry's metadata to disk as JSON.

        The tensor cache (if any) is NOT persisted — only the DAG
        structure, hexes, op_ids, params, shapes, dtypes. To persist
        tensors, save them separately keyed by hex (e.g. via torch.save
        in a follow-on directory).
        """
        path = Path(path)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(
        cls, path: str | Path, strict: bool = False,
    ) -> ProvenanceRegistry:
        """Load a registry's metadata from a JSON file written by save().

        The loaded registry has no tensor cache (tensors aren't persisted
        unless cache_to_disk was set; in that case the on-disk files are
        re-attached). Substitutions and hooks are reset to empty.

        On load, every node's recorded `op_version` is compared against
        the currently-installed `OP_REGISTRY[op_id]` version. If any
        differ, the behavior depends on `strict`:
          - strict=False (default): emit a `ProvenanceVersionWarning`
            listing the drifted nodes. Loading proceeds; the metadata
            is still useful for inspection.
          - strict=True: raise `ValueError` listing the drifted nodes.
            Use this when you depend on replay() producing values
            consistent with the original recording.

        Unknown op_ids (recorded by an op that's no longer registered
        in this process — e.g. the module wasn't imported) are listed
        in the warning's drift report. They will block replay through
        those nodes with a separate RuntimeError when actually
        attempted.
        """
        path = Path(path)
        data = json.loads(path.read_text())
        algorithm = data.get("hash_algorithm", "sha256")
        # Pre-0.2 schemas predate hash_mode; default to "full" for
        # backward compatibility (the only mode that existed then).
        mode = data.get("hash_mode", "full")
        disk_dir = data.get("cache_to_disk")
        registry = cls(
            hash_algorithm=algorithm,
            hash_mode=mode,
            cache_to_disk=disk_dir,
        )
        # Repopulate the disk-key set so get_tensor() can find the
        # already-persisted files without rewalking the directory.
        for k in data.get("disk_cache_keys", []):
            registry._disk_cache_keys.add(k)
        # Track op_version drift while iterating; warn or raise once
        # all nodes are loaded so the diagnostic is a single message.
        drifted: list[tuple[str, str, str, str]] = []
        unknown: list[tuple[str, str, str]] = []
        for entry in data["nodes"]:
            node = ProvenanceNode(
                hex=entry["hex"],
                op_id=entry["op_id"],
                op_version=entry["op_version"],
                params=entry["params"],
                input_hexes=tuple(entry["input_hexes"]),
                output_shape=tuple(tuple(s) for s in entry["output_shape"]),
                output_dtype=tuple(entry["output_dtype"]),
            )
            registry._nodes[node.hex] = node
            # Check op_version drift against the currently-installed op.
            if node.op_id in OP_REGISTRY:
                current_ver = OP_REGISTRY[node.op_id][1]
                if current_ver != node.op_version:
                    drifted.append(
                        (node.hex, node.op_id, node.op_version, current_ver)
                    )
            else:
                unknown.append((node.hex, node.op_id, node.op_version))
        if drifted or unknown:
            lines: list[str] = []
            if drifted:
                lines.append(
                    f"{len(drifted)} node(s) recorded with op_versions "
                    f"that differ from the currently-installed ops:"
                )
                # Cap the per-message detail so a huge registry doesn't
                # produce an unreadable warning.
                visible = drifted[:LOAD_DRIFT_DETAIL_LIMIT]
                for hex_id, op_id, recorded, current in visible:
                    lines.append(
                        f"  {hex_id} {op_id} recorded={recorded!r} "
                        f"current={current!r}"
                    )
                if len(drifted) > len(visible):
                    lines.append(
                        f"  ...and {len(drifted) - len(visible)} more"
                    )
            if unknown:
                lines.append(
                    f"{len(unknown)} node(s) have op_ids that are not "
                    f"currently registered (the module may not have "
                    f"been imported in this process):"
                )
                visible = unknown[:LOAD_DRIFT_DETAIL_LIMIT]
                for hex_id, op_id, recorded in visible:
                    lines.append(f"  {hex_id} {op_id} recorded={recorded!r}")
                if len(unknown) > len(visible):
                    lines.append(
                        f"  ...and {len(unknown) - len(visible)} more"
                    )
            message = "\n".join(lines)
            if strict:
                raise ValueError(message)
            warnings.warn(message, ProvenanceVersionWarning, stacklevel=2)
        return registry


# ============================================================
# Recording context
# ============================================================


@contextmanager
def record(
    cache_tensors: bool = False,
    hash_algorithm: str = _DEFAULT_HASHER_NAME,
    max_cache_size: int | None = None,
    cache_ops: Iterable[str] | None = None,
    hash_mode: HashMode = "full",
    cache_to_disk: str | Path | None = None,
) -> Iterator[ProvenanceRegistry]:
    """Activate provenance recording for decorated primitives.

    Args:
      cache_tensors: if True, store output tensors in the registry so
        `registry.get_tensor(hex)` returns them. Costs memory; off by
        default to keep the overhead near zero.
      hash_algorithm: hash function name. Defaults to whichever was
        available at import (blake3 if installed, else sha256). All
        hexes within a registry are computed with this algorithm; you
        cannot mix.
      max_cache_size: hard cap on the number of cached output tensors.
        When exceeded, the oldest cached entry is evicted (FIFO). None
        means unbounded — convenient for short pipelines but
        a memory hazard for long ones, especially on GPU.
      cache_ops: if provided, only outputs whose op_id is in this set
        are cached. Overrides `cache_tensors` for selectivity: caching
        is enabled for the listed ops and disabled for everything else.
        Use this when you want to keep, say, every Laplacian but not
        every intermediate matmul.
      hash_mode: "full" (default) hashes the full tensor bytes — slow
        on multi-MB inputs but cryptographically distinguishes content.
        "sketch" hashes `shape + dtype + SKETCH_SAMPLES = 64` evenly-
        strided samples + sum + std — ~100× faster on big tensors at
        the cost of a non-zero (but tiny) collision risk on tensors
        that happen to share the sampled positions and the two
        summary statistics. Hexes are NOT portable across modes; a
        registry serialized in one mode cannot be merged with another.
      cache_to_disk: if set, cached tensors are also persisted to a
        directory at this path (one `.pt` file per cache key). Implies
        `cache_tensors=True`. Memory eviction from `max_cache_size`
        only removes the in-memory copy; the disk copy persists and
        is reloaded on demand by `get_tensor`. Useful when a recording
        produces more cached intermediates than fit in RAM, or when
        you want to inspect the cache after the Python process exits.

    Yields:
      The active ProvenanceRegistry.
    """
    registry = ProvenanceRegistry(
        cache_tensors=cache_tensors,
        hash_algorithm=hash_algorithm,
        max_cache_size=max_cache_size,
        cache_ops=cache_ops,
        hash_mode=hash_mode,
        cache_to_disk=cache_to_disk,
    )
    prev = getattr(_local, "context", None)
    _local.context = registry
    try:
        yield registry
    finally:
        _local.context = prev


def _current_context() -> ProvenanceRegistry | None:
    return getattr(_local, "context", None)


# ============================================================
# Hashing helpers
# ============================================================


# Sentinel tag for torch.Generator canonicalization. A Generator's
# Python repr is its memory address — two generators with the same
# seed get different reprs and therefore different hexes, which
# silently breaks reproducibility. Canonicalize to {seed, device}
# so that "same seed → same hex" holds, and replay can reconstruct
# a Generator with that seed.
_TORCH_GENERATOR_TAG = "__torch_generator__"
_PROVENANCE_SIGNATURE_TAG = "__provenance_signature__"


def _canonicalize_value(v: Any) -> Any:
    """Replace non-JSON-stable objects with canonical representations.

    Currently handles:
      - torch.Generator → recorded seed + device
      - any object exposing a `_provenance_signature(self) -> dict`
        method → that dict (used by manifold instances so class-method
        provenance is deterministic; `<Manifold object at 0x7f...>`
        would otherwise leak the memory address into the hex)

    Other un-serializable values fall through to `default=str` in the
    JSON encoder; if those become a reproducibility hazard, add cases
    here.
    """
    if isinstance(v, torch.Generator):
        return {
            _TORCH_GENERATOR_TAG: True,
            "seed": int(v.initial_seed()),
            "device": str(v.device),
        }
    sig_fn = getattr(v, "_provenance_signature", None)
    if callable(sig_fn):
        return {_PROVENANCE_SIGNATURE_TAG: True, "sig": sig_fn()}
    return v


def _restore_value(v: Any) -> Any:
    """Inverse of `_canonicalize_value` — rebuild objects from params dict.

    Used by `ProvenanceRegistry.replay` to reconstruct call kwargs.
    A torch.Generator restored here is freshly seeded with the recorded
    `initial_seed()`; it does not preserve consumed-state, so replaying
    through stochastic ops that consume the generator partway will
    produce different bits past that consumption point. The replay
    docstring already warns about stochastic ops.

    For class-method provenance signatures, looks up the class in
    `_CLASS_REGISTRY` and calls its `_from_signature(sig)` to
    reconstruct an instance. If the class isn't registered, returns
    the original dict unchanged; the caller (replay) sees the tag and
    raises a clear error.
    """
    if isinstance(v, dict) and v.get(_TORCH_GENERATOR_TAG):
        g = torch.Generator(device=v["device"])
        g.manual_seed(int(v["seed"]))
        return g
    if isinstance(v, dict) and v.get(_PROVENANCE_SIGNATURE_TAG):
        sig = v.get("sig", {})
        class_name = sig.get("class")
        if class_name in _CLASS_REGISTRY:
            return _CLASS_REGISTRY[class_name](sig)
        # Fall through with the unrestored dict so replay's check
        # produces the clearer "class not registered" error.
        return v
    return v


def _tensor_content_hex(t: torch.Tensor, hasher_factory: Callable) -> str:
    """Content hash of a tensor: shape + dtype + bytes."""
    h = hasher_factory()
    h.update(str(tuple(t.shape)).encode())
    h.update(str(t.dtype).encode())
    # Hash bytes — move to CPU and contiguous first.
    h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()[:HEX_PREFIX_LEN]


def _tensor_sketch_hex(t: torch.Tensor, hasher_factory: Callable) -> str:
    """Sketch hash of a tensor: shape + dtype + strided samples + sum + std.

    Hashes O(SKETCH_SAMPLES) bytes regardless of tensor size. Used for
    `hash_mode="sketch"` registries on big tensors where the byte-hash
    in `_tensor_content_hex` dominates recording cost.

    Discriminators (so two visually-different tensors don't collide):
      - shape + dtype (cheap, very strong)
      - SKETCH_SAMPLES evenly-spaced flattened samples (catches local
        variation in the sampled positions)
      - sum (catches any global rescaling)
      - std (catches scale-preserving structural changes — sign flips,
        permutations — that would leave sum invariant)
    """
    h = hasher_factory()
    h.update(str(tuple(t.shape)).encode())
    h.update(str(t.dtype).encode())
    detached = t.detach()
    numel = detached.numel()
    if numel > 0:
        flat = detached.reshape(-1)
        stride = max(1, numel // SKETCH_SAMPLES)
        # Stride-sample at most SKETCH_SAMPLES elements; on tensors
        # smaller than SKETCH_SAMPLES, this is the whole tensor.
        samples = flat[::stride][:SKETCH_SAMPLES].to(
            device="cpu", dtype=torch.float64
        ).contiguous()
        h.update(samples.numpy().tobytes())
        # Sum + std as global discriminators. Cast to float64 for
        # cross-dtype reproducibility (a sum at float32 has reduction-
        # order-dependent rounding; float64 reductions are stabler).
        promoted = detached.to(device="cpu", dtype=torch.float64)
        # Sum is well-defined for empty above; we already guarded.
        h.update(str(float(promoted.sum().item())).encode())
        if numel > 1:
            h.update(str(float(promoted.std().item())).encode())
    return h.hexdigest()[:HEX_PREFIX_LEN]


def _op_hex(
    op_id: str,
    op_version: str,
    params_json: str,
    input_hexes: tuple[str, ...],
    hasher_factory: Callable,
) -> str:
    """Deterministic hex for an op call from its identity + inputs."""
    h = hasher_factory()
    h.update(op_id.encode())
    h.update(b"\0")
    h.update(op_version.encode())
    h.update(b"\0")
    h.update(params_json.encode())
    h.update(b"\0")
    for in_hex in input_hexes:
        h.update(in_hex.encode())
        h.update(b"\0")
    return h.hexdigest()[:HEX_PREFIX_LEN]


def _resolve_tensor_hex(t: torch.Tensor, ctx: ProvenanceRegistry) -> str:
    """Get hex for an input tensor: look up in context, else hash it.

    Dispatches on `ctx.hash_mode`: "full" uses byte-level cryptographic
    hashing (default); "sketch" uses the O(1)-bytes sketch hash.

    When `cache_tensors` is on, user-input tensors (those whose hex
    was computed here, not chained from an upstream op) are also stashed
    in the cache keyed by their content hex. This is what makes
    `replay()` work for chains where a recorded op takes a user-supplied
    tensor — without it, replay can't find that tensor's hex in the
    shadow cache.
    """
    upstream = ctx._lookup_tensor_hex(t)
    if upstream is not None:
        return upstream
    if ctx.hash_mode == "sketch":
        h = _tensor_sketch_hex(t, ctx._hasher_factory)
    else:
        h = _tensor_content_hex(t, ctx._hasher_factory)
    ctx._register_tensor_hex(t, h)
    if ctx._cache_tensors and h not in ctx._user_input_cache:
        # Cache user inputs so replay can find them. Goes into the
        # separate _user_input_cache (not bounded by max_cache_size)
        # so eviction policy on outputs is unchanged. Also mirrors
        # to disk when cache_to_disk is configured.
        ctx._user_input_cache[h] = t.detach()
        if ctx._disk_cache_dir is not None:
            torch.save(
                t.detach().cpu(),
                ctx._disk_path_for(h),
            )
            ctx._disk_cache_keys.add(h)
    return h


# ============================================================
# Decorator
# ============================================================


def with_provenance(op_id: str, op_version: str = "0.1") -> Callable:
    """Decorator: emit a provenance node when the wrapped op runs inside
    a `record()` context. Outside of recording, the decorator is
    transparent.

    All arguments — positional or keyword — are bound by parameter name
    via `inspect.signature`. This makes call signatures fully
    reconstructible for `Registry.replay`, at the cost of one hex
    invalidation if you rename a parameter (bump op_version when that
    happens).

    Restrictions:
      - The function must not use `*args` or `**kwargs` in its signature
        (named parameters only). Our math primitives don't.
    """
    def decorator(fn: Callable) -> Callable:
        sig = inspect.signature(fn)
        # *args / **kwargs would not be addressable by name, so the
        # hex would lose information and replay couldn't reconstruct
        # the call. Reject at decoration time — fail fast, not later.
        for p in sig.parameters.values():
            if p.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                raise TypeError(
                    f"@with_provenance({op_id!r}): function {fn.__qualname__} "
                    f"uses {p.kind.name.lower()} parameter {p.name!r}; "
                    f"only named parameters are supported"
                )
        if op_id in OP_REGISTRY and OP_REGISTRY[op_id][0] is not fn:
            raise ValueError(
                f"op_id {op_id!r} already registered to a different function"
            )
        OP_REGISTRY[op_id] = (fn, op_version)

        @wraps(fn)
        def wrapper(*args, **kwargs):
            ctx = _current_context()
            if ctx is None:
                return fn(*args, **kwargs)

            # Bind all arguments to parameter names — this normalizes
            # `truncated_svd(M, 3)` and `truncated_svd(M, r=3)` to the
            # same call signature, and gives us the param name for
            # every input so replay can reconstruct the call.
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()

            # Separate tensor inputs (which get hex IDs) from non-tensor
            # parameters (which get serialized to JSON). Tuples/lists
            # of tensors are unpacked into per-element hexes so that
            # e.g. a FixedRankPoint = (U, S, Vt) input contributes
            # three independent content hashes rather than being
            # str()-ed into the params blob.
            tensor_inputs: dict[str, str] = {}
            non_tensor_params: dict[str, Any] = {}
            for name, value in bound.arguments.items():
                if isinstance(value, torch.Tensor):
                    tensor_inputs[name] = _resolve_tensor_hex(value, ctx)
                elif (
                    isinstance(value, (tuple, list))
                    and len(value) > 0
                    and all(isinstance(x, torch.Tensor) for x in value)
                ):
                    for i, t in enumerate(value):
                        tensor_inputs[f"{name}[{i}]"] = _resolve_tensor_hex(t, ctx)
                else:
                    non_tensor_params[name] = _canonicalize_value(value)

            # Deterministic input_hexes: sorted by name, "name=hex" each.
            input_hexes = tuple(
                f"{name}={hex_id}"
                for name, hex_id in sorted(tensor_inputs.items())
            )
            params_json = json.dumps(
                non_tensor_params, sort_keys=True, default=str,
            )
            this_hex = _op_hex(
                op_id, op_version, params_json, input_hexes,
                ctx._hasher_factory,
            )

            # Substitution: if user has overridden this op-call's hex,
            # return their value instead of computing.
            if this_hex in ctx._substitutions:
                output = ctx._substitutions[this_hex]
            else:
                output = fn(*args, **kwargs)

            # Extract output shape/dtype metadata.
            if isinstance(output, torch.Tensor):
                shapes = (tuple(output.shape),)
                dtypes = (str(output.dtype),)
            elif isinstance(output, tuple):
                shapes = tuple(
                    tuple(o.shape) if isinstance(o, torch.Tensor) else ()
                    for o in output
                )
                dtypes = tuple(
                    str(o.dtype) if isinstance(o, torch.Tensor) else type(o).__name__
                    for o in output
                )
            else:
                shapes = ()
                dtypes = (type(output).__name__,)

            node = ProvenanceNode(
                hex=this_hex,
                op_id=op_id,
                op_version=op_version,
                params=params_json,
                input_hexes=input_hexes,
                output_shape=shapes,
                output_dtype=dtypes,
            )
            ctx._register(node, output)

            # Tag each output tensor with its hex so downstream ops chain.
            if isinstance(output, torch.Tensor):
                ctx._register_tensor_hex(output, this_hex)
            elif isinstance(output, tuple):
                for i, o in enumerate(output):
                    if isinstance(o, torch.Tensor):
                        ctx._register_tensor_hex(o, f"{this_hex}:{i}")

            # Fire any hooks registered for this op_id.
            for hook in ctx._hooks.get(op_id, ()):
                hook(node, output)

            return output

        return wrapper

    return decorator
