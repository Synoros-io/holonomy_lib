# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Synoros

"""Agent API: Python functions an LLM can call as native tools.

The pattern:
  1. Define a Python function for each inspection / intervention an
     agent might want to perform against a ProvenanceRegistry.
  2. Decorate with `@agent_tool` to register it.
  3. Call `to_anthropic_schema()` or `to_openai_schema()` to get the
     JSON-schema list the provider APIs expect.

The same registered functions are usable three ways:
  - Native LLM tool-use (Anthropic `tools=[...]`, OpenAI `functions=[...]`).
  - MCP transport (the `holonomy_lib.provenance.mcp` module wraps them).
  - Direct Python calls (just import and invoke).

Designing the schemas this way means the underlying tool surface is
provider-agnostic; if the MCP ecosystem fades or a new provider
appears, the same `@agent_tool` functions still work — only the
transport-specific schema dumper changes.

Most tools take an explicit `registry: ProvenanceRegistry` so callers
can bind the registry however they like (closure, global, dependency
injection). Transport adapters typically pre-bind the registry and
expose the tools with that argument stripped.
"""

from __future__ import annotations

import inspect
import typing
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Union, get_args, get_origin

import torch

from holonomy_lib.provenance.protocol import (
    OP_REGISTRY,
    ProvenanceRegistry,
)

# ============================================================
# Registry of decorated tools
# ============================================================


@dataclass
class ToolSpec:
    """Metadata about an @agent_tool-decorated function.

    Captured at decoration time so provider-specific schema dumpers can
    consume it without re-inspecting the function each time.

    `type_hints` carries the resolved type hints (after `__future__
    annotations` string un-stringification); use it for schema work
    rather than the raw `signature.parameters[*].annotation` so future-
    annotation modules work transparently.
    """
    name: str
    fn: Callable[..., Any]
    description: str
    signature: inspect.Signature
    docstring: str
    type_hints: dict[str, Any]


_AGENT_TOOLS: dict[str, ToolSpec] = {}


def agent_tool(
    *,
    description: str = "",
    name: str | None = None,
) -> Callable[[Callable], Callable]:
    """Decorator: register a function as an agent-callable tool.

    Args:
      description: short (1-2 sentence) human-readable description that
        gets fed to the LLM as the tool's purpose. Defaults to the
        first line of the function's docstring.
      name: registry key. Defaults to the function's `__name__`.

    The function must have type-annotated parameters. Each parameter
    becomes a JSON-schema field in the emitted tool definition; the
    function's docstring becomes the schema's `description`.
    """
    def decorator(fn: Callable) -> Callable:
        tool_name = name if name is not None else fn.__name__
        doc = (fn.__doc__ or "").strip()
        desc = description.strip() or (doc.split("\n", 1)[0] if doc else "")
        try:
            # Resolve `from __future__ import annotations` strings.
            # include_extras=True keeps Annotated[] markers if we
            # ever add them.
            hints = typing.get_type_hints(fn, include_extras=True)
        except Exception:
            # Bad annotation references would normally fail loudly;
            # fall back to raw signature annotations so a buggy hint
            # doesn't block registration of an otherwise-valid tool.
            hints = {}
        spec = ToolSpec(
            name=tool_name,
            fn=fn,
            description=desc,
            signature=inspect.signature(fn),
            docstring=doc,
            type_hints=hints,
        )
        _AGENT_TOOLS[tool_name] = spec
        return fn
    return decorator


def list_tools() -> list[ToolSpec]:
    """Return all registered tools in insertion order."""
    return list(_AGENT_TOOLS.values())


def get_tool(name: str) -> ToolSpec:
    """Look up a tool by name. Raises KeyError if unknown."""
    return _AGENT_TOOLS[name]


# ============================================================
# Type → JSON-schema mapping
# ============================================================


def _python_type_to_json_schema(annotation: Any) -> dict[str, Any]:
    """Convert a Python type annotation to a JSON-schema fragment.

    Handles the common types our tool signatures use: str, int, float,
    bool, list[T], dict, Optional[T], Union[T, None]. Falls back to
    `{"type": "string"}` for anything unrecognized (the LLM will see
    a string and the tool will validate at call time).
    """
    if annotation is inspect.Parameter.empty:
        return {"type": "string"}
    if annotation is str:
        return {"type": "string"}
    if annotation is int:
        return {"type": "integer"}
    if annotation is float:
        return {"type": "number"}
    if annotation is bool:
        return {"type": "boolean"}
    origin = get_origin(annotation)
    if origin is list:
        args = get_args(annotation)
        item_schema = (
            _python_type_to_json_schema(args[0]) if args else {"type": "string"}
        )
        return {"type": "array", "items": item_schema}
    if origin is dict:
        return {"type": "object"}
    if origin is Union:
        args = [a for a in get_args(annotation) if a is not type(None)]
        if len(args) == 1:
            schema = _python_type_to_json_schema(args[0])
            return schema
        # Multi-type unions get widened to string for now; tool surface
        # should avoid these.
        return {"type": "string"}
    return {"type": "string"}


def _tool_to_json_schema_params(spec: ToolSpec) -> dict[str, Any]:
    """Build the JSON-schema object for a tool's parameters.

    Skips the `registry: ProvenanceRegistry` parameter when present;
    transport adapters pre-bind it, so the LLM-facing schema must not
    include it.
    """
    properties: dict[str, Any] = {}
    required: list[str] = []
    for name, param in spec.signature.parameters.items():
        if name == "registry":
            continue
        # Skip parameters that the LLM has no business setting (the
        # `registry` arg is the canonical example; in the future we
        # might add a marker for others).
        resolved = spec.type_hints.get(name, param.annotation)
        if resolved is ProvenanceRegistry:
            continue
        properties[name] = _python_type_to_json_schema(resolved)
        if param.default is inspect.Parameter.empty:
            required.append(name)
    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


# ============================================================
# Provider-specific schema dumpers
# ============================================================


def to_anthropic_schema() -> list[dict[str, Any]]:
    """Emit the registered tools as an Anthropic-API `tools=[...]` list.

    Each entry has the shape:
      {"name": str, "description": str, "input_schema": {...}}
    """
    out: list[dict[str, Any]] = []
    for spec in _AGENT_TOOLS.values():
        out.append({
            "name": spec.name,
            "description": spec.description,
            "input_schema": _tool_to_json_schema_params(spec),
        })
    return out


def to_openai_schema() -> list[dict[str, Any]]:
    """Emit the registered tools as an OpenAI-API `tools=[...]` list.

    Each entry has the shape:
      {"type": "function", "function": {"name": str, "description": str,
                                          "parameters": {...}}}
    """
    out: list[dict[str, Any]] = []
    for spec in _AGENT_TOOLS.values():
        out.append({
            "type": "function",
            "function": {
                "name": spec.name,
                "description": spec.description,
                "parameters": _tool_to_json_schema_params(spec),
            },
        })
    return out


# ============================================================
# Tool constants
# ============================================================

# Max element count for `tensor_slice` to return the raw values inline.
# Above this, the tool returns only shape + summary stats and asks the
# caller to take a smaller slice. 256 = a 16x16 matrix's worth — big
# enough to inspect typical neighborhoods, small enough to fit a single
# MCP / tool-use response without bloating the agent's context window.
# Cataloged as 🔬 experimentally-set in notes/magic_numbers.md.
TENSOR_SLICE_INLINE_LIMIT: int = 256

# Default `k` for `tensor_eigenvalues` / `tensor_singular_values`.
# Ten covers typical "spectral overview" questions; callers can raise
# it explicitly when they want the full spectrum. Cataloged as 🔬.
TENSOR_SPECTRAL_DEFAULT_K: int = 10

# Python's slice() constructor takes (start, stop, step) — three
# components. Pinned here so the audit doesn't flag the bare literal
# in the slice-expression parser. Cataloged as ✅ derived since it's
# fixed by the Python language, not a tuning parameter.
PYTHON_SLICE_ARITY: int = 3


# ============================================================
# Slice expression parser
# ============================================================


# ============================================================
# Registry-navigation tools (mirror of the v0.3.0 MCP surface,
# moved here so agent.py is the canonical tool inventory)
# ============================================================


@agent_tool(description=(
    "Return the distinct op_ids present in the registry. Use this "
    "for orientation — what kinds of operations were recorded."
))
def list_ops(registry: ProvenanceRegistry) -> list[str]:
    """Sorted distinct op_ids."""
    return sorted({n.op_id for n in registry})


@agent_tool(description=(
    "Return nodes matching op_id and/or op_version filters. Each "
    "entry is a node-info dict."
))
def where(
    registry: ProvenanceRegistry,
    op_id: str | None = None,
    op_version: str | None = None,
) -> list[dict[str, Any]]:
    """Filter nodes by op."""
    return [
        n.to_dict()
        for n in registry.where(op_id=op_id, op_version=op_version)
    ]


@agent_tool(description=(
    "Return the full ProvenanceNode dict (op_id, op_version, params, "
    "input_hexes, output_shape, output_dtype) for a given hex."
))
def node_info(registry: ProvenanceRegistry, hex_id: str) -> dict[str, Any]:
    """Look up a node by hex."""
    if hex_id not in registry:
        return {"error": "hex not in registry", "hex_id": hex_id}
    return registry[hex_id].to_dict()


@agent_tool(description=(
    "Return the sorted set of all transitive upstream hexes — the "
    "Merkle DAG above the given node."
))
def ancestors(registry: ProvenanceRegistry, hex_id: str) -> dict[str, Any]:
    """Transitive ancestors of a node."""
    if hex_id not in registry:
        return {"error": "hex not in registry", "hex_id": hex_id}
    return {"hex": hex_id, "ancestors": sorted(registry.ancestors(hex_id))}


@agent_tool(description=(
    "Global mean/std/min/max for a cached tensor (over ALL elements). "
    "For batched tensors, prefer tensor_per_batch_summary — global "
    "stats average over the batch dim and can hide per-batch anomalies."
))
def get_tensor_summary(
    registry: ProvenanceRegistry, hex_id: str,
) -> dict[str, Any]:
    """Global tensor stats. Mirror of the v0.3.0 MCP tool, kept for
    quick scalar-overview use; prefer per-batch summaries for batched
    tensors."""
    t = registry.get_tensor(hex_id)
    if t is None:
        return {"error": "tensor not cached", "hex_id": hex_id}
    promoted = t.to(torch.float64)
    return {
        "hex": hex_id,
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "device": str(t.device),
        "numel": t.numel(),
        "mean": float(promoted.mean().item()),
        "std": (
            float(promoted.std().item()) if t.numel() > 1 else 0.0
        ),
        "min": float(promoted.min().item()),
        "max": float(promoted.max().item()),
    }


def _parse_index_expr(expr: str) -> int | slice | tuple:
    """Parse a numpy-style index expression into a Python index value.

    Accepts:
      - "0"           → 0
      - "-1"          → -1
      - "2:5"         → slice(2, 5, None)
      - "::2"         → slice(None, None, 2)
      - ":"           → slice(None, None, None)
      - "0, 1"        → (0, 1)
      - ":, 0"        → (slice(None), 0)
      - "0, 1:3, ::2" → (0, slice(1, 3), slice(None, None, 2))

    Rejects anything with characters outside `0-9-:, `. This is a tool
    surface exposed to LLMs; safety > expressiveness. For richer
    indexing, drop to direct Python.
    """
    if not isinstance(expr, str):
        raise TypeError(f"index expression must be a string, got {type(expr).__name__}")
    cleaned = expr.replace(" ", "")
    allowed = set("0123456789-:,")
    bad = set(cleaned) - allowed
    if bad:
        raise ValueError(
            f"index expression {expr!r} contains disallowed characters "
            f"{sorted(bad)!r}; only digits, '-', ':', ',', and spaces "
            f"are accepted"
        )
    # Tolerate a single trailing comma — `t[0,]` is valid Python and
    # LLM-generated multi-dim expressions often carry one. The interior-
    # empty-component check still fires for `"0,,1"`.
    if cleaned.endswith(","):
        cleaned = cleaned[:-1]
    parts = cleaned.split(",")
    result: list[int | slice] = []
    for p in parts:
        if not p:
            raise ValueError(
                f"index expression {expr!r} has an empty component"
            )
        if ":" in p:
            slice_args: list[int | None] = []
            for s in p.split(":"):
                slice_args.append(int(s) if s else None)
            # Python slice() takes (start, stop, step); pad with None
            # for omitted components, reject longer.
            while len(slice_args) < PYTHON_SLICE_ARITY:
                slice_args.append(None)
            if len(slice_args) > PYTHON_SLICE_ARITY:
                raise ValueError(
                    f"index expression {expr!r} has a slice with too "
                    f"many components"
                )
            result.append(slice(*slice_args))
        else:
            result.append(int(p))
    return tuple(result) if len(result) > 1 else result[0]


# ============================================================
# Inspection tools
# ============================================================


@agent_tool(description=(
    "Return a slice of a cached tensor. Index expression uses numpy "
    "syntax (e.g. '0', ':, 0', '2:5', '0, 1:3'). Returns the raw "
    "values inline if the slice has at most 256 elements; otherwise "
    "returns shape + summary stats and asks for a smaller slice."
))
def tensor_slice(
    registry: ProvenanceRegistry,
    hex_id: str,
    expr: str = ":",
) -> dict[str, Any]:
    """Slice a cached tensor by hex.

    Args:
      hex: the cached tensor's content hash.
      expr: numpy-style index expression. Examples: "0" (scalar index),
        ":, 0" (column 0 of a 2D matrix), "2:5" (slice), "0, 1:3"
        (multi-dim), ":" (everything — equivalent to no slicing).

    Returns a dict with keys `shape`, `dtype`, and either `values`
    (inline list) or `truncated=True` + `stats` summary.
    """
    t = registry.get_tensor(hex_id)
    if t is None:
        return {"error": "tensor not cached", "hex_id": hex_id}
    try:
        idx = _parse_index_expr(expr)
    except (ValueError, TypeError) as e:
        return {"error": f"bad index expression: {e}"}
    try:
        sliced = t[idx] if idx != slice(None, None, None) else t
    except (IndexError, TypeError) as e:
        return {"error": f"slice failed: {e}", "expr": expr, "tensor_shape": list(t.shape)}
    out: dict[str, Any] = {
        "shape": list(sliced.shape) if sliced.ndim > 0 else [],
        "dtype": str(sliced.dtype),
    }
    if sliced.numel() <= TENSOR_SLICE_INLINE_LIMIT:
        out["values"] = (
            sliced.tolist() if sliced.ndim > 0 else float(sliced.item())
        )
    else:
        promoted = sliced.to(torch.float64)
        out["truncated"] = True
        out["stats"] = {
            "mean": float(promoted.mean().item()),
            "std": float(promoted.std().item()) if sliced.numel() > 1 else 0.0,
            "min": float(promoted.min().item()),
            "max": float(promoted.max().item()),
        }
    return out


@agent_tool(description=(
    "Per-batch mean/std/min/max for a cached tensor. Assumes dim 0 is "
    "the batch dim; reduces over the rest. Use this instead of the "
    "v0.3 get_tensor_summary for any (B, ...) tensor where you need "
    "to see which batch element differs."
))
def tensor_per_batch_summary(
    registry: ProvenanceRegistry,
    hex_id: str,
) -> dict[str, Any]:
    """Return one summary row per batch element, instead of collapsing
    to global stats. This is the tool that fixes the v0.3.0 "average
    hides the anomaly" wall.
    """
    t = registry.get_tensor(hex_id)
    if t is None:
        return {"error": "tensor not cached", "hex_id": hex_id}
    if t.ndim == 0:
        return {"shape": [], "value": float(t.item())}
    flat = t.reshape(t.shape[0], -1).to(torch.float64)
    means = flat.mean(dim=1).tolist()
    if flat.shape[1] > 1:
        stds = flat.std(dim=1).tolist()
    else:
        stds = [0.0] * flat.shape[0]
    mins = flat.min(dim=1).values.tolist()
    maxs = flat.max(dim=1).values.tolist()
    return {
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "per_batch": [
            {"index": i, "mean": means[i], "std": stds[i],
             "min": mins[i], "max": maxs[i]}
            for i in range(t.shape[0])
        ],
    }


@agent_tool(description=(
    "Top-k eigenvalues of a cached square matrix (or batched square "
    "matrices). Symmetrizes the input before computing — pass a "
    "symmetric matrix for the canonical answer. Returns per-batch "
    "lists for batched inputs."
))
def tensor_eigenvalues(
    registry: ProvenanceRegistry,
    hex_id: str,
    k: int = TENSOR_SPECTRAL_DEFAULT_K,
) -> dict[str, Any]:
    """Top-k eigenvalues via eigvalsh. The input is symmetrized as
    `(A + A.T) / 2` for numerical stability, so non-symmetric inputs
    receive their symmetric part's eigenvalues."""
    t = registry.get_tensor(hex_id)
    if t is None:
        return {"error": "tensor not cached", "hex_id": hex_id}
    if t.ndim < 2 or t.shape[-1] != t.shape[-2]:
        return {
            "error": "tensor is not square in last two dims",
            "shape": list(t.shape),
        }
    t64 = t.to(torch.float64)
    sym = (t64 + t64.transpose(-1, -2)) * 0.5
    eigvals = torch.linalg.eigvalsh(sym)
    # eigvalsh returns ascending; flip to descending and take top-k.
    eigvals_desc = torch.flip(eigvals, dims=[-1])
    actual_k = min(k, eigvals.shape[-1])
    top = eigvals_desc[..., :actual_k]
    return {
        "shape": list(t.shape),
        "k": actual_k,
        "values": top.tolist(),
    }


@agent_tool(description=(
    "Top-k singular values of a cached matrix (or batched matrices). "
    "Returns per-batch lists for batched inputs. Use this for spectral "
    "inspection of arbitrary-shaped matrices; use tensor_eigenvalues "
    "for square symmetric ones."
))
def tensor_singular_values(
    registry: ProvenanceRegistry,
    hex_id: str,
    k: int = TENSOR_SPECTRAL_DEFAULT_K,
) -> dict[str, Any]:
    """Top-k singular values via SVD (reduced form)."""
    t = registry.get_tensor(hex_id)
    if t is None:
        return {"error": "tensor not cached", "hex_id": hex_id}
    if t.ndim < 2:
        return {
            "error": "need at least 2 dims for SVD",
            "shape": list(t.shape),
        }
    t64 = t.to(torch.float64)
    _, s, _ = torch.linalg.svd(t64, full_matrices=False)
    actual_k = min(k, s.shape[-1])
    top = s[..., :actual_k]
    return {
        "shape": list(t.shape),
        "k": actual_k,
        "values": top.tolist(),
    }


@agent_tool(description=(
    "Norm of a cached tensor. Order: 'frobenius' (default, sqrt of "
    "sum of squares) or 'spectral' (largest singular value; matrices "
    "only). Batched inputs return per-batch values."
))
def tensor_norm(
    registry: ProvenanceRegistry,
    hex_id: str,
    order: str = "frobenius",
) -> dict[str, Any]:
    """Frobenius or spectral norm."""
    t = registry.get_tensor(hex_id)
    if t is None:
        return {"error": "tensor not cached", "hex_id": hex_id}
    t64 = t.to(torch.float64)
    if order == "frobenius":
        if t64.ndim <= 1:
            value = float(torch.linalg.norm(t64).item())
            return {"order": order, "shape": list(t.shape), "value": value}
        # For batched/matrix tensors, reduce over all-but-first dim.
        if t64.ndim == 2:
            return {
                "order": order, "shape": list(t.shape),
                "value": float(torch.linalg.norm(t64).item()),
            }
        flat = t64.reshape(t64.shape[0], -1)
        per_batch = torch.linalg.norm(flat, dim=1).tolist()
        return {"order": order, "shape": list(t.shape), "per_batch": per_batch}
    if order == "spectral":
        if t64.ndim < 2:
            return {"error": "spectral norm requires at least 2 dims"}
        per = torch.linalg.matrix_norm(t64, ord=2)
        if per.ndim == 0:
            return {
                "order": order, "shape": list(t.shape),
                "value": float(per.item()),
            }
        return {
            "order": order, "shape": list(t.shape),
            "per_batch": per.tolist(),
        }
    return {"error": f"unknown order {order!r}; use 'frobenius' or 'spectral'"}


@agent_tool(description=(
    "Compare two cached tensors of matching shape. Metric: 'max_abs' "
    "(largest |a - b|), 'frobenius' (Euclidean diff), or 'cosine' "
    "(angle between flattened vectors, range -1 to 1)."
))
def tensor_compare(
    registry: ProvenanceRegistry,
    hex_a: str,
    hex_b: str,
    metric: str = "max_abs",
) -> dict[str, Any]:
    """Pairwise comparison of two cached tensors."""
    a = registry.get_tensor(hex_a)
    b = registry.get_tensor(hex_b)
    if a is None:
        return {"error": "tensor a not cached", "hex_a": hex_a}
    if b is None:
        return {"error": "tensor b not cached", "hex_b": hex_b}
    if a.shape != b.shape:
        return {
            "error": "shape mismatch",
            "a_shape": list(a.shape), "b_shape": list(b.shape),
        }
    if a.device != b.device:
        # Cross-device subtraction would raise from torch.linalg; we
        # prefer an error dict that fits the rest of the tool surface.
        return {
            "error": "device mismatch",
            "a_device": str(a.device),
            "b_device": str(b.device),
        }
    a64 = a.to(torch.float64)
    b64 = b.to(torch.float64)
    if metric == "max_abs":
        return {"metric": metric, "value": float((a64 - b64).abs().max().item())}
    if metric == "frobenius":
        return {
            "metric": metric,
            "value": float(torch.linalg.norm(a64 - b64).item()),
        }
    if metric == "cosine":
        af = a64.flatten()
        bf = b64.flatten()
        norm_a = torch.linalg.norm(af)
        norm_b = torch.linalg.norm(bf)
        # cosine is undefined when either vector is exactly zero. The
        # 1e-9 numerical-floor convention guards against rounding noise,
        # but for exact zeros we'd silently return 0.0 (which an LLM
        # could misread as "orthogonal"). Surface the degeneracy.
        if float(norm_a.item()) == 0.0 or float(norm_b.item()) == 0.0:
            return {
                "metric": metric,
                "error": "cosine undefined: one or both input tensors are zero",
            }
        cos = torch.dot(af, bf) / (norm_a * norm_b + 1e-9)
        return {"metric": metric, "value": float(cos.item())}
    return {"error": f"unknown metric {metric!r}; use 'max_abs', 'frobenius', or 'cosine'"}


def _build_substitute(
    registry: ProvenanceRegistry, target_hex: str, recipe: dict[str, Any],
) -> torch.Tensor:
    """Construct a substitute tensor from a recipe dict.

    Separated from `replay_with` so the recipe parsing is testable in
    isolation. Raises ValueError on bad recipes.
    """
    kind = recipe.get("kind")
    if kind is None:
        raise ValueError("recipe must have a 'kind' field")

    if kind == "zeros_like":
        original = registry.get_tensor(target_hex)
        if original is None:
            raise ValueError(
                f"target {target_hex!r} not cached; can't infer shape"
            )
        return torch.zeros_like(original)

    if kind == "from_hex":
        src_hex = recipe.get("hex")
        if not src_hex:
            raise ValueError("from_hex recipe requires 'hex' field")
        src = registry.get_tensor(src_hex)
        if src is None:
            raise ValueError(f"source tensor {src_hex!r} not cached")
        return src

    if kind == "perturb":
        original = registry.get_tensor(target_hex)
        if original is None:
            raise ValueError(f"target {target_hex!r} not cached")
        if "noise_std" not in recipe:
            raise ValueError("perturb recipe requires 'noise_std' field")
        if "seed" not in recipe:
            raise ValueError("perturb recipe requires 'seed' field (for reproducibility)")
        std = float(recipe["noise_std"])
        seed = int(recipe["seed"])
        g = torch.Generator(device=original.device)
        g.manual_seed(seed)
        noise = torch.randn(
            original.shape, generator=g,
            dtype=original.dtype, device=original.device,
        ) * std
        return original + noise

    if kind == "scale":
        original = registry.get_tensor(target_hex)
        if original is None:
            raise ValueError(f"target {target_hex!r} not cached")
        if "factor" not in recipe:
            raise ValueError("scale recipe requires 'factor' field")
        return original * float(recipe["factor"])

    if kind == "swap_batch":
        original = registry.get_tensor(target_hex)
        if original is None:
            raise ValueError(f"target {target_hex!r} not cached")
        if original.ndim == 0:
            raise ValueError("swap_batch requires at least 1 dim")
        if "i" not in recipe or "j" not in recipe:
            raise ValueError("swap_batch recipe requires 'i' and 'j' fields")
        i = int(recipe["i"])
        j = int(recipe["j"])
        B = original.shape[0]
        if not (0 <= i < B and 0 <= j < B):
            raise ValueError(
                f"swap_batch indices out of range: i={i}, j={j}, B={B}"
            )
        # Identity swap: return a clone (avoids the redundant double-write
        # while keeping the "always returns a fresh tensor" contract).
        if i == j:
            return original.clone()
        result = original.clone()
        result[i] = original[j].clone()
        result[j] = original[i].clone()
        return result

    if kind == "literal":
        values = recipe.get("values")
        if values is None:
            raise ValueError("literal recipe requires 'values' field")
        # Match dtype to target when cached; default to float64 otherwise.
        original = registry.get_tensor(target_hex)
        dtype = original.dtype if original is not None else torch.float64
        device = original.device if original is not None else torch.device("cpu")
        return torch.tensor(values, dtype=dtype, device=device)

    raise ValueError(
        f"unknown recipe kind {kind!r}; valid: zeros_like, from_hex, "
        f"perturb, scale, swap_batch, literal"
    )


@agent_tool(description=(
    "Substitute a cached tensor and re-execute the downstream DAG. "
    "The recipe describes how to build the substitute. Available "
    "recipe kinds: "
    "{'kind': 'zeros_like'} — fill with zeros, same shape/dtype as "
    "the target. "
    "{'kind': 'from_hex', 'hex': '<other_hex>'} — substitute with "
    "another cached tensor. "
    "{'kind': 'perturb', 'noise_std': float, 'seed': int} — original + "
    "Gaussian noise N(0, noise_std). "
    "{'kind': 'scale', 'factor': float} — multiply the original by a "
    "scalar. "
    "{'kind': 'swap_batch', 'i': int, 'j': int} — swap two batch "
    "elements (dim 0). "
    "{'kind': 'literal', 'values': [...]} — explicit nested list of "
    "values; small tensors only."
))
def replay_with(
    registry: ProvenanceRegistry,
    target_hex: str,
    recipe: dict[str, Any],
) -> dict[str, Any]:
    """Build a substitute via the recipe DSL, then call registry.replay."""
    try:
        substitute = _build_substitute(registry, target_hex, recipe)
    except ValueError as e:
        return {"error": str(e), "recipe": recipe}
    # Pre-check substitute shape against the recorded node's output
    # shape. Single-output ops only — multi-output ops have a tuple of
    # output_shapes and we'd need the `:i` suffix to disambiguate;
    # skip the check there and let registry.replay surface any error.
    if target_hex in registry:
        recorded_shapes = registry[target_hex].output_shape
        if len(recorded_shapes) == 1:
            recorded_shape = tuple(recorded_shapes[0])
            if tuple(substitute.shape) != recorded_shape:
                return {
                    "error": (
                        f"substitute shape {list(substitute.shape)} does "
                        f"not match recorded shape {list(recorded_shape)} "
                        f"for {target_hex!r}"
                    ),
                    "recipe": recipe,
                }
    try:
        new_outputs = registry.replay({target_hex: substitute})
    except Exception as e:
        return {"error": f"replay failed: {type(e).__name__}: {e}"}
    summary_entries = []
    for h, t in new_outputs.items():
        promoted = t.to(torch.float64)
        summary_entries.append({
            "hex": h,
            "shape": list(t.shape),
            "mean": float(promoted.mean().item()),
            "std": (
                float(promoted.std().item()) if t.numel() > 1 else 0.0
            ),
        })
    return {
        "replayed_count": len(new_outputs),
        "new_outputs": summary_entries,
    }


@agent_tool(description=(
    "Return the docstring and signature of a registered op. Use this "
    "for discovery — e.g., to understand what 'holonomy_lib.spectral."
    "laplacian.symmetric_normalized' actually computes."
))
def op_docstring(op_id: str) -> dict[str, Any]:
    """Lookup the registered op's signature + docstring."""
    if op_id not in OP_REGISTRY:
        return {"error": "op_id not registered", "op_id": op_id}
    fn, version = OP_REGISTRY[op_id]
    return {
        "op_id": op_id,
        "op_version": version,
        "signature": str(inspect.signature(fn)),
        "docstring": (fn.__doc__ or "").strip(),
    }
