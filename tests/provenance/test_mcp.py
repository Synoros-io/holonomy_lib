# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Synoros

"""Smoke tests for the MCP server module.

Skipped when the `mcp` SDK isn't installed (the typical dev venv).
Run with `pip install 'holonomy-lib[mcp]'` to enable.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest
import torch


def _seeded(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


pytest.importorskip("mcp")  # skip the whole module if MCP SDK absent


from holonomy_lib import provenance
from holonomy_lib.algebra import truncated_svd
from holonomy_lib.spectral import laplacian


def _build_fixture_registry() -> provenance.ProvenanceRegistry:
    A = torch.randn(1, 4, 4, dtype=torch.float64, generator=_seeded(0))
    A = (A + A.mT).abs() + torch.eye(4, dtype=torch.float64).unsqueeze(dim=0)
    with provenance.record(cache_tensors=True) as reg:
        L = laplacian.combinatorial(A)
        truncated_svd(L, r=2, mode="exact")
    return reg


class TestBuildServer:
    def test_build_server_includes_v03_nav_tools(self):
        """The v0.3.0 MCP nav-tool names (list_ops, where, node_info,
        ancestors, get_tensor_summary) must remain registered for
        backward compat with existing MCP clients."""
        from holonomy_lib.provenance.mcp import build_server
        reg = _build_fixture_registry()
        server = build_server(reg)
        if hasattr(server, "_tool_manager"):
            names = set(server._tool_manager._tools.keys())
        else:
            import asyncio
            tools = asyncio.run(server.list_tools())
            names = {t.name for t in tools}
        for expected in [
            "list_ops", "where", "node_info", "ancestors",
            "get_tensor_summary",
        ]:
            assert expected in names, f"missing tool: {expected}"

    def test_build_server_includes_v04_inspection_tools(self):
        """The new v0.4 inspection tools are registered alongside the
        v0.3 ones."""
        from holonomy_lib.provenance.mcp import build_server
        reg = _build_fixture_registry()
        server = build_server(reg)
        if not hasattr(server, "_tool_manager"):
            pytest.skip("MCP SDK version doesn't expose tool internals")
        names = set(server._tool_manager._tools.keys())
        for expected in [
            "tensor_slice", "tensor_per_batch_summary",
            "tensor_eigenvalues", "tensor_singular_values",
            "tensor_norm", "tensor_compare",
            "replay_with", "op_docstring",
        ]:
            assert expected in names, f"missing tool: {expected}"

    def test_get_tensor_summary_returns_stats(self):
        """get_tensor_summary returns shape/dtype/stats for a cached node."""
        from holonomy_lib.provenance.mcp import build_server
        reg = _build_fixture_registry()
        server = build_server(reg)
        lap_node = reg.where(
            op_id="holonomy_lib.spectral.laplacian.combinatorial",
        )[0]
        if hasattr(server, "_tool_manager"):
            tool = server._tool_manager._tools["get_tensor_summary"]
            fn = tool.fn if hasattr(tool, "fn") else tool.callable
        else:
            pytest.skip("MCP SDK version doesn't expose tool internals")
        result = fn(hex_id=lap_node.hex)
        assert result["hex"] == lap_node.hex
        assert "shape" in result
        assert "mean" in result
        assert "std" in result

    def test_list_ops_returns_distinct_op_ids(self):
        """Through the MCP wrapper, list-returning tools are wrapped
        in {"results": [...]} so FastMCP serializes a single JSON
        content item rather than one per element."""
        from holonomy_lib.provenance.mcp import build_server
        reg = _build_fixture_registry()
        server = build_server(reg)
        if not hasattr(server, "_tool_manager"):
            pytest.skip("MCP SDK version doesn't expose tool internals")
        tool = server._tool_manager._tools["list_ops"]
        fn = tool.fn if hasattr(tool, "fn") else tool.callable
        result = fn()
        assert isinstance(result, dict) and "results" in result, (
            f"expected dict with 'results' key, got {result!r}"
        )
        ops = result["results"]
        assert "holonomy_lib.spectral.laplacian.combinatorial" in ops
        assert "holonomy_lib.algebra.linear.truncated_svd" in ops


class TestLoadRegistryFromEnv:
    def test_env_unset_raises(self):
        from holonomy_lib.provenance.mcp import _load_registry_from_env
        # Make sure the env var isn't set.
        old = os.environ.pop("HOLONOMY_PROVENANCE_REGISTRY", None)
        try:
            with pytest.raises(RuntimeError, match="HOLONOMY_PROVENANCE_REGISTRY"):
                _load_registry_from_env()
        finally:
            if old is not None:
                os.environ["HOLONOMY_PROVENANCE_REGISTRY"] = old

    def test_env_missing_file_raises(self):
        from holonomy_lib.provenance.mcp import _load_registry_from_env
        with tempfile.TemporaryDirectory() as d:
            os.environ["HOLONOMY_PROVENANCE_REGISTRY"] = str(
                Path(d) / "nonexistent.json"
            )
            try:
                with pytest.raises(FileNotFoundError):
                    _load_registry_from_env()
            finally:
                os.environ.pop("HOLONOMY_PROVENANCE_REGISTRY", None)

    def test_env_pointing_at_saved_registry_loads(self):
        from holonomy_lib.provenance.mcp import _load_registry_from_env
        reg = _build_fixture_registry()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "reg.json"
            reg.save(path)
            os.environ["HOLONOMY_PROVENANCE_REGISTRY"] = str(path)
            try:
                loaded = _load_registry_from_env()
                assert len(loaded) == len(reg)
            finally:
                os.environ.pop("HOLONOMY_PROVENANCE_REGISTRY", None)
