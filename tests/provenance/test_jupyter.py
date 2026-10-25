# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Synoros

"""Smoke tests for the Jupyter cell magic.

Skipped when IPython isn't installed. The classic Jupyter SDK is
present in many dev environments because it ships as a transitive
dep of ipykernel, but we don't assume it.
"""

from __future__ import annotations

import pytest

pytest.importorskip("IPython")  # skip the whole module if IPython absent


import torch

from holonomy_lib import provenance


def _seeded(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


class TestMagicRegistration:
    def test_load_ipython_extension_registers_magics(self):
        """Smoke: extension load registers the ProvenanceMagics class."""
        from IPython.testing.globalipapp import get_ipython

        from holonomy_lib.provenance.jupyter import load_ipython_extension

        ip = get_ipython()
        # Defensive: a fresh shell might be None in test envs.
        if ip is None:
            pytest.skip("IPython global app unavailable")
        load_ipython_extension(ip)
        assert "record_provenance" in ip.magics_manager.magics["cell"]

    def test_cell_magic_records_and_exposes_registry(self):
        """Running the cell magic puts a ProvenanceRegistry in the
        user namespace under the default name `_prov`.
        """
        from IPython.testing.globalipapp import get_ipython

        from holonomy_lib.provenance.jupyter import load_ipython_extension

        ip = get_ipython()
        if ip is None:
            pytest.skip("IPython global app unavailable")
        load_ipython_extension(ip)

        # Pre-populate the user namespace so the cell can reference it.
        ip.user_ns["A"] = (
            torch.randn(1, 4, 4, dtype=torch.float64, generator=_seeded(0))
             .abs() + torch.eye(4, dtype=torch.float64).unsqueeze(0)
        )
        ip.user_ns["A"] = (ip.user_ns["A"] + ip.user_ns["A"].mT) * 0.5
        ip.user_ns["laplacian"] = __import__(
            "holonomy_lib.spectral", fromlist=["laplacian"],
        ).laplacian

        cell = "L = laplacian.combinatorial(A)"
        ip.run_cell_magic("record_provenance", "", cell)
        assert "_prov" in ip.user_ns
        reg = ip.user_ns["_prov"]
        assert isinstance(reg, provenance.ProvenanceRegistry)
        assert len(reg) == 1
        node = next(iter(reg))
        assert node.op_id == "holonomy_lib.spectral.laplacian.combinatorial"

    def test_cell_magic_respects_custom_var_name(self):
        """The line argument names the variable the registry binds to."""
        from IPython.testing.globalipapp import get_ipython

        from holonomy_lib.provenance.jupyter import load_ipython_extension

        ip = get_ipython()
        if ip is None:
            pytest.skip("IPython global app unavailable")
        load_ipython_extension(ip)

        ip.user_ns["A"] = (
            torch.randn(1, 4, 4, dtype=torch.float64, generator=_seeded(1))
             .abs() + torch.eye(4, dtype=torch.float64).unsqueeze(0)
        )
        ip.user_ns["A"] = (ip.user_ns["A"] + ip.user_ns["A"].mT) * 0.5
        ip.user_ns["laplacian"] = __import__(
            "holonomy_lib.spectral", fromlist=["laplacian"],
        ).laplacian

        cell = "L = laplacian.combinatorial(A)"
        ip.run_cell_magic("record_provenance", "my_reg", cell)
        assert "my_reg" in ip.user_ns
        assert isinstance(ip.user_ns["my_reg"], provenance.ProvenanceRegistry)
