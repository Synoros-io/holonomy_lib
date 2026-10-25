# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Synoros

"""Jupyter cell magic for inline provenance recording.

Usage in a Jupyter notebook:

    %load_ext holonomy_lib.provenance.jupyter

    %%record_provenance
    L = laplacian.combinatorial(A)
    U, S, Vt = truncated_svd(L, r=3)

After the cell runs:
  - The registry is exposed in the user namespace as `_prov` (override
    via `%%record_provenance my_reg`).
  - A Mermaid flowchart of the DAG is displayed below the cell output
    (JupyterLab renders this inline; classic Jupyter shows the
    fenced markdown).

IPython is an optional dependency declared in the `[jupyter]` extras
group. Importing this module without IPython installed raises a
clear ImportError that points to the install.
"""

from __future__ import annotations

from typing import Any

from holonomy_lib.provenance.protocol import record

try:
    from IPython.core.magic import Magics, cell_magic, magics_class
    from IPython.display import Markdown, display
except ImportError as e:
    raise ImportError(
        "holonomy_lib.provenance.jupyter requires IPython. Install "
        "it via `pip install 'holonomy-lib[jupyter]'`."
    ) from e


@magics_class
class ProvenanceMagics(Magics):
    """Cell magics for inline provenance recording in Jupyter."""

    @cell_magic
    def record_provenance(self, line: str, cell: str) -> Any:
        """Run a cell inside a `record()` context and render the DAG.

        Args (on the magic line):
          A single token: the user-namespace variable name to bind
          the resulting registry to. Defaults to `_prov`.

        Example:
            %%record_provenance my_registry
            L = laplacian.combinatorial(A)

        After execution, `my_registry` (or `_prov`) holds the
        ProvenanceRegistry and the Mermaid DAG is displayed below.
        """
        var_name = line.strip() or "_prov"
        # Capture the user namespace from the active shell. The cell's
        # `exec` runs in that namespace so user-defined symbols (A in
        # the example above) are visible.
        user_ns = self.shell.user_ns if self.shell is not None else {}
        with record() as reg:
            # `exec` inside the user namespace mirrors how Jupyter
            # normally evaluates cell content. We do NOT capture
            # stdout here — print() calls inside the cell go to the
            # cell's output area as usual.
            exec(cell, user_ns)
        user_ns[var_name] = reg
        # Render the DAG. Mermaid in a fenced markdown block is
        # supported by JupyterLab's default renderer.
        display(Markdown(f"```mermaid\n{reg.to_mermaid()}\n```"))


def load_ipython_extension(ipython) -> None:
    """Entry point for `%load_ext holonomy_lib.provenance.jupyter`."""
    ipython.register_magics(ProvenanceMagics)
