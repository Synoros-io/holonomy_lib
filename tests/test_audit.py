# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Synoros

"""Behavior tests for tools/audit.py.

The audit is the project's gate (`python tools/audit.py src/holonomy_lib --strict`).
Without tests, a regression in the audit silently breaks enforcement —
either letting magic numbers through (false negative) or blocking
legitimate code (false positive). These tests pin down the behaviors
the README/CLAUDE.md promise.
"""

from __future__ import annotations

import importlib.util
import sys
import textwrap
from pathlib import Path

import pytest

_AUDIT_PATH = Path(__file__).resolve().parents[1] / "tools" / "audit.py"
_spec = importlib.util.spec_from_file_location("holonomy_audit_tool", _AUDIT_PATH)
assert _spec is not None and _spec.loader is not None
_audit = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _audit
_spec.loader.exec_module(_audit)
load_catalog = _audit.load_catalog
scan_file = _audit.scan_file
scan_path = _audit.scan_path


def _write(tmp_path: Path, name: str, source: str) -> Path:
    """Write `source` to tmp_path/name and return the path."""
    p = tmp_path / name
    p.write_text(textwrap.dedent(source))
    return p


# --------------------------------------------------------------------
# Allowed literals — must never trip the audit
# --------------------------------------------------------------------


class TestAllowedLiterals:
    def test_zero_one_minus_one_pass(self, tmp_path):
        f = _write(tmp_path, "ok.py", """
            a = 0
            b = 1
            c = -1
            d = 0.0
            e = 1.0
            f = -1.0
        """)
        assert scan_file(f, set()) == []

    def test_halving_doubling_pass(self, tmp_path):
        f = _write(tmp_path, "ok.py", """
            half = 0.5
            two = 2
            two_f = 2.0
            neg2 = -2
        """)
        assert scan_file(f, set()) == []

    def test_numerical_floor_passes(self, tmp_path):
        f = _write(tmp_path, "ok.py", """
            EPS = 1e-9
            x = max(y, 1e-9)
        """)
        assert scan_file(f, set()) == []

    def test_thousand_and_kibi_pass(self, tmp_path):
        f = _write(tmp_path, "ok.py", """
            ms = secs * 1000
            kb = bytes / 1024
        """)
        assert scan_file(f, set()) == []


# --------------------------------------------------------------------
# Disallowed literals — must trip the audit
# --------------------------------------------------------------------


class TestDisallowedLiterals:
    def test_arbitrary_float_flagged(self, tmp_path):
        f = _write(tmp_path, "bad.py", """
            LEARNING_RATE = 0.07
        """)
        out = scan_file(f, set())
        assert len(out) == 1
        assert out[0].value == 0.07
        assert out[0].severity == "🔴"

    def test_arbitrary_int_flagged(self, tmp_path):
        f = _write(tmp_path, "bad.py", """
            BATCH_SIZE = 17
        """)
        out = scan_file(f, set())
        assert any(lit.value == 17 for lit in out)


# --------------------------------------------------------------------
# Assignment names do not exempt arbitrary values
# --------------------------------------------------------------------


class TestAssignmentNames:
    def test_loop_var_values_flagged(self, tmp_path):
        f = _write(tmp_path, "ok.py", """
            n = 17
            step = 33
        """)
        out = scan_file(f, set())
        assert [(lit.value, lit.severity) for lit in out] == [(17, "🔴"), (33, "🔴")]
        f.write_text("x=torch.tensor([0.123,0.456])")
        assert [lit.value for lit in scan_file(f, set())] == [0.123, 0.456]

    def test_display_var_values_flagged(self, tmp_path):
        f = _write(tmp_path, "ok.py", """
            log_every = 50
            verbose = 3
        """)
        out = scan_file(f, set())
        assert [(lit.value, lit.severity) for lit in out] == [(50, "🔴"), (3, "🔴")]


# --------------------------------------------------------------------
# Structural rank checks apply to their operand, not the whole line
# --------------------------------------------------------------------


class TestDerivedPatterns:
    def test_ndim_shape_assertion_marked_yellow(self, tmp_path):
        f = _write(tmp_path, "ok.py", """
            def f(X):
                assert X.ndim == 3
        """)
        out = scan_file(f, set())
        assert [(lit.value, lit.severity) for lit in out] == [(3, "🟡")]
        f.write_text("assert len(X.shape) == 3; gain = 0.123")
        out = scan_file(f, set())
        assert [(lit.value, lit.severity) for lit in out] == [(3, "🟡"), (0.123, "🔴")]

    def test_derived_neighbor_does_not_approve_literal(self, tmp_path):
        f = _write(tmp_path, "ok.py", """
            x = 1 / N * 7
        """)
        out = scan_file(f, set())
        assert [(lit.value, lit.severity) for lit in out] == [(7, "🔴")]
        f.write_text("gain=0.123 # math.sqrt")
        assert [(lit.value, lit.severity) for lit in scan_file(f, set())] == [
            (0.123, "🔴"),
        ]


# --------------------------------------------------------------------
# Catalog cross-reference — variable name in catalog → yellow
# --------------------------------------------------------------------


class TestCatalogCrossReference:
    def test_catalog_name_promotes_severity(self, tmp_path):
        f = _write(tmp_path, "src.py", """
            CATALOGED: float = 0.42
        """)
        out = scan_file(f, {"CATALOGED"})
        assert len(out) == 1
        assert out[0].severity == "🟡"

    def test_catalog_loader(self, tmp_path):
        cat = tmp_path / "magic_numbers.md"
        cat.write_text("`MY_CONST` is documented `OTHER_CONST` too\n")
        names = load_catalog(cat)
        assert names == {"MY_CONST", "OTHER_CONST"}


# --------------------------------------------------------------------
# AST coverage — function defaults, keyword args, attribute assigns
# --------------------------------------------------------------------


class TestAstCoverage:
    def test_keyword_argument_picks_up_arg_name_context(self, tmp_path):
        """Keyword arguments keep their names as context without an exemption."""
        f = _write(tmp_path, "src.py", """
            torch.clamp(t, min=0.123)
        """)
        out = scan_file(f, set())
        assert [(lit.value, lit.context_var, lit.severity) for lit in out] == [
            (0.123, "min", "🔴"),
        ]

    def test_function_default_picks_up_param_name(self, tmp_path):
        """Parameter names do not approve default values."""
        f = _write(tmp_path, "src.py", """
            def f(eps=0.123):
                return eps
        """)
        out = scan_file(f, set())
        assert [(lit.value, lit.context_var, lit.severity) for lit in out] == [
            (0.123, "eps", "🔴"),
        ]


# --------------------------------------------------------------------
# Path-level — directory excludes, missing files, syntax errors
# --------------------------------------------------------------------


class TestPathHandling:
    def test_scan_path_excludes_tests_dir(self, tmp_path):
        (tmp_path / "tests").mkdir()
        _write(tmp_path / "tests", "test_x.py", "FLAGGED = 0.99")
        _write(tmp_path, "src.py", "FLAGGED = 0.99")
        out = scan_path(tmp_path, set())
        # Only the top-level file is scanned; tests/ excluded.
        files = {lit.file_path.name for lit in out}
        assert files == {"src.py"}

    def test_scan_path_reports_failures(self, tmp_path, monkeypatch, capsys):
        bad = _write(tmp_path, "bad.py", "def broken(:")
        _write(tmp_path, "good.py", "FLAGGED = 0.42")
        errors: list[Exception] = []
        out = scan_path(tmp_path, set(), errors=errors)
        files = {lit.file_path.name for lit in out}
        assert files == {"good.py"}
        assert len(errors) == 1 and str(bad) in str(errors[0])
        assert _audit.main([str(bad), "--strict"]) == 1
        assert str(bad) in capsys.readouterr().err
        missing = tmp_path / "missing.py"
        assert _audit.main([str(missing), "--strict"]) == 1
        assert str(missing) in capsys.readouterr().err
        with pytest.raises(_audit.ScanError, match="missing.py"):
            scan_file(missing, set())

        read_text = Path.read_text

        def deny_bad(path, *args, **kwargs):
            if path == bad:
                raise PermissionError("denied")
            return read_text(path, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", deny_bad)
        assert _audit.main([str(bad), "--strict"]) == 1
        assert f"{bad}: denied" in capsys.readouterr().err

    def test_scan_path_excludes_audit_itself(self, tmp_path):
        assert scan_file(_AUDIT_PATH, set()) == []
        # Sharing the tool's basename does not exempt someone else's source.
        _write(tmp_path, "audit.py", "x = 0.99  # would be flagged otherwise")
        out = scan_path(tmp_path, set())
        assert [(lit.file_path.name, lit.value) for lit in out] == [("audit.py", 0.99)]


# --------------------------------------------------------------------
# Boolean literals — `True`/`False` must NOT be flagged (Python `bool` is `int`)
# --------------------------------------------------------------------


class TestBooleansIgnored:
    def test_true_false_not_flagged(self, tmp_path):
        f = _write(tmp_path, "ok.py", """
            flag = True
            other = False
        """)
        assert scan_file(f, set()) == []
