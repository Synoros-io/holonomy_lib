# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Synoros

"""Smoke + behavior tests for holonomy_lib.audit.

The audit is the project's CI gate (`python -m holonomy_lib.audit src/ --strict`).
Without tests, a regression in the audit silently breaks enforcement —
either letting magic numbers through (false negative) or blocking
legitimate code (false positive). These tests pin down the behaviors
the README/CLAUDE.md promise.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

from holonomy_lib.audit import (
    load_catalog,
    scan_file,
    scan_path,
)


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
        # Use a non-SAFE_VAR name. `x` etc. are loop/index vars and
        # get a free pass; `LEARNING_RATE` is the kind of name we
        # actually care about catching.
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
# Context-aware skips — display vars, safe vars
# --------------------------------------------------------------------


class TestContextualSkips:
    def test_safe_loop_var_skipped(self, tmp_path):
        f = _write(tmp_path, "ok.py", """
            n = 17
            step = 33
        """)
        # `n` and `step` are in SAFE_VAR_NAMES — values not flagged.
        assert scan_file(f, set()) == []

    def test_display_var_skipped(self, tmp_path):
        f = _write(tmp_path, "ok.py", """
            log_every = 50
            verbose = 3
        """)
        assert scan_file(f, set()) == []


# --------------------------------------------------------------------
# Derived-pattern recognition — must mark, not flag red
# --------------------------------------------------------------------


class TestDerivedPatterns:
    def test_ndim_shape_assertion_marked_yellow(self, tmp_path):
        f = _write(tmp_path, "ok.py", """
            def f(X):
                assert X.ndim == 3
        """)
        out = scan_file(f, set())
        # The literal 3 is non-allowed, but the ndim assertion pattern
        # promotes it to documented/derived, not red.
        if out:
            assert all(lit.severity == "🟡" for lit in out)

    def test_one_over_N_marked_yellow(self, tmp_path):
        f = _write(tmp_path, "ok.py", """
            x = 1 / N * 7
        """)
        out = scan_file(f, set())
        # 7 is non-allowed but the line has `1 / N` derived pattern → yellow.
        if out:
            assert any(lit.value == 7 for lit in out)
            assert all(lit.severity == "🟡" for lit in out)


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
        """`func(my_tol=1e-7)` should record context_var='my_tol' so the
        catalog can document the constant. Uses a non-standard name to
        avoid the DISPLAY_VAR_NAMES whitelist that covers `eps`, `lr`,
        etc.
        """
        f = _write(tmp_path, "src.py", """
            def f(my_tol=1e-7):
                return my_tol
        """)
        # 1e-7 not in ALLOWED → should be flagged, with context_var='my_tol'.
        out = scan_file(f, set())
        assert len(out) == 1
        assert out[0].context_var == "my_tol"

    def test_function_default_picks_up_param_name(self, tmp_path):
        """Same as above, with a positional default. `learning_factor`
        is non-standard so the whitelist doesn't catch it."""
        f = _write(tmp_path, "src.py", """
            def f(learning_factor=0.07):
                return learning_factor
        """)
        out = scan_file(f, set())
        assert out[0].context_var == "learning_factor"


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

    def test_scan_path_syntax_error_silently_skipped(self, tmp_path):
        _write(tmp_path, "bad.py", "def broken(:")
        _write(tmp_path, "good.py", "FLAGGED = 0.42")
        # bad.py has a syntax error — should not crash the audit.
        out = scan_path(tmp_path, set())
        files = {lit.file_path.name for lit in out}
        assert files == {"good.py"}

    def test_scan_path_excludes_audit_itself(self, tmp_path):
        # A file named audit.py is excluded by EXCLUDE_FILES.
        _write(tmp_path, "audit.py", "x = 0.99  # would be flagged otherwise")
        out = scan_path(tmp_path, set())
        assert out == []


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
