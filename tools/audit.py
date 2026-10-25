# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Synoros

"""Find undocumented numeric constants in source.

Per the "no magic numbers" rule (AGENTS.md invariant 2, CONVENTIONS §5),
every numerical constant in the architecture must be derived from substrate state,
corpus properties, universal invariants, or experimentally tuned with documented
procedure. This module enforces that rule by scanning Python source for numeric
literals that don't match a safe whitelist and cross-referencing them against the
`notes/magic_numbers.md` catalog.

Usage:
    python tools/audit.py <path>...      # scan files or directories
    python tools/audit.py --strict ...   # exit code 1 if any literal found

Output flags:
    🟢 Allowed   — literal is in the safe set (0, 1, -1, 0.5, 2.0, math constants)
    🟡 Documented — variable name matches an entry in magic_numbers.md
    🔴 Undocumented — literal not allowed, variable not in catalog (REVIEW)

Limitations:
    - Static AST analysis; can't detect dynamic derivation (e.g. `lr = 1/N`)
      will warn but the user should mark it as derived in the catalog
    - Pattern-match is heuristic; treat as a starting point, not ground truth
"""

from __future__ import annotations

import argparse
import ast
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

# Universal safe literals.
# Four categories, each justified explicitly:
#   1. Mathematical identities (0, 1, -1) — no debate
#   2. Halving/doubling identity 0.5 and 2 — fundamental in mathematical code:
#      symmetric/antisymmetric parts (Z ± Zᵀ)/2, midpoint mean (a + b)/2,
#      quadratic-form coefficients (½)xᵀAx, triangular numbers n(n+1)/2,
#      2π, 2-norm, second-derivative coefficients, etc. Treated as
#      identity-level since both are canonical halve/double constants; on
#      par with 0, 1, -1 for math libraries (added 2026-05-26 for holonomy-lib).
#      Note: 2 carries some false-negative risk for non-math code (e.g.,
#      "2 retries" magic numbers) — accepted tradeoff for math-library
#      readability.
#   3. Universal binary/SI unit conversions (1024 for KB↔MB↔GB,
#      1000 for s↔ms / RNG-stream decorrelation) — true universals
#   4. Conventional numerical floor 1e-9 — chosen safer than float32 eps
#      (~1.19e-7); used pervasively as anti-divide-by-zero. Posited
#      convention; documented in catalog as `numerical_floor_convention`.
ALLOWED_LITERALS: set[float] = {
    # Mathematical identities
    0, 1, -1,
    # Halving/doubling identities (symmetric part, midpoint, n(n+1)/2, 2π, etc.)
    0.5, 2, -2,
    # Universal binary unit conversions (KB↔MB↔GB)
    1024,
    # Universal SI time conversion (s↔ms) + RNG stream-offset multiplier
    1000,
    # Conventional numerical floor (divide-by-zero protection)
    # Cataloged as `numerical_floor_convention` (posited convention)
    1e-9,
}

# Files that are themselves the audit/catalog infrastructure and must not
# be scanned — they contain literals BY DESIGN (the safe-list, the posited
# constants registry, etc.) and scanning them creates circular self-reference
# noise. Self-scanning was part of the drift mode that lit up bogus
# undocumented-literal reports in 2026-05-24 cleanup.
EXCLUDE_FILES: set[str] = {
    # `spherical_harmonics.py` is a transcription of the standard
    # closed-form `Y_lm` polynomials for l ≤ 4 (Wikipedia "Table of
    # spherical harmonics", Edmonds 1957 §2.5). Every numeric in it
    # — coefficients like sqrt(35/(2π)), polynomial weights 3, 5, 7,
    # 30, 35, etc. — is locked by the mathematical formula and would
    # be wrong if anything but its specific value. Cataloging each
    # one as its own row in `magic_numbers.md` is busywork; flagging
    # individual coefficients of a closed-form polynomial expansion
    # is exactly the false-positive case the audit was designed to
    # avoid. The file is small (~200 LoC) and cross-checked by the
    # Monte-Carlo orthonormality test which would catch any
    # transcription error.
    "spherical_harmonics.py",
}

# Default directories to skip when walking a path tree. Shared between
# scan_path and extract_live_names so they stay in sync.
DEFAULT_EXCLUDE_DIRS: tuple[str, ...] = (
    "__pycache__", ".venv", ".venv-local", ".venv-hyplora",
    "tests", ".git", "node_modules",
)

# Regex matching backticked identifiers in the catalog markdown.
# Pre-compiled because we evaluate it per-line of magic_numbers.md.
_BACKTICK_NAME_RE = re.compile(r"`([A-Za-z_][A-Za-z_0-9]*)`")


@dataclass
class FoundLiteral:
    file_path: Path
    lineno: int
    col_offset: int
    value: float
    context_var: str | None = None  # the variable being assigned to, if any
    context_expr: str | None = None  # the surrounding expression source
    severity: str = "🔴"  # 🟢 / 🟡 / 🔴

    def __repr__(self) -> str:
        var_part = f" → {self.context_var}" if self.context_var else ""
        expr_part = f"  ({self.context_expr})" if self.context_expr else ""
        return (f"{self.severity} {self.file_path}:{self.lineno} "
                f"literal={self.value}{var_part}{expr_part}")


class ConstantVisitor(ast.NodeVisitor):
    """Walks AST and collects numeric literals with context."""

    def __init__(self, file_path: Path, source: str):
        self.file_path = file_path
        self.source = source
        self.source_lines = source.splitlines()
        self.literals: list[FoundLiteral] = []
        self._context_stack: list[str] = []  # innermost first

    def _line_text(self, lineno: int) -> str:
        if 0 < lineno <= len(self.source_lines):
            return self.source_lines[lineno - 1]
        return ""

    def visit_Assign(self, node: ast.Assign):
        # Capture the variable name being assigned
        targets = []
        for target in node.targets:
            if isinstance(target, ast.Name):
                targets.append(target.id)
            elif isinstance(target, ast.Attribute):
                targets.append(target.attr)
        # Visit the RHS with the target name in context
        self._context_stack.append(",".join(targets) if targets else "")
        self.generic_visit(node)
        self._context_stack.pop()

    def visit_AnnAssign(self, node: ast.AnnAssign):
        if isinstance(node.target, ast.Name):
            target = node.target.id
        elif isinstance(node.target, ast.Attribute):
            target = node.target.attr
        else:
            target = ""
        self._context_stack.append(target)
        self.generic_visit(node)
        self._context_stack.pop()

    def visit_keyword(self, node: ast.keyword):
        # Function call keyword argument — `func(arg=0.5)`
        if node.arg:
            self._context_stack.append(node.arg)
            self.generic_visit(node)
            self._context_stack.pop()
        else:
            self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef):
        """Track parameter defaults so `def f(lr=0.01)` carries `lr` as context."""
        args = node.args
        positional = list(getattr(args, "posonlyargs", [])) + list(args.args)
        n_defaults = len(args.defaults)
        n_positional = len(positional)
        for i, default in enumerate(args.defaults):
            arg_idx = n_positional - n_defaults + i
            if 0 <= arg_idx < n_positional:
                arg_name = positional[arg_idx].arg
                self._context_stack.append(arg_name)
                self.visit(default)
                self._context_stack.pop()
        for arg, kw_default in zip(args.kwonlyargs, args.kw_defaults, strict=True):
            if kw_default is None:
                continue
            self._context_stack.append(arg.arg)
            self.visit(kw_default)
            self._context_stack.pop()
        # Walk decorators, return annotation, and body separately to avoid
        # re-visiting the defaults we already handled
        for decorator in node.decorator_list:
            self.visit(decorator)
        if node.returns:
            self.visit(node.returns)
        for stmt in node.body:
            self.visit(stmt)

    visit_AsyncFunctionDef = visit_FunctionDef  # same logic for async

    def visit_Compare(self, node: ast.Compare):
        # A literal rank next to .ndim or len(.shape) is a structural
        # requirement. Only that operand is approved, never its whole line.
        def is_rank(expr: ast.expr) -> bool:
            return (
                isinstance(expr, ast.Attribute) and expr.attr == "ndim"
            ) or (
                isinstance(expr, ast.Call)
                and isinstance(expr.func, ast.Name) and expr.func.id == "len"
                and len(expr.args) == 1 and not expr.keywords
                and isinstance(expr.args[0], ast.Attribute)
                and expr.args[0].attr == "shape"
            )

        operands = [node.left, *node.comparators]
        ranks = set()
        for left, right in zip(operands, operands[1:], strict=False):
            for rank, literal in ((left, right), (right, left)):
                if (is_rank(rank) and isinstance(literal, ast.Constant)
                        and type(literal.value) is int and literal.value >= 0):
                    ranks.add(id(literal))
        for operand in operands:
            start = len(self.literals)
            self.visit(operand)
            if id(operand) in ranks:
                for found in self.literals[start:]:
                    found.severity = "🟡"

    def visit_Constant(self, node: ast.Constant):
        if not isinstance(node.value, (int, float)) or isinstance(node.value, bool):
            return
        value = float(node.value)
        if value in ALLOWED_LITERALS:
            return
        # Negative versions allowed
        if -value in ALLOWED_LITERALS:
            return
        # Outermost context (current assignment target / keyword arg name)
        context_var = self._context_stack[-1] if self._context_stack else None
        line_text = self._line_text(node.lineno)
        self.literals.append(FoundLiteral(
            file_path=self.file_path,
            lineno=node.lineno,
            col_offset=node.col_offset,
            value=value,
            context_var=context_var,
            context_expr=line_text.strip()[:80],
            severity="🔴",
        ))


def load_catalog(catalog_path: Path) -> set[str]:
    """Read magic_numbers.md and return the set of constant names documented there."""
    if not catalog_path.exists():
        return set()
    text = catalog_path.read_text()
    # Grab anything in backticks — magic_numbers.md uses `name` for constant names
    return set(_BACKTICK_NAME_RE.findall(text))


def load_catalog_status(catalog_path: Path,
                          live_names: set[str]) -> dict[str, list[str]]:
    """Scan catalog rows; return names grouped by status marker.

    Considers a name "live" if it appears in `live_names` (extracted from the
    actual source). Names NOT in live_names are archaeological and excluded
    from the status report (they're documented for posterity, not enforced).

    Returns a dict with keys: "derived" (✅), "scale_invariant" (⚖️),
    "experimentally_set" (🔬), "unresolved" (⚠️), "retired" (🏚️).
    """
    if not catalog_path.exists():
        return {}
    text = catalog_path.read_text()
    status_marks = {
        "derived": "✅",
        "scale_invariant": "⚖️",
        "experimentally_set": "🔬",
        "unresolved": "⚠️",
        "retired": "🏚️",
    }
    by_status: dict[str, list[str]] = {k: [] for k in status_marks}
    for line in text.splitlines():
        if "|" not in line:
            continue
        # Try to extract the first backticked name in the row
        names_in_row = _BACKTICK_NAME_RE.findall(line)
        if not names_in_row:
            continue
        primary_name = names_in_row[0]
        if primary_name not in live_names:
            continue  # archaeological; skip
        for status_key, mark in status_marks.items():
            if mark in line:
                by_status[status_key].append(primary_name)
                break
    return by_status


def extract_live_names(
    paths: list[Path],
    exclude_dirs: tuple[str, ...] = DEFAULT_EXCLUDE_DIRS,
) -> set[str]:
    """Walk source paths; return the set of identifier names that appear as
    AnnAssign targets, function arg names with defaults, or assignment targets.

    Used to determine which catalog entries are "live" (referenced in code).
    """
    live: set[str] = set()
    for root in paths:
        if root.is_file() and root.suffix == ".py":
            files = [root] if not _is_excluded_file(root) else []
        elif root.is_dir():
            files = [p for p in root.rglob("*.py")
                       if not any(part in exclude_dirs for part in p.parts)
                       and not _is_excluded_file(p)]
        else:
            files = []
        for path in files:
            try:
                source = path.read_text()
                tree = ast.parse(source)
            except (OSError, UnicodeDecodeError, SyntaxError):
                continue
            for node in ast.walk(tree):
                if (isinstance(node, ast.AnnAssign)
                        and isinstance(node.target, ast.Name)):
                    live.add(node.target.id)
                elif isinstance(node, ast.Assign):
                    for tgt in node.targets:
                        if isinstance(tgt, ast.Name):
                            live.add(tgt.id)
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    args = node.args
                    positional = list(args.posonlyargs) + list(args.args)
                    n_defaults = len(args.defaults)
                    n_positional = len(positional)
                    for i in range(n_defaults):
                        arg_idx = n_positional - n_defaults + i
                        if 0 <= arg_idx < n_positional:
                            live.add(positional[arg_idx].arg)
                    for arg, default in zip(
                        args.kwonlyargs, args.kw_defaults, strict=True,
                    ):
                        if default is not None:
                            live.add(arg.arg)
    return live


class ScanError(Exception):
    """A requested source path could not be read or parsed."""


def _is_excluded_file(path: Path) -> bool:
    # Only this tool is exempt, not unrelated files also called audit.py.
    return path.resolve() == Path(__file__).resolve() or path.name in EXCLUDE_FILES


def scan_file(path: Path, catalog_names: set[str]) -> list[FoundLiteral]:
    """Scan one Python file, raising ScanError if reading or parsing fails."""
    try:
        source = path.read_text()
        tree = ast.parse(source, filename=str(path))
    except (OSError, UnicodeDecodeError, SyntaxError) as error:
        raise ScanError(f"{path}: {error}") from error
    if _is_excluded_file(path):
        return []
    visitor = ConstantVisitor(path, source)
    visitor.visit(tree)
    # Cross-reference against catalog names
    for lit in visitor.literals:
        if lit.context_var and lit.context_var in catalog_names:
            # Variable name matches a catalog entry — mark documented
            lit.severity = "🟡"
    return visitor.literals


def scan_path(
    root: Path, catalog_names: set[str],
    exclude_dirs: tuple[str, ...] = DEFAULT_EXCLUDE_DIRS,
    *, errors: list[ScanError] | None = None,
) -> list[FoundLiteral]:
    """Scan a path, collecting failures in errors or raising the first failure."""
    all_literals: list[FoundLiteral] = []

    def report(error: ScanError) -> None:
        if errors is None:
            raise error
        errors.append(error)

    def scan(path: Path) -> None:
        try:
            all_literals.extend(scan_file(path, catalog_names))
        except ScanError as error:
            report(error)

    try:
        root.stat()
    except OSError as error:
        report(ScanError(f"{root}: {error}"))
        return all_literals
    if root.is_file():
        if root.suffix == ".py":
            scan(root)
        else:
            report(ScanError(f"{root}: expected a Python source file"))
    elif root.is_dir():
        def walk_error(error: OSError) -> None:
            report(ScanError(f"{root}: {error}"))

        # os.walk reports unreadable directories; Path.rglob can omit them.
        for directory, dirs, filenames in os.walk(root, onerror=walk_error):
            dirs[:] = sorted(d for d in dirs if d not in exclude_dirs)
            for filename in sorted(filenames):
                if filename.endswith(".py"):
                    scan(Path(directory) / filename)
    else:
        report(ScanError(f"{root}: expected a file or directory"))
    return all_literals


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Find undocumented constants in source.")
    ap.add_argument("paths", nargs="+", type=Path,
                    help="Files or directories to scan")
    ap.add_argument("--catalog", type=Path,
                    default=Path("notes/magic_numbers.md"),
                    help="Magic numbers catalog (default: notes/magic_numbers.md)")
    ap.add_argument("--strict", action="store_true",
                    help="Exit code 1 on undocumented literals or scan failures")
    ap.add_argument("--strict-status", action="store_true",
                    help="Exit code 1 if any LIVE catalog entry is ⚠️ Unresolved")
    ap.add_argument("--show-allowed", action="store_true",
                    help="Show 🟢 allowed literals too (verbose)")
    ap.add_argument("--show-documented", action="store_true",
                    help="Show 🟡 documented literals too")
    args = ap.parse_args(argv)

    catalog_names = load_catalog(args.catalog)
    print(f"Catalog: {args.catalog} ({len(catalog_names)} documented names)")

    # Determine which catalog entries are LIVE (appear in source)
    live_names = extract_live_names(list(args.paths))
    live_catalog = catalog_names & live_names
    archaeological = catalog_names - live_names
    print(f"  live (in source): {len(live_catalog)}, "
          f"archaeological (catalog-only): {len(archaeological)}")

    # Status counts on LIVE constants
    by_status = load_catalog_status(args.catalog, live_names)
    if by_status:
        print("\n=== Catalog status (LIVE constants only) ===")
        for status_key, mark in [("derived", "✅"), ("scale_invariant", "⚖️"),
                                    ("experimentally_set", "🔬"),
                                    ("unresolved", "⚠️"), ("retired", "🏚️")]:
            names_list = by_status.get(status_key, [])
            if names_list:
                print(f"  {mark} {status_key} ({len(names_list)}): "
                      f"{', '.join(sorted(names_list))}")

    all_literals: list[FoundLiteral] = []
    scan_errors: list[ScanError] = []
    for p in args.paths:
        all_literals.extend(scan_path(p, catalog_names, errors=scan_errors))
    for error in scan_errors:
        print(f"Scan error: {error}", file=sys.stderr)

    # Group by severity
    by_sev: dict[str, list[FoundLiteral]] = {"🔴": [], "🟡": [], "🟢": []}
    for lit in all_literals:
        by_sev.setdefault(lit.severity, []).append(lit)

    n_red = len(by_sev["🔴"])
    n_yellow = len(by_sev["🟡"])

    if by_sev["🔴"]:
        print(f"\n=== 🔴 Undocumented literals ({n_red}) ===")
        for lit in by_sev["🔴"]:
            print(f"  {lit}")
    if args.show_documented and by_sev["🟡"]:
        print(f"\n=== 🟡 Documented or derived ({n_yellow}) ===")
        for lit in by_sev["🟡"]:
            print(f"  {lit}")

    print(f"\nSummary: {n_red} undocumented, {n_yellow} documented/derived, "
          f"{len(by_sev['🟢'])} allowed, {len(scan_errors)} scan errors.")

    n_unresolved_live = len(by_status.get("unresolved", []))
    if n_unresolved_live > 0:
        print(f"  ⚠️ {n_unresolved_live} LIVE catalog entries still Unresolved")

    if args.strict and (n_red > 0 or scan_errors):
        return 1
    if args.strict_status and n_unresolved_live > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
