# holonomy-lib

Batched PyTorch primitives for geometry, spectral graph theory, discrete curvature,
topology, and tensor decompositions. Current release **v0.8.0**, licensed
**FSL-1.1-ALv2** with Synoros as licensor. Releases through 0.7.0 remain
AGPL-3.0-or-later. See `LICENSE.md`.

A primitive is done when it is batched on a configurable device, typed, cited,
numerically checked against an independent reference, and passes the constant audit
with its tunable constants cataloged in `notes/magic_numbers.md`. State any batch,
dtype, differentiation, or backend limitations. Tests belong where they protect a
correctness guarantee; test counts and coverage do not establish correctness.

## Files agents read (exhaustive — anything not listed here is NOT read)

Repo instructions and standing state:

- `AGENTS.md` (this file; `CLAUDE.md` is a symlink to it)
- `direction.md` — read at session start. Current goal, ≤1 page.
- `ledger.md` — append 3–5 lines before ending any session (date, changed, failed, archived).
- `archive/README.md` — tombstones. Read before proposing an approach; don't retry dead ends.
- `CONVENTIONS.md` — **binding coding standard, and load-bearing.** Source, tests, and
  notes cite it by section number (`CONVENTIONS §3.3`, `§1.1`, `§4.1`, …) in 15+ files.
  Do not renumber its sections or move the file; edits must preserve section numbering.

Implementation and tooling (read the relevant files before changing behavior):

- `src/holonomy_lib/` — the library; `__all__` exports define the inventory.
- `src/holonomy_explorer/` — separate local explorer package and scene adapters.
- `tests/` — existing correctness checks; read those for the affected primitive.
- `tools/audit.py` — numerical-literal audit and its explicit limitations.
- `pyproject.toml`, `uv.lock` — packaging, dependencies, and tool configuration.
- `LICENSE.md` — release license text.

Public API surface (keep in sync when adding or changing a primitive):

- `README.md` — installation, worked example, package overview, explorer, numerical
  limits, and licensing.
- `CONTENTS.md` — live flat inventory of every public primitive: signature, one-line
  summary, citation. CONVENTIONS §9.2 requires a row here for each new primitive.
- `CHANGELOG.md` — released behavior, per version.

Research canon (first-class per injected global standards; these are the evidence base for
the library's claims, not scratch notes):

- `notes/magic_numbers.md` — catalog of every posited constant with derivation and
  scale-of-validity. The static audit flags undocumented literals; it does not
  certify the derivation or numerical correctness (CONVENTIONS §5).
- `notes/research_claims_catalog.md` — the "our research" claims with explicit status fields.
- `notes/research_open_questions.md` — implementation choices not traceable to a published
  formula. Read before asserting a hyperbolic result is cited.
- `notes/substrate_provenance.md` — what synoros-substrate and this library gave each other.
- `notes/scrutiny.md` — running log of critical-review passes: checked, found, fixed.
- `notes/benchmark_baseline.md`, `notes/benchmark_optimized.md`,
  `notes/benchmark_info_geometry.md` (+ `.json`) — the benchmark series.
  `benchmark_baseline.md` is cited by `magic_numbers.md` for
  `RETRACTION_RANDOMIZED_THRESHOLD`; do not move it.
- `notes/benchmark_2026-05-26_roadmap_sweep.md`,
  `notes/benchmark_2026-07-08_pr1_quadrature_coboundary.md`,
  `notes/benchmark_2026-07-08_solvers.md` — dated run records. These are *not* disposable:
  each is linked from `README.md`, and `benchmark_2026-07-08_solvers.md` is additionally
  cited by `CHANGELOG.md`, `notes/magic_numbers.md`, and the docstring of
  `src/holonomy_lib/convex/least_squares.py`.
- `notes/validation/` — validation records (results `.md` beside the script that produced
  them). Cross-checks against geoopt, established closed forms, and synoros-substrate.
- `notes/strengthening/` — historical C1–C7 work; read the current claims catalog
  status before using it as evidence for the current implementation.
- `notes/verification/` — sympy scripts that verify closed forms symbolically.

Global standards (doc contract, testing, comments, wiring-before-benchmarks) are
injected by the harness. Machine-specific paths belong there, not in this file.

Note for outside contributors: `direction.md`, `ledger.md`, `archive/`, and the `CLAUDE.md`
symlink are local working files, gitignored and not shipped in the public repo (see
CONVENTIONS §10.4). Everything else listed above is tracked.

## Commands

Run commands from a source checkout in the synced environment. Hardware support
comes from the installed PyTorch build, not the Python version alone. Check
`torch.cuda.is_available()` before reporting accelerator results; ROCm uses `cuda`.

- install / sync deps: `uv sync --extra dev`
- test: `python -m pytest -q`
- audit: `python tools/audit.py src/ --strict`
- lint: `python -m ruff check src/ tests/`
- types: `python -m mypy src/`
- explorer: `holonomy-explorer` or `python -m holonomy_explorer`
- GPU work: use the configured workstation GPU interpreter with `PYTHONPATH=src`;
  do not install a second PyTorch copy per worktree.

**Verify (definition of done for a change):** the real affected computation agrees
with its reference; the suite, Ruff, mypy, and strict constant audit pass.
Recorded integration baselines (2026-10-25) for 0.8.0: CPU `697 passed, 36 skipped`; ROCm GPU
`712 passed, 25 skipped`; audit `0 undocumented`. These are dated observations,
not substitutes for checking a changed result.

## Invariants / non-negotiables

1. **Discuss-look-test.** Before adding a primitive, find existing implementations in
   geoopt / geomstats / pymanopt / gudhi / tensorly. Decide: port, wrap, or reimplement.
   Don't reinvent without checking.
2. **No uncataloged numerical constants.** Every literal is derived from inputs, a
   universal invariant, or an experimentally-tuned constant with a `notes/magic_numbers.md`
   entry carrying its procedure and scale-of-validity. Review the derivation as well
   as passing the heuristic audit.
3. **Citations are non-optional.** Every public mathematical function has a `References:`
   section naming its source. Properties and plain constructors are exempt (CONVENTIONS §6).
   Never ship "TODO: add citation".
4. **Batched on PyTorch.** Dense primitives accept `B ∈ {0, 1, > 1}`, preserve input
   placement and compatible dtype, and use tensor operations across batch members.
   Check numerical batch correspondence with distinct inputs. Document single-instance
   sparse APIs, host work, and unsupported differentiation explicitly.
5. **Library code only in `src/holonomy_lib`.** No application model classes or training
   loops. `src/holonomy_explorer` is a separate package for the local UI. In-progress
   mathematical primitives ship only under `holonomy_lib.experimental`, with an honest
   opening status line naming what is unresolved; keep them out of stable exports.
6. **No fluent prose without source-line verification.** If you are writing confident prose
   about how a function behaves, stop and read the file at those lines first.
7. **Wiring before benchmarks.** Never report a benchmark number until you have confirmed
   the path under test is actually the path being exercised — and never report a GPU number
   from a CPU run.
8. **No speculative features.** Build what a named consumer actually needs.
9. **No backwards-compat shims or feature flags.** This is a research library; we change it,
   we don't preserve it.
10. **No unilateral architectural choices.** When unsure, present options + tradeoffs and stop.
11. **No em dashes in user-facing docs** (`README.md`, `CONTENTS.md`, `CONVENTIONS.md`,
    docstrings). `notes/` and internal commentary are exempt. CONVENTIONS §10.2.
12. **Commit format:** `module: short summary`; the body explains what changed and why, not
    what the code does. CONVENTIONS §10.3.

**Posture:** peer, not prescription. Critical feedback is welcome and expected.
Mathematical rigor matters; engineering shortcuts that compromise it are not acceptable.
