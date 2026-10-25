# holonomy-lib

Research-grade PyTorch math library: GPU-native, batched-first geometric primitives —
hyperbolic / spherical / mixed-curvature manifolds with full autograd, differentiable
hyperbolic heat kernels, spectral graph theory, discrete Ricci flow, simplicial topology,
persistent homology, cellular sheaves, SO(3) Lie primitives, information geometry, tensor
decompositions, content-addressable provenance. Current release **v0.7.0** (AGPL-3.0,
public at `github.com/Synoros-io/holonomy_lib`).

"Working" here is stricter than "the tests pass". A primitive is done only when it is
implemented batched-first on a configurable device, type-annotated, carries a
`References:` docstring section citing the paper for its math, has unit + property +
cross-library comparison tests, passes the constant audit, and has any tunable constant
cataloged in `notes/magic_numbers.md`. Missing any one of those means it is not done.
The value of this library is that downstream researchers can *trust* the implementations;
every shortcut taken here costs that trust everywhere downstream.

## Files agents read (exhaustive — anything not listed here is NOT read)

Repo instructions and standing state:

- `AGENTS.md` (this file; `CLAUDE.md` is a symlink to it)
- `direction.md` — read at session start. Current goal, ≤1 page.
- `ledger.md` — append 3–5 lines before ending any session (date, changed, failed, archived).
- `archive/README.md` — tombstones. Read before proposing an approach; don't retry dead ends.
- `CONVENTIONS.md` — **binding coding standard, and load-bearing.** Source, tests, and
  notes cite it by section number (`CONVENTIONS §3.3`, `§1.1`, `§4.1`, …) in 15+ files.
  Do not renumber its sections or move the file; edits must preserve section numbering.

Public API surface (keep in sync when adding or changing a primitive):

- `README.md` — project overview, module table, performance highlights, roadmap.
- `CONTENTS.md` — live flat inventory of every public primitive: signature, one-line
  summary, citation. CONVENTIONS §9.2 requires a row here for each new primitive.
- `CHANGELOG.md` — released behavior, per version.

Research canon (first-class per `~/agent-standards.md`; these are the evidence base for
the library's claims, not scratch notes):

- `notes/magic_numbers.md` — catalog of every posited constant with derivation and
  scale-of-validity. The audit tool fails the build on any uncataloged literal.
- `notes/research_claims_catalog.md` — the "our research" claims with explicit status fields.
- `notes/research_open_questions.md` — implementation choices not traceable to a published
  formula. Read before asserting a hyperbolic result is cited.
- `notes/substrate_provenance.md` — what synoros-substrate and this library gave each other.
- `notes/scrutiny.md` — running log of critical-review passes: checked, found, fixed.
- `notes/benchmark_provenance_v0.2.md`, `notes/benchmark_provenance_v0.3.md` (+ `.json`)
  — versioned benchmark provenance records.
- `notes/benchmark_baseline.md`, `notes/benchmark_optimized.md`,
  `notes/benchmark_info_geometry.md` (+ `.json`) — the benchmark series.
  `benchmark_baseline.md` is cited by `magic_numbers.md` for
  `RETRACTION_RANDOMIZED_THRESHOLD`; do not move it.
- `notes/benchmark_2026-05-26_roadmap_sweep.md`,
  `notes/benchmark_2026-07-08_pr1_quadrature_coboundary.md`,
  `notes/benchmark_2026-07-08_solvers.md` — dated run records. These are *not* disposable:
  each is linked from `README.md`, and `benchmark_2026-07-08_solvers.md` is additionally
  cited by `CHANGELOG.md`, `CONTENTS.md`, `notes/magic_numbers.md`, and the docstring of
  `src/holonomy_lib/convex/least_squares.py`.
- `notes/validation/` — validation records (results `.md` beside the script that produced
  them). Cross-checks against geoopt, established closed forms, and synoros-substrate.
- `notes/strengthening/` — C1–C7 claim-strengthening work backing the claims catalog.
- `notes/verification/` — sympy scripts that verify closed forms symbolically.

Global standards (doc contract, testing, comments, wiring-before-benchmarks):
`~/agent-standards.md` — already injected in both harnesses.

Note for outside contributors: `direction.md`, `ledger.md`, `archive/`, and the `CLAUDE.md`
symlink are local working files, gitignored and not shipped in the public repo (see
CONVENTIONS §10.4). Everything else listed above is tracked.

## Commands

The repo `.venv` is **Python 3.14 + torch 2.12.0+cu130, and `torch.cuda.is_available()` is
`False`** — it is a CPU-only build, because PyTorch publishes no GPU wheels for cp314. It
is fine for tests, audit, and lint. It is **not** valid for any GPU claim. This is why the
benchmark provenance docs record `CUDA available: False`.

- install / sync deps: `uv sync --extra dev`
- test: `.venv/bin/python -m pytest -q`
- audit (constant discipline; CI gate): `.venv/bin/python -m holonomy_lib.audit src/ --strict`
- lint: `.venv/bin/python -m ruff check src/ tests/`
- types: `.venv/bin/python -m mypy src/`
- **GPU / benchmark work**: run the same commands with a Python whose torch has a GPU
  build, never a per-repo torch install: `PYTHONPATH=src <gpu-python> -m pytest -q`.
  Which interpreter that is on a given machine is a workstation fact, kept in the
  global agent standards, not in this public file.

**Verify (definition of done for a change):** full suite green **and** audit clean.
Last verified baseline: `1220 passed, 44 skipped`, audit `0 undocumented`.

## Invariants / non-negotiables

1. **Discuss-look-test.** Before adding a primitive, find existing implementations in
   geoopt / geomstats / pymanopt / gudhi / tensorly. Decide: port, wrap, or reimplement.
   Don't reinvent without checking.
2. **No uncataloged numerical constants.** Every literal is derived from inputs, a
   universal invariant, or an experimentally-tuned constant with a `notes/magic_numbers.md`
   entry carrying its procedure and scale-of-validity. The audit enforces this; failing
   audit = failing build.
3. **Citations are non-optional.** Every public function has a `References:` section naming
   the paper or textbook for its math. Never ship "TODO: add citation".
4. **GPU-first, batched-first.** Leading batch dim, `torch.Tensor`, device-agnostic. Every
   primitive must work for `B ∈ {0, 1, > 1}`; there is a test for each.
5. **Library code only.** No research experiments, no model classes, no training loops. This
   IS the library other projects depend on.
6. **No fluent prose without source-line verification.** If you are writing confident prose
   about how a function behaves, stop and read the file at those lines first.
7. **Wiring before benchmarks.** Never report a benchmark number until you have confirmed
   the path under test is actually the path being exercised — and never report a GPU number
   from the CPU-only repo `.venv`.
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
