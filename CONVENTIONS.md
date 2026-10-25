# Coding conventions for holonomy_lib

These are the binding rules every primitive in this library follows.
They're enforced by:
- the audit tool (`python -m holonomy_lib.audit src/ --strict`) for the
  numerical-constant rules,
- the test suite for the API-shape rules,
- and code review for the rest.

If you're adding a new primitive, work through this list before opening
the PR. If you're modifying an existing primitive, the conventions
constrain what's allowed.

---

## 1. API shape

### 1.1 Batched-first

Every public function takes a leading **batch dimension** `B` followed
by the math dimensions:

| primitive kind         | input shape       |
|------------------------|-------------------|
| graph / matrix         | `(B, n, n)`       |
| vector / signal        | `(B, n)`          |
| tensor of order `d`    | `(B, n_1, …, n_d)`|
| distribution / simplex | `(B, k)`          |

Single-point use is `B = 1`. Outputs preserve the batch shape:
`(B, …)` in, `(B, …)` out. **All primitives must work for
`B ∈ {0, 1, > 1}`**; there is a test for each.

### 1.2 Device and dtype propagation

Outputs land on the same device and (compatible) dtype as inputs. We
do not silently `.cpu()` or upcast. The constructors of class-based
primitives (`FixedRankManifold`, `SPDManifold`) accept `device` and
`dtype`; functional primitives infer from their tensor arguments.

### 1.3 Input validation

Validation order at function entry:

1. **Shape checks** that don't depend on dtype (square, `ndim ≥ 2`,
   `len(ranks) == d`, …) → `ValueError` with a message naming the
   offending shape.
2. **Dtype checks** (e.g. `magnetic.combinatorial` rejects complex
   `A`) → `ValueError`.
3. **Value-range checks** on parameters (`alpha ∈ [0, 1]`, `q ∈
   [0, 1)`, `r > 0`, …) → `ValueError`.
4. **Soft conditions** that are algorithmically meaningful but not
   universally enforced (asymmetric input to a symmetric-graph
   primitive; disconnected graph to a primitive whose math implies
   connectedness) → `warnings.warn(..., stacklevel=2)`.

Validation should happen **before** any heavy compute. Never use
`assert` for input validation: `python -O` strips assertions and
your guard silently disappears.

---

## 2. Graph conventions

### 2.1 Self-loops: dropped at entry

**All graph primitives treat the input adjacency `A` as a simple
graph.** The diagonal `A[i, i]` is dropped at function entry via
`holonomy_lib._graph_utils.drop_self_loops`. Self-loops do **not**
contribute to a node's degree, the random-walk transition matrix,
the magnetic phase factor, or any other downstream quantity.

Rationale: most graph math (Forman-Ricci, Ollivier-Ricci, normalized
Laplacians, magnetic Laplacian) is defined on simple graphs in the
source literature. Implicitly including self-loops changes `d_i =
Σ_j A_{ij}` and propagates inconsistently across primitives: the
combinatorial Laplacian happens to cancel them out, but
`symmetric_normalized` and `random_walk` do not, and `Ollivier-Ricci`
gets a different walk distribution. To keep the library predictable,
all graph primitives normalize at entry.

If a caller wants a primitive's behavior **with** self-loops, they
need to fold the loop weight into a parallel construction (e.g.
add `α · I` to the matrix the primitive returns); the input
adjacency itself is treated as simple.

### 2.2 Symmetry

Most graph primitives assume `A == A.mT`. When a primitive's math
requires symmetry, asymmetry triggers a `UserWarning` (not an error,
since some callers symmetrize internally and just want the warning as
a flag). The `magnetic` Laplacian is the exception: it explicitly
handles asymmetric input via the phase factor.

### 2.3 Isolated and disconnected nodes

- **Isolated nodes** (`d_i = 0`): pseudoinverse convention. Wherever
  the math has `1/d_i` or `1/sqrt(d_i)`, the isolated node gets `0`
  via `torch.where(d > 0, …, 0)`. See `Cheng-Wu (2024)` for the
  formal treatment.

- **Disconnected components**: each primitive documents its behavior.
  Ollivier curvature inflates the shortest-path distance to a large
  finite value (`DISCONNECTED_DISTANCE_MULTIPLIER`) so Sinkhorn stays
  finite; diffusion-map emits a `UserWarning` because the embedding
  contains degenerate constant-on-component modes; effective
  resistance returns 0 between components (which the docstring
  flags as "not physically meaningful, mask by component").

---

## 3. Numerical conventions

### 3.1 Floor: `1e-9`

The library-wide `numerical_floor_convention` is `1e-9`. It's the
floor used in `1 / max(x, 1e-9)`-style guards. It's listed in
`audit.ALLOWED_LITERALS` so every primitive can reuse it.

Rationale: `1e-9` is comfortably below float32 eps (`~1.19e-7`) for
safety across dtypes, and well above `torch.finfo(float64).tiny`
(`~2.22e-308`) so we never amplify true zero into a huge reciprocal.

### 3.2 Eigenvalue clamping

When computing `1 / sqrt(λ)` or `log(λ)` on eigenvalues of a
nominally PSD matrix, clamp at `torch.finfo(dtype).tiny` to absorb
float-error negatives without producing inf/NaN. See
`SPDManifold._sqrt_and_inv_sqrt` for the canonical pattern.

### 3.3 Cholesky for SPD solves

Whenever you need `S⁻¹ X` for SPD `S`, use `torch.linalg.cholesky(S)`
+ `torch.cholesky_solve(X, L)` instead of `torch.linalg.solve(S, X)`.
Cholesky is `O(n³/3)` factorize + `O(n²)` per solve; LU is `O(n³)`
factorize. For chained ops at the same `S`, expose a
`whitening=(S_sqrt, S_inv_sqrt)` kwarg so callers can precompute
once and pass through.

### 3.4 Pseudoinverse threshold

When computing the Moore-Penrose pseudoinverse of a symmetric matrix
via eigh, the threshold for "is this eigenvalue effectively zero?"
is `max_abs · 1e-9`. Compare against unclamped eigenvalues for the
where-branch; clamp the safe value for the reciprocal:

```python
safe_eig = eigvals.clamp(min=torch.finfo(dtype).tiny)
inv_eig = torch.where(eigvals > max_abs * 1e-9,
                       torch.reciprocal(safe_eig),
                       torch.zeros_like(eigvals))
```

This pattern handles isolated nodes, disconnected components, and
the all-zeros graph correctly (all eigenvalues exactly zero ⇒
pseudoinverse is zero).

---

## 4. Performance patterns

### 4.1 Host syncs (`.item()`)

Calls that force a GPU → CPU sync are a real cost. They should not
appear in inner loops without a clear reason. Specifically:

- **Convergence checks** in iterative algorithms (Sinkhorn, Lanczos)
  may use `.item()` for early stopping, but cadence should be
  configurable. `SINKHORN_SYNC_EVERY_DEFAULT = 8` is the established
  pattern.
- **Branching on tensor values** (e.g. `if finite_mask.all():`) is
  banned in the hot path; use `torch.where` to make the operation
  unconditional.

### 4.2 Pre-allocate hot-loop tensors

Don't rebuild tensors in tight loops. The Lanczos `V` matrix is
pre-allocated as `torch.empty(*batch, n, n_iter + 1)` and written
in-place per iteration, never reassembled with `torch.stack`.

### 4.3 Tile when memory is the bottleneck

The `(B, n², n)` Sinkhorn pair structure was the canonical example:
materializing it explodes RAM at modest `n`. The fix is
**tile-over-pairs** with a configurable `tile_size` parameter
(default `SINKHORN_TILE_DEFAULT = 256`). Future memory-bound
primitives should follow the same pattern.

### 4.4 Algorithm switching

Some primitives have multiple algorithms with different break-even
points (e.g. exact vs. randomized SVD in `FixedRankManifold.retraction`).
The pattern is:

- Constructor / function param `mode = "auto"` (default), `"exact"`,
  `"randomized"`.
- `"auto"` switches based on a documented threshold cataloged in
  `notes/magic_numbers.md` (`RETRACTION_RANDOMIZED_THRESHOLD = 0.25`).
- Threshold is derived from benchmark data, not hand-picked.

---

## 5. Numerical constants

### 5.1 Three categories

Every numerical literal in source code must fall into one of:

1. **Derived from inputs**: `1/N`, `1/sqrt(d)`, `n(n+1)/2`.
   No catalog entry needed; the audit treats these as ✅.
2. **Universal invariant**: `0, 1, -1, 0.5, 2, 1024, 1000, π, e,
   1e-9`. Listed in `audit.ALLOWED_LITERALS`. No catalog entry
   needed.
3. **Experimentally tuned**: `SINKHORN_TILE_DEFAULT = 256`,
   `MAGNETIC_CHARGE_DEFAULT = 0.25`, etc. **Catalog entry required**
   in `notes/magic_numbers.md` with:
   - the literature reference or empirical procedure used to pick
     the value,
   - the scale-of-validity (what changes if `n` or the input
     distribution changes substantially),
   - the file path where it's used.

The audit tool fails the build on any uncataloged literal.

### 5.2 Catalog entries

Add the constant as `UPPER_SNAKE_CASE`, with a docstring above its
definition explaining the derivation. The catalog row should not
re-derive; point at the docstring and the paper.

---

## 6. Citations

Every public function has a `References:` section in its docstring
pointing to the paper or textbook that defines its math. Format:

```
References:
  Author1, Author2 (Year). Title. Journal Volume(Issue):pages, §section
    for the specific result you're implementing.
```

If you implement an algorithm with multiple authors / variants,
cite the variant you actually used and note the alternative in a
follow-up line.

**Never ship a primitive with "TODO: add citation".** If you can't
find a paper for the math you're implementing, you're either
reinventing something that exists (look harder) or you're doing
original research (write it down first, then implement).

---

## 7. Provenance

Provenance was removed in 0.8.0.
The removal rationale and requirements for any return are in `archive/README.md`.

### 7.1 Op IDs (removed in 0.8.0)

### 7.2 Op versions (removed in 0.8.0)

### 7.3 No `*args` / `**kwargs` (removed in 0.8.0)

---

## 8. Testing

### 8.1 Three layers

Every primitive has:

1. **Unit tests**: shape correctness across `B ∈ {0, 1, several}`.
   Validation tests (rejects bad input).
2. **Property tests**: mathematical invariants (a projection is
   idempotent; a retraction lands on the manifold; a divergence
   is non-negative).
3. **Comparison tests**: agreement with an established library
   where one exists (`pymanopt`, `geoopt`, `tensorly`,
   `GraphRicciCurvature`). Tests `pytest.importorskip` the
   comparison library so the suite runs without it installed.

### 8.2 Tolerances

Use `torch.testing.assert_close` with explicit `atol` and `rtol`.
Default `atol=1e-12, rtol=0` for exact closed-form checks (e.g.
K_3 Ollivier curvature). Loosen only with a comment explaining
why (e.g. Sinkhorn convergence plateau).

### 8.3 Regression tests

When you fix a bug, add a regression test in the same commit. The
test should fail against the buggy code and pass against the fix.
A new test file isn't needed; add to the existing module's test
file under an appropriate class.

---

## 9. Documentation

### 9.1 Docstrings

Every public function has a docstring with:

- One-line summary.
- Math statement (formula, LaTeX-style ascii is fine).
- `Args:` section with shape conventions.
- `Returns:` section with shape conventions.
- `References:` section (see §6).
- Optional `Notes:` for performance, convergence caveats,
  scale-of-validity, etc.

### 9.2 CONTENTS.md inventory

Add the primitive to `CONTENTS.md` with its signature and a one-line
description. This is the public API map; users (and downstream
search engines) hit this first.

### 9.3 Magic-numbers catalog

If the primitive introduces any new tunable constant, add a row to
`notes/magic_numbers.md` per §5.

---

## 10. Repository hygiene

### 10.1 Internal modules

Files starting with `_` (e.g. `_graph_utils.py`) are internal:
not exported from package `__init__`, not part of the public API.
Import from them only inside `holonomy_lib`.

### 10.2 No em dashes in user-facing docs

`README.md`, `CONTENTS.md`, `CONVENTIONS.md`, the docstrings users
see in IDEs, use colons, commas, or semicolons instead of em-dashes.
The `notes/` directory and internal commentary is fine.

### 10.3 Commit messages

Format: `module: short summary` (e.g. `spectral: drop self-loops at
entry to graph primitives`). The body explains _what changed and
why_, not _what the code does_; that's the docstring's job.

End every commit with the `Co-Authored-By` trailer when the work
was done in pair with the Claude Code agent:

```
Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
```

### 10.4 No internal-only files in the public tree

`AGENTS.md` is the canonical agent instruction file and **is** tracked:
it carries this library's operating constraints and points here for the
coding standard. `CLAUDE.md` is a symlink to it.

Working-state files stay on local disk only, because they can name
private sibling repos: `CLAUDE.md` (the symlink), `direction.md`
(current goal), `ledger.md` (session log), and `archive/` (superseded
docs plus their tombstones). All are listed in `.gitignore`.
