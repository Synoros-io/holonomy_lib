# Scrutiny notes — running log

Running log of critical-review findings during scrutiny passes over the
library. Each entry: what was checked, what was found, what we did.

Status legend: ✅ fixed inline · 📝 documented only · 🔬 needs follow-up

---

## Pass 1 — 2026-05-27

Carry-forwards from the 2026-05-26 build session (`memory/session_state_2026-05-26.md`).
Baseline going in: 218 pass, 1 skip. Ending: 234 pass, 1 skip.

### ✅ Generator handling in `@with_provenance`

**Found**: torch.Generator was serialized via `default=str`, producing
a memory-address-based repr. Two Generators with the same seed got
different hexes — silent reproducibility break. Replay was also
broken: it would have tried to call ops with a string in place of a
Generator.

**Fix**: `_canonicalize_value(g)` returns `{__torch_generator__: True,
seed: g.initial_seed(), device: str(g.device)}`. Replay calls
`_restore_value(...)` to rebuild the Generator from this dict.

**Tests added**: `TestGeneratorCanonicalization` (4 tests covering
same-seed→same-hex, different-seed→different-hex, params shape,
replay round-trip).

### ✅ Var-args rejection in `@with_provenance`

**Found**: Decorator docstring claimed it rejected `*args/**kwargs`
but had no actual check; `inspect.signature.bind()` happily accepts
them and silently produces wrong hexes (tensors hidden inside the
var-args tuple aren't picked up as inputs).

**Fix**: Explicit `TypeError` raised at decoration time when a
`VAR_POSITIONAL` or `VAR_KEYWORD` parameter is detected.

**Tests added**: `TestVarArgsRejection` (2 tests).

### ✅ id-collision in `_tensor_id_to_hex` *(uncovered during pass)*

**Found**: `_tensor_id_to_hex` was keyed by `id(tensor)`. Python
recycles ids after GC, so a fresh tensor allocated at the same memory
address as a dead tensor would inherit the dead tensor's hex.
Demonstrated with four iterations of `laplacian.combinatorial(A + i*0.1)`:
two of four ids collided, dropping `len(reg)` from the expected 4
to 3 with wrong input chaining on the survivors.

**Fix**: Store `(weakref.ref(t), hex_id)` instead of just `hex_id`.
On lookup, return None if the weakref no longer resolves to the
same tensor (forcing fall-back to content hashing).

**Tests added**: `TestTensorIdReuse.test_id_reuse_does_not_corrupt_hex`.

### ✅ Tensor-cache bound

**Found**: `cache_tensors=True` stored every output indefinitely
— unbounded growth. For long pipelines or training loops this OOMs,
especially on GPU.

**Fix**: Added two new params to `record()`:
- `max_cache_size: Optional[int] = None` — FIFO eviction once the
  cache exceeds the cap.
- `cache_ops: Optional[Iterable[str]] = None` — selective caching by
  op_id; lets you keep, say, every Laplacian without caching every
  intermediate matmul.

Implemented via `OrderedDict` for O(1) FIFO eviction.

**Tests added**: `TestTensorCacheBounds` (4 tests).

### 📝 Disconnected-graph Ricci curvature

**Found**: For two nodes in different components, the all-pairs κ
returns ≈ 1 (high curvature), not 0. This is because the inflated
shortest-path distance (via `DISCONNECTED_DISTANCE_MULTIPLIER = 1000`)
appears in the denominator of `1 - W₁/d_G`, and W₁ is dominated by
the within-component transport cost which is small relative to the
inflated d_G. So κ → 1 for cross-component pairs.

This is HARMLESS for the Ricci-flow primitives (they mask updates by
edge presence) but SURPRISING for direct consumers of κ — the
naive expectation is "disconnected → 0 curvature".

**Action**: Documented in the `ollivier_ricci_curvature` docstring
("Disconnected components" section). Added 4 regression tests under
`TestDisconnectedComponents` pinning the observed behavior so future
refactors don't silently drift.

### ✅ HOSVD vs tensorly tolerance

**Found**: The comparison test was using `tensorly.tucker(...,
n_iter_max=1, tol=1.0)` which still runs one ALS step. Our pure-HOSVD
reconstruction was systematically ~5% worse — but only because
tensorly was doing more work, not because our HOSVD was wrong.

**Fix**: Switched to `n_iter_max=0` (which tensorly accepts and treats
as pure-HOSVD-no-ALS). Our reconstruction now matches tensorly's to
floating-point precision (relative diff < 1e-10), and the test
asserts that tight tolerance.

### ✅ Sinkhorn convergence plateau

**Found**: On wide-cost-range inputs, log-domain Sinkhorn plateaus
near a partially-converged state for hundreds of iterations
(asymmetry ~1.75e-2 from iter 100 to iter ~500), then snaps into the
symmetric basin around iter ~1000. Fixed `n_iter` either pays for
unnecessary work or stops short of convergence.

**Fix**: Added `tol: float = SINKHORN_TOL_DEFAULT` (= 1e-9) to
`ollivier_ricci_curvature` and plumbed `sinkhorn_tol` through both
`discrete_ricci_flow` and `ricci_flow_with_surgery`. Sinkhorn now
early-stops when `max|Δ log_u| < tol`. Bumped op_versions to 0.2.

**Tests added**: `test_symmetric_weighted_with_generous_budget`
verifying machine-precision symmetry with `n_iter=5000, tol=1e-12`.

---

## Pass 2 — 2026-05-27 (performance + deep scrutiny)

Driven by three parallel code-reviewer agents (provenance, discrete_geometry+
spectral, manifolds+algebra+tensor_calculus) plus a baseline benchmark
sweep (`notes/benchmark_baseline.md`). Started: 234 pass. Ending: 266 pass,
all primitives audit-clean.

### ✅ SPD numerical safety — cone-boundary inputs

**Found**: `_sqrt_and_inv_sqrt` had no eigenvalue clamp; `eigh` of a
nearly-singular SPD matrix can return slightly-negative eigenvalues
from float error, which `torch.rsqrt` then turns into `inf` and
propagates everywhere downstream. `exp` and `log` also did not
symmetrize their outputs — repeated retractions accumulated tiny
asymmetry and drifted off the SPD cone.

**Fix**: clamp eigenvalues to `torch.finfo(dtype).tiny` before
`sqrt`/`rsqrt`. Symmetrize the final `S_sqrt @ ... @ S_sqrt` outputs
of `exp` and `log`. Added the same clamp on `distance`'s
`eigvalsh(whitened)` output (which would NaN through `log(0)`).

**Tests added**: `TestNumericalRobustness` (4 tests covering n=1,
near-singular, exact-zero eigenvalue, exp/log output symmetry).

### ✅ Laplacian `_safe_inv` / `_safe_inv_sqrt` — autograd compatibility

**Found**: Both used boolean-mask indexed assignment (`out[mask] = ...`),
which produces hard-zero gradients on the masked branch instead of
the smooth `lim rsqrt → 0` gradient — and breaks `torch.vmap` /
`torch.compile` due to data-dependent control flow.

**Fix**: replaced with `torch.where(d > 0, rsqrt(d.clamp(min=tiny)),
zeros_like(d))`. Same numerical behavior, fully vectorized,
vmap/compile compatible.

### ✅ Ricci: asymmetric-input warning + Floyd-Warshall host sync

**Found** (1): `ollivier_ricci_curvature` claimed symmetric A but
didn't validate. A directed graph passed in error silently produced
meaningless curvature. **Fix**: `torch.allclose(A, A.mT, atol=1e-9)`
check with a `UserWarning` on mismatch (raises if you opt to upgrade
warnings).

**Found** (2): `_shortest_path_distances` had `if finite_mask.all():
return D` — forced a GPU→CPU sync on every call, the common fast
path. **Fix**: always run the `torch.where`-based replacement; it's
a no-op when all distances are finite.

**Found** (3): Surgery threshold uses pre-normalization initial mean
as a fixed reference while comparing post-normalization weights —
internally consistent but not obvious. **Fix**: documented in the
`ricci_flow_with_surgery` docstring.

### ✅ Provenance: `_canonical_params` was dead code

**Found**: defined at module level but only the inline wrapper logic
was used. Maintenance hazard — divergence between the dead helper and
the live serialization. **Fix**: deleted.

### ✅ Provenance: `replay(final_hex=…)` multi-output handling

**Found** (by test addition, not reviewer): The early-stop branch
in `replay()` only handled single-tensor outputs. For a multi-output
node (e.g. truncated_svd returns U/S/Vt under `hex:0`, `hex:1`,
`hex:2`), `final_hex=svd_hex` would key into `shadow[svd_hex]` which
doesn't exist. KeyError. **Fix**: when `final_hex` matches the base
of a multi-output node, return every `hex:i` entry; when it specifies
a single output (`hex:i`), return just that.

### ✅ Audit: precompile regexes, remove dead `_posited.py` exclude

Minor maintenance: `DERIVED_PATTERNS` rebuilt the same regex objects
on every literal scan. `EXCLUDE_FILES` still listed `_posited.py`
from a prior incarnation. `exclude_dirs` was duplicated across two
functions. Cleaned up.

Also: **added 18 audit smoke tests** (`tests/test_audit.py`) — the
audit IS the project's CI gate, and a regression would silently
break either by letting magic numbers through or by blocking
legitimate code. Tests cover: allowed-literal pass-through, arbitrary
literals flagged, derived-pattern recognition, catalog cross-
reference, AST coverage of function defaults / keyword args, path
handling (excludes, syntax errors, audit-self-exclusion), and
Boolean ignoring.

### ✅ Performance — Sinkhorn pair-tiling

**Problem**: `_batched_sinkhorn_w1` materialized a `(B, n², n)`
source/target structure plus a `(B, n², n, n)` per-iter broadcast —
128 MB per iter at n=64 in float64. Ollivier at n=64 took **22
seconds** in baseline.

**Fix**: tile over the n² pairs with `tile_size=SINKHORN_TILE_DEFAULT
= 256`. Source/target gathered per-tile via advanced indexing so the
full materialization never happens; inner broadcast capped at
`tile × n × n × 8` bytes (≈16 MB at n=256).

Plumbed `tile_size` through `ollivier_ricci_curvature`,
`discrete_ricci_flow`, `ricci_flow_with_surgery`. Bumped op_versions
to 0.3.

**Measured impact (CPU, float64):**
| n  | baseline | tiled  | tile + sync8 | total speedup |
|----|----------|--------|--------------|---------------|
| 16 |  34 ms   | 35 ms  |  20 ms       | 1.7×          |
| 32 | 273 ms   | 722 ms | 291 ms       | ~same         |
| 64 | 22.6 s   | 9.3 s  |  3.7 s       | **6.2×**      |

On a memory-bound GPU the wins are expected to be larger.

### ✅ Performance — Sinkhorn host-sync cadence

**Problem**: convergence-check `.item()` per Sinkhorn iter forces a
GPU→CPU sync, throttling kernel queues. With tiling there are now
n²/tile iters per tile × n_iter — even more syncs.

**Fix**: check convergence every `SINKHORN_SYNC_EVERY_DEFAULT = 8`
iters instead of every iter. At most 7 extra iters past true
convergence; 8× fewer syncs in steady-state.

### ✅ Performance — FixedRank.retraction auto-randomized

**Problem**: retraction did full SVD then sliced top-r. At
1024×1024 r=32 (r/min = 3.1%), benchmark showed 193 ms for the full
SVD when randomized SVD would be ~10 ms.

**Fix**: added `retraction_mode` constructor param ("auto" / "exact"
/ "randomized"). "auto" switches to our `truncated_svd(..., mode=
"randomized")` (Halko-Martinsson-Tropp with oversample=5, n_iter=2)
when `r ≤ 0.25 · min(m, n)`. Threshold derived from benchmark data;
cataloged as `RETRACTION_RANDOMIZED_THRESHOLD`.

Note: `torch.svd_lowrank(M, q=r)` directly (no oversample) was the
first attempt — produced 42% relative error vs exact, unacceptable
for an optimizer step. The current implementation uses our
`truncated_svd` to ensure proper oversampling.

**Measured impact (CPU, float64):**
| m × n × r        | exact   | auto    | speedup |
|------------------|---------|---------|---------|
| 64 × 64 × 8      | 0.31 ms | 0.31 ms | 1.0×    |
| 256 × 256 × 16   | 4.93 ms | 1.28 ms | 3.9×    |
| 1024 × 1024 × 32 | 132 ms  | 10.2 ms | **12.9×**|

### ✅ Performance — SPD whitening cache for chained ops

**Problem**: `exp(S, V)` and `log(S, T)` each call
`_sqrt_and_inv_sqrt(S)` internally. When the caller chains
`exp/log/distance` at the same S, that's 2-3 redundant eighs.

**Fix**: added `precompute_whitening(S)` returning `(S_sqrt,
S_inv_sqrt)` once, plus an optional `whitening=…` kwarg on each
geodesic op. Backwards-compatible; default behavior unchanged.

### ✅ Test coverage — provenance edge cases

Reviewer A flagged 5 untested paths. Added:
- `TestReplayMultiOutput`: replay through a tuple-returning parent
  feeding a child that consumes both outputs (verified the topological
  sort doesn't double-count).
- `TestReplayFinalHex`: both the base-hex-of-multi-output case and the
  specific-output (`hex:i`) case.
- `TestParentsMultiOutput`: `parents()` correctly strips `:i` from
  input_hexes.
- `TestTensorIdReuse`: regression for the id-collision bug fixed in
  Pass 1.

### ✅ CPU/GPU parity infrastructure

Added `tests/test_device_parity.py` with two suites:
- `TestParityOnGpu`: skipped when no GPU available; on CUDA / ROCm
  machines, asserts each primitive produces matching numerical
  results CPU vs GPU.
- `TestDeviceMovability`: parameterized over available devices;
  catches accidental `.cpu()` / `.cuda()` calls that would break
  device-portability.

Currently runs CPU-only (no GPU in environment); 9 GPU tests skip
silently. Infrastructure ready for ROCm / CUDA runs.

---

## Pass 3 — 2026-05-27 (new primitives + targeted perf)

Three parallel `feature-dev:code-reviewer` agents on the 8 primitives
added across two work sessions (Forman-Ricci, magnetic Laplacian, heat
kernel, effective resistance, commute time, diffusion map, Lanczos,
info_geometry). Starting: 388 tests. Ending: 400 tests, audit clean.

### ✅ Correctness — KL categorical: q=0 returns +inf when p>0

**Found**: `kl_divergence_categorical` clamped both `p` and `q` to
`1e-9` inside the log. When `p_i > 0` and `q_i = 0`, the correct
value is `+inf` (Gibbs's inequality bound; KL diverges when
`supp(p) ⊄ supp(q)`). The symmetric clamp returned `p_i · log(p_i/1e-9)
≈ 20·p_i` — finite garbage. Verified empirically: `KL([1,0] ‖ [0,1])`
returned ~20.7 instead of inf.

**Fix**: replace the symmetric clamp with a `torch.where(q > 0, log,
-inf)` branch. The `p > 0` outer guard preserves the `0·log(0/q) = 0`
convention. Documented the autograd hazard near zero in the docstring.

### ✅ Correctness — Forman-Ricci self-loop contamination

**Found**: `forman_ricci_augmented`'s triangle count uses
`(edge_mask @ edge_mask)`. A self-loop at node `i` (i.e. `A[i,i] ≠ 0`)
makes `edge_mask[i,i] = 1`, so length-2 walks `j → i → i → k` get
counted as triangles for every edge incident to `i`. Inflated κ by 2
per self-loop per endpoint. Also: the simple form's `degree`
calculation counted the self-loop as a regular edge.

**Fix**: drop the diagonal of `A` upfront in both forms via a shared
`_drop_self_loops` helper. Forman-Ricci is defined on simple graphs;
this matches the literature convention and gives invariant outputs.

### ✅ Correctness — magnetic Laplacian crashes on complex input

**Found**: passing a complex-typed `A` to `magnetic.combinatorial(A,
q=0.25)` crashed inside `torch.complex(cos, sin)` because cos/sin on
complex inputs are complex, and `torch.complex(...)` requires real
operands. The q=0 fast path masked this for that one case only.

**Fix**: shared `_validate_real_adjacency` helper rejects non-real
dtypes up front with a clear error. The phase factor `exp(i·2π·q·
(A − A^T))` is mathematically defined only for real `A`.

### ✅ Correctness — heat_kernel coefficient transfer

**Found**: `torch.as_tensor(numpy_array, device='cuda')` is unreliable
on some backends (CPU-only numpy bindings can't be placed on GPU
directly). On the CPU test path this worked; on CUDA/ROCm it would
fail at first invocation.

**Fix**: `torch.from_numpy(...).to(device=L.device, dtype=L.dtype)` is
the robust pattern. Same semantics, GPU-safe.

### ✅ Correctness — diffusion_map silent on disconnected graphs

**Found**: `diffusion_map` always drops index 0 (one null eigenvector)
even on graphs with multiple connected components. For `c` components
the indices `0..c-1` are all null modes; dropping just one leaves
`c-1` degenerate constant-on-component vectors in the output, looking
like valid slow modes (μ_j ≈ 1) but carrying no geometric information.

**Fix**: detect `n_near_zero` eigenvalues at function entry; emit a
`UserWarning` when more than one is present, telling the caller to
mask by component before interpreting cross-component distances.

### ✅ Correctness — KL Gaussian missing `mu_q` shape check

**Found**: `kl_divergence_gaussian` validated `Sigma_p` and `Sigma_q`
against `d = mu_p.shape[-1]` but not `mu_q`. A caller passing a
shape-mismatched `mu_q` got an opaque error deep in `cholesky_solve`.

**Fix**: explicit `ValueError` upfront, mirroring the existing
Sigma checks.

### ✅ Correctness — Lanczos `assert` survives `python -O`

**Found**: `_build_tridiagonal` used `assert len(betas) == n - 1`. The
`-O` flag strips asserts; a corrupted call would silently produce a
wrong-sized tridiagonal and garbage Ritz pairs.

**Fix**: replaced with `raise ValueError(...)`.

### ✅ Performance — Lanczos `V_stack` pre-allocation

**Found**: the full reorthogonalization loop did
`torch.stack(V_cols, dim=-1)` at every iteration, copying j+1
columns each time. Over `m` iterations this is `O(m² · n)` extra
allocation traffic on top of the necessary work.

**Fix**: pre-allocate `V = torch.empty(*batch, n, n_iter + 1, ...)`
once and slice `V[..., :j+1]` inside the loop — `O(1)` slice, no copy.

### 🟢 False positive — `effective_resistance` all-zeros graph

Reviewer claimed an all-zeros adjacency would produce nonsense (1/tiny
amplification). Empirically the `where(eigvals > threshold, ...)`
branch correctly returns 0 because `0 > 0` is False. Verified by
running on `A = zeros(1, 5, 5)`; the output is exactly `zeros(1, 5, 5)`.
Added a regression test that pins this down.

### Test additions

- `kl_divergence_categorical`: `test_zero_in_q_with_support_in_p_returns_inf`
- `kl_divergence_gaussian`: `test_mu_q_shape_mismatch_raises`
- `forman_ricci_augmented`: `test_self_loop_does_not_inflate_augmented`
- `magnetic.combinatorial` + `magnetic.symmetric_normalized`:
  `test_rejects_complex_input`
- `diffusion_map`: `test_warns_on_disconnected_graph`
- `heat_kernel_chebyshev`: `test_rejects_signal_with_wrong_node_dim`
- `lanczos_eigsh`: `test_batch_zero`, `test_full_iter_basis_orthonormal_tight`
- `effective_resistance`: `test_batch_zero`, `test_n_one`,
  `test_all_zeros_graph`

### Benchmarks (CPU baseline, float64)

| Primitive | n=64 | n=256 | n=1024 |
|---|---|---|---|
| `forman_ricci_simple` | 0.2 ms | 3.0 ms | 47 ms |
| `forman_ricci_augmented` | 0.4 ms | 1.9 ms | 94 ms |
| `magnetic.symmetric_normalized` | — | 2 ms (n=512) | — |
| `heat_kernel_chebyshev` (dense, K=30) | 2.3 ms | — | 56 ms (n=512) |
| `heat_kernel_chebyshev` (signal k=4) | 0.9 ms | 3.0 ms | 12 ms |
| `effective_resistance` | 0.9 ms | 3.4 ms | 17.7 ms (n=512) |
| `diffusion_map` (k=8) | 0.4 ms | 3.4 ms | 66 ms |
| `lanczos_eigsh` (k=1, n_iter=30) | — | — | 11 ms |
| `torch.linalg.eigvalsh` (reference) | — | — | 47 ms |

Lanczos vs dense eigh crossover: ~n=512. Below that, the Krylov
overhead dominates; above, Lanczos wins (4.2× at n=1024).

Heat kernel signal-path is 4-5× faster than building the dense matrix
when k_signal ≪ n — the expected win.

---

## Pass 4 — 2026-05-27 (Tier 1 additions)

Three parallel `feature-dev:code-reviewer` agents on the 4 new modules
introduced across Tier 1 (Riemannian optimizers, simplicial complexes,
Hodge Laplacians + sparse Lanczos, persistent homology). Starting:
457 pass. Ending: 461 pass, audit clean.

### ✅ Performance — Z/2 reduction essential-bar pass was O(n²)

**Found**: `_reduction.py:157` used `j in pivot_map.values()` to check
whether column `j` was a death column. The `.values()` view scan is
O(n) per call, making the outer essential-bar loop O(n²) in
`n_total`. For a typical 50-point cloud at max_dim=2, n_total is
~5000; the linear scan becomes the bottleneck of PH well before
the Python-set XOR does.

**Fix**: maintain a `death_columns: set[int]` alongside `pivot_map`
during the reduction (added immediately when `pivot_map[low] = j`
fires). The essential-bar check becomes an O(1) set lookup, restoring
the O(n) outer loop.

### ✅ Performance — DenseSimplicialComplex.boundary inner-loop hoist

**Found**: `complex.py:158` called `torch.where(prev_valid[b])[0]`
INSIDE the triple-nested `for b, j, i` loop. The result is constant
within a given `b`; recomputing it `n_k · (k+1)` times per batch is
gratuitous work + (line 159) a `.item()` host sync per call on GPU.

**Fix**: hoist `valid_prev_rows = torch.where(prev_valid[b])[0]` to
the per-batch setup block and convert to a Python list once with
`.tolist()`, eliminating both the redundant tensor ops and all
`.item()` syncs from the hot path. Same treatment applied to
`k_cols`.

### ✅ Correctness — `_scale` tuple branch was dead-code trap

**Found**: `sgd.py:54-62` had a `_scale` helper with a tuple-handling
branch ("preserve the structure"). The docstring is explicit that
`ambient_grad` is always a single `torch.Tensor` (for FixedRank, the
ambient `(B, m, n)` form per Vandereycken 2013). The tuple branch
never fires under the documented API; a future caller misled into
passing a tuple would crash deep inside `FixedRankManifold.projection`.

**Fix**: removed the `_scale` helper entirely; the step is now just
`-lr * ambient_grad` inline. Updated docstring to explicitly state
`ambient_grad` is a single tensor.

### ✅ Code quality — sparse Hodge hardcoded float64

**Found**: `_hodge_sparse` hardcoded `dtype = torch.float64`.
`SparseSimplicialComplex` has no `dtype` attribute, so the function
had no other source of information; but callers expecting float32
got silently promoted to float64.

**Fix**: added a `dtype` parameter (default `torch.float64`) to
`_hodge_sparse`, propagated through to the boundary-matrix
construction. The public `hodge_laplacian` still picks float64 by
default per the `SparseSimplicialComplex` convention; callers
needing float32 can call `_hodge_sparse` directly or add a public
override in v2.

### ✅ Conventions — `@with_provenance` decoration

**Found**: three public top-level primitives in the new code lacked
`@with_provenance` decoration:
- `simplicial.vietoris_rips_sparse`
- `simplicial.vietoris_rips_dense`
- `simplicial.pairwise_distances`

Per `CONVENTIONS.md` §7, all public top-level primitives must be
decorated.

**Fix**: decorated all three. Bumped no op_versions (these are new
primitives at v0.1).

**Carve-out documented**: `optimization.riemannian_sgd_step` is
deliberately NOT `@with_provenance`-decorated. Two reasons:
1. The `manifold` parameter is a class instance — `default=str` JSON
   serialization produces a memory-address-based string, breaking the
   hex stability contract (same issue we hit with `torch.Generator`
   before canonicalization).
2. The function is hot-loop (fires per optimizer iteration); the
   decoration overhead would dominate the actual math.

Both rationales documented inline in the function docstring. Future
v2 work could add `manifold` canonicalization analogous to
`torch.Generator`, but for v1 the carve-out is clean.

### ✅ Conventions — missing `References:` sections

**Found**: per-function `References:` blocks missing on the three
public simplicial primitives flagged above, plus `boundary()`
methods on both complex classes. Module-level docstrings reference
the right papers but CONVENTIONS.md §6 + §9.1 require per-function
references.

**Fix**: added `References:` blocks to each affected function.

### ✅ Tests — missing B=0 and max_radius=0 cases

**Found**: `RiemannianSGD` had no `B=0` test (CONVENTIONS.md §1.1
requires `B ∈ {0, 1, > 1}`). `persistence_diagrams` had no
`max_radius=0` test (the "no edges → all H_0 bars essential" case).

**Fix**: added `TestBatchZero` (SPD + FixedRank) for the optimizer,
and `TestEdgeCases.test_max_radius_zero` + `test_batch_zero` for PH.

### 🟢 Verified correct (reviewer cleared on second look)

- `_filtration.py` sort cascade: produces correct lex order via
  stable-sort passes.
- `_reduction.py` essential-bar identification: `columns[j]` empty
  check is correct and complete.
- `_reduction.py` Z/2 sign-drop: intentional, sign meaningless mod 2.
- `_persistent_h0` `n_surviving` accounting under `max_radius < inf`:
  components that never merge get genuine infinite bars.
- Vietoris-Rips edge enumeration via `torch.argwhere(adj_mask)`: the
  strict upper-triangular mask precludes duplicates.
- Riemannian SGD sign convention: `_scale(g, -lr)` then project then
  retract is correct for `g = ambient_gradient`.
- `betti_numbers` sparse-path's no-padding-subtraction: correct
  because sparse has no padding to subtract; inline comment added.

---

## Open follow-ups (not yet acted on)

### 🔬 Randomized SVD silent fallback to exact mode

When `r + oversample > min(m, n)`, `_randomized_svd` falls back to
`truncated_svd(M, r, mode="exact")` — but the OUTER recorded node
still says `mode="randomized"` and carries a `generator=...` param
that the inner exact call ignores. The outer node's hex therefore
depends on a generator that never affects the output. Either:
- Drop the silent fallback and raise; force the caller to choose a
  mode that fits the matrix.
- Or down-rank `mode` in the recorded params when falling back.

### 🔬 `_tensor_id_to_hex` unbounded growth

The weakref-keyed map is correct but still grows monotonically across
the recording session. Stale entries (those whose weakref has
expired) are evicted only when their slot happens to be re-looked-up.
For long recordings this leaks memory holding dead weakrefs. Fix:
periodic sweep or `WeakValueDictionary`-style indirection. Low
priority — leak is bounded by total tensors-allocated, not by GPU
memory.

### 🔬 Class-method provenance

Manifold methods (`FixedRankManifold.retraction` etc.) aren't
decorated yet. Need a class-aware variant of `@with_provenance` that
hashes instance state (m, n, r, device, dtype) into params. Listed
in `memory/next_session_priorities.md` §5.

### 🔬 Sparse / edge-only Ollivier curvature

All-pairs computation is O(n²) edges' worth of Sinkhorn problems even
when the graph is sparse. For large graphs (n > ~500) we should have
a sparse variant that only computes κ for actual edges. Listed in
the `ollivier_ricci_curvature` docstring as "planned".

### 🔬 Sinkhorn `.item()` host sync per iter

The new tol-based early stop calls `delta.item()`, which forces a
host sync each iteration. On GPU this throttles throughput. Two
mitigations:
- Sync every K iterations instead of every iteration.
- Use a tensor-side "did we converge?" boolean and break only when
  it has been true for the last few iterations.

Currently OK because Ollivier is O(n³)-dominant; revisit when the
sparse variant lands.

---

## 2026-10-25: 0.8.0 correctness and documentation review

This entry supersedes current-behavior claims in older passes above; their
observations remain dated records. Checked the implementation lane findings,
31 shared before/after repros, release commits, exported API signatures, and
the README example against the current source. Finding links below point to
the release record, not to local review-report files.

**Found and fixed:**

- [B01/B02](../CHANGELOG.md#v080-b01), B18/B24: projected ambient gradients
  were not Riemannian gradients for non-Euclidean metrics; repeated SPD
  eigenvalues and fixed-rank singular values broke first derivatives.
  Metric conversion and corrected spectral backwards repair the checked cases.
- [B03](../CHANGELOG.md#v080-b03), [B03b](../CHANGELOG.md#v080-b03b),
  [B04–B06](../CHANGELOG.md#v080-b04): SO(3) small-angle behavior and the
  zero-exp Jacobian were wrong; heat-kernel subtraction, quadrature tails,
  broadcast derivatives, and small-radius evaluation gave incorrect values.
  The heat paths now match the stated independent references and run under
  `no_grad`; high-dimensional and extreme-input accuracy is not certified.
- [B09/B10](../CHANGELOG.md#v080-b09),
  [B11/B12](../CHANGELOG.md#v080-curvature-derivatives),
  [B13](../CHANGELOG.md#v080-b13): curvature membership/sign branches,
  zero-curvature derivatives, transport, and large-radius Lorentz evaluation
  were repaired. Prior “κ crosses zero” and “arcsinh is finite everywhere”
  claims were stronger than the shipped implementation supported.
- [B14](../CHANGELOG.md#v080-b14), [B15](../CHANGELOG.md#v080-b15),
  B19–B23, [L1/L2 C01/C02/D01/D02](../CHANGELOG.md#v080-l1-l2-details):
  Fréchet backtracking, KL reference values, empty batches, validation,
  product dtype propagation, sampler signs, Cholesky solves, and mathematical
  docstrings were corrected.
- [C01/C01b](../CHANGELOG.md#v080-c01),
  [C06b](../CHANGELOG.md#v080-c06b), C14/C20: persistence emitted killed
  classes as essential; a filled triangle had an H1 bar and a 40-point circle
  had 361. They now have none and one respectively. Rank tolerances repair
  float32 Betti numbers. Complex construction and boundary preparation use
  tensor operations; reduction itself remains sequential.
- [C02/C03](../CHANGELOG.md#v080-c02),
  [C07/C08](../CHANGELOG.md#v080-c07), C11–C18/C21–C23:
  eigensolver breakdown/backward, sparse ordering, complex SVD, solver
  scaling, empty inputs, validation, and backend boundaries were corrected.
- [C05/C06/C10](../CHANGELOG.md#v080-c05),
  [C19](../CHANGELOG.md#v080-c19): signed diffusion powers, stationary
  normalization, resistance/component volume, and MSGNN magnetic phases now
  match their stated references. Cross-component resistance and commute time
  are `inf`; isolated diagonals remain zero.
- [C04/C09/C10/C23 (graphs)](../CHANGELOG.md#v080-graph-contracts):
  Sinkhorn now checks both marginals and warns when unconverged; the default
  failing repro value did not become the LP answer. Augmented Forman is
  binary-only, flow drops loops, and cross-component Ollivier pairs are
  masked to zero. Sheaves reject invalid maps and indices.
- [P D01–D07/D10](../CHANGELOG.md#v080-provenance): provenance was removed
  with its faulty replay, hashing, inspection, and recording paths.
  [P D08/D09/D11](../CHANGELOG.md#v080-audit): the relocated audit fails
  on unreadable/missing inputs and closes literal exemptions; removing
  decorators restores callable types. The audit remains a static heuristic.
- Test review E1-01–E1-11 and E2-01–E2-08 cut redundant/tautological checks
  and repaired retained reference, gradient, full-persistence-multiset, and
  distinct-member batch assertions. Passing counts are not a quality metric.

**Documentation:** Replaced the promotional README with a batch example,
limitations, module map, and historical measurement links. Regenerated the
public inventory, preserved convention section numbers, and corrected all
C1–C7 research statuses in the [claims catalog](research_claims_catalog.md).
The catalogue now distinguishes mathematical identities, sampled numerical
evidence, and unfinished constructions.

**Left in progress:** [B08/B17](../CHANGELOG.md#v080-heterogeneous)
heterogeneous curvature is an experimental nonmetric `dissimilarity`;
experimental eigenmaps still admits collapse. Sparse eigensolvers are
nondifferentiable, SPD/fixed-rank custom backwards are first-order, and
persistence reduction uses sequential per-member work and host metadata.
The explorer and its screenshot are integrated by the explorer lane;
this documentation pass does not certify its UI or publish new timings.
