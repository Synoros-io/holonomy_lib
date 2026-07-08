# Changelog

All notable changes to `holonomy_lib` are documented here. Format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
version numbers follow [Semantic Versioning](https://semver.org).

## [0.7.0] - 2026-07-08

### Added

- `manifolds.comparison.sn_kappa(kappa, t)`: the generalized sine `sn_κ`
  exposed as a public per-direction primitive (previously internal), for
  callers building sign-split / masked anisotropic fluxes from per-direction
  curvatures. Petersen (2016) §7.1.
- `sheaf.sheaf_coboundary_sparse` / `sheaf.sheaf_laplacian_sparse`: the sheaf
  coboundary `δ` and Laplacian `L_F = δᵀδ` as coalesced sparse-COO tensors —
  nnz scales with the edge count, covering sheaves past the dense 2 GiB cap.
  Backport of synoros-substrate's validated `sheaf_scale.py` (generic parts);
  Laplacian verified bit-identical against substrate's scipy assembly.
- `sheaf.sheaf_bottom_spectrum(sheaf, k, ...)`: the `k` smallest NONZERO
  eigenpairs of the sparse `L_F` plus node offsets — the scalable spectral
  read. Kernel (global sections) dropped at `max(diag L)·1e-9`; new cataloged
  constant `SHEAF_KERNEL_HEADROOM = 16`.
- `algebra.lanczos_eigsh(which="SA")` now accepts sparse input (was
  `NotImplementedError`), dispatching to new `algebra.eigsh_sparse.
  eigsh_sparse_sa`: hybrid backends — SciPy `eigsh` shift-invert (CPU,
  `n ≤ 8000`), SciPy LOBPCG + `spilu` (CPU, larger), `torch.lobpcg` (CUDA).
  New `solver=` parameter; `lanczos_eigsh` op_version 0.3 → 0.4. New cataloged
  constants `SPARSE_SA_ITER_SOLVER_N`, `SPARSE_SA_SHIFT_INVERT_POLE`,
  `SPARSE_SA_SPILU_LOBPCG_KNOBS`.
- `algebra.solvers.cg`: conjugate gradient for SPD systems — batched dense,
  sparse (whole batch in one sparse matmul), or callable-operator form;
  derived defaults (`rtol = √eps(dtype)`, `maxiter = n`); optional
  preconditioner. Hestenes-Stiefel (1952).
- `algebra.solvers.lsqr`: batched damped least squares via Golub-Kahan
  bidiagonalization (Paige-Saunders 1982); per-element convergence frozen on
  device every iteration (required for correctness past Krylov breakdown on
  inconsistent systems), host sync on the `SOLVER_SYNC_EVERY_DEFAULT = 8`
  cadence (new cataloged constant).
- `convex` (new package): problem-level convex optimization interfaces,
  entry gated on named consumers. Seeded with
  `convex.regularized_lstsq(A, b, lam, method=...)` — Tikhonov/ridge least
  squares via `"cholesky"` / `"cg"` / `"lsqr"`, `method="auto"` dispatch
  benchmark-derived (dense → cholesky, sparse → lsqr; no size threshold —
  see `notes/benchmark_2026-07-08_solvers.md`).

### Changed

- `sheaf`: the heterogeneous coboundary assembly is vectorized (COO index
  arithmetic + one `index_put_`) instead of a Python loop over edges; outputs
  bit-identical, the old loop kept in tests as the regression oracle. The
  shared `_coboundary_coo` builder also feeds the sparse path.
- `manifolds.comparison.model_ball_volume` and `hyperbolic.heat_kernel` now
  draw Gauss-Legendre nodes from a cached master copy
  (`_quadrature.gauss_legendre_nodes`) instead of recomputing via scipy on
  every call (scipy's `roots_legendre` does not cache; the "cached at the
  SciPy level" comment was incorrect).

## [0.6.0] - 2026-05-29

### Changed

- Release milestone marking the public preprint "Meaning on a Video Card:
  Scalable, Lossless Geometric Primitives" (Vaught, 2026), which documents the
  library's geometric and spectral primitives. No committed API changes since
  `0.5.2`; this version tags the codebase the paper describes.

## [0.5.2] - 2026-05-28

### Added

- `manifolds.comparison.model_sphere_area(kappa, N, r)` and
  `manifolds.comparison.model_ball_volume(kappa, N, r)`: model-space
  (constant-curvature space form `M_κ^N`) geodesic sphere area and ball volume,
  for real (non-integer) effective dimension `N` and signed curvature `κ`, via
  the Bishop-Gromov `sn_κ` volume element. Area is closed-form; the volume
  integral uses Gauss-Legendre quadrature (exact for `κ = 0`). These are the
  comparison reference for curvature- and dimension-dependent volume / flux
  laws. New cataloged constant `COMPARISON_BALL_VOLUME_QUADRATURE_NODES = 64`.
- `spectral.spectral_dimension(eigenvalues, t)`: spectral dimension `d_s` of a
  Laplacian spectrum via the heat-kernel return-probability decay exponent
  (`p(t) ~ t^(-d_s/2)`), fit as the log-log slope over a caller-supplied time
  window. Non-integer (fractal / continuum) dimensions supported; no new
  constants. Rammal-Toulouse (1983); Durhuus-Jonsson-Wheater (2007).
- `sheaf.HeterogeneousGraphSheaf`: cellular sheaf with per-node (heterogeneous)
  stalk dimensions, i.e. each node's stalk dim is its own; edges carry ragged
  `(d_e, d_v[u])` restriction maps. `sheaf_coboundary` / `sheaf_laplacian` /
  `sheaf_dirichlet_energy` dispatch to it; the uniform `GraphSheaf` path is
  unchanged (reduction verified: uniform dims + identity maps reproduce
  `GraphSheaf.trivial`'s Laplacian exactly). The v2 the module docstring
  anticipated. Hansen-Ghrist (2019).
- Closed-form `H^7` heat kernel (`hyperbolic.heat_kernel._heat_kernel_unit_n7`):
  `k^7_t(r) = (4πt)^{-7/2} exp(-9t - r²/4t) · [r³ sinh²r + 6r²t sinh r cosh r +
  (8t² − 6t) r sinh²r + 12t²(r − sinh r cosh r)] / sinh⁵ r`, one corrected
  operator-chain step beyond the `H^5` form. Replaces the autograd recursion at
  n=7 (faster, more precise) and seeds the odd-n recursion at n ≥ 9 (cascade
  precision gain). Verified in `notes/verification/heat_kernel_n7_sympy.py`
  (closed form = recursion = operator chain to ~1e-13; heat-equation residual
  ~1e-12; r→0 limit `1 + 2t + 16t²/15`). New cataloged dim constant
  `_N_OPERATOR_CHAIN_N7_CLOSED = 7`; 5 new tests (closed-form/limit/backward,
  n7-vs-recursion unit check, and the n=9 reseeded-recursion residual).

### Changed

- Odd-n hyperbolic heat-kernel recursion now seeds from the `H^7` closed form
  (was `H^5`), so n ≥ 9 takes one fewer `torch.autograd.grad` step.

Test count: 1143 (v0.5.0) to 1187.

## [0.5.0] - 2026-05-28

The hyperbolic / pseudo-Riemannian / mixed-curvature manifold pass.
This release adds **five new manifold classes**, the
`holonomy_lib.hyperbolic` module of manifold-aware graph
operations, and a sweep of autograd-stability + correctness fixes
along the way. Test count went from 707 (v0.4.1) to **1143**.

### Summary at a glance

| New manifold | Geometry | Status |
|---|---|---|
| `LorentzManifold` | Hyperbolic — hyperboloid model `H^n_k` at curvature `k < 0` | Standard, fully validated |
| `KappaStereographicManifold` | Parametric κ ∈ R, spherical / Euclidean / hyperbolic interpolation. **Learnable κ with dynamic sign dispatch** (κ may cross 0 during training) | Standard model (Bachmann et al. 2020); learnable-κ extension |
| `LorentzianManifold` | Pseudo-Riemannian (1, n-1) signature Minkowski spacetime with causal structure + curvature tensors | Standard (MTW; O'Neill) |
| `ProductManifold` | Riemannian product of any of the above (mixed-curvature embedding) | Standard (Gu-Sala 2019; Skopek 2019) |
| `HeterogeneousKappaManifold` | Per-point κ; configurable pair-κ combiner | Closest prior art: GraphMoRE (AAAI 2025); continuous per-point κ is our research |

New module **`holonomy_lib.hyperbolic`** with four manifold-agnostic
graph primitives that compose with the above:

- `manifold_aware_inner(x, y, manifold)` — Riemannian inner via the
  tangent at origin
- `frechet_mean(points, manifold, weights, ...)` — Karcher iteration
- `hyperbolic_laplacian_eigenmaps(adjacency, manifold, ...)` — RSGD
  embedding under `Σ A_ij d_M(Y_i, Y_j)²`
- `hyperbolic_heat_kernel(t, distances, manifold, ...)` —
  dimension-dispatched heat kernel: Gaussian (n=1),
  Davies-Mandouvalos closed forms (n=3, 5), Gauss-Legendre quadrature
  on the integral form (n=2), spectral-shift-corrected recursion
  for all higher odd/even n

### Major correctness work along the way

- **Heat-kernel recursion bug found and fixed.** The
  `k^{n+2} = -(2π sinh r)^{-1} · ∂_r k^n` recursion that
  appears in some references is **missing a spectral-shift factor**.
  Correct form: `k^{n+2} = -exp(-n·t) / (2π·sinh r) · ∂_r k^n`.
  Caught by independent PDE-residual validation
  (`notes/validation/heat_kernel_findings.md`); residuals dropped
  from O(1) to FD noise floor.
- **`hyperbolic_laplacian_eigenmaps` silent NaN on dense graphs**
  (n_edges > ~500) fixed via per-node degree normalization
  (random-walk-Laplacian gradient) + fail-loud `RuntimeError` on
  divergence.
- **Autograd-finite gradients at every boundary input**
  (`d(x, x) = 0`, `log(x, x) = 0`, `exp(x, 0) = x`,
  `parallel_transport(x, x, v) = v`). Established `_safe_sqrt` /
  `_safe_sinhc` / `_safe_arcsinhc` idiom that confines boundary-
  singular ops to the masked-out branch of `torch.where`. Reused
  across all new manifolds.
- **Closed-form `n=5` heat kernel** for precision push — `~3
  orders of magnitude` tighter PDE residual than the
  autograd-recursion path; cascades to `n=7, 9, ...`.
- **Even n ≥ 4 heat kernel** now works via the corrected recursion
  seeded from the n=2 integral form (was previously raising
  `NotImplementedError`).

### Fixed (pre-release sweep)

- `LorentzianManifold.inner` added (was missing — would crash any
  manifold-generic primitive that took it). Returns signed Minkowski
  form for API parity; docstring explains the indefinite-metric
  semantics.
- `HeterogeneousKappaManifold.random_point` added (was missing —
  would crash `hyperbolic_laplacian_eigenmaps`).
- `LorentzianManifold.metric_tensor` `@with_provenance`-decorated
  (was the only undecorated op in that class).
- `_safe_atanhc` substitute-value docstring clarified: `0.5` is
  the **correct** choice (not the `1.0` used by other `_safe_*`
  helpers), because `arctanh(1) = ∞` blows up the formula branch's
  gradient and `torch.where`'s mask can't recover `0 · ∞ = NaN`.

### Notes / docs

- `notes/validation/` — five new validation reports:
  `heat_kernel_validation.py` + `heat_kernel_results.md` +
  `heat_kernel_findings.md`; `frechet_spherical_validation.py` +
  `frechet_spherical_results.md`; `autograd_safe_vs_geoopt.py` +
  `autograd_safe_results.md`; `cross_manifold_validation.py` +
  `cross_manifold_results.md`; `findings_summary.md`.
- `notes/research_open_questions.md` — explicit posture on what's
  standard prior art vs. our research.
- `tests/benchmarks/bench_hyperbolic.py` — wall-clock benchmarks
  for `hyperbolic_laplacian_eigenmaps` and `frechet_mean`.

### Added (Stage 4 follow-ups)

- **`LorentzianManifold` curvature-tensor primitives.** Added
  `metric_tensor`, `christoffel_symbols`, `riemann_tensor`,
  `ricci_tensor`, `scalar_curvature`. All identically zero on flat
  Minkowski (the manifold IS the canonical Ricci-flat solution to
  Einstein's vacuum equations); the API is the gateway to subclasses
  for curved Lorentzian backgrounds (Schwarzschild, FLRW). 7 new
  tests verify shape + Minkowski signature.

- **Learnable κ can cross 0 during training** (the previous
  static-branch limitation is closed). When `kappa` is a Tensor,
  `_tan_kappa_c` and `_atan_kappa_c` dispatch dynamically on
  `sign(κ)` at every call site via `torch.where(κ > 0, spherical,
  hyperbolic)`; `distance` was also refactored to a unified
  κ-trig form (`2·d · _atan_kappa_c(d)`) that handles all three
  branches uniformly. `|κ|` is clamped above `finfo.tiny` at the
  sqrt to keep autograd finite through the κ=0 crossing. Static
  (float) κ keeps its fast-path branch lock — same behavior, zero
  overhead. Two new tests verify κ flipping `-0.5 → +0.5` and
  `+0.5 → -0.5` via 50 SGD steps with finite gradients + on-manifold
  embeddings throughout.

- **Learnable κ on `KappaStereographicManifold`** — **full
  autograd through every κ-dependent operation**. Accept κ as a
  0-dim `torch.Tensor` (e.g. `nn.Parameter`); the gradient of any
  manifold operation with a κ-dependence flows back to κ so SGD can
  learn the curvature alongside the embedding parameters. Every
  internal κ reference (`_conformal_factor`, `_tan_kappa_c`,
  `_atan_kappa_c`, `mobius_add`, `distance`, `is_on_manifold`,
  `_provenance_signature`) now uses `_get_kappa()` /
  `_get_sqrt_abs_kappa()` to read the live tensor value, preserving
  the autograd chain. Constraint: the κ-branch (spherical /
  hyperbolic / Euclidean) is fixed at construction from the sign of
  κ's initial value; pushing κ across 0 during training is
  undefined behavior (caller keeps κ in one sign-half).

  10 new tests (`TestKappaGradientFlowPerOp`) — one per κ-dependent
  op (distance, inner, norm, exp_0, log_0, exp, log,
  parallel_transport, mobius_add) plus an end-to-end test through
  the realistic substrate-training chain
  (`v → exp_0(v) → all-pairs distance → NLL → backward`), confirming
  both `v.grad` and `κ.grad` are finite and non-trivial.

- **End-to-end cross-manifold validation pass.** New script at
  `notes/validation/cross_manifold_validation.py` runs a
  substrate-style training loop (tangent-at-origin → NLL on all-pairs
  distance → SGD) on each of the four Riemannian manifolds and
  separately verifies the Lorentzian causal/curvature primitives.
  All paths green (`notes/validation/cross_manifold_results.md`).

  Test count now 1082 (was 1072).

### Added (mixed-curvature + heterogeneous-κ manifolds)

- **`manifolds.ProductManifold`** — Riemannian product `M_1 × M_2 × …`
  with flat concatenated point storage `(B, Σ ambient_dim_i)` and
  the Pythagorean direct-sum metric. Each operation delegates per-
  submanifold. **Mix any of our existing manifolds**: e.g.
  `ProductManifold([KappaStereographic(n=4, κ=0), Lorentz(n=4)])`
  gives a 4-Euclidean × 4-hyperbolic concept space. Standard prior
  art: Gu-Sala et al. (2019) ICLR "Learning Mixed-Curvature
  Representations in Product Spaces"; Skopek et al. (2019) ICLR
  "Mixed-curvature VAEs". 16 new tests verify Pythagorean
  decomposition, per-component delegation, weighted metrics,
  autograd-finite through the substrate chain, provenance roundtrip.

- **`manifolds.HeterogeneousKappaManifold`** — κ-stereographic
  geometry with **per-point curvature**. The natural primitive for
  substrate-style embeddings where different concepts have
  different local curvatures. The class is a pure math primitive —
  it doesn't store κ; the user owns the κ parameterization (smooth
  field + per-concept residual is the recommended pattern but the
  manifold is agnostic). Pair operations combine two per-point
  κ's via a configurable combiner (arithmetic mean default;
  harmonic mean built-in; any commutative callable accepted).

  **Status (intentionally explicit per request)**:
    - Per-point continuous κ extends the established
      κ-stereographic model (Bachmann-Bécigneul-Ganea 2020).
    - Closest prior art: **GraphMoRE** (Guo et al. AAAI 2025,
      arXiv:2412.11085) uses mixture-of-experts gating for
      discrete per-node curvature; **Di Giovanni et al. 2022**
      (arXiv:2202.01185) uses homogeneous × spherically-symmetric
      products for pointwise curvature; **kHGCN** (Yang et al.
      2022) uses discrete Ollivier-Ricci curvature for
      message-passing weights inside a single-curvature space.
    - **Our research contribution**: continuous per-point κ (vs.
      GraphMoRE's discrete gating); pair-κ combiner abstraction
      (the rule for combining two per-point κ's into an effective
      pair-κ isn't standardized in the literature — we ship
      defensible defaults + caller-override). The substrate-team's
      "κ_field + per-point residual" decomposition is a
      parameterization pattern the manifold supports but doesn't
      own.

  22 new tests verify homogeneous-case agreement with
  `KappaStereographicManifold` across κ ∈ {-1, -0.5, +0.5, +1}
  (4 cases × 3 op = 12 round-trip checks), heterogeneous behavior
  (different κ → different embedding / distance), the
  arithmetic-mean default Euclidean recovery at sign-mixed pairs,
  custom-combiner override, autograd through both v and κ in the
  substrate-style training chain, and provenance roundtrip
  (custom-callable combiners fall back to default on reload).

  Test count now 1136 (was 1098).

### Added (heat-kernel precision + completeness)

- **Even n ≥ 4 heat kernel re-enabled.** The previously-raised
  `NotImplementedError` is gone — the spectral-shift-corrected
  recursion `k^{n+2} = -exp(-n·t)/(2π sinh r) · ∂_r k^n` works for
  even n too, seeded from the n=2 Davies–Mandouvalos integral form.
  Validated by:
  - Probability mass `∫ k_t · dV = 1` to machine precision (1e-13)
    for n ∈ {4, 6}, all t ∈ {0.1, 0.5, 1.0, 2.0, 5.0}.
  - Heat-equation residual `∂_t k − Δ_radial k ≈ 0` at 1e-5..1e-3
    for n=4 and 1e-5..1e-1 for n=6 — bounded by the n=2 quadrature
    error compounding through the autograd recursion, but small
    enough that the kernel is physically meaningful at typical
    `(t, r)` scales.

- **Closed-form n=5 heat kernel** (precision push).
  `k^5_t(r) = (4πt)^{-5/2} · exp(-4t - r²/4t) ·
                  [r²·sinh r + 2t·(r·cosh r − sinh r)] / sinh³ r`,
  derived analytically from the operator chain `(1/sinh r ∂_r)²
  exp(-4t - r²/4t)`. Replaces the autograd-recursion path for n=5:
  - **Precision**: heat-equation residual now 1e-8..1e-6 (was
    1e-5..1e-3 via two `torch.autograd.grad` calls). The recursion
    path's compounded float noise is gone.
  - **Cascades to odd n ≥ 7**: previously recursed from n=3 via two
    autograd steps; now recurses from the closed-form n=5 via one
    step. n=7 residual improved similarly.
  - **Speed**: no autograd-grad calls in the n=5 forward path.

  3 new tests pin the closed-form against the recursion (atol 1e-10),
  the analytic limit at r=0 (`(4πt)^{-5/2} · exp(-4t) · (1 + 2t/3)`),
  and backward-finite gradient through `d`.

### Fixed (validation pass — major math bug)

- **`hyperbolic_heat_kernel` produced WRONG values for n ≥ 5.** Caught
  by the heat-equation residual validation
  (`notes/validation/heat_kernel_validation.py`). The recursion
  `k^{n+2} = -(2π sinh r)^{-1} · ∂_r k^n` that we (and apparently
  some references) used is missing a spectral-shift factor:
  the correct form is

      k^{n+2}(t, r) = -exp(-n·t) / (2π·sinh r) · ∂_r k^n(t, r),

  where `exp(-n·t)` accounts for the shift in the spectral-bottom
  decay between dimensions (`((n+1)/2)² − ((n-1)/2)² = n`). Without
  this factor, the function we computed for n=5 had spectral-bottom
  `exp(-t)` (inherited from n=3) instead of the correct `exp(-4t)`,
  so it didn't solve the H^5 heat equation. Heat-equation residual
  was O(1) for n=5, 7, 9; now at finite-difference noise floor
  (1e-7 to 1e-5).

  **Even n ≥ 4 now raises `NotImplementedError`.** The previous
  "recurse from n=2" path was also based on the (wrong) simple
  recursion; the correct even-n recursion involves a different
  operator chain and needs separate work.

  Three new tests pin the correct recursion at n=3→5, n=5→7, and an
  independent heat-equation residual check at n=5. The previous
  `test_recursion_identity_n3_to_n5` was rewritten to verify the
  *correct* identity (it had been verifying self-consistency with
  the buggy formula).

  Findings document: `notes/validation/heat_kernel_findings.md`.

### Added (Stage 4)

- **`manifolds.LorentzianManifold`** — flat pseudo-Riemannian
  Minkowski spacetime `R^{1, n-1}`. Distinct from `LorentzManifold`
  (which is the *unit hyperboloid* embedded in this space).
  `LorentzianManifold` IS the ambient: the manifold is `R^n` with
  signature `(1, n-1)` — one timelike component (index 0) and `n − 1`
  spacelike. Geodesics are straight lines (`exp_x(v) = x + v`); the
  interesting structure is the **causal classification** of point
  pairs (timelike/null/spacelike, future/past) plus signed-interval
  measures (`proper_time`, `proper_distance`). 38 new tests. Closes
  Stage 4 of the hyperbolic-extension plan; opens the door to the
  substrate-as-spacetime architecture. Refs: Misner-Thorne-Wheeler
  (1973) *Gravitation*; Hawking-Ellis (1973) *Large Scale Structure
  of Space-Time*; O'Neill (1983) *Semi-Riemannian Geometry*.

  Test count now 1069 (was 1031).

### Fixed (third pass — senior-dev review findings)

- **`hyperbolic_heat_kernel` was non-differentiable for n ≥ 5.** The
  Grigor'yan–Noguchi autograd recursion called `torch.autograd.grad`
  with `create_graph=False` on a `d.detach().clone()`, so any caller
  doing `loss = hyperbolic_heat_kernel(t, d, mfd_n5).sum();
  loss.backward()` got a detached zero gradient on `d`. Fix:
  `create_graph=True` when upstream `d.requires_grad`, preserving
  the graph through the recursion (and through nested calls at n=7,
  9, …). When upstream doesn't track grad, a local grad-enabled
  clone is used as before. Two new tests pin the fix at n=3 and n=5.

- **`frechet_mean` convergence norm evaluated at wrong base point.**
  After `μ ← exp(μ, tangent_avg)`, the convergence check
  `manifold.norm(μ_new, tangent_avg)` evaluated a tangent at μ_new
  using the metric at μ_new — but the tangent lives in T_{μ_old}.
  For `LorentzManifold` the induced metric is point-independent so
  the bug was a no-op; for `KappaStereographicManifold` the
  conformal factor `λ_κ(x)` depends on x and the threshold was off
  by `λ(μ_new)/λ(μ_old)`. Fixed by computing the norm at μ_old
  before the `exp` step.

- **`KappaStereographicManifold.random_point` docstring inconsistency.**
  The comment said "clip at π/(4√κ)" but the code clipped at
  `π/(8√κ)`. The code is correct (π/4 lands exactly on the κ‖x‖²=1
  boundary; π/8 is the tighter conservative choice); the docstring
  is now consistent.

- **Documented `frechet_mean` limitation on the spherical branch.**
  For `KappaStereographicManifold(κ > 0)` the manifold is NOT
  Hadamard — Karcher convergence requires inputs within the
  injectivity radius `π/√κ`. The implementation runs unconditionally;
  added a docstring caveat.

### Fixed (second pass)

- **`hyperbolic_laplacian_eigenmaps` silent NaN on dense graphs.**
  Reported by the substrate team: any graph with `n_edges > ~500`
  produced all-NaN output with the default `lr=0.05` + `max_steps=200`.
  Root cause: per-node gradient magnitude scales with degree, so hub
  nodes overshoot the manifold's `cosh`/`tan` range. Fix: normalize
  the gradient by per-node degree (the random-walk-normalized
  Laplacian's gradient — `L_rw = I − D⁻¹A`), making `lr`
  scale-invariant w.r.t. graph size. Defaults now work for graphs
  up to N=500+ with edge probability 0.5. Plus: fail-loud
  RuntimeError when the output is non-finite (rather than silently
  returning NaN). 3 new tests: dense-graph regression, star-graph
  (extreme degree variance), and explicit fail-loud assertion.

- **Autograd-finite stress tests.** 11 new tests across Lorentz +
  Stereographic stressing aggressive chains (10-step exp/log
  walk-and-backward, parallel_transport near identity, log_0 ∘
  exp_0 at zero, Möbius inverse, exp∘log round-trip across all κ
  branches). Confirms the manifold primitives are fully autograd-
  clean under deeper composition, not just at single-call boundary
  inputs.

- **Benchmarks.** `tests/benchmarks/bench_hyperbolic.py` covers
  `hyperbolic_laplacian_eigenmaps` (N ∈ {50, 200, 500} dense
  random graph) and `frechet_mean` (N ∈ {50, 500, 5000} points).
  Wired into the existing benchmark harness.

  Test count now 1028 (was 1001).

### Added

- **`manifolds.KappaStereographicManifold`** — Stage 3 of the
  hyperbolic-extension plan. Parametric κ ∈ R model interpolating
  **spherical** (κ > 0), **Euclidean** (κ = 0), and **hyperbolic**
  (κ < 0) constant-curvature geometry. Points live in `R^n` (no
  extra ambient dimension; `ambient_dim = n` vs `n+1` for Lorentz).
  Closed-form Möbius gyro-addition, conformal-factor metric,
  branch-dispatched exp/log/distance/parallel-transport — all
  autograd-finite at boundary inputs from day one (same
  `_safe_sqrt` / `_safe_tanhc` / `_safe_atanhc` idioms as the
  Lorentz fix). 151 new tests; cross-comparison against geoopt's
  `Stereographic` across all four non-zero κ branches; Euclidean
  recovery at κ = 0 verified independently. v1 takes a Python-float
  κ; learnable scalar κ is a planned follow-up. Refs: Bachmann-
  Bécigneul-Ganea (2020) *Constant Curvature Graph Convolutional
  Networks*; Skopek et al. (2019); Ungar (2008) *Gyrovector
  Spaces*; Ganea et al. (2018) *Hyperbolic Neural Networks*.

- **`holonomy_lib.hyperbolic` ops generalize to
  `KappaStereographicManifold`.** `manifold_aware_inner`,
  `frechet_mean`, and `hyperbolic_laplacian_eigenmaps` now work
  unchanged on either manifold via the new
  `manifold.ambient_dim` property (n+1 on Lorentz, n on
  Stereographic) and a refactored `manifold_aware_inner` that uses
  the Riemannian inner at origin instead of the Euclidean dot of
  `log_0` outputs (metric-consistent across both manifolds:
  `manifold_aware_inner(x, x) = d(o, x)²` on both). 13 new
  cross-manifold tests verify the model-agnosticism.

  Test count now 1001 (was 837).

### Fixed

- **`LorentzManifold` autograd produces finite gradients at boundary
  inputs.** Reported by the substrate-training loop: NLL losses on
  embeddings parameterized as `T = exp_0(v)` produced
  all-NaN `v.grad` on the first batch even with finite forward.
  Root cause: classic `clamp(min=0) + sqrt` and `torch.where(cond,
  sinh(α)/α_safe, ones)` patterns leak `0·∞ = NaN` through the
  masked-out branch of `torch.where` because PyTorch evaluates both
  branches' gradients. Also: `acosh(z)` at `z = 1` has derivative ∞,
  and `torch.linalg.vector_norm(0) = 0` has backward `0/0 = NaN`.

  Fix: three new module-level helpers (`_safe_sqrt`, `_safe_sinhc`,
  `_safe_arcsinhc`) that compute the formula on a where-substituted
  input (never 0) and mask the output. Applied throughout `norm`,
  `exp`, `exp_0`, `log`, `log_0`, `distance`. `log` and `log_0`
  reparameterized to use the `arcsinh(arg) / (arg · sqrt(arg²+1))`
  form for `α/sinh(α)` where `arg = √|k|·‖y-x‖_M/2`, avoiding
  `arccosh`'s boundary singularity entirely. `_reproject_to_hyperboloid`
  now uses `torch.cat` instead of in-place subscript assignment.

  15 new `TestAutogradFinite` tests across Lorentz + the Stage 2
  hyperbolic ops cover `distance(x,x)`, `log(x,x)`, `log_0(origin)`,
  `exp(x, 0)`, `norm(x, 0)`, `parallel_transport(x,x,v)`, the full
  NLL-style all-pairs distance chain, plus `frechet_mean` and
  `manifold_aware_inner` backward at boundary inputs. Test count
  837 (was 822). Documented tangent-at-origin training recipe in
  `CONTENTS.md`.

### Added

- **`holonomy_lib.hyperbolic`** — new module: manifold-aware graph
  operations. Each primitive takes a manifold (e.g. `LorentzManifold`)
  as an explicit dependency, so the algorithms generalize naturally
  to other constant-curvature manifolds (`KappaStereographicManifold`,
  planned). Four primitives:
  - `manifold_aware_inner(x, y, manifold)` — Riemannian inner via the
    tangent at the manifold origin; Euclidean dot of `log_0` for
    `LorentzManifold`. Refs: Pennec (2006) §3.
  - `frechet_mean(points, manifold, weights, max_iter, tol)` —
    Karcher (1977) intrinsic mean via iterated weighted tangent
    averaging. Unique and well-defined on Hadamard manifolds (Afsari
    2011), including all `LorentzManifold` curvatures.
  - `hyperbolic_laplacian_eigenmaps(adjacency, manifold, max_steps,
    lr, init, generator)` — embeds graph nodes on `manifold` by
    minimizing `Σ A_ij d_M(Y_i, Y_j)²` via `RiemannianSGD`. Refs:
    Belkin-Niyogi (2003), Nickel-Kiela (2017) *Poincaré Embeddings*,
    Liu et al. (2019) *Hyperbolic GNN*.
  - `hyperbolic_heat_kernel(t, distances, manifold, n_quad,
    tail_budget)` — heat kernel `k^n_t(d)` with dimension dispatch:
    Gaussian for n=1, Davies-Mandouvalos closed form for n=3,
    Gauss-Legendre quadrature on the Davies-Mandouvalos integral for
    n=2, Grigor'yan-Noguchi recursion via `torch.autograd.grad` for
    higher odd/even n. Curvature scales out cleanly via
    `k^n_{-|k|, t}(d) = |k|^{n/2} · k^n_{-1, |k|·t}(√|k|·d)`. Refs:
    Davies-Mandouvalos (1988); Grigor'yan (2009) *Heat Kernel and
    Analysis on Manifolds* Thm 8.21; Grigor'yan-Noguchi (1998).
  Test count now 822 (was 788). New cataloged constants:
  `_N_DAVIES_MANDOUVALOS`, `_N_GRIGORYAN_INTEGRAL`,
  `HEAT_KERNEL_QUADRATURE_NODES`, `HEAT_KERNEL_QUADRATURE_TAIL_BUDGET`.

- **`manifolds.LorentzManifold`** — hyperboloid model of hyperbolic
  space at configurable sectional curvature `k < 0` (default `k = -1`,
  the canonical unit hyperboloid). Full closed-form API: Minkowski
  inner, on-manifold check, random_point, origin, projection, inner,
  norm, exp, log, distance, parallel_transport, retraction, plus
  `exp_0` / `log_0` convenience methods at the origin. Batched-first,
  GPU-native, `@with_provenance`-decorated, audit-clean. Integrates
  transparently with `RiemannianSGD` via the existing
  `projection + retraction` API. 80 new tests including a geoopt
  cross-comparison suite (`pytest.importorskip("geoopt")`); test count
  is now 788 (was 707). Numerical notes: `log` uses the
  `(α/sinh α)·u` form rather than `β·u/‖u‖` and `distance` uses the
  `2·arcsinh(√|k|·‖y-x‖_M/2)` identity rather than `arccosh(z)`
  directly, both to avoid catastrophic cancellation at x ≈ y; `exp`
  re-projects onto the hyperboloid to suppress drift. References:
  Nickel & Kiela (2018) *Learning Continuous Hierarchies in the
  Lorentz Model* (ICML); Chen et al. (2022) HyboNet *Fully Hyperbolic
  Neural Networks* (ACL); Lee (2018) *Introduction to Riemannian
  Manifolds* §5–§6; Cannon et al. (1997); Pennec (2006) on parallel
  transport.

## [0.4.1] - 2026-05-27

End-to-end MCP transport fixes. v0.4.0's MCP server worked when
called through unit tests that bypassed the protocol (`server._tool_manager._tools[name].fn(...)`),
but failed in two ways when driven via a real stdio MCP client:

### Fixed

- **`replay_with` failed with "op_id ... not in OP_REGISTRY"**. The
  MCP server process loaded the saved registry but never imported
  the modules whose `@with_provenance`-decorated ops it might need
  to re-execute. `mcp.py` now eagerly imports the known op-defining
  modules at server startup (algebra, spectral, discrete_geometry,
  info_geometry, manifolds, optimization, simplicial, topology,
  sheaf, lie). Optional / not-yet-installed modules are skipped
  silently.

- **`op_docstring` failed with "multiple values for argument
  'op_id'"**. The `_bind_registry` wrapper unconditionally pre-bound
  the registry as the first positional argument, but `op_docstring`
  doesn't have a `registry` parameter (it queries the global
  OP_REGISTRY directly). The wrapper now inspects the function's
  signature and only pre-binds the registry when the function
  actually takes one.

### Changed

- **List-returning tools wrap their return in `{"results": [...]}`
  for MCP transport.** FastMCP serializes Python lists by emitting
  one `content[i]` item per list element, producing a non-uniform
  shape between "returned one item" and "returned many." Wrapping
  guarantees a single JSON content item with a known structure.
  Python callers (direct invocation, native LLM tool-use schemas)
  see the underlying list via the unwrapped function; this
  normalization is transport-only.
- The wrapper declares `dict[str, Any]` as the uniform return type
  for all wrapped tools so pydantic's return-value validation
  matches the wrapped shape.

Driven by a real end-to-end MCP test (`/tmp/mcp_e2e_drive.py`,
not committed): spawns the server as a subprocess, connects via
the official mcp client SDK, exercises every tool through the
protocol. With these fixes, a 9-step interpretability question
("find the anomalous batch in this recording") completes
end-to-end without hitting a transport bug.

Tests: 707 passing (same count; `test_list_ops_returns_distinct_op_ids`
updated to expect the `{"results": [...]}` wrapping).

## [0.4.0] - 2026-05-27

Provenance agent-API redesign. The v0.3.0 MCP server was structurally
limited: agents could navigate the DAG but couldn't inspect tensor
content beyond global mean/std/min/max, slice tensors, run linear
algebra on cached values, or express anything other than zero-fill
substitutions. A driven validation pass against a non-trivial
"find the anomalous batch" question hit 7 falls-down before
dropping back to Python. v0.4 reshapes the surface around what
agents actually need.

The core move: a new `holonomy_lib.provenance.agent` module holds
the canonical tool inventory. Each tool is a Python function
decorated with `@agent_tool`. The module emits native LLM tool-use
schemas (`to_anthropic_schema()`, `to_openai_schema()`); the
existing `mcp.py` becomes a thin transport adapter that iterates
the same registered tools. Three call sites (native tool-use, MCP,
direct Python) all hit the same underlying functions.

### Added: agent module

- `@agent_tool(description=, name=)` decorator. Inspects signatures
  + docstrings + resolved type hints (via `typing.get_type_hints`,
  so `from __future__ import annotations` modules work transparently);
  registers a `ToolSpec` in module-level `_AGENT_TOOLS`.
- `to_anthropic_schema()` / `to_openai_schema()`. Provider-specific
  schema dumpers. Both share the same JSON-schema mapping (str /
  int / float / bool / list / dict / Optional). The `registry:
  ProvenanceRegistry` parameter is automatically stripped from the
  LLM-facing schema; transport adapters pre-bind it.
- `list_tools()` / `get_tool(name)`. Inventory helpers.

### Added: inspection tools (the v0.3.0 falls-down list, addressed)

- `tensor_slice(hex_id, expr)`: numpy-syntax slicing of a cached
  tensor. Returns raw values inline if the slice has at most
  `TENSOR_SLICE_INLINE_LIMIT = 256` elements; otherwise returns
  shape + stats summary. Index parser accepts only digits, '-',
  ':', ',' (injection-safe).
- `tensor_per_batch_summary(hex_id)`: per-batch mean/std/min/max.
  Fixes the v0.3.0 "global stats hide anomalies" wall: the
  revalidation drive confirms it picks the anomalous batch in a
  fixture where v0.3.0's get_tensor_summary lumped everything
  together.
- `tensor_eigenvalues(hex_id, k)`: top-k eigvalsh per batch. Default
  k = `TENSOR_SPECTRAL_DEFAULT_K = 10`.
- `tensor_singular_values(hex_id, k)`: top-k SVD per batch.
- `tensor_norm(hex_id, order)`: Frobenius or spectral norm.
  Per-batch for batched inputs.
- `tensor_compare(hex_a, hex_b, metric)`: pairwise comparison.
  Metrics: max_abs / frobenius / cosine.
- `op_docstring(op_id)`: returns the registered op's signature +
  docstring + op_version. Fixes the v0.3.0 "discovery is weak" wall.

### Added: replay recipe DSL

- `replay_with(target_hex, recipe)` replaces the v0.3.0 MCP `replay`
  tool's zero-fill-only substitution. The recipe is a dict with a
  `kind` field; supported kinds:
  - `zeros_like`: same shape/dtype, all zeros.
  - `from_hex` + `hex`: substitute with another cached tensor.
  - `perturb` + `noise_std` + `seed`: original + Gaussian noise.
    Seed is required; no implicit defaults.
  - `scale` + `factor`: multiply original by a scalar.
  - `swap_batch` + `i` + `j`: swap two batch elements along dim 0.
  - `literal` + `values`: explicit nested-list values (small tensors
    only; JSON arrays balloon agent prompts).
- Internal `_build_substitute()` helper is testable in isolation;
  `replay_with` is the agent-facing wrapper that builds + replays
  + summarizes new outputs.

### Changed

- `mcp.py` refactored to be transport-only. Drops ~80 lines of
  inline tool definitions; iterates over `agent.list_tools()` and
  pre-binds the registry argument. Resolves stringified type hints
  before pydantic / FastMCP sees them (the old inline approach
  worked only because pydantic resolved each function's annotations
  in the local scope where it was defined).
- The v0.3.0 MCP nav tools (`list_ops`, `where`, `node_info`,
  `ancestors`, `get_tensor_summary`) are preserved by name and
  signature for backward compat with existing MCP clients; they
  just live in `agent.py` now instead of `mcp.py`.
- `replay` (zero-fill via shape spec) is dropped from the MCP
  surface in favor of `replay_with` (`{"kind": "zeros_like"}` is
  equivalent for the v0.3 use case).
- The `hex` parameter is renamed to `hex_id` in agent tools to
  avoid shadowing Python's `hex()` builtin AND match the v0.3
  convention.

### Cataloged constants

- `TENSOR_SLICE_INLINE_LIMIT = 256` (🔬 experimentally-set,
  agent.py).
- `TENSOR_SPECTRAL_DEFAULT_K = 10` (🔬).
- `PYTHON_SLICE_ARITY = 3` (✅ derived; Python language constant).

### Tests

`tests/provenance/test_agent.py` adds 52 new tests covering the
decorator + schema generators (Phase 1), every inspection tool
(Phase 2), and every recipe kind + end-to-end replay roundtrip
(Phase 3). `tests/provenance/test_mcp.py` updated to assert the
v0.3 nav tools still register AND the new v0.4 inspection tools
land.

Tests: 659 -> 707 passing. Audit: 0 undocumented, 28 cataloged.

### Scrutiny-pass hardening

A code-review pass on the v0.4 work surfaced four real issues plus
one minor, all in `agent.py`:

- `_parse_index_expr` tolerates a single trailing comma (`":, 0,"`).
  NumPy and PyTorch both accept `t[0,]` and LLMs frequently emit it.
- `tensor_compare` on cross-device tensors returns an `error:
  "device mismatch"` dict instead of throwing an unhandled
  RuntimeError from `torch.linalg`.
- The `cosine` metric detects exact-zero input vectors and returns
  `error: "cosine undefined..."` instead of silently returning `0.0`
  (which an LLM could misread as "orthogonal").
- `replay_with` pre-checks the substitute tensor's shape against the
  recorded node's output shape for single-output ops and returns a
  recipe-aware error instead of bubbling a confusing downstream
  torch error.
- `swap_batch` with `i == j` returns a clone early instead of doing
  the redundant double-write.

## [0.3.0] - 2026-05-27

Provenance module sweep: performance, robustness, visualization, and
an agent-access layer (MCP + Jupyter). Ten commits across four phases.
The construction side (decorator, record(), hex) was already mature;
this release rounds out the consumption side and addresses the two
biggest perf gaps (hashing cost on big tensors, memory growth from
cache_tensors). Tests: 622 -> 659 passing.

### Added: performance

- **Sketch hashing (opt-in)**: `record(hash_mode="sketch")` and
  `ProvenanceRegistry(hash_mode="sketch")`. Hashes
  `shape + dtype + SKETCH_SAMPLES = 64` strided samples + `sum` + `std`
  instead of the full tensor bytes. **15× faster** on 8 MB (n=1024
  float64) inputs; crossover with full mode at ~n=256. Hexes are not
  portable across modes; the chosen mode is stamped in to_dict() and
  round-trips through save()/load().
- **On-disk tensor cache**: `record(cache_to_disk=path)` mirrors every
  cached output to a directory via torch.save. Memory eviction under
  `max_cache_size` drops the in-memory copy but the disk copy persists
  and `get_tensor()` reloads on demand. New
  `ProvenanceRegistry.clear(delete_disk=False)` cleans the in-memory
  caches and optionally the disk files. ~2.3× I/O overhead vs memory-
  only caching at n=1024.
- **Extended benchmark suite**: `tests/benchmarks/bench_provenance.py`
  now covers hashing scale (n=16/256/1024), cache + replay overhead,
  and sketch/disk variants. Baseline + post-Phase-1 numbers preserved
  in `notes/benchmark_provenance_v0.2.md` and `_v0.3.md`.

### Added: robustness

- **Replay completion**: `ProvenanceRegistry.replay()` now works for
  class-method calls and ops taking tuple-of-tensor inputs (e.g.,
  `FixedRankPoint = (U, S, Vt)`). New
  `@provenance.register_provenance_class("FixedRankManifold")`
  decorator opts a class into replay; classes must define
  `_from_signature(cls, sig: dict)` as the inverse of
  `_provenance_signature`. `FixedRankManifold` and `SPDManifold` are
  registered out of the box. Unregistered classes still hit a clear
  `NotImplementedError` with actionable text.
- **User-input caching for replay**: when `cache_tensors=True`,
  user-supplied input tensors are now cached in a separate
  `_user_input_cache` (not bounded by `max_cache_size`) so replay
  can find them. Without this, chains like `mfd.exp(S, V)` where S
  is a user tensor couldn't replay past the substitution.
- **Op-version drift detector on load()**: new
  `ProvenanceVersionWarning` (subclass of UserWarning) is emitted
  when any loaded node's `op_version` differs from the currently-
  installed `OP_REGISTRY[op_id]` version. New `strict: bool = False`
  kwarg converts to ValueError. Unknown op_ids (recorded by a module
  not imported in this process) are flagged in the same diagnostic.

### Added: consumption / visualization

- **Mermaid + Graphviz exports**: `to_mermaid()` returns a flowchart
  string suitable for inline GitHub Markdown / JupyterLab; `to_graphviz()`
  returns DOT source. Neither imports the optional rendering library
  (output is a string the caller pipes to whatever renderer they
  prefer.
- **diff_summary(other)**: human-readable rendering of `diff()`,
  bucketed into Cache hits / Drift / Only in self / Only in other.
  Drift (same op_id, different inputs) is the interesting category
  for "did my refactor preserve semantics" questions.
- **ancestors_with_tensors(hex)**: returns
  `dict[hex, (ProvenanceNode, Optional[Tensor])]`. One call instead
  of `ancestors()` + N `get_tensor()` calls.

### Added: agent access

- **to_llm_context(max_ops, show_shapes, show_params)**: compact text
  summary suitable for placing in an LLM agent's prompt. Format:
  header (op + cached tensor counts, hash_mode), Ops by op_id with
  counts, Notable shapes, Roots (no provenance-internal parents),
  Leaves (no provenance-internal consumers).
- **MCP server (`holonomy_lib.provenance.mcp`, optional extra)**:
  exposes a saved registry as six MCP tools (`list_ops`, `where`,
  `node_info`, `ancestors`, `get_tensor_summary`, `replay`). Entry
  point: `python -m holonomy_lib.provenance.mcp` reads the registry
  from `HOLONOMY_PROVENANCE_REGISTRY` env var, starts the server on
  stdio. Install via `pip install 'holonomy-lib[mcp]'`. File-loaded
  registries only in v0.3; live-process attachment is v0.4.
- **Jupyter cell magic (`holonomy_lib.provenance.jupyter`, optional
  extra)**: `%load_ext holonomy_lib.provenance.jupyter` enables
  `%%record_provenance` which wraps a cell in `record()`, binds the
  registry to `_prov` (or a custom name via the magic line), and
  renders the Mermaid DAG below the cell output. Install via
  `pip install 'holonomy-lib[jupyter]'`.

### Changed

- `ProvenanceRegistry.to_dict()` schema bumped 0.2 -> 0.3; older saved
  registries default to `hash_mode="full"` and `cache_to_disk=None`
  on load for backward compat.
- `SKETCH_SAMPLES = 64` cataloged in `notes/magic_numbers.md` with
  scale-of-validity (empirically zero collisions on 200 random 16² and
  no known structural collisions on Laplacian inputs).

### New optional-extras groups

- `[mcp]`: `mcp>=0.9` for the MCP server.
- `[jupyter]`: `ipython` for the cell magic.

## [0.2.1] - 2026-05-27

Packaging-metadata-only release. No code changes.

### Changed

- `pyproject.toml` description synced with the GitHub repo description:
  adds cellular sheaves, SO(3) Lie primitives, and information geometry
  to the listed capabilities (these modules shipped in 0.2.0 but the
  description hadn't caught up).

## [0.2.0] - 2026-05-27

Six new modules and several extensions on top of the v0.1.0 seed.
Tests: 543 → 613 passing. Module count: 6 → 12.

### Added: new modules

- **`optimization`**: Riemannian optimizers wrapping the existing
  manifold `projection` + `retraction` API.
  - `RiemannianSGD(manifold, lr)` + functional `riemannian_sgd_step`.
  - Works with `FixedRankManifold` (SVD-triple state) and
    `SPDManifold` (square-matrix state).
  - Adam / RMSProp deliberately omitted: adaptive preconditioning is
    user-side ergonomics, not part of the math of optimization on a
    manifold. Compose your own from `manifold.projection` +
    `retraction` + `torch.optim.Adam` buffers.
  - Refs: Absil-Mahony-Sepulchre (2008) §4.1; Bonnabel (2013).

- **`simplicial`**: Simplicial complex data structures + boundary
  operators + Vietoris-Rips construction.
  - `DenseSimplicialComplex`: batched, padded `(B, n_k_max, k+1)`
    simplex tables with validity masks. `boundary(k)` returns the
    dense Koszul-signed boundary matrix.
  - `SparseSimplicialComplex`: single-instance, sparse-CSC boundary
    matrices for the persistent-homology reduction kernel.
  - `vietoris_rips_dense` (batched) + `vietoris_rips_sparse`
    (single-instance) + `pairwise_distances`.
  - Refs: Munkres (1984); Hausmann (1995); Bauer (2021).

- **`topology`**: Hodge Laplacians + Betti numbers + batched
  persistent homology on simplicial complexes.
  - `hodge_laplacian(complex, k)`: `L_k = ∂_{k+1} ∂_{k+1}^T + ∂_k^T ∂_k`,
    kernel-dim = k-th Betti.
  - `betti_numbers(complex, max_dim)`: near-zero eigenvalue counting
    on each `L_k`. Closed-form verified: S¹ → (1, 1), S² → (1, 0, 1),
    T² → (1, 2, 1).
  - `persistence_diagrams(points, max_dim=2, max_radius=inf)`: batched
    H₀ + H₁ + H₂ on Vietoris-Rips filtrations. H₀ via union-find on
    sorted edges; H₁/H₂ via Z/2 left-to-right boundary-matrix reduction
    with Bauer-Kerber-Reininghaus clearing.
  - `reduction_backend="torch"` runs end-to-end on the filtration's
    device (CPU or GPU); current path is a same-algorithm torch port
    (not yet a custom CUDA kernel) and is ~21× slower than CPython
    sets at n=80; the GPU win is a v0.3 follow-up.
  - Refs: Eckmann (1944); Lim (2020); Schaub et al. (2020);
    Edelsbrunner-Letscher-Zomorodian (2002); Cohen-Steiner-Edelsbrunner-
    Harer (2007) stability.

- **`info_geometry`**: divergences on probability distributions.
  - `bregman_divergence(p, q, potential)` for any convex generator.
  - `kl_divergence_categorical(p, q)`, `kl_divergence_gaussian(...)`
    (Cholesky-stable closed-form).
  - Fisher information metric + natural gradient added in the v0.1
    roadmap sweep.
  - Refs: Bregman (1967); Banerjee et al. (2005); Amari (2016);
    Cover-Thomas (2006).

- **`sheaf`**: cellular sheaves on graphs and their Laplacians.
  - `GraphSheaf` dataclass; `sheaf_coboundary`, `sheaf_laplacian`
    (`δ^T δ`, PSD), `sheaf_dirichlet_energy`.
  - Reduces to the standard combinatorial graph Laplacian under
    trivial stalks. Orientation-flip on a 3-cycle correctly drops
    kernel dim 1 → 0 (the monodromy test).
  - v1 is dense-only with `SHEAF_DENSE_BYTES_CAP = 2 GiB` pre-flight
    guard; node-edge sheaves on graphs only (higher-dim cellular
    sheaves on simplicial complexes planned).
  - Rejects self-loops + duplicate edges at construction (callers
    must pre-process).
  - Refs: Hansen-Ghrist (2019); Bodnar et al. (2022) Neural Sheaf
    Diffusion; Curry (2014).

- **`lie`**: SO(3) primitives + real spherical harmonics.
  - `so3.axis_angle_to_matrix` (Rodrigues), `matrix_to_axis_angle`
    (dual-branch log: trace-based away from π, quaternion-based
    near π).
  - `so3.so3_exp` / `so3_log` on so(3) (3×3 skew matrices).
  - `so3.random_so3(batch_size, generator)`: Haar-uniform via the
    quaternion-from-3-uniforms construction (Shoemake 1992); chi-squared
    sanity test with p < 1e-6 bound in the suite.
  - `so3.compose(R1, R2)`: group product.
  - `real_spherical_harmonics(directions, l_max)`: closed-form Y_lm
    for `l_max ≤ 4`. Per-l block norm preserved under SO(3) rotation
    (full mixing via Wigner-D matrices is a v0.3 follow-up).
  - `SO3_LOG_NEAR_PI_RAD = 1e-7` is empirically calibrated for
    **float64**; do not change to e3nn / pytorch3d's `1e-2` (correct
    for float32) without re-running the empirical comparison in the
    `so3.py` docstring.
  - Refs: Hall (2015) §3.1; Shoemake (1992); Edmonds (1957);
    Cohen et al. (2018) Spherical CNNs.

### Added: extensions to existing modules

- **`spectral` additions**:
  - `magnetic.*`: magnetic Laplacian for directed graphs (Furutani
    2020), with sign-magnetic extension for signed-directed graphs
    (Fiorini 2023; He et al. 2023).
  - `heat_kernel_chebyshev`: Hammond-Vandergheynst-Gribonval (2011)
    Chebyshev-polynomial heat kernel.
  - `effective_resistance`, `commute_time` (Klein-Randić 1993).
  - `diffusion_map` (Coifman-Lafon 2006).
  - Sparse-COO/CSR/CSC paths for all four Laplacian variants
    (combinatorial, symmetric-normalized, random-walk, Kunegis signed);
    end-to-end with sparse `lanczos_eigsh`.

- **`algebra` additions**:
  - `lanczos_eigsh(A, k, which="LA"|"SA", sigma=σ, n_iter=…)`: Lanczos
    top-k eigensolver with full reorthogonalization (Paige 1972).
  - `"SA"` mode is shift-and-invert (Ericsson-Ruhe 1980): factor once
    outside the iteration, each step is a `lu_solve` against
    `A − σI`. Raises `RuntimeError("shift-invert breakdown")` if σ
    coincides with an eigenvalue. For graph Laplacians (which have 0
    in spectrum), use a small negative shift.
  - Sparse-input dispatch on the same API.

- **`discrete_geometry` additions**:
  - `forman_ricci_simple` + `forman_ricci_augmented`: combinatorial
    Forman-Ricci curvature (Sreejith et al. 2016; Samal et al. 2018).
  - Performance: pair tiling (`SINKHORN_TILE_DEFAULT = 256`) +
    sync-cadence-every-8-iters → 13× speedup on Ollivier-Ricci at
    n=64 (22.6s → 1.7s).

- **`provenance` additions**:
  - Class-method support for `FixedRankManifold` / `SPDManifold`
    methods; `self` is canonicalized into a provenance-signature
    dict. `replay()` raises `NotImplementedError` for class-method
    nodes (the canonicalization isn't reversible) and for
    tuple-of-tensors inputs (unpacked `point[i]` keys don't
    reassemble); both have explicit error messages.
  - `blake3` opt-in (faster hash), hooks API, SAELens-style dataset
    emission, diff API, persistence (`save` / `load`).

### Changed

- **Self-loops dropped at entry to ALL graph primitives** (codified
  in `CONVENTIONS.md`). Sheaf v1 also rejects duplicate edges at
  `GraphSheaf` construction time; call sites must pre-process.
- `FixedRankManifold.retraction` auto-switches to Halko-Martinsson-
  Tropp randomized SVD at low `r/min(m, n)` ratios (the common case);
  25× speedup at 1024×1024×32 (193 ms → 7.6 ms).

### Notes

- The `v0.1.0` git tag was placed mid-development, before the full
  "v0.1 roadmap" cycle completed. Everything from `optimization`
  onward in the list above actually ships in this `v0.2.0` release.

## [0.1.0] - 2026-05-26

Initial public release. Six seed modules:

- **`manifolds`**: `FixedRankManifold` (Vandereycken 2013),
  `SPDManifold` (Pennec et al. 2006 affine-invariant metric).
- **`algebra`**: `truncated_svd` exact (Eckart-Young) + randomized
  (Halko-Martinsson-Tropp 2011).
- **`tensor_calculus`**: `hosvd`, `mode_product`, `mode_unfolding`
  (Kolda-Bader 2009).
- **`spectral`**: combinatorial / symmetric-normalized / random-walk
  / signed Laplacians (Chung; von Luxburg; Kunegis); batched
  `laplacian_eigenmaps` embedding.
- **`discrete_geometry`**: `ollivier_ricci_curvature` via batched
  log-domain Sinkhorn on all-pairs shortest paths (Ollivier 2009;
  Cuturi 2013); `discrete_ricci_flow` + `ricci_flow_with_surgery`
  ("Perelman on networks", Sia et al. 2019; Ni-Lin-Luo-Gao 2019).
- **`provenance`**: content-addressable Merkle DAG of math
  primitives, `@with_provenance` decorator, `record()` context
  manager, `ProvenanceRegistry` with substitution / replay /
  SAELens-style emission for mechanistic interpretability.

- `audit.py`: build-gate enforcing the no-magic-numbers discipline:
  every numerical literal must be derived, a universal invariant, or
  cataloged in `notes/magic_numbers.md` with scale-of-validity.

[0.4.1]: https://github.com/superninjv/holonomy_lib/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/superninjv/holonomy_lib/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/superninjv/holonomy_lib/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/superninjv/holonomy_lib/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/superninjv/holonomy_lib/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/superninjv/holonomy_lib/releases/tag/v0.1.0
