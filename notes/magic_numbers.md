# Magic numbers catalog — holonomy-lib

Per the **no magic numbers** rule (CONVENTIONS §5, enforced by `python tools/audit.py src/ --strict`), every numerical
constant in this library must be:

- **(a)** Derived from inputs (function of dims, structure, problem parameters)
- **(b)** A universal invariant (π, e, log N, 1/N, √N, mathematical identities)
- **(c)** Experimentally tuned with documented procedure + recorded scale-of-validity

This file catalogs every constant. Updating this file is a *precondition* for
adding any new constant.

**Constants don't transfer across scales.** A value that worked at one scale is
NOT assumed to work at another. Each entry should record the scale at which it
was determined and what to re-derive at scale.

---

## Status legend

- ✅ **Derived** — formula resolves to a value from inputs/invariants
- ⚖️ **Scale-invariant** — value is a ratio or dimensionless that auto-scales
- 🔬 **Experimentally set** — value tuned via documented sweep, scale-of-validity recorded
- ⚠️ **Unresolved** — currently picked, no derivation, marked for revisiting

---

## Library-wide ALLOWED literals (audit's safe-list, justified)

See `tools/audit.py` for the ALLOWED_LITERALS set:

| Literal | Justification |
|---|---|
| `0, 1, -1` (and float forms) | Mathematical identities |
| `0.5` and `2` (and `-2`, float forms) | Halving/doubling identities. Symmetric/antisymmetric parts `(Z ± Zᵀ)/2`, midpoint mean `(a + b)/2`, quadratic-form coefficient `(½)xᵀAx`, triangular numbers `n(n+1)/2`, `2π`, 2-norm, second-derivative coefficients. On par with `0, 1, -1` for mathematical code. Added 2026-05-26. |
| `1024, 1024.0` | Universal binary unit conversion (KB↔MB↔GB) |
| `1000, 1000.0` | Universal SI time conversion (s↔ms) AND RNG-stream-offset multiplier |
| `1e-9` (`numerical_floor_convention`) | Posited convention: anti-divide-by-zero floor, used in `1.0 / max(x, 1e-9)` and similar guards. Also reused as an absolute tolerance: the Lorentz `is_on_manifold` constraint check and the Lorentzian `causal_type` null band. Smaller than float32 eps (~1.19e-7), so in float32 it is below roundoff, and float16 cannot represent it at all (it rounds to 0; `fisher_information_categorical` raises its floor to float16's smallest normal for that reason). It is not a dtype-aware threshold (CONVENTIONS §3.1). |

---

## Cataloged constants (per-module)

### manifolds

| Name | Status | Value | Justification |
|---|---|---|---|
| `RETRACTION_RANDOMIZED_THRESHOLD` | 🔬 | `0.25` | `FixedRankManifold.retraction` switches from exact to randomized SVD when `r ≤ 0.25 · min(m, n)`. **Scale of validity**: derived from CPU benchmarks at 64-1024-square sizes (`notes/benchmark_baseline.md` 2026-05-27) — randomized SVD is strictly faster than full-and-truncate below this ratio, by 8× at r/min ≈ 12% and 37× at r/min ≈ 3%. At ratios above 25% the constant factors flip and exact wins back. Re-derive when GPU benchmarks land. Used in `src/holonomy_lib/manifolds/fixed_rank.py`. |
| `COMPARISON_BALL_VOLUME_QUADRATURE_NODES` | 🔬 | `64` | Default Gauss-Legendre node count for the model ball-volume integral `∫_0^r sn_κ(t)^{N-1} dt` in `manifolds/comparison.py`. The integrand is smooth/analytic on a finite interval, so Gauss-Legendre converges geometrically; for `κ = 0` (degree-`(N-1)` polynomial integrand) the rule is exact for `N ≤ 2·nodes`. **Scale of validity**: 64 nodes give sub-1e-12 agreement with the closed forms (Euclidean, sphere, hyperbolic) for `N` up to ~100 and `√κ·r` up to the model diameter (validated by `tests/manifolds/test_comparison.py`). Raise for very large `N` or near-diameter spherical radii. Cataloged 2026-05-28. |
| `SN_LINEAR`, `SN_QUADRATIC` | ✅ | `1/6`, `1/120` | Taylor coefficients of `sin(z)/z = 1 - z²/3! + z⁴/5!`, continued to signed κ so `sn_κ(t) = t - κt³/6 + κ²t⁵/120` near κ = 0. **Derived**: universal series coefficients; they give `sn_kappa` the correct value and κ-derivative (`-t³/6`) at and across 0. Used in `src/holonomy_lib/manifolds/comparison.py`. Cataloged 2026-10-25. |
| `KAPPA_SCALE_LINEAR`, `KAPPA_TAN_QUADRATIC`, `KAPPA_ATAN_QUADRATIC` | ✅ | `1/3`, `2/15`, `1/5` | Taylor coefficients of `tan(√q)/√q = 1 + q/3 + 2q²/15` and `atan(√q)/√q = 1 - q/3 + q²/5` with signed `q = κα²`. **Derived**: universal series coefficients; used inside a dtype-aware neighborhood of κ = 0 so exp/log maps and their κ-gradients stay correct while a learnable κ crosses 0. Used in `src/holonomy_lib/manifolds/stereographic.py`. Cataloged 2026-10-25. |
| `DISTANCE_SCALE_LINEAR`, `DISTANCE_SCALE_QUADRATIC` | ✅ | `1/6`, `3/40` | Taylor coefficients of `asin(√q)/√q = 1 + q/6 + 3q²/40`, used for the stereographic distance near κ = 0. **Derived**: universal series coefficients. Used in `src/holonomy_lib/manifolds/stereographic.py`. Cataloged 2026-10-25. |
| `SPHERICAL_SAMPLE_RADIUS_FRACTION` | 🔬 | `1/8` | Radius cap for `random_point` on a spherical (κ > 0) stereographic chart, as a fraction of the coordinate scale `1/√κ` (one quarter of the `tan` pole radius `π/(2√κ)` rounded to the existing cap). Keeps sampled points well inside the chart where the projection is well conditioned. **Scale of validity**: a sampling convenience, not a statistical prior; callers who need a specific distribution should sample tangent vectors themselves. Used in `src/holonomy_lib/manifolds/stereographic.py`. Cataloged 2026-10-25. |

### topology

(no entries yet)

### algebra

| Name | Status | Value | Justification |
|---|---|---|---|
| `RANDOMIZED_SVD_OVERSAMPLE_DEFAULT` | 🔬 | `5` | Halko-Martinsson-Tropp (2011) §1.2 standard recommendation for the oversampling parameter `p` in randomized SVD. Five extra projection columns typically yield relative spectral error below 1e-3 for matrices with smoothly decaying singular values past the truncation rank. **Scale of validity**: adequate when σ_{r+1} is meaningfully smaller than σ_r; increase to 10 (also literature-standard) when the spectrum is flat near the truncation. Used in `src/holonomy_lib/algebra/linear.py`. |
| `RANDOMIZED_SVD_N_ITER_DEFAULT` | 🔬 | `2` | Halko-Martinsson-Tropp (2011) §4.5 standard recommendation for subspace-iteration power method steps. Two iterations amplify the singular-value gap by (σ_{r+1}/σ_r)^(2q+1), sufficient for typical low-rank truncation. **Scale of validity**: fine for σ_{r+1}/σ_r ≲ 0.5; increase to 4–7 when ratio is closer to 1. Used in `src/holonomy_lib/algebra/linear.py`. (Note: value 2 is also in `ALLOWED_LITERALS` as a doubling identity; this entry documents the *semantic* role of the default.) |
| `LANCZOS_OVERSAMPLE_DEFAULT` | 🔬 | `10` | Extra Lanczos iterations beyond `k` for top-k eigensolver convergence. Saad (2011) §6.5 recommends 5–10 extra for reliable top-k Ritz convergence on well-conditioned matrices; we pick the upper end for safety. **Scale of validity**: for spectra with a clear top-k gap (extreme eigenvalues well-separated from bulk), 10 is comfortably sufficient. For interior eigenvalues or near-degenerate spectra, the caller should increase `oversample` or use shift-and-invert (`which="SA"`, `sigma=`). Used in `src/holonomy_lib/algebra/lanczos.py`. |
| `SPARSE_SA_ITER_SOLVER_N` | 🔬 | `8000` | Sparse smallest-algebraic (`which="SA"`) CPU dispatch threshold: at or below this `n` use SciPy `eigsh` shift-invert (full sparse LU), above it LOBPCG + `spilu` incomplete-LU (the LU fill-in grows too fast past this size). **Throughput threshold only** — both branches compute the same bottom-k modes (verified to ~1e-14 in `tests/algebra/test_eigsh_sparse.py`). Set in synoros-substrate `sheaf_scale.py` on sheaf Laplacians up to ~150k stalk-dims (2026-06-06). **Scale of validity**: CPU, Laplacian-like sparsity (bounded fill per row); re-derive on substantially denser matrices or different hardware. Used in `src/holonomy_lib/algebra/eigsh_sparse.py`. |
| `SPARSE_SA_SPILU_LOBPCG_KNOBS` | 🔬 | `eps_frac 1e-3, drop_tol 1e-3, fill_factor 10, tol 1e-6, maxiter 100` | The large-n CPU LOBPCG branch's numerical knobs (one entry for the set: `SPILU_EPS_FRAC`, `SPILU_DROP_TOL`, `SPILU_FILL_FACTOR`, `LOBPCG_TOL`, `LOBPCG_MAXITER`). They set CONVERGENCE RATE and achieved precision, not WHICH modes — verified against the eigsh branch. eps-shift: `spilu` factors `A − σ I` with `σ = lower − eps_frac · scale`, the Gershgorin lower bound minus a fraction of the Gershgorin scale, which makes a singular A factorable; drop_tol/fill_factor trade preconditioner quality against memory (Saad 2003 §10.4); tol/maxiter are the LOBPCG stopping criteria (Knyazev 2001). From synoros-substrate `sheaf_scale.py` (validated 2026-06-06 on sheaf Laplacians to ~150k stalk-dims). **Scale of validity**: tied to the Laplacian sparsity class; loosen drop_tol / raise fill_factor if LOBPCG stalls on denser operators. Used in `src/holonomy_lib/algebra/eigsh_sparse.py`. |
| `SOLVER_SYNC_EVERY_DEFAULT` | 🔬 | `8` | Cadence of the host-syncing (`.item()`) convergence check inside the `cg`/`lsqr` solver loops. Same value and rationale as the established `SINKHORN_SYNC_EVERY_DEFAULT` (CONVENTIONS §4.1): checking every iteration serializes GPU work on the sync; every 8 amortizes it to ~12% of iterations while wasting at most 7 cheap iterations past convergence. Per-element convergence is tracked ON DEVICE every iteration (lsqr freezes converged elements — required for correctness past Krylov breakdown, not just perf); only the all-elements host check is on this cadence. **Scale of validity**: inherited from the Sinkhorn-loop analysis; any value ≥ 1 is correct, this only trades sync overhead vs wasted iterations. Used in `src/holonomy_lib/algebra/solvers.py`. |
| `LOBPCG_SUBSPACE_BLOCKS` (algebra), `_LOBPCG_SUBSPACE_BLOCKS` (sheaf) | ✅ | `3` | LOBPCG iterates on three blocks (Ritz vectors, residuals, previous search directions; Knyazev 2001), so the native solver needs `n ≥ 3k`; smaller problems fall back to a dense eigensolver. **Derived**: structural count, not tuned. Used in `src/holonomy_lib/algebra/eigsh_sparse.py` and `src/holonomy_lib/sheaf/laplacian.py`. Cataloged 2026-10-25. |
### discrete_geometry

| Name | Status | Value | Justification |
|---|---|---|---|
| `SINKHORN_REG_DEFAULT` | 🔬 | `0.003` | Entropic regularization for the Ollivier transport problems. Chosen 2026-10-25 against exact W1 (SciPy `linprog` on the same neighborhood measures): at this value the default call converges without warnings on the karate club, P6, C8, K6, a ring of four K5, sparse and dense weighted n=16 graphs and a 3-block SBM n=48, with max edge error ≤ 4.1e-3 (karate 5.6e-4). The remaining error is entropic bias. **Scale of validity**: absolute units of edge length; rescale with the graph's distance scale. Was 0.01 before 0.8.0, which with 100 iterations left karate errors up to 0.525. Used in `src/holonomy_lib/discrete_geometry/ricci.py`. |
| `SINKHORN_N_ITER_DEFAULT` | 🔬 | `65536` | Cap on total Sinkhorn sweeps including the epsilon-scaling warm-up stages. Finished pairs stop early, so typical cost is set by convergence, not this cap; smaller caps left warnings in Ricci-flow runs on the reference graphs. **Scale of validity**: raise only for much smaller `reg` or very wide cost ranges. Used in `src/holonomy_lib/discrete_geometry/ricci.py`. |
| `SINKHORN_TOL_DEFAULT` | 🔬 | `1e-4` | Absolute tolerance on both transport marginals (checked every `SINKHORN_SYNC_EVERY_DEFAULT` sweeps). Validated together with `SINKHORN_REG_DEFAULT` against exact W1 above, in float64 and float32. Also the floor for the warm-up stages' tolerance. Was a 1e-9 bound on dual-potential change before 0.8.0. Used in `src/holonomy_lib/discrete_geometry/ricci.py`. |
| `SINKHORN_TILE_DEFAULT` | 🔬 | `256` | Pairs processed simultaneously inside the Sinkhorn loop. Peak inner-broadcast memory is `B · tile · n² · element_size`: at B=1, tile=256, n=256, float64 that is 128 MiB (8 MiB at n=64). **Scale of validity**: balances Python-loop overhead against memory pressure on the inner `(tile, n, n)` broadcast. Re-derive when running on a memory-constrained GPU or at n > ~512. Used in `src/holonomy_lib/discrete_geometry/ricci.py`. |
| `SINKHORN_SYNC_EVERY_DEFAULT` | ⚖️ | `8` | How often to host-sync Sinkhorn's convergence check (`.item()` on the delta tensor). Per-iter syncs add tens of µs of kernel-queue wait on GPU; cadence=8 amortizes that cost by 8× while only delaying convergence detection by at most 7 extra iterations. **Scale of validity**: dimensionless cadence; does not depend on input size or dtype. Cataloged 2026-05-27. Used in `src/holonomy_lib/discrete_geometry/ricci.py`. |
| `SINKHORN_EPSILON_DECAY` | ⚖️ | `0.5` | Geometric factor for epsilon-scaling: each warm-up stage halves the regularization once its marginals converge, warm-starting the dual potentials, until the target `reg`. A dimensionless schedule choice (Schmitzer 2019 uses geometric schedules). Used in `src/holonomy_lib/discrete_geometry/ricci.py`. Cataloged 2026-10-25. |
| `SINKHORN_ABSORPTION_THRESHOLD` | 🔬 | `100.0` | Absorption bound τ from Schmitzer (2019), "Stabilized sparse scaling algorithms for entropy regularized transport problems", §5: when accumulated log-scalings exceed τ they are absorbed into the dual potentials and the kernel is rebuilt, keeping the stabilized iteration finite at small `reg`. **Scale of validity**: dimensionless (log domain). Used in `src/holonomy_lib/discrete_geometry/ricci.py`. Cataloged 2026-10-25. |
| `RICCI_FLOW_SURGERY_PERIOD_DEFAULT` | 🔬 | `10` | Number of Ricci-flow steps between surgery passes. Ni-Lin-Luo-Gao (2019) §3.2 use values in [10, 15] for community detection; 10 is the more aggressive choice. **Scale of validity**: smaller graphs / faster surgery → smaller period (down to 5); huge graphs may want 20+. Used in `ricci_flow_with_surgery`. |
| `RICCI_FLOW_SURGERY_THRESHOLD_DEFAULT` | 🔬 | `3.0` | Surgery cutoff multiplier — edges with weight ≥ 3.0 × initial mean weight are removed as "necks". Sia (2019) and Ni (2019) use thresholds in [2, 3]; 3.0 is the conservative end (less aggressive removal). **Scale of validity**: graphs with wider initial-weight distributions may need a larger threshold to avoid removing legitimate heavy edges. Used in `ricci_flow_with_surgery`. |
| `FORMAN_TRIANGLE_CONTRIBUTION` | ✅ | `3.0` | Per-edge contribution from each incident triangular 2-face in augmented Forman-Ricci. **Derived**: a triangle has 3 edges, each sees one 2-face, and Forman's cell-complex prefactor on the 2-face term collapses to 1 in the graph specialization. Samal et al. (2018), §"Augmented Forman curvature". Used in `src/holonomy_lib/discrete_geometry/forman.py`. |
| `FORMAN_NODE_WEIGHT_DEFAULT` | ✅ | `1.0` | Unit vertex weights in the graph specialization of Forman curvature (Sreejith et al. 2016, eq. 1): an adjacency matrix carries edge weights only, so every vertex weight is one. **Derived** by the input representation. Used in `src/holonomy_lib/discrete_geometry/forman.py`. Cataloged 2026-10-25. |

### spectral

| Name | Status | Value | Justification |
|---|---|---|---|
| `MAGNETIC_CHARGE_DEFAULT` | 🔬 | `0.25` | Default magnetic charge `q ∈ [0, 1)` for `spectral.magnetic.*` Laplacians. Furutani et al. (2020), §5 identify `q = 1/4` as the value that maximally separates directed eigenmodes for the directed cycle and recommend it as the standard literature default. **Scale of validity**: dimensionless ratio of magnetic flux units; does not depend on `n`. Re-derive if the graph has an exceptionally heavy directional component (very asymmetric weights) where a smaller `q` may preserve more of the symmetric spectrum. Used in `src/holonomy_lib/spectral/magnetic.py`. |
| `CHEBYSHEV_ORDER_DEFAULT` | 🔬 | `30` | Default Chebyshev expansion order `K` for `heat_kernel_chebyshev`. The Chebyshev coefficients of `exp(−τy)` on `[−1, 1]` are the modified Bessel functions `I_k(τ)`, which decay super-exponentially in `k` once `k > τ`. **Scale of validity**: for diffusion times `t ≤ 5` on the symmetric-normalized Laplacian (so `τ = t · λ_max / 2 ≤ 5`), `K = 30` gives sub-1e-12 relative error. For larger `t` use `K ≈ 2τ + 10` or pass an explicit `K`. Hammond-Vandergheynst-Gribonval (2011), §6 Fig. 4 reproduces this empirically. Used in `src/holonomy_lib/spectral/heat_kernel.py`. |
| `LAMBDA_MAX_L_SYM` | ✅ | `2.0` | Canonical upper bound on the spectrum of the symmetric-normalized Laplacian `L_sym = I − D^{−1/2} A D^{−1/2}` (Chung 1997, Thm. 1.7). **Derived**: holds for any non-negative symmetric `A` regardless of `n`. Used in `src/holonomy_lib/spectral/heat_kernel.py` as the default Chebyshev rescale bound; callers using non-`L_sym` Laplacians must override. Note: `2.0` is also in `ALLOWED_LITERALS` as a doubling identity; this entry documents the *semantic* role of the default. |
| `NORMALIZED_LAPLACIAN_SPECTRAL_RADIUS` | ✅ | `2.0` | Same bound as `LAMBDA_MAX_L_SYM` (Chung 1997, Lemma 1.7), used in `diffusion_map` to scale its dtype-aware eigenvalue tolerance `eps(dtype) · n · 2`. **Derived**. Used in `src/holonomy_lib/spectral/diffusion_map.py`. Cataloged 2026-10-25. |

### info_geometry

(no entries yet)

### sheaf

| Name | Status | Value | Justification |
|---|---|---|---|
| `SHEAF_DENSE_BYTES_CAP` | 🔬 | `2 * 2**30` (= 2 GiB) | Pre-flight byte cap on the dense δ + staging tensors built inside `sheaf_coboundary`. The v1 sheaf path is dense; a graph with n_e=50k edges, n_v=10k nodes, d=16 stalks at float64 would allocate ~100 TB silently. **Scale of validity**: chosen as a generous threshold for a research-grade primitive on commodity hardware (24-32 GB host RAM); raise when running on big-memory machines, lower on memory-constrained ones. Past the cap use `sheaf_coboundary_sparse` / `sheaf_laplacian_sparse`. Used in `src/holonomy_lib/sheaf/laplacian.py`. |
| `SHEAF_KERNEL_HEADROOM` | 🔬 | `16` | Extra bottom modes requested beyond `k` in `sheaf_bottom_spectrum` so the kernel of `L_F` (global sections, λ≈0) can be dropped and still leave `k` NONZERO modes. The global-section space of a (near-)connected sheaf corpus is low-dimensional (= number of connected components for the trivial sheaf); 16 covers typical corpora, and the solve raises with instructions when the kernel exceeds the headroom (pass a larger `kernel_headroom`). From synoros-substrate `sheaf_scale.py` (`KERNEL_HEADROOM`, validated 2026-06-06). **Scale of validity**: corpora with ≤ 16-dimensional global-section space; heavily disconnected sheaves need the parameter raised. Used in `src/holonomy_lib/sheaf/laplacian.py`. |

### lie

| Name | Status | Value | Justification |
|---|---|---|---|
| `SO3_DIM` | ✅ | `3` | Mathematical dimension of SO(3): the group lives in R^{3×3} and its Lie algebra is R^3. **Derived**: hard-coded by the definition of "3D rotation"; not tunable. Used everywhere in `src/holonomy_lib/lie/so3.py` for shape checks and explicit `(3, 3)`/`(3,)` constructors. |
| `QUATERNION_DIM` | ✅ | `4` | Mathematical dimension of a quaternion. **Derived**: hard-coded by the definition; not tunable. Used in `src/holonomy_lib/lie/so3.py` for the Shoemake-1992 random-rotation construction. |
| `SO3_LOG_NEAR_PI_RAD` | 🔬 | `1e-7` | Angle threshold (in radians, measured below π) above which `so3_log` switches to the "near-π rotation" branch that recovers the axis from `Sym(R) − cos(θ) I = (1 − cos θ) a aᵀ` rather than the antisymmetric `vee((R − R^T)/2)`. The code uses `max(SO3_LOG_NEAR_PI_RAD, sqrt(eps(dtype)))`, so float32 gets a wider band automatically. The calibration below was done when the general branch used `arccos`; it now uses `atan2`, so the crossover was not re-measured. Empirically calibrated on float64: the general branch wins by 4-9 orders of magnitude for gap ∈ [1e-6, 1e-2] (general's `vee/sin` amplifies noise by ~1/gap, but float64 input precision is ~2e-16 so post-amplification accuracy stays well below the near-π branch's `O(gap²)` model error). Below gap ~ 1e-7 the `arccos((trace−1)/2)` step in the general branch becomes the precision bottleneck and the near-π branch wins. `1e-7` is the empirical crossover for float64. **Scale of validity**: dimensionless angle in radians, calibrated for float64. Float32 callers should override to ~1e-3 (their amplification tolerance is ~6 orders of magnitude weaker). Full empirical table in the module docstring of `src/holonomy_lib/lie/so3.py`. |
| `SO3_LOG_SERIES_SECOND_ORDER` | ✅ | `1/6` | Second-order coefficient of `asin(s)/s = 1 + s²/6 + ...`, written as `1/factorial(SO3_DIM)`. `so3_log` uses this small-angle series instead of the old near-zero branch, which snapped tiny rotations onto a canonical axis (it mapped `log(exp([0, 1e-7, 0]))` to `[9.9e-8, 0, 0]`). **Derived**. Used in `src/holonomy_lib/lie/so3.py`. Cataloged 2026-10-25. |
| `SO3_EXP_SINC_LINEAR`, `SO3_EXP_SINC_QUADRATIC`, `SO3_EXP_COSC_LINEAR`, `SO3_EXP_COSC_QUADRATIC` | ✅ | `1/6`, `1/120`, `1/24`, `1/720` | Reciprocal factorials in the Taylor series of Rodrigues' coefficients, `sin θ/θ = 1 - θ²/3! + θ⁴/5!` and `(1 - cos θ)/θ² = 1/2 - θ²/4! + θ⁴/6!`, used by `so3_exp` when `θ² <= sqrt(eps(dtype))` so the derivative at ω = 0 is exact and nothing divides by zero. **Derived**. Used in `src/holonomy_lib/lie/so3.py`. Cataloged 2026-10-25. |
| `_SUPPORTED_L_MAX` | ✅ | `4` | Highest degree `l` with hand-written real spherical harmonics in `real_spherical_harmonics` (Edmonds 1957 closed forms for `l ≤ 4`). **Derived** by implementation scope, not tuned. Used in `src/holonomy_lib/lie/spherical_harmonics.py`. Cataloged 2026-10-25. |

### tensor_calculus

(no entries yet)

### hyperbolic

| Name | Status | Value | Justification |
|---|---|---|---|
| `_N_DAVIES_MANDOUVALOS` | ✅ | `3` | Intrinsic dim with the Davies–Mandouvalos (1988) closed-form heat kernel `k^3_t(d) = (4πt)^{-3/2} exp(-t - d²/4t) d/sinh d`. **Derived**: 3 is fixed by the mathematics — the lowest dim where the Grigor'yan–Noguchi recursion gives an elementary closed form starting from H^1 (the Gaussian). Used in `src/holonomy_lib/hyperbolic/heat_kernel.py` as the dispatch target for n=3. Cataloged 2026-05-27. |
| `_N_DAVIES_MANDOUVALOS_N5_CLOSED` | ✅ | `5` | Intrinsic dim with the hand-derived closed form `k^5_t(r) = (4πt)^{-5/2} exp(-4t - r²/4t) · [r²·sinh r + 2t·(r·cosh r − sinh r)] / sinh³ r`. **Derived**: the operator chain `(1/sinh r ∂_r)² exp(-4t - r²/4t)` applied analytically. Using the closed form rather than an autograd recursion is both faster (no `torch.autograd.grad` calls) and more precise (no compounded float noise). Used directly for n=5 dispatch in `heat_kernel.py`. Cataloged 2026-05-28. |
| `_N_OPERATOR_CHAIN_N7_CLOSED` | ✅ | `7` | Intrinsic dim with the hand-derived closed form `k^7_t(r) = (4πt)^{-7/2} exp(-9t - r²/4t) · [r³·sinh²r + 6r²t·sinh r·cosh r + (8t² − 6t)·r·sinh²r + 12t²(r − sinh r·cosh r)] / sinh⁵ r`. **Derived**: one corrected operator-chain step `(1/sinh r ∂_r)` beyond the n=5 closed form; the integer coefficients (6, 8, 9, 12, 16, 15) are universal — from the `(1/sinh r ∂_r)³` expansion, not tuning — and are composed from allowed literals in code. Verified in `notes/verification/heat_kernel_n7_sympy.py` (closed form = recursion = operator chain to ~1e-13; H^7 heat-equation residual ~1e-12; r→0 limit `1 + 2t + 16t²/15` to 16 digits). Used directly for n=7 and as the higher-precision seed for the odd n ≥ 9 recursion. Cataloged 2026-05-29. |
| `_N_GRIGORYAN_INTEGRAL` | ✅ | `2` | Intrinsic dim with the Grigor'yan integral form for the heat kernel (`k^2_t(d) = √2 · exp(-t/4) · (4πt)^{-3/2} · ∫_d^∞ s exp(-s²/4t) / √(cosh s - cosh d) ds`). **Derived**: 2 is fixed by the mathematics — the lowest even dim where the kernel does not have an elementary closed form. Used as the recursion seed for higher even dims. Cataloged 2026-05-27. (Note: 2 is also in `ALLOWED_LITERALS`; this entry documents the *semantic* role of the seed dim.) |
| `HEAT_KERNEL_QUADRATURE_NODES` | 🔬 | `32` | Default Gauss–Legendre node count for the n=2 integral `∫_d^{d+S} s exp(-s²/4t) / √(cosh s - cosh d) ds`. Doubling-node refinement gave sub-1e-10 relative agreement on `t ∈ [1e-2, 10]`, `d ∈ [0, 5]` (validated by `tests/hyperbolic/test_heat_kernel.py::test_n2_quadrature_node_count_convergence`). **Scale of validity**: integrand is well-resolved with 32 nodes when the upper bound is set by `HEAT_KERNEL_QUADRATURE_TAIL_BUDGET`. Raise to 64+ for unusual regimes (t ≫ 10 where the support widens). Cataloged 2026-05-27. |
| `HEAT_KERNEL_QUADRATURE_TAIL_BUDGET` | 🔬 | `20.0` | Truncates the n=2 integral at `d + sqrt(4 · tail_budget · t)`: the Gaussian factor `exp(-s²/(4t))` has fallen by `exp(-tail_budget) ≈ 2e-9` there. This bounds the Gaussian factor, not a uniform relative error once the geometric factors are included (corrected 2026-10-25: the old substitution made the endpoint `d + tail_budget · t`, 1.4% low at t=0.01, d=0.2). **Scale of validity**: dimensionless multiplier of `t` in the upper-bound formula; does not need to change with input size. Increase for tighter relative-error requirements. Cataloged 2026-05-27. |
| `_GAUSSIAN_TIME_SCALE` | ✅ | `4` | The `4` in the Gaussian normalization `(4πt)^(-n/2) exp(-r²/(4t))`. **Derived** from the heat equation, not fitted. Used in `src/holonomy_lib/hyperbolic/heat_kernel.py`. Cataloged 2026-10-25. |
| `_RADIAL_SERIES_TERMS` | 🔬 | `16` | Terms (powers of `r²`, through `r³⁰`) in the small-radius series for odd n ≥ 5, used below a dtype-aware switching radius where the closed forms cancel catastrophically (n=7, d=1e-4 was off by 3.5x in float64, 2e8x in float32). The switch balances the omitted `r³²` term against `eps / r^(n-3)` cancellation. **Scale of validity**: checked against 60-digit mpmath operator chains at t ∈ {0.01, 1, 10}, near 0 and either side of the switch, n ∈ {5, 7, 9}. Used in `src/holonomy_lib/hyperbolic/heat_kernel.py`. Cataloged 2026-10-25. |
| `FRECHET_MAX_ITER_DEFAULT` | 🔬 | `100` | Iteration budget for `frechet_mean`. Convergence is checked, not assumed: when the Riemannian-gradient residual still exceeds tolerance at the budget, the function warns. **Scale of validity**: tightly clustered points converge in a few steps; widely spread points (the symmetric radius-4 H² triple) need backtracking and more iterations. Used in `src/holonomy_lib/hyperbolic/frechet_mean.py`. Cataloged 2026-10-25. |
| `FRECHET_SYNC_EVERY_DEFAULT` | ⚖️ | `8` | Host-sync cadence of the Fréchet convergence check, same rationale as `SINKHORN_SYNC_EVERY_DEFAULT` (CONVENTIONS §4.1): at most seven extra cheap iterations after convergence. Used in `src/holonomy_lib/hyperbolic/frechet_mean.py`. Cataloged 2026-10-25. |
| `FRECHET_MAX_BACKTRACKS_DEFAULT` | 🔬 | `16` | Bounded backtracking line search over step sizes `1, 1/2, ..., 2⁻¹⁵`. If no step decreases the objective, that batch member stays put and the residual warning reports it. **Scale of validity**: halving fifteen times covers step-size mismatches up to ~3e4; raise only for badly scaled inputs. Used in `src/holonomy_lib/hyperbolic/frechet_mean.py`. Cataloged 2026-10-25. |


### simplicial

| Name | Status | Value | Justification |
|---|---|---|---|
| `VR_CANDIDATE_TILE_DEFAULT` | 🔬 | `1024` | Candidate simplices processed per tile when `vietoris_rips_*` extends a complex by one dimension. Bounds the temporary integer and boolean tables to `O(1024 · simplex_width)` entries. **Scale of validity**: a memory budget with no effect on results; raise on large-memory devices for fewer kernel launches. Used in `src/holonomy_lib/simplicial/vietoris_rips.py`. Cataloged 2026-10-25. |

### optimization

`RiemannianSGD`'s `lr` default (`1e-2`) is a user choice, not an architectural constant; tune it per problem.

### experimental

| Name | Status | Value | Justification |
|---|---|---|---|
| `EIGENMAPS_MAX_STEPS_DEFAULT`, `EIGENMAPS_LR_DEFAULT` | ⚠️ | `200`, `0.05` | Work budget and step size of the experimental `hyperbolic_laplacian_eigenmaps`. Unresolved by design: its objective is attraction-only, so there is no convergence target that would justify a budget; more steps contract components further. Revisit when the objective gains a non-collapse term. Used in `src/holonomy_lib/experimental/laplacian_eigenmaps.py`. Cataloged 2026-10-25. |

---

## Re-derivation at scale

When a primitive in this library is used at a new scale (e.g., the consumer
goes from N=1000 to N=100000), the corresponding catalog entry should record
whether the constant transfers or needs re-derivation. Operations that scale
with N should be parameterized so the value auto-derives; ones that don't
need explicit re-validation at the new scale.
