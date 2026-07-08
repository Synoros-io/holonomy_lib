# Magic numbers catalog — holonomy-lib

Per the **no magic numbers** rule (see `README.md` §"Audit discipline"), every numerical
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

Same as synoros-substrate's parent library. See `src/holonomy_lib/audit.py` for the
ALLOWED_LITERALS set:

| Literal | Justification |
|---|---|
| `0, 1, -1` (and float forms) | Mathematical identities |
| `0.5` and `2` (and `-2`, float forms) | Halving/doubling identities. Symmetric/antisymmetric parts `(Z ± Zᵀ)/2`, midpoint mean `(a + b)/2`, quadratic-form coefficient `(½)xᵀAx`, triangular numbers `n(n+1)/2`, `2π`, 2-norm, second-derivative coefficients. On par with `0, 1, -1` for mathematical code. Added 2026-05-26. |
| `1024, 1024.0` | Universal binary unit conversion (KB↔MB↔GB) |
| `1000, 1000.0` | Universal SI time conversion (s↔ms) AND RNG-stream-offset multiplier |
| `1e-9` (`numerical_floor_convention`) | Posited convention: anti-divide-by-zero floor. Chosen smaller than float32 eps (~1.19e-7) for safety across dtypes. Used pervasively in `1.0 / max(x, 1e-9)` and similar guards. |

---

## Cataloged constants (per-module)

(Currently empty — populate as the library grows.)

### manifolds

| Name | Status | Value | Justification |
|---|---|---|---|
| `RETRACTION_RANDOMIZED_THRESHOLD` | 🔬 | `0.25` | `FixedRankManifold.retraction` switches from exact to randomized SVD when `r ≤ 0.25 · min(m, n)`. **Scale of validity**: derived from CPU benchmarks at 64-1024-square sizes (`notes/benchmark_baseline.md` 2026-05-27) — randomized SVD is strictly faster than full-and-truncate below this ratio, by 8× at r/min ≈ 12% and 37× at r/min ≈ 3%. At ratios above 25% the constant factors flip and exact wins back. Re-derive when GPU benchmarks land. Used in `src/holonomy_lib/manifolds/fixed_rank.py`. |
| `COMPARISON_BALL_VOLUME_QUADRATURE_NODES` | 🔬 | `64` | Default Gauss-Legendre node count for the model ball-volume integral `∫_0^r sn_κ(t)^{N-1} dt` in `manifolds/comparison.py`. The integrand is smooth/analytic on a finite interval, so Gauss-Legendre converges geometrically; for `κ = 0` (degree-`(N-1)` polynomial integrand) the rule is exact for `N ≤ 2·nodes`. **Scale of validity**: 64 nodes give sub-1e-12 agreement with the closed forms (Euclidean, sphere, hyperbolic) for `N` up to ~100 and `√κ·r` up to the model diameter (validated by `tests/manifolds/test_comparison.py`). Raise for very large `N` or near-diameter spherical radii. Cataloged 2026-05-28. |

### topology

(no entries yet)

### algebra

| Name | Status | Value | Justification |
|---|---|---|---|
| `RANDOMIZED_SVD_OVERSAMPLE_DEFAULT` | 🔬 | `5` | Halko-Martinsson-Tropp (2011) §1.2 standard recommendation for the oversampling parameter `p` in randomized SVD. Five extra projection columns typically yield relative spectral error below 1e-3 for matrices with smoothly decaying singular values past the truncation rank. **Scale of validity**: adequate when σ_{r+1} is meaningfully smaller than σ_r; increase to 10 (also literature-standard) when the spectrum is flat near the truncation. Used in `src/holonomy_lib/algebra/linear.py`. |
| `RANDOMIZED_SVD_N_ITER_DEFAULT` | 🔬 | `2` | Halko-Martinsson-Tropp (2011) §4.5 standard recommendation for subspace-iteration power method steps. Two iterations amplify the singular-value gap by (σ_{r+1}/σ_r)^(2q+1), sufficient for typical low-rank truncation. **Scale of validity**: fine for σ_{r+1}/σ_r ≲ 0.5; increase to 4–7 when ratio is closer to 1. Used in `src/holonomy_lib/algebra/linear.py`. (Note: value 2 is also in `ALLOWED_LITERALS` as a doubling identity; this entry documents the *semantic* role of the default.) |
| `LANCZOS_OVERSAMPLE_DEFAULT` | 🔬 | `10` | Extra Lanczos iterations beyond `k` for top-k eigensolver convergence. Saad (2011) §6.5 recommends 5–10 extra for reliable top-k Ritz convergence on well-conditioned matrices; we pick the upper end for safety. **Scale of validity**: for spectra with a clear top-k gap (extreme eigenvalues well-separated from bulk), 10 is comfortably sufficient. For interior eigenvalues or near-degenerate spectra, the caller should increase `oversample` or use shift-and-invert (planned). Used in `src/holonomy_lib/algebra/lanczos.py`. |

| `SPARSE_SA_ITER_SOLVER_N` | 🔬 | `8000` | Sparse smallest-algebraic (`which="SA"`) CPU dispatch threshold: at or below this `n` use SciPy `eigsh` shift-invert (full sparse LU), above it LOBPCG + `spilu` incomplete-LU (the LU fill-in grows too fast past this size). **Throughput threshold only** — both branches compute the same bottom-k modes (verified to ~1e-14 in `tests/algebra/test_eigsh_sparse.py`). Set in synoros-substrate `sheaf_scale.py` on sheaf Laplacians up to ~150k stalk-dims (2026-06-06). **Scale of validity**: CPU, Laplacian-like sparsity (bounded fill per row); re-derive on substantially denser matrices or different hardware. Used in `src/holonomy_lib/algebra/eigsh_sparse.py`. |
| `SPARSE_SA_SHIFT_INVERT_POLE` | 🔬 | `-1e-6` | Shift-invert pole for the sparse SA `eigsh` branch: a small NEGATIVE shift so `(A − σI)` stays non-singular despite a λ=0 kernel (Laplacians), placing the pole just below a PSD spectrum so the inverted operator's dominant modes are A's smallest. A pole offset in the spirit of the library's `1e-9` floor convention (three orders larger, so the pole stays clear of float64 kernel-mode noise ~1e-10). From synoros-substrate `sheaf_scale.py` (validated 2026-06-06). **Scale of validity**: PSD operators with `max(diag) ≫ 1e-6`; for spectra scaled near or below 1e-6, pass an explicit `sigma`. Used in `src/holonomy_lib/algebra/eigsh_sparse.py`. |
| `SPARSE_SA_SPILU_LOBPCG_KNOBS` | 🔬 | `eps_frac 1e-3, drop_tol 1e-3, fill_factor 10, tol 1e-6, maxiter 100` | The large-n CPU LOBPCG branch's numerical knobs (one entry for the set: `SPILU_EPS_FRAC`, `SPILU_DROP_TOL`, `SPILU_FILL_FACTOR`, `LOBPCG_TOL`, `LOBPCG_MAXITER`). They set CONVERGENCE RATE and achieved precision, not WHICH modes — verified against the eigsh branch. eps-shift `eps_frac · mean(diag A)` makes a singular A factorable by `spilu`; drop_tol/fill_factor trade preconditioner quality against memory (Saad 2003 §10.4); tol/maxiter are the LOBPCG stopping criteria (Knyazev 2001). From synoros-substrate `sheaf_scale.py` (validated 2026-06-06 on sheaf Laplacians to ~150k stalk-dims). **Scale of validity**: tied to the Laplacian sparsity class; loosen drop_tol / raise fill_factor if LOBPCG stalls on denser operators. Used in `src/holonomy_lib/algebra/eigsh_sparse.py`. |
### provenance

| Name | Status | Value | Justification |
|---|---|---|---|
| `HEX_PREFIX_LEN` | 🔬 | `16` | Length (in hex chars) of the truncated sha256 prefix used as provenance IDs. 16 chars = 64 bits ≈ collision-free up to ~2³² operations by the birthday bound. **Scale of validity**: fine for interactive mech-interp sessions and most batch runs. Bump to 32 chars (128 bits) for million-op production traces. Used in `src/holonomy_lib/provenance/protocol.py`. |
| `SKETCH_SAMPLES` | 🔬 | `64` | Number of evenly-strided samples drawn from a tensor's flattened content when `hash_mode="sketch"` is active. Combined with shape, dtype, sum, and std, gives an O(1)-bytes digest regardless of tensor size. **Scale of validity**: empirically tested for zero collisions on 200 random `(16, 16)` float64 Laplacian inputs (`tests/provenance/test_protocol.py::TestSketchHash::test_sketch_collision_rate_on_random_tensors`). Structural-collision risk (two tensors that share the strided positions + sum + std but differ elsewhere) is bounded but non-zero — full mode remains the library default. Bump if you see collisions in tensors larger than ~10⁶ elements where 64 strided samples leaves the unsampled regions too sparsely covered. Used in `src/holonomy_lib/provenance/protocol.py`. |
| `LLM_CONTEXT_MAX_OPS_DEFAULT` | 🔬 | `20` | Default cap on the per-op-id rows that `ProvenanceRegistry.to_llm_context()` lists before collapsing the rest into `"...and N more"`. Keeps agent-prompt output bounded for huge registries while staying useful for typical research-scale chains. Caller-overridable via the `max_ops` kwarg. **Scale of validity**: visual / prompt-budget tuning; not load-bearing for correctness. Used in `src/holonomy_lib/provenance/protocol.py`. |
| `DISPLAY_PREVIEW_COUNT` | 🔬 | `5` | Preview limit for roots / leaves in `to_llm_context()` and for per-op-id hex lists in `diff_summary()`. Five is enough to identify pattern (or confirm divergence in a diff) without padding the human-readable output. **Scale of validity**: visual tuning; not load-bearing for correctness. Used in `src/holonomy_lib/provenance/protocol.py`. |
| `LOAD_DRIFT_DETAIL_LIMIT` | 🔬 | `10` | Detail-line cap inside the `ProvenanceVersionWarning` emitted by `ProvenanceRegistry.load()`. Caps both the drifted-nodes and unknown-op_ids lists so a huge registry doesn't produce an unreadable warning. Slightly larger than `DISPLAY_PREVIEW_COUNT` because drift is more actionable per entry (the user typically wants to see which exact ops drifted). **Scale of validity**: visual tuning; not load-bearing for correctness. Used in `src/holonomy_lib/provenance/protocol.py`. |
| `TENSOR_SLICE_INLINE_LIMIT` | 🔬 | `256` | Max element count for `agent.tensor_slice` to return raw values inline; above this, the tool returns a stats summary and asks for a smaller slice. 256 = a 16×16 matrix worth — big enough for typical neighborhood inspection, small enough to fit a single LLM tool-response without blowing up the agent's context window. **Scale of validity**: context-budget tuning; raise if you're driving against models with much larger context windows and want to allow bigger inline slices. Used in `src/holonomy_lib/provenance/agent.py`. |
| `TENSOR_SPECTRAL_DEFAULT_K` | 🔬 | `10` | Default `k` for `agent.tensor_eigenvalues` and `agent.tensor_singular_values`. Ten covers the typical "spectral overview" question (where does the matrix concentrate? what's the gap between top values?). Callers can raise it explicitly when they want the full spectrum. **Scale of validity**: agent-prompt-budget tuning. Used in `src/holonomy_lib/provenance/agent.py`. |
| `PYTHON_SLICE_ARITY` | ✅ | `3` | Python's `slice()` constructor takes `(start, stop, step)` — three components. Pinned in `agent.py` so the slice-expression parser doesn't trip the audit's bare-literal check. Fixed by the Python language, not a tuning parameter. Used in `src/holonomy_lib/provenance/agent.py`. |

### discrete_geometry

| Name | Status | Value | Justification |
|---|---|---|---|
| `SINKHORN_REG_DEFAULT` | 🔬 | `0.01` | Cuturi (2013) §4 — middle-ground entropic regularization ε for Sinkhorn OT. Small enough to keep the entropic bias on W₁ < 1% for typical graph-metric costs; large enough to converge in O(100) iterations without underflow. **Scale of validity**: shortest-path costs in [1, ~diameter]; scale proportionally if the graph's distance scale is much wider. Used in `src/holonomy_lib/discrete_geometry/ricci.py`. |
| `SINKHORN_N_ITER_DEFAULT` | 🔬 | `100` | Cuturi (2013) §4 — typical iteration count to reach relative-change < 1e-4 for graph-metric costs at default `reg=0.01`. **Scale of validity**: smaller `reg` requires proportionally more iterations. Note: 100 iter often lands inside the convergence-plateau region for wide-cost-range inputs (asymmetry ~1.75e-2); crank to ~2000 with `tol=1e-12` for machine-precision symmetry. Used in `src/holonomy_lib/discrete_geometry/ricci.py`. |
| `SINKHORN_TOL_DEFAULT` | ⚖️ | `1e-9` | Log-domain Sinkhorn early-stop tolerance — iteration stops once max\|Δlog_u\| < tol. Reuses the library's `numerical_floor_convention`: well below typical use cases' noise floor, but not chasing machine-precision residuals. **Scale of validity**: dimensionless (acts on dual variables); does not need re-derivation across input sizes. Used in `src/holonomy_lib/discrete_geometry/ricci.py`. |
| `SINKHORN_TILE_DEFAULT` | 🔬 | `256` | Pairs processed simultaneously inside the Sinkhorn loop. Peak inner-broadcast memory = O(tile · n² · element_size); at tile=256 and float64 that caps at ~16 MB for n ≤ 256. **Scale of validity**: balances Python-loop overhead against memory pressure on the inner `(tile, n, n)` broadcast. Re-derive when running on a memory-constrained GPU or at n > ~512. Used in `src/holonomy_lib/discrete_geometry/ricci.py`. |
| `SINKHORN_SYNC_EVERY_DEFAULT` | ⚖️ | `8` | How often to host-sync Sinkhorn's convergence check (`.item()` on the delta tensor). Per-iter syncs add tens of µs of kernel-queue wait on GPU; cadence=8 amortizes that cost by 8× while only delaying convergence detection by at most 7 extra iterations. **Scale of validity**: dimensionless cadence; does not depend on input size or dtype. Cataloged 2026-05-27. Used in `src/holonomy_lib/discrete_geometry/ricci.py`. |
| `DISCONNECTED_DISTANCE_MULTIPLIER` | 🔬 | `1000.0` | Replaces +inf in the shortest-path matrix with a large finite value (= 1000 × max-finite-distance) to keep Sinkhorn numerics stable across disconnected components. **Scale of validity**: any value much larger than the graph diameter works equivalently; 1000× chosen so that exp(−d/reg) for cross-component pairs underflows to zero at the default `reg=0.01`. Used in `src/holonomy_lib/discrete_geometry/ricci.py`. |
| `RICCI_FLOW_SURGERY_PERIOD_DEFAULT` | 🔬 | `10` | Number of Ricci-flow steps between surgery passes. Ni-Lin-Luo-Gao (2019) §3.2 use values in [10, 15] for community detection; 10 is the more aggressive choice. **Scale of validity**: smaller graphs / faster surgery → smaller period (down to 5); huge graphs may want 20+. Used in `ricci_flow_with_surgery`. |
| `RICCI_FLOW_SURGERY_THRESHOLD_DEFAULT` | 🔬 | `3.0` | Surgery cutoff multiplier — edges with weight ≥ 3.0 × initial mean weight are removed as "necks". Sia (2019) and Ni (2019) use thresholds in [2, 3]; 3.0 is the conservative end (less aggressive removal). **Scale of validity**: graphs with wider initial-weight distributions may need a larger threshold to avoid removing legitimate heavy edges. Used in `ricci_flow_with_surgery`. |
| `FORMAN_TRIANGLE_CONTRIBUTION` | ✅ | `3.0` | Per-edge contribution from each incident triangular 2-face in augmented Forman-Ricci. **Derived**: a triangle has 3 edges, each sees one 2-face, and Forman's cell-complex prefactor on the 2-face term collapses to 1 in the graph specialization. Samal et al. (2018), §"Augmented Forman curvature". Used in `src/holonomy_lib/discrete_geometry/forman.py`. |

### spectral

| Name | Status | Value | Justification |
|---|---|---|---|
| `MAGNETIC_CHARGE_DEFAULT` | 🔬 | `0.25` | Default magnetic charge `q ∈ [0, 1)` for `spectral.magnetic.*` Laplacians. Furutani et al. (2020), §5 identify `q = 1/4` as the value that maximally separates directed eigenmodes for the directed cycle and recommend it as the standard literature default. **Scale of validity**: dimensionless ratio of magnetic flux units; does not depend on `n`. Re-derive if the graph has an exceptionally heavy directional component (very asymmetric weights) where a smaller `q` may preserve more of the symmetric spectrum. Used in `src/holonomy_lib/spectral/magnetic.py`. |
| `CHEBYSHEV_ORDER_DEFAULT` | 🔬 | `30` | Default Chebyshev expansion order `K` for `heat_kernel_chebyshev`. The Chebyshev coefficients of `exp(−τy)` on `[−1, 1]` are the modified Bessel functions `I_k(τ)`, which decay super-exponentially in `k` once `k > τ`. **Scale of validity**: for diffusion times `t ≤ 5` on the symmetric-normalized Laplacian (so `τ = t · λ_max / 2 ≤ 5`), `K = 30` gives sub-1e-12 relative error. For larger `t` use `K ≈ 2τ + 10` or pass an explicit `K`. Hammond-Vandergheynst-Gribonval (2011), §6 Fig. 4 reproduces this empirically. Used in `src/holonomy_lib/spectral/heat_kernel.py`. |
| `LAMBDA_MAX_L_SYM` | ✅ | `2.0` | Canonical upper bound on the spectrum of the symmetric-normalized Laplacian `L_sym = I − D^{−1/2} A D^{−1/2}` (Chung 1997, Thm. 1.7). **Derived**: holds for any non-negative symmetric `A` regardless of `n`. Used in `src/holonomy_lib/spectral/heat_kernel.py` as the default Chebyshev rescale bound; callers using non-`L_sym` Laplacians must override. Note: `2.0` is also in `ALLOWED_LITERALS` as a doubling identity; this entry documents the *semantic* role of the default. |

### info_geometry

(no entries yet)

### sheaf

| Name | Status | Value | Justification |
|---|---|---|---|
| `SHEAF_DENSE_BYTES_CAP` | 🔬 | `2 * 2**30` (= 2 GiB) | Pre-flight byte cap on the dense δ + staging tensors built inside `sheaf_coboundary`. The v1 sheaf path is dense; a graph with n_e=50k edges, n_v=10k nodes, d=16 stalks at float64 would allocate ~100 TB silently. **Scale of validity**: chosen as a generous threshold for a research-grade primitive on commodity hardware (24-32 GB host RAM); raise when running on big-memory machines, lower on memory-constrained ones. A sparse path is on the v0.2 roadmap. Used in `src/holonomy_lib/sheaf/laplacian.py`. |
| `SHEAF_KERNEL_HEADROOM` | 🔬 | `16` | Extra bottom modes requested beyond `k` in `sheaf_bottom_spectrum` so the kernel of `L_F` (global sections, λ≈0) can be dropped and still leave `k` NONZERO modes. The global-section space of a (near-)connected sheaf corpus is low-dimensional (= number of connected components for the trivial sheaf); 16 covers typical corpora, and the solve raises with instructions when the kernel exceeds the headroom (pass a larger `kernel_headroom`). From synoros-substrate `sheaf_scale.py` (`KERNEL_HEADROOM`, validated 2026-06-06). **Scale of validity**: corpora with ≤ 16-dimensional global-section space; heavily disconnected sheaves need the parameter raised. Used in `src/holonomy_lib/sheaf/laplacian.py`. |

### lie

| Name | Status | Value | Justification |
|---|---|---|---|
| `SO3_DIM` | ✅ | `3` | Mathematical dimension of SO(3): the group lives in R^{3×3} and its Lie algebra is R^3. **Derived**: hard-coded by the definition of "3D rotation"; not tunable. Used everywhere in `src/holonomy_lib/lie/so3.py` for shape checks and explicit `(3, 3)`/`(3,)` constructors. |
| `QUATERNION_DIM` | ✅ | `4` | Mathematical dimension of a quaternion. **Derived**: hard-coded by the definition; not tunable. Used in `src/holonomy_lib/lie/so3.py` for the Shoemake-1992 random-rotation construction. |
| `SO3_LOG_NEAR_ZERO_RAD` | 🔬 | `1e-6` | Angle threshold (in radians) below which `matrix_to_axis_angle` switches to the "near-zero rotation" branch and returns a canonical axis `(1, 0, 0)`. The default `sin(θ) / sin(θ_clamped)` division becomes ill-conditioned within ~√eps ≈ 1.5e-8 of zero; `1e-6` is two orders of magnitude above that and well below any rotation that meaningfully transports information. **Scale of validity**: dimensionless angle in radians; does not depend on input size. Used in `src/holonomy_lib/lie/so3.py::matrix_to_axis_angle`. |
| `SO3_LOG_NEAR_PI_RAD` | 🔬 | `1e-7` | Angle threshold (in radians, measured below π) above which `matrix_to_axis_angle` switches to the "near-π rotation" branch that recovers the axis from `(R + I)/2` rather than the antisymmetric `vee((R − R^T)/2)`. Empirically calibrated on float64: the general branch wins by 4-9 orders of magnitude for gap ∈ [1e-6, 1e-2] (general's `vee/sin` amplifies noise by ~1/gap, but float64 input precision is ~2e-16 so post-amplification accuracy stays well below the near-π branch's `O(gap²)` model error). Below gap ~ 1e-7 the `arccos((trace−1)/2)` step in the general branch becomes the precision bottleneck and the near-π branch wins. `1e-7` is the empirical crossover for float64. **Scale of validity**: dimensionless angle in radians, calibrated for float64. Float32 callers should override to ~1e-3 (their amplification tolerance is ~6 orders of magnitude weaker). Full empirical table in the module docstring of `src/holonomy_lib/lie/so3.py`. |

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
| `HEAT_KERNEL_QUADRATURE_TAIL_BUDGET` | 🔬 | `20.0` | Truncate the n=2 integral at `d + sqrt(tail_budget · t)`. The integrand decays as `exp(-s²/(4t))`, so `s² > 20·t` makes the contribution under `exp(-20) ≈ 2e-9` — below the library's `numerical_floor_convention`. **Scale of validity**: dimensionless multiplier of `t` in the upper-bound formula; does not need to change with input size. Increase for tighter relative-error requirements. Cataloged 2026-05-27. |


### optimization

(All Riemannian-optimizer hyperparameters — `lr`, `betas`, `eps`,
`weight_decay`, `momentum` — are treated as user choices, not
architectural constants. Defaults match `torch.optim` conventions
for ergonomic consistency. The audit's `DISPLAY_VAR_NAMES` covers
these names.)

### probability

(no entries yet)

---

## Re-derivation at scale

When a primitive in this library is used at a new scale (e.g., the consumer
goes from N=1000 to N=100000), the corresponding catalog entry should record
whether the constant transfers or needs re-derivation. Operations that scale
with N should be parameterized so the value auto-derives; ones that don't
need explicit re-validation at the new scale.
