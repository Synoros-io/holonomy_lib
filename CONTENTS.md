# holonomy_lib contents

A flat inventory of every public primitive: signature, one-line summary,
and the paper to cite. See `README.md` for the project overview;
this file is the API map.

Each entry has the form:

> `module.thing(signature) → returns`
> One-line what-it-does. Math citation. Cross-references.

All tensors are batched-first: leading batch dim B, then the math dims.
Shapes use `B` for batch, `n`/`m`/`r`/etc. for math.

## Conventions (binding)

- **Batched-first.** Inputs take leading batch dim. Single-point use = pass `B = 1`.
  Operations are verified for `B ∈ {0, 1, > 1}`.
- **Device-agnostic.** Operations work on whatever device the input tensor is on.
  Use `dtype` and `device` constructor params to pin precision/placement.
- **Cited.** Every primitive's docstring has a `References:` section with the
  paper. The citations below are short; the full docstring has the eq. numbers.
- **Audit-clean.** No magic numerical literals in source. Posited constants
  cataloged in `notes/magic_numbers.md` with scale-of-validity.
- **Provenance-aware.** Top-level primitives are decorated with
  `@with_provenance`, so they emit content-addressable hex IDs when called
  inside `provenance.record()`. See §Provenance.

## Available imports (canonical paths)

```python
from holonomy_lib.manifolds import (
    FixedRankManifold, HeterogeneousKappaManifold,
    KappaStereographicManifold, LorentzManifold, LorentzianManifold,
    ProductManifold, SPDManifold,
    model_ball_volume, model_sphere_area,
)
from holonomy_lib.algebra import truncated_svd
from holonomy_lib.tensor_calculus import hosvd, mode_product, mode_unfolding
from holonomy_lib.algebra import cg, eigsh_sparse_sa, lanczos_eigsh, lsqr
from holonomy_lib.convex import regularized_lstsq
from holonomy_lib.spectral import (
    laplacian, magnetic, laplacian_eigenmaps, heat_kernel_chebyshev,
    effective_resistance, commute_time, diffusion_map,
)
from holonomy_lib.discrete_geometry import (
    ollivier_ricci_curvature,
    discrete_ricci_flow,
    ricci_flow_with_surgery,
    forman_ricci_simple,
    forman_ricci_augmented,
)
from holonomy_lib.info_geometry import (
    bregman_divergence,
    kl_divergence_categorical,
    kl_divergence_gaussian,
)
from holonomy_lib.optimization import RiemannianSGD, riemannian_sgd_step
from holonomy_lib.simplicial import (
    DenseSimplicialComplex, SparseSimplicialComplex,
    pairwise_distances, vietoris_rips_dense, vietoris_rips_sparse,
)
from holonomy_lib.topology import (
    betti_numbers, hodge_laplacian, persistence_diagrams,
)
from holonomy_lib.sheaf import (
    GraphSheaf, sheaf_coboundary, sheaf_laplacian, sheaf_dirichlet_energy,
)
from holonomy_lib.lie import so3, real_spherical_harmonics
from holonomy_lib.hyperbolic import (
    frechet_mean, hyperbolic_heat_kernel,
    hyperbolic_laplacian_eigenmaps, manifold_aware_inner,
)
from holonomy_lib import provenance
```

---

## §Manifolds: `holonomy_lib.manifolds`

Riemannian manifolds with batched-first, GPU-native operations.

### `FixedRankManifold(m, n, r, device="cpu", dtype=torch.float64)`
Fixed-rank matrix manifold M_r(m, n) ⊂ R^{m×n}. Points stored as SVD triples
(U, S, Vt) of shapes (B, m, r), (B, r), (B, r, n). Manifold dim `r·(m + n − r)`.

| Method | Signature | Returns |
|---|---|---|
| `random_point` | `(batch_size=1, generator=None)` | `(U, S, Vt)` |
| `dense` | `(point)` | `(B, m, n)` |
| `projection` | `(point, Z)` | `(B, m, n)` tangent |
| `inner` | `(point, U, V)` | `(B,)` |
| `norm` | `(point, V)` | `(B,)` |
| `retraction` | `(point, tangent)` | new `(U, S, Vt)` |
| `dim` | property | int |

Refs: Vandereycken (2013), Absil-Mahony-Sepulchre (2008), Mezzadri (2007).

### `SPDManifold(n, device="cpu", dtype=torch.float64)`
Symmetric positive definite matrices P(n) with the **affine-invariant** metric
`⟨U,V⟩_S = tr(S⁻¹ U S⁻¹ V)`. Points and tangents both `(B, n, n)` symmetric.
Manifold dim `n(n+1)/2`.

| Method | Signature | Returns |
|---|---|---|
| `random_point` | `(batch_size=1, generator=None)` | `(B, n, n)` SPD |
| `is_spd` | `(S)` | `(B,)` bool |
| `projection` | `(S, Z)` | symmetric `(B, n, n)` |
| `inner` | `(S, U, V)` | `(B,)` |
| `norm` | `(S, V)` | `(B,)` |
| `exp` | `(S, V)` | `(B, n, n)` exp_S(V) |
| `log` | `(S, T)` | `(B, n, n)` log_S(T) |
| `distance` | `(S, T)` | `(B,)` geodesic |
| `retraction` | `(S, V)` | = `exp(S, V)` |

Refs: Pennec-Fillard-Ayache (2006), Bhatia (2007), Sra-Hosseini (2015).

### `LorentzManifold(n, k=-1.0, device="cpu", dtype=torch.float64)`
Hyperboloid model of `n`-dim hyperbolic space of sectional curvature `k < 0`:
`H^n_k = { x ∈ R^{n+1} : ⟨x,x⟩_M = 1/k, x_0 > 0 }` where
`⟨x,y⟩_M = -x_0·y_0 + Σ x_i·y_i`. Default `k=-1` is the canonical unit
hyperboloid (Nickel-Kiela 2018). Points and tangents stored as `(B, n+1)`
ambient tensors. Intrinsic manifold dim `n`.

| Method | Signature | Returns |
|---|---|---|
| `random_point` | `(batch_size=1, generator=None)` | `(B, n+1)` |
| `origin` | `(batch_size=1)` | `(B, n+1)` north pole |
| `is_on_manifold` | `(x, atol=1e-9)` | `(B,)` bool |
| `minkowski_inner` | `(x, y)` | `(B,)` |
| `projection` | `(x, w)` | tangent `(B, n+1)` |
| `inner` | `(x, u, v)` | `(B,)` |
| `norm` | `(x, v)` | `(B,)` |
| `exp` | `(x, v)` | `(B, n+1)` exp_x(v) |
| `log` | `(x, y)` | `(B, n+1)` log_x(y) |
| `distance` | `(x, y)` | `(B,)` geodesic |
| `parallel_transport` | `(x, y, v)` | tangent at y, `(B, n+1)` |
| `retraction` | `(x, v)` | = `exp(x, v)` |
| `exp_0` | `(v_spatial)` | `(B, n+1)` (origin shortcut) |
| `log_0` | `(y)` | `(B, n)` (origin shortcut) |

Numerical notes:

- `log` uses `(α/sinh α)·u` rather than the textbook `β·u/‖u‖` to
  avoid catastrophic cancellation at x ≈ y, and internally
  reparameterizes via `α = 2·arcsinh(arg)` where `arg = √|k|·‖y-x‖_M/2`
  so `arccosh`'s derivative singularity at z=1 never enters the
  autograd graph.
- `distance` uses the same `2·arcsinh(√|k|·‖y-x‖_M/2)` identity for
  `d(x,x) = 0` exactness AND autograd-finite gradient (the naive
  `sqrt(clamp(…, min=0))` pattern would propagate `0·∞ = NaN` through
  backward at the boundary).
- `exp` re-projects onto the hyperboloid after each call to suppress
  drift; uses `_safe_sinhc` so backward is finite at v = 0.
- `log_0`, `exp_0`, `norm` use `_safe_sqrt` / `_safe_sinhc` /
  `_safe_arcsinhc` helpers that confine boundary-singular ops to the
  masked-out branch of `torch.where`. **All Lorentz primitives have
  finite gradients at every boundary input** (x=y, v=0, y=origin),
  which is essential for tangent-at-origin training loops where
  `T = exp_0(v)` then loss involves `distance(T_i, T_j)` including
  self-pairs.

**Recommended training pattern** — parameterize the substrate as a
Euclidean tangent at origin and convert to manifold points
on-the-fly per forward pass:

```python
mfd = LorentzManifold(n=K)
v = torch.randn(N, K, requires_grad=True)  # trainable Euclidean param
optimizer = torch.optim.Adam([v], lr=1e-2)

for step in range(n_steps):
    optimizer.zero_grad()
    T = mfd.exp_0(v)                          # (N, K+1) on hyperboloid
    loss = your_loss(T, ...)                  # e.g. NLL on distances
    loss.backward()                           # v.grad stays finite even
                                              #   at boundary inputs
    optimizer.step()
```

Storing `v` (not `T`) avoids manifold-drift accumulation — each
`exp_0` puts `T` exactly on the hyperboloid, and the standard
Euclidean optimizer step on `v` is well-defined.

Refs: Nickel-Kiela (2018), Chen et al. (2022) HyboNet, Lee (2018)
*Introduction to Riemannian Manifolds*, Cannon et al. (1997),
Pennec (2006).

### `KappaStereographicManifold(n, kappa=-1.0, device="cpu", dtype=torch.float64)`
κ-stereographic model with parametric curvature κ ∈ R interpolating
**spherical** (κ > 0, projection of `S^n`), **Euclidean** (κ = 0),
and **hyperbolic** (κ < 0, Poincaré ball). Points live directly in
`R^n` (no extra ambient dimension — `ambient_dim = n`, unlike
`LorentzManifold` which uses `n+1`). Branch dispatched at
`__init__` from the sign of `kappa`. **κ may be a Python float OR
a 0-dim `torch.Tensor` (e.g. `nn.Parameter`); the manifold is fully
differentiable through every κ-dependent op**, so joint training
of embeddings `v` AND curvature `κ` works via standard SGD:

```python
import torch
from holonomy_lib.manifolds import KappaStereographicManifold

kappa = torch.nn.Parameter(torch.tensor(-1.0))         # learnable κ
mfd = KappaStereographicManifold(n=4, kappa=kappa)
v   = torch.randn(N, 4, requires_grad=True) * 0.3      # tangent-at-origin params
optimizer = torch.optim.Adam([v, kappa], lr=1e-2)

for step in range(n_steps):
    optimizer.zero_grad()
    T    = mfd.exp_0(v)                                # embeddings on manifold
    loss = your_loss(T, mfd, ...)                      # any distance-based loss
    loss.backward()                                    # both v.grad and kappa.grad finite
    optimizer.step()
```

**κ can change sign during training.** Dynamic dispatch (for Tensor κ)
re-evaluates `sign(κ)` at every operation, so SGD is free to push κ
from hyperbolic (κ<0) to spherical (κ>0) and back. For static (float)
κ the branch is locked at construction as a fast path — same numerical
behavior, no per-call sign check.

| Method | Signature | Returns |
|---|---|---|
| `random_point` | `(batch_size=1, generator=None)` | `(B, n)` |
| `origin` | `(batch_size=1)` | `(B, n)` zero vector |
| `is_on_manifold` | `(x, atol=1e-9)` | `(B,)` bool |
| `mobius_add` | `(x, y)` | `(B, n)` `x ⊕_κ y` |
| `projection` | `(x, w)` | `(B, n)` (identity — open subset of R^n) |
| `inner` | `(x, u, v)` | `(B,)` with `λ_κ(x)² · ⟨u,v⟩` |
| `norm` | `(x, v)` | `(B,)` |
| `exp` | `(x, v)` | `(B, n)` |
| `log` | `(x, y)` | `(B, n)` |
| `distance` | `(x, y)` | `(B,)` geodesic |
| `parallel_transport` | `(x, y, v)` | `(B, n)` via Möbius gyrator |
| `retraction` | `(x, v)` | = `exp(x, v)` |
| `exp_0` | `(v)` | `(B, n)` origin shortcut |
| `log_0` | `(y)` | `(B, n)` origin shortcut |

All operations are autograd-finite at the boundary inputs (`x = y`,
`v = 0`, `y = origin`) via the same `_safe_sqrt` / `_safe_tanhc` /
`_safe_atanhc` idioms as `LorentzManifold`. Convention follows
Bachmann–Bécigneul–Ganea 2020: `λ_κ(o) = 2`, so `d_κ=0(x, y) =
2·‖y - x‖_Eucl` (twice the standard Euclidean distance). Refs:
Bachmann-Bécigneul-Ganea (2020) *Constant Curvature GCN*; Skopek
et al. (2019) *Mixed-curvature VAEs*; Ungar (2008) *Gyrovector
spaces*; Ganea et al. (2018) *Hyperbolic Neural Networks* (Poincaré
ball special case).

### `LorentzianManifold(n, device="cpu", dtype=torch.float64)`
Flat pseudo-Riemannian Minkowski spacetime `R^{1, n-1}`. Distinct
from `LorentzManifold` (which is the *unit hyperboloid* embedded in
this space). `LorentzianManifold` *is* the ambient: the manifold
itself is `R^n` with signature `(1, n-1)` — one timelike component
(index 0) and `n − 1` spacelike. Tangent vectors carry a **signed**
norm-squared `⟨v, v⟩_M ∈ R` (negative = timelike, zero = null,
positive = spacelike); geodesics are straight lines (flat metric),
so `exp_x(v) = x + v`, `log_x(y) = y − x`.

The interesting structure isn't the geodesics (trivial) but the
**causal classification** of point pairs:

| Method | Signature | Returns |
|---|---|---|
| `random_point` | `(batch_size=1, generator=None)` | `(B, n)` Gaussian |
| `origin` | `(batch_size=1)` | `(B, n)` zero |
| `is_on_manifold` | `(x, atol=1e-9)` | `(B,)` all True (flat space) |
| `minkowski_inner` | `(u, v)` | `(B,)` signed ⟨u,v⟩_M |
| `inner` | `(x, u, v)` | `(B,)` signed ⟨u,v⟩_M (= `minkowski_inner`; API parity for manifold-generic callers like `ProductManifold` / `manifold_aware_inner`) |
| `norm_sq` | `(v)` | `(B,)` signed ⟨v,v⟩_M (no `norm` — would be complex) |
| `interval_sq` | `(x, y)` | `(B,)` signed `⟨y-x, y-x⟩_M` |
| `causal_type` | `(x, y, null_atol=1e-9)` | `(B,)` int64 — SPACELIKE / FUTURE_TIMELIKE / PAST_TIMELIKE / FUTURE_NULL / PAST_NULL |
| `proper_time` | `(x, y)` | `(B,)` `√(-interval_sq)` for timelike, NaN otherwise |
| `proper_distance` | `(x, y)` | `(B,)` `√interval_sq` for spacelike, NaN otherwise |
| `projection` | `(x, w)` | identity (no constraint) |
| `exp` | `(x, v)` | `x + v` |
| `log` | `(x, y)` | `y - x` |
| `retraction` | `(x, v)` | = `exp(x, v)` |
| `metric_tensor` | `(x)` | `(B, n, n)` η = diag(−1, +1, …, +1) |
| `christoffel_symbols` | `(x)` | `(B, n, n, n)` — all zero (flat) |
| `riemann_tensor` | `(x)` | `(B, n, n, n, n)` — all zero (flat) |
| `ricci_tensor` | `(x)` | `(B, n, n)` — all zero (flat) |
| `scalar_curvature` | `(x)` | `(B,)` — zero (flat) |

The curvature-tensor methods are all identically zero on flat
Minkowski — the manifold is the canonical Ricci-flat vacuum solution
to Einstein's equations — and exist as the override surface for
subclasses representing curved Lorentzian backgrounds (Schwarzschild,
FLRW). `metric_tensor` returns the constant η regardless of `x`.

Use cases:
- **Spacetime substrate**: model embeddings as events in `R^{1, n-1}`;
  use `causal_type` for causal-link inference; use `proper_time` for
  the geodesic interval between causally-connected events.
- **Indefinite-metric ML**: any task where the metric should be
  sign-indefinite (vs. positive-definite Riemannian).

Refs: Misner-Thorne-Wheeler (1973) *Gravitation* §1-§5;
Hawking-Ellis (1973) *Large Scale Structure of Space-Time* §4
(causal structure); O'Neill (1983) *Semi-Riemannian Geometry With
Applications to Relativity* §3, §5.

### `ProductManifold(manifolds, weights=None, device="cpu", dtype=torch.float64)`
Riemannian product `M_1 × M_2 × … × M_k`. Combines existing
manifolds into a single product space; useful for **mixed-curvature
embeddings** (concept = (Euclidean coords, Hyperbolic coords, …)).

Points are stored as a flat concatenated tensor `(B, Σ ambient_dim_i)`
— compatible with the existing `holonomy_lib.hyperbolic.*`
primitives (frechet_mean, laplacian_eigenmaps, …) that operate on
flat tensors. Convention: `mfd.component(x, i)` slices into
submanifold `i`'s coordinates.

The combined metric is the (optionally weighted) direct sum:
`g((u_1, …, u_k), (v_1, …, v_k))_x = Σ_i w_i · g_i(u_i, v_i)_{x_i}`,
so geodesic distance is Pythagorean:
`d² = Σ_i w_i · d_i²(x_i, y_i)`.

Each manifold operation (exp, log, projection, retraction, …)
delegates per-submanifold. **Status: well-grounded prior art —
Gu-Sala et al. (2019) "Learning Mixed-Curvature Representations
in Product Spaces" (ICLR); Skopek et al. (2019) "Mixed-curvature
VAEs" (ICLR).**

| Method | Signature | Returns |
|---|---|---|
| `random_point` | `(batch_size=1, generator=None)` | `(B, Σ ambient_dim_i)` |
| `origin` | `(batch_size=1)` | `(B, Σ ambient_dim_i)` |
| `is_on_manifold` | `(x, atol=1e-9)` | `(B,)` bool |
| `component` | `(x, index)` | submanifold-`index` slice |
| `distance` | `(x, y)` | `(B,)` Pythagorean |
| `exp` / `log` / `inner` / `norm` / `projection` / `retraction` | per-component delegation | concatenated result |
| `exp_0` / `log_0` | tangent-at-origin shortcuts | `(B, Σ dim_i)` ↔ `(B, Σ ambient_dim_i)` |

### `HeterogeneousKappaManifold(n, combiner="arithmetic_mean", device="cpu", dtype=torch.float64)`
κ-stereographic geometry **where the curvature varies per point**.
The natural primitive when concepts in an embedding space have
different local curvatures (some hierarchical / hyperbolic, some
cyclical / spherical).

The class is a **pure math primitive** — it doesn't store κ. The
user owns the κ parameterization (e.g. an `nn.Parameter(torch.randn(N))`
for per-concept κ, or a smooth `κ_field(T)` callable + per-concept
residual δ) and passes effective-κ tensors into the manifold
methods. This keeps the manifold model-agnostic and lets the user
attach any architecture on top.

**Status mix** (intentionally explicit):
- **Standard / extension**: per-point continuous κ extends the
  established κ-stereographic model (Bachmann-Bécigneul-Ganea
  2020) from a global learnable κ to per-point κ.
- **Closest published prior art**: GraphMoRE (Guo et al. 2024,
  AAAI 2025, arXiv:2412.11085) — mixture-of-experts gating
  selects per-node curvature from a discrete set; Di Giovanni
  et al. (2022, arXiv:2202.01185) — heterogeneous manifolds via
  product of a homogeneous factor and a spherically-symmetric
  factor.
- **Our research contribution**: continuous per-point κ as a real
  number (vs. GraphMoRE's discrete gating); pair-κ combiner
  abstraction (the rule for combining `(κ_x, κ_y) → κ_eff` for
  pairwise distance isn't standardized in the literature; we ship
  arithmetic-mean default, harmonic-mean built-in, and a callable
  override).

| Method | Signature | Returns |
|---|---|---|
| `origin` | `(batch_size=1)` | `(B, n)` zero vector |
| `random_point` | `(batch_size=1, kappa=None, generator=None)` | `(B, n)` — draws `v ~ N(0, σ²I)` then `exp_0(v, κ)`; κ defaults to small per-point N(0, 1) |
| `is_on_manifold` | `(x, kappa, atol=1e-9)` | `(B,)` bool |
| `exp_0` | `(v, kappa)` | `(B, n)` per-point exp at origin |
| `log_0` | `(y, kappa)` | `(B, n)` per-point log at origin |
| `distance` | `(x, k_x, y, k_y)` | `(B,)` via combiner |

Built-in combiners: `"arithmetic_mean"` (default), `"harmonic_mean"`.
Pass any commutative `Callable[[κ_x, κ_y], κ_eff]` for a custom
combiner. The homogeneous case (κ_x = κ_y) reduces exactly to
`KappaStereographicManifold(κ)` (verified by tests).

**Recommended substrate-team pattern** (from your design notes):

```python
mfd = HeterogeneousKappaManifold(n=K)
# Smooth field as a small NN / polynomial:
kappa_field: Callable[[torch.Tensor], torch.Tensor] = ...    # (B, n) -> (B,)
# Per-concept residual:
delta = torch.nn.Parameter(torch.zeros(N))

def effective_kappa(T):
    return kappa_field(T) + delta   # (N,)

# In forward pass:
T = mfd.exp_0(v, effective_kappa_at_each_point)
d_ij = mfd.distance(T[i], k_eff[i], T[j], k_eff[j])
```

Refs: Bachmann-Bécigneul-Ganea (2020); Guo et al. AAAI 2025
*GraphMoRE*; Di Giovanni et al. (2022); Yang et al. *kHGCN* (2022).

### `model_sphere_area(kappa, N, r)`, `model_ball_volume(kappa, N, r)`
Model-space (constant-curvature space form `M_κ^N`) geodesic sphere area
`S = ω_{N-1}·sn_κ(r)^{N-1}` and ball volume `V = ω_{N-1}·∫₀^r sn_κ(t)^{N-1} dt`,
for real (non-integer) effective dimension `N ≥ 1` and signed curvature `κ`
(`sn_κ` is sin / identity / sinh by the sign of `κ`). Area is closed-form; the
volume integral uses Gauss-Legendre quadrature (exact for `κ = 0`). Inputs
`(B,)` or scalar, output `(B,)`. The Bishop-Gromov comparison reference for
curvature- and dimension-dependent volume / flux laws.

Refs: Bishop-Crittenden (1964); Petersen (2016) §7.1; Dai-Wei (2019) §1.2.

---

## §Algebra: `holonomy_lib.algebra`

Linear-algebra primitives.

### `truncated_svd(M, r, mode="exact", oversample=5, n_iter=2, generator=None)`
Batched top-r SVD of `M: (..., m, n) → (U: (..., m, r), S: (..., r), Vt: (..., r, n))`.
- `mode="exact"`: full SVD then truncate (Eckart-Young optimal).
- `mode="randomized"`: Halko-Martinsson-Tropp projection; faster when r ≪ min(m, n).
  Accuracy controlled by `oversample` + `n_iter`.

Refs: Eckart-Young (1936), Halko-Martinsson-Tropp (2011).

### `lanczos_eigsh(A, k, n_iter=None, oversample=10, generator=None, which="LA", sigma=None, solver="auto")`
Extreme-k eigenpairs of symmetric `A`. `which="LA"` (default): top-k
largest-algebraic of batched dense `(B, n, n)` or sparse `(n, n)` via
Lanczos with full reorthogonalization (Paige 1972); cost
`O(B · n_iter · n²)`, vs `O(B · n³)` for dense `torch.linalg.eigh`.
`which="SA"`: smallest-k via shift-and-invert; dense runs the in-house
shift-invert Lanczos, sparse dispatches to `eigsh_sparse_sa` (below)
selected by `solver`.
Refs: Lanczos (1950), Paige (1972), Saad (2011) §6.5, Ericsson-Ruhe (1980).

### `eigsh_sparse_sa(A, k, sigma=None, solver="auto", generator=None)`
Bottom-k eigenpairs `(eigvals (k,), eigvecs (n, k))` of sparse symmetric
`A: (n, n)` (COO/CSR/CSC, 2-D single-instance). Hybrid backends:
SciPy `eigsh` shift-invert (CPU, `n ≤ 8000`), SciPy LOBPCG + `spilu`
preconditioner (CPU, larger), `torch.lobpcg` (CUDA, device-preserving,
unpreconditioned). The two SciPy branches compute the same modes
(verified ~1e-14); the size threshold is throughput-only.
Refs: Ericsson-Ruhe (1980), Knyazev (2001), Saad (2003) §10.3-10.4.

### `cg(A, b, M=None, x0=None, rtol=None, atol=0.0, maxiter=None, sync_every=8)`
Conjugate gradient for symmetric positive-definite `A x = b`. `A` is a
dense `(n, n)` / `(B, n, n)` tensor, a sparse `(n, n)` tensor (applied
to the whole batch in one sparse matmul), or a callable matvec (for
never-materialized operators, e.g. normal equations or `δᵀ(δx)`).
`b: (B, n)` or `(n,)`. Tolerances default to `sqrt(eps(dtype))`
(derived); `maxiter` defaults to `n` (finite-termination bound).
Warns on non-convergence. Refs: Hestenes-Stiefel (1952), Saad (2003) §6.7.

### `lsqr(A, b, damp=0.0, atol=None, btol=None, maxiter=None, sync_every=8)`
Iterative least squares `min ‖Ax − b‖² + damp²·‖x‖²` via Golub-Kahan
bidiagonalization — never forms `AᵀA`, so conditioning is `cond(A)`,
not squared; converges to the minimum-norm solution on rank-deficient
systems. `A` dense `(m, n)` / `(B, m, n)` or sparse 2-D; per-element
convergence is frozen on device (safe past Krylov breakdown).
Refs: Paige-Saunders (1982) Alg. LSQR + §6.

---

## §Convex: `holonomy_lib.convex`

Problem-level convex optimization interfaces ("minimize this
objective"), as opposed to the raw Krylov solvers in `algebra.solvers`.
Primitives enter only with a named downstream consumer.

### `regularized_lstsq(A, b, lam, method="auto", rtol=None, maxiter=None)`
The ridge problem `min ‖Ax − b‖² + λ‖x‖²`. Methods: `"cholesky"`
(normal equations, CONVENTIONS §3.3 pattern; exact, dense-only),
`"cg"` (CG on the never-materialized normal operator `v ↦ Aᵀ(Av)+λv`),
`"lsqr"` (`damp = √λ`; conditioning not squared, handles sparse and
rank-deficient `A`). `"auto"`: dense → cholesky, sparse → lsqr
(benchmark-derived, `notes/benchmark_2026-07-08_solvers.md`).
Refs: Tikhonov (1963), Björck (1996) §2.7, Paige-Saunders (1982) §1,
Boyd-Vandenberghe (2004) §6.3.2.

---

## §Tensor calculus: `holonomy_lib.tensor_calculus`

Multilinear algebra on tensors with leading batch dim.

### `mode_product(T, A, axis)`
n-mode product T ×_axis A. Contracts axis `axis` of `T: (B, n_1, ..., n_d)` with
the last axis of `A: (B, j, n_axis)`. Result has `j` at position `axis`.
Ref: Kolda-Bader (2009), §2.5.

### `mode_unfolding(T, axis)`
Matricize T along an axis: bring `axis` to position 1, flatten the rest.
Output `(B, n_axis, prod_of_other_modes)`. Ref: Kolda-Bader (2009), §2.4.

### `hosvd(T, ranks, mode="exact", generator=None) → (core, factors)`
Truncated Higher-Order SVD. For `T: (B, n_1, ..., n_d)`, returns
`core: (B, r_1, ..., r_d)` and a list `factors[k]: (B, n_k, r_k)` with
orthonormal columns. `T ≈ core ×_1 factors[0] ×_2 factors[1] × ... ×_d factors[d−1]`.
Refs: De Lathauwer-De Moor-Vandewalle (2000), Vannieuwenhoven et al. (2012).

---

## §Spectral: `holonomy_lib.spectral`

Graph Laplacians + spectral embedding. All take symmetric adjacency `A: (B, n, n)`.
Isolated nodes handled via Moore-Penrose convention (Cheng-Wu 2024).

### `laplacian.combinatorial(A)`
L = D − A. PSD. Eigenvalue 0 multiplicity = # connected components.
Ref: Chung (1997).

### `laplacian.symmetric_normalized(A)`
L_sym = I − D^{−1/2} A D^{−1/2}. Spectrum ⊂ [0, 2]. Ref: Chung (1997), von Luxburg (2007).

### `laplacian.random_walk(A)`
L_rw = I − D^{−1} A. Same eigenvalues as L_sym (similar via D^{1/2}). Ref: von Luxburg (2007).

### `laplacian.signed(A)`
L^σ = D^{|σ|} − A,  D^{|σ|}_{ii} = Σ_j |A_{ij}|. PSD even with negative weights.
Eigenvalue 0 iff signed graph is balanced. Ref: Kunegis et al. (2010), Thm 3.4.

### `laplacian.degree(A, signed=False)`
Weighted degree `(B, n)`. With `signed=True`, uses |A| (Kunegis convention).

### `laplacian_eigenmaps(A, k, laplacian_type="symmetric_normalized") → (eigvals, eigvecs)`
Bottom-k spectral embedding. `laplacian_type ∈ {"combinatorial", "symmetric_normalized",
"random_walk", "signed"}`. Does **not** auto-drop the trivial null eigenvector
(caller decides). Refs: Belkin-Niyogi (2003), von Luxburg (2007).

### `magnetic.combinatorial(A, q=0.25)`
Hermitian magnetic Laplacian for directed graphs:
`L^(q) = D_s − H ⊙ A_s` where `H_{ij} = exp(i·2π·q·(A_{ij} − A_{ji}))`.
Returns a complex Hermitian `(B, n, n)`; real spectrum via `linalg.eigh`.
At `q = 0` collapses to the real Laplacian of `A_s = (A + A^T)/2`.
Refs: Lieb-Loss (1993), Fanuel et al. (2017), Furutani et al. (2020).

### `magnetic.symmetric_normalized(A, q=0.25)`
Symmetric-normalized magnetic Laplacian: `L_sym^(q) = I − D_s^{−1/2}(H⊙A_s)D_s^{−1/2}`.
Spectrum ⊂ [0, 2] regardless of `q`. Use the bottom-k eigenvectors as
directed-graph eigenmaps. Refs: Furutani et al. (2020), Prop. 1.

### `heat_kernel_chebyshev(L, t, signal=None, K=30, lambda_max=2.0)`
Heat kernel `exp(−t·L)` (or `exp(−t·L) @ signal`) via Chebyshev-polynomial
expansion: `O(K · n³)` dense, or `O(K · n² · k)` for an `(n, k)` signal,
beating the `O(n³)` eigendecomposition for medium `t`. Coefficients are
modified Bessel functions `I_k(t·λ_max/2)`, computed via `scipy.special.ive`.
Refs: Hammond-Vandergheynst-Gribonval (2011), §3.

### `effective_resistance(A) → (B, n, n)`
Pairwise effective resistance `R(u, v) = (e_u − e_v)ᵀ L⁺ (e_u − e_v)`
on a weighted graph (Klein-Randić 1993). On `K_n` every edge has
`R = 2/n`; on a path `P_n` the endpoints have `R = n − 1` (series
resistance). Refs: Doyle-Snell (1984), Klein-Randić (1993).

### `commute_time(A) → (B, n, n)`
Pairwise commute time `C(u, v) = vol(A) · R(u, v)` — expected
round-trip steps of the random walk on `A`. Chandra-Raghavan-Ruzzo-
Smolensky-Tiwari (1996) identity. Ref: Lovász (1993), §5.

### `diffusion_map(A, k, t=1.0) → (transition_eigvals, embedding)`
Coifman-Lafon (2006) diffusion-map embedding at time `t`. Returns
`(B, k)` transition-matrix eigenvalues `μ_j = 1 − λ_j` and
`(B, n, k)` coordinates `Ψ_t(x_i) = (μ_j^t · φ_j(x_i))`. Drops the
trivial null eigenvector. Pairwise Euclidean distance in the
embedding is the diffusion distance. Ref: Coifman-Lafon (2006), §3.

### `spectral_dimension(eigenvalues, t) → (B,)`
Spectral dimension `d_s = -2 · slope of log p(t) vs log t`, where
`p(t) = mean_i exp(-t·λ_i)` is the heat-kernel return probability of a Laplacian
spectrum `(B, n)`. Fits the log-log slope over caller-supplied times `t` `(T,)`;
non-integer (fractal / continuum) `d_s` supported. Pick `t` inside the power-law
window (finite graphs flatten `p` as `t → ∞`). Refs: Rammal-Toulouse (1983);
Durhuus-Jonsson-Wheater (2007); Hambly-Kigami-Kumagai (2002, local).

---

## §Discrete geometry: `holonomy_lib.discrete_geometry`

Combinatorial / Ricci-style curvature on graphs. The Perelman-on-networks
thread (Ollivier curvature → flow → surgery for community detection).

### `ollivier_ricci_curvature(A, alpha=0.0, reg=0.01, n_iter=100) → (B, n, n)`
Ollivier-Ricci curvature κ on all pairs via Sinkhorn Wasserstein-1 with the
shortest-path metric. `alpha` is laziness (0 = standard Ollivier; α → 1 → LLY).
For unweighted K_n: κ(edge) = (n − 2)/(n − 1).
Refs: Ollivier (2009), Liu-Lin-Yau (2011), Cuturi (2013).

### `discrete_ricci_flow(A, n_steps, dt=1.0, alpha=0.0, normalize=True, ...) → (B, n, n)`
Iterate `w_ij(t+1) = (1 − dt · κ_ij(t)) · w_ij(t)`. Negative-curvature edges
elongate (forming necks); positive-curvature edges shrink. Optional Frobenius-norm
normalization. Refs: Sia-Jonckheere-Bogdan (2019), Ni-Lin-Luo-Gao (2019).

### `ricci_flow_with_surgery(A, n_steps, surgery_period=10, surgery_threshold=3.0, ...) → (B, n, n)`
Discrete Ricci flow with periodic edge removal: the Perelman-spirit primitive.
Every `surgery_period` steps, edges whose weight ≥ `surgery_threshold` × initial
mean weight are removed. After enough iterations, the graph splits into communities.
Inspiration: Perelman (2002, 2003 surgery, 2003 extinction). Discrete: Sia (2019),
Ni-Lin-Luo-Gao (2019), Liu-Wang-Yau-Zeng (2017).

### `forman_ricci_simple(A) → (B, n, n)`
Combinatorial Forman-Ricci curvature on every edge: no optimal-transport
solve needed, `O(B · n²)` cost. For unweighted simple graphs the formula
collapses to `κ_F(u, v) = 4 − deg(u) − deg(v)`. Cheap qualitative substitute
for Ollivier-Ricci on large graphs (Sreejith et al. 2016, eq. 1).

### `forman_ricci_augmented(A) → (B, n, n)`
Augmented form adding the 2-face (triangle) contribution:
`κ_F^aug(u, v) = κ_F_simple(u, v) + 3 · #triangles(u, v)`. Tracks
Ollivier-Ricci more closely on dense substructures while remaining
fully combinatorial. Ref: Samal et al. (2018), §"Augmented Forman".

---

## §Provenance: `holonomy_lib.provenance`

Content-addressable hex provenance for **mechanistic interpretability**. Every
decorated primitive emits a Merkle-DAG node when called inside `record()`.
Same op + same inputs ⇒ same hex (deterministic). The layer TransformerLens /
nnsight / SAELens ride on top of for geometric/spectral mech interp.

### Context managers

```python
with provenance.record(
    cache_tensors=False,    # cache output tensors for replay / inspection
    hash_algorithm=...,      # "blake3" / "sha256"
    hash_mode="full",        # "full" (crypto-grade) or "sketch" (~15× faster on big tensors)
    cache_to_disk=None,      # Path: mirror cache to .pt files; survives memory eviction
    max_cache_size=None,     # bound the in-memory output cache; disk copy retained
) as reg:
    out = pipeline(...)
    # reg is a ProvenanceRegistry
```

### `ProvenanceRegistry`: what you get inside `record()`

| Method | What it does |
|---|---|
| `reg[hex]` | Look up a ProvenanceNode by hex |
| `reg.where(op_id=..., op_version=...)` | Filter nodes by op |
| `reg.ancestors(hex)` | Walk DAG upstream |
| `reg.ancestors_with_tensors(hex)` | Ancestor subgraph paired with cached tensors |
| `reg.parents(hex)` | Direct parents |
| `reg.get_tensor(hex)` | Cached output (memory, then user-input cache, then disk) |
| `reg.on_op(op_id, callback)` | TransformerLens-style observation hook |
| `reg.substitute({hex: value})` | Context mgr: activation patching at call time |
| `reg.replay({hex: value})` | Re-execute downstream DAG with substitution |
| `reg.clear(delete_disk=False)` | Drop in-memory caches; optionally delete disk files |
| `reg.to_networkx()` | Export DAG to networkx DiGraph |
| `reg.to_dataframe()` | Export node table to pandas |
| `reg.to_dict()` | JSON-friendly export |
| `reg.to_mermaid()` | Mermaid flowchart string (inline JupyterLab / GitHub MD) |
| `reg.to_graphviz()` | Graphviz DOT source |
| `reg.to_llm_context(max_ops=20, ...)` | Compact text summary for agent prompts |
| `reg.to_sae_dataset(op_id=None)` | Yield `(tensor, metadata)` for SAE training |
| `reg.diff(other)` | Structural diff (dict form) of two recordings |
| `reg.diff_summary(other)` | Human-readable diff: Cache hits / Drift / Only in self / Only in other |
| `reg.save(path)` / `Registry.load(path, strict=False)` | JSON persistence (disk-cached tensors re-attach on load; op_version drift triggers `ProvenanceVersionWarning`) |

### Decorating new primitives

```python
from holonomy_lib.provenance import with_provenance, register_provenance_class

@with_provenance("holonomy_lib.module.op_name", op_version="0.1")
def op_name(x: torch.Tensor, k: int = 3) -> torch.Tensor:
    ...

# To opt a class into replay() of its methods, register it with both
# a `_provenance_signature(self) -> dict` AND a classmethod
# `_from_signature(cls, sig) -> instance` that inverts it:
@register_provenance_class("MyManifold")
class MyManifold:
    def _provenance_signature(self):
        return {"class": "MyManifold", "n": self.n, ...}

    @classmethod
    def _from_signature(cls, sig):
        return cls(n=sig["n"], ...)
```

Hex computation: `sha256(op_id || op_version || canonical(params) || ":".join(input_hexes))`
truncated to 16 chars. Pluggable hash function (blake3 if installed, else sha256).
Pluggable hash mode: full byte hash (default) or O(1)-bytes sketch hash.

### Agent access: MCP server and Jupyter magic

Optional extras for inspecting registries from outside Python:

```bash
pip install 'holonomy-lib[mcp]'      # MCP server for Claude / GPT / any MCP client
pip install 'holonomy-lib[jupyter]'  # %record_provenance cell magic
```

The agent inventory lives in `holonomy_lib.provenance.agent` and is
exposed to LLM agents three ways: native tool-use schemas via
`to_anthropic_schema()` / `to_openai_schema()`, MCP tools via
`python -m holonomy_lib.provenance.mcp`, and direct Python calls.
Tools: navigation (`list_ops`, `where`, `node_info`, `ancestors`),
tensor inspection (`get_tensor_summary`, `tensor_slice`,
`tensor_per_batch_summary`, `tensor_eigenvalues`,
`tensor_singular_values`, `tensor_norm`, `tensor_compare`),
counterfactual execution (`replay_with` with `zeros_like` /
`from_hex` / `perturb` / `scale` / `swap_batch` / `literal`
substitution kinds), and op docs (`op_docstring`). The MCP entry
point reads the registry file from `HOLONOMY_PROVENANCE_REGISTRY`.
On the MCP transport, list-returning tools wrap their output as
`{"results": [...]}` for a single uniform JSON content item.

The Jupyter magic wraps a cell in `record()` and renders the DAG
inline:

```python
%load_ext holonomy_lib.provenance.jupyter

%%record_provenance
L = laplacian.combinatorial(A)
U, S, Vt = truncated_svd(L, r=3)
# _prov now holds the registry; Mermaid DAG renders below the cell.
```

---

## §Information geometry: `holonomy_lib.info_geometry`

Divergences on probability distributions, treated as points on a
Riemannian manifold (Amari 2016).

### `bregman_divergence(p, q, potential)`
General Bregman divergence `D_F(p ‖ q) = F(p) − F(q) − ⟨∇F(q), p − q⟩`
for any caller-supplied convex potential `F` (which must return the
pair `(F(x), ∇F(x))` at any input). Recovers squared-Euclidean,
generalized KL, and Itakura-Saito as special cases.
Refs: Bregman (1967), Banerjee et al. (2005).

### `kl_divergence_categorical(p, q) → (B,)`
Discrete KL `KL(p ‖ q) = Σ_i p_i (log p_i − log q_i)` with the
0·log(0/x) = 0 convention. Both `p` and `q` must be on the simplex.
Refs: Cover-Thomas (2006), Amari (2016) §2.4.

### `kl_divergence_gaussian(mu_p, Sigma_p, mu_q, Sigma_q) → (B,)`
Closed-form KL between two multivariate Gaussians. Cholesky-stable:
factors `Σ_q` once and reuses the factorization for the trace and
Mahalanobis terms; pulls `log det Σ` directly from the Cholesky
diagonal. Ref: Petersen-Pedersen Matrix Cookbook eq. 380.

---

## §Topology: `holonomy_lib.topology`

Hodge Laplacians and persistent homology on simplicial complexes.
Built on `holonomy_lib.simplicial`.

### `hodge_laplacian(complex, k) → Tensor`
The k-th Hodge Laplacian `L_k = ∂_{k+1} ∂_{k+1}^T + ∂_k^T ∂_k`. Its
kernel has dimension equal to the k-th Betti number (Hodge
decomposition). Dense complex → `(B, n_k_max, n_k_max)`; sparse →
`(n_k, n_k)` dense Tensor (the product of two sparse boundary
matrices is generally dense).
Refs: Eckmann (1944), Lim (2020), Schaub et al. (2020).

### `betti_numbers(complex, max_dim, threshold=1e-9) → Tensor`
`(β_0, …, β_max_dim)` via near-zero eigenvalue counting on each
`L_k`. Closed-form verification: S¹ gives `(1, 1)`, S² gives
`(1, 0, 1)`, T² gives `(1, 2, 1)`.

### `persistence_diagrams(points_or_distances, max_dim=2, max_radius=inf) → (diagrams, masks)`
Persistent homology of the Vietoris-Rips filtration over `(B, n, d)`
point clouds (or `(B, n, n)` distance matrices). Returns per-dim
batched padded tensors `diagrams[k]: (B, max_pairs_k, 2)` of
`(birth, death)` pairs + matching validity masks. H₀ via batched
union-find on sorted filtration edges; H_{1..max_dim} via Z/2
left-to-right boundary-matrix reduction (Edelsbrunner-Letscher-
Zomorodian 2002) with the Bauer-Kerber-Reininghaus clearing
optimization. Refs: Edelsbrunner-Letscher-Zomorodian (2002);
Cohen-Steiner-Edelsbrunner-Harer (2007) stability; Bauer (2021).

---

## §Simplicial: `holonomy_lib.simplicial`

Simplicial complex data structures + boundary operators. Foundation
for `holonomy_lib.topology` (Hodge Laplacians + persistent homology).
Two representations:

### `DenseSimplicialComplex(simplices_by_dim, valid_mask, n_vertices, ...)`
Batched, padded simplex tables. `simplices_by_dim[k]` is
`(B, n_k_max, k+1)` int with a `(B, n_k_max)` validity mask.
`boundary(k) → (B, n_{k-1}_max, n_k_max)` dense Tensor with the
Koszul signs.

### `SparseSimplicialComplex(simplices_by_dim, n_vertices, ...)`
Single-instance, no batch dim. `boundary(k) → sparse-CSC`. Used by
persistent homology where the matrix-reduction kernel walks the
sparse boundary column by column.

### `vietoris_rips_sparse(distances, max_radius, max_dim) → SparseSimplicialComplex`
VR complex from a single `(n, n)` distance matrix. Incremental
k-simplex construction via shared-(k-1)-face extension.

### `vietoris_rips_dense(distances, max_radius, max_dim, dtype=…) → DenseSimplicialComplex`
Batched VR construction from `(B, n, n)` distance matrices. Pads
per-dim to the max simplex count across the batch.

### `pairwise_distances(points) → Tensor`
Euclidean distance matrix from `(n, d)` or `(B, n, d)` points.
Refs: Munkres (1984), §1; Hausmann (1995); Bauer (2021).

---

## §Optimization: `holonomy_lib.optimization`

Riemannian optimizers wrapping the existing manifold `projection` +
`retraction` API. The flow is: caller computes the ambient gradient,
the optimizer projects it to the tangent space, scales by `-lr`, and
retracts back onto the manifold.

### `RiemannianSGD(manifold, lr=1e-2)`
Stateful Riemannian steepest-descent wrapper. `opt.step(point,
ambient_grad)` returns the new point. Works with `FixedRankManifold`
(point = `(U, S, Vt)` triple; ambient grad is `(B, m, n)`) and
`SPDManifold` (point and ambient grad both `(B, n, n)`).
Refs: Absil-Mahony-Sepulchre (2008), §4.1; Bonnabel (2013).

### `riemannian_sgd_step(manifold, point, ambient_grad, lr)`
Functional one-step API for use in custom training loops; the
`RiemannianSGD` class is a thin wrapper around it.

No `RiemannianAdam` in v1: adaptive step-size schemes (Adam, RMSProp,
AdamW, ...) are user-side ergonomics, not part of the math of
optimization on a manifold. The Riemannian gradient step *is* the
SGD primitive; adaptive preconditioning happens in user code by
rescaling `ambient_grad` before calling `step()`.

---

## §Sheaf: `holonomy_lib.sheaf`

Cellular sheaves on graphs and their Laplacians. A cellular sheaf
attaches a finite-dim vector space (a "stalk") to each simplex of a
graph and a linear restriction map for each face relation; the sheaf
Laplacian generalizes the graph Laplacian to track disagreement of
node-stalk values pushed up to each incident edge stalk. The dense
path carries a `SHEAF_DENSE_BYTES_CAP = 2 GiB` pre-flight guard;
past it, use the sparse path (`sheaf_coboundary_sparse`,
`sheaf_laplacian_sparse`, `sheaf_bottom_spectrum`). Restricted to
node-edge sheaves (no 2-cells yet).

### `GraphSheaf(edges, stalk_dim, restrictions_u, restrictions_v, n_nodes, ...)`
Dataclass holding edge list `(n_e, 2)`, per-edge restriction maps
`(n_e, d_e, d_v)` for each endpoint, and stalk dimensions. Rejects
self-loops + duplicate edges at construction (call sites must
pre-process). Trivial-sheaf factory `GraphSheaf.trivial(edges,
n_nodes)` builds the sheaf whose Laplacian equals the standard
combinatorial graph Laplacian.

### `HeterogeneousGraphSheaf(n_nodes, edges, node_stalk_dims, edge_stalk_dims, F_left, F_right)`
Cellular sheaf with PER-NODE stalk dimensions: node `i` carries its own
`d_v[i]`-dim stalk; edges carry ragged `(d_e[e], d_v[u])` / `(d_e[e], d_v[v])`
restriction maps stored as tuples. `sheaf_coboundary` / `sheaf_laplacian` /
`sheaf_dirichlet_energy` accept it (the uniform `GraphSheaf` is the special case
of one global `d_v`). Use when a node's stalk dim is its own quantity, e.g. a
per-node rank, not a single global K.

### `sheaf_coboundary(sheaf) → Tensor`
Coboundary operator `δ: C⁰ → C¹` as a `(n_e · d_e, n_v · d_v)` dense
matrix. Action on a node-cochain `x ∈ R^{n_v·d_v}` is the per-edge
disagreement `F_{u≤e}(x_u) − F_{v≤e}(x_v)`.

### `sheaf_laplacian(sheaf) → Tensor`
The sheaf Laplacian `L_F = δ^T δ` as a `(n_v·d_v, n_v·d_v)` dense
PSD matrix. Reduces to the standard graph Laplacian for the trivial
sheaf; orientation-flip on a 3-cycle drops kernel dim from 1 to 0
(the monodromy test).

### `sheaf_dirichlet_energy(sheaf, x) → (B,)`
Quadratic form `x^T L_F x`, batched-first over `x: (B, n_v·d_v)`.

### `sheaf_coboundary_sparse(sheaf) → sparse COO`
Same operator as `sheaf_coboundary` as a coalesced sparse-COO
`(Σd_e, Σd_v)`; nnz scales with edges, not with `Σd_e · Σd_v`. For
sheaves past the dense byte cap.

### `sheaf_laplacian_sparse(sheaf) → sparse COO`
`L_F = δ^T δ` assembled sparsely; block-sparsity of the graph
adjacency, symmetric PSD, matches the dense `sheaf_laplacian`.

### `sheaf_bottom_spectrum(sheaf, k, drop_kernel=True, kernel_headroom=16, solver="auto", generator=None)`
The `k` smallest NONZERO eigenpairs of the sparse `L_F` plus node
offsets `(eigvals (k,), eigvecs (Σd_v, k), offsets (n_nodes+1,))` —
the scalable spectral read (the kernel = global sections is dropped
at threshold `max(diag L)·1e-9`). Backends via `algebra.eigsh_sparse`.

Refs: Hansen-Ghrist (2019) *Toward a spectral theory of cellular
sheaves* (J. Appl. Comput. Topol. 3); Bodnar et al. (2022) *Neural
sheaf diffusion* (NeurIPS); Curry (2014) PhD thesis.

---

## §Lie: `holonomy_lib.lie`

Lie group primitives. v1 covers SO(3) (the rotation group of R³) +
real spherical harmonics; SE(3) / SU(2) / SL(n) are planned. Single
flat namespace `so3` for the SO(3) primitives; spherical harmonics
exposed at the top level.

### `so3.axis_angle_to_matrix(axis, angle) → (B, 3, 3)`
Rodrigues formula `R = I + sin(θ) K + (1 − cos θ) K²` with
`K = axis^∧`. Batched-first, `axis: (B, 3)`, `angle: (B,)`.

### `so3.matrix_to_axis_angle(R) → (axis (B, 3), angle (B,))`
Inverse log map. Dual-branch: trace-based formula away from π,
quaternion-based formula in the near-π regime (gap from π below
`SO3_LOG_NEAR_PI_RAD = 1e-7`, empirically calibrated for float64;
see `so3.py` docstring at the constant; don't "fix" to e3nn's `1e-2`
without re-running the empirical comparison).

### `so3.so3_exp(omega) → (B, 3, 3)`, `so3.so3_log(R) → (B, 3, 3)`
Matrix exp / log on so(3) (3×3 skew-symmetric matrices), built on
the axis-angle pair above.

### `so3.random_so3(batch_size, generator=None, device=…, dtype=…) → (B, 3, 3)`
Haar-uniform sampling on SO(3) via quaternion construction from 3
uniforms on [0, 1) (Shoemake 1992). Chi-squared sanity test in the
suite (p < 1e-6 bound).

### `so3.compose(R1, R2) → (B, 3, 3)`
Group product `R1 @ R2`. Trivial; included for API symmetry.

### `real_spherical_harmonics(directions, l_max) → list[(B, 2l+1)]`
Closed-form real Y_lm for `l_max ≤ 4`, evaluated at unit direction
vectors `directions: (B, 3)`. One tensor per `l` from `0` to `l_max`,
each of width `2l + 1`. Per-l block norm preserved under SO(3)
rotation (full mixing via Wigner-D matrices is a v0.3 follow-up). Audit
exempt: file is a transcription of Wikipedia's "Table of spherical
harmonics" + Monte-Carlo orthonormality test in the suite.

Refs: Hall (2015) §3.1; Shoemake (1992) *Uniform random rotations*;
Edmonds (1957) *Angular Momentum in Quantum Mechanics*; Cohen et al.
(2018) *Spherical CNNs*.

---

## §Hyperbolic: `holonomy_lib.hyperbolic`

Manifold-aware graph operations. Each primitive takes a manifold
object (e.g. `LorentzManifold`) as an explicit dependency, so the
algorithms generalize to other constant-curvature manifolds
(`KappaStereographicManifold`, etc.) without rewrites. This is the
layer where graph algorithms meet differential geometry: intrinsic
means, manifold-valued Laplacian eigenmaps, and the manifold heat
kernel.

### `manifold_aware_inner(x, y, manifold) → (B,)`
Riemannian inner product of `x` and `y` via the tangent at the
manifold origin: `⟨log_o(x), log_o(y)⟩_o`. For `LorentzManifold`
this reduces to the Euclidean dot product of the spatial parts of
`log_0(x)` and `log_0(y)`. Symmetric, non-negative on the diagonal
(equals `d(o, x)²` for x = y). Refs: Pennec (2006) §3.

### `frechet_mean(points, manifold, weights=None, max_iter=100, tol=1e-9) → (B, n+1)`
Weighted Karcher (1977) iteration for the Fréchet mean — the
geodesically-convex minimizer of `Σ_i w_i · d_M(μ, p_i)²` on a
Hadamard manifold. Iterates
`μ_{t+1} = exp_{μ_t}(Σ w_i log_{μ_t}(p_i) / Σ w_i)` until the update
tangent's Riemannian norm (at `μ_t`, before the step) falls below
`tol`. Guaranteed convergence on `LorentzManifold` (any `k < 0`) and
`KappaStereographicManifold(κ < 0)`; for κ > 0 the manifold is NOT
Hadamard and convergence requires inputs within the injectivity
radius `π/√κ` (caller's responsibility). Refs: Karcher (1977),
Pennec (2006) §4, Afsari (2011) on uniqueness.

### `hyperbolic_laplacian_eigenmaps(adjacency, manifold, max_steps=200, lr=0.05, init=None, generator=None) → (B, N, n+1)`
Embed graph nodes on `manifold` by minimizing `Σ_{ij} A_{ij} ·
d_M(Y_i, Y_j)²` via `RiemannianSGD`. Output shape `(B, N,
ambient_dim)`. Cost per step is `O(B · N² · ambient_dim)` from the
pairwise `log` evaluation; suited to graphs with `N ≤ a few hundred`.
Refs: Belkin-Niyogi (2003), Nickel-Kiela (2017) *Poincaré Embeddings*,
Liu et al. (2019) *Hyperbolic GNN*.

### `hyperbolic_heat_kernel(t, distances, manifold, n_quad=32, tail_budget=20.0) → tensor`
Heat kernel `k^n_t(d)` on the hyperbolic manifold. Dimension dispatch:
- `n=1`: Gaussian on R (degenerate boundary case).
- `n=2`: Gauss–Legendre quadrature on the Davies–Mandouvalos integral
  representation (32 nodes by default).
- `n=3`: Davies–Mandouvalos (1988) closed form
  `(4πt)^{-3/2} · exp(-t - d²/4t) · d/sinh d`.
- `n=5`: hand-derived closed form
  `(4πt)^{-5/2} · exp(-4t - d²/4t) · [d²·sinh d + 2t·(d·cosh d − sinh d)] / sinh³ d`
  — the operator chain `(1/sinh r · ∂_r)² exp(-4t - r²/4t)` expanded
  analytically. Faster and ~3 orders of magnitude more precise than
  the autograd-recursion alternative (validated by
  `notes/validation/heat_kernel_results.md`).
- Higher `n` (odd ≥ 7, even ≥ 4): spectral-shift-corrected
  Grigor'yan recursion
  `k^{n+2}(t, d) = -exp(-n·t) / (2π·sinh d) · ∂_d k^n(t, d)`
  via `torch.autograd.grad`, seeded at n=5 (odd) or n=2 (even).
  `create_graph=True` preserves the autograd chain for backward
  through `distances`. The `exp(-n·t)` factor is the
  spectral-bottom shift between dimensions
  (`((n+1)/2)² − ((n-1)/2)² = n`); omitting it was a bug in the
  original implementation, caught by independent PDE-residual
  validation.

Curvature scales out: `k^n_{−|k|, t}(d) = |k|^{n/2} · k^n_{−1, |k|·t}(√|k| · d)`.

Refs: Davies–Mandouvalos (1988); Grigor'yan (2009) *Heat Kernel and
Analysis on Manifolds* Theorem 8.21; Grigor'yan–Noguchi (1998).
Validation: `notes/validation/heat_kernel_findings.md`.

---

## Planned primitives

Open frontiers the library does not yet cover:
- Wigner-D matrices (real basis) for higher-l rotation actions on
  spherical-harmonic features. Today `real_spherical_harmonics` only
  preserves per-l block norms under rotation; the full mixing matrix
  is the natural next step.
- Optimal transport extensions: Gromov-Wasserstein (Mémoli 2011) for
  metric-measure-space comparison, Sinkhorn divergences (de-biased OT).
- GPU-resident custom CUDA kernel for the Z/2 PH boundary-matrix
  reduction (current torch path is a same-algorithm port that's
  ~21× slower than CPython sets at n=80; the win is a future kernel).
- Sparse-input shift-and-invert via iterative solver (CG/MINRES) for
  sparse SA Lanczos.
- Further manifolds: sphere, Stiefel, Grassmann.
(`ProductManifold` and `HeterogeneousKappaManifold` now ship;
`KappaStereographicManifold` supports κ-sign crossing during
training — the previous static-branch limitation is closed.)
- Higher-dimensional cellular sheaves on simplicial complexes (with
  2-cells / faces and the corresponding chain identity ∂_1 ∘ ∂_2 = 0).
- SE(3) / SU(2) / SL(n) Lie group primitives.
