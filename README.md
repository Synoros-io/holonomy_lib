# holonomy_lib

> **A research-grade PyTorch math library**: GPU-native, batched-first,
> audit-clean, with every primitive grounded in a citation.
> Differential geometry, spectral graph theory, discrete Ricci flow,
> tensor decompositions, Riemannian optimization, simplicial topology,
> batched persistent homology, and content-addressable provenance for
> mechanistic interpretability, all under one roof. Developed by
> independent and Synoros researchers for the *substrate* research.

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![tests: 1220 passing](https://img.shields.io/badge/tests-1220%20passing-brightgreen.svg)](#testing)
[![audit: clean](https://img.shields.io/badge/audit-clean-brightgreen.svg)](#audit-discipline)

---

## What this is

A consolidated PyTorch math library for research at the intersection of
**differential geometry**, **spectral graph theory**, **computational
topology**, and **mechanistic interpretability**: the mathematics that
modern ML keeps reinventing project by project. Fourteen modules, 1220
tests, every numerical constant derived or cataloged with a
scale-of-validity, every primitive cited to the paper that defines it.

The name **holonomy** comes from differential geometry: the
transformation a vector accumulates when parallel-transported around a
closed loop. It captures the library's character: every operation
defined by the geometry it preserves, every result traceable back to
its inputs through a content-addressable provenance DAG.

| Module | Primitives | What it gives you |
|---|---|---|
| `manifolds` | `FixedRankManifold`, `SPDManifold`, `LorentzManifold`, `KappaStereographicManifold`, `LorentzianManifold`, **`ProductManifold`**, **`HeterogeneousKappaManifold`**, **`comparison.*`** | Riemannian geometry on low-rank matrices, SPD cones, the hyperboloid model of hyperbolic space at curvature `k < 0`, the κ-stereographic model with parametric κ ∈ R interpolating spherical / Euclidean / hyperbolic, the pseudo-Riemannian (1, n-1)-signature Minkowski spacetime, mixed-curvature product manifolds, and per-point-κ heterogeneous geometry for substrate-style embeddings (Vandereycken 2013; Pennec et al. 2006; Nickel-Kiela 2018; Bachmann-Bécigneul-Ganea 2020; MTW 1973; O'Neill 1983; Gu-Sala et al. 2019; Skopek et al. 2019; Di Giovanni et al. 2022; Guo et al. AAAI 2025 GraphMoRE), plus Bishop-Gromov model-space sphere-area / ball-volume comparison references at real (non-integer) dimension and signed curvature |
| `algebra` | `truncated_svd` (exact + randomized), `lanczos_eigsh` (dense + sparse, LA/SA), **`eigsh_sparse_sa`**, **`cg`**, **`lsqr`** | Halko-Martinsson-Tropp randomized SVD; Eckart-Young exact; Lanczos top-k eigensolver with full reorthogonalization (Paige 1972); sparse bottom-k via hybrid shift-invert / LOBPCG+ILU backends (Ericsson-Ruhe 1980; Knyazev 2001); conjugate gradient (Hestenes-Stiefel 1952) and LSQR (Paige-Saunders 1982) iterative solvers |
| **`convex`** | **`regularized_lstsq`** | Problem-level convex optimization interfaces gated on named downstream consumers; ridge/Tikhonov least squares dispatching to Cholesky normal equations, CG on the normal operator, or LSQR with damping (Tikhonov 1963; Paige-Saunders 1982; Boyd-Vandenberghe 2004 §6.3.2) |
| `tensor_calculus` | `hosvd`, `mode_product`, `mode_unfolding` | Truncated HOSVD with Kolda-Bader n-mode product |
| `spectral` | `combinatorial`/`symmetric_normalized`/`random_walk`/`signed` Laplacians, `laplacian_eigenmaps`, `magnetic.*` (directed), `heat_kernel_chebyshev`, **`effective_resistance`**, **`commute_time`**, **`diffusion_map`**, **`spectral_dimension`** | Chung; von Luxburg; Kunegis (signed); Furutani 2020 (magnetic Hermitian); Hammond-Vandergheynst-Gribonval 2011 (Chebyshev heat kernel); Klein-Randić 1993 (resistance); Coifman-Lafon 2006 (diffusion maps); Rammal-Toulouse 1983 (spectral dimension) |
| `discrete_geometry` | `ollivier_ricci_curvature`, `discrete_ricci_flow`, `ricci_flow_with_surgery`, `forman_ricci_simple`, `forman_ricci_augmented` | Sinkhorn-W₁ Ollivier on graphs (Ollivier 2009; Cuturi 2013; Sia/Ni-Lin-Luo-Gao 2019), the **Perelman-on-networks** flow + surgery primitive, and the cheap combinatorial Forman alternative (Sreejith et al. 2016; Samal et al. 2018) |
| **`info_geometry`** | **`bregman_divergence`**, **`kl_divergence_categorical`**, **`kl_divergence_gaussian`** | Bregman divergence for any convex generator plus closed-form KL for the standard exponential families (Bregman 1967; Banerjee et al. 2005; Amari 2016) |
| **`optimization`** | **`RiemannianSGD`** | Steepest descent on `FixedRankManifold` / `SPDManifold` / `LorentzManifold` via the existing projection + retraction API (Absil-Mahony-Sepulchre 2008, §4.1) |
| **`simplicial`** | **`DenseSimplicialComplex`**, **`SparseSimplicialComplex`**, **`vietoris_rips_*`** | Simplicial complex data structures + boundary operators + Vietoris-Rips construction; foundation for Hodge + persistent homology (Munkres 1984; Hausmann 1995; Bauer 2021) |
| **`topology`** | **`hodge_laplacian`**, **`betti_numbers`**, **`persistence_diagrams`** | Hodge Laplacians + Betti numbers on simplicial complexes (Eckmann 1944; Lim 2020), plus batched persistent homology H₀+H₁+H₂ of Vietoris-Rips filtrations via union-find + Z/2 matrix reduction (Edelsbrunner-Letscher-Zomorodian 2002; Cohen-Steiner-Edelsbrunner-Harer 2007 stability) |
| **`sheaf`** | **`GraphSheaf`**, **`HeterogeneousGraphSheaf`**, **`sheaf_coboundary`**/**`sheaf_coboundary_sparse`**, **`sheaf_laplacian`**/**`sheaf_laplacian_sparse`**, **`sheaf_dirichlet_energy`**, **`sheaf_bottom_spectrum`** | Cellular sheaves on graphs and their Laplacians, uniform or per-node (heterogeneous) stalk dims (Hansen-Ghrist 2019); reduces to the standard graph Laplacian under trivial stalks; sparse-COO coboundary/Laplacian plus bottom-k-nonzero spectral read past the dense byte cap; the spectral foundation behind Neural Sheaf Diffusion (Bodnar et al. 2022) |
| **`lie`** | **`so3.{exp,log,axis_angle,random_so3,compose}`**, **`real_spherical_harmonics`** | SO(3) primitives: Rodrigues / matrix log with empirically-calibrated near-π branch, Haar-uniform sampling (Shoemake 1992), composition; real spherical harmonics Y_lm for l ≤ 4 (Edmonds 1957), the natural basis for SO(3)-equivariant functions on the sphere |
| `provenance` | `@with_provenance`, `record()`, `ProvenanceRegistry` | Content-addressable Merkle DAG of math primitives; substitution / replay / SAELens emission for mech interp |
| **`hyperbolic`** | **`frechet_mean`**, **`hyperbolic_laplacian_eigenmaps`**, **`manifold_aware_inner`**, **`hyperbolic_heat_kernel`** | Manifold-aware graph operations: Karcher (1977) intrinsic mean, RSGD-based hyperbolic Laplacian eigenmaps (Belkin-Niyogi + Nickel-Kiela 2017), tangent-at-origin inner product (Pennec 2006), and the dimension-dispatched H^n heat kernel (closed forms: Davies-Mandouvalos 1988 at n=3, hand-derived operator-chain at n=5 and n=7; Grigor'yan-Noguchi recursion above, seeded from n=7) |

---

## Why use this

Existing libraries cover slices of what's here, but none cover all four
properties this library guarantees:

1. **Breadth**: Riemannian manifolds, spectral graph theory, tensor
   decompositions, Ricci-curvature, and content-addressable provenance,
   under one import root.
2. **GPU-native, batched-first**: every operation takes a leading
   batch dim, runs on `cuda`/`rocm`/`mps`/`cpu`, verified for `B ∈ {0,
   1, > 1}`.
3. **Audit-clean**: every numerical constant is derived, a universal
   invariant, or experimentally tuned with documented scale-of-validity.
   Enforced by `python -m holonomy_lib.audit src/ --strict`.
4. **Cited**: every public function has a `References:` section
   pointing to the paper that defines its math. No "trust me"
   implementations.

| | this lib | geoopt | geomstats | pymanopt | gudhi | ripser |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| Riemannian manifolds + optimizers | ✓ | ✓ | ✓ | ✓ | – | – |
| Spectral graph theory (4+ Laplacians) | ✓ | – | – | – | – | – |
| Magnetic Laplacian (directed graphs) | ✓ | – | – | – | – | – |
| Ollivier-Ricci + Forman-Ricci curvature | ✓ | – | – | – | – | – |
| Discrete Ricci flow + surgery | ✓ | – | – | – | – | – |
| Tucker / HOSVD | ✓ | – | – | – | – | – |
| Chebyshev heat kernel + diffusion maps | ✓ | – | – | – | – | – |
| Simplicial complexes + Hodge Laplacians | ✓ | – | – | – | ✓ | – |
| Batched persistent homology (H₀+H₁+H₂) | ✓ | – | – | – | – | – |
| GPU-native (PyTorch) | ✓ | ✓ | partial | – | – | – |
| Batched-first | ✓ | ✓ | partial | – | – | – |
| Content-addressable provenance | ✓ | – | – | – | – | – |
| Audit / no-magic-numbers | ✓ | – | – | – | – | – |
| Information geometry (Bregman + KL) | ✓ | – | ✓ | – | – | – |
| Differentiable hyperbolic heat kernel (all `n`) | ✓ | – | – | – | – | – |
| Pseudo-Riemannian (Lorentzian) manifold | ✓ | – | – | – | – | – |
| Per-point κ (heterogeneous curvature) | ✓ | – | – | – | – | – |
| Mixed-curvature product manifold | ✓ | – | – | – | – | – |
| Learnable κ that can cross 0 mid-training | ✓ | – | – | – | – | – |

---

## Installation

The library has a small dependency surface: `torch >= 2.0`, `numpy`,
`scipy`. Everything else, Riemannian optimizers, simplicial complexes,
Hodge Laplacians, persistent homology, is shipped natively. You do
**not** need to install `pymanopt`, `geoopt`, `gudhi`, `ripser`, or
similar to use the corresponding primitives.

### Standard install (CPU or default CUDA)

```bash
pip install holonomy-lib
```

This pulls torch's default wheel (CPU or CUDA 12, depending on
platform) automatically. Python ≥ 3.12.

### ROCm / older CUDA / specific torch wheel

Install your preferred torch wheel **first**, then the library. pip /
uv will respect the already-installed torch:

```bash
# AMD ROCm 6.4:
pip install --index-url https://download.pytorch.org/whl/rocm6.4 torch
pip install holonomy-lib

# CUDA 12.1 specifically:
pip install --index-url https://download.pytorch.org/whl/cu121 torch
pip install holonomy-lib

# CPU only:
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install holonomy-lib
```

### From source (development)

```bash
git clone https://github.com/Synoros-io/holonomy_lib
cd holonomy_lib
uv venv
uv pip install -e ".[dev]"
```

### Optional extras

- `holonomy-lib[provenance-extras]`: `blake3` (faster hash), `networkx`
  (DAG export), `pandas` (DataFrame export). Used only inside specific
  provenance helpers; the library degrades gracefully without them.
- `holonomy-lib[comparison]`: pymanopt, geoopt, geomstats, tensorly,
  gudhi, ripser, GraphRicciCurvature, networkx, autograd. Required
  ONLY for running the cross-comparison test suite locally; the
  library itself never imports these.
- `holonomy-lib[dev]`: pytest, ruff, mypy.
- `holonomy-lib[all]`: provenance-extras + dev (the typical
  contributor install).

---

## Quick start

```python
import torch
from holonomy_lib.manifolds import SPDManifold
from holonomy_lib.optimization import RiemannianSGD
from holonomy_lib.spectral import laplacian, laplacian_eigenmaps
from holonomy_lib.discrete_geometry import ricci_flow_with_surgery
from holonomy_lib.topology import persistence_diagrams
from holonomy_lib import provenance

# 1. Riemannian geometry on SPD covariance matrices
mfd = SPDManifold(n=8, dtype=torch.float64)
S = mfd.random_point(batch_size=4)       # (4, 8, 8) SPD
T = mfd.random_point(batch_size=4)
d = mfd.distance(S, T)                    # affine-invariant geodesic
V = mfd.log(S, T)                         # Lie-algebra-style log
T_recon = mfd.exp(S, V)                   # exp_S(log_S(T)) ≈ T

# 2. Riemannian gradient descent ON the SPD manifold
opt = RiemannianSGD(mfd, lr=0.5)
point = S.clone()
for _ in range(50):
    ambient_grad = -mfd.log(point, T)     # gradient of (1/2) d(point, T)^2
    point = opt.step(point, ambient_grad)
# `point` now sits on the SPD manifold, close to T.

# 3. Graph spectral embedding
A = (torch.rand(1, 50, 50) > 0.7).double()
A = (A + A.mT) * 0.5                      # symmetrize
vals, vecs = laplacian_eigenmaps(A, k=4, laplacian_type="symmetric_normalized")

# 4. Perelman-on-networks: community detection via Ricci flow + surgery
A_after = ricci_flow_with_surgery(
    A, n_steps=20, surgery_period=5, surgery_threshold=3.0,
    dt=0.5, alpha=0.0,
)
# Disconnected components in A_after correspond to detected communities.

# 5. Batched persistent homology on point clouds
points = torch.randn(8, 30, 2, dtype=torch.float64)   # 8 point clouds of 30 pts
diagrams, masks = persistence_diagrams(
    points, max_dim=2, max_radius=2.5,
)
# diagrams[0]: (8, max_h0, 2)  birth/death pairs for H_0 (components)
# diagrams[1]: (8, max_h1, 2)  for H_1 (loops)
# diagrams[2]: (8, max_h2, 2)  for H_2 (voids)
# masks[k] tells you which pair-slots are valid per batch element.

# 6. Mech-interp-style provenance: every primitive emits a Merkle DAG node
with provenance.record(cache_tensors=True) as reg:
    L = laplacian.combinatorial(A)
    vals, vecs = laplacian_eigenmaps(A, k=4)

# Look up any operation by content-addressable hex
for node in reg:
    print(f"{node.hex}  {node.op_id}")
```

See `CONTENTS.md` for the complete inventory of primitives, signatures,
and citations.

---

## Performance

Benchmarks: `notes/benchmark_baseline.md` (before optimization),
`notes/benchmark_optimized.md` (post-Phase-3 fixes),
`notes/benchmark_2026-05-26_roadmap_sweep.md` (v0.1 roadmap items),
`notes/benchmark_2026-07-08_pr1_quadrature_coboundary.md` (quadrature
cache + vectorized heterogeneous coboundary), and
`notes/benchmark_2026-07-08_solvers.md` (`regularized_lstsq` method
dispatch). All times CPU, single-thread, PyTorch 2.12, float64.

### Highlight: discrete Ricci curvature

The signature primitive, Ollivier-Ricci curvature via batched
log-domain Sinkhorn over all-pairs shortest-path costs, got two
optimizations:

- **Pair tiling**: the Sinkhorn dual update used to materialize a `(B,
  n², n, n)` intermediate (128 MB per iter at n=64). Tiled
  implementation processes pairs in chunks of `SINKHORN_TILE_DEFAULT =
  256`, capping the inner broadcast at ~16 MB.
- **Sync cadence**: the `.item()` convergence check used to fire every
  iter, forcing a GPU→CPU sync. Now checks every 8 iters; same
  asymptotic work, 8× fewer host syncs.

| graph size (n) | before | after | **speedup** |
|---:|---:|---:|---:|
| 16  | 34.0 ms | 18.0 ms | 1.9× |
| 32  | 273 ms  | 133 ms  | 2.1× |
| **64** | **22.6 s** | **1.7 s** | **13×** |

### Highlight: Riemannian retraction on low-rank manifolds

`FixedRankManifold.retraction` used to do a full SVD then slice top-r.
At low `r/min(m, n)` ratios (the common case for the fixed-rank
manifold), it now auto-switches to Halko-Martinsson-Tropp randomized
SVD with documented oversampling.

| m × n × r | before (full SVD) | after (auto) | **speedup** |
|---|---:|---:|---:|
| 64 × 64 × 8 | 0.31 ms | 0.31 ms | 1.0× (parity) |
| 256 × 256 × 16 | 7.4 ms | 1.3 ms | 5.8× |
| **1024 × 1024 × 32** | **193 ms** | **7.6 ms** | **25×** |

### Highlight: Lanczos vs dense `eigh` on big symmetric matrices

The library's `algebra.lanczos_eigsh` with full reorthogonalization
beats `torch.linalg.eigvalsh` once the matrix is big enough that
computing the full spectrum becomes wasteful. Single-batch top-1
eigenvalue at CPU, float64:

| n | dense `eigvalsh` | `lanczos_eigsh` (n_iter=30) | **speedup** |
|---:|---:|---:|---:|
| 128 | 0.44 ms | 2.66 ms | 0.2× (Lanczos overhead dominates) |
| 512 | 7.62 ms | 4.84 ms | 1.6× |
| **1024** | **46.5 ms** | **11.0 ms** | **4.2×** |

The same `lanczos_eigsh` accepts sparse-CSC inputs (via the dispatch
added in Phase 3), so it's the natural top-k path on the sparse-Hodge
Laplacians produced by the `topology` module.

### Highlight: shift-and-invert Lanczos for smallest eigenvalues

`lanczos_eigsh(A, k, which="SA", sigma=σ)` runs Lanczos on
`(A − σI)^{-1}` so the dominant Ritz values converge to the
eigenvalues of `A` closest to σ (Ericsson-Ruhe 1980). LU-factor is
done once outside the iteration; each step is a `lu_solve`. Where the
factorization cost is amortized over enough iterations, it beats both
LA Lanczos (which has to do many iterations to converge on the small
end of the spectrum) and dense `eigvalsh` (which always pays `O(n³)`):

| n | dense `eigvalsh` | `lanczos_eigsh` LA, n_iter=60 | `lanczos_eigsh` SA, n_iter=40 |
|---:|---:|---:|---:|
| 64 | 0.23 ms | 2.83 ms | 2.56 ms |
| 256 | 3.34 ms | 4.89 ms | 4.77 ms |
| **1024** | **80.9 ms** | 24.5 ms | **18.9 ms** |

SA mode raises `RuntimeError("shift-invert breakdown")` if σ coincides
with an eigenvalue of `A`; for graph Laplacians (which have 0 in
spectrum) use a small negative shift.

### Highlight: sparse smallest-algebraic eigensolver

The table above is the dense in-house SA path. Sparse `A`
(COO/CSR/CSC) dispatches instead to `algebra.eigsh_sparse_sa`, wired
automatically into `lanczos_eigsh(which="SA")`: SciPy `eigsh`
shift-invert (CPU, `n ≤ SPARSE_SA_ITER_SOLVER_N`), SciPy `lobpcg` +
`spilu` preconditioning (CPU, larger `n`), or `torch.lobpcg` (CUDA,
unpreconditioned; validated at test scale only). Verified to ~1e-8
against the dense SA path and ~1e-10 against dense `eigvalsh`. On a
sparse sheaf Laplacian too large for the dense path (10,000 nodes,
Σd_e = 35,868, dense δ would be 5.4 GiB), the sparse assembly (nnz =
192,329) takes 0.12 s and the bottom-16 nonzero modes converge via
LOBPCG+spilu in ~126 s
(`notes/validation/sparse_sheaf_spectral_results.md`).

### Highlight: sparse Laplacian backend

All four Laplacian variants (combinatorial, symmetric-normalized,
random-walk, signed) accept sparse-COO/CSR/CSC adjacency and return
a sparse-COO Laplacian on the same device. Combined with the sparse
`lanczos_eigsh` path, you get end-to-end sparse spectral chains
without materializing the dense `(n, n)`. The crossover is at very
small `n` because sparse construction time stays nearly flat while
dense scales `O(n²)`:

| n | density | dense `L = D − A` | sparse `L` |
|---:|---:|---:|---:|
| 256 | 0.05 | 0.25 ms | **0.21 ms** |
| 1024 | 0.01 | 3.2 ms | **0.23 ms** (14×) |
| 4096 | 0.003 | n/a (16 GB) | **0.30 ms** |

### Highlight: iterative Krylov solvers

`algebra.cg` (Hestenes-Stiefel 1952) and `algebra.lsqr`
(Paige-Saunders 1982) back `convex.regularized_lstsq`'s `method="auto"`
dispatch. Dense ridge problems on well-conditioned Gaussian `A` (CPU,
float64, `λ = 0.1`):

| m×n | cholesky | cg (normal op) | lsqr |
|---|---:|---:|---:|
| 128×64 | 0.1 ms | 2.4 ms | 7.5 ms |
| 512×256 | 1.5 ms | 4.9 ms | 11.1 ms |
| **2048×1024** | 40.6 ms | 36.9 ms | 41.4 ms |

Cholesky dominates through `n=256` and reaches rough parity by
`n≈1024-2048` (run-to-run noise territory); the iterative timings are
best-case here since Gaussian `A` is well-conditioned, and the gap
would widen back in Cholesky's favor on ill-conditioned problems (in
wall time; LSQR wins there on accuracy). `method="auto"` picks
`"cholesky"` for dense `A` (no benchmarked crossover to threshold on)
and `"lsqr"` for sparse `A` (Cholesky would need the dense Gram
matrix); full read in `notes/benchmark_2026-07-08_solvers.md`.

### Highlight: Batched persistent homology

`topology.persistence_diagrams` computes H₀ + H₁ + H₂ for a batch of
point clouds in parallel. H₀ runs via union-find on sorted filtration
edges (no boundary-matrix reduction needed). H₁ and H₂ use Z/2
left-to-right reduction (Edelsbrunner-Letscher-Zomorodian 2002) on
sparse-CSC boundary matrices, batching across point clouds.

Closed-form verification: a noisy 30-point unit circle reliably
recovers one persistent H₁ bar (the loop) with persistence > 0.2 in
the default `max_radius` range; the bottleneck stability theorem
(Cohen-Steiner-Edelsbrunner-Harer 2007) is verified under
ε-perturbation in the test suite.

The `reduction_backend="torch"` path runs end-to-end on the
filtration's device (CPU or GPU), but is **not yet a custom CUDA
kernel**, just a torch-tensor port of the same sequential algorithm.
For small inputs CPython sets are faster (43 ms vs 903 ms on an
80-point circle); the torch path is a foundation for the v0.2 GPU
kernel rather than an immediate win. The default backend stays
`"python"`.

---

## Audit discipline

Every numerical constant must be in one of three categories:

| Category | Example |
|---|---|
| ✅ **Derived from inputs** | `1 / N` for normalization; `1 / sqrt(d)` for Laplacian normalization |
| ⚖️ **Universal invariant** | `1e-9` numerical floor; `0.5` halving; `2π`; `1024` (KB↔MB) |
| 🔬 **Experimentally tuned** | `SINKHORN_TILE_DEFAULT = 256`, cataloged with scale-of-validity |

Each constant in category 🔬 has a row in [`notes/magic_numbers.md`](notes/magic_numbers.md)
with the procedure used to pick it, the regime where it's valid, and what
to re-derive when scale changes. The audit tool
(`python -m holonomy_lib.audit src/ --strict`) fails on any uncataloged
literal; run it before every commit.

---

## Provenance for mechanistic interpretability

Every public primitive is decorated with `@with_provenance`. Inside a
`provenance.record()` block, calls emit Merkle-DAG nodes whose hex
identity is `hash(op_id || op_version || canonical(params) ||
input_hexes)`. Same op, same inputs ⇒ same hex (deterministic, content-
addressable).

This unlocks TransformerLens-style activation patching and SAELens-
style dataset emission for the mathematical primitives, not just neural
network internals:

```python
with provenance.record(cache_tensors=True) as reg:
    out = pipeline(A)

# Find a specific operation
lap_node = reg.where(op_id="holonomy_lib.spectral.laplacian.combinatorial")[0]

# Ablation: substitute zeros and replay only the downstream computation
new = reg.replay({lap_node.hex: torch.zeros_like(reg.get_tensor(lap_node.hex))})
# `new` contains the re-executed outputs of every node downstream of the substitution.

# Emit a SAELens-style dataset for training feature extractors
for tensor, metadata in reg.to_sae_dataset(op_id="holonomy_lib.algebra.linear.truncated_svd"):
    yield tensor, metadata
```

Pluggable hash function (blake3 if installed, else SHA-256). Persist
the DAG with `reg.save(path)` / `ProvenanceRegistry.load(path)`.

**Performance modes** (v0.3): `record(hash_mode="sketch")` swaps the
byte-level content hash for an O(1)-bytes sketch (shape + dtype + 64
strided samples + sum + std). About 15× faster than full mode on 8 MB
tensors; crossover at ~n=256. `record(cache_to_disk=path)` mirrors the
output cache to torch.save'd files, so memory eviction from
`max_cache_size` doesn't lose tensors: `get_tensor()` reloads on
demand. Both default off.

**Inspection** (v0.3): `reg.to_mermaid()`, `reg.to_graphviz()` for
visualization; `reg.diff_summary(other)` for "did my refactor preserve
semantics" comparisons; `reg.to_llm_context()` for a compact text
summary suitable for an agent prompt.

**Agent access** (v0.3, optional extras): `pip install 'holonomy-lib[mcp]'`
adds an MCP server (`python -m holonomy_lib.provenance.mcp`) that
exposes the registry as agent tools for Claude, GPT, or any MCP
client. `pip install 'holonomy-lib[jupyter]'` adds a
`%%record_provenance` cell magic that records and renders the DAG
inline.

---

## Testing

```bash
# Full test suite
uv run pytest

# Just one module
uv run pytest tests/manifolds

# Run the audit (build gate)
uv run python -m holonomy_lib.audit src/ --strict

# Run benchmarks (excluded from the test suite; runs on demand)
uv run python -m tests.benchmarks.run --out notes/benchmark_latest.md

# Run on a GPU machine; parity tests light up automatically
uv run pytest tests/test_device_parity.py
```

**Comparison tests** run against established libraries when installed:
`pymanopt` for FixedRankManifold, `geoopt` for SPDManifold, `tensorly`
for HOSVD, `scipy.sparse.csgraph` for Laplacians,
`GraphRicciCurvature` + `networkx` for Ollivier-Ricci. The tests skip
silently if a comparison library isn't installed.

---

## Project structure

```
holonomy_lib/
├── src/holonomy_lib/          # the library
│   ├── manifolds/             # FixedRank, SPD, Lorentz, κ-stereographic, Lorentzian, Product, heterogeneous-κ
│   ├── hyperbolic/            # manifold-aware graph ops: Fréchet mean, eigenmaps, heat kernel
│   ├── algebra/               # truncated_svd, lanczos_eigsh (dense + sparse, LA/SA), eigsh_sparse_sa, cg, lsqr
│   ├── convex/                # regularized_lstsq (Tikhonov/ridge least squares)
│   ├── tensor_calculus/       # hosvd, mode_product, mode_unfolding
│   ├── spectral/              # 4 Laplacians (incl. magnetic + sign-magnetic), eigenmaps, heat kernel, resistance, diffusion maps
│   ├── discrete_geometry/     # Ollivier-Ricci, discrete Ricci flow, surgery, Forman-Ricci
│   ├── info_geometry/         # Bregman + KL divergences, Fisher metric, natural gradient
│   ├── optimization/          # RiemannianSGD on FixedRank / SPD
│   ├── simplicial/            # Dense + Sparse complexes, Vietoris-Rips
│   ├── topology/              # Hodge Laplacians, Betti, persistence diagrams (H₀+H₁+H₂)
│   ├── sheaf/                 # cellular sheaves on graphs, sheaf Laplacians (dense + sparse), sparse bottom-k spectrum
│   ├── lie/                   # SO(3) primitives, real spherical harmonics (l ≤ 4)
│   ├── provenance/            # content-addressable hex protocol
│   └── audit.py               # audit gate: no magic numbers
├── tests/                     # 1220 tests across all modules
│   └── benchmarks/            # device-agnostic timing harness
├── notes/
│   ├── magic_numbers.md       # cataloged constants with scale-of-validity
│   ├── scrutiny.md            # findings + fixes from review passes
│   ├── benchmark_baseline.md  # before optimization
│   └── benchmark_optimized.md # after
├── CHANGELOG.md               # release history
└── CONTENTS.md                # primitive inventory and quick reference
```

---

## Roadmap

See [`CHANGELOG.md`](CHANGELOG.md) for the full release history.

**v0.7.0** (current): sparse spectral scale-out + iterative solvers.

- `sheaf.sheaf_coboundary_sparse` / `sheaf.sheaf_laplacian_sparse` /
  `sheaf.sheaf_bottom_spectrum`: sparse-COO sheaf coboundary and
  Laplacian, plus a bottom-k-nonzero spectral read, for sheaves past
  the dense `SHEAF_DENSE_BYTES_CAP = 2 GiB` guard.
- `algebra.lanczos_eigsh(which="SA")` now accepts sparse input,
  dispatching to the new `algebra.eigsh_sparse_sa` hybrid backend
  (SciPy `eigsh` shift-invert / SciPy `lobpcg` + `spilu` / `torch.lobpcg`).
- `algebra.cg` / `algebra.lsqr`: iterative Krylov solvers
  (Hestenes-Stiefel 1952; Paige-Saunders 1982), dense, sparse, or
  callable-operator form.
- New `convex` package, seeded with `convex.regularized_lstsq`
  (Tikhonov/ridge least squares, `method="auto"` dispatch).
- `manifolds.sn_kappa` exposed as a public per-direction
  generalized-sine primitive (was internal `_sn_kappa`).
- Perf: cached Gauss-Legendre quadrature nodes, vectorized
  heterogeneous sheaf coboundary assembly.
- Test count 1187 (v0.6.0) → 1220; audit clean.

**v0.6.0**: public preprint release milestone ("Meaning on a Video
Card: Scalable, Lossless Geometric Primitives", Vaught 2026); no
committed API changes since v0.5.2.

**v0.5.2**: measurement + structure primitives.

- `manifolds.comparison`: Bishop-Gromov model-space sphere area / ball
  volume at real (non-integer) dimension and signed curvature.
- `spectral.spectral_dimension`: spectral dimension `d_s` from a
  Laplacian spectrum (non-integer / fractal dimensions supported).
- `sheaf.HeterogeneousGraphSheaf`: cellular sheaf with per-node
  (heterogeneous) stalk dimensions; reduces to `GraphSheaf` under
  uniform dims.
- Closed-form `H^7` heat kernel, one operator-chain step beyond `H^5`;
  the odd-n heat-kernel recursion now seeds from it.

**v0.5.0**: hyperbolic / pseudo-Riemannian / mixed-curvature
manifold pass.

- Five new manifold classes: `LorentzManifold` (hyperboloid model of
  `H^n_k`), `KappaStereographicManifold` (parametric κ interpolating
  spherical / Euclidean / hyperbolic, with learnable κ that can cross
  0 mid-training via dynamic sign dispatch), `LorentzianManifold`
  (pseudo-Riemannian `(1, n-1)` Minkowski spacetime with causal
  structure + flat-background curvature tensors), `ProductManifold`
  (mixed-curvature direct product), and `HeterogeneousKappaManifold`
  (per-point κ with a configurable pair-κ combiner).
- New `holonomy_lib.hyperbolic` module of manifold-agnostic graph
  operations: `manifold_aware_inner`, `frechet_mean` (Karcher
  iteration), `hyperbolic_laplacian_eigenmaps` (RSGD embedding), and
  `hyperbolic_heat_kernel` (dimension-dispatched H^n heat kernel).
- Correctness: found and fixed a missing spectral-shift factor in the
  heat-kernel dimension recursion that appears in some references;
  added a closed-form n=5 kernel (~3 orders of magnitude tighter PDE
  residual); even-n now works via the corrected recursion seeded from
  the n=2 integral form; autograd-finite gradients at every boundary
  input, established across all new manifolds.
- Test count 707 → 1143; audit clean.

**v0.4.1**: end-to-end MCP transport fixes.

- `mcp.py` eagerly imports all op-defining submodules at server
  startup so `OP_REGISTRY` is populated before `replay_with` runs
  (loaded registries reference ops the server process otherwise
  never touches).
- `_bind_registry` now inspects each tool's signature and only
  pre-binds the `registry` argument when the function actually
  declares one (`op_docstring` queries global state and doesn't).
- List-returning tools wrap their return in `{"results": [...]}`
  on the MCP transport so FastMCP serializes a single JSON content
  item instead of one per element. Python callers still see the
  underlying list via the unwrapped function; normalization is
  transport-only.

**v0.4.0**: provenance agent-API redesign.

- New `holonomy_lib.provenance.agent` module holds the canonical
  agent tool inventory. Each tool is a Python function decorated
  with `@agent_tool`; `to_anthropic_schema()` / `to_openai_schema()`
  emit native LLM tool-use JSON.
- Inspection tools (each callable as a native LLM tool, an MCP tool,
  or directly from Python): `tensor_slice` (numpy-syntax indexing),
  `tensor_per_batch_summary`, `tensor_eigenvalues`,
  `tensor_singular_values`, `tensor_norm`, `tensor_compare`,
  `op_docstring`. Replaces v0.3.0's global-stats-only
  `get_tensor_summary`, which couldn't see per-batch anomalies.
- `replay_with(target_hex, recipe)` substitution DSL: kinds are
  `zeros_like`, `from_hex`, `perturb` (Gaussian noise with required
  seed), `scale`, `swap_batch`, `literal`. Replaces v0.3.0's
  zero-fill-only MCP `replay`.
- `mcp.py` refactored to pure transport: iterates the agent
  inventory and pre-binds the registry argument. Same v0.3 nav
  tools by name (back-compat); the new inspection + replay_with
  tools land alongside.

**v0.3.0**: provenance module sweep.

- Performance: opt-in sketch hashing (15× faster on 8 MB tensors via
  shape + dtype + 64 strided samples + sum + std) and on-disk tensor
  cache (memory eviction retains the disk copy; `get_tensor()`
  reloads on demand).
- Robustness: `replay()` now works for class-method calls and
  tuple-of-tensor inputs (FixedRankPoint = (U, S, Vt)). Op-version
  drift detector on `load()` emits `ProvenanceVersionWarning` with
  optional `strict=True` escalation.
- Visualization: `to_mermaid()`, `to_graphviz()`, `diff_summary(other)`
  with Cache hits / Drift / Only-in-self / Only-in-other categories,
  `ancestors_with_tensors(hex)` convenience.
- Agent access: `to_llm_context()` text summary; MCP server
  (`pip install 'holonomy-lib[mcp]'`); Jupyter `%record_provenance`
  cell magic (`pip install 'holonomy-lib[jupyter]'`).

**v0.2.0**: six new modules and several extensions since
the v0.1.0 seed.

- New modules: `optimization` (RiemannianSGD), `simplicial`
  (Dense / Sparse complexes + Vietoris-Rips), `topology` (Hodge
  Laplacians + Betti + batched persistent homology H₀+H₁+H₂),
  `info_geometry` (Bregman + KL + Fisher metric + natural gradient),
  `sheaf` (cellular sheaves on graphs + sheaf Laplacians), `lie`
  (SO(3) primitives + real spherical harmonics for l ≤ 4).
- Spectral additions: Forman-Ricci curvature, magnetic Laplacian,
  sign-magnetic Laplacian, Chebyshev heat kernel, effective
  resistance, commute time, diffusion maps, and sparse-COO/CSR/CSC
  paths for all four Laplacian variants.
- Algebra additions: `lanczos_eigsh` with LA (largest algebraic) and
  shift-and-invert SA (smallest algebraic) modes.
- Class-method provenance for `FixedRankManifold` / `SPDManifold`;
  device-agnostic torch reduction backend for persistent homology
  (foundation for a future custom CUDA kernel).

**v0.1.0**: initial public release. `manifolds`, `algebra`,
`tensor_calculus`, `spectral` (4 Laplacians + eigenmaps),
`discrete_geometry` (Ollivier-Ricci + flow + surgery), `provenance`.

**Frontiers**: Wigner-D matrices (real basis) to complete the
SO(3) equivariance story so spherical-harmonic features mix
correctly under rotation; optimal transport extensions
(Gromov-Wasserstein for metric-measure-space comparison, Sinkhorn
divergences for de-biased OT); GPU-resident custom CUDA kernel for
the Z/2 PH reduction (current torch path is sequential with a
per-column CPU sync); higher-dim cellular sheaves on simplicial
complexes (2-cells / faces and the ∂₁∘∂₂ = 0 chain identity);
further manifolds (sphere, Stiefel, Grassmann);
SE(3) / SU(2) / SL(n) Lie group primitives.
(Sparse-input shift-and-invert for smallest-algebraic eigenpairs
shipped in v0.7.0 as `algebra.eigsh_sparse_sa`.)
Contributions welcome via PR.

---

## Citation

If this library helps your research, please cite it:

```bibtex
@software{holonomy_lib,
  author = {John Vaught},
  title = {holonomy\_lib: GPU-native research math for differential
           geometry, spectral graph theory, and mechanistic interpretability},
  year = {2026},
  url = {https://github.com/Synoros-io/holonomy_lib},
}
```

The library implements algorithms from many sources; please also cite
the original paper for the specific primitive you use (each public
function lists its references in its docstring).

---

## Contributing

See [`CONVENTIONS.md`](CONVENTIONS.md) for the full coding standards
(batched-first API shape, self-loop policy, numerical conventions,
performance patterns, magic-numbers catalog, citation requirements,
provenance, testing). Contributions welcome. Hard constraints
(binding for any code in this repo):

- **Citations are non-optional.** Every public function has a
  `References:` section pointing to the paper for its math.
- **Every numerical constant has a derivation or a catalog entry** in
  `notes/magic_numbers.md`. The audit tool enforces this.
- **Tests before merge**: unit tests for correctness, property tests
  for invariants, comparison tests against established libraries where
  one exists.
- **GPU-first, batched-first**: operations take a leading batch dim,
  work on `torch.Tensor` on `cuda`/`rocm`/`cpu`. Verify shapes for
  `B ∈ {0, 1, > 1}`.

Open an issue first for non-trivial additions so we can align on
approach.

---

## License

holonomy_lib is free software under the **GNU Affero General Public License v3.0 or later** (AGPL-3.0-or-later). See [LICENSE](LICENSE). Note the AGPL's network clause (§13): anyone who runs a modified version to provide a network service must make their modified source available to that service's users.

Copyright © 2026 Synoros.

**Commercial licensing.** The AGPL is not the only option. The copyright holder offers holonomy_lib under separate commercial terms for uses the AGPL does not permit, for example embedding it in a proprietary product or network service without the AGPL's source-disclosure obligations. For a commercial license, contact [contact@synoros.io](mailto:contact@synoros.io).
