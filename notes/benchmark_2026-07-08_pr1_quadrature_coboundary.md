# PR 1 benchmark record — quadrature cache + vectorized heterogeneous coboundary

Date: 2026-07-08. CPU float64, workstation (no CUDA available in this venv build).
Method: 5-rep mean after warmup, `perf_counter`; script in session scratchpad
(equivalent measurements reproducible via `tests/benchmarks/bench_manifolds.py`
case `comparison.model_ball_volume` and the regression tests in
`tests/sheaf/test_heterogeneous_graph_sheaf.py`).

## Heterogeneous coboundary: per-edge Python loop → vectorized COO assembly

Random ragged sheaves, node/edge stalk dims uniform in [1, 4]:

| n_edges | loop (old) | vectorized | speedup | outputs |
|---|---|---|---|---|
| 100 | 1.11 ms | 0.56 ms | 2.0x | bit-identical |
| 2 000 | 24.6 ms | 12.6 ms | 2.0x | bit-identical |
| 10 000 | 250 ms | 213 ms | 1.2x | bit-identical |

Component breakdown at n_edges = 10 000 (dense δ is (24 911, 4 991) ≈ 949 MiB):

- `torch.zeros` for the dense δ: **171 ms** — shared floor for BOTH paths;
  the dense path is allocation-bound at this scale by design (2 GiB cap).
- `_coboundary_coo` (index arithmetic + ragged flatten): 46 ms, of which the
  `torch.cat` over the 2·n_e ragged restriction maps is 18 ms.
- The scatter itself (`index_put_`): ~6 ms.

Read: assembly work dropped ~80 ms → ~50 ms; the remaining cost is the dense
allocation, which is exactly what the sparse path (`sheaf_coboundary_sparse`,
next PR) removes. `_coboundary_coo` is the shared builder for both.

On CUDA the old loop was structurally worse than these CPU numbers suggest
(2·n_e slice-assign kernel launches + `.tolist()` host syncs per call); the
vectorized path is a fixed ~30 kernels regardless of n_edges.

## model_ball_volume: cached Gauss-Legendre nodes

`scipy.special.roots_legendre` does NOT cache (only `scipy.integrate.fixed_quad`'s
private wrapper does); previously every call recomputed the rule and round-tripped
numpy→torch. Now `_quadrature.gauss_legendre_nodes` serves a cached float64 CPU
master per `n_quad` with a per-call dtype/device cast.

- B=4096, n_quad=64: 4.3 ms/call after caching. The scipy recompute+convert was
  ~0.1-0.2 ms/call of pure overhead — negligible per call at B=4096, but it was
  a per-call host-side serialization point, which matters for the substrate
  call pattern (`force_decay` at B=1 in tight loops) and for any CUDA caller
  (host-side numpy work forces sync points).

Same change applied to `hyperbolic/heat_kernel.py` (its "cached at the SciPy
level" comment was incorrect and is removed).
