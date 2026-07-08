# regularized_lstsq method dispatch — benchmark record (2026-07-08)

CPU float64, workstation. 3-rep mean after warmup, `perf_counter`.
Dense ridge problems, `A: (2n, n)` standard Gaussian (well-conditioned;
`λ = 0.1`), default tolerances (`sqrt(eps)`).

| m×n | cholesky | cg (normal op) | lsqr |
|---|---|---|---|
| 128×64 | 0.1 ms | 2.4 ms | 7.5 ms |
| 512×256 | 1.5 ms | 4.9 ms | 11.1 ms |
| 2048×1024 | 40.6 ms | 36.9 ms | 41.4 ms |
| 4096×2048 | 271.7 ms | 292.7 ms | 222.7 ms |

Read:
- Cholesky dominates by 3-24x through n=256 and is at parity (±25%,
  run-to-run noise territory) at n=1024-2048.
- The iterative timings are BEST-CASE for cg/lsqr: Gaussian `A` is
  well-conditioned, and Krylov iteration counts scale with `√cond(A)`
  while Cholesky's flops are fixed. On ill-conditioned problems the
  gap widens in Cholesky's favor (in wall time; in *accuracy* LSQR
  wins there — see the least_squares docstring).
- Conclusion for `method="auto"`: dense → `"cholesky"` with NO size
  threshold (no benchmarked crossing to derive one from); sparse →
  `"lsqr"` (Cholesky would need the dense Gram matrix). Users pick
  `"cg"`/`"lsqr"` explicitly for memory-bound dense (`n×n` Gram matrix
  too large) or accuracy-critical ill-conditioned problems.

The `SOLVER_SYNC_EVERY_DEFAULT = 8` cadence follows the established
`SINKHORN_SYNC_EVERY_DEFAULT` pattern (CONVENTIONS §4.1) — its
justification is the host-sync amortization argument, not this sweep.
