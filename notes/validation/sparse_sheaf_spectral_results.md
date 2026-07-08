# Sparse sheaf spectral path — validation record (2026-07-08)

Backport of synoros-substrate `v4/sheaf_scale.py` (its docstring: "the
sparse-sheaf spectral primitive to FORK INTO holonomy_lib once proven";
validated there 2026-06-06, Spearman 0.98 vs the dense read at n=824).
Generic parts only: sparse coboundary/Laplacian + bottom-k-nonzero solve.
The footprint/query machinery (`SpectralGeometry`) stays in substrate.

## Parity vs substrate's validated implementation

Random ragged sheaf, n=120 nodes (dims 1-4), 194 edges (dims 1-3),
Σd_v=318, seed 42; compared in one process against
`synoros_substrate.v4.sheaf_scale.{sparse_sheaf_laplacian, _bottom_k_eigsh}`:

- `sheaf_laplacian_sparse` vs substrate's scipy CSR assembly:
  **max |diff| = 0.0** (bit-identical).
- `sheaf_bottom_spectrum(k=8)` vs substrate's eigsh + kernel-drop:
  **max |diff| = 9.4e-11** on the eigenvalues.
- Node offsets identical to substrate's `voff`.

Incidental finding: that random ragged sheaf has a genuine 11-dim kernel
(confirmed against dense `eigvalsh`; nodes whose stalk dim exceeds the
rank of their incident restriction maps contribute global sections), so
`kernel_headroom=8` correctly raised with instructions and the default
16 cleared it. The failure mode works as designed.

## Backend agreement (tests/algebra/test_eigsh_sparse.py)

- Every backend (`eigsh`, `lobpcg`, `torch_lobpcg`, `auto`) vs dense
  `torch.linalg.eigvalsh` bottom-k: eigsh exact to 1e-10; LOBPCG branches
  to their 1e-6 stopping tolerance (values in practice ~1e-11 for
  torch_lobpcg on graph Laplacians at n=60).
- SciPy eigsh vs SciPy lobpcg+spilu on the same operator: ≤ 1e-7
  (values; the substrate "same modes" claim re-verified in-repo).
- Sparse SA vs dense in-house shift-invert Lanczos: ≤ 1e-8.

## Scale sanity (past the dense cap)

Random ragged sheaf: 10 000 nodes, 17 997 edges, Σd_v=20 049,
Σd_e=35 868. Dense δ would be 5.4 GiB → the dense path refuses (cap).
Sparse path: L_F nnz=192 329 assembled in 0.12 s; bottom-16 nonzero
modes via auto→lobpcg+spilu in ~126 s (a deliberately nasty spectrum:
~600-dim near-kernel from dims-1-3 random maps, smallest nonzero modes
at 1e-7). Real corpora (substrate's ConceptNet-scale sheaves) have
small kernels and were validated to ~150k stalk-dims in substrate.

CUDA note: `torch.lobpcg` branch is unpreconditioned; validated at test
scale only (see `SPARSE_SA_*` catalog entries and the module docstring
of `algebra/eigsh_sparse.py` for the scale-of-validity statement).
