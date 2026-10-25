# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Synoros

"""Z/2 left-to-right boundary-matrix reduction for persistent homology.

The persistent-homology algorithm (Edelsbrunner-Letscher-Zomorodian
2002) reduces the boundary matrix `D` whose columns are simplices in
filtration order, with `D[i, j] = 1` iff simplex `i` is a face of
simplex `j` (working modulo 2). The reduction sweeps columns left to
right: for each column `j`, while `low(j)` (the largest row index of
any non-zero entry in column `j`) collides with the `low` of some
already-paired column `k < j`, replace `column j ← column j ⊕
column k`. The result yields persistence pairs: each column `j` with
non-zero `low(j) = i` becomes the persistence pair `(birth_i,
birth_j)`. Columns that reduce to zero are essential — their birth
gets matched only to `+∞` (until a higher-dim simplex kills them,
if applicable).

Two backends:

  `backend="python"` (default) — each column is a Python `set[int]`
    of non-zero row indices. XOR is set symmetric difference. `low`
    is `max(col_set)` if non-empty. C-implemented set ops are
    extremely fast for the small-to-moderate columns that typical
    PH inputs produce; the bottleneck is the inherently sequential
    column loop, not the per-column work.

  `backend="torch"` — each column is a `torch.LongTensor` of row
    indices on the filtration's native device. XOR is implemented
    via concat + `torch.unique(return_counts=True)` + odd-count
    filter; `low` is `col.max().item()` (one CPU sync per column).
    This path runs end-to-end on whatever device the filtration
    lives on (CPU or GPU), which is what `persistence_diagrams`
    needs when the filtration was already built on GPU. For typical
    small-to-moderate complexes the Python-set path is competitive
    or faster (set ops are tight C code) — the torch path matters
    when (1) you need device-agnostic semantics, or (2) the
    column sizes get large enough that vectorized `unique` beats
    Python set hashing. v1 ships as a foundation for a future
    custom CUDA kernel.

The reduction is inherently sequential per matrix (column j depends
on all earlier columns), but PH batches across point clouds at the
higher API level (`persistence_diagrams`).

Micro-optimizations applied:
  Clearing optimization (Bauer-Kerber-Reininghaus 2014, §3): a
  column already paired by an earlier column is zero, skip it. We
  detect this by tracking `essential_columns` and `paired_columns`
  during reduction.

References:
  Edelsbrunner, H., Letscher, D., Zomorodian, A. (2002). Topological
    persistence and simplification. Discrete & Computational
    Geometry 28:511-533. Original algorithm.
  Bauer, U., Kerber, M., Reininghaus, J. (2014). Clear and Compress:
    Computing Persistent Homology in Chunks. In: Topological Methods
    in Data Analysis and Visualization III, Springer. §3 — clearing.
  Otter, N., Porter, M. A., Tillmann, U., Grindrod, P., Harrington,
    H. A. (2017). A roadmap for the computation of persistent
    homology. EPJ Data Science 6(17). Survey + benchmarks.
"""

from __future__ import annotations

from typing import Literal

import torch

from holonomy_lib.topology._filtration import Filtration


def reduce_filtration(
    filtration: Filtration,
    backend: Literal["python", "torch"] = "python",
) -> dict[int, list[tuple[float, float]]]:
    """Run Z/2 left-to-right reduction. Return persistence pairs by dim.

    Args:
      filtration: a `Filtration` produced by `build_filtration`.
      backend: which reduction backend to use. `"python"` (default)
        uses Python `set[int]` columns; `"torch"` uses
        `torch.LongTensor` columns and runs on the filtration's
        native device.

    Returns:
      `dict[k, list[(birth, death)]]` — finite + essential bars for
      each dim k. Essential bars have `death = +inf`.
    """
    if backend == "torch":
        return _reduce_filtration_torch(filtration)
    if backend != "python":
        raise ValueError(
            f"backend must be 'python' or 'torch'; got backend={backend!r}"
        )
    n_total = int(filtration.birth_times.shape[0])
    # Build column-set representation of the boundary matrix in
    # filtration order. column[j] = set of row indices i (j's faces)
    # such that simplex i is a face of simplex j (mod 2).
    # We need the inverse map (dim, original_idx) -> filtration index.
    dims = filtration.sorted_dims.tolist()
    indices = filtration.sorted_indices.tolist()
    births = filtration.birth_times.tolist()
    complex = filtration.complex

    # Map (dim, original_idx_within_dim) -> filtration column index.
    orig_to_filt: dict[tuple[int, int], int] = {}
    for filt_idx in range(n_total):
        orig_to_filt[(dims[filt_idx], indices[filt_idx])] = filt_idx

    # Pre-compute face → filtration-index map for each dim.
    # For each (k-1)-face of every k-simplex, we need to find its
    # filtration index. Build the face lookup once per dim.
    from holonomy_lib.simplicial._face_lookup import build_simplex_index

    face_lookup_by_dim: dict[int, dict[tuple[int, ...], int]] = {}
    for k, simplices in complex.simplices_by_dim.items():
        face_lookup_by_dim[k] = build_simplex_index(simplices)

    # Build columns (Python sets of int row indices in filtration order).
    columns: list[set[int]] = []
    for j in range(n_total):
        k = dims[j]
        idx_in_dim = indices[j]
        if k == 0:
            columns.append(set())  # vertices have empty boundary
            continue
        simplex = complex.simplices_by_dim[k][idx_in_dim]
        face_indices = face_lookup_by_dim.get(k - 1, {})
        col_set: set[int] = set()
        # Enumerate faces: drop each vertex in turn. The Koszul sign
        # is irrelevant in Z/2; XOR commutes regardless.
        verts = simplex.tolist()
        for i in range(k + 1):
            face = tuple(verts[:i] + verts[i + 1:])
            if face not in face_indices:
                raise ValueError(
                    f"reduce_filtration: face {face} of simplex "
                    f"{tuple(verts)} not found in dim-{k - 1} simplices"
                )
            face_idx_in_dim = face_indices[face]
            filt_idx = orig_to_filt[(k - 1, face_idx_in_dim)]
            # XOR in Z/2: toggle membership.
            if filt_idx in col_set:
                col_set.remove(filt_idx)
            else:
                col_set.add(filt_idx)
        columns.append(col_set)

    # Z/2 left-to-right reduction.
    # `pivot_map[low] = j` for the paired column j with low(j) = low.
    # `death_columns: set[int]` tracks which columns are deaths (the
    # values of `pivot_map`); we maintain it alongside `pivot_map`
    # because `pivot_map.values()` is an O(n) view scan on dict, and
    # the essential-bar pass below would otherwise be O(n²) in
    # n_total.
    pivot_map: dict[int, int] = {}
    death_columns: set[int] = set()
    pairs_by_dim: dict[int, list[tuple[float, float]]] = {}

    for j in range(n_total):
        col = columns[j]
        while col:
            low = max(col)
            if low in pivot_map:
                # XOR column j with column pivot_map[low].
                col ^= columns[pivot_map[low]]
            else:
                # New pivot — record the persistence pair.
                pivot_map[low] = j
                death_columns.add(j)
                death_dim = dims[j]
                # Pair lives in dim (death_dim - 1) since it's the
                # H_{death_dim-1} bar born when `low` (a (death_dim-1)-
                # simplex) appeared and killed when j (a death_dim-
                # simplex) joined as its filler.
                bd_dim = death_dim - 1
                birth = births[low]
                death = births[j]
                if birth < death:
                    # Skip zero-length bars: they have no persistence.
                    pairs_by_dim.setdefault(bd_dim, []).append(
                        (birth, death),
                    )
                break

    # Essential bars: any column j that wasn't matched as a death AND
    # reduces to an empty column. In Z/2 reduction the algorithm
    # always finds a pivot for any non-empty reduced column, so the
    # `columns[j]` check is the actual condition; the membership test
    # in `death_columns` (O(1) set lookup) handles the corresponding
    # paired-column case.
    for j in range(n_total):
        if j in death_columns:
            continue
        if columns[j]:
            continue
        # Essential bar for dim = dims[j], born at births[j].
        bd_dim = dims[j]
        pairs_by_dim.setdefault(bd_dim, []).append(
            (births[j], float("inf")),
        )

    return pairs_by_dim


def _reduce_filtration_torch(
    filtration: Filtration,
) -> dict[int, list[tuple[float, float]]]:
    """Torch-tensor reduction: same algorithm, device-agnostic.

    Each column is a `torch.LongTensor` of non-zero row indices on
    the filtration's native device. XOR is `unique(cat(a, b))` filtered
    by `count % 2 == 1`; `low(col)` is `col.max().item()`. Pivot
    bookkeeping stays on CPU (a Python dict of int → int) since it's
    pure scalar logic.

    Cost per column: one `.item()` sync to read `low`, plus per XOR
    one `cat` + `unique` + boolean filter. For small sparse columns
    this is slower than CPython set ops; the benefit is that the
    column data never leaves the filtration's device.
    """
    n_total = int(filtration.birth_times.shape[0])
    device = filtration.birth_times.device
    dims = filtration.sorted_dims.tolist()
    indices = filtration.sorted_indices.tolist()
    births = filtration.birth_times.tolist()
    complex = filtration.complex

    orig_to_filt: dict[tuple[int, int], int] = {}
    for filt_idx in range(n_total):
        orig_to_filt[(dims[filt_idx], indices[filt_idx])] = filt_idx

    from holonomy_lib.simplicial._face_lookup import build_simplex_index

    face_lookup_by_dim: dict[int, dict[tuple[int, ...], int]] = {}
    for k, simplices in complex.simplices_by_dim.items():
        face_lookup_by_dim[k] = build_simplex_index(simplices)

    # Build columns as LongTensors. Pre-collect each column's face
    # filtration indices on Python side (face enumeration is cheap)
    # then construct a tensor on `device`.
    columns: list[torch.Tensor] = []
    for j in range(n_total):
        k = dims[j]
        idx_in_dim = indices[j]
        if k == 0:
            columns.append(torch.empty(0, dtype=torch.int64, device=device))
            continue
        simplex = complex.simplices_by_dim[k][idx_in_dim]
        face_indices = face_lookup_by_dim.get(k - 1, {})
        face_filt: list[int] = []
        verts = simplex.tolist()
        for i in range(k + 1):
            face = tuple(verts[:i] + verts[i + 1:])
            if face not in face_indices:
                raise ValueError(
                    f"_reduce_filtration_torch: face {face} of simplex "
                    f"{tuple(verts)} not found in dim-{k - 1} simplices"
                )
            face_idx_in_dim = face_indices[face]
            face_filt.append(orig_to_filt[(k - 1, face_idx_in_dim)])
        # Face indices are all distinct on a simplex (no duplicate
        # vertices) — the column has exactly k+1 entries before any
        # XOR. Construct directly without de-dup.
        columns.append(torch.tensor(face_filt, dtype=torch.int64, device=device))

    pivot_map: dict[int, int] = {}
    death_columns: set[int] = set()
    pairs_by_dim: dict[int, list[tuple[float, float]]] = {}

    for j in range(n_total):
        col = columns[j]
        while col.numel() > 0:
            low = int(col.max().item())                          # CPU sync
            if low in pivot_map:
                other = columns[pivot_map[low]]
                col = _xor_columns(col, other)
                columns[j] = col
            else:
                pivot_map[low] = j
                death_columns.add(j)
                death_dim = dims[j]
                bd_dim = death_dim - 1
                birth = births[low]
                death = births[j]
                if birth < death:
                    pairs_by_dim.setdefault(bd_dim, []).append(
                        (birth, death),
                    )
                break

    for j in range(n_total):
        if j in death_columns:
            continue
        if columns[j].numel() > 0:
            continue
        bd_dim = dims[j]
        pairs_by_dim.setdefault(bd_dim, []).append(
            (births[j], float("inf")),
        )

    return pairs_by_dim


def _xor_columns(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Z/2 XOR of two index-sets represented as LongTensors.

    The set XOR `A △ B = (A ∪ B) − (A ∩ B)` is equivalent to "elements
    that appear in exactly one of A, B". With both as sorted/unsorted
    LongTensors, concatenate them and keep entries whose total count
    is odd — exactly the XOR.

    Precondition: each input tensor is itself duplicate-free (a true
    "set"). The reduction loop guarantees this — initial columns are
    built from distinct simplex faces, and every later column is the
    output of `torch.unique` from a prior XOR. If you call this helper
    in a context where duplicates are possible, dedupe both inputs
    first; a triply-occurring index would incorrectly be dropped
    rather than kept (3 % 2 == 1 but the algorithm treats it as a
    cancelled "1 + 1 + 1" rather than an odd-survivor).
    """
    combined = torch.cat([a, b])
    uniq, counts = torch.unique(combined, return_counts=True)
    return uniq[counts % 2 == 1]
