# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Synoros

"""Z/2 left-to-right boundary-matrix reduction for persistent homology.

Boundary columns are prepared by tensor face enumeration and row joins.
The Python backend transfers their indices and filtration metadata to the
host together, once per filtration. The torch backend keeps columns on the
input device and transfers only metadata in that bulk operation.

Reduction is sequential: a column's pivot depends on earlier reduced
columns. Python uses set symmetric differences; torch uses odd occurrence
counts in concatenated index tensors. The torch backend synchronizes once
per pivot attempt, including every attempt after an XOR. It is not a fully
asynchronous GPU reduction. Neither backend implements clearing.

A nonzero reduced column pairs its lowest face with a death. A zero column
is essential only when no later column has paired that birth with a death.

References:
  Edelsbrunner, H., Letscher, D., Zomorodian, A. (2002). Topological
    persistence and simplification. Discrete & Computational Geometry
    28:511-533. Left-to-right boundary reduction.
"""

from __future__ import annotations

from typing import Literal

import torch

from holonomy_lib.simplicial._face_lookup import enumerate_faces, lookup_rows
from holonomy_lib.topology._filtration import Filtration


def _boundary_columns(filtration: Filtration) -> torch.Tensor:
    """Build the padded boundary matrix of a filtration, with -1 for unused slots.

    Takes a filtration whose complex is closed under taking faces. Returns an
    `(n_total, max_dim + 1)` int64 tensor. Row j holds the filtration ids of the
    faces of the simplex at filtration position j, and vertex rows are all -1.
    """
    complex = filtration.complex
    n_total = filtration.birth_times.numel()
    device = filtration.birth_times.device
    columns = torch.full(
        (n_total, complex.max_dim + 1), -1, dtype=torch.int64, device=device,
    )
    filtration_ids = torch.arange(n_total, device=device)
    inverse_by_dim: dict[int, torch.Tensor] = {}
    # Map each simplex's row within its dimension to its position in the
    # filtration. Faces are found by row, so this inverse turns them into
    # filtration ids.
    for dim, simplices in complex.simplices_by_dim.items():
        selected = filtration.sorted_dims == dim
        inverse = torch.empty(simplices.shape[0], dtype=torch.int64, device=device)
        inverse[filtration.sorted_indices[selected]] = filtration_ids[selected]
        inverse_by_dim[dim] = inverse
    for dim, simplices in complex.simplices_by_dim.items():
        if dim == 0 or simplices.shape[0] == 0:
            continue
        # Each simplex has dim+1 faces. enumerate_faces lists them as vertex
        # tuples, and lookup_rows finds each one's row in the dimension below.
        # The reshape gives every simplex its own row of face rows.
        faces, _ = enumerate_faces(simplices)
        face_ids = lookup_rows(
            complex.simplices_by_dim[dim - 1], faces.reshape(-1, dim),
        ).reshape(simplices.shape[0], dim + 1)
        columns[inverse_by_dim[dim], :dim + 1] = inverse_by_dim[dim - 1][face_ids]
    return columns


def _host_records(
    filtration: Filtration, columns: torch.Tensor | None,
) -> tuple[list[int], list[float], list[list[int]]]:
    """Copy dims, birth times, and optional boundary columns to host lists.

    Takes a filtration and the padded boundary matrix or None. Returns three
    lists: the dim of each simplex, its birth time, and the columns as nested
    lists of filtration ids (empty when `columns` is None). Everything goes
    over in one device-to-host copy as raw bytes, so int64 indices never pass
    through a float dtype.
    """
    dims = filtration.sorted_dims.contiguous()
    births = filtration.birth_times.contiguous()
    chunks = [dims.view(torch.uint8), births.view(torch.uint8)]
    if columns is not None:
        chunks.append(columns.contiguous().view(torch.uint8).flatten())
    host = torch.cat(chunks).cpu()
    dims_end = dims.numel() * dims.element_size()
    births_end = dims_end + births.numel() * births.element_size()
    host_dims = host[:dims_end].view(dims.dtype).tolist()
    host_births = host[dims_end:births_end].view(births.dtype).tolist()
    # Float births can leave the columns at a byte offset that is not a
    # multiple of 8, and an int64 view needs that alignment. Cloning the slice
    # gives it a fresh host buffer that starts at offset 0. This is a host
    # copy, not another device transfer.
    host_columns = (
        host[births_end:].clone().view(columns.dtype).reshape(columns.shape).tolist()
        if columns is not None else []
    )
    return host_dims, host_births, host_columns


def reduce_filtration(
    filtration: Filtration,
    backend: Literal["python", "torch"] = "python",
) -> dict[int, list[tuple[float, float]]]:
    """Run sequential Z/2 reduction and return persistence pairs by dimension.

    Args:
      filtration: the simplex ordering produced by `build_filtration`.
      backend: `"python"` for host set columns, or `"torch"` for index
        tensors on the input device. Torch synchronizes for each pivot
        attempt because the next column operation depends on that pivot.

    Returns:
      A dictionary keyed by homology dimension, holding the finite and
      essential `(birth, death)` pairs. Essential bars have infinite death;
      zero-length bars are omitted.

    References:
      Edelsbrunner, Letscher, Zomorodian (2002). Topological persistence
        and simplification. Discrete & Computational Geometry 28:511-533.
    """
    if backend not in ("python", "torch"):
        raise ValueError(f"backend must be 'python' or 'torch'; got {backend!r}")
    boundary = _boundary_columns(filtration)
    # The python backend needs the boundary on the host as sets. The torch
    # backend keeps it on the device, so only dims and births are copied.
    dims, births, host_columns = _host_records(
        filtration, boundary if backend == "python" else None,
    )
    if backend == "torch":
        pairs, pivots, empty = _reduce_torch(boundary, dims)
    else:
        # One set of filtration ids per column. A dim-0 simplex has no
        # boundary, so its column starts out empty.
        columns = [set(row[:dim + 1]) if dim else set()
                   for row, dim in zip(host_columns, dims, strict=True)]
        pairs, pivots, empty = _reduce_python(columns)

    # Each pair joins a birth simplex to the simplex that kills it. A bar whose
    # death is not after its birth has zero length and is dropped.
    pairs_by_dim: dict[int, list[tuple[float, float]]] = {}
    for birth_id, death_id in pairs:
        birth, death = births[birth_id], births[death_id]
        if birth < death:
            pairs_by_dim.setdefault(dims[birth_id], []).append((birth, death))

    # A zero column creates a class at its own index. If some column later has
    # that index as its low, the class died there. The rest never die.
    essential = torch.tensor(empty, dtype=torch.bool)
    essential[list(pivots)] = False
    for index in essential.nonzero().flatten().tolist():
        pairs_by_dim.setdefault(dims[index], []).append((births[index], float("inf")))
    return pairs_by_dim


def _reduce_python(
    columns: list[set[int]],
) -> tuple[list[tuple[int, int]], dict[int, int], list[bool]]:
    """Reduce the boundary columns left to right over Z/2.

    Takes one set of filtration ids per simplex, in filtration order. Returns
    the (low, index) pairs, a map from each low to the column that owns it,
    and a flag per column that is True when the reduced column is zero.
    Assumes the columns are in filtration order, since each column's pivot
    depends only on the columns before it.
    """
    pivot_map: dict[int, int] = {}
    pairs: list[tuple[int, int]] = []
    for index, column in enumerate(columns):
        while column:
            # The low is the largest filtration id left in the column, which is
            # the latest-born face. If an earlier column already owns that low,
            # add that column in over Z/2 to cancel the face, then look again.
            # Otherwise this column claims the low and is done.
            low = max(column)
            if low in pivot_map:
                column ^= columns[pivot_map[low]]
            else:
                pivot_map[low] = index
                pairs.append((low, index))
                break
    return pairs, pivot_map, [not column for column in columns]


def _reduce_torch(
    boundary: torch.Tensor, dims: list[int],
) -> tuple[list[tuple[int, int]], dict[int, int], list[bool]]:
    """Reduce the boundary columns on the device; same output as the python path.

    Takes the padded boundary matrix and the dim of each simplex. Returns the
    same triple as `_reduce_python`. Each pass through the loop reads the pivot
    back with `.item()`, so every pivot attempt costs one device sync.
    """
    # Keep only the real faces of each row. Vertex rows have none, and the -1
    # padding is cut off.
    columns = [row[:dim + 1] if dim else row[:0]
               for row, dim in zip(boundary.unbind(), dims, strict=True)]
    pivot_map: dict[int, int] = {}
    pairs: list[tuple[int, int]] = []
    for index, column in enumerate(columns):
        while column.numel():
            # .item() copies the pivot to the host because the branch below
            # depends on its value. This is the one sync per attempt.
            low = int(column.max().item())
            if low in pivot_map:
                column = _xor_columns(column, columns[pivot_map[low]])
                columns[index] = column
            else:
                pivot_map[low] = index
                pairs.append((low, index))
                break
    return pairs, pivot_map, [column.numel() == 0 for column in columns]


def _xor_columns(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Return the Z/2 sum of two boundary columns as a sorted index tensor.

    Takes two 1-D index tensors. Returns the entries that appear in exactly one
    input, sorted ascending. Assumes no repeats within either input, since a
    repeat would be counted as a second occurrence and cancel wrongly.
    """
    unique, counts = torch.unique(torch.cat((a, b)), return_counts=True)
    return unique[counts % 2 == 1]
