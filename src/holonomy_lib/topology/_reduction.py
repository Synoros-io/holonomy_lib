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
    """Return padded face indices in filtration order, with -1 for padding."""
    complex = filtration.complex
    n_total = filtration.birth_times.numel()
    device = filtration.birth_times.device
    columns = torch.full(
        (n_total, complex.max_dim + 1), -1, dtype=torch.int64, device=device,
    )
    filtration_ids = torch.arange(n_total, device=device)
    inverse_by_dim: dict[int, torch.Tensor] = {}
    # Dimension progression is structural; each join covers all simplices.
    for dim, simplices in complex.simplices_by_dim.items():
        selected = filtration.sorted_dims == dim
        inverse = torch.empty(simplices.shape[0], dtype=torch.int64, device=device)
        inverse[filtration.sorted_indices[selected]] = filtration_ids[selected]
        inverse_by_dim[dim] = inverse
    for dim, simplices in complex.simplices_by_dim.items():
        if dim == 0 or simplices.shape[0] == 0:
            continue
        faces, _ = enumerate_faces(simplices)
        face_ids = lookup_rows(
            complex.simplices_by_dim[dim - 1], faces.reshape(-1, dim),
        ).reshape(simplices.shape[0], dim + 1)
        columns[inverse_by_dim[dim], :dim + 1] = inverse_by_dim[dim - 1][face_ids]
    return columns


def _host_records(
    filtration: Filtration, columns: torch.Tensor | None,
) -> tuple[list[int], list[float], list[list[int]]]:
    """Transfer metadata and optional columns once, preserving their dtypes.

    Byte packing avoids converting integer indices through a floating dtype.
    The subsequent views and list conversions all operate on host memory.
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
    # Float metadata can leave the byte offset unaligned for int64. A host
    # clone resets that offset without another device transfer.
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
      A dictionary of finite and essential `(birth, death)` pairs. Essential
      bars have infinite death; zero-length bars are omitted.

    References:
      Edelsbrunner, Letscher, Zomorodian (2002). Topological persistence
        and simplification. Discrete & Computational Geometry 28:511-533.
    """
    if backend not in ("python", "torch"):
        raise ValueError(f"backend must be 'python' or 'torch'; got {backend!r}")
    boundary = _boundary_columns(filtration)
    dims, births, host_columns = _host_records(
        filtration, boundary if backend == "python" else None,
    )
    if backend == "torch":
        pairs, pivots, empty = _reduce_torch(boundary, dims)
    else:
        columns = [set(row[:dim + 1]) if dim else set()
                   for row, dim in zip(host_columns, dims, strict=True)]
        pairs, pivots, empty = _reduce_python(columns)

    pairs_by_dim: dict[int, list[tuple[float, float]]] = {}
    for birth_id, death_id in pairs:
        birth, death = births[birth_id], births[death_id]
        if birth < death:
            pairs_by_dim.setdefault(dims[birth_id], []).append((birth, death))

    # A zero column births a class, but that class may have died later.
    essential = torch.tensor(empty, dtype=torch.bool)
    essential[list(pivots)] = False
    for index in essential.nonzero().flatten().tolist():
        pairs_by_dim.setdefault(dims[index], []).append((births[index], float("inf")))
    return pairs_by_dim


def _reduce_python(
    columns: list[set[int]],
) -> tuple[list[tuple[int, int]], dict[int, int], list[bool]]:
    """Reduce host columns in order; pivots depend on earlier columns."""
    pivot_map: dict[int, int] = {}
    pairs: list[tuple[int, int]] = []
    for index, column in enumerate(columns):
        while column:
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
    """Reduce device columns, synchronizing once per dependent pivot attempt."""
    columns = [row[:dim + 1] if dim else row[:0]
               for row, dim in zip(boundary.unbind(), dims, strict=True)]
    pivot_map: dict[int, int] = {}
    pairs: list[tuple[int, int]] = []
    for index, column in enumerate(columns):
        while column.numel():
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
    """Return Z/2 symmetric difference; each input must be duplicate-free."""
    unique, counts = torch.unique(torch.cat((a, b)), return_counts=True)
    return unique[counts % 2 == 1]
