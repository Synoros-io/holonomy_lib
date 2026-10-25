"""Synchronized warm-up and repeated timings of the actual library paths."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from holonomy_lib.manifolds import KappaStereographicManifold
from holonomy_lib.spectral import heat_kernel_chebyshev
from holonomy_lib.spectral.laplacian import symmetric_normalized
from holonomy_lib.topology import persistence_diagrams

from .common import (
    DEFAULT_REPEATS,
    DEFAULT_SEED,
    DEFAULT_SWEEP,
    MAX_BATCH_SIZE,
    MAX_NODES,
    MAX_PERSISTENCE_BATCH,
    MAX_PERSISTENCE_SIZE,
    MAX_REPEATS,
    Context,
    Request,
    choice,
    integer,
)
from .datasets import MAX_SEED
from .topology import reference_check

# Keep the initial persistence sweep small enough for interactive CPU reduction.
DEFAULT_PROBLEM_SIZE = 8
BENCHMARK_GRAPH_DENSITY = 0.25  # Example Erdős-Rényi connection probability.
BENCHMARK_POINT_SCALE = 0.2  # Uniform square sits inside the κ=-1 unit ball.


def _operation(
    primitive: str,
    batch: int,
    size: int,
    context: Context,
    seed: int,
) -> Callable[[], Any]:
    generator = context.generator(seed)
    options: dict[str, Any] = {"device": context.device, "dtype": context.dtype}
    if primitive == "heat":
        upper = (
            (
                torch.rand((batch, size, size), generator=generator, **options)
                < BENCHMARK_GRAPH_DENSITY
            )
            .to(context.dtype)
            .triu(1)
        )
        adjacency = upper + upper.mT
        # Matrix construction is outside the measured primitive call.
        laplacian = symmetric_normalized(adjacency)
        signal = torch.ones((batch, size, 1), **options)
        return lambda: heat_kernel_chebyshev(laplacian, 1.0, signal=signal)
    points = torch.rand((batch, size, 2), generator=generator, **options)
    if primitive == "distance":
        points = (points - 0.5) * (2 * BENCHMARK_POINT_SCALE)
        manifold = KappaStereographicManifold(n=2, kappa=-1, **options)
        return lambda: manifold.distance(points[:, :, None, :], points[:, None, :, :])
    return lambda: persistence_diagrams(points, max_dim=1)


def _probe(output: Any) -> Request:
    if isinstance(output, torch.Tensor):
        return {
            "shape": list(output.shape),
            "all_finite": bool(torch.isfinite(output).all()),
        }
    diagrams, masks = output
    return {
        "diagram_shapes": [list(item.shape) for item in diagrams],
        "valid_bars": [int(mask.sum()) for mask in masks],
    }


@torch.no_grad()
def compute(request: Request) -> Request:
    """Sweep batch sizes, warming each shape before synchronized measurements.

    Repeats and device/shape sweeps are deliberately sequential to avoid timing
    competing work. The tensor operation itself receives the whole batch.

    References:
        Hammond, Vandergheynst and Gribonval (2011), Wavelets on graphs.
        Bachmann et al. (2020), Constant Curvature Graph Convolutional Networks.
        Edelsbrunner, Letscher and Zomorodian (2002), Topological persistence.
    """
    selected = Context.from_request(request)
    primitive = choice(
        request, "primitive", "heat", ("heat", "distance", "persistence")
    )
    size_cap = MAX_PERSISTENCE_SIZE if primitive == "persistence" else MAX_NODES
    batch_cap = MAX_PERSISTENCE_BATCH if primitive == "persistence" else MAX_BATCH_SIZE
    size = integer(request, "size", DEFAULT_PROBLEM_SIZE, 1, size_cap)
    repeats = integer(request, "repeats", DEFAULT_REPEATS, 1, MAX_REPEATS)
    seed = integer(request, "seed", DEFAULT_SEED, 0, MAX_SEED)
    batches = request.get(
        "batches", [batch for batch in DEFAULT_SWEEP if batch <= batch_cap]
    )
    if (
        not isinstance(batches, list)
        or not batches
        or len(batches) > len(DEFAULT_SWEEP)
    ):
        raise ValueError(f"batches must contain 1 to {len(DEFAULT_SWEEP)} batch sizes")
    if any(type(batch) is not int or not 1 <= batch <= batch_cap for batch in batches):
        raise ValueError(
            f"each batch size must be an integer between 1 and {batch_cap}"
        )
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(
            selected.device
            if selected.device.type == "cuda"
            else torch.device("cuda", 0)
        )
    rows: list[Request] = []
    warnings: list[str] = []
    references: list[Request] = []
    for device in devices:
        context = Context(device, selected.dtype)
        if primitive == "persistence":
            reference = reference_check(context)
            references.append({"device": str(device), **reference})
            if not reference["passed"]:
                warnings.append(
                    f"{device}: square H1 reference failed; "
                    "timing the current provisional reduction."
                )
        for batch in sorted(set(batches)):
            operation = _operation(primitive, batch, size, context, seed)
            probe = _probe(context.call(operation))
            times = []
            for _ in range(repeats):
                start = context.elapsed_ms
                context.call(operation)
                times.append(context.elapsed_ms - start)
            timings = torch.tensor(times, dtype=torch.float64)
            median = timings.quantile(0.5)
            rows.append(
                {
                    "device": str(device),
                    "batch": batch,
                    "min_ms": timings.min(),
                    "median_ms": median,
                    "max_ms": timings.max(),
                    "samples_ms": times,
                    "per_item_ms": median / batch,
                    "per_item_min_ms": timings.min() / batch,
                    "per_item_max_ms": timings.max() / batch,
                    "probe": probe,
                }
            )
        selected.elapsed_ms += context.elapsed_ms
        selected.calls.extend(context.calls)
    return selected.finish(
        rows=rows,
        warnings=warnings,
        reference_checks=references,
        parameters={
            "primitive": primitive,
            "size": size,
            "repeats": repeats,
            "seed": seed,
            "batches": sorted(set(batches)),
            "warmups": 1,
        },
        timing_note=(
            "Wall time, synchronized before and after each call; "
            "input generation excluded."
        ),
        persistence_note=(
            "Persistence reduction is sequential per batch member, with CPU "
            "bookkeeping even for GPU inputs. It is expected to scale differently."
        ),
    )
