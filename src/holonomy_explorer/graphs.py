"""Graph scene: adapters for spectral and discrete-geometric primitives."""

from __future__ import annotations

import torch

from holonomy_lib import discrete_geometry, spectral
from holonomy_lib.spectral.laplacian import symmetric_normalized

from . import datasets
from .common import (
    MAX_FLOW_FRAMES,
    MAX_FLOW_STEPS,
    Context,
    Request,
    choice,
    integer,
    number,
)

# Small initial flow, leaving room for the user to increase the step budget.
DEFAULT_FLOW_STEPS = 6
DEFAULT_SURGERY_PERIOD = 2
DEFAULT_SURGERY_THRESHOLD = 1.3
DEFAULT_FLOW_DT = 0.5
MAX_SURGERY_THRESHOLD = 10.0
# With the library's default K=30, keep the heat slider in a modest t range.
MIN_HEAT_TIME = 0.01
MAX_HEAT_TIME = 10.0
REFERENCE_EPS_MULTIPLIER = 64  # Roundoff allowance for a six-node eigensolve.


def components(adjacency: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return component labels and sizes for one (N,N) graph.

    Repeated squaring doubles the reachable path length each round, so this
    loop is over log2(N) path lengths, never over individual nodes or edges.

    References:
        Warshall (1962), A Theorem on Boolean Matrices.
    """
    nodes = adjacency.shape[-1]
    reachable = (adjacency > 0) | torch.eye(
        nodes,
        device=adjacency.device,
        dtype=torch.bool,
    )
    for _ in range((nodes - 1).bit_length()):
        reachable = (reachable.to(adjacency.dtype) @ reachable.to(adjacency.dtype)) > 0
    indices = torch.arange(nodes, device=adjacency.device)
    first = torch.where(reachable, indices[None, :], nodes).amin(-1)
    _, labels, sizes = torch.unique(first, return_inverse=True, return_counts=True)
    return labels, sizes


def _resistance_reference(context: Context) -> Request:
    adjacency = context.tensor(
        [
            [
                [0, 1, 1, 0, 0, 0],
                [1, 0, 1, 0, 0, 0],
                [1, 1, 0, 1, 0, 0],
                [0, 0, 1, 0, 1, 1],
                [0, 0, 0, 1, 0, 1],
                [0, 0, 0, 1, 1, 0],
            ]
        ]
    )
    resistance = context.call(spectral.effective_resistance, adjacency)
    observed = torch.stack((resistance[0, 0, 1], resistance[0, 2, 3]))
    expected = context.tensor([2 / 3, 1])  # Triangle edge and unit bridge.
    error = (observed - expected).abs().max()
    return {
        "observed": observed,
        "expected": expected,
        "max_abs_error": error,
        "passed": bool(
            error < torch.finfo(context.dtype).eps * REFERENCE_EPS_MULTIPLIER
        ),
    }


@torch.no_grad()
def compute(request: Request) -> Request:
    """Return the graph, layout and a single library-computed overlay.

    Flow frames are sequential in step count. Each frame starts at the original
    adjacency: chunked calls reset surgery's cutoff and normalization reference.

    References:
        Belkin and Niyogi (2003), Laplacian Eigenmaps.
        Ollivier (2009), Ricci curvature of Markov chains on metric spaces.
        Forman (2003), Bochner's method for cell complexes.
        Ni et al. (2019), Community detection on networks with Ricci flow.
    """
    context = Context.from_request(request)
    adjacency, parameters = datasets.graph(request, context)
    nodes = adjacency.shape[-1]
    source = integer(request, "source", 0, 0, nodes - 1)
    overlay = choice(
        request,
        "overlay",
        "forman",
        (
            "none",
            "ollivier",
            "forman",
            "flow",
            "heat",
            "resistance",
            "spectrum",
        ),
    )
    eigenvalues, vectors = context.call(
        spectral.laplacian_eigenmaps,
        adjacency,
        k=min(3, nodes),
    )
    coordinates = torch.zeros((nodes, 2), device=context.device, dtype=context.dtype)
    coordinates[:, : vectors.shape[-1] - 1] = vectors[0, :, 1:3]
    edges = torch.triu_indices(nodes, nodes, 1, device=context.device)
    present = adjacency[0, edges[0], edges[1]] > 0
    edges = edges[:, present].mT
    labels, sizes = components(adjacency[0])
    warnings: list[str] = []
    if len(sizes) > 1:
        warnings.append(
            "Disconnected graph: the spectral layout can overlap vertices in "
            "degenerate component modes. Cross-component resistance is infinite."
        )
    result: Request = {
        "nodes": nodes,
        "edges": edges,
        "weights": adjacency[0, edges[:, 0], edges[:, 1]],
        "coordinates": coordinates,
        "overlay": overlay,
        "source": source,
        "parameters": parameters,
        "component_labels": labels,
        "component_sizes": sizes,
        "warnings": warnings,
    }
    parameters["overlay"] = overlay
    if overlay in ("ollivier", "forman"):
        function = (
            discrete_geometry.ollivier_ricci_curvature
            if overlay == "ollivier"
            else discrete_geometry.forman_ricci_simple
        )
        curvature = context.call(function, adjacency)
        values = curvature[0, edges[:, 0], edges[:, 1]]
        result["edge_values"] = values
        result["range"] = [values.min(), values.max()] if values.numel() else [0, 0]
        if overlay == "ollivier":
            warnings.append(
                "Ollivier values use the library's regularized Sinkhorn approximation."
            )
    elif overlay == "heat":
        time = number(request, "time", 1, MIN_HEAT_TIME, MAX_HEAT_TIME)
        laplacian = context.call(symmetric_normalized, adjacency)
        signal = torch.zeros((1, nodes, 1), device=context.device, dtype=context.dtype)
        signal[:, source, :] = 1
        heat = context.call(
            spectral.heat_kernel_chebyshev, laplacian, time, signal=signal
        )
        result["node_values"] = heat[0, :, 0]
        result["value_label"] = "Heat amplitude (normalized Laplacian)"
        parameters.update(time=time, source=source)
    elif overlay == "resistance":
        resistance = context.call(spectral.effective_resistance, adjacency)[0, source]
        result["node_values"] = torch.where(
            labels == labels[source],
            resistance,
            float("inf"),
        )
        result["value_label"] = "Effective resistance (unit-edge resistance units)"
        reference = _resistance_reference(context)
        result["reference_check"] = reference
        if not reference["passed"]:
            warnings.append(
                "Library resistance failed a triangle-and-bridge reference check "
                "in this dtype. Values below are the unmodified library output."
            )
        parameters["source"] = source
    elif overlay == "flow":
        steps = integer(request, "steps", DEFAULT_FLOW_STEPS, 1, MAX_FLOW_STEPS)
        period = integer(request, "surgery_period", DEFAULT_SURGERY_PERIOD, 1, steps)
        threshold = number(
            request,
            "surgery_threshold",
            DEFAULT_SURGERY_THRESHOLD,
            torch.finfo(context.dtype).eps,
            MAX_SURGERY_THRESHOLD,
        )
        dt = number(request, "dt", DEFAULT_FLOW_DT, torch.finfo(context.dtype).eps, 1)
        frame_steps = list(range(period, steps + 1, period))
        if not frame_steps or frame_steps[-1] != steps:
            frame_steps.append(steps)
        if len(frame_steps) > MAX_FLOW_FRAMES:
            raise ValueError(
                f"choose a longer surgery period: at most {MAX_FLOW_FRAMES} frames"
            )
        frames = [adjacency[0]]
        for step in frame_steps:
            frames.append(
                context.call(
                    discrete_geometry.ricci_flow_with_surgery,
                    adjacency,
                    n_steps=step,
                    surgery_period=period,
                    surgery_threshold=threshold,
                    dt=dt,
                )[0]
            )
        final_labels, final_sizes = components(frames[-1])
        result.update(
            frames=torch.stack(frames),
            frame_steps=[0, *frame_steps],
            final_labels=final_labels,
            final_sizes=final_sizes,
            flow_method=(
                "Each frame reruns from the original graph "
                "to its cumulative step count."
            ),
        )
        parameters.update(
            steps=steps, surgery_period=period, surgery_threshold=threshold, dt=dt
        )
        if parameters["graph"] == "karate":
            factions = torch.tensor(datasets.KARATE_FACTIONS, device=context.device)
            mismatch = (final_labels != factions).sum()
            result["faction_wrong_side"] = (
                torch.minimum(mismatch, nodes - mismatch)
                if len(final_sizes) == 2
                else None
            )
            result["faction_note"] = (
                "Best label assignment against the observed post-split clubs."
                if len(final_sizes) == 2
                else "Wrong-side count requires exactly two final components; "
                "this flow did not produce two."
            )
    if overlay in ("ollivier", "flow") and parameters["weighted"]:
        warnings.append(
            "Weighted Ollivier/flow currently use adjacency values both as path "
            "lengths and as random-walk mass. Interpret weighted results with care."
        )
    laplacian = context.call(symmetric_normalized, adjacency)
    spectrum = context.call(torch.linalg.eigvalsh, laplacian)[0]
    result.update(spectrum=spectrum, spectral_gap=spectrum[1] if nodes > 1 else None)
    return context.finish(**result)
