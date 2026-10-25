"""One persistence computation supplies the complete interactive filtration."""

from __future__ import annotations

import torch

from holonomy_lib.simplicial import pairwise_distances
from holonomy_lib.topology import persistence_diagrams

from .common import Context, Request
from .datasets import cloud


def reference_check(context: Context) -> Request:
    """Check the square's sole H1 interval against [1, sqrt(2)).

    References:
        Edelsbrunner and Harer (2010), Computational Topology, VII.
    """
    square = context.tensor([[[0, 0], [1, 0], [1, 1], [0, 1]]])
    diagrams, masks = context.call(persistence_diagrams, square, max_dim=1)
    bars = diagrams[1][masks[1]]
    # Zero-length intervals do not change persistence and may be retained by a
    # future reduction backend. The check concerns positive-lifetime H1 only.
    bars = bars[bars[:, 1] > bars[:, 0]]
    expected = context.tensor([[1, 2**0.5]])
    passed = bars.shape == expected.shape and torch.allclose(bars, expected)
    return {"passed": bool(passed), "h1": bars, "expected_h1": expected}


@torch.no_grad()
def compute(request: Request) -> Request:
    """Return bars and all radius events so slider movement needs no recompute.

    Reduction in the dependency is sequential per batch member and uses CPU
    bookkeeping even with GPU tensors. Counts below use [birth, death), including
    all simplices with diameter <= ε at a tied filtration event.

    References:
        Edelsbrunner, Letscher and Zomorodian (2002), Topological persistence
        and simplification. Discrete & Computational Geometry 28:511-533.
    """
    context = Context.from_request(request)
    points = cloud(request, context)
    distances = context.call(pairwise_distances, points[None])[0]
    if not bool(torch.isfinite(distances).all()):
        raise ValueError(
            "pairwise distances overflowed; use smaller coordinates or float64"
        )
    diagrams, masks = context.call(persistence_diagrams, points[None], max_dim=1)
    bars = [diagram[0, mask[0]] for diagram, mask in zip(diagrams, masks, strict=True)]
    indices = torch.arange(len(points), device=context.device)
    edges = torch.combinations(indices, r=2)
    triangles = torch.combinations(indices, r=3)
    edge_radii = distances[edges[:, 0], edges[:, 1]]
    triangle_radii = torch.stack(
        (
            distances[triangles[:, 0], triangles[:, 1]],
            distances[triangles[:, 0], triangles[:, 2]],
            distances[triangles[:, 1], triangles[:, 2]],
        ),
        -1,
    ).amax(-1)
    events = torch.unique(torch.cat((context.tensor([0]), edge_radii)), sorted=True)
    betti = torch.stack(
        [
            (
                (bar[None, :, 0] <= events[:, None])
                & (events[:, None] < bar[None, :, 1])
            ).sum(-1)
            for bar in bars
        ],
        -1,
    )
    reference = reference_check(context)
    warnings = (
        []
        if reference["passed"]
        else [
            "This library checkout fails the square H1 reference: it emits extra "
            "infinite bars. The bars and Betti counts shown here are provisional "
            "library results, not corrected by the explorer."
        ]
    )
    return context.finish(
        points=points,
        distances=distances,
        bars=bars,
        edges=edges,
        edge_radii=edge_radii,
        triangles=triangles,
        triangle_radii=triangle_radii,
        events=events,
        betti=betti,
        edge_counts=(edge_radii[None] <= events[:, None]).sum(-1),
        triangle_counts=(triangle_radii[None] <= events[:, None]).sum(-1),
        reference_check=reference,
        warnings=warnings,
        parameters={key: value for key, value in request.items() if key != "points"},
    )
