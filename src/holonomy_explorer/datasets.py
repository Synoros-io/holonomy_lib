"""Small reproducible inputs; the explorer has no dataset download step."""

from __future__ import annotations

import math

import torch

from .common import (
    DEFAULT_NODES,
    DEFAULT_NOISE,
    DEFAULT_POINTS,
    DEFAULT_SEED,
    MAX_NODES,
    MAX_POINTS,
    Context,
    Request,
    choice,
    integer,
    number,
    points_tensor,
)

# Zachary (1977), Journal of Anthropological Research 33, 452-473, Table 3.
# Zero-based, unweighted transcription of the 34-member, 78-edge network.
KARATE_NODES = 34
KARATE_EDGE_COUNT = 78
KARATE_DEGREES = (16, 17)  # Vertices 0 and 33, checked against the transcription.
KARATE_EDGES = (
    (0, 1),
    (0, 2),
    (0, 3),
    (0, 4),
    (0, 5),
    (0, 6),
    (0, 7),
    (0, 8),
    (0, 10),
    (0, 11),
    (0, 12),
    (0, 13),
    (0, 17),
    (0, 19),
    (0, 21),
    (0, 31),
    (1, 2),
    (1, 3),
    (1, 7),
    (1, 13),
    (1, 17),
    (1, 19),
    (1, 21),
    (1, 30),
    (2, 3),
    (2, 7),
    (2, 8),
    (2, 9),
    (2, 13),
    (2, 27),
    (2, 28),
    (2, 32),
    (3, 7),
    (3, 12),
    (3, 13),
    (4, 6),
    (4, 10),
    (5, 6),
    (5, 10),
    (5, 16),
    (6, 16),
    (8, 30),
    (8, 32),
    (8, 33),
    (9, 33),
    (13, 33),
    (14, 32),
    (14, 33),
    (15, 32),
    (15, 33),
    (18, 32),
    (18, 33),
    (19, 33),
    (20, 32),
    (20, 33),
    (22, 32),
    (22, 33),
    (23, 25),
    (23, 27),
    (23, 29),
    (23, 32),
    (23, 33),
    (24, 25),
    (24, 27),
    (24, 31),
    (25, 31),
    (26, 29),
    (26, 33),
    (27, 33),
    (28, 31),
    (28, 33),
    (29, 32),
    (29, 33),
    (30, 32),
    (30, 33),
    (31, 32),
    (31, 33),
    (32, 33),
)
# Observed club after fission, not the faction predicted by a graph algorithm.
KARATE_FACTIONS = (
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    1,
    0,
    0,
    0,
    0,
    1,
    1,
    0,
    0,
    1,
    0,
    1,
    0,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
)
# Initial controls are examples, not fitted model parameters.
DEFAULT_BLOCKS = 3
DEFAULT_P_IN = 0.5
DEFAULT_P_OUT = 0.05
DEFAULT_RADIUS = 0.3
DEFAULT_CLIQUE_SIZE = 5
DEFAULT_CLIQUES = 4
MAX_SEED = 2**32 - 1  # Portable unsigned 32-bit seed range.
CIRCLE_SEPARATION = 3.0  # Unit circles remain disjoint before adding noise.
ANNULUS_INNER_RADIUS = 0.5
CLUSTER_SPREAD = 0.15  # Example Gaussian standard deviation, coordinate units.


def graph(request: Request, context: Context) -> tuple[torch.Tensor, Request]:
    """Generate an undirected simple adjacency of shape (1, N, N).

    References:
        Zachary (1977), An Information Flow Model for Conflict and Fission
        in Small Groups, J. Anthropological Research 33:452-473.
        Holland, Laskey and Leinhardt (1983), Stochastic blockmodels.
    """
    kind = choice(
        request,
        "graph",
        "karate",
        (
            "karate",
            "sbm",
            "barbell",
            "cliques",
            "geometric",
            "custom",
        ),
    )
    seed = integer(request, "seed", DEFAULT_SEED, 0, MAX_SEED)
    generator = context.generator(seed)
    metadata: Request = {"graph": kind, "seed": seed, "weighted": False}
    if kind == "karate":
        adjacency = torch.zeros(
            (KARATE_NODES, KARATE_NODES),
            device=context.device,
            dtype=context.dtype,
        )
        edges = torch.tensor(KARATE_EDGES, device=context.device)
        adjacency[edges[:, 0], edges[:, 1]] = 1
        metadata["citation"] = "Zachary (1977), J. Anthropol. Res. 33:452–473"
        metadata["factions"] = KARATE_FACTIONS
    elif kind == "custom":
        adjacency, weighted = _edge_list(request, context)
        metadata["weighted"] = weighted
    elif kind in ("barbell", "cliques"):
        size = integer(request, "clique_size", DEFAULT_CLIQUE_SIZE, 2, MAX_NODES // 2)
        count = (
            2
            if kind == "barbell"
            else integer(request, "cliques", DEFAULT_CLIQUES, 2, MAX_NODES // 2)
        )
        if size * count > MAX_NODES:
            raise ValueError(f"clique_size × cliques must be at most {MAX_NODES}")
        labels = torch.arange(size * count, device=context.device) // size
        adjacency = (labels[:, None] == labels[None, :]).to(context.dtype)
        bridges = torch.arange(count - (kind == "barbell"), device=context.device)
        adjacency[bridges * size + size - 1, ((bridges + 1) % count) * size] = 1
        metadata.update(clique_size=size, cliques=count)
    else:
        nodes = integer(request, "nodes", DEFAULT_NODES, 3, MAX_NODES)
        if kind == "sbm":
            blocks = integer(request, "blocks", DEFAULT_BLOCKS, 1, nodes)
            p_in = number(request, "p_in", DEFAULT_P_IN, 0, 1)
            p_out = number(request, "p_out", DEFAULT_P_OUT, 0, 1)
            labels = torch.arange(nodes, device=context.device) * blocks // nodes
            probability = torch.where(
                labels[:, None] == labels[None, :],
                p_in,
                p_out,
            )
            draws = torch.rand(
                (nodes, nodes),
                generator=generator,
                device=context.device,
                dtype=context.dtype,
            )
            adjacency = (draws < probability).to(context.dtype).triu(1)
            metadata.update(nodes=nodes, blocks=blocks, p_in=p_in, p_out=p_out)
        else:
            radius = number(request, "radius", DEFAULT_RADIUS, 0, math.sqrt(2))
            points = torch.rand(
                (nodes, 2),
                generator=generator,
                device=context.device,
                dtype=context.dtype,
            )
            adjacency = (torch.cdist(points, points) <= radius).to(context.dtype)
            metadata.update(nodes=nodes, radius=radius)
    adjacency = torch.maximum(adjacency, adjacency.mT)
    adjacency.fill_diagonal_(0)
    return adjacency.unsqueeze(0), metadata


def _edge_list(request: Request, context: Context) -> tuple[torch.Tensor, bool]:
    text = request.get("edges", "")
    if not isinstance(text, str):
        raise ValueError("edges must be text: i j [weight] per line")
    parsed: dict[tuple[int, int], float] = {}
    largest = -1
    # Text parsing is outside the tensor compute path. Reject conflicting duplicate
    # edges so that input order cannot silently change the graph's meaning.
    for line_number, line in enumerate(text.splitlines(), start=1):
        parts = line.split("#", 1)[0].split()
        if not parts:
            continue
        if len(parts) not in (2, 3):
            raise ValueError(f"line {line_number}: expected i j [weight]")
        left, right = int(parts[0]), int(parts[1])
        weight = float(parts[2]) if len(parts) == 3 else 1.0
        if not 0 <= min(left, right) <= max(left, right) < MAX_NODES:
            raise ValueError(f"node IDs must be between 0 and {MAX_NODES - 1}")
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError("edge weights must be finite and positive")
        largest = max(largest, left, right)
        if left == right:
            continue
        key = (min(left, right), max(left, right))
        if key in parsed and parsed[key] != weight:
            raise ValueError(f"conflicting weights for edge {key}")
        parsed[key] = weight
    nodes = integer(request, "nodes", max(1, largest + 1), 1, MAX_NODES)
    if largest >= nodes:
        raise ValueError("nodes must include the largest edge endpoint")
    adjacency = torch.zeros((nodes, nodes), device=context.device, dtype=context.dtype)
    if parsed:
        endpoints = torch.tensor(list(parsed), device=context.device)
        weights = context.tensor(list(parsed.values()))
        if not bool((torch.isfinite(weights) & (weights > 0)).all()):
            raise ValueError(
                "edge weights must stay finite and positive in the chosen dtype"
            )
        adjacency[endpoints[:, 0], endpoints[:, 1]] = weights
    return adjacency, any(weight != 1 for weight in parsed.values())


def cloud(request: Request, context: Context) -> torch.Tensor:
    """Generate (N, 2) Euclidean points or use the user's clicks.

    References:
        The polar area element r dr dθ gives uniform annulus area sampling.
    """
    kind = choice(
        request,
        "cloud",
        "circle",
        (
            "circle",
            "two_circles",
            "figure_eight",
            "annulus",
            "clusters",
            "custom",
        ),
    )
    if kind == "custom":
        return points_tensor(request, context, MAX_POINTS)
    count = integer(request, "points_count", DEFAULT_POINTS, 1, MAX_POINTS)
    seed = integer(request, "seed", DEFAULT_SEED, 0, MAX_SEED)
    noise = number(request, "noise", DEFAULT_NOISE, 0, 1)
    generator = context.generator(seed)
    random = torch.rand(
        (count, 2),
        generator=generator,
        device=context.device,
        dtype=context.dtype,
    )
    angle = random[:, 0] * math.tau
    points = torch.stack((angle.cos(), angle.sin()), dim=-1)
    if kind == "two_circles":
        points[:, 0] += (
            torch.arange(count, device=context.device) % 2
        ) * CIRCLE_SEPARATION
        points[:, 0] -= CIRCLE_SEPARATION / 2
    elif kind == "figure_eight":
        points = torch.stack((angle.sin(), (2 * angle).sin() / 2), dim=-1)
    elif kind == "annulus":
        radius = (
            ANNULUS_INNER_RADIUS**2 + random[:, 1] * (1 - ANNULUS_INNER_RADIUS**2)
        ).sqrt()
        points *= radius[:, None]
    elif kind == "clusters":
        centers = context.tensor(((-1, -1), (1, -1), (0, 1)))
        points = centers[torch.arange(count, device=context.device) % len(centers)]
        points = points + CLUSTER_SPREAD * torch.randn(
            (count, 2),
            generator=generator,
            device=context.device,
            dtype=context.dtype,
        )
    return points + noise * torch.randn(
        (count, 2),
        generator=generator,
        device=context.device,
        dtype=context.dtype,
    )
