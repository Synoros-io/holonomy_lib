"""Stereographic distances, geodesics, and a checked mean estimate."""

from __future__ import annotations

import math

import torch

from holonomy_lib.hyperbolic import frechet_mean
from holonomy_lib.manifolds import KappaStereographicManifold

from .common import (
    CHART_POLE_FRACTION,
    CURVE_SAMPLES,
    DEFAULT_GRID_EXTENT,
    GRID_RAYS,
    GRID_RINGS,
    MAX_CURVATURE_POINTS,
    Context,
    Request,
    number,
    points_tensor,
)

# Residual tolerance is a display diagnostic, scaled to the dtype's resolution.
MEAN_RESIDUAL_EPS_POWER = 0.5
# An initial asymmetric triangle inside every displayed chart at |κ| <= 2.
DEFAULT_CURVATURE_POINTS = ((-0.35, -0.15), (0.35, -0.15), (0.0, 0.4))


@torch.no_grad()
def compute(request: Request) -> Request:
    """Evaluate one chart with a leading singleton scene batch.

    Points outside the chart remain in the response but are excluded from all
    geometry. The library mean uses sequential Karcher iterations; its residual
    is reported because an iteration cap alone does not establish convergence.

    References:
        Bachmann et al. (2020), Constant Curvature Graph Convolutional Networks.
        Karcher (1977), Riemannian center of mass and mollifier smoothing.
    """
    context = Context.from_request(request)
    kappa = number(request, "kappa", -1, -2, 2)
    points = points_tensor(request, context, MAX_CURVATURE_POINTS)
    manifold = KappaStereographicManifold(
        n=2,
        kappa=kappa,
        device=context.device,
        dtype=context.dtype,
    )
    # The negative-curvature ball is open. Use the documented hemispherical
    # chart for positive κ as well, without relying on a membership sign bug.
    valid = (
        points.square().sum(-1) * abs(kappa) < 1
        if kappa != 0
        else torch.ones(len(points), device=context.device, dtype=torch.bool)
    )
    indices = valid.nonzero().flatten()
    included = points[valid]
    distance = context.call(
        manifold.distance,
        included[None, :, None, :],
        included[None, None, :, :],
    )[0]
    raw_pairs = request.get("pairs")
    if raw_pairs is None:
        pairs = torch.combinations(indices, r=2)
    else:
        if not isinstance(raw_pairs, list) or len(raw_pairs) > MAX_CURVATURE_POINTS**2:
            raise ValueError("pairs must be a list of point-index pairs")
        if raw_pairs:
            if any(
                not isinstance(pair, list)
                or len(pair) != 2
                or any(type(index) is not int for index in pair)
                for pair in raw_pairs
            ):
                raise ValueError("each pair must contain two integer indices")
            pairs = torch.tensor(raw_pairs, device=context.device, dtype=torch.long)
            if bool(((pairs < 0) | (pairs >= len(points))).any()):
                raise ValueError("pair index is outside the point list")
            pairs = pairs[valid[pairs].all(-1)]
        else:
            pairs = torch.empty((0, 2), device=context.device, dtype=torch.long)
    steps = torch.linspace(
        0, 1, CURVE_SAMPLES, device=context.device, dtype=context.dtype
    )
    start, end = points[pairs[:, 0]], points[pairs[:, 1]]
    tangent = context.call(manifold.log, start, end)
    curves = context.call(
        manifold.exp, start[:, None, :], tangent[:, None, :] * steps[None, :, None]
    )
    extent = DEFAULT_GRID_EXTENT
    if kappa > 0:
        # exp_0 reaches the hemisphere boundary at tangent radius π/(4√κ).
        extent = min(extent, CHART_POLE_FRACTION * math.pi / (4 * math.sqrt(kappa)))
    angles = torch.linspace(
        0, math.tau, CURVE_SAMPLES, device=context.device, dtype=context.dtype
    )
    radii = torch.arange(1, GRID_RINGS + 1, device=context.device, dtype=context.dtype)
    radii *= extent / GRID_RINGS
    rings = radii[:, None, None] * torch.stack((angles.cos(), angles.sin()), -1)[None]
    ray_angles = (
        torch.arange(GRID_RAYS, device=context.device, dtype=context.dtype)
        * math.tau
        / GRID_RAYS
    )
    rays = (
        extent
        * steps[None, :, None]
        * torch.stack((ray_angles.cos(), ray_angles.sin()), -1)[:, None]
    )
    grid = context.call(manifold.exp_0, torch.cat((rings, rays)))
    mean: Request | None = None
    warnings: list[str] = []
    if kappa <= 0 and len(included):
        try:
            estimate = context.call(frechet_mean, included[None], manifold)
            average_log = context.call(
                manifold.log, estimate[:, None, :], included[None]
            ).mean(1)
            residual = context.call(manifold.norm, estimate, average_log)[0]
            converged = bool(
                residual <= torch.finfo(context.dtype).eps ** MEAN_RESIDUAL_EPS_POWER
            )
            mean = {"point": estimate[0], "residual": residual, "converged": converged}
            if not converged:
                warnings.append(
                    "Fréchet iteration has not converged; "
                    "the marked point is only an estimate."
                )
        except (ValueError, RuntimeError) as error:
            warnings.append(f"Fréchet mean unavailable: {error}")
    return context.finish(
        points=points,
        valid=valid,
        included_indices=indices,
        distances=distance,
        pairs=pairs,
        curves=curves,
        grid=grid,
        mean=mean,
        warnings=warnings,
        boundary_radius=1 / math.sqrt(abs(kappa)) if kappa else None,
        parameters={"kappa": kappa, "curve_samples": CURVE_SAMPLES},
        chart_note=(
            "Open hyperbolic ball"
            if kappa < 0
            else "Conservative hemispherical chart"
            if kappa > 0
            else "Flat chart: distance is 2 × Euclidean coordinate distance"
        ),
    )
