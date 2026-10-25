"""Request validation, compute timing, and the JSON boundary."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from inspect import signature
from time import perf_counter
from typing import Any, TypeVar

import torch

# These are UI resource budgets, not numerical accuracy claims. They bound dense
# graph work, cubic VR simplex counts, and repeated sweeps on a laptop CPU.
MAX_NODES = 64
MAX_POINTS = 40
MAX_CURVATURE_POINTS = 12
MAX_BATCH_SIZE = 256
MAX_PERSISTENCE_BATCH = 16
MAX_PERSISTENCE_SIZE = 16
MAX_REPEATS = 7
MAX_FLOW_STEPS = 24
MAX_FLOW_FRAMES = 8
MAX_REQUEST_BYTES = 256 * 1024
REQUEST_TIMEOUT_SECONDS = 20

# Presentation and sampling defaults are choices for a small first view.
DEFAULT_SEED = 0
DEFAULT_NODES = 24
DEFAULT_POINTS = 24
DEFAULT_NOISE = 0.05
DEFAULT_REPEATS = 3
DEFAULT_SWEEP = (1, 4, 16, 64, 256)
CURVE_SAMPLES = 65
GRID_RINGS = 5
GRID_RAYS = 12
# Stay away from the positive-curvature chart pole when sampling the grid.
CHART_POLE_FRACTION = 0.8
DEFAULT_GRID_EXTENT = 1.0
SECONDS_TO_MS = 1000
DTYPES = {"float32": torch.float32, "float64": torch.float64}
Request = dict[str, Any]
T = TypeVar("T")


def integer(request: Request, key: str, default: int, low: int, high: int) -> int:
    """Read an integer without silently truncating fractional input."""
    value = request.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    if not low <= value <= high:
        raise ValueError(f"{key} must be between {low} and {high}")
    return value


def number(
    request: Request, key: str, default: float, low: float, high: float
) -> float:
    """Read a finite scalar within the control's documented domain."""
    value = request.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"{key} must be a number")
    value = float(value)
    if not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{key} must be finite and between {low} and {high}")
    return value


def choice(request: Request, key: str, default: str, options: tuple[str, ...]) -> str:
    """Read a named option, rejecting unexpected routes through the adapter."""
    value = request.get(key, default)
    if value not in options:
        raise ValueError(f"{key} must be one of {', '.join(options)}")
    return str(value)


def json_safe(value: Any) -> Any:
    """Detach tensors once at the boundary; JSON has no NaN or infinity."""
    if isinstance(value, torch.Tensor):
        return json_safe(value.detach().cpu().tolist())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


@dataclass
class Context:
    """Keep chosen tensor placement and synchronized library-call time together."""

    device: torch.device
    dtype: torch.dtype
    elapsed_ms: float = 0.0
    calls: list[Request] = field(default_factory=list)

    @classmethod
    def from_request(cls, request: Request) -> Context:
        dtype_name = choice(request, "dtype", "float64", tuple(DTYPES))
        device = torch.device(str(request.get("device", "cpu")))
        if device.type not in ("cpu", "cuda"):
            raise ValueError("device must be cpu or cuda (CUDA also names ROCm)")
        if device.type == "cpu":
            device = torch.device("cpu")
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise ValueError(
                    "CUDA/ROCm is not available in this Python environment"
                )
            device = torch.device("cuda", device.index or 0)
        # Allocate now so an invalid GPU index fails before any computation.
        torch.empty(0, device=device, dtype=DTYPES[dtype_name])
        return cls(device, DTYPES[dtype_name])

    def sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def call(self, function: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Time only the requested library operation, including device completion."""
        self.sync()
        start = perf_counter()
        try:
            return function(*args, **kwargs)
        finally:
            self.sync()
            duration = (perf_counter() - start) * SECONDS_TO_MS
            self.elapsed_ms += duration
            self.calls.append(
                {
                    "function": f"{function.__module__}.{function.__qualname__}",
                    "elapsed_ms": duration,
                    "device": str(self.device),
                    "parameters": _scalar_arguments(function, args, kwargs),
                }
            )

    def tensor(self, data: Any) -> torch.Tensor:
        return torch.as_tensor(data, device=self.device, dtype=self.dtype)

    def generator(self, seed: int) -> torch.Generator:
        return torch.Generator(device=self.device).manual_seed(seed)

    def finish(self, **values: Any) -> Request:
        return dict(
            json_safe(values),
            device=str(self.device),
            dtype=str(self.dtype).removeprefix("torch."),
            elapsed_ms=self.elapsed_ms,
            calls=json_safe(self.calls),
        )


def _scalar_arguments(
    function: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: Request,
) -> Request:
    # Read defaults from the dependency instead of copying them into the UI.
    # Some torch C builtins have no Python signature; their inputs are already
    # described by the scene's parameters and data arrays.
    try:
        bound = signature(function).bind(*args, **kwargs)
    except (ValueError, TypeError):
        return {}
    bound.apply_defaults()
    return {
        key: value
        for key, value in bound.arguments.items()
        if isinstance(value, (str, int, float, bool)) or value is None
    }


def points_tensor(request: Request, context: Context, cap: int) -> torch.Tensor:
    """Read a finite (N, 2) point array; empty clicks are a valid cloud."""
    raw = request.get("points", [])
    if not isinstance(raw, list) or len(raw) > cap:
        raise ValueError(f"points must be a list containing at most {cap} points")
    if not raw:
        return context.tensor([]).reshape(0, 2)
    points = context.tensor(raw)
    if points.shape != (len(raw), 2) or not bool(torch.isfinite(points).all()):
        raise ValueError("points must contain finite [x, y] pairs")
    return points
