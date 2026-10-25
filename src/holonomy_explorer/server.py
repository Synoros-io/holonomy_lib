"""A small standard-library HTTP transport; computation lives in scene modules."""

from __future__ import annotations

import json
import sys
import threading
from collections.abc import Callable
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from typing import Any
from urllib.parse import urlsplit

import torch

from . import batching, common, curvature, datasets, graphs, topology
from .common import Request, json_safe

ROUTES: dict[str, Callable[[Request], Request]] = {
    "graphs": graphs.compute,
    "curvature": curvature.compute,
    "topology": topology.compute,
    "batching": batching.compute,
}
CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
}


def configuration(request: Request) -> Request:
    """Expose limits and defaults so the browser does not invent compute values."""
    context = common.Context.from_request(request)
    return context.finish(
        cuda_available=torch.cuda.is_available(),
        limits={
            "nodes": common.MAX_NODES,
            "points": common.MAX_POINTS,
            "curvature_points": common.MAX_CURVATURE_POINTS,
            "batch": common.MAX_BATCH_SIZE,
            "persistence_batch": common.MAX_PERSISTENCE_BATCH,
            "persistence_size": common.MAX_PERSISTENCE_SIZE,
            "repeats": common.MAX_REPEATS,
            "flow_steps": common.MAX_FLOW_STEPS,
            "flow_frames": common.MAX_FLOW_FRAMES,
        },
        defaults={
            "nodes": common.DEFAULT_NODES,
            "points_count": common.DEFAULT_POINTS,
            "seed": common.DEFAULT_SEED,
            "noise": common.DEFAULT_NOISE,
            "blocks": datasets.DEFAULT_BLOCKS,
            "p_in": datasets.DEFAULT_P_IN,
            "p_out": datasets.DEFAULT_P_OUT,
            "radius": datasets.DEFAULT_RADIUS,
            "clique_size": datasets.DEFAULT_CLIQUE_SIZE,
            "cliques": datasets.DEFAULT_CLIQUES,
            "steps": graphs.DEFAULT_FLOW_STEPS,
            "surgery_period": graphs.DEFAULT_SURGERY_PERIOD,
            "surgery_threshold": graphs.DEFAULT_SURGERY_THRESHOLD,
            "dt": graphs.DEFAULT_FLOW_DT,
            "repeats": common.DEFAULT_REPEATS,
            "size": batching.DEFAULT_PROBLEM_SIZE,
            "batches": common.DEFAULT_SWEEP,
        },
        heat_time_range=[graphs.MIN_HEAT_TIME, graphs.MAX_HEAT_TIME],
        max_seed=datasets.MAX_SEED,
        max_surgery_threshold=graphs.MAX_SURGERY_THRESHOLD,
        curvature_points=curvature.DEFAULT_CURVATURE_POINTS,
    )


def _reject_constant(value: str) -> Any:
    raise ValueError(f"JSON must not contain {value}")


def create_server(host: str, port: int, device: str, dtype: str) -> ThreadingHTTPServer:
    """Build the server without starting it, for CLI and embedded local use."""
    defaults = {"device": device, "dtype": dtype}
    compute_lock = threading.Lock()
    static = resources.files("holonomy_explorer").joinpath("static")

    class Handler(BaseHTTPRequestHandler):
        def setup(self) -> None:
            super().setup()
            self.connection.settimeout(common.REQUEST_TIMEOUT_SECONDS)

        def _send(self, status: HTTPStatus, payload: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cache-Control", "no-store")
            self.send_header(
                "Content-Security-Policy",
                (
                    "default-src 'self'; script-src 'self'; style-src 'self'; "
                    "connect-src 'self'; img-src 'self'; object-src 'none'; "
                    "base-uri 'none'; frame-ancestors 'none'"
                ),
            )
            self.end_headers()
            self.wfile.write(payload)

        def _json(self, status: HTTPStatus, payload: Request) -> None:
            encoded = json.dumps(json_safe(payload), allow_nan=False).encode("utf-8")
            self._send(status, encoded, "application/json; charset=utf-8")

        def do_GET(self) -> None:
            path = urlsplit(self.path).path
            name = "index.html" if path == "/" else path.removeprefix("/static/")
            if path != "/" and not path.startswith("/static/"):
                self._json(HTTPStatus.BAD_REQUEST, {"error": "unknown resource"})
                return
            # Only flat package assets can be served, never arbitrary filesystem paths.
            if not name or "/" in name or "\\" in name or name.startswith("."):
                self._json(HTTPStatus.BAD_REQUEST, {"error": "invalid resource path"})
                return
            asset = static.joinpath(name)
            suffix = "." + name.rsplit(".", 1)[-1]
            if not asset.is_file() or suffix not in CONTENT_TYPES:
                self._json(HTTPStatus.BAD_REQUEST, {"error": "unknown resource"})
                return
            self._send(HTTPStatus.OK, asset.read_bytes(), CONTENT_TYPES[suffix])

        def do_POST(self) -> None:
            try:
                if self.headers.get_content_type() != "application/json":
                    raise ValueError("Content-Type must be application/json")
                origin = self.headers.get("Origin")
                if origin and urlsplit(origin).netloc != self.headers.get("Host"):
                    raise ValueError("cross-origin requests are not accepted")
                if self.headers.get("Transfer-Encoding"):
                    raise ValueError("chunked request bodies are not supported")
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= common.MAX_REQUEST_BYTES:
                    raise ValueError(
                        f"body must contain 1 to {common.MAX_REQUEST_BYTES} bytes"
                    )
                raw = self.rfile.read(length)
                if len(raw) != length:
                    raise ValueError("incomplete request body")
                request = json.loads(raw, parse_constant=_reject_constant)
                if not isinstance(request, dict):
                    raise ValueError("request must be a JSON object")
                route = urlsplit(self.path).path
                if not route.startswith("/api/"):
                    raise ValueError("unknown API route")
                name = route.removeprefix("/api/")
                function = configuration if name == "config" else ROUTES.get(name)
                if function is None:
                    raise ValueError("unknown API route")
                if not compute_lock.acquire(blocking=False):
                    raise ValueError(
                        "another computation is running; try again when it finishes"
                    )
                try:
                    response = function({**defaults, **request})
                finally:
                    compute_lock.release()
                self._json(HTTPStatus.OK, response)
            except Exception as error:
                # Library exceptions are visible beside the triggering control.
                # HTTP/JSON parsing errors follow the same small response contract.
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})

        def log_message(self, format: str, *args: Any) -> None:
            print(f"explorer: {format % args}", file=sys.stderr)

    return ThreadingHTTPServer((host, port), Handler)
