# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Synoros

"""Microbenchmark harness — device-agnostic, deterministic, lightweight.

Each benchmark module imports `Bench` and registers cases via
`Bench.case(name, fn, *, sizes, devices, dtypes)`. Running the harness
times each case at each (size, device, dtype) combination, takes the
median of several timed iterations, and writes a JSON + Markdown report.

Why not pytest-benchmark? pytest-benchmark is great but conflates "is
this test green?" with "is this primitive fast?" — we want the latter
to be reportable independently, and we want median-of-N timings with
explicit warmup so noisy timings don't dominate. This harness is
~100 lines and exactly what we need.

Usage:
  uv run python -m tests.benchmarks.run               # all benchmarks
  uv run python -m tests.benchmarks.run --module spd  # one module
  uv run python -m tests.benchmarks.run --device cuda # GPU only
  uv run python -m tests.benchmarks.run --out path.md # custom output
"""

from __future__ import annotations

import json
import statistics
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch


@dataclass
class BenchResult:
    name: str
    size: dict[str, Any]          # named dims, e.g. {"B": 1, "n": 64, "r": 8}
    device: str                    # "cpu" / "cuda" / "cuda:0" / "mps"
    dtype: str                     # "float32" / "float64"
    median_seconds: float
    min_seconds: float
    iterations: int                # number of timed iterations
    notes: str = ""


@dataclass
class _Case:
    name: str
    fn: Callable[..., Any]         # zero-arg callable that runs ONE iteration
    setup: Callable[[dict, str, torch.dtype], Callable[..., Any]] | None = None
    sizes: list[dict[str, Any]] = field(default_factory=list)
    notes: str = ""


class Bench:
    """Registry of benchmark cases for one module."""

    def __init__(self, module_name: str):
        self.module_name: str = module_name
        self._cases: list[_Case] = []

    def case(
        self,
        name: str,
        setup: Callable[[dict, str, torch.dtype], Callable[..., Any]],
        sizes: list[dict[str, Any]],
        notes: str = "",
    ) -> None:
        """Register a case.

        Args:
          name: human-readable benchmark name.
          setup: function `setup(size, device, dtype) -> fn` returning a
            zero-arg callable that performs ONE iteration. Inputs should
            be constructed inside setup so each `fn()` call is timed for
            the operation only, not for the allocations.
          sizes: list of size dicts to sweep. The setup() function gets
            one element of this list per benchmark run.
          notes: free-form notes about what the case measures.
        """
        self._cases.append(_Case(name=name, fn=lambda: None,
                                  setup=setup, sizes=sizes, notes=notes))

    def run(
        self,
        devices: Iterable[str],
        dtypes: Iterable[torch.dtype],
        iterations: int = 20,
        warmup: int = 3,
    ) -> list[BenchResult]:
        """Run every (case, size, device, dtype) combination."""
        results: list[BenchResult] = []
        for case in self._cases:
            for size in case.sizes:
                for device in devices:
                    for dtype in dtypes:
                        try:
                            fn = case.setup(size, device, dtype)
                        except Exception as e:
                            # Some combinations are invalid (e.g. CUDA on a
                            # CPU-only build, or an op that doesn't run at
                            # this dtype). Record the skip and continue.
                            results.append(BenchResult(
                                name=f"{self.module_name}.{case.name}",
                                size=size, device=device, dtype=str(dtype),
                                median_seconds=float("nan"),
                                min_seconds=float("nan"),
                                iterations=0,
                                notes=f"setup failed: {type(e).__name__}: {e}",
                            ))
                            continue
                        # Warmup — JIT-compile, allocator-prime, etc.
                        for _ in range(warmup):
                            fn()
                        if device.startswith("cuda"):
                            torch.cuda.synchronize()
                        # Time
                        per_iter: list[float] = []
                        for _ in range(iterations):
                            if device.startswith("cuda"):
                                torch.cuda.synchronize()
                            t0 = time.perf_counter()
                            fn()
                            if device.startswith("cuda"):
                                torch.cuda.synchronize()
                            per_iter.append(time.perf_counter() - t0)
                        results.append(BenchResult(
                            name=f"{self.module_name}.{case.name}",
                            size=size, device=device, dtype=str(dtype),
                            median_seconds=statistics.median(per_iter),
                            min_seconds=min(per_iter),
                            iterations=iterations,
                            notes=case.notes,
                        ))
        return results


def write_markdown(results: list[BenchResult], path: str | Path) -> None:
    """Write results as a Markdown table grouped by benchmark name."""
    lines: list[str] = []
    lines.append("# Benchmark results\n")
    lines.append(f"_torch {torch.__version__}, "
                  f"CUDA available: {torch.cuda.is_available()}_\n")
    by_name: dict[str, list[BenchResult]] = {}
    for r in results:
        by_name.setdefault(r.name, []).append(r)
    for name, rows in sorted(by_name.items()):
        lines.append(f"\n## `{name}`\n")
        if rows[0].notes:
            lines.append(f"_{rows[0].notes}_\n")
        lines.append("| size | device | dtype | median (ms) | min (ms) | iters |")
        lines.append("|---|---|---|---:|---:|---:|")
        for r in rows:
            size_s = " ".join(f"{k}={v}" for k, v in r.size.items())
            if r.median_seconds != r.median_seconds:  # NaN
                lines.append(
                    f"| {size_s} | {r.device} | {r.dtype} "
                    f"| — | — | 0 (skipped: {r.notes}) |"
                )
            else:
                lines.append(
                    f"| {size_s} | {r.device} | {r.dtype} "
                    f"| {r.median_seconds*1000:.3f} "
                    f"| {r.min_seconds*1000:.3f} | {r.iterations} |"
                )
    Path(path).write_text("\n".join(lines))


def write_json(results: list[BenchResult], path: str | Path) -> None:
    """Write results as a JSON array suitable for diffing across runs."""
    payload = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "results": [
            {
                "name": r.name, "size": r.size, "device": r.device,
                "dtype": r.dtype, "median_seconds": r.median_seconds,
                "min_seconds": r.min_seconds, "iterations": r.iterations,
                "notes": r.notes,
            }
            for r in results
        ],
    }
    Path(path).write_text(json.dumps(payload, indent=2))


def detect_devices(explicit: str | None = None) -> list[str]:
    """Pick which devices to benchmark on."""
    if explicit is not None:
        return [explicit]
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        devices.append("mps")
    return devices
