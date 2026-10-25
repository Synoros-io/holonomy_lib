# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Synoros

"""Microbenchmark harness with device-aware timing and failure records.

Each benchmark module imports `Bench` and registers cases via
`Bench.case(name, setup, sizes, notes)`. Running the harness
times each case at each (size, device, dtype) combination, takes the
median of several timed iterations, and writes a JSON + Markdown report.

Why not pytest-benchmark? pytest-benchmark is great but conflates "is
this test green?" with "is this primitive fast?"; we want the latter
to be reportable independently, and we want median-of-N timings with
explicit warmup so noisy timings don't dominate. This harness is
independent of pytest collection.

Usage:
  uv run python -m tests.benchmarks.run               # all benchmarks
  uv run python -m tests.benchmarks.run --module manifolds
  uv run python -m tests.benchmarks.run --device cuda # GPU only
  uv run python -m tests.benchmarks.run --out path.md # custom output
"""

from __future__ import annotations

import json
import statistics
import time
import warnings
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

# Preserve the original harness sampling budget; these are timing defaults,
# not accuracy guarantees. Both can be overridden by the caller or CLI.
DEFAULT_ITERATIONS = 20
DEFAULT_WARMUP = 3
# SI unit conversion for the printed report; JSON retains seconds.
MILLISECONDS_PER_SECOND = 1000


@dataclass
class BenchResult:
    name: str
    size: dict[str, Any]          # named dims, e.g. {"B": 1, "n": 64, "r": 8}
    device: str                    # "cpu" / "cuda" / "cuda:0" / "mps"
    dtype: str                     # "float32" / "float64"
    median_seconds: float | None
    min_seconds: float | None
    iterations: int                # number of timed iterations
    notes: str = ""


@dataclass
class _Case:
    name: str
    setup: Callable[[dict[str, Any], str, torch.dtype], Callable[[], Any]]
    sizes: list[dict[str, Any]]
    notes: str = ""


def _synchronize(device: str) -> None:
    """Wait for work on the device whose operation is being timed."""
    target = torch.device(device)
    if target.type == "cuda":
        torch.cuda.synchronize(target)
    elif target.type == "mps":
        torch.mps.synchronize()


class Bench:
    """Registry of benchmark cases for one module."""

    def __init__(self, module_name: str):
        self.module_name: str = module_name
        self._cases: list[_Case] = []

    def case(
        self,
        name: str,
        setup: Callable[[dict[str, Any], str, torch.dtype], Callable[[], Any]],
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
          sizes: list of size dicts to sweep, smallest first. The setup()
            function gets one element of this list per benchmark run.
          notes: free-form notes about what the case measures.
        """
        self._cases.append(_Case(name=name, setup=setup, sizes=sizes, notes=notes))

    def run(
        self,
        devices: Iterable[str],
        dtypes: Iterable[torch.dtype],
        iterations: int = DEFAULT_ITERATIONS,
        warmup: int = DEFAULT_WARMUP,
        smallest: bool = False,
    ) -> list[BenchResult]:
        """Run the sweep, or only the first (smallest) size of each case.

        Failed setup, warmup or timing produces null measurements with the
        failure reason in notes. Warnings are retained there and displayed
        after each measurement, outside the timed interval.
        """
        if iterations <= 0:
            raise ValueError("iterations must be positive")
        if warmup < 0:
            raise ValueError("warmup must be nonnegative")
        devices = tuple(devices)
        dtypes = tuple(dtypes)
        results: list[BenchResult] = []
        for case in self._cases:
            for size in case.sizes[:1] if smallest else case.sizes:
                for device in devices:
                    for dtype in dtypes:
                        notes = [case.notes] if case.notes else []
                        per_iter: list[float] = []
                        phase = "setup"
                        with warnings.catch_warnings(record=True) as captured:
                            try:
                                fn = case.setup(size, device, dtype)
                                phase = "warmup"
                                for _ in range(warmup):
                                    fn()
                                _synchronize(device)
                                phase = "timing"
                                for _ in range(iterations):
                                    _synchronize(device)
                                    t0 = time.perf_counter()
                                    fn()
                                    _synchronize(device)
                                    per_iter.append(time.perf_counter() - t0)
                            except Exception as error:
                                # Partial timings are not a completed measurement.
                                per_iter.clear()
                                notes.append(
                                    f"{phase} failed: {type(error).__name__}: {error}"
                                )
                        for warning in captured:
                            warning_note = (
                                f"{warning.category.__name__}: {warning.message}"
                            )
                            if warning_note not in notes:
                                notes.append(warning_note)
                            warnings.showwarning(
                                warning.message, warning.category,
                                warning.filename, warning.lineno, line=warning.line,
                            )
                        results.append(BenchResult(
                            name=f"{self.module_name}.{case.name}",
                            size=size, device=device, dtype=str(dtype),
                            median_seconds=statistics.median(per_iter)
                            if per_iter else None,
                            min_seconds=min(per_iter) if per_iter else None,
                            iterations=len(per_iter),
                            notes="; ".join(notes),
                        ))
        return results


def format_results_table(results: list[BenchResult]) -> str:
    """Format every measurement and its notes for stdout or Markdown."""
    lines = [
        "| name | size | device | dtype | median (ms) | min (ms) | iters | notes |",
        "|---|---|---|---|---:|---:|---:|---|",
    ]
    for result in results:
        size = " ".join(f"{key}={value}" for key, value in result.size.items())
        notes = result.notes.replace("|", "\\|").replace("\n", " ")
        median = (
            f"{result.median_seconds * MILLISECONDS_PER_SECOND:.3f}"
            if result.median_seconds is not None else "n/a"
        )
        minimum = (
            f"{result.min_seconds * MILLISECONDS_PER_SECOND:.3f}"
            if result.min_seconds is not None else "n/a"
        )
        lines.append(
            f"| {result.name} | {size} | {result.device} | {result.dtype} "
            f"| {median} | {minimum} | {result.iterations} | {notes} |"
        )
    return "\n".join(lines)


def write_markdown(results: list[BenchResult], path: str | Path) -> None:
    """Write every measurement with its own failure or warning notes."""
    header = (
        f"# Benchmark results\n\n_torch {torch.__version__}, "
        f"CUDA available: {torch.cuda.is_available()}_\n\n"
    )
    Path(path).write_text(header + format_results_table(results) + "\n")


def write_json(results: list[BenchResult], path: str | Path) -> None:
    """Write strict JSON with null measurements for unavailable results."""
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
    Path(path).write_text(json.dumps(payload, indent=2, allow_nan=False))


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
