# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Synoros

"""Benchmarks for holonomy_lib.provenance overhead.

Per-call overhead of the @with_provenance decorator (inspect.signature,
content-hash, JSON canon) shows up on every primitive call inside a
record() context. We measure with a trivial op so the timing reflects
the decorator, not the op.

Three groups of cases:
1. Decorator overhead at small sizes — outside record, inside record,
   chained inside record. Establishes the per-call cost floor.
2. Hashing scale — recording an op whose input is a 2D float64 tensor
   from 16² (2 KB) up to 1024² (8 MB). Isolates the bytes-hash cost
   inside `_tensor_content_hex`; phase 1b sketch-hashing should crush
   this curve.
3. Cache + replay — `cache_tensors=True` overhead and replay vs
   direct-call cost. Phase 1c (disk cache) and phase 2a (replay
   completion) re-measure these.
"""

from __future__ import annotations

import tempfile

import torch

from holonomy_lib import provenance
from holonomy_lib.provenance import with_provenance
from holonomy_lib.spectral import laplacian
from tests.benchmarks.harness import Bench

bench = Bench("provenance")


# Trivial op so the timing reflects the decorator, not the math.
@with_provenance("bench.identity", op_version="0.1")
def _identity(x: torch.Tensor) -> torch.Tensor:
    return x


@with_provenance("bench.double", op_version="0.1")
def _double(x: torch.Tensor) -> torch.Tensor:
    return x * 2


def _setup_decorator_outside_record(size, device, dtype):
    g = torch.Generator(device="cpu"); g.manual_seed(0)
    x = torch.randn(size["B"], size["n"], generator=g, dtype=dtype).to(device)
    def fn():
        return _identity(x)
    return fn


def _setup_decorator_inside_record(size, device, dtype):
    g = torch.Generator(device="cpu"); g.manual_seed(0)
    x = torch.randn(size["B"], size["n"], generator=g, dtype=dtype).to(device)
    def fn():
        with provenance.record() as _:
            _identity(x)
    return fn


def _setup_chained_inside_record(size, device, dtype):
    """combinatorial → identity chain — exercises tensor chaining."""
    n = size["n"]
    g = torch.Generator(device="cpu"); g.manual_seed(0)
    A = torch.rand(size["B"], n, n, generator=g, dtype=dtype)
    A = (A + A.mT) * 0.5
    A = A.to(device)
    def fn():
        with provenance.record() as _:
            L = laplacian.combinatorial(A)
            _identity(L)
    return fn


# ----- group 2: hashing scale -----

def _setup_hashing_big_tensor(size, device, dtype):
    """Records identity on a fresh 2D tensor of shape (n, n).

    Each fn() call recreates a fresh tensor so its content hash isn't
    short-circuited by the per-tensor id cache. This is the workload
    where _tensor_content_hex byte-hashing dominates.
    """
    n = size["n"]
    def fn():
        # Build a fresh tensor inside the timed iteration so the cache
        # doesn't short-circuit hashing. NOTE: this includes a small
        # alloc cost, but that's constant across the size sweep so the
        # delta is still meaningful.
        x = torch.zeros(n, n, dtype=dtype, device=device)
        with provenance.record() as _:
            _identity(x)
    return fn


# ----- group 3: cache + replay -----

def _setup_recording_with_cache(size, device, dtype):
    """Recording a 2-op chain with cache_tensors=True.

    Measures the cache_put overhead per output. At larger sizes, the
    eventual disk-backed cache (phase 1c) re-uses this case to compare
    in-memory vs on-disk costs.
    """
    n = size["n"]
    g = torch.Generator(device="cpu"); g.manual_seed(0)
    A = torch.rand(size["B"], n, n, generator=g, dtype=dtype)
    A = (A + A.mT) * 0.5
    A = A.to(device)
    def fn():
        with provenance.record(cache_tensors=True) as _:
            L = laplacian.combinatorial(A)
            _identity(L)
    return fn


def _setup_replay_overhead(size, device, dtype):
    """Time replay() on a 2-op chain after substituting the middle node.

    Setup runs the chain once to populate the registry, then each timed
    fn() substitutes the laplacian output with a fresh tensor of matching
    shape — this forces replay() to walk the DAG and re-execute the
    downstream `_double` node. Measures DAG-traversal + topological-sort
    + single-op dispatch overhead.
    """
    n = size["n"]
    g = torch.Generator(device="cpu"); g.manual_seed(0)
    A = torch.rand(size["B"], n, n, generator=g, dtype=dtype)
    A = (A + A.mT) * 0.5
    A = A.to(device)
    with provenance.record(cache_tensors=True) as reg:
        L = laplacian.combinatorial(A)
        _double(L)
    laplacian_hex = reg.where(op_id="holonomy_lib.spectral.laplacian.combinatorial")[0].hex
    substitute_tensor = torch.zeros_like(L)
    def fn():
        reg.replay({laplacian_hex: substitute_tensor})
    return fn


def _setup_replay_baseline(size, device, dtype):
    """Direct _double call on a cached laplacian — replay's lower bound."""
    n = size["n"]
    g = torch.Generator(device="cpu"); g.manual_seed(0)
    A = torch.rand(size["B"], n, n, generator=g, dtype=dtype)
    A = (A + A.mT) * 0.5
    A = A.to(device)
    L = laplacian.combinatorial(A)
    substitute_tensor = torch.zeros_like(L)
    def fn():
        # The math replay actually re-runs at the affected step.
        _double(substitute_tensor)
    return fn


# ----- group 4: phase-1 modes (sketch, disk cache) -----

def _setup_hashing_big_tensor_sketch(size, device, dtype):
    """Sketch-mode variant of hashing_big_tensor.

    Compare against the full-mode case at the same sizes to see the
    sketch-vs-full speedup. Sketch wins above ~256² float64.
    """
    n = size["n"]
    def fn():
        x = torch.zeros(n, n, dtype=dtype, device=device)
        with provenance.record(hash_mode="sketch") as _:
            _identity(x)
    return fn


def _setup_recording_with_disk_cache(size, device, dtype):
    """Recording with cache_to_disk set: each output is torch.save'd.

    Uses a single shared tempdir for the bench run. Files accumulate
    across iterations — the I/O cost shown is the per-call write, not
    a steady-state where the LRU is in play.
    """
    n = size["n"]
    g = torch.Generator(device="cpu"); g.manual_seed(0)
    A = torch.rand(size["B"], n, n, generator=g, dtype=dtype)
    A = (A + A.mT) * 0.5
    A = A.to(device)
    tmpdir = tempfile.mkdtemp(prefix="bench_disk_cache_")
    def fn():
        with provenance.record(cache_to_disk=tmpdir) as _:
            L = laplacian.combinatorial(A)
            _identity(L)
    return fn


_sizes = [
    {"B": 1,  "n": 16},
    {"B": 1,  "n": 128},
    {"B": 16, "n": 16},
]

# Bigger sizes for the hashing / cache / replay sweep. Pure-2D shape
# (no batch dim) so the totals are easy to read in bytes:
#   n=16 → 2 KiB ; n=256 → 0.5 MiB ; n=1024 → 8 MiB at float64.
_big_sizes = [
    {"B": 1, "n": 16},
    {"B": 1, "n": 256},
    {"B": 1, "n": 1024},
]


bench.case("decorator_outside_record", _setup_decorator_outside_record, _sizes,
            notes="Decorator no-op path — should be one attribute lookup.")
bench.case("decorator_inside_record", _setup_decorator_inside_record, _sizes,
            notes="Full provenance overhead per call: bind args, hash, register.")
bench.case("chained_inside_record", _setup_chained_inside_record, _sizes,
            notes="Two-op chain — exercises tensor-id → hex lookup.")
bench.case("hashing_big_tensor", _setup_hashing_big_tensor, _big_sizes,
            notes="Fresh-tensor identity, isolates _tensor_content_hex byte cost. "
                  "Phase 1b sketch-hashing should flatten this curve.")
bench.case("recording_with_cache", _setup_recording_with_cache, _big_sizes,
            notes="laplacian → identity chain with cache_tensors=True. "
                  "Baseline for phase 1c disk-cache comparison.")
bench.case("replay_overhead", _setup_replay_overhead, _big_sizes,
            notes="replay() of a 2-op chain after substituting the middle "
                  "node — DAG walk + topo-sort + single-op dispatch.")
bench.case("replay_baseline", _setup_replay_baseline, _big_sizes,
            notes="Direct call to the affected op with the substituted "
                  "tensor. The math floor that replay overhead sits on.")
bench.case("hashing_big_tensor_sketch", _setup_hashing_big_tensor_sketch, _big_sizes,
            notes="Sketch-mode hashing — compare against hashing_big_tensor "
                  "(full mode) at matching sizes; crossover ~n=256.")
bench.case("recording_with_disk_cache", _setup_recording_with_disk_cache, _big_sizes,
            notes="Same chain as recording_with_cache but with cache_to_disk "
                  "set — measures torch.save overhead per output.")
