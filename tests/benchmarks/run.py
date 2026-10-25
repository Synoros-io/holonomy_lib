# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Synoros

"""Entry point: run all benchmarks and write results.

Examples:
  uv run python -m tests.benchmarks.run
  uv run python -m tests.benchmarks.run --module algebra
  uv run python -m tests.benchmarks.run --module spd --device cpu
  uv run python -m tests.benchmarks.run --out notes/benchmark_2026-05-27.md
  uv run python -m tests.benchmarks.run --iterations 50
"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path

import torch

from tests.benchmarks.harness import (
    detect_devices,
    write_json,
    write_markdown,
)

MODULES = [
    "bench_algebra",
    "bench_manifolds",
    "bench_hyperbolic",
    "bench_spectral",
    "bench_tensor_calculus",
    "bench_discrete_geometry",
    "bench_info_geometry",
    "bench_topology",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--module", default=None,
                    help="One bench module (without prefix) or 'all'.")
    p.add_argument("--device", default=None,
                    help="One device (cpu / cuda / mps); default auto-detect.")
    p.add_argument("--iterations", type=int, default=20)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--out", default=None,
                    help="Output .md path (default: notes/benchmark_<n>.md).")
    p.add_argument("--out-json", default=None,
                    help="Output .json path (default: alongside --out).")
    p.add_argument("--dtypes", nargs="+", default=["float64", "float32"])
    args = p.parse_args()

    if args.module is None or args.module == "all":
        modules = MODULES
    else:
        modules = [args.module if args.module.startswith("bench_")
                    else f"bench_{args.module}"]

    dtypes = [getattr(torch, d) for d in args.dtypes]
    devices = detect_devices(args.device)

    print(f"Running benchmarks on devices={devices}, dtypes={args.dtypes}, "
           f"iterations={args.iterations}")

    all_results = []
    for mod_name in modules:
        print(f"  • {mod_name}")
        mod = importlib.import_module(f"tests.benchmarks.{mod_name}")
        bench = mod.bench
        results = bench.run(
            devices=devices, dtypes=dtypes,
            iterations=args.iterations, warmup=args.warmup,
        )
        all_results.extend(results)

    out_md = Path(args.out) if args.out else Path(
        "notes/benchmark_latest.md",
    )
    out_json = Path(args.out_json) if args.out_json else out_md.with_suffix(".json")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    write_markdown(all_results, out_md)
    write_json(all_results, out_json)
    print(f"\nWrote {out_md}\n     {out_json}")


if __name__ == "__main__":
    main()
