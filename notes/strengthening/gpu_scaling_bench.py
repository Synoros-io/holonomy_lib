"""Same-machine CPU-vs-CUDA batch scaling of the H^n heat kernel.

Run on a commodity GPU (RTX 3060) to show where GPU evaluation overtakes
CPU as batch grows, for the closed-form paths (n=5, 7) and a recursion
path (n=9). Same-machine comparison (both devices on one host) so the
CPU/GPU ratio is meaningful.

Run (on the instance):
  PYTHONPATH=src <python with a GPU torch build> notes/strengthening/gpu_scaling_bench.py
"""
from __future__ import annotations

import time

import torch

from holonomy_lib.hyperbolic import hyperbolic_heat_kernel
from holonomy_lib.manifolds import LorentzManifold

DT = torch.float64


def bench(device: str, n: int, batch: int, iters: int = 15, warmup: int = 3):
    mfd = LorentzManifold(n=n, device=device, dtype=DT)
    t = torch.tensor(0.5, dtype=DT, device=device)
    try:
        for _ in range(warmup):
            d = torch.linspace(0.1, 3.0, batch, dtype=DT, device=device).requires_grad_(True)
            hyperbolic_heat_kernel(t, d, mfd).sum().backward()
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            d = torch.linspace(0.1, 3.0, batch, dtype=DT, device=device).requires_grad_(True)
            hyperbolic_heat_kernel(t, d, mfd).sum().backward()
        if device == "cuda":
            torch.cuda.synchronize()
        return (time.perf_counter() - t0) / iters * 1e3  # ms
    except RuntimeError as e:
        return float("nan") if "out of memory" in str(e).lower() else None


def main():
    print(f"device: {torch.cuda.get_device_name(0)}, torch {torch.__version__}")
    print(f"{'n':>2} {'batch':>9} {'cpu ms':>10} {'cuda ms':>10} {'cpu/cuda':>9}")
    print("-" * 46)
    for n in (5, 7, 9):
        for batch in (4096, 65536, 262144, 1048576):
            c = bench("cpu", n, batch)
            g = bench("cuda", n, batch)
            ratio = (c / g) if (c and g and g == g) else float("nan")
            print(f"{n:>2} {batch:>9} {c:>10.2f} {g:>10.2f} {ratio:>8.1f}x")
        print()


if __name__ == "__main__":
    main()
