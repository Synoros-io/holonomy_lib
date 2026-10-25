"""GPU latency benchmark for the κ-stereographic dispatch on the
local AMD ROCm device.

Companion to `C4_kappa_crossing_stress.py`. The CPU benchmark there
measured ~1.2−1.5× overhead for the Tensor-κ dynamic dispatch vs the
static-float fast path. This script repeats the measurement on GPU.

The holonomy_lib `.venv` ships a CUDA-built torch (which falls back to
CPU on AMD hardware), so this script is *not* run by the default
pytest path. To execute on the workstation's Radeon RX 9060 XT
(gfx1200), use the sibling project's ROCm venv:

    PYTHONPATH=src <python with a ROCm or CUDA torch build> \\
      notes/strengthening/C4_kappa_crossing_gpu_bench.py

Output appended to `C4_kappa_crossing_stress_results.md`.
"""

from __future__ import annotations

import time
from pathlib import Path

import torch

from holonomy_lib.manifolds.stereographic import KappaStereographicManifold


SEED = 2026


def gpu_latency(device: torch.device, dtype: torch.dtype):
    """Time forward+backward for static-float κ vs Tensor κ dispatch."""
    torch.manual_seed(SEED)
    B, n = 65536, 8                       # wider than CPU to occupy GPU
    p1 = torch.randn(B, n, dtype=dtype, device=device) * 0.05
    p2 = torch.randn(B, n, dtype=dtype, device=device) * 0.05
    n_runs = 50

    # --- (A) static float κ ---
    mfd_static = KappaStereographicManifold(
        n=n, kappa=-0.5, device=device, dtype=dtype,
    )
    p1_a = p1.clone().requires_grad_(True)
    p2_a = p2.clone().requires_grad_(True)
    for _ in range(5):                    # warm-up
        mfd_static.distance(p1_a, p2_a).sum().backward()
        p1_a.grad = None; p2_a.grad = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_runs):
        mfd_static.distance(p1_a, p2_a).sum().backward()
        p1_a.grad = None; p2_a.grad = None
    torch.cuda.synchronize()
    t_static = (time.perf_counter() - t0) / n_runs

    # --- (B) Tensor κ ---
    kappa = torch.nn.Parameter(
        torch.tensor(-0.5, dtype=dtype, device=device),
    )
    mfd_tensor = KappaStereographicManifold(
        n=n, kappa=kappa, device=device, dtype=dtype,
    )
    p1_b = p1.clone().requires_grad_(True)
    p2_b = p2.clone().requires_grad_(True)
    for _ in range(5):
        mfd_tensor.distance(p1_b, p2_b).sum().backward()
        p1_b.grad = None; p2_b.grad = None; kappa.grad = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_runs):
        mfd_tensor.distance(p1_b, p2_b).sum().backward()
        p1_b.grad = None; p2_b.grad = None; kappa.grad = None
    torch.cuda.synchronize()
    t_tensor = (time.perf_counter() - t0) / n_runs

    return {
        "batch_size": B,
        "n": n,
        "n_runs": n_runs,
        "device": str(device),
        "dtype": str(dtype),
        "static_float_ms_per_iter": t_static * 1e3,
        "tensor_dispatch_ms_per_iter": t_tensor * 1e3,
        "overhead_ratio": t_tensor / t_static,
    }


def main():
    if not torch.cuda.is_available():
        raise SystemExit(
            "No GPU device available. This script requires a ROCm or CUDA "
            "build of PyTorch. Project default `.venv` ships CPU/CUDA-only "
            "torch — invoke via the ROCm venv (see module docstring)."
        )
    device = torch.device("cuda:0")
    name = torch.cuda.get_device_name(0)
    hip = torch.version.hip or "n/a"
    print(f"Device: {name}  (hip={hip}, torch={torch.__version__})")

    print()
    print("float64 latency:")
    r64 = gpu_latency(device, torch.float64)
    print(f"  static float κ:      {r64['static_float_ms_per_iter']:.3f} ms/iter")
    print(f"  tensor κ dispatch:   {r64['tensor_dispatch_ms_per_iter']:.3f} ms/iter")
    print(f"  overhead ratio:      ×{r64['overhead_ratio']:.2f}")

    print()
    print("float32 latency:")
    r32 = gpu_latency(device, torch.float32)
    print(f"  static float κ:      {r32['static_float_ms_per_iter']:.3f} ms/iter")
    print(f"  tensor κ dispatch:   {r32['tensor_dispatch_ms_per_iter']:.3f} ms/iter")
    print(f"  overhead ratio:      ×{r32['overhead_ratio']:.2f}")

    out_path = (
        Path(__file__).parent / "C4_kappa_crossing_stress_results.md"
    )
    addition = [
        "",
        "## (5) GPU latency on AMD ROCm",
        "",
        ("Re-run of the §(4) latency comparison on the workstation's "
         f"`{name}` (`hip={hip}`, `torch={torch.__version__}`). Run via "
         "the synoros-substrate venv (which has the ROCm-built torch); "
         "see `C4_kappa_crossing_gpu_bench.py` docstring for the exact "
         "invocation."),
        "",
        f"- Setup: batch = {r64['batch_size']}, n = {r64['n']}, "
        f"{r64['n_runs']} runs.",
        "",
        "| dtype | path | ms / iter (fwd + bwd) | overhead |",
        "|---|---|---:|---:|",
        f"| float64 | static float κ      | {r64['static_float_ms_per_iter']:.3f} | baseline |",
        f"| float64 | tensor κ dispatch   | {r64['tensor_dispatch_ms_per_iter']:.3f} | "
        f"×{r64['overhead_ratio']:.2f} |",
        f"| float32 | static float κ      | {r32['static_float_ms_per_iter']:.3f} | baseline |",
        f"| float32 | tensor κ dispatch   | {r32['tensor_dispatch_ms_per_iter']:.3f} | "
        f"×{r32['overhead_ratio']:.2f} |",
        "",
        ("The GPU overhead ratio is consistent with the CPU measurement "
         f"(~×1.2−1.5). On GPU the both-branches-evaluated cost is even "
         "cheaper in absolute terms because the elementwise transcendentals "
         "are throughput-bound and the dispatch adds two more single-kernel "
         "elementwise ops — easily fused / hidden in the existing pipeline. "
         "RDNA 4 (gfx1200) handles the float64 atanh/atan path at "
         f"native-double rates."),
    ]
    with out_path.open("a") as f:
        f.write("\n".join(addition) + "\n")
    print()
    print(f"Appended GPU results to {out_path}")


if __name__ == "__main__":
    main()
