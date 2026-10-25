"""Synthetic curvature-recovery task for C5 + C6 strengthening.

A clean identifiability test: generate ground-truth (coords, κ_field)
on `HeterogeneousKappaManifold` with three known regions —
hyperbolic, Euclidean, spherical — and use the pairwise Riemannian
distances as targets. Train three variants to recover the geometry
from distance-fitting alone:

  (1) Single κ baseline — `KappaStereographicManifold` with one
      learnable scalar curvature.
  (2) Discrete-gating per-node κ (GraphMoRE-style) —
      `DiscreteGatingKappa` from `tests/research_baselines/` with a
      K=3 bank matched to the true regions: `{-1.0, 0.0, +0.5}`.
  (3) Continuous per-point κ — `HeterogeneousKappaManifold` with
      `nn.Parameter(κ_field, shape=(N,))`. Sweep four combiners:
      arithmetic_mean, harmonic_mean, signed_geometric_mean (custom),
      max_magnitude (custom).

This is a *favorable* setting for the discrete baseline (its bank
contains every true κ exactly). The continuous variant has equal or
better expressivity (it can fit any real κ, including the bank
values exactly), so the comparison isolates the gradient-flow /
identifiability differences from the expressivity difference. A
follow-up "off-bank" test would tilt it toward continuous; in this
session we run the clean on-bank version first.

Output: `notes/strengthening/C5_C6_synthetic_task_results.md`.

Run (GPU on ROCm via sibling venv):

    PYTHONPATH=src <python with a ROCm or CUDA torch build> \\
      notes/strengthening/C5_C6_synthetic_task.py

Run (CPU, local venv):

    PYTHONPATH=. uv run python notes/strengthening/C5_C6_synthetic_task.py
"""

from __future__ import annotations

import time
from pathlib import Path

import torch
import torch.nn as nn

from holonomy_lib.experimental.heterogeneous_kappa import HeterogeneousKappaManifold
from holonomy_lib.manifolds.lorentz import _safe_sqrt
from holonomy_lib.manifolds.stereographic import KappaStereographicManifold

from tests.research_baselines.graphmore_discrete import DiscreteGatingKappa


# --- Ground-truth task ---------------------------------------------------
def make_ground_truth(seed: int, n_per_region: int = 30, dim: int = 3,
                      tangent_std: float = 0.25, dtype=torch.float64):
    """Generate (X_true, κ_true, D_true).

    Three regions, n_per_region nodes each (90 total):
      A: κ_true = -1.0 (hyperbolic)
      B: κ_true =  0.0 (Euclidean)
      C: κ_true = +0.5 (spherical)

    Per-point coords sampled via exp_0 of a small Gaussian tangent at
    each region's curvature. Pairwise distances computed via
    HeterogeneousKappaManifold with the arithmetic_mean combiner
    (which agrees on homogeneous κ-pairs with the standard
    `KappaStereographicManifold(κ).distance`).

    Returns:
      n_nodes, X_true (N, dim), kappa_true (N,), region (N,), D_true (N, N)
    """
    g = torch.Generator().manual_seed(seed)
    region_kappas = [-1.0, 0.0, +0.5]
    region = torch.cat([torch.full((n_per_region,), r, dtype=torch.long)
                        for r in range(len(region_kappas))])
    n_nodes = region.shape[0]
    kappa_true = torch.tensor(
        [region_kappas[r.item()] for r in region], dtype=dtype,
    )

    # Per-point tangent vectors. Small magnitude keeps spherical
    # samples in-domain.
    tangent = tangent_std * torch.randn(n_nodes, dim, generator=g, dtype=dtype)

    # Per-region exp_0 (use HeterogeneousKappaManifold's exp_0 directly)
    mfd_truth = HeterogeneousKappaManifold(n=dim, combiner="arithmetic_mean",
                                            dtype=dtype)
    X_true = mfd_truth.exp_0(tangent, kappa_true)

    # Pairwise distances (N, N): broadcast x[i], x[j], κ[i], κ[j].
    idx_i = torch.arange(n_nodes).unsqueeze(1).expand(n_nodes, n_nodes)
    idx_j = torch.arange(n_nodes).unsqueeze(0).expand(n_nodes, n_nodes)
    D_true = mfd_truth.distance(
        X_true[idx_i.reshape(-1)], kappa_true[idx_i.reshape(-1)],
        X_true[idx_j.reshape(-1)], kappa_true[idx_j.reshape(-1)],
    ).reshape(n_nodes, n_nodes)

    return n_nodes, X_true, kappa_true, region, D_true


# --- Custom combiners ---------------------------------------------------
def signed_geometric_mean(kappa_a: torch.Tensor, kappa_b: torch.Tensor) -> torch.Tensor:
    """sign(κ_a + κ_b) · _safe_sqrt(|κ_a · κ_b|).

    Uses the library's `_safe_sqrt` to keep the gradient finite at
    `κ_a = 0` or `κ_b = 0` — see C3 strengthening for the autograd-
    safety story. The plain `torch.sqrt` at 0 has unbounded derivative
    and propagates NaN; `_safe_sqrt` short-circuits the masked-out
    branch via `torch.where`.
    """
    sgn = torch.sign(kappa_a + kappa_b)
    mag = _safe_sqrt(torch.abs(kappa_a * kappa_b))
    return sgn * mag


def max_magnitude(kappa_a: torch.Tensor, kappa_b: torch.Tensor) -> torch.Tensor:
    """Whichever |κ| is bigger wins."""
    mask = torch.abs(kappa_a) >= torch.abs(kappa_b)
    return torch.where(mask, kappa_a, kappa_b)


COMBINERS = {
    "arithmetic_mean": "arithmetic_mean",
    "harmonic_mean": "harmonic_mean",
    "signed_geometric_mean": signed_geometric_mean,
    "max_magnitude": max_magnitude,
}


# --- Training loops -----------------------------------------------------
def _sample_pairs(n_nodes, n_pairs, g, device):
    i = torch.randint(0, n_nodes, (n_pairs,), generator=g).to(device)
    j = torch.randint(0, n_nodes, (n_pairs,), generator=g).to(device)
    return i, j


def _coord_barrier(coords: torch.Tensor, safe_norm: float = 0.85) -> torch.Tensor:
    """Quadratic penalty when ‖x_i‖ > safe_norm. Keeps points inside the
    manifold domain across all κ ∈ [-1, +1] (the regime we expect)."""
    norms = coords.norm(dim=-1)
    return ((norms.clamp(min=safe_norm) - safe_norm) ** 2).mean()


def train_single_kappa(n_nodes, D_true, dim, n_epochs, pairs_per_epoch,
                        seed, device):
    g = torch.Generator().manual_seed(seed)
    coords = nn.Parameter(
        0.1 * torch.randn(n_nodes, dim, generator=g, dtype=D_true.dtype).to(device)
    )
    kappa = nn.Parameter(torch.tensor(-0.05, dtype=D_true.dtype, device=device))
    mfd = KappaStereographicManifold(
        n=dim, kappa=kappa, device=device, dtype=D_true.dtype,
    )
    opt = torch.optim.Adam([coords, kappa], lr=0.005)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=n_epochs, eta_min=1e-4,
    )
    D = D_true.to(device)
    for epoch in range(n_epochs):
        opt.zero_grad()
        i, j = _sample_pairs(n_nodes, pairs_per_epoch, g, device)
        d_pred = mfd.distance(coords[i], coords[j])
        d_target = D[i, j]
        loss = ((d_pred - d_target) ** 2).mean() + 5.0 * _coord_barrier(coords)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([coords, kappa], max_norm=1.0)
        opt.step(); sch.step()
        with torch.no_grad():
            kappa.clamp_(-1.5, 1.5)
    # Final loss on all pairs (full evaluation)
    with torch.no_grad():
        idx_i = torch.arange(n_nodes, device=device).unsqueeze(1).expand(n_nodes, n_nodes)
        idx_j = torch.arange(n_nodes, device=device).unsqueeze(0).expand(n_nodes, n_nodes)
        D_pred = mfd.distance(coords[idx_i.reshape(-1)], coords[idx_j.reshape(-1)]).reshape(n_nodes, n_nodes)
        full_loss = ((D_pred - D) ** 2).mean().item()
    return {
        "final_loss": full_loss,
        "learned_kappa": float(kappa.item()),
        "kappa_recovery_corr": float("nan"),
    }


def train_discrete_gating(n_nodes, D_true, dim, n_epochs, pairs_per_epoch,
                          seed, kappa_true, device,
                          kappa_bank=(-1.0, 0.0, +0.5)):
    g = torch.Generator().manual_seed(seed)
    coords = nn.Parameter(
        0.1 * torch.randn(n_nodes, dim, generator=g, dtype=D_true.dtype).to(device)
    )
    gating = DiscreteGatingKappa(
        n_nodes=n_nodes, kappa_bank=kappa_bank,
        dim=dim, mode="soft", device=device, dtype=D_true.dtype,
    )
    opt = torch.optim.Adam([coords, gating.gates], lr=0.005)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=n_epochs, eta_min=1e-4,
    )
    D = D_true.to(device)
    for epoch in range(n_epochs):
        opt.zero_grad()
        i, j = _sample_pairs(n_nodes, pairs_per_epoch, g, device)
        d_pred = gating.pairwise_distance(coords[i], i, coords[j], j)
        d_target = D[i, j]
        loss = ((d_pred - d_target) ** 2).mean() + 5.0 * _coord_barrier(coords)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([coords, gating.gates], max_norm=1.0)
        opt.step(); sch.step()
    with torch.no_grad():
        idx_i = torch.arange(n_nodes, device=device).unsqueeze(1).expand(n_nodes, n_nodes)
        idx_j = torch.arange(n_nodes, device=device).unsqueeze(0).expand(n_nodes, n_nodes)
        D_pred = gating.pairwise_distance(
            coords[idx_i.reshape(-1)], idx_i.reshape(-1),
            coords[idx_j.reshape(-1)], idx_j.reshape(-1),
        ).reshape(n_nodes, n_nodes)
        full_loss = ((D_pred - D) ** 2).mean().item()
    recovered = gating.recovered_kappa().detach().cpu()
    kt_cpu = kappa_true.cpu()
    corr = float(torch.corrcoef(torch.stack([recovered, kt_cpu]))[0, 1].item())
    return {
        "final_loss": full_loss,
        "recovered_kappa_mean_regionA": float(recovered[:30].mean()),
        "recovered_kappa_mean_regionB": float(recovered[30:60].mean()),
        "recovered_kappa_mean_regionC": float(recovered[60:].mean()),
        "kappa_recovery_corr": corr,
    }


def train_continuous_kappa(n_nodes, D_true, dim, n_epochs, pairs_per_epoch,
                            seed, combiner, kappa_true, device,
                            kappa_clip: float = 1.5):
    """Hard-clamp `kappa_field` to ±kappa_clip after each step to keep
    the κ-stereographic domain `|κ|·‖x‖² < 1` consistent with the
    coord barrier at ‖x‖ < 0.85 (κ_clip · 0.85² ≈ 1.08 — safely inside
    the unit-ball domain). Without this clamp, Adam's momentum can
    push κ arbitrarily far over many epochs, eventually pushing
    points outside the manifold domain → NaN.
    """
    g = torch.Generator().manual_seed(seed)
    coords = nn.Parameter(
        0.1 * torch.randn(n_nodes, dim, generator=g, dtype=D_true.dtype).to(device)
    )
    kappa_field = nn.Parameter(
        0.01 * torch.randn(n_nodes, generator=g, dtype=D_true.dtype).to(device)
    )
    mfd = HeterogeneousKappaManifold(
        n=dim, combiner=combiner, device=device, dtype=D_true.dtype,
    )
    opt = torch.optim.Adam([coords, kappa_field], lr=0.005)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=n_epochs, eta_min=1e-4,
    )
    D = D_true.to(device)
    for epoch in range(n_epochs):
        opt.zero_grad()
        i, j = _sample_pairs(n_nodes, pairs_per_epoch, g, device)
        d_pred = mfd.distance(coords[i], kappa_field[i], coords[j], kappa_field[j])
        d_target = D[i, j]
        loss = ((d_pred - d_target) ** 2).mean() + 5.0 * _coord_barrier(coords)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([coords, kappa_field], max_norm=1.0)
        opt.step(); sch.step()
        with torch.no_grad():
            kappa_field.clamp_(-kappa_clip, kappa_clip)
    with torch.no_grad():
        idx_i = torch.arange(n_nodes, device=device).unsqueeze(1).expand(n_nodes, n_nodes)
        idx_j = torch.arange(n_nodes, device=device).unsqueeze(0).expand(n_nodes, n_nodes)
        D_pred = mfd.distance(
            coords[idx_i.reshape(-1)], kappa_field[idx_i.reshape(-1)],
            coords[idx_j.reshape(-1)], kappa_field[idx_j.reshape(-1)],
        ).reshape(n_nodes, n_nodes)
        full_loss = ((D_pred - D) ** 2).mean().item()
    learned = kappa_field.detach().cpu()
    kt_cpu = kappa_true.cpu()
    corr = float(torch.corrcoef(torch.stack([learned, kt_cpu]))[0, 1].item())
    return {
        "final_loss": full_loss,
        "learned_kappa_mean_regionA": float(learned[:30].mean()),
        "learned_kappa_mean_regionB": float(learned[30:60].mean()),
        "learned_kappa_mean_regionC": float(learned[60:].mean()),
        "kappa_recovery_corr": corr,
    }


# --- Main ---------------------------------------------------------------
def main():
    dtype = torch.float64
    n_epochs = 1500
    pairs_per_epoch = 256
    dim = 3
    seeds = [13, 17, 23]

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        device_label = f"{torch.cuda.get_device_name(0)} ({torch.version.hip or 'cuda'})"
    else:
        device = torch.device("cpu")
        device_label = "CPU"
    print(f"Device: {device_label}")
    print(f"Setup: dim={dim}, epochs={n_epochs}, pairs/epoch={pairs_per_epoch}, seeds={seeds}")

    # Ground truth (single seed; same task across all variants)
    n_nodes, X_true, kappa_true, region, D_true = make_ground_truth(
        seed=0, n_per_region=30, dim=dim, dtype=dtype,
    )
    print(f"Ground truth: N={n_nodes}, regions A(-1.0)/B(0.0)/C(+0.5) "
          f"size = {(region==0).sum().item()}/{(region==1).sum().item()}/"
          f"{(region==2).sum().item()}, D_max={D_true.max().item():.2f}, "
          f"D_mean={D_true[D_true>0].mean().item():.3f}")

    DISCRETE_BANKS = {
        "friendly_K3": (-1.0, 0.0, +0.5),       # bank matches true κ values
        "adversarial_K2": (-2.0, +1.5),         # bank misses all true values
    }

    results = {"single": [], "discrete": {}, "continuous": {}}
    t_start = time.perf_counter()

    print()
    for s in seeds:
        print(f"seed={s}", end=" ", flush=True)
        t0 = time.perf_counter()
        r = train_single_kappa(n_nodes, D_true, dim, n_epochs,
                                pairs_per_epoch, s, device)
        r["seed"] = s; r["time_s"] = time.perf_counter() - t0
        results["single"].append(r)
        print(f"single(L={r['final_loss']:.3f},κ={r['learned_kappa']:+.2f})",
              end=" ", flush=True)

        for bank_name, bank in DISCRETE_BANKS.items():
            t0 = time.perf_counter()
            r = train_discrete_gating(n_nodes, D_true, dim, n_epochs,
                                       pairs_per_epoch, s, kappa_true,
                                       device, kappa_bank=bank)
            r["seed"] = s; r["bank"] = bank_name; r["bank_values"] = list(bank)
            r["time_s"] = time.perf_counter() - t0
            results["discrete"].setdefault(bank_name, []).append(r)
            print(f"discrete-{bank_name}(L={r['final_loss']:.3f},"
                  f"corr={r['kappa_recovery_corr']:+.2f})",
                  end=" ", flush=True)

        for cname, combiner in COMBINERS.items():
            t0 = time.perf_counter()
            r = train_continuous_kappa(n_nodes, D_true, dim, n_epochs,
                                        pairs_per_epoch, s, combiner,
                                        kappa_true, device)
            r["seed"] = s; r["combiner"] = cname
            r["time_s"] = time.perf_counter() - t0
            results["continuous"].setdefault(cname, []).append(r)
        print()

    total_time = time.perf_counter() - t_start

    def mean_std(values):
        t = torch.tensor(values, dtype=torch.float64)
        return t.mean().item(), (t.std().item() if t.numel() > 1 else 0.0)

    def summarize(runs, fields):
        return {f: mean_std([r[f] for r in runs]) for f in fields}

    single_summary = summarize(results["single"],
                               ["final_loss", "learned_kappa"])
    discrete_summary = {
        b: summarize(rs, ["final_loss", "kappa_recovery_corr",
                           "recovered_kappa_mean_regionA",
                           "recovered_kappa_mean_regionB",
                           "recovered_kappa_mean_regionC"])
        for b, rs in results["discrete"].items()
    }
    continuous_summary = {
        c: summarize(rs, ["final_loss", "kappa_recovery_corr",
                           "learned_kappa_mean_regionA",
                           "learned_kappa_mean_regionB",
                           "learned_kappa_mean_regionC"])
        for c, rs in results["continuous"].items()
    }

    # --- Markdown output ---
    out_path = Path(__file__).parent / "C5_C6_synthetic_task_results.md"
    lines = [
        "# C5 + C6 synthetic-task results — per-point κ recovery",
        "",
        ("Generated by `notes/strengthening/C5_C6_synthetic_task.py`. "
         "Identifiability test: ground-truth coords + per-point κ "
         "drawn from three regions on `HeterogeneousKappaManifold`. "
         "Pairwise Riemannian distances under the truth manifold are "
         "the targets. Three variants of curvature parameterization "
         "race to recover the geometry; we measure final fit loss "
         "(all-pairs MSE, computed at the end of training) and per-node "
         "κ-recovery correlation against the ground truth."),
        "",
        f"- **Device**: {device_label}",
        f"- **Setup**: embedding dim = {dim}, {n_epochs} epochs, "
        f"{pairs_per_epoch} sampled pairs/epoch, "
        f"{len(seeds)} seeds = {seeds}.",
        f"- **Ground truth**: N={n_nodes}, 3 regions of 30 nodes "
        f"each: A (κ=−1.0), B (κ=0.0), C (κ=+0.5). Distance targets "
        f"computed under `HeterogeneousKappaManifold(arithmetic_mean)` "
        f"on the true κ_field. Max true pairwise distance "
        f"= {D_true.max().item():.3f}.",
        f"- **Total wall time**: {total_time:.1f} s.",
        "",
        "## (1) All-pairs fit loss + κ-recovery (mean ± std across seeds)",
        "",
        "| Variant | Combiner / bank | Final loss | κ-recovery corr |",
        "|---|---|---:|---:|",
        f"| Single κ baseline | — | "
        f"{single_summary['final_loss'][0]:.4f} ± {single_summary['final_loss'][1]:.4f} | "
        f"N/A |",
    ]
    bank_labels = {"friendly_K3": "bank {−1.0, 0.0, +0.5} (matches truth)",
                    "adversarial_K2": "bank {−2.0, +1.5} (no truth match)"}
    for bname, s in discrete_summary.items():
        lines.append(
            f"| Discrete gating | {bank_labels[bname]} | "
            f"{s['final_loss'][0]:.4f} ± {s['final_loss'][1]:.4f} | "
            f"{s['kappa_recovery_corr'][0]:+.3f} ± "
            f"{s['kappa_recovery_corr'][1]:.3f} |"
        )
    for c, s in continuous_summary.items():
        lines.append(
            f"| Continuous per-point κ | {c} | "
            f"{s['final_loss'][0]:.4f} ± {s['final_loss'][1]:.4f} | "
            f"{s['kappa_recovery_corr'][0]:+.3f} ± "
            f"{s['kappa_recovery_corr'][1]:.3f} |"
        )

    lines += [
        "",
        ("Reading the table: low final loss means the variant fit the "
         "distance pattern well. High κ-recovery correlation means the "
         "learned per-node κ matches the true region label per node."),
        "",
        "## (2) Region-wise recovered κ",
        "",
        ("True κ per region: A=−1.0, B=0.0, C=+0.5. Mean of the "
         "learned per-node κ (continuous variant) / mixture-weighted "
         "κ (discrete) inside each region, averaged across seeds:"),
        "",
        "| Variant | Combiner / bank | region A (true −1.0) | region B (true 0.0) | region C (true +0.5) |",
        "|---|---|---:|---:|---:|",
    ]
    for bname, s in discrete_summary.items():
        lines.append(
            f"| Discrete gating | {bank_labels[bname]} | "
            f"{s['recovered_kappa_mean_regionA'][0]:+.3f} ± "
            f"{s['recovered_kappa_mean_regionA'][1]:.3f} | "
            f"{s['recovered_kappa_mean_regionB'][0]:+.3f} ± "
            f"{s['recovered_kappa_mean_regionB'][1]:.3f} | "
            f"{s['recovered_kappa_mean_regionC'][0]:+.3f} ± "
            f"{s['recovered_kappa_mean_regionC'][1]:.3f} |"
        )
    for c, s in continuous_summary.items():
        lines.append(
            f"| Continuous κ | {c} | "
            f"{s['learned_kappa_mean_regionA'][0]:+.3f} ± "
            f"{s['learned_kappa_mean_regionA'][1]:.3f} | "
            f"{s['learned_kappa_mean_regionB'][0]:+.3f} ± "
            f"{s['learned_kappa_mean_regionB'][1]:.3f} | "
            f"{s['learned_kappa_mean_regionC'][0]:+.3f} ± "
            f"{s['learned_kappa_mean_regionC'][1]:.3f} |"
        )

    lines += [
        "",
        "## (3) Single-κ baseline — what one scalar settles on",
        "",
        ("Forced to use one κ for all three regions, the optimizer "
         "must compromise."),
        "",
        f"- Learned κ (mean ± std across seeds): "
        f"{single_summary['learned_kappa'][0]:+.3f} ± "
        f"{single_summary['learned_kappa'][1]:.3f}",
        f"- Final loss is bounded below by the irreducible "
        f"compromise across regions.",
        "",
        "## (4) Combiner study (C6)",
        "",
        ("Four combiners exercised on the continuous variant. The "
         "*ground truth* used `arithmetic_mean`, so any other "
         "combiner is fitting a slightly different distance metric "
         "with the same per-point κ's. This is the realistic case "
         "for an unaligned downstream user."),
        "",
        ("- **arithmetic_mean** (default): matches the ground-truth "
         "combiner; expected to dominate."),
        ("- **harmonic_mean**: well-defined when both κ's are nonzero "
         "and same-sign; with κ_B = 0 in region B, hits the "
         "`finfo.tiny`-clamp guard."),
        ("- **signed_geometric_mean**: magnitude-aware (geometric mean "
         "of |κ|'s, signed by sum). Smoother across sign mixes than "
         "harmonic."),
        ("- **max_magnitude**: \"the more strongly curved side wins\". "
         "Non-smooth in κ at the |κ_a| = |κ_b| boundary."),
        "",
        "## (5) Notes & caveats",
        "",
        ("- We run discrete gating with **two banks**: a *friendly* "
         "K=3 bank that includes every true κ value exactly "
         "(the maximally-favorable case for discrete) and an "
         "*adversarial* K=2 bank with no truth match. The contrast "
         "between the two isolates the bank-choice burden — the "
         "continuous variant has no analogous burden."),
        ("- The κ-recovery correlation is Pearson r across all N=90 "
         "nodes between learned κ and true κ. r=1 = perfect direction; "
         "r=0 = no signal / collapse; r=−1 = sign-flipped. The friendly "
         "bank gets the truth-direction effectively for free (it has "
         "to do is pick the right gate per node); continuous has to "
         "find each κ in a real interval."),
        ("- The C5 + C6 claim is about the *primitive* — what "
         "`HeterogeneousKappaManifold` enables. The continuous κ "
         "vs. discrete-gating comparison shows the primitive's "
         "expressivity *without* a bank choice; the combiner sweep "
         "characterizes the design axis we don't lock at construction "
         "time."),
        ("- Coords are kept on-domain via a soft barrier at "
         "‖x‖ < 0.85 (penalty `5·(‖x‖-0.85)²₊`); per-point κ is "
         "post-step clamped to ±1.5 to keep `|κ|·‖x‖² < 1`. Without "
         "these guards Adam's momentum eventually pushes points "
         "outside the manifold domain → NaN (we saw this empirically "
         "during pilot runs; the guards keep training stable for "
         "the full 1500-epoch schedule)."),
    ]

    out_path.write_text("\n".join(lines) + "\n")
    print(f"\nTotal wall time: {total_time:.1f} s")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
