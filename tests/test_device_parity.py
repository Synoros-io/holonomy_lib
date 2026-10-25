# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Synoros

"""Cross-device parity tests — every primitive produces the same
numerical results on CPU and (when available) CUDA / MPS.

These tests run automatically on CPU and on whatever GPU is available
through `torch.cuda.is_available()` / `torch.backends.mps.is_available()`.
When no GPU is present (the common case for CPU-only torch builds),
the GPU parametrization simply skips. Run on a CUDA / ROCm machine
to actually exercise the device-portable paths.

Why these tests matter: primitives may have device-conditional code
paths (e.g. `torch.svd_lowrank` falls back to a different algorithm
on CPU vs GPU). The tests pin down that the numerical contract is
the same regardless.
"""

from __future__ import annotations

import pytest
import torch

from holonomy_lib.algebra import truncated_svd
from holonomy_lib.discrete_geometry import ollivier_ricci_curvature
from holonomy_lib.manifolds import FixedRankManifold, LorentzManifold, SPDManifold
from holonomy_lib.spectral import laplacian
from holonomy_lib.tensor_calculus import hosvd


def _available_devices() -> list[str]:
    devs = ["cpu"]
    if torch.cuda.is_available():
        devs.append("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        devs.append("mps")
    return devs


_HAS_GPU = len(_available_devices()) > 1


def _seeded(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


@pytest.fixture(params=_available_devices())
def device(request):
    return request.param


@pytest.mark.skipif(not _HAS_GPU, reason="no GPU available — parity tests are CPU-only")
class TestParityOnGpu:
    """Smoke parity: each primitive on GPU matches the CPU result.

    We do exact comparisons (atol=0) on operations that are
    deterministic (no random projection); approximate comparisons
    on randomized SVD. eigh on different backends (LAPACK vs cuSOLVER /
    rocSOLVER) is allowed to disagree by ~1e-10 on float64.
    """

    @pytest.mark.parametrize("op_name", [
        "combinatorial", "symmetric_normalized", "random_walk", "signed",
    ])
    def test_laplacian_parity(self, op_name):
        A = torch.randn(2, 8, 8, dtype=torch.float64, generator=_seeded(1))
        A = (A + A.mT).abs()
        op = getattr(laplacian, op_name)
        L_cpu = op(A)
        L_gpu = op(A.to("cuda"))
        torch.testing.assert_close(L_cpu, L_gpu.cpu(), atol=1e-12, rtol=0)

    def test_truncated_svd_exact_parity(self):
        M = torch.randn(2, 16, 12, dtype=torch.float64, generator=_seeded(2))
        U_c, S_c, Vt_c = truncated_svd(M, r=4, mode="exact")
        U_g, S_g, Vt_g = truncated_svd(M.to("cuda"), r=4, mode="exact")
        # SVD has sign ambiguity per column; compare singular values (gauge-invariant).
        torch.testing.assert_close(S_c, S_g.cpu(), atol=1e-10, rtol=0)

    def test_hosvd_reconstruction_parity(self):
        T = torch.randn(1, 6, 7, 8, dtype=torch.float64, generator=_seeded(3))
        from holonomy_lib.tensor_calculus import mode_product
        c_c, f_c = hosvd(T, ranks=(3, 4, 5), mode="exact")
        c_g, f_g = hosvd(T.to("cuda"), ranks=(3, 4, 5), mode="exact")
        # Compare reconstructions (basis-rotation-invariant within truncation).
        rec_c = mode_product(c_c, f_c[0], axis=1)
        rec_c = mode_product(rec_c, f_c[1], axis=2)
        rec_c = mode_product(rec_c, f_c[2], axis=3)
        rec_g = mode_product(c_g, f_g[0], axis=1)
        rec_g = mode_product(rec_g, f_g[1], axis=2)
        rec_g = mode_product(rec_g, f_g[2], axis=3)
        torch.testing.assert_close(rec_c, rec_g.cpu(), atol=1e-10, rtol=0)

    def test_spd_distance_parity(self):
        mfd_c = SPDManifold(n=4, device="cpu", dtype=torch.float64)
        mfd_g = SPDManifold(n=4, device="cuda", dtype=torch.float64)
        S_c = mfd_c.random_point(batch_size=3, generator=_seeded(4))
        T_c = mfd_c.random_point(batch_size=3, generator=_seeded(5))
        d_c = mfd_c.distance(S_c, T_c)
        d_g = mfd_g.distance(S_c.to("cuda"), T_c.to("cuda"))
        torch.testing.assert_close(d_c, d_g.cpu(), atol=1e-9, rtol=1e-9)

    def test_lorentz_distance_parity(self):
        mfd_c = LorentzManifold(n=4, device="cpu", dtype=torch.float64)
        mfd_g = LorentzManifold(n=4, device="cuda", dtype=torch.float64)
        x_c = mfd_c.random_point(batch_size=3, generator=_seeded(40))
        y_c = mfd_c.random_point(batch_size=3, generator=_seeded(41))
        d_c = mfd_c.distance(x_c, y_c)
        d_g = mfd_g.distance(x_c.to("cuda"), y_c.to("cuda"))
        torch.testing.assert_close(d_c, d_g.cpu(), atol=1e-10, rtol=1e-10)

    def test_lorentz_exp_parity(self):
        mfd_c = LorentzManifold(n=4, device="cpu", dtype=torch.float64)
        mfd_g = LorentzManifold(n=4, device="cuda", dtype=torch.float64)
        x_c = mfd_c.random_point(batch_size=2, generator=_seeded(42))
        v_seed = torch.randn(2, mfd_c.n + 1, dtype=torch.float64,
                              generator=_seeded(43))
        v_c = mfd_c.projection(x_c, v_seed) * 0.1
        out_c = mfd_c.exp(x_c, v_c)
        out_g = mfd_g.exp(x_c.to("cuda"), v_c.to("cuda"))
        torch.testing.assert_close(out_c, out_g.cpu(), atol=1e-10, rtol=1e-10)

    def test_fixed_rank_retraction_exact_parity(self):
        mfd_c = FixedRankManifold(m=8, n=8, r=3, retraction_mode="exact")
        mfd_g = FixedRankManifold(m=8, n=8, r=3, device="cuda",
                                    retraction_mode="exact")
        pt = mfd_c.random_point(batch_size=1, generator=_seeded(6))
        tangent = mfd_c.projection(
            pt, torch.randn(1, 8, 8, dtype=torch.float64, generator=_seeded(7)),
        )
        out_c = mfd_c.retraction(pt, tangent)
        out_g = mfd_g.retraction(
            tuple(t.to("cuda") for t in pt), tangent.to("cuda"),
        )
        # SVD sign / basis freedom — compare reconstruction.
        rec_c = out_c[0] @ torch.diag_embed(out_c[1]) @ out_c[2]
        rec_g = out_g[0] @ torch.diag_embed(out_g[1]) @ out_g[2]
        torch.testing.assert_close(rec_c, rec_g.cpu(), atol=1e-10, rtol=0)

    def test_ollivier_parity(self):
        # Small graph so the test is fast.
        torch.manual_seed(8)
        A = (torch.rand(1, 8, 8, dtype=torch.float64) > 0.4).to(torch.float64)
        A = (A + A.mT) * 0.5
        A.diagonal(dim1=-2, dim2=-1).zero_()
        k_c = ollivier_ricci_curvature(A, alpha=0.0, reg=0.05, n_iter=50)
        k_g = ollivier_ricci_curvature(
            A.to("cuda"), alpha=0.0, reg=0.05, n_iter=50,
        )
        # Sinkhorn is iterative — small numerical differences accumulate;
        # 1e-6 atol is the practical bound across backends.
        torch.testing.assert_close(k_c, k_g.cpu(), atol=1e-6, rtol=0)


class TestDeviceMovability:
    """Every primitive's outputs land on the same device as the inputs.
    Catches accidental `.cpu()` / `.cuda()` calls in the implementation.
    """

    def test_laplacian_preserves_input_device(self, device):
        A = torch.randn(1, 5, 5, dtype=torch.float64,
                        generator=_seeded(20)).to(device)
        A = (A + A.mT).abs()
        L = laplacian.combinatorial(A)
        assert L.device.type == torch.device(device).type

    def test_truncated_svd_preserves_input_device(self, device):
        M = torch.randn(1, 8, 6, dtype=torch.float64,
                        generator=_seeded(21)).to(device)
        U, S, Vt = truncated_svd(M, r=3, mode="exact")
        for t in (U, S, Vt):
            assert t.device.type == torch.device(device).type

    def test_hosvd_preserves_input_device(self, device):
        T = torch.randn(1, 4, 5, 6, dtype=torch.float64,
                        generator=_seeded(22)).to(device)
        core, factors = hosvd(T, ranks=(2, 3, 4), mode="exact")
        assert core.device.type == torch.device(device).type
        for f in factors:
            assert f.device.type == torch.device(device).type

    def test_lorentz_preserves_input_device(self, device):
        mfd = LorentzManifold(n=3, device=device, dtype=torch.float64)
        x = mfd.random_point(batch_size=2, generator=_seeded(23))
        y = mfd.random_point(batch_size=2, generator=_seeded(24))
        d = mfd.distance(x, y)
        v = mfd.log(x, y)
        assert d.device.type == torch.device(device).type
        assert v.device.type == torch.device(device).type
        assert mfd.origin(2).device.type == torch.device(device).type
