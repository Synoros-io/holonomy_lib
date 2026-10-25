# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Synoros

"""Tests for holonomy_lib.provenance.protocol.

Covers:
  1. Determinism — same op + inputs → same hex.
  2. Sensitivity — different params or inputs → different hex.
  3. Transparency outside record() — decorated functions behave normally.
  4. Recording — primitives emit nodes when inside record().
  5. DAG chaining — output of one op becomes input of another with
     the correct provenance edge.
  6. Substitution — TransformerLens-style activation patching applied
     to math primitives.
  7. Interop — to_networkx / to_dataframe / to_dict / to_sae_dataset.
  8. Hooks — observe ops without changing behavior.
  9. Diff — compare two recordings.
  10. Persistence — save/load registry to disk.
  11. Hash algorithm pluggability — blake3 vs sha256.
"""

from __future__ import annotations

import tempfile
import warnings
from pathlib import Path

import pytest
import torch

from holonomy_lib import provenance
from holonomy_lib.algebra import truncated_svd
from holonomy_lib.spectral import laplacian


def _seeded(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


# --------------------------------------------------------------------
# Transparency outside record()
# --------------------------------------------------------------------


class TestTransparency:
    def test_decorated_function_works_without_recording(self):
        """Calling a decorated primitive outside record() returns identical
        values to its undecorated semantics (no behavioral change)."""
        M = torch.randn(5, 7, dtype=torch.float64, generator=_seeded(0))
        U, S, Vt = truncated_svd(M, r=3, mode="exact")
        assert U.shape == (5, 3)
        assert S.shape == (3,)
        assert Vt.shape == (3, 7)

    def test_no_registry_outside_recording(self):
        """No global state pollution: calling a decorated op outside
        record() does not leave anything behind."""
        # Just verifying the call returns and doesn't raise.
        M = torch.randn(4, 4, dtype=torch.float64)
        L = laplacian.combinatorial(M)
        assert L.shape == (4, 4)


# --------------------------------------------------------------------
# Recording — basic capture
# --------------------------------------------------------------------


class TestRecording:
    def test_recording_captures_op(self):
        M = torch.randn(1, 5, 7, dtype=torch.float64, generator=_seeded(1))
        with provenance.record() as reg:
            U, S, Vt = truncated_svd(M, r=3, mode="exact")
        assert len(reg) == 1
        node = next(iter(reg))
        assert node.op_id == "holonomy_lib.algebra.linear.truncated_svd"
        assert node.op_version == "0.1"
        # Output shape captured per-tensor (truncated_svd returns 3 tensors).
        assert len(node.output_shape) == 3
        assert node.output_shape[0] == (1, 5, 3)
        assert node.output_shape[1] == (1, 3)
        assert node.output_shape[2] == (1, 3, 7)

    def test_recording_captures_params(self):
        M = torch.randn(1, 5, 5, dtype=torch.float64, generator=_seeded(2))
        with provenance.record() as reg:
            truncated_svd(M, r=2, mode="exact")
        node = next(iter(reg))
        params = node.parsed_params()
        assert params["r"] == 2
        assert params["mode"] == "exact"


# --------------------------------------------------------------------
# Determinism — same op + same inputs → same hex
# --------------------------------------------------------------------


class TestDeterminism:
    def test_same_inputs_same_hex(self):
        M = torch.randn(1, 5, 5, dtype=torch.float64, generator=_seeded(3))
        with provenance.record() as reg1:
            truncated_svd(M, r=2, mode="exact")
        with provenance.record() as reg2:
            truncated_svd(M, r=2, mode="exact")
        h1 = next(iter(reg1)).hex
        h2 = next(iter(reg2)).hex
        assert h1 == h2, "same op + same inputs → same hex required"

    def test_different_params_different_hex(self):
        M = torch.randn(1, 5, 5, dtype=torch.float64, generator=_seeded(4))
        with provenance.record() as reg:
            truncated_svd(M, r=2, mode="exact")
            truncated_svd(M, r=3, mode="exact")
        hexes = [n.hex for n in reg]
        assert len(set(hexes)) == 2, "different r should yield different hex"

    def test_different_inputs_different_hex(self):
        M1 = torch.randn(1, 5, 5, dtype=torch.float64, generator=_seeded(5))
        M2 = torch.randn(1, 5, 5, dtype=torch.float64, generator=_seeded(6))
        with provenance.record() as reg:
            truncated_svd(M1, r=2, mode="exact")
            truncated_svd(M2, r=2, mode="exact")
        hexes = [n.hex for n in reg]
        assert len(set(hexes)) == 2, "different inputs should yield different hex"


# --------------------------------------------------------------------
# DAG chaining — outputs of one op become inputs of the next
# --------------------------------------------------------------------


class TestDagChaining:
    def test_chained_ops_have_correct_edges(self):
        """L = combinatorial(A); U, _, _ = truncated_svd(L, ...)
        The truncated_svd node's input_hexes should include the
        combinatorial node's hex.
        """
        A = torch.randn(1, 6, 6, dtype=torch.float64, generator=_seeded(7))
        A = (A + A.mT).abs()  # symmetric non-negative
        with provenance.record() as reg:
            L = laplacian.combinatorial(A)
            U, S, Vt = truncated_svd(L, r=3, mode="exact")

        # Two nodes recorded
        assert len(reg) == 2
        # Find each by op_id
        comb_node = reg.where(op_id="holonomy_lib.spectral.laplacian.combinatorial")[0]
        svd_node = reg.where(op_id="holonomy_lib.algebra.linear.truncated_svd")[0]
        # SVD's input_hexes are name=hex; comb_node.hex appears as the
        # hex part of one of those.
        hex_parts = [h.partition("=")[2] for h in svd_node.input_hexes]
        assert comb_node.hex in hex_parts

    def test_ancestors_walk(self):
        A = torch.randn(1, 5, 5, dtype=torch.float64, generator=_seeded(8))
        A = (A + A.mT).abs()
        with provenance.record() as reg:
            L = laplacian.combinatorial(A)
            U, _, _ = truncated_svd(L, r=2, mode="exact")
        svd_hex = reg.where(op_id="holonomy_lib.algebra.linear.truncated_svd")[0].hex
        ancestors = reg.ancestors(svd_hex)
        # Should include the combinatorial node + the SVD itself
        assert len(ancestors) == 2


# --------------------------------------------------------------------
# Tensor caching
# --------------------------------------------------------------------


class TestTensorCache:
    def test_cache_tensors_off_by_default(self):
        M = torch.randn(1, 5, 5, dtype=torch.float64, generator=_seeded(9))
        with provenance.record() as reg:
            truncated_svd(M, r=2, mode="exact")
        h = next(iter(reg)).hex
        assert reg.get_tensor(h) is None

    def test_cache_tensors_when_enabled(self):
        M = torch.randn(1, 5, 5, dtype=torch.float64, generator=_seeded(10))
        with provenance.record(cache_tensors=True) as reg:
            U, S, Vt = truncated_svd(M, r=2, mode="exact")
        # Multi-output: U/S/Vt cached under hex:0, hex:1, hex:2
        node = next(iter(reg))
        cached_U = reg.get_tensor(f"{node.hex}:0")
        assert cached_U is not None
        torch.testing.assert_close(cached_U, U)


# --------------------------------------------------------------------
# Substitution — TransformerLens-style activation patching
# --------------------------------------------------------------------


class TestSubstitution:
    def test_substitute_at_op_call(self):
        """When we substitute a hex, that op call returns the substituted
        value instead of computing.
        """
        A = torch.randn(1, 4, 4, dtype=torch.float64, generator=_seeded(11))
        A = (A + A.mT).abs()
        # First pass: record to learn the hex of the combinatorial call
        with provenance.record() as reg:
            L = laplacian.combinatorial(A)
        target_hex = next(iter(reg)).hex

        # Second pass: substitute that hex with zeros
        fake_L = torch.zeros(1, 4, 4, dtype=torch.float64)
        with provenance.record() as reg2:
            with reg2.substitute({target_hex: fake_L}):
                L_patched = laplacian.combinatorial(A)
        torch.testing.assert_close(L_patched, fake_L, atol=0, rtol=0)


# --------------------------------------------------------------------
# Mech-interp interop — networkx + dataframe + dict exports
# --------------------------------------------------------------------


class TestInterop:
    def test_to_dict(self):
        A = torch.randn(1, 4, 4, dtype=torch.float64, generator=_seeded(12))
        A = (A + A.mT).abs()
        with provenance.record() as reg:
            laplacian.combinatorial(A)
        d = reg.to_dict()
        assert "nodes" in d
        assert len(d["nodes"]) == 1
        assert d["nodes"][0]["op_id"] == "holonomy_lib.spectral.laplacian.combinatorial"

    def test_to_networkx(self):
        nx = pytest.importorskip("networkx")
        A = torch.randn(1, 5, 5, dtype=torch.float64, generator=_seeded(13))
        A = (A + A.mT).abs()
        with provenance.record() as reg:
            L = laplacian.combinatorial(A)
            truncated_svd(L, r=2, mode="exact")
        G = reg.to_networkx()
        assert isinstance(G, nx.DiGraph)
        # Three nodes: the raw input A (leaf, no op), the combinatorial
        # Laplacian, and the truncated SVD. Two edges: A → L → SVD.
        assert G.number_of_nodes() == 3
        assert G.number_of_edges() == 2
        # Op-produced nodes carry op_id; leaf input nodes don't.
        op_nodes = [h for h in G.nodes if "op_id" in G.nodes[h]]
        leaf_nodes = [h for h in G.nodes if "op_id" not in G.nodes[h]]
        assert len(op_nodes) == 2
        assert len(leaf_nodes) == 1

    def test_to_dataframe(self):
        pd = pytest.importorskip("pandas")
        A = torch.randn(1, 4, 4, dtype=torch.float64, generator=_seeded(14))
        A = (A + A.mT).abs()
        with provenance.record() as reg:
            laplacian.combinatorial(A)
            truncated_svd(A, r=2, mode="exact")
        df = reg.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "op_id" in df.columns
        assert "hex" in df.columns


# --------------------------------------------------------------------
# Mech-interp demo — full workflow
# --------------------------------------------------------------------


class TestMechInterpDemo:
    """End-to-end demo: a small "pipeline" gets traced, a DAG is built,
    one node is identified by op_id, substituted, and the downstream
    computation is observed to change. This is the workflow a mech
    interp researcher would do — TransformerLens for math primitives.
    """

    def test_ablation_workflow(self):
        A = torch.randn(1, 6, 6, dtype=torch.float64, generator=_seeded(15))
        A = (A + A.mT).abs()

        def pipeline(adj):
            L = laplacian.symmetric_normalized(adj)
            U, S, Vt = truncated_svd(L, r=3, mode="exact")
            return U @ torch.diag_embed(S)

        # Baseline
        with provenance.record() as reg_base:
            out_base = pipeline(A)

        # Find the Laplacian node, prepare a "zero ablation"
        lap_node = reg_base.where(
            op_id="holonomy_lib.spectral.laplacian.symmetric_normalized",
        )[0]
        zero_L = torch.zeros(1, 6, 6, dtype=torch.float64)

        # Ablate: re-run with the Laplacian replaced by zeros.
        with provenance.record() as reg_ablated:
            with reg_ablated.substitute({lap_node.hex: zero_L}):
                out_ablated = pipeline(A)

        # Sanity checks:
        #   1. Baseline and ablation produced different outputs.
        diff = (out_base - out_ablated).abs().max().item()
        assert diff > 0.1, (
            f"ablation should change downstream output; diff={diff}"
        )
        #   2. Ablated DAG has the substituted Laplacian.
        ablated_lap = reg_ablated.where(
            op_id="holonomy_lib.spectral.laplacian.symmetric_normalized",
        )[0]
        assert ablated_lap.hex == lap_node.hex, (
            "same op call → same hex even with substitution active"
        )


# --------------------------------------------------------------------
# Hooks — observation without mutation
# --------------------------------------------------------------------


class TestHooks:
    def test_hook_fires_on_matching_op_id(self):
        A = torch.randn(1, 4, 4, dtype=torch.float64, generator=_seeded(20))
        A = (A + A.mT).abs()
        captured = []
        with provenance.record() as reg:
            reg.on_op(
                "holonomy_lib.spectral.laplacian.combinatorial",
                lambda node, out: captured.append((node.hex, out.shape)),
            )
            L = laplacian.combinatorial(A)
        assert len(captured) == 1
        hex_id, shape = captured[0]
        assert shape == (1, 4, 4)
        assert hex_id == next(iter(reg)).hex

    def test_hook_does_not_fire_for_other_ops(self):
        A = torch.randn(1, 4, 4, dtype=torch.float64, generator=_seeded(21))
        A = (A + A.mT).abs()
        captured = []
        with provenance.record() as reg:
            # Hook for an op we don't call
            reg.on_op("holonomy_lib.algebra.linear.truncated_svd",
                        lambda n, o: captured.append(n))
            laplacian.combinatorial(A)
        assert captured == []

    def test_multiple_hooks_for_same_op_fire_in_order(self):
        A = torch.randn(1, 4, 4, dtype=torch.float64, generator=_seeded(22))
        A = (A + A.mT).abs()
        order = []
        with provenance.record() as reg:
            reg.on_op("holonomy_lib.spectral.laplacian.combinatorial",
                        lambda n, o: order.append("first"))
            reg.on_op("holonomy_lib.spectral.laplacian.combinatorial",
                        lambda n, o: order.append("second"))
            laplacian.combinatorial(A)
        assert order == ["first", "second"]

    def test_hook_observes_does_not_mutate(self):
        """Hook output is not used; the math primitive's actual return
        value is what the caller sees.
        """
        A = torch.randn(1, 4, 4, dtype=torch.float64, generator=_seeded(23))
        A = (A + A.mT).abs()
        with provenance.record() as reg:
            # Hook that "tries" to mutate (its return value is ignored)
            reg.on_op("holonomy_lib.spectral.laplacian.combinatorial",
                        lambda n, o: torch.zeros_like(o))
            L = laplacian.combinatorial(A)
        # L is the real Laplacian, not zeros
        assert not torch.allclose(L, torch.zeros_like(L))


# --------------------------------------------------------------------
# SAELens-ready dataset emission
# --------------------------------------------------------------------


class TestSaeDataset:
    def test_emits_cached_activations_with_metadata(self):
        A = torch.randn(1, 5, 5, dtype=torch.float64, generator=_seeded(30))
        A = (A + A.mT).abs()
        with provenance.record(cache_tensors=True) as reg:
            laplacian.combinatorial(A)
            truncated_svd(A, r=2, mode="exact")
        records = list(reg.to_sae_dataset())
        # combinatorial yields 1 tensor; truncated_svd yields 3 (multi-output)
        assert len(records) == 4
        for tensor, meta in records:
            assert isinstance(tensor, torch.Tensor)
            assert "hex" in meta
            assert "op_id" in meta

    def test_filter_by_op_id(self):
        A = torch.randn(1, 5, 5, dtype=torch.float64, generator=_seeded(31))
        A = (A + A.mT).abs()
        with provenance.record(cache_tensors=True) as reg:
            laplacian.combinatorial(A)
            truncated_svd(A, r=2, mode="exact")
        only_svd = list(reg.to_sae_dataset(
            op_id="holonomy_lib.algebra.linear.truncated_svd",
        ))
        assert len(only_svd) == 3  # U, S, Vt
        assert all(
            meta["op_id"] == "holonomy_lib.algebra.linear.truncated_svd"
            for _, meta in only_svd
        )


# --------------------------------------------------------------------
# Run diffing
# --------------------------------------------------------------------


class TestDiff:
    def test_identical_runs_diff_empty(self):
        A = torch.randn(1, 4, 4, dtype=torch.float64, generator=_seeded(40))
        A = (A + A.mT).abs()
        with provenance.record() as reg1:
            laplacian.combinatorial(A)
        with provenance.record() as reg2:
            laplacian.combinatorial(A)
        d = reg1.diff(reg2)
        assert d["only_in_self"] == {}
        assert d["only_in_other"] == {}
        assert len(d["shared"]) == 1

    def test_different_inputs_show_divergence(self):
        A = torch.randn(1, 4, 4, dtype=torch.float64, generator=_seeded(41))
        A = (A + A.mT).abs()
        B = torch.randn(1, 4, 4, dtype=torch.float64, generator=_seeded(42))
        B = (B + B.mT).abs()
        with provenance.record() as reg1:
            laplacian.combinatorial(A)
        with provenance.record() as reg2:
            laplacian.combinatorial(B)
        d = reg1.diff(reg2)
        # Both runs have the combinatorial op, but with different hexes
        op = "holonomy_lib.spectral.laplacian.combinatorial"
        assert op in d["only_in_self"]
        assert op in d["only_in_other"]
        assert d["shared"].get(op, []) == []

    def test_different_op_ids_show_in_only_lists(self):
        A = torch.randn(1, 4, 4, dtype=torch.float64, generator=_seeded(43))
        A = (A + A.mT).abs()
        with provenance.record() as reg1:
            laplacian.combinatorial(A)
        with provenance.record() as reg2:
            laplacian.symmetric_normalized(A)
        d = reg1.diff(reg2)
        assert "holonomy_lib.spectral.laplacian.combinatorial" in d["op_ids_only_in_self"]
        assert "holonomy_lib.spectral.laplacian.symmetric_normalized" in d["op_ids_only_in_other"]


# --------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------


class TestPersistence:
    def test_save_and_load_roundtrip(self):
        A = torch.randn(1, 4, 4, dtype=torch.float64, generator=_seeded(50))
        A = (A + A.mT).abs()
        with provenance.record() as reg:
            laplacian.combinatorial(A)
            truncated_svd(A, r=2, mode="exact")

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "registry.json"
            reg.save(path)
            assert path.exists()
            loaded = provenance.ProvenanceRegistry.load(path)

        assert len(loaded) == len(reg)
        # Hexes survive
        original_hexes = {n.hex for n in reg}
        loaded_hexes = {n.hex for n in loaded}
        assert original_hexes == loaded_hexes
        # Algorithm preserved
        assert loaded.hash_algorithm == reg.hash_algorithm


# --------------------------------------------------------------------
# Causal replay — downstream DAG re-execution
# --------------------------------------------------------------------


class TestReplay:
    def test_replay_requires_tensor_cache(self):
        A = torch.randn(1, 4, 4, dtype=torch.float64, generator=_seeded(70))
        A = (A + A.mT).abs()
        with provenance.record() as reg:
            laplacian.combinatorial(A)
        with pytest.raises(ValueError, match="cache_tensors"):
            reg.replay({"abc": torch.zeros(4)})

    def test_replay_rejects_unknown_target(self):
        A = torch.randn(1, 4, 4, dtype=torch.float64, generator=_seeded(71))
        A = (A + A.mT).abs()
        with provenance.record(cache_tensors=True) as reg:
            laplacian.combinatorial(A)
        with pytest.raises(KeyError, match="not in registry"):
            reg.replay({"0000000000000000": torch.zeros(1, 4, 4)})

    def test_replay_substitutes_and_propagates(self):
        """Replay rebuilds the downstream chain from cached upstream."""
        A = torch.randn(1, 5, 5, dtype=torch.float64, generator=_seeded(72))
        A = (A + A.mT).abs() + torch.eye(5, dtype=torch.float64).unsqueeze(dim=0)

        with provenance.record(cache_tensors=True) as reg:
            L = laplacian.combinatorial(A)
            U, S, Vt = truncated_svd(L, r=3, mode="exact")

        # Find the Laplacian node; substitute it with zeros.
        lap_hex = reg.where(
            op_id="holonomy_lib.spectral.laplacian.combinatorial",
        )[0].hex
        new_outputs = reg.replay({lap_hex: torch.zeros(1, 5, 5, dtype=torch.float64)})

        # The SVD should have been re-executed; outputs differ from cache.
        svd_hex = reg.where(op_id="holonomy_lib.algebra.linear.truncated_svd")[0].hex
        original_U = reg.get_tensor(f"{svd_hex}:0")
        new_U = new_outputs[f"{svd_hex}:0"]
        diff = (original_U - new_U).abs().max().item()
        assert diff > 0.01, (
            f"replay should produce different U for zero'd Laplacian; diff={diff}"
        )

    def test_replay_skips_unaffected_nodes(self):
        """Nodes upstream of the substitution target are NOT in the output;
        only descendants are re-executed.
        """
        A = torch.randn(1, 4, 4, dtype=torch.float64, generator=_seeded(73))
        A = (A + A.mT).abs() + torch.eye(4, dtype=torch.float64).unsqueeze(dim=0)
        with provenance.record(cache_tensors=True) as reg:
            L = laplacian.combinatorial(A)
            U, S, Vt = truncated_svd(L, r=2, mode="exact")

        lap_hex = reg.where(
            op_id="holonomy_lib.spectral.laplacian.combinatorial",
        )[0].hex
        svd_hex = reg.where(op_id="holonomy_lib.algebra.linear.truncated_svd")[0].hex
        new_outputs = reg.replay({lap_hex: torch.zeros(1, 4, 4, dtype=torch.float64)})

        # Only SVD outputs in new_outputs (combinatorial wasn't re-executed
        # since it was substituted, not downstream of itself)
        assert lap_hex not in new_outputs
        assert any(h.startswith(svd_hex) for h in new_outputs)


# --------------------------------------------------------------------
# Hash-algorithm pluggability
# --------------------------------------------------------------------


class TestReplayMultiOutput:
    """Multi-output parents (ops returning tuples) feeding a single child
    must replay correctly. The topological sort in `replay()` uses
    `children_of` with one entry per parent-input edge, which means a
    child consuming two outputs of the same parent appears twice in
    children_of[parent]; the indegree count also counts both edges.
    The two cancel out — verifying this here so it stays correct.
    """

    def test_replay_multi_output_parent_to_single_child(self):
        from holonomy_lib.provenance import with_provenance

        @with_provenance("test.replay_two_outputs", op_version="0.1")
        def two_outputs(M: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return M + 1, M - 1

        @with_provenance("test.replay_consume_both", op_version="0.1")
        def consume_both(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
            return A * B

        @with_provenance("test.replay_head", op_version="0.1")
        def head(M: torch.Tensor) -> torch.Tensor:
            return M * 2

        M = torch.randn(1, 3, 3, dtype=torch.float64, generator=_seeded(120))
        with provenance.record(cache_tensors=True) as reg:
            H = head(M)
            A, B = two_outputs(H)
            consume_both(A, B)

        head_hex = reg.where(op_id="test.replay_head")[0].hex
        two_hex = reg.where(op_id="test.replay_two_outputs")[0].hex
        cb_hex = reg.where(op_id="test.replay_consume_both")[0].hex

        new = reg.replay({head_hex: torch.zeros_like(reg.get_tensor(head_hex))})
        # Both downstream nodes must appear in new (consume_both depends
        # on BOTH of two_outputs's outputs — if indegree counting were
        # wrong, consume_both would be silently dropped).
        assert f"{two_hex}:0" in new
        assert f"{two_hex}:1" in new
        assert cb_hex in new


class TestReplayFinalHex:
    """The `final_hex` argument to replay() short-circuits once that
    node has been re-executed. Previously untested."""

    def test_replay_final_hex_short_circuits(self):
        from holonomy_lib.algebra import truncated_svd
        from holonomy_lib.spectral import laplacian
        A = torch.randn(1, 5, 5, dtype=torch.float64, generator=_seeded(125))
        A = (A + A.mT).abs() + torch.eye(5, dtype=torch.float64).unsqueeze(0)
        with provenance.record(cache_tensors=True) as reg:
            L = laplacian.combinatorial(A)
            truncated_svd(L, r=2, mode="exact")

        lap_hex = reg.where(
            op_id="holonomy_lib.spectral.laplacian.combinatorial",
        )[0].hex
        svd_hex = reg.where(
            op_id="holonomy_lib.algebra.linear.truncated_svd",
        )[0].hex
        # Ask for early stop at the SVD's main hex. The SVD is multi-
        # output so the result holds U/S/Vt under svd_hex:0/1/2.
        new = reg.replay(
            {lap_hex: torch.zeros_like(reg.get_tensor(lap_hex))},
            final_hex=svd_hex,
        )
        # Three SVD outputs returned; no extraneous downstream entries
        # (which would indicate the early-stop didn't fire).
        assert len(new) == 3
        assert all(k.split(":")[0] == svd_hex for k in new)

    def test_replay_final_hex_specific_output(self):
        """Asking for `svd_hex:1` (the singular values specifically)
        returns only that output, not all three."""
        from holonomy_lib.algebra import truncated_svd
        from holonomy_lib.spectral import laplacian
        A = torch.randn(1, 5, 5, dtype=torch.float64, generator=_seeded(126))
        A = (A + A.mT).abs() + torch.eye(5, dtype=torch.float64).unsqueeze(0)
        with provenance.record(cache_tensors=True) as reg:
            L = laplacian.combinatorial(A)
            truncated_svd(L, r=2, mode="exact")
        lap_hex = reg.where(
            op_id="holonomy_lib.spectral.laplacian.combinatorial",
        )[0].hex
        svd_hex = reg.where(
            op_id="holonomy_lib.algebra.linear.truncated_svd",
        )[0].hex
        S_key = f"{svd_hex}:1"
        new = reg.replay(
            {lap_hex: torch.zeros_like(reg.get_tensor(lap_hex))},
            final_hex=S_key,
        )
        assert list(new) == [S_key]


class TestParentsMultiOutput:
    """`parents()` extracts base hexes from `name=hex:i`-form input_hexes."""

    def test_parents_strips_output_index(self):
        from holonomy_lib.algebra import truncated_svd
        A = torch.randn(1, 5, 5, dtype=torch.float64, generator=_seeded(130))
        with provenance.record() as reg:
            U, S, Vt = truncated_svd(A, r=2, mode="exact")
            # Chain U → another op so parents() walks across hex:0 → svd hex
            from holonomy_lib.spectral import laplacian
            laplacian.combinatorial(U @ U.mT)
        comb_node = reg.where(
            op_id="holonomy_lib.spectral.laplacian.combinatorial",
        )[0]
        # The Laplacian's input was U @ U.mT (an unrecorded compute);
        # so its input is a leaf, not the SVD itself. Instead exercise
        # parents() with an input chain that produces multi-output edge.
        # Build a simple chain manually.
        from holonomy_lib.provenance import with_provenance

        @with_provenance("test.parents_multi", op_version="0.1")
        def emit_pair(M: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return M, M.mT

        @with_provenance("test.parents_consume", op_version="0.1")
        def consume_first(X: torch.Tensor) -> torch.Tensor:
            return X * 2

        with provenance.record() as reg2:
            pair = emit_pair(A)
            consume_first(pair[0])
        pair_hex = reg2.where(op_id="test.parents_multi")[0].hex
        cons_hex = reg2.where(op_id="test.parents_consume")[0].hex
        parents = reg2.parents(cons_hex)
        # parents() should resolve "X=pair_hex:0" → base pair_hex
        assert len(parents) == 1
        assert parents[0].hex == pair_hex


class TestTensorIdReuse:
    """Python recycles `id()` after garbage collection. Earlier
    versions of the registry keyed `_tensor_id_to_hex` on raw `id(t)`
    and returned the dead tensor's hex when the id was reused for a
    new tensor — silently producing wrong provenance hashes. The fix
    stores a weakref alongside the hex and treats stale entries as
    missing.
    """

    def test_id_reuse_does_not_corrupt_hex(self):
        A = torch.randn(1, 4, 4, dtype=torch.float64, generator=_seeded(95))
        A = (A + A.mT).abs()
        with provenance.record() as reg:
            # Each iteration's intermediate tensor is unreferenced after
            # the call returns — Python is free to reuse its id for the
            # next iteration's allocation.
            for i in range(4):
                laplacian.combinatorial(A + i * 0.1)
        # All four inputs differ in content → four distinct nodes.
        assert len(reg) == 4
        hexes = [n.hex for n in reg]
        assert len(set(hexes)) == 4


class TestTensorCacheBounds:
    """Cache bounding — `max_cache_size` (FIFO eviction) and
    `cache_ops` (selective caching by op_id)."""

    def test_max_cache_size_evicts_oldest(self):
        """With max_cache_size=N, only the latest N cached entries
        survive; earlier ones get evicted in FIFO order."""
        A = torch.randn(1, 4, 4, dtype=torch.float64, generator=_seeded(90))
        A = (A + A.mT).abs()
        # Run several combinatorial Laplacian calls on different inputs
        with provenance.record(
            cache_tensors=True, max_cache_size=2,
        ) as reg:
            for i in range(5):
                laplacian.combinatorial(A + i * 0.1)
        # Five recorded nodes, but cache holds at most 2.
        assert len(reg) == 5
        cached_count = sum(
            1 for n in reg if reg.get_tensor(n.hex) is not None
        )
        assert cached_count == 2, (
            f"cache should hold exactly max_cache_size=2 entries, "
            f"got {cached_count}"
        )

    def test_max_cache_size_keeps_most_recent(self):
        """The entries that survive are the most recent ones."""
        A = torch.randn(1, 4, 4, dtype=torch.float64, generator=_seeded(91))
        A = (A + A.mT).abs()
        with provenance.record(
            cache_tensors=True, max_cache_size=2,
        ) as reg:
            for i in range(4):
                laplacian.combinatorial(A + i * 0.1)
        # Nodes in recording order
        nodes = list(reg)
        # Last 2 nodes should have cache entries; first 2 should not.
        assert reg.get_tensor(nodes[-1].hex) is not None
        assert reg.get_tensor(nodes[-2].hex) is not None
        assert reg.get_tensor(nodes[0].hex) is None
        assert reg.get_tensor(nodes[1].hex) is None

    def test_cache_ops_selective(self):
        """cache_ops restricts caching to the named op_ids."""
        A = torch.randn(1, 5, 5, dtype=torch.float64, generator=_seeded(92))
        A = (A + A.mT).abs() + torch.eye(5, dtype=torch.float64).unsqueeze(0)
        with provenance.record(
            cache_ops=["holonomy_lib.spectral.laplacian.combinatorial"],
        ) as reg:
            laplacian.combinatorial(A)
            truncated_svd(A, r=2, mode="exact")
        comb_node = reg.where(
            op_id="holonomy_lib.spectral.laplacian.combinatorial",
        )[0]
        svd_node = reg.where(
            op_id="holonomy_lib.algebra.linear.truncated_svd",
        )[0]
        assert reg.get_tensor(comb_node.hex) is not None
        # truncated_svd is a multi-output op (hex:0, hex:1, hex:2)
        assert reg.get_tensor(f"{svd_node.hex}:0") is None

    def test_invalid_max_cache_size(self):
        with pytest.raises(ValueError, match="max_cache_size"):
            with provenance.record(max_cache_size=0) as _:
                pass


class TestGeneratorCanonicalization:
    """torch.Generator params canonicalize to (seed, device) so that two
    Generators with the same seed produce the same hex and replay can
    reconstruct them."""

    # Use a matrix large enough that randomized mode does not fall
    # back to exact (which would silently drop the generator from
    # params). `ell = r + oversample = 2 + 5 = 7` must be ≤ min(m, n);
    # 12×12 with r=2 satisfies this.
    _M_SHAPE = (1, 12, 12)
    _R = 2

    def test_same_seed_different_generator_same_hex(self):
        """Two Generators built from the same seed → same hex, even
        though their default str() reprs differ by memory address.
        """
        M = torch.randn(*self._M_SHAPE, dtype=torch.float64, generator=_seeded(80))
        g_a = _seeded(123)
        g_b = _seeded(123)
        # Sanity: the two Generators are distinct Python objects
        assert g_a is not g_b
        with provenance.record() as r1:
            truncated_svd(M, r=self._R, mode="randomized", generator=g_a)
        with provenance.record() as r2:
            truncated_svd(M, r=self._R, mode="randomized", generator=g_b)
        h1 = next(iter(r1)).hex
        h2 = next(iter(r2)).hex
        assert h1 == h2, (
            "same-seed generators must produce identical hexes; got "
            f"{h1!r} vs {h2!r}"
        )

    def test_different_seed_different_hex(self):
        M = torch.randn(*self._M_SHAPE, dtype=torch.float64, generator=_seeded(81))
        with provenance.record() as r1:
            truncated_svd(M, r=self._R, mode="randomized", generator=_seeded(1))
        with provenance.record() as r2:
            truncated_svd(M, r=self._R, mode="randomized", generator=_seeded(2))
        h1 = next(iter(r1)).hex
        h2 = next(iter(r2)).hex
        assert h1 != h2

    def test_params_carry_canonical_generator(self):
        """The serialized params should contain the canonical form
        (seed + device), not a memory-address string."""
        M = torch.randn(*self._M_SHAPE, dtype=torch.float64, generator=_seeded(82))
        with provenance.record() as reg:
            truncated_svd(
                M, r=self._R, mode="randomized", generator=_seeded(777),
            )
        params = next(iter(reg)).parsed_params()
        g = params["generator"]
        assert isinstance(g, dict)
        assert g["seed"] == 777
        assert "device" in g

    def test_replay_reconstructs_generator_param(self):
        """Replay can re-execute through an op that consumes a Generator.

        The canonical-form params reconstruct into a real Generator with
        the recorded seed, so the op can run. (Generator state past
        seeding doesn't survive replay — replay's docstring warns about
        stochastic ops — but reconstruction itself must not raise.)
        """
        M = torch.randn(*self._M_SHAPE, dtype=torch.float64, generator=_seeded(83))
        with provenance.record(cache_tensors=True) as reg:
            # Chain: laplacian.combinatorial(A) → truncated_svd(L, generator=g)
            A = M @ M.mT + torch.eye(M.shape[-1], dtype=M.dtype).unsqueeze(0)
            L = laplacian.combinatorial(A)
            truncated_svd(L, r=self._R, mode="randomized", generator=_seeded(42))

        # Substitute the upstream Laplacian → replay must re-execute the
        # randomized SVD, which means reconstructing its Generator param.
        lap_hex = reg.where(
            op_id="holonomy_lib.spectral.laplacian.combinatorial",
        )[0].hex
        fake_L = torch.zeros_like(reg.get_tensor(lap_hex))
        new = reg.replay({lap_hex: fake_L})
        # The SVD's outputs come back under its hex:i; the assertion is
        # just that replay completes without raising.
        svd_hex = reg.where(
            op_id="holonomy_lib.algebra.linear.truncated_svd",
        )[0].hex
        assert any(h.startswith(svd_hex) for h in new)


class TestVarArgsRejection:
    def test_rejects_var_positional(self):
        with pytest.raises(TypeError, match="var_positional"):
            @provenance.with_provenance("test.bad_var_args", op_version="0.1")
            def bad(*tensors):  # noqa: ARG001
                return tensors[0]

    def test_rejects_var_keyword(self):
        with pytest.raises(TypeError, match="var_keyword"):
            @provenance.with_provenance("test.bad_var_kwargs", op_version="0.1")
            def bad(x: torch.Tensor, **kw):  # noqa: ARG001
                return x


class TestHashAlgorithm:
    def test_sha256_is_always_available(self):
        A = torch.randn(1, 4, 4, dtype=torch.float64, generator=_seeded(60))
        A = (A + A.mT).abs()
        with provenance.record(hash_algorithm="sha256") as reg:
            laplacian.combinatorial(A)
        assert reg.hash_algorithm == "sha256"
        assert len(reg) == 1

    def test_unknown_algorithm_rejected(self):
        with pytest.raises(ValueError, match="hash_algorithm"):
            with provenance.record(hash_algorithm="md5") as _:
                pass

    def test_different_algorithms_produce_different_hexes(self):
        """sha256 and blake3 hash the same content to different hex values."""
        try:
            import blake3  # noqa: F401
        except ImportError:
            pytest.skip("blake3 not installed")
        A = torch.randn(1, 4, 4, dtype=torch.float64, generator=_seeded(61))
        A = (A + A.mT).abs()
        with provenance.record(hash_algorithm="sha256") as r1:
            laplacian.combinatorial(A)
        with provenance.record(hash_algorithm="blake3") as r2:
            laplacian.combinatorial(A)
        h1 = next(iter(r1)).hex
        h2 = next(iter(r2)).hex
        assert h1 != h2, "different hash algos must produce different hexes"


# --------------------------------------------------------------------
# Sketch hash mode (phase 1b)
# --------------------------------------------------------------------


class TestSketchHash:
    """`hash_mode='sketch'` trades crypto-grade content hashing for an
    O(SKETCH_SAMPLES)-bytes digest. Must be deterministic and must
    distinguish typical-research-scale perturbations.
    """

    def _record_sketch(self, A: torch.Tensor) -> str:
        with provenance.record(hash_mode="sketch") as reg:
            laplacian.combinatorial(A)
        return next(iter(reg)).hex

    def test_default_mode_is_full(self):
        """Default behavior must NOT change — sketch is opt-in."""
        with provenance.record() as reg:
            assert reg.hash_mode == "full"

    def test_sketch_is_deterministic(self):
        """Same tensor → same hex under sketch mode."""
        A = torch.randn(1, 16, 16, dtype=torch.float64, generator=_seeded(70))
        A = (A + A.mT).abs()
        h1 = self._record_sketch(A.clone())
        h2 = self._record_sketch(A.clone())
        assert h1 == h2

    def test_sketch_distinguishes_different_tensors(self):
        """Two unrelated tensors get different sketch hexes."""
        A = torch.randn(1, 16, 16, dtype=torch.float64, generator=_seeded(71))
        A = (A + A.mT).abs()
        B = torch.randn(1, 16, 16, dtype=torch.float64, generator=_seeded(72))
        B = (B + B.mT).abs()
        assert self._record_sketch(A) != self._record_sketch(B)

    def test_sketch_detects_perturbation(self):
        """A small additive perturbation changes the sum + std discriminators,
        so the sketch hex changes even if the strided samples happen to
        land on positions that didn't move. This is the empirical edge
        the sum/std discriminators protect.
        """
        A = torch.randn(1, 32, 32, dtype=torch.float64, generator=_seeded(73))
        A = (A + A.mT).abs()
        A_perturbed = A.clone()
        A_perturbed[0, 5, 7] += 0.01  # tiny single-element change
        assert self._record_sketch(A) != self._record_sketch(A_perturbed)

    def test_sketch_versus_full_produce_different_hexes(self):
        """A registry in sketch mode and one in full mode hash the same
        tensor to different hexes (different paths through the hasher).
        Recordings are not interchangeable.
        """
        A = torch.randn(1, 8, 8, dtype=torch.float64, generator=_seeded(74))
        A = (A + A.mT).abs()
        with provenance.record(hash_mode="full") as r_full:
            laplacian.combinatorial(A.clone())
        with provenance.record(hash_mode="sketch") as r_sketch:
            laplacian.combinatorial(A.clone())
        assert next(iter(r_full)).hex != next(iter(r_sketch)).hex

    def test_invalid_mode_rejected(self):
        with pytest.raises(ValueError, match="hash_mode"):
            with provenance.record(hash_mode="bogus") as _:  # type: ignore[arg-type]
                pass

    def test_sketch_mode_round_trips_through_save_load(self):
        """Saved sketch registries reload in sketch mode."""
        A = torch.randn(1, 4, 4, dtype=torch.float64, generator=_seeded(75))
        A = (A + A.mT).abs()
        with provenance.record(hash_mode="sketch") as reg:
            laplacian.combinatorial(A)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "reg.json"
            reg.save(path)
            reloaded = provenance.ProvenanceRegistry.load(path)
        assert reloaded.hash_mode == "sketch"

    def test_sketch_collision_rate_on_random_tensors(self):
        """Empirical collision check on N=200 random tensors. We don't
        need 10⁴ here — that's research-scale, takes too long in CI.
        200 distinct tensors should give 200 distinct hexes; if they
        don't, the sketch is structurally broken.
        """
        hexes: set[str] = set()
        for seed in range(200):
            t = torch.randn(16, 16, dtype=torch.float64, generator=_seeded(seed + 1000))
            with provenance.record(hash_mode="sketch") as reg:
                laplacian.combinatorial(t.abs())
            hexes.add(next(iter(reg)).hex)
        assert len(hexes) == 200, (
            f"sketch hash collision: {200 - len(hexes)} collisions in 200 tensors"
        )

    def test_sketch_handles_tiny_tensors(self):
        """SKETCH_SAMPLES=64 vs a 4-element tensor: stride=1, all 4 elements
        sampled. Must not raise and must distinguish two different tiny
        tensors.
        """
        a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
        b = torch.tensor([[1.0, 2.0], [3.0, 5.0]], dtype=torch.float64)
        with provenance.record(hash_mode="sketch") as ra:
            laplacian.combinatorial(a.unsqueeze(0))
        with provenance.record(hash_mode="sketch") as rb:
            laplacian.combinatorial(b.unsqueeze(0))
        assert next(iter(ra)).hex != next(iter(rb)).hex


# --------------------------------------------------------------------
# Disk-backed tensor cache (phase 1c)
# --------------------------------------------------------------------


class TestDiskCache:
    """`cache_to_disk=path` mirrors the in-memory cache to disk. Memory
    eviction from max_cache_size only drops the in-memory copy; disk
    survives and get_tensor() reloads on demand.
    """

    def _build(self, seed: int):
        A = torch.randn(1, 4, 4, dtype=torch.float64, generator=_seeded(seed))
        return (A + A.mT).abs()

    def test_cache_to_disk_implies_cache_tensors(self):
        """Setting cache_to_disk auto-enables caching — no separate kwarg."""
        with tempfile.TemporaryDirectory() as d:
            with provenance.record(cache_to_disk=d) as reg:
                pass
            assert reg._cache_tensors is True

    def test_round_trip_through_disk(self):
        """A tensor written to disk loads back equal."""
        with tempfile.TemporaryDirectory() as d:
            A = self._build(80)
            with provenance.record(cache_to_disk=d) as reg:
                laplacian.combinatorial(A)
            node = next(iter(reg))
            # Force a disk reload by clearing the in-memory cache.
            in_mem = reg.get_tensor(node.hex)
            assert in_mem is not None
            reg._tensor_cache.clear()
            assert reg.get_tensor(node.hex) is not None
            torch.testing.assert_close(reg.get_tensor(node.hex), in_mem)

    def test_disk_files_use_safe_filenames(self):
        """Multi-output keys with `:` separators map to `__` on disk."""
        with tempfile.TemporaryDirectory() as d:
            A = self._build(81)
            with provenance.record(cache_to_disk=d) as reg:
                # truncated_svd is a multi-output op (U, S, Vt) → its
                # cache keys include ':0', ':1', ':2' suffixes.
                truncated_svd(A, r=2)
            disk_files = list(Path(d).iterdir())
            assert disk_files, "no files written to disk cache"
            # No filename should contain a raw ':' (Windows-incompatible).
            for f in disk_files:
                assert ":" not in f.name, f"unsafe filename: {f.name}"

    def test_lru_eviction_retains_disk_copy(self):
        """When max_cache_size evicts a tensor from memory, the disk
        copy persists; subsequent get_tensor() reloads it.
        """
        with tempfile.TemporaryDirectory() as d:
            with provenance.record(
                cache_to_disk=d, max_cache_size=1,
            ) as reg:
                laplacian.combinatorial(self._build(82))
                laplacian.combinatorial(self._build(83))
            hexes = [n.hex for n in reg]
            assert len(hexes) == 2
            # max_cache_size=1: only one is in memory at end of recording.
            assert len(reg._tensor_cache) == 1
            # But BOTH should be on disk and retrievable.
            for h in hexes:
                assert reg.get_tensor(h) is not None

    def test_clear_keeps_disk_by_default(self):
        """clear() without args only drops in-memory; disk persists."""
        with tempfile.TemporaryDirectory() as d:
            with provenance.record(cache_to_disk=d) as reg:
                laplacian.combinatorial(self._build(84))
            reg.clear()
            assert len(reg._tensor_cache) == 0
            assert len(list(Path(d).iterdir())) > 0  # disk untouched

    def test_clear_with_delete_disk_removes_files(self):
        """clear(delete_disk=True) removes the .pt files."""
        with tempfile.TemporaryDirectory() as d:
            with provenance.record(cache_to_disk=d) as reg:
                laplacian.combinatorial(self._build(85))
            assert list(Path(d).iterdir())  # something written
            reg.clear(delete_disk=True)
            # No .pt files remain (directory itself may still exist).
            pt_files = [f for f in Path(d).iterdir() if f.suffix == ".pt"]
            assert pt_files == []

    def test_save_and_load_preserves_disk_attachment(self):
        """A saved disk-cached registry reloads pointing at the same
        cache directory; tensors are still retrievable.
        """
        with tempfile.TemporaryDirectory() as cache_dir:
            with tempfile.TemporaryDirectory() as meta_dir:
                A = self._build(86)
                with provenance.record(cache_to_disk=cache_dir) as reg:
                    laplacian.combinatorial(A)
                meta = Path(meta_dir) / "reg.json"
                reg.save(meta)
                reloaded = provenance.ProvenanceRegistry.load(meta)
                assert str(reloaded._disk_cache_dir) == str(
                    Path(cache_dir).expanduser().resolve()
                )
                # Tensor cache survives the round-trip via disk.
                node = next(iter(reloaded))
                assert reloaded.get_tensor(node.hex) is not None


# --------------------------------------------------------------------
# op_version drift detection on load() (phase 2b)
# --------------------------------------------------------------------


class TestLoadDrift:
    """When the currently-installed op_version differs from the
    version recorded at save-time, load() flags the drift so users
    don't silently replay against semantics that have changed."""

    def _make_registry(self) -> tuple[provenance.ProvenanceRegistry, str]:
        """Helper: record one laplacian call, return (registry, op_id)."""
        A = torch.randn(1, 4, 4, dtype=torch.float64, generator=_seeded(90))
        A = (A + A.mT).abs()
        with provenance.record() as reg:
            laplacian.combinatorial(A)
        return reg, "holonomy_lib.spectral.laplacian.combinatorial"

    def test_no_warning_when_versions_match(self):
        """The happy path: same versions on both ends, silent load."""
        reg, _ = self._make_registry()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "reg.json"
            reg.save(path)
            with warnings.catch_warnings():
                warnings.simplefilter("error", provenance.ProvenanceVersionWarning)
                provenance.ProvenanceRegistry.load(path)  # must not raise

    def test_warning_on_version_drift(self):
        """Monkeypatch OP_REGISTRY to bump a version after save; load
        must emit ProvenanceVersionWarning naming the drifted op.
        """
        reg, op_id = self._make_registry()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "reg.json"
            reg.save(path)
            original_fn, original_ver = provenance.OP_REGISTRY[op_id]
            try:
                provenance.OP_REGISTRY[op_id] = (original_fn, "99.0-drifted")
                with pytest.warns(
                    provenance.ProvenanceVersionWarning,
                    match=op_id,
                ):
                    provenance.ProvenanceRegistry.load(path)
            finally:
                provenance.OP_REGISTRY[op_id] = (original_fn, original_ver)

    def test_strict_raises_on_drift(self):
        """strict=True converts the warning into a ValueError."""
        reg, op_id = self._make_registry()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "reg.json"
            reg.save(path)
            original_fn, original_ver = provenance.OP_REGISTRY[op_id]
            try:
                provenance.OP_REGISTRY[op_id] = (original_fn, "99.0-drifted")
                with pytest.raises(ValueError, match="differ from"):
                    provenance.ProvenanceRegistry.load(path, strict=True)
            finally:
                provenance.OP_REGISTRY[op_id] = (original_fn, original_ver)

    def test_unknown_op_flagged_in_warning(self):
        """A node whose op_id is no longer registered (e.g., module
        not imported) is listed in the warning alongside drifted ops.
        """
        reg, op_id = self._make_registry()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "reg.json"
            reg.save(path)
            original = provenance.OP_REGISTRY.pop(op_id)
            try:
                with pytest.warns(
                    provenance.ProvenanceVersionWarning,
                    match="not currently registered",
                ):
                    provenance.ProvenanceRegistry.load(path)
            finally:
                provenance.OP_REGISTRY[op_id] = original


# --------------------------------------------------------------------
# Visualization + diff renderer (phase 3a)
# --------------------------------------------------------------------


class TestMermaidExport:
    def test_mermaid_contains_flowchart_header(self):
        A = torch.randn(1, 4, 4, dtype=torch.float64, generator=_seeded(100))
        A = (A + A.mT).abs()
        with provenance.record() as reg:
            laplacian.combinatorial(A)
        out = reg.to_mermaid()
        assert out.startswith("flowchart TD")

    def test_mermaid_lists_nodes_and_edges(self):
        """A 2-op chain produces one node line per op and one edge
        line connecting them."""
        A = torch.randn(1, 5, 5, dtype=torch.float64, generator=_seeded(101))
        A = (A + A.mT).abs() + torch.eye(5, dtype=torch.float64).unsqueeze(dim=0)
        with provenance.record() as reg:
            L = laplacian.combinatorial(A)
            truncated_svd(L, r=2, mode="exact")
        out = reg.to_mermaid()
        lap = reg.where(op_id="holonomy_lib.spectral.laplacian.combinatorial")[0]
        svd = reg.where(op_id="holonomy_lib.algebra.linear.truncated_svd")[0]
        # Both nodes appear with their hex IDs.
        assert lap.hex in out
        assert svd.hex in out
        # Edge from laplacian to svd is present.
        assert f"{lap.hex} --> {svd.hex}" in out

    def test_mermaid_safe_on_empty_registry(self):
        with provenance.record() as reg:
            pass
        out = reg.to_mermaid()
        # Returns valid (if empty) Mermaid source.
        assert "flowchart TD" in out


class TestGraphvizExport:
    def test_dot_contains_digraph_header(self):
        A = torch.randn(1, 4, 4, dtype=torch.float64, generator=_seeded(102))
        A = (A + A.mT).abs()
        with provenance.record() as reg:
            laplacian.combinatorial(A)
        out = reg.to_graphviz()
        assert out.startswith("digraph provenance {")
        assert out.endswith("}")

    def test_dot_lists_nodes_and_edges(self):
        A = torch.randn(1, 5, 5, dtype=torch.float64, generator=_seeded(103))
        A = (A + A.mT).abs() + torch.eye(5, dtype=torch.float64).unsqueeze(dim=0)
        with provenance.record() as reg:
            L = laplacian.combinatorial(A)
            truncated_svd(L, r=2, mode="exact")
        out = reg.to_graphviz()
        lap = reg.where(op_id="holonomy_lib.spectral.laplacian.combinatorial")[0]
        svd = reg.where(op_id="holonomy_lib.algebra.linear.truncated_svd")[0]
        assert f'"{lap.hex}"' in out
        assert f'"{svd.hex}"' in out
        assert f'"{lap.hex}" -> "{svd.hex}";' in out


class TestDiffSummary:
    def test_identical_registries_report_no_differences(self):
        A = torch.randn(1, 4, 4, dtype=torch.float64, generator=_seeded(110))
        A = (A + A.mT).abs()
        with provenance.record() as r1:
            laplacian.combinatorial(A)
        with provenance.record() as r2:
            laplacian.combinatorial(A.clone())
        out = r1.diff_summary(r2)
        # Same content, identical hexes → all shared, no drift.
        assert "Cache hits" in out
        # Empty drift / new / gone sections shouldn't appear.
        assert "Drift" not in out
        assert "Only in self" not in out
        assert "Only in other" not in out

    def test_truly_identical_registries(self):
        """An empty self vs empty other returns the no-diff sentinel."""
        with provenance.record() as r1:
            pass
        with provenance.record() as r2:
            pass
        assert r1.diff_summary(r2) == "(No differences.)"

    def test_drift_categorizes_same_op_different_hex(self):
        """Same op_id called on different inputs → categorized as drift,
        not as new/gone.
        """
        A = torch.randn(1, 4, 4, dtype=torch.float64, generator=_seeded(111))
        A = (A + A.mT).abs()
        B = torch.randn(1, 4, 4, dtype=torch.float64, generator=_seeded(112))
        B = (B + B.mT).abs()
        with provenance.record() as r1:
            laplacian.combinatorial(A)
        with provenance.record() as r2:
            laplacian.combinatorial(B)
        out = r1.diff_summary(r2)
        assert "Drift" in out
        assert "holonomy_lib.spectral.laplacian.combinatorial" in out
        # Not classified as Only-in-self or Only-in-other (which would
        # mean the op_id was unique to one side).
        assert "Only in self" not in out
        assert "Only in other" not in out

    def test_only_in_self_categorizes_unique_op_id(self):
        """An op that only ran on the self side appears in
        'Only in self'.
        """
        A = torch.randn(1, 4, 4, dtype=torch.float64, generator=_seeded(113))
        A = (A + A.mT).abs() + torch.eye(4, dtype=torch.float64).unsqueeze(dim=0)
        with provenance.record() as r1:
            L = laplacian.combinatorial(A)
            truncated_svd(L, r=2, mode="exact")
        with provenance.record() as r2:
            laplacian.combinatorial(A.clone())
        out = r1.diff_summary(r2)
        assert "Only in self" in out
        assert "holonomy_lib.algebra.linear.truncated_svd" in out


# --------------------------------------------------------------------
# ancestors_with_tensors convenience (phase 3b)
# --------------------------------------------------------------------


class TestAncestorsWithTensors:
    def test_returns_node_and_tensor_pairs(self):
        """For a 2-op chain, ancestors_with_tensors of the leaf returns
        the upstream node + its cached tensor."""
        A = torch.randn(1, 5, 5, dtype=torch.float64, generator=_seeded(120))
        A = (A + A.mT).abs() + torch.eye(5, dtype=torch.float64).unsqueeze(dim=0)
        with provenance.record(cache_tensors=True) as reg:
            L = laplacian.combinatorial(A)
            U, S, Vt = truncated_svd(L, r=2, mode="exact")
        lap_node = reg.where(
            op_id="holonomy_lib.spectral.laplacian.combinatorial",
        )[0]
        svd_node = reg.where(
            op_id="holonomy_lib.algebra.linear.truncated_svd",
        )[0]
        ancestors = reg.ancestors_with_tensors(svd_node.hex)
        assert lap_node.hex in ancestors
        anc_node, anc_tensor = ancestors[lap_node.hex]
        assert anc_node is lap_node
        assert anc_tensor is not None
        torch.testing.assert_close(anc_tensor, L)

    def test_tensor_is_none_when_not_cached(self):
        """Without cache_tensors=True the tensor slot is None but the
        node info is still returned.
        """
        A = torch.randn(1, 4, 4, dtype=torch.float64, generator=_seeded(121))
        A = (A + A.mT).abs() + torch.eye(4, dtype=torch.float64).unsqueeze(dim=0)
        with provenance.record() as reg:  # no caching
            L = laplacian.combinatorial(A)
            truncated_svd(L, r=2, mode="exact")
        svd_hex = reg.where(
            op_id="holonomy_lib.algebra.linear.truncated_svd",
        )[0].hex
        ancestors = reg.ancestors_with_tensors(svd_hex)
        for _hex, (node, tensor) in ancestors.items():
            assert node is reg._nodes[_hex]
            assert tensor is None


# --------------------------------------------------------------------
# LLM-context summary (phase 4a)
# --------------------------------------------------------------------


class TestLLMContext:
    def _build(self, seed: int):
        A = torch.randn(1, 5, 5, dtype=torch.float64, generator=_seeded(seed))
        return (A + A.mT).abs() + torch.eye(5, dtype=torch.float64).unsqueeze(dim=0)

    def test_header_lists_counts_and_hash_mode(self):
        with provenance.record(cache_tensors=True) as reg:
            laplacian.combinatorial(self._build(130))
        out = reg.to_llm_context()
        assert "Provenance registry: 1 ops" in out
        assert "hash_mode=full" in out

    def test_lists_ops_by_op_id_with_counts(self):
        """3 laplacian calls + 1 SVD call → grouped counts in output."""
        with provenance.record() as reg:
            for s in (130, 131, 132):
                L = laplacian.combinatorial(self._build(s))
            truncated_svd(L, r=2, mode="exact")
        out = reg.to_llm_context()
        assert "Ops by op_id:" in out
        assert "holonomy_lib.spectral.laplacian.combinatorial x 3" in out
        assert "holonomy_lib.algebra.linear.truncated_svd x 1" in out

    def test_lists_roots_and_leaves(self):
        """In a 2-op chain, laplacian is the root and truncated_svd is
        a leaf (truncated_svd's single output isn't consumed)."""
        with provenance.record() as reg:
            L = laplacian.combinatorial(self._build(133))
            truncated_svd(L, r=2, mode="exact")
        lap = reg.where(op_id="holonomy_lib.spectral.laplacian.combinatorial")[0]
        svd = reg.where(op_id="holonomy_lib.algebra.linear.truncated_svd")[0]
        out = reg.to_llm_context()
        assert "Roots:" in out
        assert lap.hex in out
        assert "Leaves:" in out
        assert svd.hex in out

    def test_max_ops_caps_output(self):
        """With max_ops=1 the ops list shouldn't list both ops."""
        with provenance.record() as reg:
            for s in (140, 141, 142):
                L = laplacian.combinatorial(self._build(s))
            truncated_svd(L, r=2, mode="exact")
        out = reg.to_llm_context(max_ops=1)
        assert "...and 1 more op_id(s)" in out

    def test_empty_registry_returns_header_only(self):
        with provenance.record() as reg:
            pass
        out = reg.to_llm_context()
        assert out == "Provenance registry: 0 ops, 0 cached tensors, hash_mode=full"
