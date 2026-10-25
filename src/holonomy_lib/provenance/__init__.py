# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Synoros

"""holonomy_lib.provenance — content-addressable hex provenance for math primitives.

The point of this module is to give mechanistic-interpretability researchers
a library where activation tracing, ablation, and circuit analysis are
*native* operations rather than something you bolt on with hooks. Every
primitive decorated with `@with_provenance` emits, when called inside a
`record()` context, a hex-fingerprint node in a Merkle DAG. The hex is
computed deterministically as

    hex(call) = sha256(op_id + op_version + canonical(params)
                       + [hex(input) for input in inputs])

so the same operation on the same inputs always produces the same hex.
For user-supplied tensors with no upstream op, hex = content hash of
the tensor bytes (+ shape + dtype). The DAG composes from there.

Interop targets — designed-in from the start:
  - NetworkX `DiGraph` export (`registry.to_networkx()`) for any graph
    analysis tool.
  - Pandas DataFrame export (`registry.to_dataframe()`) for SAELens-style
    training over `(op_id, params, hex, output_shape)` records.
  - JSON-friendly `to_dict()` for arbitrary downstream consumption.
  - Substitution-at-call (`registry.substitute({hex: replacement})`)
    for TransformerLens-style activation patching applied to math
    primitives rather than transformer layers.

What this module does NOT try to be:
  - A neural-network hooks library (TransformerLens / nnsight already
    exist; this complements them at the math-primitive layer).
  - A full reactive computation graph (no auto-replay of downstream ops
    on substitution — caller re-runs).
  - An MLOps experiment tracker (MLflow / W&B exist for that).
"""

from holonomy_lib.provenance.protocol import (
    OP_REGISTRY,
    ProvenanceNode,
    ProvenanceRegistry,
    ProvenanceVersionWarning,
    record,
    register_provenance_class,
    with_provenance,
)

__all__ = [
    "OP_REGISTRY",
    "ProvenanceNode",
    "ProvenanceRegistry",
    "ProvenanceVersionWarning",
    "record",
    "register_provenance_class",
    "with_provenance",
]
