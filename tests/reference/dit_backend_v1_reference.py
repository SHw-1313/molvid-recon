"""Explicit v1 backend reference used by B parity tests.

Reference source commit: a22f60c4ffd1f502ab300352a00eafe841b1f8d2.
The implementation lives in MolecularDiT's explicit reference dispatch. This
module prevents parity tests from silently selecting the optimized runtime.
"""

from __future__ import annotations

from typing import Any

from module.molecular_dit import MolecularDiT
from module.state_detail_latent_adapter import StateDetailLatentAdapter


REFERENCE_SOURCE_COMMIT = "a22f60c4ffd1f502ab300352a00eafe841b1f8d2"
REFERENCE_EXECUTION_BACKEND = "reference"


def build_reference_model(
    adapter: StateDetailLatentAdapter,
    *,
    scalar_width: int = 256,
    vector_width: int = 128,
    depth: int = 4,
    heads: int = 8,
    ffn_multiplier: int = 4,
    dropout: float = 0.0,
    **kwargs: Any,
) -> MolecularDiT:
    """Construct the frozen reference execution path explicitly."""

    return MolecularDiT(
        adapter=adapter,
        scalar_width=scalar_width,
        vector_width=vector_width,
        depth=depth,
        heads=heads,
        ffn_multiplier=ffn_multiplier,
        dropout=dropout,
        execution_backend=REFERENCE_EXECUTION_BACKEND,
        **kwargs,
    )
