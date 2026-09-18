"""Four-field latent errors and explicit codec-oracle generation gaps."""

from __future__ import annotations

from typing import Any, Mapping

from ..flow.objective import four_field_loss
from ..latent.types import LatentBatch, LatentFields


def latent_field_errors(
    generated: LatentFields,
    oracle: LatentFields,
    batch: LatentBatch,
) -> dict[str, Any]:
    """Report future-only valid-coefficient MSE without hiding empty fields."""

    if batch.observed_mask is None:
        raise ValueError("latent evaluation requires an observation mask")
    loss = four_field_loss(generated, oracle, batch)
    return {
        "total_mse": float(loss.total.detach().cpu()),
        "fields": {
            name: {
                "mse": float(loss.fields[name].detach().cpu()),
                "valid_elements": loss.valid_elements[name],
            }
            for name in generated.names()
        },
        "aggregation": "equal_mean_of_four_masked_field_mse",
    }


def _numeric_gap(generated: Any, oracle: Any) -> Any:
    if isinstance(generated, Mapping) and isinstance(oracle, Mapping):
        return {
            key: _numeric_gap(generated[key], oracle[key])
            for key in generated
            if key in oracle
        }
    if isinstance(generated, (int, float)) and isinstance(oracle, (int, float)):
        if isinstance(generated, bool) or isinstance(oracle, bool):
            return None
        return float(generated) - float(oracle)
    return None


def oracle_vs_generated(
    oracle_metrics: Mapping[str, Any],
    generated_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    """Keep the reconstruction floor distinct from the generation error."""

    return {
        "codec_oracle": dict(oracle_metrics),
        "generated_result": dict(generated_metrics),
        "generation_gap": _numeric_gap(generated_metrics, oracle_metrics),
    }
