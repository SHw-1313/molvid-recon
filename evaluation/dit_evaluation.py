"""Report plumbing that keeps codec-oracle and generated-latent results separate."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional
import time

import torch
from torch import Tensor

from .codec_evaluation import _metrics, aligned_rmsf_metrics, dynamic_acf_metrics
from module.state_detail_latent_adapter import LatentFieldSet


DIT_EVALUATION_SCHEMA = "pvb.dit.state_detail.evaluation.v1"


def _coordinates(decoded: Any) -> Tensor:
    for name in ("x_hat", "coordinates", "x"):
        if hasattr(decoded, name):
            value = getattr(decoded, name)
            if isinstance(value, Tensor):
                return value
    if isinstance(decoded, Tensor):
        return decoded
    raise TypeError("decoder output does not expose coordinates")


def _subtract_numeric(generated: Any, oracle: Any) -> Any:
    if isinstance(generated, Mapping) and isinstance(oracle, Mapping):
        return {
            key: _subtract_numeric(generated[key], oracle[key])
            for key in generated
            if key in oracle
        }
    if isinstance(generated, (int, float)) and isinstance(oracle, (int, float)):
        return float(generated) - float(oracle)
    return None


def trajectory_metrics(
    prediction: Tensor,
    target: Tensor,
    batch: Any,
    *,
    history_frames: Optional[int] = None,
) -> dict[str, Any]:
    """Reuse the corrected codec metrics and attach pilot-only report fields."""
    result = dict(_metrics(prediction, target, batch, history_frames=history_frames))
    if history_frames is None:
        result["rmsf"] = aligned_rmsf_metrics(prediction, target, batch)
        result["dynamic"] = dynamic_acf_metrics(prediction, target, batch)
        result["observation_boundary"] = {}
    else:
        temporal = result["temporal"]
        result["rmsf"] = {
            "observed": temporal["observed"]["rmsf"],
            "future": temporal["future"]["rmsf"],
            "full_diagnostic": temporal["full_diagnostic"]["rmsf"],
        }
        result["dynamic"] = {
            "observed": temporal["observed"]["dynamic"],
            "future": temporal["future"]["dynamic"],
            "full_diagnostic": temporal["full_diagnostic"]["dynamic"],
        }
        result["observation_boundary"] = result["boundary"]
    return result


def diversity_diagnostics(samples: Tensor) -> dict[str, Any]:
    """Summarize sample-to-sample spread without selecting a best-of-N result."""
    if samples.ndim != 4 or samples.shape[-1] != 3:
        raise ValueError("samples must have shape [S,T,N,3]")
    count = int(samples.shape[0])
    if count < 2:
        return {"sample_count": count, "pairwise_rmsd": None, "primary_metric": "not_applicable"}
    pairs = []
    for first in range(count):
        for second in range(first + 1, count):
            pairs.append((samples[first] - samples[second]).square().mean().sqrt())
    return {
        "sample_count": count,
        "pairwise_rmsd": float(torch.stack(pairs).mean().detach().cpu()),
        "primary_metric": "not_best_of_n",
    }


@dataclass(frozen=True)
class DiTEvaluationResult:
    codec_oracle: Mapping[str, Any]
    generated_result: Mapping[str, Any]
    generation_gap: Mapping[str, Any]
    latent_flow_loss: Mapping[str, float]
    runtime: Mapping[str, Any]
    diversity: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = DIT_EVALUATION_SCHEMA

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "codec_oracle": dict(self.codec_oracle),
            "generated_result": dict(self.generated_result),
            "generation_gap": dict(self.generation_gap),
            "latent_flow_loss": dict(self.latent_flow_loss),
            "runtime": dict(self.runtime),
            "diversity": dict(self.diversity),
        }


@torch.no_grad()
def evaluate_codec_oracle(
    codec: Any,
    latent: Any,
    batch: Any,
    *,
    device: str | torch.device | None = None,
    history_frames: Optional[int] = None,
) -> dict[str, Any]:
    codec.eval()
    decoded = codec.decode(latent)
    prediction = _coordinates(decoded)
    target = torch.as_tensor(batch.x, device=prediction.device, dtype=prediction.dtype)
    return trajectory_metrics(prediction, target, batch, history_frames=history_frames)


@torch.no_grad()
def evaluate_generated_latent(
    codec: Any,
    generated_latent: Any,
    batch: Any,
    *,
    history_frames: Optional[int] = None,
) -> dict[str, Any]:
    codec.eval()
    start = time.perf_counter()
    decoded = codec.decode(generated_latent)
    prediction = _coordinates(decoded)
    elapsed = time.perf_counter() - start
    target = torch.as_tensor(batch.x, device=prediction.device, dtype=prediction.dtype)
    result = trajectory_metrics(prediction, target, batch, history_frames=history_frames)
    result["decode_runtime_s"] = elapsed
    return result


def evaluate_oracle_vs_generated(
    codec: Any,
    *,
    oracle_latent: Any,
    generated_latent: Any,
    batch: Any,
    latent_flow_loss: Optional[Mapping[str, float]] = None,
    trunk_runtime: Optional[Mapping[str, Any]] = None,
    diversity: Optional[Mapping[str, Any]] = None,
    history_frames: Optional[int] = None,
) -> DiTEvaluationResult:
    oracle = evaluate_codec_oracle(
        codec, oracle_latent, batch, history_frames=history_frames
    )
    generated = evaluate_generated_latent(
        codec, generated_latent, batch, history_frames=history_frames
    )
    return DiTEvaluationResult(
        codec_oracle=oracle,
        generated_result=generated,
        generation_gap=_subtract_numeric(generated, oracle),
        latent_flow_loss=dict(latent_flow_loss or {}),
        runtime=dict(trunk_runtime or {}),
        diversity=dict(diversity or {}),
    )


def report_markdown(report: Mapping[str, Any]) -> str:
    """Render a compact review-safe report without collapsing the oracle gap."""
    lines = [
        "# State/detail DiT evaluation",
        "",
        "Codec-oracle and generated-latent results are reported separately.",
        "",
        "| Quantity | Value |",
        "|---|---:|",
    ]
    for section, label in (
        ("codec_oracle", "Codec oracle"),
        ("generated_result", "Generated result"),
        ("generation_gap", "Generation gap"),
    ):
        section_value = report.get(section, {})
        future = section_value.get("future", {}) if isinstance(section_value, Mapping) else {}
        if isinstance(future, Mapping):
            lines.append(f"| {label} future aligned RMSD | {future.get('aligned_rmsd', 'n/a')} |")
            lines.append(f"| {label} future dRMSD | {future.get('drmsd', 'n/a')} |")
    flow = report.get("latent_flow_loss", {})
    if isinstance(flow, Mapping):
        for key, value in flow.items():
            lines.append(f"| Latent flow {key} | {value} |")
    return "\n".join(lines) + "\n"


__all__ = [
    "DIT_EVALUATION_SCHEMA",
    "DiTEvaluationResult",
    "evaluate_codec_oracle",
    "evaluate_generated_latent",
    "evaluate_oracle_vs_generated",
    "diversity_diagnostics",
    "report_markdown",
    "trajectory_metrics",
]
