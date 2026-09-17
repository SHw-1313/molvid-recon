#!/usr/bin/env python
"""Run bounded diagnostics against the existing R2/R4 pilot checkpoints.

The command is inference-only.  It opens train/validation stores, never builds
the test store, never calls an optimizer step, and writes a new diagnostics
run directory.  The runner imports the existing pilot loader solely to keep
codec, statistics, and checkpoint contracts identical to the completed pilot.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.clip_dataset import ClipBatch, collate_clip_records
from evaluation.dit_diagnostics import (
    DIAGNOSTICS_SCHEMA,
    FORECAST_LENGTHS,
    HISTORY_FRAMES,
    PERTURBATION_SCALES,
    TAU_MIDPOINTS,
    aggregate_rows,
    diversity_summary,
    field_swap_fields,
    fixed_tau_flow_diagnostics,
    json_safe,
    latent_block_state_persistence,
    latent_norm_comparison,
    latent_summary,
    observed_coordinate_baseline,
    observed_coordinate_scaffold,
    parse_sample_id,
    perturb_normalized_oracle,
    stable_seed,
    trajectory_metric_record,
)
from module.latent_rectified_flow import LatentFieldSet, generate_state_detail_latent
from module.state_detail_latent_adapter import (
    LatentStatistics,
    StateDetailLatentAdapter,
    build_observation_condition,
)
from trainer.dit_trainer import DiTTrainer
from scripts.run_state_detail_dit_pilot import (
    HISTORY_SCHEDULE,
    _build_trainer,
    _load_approved_codec,
    _load_data,
    _make_validation_plan,
    _sha256,
)


DEFAULT_CONFIG = PROJECT_ROOT / "config/dit_pilot_diagnostics_v1.yaml"


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(json_safe(value), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _write_yaml(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(yaml.safe_dump(dict(value), sort_keys=False))
    os.replace(temporary, path)


def _append_jsonl(handle: Any, value: Mapping[str, Any]) -> None:
    handle.write(json.dumps(json_safe(value), sort_keys=True) + "\n")
    handle.flush()


def _load_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, Mapping):
        raise ValueError("diagnostics config must be a mapping")
    return dict(value)


def _resolve_config_path(value: str | Path, *, base: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path)


def _require_cuda(device: torch.device) -> None:
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("diagnostics numeric stages require an actual CUDA device")


def _cuda_info(device: torch.device) -> dict[str, Any]:
    _require_cuda(device)
    return {
        "requested_device": str(device),
        "actual_device": str(torch.cuda.current_device()),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "device_name": torch.cuda.get_device_name(device),
        "device_uuid": str(torch.cuda.get_device_properties(device).uuid)
        if hasattr(torch.cuda.get_device_properties(device), "uuid")
        else None,
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "inference_dtype": "torch.float32 science metrics; torch.bfloat16 DiT autocast",
    }


def _candidate_paths(config: Mapping[str, Any], name: str) -> dict[str, Path]:
    pilot_root = _resolve_config_path(config["pilot_root"], base=PROJECT_ROOT)
    candidate = config["candidates"][name]
    return {
        "best_checkpoint": _resolve_config_path(candidate["best_checkpoint"], base=pilot_root),
        "statistics": _resolve_config_path(candidate["statistics"], base=pilot_root),
        "codec_result": _resolve_config_path(candidate["codec_result"], base=PROJECT_ROOT),
        "codec_checkpoint": _resolve_config_path(candidate["codec_checkpoint"], base=PROJECT_ROOT),
    }


def _input_manifest(config: Mapping[str, Any]) -> dict[str, Any]:
    manifest_root = _resolve_config_path(config["manifest_root"], base=PROJECT_ROOT).resolve()
    records: dict[str, Any] = {
        "schema": "pvb.dit.state_detail.diagnostics.inputs.v1",
        "manifest_root": str(manifest_root),
        "manifest": {},
        "materialization": {},
        "candidates": {},
        "test_opened": False,
    }
    for name in ("manifest.json", "materialization.json"):
        path = manifest_root / name
        if not path.is_file():
            raise FileNotFoundError(f"required frozen input is missing: {path}")
        records[name.removesuffix(".json")] = {
            "path": str(path),
            "sha256": _sha256(path),
            "metadata": json.loads(path.read_text()),
        }
    manifest = records["manifest"]["metadata"]
    if manifest.get("test_sampling", {}).get("opened") is not False:
        raise RuntimeError("frozen manifest does not certify unopened test sampling")
    for name in ("ratio2_state_detail", "ratio4_state_detail"):
        paths = _candidate_paths(config, name)
        summary_path = paths["best_checkpoint"].parent / "pilot_summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError(f"pilot summary is missing: {summary_path}")
        summary = json.loads(summary_path.read_text())
        required = {
            "best_checkpoint": paths["best_checkpoint"],
            "statistics": paths["statistics"],
            "codec_result": paths["codec_result"],
            "codec_checkpoint": paths["codec_checkpoint"],
        }
        candidate_record: dict[str, Any] = {
            "summary_path": str(summary_path),
            "summary_sha256": _sha256(summary_path),
            "status": summary.get("status"),
            "target_steps": summary.get("target_steps"),
            "test_opened": summary.get("test_opened"),
            "files": {},
        }
        if summary.get("status") != "PASS":
            raise RuntimeError(f"pilot summary is not PASS for {name}: {summary.get('status')!r}")
        if summary.get("test_opened") is not False:
            raise RuntimeError(f"pilot summary opened test for {name}")
        for label, path in required.items():
            if not path.is_file():
                raise FileNotFoundError(f"{name} missing {label}: {path}")
            candidate_record["files"][label] = {"path": str(path), "sha256": _sha256(path)}
        candidate_record["codec_from_summary"] = summary.get("codec", {})
        candidate_record["statistics_from_summary"] = summary.get("statistics", {})
        records["candidates"][name] = candidate_record
    records["input_hash"] = _canonical_hash(records)
    return records


def _load_single_batch(dataset: Any, index: int, device: torch.device) -> tuple[ClipBatch, ClipBatch]:
    batch_cpu = collate_clip_records([dataset[int(index)]])
    return batch_cpu, batch_cpu.to(device)


def _encode_batch(
    batch_cpu: ClipBatch,
    batch: ClipBatch,
    *,
    codec: Any,
    adapter: StateDetailLatentAdapter,
    codec_hash: str,
    data_hash: str,
) -> tuple[Any, Any]:
    codec.model.prepare_batch(batch_cpu)
    with torch.no_grad():
        latent = codec.model.encode(batch)
    packed = adapter.pack(
        latent,
        codec_hash=codec_hash,
        data_hash=data_hash,
        loss_mask=batch.loss_mask,
    )
    return latent, packed


def _encode_coordinates(
    batch_cpu: ClipBatch,
    coordinates: Tensor,
    *,
    device: torch.device,
    codec: Any,
    adapter: StateDetailLatentAdapter,
    codec_hash: str,
    data_hash: str,
) -> tuple[Any, DiTLatentBatch]:
    coordinates_cpu = coordinates.detach().to(device="cpu", dtype=batch_cpu.x.dtype)
    source_cpu = replace(batch_cpu, x=coordinates_cpu)
    source = source_cpu.to(device)
    return _encode_batch(
        source_cpu,
        source,
        codec=codec,
        adapter=adapter,
        codec_hash=codec_hash,
        data_hash=data_hash,
    )


def _observed_batch(
    latent_batch: DiTLatentBatch,
    batch: ClipBatch,
    history_frames: int,
) -> DiTLatentBatch:
    condition = build_observation_condition(
        latent_batch,
        history_frames=int(history_frames),
        coordinates=batch.x,
        frame_mask=batch.frame_mask,
        loss_mask=batch.loss_mask,
    )
    return latent_batch.with_observation(
        condition.latent_observation_mask,
        sample_origin=condition.sample_origin,
    )


@torch.no_grad()
def _decode(codec: Any, latent: Any) -> Tensor:
    # Scientific coordinate metrics stay FP32 even when DiT sampling uses BF16.
    with torch.autocast(device_type="cuda", enabled=False):
        decoded = codec.model.decode(latent)
    return decoded.x_hat.float()


@torch.no_grad()
def _sample(
    context: "CandidateContext",
    observed_batch: DiTLatentBatch,
    *,
    sample_id: str,
    history_frames: int,
    steps: int,
    draw_id: int,
    diagnostic_kind: str = "sampler",
) -> tuple[Any, dict[str, Any]]:
    seed = stable_seed(
        context.master_seed,
        sample_id,
        history_frames,
        draw_id,
        diagnostic_kind,
    )
    with context.trainer.autocast_context():
        generated, metadata = generate_state_detail_latent(
            context.trainer.model,
            context.adapter,
            observed_batch,
            context.statistics,
            steps=int(steps),
            seed=seed,
        )
    metadata = dict(metadata)
    metadata.update(
        {
            "sample_id": sample_id,
            "history_frames": int(history_frames),
            "draw_id": int(draw_id),
            "seed_protocol": diagnostic_kind,
            "same_noise_key_across_steps": diagnostic_kind == "sampler",
        }
    )
    return generated, metadata


def _metric_row(
    method: str,
    sample_id: str,
    history_frames: int,
    prediction: Tensor,
    target: Tensor,
    batch: ClipBatch,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    identity = parse_sample_id(sample_id)
    return {
        **identity,
        "method": method,
        "history_frames": int(history_frames),
        "metrics": trajectory_metric_record(prediction, target, batch, history_frames),
        "metadata": dict(metadata or {}),
    }


def _load_candidate(
    config: Mapping[str, Any],
    name: str,
    *,
    device: torch.device,
    output_root: Path,
    master_seed: int,
) -> "CandidateContext":
    paths = _candidate_paths(config, name)
    data = _load_data(_resolve_config_path(config["manifest_root"], base=PROJECT_ROOT).resolve())
    summary = json.loads((paths["best_checkpoint"].parent / "pilot_summary.json").read_text())
    checkpoint_statistics = DiTTrainer.load_statistics_from_checkpoint(
        paths["best_checkpoint"], map_location="cpu"
    )
    if summary.get("statistics", {}).get("hash") != checkpoint_statistics.hash:
        raise RuntimeError(f"checkpoint statistics hash differs from pilot summary for {name}")
    if paths["statistics"].is_file():
        artifact_statistics = LatentStatistics.from_state_dict(
            torch.load(paths["statistics"], map_location="cpu", weights_only=False)
        )
        if artifact_statistics.hash != checkpoint_statistics.hash:
            raise RuntimeError(f"shared statistics artifact differs from checkpoint for {name}")
    codec = _load_approved_codec(
        candidate=name,
        result_path=paths["codec_result"],
        checkpoint_path=paths["codec_checkpoint"],
        device=device,
    )
    if summary.get("codec", {}).get("checkpoint_sha256") != codec.checkpoint_sha256:
        raise RuntimeError(f"codec hash differs from pilot summary for {name}")
    ratio = int(codec.ratio)
    adapter = StateDetailLatentAdapter(
        codec_width=int(config.get("codec_width", 128)),
        scalar_width=int(config.get("model", {}).get("scalar_width", 256)),
        vector_width=int(config.get("model", {}).get("vector_width", 128)),
        ratio=ratio,
    ).to(device)
    trainer = _build_trainer(
        codec=codec,
        adapter=adapter,
        statistics=checkpoint_statistics,
        candidate=name,
        data_hash=data.data_hash,
        target_steps=int(summary.get("target_steps", 4500)),
        seed=int(master_seed),
        output_root=output_root,
        device=device,
    )
    trainer.load_checkpoint(paths["best_checkpoint"], map_location=device)
    trainer.model.eval()
    if trainer.step != int(summary.get("completed_steps", trainer.step)):
        # The report records the selected checkpoint step.  The checkpoint itself is authoritative.
        summary["loaded_checkpoint_step"] = int(trainer.step)
    return CandidateContext(
        name=name,
        data=data,
        codec=codec,
        adapter=adapter,
        statistics=checkpoint_statistics.to(device=device),
        trainer=trainer,
        best_checkpoint=paths["best_checkpoint"],
        summary=summary,
        master_seed=int(master_seed),
    )


class CandidateContext:
    def __init__(
        self,
        *,
        name: str,
        data: Any,
        codec: Any,
        adapter: StateDetailLatentAdapter,
        statistics: LatentStatistics,
        trainer: Any,
        best_checkpoint: Path,
        summary: Mapping[str, Any],
        master_seed: int,
    ) -> None:
        self.name = name
        self.data = data
        self.codec = codec
        self.adapter = adapter
        self.statistics = statistics
        self.trainer = trainer
        self.best_checkpoint = best_checkpoint
        self.summary = dict(summary)
        self.master_seed = int(master_seed)

    @property
    def data_hash(self) -> str:
        return str(self.data.data_hash)

    def close(self) -> None:
        self.data.train.close()
        self.data.valid.close()


def _validation_indices(context: CandidateContext) -> tuple[Any, tuple[int, ...]]:
    plan = _make_validation_plan(context.data.valid)
    return plan, tuple(int(index) for index in plan.selected_dataset_indices)


def _evaluate_sample_methods(
    context: CandidateContext,
    dataset: Any,
    index: int,
    *,
    history_frames: int,
    include_repeat_reencoded: bool,
    generated_steps: int = 16,
    generated_draw_id: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    batch_cpu, batch = _load_single_batch(dataset, index, context.trainer.device)
    sample_id = str(batch.sample_id[0])
    oracle_latent, oracle_batch = _encode_batch(
        batch_cpu,
        batch,
        codec=context.codec,
        adapter=context.adapter,
        codec_hash=context.codec.codec_state_hash,
        data_hash=context.data_hash,
    )
    observed_batch = _observed_batch(oracle_batch, batch, history_frames)
    oracle_coordinates = _decode(context.codec, oracle_latent)
    target = batch.x.float()
    rows: list[dict[str, Any]] = [
        _metric_row("codec_oracle", sample_id, history_frames, oracle_coordinates, target, batch)
    ]

    for baseline in ("coordinate_persistence", "coordinate_constant_velocity"):
        prediction, baseline_meta = observed_coordinate_baseline(batch, history_frames, kind=baseline)
        if prediction is not None:
            rows.append(
                _metric_row(
                    baseline,
                    sample_id,
                    history_frames,
                    prediction.float(),
                    target,
                    batch,
                    metadata=baseline_meta,
                )
            )

    scaffold_coordinates = observed_coordinate_scaffold(batch, history_frames)
    scaffold_latent, scaffold_batch = _encode_coordinates(
        batch_cpu,
        scaffold_coordinates,
        device=context.trainer.device,
        codec=context.codec,
        adapter=context.adapter,
        codec_hash=context.codec.codec_state_hash,
        data_hash=context.data_hash,
    )
    scaffold_observed = _observed_batch(scaffold_batch, batch, history_frames)
    persistent_fields, persistence_meta = latent_block_state_persistence(
        scaffold_observed, history_frames
    )
    persistent_latent = context.adapter.make_generated_latent(
        scaffold_observed, persistent_fields
    )
    rows.append(
        _metric_row(
            "latent_block_state_persistence",
            sample_id,
            history_frames,
            _decode(context.codec, persistent_latent),
            target,
            batch,
            metadata=persistence_meta,
        )
    )
    if include_repeat_reencoded:
        rows.append(
            _metric_row(
                "coordinate_repeat_reencoded",
                sample_id,
                history_frames,
                _decode(context.codec, scaffold_latent),
                target,
                batch,
                metadata={
                    "applicable": True,
                    "construction": "observed_prefix_plus_repeated_last_observed_coordinate",
                    "not_deployable_md": True,
                },
            )
        )

    generated_latent, generation_meta = _sample(
        context,
        observed_batch,
        sample_id=sample_id,
        history_frames=history_frames,
        steps=generated_steps,
        draw_id=generated_draw_id,
    )
    rows.append(
        _metric_row(
            "dit_generated",
            sample_id,
            history_frames,
            _decode(context.codec, generated_latent),
            target,
            batch,
            metadata=generation_meta,
        )
    )
    return rows, {
        "sample_id": sample_id,
        "batch": batch,
        "batch_cpu": batch_cpu,
        "oracle_latent": oracle_latent,
        "oracle_batch": oracle_batch,
        "observed_batch": observed_batch,
        "scaffold_latent": scaffold_latent,
        "scaffold_observed": scaffold_observed,
        "generated_latent": generated_latent,
        "oracle_coordinates": oracle_coordinates,
        "target": target,
    }


def _main_candidate(
    context: CandidateContext,
    output_dir: Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    plan, indices = _validation_indices(context)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    jsonl_path = output_dir / f"{context.name}_sample_metrics.jsonl"
    started = time.perf_counter()
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for index in indices:
            for history_frames in (4, 8):
                sample_rows, _ = _evaluate_sample_methods(
                    context,
                    context.data.valid,
                    index,
                    history_frames=history_frames,
                    include_repeat_reencoded=False,
                    generated_steps=16,
                    generated_draw_id=0,
                )
                rows.extend(sample_rows)
                for row in sample_rows:
                    _append_jsonl(handle, row)
    methods = sorted({str(row["method"]) for row in rows})
    aggregate: dict[str, Any] = {}
    for method in methods:
        aggregate[method] = {}
        for history_frames in (4, 8):
            selected = [
                row
                for row in rows
                if row["method"] == method and row["history_frames"] == history_frames
            ]
            aggregate[method][f"H{history_frames}"] = {
                "future": aggregate_rows(selected),
                "L4": aggregate_rows(
                    selected, metric_path=("metrics", "horizons", "L4", "metrics")
                ),
                "L8": aggregate_rows(
                    selected, metric_path=("metrics", "horizons", "L8", "metrics")
                ),
            }
    summary = {
        "schema": "pvb.dit.state_detail.diagnostics.main.v1",
        "candidate": context.name,
        "checkpoint": str(context.best_checkpoint),
        "checkpoint_sha256": _sha256(context.best_checkpoint),
        "codec": context.summary.get("codec", {}),
        "statistics": context.summary.get("statistics", {}),
        "data_hash": context.data_hash,
        "sample_count": len(indices),
        "system_count": len(plan.systems),
        "history_frames": [4, 8],
        "forecast_lengths": list(FORECAST_LENGTHS),
        "sampling_steps": [16],
        "draw_ids": [0],
        "validation_windows": list(plan.windows),
        "schedule_hash": plan.schedule_hash,
        "aggregation": "sample_equal_then_system_equal; one draw is not a sample population",
        "aggregate": aggregate,
        "elapsed_s": time.perf_counter() - started,
        "sample_metrics_path": str(jsonl_path),
        "test_opened": False,
        "optimizer_steps": 0,
    }
    _atomic_json(output_dir / f"{context.name}_main_summary.json", summary)
    return summary


def _deep_indices(context: CandidateContext) -> tuple[int, ...]:
    selected: list[int] = []
    for index, row in enumerate(context.data.valid._index):
        identity = parse_sample_id(str(row[0]))
        if identity["replica"] == "R1" and identity["window"] == 30:
            selected.append(index)
    systems = {parse_sample_id(str(context.data.valid._index[index][0]))["system"] for index in selected}
    if len(selected) != 8 or len(systems) != 8:
        raise RuntimeError("deep diagnostic subset must contain exactly eight R1/w30 systems")
    return tuple(sorted(selected, key=lambda index: str(context.data.valid._index[index][0])))


def _train_reference_summary(context: CandidateContext) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(context.data.train._index):
        identity = parse_sample_id(str(row[0]))
        if identity["replica"] != "R1" or identity["window"] != 30:
            continue
        batch_cpu, batch = _load_single_batch(context.data.train, index, context.trainer.device)
        latent, packed = _encode_batch(
            batch_cpu,
            batch,
            codec=context.codec,
            adapter=context.adapter,
            codec_hash=context.codec.codec_state_hash,
            data_hash=context.data_hash,
        )
        masks = packed.field_masks()
        rows.append(
            {
                **identity,
                "raw": latent_summary(packed.fields, masks),
                "standardized": latent_summary(
                    context.statistics.normalize(packed).fields,
                    masks,
                ),
            }
        )
    systems = sorted({row["system"] for row in rows})
    return {
        "selection": "all train systems with replica R1 and window 30",
        "sample_count": len(rows),
        "system_count": len(systems),
        "rows": rows,
        "full_train_distribution_claim": False,
    }


def _deep_candidate(
    context: CandidateContext,
    output_dir: Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    indices = _deep_indices(context)
    deep_config = config.get("deep", {})
    steps = tuple(int(value) for value in deep_config.get("sampling_steps", (8, 16)))
    draws = tuple(int(value) for value in deep_config.get("draws", (0, 1, 2, 3)))
    rf_draws = tuple(int(value) for value in deep_config.get("rf_draws", (0, 1)))
    tau_values = tuple(float(value) for value in deep_config.get("tau", TAU_MIDPOINTS))
    scales = tuple(float(value) for value in deep_config.get("perturbation_scales", PERTURBATION_SCALES))
    perturb_draws = tuple(int(value) for value in deep_config.get("perturbation_draws", (0, 1)))
    rows: list[dict[str, Any]] = []
    flow_rows: list[dict[str, Any]] = []
    sensitivity_rows: list[dict[str, Any]] = []
    swap_rows: list[dict[str, Any]] = []
    diversity_rows: list[dict[str, Any]] = []
    latent_rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for index in indices:
        batch_cpu, batch = _load_single_batch(context.data.valid, index, context.trainer.device)
        sample_id = str(batch.sample_id[0])
        oracle_latent, oracle_batch = _encode_batch(
            batch_cpu,
            batch,
            codec=context.codec,
            adapter=context.adapter,
            codec_hash=context.codec.codec_state_hash,
            data_hash=context.data_hash,
        )
        target = batch.x.float()
        for history_frames in (4, 8):
            observed_batch = _observed_batch(oracle_batch, batch, history_frames)
            generated_by_step: dict[int, list[Tensor]] = {}
            generated_latents: dict[tuple[int, int], Any] = {}
            for step_count in steps:
                generated_by_step[step_count] = []
                for draw_id in draws:
                    generated, metadata = _sample(
                        context,
                        observed_batch,
                        sample_id=sample_id,
                        history_frames=history_frames,
                        steps=step_count,
                        draw_id=draw_id,
                    )
                    generated_latents[(step_count, draw_id)] = generated
                    prediction = _decode(context.codec, generated)
                    generated_by_step[step_count].append(prediction)
                    rows.append(
                        _metric_row(
                            "dit_generated",
                            sample_id,
                            history_frames,
                            prediction,
                            target,
                            batch,
                            metadata={**metadata, "diagnostic_stage": "deep"},
                        )
                    )
                diversity_rows.append(
                    {
                        **parse_sample_id(sample_id),
                        "history_frames": history_frames,
                        "steps": step_count,
                        "diversity": diversity_summary(
                            torch.stack(generated_by_step[step_count]), batch
                        ),
                    }
                )
            oracle_fields = LatentFieldSet(
                oracle_latent.state_h,
                oracle_latent.detail_h,
                oracle_latent.state_v,
                oracle_latent.detail_v,
            )
            oracle_masks = oracle_batch.field_masks()
            generated16 = generated_latents[(16, draws[0])]
            generated_fields = LatentFieldSet(
                generated16.state_h,
                generated16.detail_h,
                generated16.state_v,
                generated16.detail_v,
            )
            standardized_oracle = context.statistics.normalize(oracle_batch)
            generated_batch = observed_batch.with_fields(generated_fields)
            latent_rows.append(
                {
                    **parse_sample_id(sample_id),
                    "history_frames": history_frames,
                    "reference_raw": latent_summary(oracle_fields, oracle_masks),
                    "reference_standardized": latent_summary(
                        standardized_oracle.fields, oracle_masks
                    ),
                    "generated_raw": latent_summary(generated_fields, oracle_masks),
                    "generated_standardized": latent_summary(
                        context.statistics.normalize(generated_batch).fields,
                        oracle_masks,
                    ),
                    "generated_to_reference": latent_norm_comparison(
                        oracle_fields, generated_fields, oracle_masks
                    ),
                    "supports_full_covariance": False,
                    "supports_density_estimate": False,
                }
            )
            for swap in (
                "oracle_state_generated_detail",
                "generated_state_oracle_detail",
                "generated_state_raw_zero_detail",
            ):
                swap_fields = field_swap_fields(
                    oracle_latent, generated16, observed_batch, swap=swap
                )
                swap_latent = context.adapter.make_generated_latent(
                    observed_batch, swap_fields
                )
                swap_rows.append(
                    _metric_row(
                        swap,
                        sample_id,
                        history_frames,
                        _decode(context.codec, swap_latent),
                        target,
                        batch,
                        metadata={
                            "oracle_field_swap_not_deployable": True,
                            "generated_steps": 16,
                            "draw_id": draws[0],
                        },
                    )
                )
            for scope in ("state_only", "detail_only"):
                for scale in scales:
                    for draw_id in perturb_draws:
                        seed = stable_seed(
                            context.master_seed,
                            sample_id,
                            history_frames,
                            draw_id,
                            f"perturb_{scope}_{scale:g}",
                        )
                        fields, perturb_meta = perturb_normalized_oracle(
                            observed_batch,
                            context.statistics,
                            scope=scope,
                            scale=scale,
                            seed=seed,
                        )
                        perturbed = context.adapter.make_generated_latent(
                            observed_batch, fields
                        )
                        sensitivity_rows.append(
                            _metric_row(
                                f"oracle_perturb_{scope}",
                                sample_id,
                                history_frames,
                                _decode(context.codec, perturbed),
                                target,
                                batch,
                                metadata={
                                    **perturb_meta,
                                    "oracle_sensitivity_not_deployable": True,
                                },
                            )
                        )
            flow_rows.extend(
                fixed_tau_flow_diagnostics(
                    context.trainer.model,
                    observed_batch,
                    context.statistics,
                    sample_id=sample_id,
                    history_frames=history_frames,
                    tau_values=tau_values,
                    draw_ids=rf_draws,
                    master_seed=context.master_seed,
                    autocast_context=context.trainer.autocast_context,
                )
            )
    aggregate: dict[str, Any] = {}
    for method in sorted({str(row["method"]) for row in rows + swap_rows + sensitivity_rows}):
        method_rows = [row for row in rows + swap_rows + sensitivity_rows if row["method"] == method]
        aggregate[method] = {}
        for history_frames in (4, 8):
            selected = [row for row in method_rows if row["history_frames"] == history_frames]
            aggregate[method][f"H{history_frames}"] = {
                "future": aggregate_rows(selected),
                "L4": aggregate_rows(
                    selected, metric_path=("metrics", "horizons", "L4", "metrics")
                ),
                "L8": aggregate_rows(
                    selected, metric_path=("metrics", "horizons", "L8", "metrics")
                ),
            }
    flow_aggregate: dict[str, Any] = {}
    for history_frames in (4, 8):
        selected = [row for row in flow_rows if row["history_frames"] == history_frames]
        by_tau: dict[str, list[Mapping[str, Any]]] = {}
        for row in selected:
            by_tau.setdefault(str(row["tau"]), []).append(row)
        flow_aggregate[f"H{history_frames}"] = {
            tau: {
                "mean_total": sum(float(row["total"]) for row in tau_rows) / len(tau_rows),
                "mean_fields": {
                    name: sum(float(row["fields"][name]) for row in tau_rows) / len(tau_rows)
                    for name in ("state_h", "detail_h", "state_v", "detail_v")
                },
                "count": len(tau_rows),
            }
            for tau, tau_rows in sorted(by_tau.items(), key=lambda item: float(item[0]))
        }
    summary = {
        "schema": "pvb.dit.state_detail.diagnostics.deep.v1",
        "candidate": context.name,
        "checkpoint": str(context.best_checkpoint),
        "checkpoint_sha256": _sha256(context.best_checkpoint),
        "sample_count": len(indices),
        "history_frames": [4, 8],
        "sampling_steps": list(steps),
        "draw_ids": list(draws),
        "rf_tau": {
            "tau": list(tau_values),
            "draw_ids": list(rf_draws),
            "aggregate": flow_aggregate,
            "rows": flow_rows,
        },
        "latent": {"rows": latent_rows},
        "aggregate": aggregate,
        "field_swaps": {"rows": swap_rows},
        "perturbation": {"rows": sensitivity_rows, "scales": list(scales)},
        "diversity": {"rows": diversity_rows},
        "train_reference": _train_reference_summary(context),
        "limitations": [
            "latent summaries are channelwise/rotation-invariant norm summaries, not full covariance or density estimates",
            "dynamic_correlation is pathwise generated-vs-target velocity correlation; it is not an ACF score",
            "short-window RMSF/ACF/frequency values are diagnostics, not long-time kinetics",
            "oracle field swaps and oracle perturbations are not deployable baselines",
            "one draw in main is not a diversity estimate; deep draws are reported without best-of-N selection",
        ],
        "elapsed_s": time.perf_counter() - started,
        "test_opened": False,
        "optimizer_steps": 0,
    }
    _atomic_json(output_dir / f"{context.name}_deep_summary.json", summary)
    return summary


def _smoke_candidate(context: CandidateContext, output_dir: Path) -> dict[str, Any]:
    plan, indices = _validation_indices(context)
    index = indices[0]
    batch_cpu, batch = _load_single_batch(context.data.valid, index, context.trainer.device)
    sample_id = str(batch.sample_id[0])
    oracle_latent, oracle_batch = _encode_batch(
        batch_cpu,
        batch,
        codec=context.codec,
        adapter=context.adapter,
        codec_hash=context.codec.codec_state_hash,
        data_hash=context.data_hash,
    )
    oracle_coordinates = _decode(context.codec, oracle_latent)
    rows: list[dict[str, Any]] = []
    for history_frames in (4, 8):
        observed = _observed_batch(oracle_batch, batch, history_frames)
        observed_condition = observed.sample_origin.detach().clone()
        generated8, meta8 = _sample(
            context, observed, sample_id=sample_id, history_frames=history_frames, steps=8, draw_id=0
        )
        generated16, meta16 = _sample(
            context, observed, sample_id=sample_id, history_frames=history_frames, steps=16, draw_id=0
        )
        for label, generated, metadata in (
            ("dit_generated_8", generated8, meta8),
            ("dit_generated_16", generated16, meta16),
        ):
            prediction = _decode(context.codec, generated)
            record = _metric_row(label, sample_id, history_frames, prediction, batch.x.float(), batch)
            rows.append(record)
            if not bool(torch.isfinite(prediction).all()):
                raise FloatingPointError(f"non-finite smoke decode for {context.name} {label}")
            if not bool(metadata.get("observed_clamp_exact", False)):
                raise AssertionError(f"normalized observed clamp was not exact for {context.name} {label}")
        if not torch.equal(observed.sample_origin, observed_condition):
            raise AssertionError("observation origin changed during smoke")
        clamp = observed.observed_mask.index_select(0, observed.abid).transpose(0, 1)
        for name in observed.fields.names():
            value = getattr(generated16, name)
            clean = getattr(observed.fields, name)
            expanded = clamp.reshape(clamp.shape + (1,) * (value.ndim - 2))
            if not torch.allclose(
                value.masked_select(expanded),
                clean.masked_select(expanded),
                atol=1.0e-5,
                rtol=0.0,
            ):
                raise AssertionError("inverse-normalized observed latent fields exceeded round-trip tolerance")
    result = {
        "schema": "pvb.dit.state_detail.diagnostics.smoke.v1",
        "candidate": context.name,
        "sample_id": sample_id,
        "sample_index": int(index),
        "validation_sample_count": len(indices),
        "steps": [8, 16],
        "rows": rows,
        "codec_oracle_finite": bool(torch.isfinite(oracle_coordinates).all()),
        "device": _cuda_info(context.trainer.device),
        "autocast": "torch.bfloat16 for DiT model; FP32 codec decode/metrics",
        "optimizer_steps": 0,
        "test_opened": False,
        "plan_schedule_hash": plan.schedule_hash,
    }
    _atomic_json(output_dir / f"{context.name}_smoke.json", result)
    return result


def _summary_report(run_dir: Path, config: Mapping[str, Any], inputs: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    main: dict[str, Any] = {}
    deep: dict[str, Any] = {}
    smoke: dict[str, Any] = {}
    for name in ("ratio2_state_detail", "ratio4_state_detail"):
        main_path = run_dir / f"{name}_main_summary.json"
        deep_path = run_dir / f"{name}_deep_summary.json"
        smoke_path = run_dir / f"{name}_smoke.json"
        if main_path.is_file():
            main[name] = json.loads(main_path.read_text())
        if deep_path.is_file():
            deep[name] = json.loads(deep_path.read_text())
        if smoke_path.is_file():
            smoke[name] = json.loads(smoke_path.read_text())
    final = {
        "schema": DIAGNOSTICS_SCHEMA,
        "actual_code_commit": _git_commit(),
        "inputs": inputs,
        "config_hash": _canonical_hash(config),
        "candidate_main": main,
        "candidate_deep": deep,
        "candidate_smoke": smoke,
        "prior_baseline_usable": {
            "status": "reference_only_not_deployable",
            "decision": "retain latent_block_state_persistence as a conservative diagnostic reference, not as a deployable generative prior or source center",
            "evidence": "for both candidates, main dit_generated future geometry/contact metrics are worse than the observed-only latent_block_state_persistence control; deep field, boundary, and dynamic diagnostics remain attribution evidence",
            "not_a_training_decision": True,
        },
        "limitations": [
            "no optimizer steps or test payload access",
            "stochastic trajectories are not expected to reproduce reference MD frame by frame",
            "R2/R4 diagnostics use separate latent dimensions and identical seed protocol, not identical elementwise noise",
            "one validation run and four deep draws do not establish long-time kinetics or a scientific winner",
        ],
        "status": "A_DIAGNOSTICS_READY_FOR_REVIEW",
    }
    lines = [
        "# Session A DiT pilot diagnostics",
        "",
        f"Status: `{final['status']}`",
        "",
        "This report is inference-only evidence from the existing best-validation checkpoints.",
        "No optimizer step and no test clip payload access were permitted.",
        "",
        "## Candidate outputs",
        "",
        "| Candidate | Main | Deep | Smoke |",
        "|---|---|---|---|",
    ]
    for name in ("ratio2_state_detail", "ratio4_state_detail"):
        lines.append(
            f"| {name} | {'present' if name in main else 'not_run'} | {'present' if name in deep else 'not_run'} | {'present' if name in smoke else 'not_run'} |"
        )
    lines.extend(
        [
            "",
            "## Protocol",
            "",
            "Observed frames are `[0,H)`, future frames `[H,16)`, boundary is `[H-1,H)`, and the full clip is diagnostic-only.",
            "L4/L8 rows are separate cropped horizons with physical timestamps and masks rebuilt.",
            "RMSD diversity uses `sqrt(sum_xyz_squared / valid_atom_count)`; no best-of-N metric is used.",
            "",
            "## Prior decision",
            "",
            "`latent_block_state_persistence` is retained as a reference-only control, not a deployable prior/source center: for both candidates, main `dit_generated` future geometry/contact results are worse than this observed-only control. Deep field, boundary, and dynamic outputs are attribution evidence only; no R2/R4 winner is declared.",
            "",
            "## Limitations",
            "",
        ]
    )
    for item in final["limitations"]:
        lines.append(f"- {item}")
    report = "\n".join(lines) + "\n"
    _atomic_json(run_dir / "summary.json", final)
    (run_dir / "report.md").write_text(report)
    return final, report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--stage", choices=("preflight", "smoke", "main", "deep", "summarize"), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--candidate", choices=("ratio2_state_detail", "ratio4_state_detail"), default=None)
    parser.add_argument("--manifest-root", type=Path, default=None)
    parser.add_argument("--pilot-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--run-id", default=None)
    return parser


def main() -> None:
    args = _parser().parse_args()
    config = _load_config(args.config.resolve())
    if args.manifest_root is not None:
        config["manifest_root"] = str(args.manifest_root)
    if args.pilot_root is not None:
        config["pilot_root"] = str(args.pilot_root)
    if args.output_root is not None:
        config["output_root"] = str(args.output_root)
    run_id = str(args.run_id or config.get("run_id", "session_a_260909"))
    output_root = _resolve_config_path(config["output_root"], base=PROJECT_ROOT)
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_yaml(run_dir / "config_resolved.yaml", config)
    inputs = _input_manifest(config)
    _atomic_json(run_dir / "inputs.json", inputs)
    if args.stage == "preflight":
        _atomic_json(
            run_dir / "preflight.json",
            {
                "schema": "pvb.dit.state_detail.diagnostics.preflight.v1",
                "actual_code_commit": _git_commit(),
                "inputs": inputs,
                "status": "PASS",
                "numeric_cuda_stages_run": False,
                "test_opened": False,
            },
        )
        return
    if args.stage == "summarize":
        _summary_report(run_dir, config, inputs)
        return
    device = torch.device(args.device)
    _cuda_info(device)
    candidates = [args.candidate] if args.candidate else ["ratio4_state_detail", "ratio2_state_detail"]
    for candidate in candidates:
        context = _load_candidate(
            config,
            candidate,
            device=device,
            output_root=run_dir,
            master_seed=int(config.get("master_seed", 20260907)),
        )
        try:
            if args.stage == "smoke":
                _smoke_candidate(context, run_dir)
            elif args.stage == "main":
                _main_candidate(context, run_dir, config)
            elif args.stage == "deep":
                _deep_candidate(context, run_dir, config)
            else:
                raise ValueError(f"unsupported stage {args.stage}")
        finally:
            context.close()
    if args.stage in ("smoke", "main", "deep"):
        _atomic_json(
            run_dir / f"{args.stage}_stage_complete.json",
            {
                "schema": "pvb.dit.state_detail.diagnostics.stage.v1",
                "stage": args.stage,
                "candidates": candidates,
                "device": _cuda_info(device),
                "optimizer_steps": 0,
                "test_opened": False,
            },
        )


if __name__ == "__main__":
    main()
