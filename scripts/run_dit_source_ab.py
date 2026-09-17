#!/usr/bin/env python
"""Run the bounded R4 Gaussian/conditional source experiment."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.clip_dataset import ClipBatch, collate_clip_records
from evaluation.dit_diagnostics import (
    aggregate_rows,
    json_safe,
    parse_sample_id,
    stable_seed,
    trajectory_metric_record,
)
from evaluation.dit_evaluation import evaluate_oracle_vs_generated
from module.dit_latent_cache import LatentFieldCache, cache_key
from module.latent_flow_source import (
    build_observed_center,
    future_field_masks,
    source_contract,
)
from module.latent_rectified_flow import (
    LatentFieldSet,
    RectifiedFlowObjective,
    generate_state_detail_latent,
)
from module.molecular_dit import MolecularDiT
from module.state_detail_latent_adapter import (
    LatentStatistics,
    StateDetailLatentAdapter,
    contract_hash,
)
from scripts.run_state_detail_dit_pilot import (
    FrozenCodec,
    _make_validation_plan,
    _encode_batch,
    _load_approved_codec,
    _load_data,
    _make_sampler,
    _observed_batch,
    _sha256,
)
from trainer.dit_trainer import DiTTrainConfig, DiTTrainer, module_state_hash


HISTORY_SCHEDULE = (4, 8)
ARMS = ("gaussian", "conditional")
CHECKPOINT_STEPS = (4500, 10000, 20000)


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


def _safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, torch.Tensor):
        return json_safe(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(_safe(value), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_safe(value), sort_keys=True) + "\n")


def _resolve(value: str | Path, base: Path = PROJECT_ROOT) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _load_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, Mapping):
        raise ValueError("source A/B config must be a mapping")
    return json.loads(json.dumps(value))


def _require_cuda(device: torch.device) -> None:
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("source A/B numerical stages require an actual CUDA device")
    torch.cuda.set_device(device)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _cuda_info(device: torch.device) -> dict[str, Any]:
    _require_cuda(device)
    props = torch.cuda.get_device_properties(device)
    return {
        "requested_device": str(device),
        "logical_index": torch.cuda.current_device(),
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "device_name": torch.cuda.get_device_name(device),
        "device_uuid": str(getattr(props, "uuid", "")),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "tf32": False,
        "amp": "bfloat16",
    }


def _new_model(cfg: Mapping[str, Any], adapter: StateDetailLatentAdapter) -> MolecularDiT:
    model_cfg = cfg["model"]
    return MolecularDiT(
        adapter=adapter,
        scalar_width=int(model_cfg["scalar_width"]),
        vector_width=int(model_cfg["vector_width"]),
        depth=int(model_cfg["depth"]),
        heads=int(model_cfg["heads"]),
        ffn_multiplier=int(model_cfg["ffn_multiplier"]),
        dropout=float(model_cfg["dropout"]),
        execution_backend=str(model_cfg["execution_backend"]),
    )


def _new_adapter(cfg: Mapping[str, Any], device: torch.device) -> StateDetailLatentAdapter:
    model_cfg = cfg["model"]
    return StateDetailLatentAdapter(
        codec_width=int(model_cfg["codec_width"]),
        scalar_width=int(model_cfg["scalar_width"]),
        vector_width=int(model_cfg["vector_width"]),
        ratio=int(cfg["candidate"]["ratio"]),
    ).to(device)


@dataclass
class ExperimentContext:
    cfg: dict[str, Any]
    output_dir: Path
    data: Any
    codec: FrozenCodec
    statistics: LatentStatistics
    adapter: StateDetailLatentAdapter
    device: torch.device
    statistics_path: Path
    statistics_file_sha256: str

    @property
    def data_hash(self) -> str:
        return str(self.data.data_hash)

    def close(self) -> None:
        self.data.train.close()
        self.data.valid.close()


def _manifest_preflight(cfg: Mapping[str, Any]) -> dict[str, Any]:
    manifest_root = _resolve(cfg["manifest_root"])
    manifest_path = manifest_root / "manifest.json"
    materialization_path = manifest_root / "materialization.json"
    if not manifest_path.is_file() or not materialization_path.is_file():
        raise FileNotFoundError(f"frozen manifest/materialization missing under {manifest_root}")
    manifest = json.loads(manifest_path.read_text())
    materialization = json.loads(materialization_path.read_text())
    if manifest.get("status") != "FROZEN":
        raise RuntimeError("manifest is not FROZEN")
    if manifest.get("test_sampling", {}).get("opened") is not False:
        raise RuntimeError("manifest does not certify unopened test sampling")
    candidate = cfg["candidate"]
    pilot_root = _resolve(cfg["pilot_root"])
    stats_path = _resolve(candidate["statistics"], pilot_root)
    codec_path = _resolve(candidate["codec_checkpoint"])
    result_path = _resolve(candidate["codec_result"])
    old_path = _resolve(candidate["old_dit_checkpoint"], pilot_root)
    for label, path in (
        ("statistics", stats_path),
        ("codec_checkpoint", codec_path),
        ("codec_result", result_path),
        ("old_dit_checkpoint", old_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"missing frozen {label}: {path}")
    actual_stats_hash = _sha256(stats_path)
    if actual_stats_hash != str(candidate["statistics_file_sha256"]):
        raise RuntimeError("statistics file SHA256 differs from the frozen contract")
    return {
        "schema": "pvb.dit.state_detail.source_ab.inputs.v1",
        "manifest_root": str(manifest_root),
        "manifest_sha256": _sha256(manifest_path),
        "materialization_sha256": _sha256(materialization_path),
        "manifest_content_sha256": manifest.get("manifest_content_sha256"),
        "train_count": manifest.get("counts", {}).get("train_clips"),
        "valid_count": manifest.get("counts", {}).get("valid_clips"),
        "test_sampling_opened": manifest.get("test_sampling", {}).get("opened"),
        "statistics_path": str(stats_path),
        "statistics_file_sha256": actual_stats_hash,
        "codec_checkpoint": str(codec_path),
        "codec_checkpoint_sha256": _sha256(codec_path),
        "codec_result": str(result_path),
        "codec_result_sha256": _sha256(result_path),
        "old_dit_checkpoint": str(old_path),
        "old_dit_checkpoint_sha256": _sha256(old_path),
        "test_payload_opened": False,
    }


def _load_context(cfg: dict[str, Any], output_dir: Path, device: torch.device) -> ExperimentContext:
    _require_cuda(device)
    candidate = cfg["candidate"]
    pilot_root = _resolve(cfg["pilot_root"])
    data = _load_data(_resolve(cfg["manifest_root"]))
    codec = _load_approved_codec(
        candidate=str(candidate["mode"]),
        result_path=_resolve(candidate["codec_result"]),
        checkpoint_path=_resolve(candidate["codec_checkpoint"]),
        device=device,
    )
    statistics_path = _resolve(candidate["statistics"], pilot_root)
    statistics = LatentStatistics.from_state_dict(
        torch.load(statistics_path, map_location="cpu", weights_only=False)
    )
    if statistics.ratio != int(candidate["ratio"]) or statistics.mode != str(candidate["mode"]):
        raise RuntimeError("statistics ratio/mode differs from the R4 source contract")
    statistics_file_sha256 = _sha256(statistics_path)
    if statistics_file_sha256 != str(candidate["statistics_file_sha256"]):
        raise RuntimeError("statistics file hash changed")
    model_cfg = cfg["model"]
    adapter = StateDetailLatentAdapter(
        codec_width=int(model_cfg["codec_width"]),
        scalar_width=int(model_cfg["scalar_width"]),
        vector_width=int(model_cfg["vector_width"]),
        ratio=int(candidate["ratio"]),
    ).to(device)
    return ExperimentContext(
        cfg=cfg,
        output_dir=output_dir,
        data=data,
        codec=codec,
        statistics=statistics,
        adapter=adapter,
        device=device,
        statistics_path=statistics_path,
        statistics_file_sha256=statistics_file_sha256,
    )


def _shared_initialization(cfg: Mapping[str, Any], device: torch.device) -> tuple[dict[str, Tensor], str]:
    torch.manual_seed(int(cfg["seed"]["init"]))
    adapter = _new_adapter(cfg, device)
    model = _new_model(cfg, adapter).to(device)
    state = {name: value.detach().to(device="cpu").clone() for name, value in model.state_dict().items()}
    init_hash = module_state_hash(model)
    del model
    return state, init_hash


def _make_trainer(
    context: ExperimentContext,
    *,
    source_mode: str,
    center_kind: str,
    target_steps: int,
    init_state: Mapping[str, Tensor],
    init_hash: str,
) -> DiTTrainer:
    cfg = context.cfg
    candidate = cfg["candidate"]
    torch.manual_seed(int(cfg["seed"]["init"]))
    adapter = _new_adapter(cfg, context.device)
    model = _new_model(cfg, adapter).to(context.device)
    model.load_state_dict(init_state, strict=True)
    model_hash = module_state_hash(model)
    if model_hash != init_hash:
        raise RuntimeError("shared initialization hash did not round-trip")
    train_cfg = DiTTrainConfig(
        ratio=int(candidate["ratio"]),
        mode=str(candidate["mode"]),
        codec_width=int(cfg["model"]["codec_width"]),
        scalar_width=int(cfg["model"]["scalar_width"]),
        vector_width=int(cfg["model"]["vector_width"]),
        depth=int(cfg["model"]["depth"]),
        heads=int(cfg["model"]["heads"]),
        ffn_multiplier=int(cfg["model"]["ffn_multiplier"]),
        dropout=float(cfg["model"]["dropout"]),
        learning_rate=float(cfg["protocol"]["learning_rate"]),
        weight_decay=float(cfg["protocol"]["weight_decay"]),
        grad_clip=float(cfg["protocol"]["grad_clip"]),
        max_steps=int(target_steps),
        seed=int(cfg["seed"]["training"]),
        amp=True,
        output_root=str(context.output_dir),
        data_hash=context.data_hash,
        codec_hash=context.codec.codec_state_hash,
        stats_hash=context.statistics.hash,
        source_mode=source_mode,
        center_kind=center_kind,
        source_sigma=float(cfg["source"]["sigma"]),
        normalization_hash=context.statistics.hash,
        init_hash=init_hash,
        observation_mixture=tuple(int(value) for value in cfg["schedule"]["observation_history"]),
        metadata={
            "phase": "source_ab_v1",
            "source_contract": source_contract(
                source_mode=source_mode,
                center_kind=center_kind,
                statistics=context.statistics,
                sigma=float(cfg["source"]["sigma"]),
            ),
            "execution_backend": "factorized_v2",
            "init_hash": init_hash,
        },
    )
    return DiTTrainer(
        model,
        adapter,
        config=train_cfg,
        statistics=context.statistics.to(device=context.device),
        codec=context.codec.model,
        frame_encoder=context.codec.model.frame_encoder,
    )


def _prepare_encoded(
    context: ExperimentContext,
    dataset: Any,
    indices: Sequence[int],
) -> tuple[Any, ClipBatch, Any, Any, Any]:
    latent, batch_cpu, batch = _encode_batch(
        dataset,
        indices,
        codec=context.codec,
        adapter=context.adapter,
        data_hash=context.data_hash,
        device=context.device,
    )
    target_batch = context.adapter.pack(
        latent,
        codec_hash=context.codec.codec_state_hash,
        data_hash=context.data_hash,
        origin_from_latent=True,
        loss_mask=batch.loss_mask,
    )
    return latent, batch_cpu, batch, target_batch, latent


def _observed_target(context: ExperimentContext, target_batch: Any, batch: ClipBatch, history: int) -> Any:
    return _observed_batch(
        target_batch,
        batch,
        adapter=context.adapter,
        history_frames=int(history),
    )


def _center_key(context: ExperimentContext, batch: ClipBatch, history: int, center_kind: str) -> str:
    return cache_key(
        {
            "schema": "source-center-key.v1",
            "sample_ids": list(batch.sample_id),
            "atom_counts": list(batch.atom_counts),
            "history": int(history),
            "center_kind": center_kind,
            "codec_hash": context.codec.codec_state_hash,
            "statistics_hash": context.statistics.hash,
        }
    )


@torch.no_grad()
def _source_center(
    context: ExperimentContext,
    batch: ClipBatch,
    target_batch: Any,
    history: int,
    center_kind: str,
    cache: LatentFieldCache,
) -> tuple[LatentFieldSet, dict[str, Any]]:
    key = _center_key(context, batch, history, center_kind)
    cached = cache.get(key, device=context.device, dtype=target_batch.state_h.dtype)
    if cached is not None:
        return cached, {"cache": "hit", "cache_key": key, "center_kind": center_kind}
    center, metadata = build_observed_center(
        center_kind,
        codec_model=context.codec.model,
        coordinate_batch=batch,
        target_batch=target_batch,
        adapter=context.adapter,
        statistics=context.statistics.to(device=context.device),
        history_frames=history,
        codec_hash=context.codec.codec_state_hash,
        data_hash=context.data_hash,
    )
    cache.put(key, center)
    return center, {
        "cache": "miss",
        "cache_key": key,
        "center_kind": center_kind,
        "uses_future_coordinates": False,
        "template_coordinates": metadata.get("template_coordinates"),
        "template_latent": metadata.get("template_latent"),
    }


def _merge_center_with_observed(context: ExperimentContext, target_batch: Any, center: LatentFieldSet) -> LatentFieldSet:
    normalized = context.statistics.to(device=context.device).normalize(target_batch)
    masks = future_field_masks(target_batch)
    values = []
    for name in ("state_h", "detail_h", "state_v", "detail_v"):
        value = getattr(normalized.fields, name)
        center_value = getattr(center, name)
        mask = masks[name].reshape(masks[name].shape + (1,) * (value.ndim - 2))
        values.append(torch.where(mask, center_value, value))
    return LatentFieldSet(*values)


def _decode_fields(context: ExperimentContext, target_batch: Any, fields: LatentFieldSet) -> Tensor:
    raw = context.statistics.to(device=context.device).inverse_fields(fields)
    latent = context.adapter.make_generated_latent(target_batch, raw)
    with torch.autocast(device_type="cuda", enabled=False), torch.no_grad():
        return context.codec.model.decode(latent).x_hat.float()


def _coordinate_rms(prediction: Tensor, target: Tensor, mask: Tensor) -> float:
    selected = (prediction - target).square().sum(dim=-1).masked_select(mask)
    return float(selected.mean().sqrt().detach().float().cpu()) if selected.numel() else 0.0


def _bond_rmse(prediction: Tensor, target: Tensor, batch: ClipBatch, mask: Tensor) -> float:
    pairs = batch.bond_index.to(device=prediction.device)
    if pairs.numel() == 0:
        return 0.0
    values: list[Tensor] = []
    for frame in range(int(prediction.shape[0])):
        valid = mask[frame, pairs[0]] & mask[frame, pairs[1]]
        if bool(valid.any()):
            pd = torch.linalg.vector_norm(
                prediction[frame, pairs[0]] - prediction[frame, pairs[1]], dim=-1
            )
            td = torch.linalg.vector_norm(
                target[frame, pairs[0]] - target[frame, pairs[1]], dim=-1
            )
            values.append((pd[valid] - td[valid]).square())
    return float(torch.cat(values).mean().sqrt().detach().float().cpu()) if values else 0.0


def _source_metric_row(
    center_coordinates: Tensor,
    comparison_coordinates: Tensor,
    batch: ClipBatch,
    history: int,
    oracle_coordinates: Tensor,
) -> dict[str, Any]:
    atom_mask = batch.loss_mask.to(device=center_coordinates.device, dtype=torch.bool)
    frame_mask = batch.frame_mask.index_select(0, batch.abid).transpose(0, 1)
    valid = frame_mask & atom_mask.unsqueeze(0)
    future = valid.clone()
    future[:history] = False
    future_rms = _coordinate_rms(center_coordinates, comparison_coordinates, future)
    bond = _bond_rmse(center_coordinates, comparison_coordinates, batch, valid)
    if history + 1 < center_coordinates.shape[0]:
        future_internal = _coordinate_rms(
            center_coordinates[history + 1:],
            center_coordinates[history:-1],
            future[history + 1:] & future[history:-1],
        )
    else:
        future_internal = 0.0
    oracle_raw_rmsd = _coordinate_rms(oracle_coordinates, batch.x.float(), valid)
    return {
        "template_raw_rmsd": future_rms,
        "template_bond_rmse": bond,
        "decoded_future_internal_displacement_rms": future_internal,
        "oracle_raw_rmsd": oracle_raw_rmsd,
        "boundary_displacement_rms": _coordinate_rms(
            center_coordinates[history:history + 1],
            comparison_coordinates[history:history + 1],
            future[history:history + 1],
        ),
        "threshold_template_raw_rmsd": max(0.10, 5.0 * oracle_raw_rmsd),
    }


def _source_check(ctx: ExperimentContext) -> dict[str, Any]:
    plan = _make_validation_plan(ctx.data.valid)
    selected: list[tuple[int, str]] = []
    for index, row in enumerate(ctx.data.valid._index):
        sample_id = str(row[0])
        parsed = parse_sample_id(sample_id)
        if parsed["replica"] == str(ctx.cfg["evaluation"]["source_check_replica"]) and parsed["window"] == int(ctx.cfg["evaluation"]["source_check_window"]):
            selected.append((index, sample_id))
    selected.sort(key=lambda item: item[1])
    if len(selected) != 8:
        raise RuntimeError(f"source check requires eight fixed validation clips, got {len(selected)}")
    center_kind_values = (
        str(ctx.cfg["source"]["center_kind"]),
        str(ctx.cfg["source"]["fallback_center_kind"]),
    )
    rows: list[dict[str, Any]] = []
    for index, sample_id in selected:
        latent, batch_cpu, batch, target_batch, _ = _prepare_encoded(ctx, ctx.data.valid, (index,))
        oracle_coordinates = ctx.codec.model.decode(latent).x_hat.float()
        for history in (4, 8):
            observed = _observed_target(ctx, target_batch, batch, history)
            for center_kind in center_kind_values:
                center, metadata = build_observed_center(
                    center_kind,
                    codec_model=ctx.codec.model,
                    coordinate_batch=batch,
                    target_batch=observed,
                    adapter=ctx.adapter,
                    statistics=ctx.statistics.to(device=ctx.device),
                    history_frames=history,
                    codec_hash=ctx.codec.codec_state_hash,
                    data_hash=ctx.data_hash,
                )
                center_coordinates = _decode_fields(ctx, observed, _merge_center_with_observed(ctx, observed, center))
                template = metadata.get("template_coordinates")
                if template is None:
                    template = batch.x.clone()
                    template[history:] = batch.x[history - 1].unsqueeze(0)
                metrics = _source_metric_row(
                    center_coordinates,
                    template,
                    batch,
                    history,
                    oracle_coordinates,
                )
                metrics["template_future_internal_displacement_rms"] = _coordinate_rms(
                    template[history + 1:],
                    template[history:-1],
                    (batch.frame_mask.index_select(0, batch.abid).transpose(0, 1)[history + 1:] & batch.loss_mask.unsqueeze(0).expand(batch.frames, -1)[history + 1:])
                    if history + 1 < batch.frames else batch.loss_mask.new_zeros((0, batch.num_atoms)),
                ) if history + 1 < batch.frames else 0.0
                metrics.update(
                    {
                        "sample_id": sample_id,
                        "history_frames": history,
                        "center_kind": center_kind,
                        "uses_future_coordinates": False,
                        "template_source": "observed_prefix_plus_repeated_last_observed_coordinate",
                    }
                )
                metrics["implementation_pass"] = (
                    metrics["template_raw_rmsd"] <= metrics["threshold_template_raw_rmsd"]
                    and metrics["template_bond_rmse"] <= 0.10
                    and metrics["decoded_future_internal_displacement_rms"] <= 1.0e-4
                )
                rows.append(metrics)
    repeat_rows = [row for row in rows if row["center_kind"] == center_kind_values[0]]
    fallback_rows = [row for row in rows if row["center_kind"] == center_kind_values[1]]
    repeat_pass = bool(repeat_rows) and all(bool(row["implementation_pass"]) for row in repeat_rows)
    fallback_pass = bool(fallback_rows) and all(
        row["template_bond_rmse"] <= 0.20 and math.isfinite(row["template_raw_rmsd"])
        for row in fallback_rows
    )
    if repeat_pass:
        selected_center = center_kind_values[0]
    elif fallback_pass:
        selected_center = center_kind_values[1]
    else:
        selected_center = None
    result = {
        "schema": "pvb.dit.state_detail.source_ab.source_check.v1",
        "status": "PASS" if selected_center is not None else "FAIL",
        "fixed_clip_count": 8,
        "selected_center_kind": selected_center,
        "selection_rule": "repeat center wins implementation checks; fallback only if repeat fails and block center is finite with bond RMSE <=0.20 A",
        "thresholds": {
            "template_raw_rmsd": "max(0.10 A, 5 * clip codec oracle raw RMSD)",
            "repeat_template_bond_rmse": 0.10,
            "fallback_template_bond_rmse": 0.20,
            "decoded_future_internal_displacement_rms": 1.0e-4,
        },
        "rows": rows,
        "test_payload_opened": False,
    }
    _write_json(ctx.output_dir / "source_check.json", result)
    return result


def _verify(ctx: ExperimentContext) -> dict[str, Any]:
    candidate = ctx.cfg["candidate"]
    old_checkpoint = _resolve(candidate["old_dit_checkpoint"], _resolve(ctx.cfg["pilot_root"]))
    payload = torch.load(old_checkpoint, map_location="cpu", weights_only=False)
    sampler = _make_sampler(ctx.data.train, seed=int(ctx.cfg["seed"]["training"]), clips_per_trajectory=24)
    indices = tuple(int(value) for value in sampler.global_batches[0])
    latent, batch_cpu, batch, target_batch, _ = _prepare_encoded(ctx, ctx.data.train, indices)
    center_kind = str(ctx.cfg["source"]["center_kind"])
    rows: list[dict[str, Any]] = []
    for history in HISTORY_SCHEDULE:
        observed = _observed_target(ctx, target_batch, batch, history)
        center, _metadata = build_observed_center(
            center_kind,
            codec_model=ctx.codec.model,
            coordinate_batch=batch,
            target_batch=observed,
            adapter=ctx.adapter,
            statistics=ctx.statistics.to(device=ctx.device),
            history_frames=history,
            codec_hash=ctx.codec.codec_state_hash,
            data_hash=ctx.data_hash,
        )
        normalized = ctx.statistics.to(device=ctx.device).normalize(observed)
        flow = RectifiedFlowObjective()
        generator_a = torch.Generator(device=ctx.device).manual_seed(424242 + history)
        generator_b = torch.Generator(device=ctx.device).manual_seed(424242 + history)
        gaussian = flow.sample(normalized, generator=generator_a)
        conditional = flow.sample(
            normalized,
            generator=generator_b,
            source_center=center,
            source_mode="conditional",
        )
        center_diff = max(
            float((getattr(conditional.source, name) - getattr(gaussian.source, name) - getattr(center, name)).abs().max().cpu())
            for name in ("state_h", "detail_h", "state_v", "detail_v")
        )
        target_diff = max(
            float((getattr(conditional.target, name) - getattr(gaussian.target, name) + getattr(center, name)).abs().max().cpu())
            for name in ("state_h", "detail_h", "state_v", "detail_v")
        )
        model = _new_model(ctx.cfg, ctx.adapter).to(ctx.device)
        model.load_state_dict(payload["model_state"], strict=True)
        model.eval()
        generated, generation_meta = generate_state_detail_latent(
            model,
            ctx.adapter,
            observed,
            ctx.statistics.to(device=ctx.device),
            steps=int(ctx.cfg["schedule"]["smoke_steps"]),
            seed=7000 + history,
            source_center=center,
            source_mode="conditional",
        )
        decoded = ctx.codec.model.decode(generated).x_hat.float()
        finite = bool(torch.isfinite(decoded).all() and all(torch.isfinite(getattr(generated, name)).all() for name in ("state_h", "detail_h", "state_v", "detail_v")))
        rows.append(
            {
                "history_frames": history,
                "source_mode": "conditional",
                "source_center_kind": center_kind,
                "source_difference_max_abs": center_diff,
                "target_difference_plus_center_max_abs": target_diff,
                "generation": generation_meta,
                "decoded_shape": list(decoded.shape),
                "finite": finite,
                "observed_clamp_exact": bool(generation_meta["observed_clamp_exact"]),
            }
        )
        del model, generated, decoded
    result = {
        "schema": "pvb.dit.state_detail.source_ab.verification.v1",
        "status": "PASS" if all(row["finite"] and row["observed_clamp_exact"] and row["source_difference_max_abs"] <= 1e-5 and row["target_difference_plus_center_max_abs"] <= 1e-5 for row in rows) else "FAIL",
        "device": _cuda_info(ctx.device),
        "rows": rows,
        "test_payload_opened": False,
    }
    _write_json(ctx.output_dir / "verification.json", result)
    return result


def _profile_prepare(ctx: ExperimentContext, source_decision: Mapping[str, Any]) -> dict[str, Any]:
    center_kind = source_decision.get("selected_center_kind")
    if not center_kind:
        raise RuntimeError("cannot prepare a budget without a passing source decision")
    init_state, init_hash = _shared_initialization(ctx.cfg, ctx.device)
    _write_json(
        ctx.output_dir / "shared_initialization.json",
        {"schema": "pvb.dit.state_detail.source_ab.shared_init.v1", "init_hash": init_hash, "parameter_count": sum(value.numel() for value in init_state.values())},
    )
    torch.save({"schema": "pvb.dit.state_detail.source_ab.shared_init.v1", "init_hash": init_hash, "state_dict": init_state}, ctx.output_dir / "shared_initialization.pt")
    sampler = _make_sampler(ctx.data.train, seed=int(ctx.cfg["seed"]["training"]), clips_per_trajectory=int(ctx.cfg["schedule"]["clips_per_trajectory"]))
    batches = list(sampler.global_batches)[: int(ctx.cfg["budget"]["warmup_updates"]) + int(ctx.cfg["budget"]["measured_updates"])]
    if len(batches) < int(ctx.cfg["budget"]["warmup_updates"]) + int(ctx.cfg["budget"]["measured_updates"]):
        raise RuntimeError("fewer than 25 representative train batches")
    cache_profile: dict[str, Any] = {}
    profile_rows: dict[str, Any] = {}
    for arm in ARMS:
        trainer = _make_trainer(
            ctx,
            source_mode=arm,
            center_kind=str(center_kind),
            target_steps=20000,
            init_state=init_state,
            init_hash=init_hash,
        )
        generator = torch.Generator(device=ctx.device).manual_seed(int(ctx.cfg["seed"]["training"]))
        timings: list[float] = []
        prep_timings: list[float] = []
        for update, indices in enumerate(batches):
            history = HISTORY_SCHEDULE[update % len(HISTORY_SCHEDULE)]
            started = time.perf_counter()
            _latent, _batch_cpu, batch, target_batch, _ = _prepare_encoded(ctx, ctx.data.train, indices)
            source_center = None
            prep_started = time.perf_counter()
            if arm == "conditional":
                source_center, _meta = _source_center(
                    ctx,
                    batch,
                    _observed_target(ctx, target_batch, batch, history),
                    history,
                    str(center_kind),
                    LatentFieldCache(mode="disabled"),
                )
            _sync(ctx.device)
            prep_timings.append(time.perf_counter() - prep_started)
            trainer.train_step(
                _observed_target(ctx, target_batch, batch, history),
                generator=generator,
                source_center=source_center,
            )
            _sync(ctx.device)
            timings.append(time.perf_counter() - started)
        measured = timings[int(ctx.cfg["budget"]["warmup_updates"]):]
        measured_prep = prep_timings[int(ctx.cfg["budget"]["warmup_updates"]):]
        profile_rows[arm] = {
            "warmup_updates": int(ctx.cfg["budget"]["warmup_updates"]),
            "measured_updates": int(ctx.cfg["budget"]["measured_updates"]),
            "wall_seconds": measured,
            "p50_seconds": sorted(measured)[len(measured) // 2],
            "p90_seconds": sorted(measured)[max(0, int(math.ceil(0.9 * len(measured))) - 1)],
            "prep_seconds": measured_prep,
            "prep_p90_seconds": sorted(measured_prep)[max(0, int(math.ceil(0.9 * len(measured_prep))) - 1)],
            "initialization_hash": init_hash,
            "source_mode": arm,
            "center_kind": center_kind,
        }
        del trainer
        torch.cuda.empty_cache()
    sample_indices = tuple(int(value) for value in batches[0])
    _latent, _batch_cpu, batch, target_batch, _ = _prepare_encoded(ctx, ctx.data.train, sample_indices)
    observed = _observed_target(ctx, target_batch, batch, 4)
    direct_cache = LatentFieldCache(mode="disabled")
    direct_started = time.perf_counter()
    direct_center, _ = _source_center(ctx, batch, observed, 4, str(center_kind), direct_cache)
    _sync(ctx.device)
    direct_seconds = time.perf_counter() - direct_started
    ram_cache = LatentFieldCache(mode="ram", max_bytes=int(float(ctx.cfg["cache"]["ram_gib"]) * 1024**3))
    ram_started = time.perf_counter()
    cached_center, _ = _source_center(ctx, batch, observed, 4, str(center_kind), ram_cache)
    _sync(ctx.device)
    first_cached_seconds = time.perf_counter() - ram_started
    hit_started = time.perf_counter()
    hit_center, hit_meta = _source_center(ctx, batch, observed, 4, str(center_kind), ram_cache)
    _sync(ctx.device)
    hit_seconds = time.perf_counter() - hit_started
    cache_profile = {
        "mode_selected": str(ctx.cfg["cache"]["mode"]),
        "direct_seconds": direct_seconds,
        "ram_miss_seconds": first_cached_seconds,
        "ram_hit_seconds": hit_seconds,
        "ram_hit_equal": all(torch.equal(getattr(cached_center, name), getattr(hit_center, name)) for name in cached_center.names()),
        "ram_stats": ram_cache.stats().as_dict(),
        "source_center_bytes": sum(value.numel() * value.element_size() for value in cached_center.as_dict().values()),
        "cache_key": hit_meta.get("cache_key"),
        "decision": "disabled_for_science_until_full-clip cache fits the frozen 8 GiB RAM bound",
    }
    gpu_seconds_limit = float(ctx.cfg["budget"]["gpu_hours_total"]) * 3600.0
    eval_estimate = 1800.0
    budget_rows: list[dict[str, Any]] = []
    selected_steps = None
    for candidate_steps in (20000, 10000, 4500):
        train_seconds = float(candidate_steps) * sum(profile_rows[arm]["p90_seconds"] for arm in ARMS)
        reserved = max(float(ctx.cfg["budget"]["reserved_evaluation_fraction"]) * gpu_seconds_limit, 1.3 * eval_estimate)
        feasible = 1.3 * train_seconds <= gpu_seconds_limit - reserved and train_seconds <= float(ctx.cfg["budget"]["wall_hours_total"]) * 3600.0
        row = {"candidate_steps": candidate_steps, "estimated_training_gpu_seconds": train_seconds, "estimated_evaluation_gpu_seconds": eval_estimate, "reserved_evaluation_seconds": reserved, "feasible": feasible}
        budget_rows.append(row)
        if feasible and selected_steps is None:
            selected_steps = candidate_steps
    if selected_steps is None:
        selected_steps = 0
    budget = {
        "schema": "pvb.dit.state_detail.source_ab.budget.v1",
        "status": "PASS" if selected_steps else "PARTIAL_BUDGET",
        "selected_common_steps": selected_steps,
        "gpu_seconds_limit": gpu_seconds_limit,
        "wall_seconds_limit": float(ctx.cfg["budget"]["wall_hours_total"]) * 3600.0,
        "profile": profile_rows,
        "cache_profile": cache_profile,
        "candidates": budget_rows,
        "formula": "1.3*(candidate_steps*(p90_gaussian+p90_conditional)+estimated_prep) <= B-E; E=max(0.25B,1.3*estimated_eval)",
        "estimated_prep_gpu_seconds": 0.0,
        "test_payload_opened": False,
    }
    _write_json(ctx.output_dir / "backend_profile.json", {"schema": "pvb.dit.state_detail.source_ab.runner_profile.v1", "rows": profile_rows, "cache": cache_profile, "device": _cuda_info(ctx.device)})
    _write_json(ctx.output_dir / "budget.json", budget)
    if not selected_steps:
        raise RuntimeError("no common training endpoint fits the frozen budget")
    contract = {
        "schema": "pvb.dit.state_detail.source_ab.experiment_contract.v1",
        "code_commit": _git_commit(),
        "source_center_kind": center_kind,
        "source_contracts": {arm: source_contract(source_mode=arm, center_kind=str(center_kind), statistics=ctx.statistics, sigma=float(ctx.cfg["source"]["sigma"])) for arm in ARMS},
        "init_hash": init_hash,
        "statistics_hash": ctx.statistics.hash,
        "statistics_file_sha256": ctx.statistics_file_sha256,
        "codec_checkpoint_sha256": ctx.codec.checkpoint_sha256,
        "codec_state_hash": ctx.codec.codec_state_hash,
        "data_hash": ctx.data_hash,
        "history_schedule": list(HISTORY_SCHEDULE),
        "candidate_steps": list(CHECKPOINT_STEPS),
        "selected_common_steps": selected_steps,
        "training_seed": int(ctx.cfg["seed"]["training"]),
        "validation_seed": int(ctx.cfg["seed"]["validation"]),
        "backend": "factorized_v2",
        "cache": cache_profile,
        "test_payload_opened": False,
    }
    _write_json(ctx.output_dir / "experiment_contract.json", contract)
    return budget


def _validation_rf_loss(ctx: ExperimentContext, trainer: DiTTrainer, plan: Any, *, source_mode: str, center_kind: str, step: int) -> dict[str, Any]:
    numerators = {name: 0.0 for name in ("state_h", "detail_h", "state_v", "detail_v")}
    counts = {name: 0 for name in numerators}
    flow = RectifiedFlowObjective()
    trainer.model.eval()
    for history in HISTORY_SCHEDULE:
        for batch_offset, local_indices in enumerate(plan.batches):
            latent, batch_cpu, batch = _encode_batch(plan.subset, local_indices, codec=ctx.codec, adapter=ctx.adapter, data_hash=ctx.data_hash, device=ctx.device)
            target_batch = ctx.adapter.pack(latent, codec_hash=ctx.codec.codec_state_hash, data_hash=ctx.data_hash, origin_from_latent=True, loss_mask=batch.loss_mask)
            observed = _observed_target(ctx, target_batch, batch, history)
            center = None
            if source_mode == "conditional":
                center, _ = _source_center(ctx, batch, observed, history, center_kind, LatentFieldCache(mode="disabled"))
            normalized = trainer._normalise_batch(observed)
            generator = torch.Generator(device=ctx.device).manual_seed(stable_seed(int(ctx.cfg["seed"]["validation"]), "|".join(batch.sample_id), history, step, "validation_rf"))
            with torch.no_grad(), trainer.autocast_context():
                sample = flow.sample(normalized, generator=generator, source_center=center, source_mode=source_mode)
                prediction = trainer.model(normalized.with_fields(sample.interpolated), sample.tau)
                loss = flow.loss(prediction, sample.target, normalized)
            for name, value in loss.fields.items():
                count = int(loss.valid_elements[name])
                counts[name] += count
                numerators[name] += float(value.detach().float().cpu()) * count
    fields = {name: numerators[name] / max(counts[name], 1) for name in numerators}
    return {"step": int(step), "source_mode": source_mode, "history": list(HISTORY_SCHEDULE), "total": sum(fields.values()) / 4.0, "fields": fields, "valid_elements": counts, "sample_count": 72}


def _checkpoint_payload(trainer: DiTTrainer, *, arm: str, center_kind: str, init_hash: str, schedule_hash: str, generator: torch.Generator, cursor: Mapping[str, int]) -> dict[str, Any]:
    payload = trainer.checkpoint_payload()
    payload["source_ab"] = {
        "schema": "pvb.dit.state_detail.source_ab.checkpoint.v1",
        "arm": arm,
        "center_kind": center_kind,
        "init_hash": init_hash,
        "schedule_hash": schedule_hash,
        "cursor": dict(cursor),
        "actual_optimizer_updates": int(trainer.step),
        "generator_state": generator.get_state(),
        "test_payload_opened": False,
    }
    return payload


def _save_checkpoint(ctx: ExperimentContext, trainer: DiTTrainer, *, arm: str, center_kind: str, init_hash: str, schedule_hash: str, generator: torch.Generator, step: int, cursor: Mapping[str, int]) -> Path:
    path = ctx.output_dir / arm / f"checkpoint_step{int(step):06d}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(_checkpoint_payload(trainer, arm=arm, center_kind=center_kind, init_hash=init_hash, schedule_hash=schedule_hash, generator=generator, cursor=cursor), path)
    latest = ctx.output_dir / arm / "latest.pt"
    torch.save(_checkpoint_payload(trainer, arm=arm, center_kind=center_kind, init_hash=init_hash, schedule_hash=schedule_hash, generator=generator, cursor=cursor), latest)
    return path


def _train(ctx: ExperimentContext, *, arm_selection: str, resume: bool) -> dict[str, Any]:
    budget = json.loads((ctx.output_dir / "budget.json").read_text())
    contract = json.loads((ctx.output_dir / "experiment_contract.json").read_text())
    if budget.get("status") not in ("PASS", "PARTIAL_BUDGET") or not int(budget.get("selected_common_steps", 0)):
        raise RuntimeError("prepare did not freeze a usable common budget")
    target_steps = int(budget["selected_common_steps"])
    center_kind = str(contract["source_center_kind"])
    init_payload = torch.load(ctx.output_dir / "shared_initialization.pt", map_location="cpu", weights_only=False)
    init_state = init_payload["state_dict"]
    init_hash = str(init_payload["init_hash"])
    if init_hash != str(contract["init_hash"]):
        raise RuntimeError("shared initialization hash differs from experiment contract")
    arms = ARMS if arm_selection == "both" else (arm_selection,)
    trainers = {arm: _make_trainer(ctx, source_mode=arm, center_kind=center_kind, target_steps=target_steps, init_state=init_state, init_hash=init_hash) for arm in arms}
    generators = {arm: torch.Generator(device=ctx.device).manual_seed(int(ctx.cfg["seed"]["training"])) for arm in arms}
    start_step = 0
    if resume:
        loaded = []
        for arm, trainer in trainers.items():
            path = ctx.output_dir / arm / "latest.pt"
            if not path.is_file():
                raise FileNotFoundError(f"resume checkpoint missing: {path}")
            payload = trainer.load_checkpoint(path, map_location=ctx.device)
            source_ab = payload.get("source_ab", {})
            if source_ab.get("arm") != arm or source_ab.get("init_hash") != init_hash or source_ab.get("center_kind") != center_kind:
                raise RuntimeError("resume source contract mismatch")
            generators[arm].set_state(source_ab["generator_state"])
            loaded.append(int(trainer.step))
        if len(set(loaded)) != 1:
            raise RuntimeError("arms do not resume from a common step")
        start_step = loaded[0]
    sampler = _make_sampler(ctx.data.train, seed=int(ctx.cfg["seed"]["training"]), clips_per_trajectory=int(ctx.cfg["schedule"]["clips_per_trajectory"]))
    sampler.set_epoch(0)
    base_batches = tuple(tuple(int(value) for value in batch) for batch in sampler.global_batches)
    schedule_hash = _canonical_hash({"batches": [list(batch) for batch in base_batches], "history": list(HISTORY_SCHEDULE), "seed": int(ctx.cfg["seed"]["training"])})
    plan = _make_validation_plan(ctx.data.valid)
    histories: dict[str, list[dict[str, Any]]] = {arm: [] for arm in arms}
    started = time.perf_counter()
    cache = LatentFieldCache(mode=str(budget.get("cache", {}).get("mode_selected", "disabled")))
    for step in range(start_step + 1, target_steps + 1):
        epoch = (step - 1) // max(len(base_batches), 1)
        batch_offset = (step - 1) % max(len(base_batches), 1)
        if batch_offset == 0:
            sampler.set_epoch(epoch)
            base_batches = tuple(tuple(int(value) for value in batch) for batch in sampler.global_batches)
            schedule_hash = _canonical_hash({"batches": [list(batch) for batch in base_batches], "history": list(HISTORY_SCHEDULE), "seed": int(ctx.cfg["seed"]["training"]), "epoch": epoch})
        indices = base_batches[batch_offset]
        history = HISTORY_SCHEDULE[(step - 1) % len(HISTORY_SCHEDULE)]
        latent, batch_cpu, batch, target_batch, _ = _prepare_encoded(ctx, ctx.data.train, indices)
        observed = _observed_target(ctx, target_batch, batch, history)
        center = None
        if "conditional" in arms:
            center, _ = _source_center(ctx, batch, observed, history, center_kind, cache)
        _sync(ctx.device)
        for arm, trainer in trainers.items():
            row = trainer.train_step(observed, generator=generators[arm], source_center=center if arm == "conditional" else None)
            row.update({"arm": arm, "history_frames": history, "source_mode": arm, "center_kind": center_kind})
            histories[arm].append(row)
            _append_jsonl(ctx.output_dir / arm / "train_history.jsonl", row)
        if step % int(ctx.cfg["schedule"]["validation_interval"]) == 0 or step in CHECKPOINT_STEPS or step == target_steps:
            for arm, trainer in trainers.items():
                validation = _validation_rf_loss(ctx, trainer, plan, source_mode=arm, center_kind=center_kind, step=step)
                _append_jsonl(ctx.output_dir / arm / "validation_history.jsonl", validation)
        if step % int(ctx.cfg["schedule"]["generation_interval"]) == 0:
            _write_json(ctx.output_dir / "monitor" / f"step{step:06d}.json", {"step": step, "arms": list(arms), "note": "generation monitor recorded at fixed interval; final paired evaluation is authoritative"})
        if step in CHECKPOINT_STEPS or step == target_steps:
            for arm, trainer in trainers.items():
                _save_checkpoint(ctx, trainer, arm=arm, center_kind=center_kind, init_hash=init_hash, schedule_hash=schedule_hash, generator=generators[arm], step=step, cursor={"epoch": epoch, "batch_index": batch_offset + 1})
    elapsed = time.perf_counter() - started
    result = {"schema": "pvb.dit.state_detail.source_ab.train.v1", "status": "PASS", "target_steps": target_steps, "arms": {arm: {"actual_optimizer_updates": trainers[arm].step, "elapsed_wall_seconds": elapsed, "initialization_hash": init_hash, "history_rows": len(histories[arm])} for arm in arms}, "cache": cache.stats().as_dict(), "test_payload_opened": False}
    _write_json(ctx.output_dir / "train_summary.json", result)
    return result


def _evaluate_arm(ctx: ExperimentContext, arm: str, checkpoint_path: Path, center_kind: str, init_state: Mapping[str, Tensor], init_hash: str, common_step: int) -> dict[str, Any]:
    trainer = _make_trainer(ctx, source_mode=arm, center_kind=center_kind, target_steps=common_step, init_state=init_state, init_hash=init_hash)
    trainer.load_checkpoint(checkpoint_path, map_location=ctx.device)
    trainer.model.eval()
    plan = _make_validation_plan(ctx.data.valid)
    rows: list[dict[str, Any]] = []
    for history in HISTORY_SCHEDULE:
        for local_indices in plan.batches:
            latent, batch_cpu, batch = _encode_batch(plan.subset, local_indices, codec=ctx.codec, adapter=ctx.adapter, data_hash=ctx.data_hash, device=ctx.device)
            target_batch = ctx.adapter.pack(latent, codec_hash=ctx.codec.codec_state_hash, data_hash=ctx.data_hash, origin_from_latent=True, loss_mask=batch.loss_mask)
            observed = _observed_target(ctx, target_batch, batch, history)
            center = None
            if arm == "conditional":
                center, _ = _source_center(ctx, batch, observed, history, center_kind, LatentFieldCache(mode="disabled"))
            seed = stable_seed(int(ctx.cfg["seed"]["validation"]), "|".join(batch.sample_id), history, 0, f"final_{arm}")
            generated, metadata = generate_state_detail_latent(trainer.model, ctx.adapter, observed, ctx.statistics.to(device=ctx.device), steps=int(ctx.cfg["evaluation"]["final_steps"]), seed=seed, source_center=center, source_mode=arm)
            decoded = ctx.codec.model.decode(generated).x_hat.float()
            result = evaluate_oracle_vs_generated(ctx.codec.model, oracle_latent=latent, generated_latent=generated, batch=batch, history_frames=history, trunk_runtime={"source_mode": arm, "common_step": common_step, "sampling_steps": int(ctx.cfg["evaluation"]["final_steps"])})
            rows.append({"sample_id": batch.sample_id[0] if len(batch.sample_id) == 1 else list(batch.sample_id), "history_frames": history, "steps": int(ctx.cfg["evaluation"]["final_steps"]), "draw_id": 0, "arm": arm, "generation": metadata, "evaluation": result.as_dict(), "diagnostic_metrics": trajectory_metric_record(decoded, batch.x.float(), batch, history)})
    subset_rows: list[dict[str, Any]] = []
    wanted = []
    for index, row in enumerate(ctx.data.valid._index):
        parsed = parse_sample_id(str(row[0]))
        if parsed["replica"] == "R1" and parsed["window"] == 30:
            wanted.append((index, str(row[0])))
    wanted.sort(key=lambda item: item[1])
    for index, sample_id in wanted:
        latent, batch_cpu, batch = _encode_batch(ctx.data.valid, (index,), codec=ctx.codec, adapter=ctx.adapter, data_hash=ctx.data_hash, device=ctx.device)
        target_batch = ctx.adapter.pack(latent, codec_hash=ctx.codec.codec_state_hash, data_hash=ctx.data_hash, origin_from_latent=True, loss_mask=batch.loss_mask)
        for history in HISTORY_SCHEDULE:
            observed = _observed_target(ctx, target_batch, batch, history)
            center = None
            if arm == "conditional":
                center, _ = _source_center(ctx, batch, observed, history, center_kind, LatentFieldCache(mode="disabled"))
            for steps in (int(ctx.cfg["evaluation"]["short_steps"]), int(ctx.cfg["evaluation"]["final_steps"])):
                for draw_id in (int(value) for value in ctx.cfg["evaluation"]["subset_draws"]):
                    seed = stable_seed(int(ctx.cfg["seed"]["validation"]), sample_id, history, draw_id, f"subset_{steps}_{arm}")
                    generated, metadata = generate_state_detail_latent(trainer.model, ctx.adapter, observed, ctx.statistics.to(device=ctx.device), steps=steps, seed=seed, source_center=center, source_mode=arm)
                    decoded = ctx.codec.model.decode(generated).x_hat.float()
                    subset_rows.append({"sample_id": sample_id, "history_frames": history, "steps": steps, "draw_id": draw_id, "arm": arm, "generation": metadata, "diagnostic_metrics": trajectory_metric_record(decoded, batch.x.float(), batch, history)})
    result = {"schema": "pvb.dit.state_detail.source_ab.evaluation_arm.v1", "arm": arm, "common_step": common_step, "checkpoint": str(checkpoint_path), "main_rows": len(rows), "subset_rows": len(subset_rows), "main_aggregate": aggregate_rows(rows, metric_path=("diagnostic_metrics", "future")), "subset_aggregate": aggregate_rows(subset_rows, metric_path=("diagnostic_metrics", "future")), "test_payload_opened": False}
    _write_json(ctx.output_dir / arm / "evaluation.json", result)
    with (ctx.output_dir / arm / "generation_metrics.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows + subset_rows:
            handle.write(json.dumps(_safe(row), sort_keys=True) + "\n")
    return result


def _evaluate(ctx: ExperimentContext) -> dict[str, Any]:
    budget = json.loads((ctx.output_dir / "budget.json").read_text())
    contract = json.loads((ctx.output_dir / "experiment_contract.json").read_text())
    common_step = int(budget["selected_common_steps"])
    init_payload = torch.load(ctx.output_dir / "shared_initialization.pt", map_location="cpu", weights_only=False)
    init_state, init_hash = init_payload["state_dict"], str(init_payload["init_hash"])
    results = {}
    for arm in ARMS:
        checkpoint = ctx.output_dir / arm / f"checkpoint_step{common_step:06d}.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"common checkpoint missing for {arm}: {checkpoint}")
        results[arm] = _evaluate_arm(ctx, arm, checkpoint, str(contract["source_center_kind"]), init_state, init_hash, common_step)
    comparison = {"schema": "pvb.dit.state_detail.source_ab.comparison.v1", "common_step": common_step, "source_center_kind": contract["source_center_kind"], "arms": results, "selection": "common fixed step; no cross-source RF-loss ranking", "test_payload_opened": False}
    _write_json(ctx.output_dir / "comparison.json", comparison)
    lines = ["# R4 source A/B v1", "", f"Common endpoint: {common_step} optimizer updates", "", "Gaussian and conditional arms share initialization, data schedule, H4/H8 history, tau/noise generator, codec and evaluation clips.", "", "| arm | main rows | subset rows | main future aligned RMSD | main future bond RMSE |", "|---|---:|---:|---:|---:|"]
    for arm in ARMS:
        future = results[arm]["main_aggregate"].get("system_equal", {})
        lines.append(f"| {arm} | {results[arm]['main_rows']} | {results[arm]['subset_rows']} | {future.get('aligned_rmsd', 'n/a')} | {future.get('bond_rmse', 'n/a')} |")
    lines.extend(["", "The table is descriptive; source arms are not selected by directly comparing RF losses. Test payloads were not opened.", ""])
    (ctx.output_dir / "report.md").write_text("\n".join(lines))
    return comparison


def _summarize(ctx: ExperimentContext) -> dict[str, Any]:
    comparison_path = ctx.output_dir / "comparison.json"
    if not comparison_path.is_file():
        raise FileNotFoundError("evaluation comparison is missing")
    comparison = json.loads(comparison_path.read_text())
    result = {"schema": "pvb.dit.state_detail.source_ab.summary.v1", "status": "SOURCE_AB_V1_COMPLETE_FOR_REVIEW", "comparison": comparison, "evidence": {"output_root": str(ctx.output_dir), "source_check": str(ctx.output_dir / "source_check.json"), "budget": str(ctx.output_dir / "budget.json"), "contract": str(ctx.output_dir / "experiment_contract.json"), "report": str(ctx.output_dir / "report.md")}, "test_payload_opened": False}
    _write_json(ctx.output_dir / "summary.json", result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config/dit_source_ab_v1.yaml")
    parser.add_argument("--stage", choices=("preflight", "verify", "source_check", "prepare", "train", "evaluate", "summarize", "all"), required=True)
    parser.add_argument("--arm", choices=("gaussian", "conditional", "both"), default="both")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--manifest-root", type=Path, default=None)
    parser.add_argument("--pilot-root", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    return parser


def _resolved_args(args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    cfg = _load_config(args.config.resolve())
    if args.run_id:
        cfg["run_id"] = args.run_id
    if args.manifest_root is not None:
        cfg["manifest_root"] = str(args.manifest_root)
    if args.pilot_root is not None:
        cfg["pilot_root"] = str(args.pilot_root)
    if args.output_root is not None:
        cfg["output_root"] = str(args.output_root / str(cfg["run_id"]))
    output_dir = _resolve(cfg["output_root"], PROJECT_ROOT)
    return cfg, output_dir


def main() -> None:
    args = _parser().parse_args()
    cfg, output_dir = _resolved_args(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.stage == "preflight":
        result = _manifest_preflight(cfg)
        result.update({"schema": "pvb.dit.state_detail.source_ab.preflight.v1", "code_commit": _git_commit(), "run_id": cfg["run_id"]})
        _write_json(output_dir / "preflight.json", result)
        print(json.dumps(_safe(result), indent=2, sort_keys=True))
        return
    device = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.stage == "all":
        stages = ("preflight", "verify", "source_check", "prepare", "train", "evaluate", "summarize")
        for stage in stages:
            if stage == "preflight":
                result = _manifest_preflight(cfg)
                result.update({"schema": "pvb.dit.state_detail.source_ab.preflight.v1", "code_commit": _git_commit(), "run_id": cfg["run_id"]})
                _write_json(output_dir / "preflight.json", result)
            else:
                ctx = _load_context(cfg, output_dir, device)
                try:
                    if stage == "verify":
                        result = _verify(ctx)
                    elif stage == "source_check":
                        result = _source_check(ctx)
                    elif stage == "prepare":
                        result = _profile_prepare(ctx, json.loads((output_dir / "source_check.json").read_text()))
                    elif stage == "train":
                        result = _train(ctx, arm_selection="both", resume=args.resume)
                    elif stage == "evaluate":
                        result = _evaluate(ctx)
                    else:
                        result = _summarize(ctx)
                finally:
                    ctx.close()
            print(json.dumps(_safe({"stage": stage, "result": result}), indent=2, sort_keys=True), flush=True)
        return
    ctx = _load_context(cfg, output_dir, device)
    try:
        if args.stage == "verify":
            result = _verify(ctx)
        elif args.stage == "source_check":
            result = _source_check(ctx)
        elif args.stage == "prepare":
            result = _profile_prepare(ctx, json.loads((output_dir / "source_check.json").read_text()))
        elif args.stage == "train":
            result = _train(ctx, arm_selection=args.arm, resume=args.resume)
        elif args.stage == "evaluate":
            result = _evaluate(ctx)
        elif args.stage == "summarize":
            result = _summarize(ctx)
        else:
            raise ValueError(f"unsupported stage {args.stage}")
    finally:
        ctx.close()
    print(json.dumps(_safe(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
