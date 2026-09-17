#!/usr/bin/env python
"""Reassess the frozen R4 source-A/B checkpoints without training."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.clip_dataset import ClipMMapDataset, collate_clip_records
from evaluation.codec_evaluation import _frame_view
from evaluation.dit_diagnostics import observed_coordinate_baseline, parse_sample_id
from evaluation.dit_reassessment import (
    DRAW_IDS,
    FIXED_EULER_STEPS,
    HISTORIES,
    REASSESSMENT_SCHEMA,
    aggregate_generated_rows,
    future_diversity,
    generate_fixed_noise_latent,
    make_fixed_noise,
    _mean_mapping,
    reassessment_seed,
    reassessment_trajectory_metrics,
    reaggregate_legacy_jsonl,
    seed_contract,
    sha256_file,
    tensor_bytes_hash,
)
from module.latent_flow_source import build_observed_center
from scripts.run_dit_source_ab import (
    _load_context,
    _new_model,
    _observed_target,
    _resolve,
)
from scripts.run_state_detail_dit_pilot import _encode_batch, _load_data
from trainer.dit_trainer import module_state_hash


ARM_NAMES = ("gaussian", "conditional")
SAMPLE_ID_RE = re.compile(r"^(?P<system>.+)_(?P<replica>R[0-9]+)_w(?P<window>[0-9]+)$")


def _git_commit() -> str:
    override = os.environ.get("MOLVID_SOURCE_COMMIT", "").strip()
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return override or "unknown"


def _code_inventory() -> dict[str, str]:
    paths = (
        "scripts/run_dit_source_reassessment.py",
        "scripts/write_dit_reassessment_interpretation.py",
        "evaluation/dit_reassessment.py",
        "evaluation/dit_reassessment_interpretation.py",
        "scripts/run_dit_source_ab.py",
        "scripts/run_state_detail_dit_pilot.py",
        "module/latent_rectified_flow.py",
        "module/latent_flow_source.py",
        "module/state_detail_latent_adapter.py",
        "module/state_detail_codec_v2.py",
        "module/molecular_dit.py",
        "evaluation/codec_evaluation.py",
        "data/clip_dataset.py",
    )
    return {path: sha256_file(PROJECT_ROOT / path) for path in paths}


def _working_diff_sha256() -> str:
    try:
        diff = subprocess.check_output(
            ["git", "diff", "--no-ext-diff", "--binary"],
            cwd=PROJECT_ROOT,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return hashlib.sha256(diff).hexdigest()


def _gpu_inventory() -> dict[str, Any]:
    result: dict[str, Any] = {}
    commands = {
        "devices": ["nvidia-smi", "--query-gpu=index,uuid,memory.used,memory.total", "--format=csv,noheader,nounits"],
        "compute_apps": ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv,noheader,nounits"],
    }
    for name, command in commands.items():
        try:
            output = subprocess.check_output(command, text=True, stderr=subprocess.STDOUT)
            result[name] = [line.strip() for line in output.splitlines() if line.strip()]
        except (OSError, subprocess.CalledProcessError) as exc:
            result[name] = {"error": str(exc)}
    return result


def _safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            item = value.detach().float().item()
            return item if math.isfinite(item) else None
        return value.detach().float().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(_safe(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_safe(row), sort_keys=True) + "\n")
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"expected a JSON object: {path}")
    return dict(value)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if isinstance(value, Mapping):
                rows.append(dict(value))
    return rows


def _load_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("reassessment config must be a mapping")
    return json.loads(json.dumps(value))


def _resolved_args(args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    cfg = _load_config(args.config.resolve())
    if args.run_id:
        cfg["run_id"] = args.run_id
    if args.output_root is not None:
        cfg["output_root"] = str(args.output_root / str(cfg["run_id"]))
    output_dir = _resolve(cfg["output_root"], PROJECT_ROOT)
    return cfg, output_dir


def _checkpoint_paths(cfg: Mapping[str, Any]) -> dict[str, Path]:
    root = Path(str(cfg["legacy_output_root"]))
    return {
        arm: root / arm / "checkpoint_step020000.pt"
        for arm in ARM_NAMES
    }


def _preflight(cfg: Mapping[str, Any], output_dir: Path) -> dict[str, Any]:
    manifest_root = _resolve(cfg["manifest_root"])
    manifest_path = manifest_root / "manifest.json"
    materialization_path = manifest_root / "materialization.json"
    if not manifest_path.is_file() or not materialization_path.is_file():
        raise FileNotFoundError(f"frozen manifest/materialization missing: {manifest_root}")
    manifest = _read_json(manifest_path)
    materialization = _read_json(materialization_path)
    if manifest.get("status") != "FROZEN":
        raise RuntimeError("manifest is not FROZEN")
    if manifest.get("test_sampling", {}).get("opened") is not False:
        raise RuntimeError("manifest does not certify unopened test sampling")
    candidate = cfg["candidate"]
    pilot_root = _resolve(cfg["pilot_root"])
    statistics_path = _resolve(candidate["statistics"], pilot_root)
    codec_path = _resolve(candidate["codec_checkpoint"])
    codec_result = _resolve(candidate["codec_result"])
    checkpoint_paths = _checkpoint_paths(cfg)
    required = {
        "statistics": statistics_path,
        "codec_checkpoint": codec_path,
        "codec_result": codec_result,
        **{f"{arm}_checkpoint": path for arm, path in checkpoint_paths.items()},
        **{f"{arm}_generation_jsonl": Path(str(cfg["legacy_output_root"])) / arm / "generation_metrics.jsonl" for arm in ARM_NAMES},
    }
    missing = {label: str(path) for label, path in required.items() if not path.is_file()}
    if missing:
        raise FileNotFoundError(f"reassessment inputs missing: {missing}")
    result = {
        "schema": f"{REASSESSMENT_SCHEMA}.preflight.v1",
        "code_commit": _git_commit(),
        "baseline_commit": "5c2754fcce44ed77dad77db09db709408fee7634",
        "actual_head": _git_commit(),
        "code_file_sha256": _code_inventory(),
        "working_tree_diff_sha256": _working_diff_sha256(),
        "gpu_inventory_at_preflight": _gpu_inventory(),
        "branch_contract": "perf/dit-factorized-backend-v2",
        "manifest_root": str(manifest_root),
        "manifest_sha256": sha256_file(manifest_path),
        "materialization_sha256": sha256_file(materialization_path),
        "manifest_content_sha256": manifest.get("manifest_content_sha256"),
        "train_clip_count": manifest.get("counts", {}).get("train_clips_materialized"),
        "valid_clip_count": manifest.get("counts", {}).get("valid_clips_materialized"),
        "test_sampling_opened": manifest.get("test_sampling", {}).get("opened"),
        "inputs": {
            label: {"path": str(path), "sha256": sha256_file(path)}
            for label, path in required.items()
        },
        "legacy_output_root": str(cfg["legacy_output_root"]),
        "shared_adapter_legacy_diagnostic": True,
        "test_payload_opened": False,
    }
    _write_json(output_dir / "preflight.json", result)
    return result


def _dataset_ids(dataset: ClipMMapDataset) -> list[str]:
    return [str(row[0]) for row in dataset._index]


def _choose_system_clips(
    dataset: ClipMMapDataset,
    *,
    system_count: int,
    replica: str,
    window: int,
) -> list[dict[str, Any]]:
    by_system: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for index, row in enumerate(dataset._index):
        sample_id = str(row[0])
        parsed = parse_sample_id(sample_id)
        by_system[str(parsed["system"])].append((index, sample_id))
    systems = sorted(by_system)
    if len(systems) < int(system_count):
        raise RuntimeError(f"split contains only {len(systems)} systems; need {system_count}")
    selected: list[dict[str, Any]] = []
    for system in systems[: int(system_count)]:
        candidates = by_system[system]
        preferred = [
            item for item in candidates
            if parse_sample_id(item[1])["replica"] == str(replica)
            and parse_sample_id(item[1])["window"] == int(window)
        ]
        index, sample_id = sorted(preferred or candidates, key=lambda item: item[1])[0]
        parsed = parse_sample_id(sample_id)
        selected.append(
            {
                "dataset_index": int(index),
                "sample_id": sample_id,
                "system": parsed["system"],
                "replica": parsed["replica"],
                "window": int(parsed["window"]),
                "selection_rule": (
                    "sorted_systems_then_R1_window30"
                    if preferred
                    else "sorted_systems_then_lexicographically_first_available_clip"
                ),
            }
        )
    return selected


def _plan(cfg: Mapping[str, Any], output_dir: Path) -> dict[str, Any]:
    data = _load_data(_resolve(cfg["manifest_root"]))
    try:
        evaluation_cfg = cfg["evaluation"]
        valid = _choose_system_clips(
            data.valid,
            system_count=int(evaluation_cfg["validation_system_count"]),
            replica=str(evaluation_cfg["validation_replica"]),
            window=int(evaluation_cfg["validation_window"]),
        )
        train = _choose_system_clips(
            data.train,
            system_count=int(evaluation_cfg["train_system_count"]),
            replica=str(evaluation_cfg["validation_replica"]),
            window=int(evaluation_cfg["validation_window"]),
        )
        rollout_systems = [row["system"] for row in valid]
        result = {
            "schema": f"{REASSESSMENT_SCHEMA}.plan.v1",
            "selection": {
                "validation": valid,
                "train": train,
                "validation_systems": rollout_systems,
                "rollout_smoke_systems": rollout_systems[: int(cfg["rollout"]["smoke_systems"])],
                "rollout_full_systems": rollout_systems[: int(cfg["rollout"]["full_systems"])],
                "quality_independent": True,
            },
            "train_index_sha256": sha256_file(data.manifest_root / "clip_store" / "train" / "index.txt"),
            "valid_index_sha256": sha256_file(data.manifest_root / "clip_store" / "valid" / "index.txt"),
            "data_hash": data.data_hash,
            "window_contract": {
                "clip_len_frames": int(cfg["rollout"]["clip_len"]),
                "window_stride_frames": int(cfg["rollout"]["window_stride_frames"]),
                "first_window": int(cfg["rollout"]["first_window"]),
                "source": "trajectory_clips.ClipPreprocessConfig and T1 materialized sample IDs",
            },
            "test_payload_opened": False,
        }
    finally:
        data.train.close()
        data.valid.close()
    _write_json(output_dir / "plan.json", result)
    return result


def _require_cuda(device: torch.device) -> None:
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("A reassessment numerical stage requires an actual idle CUDA device; CPU fallback is prohibited")
    torch.cuda.set_device(device)


def _cuda_info(device: torch.device) -> dict[str, Any]:
    _require_cuda(device)
    properties = torch.cuda.get_device_properties(device)
    return {
        "requested_device": str(device),
        "logical_index": int(torch.cuda.current_device()),
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "device_name": torch.cuda.get_device_name(device),
        "device_uuid": str(getattr(properties, "uuid", "")),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "tf32": False,
        "amp": "not used for sampling; frozen checkpoint/model semantics preserved",
    }


def _load_models(ctx: Any, cfg: Mapping[str, Any]) -> tuple[dict[str, torch.nn.Module], dict[str, Any]]:
    paths = _checkpoint_paths(cfg)
    models: dict[str, torch.nn.Module] = {}
    metadata: dict[str, Any] = {}
    for arm, path in paths.items():
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, Mapping) or payload.get("schema") != "pvb.dit.state_detail.checkpoint.v2":
            raise RuntimeError(f"{arm} checkpoint schema is not the frozen DiT v2 checkpoint: {path}")
        source = payload.get("source_ab", {})
        if not isinstance(source, Mapping) or source.get("arm") != arm:
            raise RuntimeError(f"{arm} checkpoint source contract does not match arm")
        if int(source.get("actual_optimizer_updates", -1)) != 20000:
            raise RuntimeError(f"{arm} checkpoint is not completed step 20000")
        model_state = payload.get("model_state")
        if not isinstance(model_state, Mapping):
            raise RuntimeError(f"{arm} checkpoint lacks model_state")
        model = _new_model(cfg, ctx.adapter).to(ctx.device)
        model.load_state_dict(model_state, strict=True)
        model.eval()
        models[arm] = model
        metadata[arm] = {
            "checkpoint": str(path),
            "checkpoint_sha256": sha256_file(path),
            "source_contract": dict(source),
            "model_state_hash": module_state_hash(model),
            "shared_adapter_legacy_diagnostic": True,
        }
        del payload
    return models, metadata


def _clip_record_metadata(sample_id: str, *, split: str) -> dict[str, Any]:
    parsed = parse_sample_id(str(sample_id))
    return {
        "split": split,
        "sample_id": str(sample_id),
        "system": parsed["system"],
        "replica": parsed["replica"],
        "window": int(parsed["window"]),
    }


def _prepare_clip(ctx: Any, dataset: ClipMMapDataset, index: int) -> tuple[Any, Any, Any, Any]:
    latent, _batch_cpu, batch = _encode_batch(
        dataset,
        (int(index),),
        codec=ctx.codec,
        adapter=ctx.adapter,
        data_hash=ctx.data_hash,
        device=ctx.device,
    )
    target_batch = ctx.adapter.pack(
        latent,
        codec_hash=ctx.codec.codec_state_hash,
        data_hash=ctx.data_hash,
        origin_from_latent=True,
        loss_mask=batch.loss_mask,
    )
    return latent, batch, target_batch, _clip_record_metadata(dataset._index[int(index)][0], split="")


def _small_center_meta(metadata: Mapping[str, Any]) -> dict[str, Any]:
    result = {
        key: value
        for key, value in metadata.items()
        if key not in {"template_coordinates", "template_latent"}
    }
    result["uses_future_coordinates"] = False
    return result


def _evaluate(cfg: Mapping[str, Any], output_dir: Path, device: torch.device) -> dict[str, Any]:
    _require_cuda(device)
    ctx = _load_context(dict(cfg), output_dir, device)
    models: dict[str, torch.nn.Module] = {}
    try:
        models, checkpoint_meta = _load_models(ctx, cfg)
        plan = _read_json(output_dir / "plan.json")
        generated_rows: list[dict[str, Any]] = []
        baseline_rows: list[dict[str, Any]] = []
        per_frame_rows: list[dict[str, Any]] = []
        diversity_rows: list[dict[str, Any]] = []
        prediction_arrays: dict[str, np.ndarray] = {}
        prediction_index: list[dict[str, Any]] = []
        statistics = ctx.statistics.to(device=device)
        for split in ("valid", "train"):
            dataset = ctx.data.valid if split == "valid" else ctx.data.train
            selection_key = "validation" if split == "valid" else split
            for clip in plan["selection"][selection_key]:
                latent, batch, target_batch, clip_meta = _prepare_clip(
                    ctx, dataset, int(clip["dataset_index"])
                )
                clip_meta["split"] = split
                with torch.no_grad():
                    oracle_coordinates = ctx.codec.model.decode(latent).x_hat.float()
                for history in HISTORIES:
                    observed = _observed_target(ctx, target_batch, batch, history)
                    oracle_metrics = reassessment_trajectory_metrics(
                        oracle_coordinates, batch.x.float(), batch, history
                    )
                    persistence, persistence_meta = observed_coordinate_baseline(
                        batch, history, kind="coordinate_persistence"
                    )
                    if persistence is None:
                        raise RuntimeError("coordinate persistence baseline unexpectedly unavailable")
                    baseline_rows.append(
                        {
                            **clip_meta,
                            "arm": "baseline",
                            "baseline_kind": "oracle_reconstruction",
                            "history_frames": history,
                            "steps": None,
                            "draw": None,
                            "seed": None,
                            "checkpoint_sha256": None,
                            "checkpoint_hash_reason": "codec oracle baseline does not use a DiT checkpoint",
                            "metrics": oracle_metrics,
                            "baseline_metadata": {"codec_state_hash": ctx.codec.codec_state_hash},
                        }
                    )
                    baseline_rows.append(
                        {
                            **clip_meta,
                            "arm": "baseline",
                            "baseline_kind": "repeat_last_observed_frame",
                            "history_frames": history,
                            "steps": None,
                            "draw": None,
                            "seed": None,
                            "checkpoint_sha256": None,
                            "checkpoint_hash_reason": "coordinate-only observed baseline does not use a DiT checkpoint",
                            "metrics": reassessment_trajectory_metrics(
                                persistence, batch.x.float(), batch, history
                            ),
                            "baseline_metadata": persistence_meta,
                        }
                    )
                    normalized = statistics.normalize(observed)
                    noises: dict[int, tuple[Any, dict[str, Any]]] = {}
                    for draw in DRAW_IDS:
                        seed = reassessment_seed(
                            int(cfg["seed"]["validation"]),
                            str(clip_meta["sample_id"]),
                            history,
                            draw,
                        )
                        noises[draw] = make_fixed_noise(normalized, seed=seed)
                    for arm in ARM_NAMES:
                        arm_center = None
                        center_metadata: dict[str, Any] = {
                            "center_kind": "gaussian_zero_center",
                            "uses_future_coordinates": False,
                        }
                        if arm == "conditional":
                            arm_center, raw_center_metadata = build_observed_center(
                                str(cfg["evaluation"]["center_kind"]),
                                codec_model=ctx.codec.model,
                                coordinate_batch=batch,
                                target_batch=observed,
                                adapter=ctx.adapter,
                                statistics=statistics,
                                history_frames=history,
                                codec_hash=ctx.codec.codec_state_hash,
                                data_hash=ctx.data_hash,
                            )
                            center_metadata = _small_center_meta(raw_center_metadata)
                        for steps in FIXED_EULER_STEPS:
                            draw_coordinates: list[Tensor] = []
                            for draw in DRAW_IDS:
                                noise, noise_meta = noises[draw]
                                torch.cuda.synchronize(ctx.device)
                                started = time.perf_counter()
                                generated, generation = generate_fixed_noise_latent(
                                    models[arm],
                                    ctx.adapter,
                                    observed,
                                    statistics,
                                    noise=noise,
                                    steps=steps,
                                    source_center=arm_center,
                                    source_mode=arm,
                                )
                                with torch.no_grad():
                                    decoded = ctx.codec.model.decode(generated).x_hat.float()
                                torch.cuda.synchronize(ctx.device)
                                elapsed = time.perf_counter() - started
                                draw_coordinates.append(decoded.detach())
                                key = f"{split}_{clip_meta['sample_id']}_{arm}_H{history}_S{steps}_D{draw}".replace("/", "_")
                                prediction_arrays[key] = decoded.detach().cpu().numpy().astype(np.float32)
                                prediction_index.append(
                                    {
                                        **clip_meta,
                                        "arm": arm,
                                        "history_frames": history,
                                        "steps": steps,
                                        "draw": draw,
                                        "seed": noises[draw][1]["seed"],
                                        "checkpoint_sha256": checkpoint_meta[arm]["checkpoint_sha256"],
                                        "prediction_key": key,
                                    }
                                )
                                metrics = reassessment_trajectory_metrics(
                                    decoded, batch.x.float(), batch, history
                                )
                                row = {
                                    **clip_meta,
                                    "arm": arm,
                                    "diagnostic_label": "shared_adapter_legacy_diagnostic",
                                    "history_frames": history,
                                    "steps": steps,
                                    "draw": draw,
                                    "seed": noises[draw][1]["seed"],
                                    "checkpoint_sha256": checkpoint_meta[arm]["checkpoint_sha256"],
                                    "checkpoint_step": 20000,
                                    "epsilon_contract": seed_contract(int(cfg["seed"]["validation"])),
                                    "epsilon_noise_sha256": noise_meta["noise_sha256"],
                                    "source_center": center_metadata,
                                    "generation": {
                                        **generation,
                                        "wall_seconds_including_decode": elapsed,
                                    },
                                    "metrics": metrics,
                                    "prediction_key": key,
                                }
                                generated_rows.append(row)
                                for frame in metrics["frame_curve"]:
                                    per_frame_rows.append(
                                        {
                                            **clip_meta,
                                            "arm": arm,
                                            "history_frames": history,
                                            "steps": steps,
                                            "draw": draw,
                                            "seed": noises[draw][1]["seed"],
                                            "checkpoint_sha256": checkpoint_meta[arm]["checkpoint_sha256"],
                                            **frame,
                                        }
                                    )
                            diversity_rows.append(
                                {
                                    **clip_meta,
                                    "arm": arm,
                                    "history_frames": history,
                                    "steps": steps,
                                    "draw": None,
                                    "draw_count": len(draw_coordinates),
                                    "seed": None,
                                    "checkpoint_sha256": checkpoint_meta[arm]["checkpoint_sha256"],
                                    "diversity": future_diversity(
                                        torch.stack(draw_coordinates), batch, history
                                    ),
                                }
                            )
                            del draw_coordinates
                del latent, batch, target_batch
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_jsonl(output_dir / "generated_rows.jsonl", generated_rows)
        _write_jsonl(output_dir / "baseline_rows.jsonl", baseline_rows)
        _write_jsonl(output_dir / "per_frame_metrics.jsonl", per_frame_rows)
        _write_jsonl(output_dir / "diversity_rows.jsonl", diversity_rows)
        _write_jsonl(output_dir / "prediction_index.jsonl", prediction_index)
        np.savez_compressed(output_dir / "predictions.npz", **prediction_arrays)
        grouped: list[dict[str, Any]] = []
        groups: dict[tuple[str, str, int, int], list[dict[str, Any]]] = defaultdict(list)
        for row in generated_rows:
            groups[(str(row["split"]), str(row["arm"]), int(row["history_frames"]), int(row["steps"]))].append(row)
        for (split, arm, history, steps), group in sorted(groups.items()):
            grouped.append(
                {
                    "split": split,
                    "arm": arm,
                    "history_frames": history,
                    "steps": steps,
                    "aggregate": aggregate_generated_rows(group),
                }
            )
        result = {
            "schema": f"{REASSESSMENT_SCHEMA}.evaluation.v1",
            "status": "PASS",
            "device": _cuda_info(device),
            "checkpoint_metadata": checkpoint_meta,
            "row_count": len(generated_rows),
            "baseline_row_count": len(baseline_rows),
            "per_frame_row_count": len(per_frame_rows),
            "diversity_row_count": len(diversity_rows),
            "baseline_groups": _aggregate_baselines(baseline_rows),
            "groups": grouped,
            "files": {
                "generated_rows": str(output_dir / "generated_rows.jsonl"),
                "baseline_rows": str(output_dir / "baseline_rows.jsonl"),
                "per_frame_metrics": str(output_dir / "per_frame_metrics.jsonl"),
                "diversity_rows": str(output_dir / "diversity_rows.jsonl"),
                "predictions": str(output_dir / "predictions.npz"),
            },
            "test_payload_opened": False,
        }
        _write_json(output_dir / "evaluation_summary.json", result)
        return result
    finally:
        ctx.close()


def _cuda_check(cfg: Mapping[str, Any], output_dir: Path, device: torch.device) -> dict[str, Any]:
    """Run a bounded one-clip CUDA contract check before the full evaluation."""

    _require_cuda(device)
    ctx = _load_context(dict(cfg), output_dir, device)
    try:
        models, checkpoint_meta = _load_models(ctx, cfg)
        plan = _read_json(output_dir / "plan.json")
        clip = plan["selection"]["validation"][0]
        latent, batch, target_batch, clip_meta = _prepare_clip(
            ctx, ctx.data.valid, int(clip["dataset_index"])
        )
        observed4 = _observed_target(ctx, target_batch, batch, 4)
        observed8 = _observed_target(ctx, target_batch, batch, 8)
        statistics = ctx.statistics.to(device=device)
        normalized4 = statistics.normalize(observed4)
        normalized8 = statistics.normalize(observed8)
        seed4 = reassessment_seed(int(cfg["seed"]["validation"]), str(clip_meta["sample_id"]), 4, 0)
        seed8 = reassessment_seed(int(cfg["seed"]["validation"]), str(clip_meta["sample_id"]), 8, 0)
        noise4, noise4_meta = make_fixed_noise(normalized4, seed=seed4)
        noise4_repeat, noise4_repeat_meta = make_fixed_noise(normalized4, seed=seed4)
        noise8, noise8_meta = make_fixed_noise(normalized8, seed=seed8)
        if noise4_meta["noise_sha256"] != noise4_repeat_meta["noise_sha256"]:
            raise RuntimeError("fixed-noise generator is not deterministic")
        center4, center4_meta = build_observed_center(
            str(cfg["evaluation"]["center_kind"]),
            codec_model=ctx.codec.model,
            coordinate_batch=batch,
            target_batch=observed4,
            adapter=ctx.adapter,
            statistics=statistics,
            history_frames=4,
            codec_hash=ctx.codec.codec_state_hash,
            data_hash=ctx.data_hash,
        )
        center8, center_meta = build_observed_center(
            str(cfg["evaluation"]["center_kind"]),
            codec_model=ctx.codec.model,
            coordinate_batch=batch,
            target_batch=observed8,
            adapter=ctx.adapter,
            statistics=statistics,
            history_frames=8,
            codec_hash=ctx.codec.codec_state_hash,
            data_hash=ctx.data_hash,
        )
        checks: list[dict[str, Any]] = []
        cases = (
            ("gaussian", 4, 8, observed4, noise4, None),
            ("conditional", 4, 8, observed4, noise4, center4),
            ("gaussian", 8, 32, observed8, noise8, None),
            ("conditional", 8, 16, observed8, noise8, center8),
        )
        for arm, history, steps, observed, noise, center in cases:
            generated, generation = generate_fixed_noise_latent(
                models[arm],
                ctx.adapter,
                observed,
                statistics,
                noise=noise,
                steps=steps,
                source_center=center,
                source_mode=arm,
            )
            with torch.no_grad():
                decoded = ctx.codec.model.decode(generated).x_hat.float()
            torch.cuda.synchronize(device)
            checks.append(
                {
                    "arm": arm,
                    "history_frames": history,
                    "steps": steps,
                    "noise_sha256": tensor_bytes_hash(noise),
                    "observed_clamp_exact": generation["observed_clamp_exact"],
                    "decoded_shape": list(decoded.shape),
                    "decoded_finite": bool(torch.isfinite(decoded).all()),
                    "checkpoint_sha256": checkpoint_meta[arm]["checkpoint_sha256"],
                }
            )
            if not bool(torch.isfinite(decoded).all()):
                raise RuntimeError(f"CUDA check produced non-finite coordinates for {arm} H{history} S{steps}")
        result = {
            "schema": f"{REASSESSMENT_SCHEMA}.cuda_check.v1",
            "status": "PASS",
            "device": _cuda_info(device),
            "clip": clip_meta,
            "seed_contract": seed_contract(int(cfg["seed"]["validation"])),
            "fixed_noise": {
                "H4_draw0": noise4_meta,
                "H4_repeat": noise4_repeat_meta,
                "H8_draw0": noise8_meta,
                "same_H4_hash_on_repeat": noise4_meta["noise_sha256"] == noise4_repeat_meta["noise_sha256"],
            },
            "conditional_center": {
                "H4": _small_center_meta(center4_meta),
                "H8": _small_center_meta(center_meta),
            },
            "checks": checks,
            "test_payload_opened": False,
        }
        _write_json(output_dir / "cuda_check.json", result)
        return result
    finally:
        ctx.close()


def _inverse_clip_gauge(record: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    alignment = record.get("alignment")
    if not isinstance(alignment, Mapping):
        raise ValueError("clip lacks the recorded preprocessing alignment transform")
    rotation = np.asarray(alignment["rotation"], dtype=np.float64)
    translation = np.asarray(alignment["translation"], dtype=np.float64)
    center = np.asarray(alignment["center_angstrom"], dtype=np.float64)
    if rotation.shape != (16, 3, 3) or translation.shape != (16, 3):
        raise ValueError("alignment transform has an unexpected shape")
    def inverse(value: Any) -> np.ndarray:
        centered = np.asarray(value, dtype=np.float64) + center[None, None, :] - translation[:, None, :]
        return np.einsum("tnc,tdc->tnd", centered, rotation).astype(np.float32)
    return inverse(record["x"]), inverse(record["bpos"])


def _block_positions_from_atoms(coordinates: np.ndarray, block_id: Any) -> np.ndarray:
    """Recover repeated block-center coordinates without future observations."""

    coordinates = np.asarray(coordinates, dtype=np.float32)
    ids = np.asarray(block_id, dtype=np.int64).reshape(-1)
    if coordinates.ndim != 3 or coordinates.shape[-1] != 3 or ids.size != coordinates.shape[1]:
        raise ValueError("coordinates/block_id shape mismatch while constructing generated bpos")
    result = np.empty_like(coordinates)
    for block in np.unique(ids):
        members = ids == int(block)
        result[:, members] = coordinates[:, members].mean(axis=1, keepdims=True)
    return result


def _load_rollout_track(
    dataset: ClipMMapDataset,
    *,
    system: str,
    replica: str,
    first_window: int,
    clip_len: int,
    window_stride_frames: int,
) -> dict[str, Any]:
    ids = {str(row[0]): int(index) for index, row in enumerate(dataset._index)}
    sample_ids = [
        f"{system}_{replica}_w{int(first_window + offset):06d}"
        for offset in range(2)
    ]
    if any(sample_id not in ids for sample_id in sample_ids):
        raise FileNotFoundError(f"missing contiguous rollout windows: {sample_ids}")
    records = [dataset[ids[sample_id]] for sample_id in sample_ids]
    dt_values = []
    global_x: list[np.ndarray] = []
    global_bpos: list[np.ndarray] = []
    global_time: list[np.ndarray] = []
    gauge_transforms: list[dict[str, Any]] = []
    topology_keys = ("atype", "btype", "block_id", "component_id", "atom_source_index", "edge_mask", "loss_mask", "align_mask", "bond_index", "atom_identity", "topology_fingerprint", "time_bucket_id", "task", "source", "split", "system_id", "replica", "coordinate_unit", "sampled_delta_time_ps", "native_delta_time_ps", "source_stride", "timestamp_provenance")
    for offset, record in enumerate(records):
        if int(np.asarray(record["x"]).shape[0]) != int(clip_len):
            raise ValueError("rollout requires the frozen 16-frame clip length")
        local_time = np.asarray(record["time_ps"], dtype=np.float64)
        delta = np.asarray(record["delta_time_ps"], dtype=np.float64)
        if delta.size == 0 or not np.allclose(delta, delta[0], rtol=1e-6, atol=1e-5):
            raise ValueError("rollout requires a constant physical dt per clip")
        dt_values.append(float(delta[0]))
        x_raw, bpos_raw = _inverse_clip_gauge(record)
        global_x.append(x_raw)
        global_bpos.append(bpos_raw)
        global_time.append(local_time + float(first_window + offset) * float(window_stride_frames) * float(delta[0]))
        alignment = dict(record["alignment"])
        gauge_transforms.append({"sample_id": record["sample_id"], "alignment": alignment})
        if offset:
            reference = records[0]
            for key in ("atype", "btype", "block_id", "component_id", "atom_source_index", "bond_index", "atom_identity"):
                if not np.array_equal(np.asarray(record[key]), np.asarray(reference[key])):
                    raise ValueError(f"contiguous rollout windows disagree on topology field {key}")
        if offset and not np.isclose(global_time[-1][0], global_time[-2][-1] + dt_values[-1], rtol=1e-6, atol=1e-4):
            raise ValueError("rollout windows are not contiguous under the recorded window stride and dt")
    if not np.allclose(dt_values, dt_values[0], rtol=1e-6, atol=1e-5):
        raise ValueError("rollout windows have inconsistent dt")
    base = records[0]
    combined = dict(base)
    combined["sample_id"] = f"{system}_{replica}_rollout_w{first_window:06d}_32f"
    combined["x"] = np.concatenate(global_x, axis=0)
    combined["bpos"] = np.concatenate(global_bpos, axis=0)
    combined["time_ps"] = np.concatenate(global_time, axis=0).astype(np.float32)
    combined["delta_time_ps"] = np.diff(combined["time_ps"]).astype(np.float32)
    combined["alignment"] = {"coordinate_gauge": "inverse_preprocessing_raw_source_gauge", "source_window_transforms": gauge_transforms}
    combined["source"] = "reassessment_rollout_combined_contiguous_windows"
    return {
        "system": system,
        "replica": replica,
        "first_window": int(first_window),
        "sample_ids": sample_ids,
        "dt_ps": float(dt_values[0]),
        "global_x": combined["x"],
        "global_bpos": combined["bpos"],
        "global_time_ps": combined["time_ps"],
        "combined_record": combined,
        "gauge_transforms": gauge_transforms,
        "topology_keys": topology_keys,
        "gauge_contract": {
            "source": "inverse of each clip's recorded align_and_center transform",
            "formula": "x_raw = (x_stored + center - translation) @ rotation.T",
            "single_coordinate_gauge_across_32_frames": True,
            "model_segment_origin_is_recorded": True,
        },
    }


def _rollout_record(
    track: Mapping[str, Any],
    *,
    prefix: np.ndarray,
    prefix_bpos: np.ndarray,
    segment: int,
    prefix_kind: str,
) -> dict[str, Any]:
    base = dict(track["combined_record"])
    dt = float(track["dt_ps"])
    future_fill = np.repeat(prefix[-1:,:,:], 8, axis=0)
    x = np.concatenate((prefix, future_fill), axis=0).astype(np.float32)
    bpos = np.concatenate((prefix_bpos, np.repeat(prefix_bpos[-1:,:,:], 8, axis=0)), axis=0).astype(np.float32)
    base.update(
        {
            "sample_id": f"{track['system']}_{track['replica']}_rollout_{prefix_kind}_segment{segment}",
            "x": x,
            "bpos": bpos,
            "time_ps": (np.arange(16, dtype=np.float32) * dt),
            "delta_time_ps": np.full(15, dt, dtype=np.float32),
            "alignment": {"coordinate_gauge": "global_rollout_gauge", "prefix_kind": prefix_kind, "future_fill": "repeat_last_prefix"},
        }
    )
    return base


def _rollout_metric_batch(track: Mapping[str, Any], start: int) -> Any:
    record = dict(track["combined_record"])
    record["sample_id"] = f"{track['system']}_{track['replica']}_rollout_metric_segment{start}"
    record["x"] = np.asarray(track["global_x"][start:start + 16], dtype=np.float32)
    record["bpos"] = np.asarray(track["global_bpos"][start:start + 16], dtype=np.float32)
    record["time_ps"] = np.asarray(track["global_time_ps"][start:start + 16], dtype=np.float32)
    record["delta_time_ps"] = np.diff(record["time_ps"]).astype(np.float32)
    return collate_clip_records([record])


def _full_rollout_metric_batch(track: Mapping[str, Any]) -> Any:
    return collate_clip_records([dict(track["combined_record"])])


def _run_rollout(
    cfg: Mapping[str, Any],
    output_dir: Path,
    device: torch.device,
    *,
    system_limit: int,
    output_name: str,
) -> dict[str, Any]:
    _require_cuda(device)
    ctx = _load_context(dict(cfg), output_dir, device)
    try:
        models, checkpoint_meta = _load_models(ctx, cfg)
        plan = _read_json(output_dir / "plan.json")
        systems = list(plan["selection"]["validation_systems"])[: int(system_limit)]
        tracks: list[dict[str, Any]] = []
        unavailable: list[dict[str, Any]] = []
        for system in systems:
            try:
                tracks.append(
                    _load_rollout_track(
                        ctx.data.valid,
                        system=str(system),
                        replica=str(cfg["evaluation"]["validation_replica"]),
                        first_window=int(cfg["rollout"]["first_window"]),
                        clip_len=int(cfg["rollout"]["clip_len"]),
                        window_stride_frames=int(cfg["rollout"]["window_stride_frames"]),
                    )
                )
            except (FileNotFoundError, ValueError, KeyError) as exc:
                unavailable.append({"system": str(system), "status": "ROLLOUT_DATA_UNAVAILABLE", "reason": str(exc)})
        if unavailable:
            _write_json(output_dir / output_name / "rollout_unavailable.json", {"status": "ROLLOUT_DATA_UNAVAILABLE", "rows": unavailable, "test_payload_opened": False})
        if not tracks:
            result = {
                "schema": f"{REASSESSMENT_SCHEMA}.rollout.v1",
                "status": "ROLLOUT_DATA_UNAVAILABLE",
                "system_count": 0,
                "unavailable": unavailable,
                "test_payload_opened": False,
            }
            _write_json(output_dir / output_name / "rollout_summary.json", result)
            return result
        segment_rows: list[dict[str, Any]] = []
        frame_rows: list[dict[str, Any]] = []
        full_rows: list[dict[str, Any]] = []
        arrays: dict[str, np.ndarray] = {}
        arrays_index: list[dict[str, Any]] = []
        rollout_cfg = cfg["rollout"]
        statistics = ctx.statistics.to(device=device)
        for track in tracks:
            actual = torch.as_tensor(track["global_x"], device=device, dtype=torch.float32)
            actual_bpos = np.asarray(track["global_bpos"], dtype=np.float32)
            for prefix_kind in ("true_prefix", "generated_prefix"):
                for arm in ARM_NAMES:
                    for draw in (int(value) for value in rollout_cfg["draws"]):
                        generated_segments: list[Tensor] = []
                        generated_prefix = actual[:8]
                        gauge_rows: list[dict[str, Any]] = []
                        for segment in range(int(rollout_cfg["segments"])):
                            if prefix_kind == "true_prefix":
                                prefix = actual[segment * 8:(segment + 1) * 8]
                                prefix_bpos = torch.as_tensor(actual_bpos[segment * 8:(segment + 1) * 8], device=device)
                            else:
                                prefix = generated_prefix
                                prefix_bpos = torch.as_tensor(
                                    _block_positions_from_atoms(
                                        prefix.detach().cpu().numpy(),
                                        track["combined_record"]["block_id"],
                                    ),
                                    device=device,
                                )
                            record = _rollout_record(
                                track,
                                prefix=prefix.detach().cpu().numpy(),
                                prefix_bpos=prefix_bpos.detach().cpu().numpy(),
                                segment=segment,
                                prefix_kind=prefix_kind,
                            )
                            batch_cpu = collate_clip_records([record])
                            ctx.codec.model.prepare_batch(batch_cpu)
                            batch = batch_cpu.to(device, non_blocking=True)
                            with torch.no_grad():
                                latent = ctx.codec.model.encode(batch)
                            target_batch = ctx.adapter.pack(
                                latent,
                                codec_hash=ctx.codec.codec_state_hash,
                                data_hash=ctx.data_hash,
                                origin_from_latent=True,
                                loss_mask=batch.loss_mask,
                            )
                            observed = _observed_target(ctx, target_batch, batch, 8)
                            center = None
                            center_metadata = {"center_kind": "gaussian_zero_center", "uses_future_coordinates": False}
                            if arm == "conditional":
                                center, raw_center_metadata = build_observed_center(
                                    str(cfg["evaluation"]["center_kind"]),
                                    codec_model=ctx.codec.model,
                                    coordinate_batch=batch,
                                    target_batch=observed,
                                    adapter=ctx.adapter,
                                    statistics=statistics,
                                    history_frames=8,
                                    codec_hash=ctx.codec.codec_state_hash,
                                    data_hash=ctx.data_hash,
                                )
                                center_metadata = _small_center_meta(raw_center_metadata)
                            normalized = statistics.normalize(observed)
                            seed_id = f"{track['system']}_{track['replica']}_w{track['first_window']:06d}_rollout_segment{segment}"
                            seed = reassessment_seed(int(cfg["seed"]["rollout"]), seed_id, 8, draw)
                            noise, noise_meta = make_fixed_noise(normalized, seed=seed)
                            generated, generation = generate_fixed_noise_latent(
                                models[arm],
                                ctx.adapter,
                                observed,
                                statistics,
                                noise=noise,
                                steps=int(rollout_cfg["steps"]),
                                source_center=center,
                                source_mode=arm,
                            )
                            with torch.no_grad():
                                decoded = ctx.codec.model.decode(generated).x_hat.float()
                            torch.cuda.synchronize(device)
                            predicted_future = decoded[8:].detach()
                            generated_segments.append(predicted_future)
                            if prefix_kind == "generated_prefix":
                                generated_prefix = predicted_future
                            metric_batch = _rollout_metric_batch(track, segment * 8)
                            metric_batch = metric_batch.to(device)
                            pred_window = torch.cat((prefix, predicted_future), dim=0).unsqueeze(0)
                            # The metric helper consumes time-major coordinates, so use the
                            # single-sample batch's metadata and remove the temporary batch axis.
                            segment_metrics = reassessment_trajectory_metrics(
                                pred_window.squeeze(0),
                                metric_batch.x.float(),
                                metric_batch,
                                8,
                            )
                            global_start = segment * 8
                            row = {
                                "schema": f"{REASSESSMENT_SCHEMA}.rollout_segment.v1",
                                "split": "valid",
                                "system": track["system"],
                                "replica": track["replica"],
                                "window": track["first_window"],
                                "sample_id": track["sample_ids"][0],
                                "prefix_kind": prefix_kind,
                                "arm": arm,
                                "history_frames": 8,
                                "steps": int(rollout_cfg["steps"]),
                                "draw": draw,
                                "seed": seed,
                                "checkpoint_sha256": checkpoint_meta[arm]["checkpoint_sha256"],
                                "segment": segment,
                                "global_frame_interval": [global_start, global_start + 16],
                                "target_future_global_interval": [global_start + 8, global_start + 16],
                                "epsilon_noise_sha256": noise_meta["noise_sha256"],
                                "source_center": center_metadata,
                                "generation": generation,
                                "gauge": {
                                    **track["gauge_contract"],
                                    "prefix_source": "real_observed_prefix" if prefix_kind == "true_prefix" else "previous_generated_coordinates_only",
                                    "model_sample_origin": [float(value) for value in target_batch.sample_origin[0].detach().cpu()],
                                    "global_time_ps_interval": [float(track["global_time_ps"][global_start]), float(track["global_time_ps"][global_start + 15])],
                                    "local_model_time_ps_interval": [0.0, float(15 * track["dt_ps"])],
                                },
                                "metrics": segment_metrics,
                            }
                            segment_rows.append(row)
                            for frame in segment_metrics["frame_curve"]:
                                frame_rows.append({
                                    **{key: row[key] for key in ("split", "system", "replica", "window", "sample_id", "prefix_kind", "arm", "history_frames", "steps", "draw", "seed", "checkpoint_sha256")},
                                    "frame": global_start + int(frame["frame"]),
                                    "global_time_ps": float(track["global_time_ps"][global_start + int(frame["frame"])]),
                                    "local_frame": int(frame["frame"]),
                                    "aligned_rmsd": frame["aligned_rmsd"],
                                    "bond_rmse": frame["bond_rmse"],
                                    "contact_f1": frame.get("contact_f1"),
                                    "future": bool(frame["future"]),
                                })
                            del latent, batch, target_batch, observed, normalized, noise, generated, decoded, metric_batch
                        full_prediction = torch.cat((actual[:8], *generated_segments), dim=0)
                        full_batch = _full_rollout_metric_batch(track).to(device)
                        full_metrics = reassessment_trajectory_metrics(
                            full_prediction,
                            actual,
                            full_batch,
                            8,
                        )
                        key = f"{track['system']}_{prefix_kind}_{arm}_D{draw}"
                        arrays[key] = full_prediction.detach().cpu().numpy().astype(np.float32)
                        arrays_index.append({"key": key, "system": track["system"], "prefix_kind": prefix_kind, "arm": arm, "draw": draw})
                        full_rows.append(
                            {
                                "schema": f"{REASSESSMENT_SCHEMA}.rollout_path.v1",
                                "split": "valid",
                                "system": track["system"],
                                "replica": track["replica"],
                                "window": track["first_window"],
                                "sample_id": track["sample_ids"][0],
                                "prefix_kind": prefix_kind,
                                "arm": arm,
                                "history_frames": 8,
                                "steps": int(rollout_cfg["steps"]),
                                "draw": draw,
                                "seed": None,
                                "checkpoint_sha256": checkpoint_meta[arm]["checkpoint_sha256"],
                                "frame_count": 32,
                                "endpoint_frame_24": 24,
                                "prediction_key": key,
                                "metrics": full_metrics,
                            }
                        )
        output_path = output_dir / output_name
        output_path.mkdir(parents=True, exist_ok=True)
        _write_jsonl(output_path / "segment_rows.jsonl", segment_rows)
        _write_jsonl(output_path / "per_frame_rows.jsonl", frame_rows)
        _write_jsonl(output_path / "path_rows.jsonl", full_rows)
        _write_jsonl(output_path / "prediction_index.jsonl", arrays_index)
        np.savez_compressed(output_path / "predictions.npz", **arrays)
        summary = {
            "schema": f"{REASSESSMENT_SCHEMA}.rollout.v1",
            "status": "PASS" if not unavailable else "PASS_WITH_ROLLOUT_DATA_UNAVAILABLE",
            "device": _cuda_info(device),
            "system_count": len(tracks),
            "systems": [track["system"] for track in tracks],
            "unavailable": unavailable,
            "segment_row_count": len(segment_rows),
            "per_frame_row_count": len(frame_rows),
            "path_row_count": len(full_rows),
            "window_contract": {
                "first_window": int(cfg["rollout"]["first_window"]),
                "clip_len": int(cfg["rollout"]["clip_len"]),
                "window_stride_frames": int(cfg["rollout"]["window_stride_frames"]),
                "dt_ps_preserved": True,
                "time_reset_per_model_segment": True,
                "global_time_saved": True,
            },
            "test_payload_opened": False,
        }
        _write_json(output_path / "rollout_summary.json", summary)
        _write_json(output_path / "gauge_transforms.json", {track["system"]: track["gauge_transforms"] for track in tracks})
        return summary
    finally:
        ctx.close()


def _aggregate_baselines(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["split"]), int(row["history_frames"]), str(row["baseline_kind"]))].append(row)
    result = []
    for (split, history, kind), group in sorted(groups.items()):
        future = [row["metrics"].get("future", {}) for row in group]
        result.append(
            {
                "split": split,
                "history_frames": history,
                "baseline_kind": kind,
                "clip_count": len(group),
                "system_count": len({row["system"] for row in group}),
                "system_equal": _mean_mapping(future),
            }
        )
    return result


def _plot_evaluation(output_dir: Path) -> str | None:
    rows = _read_jsonl(output_dir / "per_frame_metrics.jsonl")
    if not rows:
        return None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        _write_json(output_dir / "plot_status.json", {"status": "UNAVAILABLE", "reason": str(exc)})
        return None
    grouped: dict[tuple[str, int, int], dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        value = row.get("aligned_rmsd")
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            grouped[(str(row["arm"]), int(row["history_frames"]), int(row["steps"]))][int(row["frame"])].append(float(value))
    figure, axis = plt.subplots(figsize=(9, 5))
    for (arm, history, steps), frame_values in sorted(grouped.items()):
        x = sorted(frame_values)
        y = [sum(frame_values[frame]) / len(frame_values[frame]) for frame in x]
        if steps == 16:
            axis.plot(x, y, label=f"{arm}, H{history}, {steps} steps")
    axis.set_xlabel("frame")
    axis.set_ylabel("per-frame aligned RMSD (Å)")
    axis.set_title("R4 source reassessment: generated geometry curve")
    axis.legend(fontsize=8)
    axis.grid(alpha=0.25)
    path = output_dir / "geometry_curves.png"
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return str(path)


def _summarize(cfg: Mapping[str, Any], output_dir: Path) -> dict[str, Any]:
    preflight = _read_json(output_dir / "preflight.json") if (output_dir / "preflight.json").is_file() else {}
    plan = _read_json(output_dir / "plan.json") if (output_dir / "plan.json").is_file() else {}
    legacy = _read_json(output_dir / "legacy_reaggregation.json") if (output_dir / "legacy_reaggregation.json").is_file() else {}
    cuda_check = _read_json(output_dir / "cuda_check.json") if (output_dir / "cuda_check.json").is_file() else None
    evaluation = _read_json(output_dir / "evaluation_summary.json") if (output_dir / "evaluation_summary.json").is_file() else None
    rollout_full = _read_json(output_dir / "rollout" / "rollout_summary.json") if (output_dir / "rollout" / "rollout_summary.json").is_file() else None
    rollout_smoke = _read_json(output_dir / "rollout_smoke" / "rollout_summary.json") if (output_dir / "rollout_smoke" / "rollout_summary.json").is_file() else None
    if evaluation is None:
        status = "A_SOURCE_REASSESSMENT_V2_PARTIAL_NO_CUDA_EVALUATION"
    elif rollout_full is None:
        status = "A_SOURCE_REASSESSMENT_V2_PARTIAL_ROLLOUT_NOT_RUN"
    elif rollout_full.get("status") == "ROLLOUT_DATA_UNAVAILABLE":
        status = "A_SOURCE_REASSESSMENT_V2_COMPLETE_ROLLOUT_DATA_UNAVAILABLE"
    else:
        status = "A_SOURCE_REASSESSMENT_V2_COMPLETE"
    plot = _plot_evaluation(output_dir) if evaluation is not None else None
    lines = [
        "# R4 source checkpoint reassessment v2",
        "",
        f"Status: `{status}`",
        "",
        "This is a descriptive `shared_adapter_legacy_diagnostic`; the Gaussian/conditional checkpoint difference is not an independent source ablation.",
        "The frozen R4 codec/statistics, DiT/RF semantics, train/validation manifests, and test-sealed boundary were preserved.",
        "",
        "## Inputs and selection",
        "",
        f"- Code HEAD: `{preflight.get('actual_head', _git_commit())}`; required baseline: `5c2754fcce44ed77dad77db09db709408fee7634`.",
        f"- Legacy JSONL row check by arm: `{legacy.get('row_count_check')}`; per-arm counts `{legacy.get('actual_row_counts_per_arm', {})}` vs expected `{legacy.get('expected_row_counts_per_arm', {'main_batch': 52, 'subset_clip': 128})}`. Main batch rows were never split into fake systems.",
        f"- Validation selection: `{len(plan.get('selection', {}).get('validation', []))}` systems, fixed R1/window30; train selection: `{len(plan.get('selection', {}).get('train', []))}` sorted systems, one preselected clip each.",
        "- Test payload opened: `false`.",
        "",
        "## Existing JSONL reaggregation",
        "",
        "Legacy main rows are reported as batch-row means with `system_equal_not_available`. Legacy subset rows are averaged draw → clip → system, separately for H and Euler steps.",
        "",
    ]
    for arm, value in sorted(legacy.get("arms", {}).items()):
        lines.append(f"- `{arm}`: {value.get('row_counts', {})} rows; checkpoint SHA256 `{value.get('checkpoint_sha256', '')}`.")
    lines.extend(["", "## New single-segment results", "", f"CUDA contract check: `{None if cuda_check is None else cuda_check.get('status')}`.", ""])
    if evaluation is None:
        lines.append("No CUDA evaluation artifact exists. Numerical sampling was not run; no CPU fallback was used. See `preflight.json.gpu_inventory_at_preflight` for the confirmed occupied UUID/PID list.")
    else:
        lines.extend([
            f"Generated rows: `{evaluation.get('row_count')}`; baseline rows: `{evaluation.get('baseline_row_count')}`; diversity rows: `{evaluation.get('diversity_row_count')}`.",
            "",
            "| split | arm | H | steps | systems | future aligned RMSD | bond RMSE | contact F1 |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ])
        for group in evaluation.get("groups", []):
            metrics = group.get("aggregate", {}).get("system_equal", {})
            lines.append(
                f"| {group.get('split')} | {group.get('arm')} | {group.get('history_frames')} | {group.get('steps')} | {group.get('aggregate', {}).get('system_count')} | {metrics.get('aligned_rmsd', 'n/a')} | {metrics.get('bond_rmse', 'n/a')} | {metrics.get('contact_f1', 'n/a')} |"
            )
        lines.extend(["", "Baseline system-equal summaries (one oracle and one repeat-last row per clip/H):"])
        for baseline in evaluation.get("baseline_groups", []):
            metrics = baseline.get("system_equal", {})
            lines.append(
                f"- `{baseline.get('split')}` H{baseline.get('history_frames')} `{baseline.get('baseline_kind')}`: RMSD `{metrics.get('aligned_rmsd', 'n/a')}`, bond `{metrics.get('bond_rmse', 'n/a')}`, contact F1 `{metrics.get('contact_f1', 'n/a')}`."
            )
        lines.extend([
            "",
            "Oracle reconstruction and repeat-last-observed-frame baselines were computed once per clip/H. Generated coordinate arrays are reused by all metrics through `predictions.npz`; per-frame bond/RMSD curves are in `per_frame_metrics.jsonl`.",
            "",
            "Interpretation checks required by this run:",
            "- Training-system versus validation-system differences must be read from the separate split rows, not inferred from old batch rows.",
            "- H4 first predicted frames are available in the per-frame curve; compare them with later future frames to distinguish first-block failure from rollout growth.",
            "- RMSF ratio, displacement summaries, diversity, and velocity-lag1 ACF are reported separately; near-constant Pearson/ACF signals are null with reasons.",
            "- 8/16/32 steps remain separate groups and share the same epsilon seed for each sample/H/draw across both arms.",
        ])
    lines.extend(["", "## Short rollout", ""])
    if rollout_smoke is None:
        lines.append("The required two-system H8 smoke was not run.")
    else:
        lines.append(f"Smoke status: `{rollout_smoke.get('status')}`, systems `{rollout_smoke.get('system_count')}`.")
    if rollout_full is None:
        lines.append("The eight-system rollout was not run.")
    elif rollout_full.get("status") == "ROLLOUT_DATA_UNAVAILABLE":
        lines.append("`ROLLOUT_DATA_UNAVAILABLE`: no continuous w30/w31 pair was available; no interpolation or discontinuous concatenation was used.")
    else:
        lines.append(f"Full rollout status: `{rollout_full.get('status')}`, systems `{rollout_full.get('system_count')}`, segment rows `{rollout_full.get('segment_row_count')}`.")
        lines.append("Rollout records preserve a single inverse-preprocessing gauge, local model time reset with the real dt, and global physical time. True-prefix and generated-prefix rows are separate.")
        lines.append("Compare segment 2/3 rows in `rollout/segment_rows.jsonl` to test whether generated prefixes amplify geometry or motion errors.")
    lines.extend([
        "",
        "## Reproduction",
        "",
        "```bash",
        "enter-container  # then: conda activate torch-ito",
        "python scripts/run_dit_source_reassessment.py --config config/dit_source_reassessment_v2.yaml --stage preflight",
        "python scripts/run_dit_source_reassessment.py --config config/dit_source_reassessment_v2.yaml --stage plan",
        "python scripts/run_dit_source_reassessment.py --config config/dit_source_reassessment_v2.yaml --stage reaggregate",
        "python scripts/run_dit_source_reassessment.py --config config/dit_source_reassessment_v2.yaml --stage cuda_check --device cuda:IDLE",
        "python scripts/run_dit_source_reassessment.py --config config/dit_source_reassessment_v2.yaml --stage evaluate --device cuda:IDLE",
        "python scripts/run_dit_source_reassessment.py --config config/dit_source_reassessment_v2.yaml --stage rollout_smoke --device cuda:IDLE",
        "python scripts/run_dit_source_reassessment.py --config config/dit_source_reassessment_v2.yaml --stage rollout --device cuda:IDLE",
        "python scripts/run_dit_source_reassessment.py --config config/dit_source_reassessment_v2.yaml --stage summarize",
        "```",
        "",
        f"Plot: `{plot or 'not generated'}`.",
    ])
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    handoff = [
        "# Session A handoff",
        "",
        f"- Status: `{status}`.",
        f"- Run root: `{output_dir}`.",
        f"- Code HEAD recorded by preflight: `{preflight.get('actual_head', _git_commit())}`.",
        f"- Legacy reaggregation: `{legacy.get('row_count_check')}` by arm; main batch rows remain unsplittable.",
        f"- CUDA single-segment evaluation: `{evaluation is not None}`; rollout smoke: `{rollout_smoke is not None}`; rollout full: `{rollout_full is not None}`.",
        "- Test payload opened: `false`; no training was run.",
        "- Continue only on one confirmed idle CUDA GPU; if unavailable, retain this run as partial and do not use a CPU numerical fallback.",
    ]
    (output_dir / "HANDOFF.md").write_text("\n".join(handoff) + "\n", encoding="utf-8")
    result = {
        "schema": f"{REASSESSMENT_SCHEMA}.summary.v1",
        "status": status,
        "output_root": str(output_dir),
        "legacy_reaggregation": str(output_dir / "legacy_reaggregation.json"),
        "cuda_check": None if cuda_check is None else str(output_dir / "cuda_check.json"),
        "evaluation": None if evaluation is None else str(output_dir / "evaluation_summary.json"),
        "rollout_smoke": None if rollout_smoke is None else str(output_dir / "rollout_smoke" / "rollout_summary.json"),
        "rollout": None if rollout_full is None else str(output_dir / "rollout" / "rollout_summary.json"),
        "report": str(output_dir / "report.md"),
        "handoff": str(output_dir / "HANDOFF.md"),
        "plot": plot,
        "test_payload_opened": False,
    }
    _write_json(output_dir / "summary.json", result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config/dit_source_reassessment_v2.yaml")
    parser.add_argument("--stage", choices=("preflight", "plan", "reaggregate", "cuda_check", "evaluate", "rollout_smoke", "rollout", "summarize", "all"), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--output-root", type=Path, default=None)
    return parser


def main() -> None:
    args = _parser().parse_args()
    cfg, output_dir = _resolved_args(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.stage in {"preflight", "plan", "reaggregate", "summarize"}:
        if args.stage == "preflight":
            result = _preflight(cfg, output_dir)
        elif args.stage == "plan":
            result = _plan(cfg, output_dir)
        elif args.stage == "reaggregate":
            checkpoint_paths = _checkpoint_paths(cfg)
            result = reaggregate_legacy_jsonl(cfg["legacy_output_root"], checkpoint_paths=checkpoint_paths)
            _write_json(output_dir / "legacy_reaggregation.json", result)
            rows = result.pop("normalized_rows", [])
            _write_jsonl(output_dir / "legacy_rows.jsonl", rows)
            _write_json(output_dir / "legacy_reaggregation.json", result)
        else:
            result = _summarize(cfg, output_dir)
        print(json.dumps(_safe(result), indent=2, sort_keys=True))
        return
    if args.stage == "all":
        _preflight(cfg, output_dir)
        _plan(cfg, output_dir)
        checkpoint_paths = _checkpoint_paths(cfg)
        legacy = reaggregate_legacy_jsonl(cfg["legacy_output_root"], checkpoint_paths=checkpoint_paths)
        rows = legacy.pop("normalized_rows", [])
        _write_jsonl(output_dir / "legacy_rows.jsonl", rows)
        _write_json(output_dir / "legacy_reaggregation.json", legacy)
        device = torch.device(args.device)
        _cuda_check(cfg, output_dir, device)
        _evaluate(cfg, output_dir, device)
        _run_rollout(cfg, output_dir, device, system_limit=int(cfg["rollout"]["smoke_systems"]), output_name="rollout_smoke")
        _run_rollout(cfg, output_dir, device, system_limit=int(cfg["rollout"]["full_systems"]), output_name="rollout")
        result = _summarize(cfg, output_dir)
        print(json.dumps(_safe(result), indent=2, sort_keys=True))
        return
    device = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.stage == "evaluate":
        result = _evaluate(cfg, output_dir, device)
    elif args.stage == "cuda_check":
        result = _cuda_check(cfg, output_dir, device)
    elif args.stage == "rollout_smoke":
        result = _run_rollout(cfg, output_dir, device, system_limit=int(cfg["rollout"]["smoke_systems"]), output_name="rollout_smoke")
    elif args.stage == "rollout":
        result = _run_rollout(cfg, output_dir, device, system_limit=int(cfg["rollout"]["full_systems"]), output_name="rollout")
    else:
        raise ValueError(f"unsupported stage: {args.stage}")
    print(json.dumps(_safe(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
