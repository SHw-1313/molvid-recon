#!/usr/bin/env python
"""Run the clean baseline and Round 1 of R4 DiT architecture sequential v1."""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
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

from data.clip_batching import TaskAwareClipBatchSampler
from data.clip_dataset import ClipMMapDataset
from evaluation.dit_architecture_sequential_v1 import (
    PHASE,
    assert_independent_trainers,
    generation_seed,
    load_sequential_checkpoint,
    sha256_file,
)
from evaluation.dit_diagnostics import parse_sample_id
from evaluation.dit_reassessment import (
    aggregate_generated_rows,
    future_diversity,
    generate_fixed_noise_latent,
    make_fixed_noise,
    reassessment_trajectory_metrics,
)
from module.latent_flow_source import (
    build_observed_center,
    repeat_last_coordinate_template,
)
from module.molecular_dit import MolecularDiT
from module.state_detail_latent_adapter import (
    LatentStatistics,
    StateDetailLatentAdapter,
)
from scripts.run_state_detail_dit_pilot import (
    FrozenCodec,
    _atomic_json,
    _atomic_torch_save,
    _encode_batch,
    _load_approved_codec,
    _load_data,
    _make_sampler,
    _make_validation_plan,
    _observed_batch,
)
from trainer.dit_trainer import DiTTrainConfig, DiTTrainer, module_state_hash


SCHEMA = "pvb.dit.architecture_sequential.v1"
ROUND1_ARMS = ("control", "candidate")
NORM_SOURCE = {"control": "post_adaln", "candidate": "pre_adaln"}


def _safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().item()
        return value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_json(path, _safe(value))


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_safe(value), sort_keys=True) + "\n")


def _truncate_jsonl(path: Path, *, step_key: str, maximum_step: int) -> int:
    if not path.is_file():
        return 0
    lines = path.read_text(encoding="utf-8").splitlines()
    rows = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if any(value.strip() for value in lines[line_number:]):
                raise RuntimeError(f"malformed non-terminal JSONL row in {path}")
            break
        if step_key not in row:
            raise RuntimeError(f"{path} row lacks resume key {step_key}")
        if int(row[step_key]) <= int(maximum_step):
            rows.append(row)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    os.replace(temporary, path)
    return len(rows)


def _load_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("architecture sequential config must be a mapping")
    if value.get("schema") != "pvb.dit.architecture_sequential.config.v1":
        raise ValueError("unsupported architecture sequential config schema")
    return json.loads(json.dumps(value))


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(_safe(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _clip_index_semantic_hash(path: Path) -> str:
    """Hash clip identity/shape metadata while ignoring materialization offsets."""

    digest = hashlib.sha256()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 6:
                raise RuntimeError(f"malformed clip index row {line_number}: {path}")
            digest.update("\t".join((fields[0], *fields[3:])).encode("utf-8"))
            digest.update(b"\n")
    return digest.hexdigest()


def _code_provenance() -> dict[str, Any]:
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
    ).strip()
    files = (
        "module/molecular_dit.py",
        "module/dit_backend_v2.py",
        "trainer/dit_trainer.py",
        "scripts/run_dit_source_ab.py",
        "scripts/run_dit_architecture_sequential_v1.py",
        "evaluation/dit_architecture_sequential_v1.py",
        "config/dit_architecture_sequential_v1.yaml",
        "config/dit_architecture_sequential_v1.neibu.yaml",
        "tests/test_dit_architecture_sequential_cuda.py",
    )
    hashes = {}
    for relative in files:
        path = PROJECT_ROOT / relative
        hashes[relative] = sha256_file(path) if path.is_file() else None
    return {
        "head": head,
        "files": hashes,
        "source_hash": _canonical_hash(hashes),
    }


def _require_cuda(device: torch.device) -> None:
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("architecture sequential numerical stages require actual CUDA")
    torch.cuda.set_device(device)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _sync(device: torch.device) -> None:
    torch.cuda.synchronize(device)


def _cuda_info(device: torch.device) -> dict[str, Any]:
    _require_cuda(device)
    properties = torch.cuda.get_device_properties(device)
    return {
        "requested_device": str(device),
        "logical_index": int(torch.cuda.current_device()),
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "name": torch.cuda.get_device_name(device),
        "uuid": str(getattr(properties, "uuid", "")),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "tf32": False,
        "amp": "bfloat16",
    }


@dataclass
class SequentialContext:
    cfg: dict[str, Any]
    output_dir: Path
    data: Any
    codec: FrozenCodec
    statistics: LatentStatistics
    statistics_device: LatentStatistics
    device: torch.device

    @property
    def data_hash(self) -> str:
        return str(self.data.data_hash)

    def close(self) -> None:
        self.data.train.close()
        self.data.valid.close()


def _new_adapter(cfg: Mapping[str, Any], device: torch.device) -> StateDetailLatentAdapter:
    model = cfg["model"]
    return StateDetailLatentAdapter(
        codec_width=int(model["codec_width"]),
        scalar_width=int(model["scalar_width"]),
        vector_width=int(model["vector_width"]),
        ratio=int(cfg["candidate"]["ratio"]),
    ).to(device)


def _new_model(
    cfg: Mapping[str, Any],
    adapter: StateDetailLatentAdapter,
    *,
    ffn_norm_source: str,
) -> MolecularDiT:
    model = cfg["model"]
    return MolecularDiT(
        adapter=adapter,
        scalar_width=int(model["scalar_width"]),
        vector_width=int(model["vector_width"]),
        depth=int(model["depth"]),
        heads=int(model["heads"]),
        ffn_multiplier=int(model["ffn_multiplier"]),
        dropout=float(model["dropout"]),
        execution_backend=str(model["execution_backend"]),
        ffn_norm_source=str(ffn_norm_source),
    )


def _load_context(
    cfg: dict[str, Any], output_dir: Path, device: torch.device
) -> SequentialContext:
    _require_cuda(device)
    candidate = cfg["candidate"]
    data = _load_data(_resolve(cfg["manifest_root"]))
    codec = _load_approved_codec(
        candidate=str(candidate["mode"]),
        result_path=_resolve(candidate["codec_result"]),
        checkpoint_path=_resolve(candidate["codec_checkpoint"]),
        device=device,
    )
    statistics_path = _resolve(candidate["statistics"])
    statistics = LatentStatistics.from_state_dict(
        torch.load(statistics_path, map_location="cpu", weights_only=False)
    )
    if statistics.ratio != 4 or statistics.mode != "ratio4_state_detail":
        raise RuntimeError("sequential v1 requires frozen R4 statistics")
    if sha256_file(statistics_path) != str(candidate["statistics_file_sha256"]):
        raise RuntimeError("statistics file SHA256 differs from the frozen contract")
    return SequentialContext(
        cfg=cfg,
        output_dir=output_dir,
        data=data,
        codec=codec,
        statistics=statistics,
        statistics_device=statistics.to(device=device),
        device=device,
    )


def _selection(cfg: Mapping[str, Any]) -> dict[str, Any]:
    data = _load_data(_resolve(cfg["manifest_root"]))
    try:
        def choose(dataset: Any, count: int) -> list[dict[str, Any]]:
            by_system = {}
            for index, row in enumerate(dataset._index):
                sample_id = str(row[0])
                parsed = parse_sample_id(sample_id)
                if parsed["replica"] == str(cfg["evaluation"]["fixed_replica"]) and parsed["window"] == int(cfg["evaluation"]["fixed_window"]):
                    by_system.setdefault(
                        parsed["system"],
                        {
                            "dataset_index": index,
                            "sample_id": sample_id,
                            "system": parsed["system"],
                            "replica": parsed["replica"],
                            "window": parsed["window"],
                        },
                    )
            values = [by_system[key] for key in sorted(by_system)]
            if len(values) < count:
                raise RuntimeError(f"fixed selection has {len(values)} systems; need {count}")
            return values[:count]

        quick_train = choose(data.train, int(cfg["evaluation"]["quick_train_systems"]))
        quick_valid = choose(data.valid, int(cfg["evaluation"]["quick_valid_systems"]))
        validation = _make_validation_plan(data.valid)
        final_valid = [
            {
                "dataset_index": int(index),
                "sample_id": str(sample_id),
                "system": parse_sample_id(str(sample_id))["system"],
                "replica": parse_sample_id(str(sample_id))["replica"],
                "window": parse_sample_id(str(sample_id))["window"],
            }
            for index, sample_id in zip(
                validation.selected_dataset_indices, validation.selected_sample_ids
            )
        ]
        smoke_cfg = cfg["evaluation"]["smoke"]
        smoke_root = _resolve(smoke_cfg["root"])
        smoke_index = smoke_root / "train" / "index.txt"
        available = {}
        with smoke_index.open("r", encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                sample_id = line.split("\t", 1)[0]
                available[sample_id] = index
        smoke = []
        for system in smoke_cfg["systems"]:
            for replica in smoke_cfg["replicas"]:
                sample_id = f"{system}_{replica}_w{int(smoke_cfg['window']):06d}"
                if sample_id not in available:
                    raise RuntimeError(f"historical smoke clip is missing: {sample_id}")
                smoke.append(
                    {
                        "dataset_index": int(available[sample_id]),
                        "sample_id": sample_id,
                        "system": str(system),
                        "replica": str(replica),
                        "window": int(smoke_cfg["window"]),
                    }
                )
        return {
            "schema": f"{SCHEMA}.selection.v1",
            "quick_train": quick_train,
            "quick_valid": quick_valid,
            "final_valid": final_valid,
            "smoke": smoke,
            "smoke_root": str(smoke_root),
            "validation_schedule_hash": validation.schedule_hash,
            "train_index_sha256": sha256_file(data.manifest_root / "clip_store" / "train" / "index.txt"),
            "valid_index_sha256": sha256_file(data.manifest_root / "clip_store" / "valid" / "index.txt"),
            "quality_independent": True,
            "test_payload_opened": False,
        }
    finally:
        data.train.close()
        data.valid.close()


def _capacity_status(cfg: Mapping[str, Any]) -> dict[str, Any]:
    root = _resolve(cfg["baseline"]["capacity_run_root"])
    experiment = str(cfg["baseline"]["experiment_id"])
    summary_path = root / experiment / "train_summary.json"
    if not summary_path.is_file():
        return {
            "available": False,
            "reason": "completed C48 train_summary.json is not present",
            "summary": str(summary_path),
        }
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "PASS":
        return {
            "available": False,
            "reason": f"C48 summary status is {summary.get('status')!r}, not PASS",
            "summary": str(summary_path),
        }
    recorded = Path(str(summary.get("final_checkpoint", "")))
    checkpoint = root / experiment / "checkpoints" / recorded.name
    if not checkpoint.is_file():
        return {
            "available": False,
            "reason": "C48 recorded final checkpoint is not locally present",
            "summary": str(summary_path),
            "recorded_checkpoint": str(recorded),
            "local_checkpoint": str(checkpoint),
        }
    if checkpoint.stat().st_mtime_ns > summary_path.stat().st_mtime_ns:
        return {
            "available": False,
            "reason": "C48 checkpoint is newer than its completion summary",
            "summary": str(summary_path),
            "local_checkpoint": str(checkpoint),
        }
    return {
        "available": True,
        "summary": str(summary_path),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "actual_optimizer_updates": int(summary["actual_optimizer_updates"]),
        "tokens_seen": int(summary["tokens_seen"]),
        "status": summary["status"],
    }


def _preflight(cfg: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    candidate = cfg["candidate"]
    manifest_root = _resolve(cfg["manifest_root"])
    manifest_path = manifest_root / "manifest.json"
    materialization_path = manifest_root / "materialization.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    materialization = json.loads(materialization_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "FROZEN":
        raise RuntimeError("manifest is not frozen")
    if manifest.get("test_sampling", {}).get("opened") is not False:
        raise RuntimeError("manifest does not certify sealed test sampling")
    if materialization.get("materialized") is not True:
        raise RuntimeError("train/validation materialization is incomplete")
    codec_path = _resolve(candidate["codec_checkpoint"])
    statistics_path = _resolve(candidate["statistics"])
    if sha256_file(codec_path) != str(candidate["codec_checkpoint_sha256"]):
        raise RuntimeError("codec checkpoint SHA256 differs from the frozen contract")
    if sha256_file(statistics_path) != str(candidate["statistics_file_sha256"]):
        raise RuntimeError("statistics SHA256 differs from the frozen contract")
    selection = _selection(cfg)
    selection_path = output_dir / "selection.json"
    if selection_path.is_file():
        existing = json.loads(selection_path.read_text(encoding="utf-8"))
        if existing != selection:
            raise RuntimeError("existing fixed selection differs; refusing reselection")
    else:
        _write_json(selection_path, selection)
    result = {
        "schema": f"{SCHEMA}.preflight.v1",
        "status": "PASS",
        "manifest": str(manifest_root),
        "manifest_sha256": sha256_file(manifest_path),
        "materialization_sha256": sha256_file(materialization_path),
        "counts": manifest.get("counts", {}),
        "codec_checkpoint": str(codec_path),
        "codec_checkpoint_sha256": sha256_file(codec_path),
        "statistics": str(statistics_path),
        "statistics_file_sha256": sha256_file(statistics_path),
        "capacity_c48": _capacity_status(cfg),
        "selection": str(selection_path),
        "code": _code_provenance(),
        "test_payload_opened": False,
    }
    _write_json(output_dir / "preflight.json", result)
    return result


def _new_trainer(
    ctx: SequentialContext,
    *,
    ffn_norm_source: str,
    max_steps: int,
    initial_state: Mapping[str, torch.Tensor],
    initial_hash: str,
    arm: str,
    round_name: str,
) -> DiTTrainer:
    torch.manual_seed(int(ctx.cfg["evaluation"]["training_seed_base"]))
    adapter = _new_adapter(ctx.cfg, ctx.device)
    model = _new_model(ctx.cfg, adapter, ffn_norm_source=ffn_norm_source).to(ctx.device)
    model.load_state_dict(initial_state, strict=True)
    if module_state_hash(model) != initial_hash:
        raise RuntimeError("model state hash did not round-trip into an independent model")
    protocol = ctx.cfg["protocol"]
    model_cfg = ctx.cfg["model"]
    config = DiTTrainConfig(
        ratio=4,
        mode="ratio4_state_detail",
        codec_width=int(model_cfg["codec_width"]),
        scalar_width=int(model_cfg["scalar_width"]),
        vector_width=int(model_cfg["vector_width"]),
        depth=int(model_cfg["depth"]),
        heads=int(model_cfg["heads"]),
        ffn_multiplier=int(model_cfg["ffn_multiplier"]),
        dropout=float(model_cfg["dropout"]),
        learning_rate=float(protocol["learning_rate"]),
        weight_decay=float(protocol["weight_decay"]),
        grad_clip=float(protocol["grad_clip"]),
        max_steps=int(max_steps),
        seed=int(ctx.cfg["evaluation"]["training_seed_base"]),
        amp=True,
        output_root=str(ctx.output_dir / round_name / arm),
        data_hash=ctx.data_hash,
        codec_hash=ctx.codec.codec_state_hash,
        stats_hash=ctx.statistics.hash,
        source_mode=str(ctx.cfg["source"]["mode"]),
        center_kind=str(ctx.cfg["source"]["center_kind"]),
        source_sigma=float(ctx.cfg["source"]["sigma"]),
        normalization_hash=ctx.statistics.hash,
        init_hash=initial_hash,
        observation_mixture=tuple(int(value) for value in ctx.cfg["schedule"]["observation_history"]),
        metadata={
            "phase": PHASE,
            "round": round_name,
            "arm": arm,
            "execution_backend": str(model_cfg["execution_backend"]),
            "ffn_norm_source": ffn_norm_source,
        },
    )
    return DiTTrainer(
        model,
        adapter,
        config=config,
        statistics=ctx.statistics_device,
        codec=ctx.codec.model,
        frame_encoder=ctx.codec.model.frame_encoder,
    )


def _optimizer_to_contract_device(optimizer: torch.optim.Optimizer) -> None:
    for state in optimizer.state.values():
        for key, value in tuple(state.items()):
            if not isinstance(value, torch.Tensor):
                continue
            if key == "step" and value.ndim == 0:
                state[key] = value.detach().to(device="cpu")
            else:
                parameter_device = optimizer.param_groups[0]["params"][0].device
                state[key] = value.detach().to(device=parameter_device)


def _restore_parent(
    trainer: DiTTrainer,
    payload: Mapping[str, Any],
    *,
    expected_codec_hash: str,
    expected_statistics_hash: str,
) -> None:
    if payload.get("codec_hash") != expected_codec_hash:
        raise RuntimeError("parent checkpoint codec hash mismatch")
    if payload.get("statistics_hash") != expected_statistics_hash:
        raise RuntimeError("parent checkpoint statistics hash mismatch")
    trainer.model.load_state_dict(payload["model_state"], strict=True)
    trainer.optimizer.load_state_dict(copy.deepcopy(payload["optimizer_state"]))
    _optimizer_to_contract_device(trainer.optimizer)
    if payload.get("scaler_state"):
        trainer.scaler.load_state_dict(copy.deepcopy(payload["scaler_state"]))
    trainer.step = int(payload["step"])
    successful_updates = payload.get("successful_optimizer_updates")
    if successful_updates is None:
        capacity = payload.get("capacity", {})
        if capacity.get("schema") != "pvb.dit.capacity_data.runner.v1.checkpoint.v1":
            raise RuntimeError("parent checkpoint lacks successful-update provenance")
        successful_updates = capacity.get("successful_optimizer_updates")
    trainer.successful_updates = int(successful_updates)
    if trainer.step != trainer.successful_updates:
        raise RuntimeError("parent step and successful update count disagree")


def _checkpoint_payload(
    trainer: DiTTrainer,
    *,
    round_name: str,
    arm: str,
    parent_path: str | None,
    parent_sha256: str | None,
    parent_model_hash: str,
    cursor: Mapping[str, int],
    generator: torch.Generator,
    added_tokens: int,
    added_updates: int,
    schedule_hash: str,
    unique_variable: Mapping[str, Any],
) -> dict[str, Any]:
    payload = trainer.checkpoint_payload()
    payload["sequential"] = {
        "schema": f"{SCHEMA}.checkpoint.v1",
        "round": round_name,
        "arm": arm,
        "parent_checkpoint": parent_path,
        "parent_checkpoint_sha256": parent_sha256,
        "parent_model_state_hash": parent_model_hash,
        "cursor": {"epoch": int(cursor["epoch"]), "batch_index": int(cursor["batch_index"])},
        "training_generator_state": generator.get_state(),
        "added_future_atom_frame_tokens": int(added_tokens),
        "added_successful_updates": int(added_updates),
        "schedule_hash": schedule_hash,
        "unique_variable": dict(unique_variable),
        "code": _code_provenance(),
        "test_payload_opened": False,
    }
    return payload


def _save_checkpoint(
    directory: Path,
    trainer: DiTTrainer,
    *,
    round_name: str,
    arm: str,
    parent_path: str | None,
    parent_sha256: str | None,
    parent_model_hash: str,
    cursor: Mapping[str, int],
    generator: torch.Generator,
    added_tokens: int,
    added_updates: int,
    schedule_hash: str,
    unique_variable: Mapping[str, Any],
    final: bool,
) -> Path:
    payload = _checkpoint_payload(
        trainer,
        round_name=round_name,
        arm=arm,
        parent_path=parent_path,
        parent_sha256=parent_sha256,
        parent_model_hash=parent_model_hash,
        cursor=cursor,
        generator=generator,
        added_tokens=added_tokens,
        added_updates=added_updates,
        schedule_hash=schedule_hash,
        unique_variable=unique_variable,
    )
    name = "checkpoint_final.pt" if final else f"checkpoint_step{trainer.step:06d}.pt"
    path = directory / name
    _atomic_torch_save(path, payload)
    _atomic_torch_save(directory / "latest.pt", payload)
    return path


def _state_from_payload(
    cfg: Mapping[str, Any],
    payload: Mapping[str, Any],
    device: torch.device,
    *,
    ffn_norm_source: str,
) -> tuple[dict[str, torch.Tensor], str]:
    adapter = _new_adapter(cfg, device)
    model = _new_model(cfg, adapter, ffn_norm_source=ffn_norm_source).to(device)
    model.load_state_dict(payload["model_state"], strict=True)
    state = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }
    state_hash = module_state_hash(model)
    del model, adapter
    return state, state_hash


def _capacity_payload(
    ctx: SequentialContext,
) -> tuple[Mapping[str, Any], Mapping[str, Any], Path]:
    status = _capacity_status(ctx.cfg)
    if not status["available"]:
        raise FileNotFoundError(status["reason"])
    checkpoint = Path(status["checkpoint"])
    summary = json.loads(Path(status["summary"]).read_text(encoding="utf-8"))
    capacity_root = _resolve(ctx.cfg["baseline"]["capacity_run_root"])
    capacity_contract_path = capacity_root / "experiment_contract.json"
    capacity_preflight_path = capacity_root / "preflight.json"
    if not capacity_contract_path.is_file() or not capacity_preflight_path.is_file():
        raise RuntimeError("C48 lacks its frozen experiment contract or preflight provenance")
    capacity_contract = json.loads(capacity_contract_path.read_text(encoding="utf-8"))
    capacity_preflight = json.loads(capacity_preflight_path.read_text(encoding="utf-8"))
    expected_code_commit = str(ctx.cfg["baseline"]["expected_code_commit"])
    if capacity_contract.get("code_commit") != expected_code_commit:
        raise RuntimeError("C48 experiment contract was not frozen at the audited code commit")
    if capacity_preflight.get("test_payload_opened") is not False:
        raise RuntimeError("C48 preflight does not certify a sealed test payload")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("schema") != "pvb.dit.state_detail.checkpoint.v2":
        raise RuntimeError("C48 generic checkpoint schema mismatch")
    capacity = payload.get("capacity", {})
    expected = {
        "schema": "pvb.dit.capacity_data.runner.v1.checkpoint.v1",
        "experiment_id": "C48",
        "source_mode": "conditional",
        "data_scale": "base48",
        "depth": 4,
        "test_payload_opened": False,
    }
    for key, value in expected.items():
        if capacity.get(key) != value:
            raise RuntimeError(f"C48 capacity contract mismatch at {key}")
    contracted_experiment = capacity_contract.get("experiments", {}).get("C48", {})
    for key, value in (
        ("source_mode", "conditional"),
        ("data_scale", "base48"),
        ("depth", 4),
    ):
        if contracted_experiment.get(key) != value:
            raise RuntimeError(f"C48 frozen experiment contract mismatch at {key}")
    if payload.get("data_hash") != capacity_contract.get("data_hash"):
        raise RuntimeError("C48 checkpoint/data-contract hash mismatch")
    if capacity.get("data_hash") != payload.get("data_hash"):
        raise RuntimeError("C48 nested/generic data hashes disagree")
    config = payload.get("config", {})
    fixed = {
        "ratio": 4,
        "mode": "ratio4_state_detail",
        "scalar_width": 256,
        "vector_width": 128,
        "depth": 4,
        "heads": 8,
        "ffn_multiplier": 4,
        "dropout": 0.0,
        "source_mode": "conditional",
        "center_kind": "repeat_last_coordinate_encode",
        "source_sigma": 1.0,
    }
    for key, value in fixed.items():
        if config.get(key) != value:
            raise RuntimeError(f"C48 frozen recipe mismatch at {key}")
    expected_adapter = _new_adapter(ctx.cfg, ctx.device)
    try:
        if dict(payload.get("adapter_contract", {})) != dict(expected_adapter.contract()):
            raise RuntimeError("C48 adapter contract mismatch")
    finally:
        del expected_adapter
    if payload.get("codec_hash") != ctx.codec.codec_state_hash:
        raise RuntimeError("C48 frozen codec state hash mismatch")
    if payload.get("statistics_hash") != ctx.statistics.hash:
        raise RuntimeError("C48 statistics contract hash mismatch")
    expected_frozen_hashes = {
        "codec": module_state_hash(ctx.codec.model),
        "frame_encoder": module_state_hash(ctx.codec.model.frame_encoder),
    }
    if dict(payload.get("frozen_hashes", {})) != expected_frozen_hashes:
        raise RuntimeError("C48 frozen codec/frame-encoder state hash mismatch")
    source_contract = payload.get("source_contract", {})
    for key, value in (
        ("source_mode", "conditional"),
        ("center_kind", "repeat_last_coordinate_encode"),
        ("source_sigma", 1.0),
        ("normalization_hash", ctx.statistics.hash),
    ):
        if source_contract.get(key) != value:
            raise RuntimeError(f"C48 checkpoint source contract mismatch at {key}")
    if not isinstance(payload.get("optimizer_state"), Mapping):
        raise RuntimeError("C48 checkpoint lacks an optimizer state")
    if not isinstance(payload.get("scaler_state"), Mapping):
        raise RuntimeError("C48 checkpoint lacks an AMP scaler state")
    if not isinstance(payload.get("rng_state"), Mapping):
        raise RuntimeError("C48 checkpoint lacks generic RNG state")
    if not isinstance(capacity.get("generator_state"), torch.Tensor):
        raise RuntimeError("C48 checkpoint lacks its independent training generator state")
    if not isinstance(capacity.get("cursor"), Mapping) or not capacity.get("schedule_hash"):
        raise RuntimeError("C48 checkpoint lacks its sampler cursor or schedule hash")
    capacity_manifest = _resolve(ctx.cfg["baseline"]["capacity_data_manifest"])
    original_manifest = _resolve(ctx.cfg["manifest_root"])
    for original_split, capacity_split in (("train", "train48"), ("valid", "valid")):
        original_index = original_manifest / "clip_store" / original_split / "index.txt"
        capacity_index = capacity_manifest / "clip_store" / capacity_split / "index.txt"
        if _clip_index_semantic_hash(original_index) != _clip_index_semantic_hash(capacity_index):
            raise RuntimeError(
                f"C48 {capacity_split} clip identities/shapes differ from frozen {original_split}"
            )
    if int(payload.get("step", -1)) != int(summary["actual_optimizer_updates"]):
        raise RuntimeError("C48 summary/checkpoint update count mismatch")
    if int(capacity.get("successful_optimizer_updates", -1)) != int(payload["step"]):
        raise RuntimeError("C48 capacity successful update count mismatch")
    if int(payload["step"]) != int(ctx.cfg["baseline"]["expected_successful_updates"]):
        raise RuntimeError("C48 successful-update exposure differs from the audited budget")
    if int(summary.get("tokens_seen", -1)) != int(
        ctx.cfg["baseline"]["expected_effective_atom_frame_tokens"]
    ):
        raise RuntimeError("C48 effective-token exposure differs from the audited budget")
    if int(capacity.get("tokens_seen", -1)) != int(summary["tokens_seen"]):
        raise RuntimeError("C48 summary/checkpoint token exposure mismatch")
    return payload, summary, checkpoint


def _validated_capacity_status(ctx: SequentialContext) -> dict[str, Any]:
    status = _capacity_status(ctx.cfg)
    if not status["available"]:
        return status
    try:
        _payload, _summary, _checkpoint = _capacity_payload(ctx)
    except (OSError, EOFError, KeyError, RuntimeError, ValueError) as error:
        return {
            **status,
            "available": False,
            "reuse_validation": "REJECT",
            "reason": f"completed C48 failed strict reuse validation: {error}",
        }
    return {**status, "reuse_validation": "PASS"}


def _import_baseline(ctx: SequentialContext) -> dict[str, Any]:
    payload, summary, checkpoint = _capacity_payload(ctx)
    state, state_hash = _state_from_payload(
        ctx.cfg, payload, ctx.device, ffn_norm_source="post_adaln"
    )
    trainer = _new_trainer(
        ctx,
        ffn_norm_source="post_adaln",
        max_steps=max(int(payload["step"]), 1),
        initial_state=state,
        initial_hash=state_hash,
        arm="C48",
        round_name="baseline",
    )
    _restore_parent(
        trainer,
        payload,
        expected_codec_hash=ctx.codec.codec_state_hash,
        expected_statistics_hash=ctx.statistics.hash,
    )
    source_rng = payload.get("rng_state")
    generator = torch.Generator(device=ctx.device).manual_seed(
        int(ctx.cfg["evaluation"]["training_seed_base"])
    )
    capacity = payload["capacity"]
    if isinstance(capacity.get("generator_state"), torch.Tensor):
        generator.set_state(capacity["generator_state"].detach().to(device="cpu"))
    output = ctx.output_dir / "baseline"
    final_path = _save_checkpoint(
        output,
        trainer,
        round_name="baseline",
        arm="C48",
        parent_path=str(checkpoint),
        parent_sha256=sha256_file(checkpoint),
        parent_model_hash=state_hash,
        cursor=capacity.get("cursor", {"epoch": 0, "batch_index": 0}),
        generator=generator,
        added_tokens=0,
        added_updates=0,
        schedule_hash=str(capacity.get("schedule_hash", "")),
        unique_variable={"baseline": "audited_capacity_C48_import", "ffn_norm_source": "post_adaln"},
        final=True,
    )
    if source_rng is not None:
        migrated = torch.load(final_path, map_location="cpu", weights_only=False)
        migrated["rng_state"] = source_rng
        _atomic_torch_save(final_path, migrated)
        _atomic_torch_save(output / "latest.pt", migrated)
    result = {
        "schema": f"{SCHEMA}.baseline.v1",
        "status": "PASS",
        "source": "capacity_C48",
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_sha256": sha256_file(checkpoint),
        "source_summary": summary,
        "checkpoint": str(final_path),
        "checkpoint_sha256": sha256_file(final_path),
        "model_state_hash": module_state_hash(trainer.model),
        "adapter_state_hash": module_state_hash(trainer.adapter),
        "initialization_hash": state_hash,
        "parameter_count": sum(parameter.numel() for parameter in trainer.model.parameters()),
        "successful_updates": trainer.successful_updates,
        "tokens_seen_before_import": int(summary["tokens_seen"]),
        "config_hash": _canonical_hash(ctx.cfg),
        "data_hash": ctx.data_hash,
        "codec_state_hash": ctx.codec.codec_state_hash,
        "statistics_hash": ctx.statistics.hash,
        "code": _code_provenance(),
        "ffn_norm_source": "post_adaln",
        "independent_adapter": trainer.adapter is trainer.model.adapter,
        "test_payload_opened": False,
    }
    _write_json(output / "summary.json", result)
    return result


def _initial_state(
    cfg: Mapping[str, Any],
    device: torch.device,
    *,
    ffn_norm_source: str,
) -> tuple[dict[str, torch.Tensor], str]:
    torch.manual_seed(int(cfg["evaluation"]["training_seed_base"]))
    adapter = _new_adapter(cfg, device)
    model = _new_model(cfg, adapter, ffn_norm_source=ffn_norm_source).to(device)
    state = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }
    return state, module_state_hash(model)


def _next_batch(
    sampler: Any, cursor: dict[str, int], plan_cache: dict[str, Any]
) -> tuple[tuple[int, ...], str]:
    while True:
        epoch = int(cursor["epoch"])
        if int(plan_cache.get("epoch", -1)) != epoch:
            sampler.set_epoch(epoch)
            batches = tuple(
                tuple(int(value) for value in row) for row in sampler.global_batches
            )
            if not batches:
                raise RuntimeError("training sampler produced no batches")
            plan_cache.clear()
            plan_cache.update(
                {
                    "epoch": epoch,
                    "batches": batches,
                    "schedule_hash": _canonical_hash(
                        {
                            "epoch": epoch,
                            "batches": [list(row) for row in batches],
                            "sampler": sampler.state_dict(),
                        }
                    ),
                }
            )
        batches = plan_cache["batches"]
        if int(cursor["batch_index"]) < len(batches):
            result = batches[int(cursor["batch_index"])]
            cursor["batch_index"] += 1
            return result, str(plan_cache["schedule_hash"])
        cursor["epoch"] += 1
        cursor["batch_index"] = 0
        plan_cache.clear()


def _batch_tokens(
    specs: Sequence[Any], indices: Sequence[int], history_frames: int
) -> int:
    history = int(history_frames)
    total = 0
    for index in indices:
        spec = specs[int(index)]
        future_frames = int(spec.frames) - history
        if future_frames < 1:
            raise RuntimeError("training batch has no future frames")
        total += int(spec.atoms) * future_frames
    return total


def _prepare_batch(
    ctx: SequentialContext,
    dataset: Any,
    indices: Sequence[int],
    adapter: StateDetailLatentAdapter,
    history: int,
) -> tuple[Any, Any, Any]:
    latent, _batch_cpu, batch = _encode_batch(
        dataset,
        indices,
        codec=ctx.codec,
        adapter=adapter,
        data_hash=ctx.data_hash,
        device=ctx.device,
    )
    target = adapter.pack(
        latent,
        codec_hash=ctx.codec.codec_state_hash,
        data_hash=ctx.data_hash,
        origin_from_latent=True,
        loss_mask=batch.loss_mask,
    )
    observed = _observed_batch(
        target, batch, adapter=adapter, history_frames=int(history)
    )
    return latent, batch, observed


def _source_center(
    ctx: SequentialContext,
    batch: Any,
    observed: Any,
    adapter: StateDetailLatentAdapter,
    history: int,
) -> Any:
    center, metadata = build_observed_center(
        str(ctx.cfg["source"]["center_kind"]),
        codec_model=ctx.codec.model,
        coordinate_batch=batch,
        target_batch=observed,
        adapter=adapter,
        statistics=ctx.statistics_device,
        history_frames=int(history),
        codec_hash=ctx.codec.codec_state_hash,
        data_hash=ctx.data_hash,
    )
    if metadata.get("uses_future_coordinates") is not False:
        raise RuntimeError("conditional source unexpectedly uses future coordinates")
    return center


def _tokens_for_updates(ctx: SequentialContext, updates: int, seed: int) -> int:
    sampler = _make_sampler(
        ctx.data.train,
        seed=int(seed),
        clips_per_trajectory=int(ctx.cfg["schedule"]["clips_per_trajectory"]),
    )
    cursor = {"epoch": 0, "batch_index": 0}
    plan_cache: dict[str, Any] = {}
    specs = ctx.data.train.clip_spec_table()
    total = 0
    history_order = tuple(int(value) for value in ctx.cfg["schedule"]["history_order"])
    for update in range(int(updates)):
        indices, _ = _next_batch(sampler, cursor, plan_cache)
        history = history_order[update % len(history_order)]
        total += _batch_tokens(specs, indices, history)
    return total


def _profile(ctx: SequentialContext) -> dict[str, Any]:
    warmup = int(ctx.cfg["budget"]["profile_warmup_updates"])
    measured = int(ctx.cfg["budget"]["profile_measured_updates"])
    total_profile = warmup + measured
    baseline_summary_path = ctx.output_dir / "baseline" / "summary.json"
    parent_payload = None
    parent_path = None
    if baseline_summary_path.is_file():
        baseline_summary = json.loads(baseline_summary_path.read_text(encoding="utf-8"))
        parent_path = Path(baseline_summary["checkpoint"])
        parent_payload = torch.load(parent_path, map_location="cpu", weights_only=False)
        state, state_hash = _state_from_payload(
            ctx.cfg, parent_payload, ctx.device, ffn_norm_source="post_adaln"
        )
        parent_step = int(parent_payload["step"])
    else:
        state, state_hash = _initial_state(
            ctx.cfg, ctx.device, ffn_norm_source="post_adaln"
        )
        parent_step = 0
    capacity_reuse = (
        {"available": False, "reason": "baseline checkpoint already exists"}
        if parent_payload is not None
        else _validated_capacity_status(ctx)
    )
    sampler = _make_sampler(
        ctx.data.train,
        seed=int(ctx.cfg["evaluation"]["training_seed_base"]) + 1,
        clips_per_trajectory=int(ctx.cfg["schedule"]["clips_per_trajectory"]),
    )
    cursor = {"epoch": 0, "batch_index": 0}
    plan_cache: dict[str, Any] = {}
    batches = [
        _next_batch(sampler, cursor, plan_cache)[0] for _ in range(total_profile)
    ]
    rows = {}
    for arm in ROUND1_ARMS:
        trainer = _new_trainer(
            ctx,
            ffn_norm_source=NORM_SOURCE[arm],
            max_steps=max(parent_step + total_profile, 1),
            initial_state=state,
            initial_hash=state_hash,
            arm=arm,
            round_name="round1_profile",
        )
        if parent_payload is not None:
            _restore_parent(
                trainer,
                parent_payload,
                expected_codec_hash=ctx.codec.codec_state_hash,
                expected_statistics_hash=ctx.statistics.hash,
            )
        generator = torch.Generator(device=ctx.device).manual_seed(
            int(ctx.cfg["evaluation"]["training_seed_base"]) + 1
        )
        timings = []
        torch.cuda.reset_peak_memory_stats(ctx.device)
        for update, indices in enumerate(batches):
            history = int(ctx.cfg["schedule"]["history_order"][update % 2])
            _sync(ctx.device)
            started = time.perf_counter()
            _latent, batch, observed = _prepare_batch(
                ctx, ctx.data.train, indices, trainer.adapter, history
            )
            center = _source_center(
                ctx, batch, observed, trainer.adapter, history
            )
            trainer.train_step(observed, generator=generator, source_center=center)
            _sync(ctx.device)
            timings.append(time.perf_counter() - started)
        generation_timings = []
        generation_metrics = []
        generation_batches = int(ctx.cfg["budget"]["profile_generation_batches"])
        for generation_index in range(generation_batches):
            selected = batches[(warmup + generation_index) % len(batches)]
            indices = (int(selected[0]),)
            history = int(
                ctx.cfg["schedule"]["history_order"][generation_index % 2]
            )
            _sync(ctx.device)
            started = time.perf_counter()
            _latent, batch, observed = _prepare_batch(
                ctx, ctx.data.train, indices, trainer.adapter, history
            )
            center = _source_center(
                ctx, batch, observed, trainer.adapter, history
            )
            normalized = ctx.statistics_device.normalize(observed)
            noise, _noise_meta = make_fixed_noise(
                normalized,
                seed=generation_seed(
                    int(ctx.cfg["evaluation"]["generation_seed"]),
                    str(batch.sample_id[0]),
                    history,
                    0,
                ),
            )
            generated, generation = generate_fixed_noise_latent(
                trainer.model,
                trainer.adapter,
                observed,
                ctx.statistics_device,
                noise=noise,
                steps=int(ctx.cfg["evaluation"]["euler_steps"]),
                source_center=center,
                source_mode="conditional",
            )
            decoded = ctx.codec.model.decode(generated).x_hat.float()
            metrics = reassessment_trajectory_metrics(
                decoded, batch.x.float(), batch, history
            )
            _sync(ctx.device)
            generation_timings.append(time.perf_counter() - started)
            if not bool(torch.isfinite(decoded).all()) or not bool(
                generation["observed_clamp_exact"]
            ):
                raise RuntimeError("profile generation/decode produced invalid output")
            generation_metrics.append(
                {
                    "sample_id": str(batch.sample_id[0]),
                    "history_frames": history,
                    "bond_rmse": metrics["future"]["bond_rmse"],
                    "contact_f1": metrics["future"]["contact_f1"],
                }
            )
        values = sorted(timings[warmup:])
        generation_values = sorted(generation_timings)
        rows[arm] = {
            "ffn_norm_source": NORM_SOURCE[arm],
            "warmup_updates": warmup,
            "measured_updates": measured,
            "seconds": timings[warmup:],
            "p50_seconds": values[len(values) // 2],
            "p90_seconds": values[max(0, math.ceil(0.9 * len(values)) - 1)],
            "generation_evaluation": {
                "batches": generation_batches,
                "seconds": generation_timings,
                "p50_seconds": generation_values[len(generation_values) // 2],
                "p90_seconds": generation_values[
                    max(0, math.ceil(0.9 * len(generation_values)) - 1)
                ],
                "includes": [
                    "clip_read",
                    "codec_encode",
                    "source_center",
                    "euler16",
                    "decode",
                    "cuda_metrics",
                ],
                "metrics": generation_metrics,
            },
            "peak_memory_bytes": int(torch.cuda.max_memory_allocated(ctx.device)),
            "parent_step": parent_step,
        }
        del trainer
        torch.cuda.empty_cache()
    limit = float(ctx.cfg["budget"]["gpu_hours_total"]) * 3600.0
    reserve = float(ctx.cfg["budget"]["reserve_fraction"]) * limit
    training_limit = limit - reserve
    p90 = max(float(value["p90_seconds"]) for value in rows.values())
    baseline_needed = not bool(capacity_reuse["available"]) and parent_payload is None
    baseline_updates = 0
    round_updates = 0
    choices = []
    for baseline_candidate in (
        ctx.cfg["budget"]["baseline_candidate_updates"] if baseline_needed else [0]
    ):
        baseline_cost = 1.3 * int(baseline_candidate) * p90
        for round_candidate in ctx.cfg["budget"]["round_candidate_updates"]:
            paired_cost = 2.0 * int(round_candidate) * p90
            future_factor = 5.25
            estimated = baseline_cost + future_factor * paired_cost
            feasible = estimated <= training_limit
            choices.append(
                {
                    "baseline_updates": int(baseline_candidate),
                    "round_updates": int(round_candidate),
                    "estimated_training_gpu_seconds": estimated,
                    "feasible": feasible,
                }
            )
            if feasible and not round_updates:
                baseline_updates = int(baseline_candidate)
                round_updates = int(round_candidate)
        if round_updates:
            break
    if not round_updates:
        raise RuntimeError("no baseline/Round 1 budget leaves the required evaluation reserve")
    round_seed = int(ctx.cfg["evaluation"]["training_seed_base"]) + 1
    extension_updates = int(
        math.ceil(
            round_updates * float(ctx.cfg["budget"]["inconclusive_extension_fraction"])
        )
    )
    round_tokens = _tokens_for_updates(ctx, round_updates, round_seed)
    extended_tokens = _tokens_for_updates(
        ctx, round_updates + extension_updates, round_seed
    )
    result = {
        "schema": f"{SCHEMA}.budget.v1",
        "status": "PASS",
        "frozen_before_training": True,
        "gpu_seconds_total": limit,
        "wall_seconds_total": float(ctx.cfg["budget"]["wall_hours_total"]) * 3600.0,
        "evaluation_recovery_reserve_seconds": reserve,
        "baseline_reuse": capacity_reuse,
        "baseline_reuse_available": bool(capacity_reuse["available"]),
        "selected_baseline_updates": baseline_updates,
        "selected_round1_added_updates": round_updates,
        "selected_round1_future_atom_frame_tokens_per_arm": round_tokens,
        "selected_round1_extension_added_updates": extension_updates,
        "selected_round1_extension_future_atom_frame_tokens_per_arm": (
            extended_tokens - round_tokens
        ),
        "selected_round1_extended_total_updates": round_updates + extension_updates,
        "selected_round1_extended_total_future_atom_frame_tokens_per_arm": extended_tokens,
        "profile": rows,
        "choices": choices,
        "future_round_cost_multiplier": {
            "round1": 1.0,
            "round2_decode_backward": 2.0,
            "round3_refiner": 0.5,
            "round4_corruption": 1.25,
            "round1_inconclusive_extension_contingency": 0.5,
            "total": 5.25,
        },
        "device": _cuda_info(ctx.device),
        "code": _code_provenance(),
        "test_payload_opened": False,
    }
    _write_json(ctx.output_dir / "budget.json", result)
    _write_json(ctx.output_dir / "profile.json", {"schema": f"{SCHEMA}.profile.v1", "rows": rows})
    return result


def _load_budget(ctx: SequentialContext) -> dict[str, Any]:
    path = ctx.output_dir / "budget.json"
    if not path.is_file():
        raise FileNotFoundError("profile/budget is missing")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") != "PASS" or not value.get("frozen_before_training"):
        raise RuntimeError("budget is not a frozen passing budget")
    return value


def _train_baseline(ctx: SequentialContext, *, resume: bool) -> dict[str, Any]:
    output = ctx.output_dir / "baseline"
    summary_path = output / "summary.json"
    if summary_path.is_file() and not resume:
        raise RuntimeError("baseline summary exists; refusing to overwrite it")
    capacity = _validated_capacity_status(ctx)
    existing_training_artifacts = [
        path
        for path in (
            output / "latest.pt",
            output / "checkpoint_final.pt",
            output / "train_history.jsonl",
            output / "validation_history.jsonl",
        )
        if path.exists()
    ]
    if capacity["available"]:
        if resume or existing_training_artifacts:
            raise RuntimeError(
                "a completed C48 appeared after scratch baseline artifacts; refusing to switch lineage"
            )
        return _import_baseline(ctx)
    budget = _load_budget(ctx)
    target = int(budget["selected_baseline_updates"])
    if target < 1:
        raise RuntimeError("no reusable C48 and frozen budget did not allocate baseline updates")
    state, state_hash = _initial_state(
        ctx.cfg, ctx.device, ffn_norm_source="post_adaln"
    )
    trainer = _new_trainer(
        ctx,
        ffn_norm_source="post_adaln",
        max_steps=target,
        initial_state=state,
        initial_hash=state_hash,
        arm="scratch_C48",
        round_name="baseline",
    )
    generator = torch.Generator(device=ctx.device).manual_seed(
        int(ctx.cfg["evaluation"]["training_seed_base"])
    )
    sampler = _make_sampler(
        ctx.data.train,
        seed=int(ctx.cfg["evaluation"]["training_seed_base"]),
        clips_per_trajectory=int(ctx.cfg["schedule"]["clips_per_trajectory"]),
    )
    cursor = {"epoch": 0, "batch_index": 0}
    plan_cache: dict[str, Any] = {}
    specs = ctx.data.train.clip_spec_table()
    tokens = 0
    started_step = 0
    latest = output / "latest.pt"
    history_path = output / "train_history.jsonl"
    validation_path = output / "validation_history.jsonl"
    validation_rows = 0
    if resume:
        payload = trainer.load_checkpoint(latest, map_location=ctx.device)
        sequential = payload.get("sequential", {})
        if sequential.get("round") != "baseline":
            raise RuntimeError("baseline resume checkpoint contract mismatch")
        generator.set_state(sequential["training_generator_state"].detach().cpu())
        cursor = {
            "epoch": int(sequential["cursor"]["epoch"]),
            "batch_index": int(sequential["cursor"]["batch_index"]),
        }
        tokens = int(sequential["added_future_atom_frame_tokens"])
        started_step = trainer.successful_updates
        _truncate_jsonl(
            history_path,
            step_key="successful_optimizer_updates",
            maximum_step=trainer.successful_updates,
        )
        validation_rows = _truncate_jsonl(
            validation_path,
            step_key="step",
            maximum_step=trainer.successful_updates,
        )
    elif latest.exists() or history_path.exists() or validation_path.exists():
        raise RuntimeError("fresh baseline run refuses existing training artifacts")
    torch.cuda.reset_peak_memory_stats(ctx.device)
    started = time.perf_counter()
    schedule_hash = ""
    while trainer.successful_updates < target:
        update = trainer.successful_updates
        indices, schedule_hash = _next_batch(sampler, cursor, plan_cache)
        history = int(ctx.cfg["schedule"]["history_order"][update % 2])
        _latent, batch, observed = _prepare_batch(
            ctx, ctx.data.train, indices, trainer.adapter, history
        )
        center = _source_center(ctx, batch, observed, trainer.adapter, history)
        row = trainer.train_step(observed, generator=generator, source_center=center)
        batch_tokens = _batch_tokens(specs, indices, history)
        tokens += batch_tokens
        row.update(
            {
                "round": "baseline",
                "arm": "scratch_C48",
                "history_frames": history,
                "batch_tokens": batch_tokens,
                "tokens_seen": tokens,
                "successful_optimizer_updates": trainer.successful_updates,
            }
        )
        _append_jsonl(history_path, row)
        if trainer.successful_updates % int(ctx.cfg["schedule"]["validation_interval"]) == 0:
            validation = _validation_rf(
                ctx, trainer, step=trainer.successful_updates
            )
            _append_jsonl(validation_path, validation)
            validation_rows += 1
        if trainer.successful_updates % int(ctx.cfg["schedule"]["checkpoint_interval"]) == 0:
            _save_checkpoint(
                output,
                trainer,
                round_name="baseline",
                arm="scratch_C48",
                parent_path=None,
                parent_sha256=None,
                parent_model_hash=state_hash,
                cursor=cursor,
                generator=generator,
                added_tokens=tokens,
                added_updates=trainer.successful_updates,
                schedule_hash=schedule_hash,
                unique_variable={"baseline": "clean_scratch_C48", "ffn_norm_source": "post_adaln"},
                final=False,
            )
    final_path = _save_checkpoint(
        output,
        trainer,
        round_name="baseline",
        arm="scratch_C48",
        parent_path=None,
        parent_sha256=None,
        parent_model_hash=state_hash,
        cursor=cursor,
        generator=generator,
        added_tokens=tokens,
        added_updates=trainer.successful_updates,
        schedule_hash=schedule_hash,
        unique_variable={"baseline": "clean_scratch_C48", "ffn_norm_source": "post_adaln"},
        final=True,
    )
    elapsed = time.perf_counter() - started
    result = {
        "schema": f"{SCHEMA}.baseline.v1",
        "status": "PASS" if trainer.successful_updates == target else "PARTIAL",
        "source": "clean_scratch_C48",
        "checkpoint": str(final_path),
        "checkpoint_sha256": sha256_file(final_path),
        "model_state_hash": module_state_hash(trainer.model),
        "adapter_state_hash": module_state_hash(trainer.adapter),
        "initialization_hash": state_hash,
        "successful_updates": trainer.successful_updates,
        "updates_this_invocation": trainer.successful_updates - started_step,
        "tokens_seen": tokens,
        "validation_rows": validation_rows,
        "gpu_hours_this_invocation": elapsed / 3600.0,
        "future_atom_frame_tokens_per_second": tokens / max(elapsed, 1.0e-12),
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated(ctx.device)),
        "config_hash": _canonical_hash(ctx.cfg),
        "data_hash": ctx.data_hash,
        "codec_state_hash": ctx.codec.codec_state_hash,
        "statistics_hash": ctx.statistics.hash,
        "code": _code_provenance(),
        "ffn_norm_source": "post_adaln",
        "maturity": "pilot_only" if target < 20000 else "planned_clean_baseline",
        "capacity_reuse": capacity,
        "test_payload_opened": False,
    }
    _write_json(summary_path, result)
    return result


def _require_pass_summary(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} summary is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") != "PASS":
        raise RuntimeError(f"{label} status is not PASS")
    return value


def _baseline_checkpoint(ctx: SequentialContext) -> Path:
    summary_path = ctx.output_dir / "baseline" / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError("clean baseline summary is missing")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "PASS":
        raise RuntimeError("clean baseline is not complete")
    path = Path(summary["checkpoint"])
    if not path.is_file() or sha256_file(path) != summary["checkpoint_sha256"]:
        raise RuntimeError("clean baseline checkpoint is missing or changed")
    return path


def _round1_resume_paths(ctx: SequentialContext) -> dict[str, Path]:
    root = ctx.output_dir / "round1"
    latest = {arm: root / arm / "latest.pt" for arm in ROUND1_ARMS}
    if not all(path.is_file() for path in latest.values()):
        raise FileNotFoundError("Round 1 resume requires checkpoints for both arms")

    def exposure(path: Path, arm: str) -> tuple[Any, ...]:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        sequential = payload.get("sequential", {})
        if sequential.get("round") != "round1" or sequential.get("arm") != arm:
            raise RuntimeError(f"Round 1 resume contract mismatch for {arm}: {path}")
        return (
            int(payload.get("successful_optimizer_updates", -1)),
            int(sequential.get("added_successful_updates", -1)),
            int(sequential.get("added_future_atom_frame_tokens", -1)),
            dict(sequential.get("cursor", {})),
        )

    latest_exposure = {
        arm: exposure(path, arm) for arm, path in latest.items()
    }
    if latest_exposure["control"] == latest_exposure["candidate"]:
        return latest
    names = None
    for arm in ROUND1_ARMS:
        available = {
            path.name
            for path in (root / arm).glob("checkpoint_step*.pt")
            if path.is_file()
        }
        names = available if names is None else names & available
    if not names:
        raise RuntimeError("Round 1 arms have divergent latest checkpoints and no common checkpoint")
    for name in sorted(names, reverse=True):
        paths = {arm: root / arm / name for arm in ROUND1_ARMS}
        values = {arm: exposure(path, arm) for arm, path in paths.items()}
        if values["control"] == values["candidate"]:
            return paths
    raise RuntimeError("Round 1 has no common checkpoint with equal paired exposure")


def _validation_rf(
    ctx: SequentialContext,
    trainer: DiTTrainer,
    *,
    step: int,
    max_atom_frame_tokens: int | None = None,
) -> dict[str, Any]:
    plan = _make_validation_plan(ctx.data.valid)
    batches = plan.batches
    if max_atom_frame_tokens is not None:
        cap = int(max_atom_frame_tokens)
        if cap < 1:
            raise ValueError("validation token cap must be positive")
        sampler = TaskAwareClipBatchSampler(
            plan.subset,
            max_tokens=cap,
            seed=20260907,
            shuffle=False,
            replacement=False,
            oversize_policy="error",
        )
        batches = tuple(tuple(int(index) for index in batch) for batch in sampler.global_batches)
        if tuple(sorted(index for batch in batches for index in batch)) != tuple(range(len(plan.subset))):
            raise RuntimeError("capped validation batching must cover each selected clip exactly once")
    numerators = {name: 0.0 for name in ("state_h", "detail_h", "state_v", "detail_v")}
    counts = {name: 0 for name in numerators}
    was_training = trainer.model.training
    trainer.model.eval()
    for history in (4, 8):
        for local_indices in batches:
            _latent, batch, observed = _prepare_batch(
                ctx, plan.subset, local_indices, trainer.adapter, history
            )
            center = _source_center(ctx, batch, observed, trainer.adapter, history)
            normalized = trainer._normalise_batch(observed)
            seed = generation_seed(
                int(ctx.cfg["evaluation"]["generation_seed"]),
                "|".join(batch.sample_id),
                history,
                0,
            )
            generator = torch.Generator(device=ctx.device).manual_seed(seed)
            with torch.no_grad(), trainer.autocast_context():
                sample = trainer.flow.sample(
                    normalized,
                    generator=generator,
                    source_center=center,
                    source_mode="conditional",
                )
                prediction = trainer.model(
                    normalized.with_fields(sample.interpolated), sample.tau
                )
                loss = trainer.flow.loss(prediction, sample.target, normalized)
            for name, value in loss.fields.items():
                count = int(loss.valid_elements[name])
                numerators[name] += float(value.detach().float().cpu()) * count
                counts[name] += count
    if was_training:
        trainer.model.train()
    fields = {
        name: numerators[name] / max(counts[name], 1)
        for name in numerators
    }
    return {
        "step": int(step),
        "total": sum(fields.values()) / 4.0,
        "fields": fields,
        "valid_elements": counts,
        "validation_clips": len(plan.selected_sample_ids),
        "validation_batches": len(batches),
        "validation_max_atom_frame_tokens": None if max_atom_frame_tokens is None else int(max_atom_frame_tokens),
        "seed_excludes_arm_round_checkpoint_step": True,
    }


def _round1_exposure_plan(
    output_dir: Path, budget: Mapping[str, Any], *, resume: bool
) -> dict[str, Any]:
    base_delta = int(budget["selected_round1_added_updates"])
    extension_delta = int(budget["selected_round1_extension_added_updates"])
    prior_summary: dict[str, Any] = {}
    common_extension = False
    summary_path = output_dir / "round1" / "train_summary.json"
    decision_path = output_dir / "round1" / "decision.json"
    if resume and summary_path.is_file():
        prior_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        decision = (
            json.loads(decision_path.read_text(encoding="utf-8"))
            if decision_path.is_file()
            else {}
        )
        if decision.get("status") != "NEEDS_COMMON_EXTENSION":
            raise RuntimeError(
                "completed Round 1 can resume only after a NEEDS_COMMON_EXTENSION decision"
            )
        if prior_summary.get("status") != "PASS":
            raise RuntimeError("Round 1 extension requires a passing initial paired run")
        if bool(prior_summary.get("common_extension_applied", False)):
            raise RuntimeError("the one allowed Round 1 common extension was already applied")
        if int(prior_summary.get("added_successful_updates_per_arm", -1)) != base_delta:
            raise RuntimeError("initial Round 1 exposure differs from the frozen base budget")
        common_extension = True
    return {
        "base_delta": base_delta,
        "extension_delta": extension_delta,
        "target_delta": base_delta + (extension_delta if common_extension else 0),
        "common_extension": common_extension,
        "prior_summary": prior_summary,
    }


def _train_round1(ctx: SequentialContext, *, resume: bool) -> dict[str, Any]:
    _require_pass_summary(
        ctx.output_dir / "round1" / "smoke.json", "Round 1 real-clip smoke"
    )
    budget = _load_budget(ctx)
    exposure = _round1_exposure_plan(ctx.output_dir, budget, resume=resume)
    base_delta = int(exposure["base_delta"])
    extension_delta = int(exposure["extension_delta"])
    delta = int(exposure["target_delta"])
    common_extension = bool(exposure["common_extension"])
    prior_summary = dict(exposure["prior_summary"])
    parent_path = _baseline_checkpoint(ctx)
    parent_sha = sha256_file(parent_path)
    parent = torch.load(parent_path, map_location="cpu", weights_only=False)
    state, parent_model_hash = _state_from_payload(
        ctx.cfg, parent, ctx.device, ffn_norm_source="post_adaln"
    )
    parent_step = int(parent["step"])
    target_step = parent_step + delta
    trainers = {
        arm: _new_trainer(
            ctx,
            ffn_norm_source=NORM_SOURCE[arm],
            max_steps=target_step,
            initial_state=state,
            initial_hash=parent_model_hash,
            arm=arm,
            round_name="round1",
        )
        for arm in ROUND1_ARMS
    }
    for trainer in trainers.values():
        _restore_parent(
            trainer,
            parent,
            expected_codec_hash=ctx.codec.codec_state_hash,
            expected_statistics_hash=ctx.statistics.hash,
        )
    assert_independent_trainers(trainers["control"], trainers["candidate"])
    seed = int(ctx.cfg["evaluation"]["training_seed_base"]) + 1
    generators = {
        arm: torch.Generator(device=ctx.device).manual_seed(seed)
        for arm in ROUND1_ARMS
    }
    sampler = _make_sampler(
        ctx.data.train,
        seed=seed,
        clips_per_trajectory=int(ctx.cfg["schedule"]["clips_per_trajectory"]),
    )
    cursor = {"epoch": 0, "batch_index": 0}
    plan_cache: dict[str, Any] = {}
    specs = ctx.data.train.clip_spec_table()
    tokens = {arm: 0 for arm in ROUND1_ARMS}
    start_added = 0
    if resume:
        restored = {}
        resume_paths = _round1_resume_paths(ctx)
        for arm in ROUND1_ARMS:
            payload = trainers[arm].load_checkpoint(
                resume_paths[arm], map_location=ctx.device
            )
            sequential = payload.get("sequential", {})
            if sequential.get("round") != "round1" or sequential.get("arm") != arm:
                raise RuntimeError(f"Round 1 resume contract mismatch for {arm}")
            generators[arm].set_state(
                sequential["training_generator_state"].detach().cpu()
            )
            restored[arm] = sequential
            tokens[arm] = int(sequential["added_future_atom_frame_tokens"])
        first = restored["control"]
        second = restored["candidate"]
        if (
            first["cursor"] != second["cursor"]
            or tokens["control"] != tokens["candidate"]
            or first["added_successful_updates"] != second["added_successful_updates"]
        ):
            raise RuntimeError("Round 1 arms do not resume at the same exposure")
        cursor = {
            "epoch": int(first["cursor"]["epoch"]),
            "batch_index": int(first["cursor"]["batch_index"]),
        }
        start_added = int(first["added_successful_updates"])
        if any(
            trainer.successful_updates != parent_step + start_added
            for trainer in trainers.values()
        ):
            raise RuntimeError("Round 1 checkpoint step does not match its added exposure")
        for arm in ROUND1_ARMS:
            _truncate_jsonl(
                ctx.output_dir / "round1" / arm / "train_history.jsonl",
                step_key="added_successful_updates",
                maximum_step=start_added,
            )
            _truncate_jsonl(
                ctx.output_dir / "round1" / arm / "validation_history.jsonl",
                step_key="added_successful_updates",
                maximum_step=start_added,
            )
    else:
        for arm in ROUND1_ARMS:
            if (ctx.output_dir / "round1" / arm / "latest.pt").exists():
                raise RuntimeError("fresh Round 1 run refuses an existing arm checkpoint")
    compute_seconds = {arm: 0.0 for arm in ROUND1_ARMS}
    torch.cuda.reset_peak_memory_stats(ctx.device)
    started = time.perf_counter()
    schedule_hash = ""
    while trainers["control"].successful_updates < target_step:
        added = trainers["control"].successful_updates - parent_step
        if trainers["candidate"].successful_updates - parent_step != added:
            raise RuntimeError("Round 1 arms have unequal successful updates")
        indices, schedule_hash = _next_batch(sampler, cursor, plan_cache)
        history = int(ctx.cfg["schedule"]["history_order"][added % 2])
        _latent, batch, observed = _prepare_batch(
            ctx, ctx.data.train, indices, trainers["control"].adapter, history
        )
        center = _source_center(
            ctx, batch, observed, trainers["control"].adapter, history
        )
        batch_tokens = _batch_tokens(specs, indices, history)
        for arm in ROUND1_ARMS:
            _sync(ctx.device)
            arm_started = time.perf_counter()
            row = trainers[arm].train_step(
                observed, generator=generators[arm], source_center=center
            )
            _sync(ctx.device)
            compute_seconds[arm] += time.perf_counter() - arm_started
            tokens[arm] += batch_tokens
            row.update(
                {
                    "round": 1,
                    "arm": arm,
                    "ffn_norm_source": NORM_SOURCE[arm],
                    "history_frames": history,
                    "batch_tokens": batch_tokens,
                    "added_future_atom_frame_tokens": tokens[arm],
                    "added_successful_updates": trainers[arm].successful_updates - parent_step,
                    "parent_step": parent_step,
                }
            )
            _append_jsonl(
                ctx.output_dir / "round1" / arm / "train_history.jsonl", row
            )
        if not torch.equal(
            generators["control"].get_state(), generators["candidate"].get_state()
        ):
            raise RuntimeError("Round 1 tau/epsilon generators diverged")
        current_added = trainers["control"].successful_updates - parent_step
        if current_added % int(ctx.cfg["schedule"]["validation_interval"]) == 0:
            for arm in ROUND1_ARMS:
                validation = _validation_rf(
                    ctx, trainers[arm], step=trainers[arm].successful_updates
                )
                validation["arm"] = arm
                validation["added_successful_updates"] = current_added
                _append_jsonl(
                    ctx.output_dir / "round1" / arm / "validation_history.jsonl",
                    validation,
                )
        if current_added % int(ctx.cfg["schedule"]["checkpoint_interval"]) == 0:
            for arm in ROUND1_ARMS:
                _save_checkpoint(
                    ctx.output_dir / "round1" / arm,
                    trainers[arm],
                    round_name="round1",
                    arm=arm,
                    parent_path=str(parent_path),
                    parent_sha256=parent_sha,
                    parent_model_hash=parent_model_hash,
                    cursor=cursor,
                    generator=generators[arm],
                    added_tokens=tokens[arm],
                    added_updates=current_added,
                    schedule_hash=schedule_hash,
                    unique_variable={
                        "ffn_norm_source": NORM_SOURCE[arm],
                        "common_extension": common_extension,
                    },
                    final=False,
                )
    elapsed = time.perf_counter() - started
    expected_tokens = int(
        budget[
            "selected_round1_extended_total_future_atom_frame_tokens_per_arm"
            if common_extension
            else "selected_round1_future_atom_frame_tokens_per_arm"
        ]
    )
    if tokens["control"] != tokens["candidate"] or tokens["control"] != expected_tokens:
        raise RuntimeError("Round 1 token exposure differs from the frozen budget")
    checkpoints = {}
    for arm in ROUND1_ARMS:
        checkpoints[arm] = _save_checkpoint(
            ctx.output_dir / "round1" / arm,
            trainers[arm],
            round_name="round1",
            arm=arm,
            parent_path=str(parent_path),
            parent_sha256=parent_sha,
            parent_model_hash=parent_model_hash,
            cursor=cursor,
            generator=generators[arm],
            added_tokens=tokens[arm],
            added_updates=delta,
            schedule_hash=schedule_hash,
            unique_variable={
                "ffn_norm_source": NORM_SOURCE[arm],
                "common_extension": common_extension,
            },
            final=True,
        )
    prior_gpu_hours = float(prior_summary.get("gpu_hours", 0.0))
    prior_wall_seconds = float(prior_summary.get("wall_seconds", 0.0))
    prior_compute_seconds = prior_summary.get("arm_compute_seconds", {})
    cumulative_compute_seconds = {
        arm: float(prior_compute_seconds.get(arm, 0.0)) + compute_seconds[arm]
        for arm in ROUND1_ARMS
    }
    result = {
        "schema": f"{SCHEMA}.round1.training.v1",
        "status": "PASS",
        "parent_checkpoint": str(parent_path),
        "parent_checkpoint_sha256": parent_sha,
        "parent_model_state_hash": parent_model_hash,
        "initialization_hash": parent_model_hash,
        "unique_variable": "ffn_norm_source: post_adaln versus pre_adaln",
        "config_hash": _canonical_hash(ctx.cfg),
        "data_hash": ctx.data_hash,
        "codec_state_hash": ctx.codec.codec_state_hash,
        "statistics_hash": ctx.statistics.hash,
        "code": _code_provenance(),
        "parent_successful_updates": parent_step,
        "base_budget_successful_updates_per_arm": base_delta,
        "common_extension_applied": common_extension,
        "extension_successful_updates_per_arm": (extension_delta if common_extension else 0),
        "added_successful_updates_per_arm": delta,
        "successful_optimizer_updates": {
            arm: trainers[arm].successful_updates for arm in ROUND1_ARMS
        },
        "added_future_atom_frame_tokens_per_arm": expected_tokens,
        "updates_this_invocation_per_arm": delta - start_added,
        "wall_seconds_this_invocation": elapsed,
        "wall_seconds": prior_wall_seconds + elapsed,
        "gpu_hours_this_invocation": elapsed / 3600.0,
        "gpu_hours": prior_gpu_hours + elapsed / 3600.0,
        "future_atom_frame_tokens_per_second_this_invocation": (
            2.0
            * max(
                expected_tokens
                - int(prior_summary.get("added_future_atom_frame_tokens_per_arm", 0)),
                0,
            )
            / max(elapsed, 1.0e-12)
        ),
        "future_atom_frame_tokens_per_second": (
            2.0 * expected_tokens / max(prior_wall_seconds + elapsed, 1.0e-12)
        ),
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated(ctx.device)),
        "arm_compute_seconds_this_invocation": compute_seconds,
        "arm_compute_seconds": cumulative_compute_seconds,
        "checkpoints": {
            arm: {
                "path": str(path),
                "sha256": sha256_file(path),
                "model_state_hash": module_state_hash(trainers[arm].model),
                "adapter_state_hash": module_state_hash(trainers[arm].adapter),
                "optimizer_state_owned": trainers[arm].optimizer is not trainers[ROUND1_ARMS[1 - ROUND1_ARMS.index(arm)]].optimizer,
                "scaler_owned": trainers[arm].scaler is not trainers[ROUND1_ARMS[1 - ROUND1_ARMS.index(arm)]].scaler,
            }
            for arm, path in checkpoints.items()
        },
        "resume_start_added_updates": start_added,
        "test_payload_opened": False,
    }
    _write_json(ctx.output_dir / "round1" / "train_summary.json", result)
    return result


def _smoke_round1(ctx: SequentialContext) -> dict[str, Any]:
    _require_pass_summary(
        ctx.output_dir / "baseline" / "evaluation_quick" / "baseline" / "summary.json",
        "clean baseline compact generation evaluation",
    )
    parent_path = _baseline_checkpoint(ctx)
    parent = torch.load(parent_path, map_location="cpu", weights_only=False)
    state, parent_hash = _state_from_payload(
        ctx.cfg, parent, ctx.device, ffn_norm_source="post_adaln"
    )
    selection = json.loads((ctx.output_dir / "selection.json").read_text(encoding="utf-8"))
    smoke_updates = len(selection["smoke"])
    trainers = {
        arm: _new_trainer(
            ctx,
            ffn_norm_source=NORM_SOURCE[arm],
            max_steps=int(parent["step"]) + smoke_updates,
            initial_state=state,
            initial_hash=parent_hash,
            arm=arm,
            round_name="round1_smoke",
        )
        for arm in ROUND1_ARMS
    }
    for trainer in trainers.values():
        _restore_parent(
            trainer,
            parent,
            expected_codec_hash=ctx.codec.codec_state_hash,
            expected_statistics_hash=ctx.statistics.hash,
        )
    assert_independent_trainers(trainers["control"], trainers["candidate"])
    smoke_dataset = ClipMMapDataset(Path(selection["smoke_root"]) / "train")
    rows = []
    try:
        generators = {
            arm: torch.Generator(device=ctx.device).manual_seed(2026091401)
            for arm in ROUND1_ARMS
        }
        history_order = tuple(
            int(value) for value in ctx.cfg["schedule"]["history_order"]
        )
        for smoke_index, item in enumerate(selection["smoke"]):
            replica = str(item["replica"])
            indices = (int(item["dataset_index"]),)
            training_history = history_order[smoke_index % len(history_order)]
            _latent, batch, observed = _prepare_batch(
                ctx,
                smoke_dataset,
                indices,
                trainers["control"].adapter,
                training_history,
            )
            center = _source_center(
                ctx,
                batch,
                observed,
                trainers["control"].adapter,
                training_history,
            )
            step_rows = {}
            for arm in ROUND1_ARMS:
                step_rows[arm] = trainers[arm].train_step(
                    observed,
                    generator=generators[arm],
                    source_center=center,
                )
            if not torch.equal(
                generators["control"].get_state(), generators["candidate"].get_state()
            ):
                raise RuntimeError("Round 1 smoke tau/epsilon generators diverged")
            for history in (4, 8):
                _latent, batch, observed = _prepare_batch(
                    ctx,
                    smoke_dataset,
                    indices,
                    trainers["control"].adapter,
                    history,
                )
                center = _source_center(
                    ctx, batch, observed, trainers["control"].adapter, history
                )
                normalized = ctx.statistics_device.normalize(observed)
                noise, noise_meta = make_fixed_noise(
                    normalized,
                    seed=generation_seed(
                        20260914, "|".join(batch.sample_id), history, 0
                    ),
                )
                for arm in ROUND1_ARMS:
                    generated, generation = generate_fixed_noise_latent(
                        trainers[arm].model,
                        trainers[arm].adapter,
                        observed,
                        ctx.statistics_device,
                        noise=noise,
                        steps=8,
                        source_center=center,
                        source_mode="conditional",
                    )
                    decoded = ctx.codec.model.decode(generated).x_hat.float()
                    rows.append(
                        {
                            "replica": replica,
                            "history_frames": history,
                            "training_history_frames": training_history,
                            "arm": arm,
                            "sample_ids": list(batch.sample_id),
                            "loss": step_rows[arm]["loss"],
                            "finite": bool(torch.isfinite(decoded).all()),
                            "observed_clamp_exact": bool(generation["observed_clamp_exact"]),
                            "noise_sha256": noise_meta["noise_sha256"],
                            "decoded_shape": list(decoded.shape),
                        }
                    )
        passed = bool(rows) and all(
            row["finite"] and row["observed_clamp_exact"] for row in rows
        ) and all(
            trainer.successful_updates == int(parent["step"]) + smoke_updates
            for trainer in trainers.values()
        )
        result = {
            "schema": f"{SCHEMA}.round1.smoke.v1",
            "status": "PASS" if passed else "FAIL",
            "systems": list(ctx.cfg["evaluation"]["smoke"]["systems"]),
            "trajectory_count": len(selection["smoke"]),
            "optimizer_updates_per_arm": smoke_updates,
            "batching": "serial_single_clip",
            "rows": rows,
            "ranking_use": False,
            "device": _cuda_info(ctx.device),
            "test_payload_opened": False,
        }
        _write_json(ctx.output_dir / "round1" / "smoke.json", result)
        return result
    finally:
        smoke_dataset.close()


def _amplitude_error(metrics: Mapping[str, Any]) -> float | None:
    rmsf = metrics.get("future", {}).get("rmsf", {})
    prediction = rmsf.get("prediction_atom_values_angstrom")
    target = rmsf.get("target_atom_values_angstrom")
    if not isinstance(prediction, list) or not isinstance(target, list) or len(prediction) != len(target) or not prediction:
        return None
    return sum(abs(float(first) - float(second)) for first, second in zip(prediction, target)) / len(prediction)


def _save_prediction(path: Path, coordinates: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        coordinates=coordinates.detach().cpu().to(torch.float16).numpy(),
    )
    os.replace(temporary, path)


def _evaluation_items(
    ctx: SequentialContext, scope: str
) -> list[tuple[str, Any, Mapping[str, Any]]]:
    selection = json.loads((ctx.output_dir / "selection.json").read_text(encoding="utf-8"))
    if scope == "quick":
        return [
            *(("train", ctx.data.train, item) for item in selection["quick_train"]),
            *(("valid", ctx.data.valid, item) for item in selection["quick_valid"]),
        ]
    if scope == "final":
        return [("valid", ctx.data.valid, item) for item in selection["final_valid"]]
    raise ValueError("evaluation scope must be quick or final")


def _evaluate_checkpoint(
    ctx: SequentialContext,
    *,
    checkpoint: Path,
    label: str,
    round_name: str,
    scope: str,
    include_references: bool = False,
) -> dict[str, Any]:
    loaded = load_sequential_checkpoint(
        checkpoint,
        device=ctx.device,
        expected_data_hash=ctx.data_hash,
        expected_codec_hash=ctx.codec.codec_state_hash,
        expected_statistics_hash=ctx.statistics.hash,
    )
    output = ctx.output_dir / round_name / f"evaluation_{scope}" / label
    rows = []
    diversity_rows = []
    reference_root = ctx.output_dir / "baseline" / f"references_{scope}"
    reference_summary_path = reference_root / "summary.json"
    existing_references = None
    reference_rows: list[dict[str, Any]] | None = None
    if include_references:
        if reference_summary_path.is_file():
            existing_references = json.loads(
                reference_summary_path.read_text(encoding="utf-8")
            )
            if (
                existing_references.get("status") != "PASS"
                or existing_references.get("scope") != scope
                or existing_references.get("data_hash") != ctx.data_hash
                or existing_references.get("codec_state_hash")
                != ctx.codec.codec_state_hash
                or existing_references.get("selection_sha256")
                != sha256_file(ctx.output_dir / "selection.json")
                or existing_references.get("metric_implementation_sha256")
                != sha256_file(PROJECT_ROOT / "evaluation/dit_reassessment.py")
            ):
                raise RuntimeError("existing fixed reference controls do not match this run")
            reference_rows_path = Path(str(existing_references.get("rows", "")))
            if (
                not reference_rows_path.is_file()
                or existing_references.get("rows_sha256")
                != sha256_file(reference_rows_path)
            ):
                raise RuntimeError("existing fixed reference rows are missing or changed")
        else:
            reference_rows = []
    draws = (
        tuple(int(value) for value in ctx.cfg["evaluation"]["quick_draws"])
        if scope == "quick"
        else tuple(int(value) for value in ctx.cfg["evaluation"]["final_draws"])
    )
    prediction_root = output / "predictions"
    try:
        for split, dataset, item in _evaluation_items(ctx, scope):
            sample_id = str(item["sample_id"])
            oracle_coordinates = None
            for history in (4, 8):
                latent, batch, observed = _prepare_batch(
                    ctx,
                    dataset,
                    (int(item["dataset_index"]),),
                    loaded.adapter,
                    history,
                )
                center = _source_center(
                    ctx, batch, observed, loaded.adapter, history
                )
                normalized = ctx.statistics_device.normalize(observed)
                if reference_rows is not None:
                    if oracle_coordinates is None:
                        oracle_coordinates = ctx.codec.model.decode(latent).x_hat.float()
                    controls = {
                        "oracle_decode": oracle_coordinates,
                        "repeat_last": repeat_last_coordinate_template(batch, history),
                    }
                    for control_label, control_coordinates in controls.items():
                        control_metrics = reassessment_trajectory_metrics(
                            control_coordinates, batch.x.float(), batch, history
                        )
                        control_metrics["future"]["amplitude_error_angstrom"] = (
                            _amplitude_error(control_metrics)
                        )
                        reference_rows.append(
                            {
                                "schema": f"{SCHEMA}.reference_row.v1",
                                "label": control_label,
                                "split": split,
                                "sample_id": sample_id,
                                "system": str(item["system"]),
                                "replica": str(item["replica"]),
                                "window": int(item["window"]),
                                "history_frames": history,
                                "draw": 0,
                                "metrics": control_metrics,
                                "test_payload_opened": False,
                            }
                        )
                coordinates = []
                for draw in draws:
                    seed = generation_seed(
                        int(ctx.cfg["evaluation"]["generation_seed"]),
                        sample_id,
                        history,
                        draw,
                    )
                    noise, noise_meta = make_fixed_noise(normalized, seed=seed)
                    _sync(ctx.device)
                    started = time.perf_counter()
                    generated, generation = generate_fixed_noise_latent(
                        loaded.model,
                        loaded.adapter,
                        observed,
                        ctx.statistics_device,
                        noise=noise,
                        steps=int(ctx.cfg["evaluation"]["euler_steps"]),
                        source_center=center,
                        source_mode="conditional",
                    )
                    decoded = ctx.codec.model.decode(generated).x_hat.float()
                    _sync(ctx.device)
                    elapsed = time.perf_counter() - started
                    metrics = reassessment_trajectory_metrics(
                        decoded, batch.x.float(), batch, history
                    )
                    metrics["future"]["amplitude_error_angstrom"] = _amplitude_error(metrics)
                    name = (
                        f"{split}__{sample_id}__H{history}__draw{draw}.npz"
                        .replace("/", "_")
                    )
                    prediction_path = prediction_root / name
                    _save_prediction(prediction_path, decoded)
                    coordinates.append(decoded.detach())
                    rows.append(
                        {
                            "schema": f"{SCHEMA}.generation_row.v1",
                            "round": round_name,
                            "label": label,
                            "split": split,
                            "sample_id": sample_id,
                            "system": str(item["system"]),
                            "replica": str(item["replica"]),
                            "window": int(item["window"]),
                            "history_frames": history,
                            "steps": int(ctx.cfg["evaluation"]["euler_steps"]),
                            "draw": draw,
                            "seed": seed,
                            "noise_sha256": noise_meta["noise_sha256"],
                            "checkpoint": str(checkpoint),
                            "checkpoint_sha256": loaded.checkpoint_sha256,
                            "model_state_hash": loaded.model_state_hash,
                            "adapter_is_model_adapter": loaded.adapter is loaded.model.adapter,
                            "generation": {
                                **generation,
                                "wall_seconds_including_decode": elapsed,
                            },
                            "metrics": metrics,
                            "prediction_coordinates": str(prediction_path),
                            "test_payload_opened": False,
                        }
                    )
                if len(coordinates) > 1:
                    diversity_rows.append(
                        {
                            "split": split,
                            "sample_id": sample_id,
                            "system": str(item["system"]),
                            "history_frames": history,
                            "diversity": future_diversity(
                                torch.stack(coordinates), batch, history
                            ),
                        }
                    )
                del latent, batch, observed, coordinates
            del oracle_coordinates
        if reference_rows is not None:
            reference_root.mkdir(parents=True, exist_ok=True)
            reference_rows_path = reference_root / "rows.jsonl"
            reference_rows_path.write_text(
                "".join(
                    json.dumps(_safe(row), sort_keys=True) + "\n"
                    for row in reference_rows
                ),
                encoding="utf-8",
            )
            reference_aggregates = {}
            for control_label in ("oracle_decode", "repeat_last"):
                reference_aggregates[control_label] = {}
                for split in sorted({row["split"] for row in reference_rows}):
                    reference_aggregates[control_label][split] = {}
                    for history in (4, 8):
                        selected = [
                            row
                            for row in reference_rows
                            if row["label"] == control_label
                            and row["split"] == split
                            and row["history_frames"] == history
                        ]
                        reference_aggregates[control_label][split][f"H{history}"] = (
                            aggregate_generated_rows(selected)
                        )
            existing_references = {
                "schema": f"{SCHEMA}.references.v1",
                "status": "PASS",
                "scope": scope,
                "data_hash": ctx.data_hash,
                "codec_state_hash": ctx.codec.codec_state_hash,
                "selection_sha256": sha256_file(ctx.output_dir / "selection.json"),
                "metric_implementation_sha256": sha256_file(
                    PROJECT_ROOT / "evaluation/dit_reassessment.py"
                ),
                "oracle_decode_reused_across_histories_per_clip": True,
                "rows": str(reference_rows_path),
                "rows_sha256": sha256_file(reference_rows_path),
                "row_count": len(reference_rows),
                "aggregates": reference_aggregates,
                "test_payload_opened": False,
            }
            _write_json(reference_summary_path, existing_references)
        rows_path = output / "generation_rows.jsonl"
        rows_path.parent.mkdir(parents=True, exist_ok=True)
        rows_path.write_text(
            "".join(json.dumps(_safe(row), sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
        _write_json(output / "diversity.json", {"rows": diversity_rows})
        aggregates = {}
        for split in sorted({row["split"] for row in rows}):
            aggregates[split] = {}
            for history in (4, 8):
                selected = [
                    row
                    for row in rows
                    if row["split"] == split and row["history_frames"] == history
                ]
                aggregates[split][f"H{history}"] = aggregate_generated_rows(selected)
        result = {
            "schema": f"{SCHEMA}.evaluation.v1",
            "status": "PASS",
            "round": round_name,
            "label": label,
            "scope": scope,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": loaded.checkpoint_sha256,
            "model_state_hash": loaded.model_state_hash,
            "adapter_identity": loaded.adapter is loaded.model.adapter,
            "adapter_state_hash": loaded.adapter_state_hash,
            "parameter_count": sum(parameter.numel() for parameter in loaded.model.parameters()),
            "row_count": len(rows),
            "aggregates": aggregates,
            "diversity_rows": len(diversity_rows),
            "generation_latency_seconds": {
                "mean": float(np.mean([
                    row["generation"]["wall_seconds_including_decode"] for row in rows
                ])),
                "p90": float(np.quantile([
                    row["generation"]["wall_seconds_including_decode"] for row in rows
                ], 0.9)),
                "includes_source": False,
                "includes_euler_and_decode": True,
            },
            "rows": str(rows_path),
            "references": existing_references,
            "test_payload_opened": False,
        }
        _write_json(output / "summary.json", result)
        return result
    finally:
        del loaded
        torch.cuda.empty_cache()


def _evaluate_baseline(ctx: SequentialContext, scope: str) -> dict[str, Any]:
    return _evaluate_checkpoint(
        ctx,
        checkpoint=_baseline_checkpoint(ctx),
        label="baseline",
        round_name="baseline",
        scope=scope,
        include_references=True,
    )


def _round1_checkpoints(ctx: SequentialContext) -> dict[str, Path]:
    summary_path = ctx.output_dir / "round1" / "train_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError("Round 1 training summary is missing")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "PASS":
        raise RuntimeError("Round 1 training did not pass")
    result = {}
    for arm in ROUND1_ARMS:
        path = Path(summary["checkpoints"][arm]["path"])
        if sha256_file(path) != summary["checkpoints"][arm]["sha256"]:
            raise RuntimeError(f"Round 1 {arm} checkpoint changed")
        result[arm] = path
    return result


def _evaluate_round1(ctx: SequentialContext, scope: str) -> dict[str, Any]:
    results = {}
    for arm, checkpoint in _round1_checkpoints(ctx).items():
        results[arm] = _evaluate_checkpoint(
            ctx,
            checkpoint=checkpoint,
            label=arm,
            round_name="round1",
            scope=scope,
        )
    return {
        "schema": f"{SCHEMA}.round1.evaluation_pair.v1",
        "scope": scope,
        "arms": results,
        "sequential_loading": True,
        "test_payload_opened": False,
    }


def _metric(summary: Mapping[str, Any], history: int, key: str) -> float:
    value = summary["aggregates"]["valid"][f"H{history}"]["system_equal"].get(key)
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise RuntimeError(f"missing numeric valid H{history} metric {key}")
    return float(value)


def _direction_count(
    control: Mapping[str, Any],
    candidate: Mapping[str, Any],
    history: int,
    key: str,
) -> int:
    first = {
        row["system"]: row["future"].get(key)
        for row in control["aggregates"]["valid"][f"H{history}"]["system_rows"]
    }
    second = {
        row["system"]: row["future"].get(key)
        for row in candidate["aggregates"]["valid"][f"H{history}"]["system_rows"]
    }
    if set(first) != set(second):
        raise RuntimeError("paired evaluation systems differ")
    return sum(
        isinstance(first[system], (int, float))
        and isinstance(second[system], (int, float))
        and float(second[system]) < float(first[system])
        for system in first
    )


def _round1_report_and_plots(
    ctx: Any,
    decision: Mapping[str, Any],
    evidence: Mapping[str, Mapping[str, Any]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = ctx.output_dir / "round1" / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, history in zip(axes, (4, 8)):
        control_rows = {
            row["system"]: row["future"].get("amplitude_error_angstrom")
            for row in evidence["control"]["aggregates"]["valid"][f"H{history}"]["system_rows"]
        }
        candidate_rows = {
            row["system"]: row["future"].get("amplitude_error_angstrom")
            for row in evidence["candidate"]["aggregates"]["valid"][f"H{history}"]["system_rows"]
        }
        systems = sorted(set(control_rows) & set(candidate_rows))
        pairs = [
            (float(control_rows[system]), float(candidate_rows[system]))
            for system in systems
            if isinstance(control_rows[system], (int, float))
            and isinstance(candidate_rows[system], (int, float))
        ]
        if pairs:
            first, second = zip(*pairs)
            bound = max((*first, *second, 1.0e-6))
            axis.scatter(first, second)
            axis.plot((0.0, bound), (0.0, bound), linestyle="--", color="black")
        axis.set(title=f"H{history}", xlabel="post-AdaLN E_amp (A)", ylabel="pre-AdaLN E_amp (A)")
    figure.suptitle("Round 1 per-system amplitude error")
    figure.tight_layout()
    figure.savefig(plot_dir / "amplitude_pair.png", dpi=160)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for axis, history in zip(axes, (4, 8)):
        for arm in ROUND1_ARMS:
            rows = [
                json.loads(line)
                for line in Path(evidence[arm]["rows"]).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            by_frame: dict[int, list[float]] = {}
            for row in rows:
                if row["split"] != "valid" or int(row["history_frames"]) != history:
                    continue
                for frame in row["metrics"]["frame_curve"]:
                    value = frame.get("bond_rmse")
                    if frame.get("future") and isinstance(value, (int, float)):
                        by_frame.setdefault(int(frame["frame"]), []).append(float(value))
            frames = sorted(by_frame)
            values = [sum(by_frame[frame]) / len(by_frame[frame]) for frame in frames]
            axis.plot(frames, values, marker="o", label=arm)
        axis.set(title=f"H{history}", xlabel="frame", ylabel="bond RMSE (A)")
        axis.legend()
    figure.suptitle("Round 1 valid future bond curve")
    figure.tight_layout()
    figure.savefig(plot_dir / "future_bond_curve.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 4))
    for arm in ROUND1_ARMS:
        path = ctx.output_dir / "round1" / arm / "train_history.jsonl"
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        axis.plot(
            [int(row["added_successful_updates"]) for row in rows],
            [float(row["loss"]) for row in rows],
            label=arm,
            alpha=0.8,
        )
    axis.set(xlabel="added successful update", ylabel="RF loss", title="Round 1 matched continuation")
    axis.legend()
    figure.tight_layout()
    figure.savefig(plot_dir / "training_loss.png", dpi=160)
    plt.close(figure)

    lines = [
        "# Round 1: pre-AdaLN vector magnitude",
        "",
        f"- Decision: `{decision['status']}`",
        f"- Evidence scope: `{decision['evaluation_scope']}`",
        f"- Selected arm: `{decision['selected_arm']}`",
        f"- Unique variable: {decision['unique_variable']}",
        f"- Added successful updates per arm: {decision['successful_updates_per_arm']}",
        f"- Future atom-frame tokens per arm: {decision['future_atom_frame_tokens_per_arm']}",
        f"- GPU-hours: {decision['gpu_hours']:.6f}",
        "",
        "| H | E_amp relative improvement | systems improved | bond relative change | contact F1 change |",
        "|---|---:|---:|---:|---:|",
    ]
    for history in (4, 8):
        key = f"H{history}"
        guard = decision["guardrails"][key]
        lines.append(
            f"| {key} | {decision['relative_improvement'][key]:.6f} | "
            f"{decision['systems_same_direction'][key]} | "
            f"{guard['bond_relative_change']:.6f} | {guard['contact_f1_change']:.6f} |"
        )
    lines.extend(("", decision["reason"], "", f"Remaining risk: {decision['remaining_risk']}", ""))
    (ctx.output_dir / "round1" / "report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def _decide_round1(ctx: Any) -> dict[str, Any]:
    checkpoints = _round1_checkpoints(ctx)

    def current_evidence(scope: str) -> dict[str, dict[str, Any]]:
        root = ctx.output_dir / "round1" / f"evaluation_{scope}"
        result = {}
        for arm in ROUND1_ARMS:
            path = root / arm / "summary.json"
            if not path.is_file():
                raise FileNotFoundError(f"Round 1 {scope} evaluation is missing for {arm}")
            summary = json.loads(path.read_text(encoding="utf-8"))
            if summary.get("status") != "PASS":
                raise RuntimeError(f"Round 1 {scope} evaluation did not pass for {arm}")
            if summary.get("checkpoint_sha256") != sha256_file(checkpoints[arm]):
                raise RuntimeError(
                    f"Round 1 {scope} evaluation is stale for {arm}; re-evaluate current checkpoint"
                )
            result[arm] = summary
        return result

    quick = current_evidence("quick")
    amplitude_key = "amplitude_error_angstrom"
    quick_improvement = {}
    quick_direction = {}
    for history in (4, 8):
        control = _metric(quick["control"], history, amplitude_key)
        candidate = _metric(quick["candidate"], history, amplitude_key)
        quick_improvement[f"H{history}"] = (control - candidate) / max(abs(control), 1.0e-12)
        quick_direction[f"H{history}"] = _direction_count(
            quick["control"], quick["candidate"], history, amplitude_key
        )
    threshold = float(ctx.cfg["decision"]["primary_relative_improvement"])
    minimum_systems = int(ctx.cfg["decision"]["minimum_systems_same_direction"])
    quick_promising = (
        sum(quick_improvement.values()) / 2.0 >= threshold
        and min(quick_improvement.values()) > -float(ctx.cfg["decision"]["opposite_trend_tolerance"])
        and min(quick_direction.values()) >= minimum_systems
    )
    final_root = ctx.output_dir / "round1" / "evaluation_final"
    if quick_promising and not all(
        (final_root / arm / "summary.json").is_file() for arm in ROUND1_ARMS
    ):
        result = {
            "schema": f"{SCHEMA}.decision.v1",
            "round": 1,
            "status": "NEEDS_FINAL_EVALUATION",
            "parent_checkpoint": str(_baseline_checkpoint(ctx)),
            "unique_variable": "ffn_norm_source",
            "quick_primary_improvement": quick_improvement,
            "quick_systems_same_direction": quick_direction,
            "next_command": "evaluate-round1 --scope final, then decide-round1",
            "test_payload_opened": False,
        }
        _write_json(ctx.output_dir / "round1" / "decision.json", result)
        return result
    evidence = quick
    scope = "quick"
    if quick_promising:
        evidence = current_evidence("final")
        scope = "final"
    improvement = {}
    direction = {}
    guardrails = {}
    for history in (4, 8):
        control_amp = _metric(evidence["control"], history, amplitude_key)
        candidate_amp = _metric(evidence["candidate"], history, amplitude_key)
        improvement[f"H{history}"] = (
            control_amp - candidate_amp
        ) / max(abs(control_amp), 1.0e-12)
        direction[f"H{history}"] = _direction_count(
            evidence["control"], evidence["candidate"], history, amplitude_key
        )
        control_bond = _metric(evidence["control"], history, "bond_rmse")
        candidate_bond = _metric(evidence["candidate"], history, "bond_rmse")
        control_contact = _metric(evidence["control"], history, "contact_f1")
        candidate_contact = _metric(evidence["candidate"], history, "contact_f1")
        guardrails[f"H{history}"] = {
            "bond_relative_change": (
                candidate_bond - control_bond
            ) / max(abs(control_bond), 1.0e-12),
            "contact_f1_change": candidate_contact - control_contact,
        }
    primary_pass = (
        sum(improvement.values()) / 2.0 >= threshold
        and min(improvement.values()) > -float(ctx.cfg["decision"]["opposite_trend_tolerance"])
        and min(direction.values()) >= minimum_systems
    )
    guard_pass = all(
        row["bond_relative_change"] <= float(ctx.cfg["decision"]["bond_relative_worsening"])
        and row["contact_f1_change"] >= -float(ctx.cfg["decision"]["contact_f1_absolute_drop"])
        for row in guardrails.values()
    )
    train = json.loads(
        (ctx.output_dir / "round1" / "train_summary.json").read_text(encoding="utf-8")
    )
    extension_applied = bool(train.get("common_extension_applied", False))
    if primary_pass and guard_pass:
        status = "KEEP"
        selected_arm = "candidate"
        reason = "amplitude error passed the paired threshold and geometry guardrails"
    elif primary_pass:
        status = "TRADEOFF"
        selected_arm = "control"
        reason = "amplitude improved but a geometry guardrail failed"
    elif any(value > 0 for value in improvement.values()) and not extension_applied:
        status = "NEEDS_COMMON_EXTENSION"
        selected_arm = None
        reason = "sub-threshold or heterogeneous pilot evidence requires one matched extension"
    elif any(value > 0 for value in improvement.values()):
        status = "INCONCLUSIVE"
        selected_arm = "control"
        reason = "one matched extension remained sub-threshold or heterogeneous; retain the simpler control"
    else:
        status = "REJECT"
        selected_arm = "control"
        reason = "pre-AdaLN amplitude input did not improve the paired primary metric"
    result = {
        "schema": f"{SCHEMA}.decision.v1",
        "round": 1,
        "status": status,
        "parent_checkpoint": train["parent_checkpoint"],
        "parent_checkpoint_sha256": train["parent_checkpoint_sha256"],
        "unique_variable": "ffn_norm_source: post_adaln versus pre_adaln",
        "evaluation_scope": scope,
        "primary_metric": "system-equal per-atom future RMSF absolute error",
        "relative_improvement": improvement,
        "systems_same_direction": direction,
        "guardrails": guardrails,
        "selected_arm": selected_arm,
        "selected_checkpoint": (
            None if selected_arm is None else str(checkpoints[selected_arm])
        ),
        "selected_checkpoint_sha256": (
            None if selected_arm is None else sha256_file(checkpoints[selected_arm])
        ),
        "initialization_hash": train["initialization_hash"],
        "config_hash": train["config_hash"],
        "data_hash": train["data_hash"],
        "codec_state_hash": train["codec_state_hash"],
        "statistics_hash": train["statistics_hash"],
        "code": train["code"],
        "checkpoints": train["checkpoints"],
        "successful_optimizer_updates": train["successful_optimizer_updates"],
        "system_metrics": {
            f"H{history}": {
                arm: [
                    {
                        "system": row["system"],
                        "amplitude_error_angstrom": row["future"].get(
                            "amplitude_error_angstrom"
                        ),
                        "bond_rmse_angstrom": row["future"].get("bond_rmse"),
                        "contact_f1": row["future"].get("contact_f1"),
                    }
                    for row in evidence[arm]["aggregates"]["valid"][f"H{history}"]["system_rows"]
                ]
                for arm in ROUND1_ARMS
            }
            for history in (4, 8)
        },
        "successful_updates_per_arm": train["added_successful_updates_per_arm"],
        "future_atom_frame_tokens_per_arm": train["added_future_atom_frame_tokens_per_arm"],
        "gpu_hours": train["gpu_hours"],
        "reason": reason,
        "remaining_risk": "one training seed; sequential continuation gain is not a scaling-law result",
        "next_command": (
            "python scripts/run_dit_architecture_sequential_v1.py --stage train-round1 --resume; then evaluate-round1 --scope quick and decide-round1"
            if status == "NEEDS_COMMON_EXTENSION"
            else None
        ),
        "test_payload_opened": False,
    }
    _write_json(ctx.output_dir / "round1" / "decision.json", result)
    if status in ("KEEP", "REJECT", "TRADEOFF", "INCONCLUSIVE"):
        _round1_report_and_plots(ctx, result, evidence)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config/dit_architecture_sequential_v1.yaml",
    )
    parser.add_argument(
        "--stage",
        required=True,
        choices=(
            "preflight",
            "profile",
            "baseline",
            "smoke-round1",
            "train-round1",
            "evaluate-baseline",
            "evaluate-round1",
            "decide-round1",
        ),
    )
    parser.add_argument("--scope", choices=("quick", "final"), default="quick")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    cfg = _load_config(args.config.resolve())
    if args.run_id:
        cfg["run_id"] = str(args.run_id)
    output_dir = _resolve(cfg["output_root"]) / str(cfg["run_id"])
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.stage == "preflight":
        result = _preflight(cfg, output_dir)
        print(json.dumps(_safe(result), indent=2, sort_keys=True))
        return
    if args.stage == "decide-round1":
        metadata_context = argparse.Namespace(cfg=cfg, output_dir=output_dir)
        result = _decide_round1(metadata_context)
        print(json.dumps(_safe(result), indent=2, sort_keys=True))
        return
    device = torch.device(args.device)
    ctx = _load_context(cfg, output_dir, device)
    try:
        if args.stage == "profile":
            result = _profile(ctx)
        elif args.stage == "baseline":
            result = _train_baseline(ctx, resume=bool(args.resume))
        elif args.stage == "smoke-round1":
            result = _smoke_round1(ctx)
        elif args.stage == "train-round1":
            result = _train_round1(ctx, resume=bool(args.resume))
        elif args.stage == "evaluate-baseline":
            result = _evaluate_baseline(ctx, args.scope)
        elif args.stage == "evaluate-round1":
            result = _evaluate_round1(ctx, args.scope)
        elif args.stage == "decide-round1":
            raise AssertionError("decide-round1 is handled before CUDA context creation")
        else:
            raise AssertionError(args.stage)
        print(json.dumps(_safe(result), indent=2, sort_keys=True))
    finally:
        ctx.close()


if __name__ == "__main__":
    main()
