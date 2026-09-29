"""Run the bounded real-artifact CUDA smoke or static decoder engineering pilot."""

from __future__ import annotations

import argparse
from dataclasses import replace
import gc
import json
from pathlib import Path
import re
import subprocess
import time
from typing import Any, Mapping, Sequence

import torch

from molvid.checkpoints import load_codec_artifact
from molvid.cli.train_frame_joint import (
    _load_statistics,
    _open_data,
    _validate_frame_joint_config,
)
from molvid.config import load_config
from molvid.data.batch import collate_clip_records
from molvid.data.sampling import get_clip_specs
from molvid.data.static import trajectory_record_to_static
from molvid.flow.objective import FrameRectifiedFlowObjective
from molvid.losses.sampled_physical import PhysicalAuxiliaryConfig
from molvid.model import FrameJointModel
from molvid.runtime import atomic_write_json, configure_device, seed_all, sha256_file
from molvid.training.batches import prepare_frame_joint_batch
from molvid.training.dit import module_state_hash
from molvid.training.joint import (
    FrameJointTrainer,
    JointLossConfig,
    frame_joint_loss,
)
from molvid.training.static_decoder import (
    StaticDecoderAuxiliaryTrainer,
    StaticDecoderLossConfig,
    StaticPerturbationConfig,
    parameter_gradient_norm,
    prepare_static_decoder_batch,
    resolve_static_gamma,
    static_clash_loss_function,
    static_decoder_loss,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--parent-sha256", required=True)
    parser.add_argument("--auxiliary-recipe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "pilot"), required=True)
    parser.add_argument("--updates", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260929)
    return parser.parse_args()


def _commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        text=True,
        cwd=Path(__file__).resolve().parents[1],
    ).strip()


def _system_replica(record: Mapping[str, Any]) -> tuple[str, str]:
    system = str(record.get("system_id", ""))
    replica = str(record.get("replica", ""))
    if system and replica:
        return system, replica
    match = re.match(r"^(atlas_.+)_R([123])_", str(record.get("sample_id", "")))
    if match is None:
        raise ValueError("train record lacks auditable system/replica provenance")
    return match.group(1), f"R{match.group(2)}"


def _select_three_system_grid(dataset: Any) -> tuple[list[Mapping[str, Any]], list[dict[str, Any]]]:
    # Select complete systems by their largest replica atom count.  Gradient
    # calibration exercises the full dynamic objective, so choosing the
    # cheapest auditable grid keeps this nominally static probe a low-memory
    # neighbour on a shared GPU without changing any scientific contract.
    candidates: dict[str, dict[str, tuple[int, int]]] = {}
    for item in get_clip_specs(dataset):
        match = re.match(r"^atlas_(.+)_(R[123])_", item.sample_id)
        if match is None:
            continue
        system, replica = match.groups()
        candidates.setdefault(system, {}).setdefault(
            replica, (int(item.index), int(item.atoms))
        )
    complete = [
        (max(value[replica][1] for replica in ("R1", "R2", "R3")), system)
        for system, value in candidates.items()
        if all(replica in value for replica in ("R1", "R2", "R3"))
    ]
    selected_systems = [system for _, system in sorted(complete)[:3]]
    if len(selected_systems) != 3:
        raise RuntimeError("train data did not provide three complete 3-replica systems")
    indices: dict[tuple[str, str], int] = {}
    for system in selected_systems:
        for replica in ("R1", "R2", "R3"):
            indices[(system, replica)] = candidates[system][replica][0]
    expected = [(system, f"R{replica}") for system in selected_systems for replica in (1, 2, 3)]
    if len(selected_systems) != 3 or any(key not in indices for key in expected):
        raise RuntimeError("train data did not provide a complete 3-system x 3-replica grid")
    ordered = [dataset[indices[key]] for key in expected]
    provenance = []
    for key, record in zip(expected, ordered):
        actual_system, actual_replica = _system_replica(record)
        if (actual_system, actual_replica) != key:
            raise RuntimeError("clip index metadata disagrees with decoded record provenance")
        # On-disk clips are unpadded and may omit ``frame_mask``; validation
        # materializes the all-valid default only when the record is consumed.
        source_frames = int(torch.as_tensor(record["x"]).shape[0])
        valid_frames = int(
            torch.as_tensor(
                record.get("frame_mask", torch.ones(source_frames, dtype=torch.bool))
            ).sum()
        )
        frame = max(0, valid_frames // 2)
        if frame >= valid_frames:
            frame = valid_frames - 1
        provenance.append(
            {
                "dataset_index": indices[key],
                "system_id": key[0],
                "replica": key[1],
                "source_sample_id": str(record["sample_id"]),
                "source_frame_index": frame,
                "source_time_ps": float(record["time_ps"][frame]),
                "atoms": int(record["x"].shape[1]),
            }
        )
    return ordered, provenance


def _release_unused_cuda_cache(device: torch.device) -> dict[str, int]:
    """Release completed phase workspaces while preserving live model state."""

    gc.collect()
    torch.cuda.synchronize(device)
    before = int(torch.cuda.memory_reserved(device))
    torch.cuda.empty_cache()
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes_before_release": before,
        "reserved_bytes_after_release": int(torch.cuda.memory_reserved(device)),
    }


def _static_records(
    records: Sequence[Mapping[str, Any]], provenance: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    return [
        trajectory_record_to_static(
            record,
            frame_index=int(meta["source_frame_index"]),
            sample_id=f"static_decoder::{meta['system_id']}::{meta['replica']}::{meta['source_frame_index']}",
        )
        for record, meta in zip(records, provenance)
    ]


def _make_model(
    raw: Mapping[str, Any],
    *,
    device: torch.device,
    normalization_hash: str,
    parent: Path,
    parent_sha256: str,
    data_hash: str,
):
    artifact = load_codec_artifact(
        raw["codec"]["checkpoint"],
        expected_sha256=str(raw["codec"]["sha256"]),
        device=device,
    )
    statistics = _load_statistics(
        raw["statistics"],
        device=device,
        normalization_data_hash=normalization_hash,
        codec_sha256=artifact.report.source_sha256,
    )
    model_config = raw["model"]
    model = FrameJointModel.from_codec(
        artifact.model,
        statistics,
        scalar_width=int(model_config["scalar_width"]),
        vector_width=int(model_config["vector_width"]),
        depth=int(model_config["depth"]),
        heads=int(model_config["heads"]),
        geometry_enabled=bool(model_config.get("geometry_enabled", False)),
        motion_enabled=bool(model_config.get("motion_enabled", False)),
    ).to(device)
    parent_payload = torch.load(parent, map_location="cpu", weights_only=False)
    loss_config = JointLossConfig.resolve(parent_payload["contracts"]["loss"])
    rates = raw["training"]["stages"][0]["learning_rates"]
    dynamic = FrameJointTrainer(
        model,
        loss_config=loss_config,
        learning_rates=rates,
        weight_decay=float(raw["training"].get("weight_decay", 0.01)),
        grad_clip=float(raw["training"].get("grad_clip", 1.0)),
        stage="continuation",
        amp=True,
        data_hash=data_hash,
        teacher_artifact_sha256=artifact.report.source_sha256,
        schedule_contract={"engineering_pilot_only": True},
    )
    warm = dynamic.load_warm_start(parent, expected_sha256=parent_sha256)
    return artifact, model, dynamic, warm


def _dynamic_decoder_gradient_norm(
    model: FrameJointModel,
    loss_config: JointLossConfig,
    record: Mapping[str, Any],
    *,
    device: torch.device,
    generator: torch.Generator,
) -> float:
    batch = collate_clip_records([record])
    history = 4 if str(record["time_bucket_id"]).startswith("fixed_history_") else min(8, batch.frames - 1)
    prepared = prepare_frame_joint_batch(
        model.target_teacher,
        batch,
        device=device,
        normalizer=model,
        history_frames=history,
    )
    objective = FrameRectifiedFlowObjective()
    sample = objective.sample(
        prepared.normalized_target,
        prepared.source_center,
        generator=generator,
    )
    output = model(
        sample.interpolated,
        context=prepared.observed_context,
        query=prepared.query,
        flow_time=sample.flow_time,
        clean_future=prepared.target_future,
        decode_generated=True,
        decode_clean=True,
        decode_near=True,
    )
    flow = objective.loss(output.velocity, sample.target_velocity)
    losses = frame_joint_loss(
        output,
        flow,
        prepared,
        sample.flow_time,
        stage="continuation",
        config=loss_config,
    )
    return float(
        parameter_gradient_norm(
            losses.total,
            tuple(model.decoder.parameters()),
        )
    )


def _static_decoder_gradient_norm(
    model: FrameJointModel,
    record: Mapping[str, Any],
    *,
    device: torch.device,
    generator: torch.Generator,
    physical_config: PhysicalAuxiliaryConfig,
) -> tuple[float, dict[str, Any]]:
    prepared = prepare_static_decoder_batch(
        model.target_teacher,
        collate_clip_records([record]),
        device=device,
        generator=generator,
        perturbation=StaticPerturbationConfig(
            sigma_per_axis_angstrom=(0.05, 0.10, 0.20),
            clean_probability=0.0,
        ),
    )
    output = model.decoder(prepared.observed_context)
    losses = static_decoder_loss(
        output,
        prepared,
        StaticDecoderLossConfig(
            coordinate=1.0,
            bond=0.1,
            clash=physical_config.clash_weight,
            gamma=1.0,
        ),
        clash_loss_function=static_clash_loss_function(physical_config),
    )
    norm = parameter_gradient_norm(losses.unscaled_total, tuple(model.decoder.parameters()))
    return float(norm), {
        "sample_id": record["sample_id"],
        "nominal_sigma_per_axis_angstrom": float(prepared.nominal_sigma_per_axis_angstrom[0]),
        "realized_rms_displacement_angstrom": float(prepared.realized_rms_displacement_angstrom[0]),
        "raw_coordinate_mse": float(losses.raw["coordinate_mse"].detach()),
        "raw_bond_distance": float(losses.raw["bond_distance"].detach()),
        "bpos_exact_anchor_count": prepared.bpos_exact_anchor_count,
        "bpos_fallback_count": prepared.bpos_fallback_count,
    }


def _calibrate_gamma(
    model: FrameJointModel,
    dynamic_loss: JointLossConfig,
    dynamic_records: Sequence[Mapping[str, Any]],
    static_records: Sequence[Mapping[str, Any]],
    *,
    device: torch.device,
    seed: int,
    physical_config: PhysicalAuxiliaryConfig,
) -> tuple[float, dict[str, Any]]:
    torch.cuda.reset_peak_memory_stats(device)
    dynamic_generator = torch.Generator(device=device).manual_seed(seed + 101)
    static_generator = torch.Generator(device=device).manual_seed(seed + 202)
    dynamic_norms = []
    static_norms = []
    perturbation_rows = []
    for dynamic_record, static_record in zip(dynamic_records, static_records):
        dynamic_norms.append(
            _dynamic_decoder_gradient_norm(
                model,
                dynamic_loss,
                dynamic_record,
                device=device,
                generator=dynamic_generator,
            )
        )
        norm, row = _static_decoder_gradient_norm(
            model,
            static_record,
            device=device,
            generator=static_generator,
            physical_config=physical_config,
        )
        static_norms.append(norm)
        perturbation_rows.append(row)
    gamma, report = resolve_static_gamma(dynamic_norms, static_norms, target_ratio=0.075)
    report["perturbation_diagnostics"] = perturbation_rows
    report["clash_weight"] = physical_config.clash_weight
    report["clash_rule"] = "A1 released topology-aware physical clash"
    report["clash_table_provenance"] = dict(physical_config.table_provenance or {})
    report["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
    report["allocated_bytes_at_end"] = int(torch.cuda.memory_allocated(device))
    report["reserved_bytes_at_end"] = int(torch.cuda.memory_reserved(device))
    return gamma, report


def _inactive_state(model: FrameJointModel) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if not name.startswith("decoder.")
    }


def _inactive_max_abs(model: FrameJointModel, before: Mapping[str, torch.Tensor]) -> float:
    result = 0.0
    current = dict(model.named_parameters())
    for name, value in before.items():
        result = max(result, float((current[name].detach() - value).abs().max()))
    return result


def _one_precision_smoke(
    raw: Mapping[str, Any],
    *,
    device: torch.device,
    normalization_hash: str,
    data_hash: str,
    parent: Path,
    parent_sha256: str,
    static_records: Sequence[Mapping[str, Any]],
    gamma: float,
    amp: bool,
    seed: int,
    physical_config: PhysicalAuxiliaryConfig,
) -> dict[str, Any]:
    artifact, model, dynamic, warm = _make_model(
        raw,
        device=device,
        normalization_hash=normalization_hash,
        parent=parent,
        parent_sha256=parent_sha256,
        data_hash=data_hash,
    )
    cpu_batch = collate_clip_records(static_records)
    generator = torch.Generator(device=device).manual_seed(seed + (1000 if amp else 0))
    prepared = prepare_static_decoder_batch(
        model.target_teacher,
        cpu_batch,
        device=device,
        generator=generator,
        perturbation=StaticPerturbationConfig(
            sigma_per_axis_angstrom=(0.05, 0.10, 0.20),
            clean_probability=0.20,
        ),
    )
    model.eval()
    with torch.no_grad():
        prediction = model.decoder(prepared.observed_context).coordinates
        mutated = replace(prepared, target_coordinates=prepared.target_coordinates + 777.0)
        repeated = model.decoder(mutated.observed_context).coordinates
    target_mutation_max_abs = float((prediction - repeated).abs().max())
    if target_mutation_max_abs != 0.0:
        raise RuntimeError("static target mutation changed decoder prediction")

    # Constructing the optional branch and not calling it is the disabled path.
    off_before = model.decoder(prepared.observed_context).coordinates.detach().clone()
    static_trainer = StaticDecoderAuxiliaryTrainer(
        model,
        loss_config=StaticDecoderLossConfig(
            coordinate=1.0,
            bond=0.1,
            clash=physical_config.clash_weight,
            gamma=gamma,
        ),
        optimizer=dynamic.optimizer,
        grad_clip=float(raw["training"].get("grad_clip", 1.0)),
        amp=amp,
        clash_loss_function=static_clash_loss_function(physical_config),
    )
    off_after = model.decoder(prepared.observed_context).coordinates.detach()
    disabled_path_max_abs = float((off_before - off_after).abs().max())
    if disabled_path_max_abs != 0.0:
        raise RuntimeError("constructing disabled static auxiliary changed the model")

    teacher_hash_before = module_state_hash(model.target_teacher)
    inactive_before = _inactive_state(model)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    metrics = static_trainer.train_step(prepared)
    torch.cuda.synchronize(device)
    seconds = time.perf_counter() - started
    inactive_error = _inactive_max_abs(model, inactive_before)
    teacher_hash_after = module_state_hash(model.target_teacher)
    if inactive_error != 0.0 or teacher_hash_before != teacher_hash_after:
        raise RuntimeError("static update changed an inactive module")
    optimizer_decoder_ids = {id(parameter) for parameter in model.decoder.parameters()}
    inactive_optimizer_slots = sum(
        parameter in dynamic.optimizer.state
        for group in dynamic.optimizer.param_groups
        for parameter in group["params"]
        if id(parameter) not in optimizer_decoder_ids
    )
    if inactive_optimizer_slots:
        raise RuntimeError("static update created optimizer moments outside decoder")
    return {
        "precision": "bf16" if amp else "fp32",
        "teacher_sha256": artifact.report.source_sha256,
        "warm_start_loaded": len(warm.loaded),
        "warm_start_initialized": len(warm.initialized),
        "target_mutation_max_abs": target_mutation_max_abs,
        "disabled_static_path_dynamic_decode_max_abs": disabled_path_max_abs,
        "inactive_parameter_max_abs": inactive_error,
        "teacher_state_unchanged": teacher_hash_before == teacher_hash_after,
        "inactive_optimizer_state_entries": inactive_optimizer_slots,
        "step_seconds": seconds,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "metrics": metrics,
        "trainer_contract": static_trainer.contract(),
    }


def _largest_profile(
    dataset: Any,
    model: FrameJointModel,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    gamma: float,
    seed: int,
    physical_config: PhysicalAuxiliaryConfig,
) -> dict[str, Any]:
    largest = max(get_clip_specs(dataset), key=lambda item: (item.atoms, item.sample_id))
    record = dataset[int(largest.index)]
    valid = int(torch.as_tensor(record["frame_mask"]).sum())
    frame = max(0, valid // 2)
    static = trajectory_record_to_static(record, frame_index=frame)
    prepared = prepare_static_decoder_batch(
        model.target_teacher,
        collate_clip_records([static]),
        device=device,
        generator=torch.Generator(device=device).manual_seed(seed + 303),
        perturbation=StaticPerturbationConfig(
            sigma_per_axis_angstrom=(0.05, 0.10, 0.20), clean_probability=0.20
        ),
    )
    trainer = StaticDecoderAuxiliaryTrainer(
        model,
        loss_config=StaticDecoderLossConfig(
            coordinate=1.0,
            bond=0.1,
            clash=physical_config.clash_weight,
            gamma=gamma,
        ),
        optimizer=optimizer,
        amp=True,
        clash_loss_function=static_clash_loss_function(physical_config),
    )
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    metrics = trainer.train_step(prepared)
    torch.cuda.synchronize(device)
    return {
        "dataset_index": int(largest.index),
        "sample_id": str(record["sample_id"]),
        "atoms": int(largest.atoms),
        "source_frame_index": frame,
        "seconds": time.perf_counter() - started,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "metrics": metrics,
    }


def main() -> int:
    args = _arguments()
    if args.mode == "pilot" and not 100 <= args.updates <= 200:
        raise ValueError("engineering pilot requires 100--200 successful updates")
    raw = load_config(args.config, schema="molvid.frame_joint.train.v1")
    _validate_frame_joint_config(raw)
    auxiliary_recipe = json.loads(args.auxiliary_recipe.read_text(encoding="utf-8"))
    if auxiliary_recipe.get("schema") != "molvid.A.resolved_auxiliary.v1":
        raise ValueError("auxiliary recipe is not A's frozen resolved recipe")
    if auxiliary_recipe.get("parent_sha256") != args.parent_sha256:
        raise ValueError("engineering parent differs from A auxiliary calibration parent")
    physical_config = PhysicalAuxiliaryConfig.resolve(auxiliary_recipe["physical"])
    device = configure_device(args.device, deterministic=True)
    if device.type != "cuda":
        raise RuntimeError("static decoder smoke/pilot requires CUDA")
    seed_all(args.seed)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    train, valid, data_hash, normalization_hash, relationship, close = _open_data(raw["data"])
    del valid
    started = time.perf_counter()
    try:
        records, provenance = _select_three_system_grid(train)
        static_records = _static_records(records, provenance)
        artifact, calibration_model, dynamic, warm = _make_model(
            raw,
            device=device,
            normalization_hash=normalization_hash,
            parent=args.parent,
            parent_sha256=args.parent_sha256,
            data_hash=data_hash,
        )
        gamma, calibration = _calibrate_gamma(
            calibration_model,
            dynamic.loss_config,
            records,
            static_records,
            device=device,
            seed=args.seed,
            physical_config=physical_config,
        )
        atomic_write_json(output / "gamma_calibration.json", calibration)
        smoke = [
            _one_precision_smoke(
                raw,
                device=device,
                normalization_hash=normalization_hash,
                data_hash=data_hash,
                parent=args.parent,
                parent_sha256=args.parent_sha256,
                static_records=static_records,
                gamma=gamma,
                amp=amp,
                seed=args.seed,
                physical_config=physical_config,
            )
            for amp in (False, True)
        ]
        phase_cache_releases = {
            "after_gradient_calibration_and_precision_smoke": (
                _release_unused_cuda_cache(device)
            )
        }
        _, largest_model, largest_dynamic, _ = _make_model(
            raw,
            device=device,
            normalization_hash=normalization_hash,
            parent=args.parent,
            parent_sha256=args.parent_sha256,
            data_hash=data_hash,
        )
        largest = _largest_profile(
            train,
            largest_model,
            largest_dynamic.optimizer,
            device=device,
            gamma=gamma,
            seed=args.seed,
            physical_config=physical_config,
        )
        del largest_model, largest_dynamic
        phase_cache_releases["after_largest_sample_profile"] = (
            _release_unused_cuda_cache(device)
        )
        pilot = None
        if args.mode == "pilot":
            static_trainer = StaticDecoderAuxiliaryTrainer(
                calibration_model,
                loss_config=StaticDecoderLossConfig(
                    coordinate=1.0,
                    bond=0.1,
                    clash=physical_config.clash_weight,
                    gamma=gamma,
                ),
                optimizer=dynamic.optimizer,
                amp=True,
                clash_loss_function=static_clash_loss_function(physical_config),
            )
            static_generator = torch.Generator(device=device).manual_seed(args.seed + 404)
            metrics_path = output / "pilot_metrics.jsonl"
            pilot_started = time.perf_counter()
            pilot_peak_allocated = int(torch.cuda.memory_allocated(device))
            pilot_peak_reserved = int(torch.cuda.memory_reserved(device))
            cpu_static_batch = collate_clip_records(static_records)
            with metrics_path.open("w", encoding="utf-8") as handle:
                for update in range(args.updates):
                    prepared = prepare_static_decoder_batch(
                        calibration_model.target_teacher,
                        cpu_static_batch,
                        device=device,
                        generator=static_generator,
                        perturbation=StaticPerturbationConfig(
                            sigma_per_axis_angstrom=(0.05, 0.10, 0.20),
                            clean_probability=0.20,
                        ),
                    )
                    row = static_trainer.train_step(prepared)
                    row["update_index"] = update
                    del prepared
                    row["cuda_allocated_bytes"] = int(torch.cuda.memory_allocated(device))
                    row["cuda_reserved_bytes"] = int(torch.cuda.memory_reserved(device))
                    pilot_peak_allocated = max(
                        pilot_peak_allocated, row["cuda_allocated_bytes"]
                    )
                    pilot_peak_reserved = max(
                        pilot_peak_reserved, row["cuda_reserved_bytes"]
                    )
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
                    handle.flush()
            torch.cuda.synchronize(device)
            checkpoint = output / f"engineering_pilot_decoder_step_{args.updates:08d}.pt"
            torch.save(
                {
                    "schema": "molvid.static_decoder.engineering_pilot.v1",
                    "formal_parent_allowed": False,
                    "parent_sha256": args.parent_sha256,
                    "code_commit": _commit(),
                    "successful_static_updates": args.updates,
                    "decoder_state": calibration_model.decoder.state_dict(),
                    "optimizer_state": dynamic.optimizer.state_dict(),
                    "static_contract": static_trainer.contract(),
                    "static_generator_state": static_generator.get_state().cpu(),
                },
                checkpoint,
            )
            pilot = {
                "successful_static_updates": args.updates,
                "seconds": time.perf_counter() - pilot_started,
                "peak_allocated_bytes": pilot_peak_allocated,
                "peak_reserved_bytes": pilot_peak_reserved,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256_file(checkpoint),
                "formal_parent_allowed": False,
                "metrics": str(metrics_path),
            }
        result = {
            "schema": "molvid.static_decoder.smoke_pilot.v1",
            "mode": args.mode,
            "code_commit": _commit(),
            "device": str(device),
            "cuda_device_name": torch.cuda.get_device_name(device),
            "parent": str(args.parent.resolve()),
            "parent_sha256": args.parent_sha256,
            "auxiliary_recipe": str(args.auxiliary_recipe.resolve()),
            "auxiliary_recipe_sha256": sha256_file(args.auxiliary_recipe),
            "teacher_sha256": artifact.report.source_sha256,
            "data_hash": data_hash,
            "normalization_source_data_hash": normalization_hash,
            "data_relationship": relationship,
            "train_system_count": 3,
            "train_replica_count": 9,
            "static_sources": provenance,
            "perturbation": StaticPerturbationConfig().contract(),
            "loss": StaticDecoderLossConfig(
                coordinate=1.0,
                bond=0.1,
                clash=physical_config.clash_weight,
                gamma=gamma,
            ).contract(),
            "gamma_calibration": str(output / "gamma_calibration.json"),
            "warm_start_loaded": len(warm.loaded),
            "warm_start_initialized": len(warm.initialized),
            "smoke": smoke,
            "largest_profile": largest,
            "phase_cache_releases": phase_cache_releases,
            "pilot": pilot,
            "elapsed_seconds": time.perf_counter() - started,
            "test_opened": False,
            "limitations": [
                "engineering parent is historical P2-GM, not the released A1 formal checkpoint",
                "static clash reuses the released A1 thresholds and normalization",
                "engineering pilot checkpoint is prohibited as a formal three-arm parent",
            ],
        }
        atomic_write_json(output / "result.json", result)
        print(json.dumps(result, sort_keys=True))
    finally:
        close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
