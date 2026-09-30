"""CUDA preservation and startup-gradient checks for the C4-to-C8 expansion."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import torch

from molvid.cli.train_frame_joint import _open_data
from molvid.data.batch import collate_clip_records
from molvid.data.sampling import get_clip_specs
from molvid.flow.objective import FrameRectifiedFlowObjective
from molvid.runtime import atomic_write_json, configure_device, seed_all
from molvid.training.batches import prepare_frame_joint_batch
from molvid.training.dit import module_state_hash
from tools.gm_capacity_common import (
    build_capacity_trainer,
    load_capacity_release,
    load_parent_model,
    parameter_ledger,
)


SEED = 20260928
NEW_IDENTITY_BLOCKS = (1, 3, 5, 7)
NEW_GEOMETRY_BLOCKS = (2, 6)


def _grad_norm(parameters) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            total += float(parameter.grad.detach().float().square().sum())
    return math.sqrt(total)


def _block_gradients(block: torch.nn.Module) -> dict[str, float]:
    named = list(block.named_parameters())
    gate = [
        parameter for name, parameter in named
        if "adaln.modulation" in name or "residual_gate" in name
    ]
    inner = [
        parameter for name, parameter in named
        if "adaln.modulation" not in name and "residual_gate" not in name
    ]
    return {
        "all": _grad_norm(parameter for _, parameter in named),
        "residual_gates": _grad_norm(gate),
        "inner": _grad_norm(inner),
    }


def _geometry_gradients(block: torch.nn.Module) -> dict[str, float]:
    local = block.local_geometry
    if local is None:
        raise RuntimeError("requested new geometry block has no G branch")
    named = list(local.named_parameters())
    return {
        "all": _grad_norm(parameter for _, parameter in named),
        "residual_gates": _grad_norm(
            parameter for name, parameter in named if "gate" in name and "vector_gate" not in name
        ),
        "inner": _grad_norm(
            parameter for name, parameter in named
            if name not in {"scalar_gate", "vector_residual_gate"}
        ),
    }


def _max_abs(first: torch.Tensor, second: torch.Tensor) -> float:
    return float((first.float() - second.float()).abs().max())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gradient-updates", type=int, default=3)
    args = parser.parse_args()
    if args.gradient_updates < 2 or args.gradient_updates > 8:
        raise ValueError("gradient-updates must lie in [2,8]")
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    device = configure_device("cuda", deterministic=True)
    seed_all(SEED)
    bundle = load_capacity_release(args.release)
    parent = load_parent_model(bundle, device=device)
    trainer, warm = build_capacity_trainer(
        bundle,
        parent,
        depth=8,
        device=device,
        schedule_contract={"schema": "molvid.capacity.cuda_check.v1"},
    )
    parent_model = parent["model"].eval()
    expanded = trainer.base_model.eval()
    train, _, data_hash, _, _, close_data = _open_data(bundle.configuration["data"])
    evidence: dict[str, object] = {
        "schema": "molvid.capacity.cuda_check.v1",
        "release_sha256": bundle.checkpoint_sha256,
        "data_hash": data_hash,
        "warm_start_report": warm.as_dict(),
        "parameter_ledger": parameter_ledger(expanded),
        "function_preservation": [],
        "gradient_updates": [],
        "test_opened": False,
    }
    try:
        ordinary = [
            spec for spec in get_clip_specs(train)
            if not spec.time_bucket_id.startswith("fixed_history_")
        ]
        if not ordinary:
            raise RuntimeError("capacity CUDA check found no ordinary training record")
        chosen = min(ordinary, key=lambda spec: (spec.atoms, spec.effective_tokens, spec.sample_id))
        cpu = collate_clip_records([train[chosen.index]])
        batches = {}
        objective = FrameRectifiedFlowObjective()
        for history in (4, 8):
            batch = prepare_frame_joint_batch(
                expanded.target_teacher,
                cpu,
                device=device,
                normalizer=expanded,
                history_frames=history,
            )
            batches[history] = batch
            draw = objective.sample(
                batch.normalized_target,
                batch.source_center,
                generator=torch.Generator(device=device).manual_seed(SEED + history),
            )
            for amp in (False, True):
                with torch.no_grad(), torch.autocast(
                    "cuda", dtype=torch.bfloat16, enabled=amp
                ):
                    first = parent_model(
                        draw.interpolated,
                        context=batch.observed_context,
                        query=batch.query,
                        flow_time=draw.flow_time,
                        decode_generated=True,
                    )
                    second = expanded(
                        draw.interpolated,
                        context=batch.observed_context,
                        query=batch.query,
                        flow_time=draw.flow_time,
                        decode_generated=True,
                    )
                assert first.generated is not None and second.generated is not None
                row = {
                    "history_frames": history,
                    "precision": "bf16" if amp else "fp32",
                    "velocity_h_max_abs": _max_abs(first.velocity.h, second.velocity.h),
                    "velocity_v_max_abs": _max_abs(first.velocity.v, second.velocity.v),
                    "endpoint_h_max_abs": _max_abs(first.endpoint.h, second.endpoint.h),
                    "endpoint_v_max_abs": _max_abs(first.endpoint.v, second.endpoint.v),
                    "coordinates_max_abs_A": _max_abs(
                        first.generated.coordinates, second.generated.coordinates
                    ),
                }
                if any(value != 0.0 for key, value in row.items() if key.endswith(("_abs", "_abs_A"))):
                    raise RuntimeError(f"C8 warm start does not preserve the C4 function: {row}")
                evidence["function_preservation"].append(row)

        teacher_before = module_state_hash(expanded.target_teacher)
        batch = batches[4]
        main_generator = torch.Generator(device=device).manual_seed(SEED + 1000)
        sampled_generator = torch.Generator(device=device).manual_seed(SEED + 2000)
        for update in range(args.gradient_updates):
            metrics = trainer.train_step(
                batch,
                generator=main_generator,
                sampled_generator=sampled_generator,
                auxiliary_active=False,
            )
            row = {
                "update": update + 1,
                "loss": metrics["loss"],
                "grad_norm": metrics["grad_norm"],
                "identity_blocks": {
                    str(index): _block_gradients(expanded.dit.blocks[index - 1])
                    for index in NEW_IDENTITY_BLOCKS
                },
                "new_geometry": {
                    str(index): _geometry_gradients(expanded.dit.blocks[index - 1])
                    for index in NEW_GEOMETRY_BLOCKS
                },
            }
            evidence["gradient_updates"].append(row)
        first = evidence["gradient_updates"][0]
        last = evidence["gradient_updates"][-1]
        for index in NEW_IDENTITY_BLOCKS:
            if first["identity_blocks"][str(index)]["residual_gates"] <= 0:
                raise RuntimeError(f"new identity block {index} lacks startup gate gradient")
            if last["identity_blocks"][str(index)]["inner"] <= 0:
                raise RuntimeError(f"new identity block {index} lacks subsequent inner gradient")
        for index in NEW_GEOMETRY_BLOCKS:
            if first["new_geometry"][str(index)]["residual_gates"] <= 0:
                raise RuntimeError(f"new G branch {index} lacks startup gate gradient")
            if last["new_geometry"][str(index)]["inner"] <= 0:
                raise RuntimeError(f"new G branch {index} lacks subsequent inner gradient")
        evidence["teacher_unchanged"] = teacher_before == module_state_hash(
            expanded.target_teacher
        )
        if evidence["teacher_unchanged"] is not True:
            raise RuntimeError("capacity CUDA check changed the frozen teacher")
        evidence["sample_id"] = chosen.sample_id
        evidence["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
        evidence["gpu_hours"] = (time.time() - started) / 3600.0
        evidence["passed"] = True
        atomic_write_json(args.output / "cuda_check.json", evidence)
        print(json.dumps({
            "passed": True,
            "sample_id": chosen.sample_id,
            "peak_allocated_bytes": evidence["peak_allocated_bytes"],
            "output": str(args.output / "cuda_check.json"),
        }, sort_keys=True))
    finally:
        atomic_write_json(args.output / "cuda_check_partial.json", evidence)
        close_data()


if __name__ == "__main__":
    main()
