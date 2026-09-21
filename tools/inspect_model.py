"""Inspect a verified current codec or DiT and one real validation forward."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn

from molvid.checkpoints import (
    load_codec_artifact,
    load_dit_inference,
    load_frame_joint_inference,
)
from molvid.data.batch import ClipBatch, collate_clip_records
from molvid.data.manifest import load_datasets
from molvid.data.store import ClipMMapDataset
from molvid.flow.objective import FrameRectifiedFlowObjective
from molvid.generation import _block_positions, _observed_scaffold
from molvid.latent.conditioning import build_observation_condition
from molvid.runtime import atomic_write_json, configure_device
from molvid.training.batches import prepare_batch_then_to_device, prepare_frame_joint_batch
from molvid.training.dit import module_state_hash
from molvid.training.joint import JointLossConfig, frame_joint_loss


def model_summary(model: nn.Module) -> dict[str, Any]:
    """Report owned parameters per module, including frozen state."""

    modules = []
    for name, module in model.named_modules():
        owned = tuple(module.parameters(recurse=False))
        modules.append({
            "path": name or ".",
            "type": type(module).__name__,
            "parameters": sum(parameter.numel() for parameter in owned),
            "frozen_parameters": sum(
                parameter.numel() for parameter in owned if not parameter.requires_grad
            ),
        })
    return {
        "tree": str(model),
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "frozen_names": [
            name for name, parameter in model.named_parameters() if not parameter.requires_grad
        ],
        "modules": modules,
    }


@torch.no_grad()
def inspect_codec_forward(codec: nn.Module, template: ClipBatch, device: torch.device) -> dict[str, Any]:
    coordinate = prepare_batch_then_to_device(codec, template, device)
    latent = codec.encode(coordinate)
    decoded = codec.decode(latent)
    return {
        "coordinate": list(coordinate.x.shape),
        "state_h": list(latent.state_h.shape),
        "state_v": list(latent.state_v.shape),
        "detail_h": list(latent.detail_h.shape),
        "detail_v": list(latent.detail_v.shape),
        "reconstruction": list(decoded.x_hat.shape),
    }


@torch.no_grad()
def inspect_dit_forward(
    codec: nn.Module, model: nn.Module, adapter: nn.Module,
    template: ClipBatch, *, device: torch.device, codec_hash: str,
    data_hash: str, history_frames: int,
) -> dict[str, Any]:
    """Run one DiT forward on a repeated-prefix scaffold, never true future coordinates."""

    scaffold = _observed_scaffold(template, template.x[:history_frames], history_frames)
    coordinate = prepare_batch_then_to_device(codec, scaffold, device)
    coordinate = replace(
        coordinate, bpos=_block_positions(coordinate.x, coordinate.block_id)
    )
    latent = codec.encode(coordinate)
    packed = adapter.from_codec_latent(
        latent, codec_hash=codec_hash, data_hash=data_hash,
        origin_from_latent=True, loss_mask=coordinate.loss_mask,
    )
    observation = build_observation_condition(
        packed, history_frames=history_frames, coordinates=coordinate.x,
        frame_mask=coordinate.frame_mask, loss_mask=coordinate.loss_mask,
    )
    observed = packed.with_observation(
        observation.latent_observation_mask, sample_origin=observation.sample_origin
    )
    tau = torch.full((observed.batch_size,), 0.5, device=device)
    velocity = model(observed, tau)
    return {
        "coordinate_scaffold": list(coordinate.x.shape),
        "latent_state_h": list(observed.state_h.shape),
        "velocity": {name: list(value.shape) for name, value in velocity.as_dict().items()},
        "future_condition": "observed_prefix_only",
    }


def inspect_frame_joint_forward(
    model: nn.Module,
    template: ClipBatch,
    *,
    device: torch.device,
    history_frames: int,
    loss_config: JointLossConfig,
) -> dict[str, Any]:
    prepared = prepare_frame_joint_batch(
        model.target_teacher,
        template,
        device=device,
        normalizer=model,
        history_frames=history_frames,
    )
    objective = FrameRectifiedFlowObjective()
    sample = objective.sample(
        prepared.normalized_target,
        prepared.source_center,
        generator=torch.Generator(device=device).manual_seed(0),
        flow_time=torch.full(
            (prepared.normalized_target.batch_size,), 0.95, device=device
        ),
    )
    model.zero_grad(set_to_none=True)
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
    losses = frame_joint_loss(
        output,
        objective.loss(output.velocity, sample.target_velocity),
        prepared,
        sample.flow_time,
        stage="joint",
        config=loss_config,
    )
    losses.total.backward()
    gradients = {}
    for name in ("history_encoder", "dit", "decoder", "target_teacher"):
        module = getattr(model, name)
        values = list(module.named_parameters())
        gradients[name] = {
            "trainable": sum(parameter.numel() for _, parameter in values if parameter.requires_grad),
            "gradient_nonzero_tensors": sum(
                parameter.grad is not None and bool(torch.any(parameter.grad != 0))
                for _, parameter in values
                if parameter.requires_grad
            ),
            "gradient_missing": [
                parameter_name
                for parameter_name, parameter in values
                if parameter.requires_grad and parameter.grad is None
            ],
        }
    return {
        "observed_h": list(prepared.observed_context.latent.h.shape),
        "observed_v": list(prepared.observed_context.latent.v.shape),
        "future_target_h": list(prepared.target_future.h.shape),
        "future_target_v": list(prepared.target_future.v.shape),
        "velocity_h": list(output.velocity.h.shape),
        "velocity_v": list(output.velocity.v.shape),
        "generated_coordinates": list(output.generated.coordinates.shape),
        "history_memory_h": list(output.history_memory.h.shape),
        "history_memory_v": list(output.history_memory.v.shape),
        "physical_time_shape": list(prepared.query.time_ps.shape),
        "flow_time_shape": list(sample.flow_time.shape),
        "future_condition": "observed_prefix_topology_and_query_time_only",
        "loss": float(losses.total.detach()),
        "gradient_destinations": gradients,
        "teacher_state_hash": module_state_hash(model.target_teacher),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("codec", "dit", "frame_joint"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--codec", type=Path)
    parser.add_argument("--codec-sha256")
    parser.add_argument("--manifest-root", type=Path)
    parser.add_argument("--store", type=Path)
    parser.add_argument("--valid-index", type=int, required=True)
    parser.add_argument("--history", type=int, choices=(4, 8), default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.kind in {"dit", "frame_joint"} and (args.codec is None or not args.codec_sha256):
        parser.error("DiT/Frame Joint inspection requires --codec and --codec-sha256")
    if (args.manifest_root is None) == (args.store is None):
        parser.error("provide exactly one of --manifest-root or --store")
    if args.kind in {"codec", "dit"} and args.manifest_root is None:
        parser.error("codec/DiT inspection requires --manifest-root")
    device = configure_device(args.device, deterministic=True)
    splits = load_datasets(args.manifest_root) if args.manifest_root is not None else None
    store = ClipMMapDataset(args.store) if args.store is not None else None
    try:
        dataset = splits.valid if splits is not None else store
        assert dataset is not None
        if args.valid_index < 0 or args.valid_index >= len(dataset):
            raise IndexError("validation clip index is out of range")
        template = collate_clip_records([dataset[args.valid_index]])
        if args.kind == "codec":
            artifact = load_codec_artifact(
                args.checkpoint, expected_sha256=args.checkpoint_sha256, device=device
            )
            model = artifact.model.eval()
            forward = inspect_codec_forward(model, template, device)
            identity = artifact.report.source_sha256
        elif args.kind == "dit":
            assert splits is not None
            loaded = load_dit_inference(
                args.checkpoint, expected_sha256=args.checkpoint_sha256,
                codec_path=args.codec, codec_sha256=args.codec_sha256, device=device,
            )
            if loaded["data_hash"] != splits.data_hash:
                raise ValueError("DiT and inspection manifest data hashes differ")
            model = loaded["model"].eval()
            forward = inspect_dit_forward(
                loaded["codec"].eval(), model, loaded["adapter"], template,
                device=device, codec_hash=loaded["codec_hash"],
                data_hash=loaded["data_hash"], history_frames=args.history,
            )
            identity = args.checkpoint_sha256
        else:
            loaded = load_frame_joint_inference(
                args.checkpoint,
                expected_sha256=args.checkpoint_sha256,
                codec_path=args.codec,
                codec_sha256=args.codec_sha256,
                device=device,
            )
            model = loaded["model"]
            forward = inspect_frame_joint_forward(
                model,
                template,
                device=device,
                history_frames=args.history,
                loss_config=JointLossConfig.resolve(
                    loaded["payload"]["contracts"]["loss"]
                ),
            )
            identity = args.checkpoint_sha256
        result = {
            "schema_version": "molvid.model.inspection.v1",
            "kind": args.kind, "checkpoint_sha256": identity,
            "sample_id": template.sample_id[0],
            "device": str(device),
            "dtype": str(next(model.parameters()).dtype),
            "model": model_summary(model), "forward_shapes": forward,
        }
        if args.output is not None:
            atomic_write_json(args.output, result)
            print(json.dumps({
                "output": str(args.output),
                "kind": args.kind,
                "total_parameters": result["model"]["total_parameters"],
                "trainable_parameters": result["model"]["trainable_parameters"],
                "forward_shapes": forward,
            }, sort_keys=True))
        else:
            print(json.dumps(result, sort_keys=True))
        return 0
    finally:
        if splits is not None:
            splits.close()
        if store is not None:
            store.close()


if __name__ == "__main__":
    raise SystemExit(main())
