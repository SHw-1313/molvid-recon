"""Inspect a verified current codec or DiT and one real validation forward."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn

from molvid.checkpoints import load_codec_artifact, load_dit_inference
from molvid.data.batch import ClipBatch, collate_clip_records
from molvid.data.manifest import load_datasets
from molvid.generation import _block_positions, _observed_scaffold
from molvid.latent.conditioning import build_observation_condition
from molvid.runtime import configure_device
from molvid.training.batches import prepare_batch_then_to_device


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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("codec", "dit"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--codec", type=Path)
    parser.add_argument("--codec-sha256")
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--valid-index", type=int, required=True)
    parser.add_argument("--history", type=int, choices=(4, 8), default=8)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    if args.kind == "dit" and (args.codec is None or not args.codec_sha256):
        parser.error("DiT inspection requires --codec and --codec-sha256")
    device = configure_device(args.device, deterministic=True)
    splits = load_datasets(args.manifest_root)
    try:
        if args.valid_index < 0 or args.valid_index >= len(splits.valid):
            raise IndexError("validation clip index is out of range")
        template = collate_clip_records([splits.valid[args.valid_index]])
        if args.kind == "codec":
            artifact = load_codec_artifact(
                args.checkpoint, expected_sha256=args.checkpoint_sha256, device=device
            )
            model = artifact.model.eval()
            forward = inspect_codec_forward(model, template, device)
            identity = artifact.report.source_sha256
        else:
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
        print(json.dumps({
            "schema_version": "molvid.model.inspection.v1",
            "kind": args.kind, "checkpoint_sha256": identity,
            "sample_id": template.sample_id[0],
            "model": model_summary(model), "forward_shapes": forward,
        }, sort_keys=True))
        return 0
    finally:
        splits.close()


if __name__ == "__main__":
    raise SystemExit(main())
