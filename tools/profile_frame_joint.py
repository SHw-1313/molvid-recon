"""Profile one real maximum-system Frame Joint training update on CUDA."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any, Callable

import torch

from molvid.checkpoints import load_frame_joint_inference
from molvid.data.batch import collate_clip_records
from molvid.data.manifest import load_datasets
from molvid.data.sampling import get_clip_specs
from molvid.data.store import ClipMMapDataset
from molvid.flow.objective import FrameRectifiedFlowObjective
from molvid.runtime import atomic_write_json, configure_device
from molvid.training.batches import prepare_frame_joint_batch
from molvid.training.joint import FrameJointTrainer, JointLossConfig, frame_joint_loss


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--codec", type=Path, required=True)
    parser.add_argument("--codec-sha256", required=True)
    data = parser.add_mutually_exclusive_group(required=True)
    data.add_argument("--train-store", type=Path)
    data.add_argument("--manifest-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--history", type=int, default=8)
    return parser.parse_args()


def _measure(device: torch.device, function: Callable[[], Any]) -> tuple[Any, dict[str, float | int]]:
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    allocated = torch.cuda.memory_allocated(device)
    started = time.perf_counter()
    value = function()
    torch.cuda.synchronize(device)
    seconds = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated(device)
    return value, {
        "seconds": seconds,
        "allocated_before_bytes": allocated,
        "peak_allocated_bytes": peak,
        "incremental_peak_bytes": peak - allocated,
    }


def main() -> int:
    args = _arguments()
    device = configure_device(args.device, deterministic=True)
    loaded = load_frame_joint_inference(
        args.checkpoint,
        expected_sha256=args.checkpoint_sha256,
        codec_path=args.codec,
        codec_sha256=args.codec_sha256,
        device=device,
    )
    model = loaded["model"]
    payload = loaded["payload"]
    contracts = payload["contracts"]
    splits = load_datasets(args.manifest_root) if args.manifest_root is not None else None
    dataset = splits.train if splits is not None else ClipMMapDataset(args.train_store)
    try:
        specs = get_clip_specs(dataset)
        counts = [int(item.atoms) for item in specs]
        index = max(range(len(dataset)), key=counts.__getitem__)
        record = dataset[index]
        batch = collate_clip_records([record])
        if batch.atom_count != max(counts):
            raise RuntimeError("maximum-system selection changed while loading")

        # Warm topology/device caches once, then report a steady-state target
        # preparation pass separately from the trainable graph.
        prepare_frame_joint_batch(
            model.target_teacher,
            batch,
            device=device,
            normalizer=model,
            history_frames=args.history,
        )
        prepared, preparation = _measure(
            device,
            lambda: prepare_frame_joint_batch(
                model.target_teacher,
                batch,
                device=device,
                normalizer=model,
                history_frames=args.history,
            ),
        )
        flow = FrameRectifiedFlowObjective()
        generator = torch.Generator(device=device).manual_seed(260920)
        flow_time = torch.full(
            (prepared.normalized_target.batch_size,),
            0.95,
            device=device,
            dtype=prepared.normalized_target.h.dtype,
        )
        sample = flow.sample(
            prepared.normalized_target,
            prepared.source_center,
            generator=generator,
            flow_time=flow_time,
        )

        model.eval()
        with torch.no_grad():
            history, history_profile = _measure(
                device, lambda: model.history_encoder(prepared.observed_context)
            )
            velocity, dit_profile = _measure(
                device,
                lambda: model.dit(
                    sample.interpolated,
                    context=prepared.observed_context,
                    query=prepared.query,
                    history_memory=history,
                    flow_time=flow_time,
                ),
            )
            endpoint = model.inverse(
                sample.interpolated.with_features(
                    sample.interpolated.h + (1.0 - flow_time[0]) * velocity.h,
                    sample.interpolated.v + (1.0 - flow_time[0]) * velocity.v,
                )
            )
            _, decoder_profile = _measure(
                device,
                lambda: model.decoder(
                    prepared.observed_context, endpoint, prepared.query
                ),
            )
        del history, velocity, endpoint
        torch.cuda.empty_cache()

        optimization = contracts["optimization"]
        trainer = FrameJointTrainer(
            model,
            loss_config=JointLossConfig.resolve(contracts["loss"]),
            learning_rates=optimization["base_learning_rates"],
            weight_decay=float(optimization["weight_decay"]),
            grad_clip=float(optimization["grad_clip"]),
            stage="joint",
            amp=bool(optimization["amp_bfloat16"]),
            data_hash=str(contracts["data_hash"]),
            teacher_artifact_sha256=str(contracts["teacher_artifact_sha256"]),
        )
        model.train()
        trainer.optimizer.zero_grad(set_to_none=True)
        output, full_forward = _measure(
            device,
            lambda: model(
                sample.interpolated,
                context=prepared.observed_context,
                query=prepared.query,
                flow_time=flow_time,
                clean_future=prepared.target_future,
                decode_generated=True,
                decode_clean=True,
                decode_near=True,
            ),
        )
        losses, loss_profile = _measure(
            device,
            lambda: frame_joint_loss(
                output,
                flow.loss(output.velocity, sample.target_velocity),
                prepared,
                flow_time,
                stage="joint",
                config=trainer.loss_config,
            ),
        )
        _, backward_profile = _measure(device, losses.total.backward)
        _, optimizer_profile = _measure(device, trainer.optimizer.step)

        result = {
            "schema_version": "molvid.frame_joint.cuda_profile.v1",
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device),
            "checkpoint_sha256": args.checkpoint_sha256,
            "sample_id": batch.sample_id[0],
            "atoms": batch.atom_count,
            "frames": batch.frames,
            "history_frames": args.history,
            "query_frames": batch.frames - args.history,
            "flow_time": 0.95,
            "profiles": {
                "teacher_prepare_steady_state": preparation,
                "history_encoder_forward": history_profile,
                "dit_forward": dit_profile,
                "decoder_forward": decoder_profile,
                "full_joint_forward": full_forward,
                "loss_construction": loss_profile,
                "backward": backward_profile,
                "optimizer_step": optimizer_profile,
            },
            "loss": float(losses.total.detach()),
            "peak_allocated_bytes": max(
                int(profile["peak_allocated_bytes"])
                for profile in (
                    preparation,
                    history_profile,
                    dit_profile,
                    decoder_profile,
                    full_forward,
                    loss_profile,
                    backward_profile,
                    optimizer_profile,
                )
            ),
        }
        args.output.mkdir(parents=True, exist_ok=True)
        atomic_write_json(args.output / "profile_max_system.json", result)
        print(json.dumps(result, sort_keys=True))
        return 0
    finally:
        if splits is not None:
            splits.close()
        else:
            dataset.close()


if __name__ == "__main__":
    raise SystemExit(main())
