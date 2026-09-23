"""Check that actual-source generation is exactly future-target independent on CUDA."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import torch

from molvid.checkpoints import load_frame_joint_inference
from molvid.cli.train_frame_joint import _open_data, _validate_frame_joint_config
from molvid.config import load_config
from molvid.data.batch import collate_clip_records
from molvid.data.sampling import get_clip_specs
from molvid.flow.objective import FrameRectifiedFlowObjective
from molvid.flow.source import sample_frame_source
from molvid.runtime import atomic_write_json, configure_device
from molvid.training.batches import PreparedFrameJointBatch, prepare_frame_joint_batch
from molvid.training.dit import module_state_hash


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--parent-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _latent_max_abs(
    first: PreparedFrameJointBatch,
    second: PreparedFrameJointBatch,
) -> float:
    return max(
        float((first.source_center.h - second.source_center.h).abs().max()),
        float((first.source_center.v - second.source_center.v).abs().max()),
    )


def main() -> int:
    args = _arguments()
    raw = load_config(args.config, schema="molvid.frame_joint.train.v1")
    _validate_frame_joint_config(raw)
    device = configure_device(str(raw["training"].get("device", "cuda")), deterministic=True)
    train, valid, _data_hash, _normalization_hash, _relationship, close = _open_data(
        raw["data"]
    )
    del valid
    try:
        loaded = load_frame_joint_inference(
            args.parent,
            expected_sha256=args.parent_sha256,
            codec_path=raw["codec"]["checkpoint"],
            codec_sha256=str(raw["codec"]["sha256"]),
            device=device,
        )
        model = loaded["model"].eval()
        specs = get_clip_specs(train)
        chosen = min(
            (
                item
                for item in specs
                if item.frames == 16
                and not item.time_bucket_id.startswith("fixed_history")
            ),
            key=lambda item: (item.atoms, item.index),
        )
        original_cpu = collate_clip_records([train[chosen.index]])
        history = 4
        mutated_x = original_cpu.x.clone()
        mutated_bpos = original_cpu.bpos.clone()
        mutation = torch.randn(
            mutated_x[history:].shape,
            generator=torch.Generator().manual_seed(17),
        ) * 50.0
        mutated_x[history:] += mutation
        mutated_bpos[history:] += mutation
        mutated_cpu = replace(original_cpu, x=mutated_x, bpos=mutated_bpos)
        prepared = [
            prepare_frame_joint_batch(
                model.target_teacher,
                value,
                device=device,
                normalizer=model,
                history_frames=history,
            )
            for value in (original_cpu, original_cpu, mutated_cpu)
        ]
        sampled_generator = torch.Generator(device=device).manual_seed(91)
        sources = tuple(
            sample_frame_source(prepared[0].source_center, generator=sampled_generator)[0]
            for _ in range(2)
        )
        source_draw_max_abs = max(
            float((sources[0].h - sources[1].h).abs().max()),
            float((sources[0].v - sources[1].v).abs().max()),
        )
        main_generator = torch.Generator(device=device).manual_seed(92)
        flow_sample = FrameRectifiedFlowObjective().sample(
            prepared[0].normalized_target,
            prepared[0].source_center,
            generator=main_generator,
        )
        teacher_before = module_state_hash(model.target_teacher)
        with torch.no_grad():
            outputs = [
                model(
                    flow_sample.interpolated,
                    context=value.observed_context,
                    query=value.query,
                    flow_time=flow_sample.flow_time,
                    sampled_sources=sources,
                    sampled_steps=4,
                )
                for value in prepared
            ]
        teacher_after = module_state_hash(model.target_teacher)

        def output_delta(left: int, right: int) -> float:
            return max(
                float((first.coordinates - second.coordinates).abs().max())
                for first, second in zip(
                    outputs[left].sampled, outputs[right].sampled, strict=True
                )
            )

        result = {
            "schema": "molvid.frame_gm.p3_target_mutation_cuda_check.v1",
            "sample_id": chosen.sample_id,
            "atoms": int(chosen.atoms),
            "frames": int(chosen.frames),
            "history_frames": history,
            "sampled_draws": 2,
            "sampled_euler_steps": 4,
            "source_draws_distinct_max_abs": source_draw_max_abs,
            "observed_coordinates_equal_after_future_mutation": torch.equal(
                prepared[0].observed_context.coordinates,
                prepared[2].observed_context.coordinates,
            ),
            "origin_equal_after_future_mutation": torch.equal(
                prepared[0].observed_context.sample_origin,
                prepared[2].observed_context.sample_origin,
            ),
            "source_control_max_abs": _latent_max_abs(prepared[0], prepared[1]),
            "source_mutation_max_abs": _latent_max_abs(prepared[0], prepared[2]),
            "output_control_max_abs": output_delta(0, 1),
            "output_mutation_max_abs": output_delta(0, 2),
            "teacher_state_hash_before": teacher_before,
            "teacher_state_hash_after": teacher_after,
            "teacher_state_unchanged": teacher_before == teacher_after,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "device": str(device),
            "cuda_device_name": torch.cuda.get_device_name(device),
            "test_opened": False,
        }
        passed = bool(
            result["source_draws_distinct_max_abs"] > 0.0
            and result["observed_coordinates_equal_after_future_mutation"]
            and result["origin_equal_after_future_mutation"]
            and result["source_control_max_abs"] == 0.0
            and result["source_mutation_max_abs"] == 0.0
            and result["output_control_max_abs"] == 0.0
            and result["output_mutation_max_abs"] == 0.0
            and result["teacher_state_unchanged"]
        )
        result["passed"] = passed
        atomic_write_json(args.output, result)
        print(json.dumps(result, sort_keys=True))
        if not passed:
            raise RuntimeError("P3 future-target mutation CUDA check failed")
    finally:
        close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
