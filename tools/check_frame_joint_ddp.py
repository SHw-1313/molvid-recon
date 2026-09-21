"""Check one-vs-two GPU update equivalence and exact two-rank resume."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.distributed as dist
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel

from molvid.checkpoints import capture_rng_state, load_frame_joint_inference
from molvid.data.batch import collate_clip_records
from molvid.data.store import ClipMMapDataset
from molvid.flow.objective import FrameRectifiedFlowObjective, broadcast_sample_time
from molvid.runtime import atomic_write_json, configure_device, sha256_file
from molvid.training.batches import PreparedFrameJointBatch, prepare_frame_joint_batch
from molvid.training.joint import FrameJointTrainer, JointLossConfig, frame_joint_loss


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--codec", type=Path, required=True)
    parser.add_argument("--codec-sha256", required=True)
    parser.add_argument("--train-store", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _trainer(model: torch.nn.Module, contracts: Mapping[str, Any]) -> FrameJointTrainer:
    optimization = contracts["optimization"]
    return FrameJointTrainer(
        model,
        loss_config=JointLossConfig.resolve(contracts["loss"]),
        learning_rates=optimization["base_learning_rates"],
        weight_decay=float(optimization["weight_decay"]),
        grad_clip=float(optimization["grad_clip"]),
        stage="joint",
        amp=False,
        data_hash=str(contracts["data_hash"]),
        teacher_artifact_sha256=str(contracts["teacher_artifact_sha256"]),
    )


def _fixed_step(
    trainer: FrameJointTrainer,
    batch: PreparedFrameJointBatch,
    *,
    label: str,
    flow_time_values: tuple[float, ...],
) -> float:
    target = batch.normalized_target
    source = target.with_features(torch.zeros_like(target.h), torch.zeros_like(target.v))
    flow_time = torch.tensor(
        flow_time_values, device=target.h.device, dtype=target.h.dtype
    )
    if flow_time.shape != (target.batch_size,):
        raise ValueError("fixed DDP flow times must match the local batch")
    h_time = broadcast_sample_time(flow_time, target.h, sample_ids=target.abid)
    v_time = broadcast_sample_time(flow_time, target.v, sample_ids=target.abid)
    interpolated = target.with_features(
        source.h + h_time * (target.h - source.h),
        source.v + v_time * (target.v - source.v),
    )
    velocity = target.with_features(target.h - source.h, target.v - source.v)
    trainer.model.train()
    output = trainer.model(
        interpolated,
        context=batch.observed_context,
        query=batch.query,
        flow_time=flow_time,
        clean_future=batch.target_future,
        decode_generated=True,
        decode_clean=True,
        decode_near=True,
    )
    print(f"[{label}] forward complete", flush=True)
    objective = FrameRectifiedFlowObjective()
    losses = frame_joint_loss(
        output,
        objective.loss(output.velocity, velocity),
        batch,
        flow_time,
        stage="joint",
        config=trainer.loss_config,
    )
    print(f"[{label}] loss complete", flush=True)
    trainer.optimizer.zero_grad(set_to_none=True)
    scaled = trainer._distributed_total(losses)
    scaled.backward()
    print(f"[{label}] backward complete", flush=True)
    active = [parameter for parameter in trainer.base_model.parameters() if parameter.requires_grad]
    torch.nn.utils.clip_grad_norm_(active, trainer.grad_clip)
    trainer.optimizer.step()
    print(f"[{label}] optimizer complete", flush=True)
    trainer.step += 1
    trainer.successful_updates += 1
    return float(losses.total.detach())


def _cpu_state(model: torch.nn.Module) -> dict[str, Tensor]:
    base = model.module if hasattr(model, "module") else model
    return {name: value.detach().cpu().clone() for name, value in base.state_dict().items()}


def _state_max_abs(model: torch.nn.Module, expected: Mapping[str, Tensor]) -> float:
    base = model.module if hasattr(model, "module") else model
    maximum = 0.0
    for name, value in base.named_parameters():
        if not value.requires_grad:
            continue
        current = value.detach().cpu()
        reference = expected[name]
        if value.dtype.is_floating_point:
            maximum = max(maximum, float((current - reference).abs().max()))
        elif not torch.equal(current, reference):
            return float("inf")
    return maximum


def _mark(rank: int, message: str) -> None:
    print(f"[rank{rank}] {message}", flush=True)


def _optimizer_max_abs(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    if left["param_groups"] != right["param_groups"] or set(left["state"]) != set(right["state"]):
        return float("inf")
    maximum = 0.0
    for parameter_id, left_slot in left["state"].items():
        right_slot = right["state"][parameter_id]
        if set(left_slot) != set(right_slot):
            return float("inf")
        for name, left_value in left_slot.items():
            right_value = right_slot[name]
            if isinstance(left_value, Tensor):
                if left_value.dtype.is_floating_point:
                    maximum = max(
                        maximum,
                        float((left_value.cpu() - right_value.cpu()).abs().max()),
                    )
                elif not torch.equal(left_value.cpu(), right_value.cpu()):
                    return float("inf")
            elif left_value != right_value:
                return float("inf")
    return maximum


def _load(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    return load_frame_joint_inference(
        args.checkpoint,
        expected_sha256=args.checkpoint_sha256,
        codec_path=args.codec,
        codec_sha256=args.codec_sha256,
        device=device,
    )


def main() -> int:
    args = _arguments()
    world = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if world != 2:
        raise ValueError("this check must run under torchrun --nproc-per-node=2")
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", init_method="env://", device_id=torch.device(f"cuda:{local_rank}"))
    device = configure_device(f"cuda:{local_rank}", deterministic=True)
    loaded = _load(args, device)
    _mark(rank, "loaded source checkpoint")
    model = loaded["model"]
    contracts = loaded["payload"]["contracts"]
    dataset = ClipMMapDataset(args.train_store)
    try:
        records = [dataset[0], dataset[1]]
        local_batch = collate_clip_records([records[rank]])
        local = prepare_frame_joint_batch(
            model.target_teacher,
            local_batch,
            device=device,
            normalizer=model,
            history_frames=8,
        )
        initial_state = _cpu_state(model)
        _mark(rank, "prepared local sample")
        global_batch = collate_clip_records(records)
        global_prepared = prepare_frame_joint_batch(
            model.target_teacher,
            global_batch,
            device=device,
            normalizer=model,
            history_frames=8,
        )
        reference_trainer = _trainer(model, contracts)
        reference_loss = _fixed_step(
            reference_trainer,
            global_prepared,
            label=f"rank{rank}-reference",
            flow_time_values=(0.95, 0.50),
        )
        unused = [
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and parameter.grad is None
        ]
        _mark(rank, f"single-GPU unused trainable parameters: {unused}")
        _mark(rank, "finished single-GPU reference update")
        reference_state = _cpu_state(model)
        del global_prepared, reference_trainer
        model.load_state_dict(initial_state, strict=True)
        torch.cuda.empty_cache()
        dist.barrier(device_ids=[local_rank])
        _mark(rank, "starting DDP update")

        wrapped = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            find_unused_parameters=True,
        )
        trainer = _trainer(wrapped, contracts)
        local_loss = _fixed_step(
            trainer,
            local,
            label=f"rank{rank}-ddp",
            flow_time_values=((0.95,) if rank == 0 else (0.50,)),
        )
        torch.cuda.synchronize(device)
        _mark(rank, "finished DDP update")
        update_error = _state_max_abs(wrapped, reference_state)
        _mark(rank, "finished update comparison")

        generator = torch.Generator(device=device).manual_seed(260920 + rank)
        rank_state = {
            "cursor": {"epoch": 0, "batch_index": rank + 1},
            "rng_state": capture_rng_state(),
            "training_generator_state": generator.get_state().detach().cpu(),
        }
        gathered: list[Any] = [None, None]
        dist.all_gather_object(gathered, rank_state)
        checkpoint = args.output / "ddp_resume_step_00000001.pt"
        if rank == 0:
            args.output.mkdir(parents=True, exist_ok=True)
            trainer.save_checkpoint(
                checkpoint,
                cursor={"epoch": 0, "batch_index": 1},
                generator=generator,
                rank_states={str(index): state for index, state in enumerate(gathered)},
            )
        dist.barrier(device_ids=[local_rank])
        checkpoint_sha = sha256_file(checkpoint)
        before_model = _cpu_state(wrapped)
        before_optimizer = trainer.optimizer.state_dict()
        _mark(rank, "saved resume checkpoint")

        del trainer, wrapped, model
        torch.cuda.empty_cache()
        resumed_loaded = _load(args, device)
        resumed_model = resumed_loaded["model"]
        resumed_wrapped = DistributedDataParallel(
            resumed_model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            find_unused_parameters=True,
        )
        resumed = _trainer(resumed_wrapped, contracts)
        resumed_generator = torch.Generator(device=device)
        resumed.load_checkpoint(
            checkpoint,
            generator=resumed_generator,
            expected_sha256=checkpoint_sha,
            rank=rank,
        )
        resume_model_error = _state_max_abs(resumed_wrapped, before_model)
        resume_optimizer_error = _optimizer_max_abs(
            resumed.optimizer.state_dict(), before_optimizer
        )
        generator_equal = torch.equal(
            resumed_generator.get_state().cpu(), generator.get_state().cpu()
        )
        local_result = {
            "rank": rank,
            "local_loss": local_loss,
            "reference_loss": reference_loss,
            "single_vs_ddp_model_max_abs": update_error,
            "resume_model_max_abs": resume_model_error,
            "resume_optimizer_max_abs": resume_optimizer_error,
            "generator_state_equal": generator_equal,
        }
        results: list[Any] = [None, None]
        dist.all_gather_object(results, local_result)
        _mark(rank, "finished strict resume comparison")
        if rank == 0:
            output = {
                "schema_version": "molvid.frame_joint.ddp_check.v1",
                "world_size": 2,
                "global_batch_sample_ids": [
                    str(records[0]["sample_id"]), str(records[1]["sample_id"])
                ],
                "global_flow_times": [0.95, 0.50],
                "threshold_applicability_differs_by_rank": True,
                "single_gpu_loss": results[0]["reference_loss"],
                "ddp_rank_losses": [row["local_loss"] for row in results],
                "single_vs_ddp_model_max_abs": max(
                    row["single_vs_ddp_model_max_abs"] for row in results
                ),
                "resume_model_max_abs": max(row["resume_model_max_abs"] for row in results),
                "resume_optimizer_max_abs": max(row["resume_optimizer_max_abs"] for row in results),
                "per_rank_generator_state_equal": all(
                    row["generator_state_equal"] for row in results
                ),
                "resume_checkpoint_sha256": checkpoint_sha,
                "tolerance": 2.0e-6,
                "passed": bool(
                    all(
                        row["single_vs_ddp_model_max_abs"] <= 2.0e-6
                        for row in results
                    )
                    and all(row["resume_model_max_abs"] == 0 for row in results)
                    and all(row["resume_optimizer_max_abs"] == 0 for row in results)
                    and all(row["generator_state_equal"] for row in results)
                ),
            }
            atomic_write_json(args.output / "ddp_check.json", output)
            print(json.dumps(output, sort_keys=True))
            if not output["passed"]:
                raise RuntimeError("Frame Joint DDP/resume check failed")
        dist.barrier(device_ids=[local_rank])
        return 0
    finally:
        dataset.close()
        dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
