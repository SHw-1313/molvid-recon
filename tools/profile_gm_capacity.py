"""Profile matched C4/C8 training strata, parameters, FLOPs, memory, and cost."""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
import math
from pathlib import Path
import re
import time

import torch
from torch.utils.flop_counter import FlopCounterMode

from molvid.cli.train_frame_joint import _open_data, _sampler
from molvid.data.batch import collate_clip_records
from molvid.flow.objective import FrameRectifiedFlowObjective
from molvid.runtime import atomic_write_json, configure_device, seed_all
from molvid.training.batches import prepare_frame_joint_batch
from molvid.training.dit import module_state_hash
from molvid.training.execution import execution_schedule
from tools.gm_capacity_common import (
    CAPACITY_DEPTHS,
    build_capacity_trainer,
    load_capacity_release,
    load_parent_model,
    parameter_ledger,
)


SEED = 20260928
FORMAL_GPU_HOURS = 32.0
PILOT_UPDATES = 128
FORMAL_CHECKPOINT_GRANULARITY = 1000


def _representatives(entries):
    result = {}
    for entry in entries:
        result.setdefault((entry.view, entry.time_bucket_id, entry.history_frames), entry)
    return list(result.values())


def _stress_representatives(entries, specs):
    """Choose the largest packed atom-frame batch in each fixed stratum."""

    by_index = {item.index: item for item in specs}
    result = {}
    scores = {}
    for entry in entries:
        key = (entry.view, entry.time_bucket_id, entry.history_frames)
        score = sum(
            int(by_index[index].atoms) * int(by_index[index].frames)
            for index in entry.indices
        )
        if key not in result or score > scores[key]:
            result[key] = entry
            scores[key] = score
    return list(result.values())


def _module_flops(counts, depth: int) -> dict[str, object]:
    serialized = {
        str(module): {str(operator): int(value) for operator, value in operators.items()}
        for module, operators in counts.items()
    }
    block_flops = {}
    for index in range(depth):
        by_component = {}
        pattern = re.compile(rf"(?:^|\.)dit\.blocks\.{index}\.([^.]+)$")
        for module, operators in serialized.items():
            match = pattern.search(module)
            if match is not None:
                by_component[match.group(1)] = sum(operators.values())
        block_flops[str(index + 1)] = {
            "supported_operator_flops": sum(by_component.values()),
            "components": dict(sorted(by_component.items())),
        }
    return {
        "definition": "torch.utils.flop_counter supported operators on one real DiT forward",
        "per_block": block_flops,
        "raw_module_counts": serialized,
    }


def finalize_budget(evidence: dict[str, object]) -> dict[str, object]:
    arms = evidence["arms"]
    updates = {int(value["updates_per_effective_epoch"]) for value in arms.values()}
    if len(updates) != 1:
        raise RuntimeError("C4/C8 profiles disagree on updates per effective epoch")
    updates_per_epoch = updates.pop()
    pair_seconds_per_update = sum(
        float(value["weighted_seconds_per_update"]) for value in arms.values()
    )
    pair_per_epoch = sum(
        float(value["projected_gpu_hours_per_effective_epoch"])
        for value in arms.values()
    )
    projected_two_epochs = 1.2 * 2.0 * pair_per_epoch
    projected_one_epoch = 1.2 * pair_per_epoch
    if projected_two_epochs <= FORMAL_GPU_HOURS:
        formal_updates = 2 * updates_per_epoch
        effective_epochs = 2.0
        adjustment = None
    elif projected_one_epoch <= FORMAL_GPU_HOURS:
        formal_updates = updates_per_epoch
        effective_epochs = 1.0
        adjustment = {
            "reason": "two matched epochs exceed the 32 GPU-hour formal cap",
            "paired_symmetric": True,
        }
    else:
        maximum = math.floor(
            FORMAL_GPU_HOURS * 3600.0 / (1.2 * pair_seconds_per_update)
        )
        formal_updates = (
            maximum // FORMAL_CHECKPOINT_GRANULARITY
        ) * FORMAL_CHECKPOINT_GRANULARITY
        if formal_updates < 512:
            raise RuntimeError(
                "the 32 GPU-hour cap cannot support a meaningful matched capacity run"
            )
        effective_epochs = formal_updates / updates_per_epoch
        adjustment = {
            "reason": "one matched effective epoch exceeds the 32 GPU-hour formal cap",
            "paired_symmetric": True,
            "checkpoint_granularity": FORMAL_CHECKPOINT_GRANULARITY,
            "unrounded_maximum_updates": maximum,
            "selected_updates_per_arm": formal_updates,
            "selected_effective_epoch_fraction": effective_epochs,
        }
    projected_selected = (
        1.2 * pair_seconds_per_update * formal_updates / 3600.0
    )
    evidence.update({
        "runtime_headroom_factor": 1.2,
        "projected_pair_gpu_hours_one_epoch": projected_one_epoch,
        "projected_pair_gpu_hours_two_epochs": projected_two_epochs,
        "chosen_effective_epochs": effective_epochs,
        "formal_updates_per_arm": formal_updates,
        "projected_selected_formal_gpu_hours": projected_selected,
        "budget_adjustment": adjustment,
        "measured_profile_update_gpu_hours": sum(
            float(row["input_seconds"])
            + float(row["base"]["seconds"])
            + float(row["auxiliary"]["seconds"])
            for value in arms.values()
            for row in value["representative_rows"]
        ) / 3600.0,
    })
    evidence["passed"] = True
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-atom-frames", type=int, default=30000)
    parser.add_argument("--finalize-measurements", type=Path)
    args = parser.parse_args()
    if args.max_atom_frames < 1000:
        raise ValueError("max-atom-frames is implausibly small")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.finalize_measurements is not None:
        bundle = load_capacity_release(args.release)
        evidence = json.loads(args.finalize_measurements.read_text(encoding="utf-8"))
        if (
            evidence.get("schema") != "molvid.capacity.profile.v1"
            or evidence.get("parent_checkpoint_sha256") != bundle.checkpoint_sha256
            or set(evidence.get("arms", {})) != {"C4", "C8"}
        ):
            raise ValueError("profile measurements are incomplete or use a different parent")
        finalize_budget(evidence)
        atomic_write_json(args.output / "profile.json", evidence)
        print(json.dumps({
            "formal_updates_per_arm": evidence["formal_updates_per_arm"],
            "chosen_effective_epochs": evidence["chosen_effective_epochs"],
            "projected_selected_formal_gpu_hours": evidence["projected_selected_formal_gpu_hours"],
        }, sort_keys=True))
        return
    started = time.time()
    device = configure_device("cuda", deterministic=True)
    bundle = load_capacity_release(args.release)
    raw = copy.deepcopy(bundle.configuration)
    raw["training"]["max_atom_frames_per_gpu"] = args.max_atom_frames
    train, _, data_hash, _, _, close_data = _open_data(raw["data"])
    parent = load_parent_model(bundle, device=device)
    evidence: dict[str, object] = {
        "schema": "molvid.capacity.profile.v1",
        "parent_checkpoint_sha256": bundle.checkpoint_sha256,
        "data_hash": data_hash,
        "max_atom_frames_per_gpu": args.max_atom_frames,
        "pilot_updates_per_arm": PILOT_UPDATES,
        "arms": {},
        "test_opened": False,
    }
    try:
        sampler = _sampler(train, raw["training"], seed=SEED, rank=0, world=1)
        sampler.set_epoch(0)
        entries = execution_schedule(
            sampler.global_batches,
            sampler.specs,
            histories=(4, 8),
            seed=SEED,
            epoch=0,
        )
        representatives = _representatives(entries)
        stress_representatives = _stress_representatives(entries, sampler.specs)
        frequencies = Counter(
            (entry.view, entry.time_bucket_id, entry.history_frames) for entry in entries
        )
        if len(representatives) != 9 or len(stress_representatives) != 9:
            raise RuntimeError("capacity profile does not cover all nine data/H strata")
        for depth in CAPACITY_DEPTHS:
            seed_all(SEED)
            trainer, warm = build_capacity_trainer(
                bundle,
                parent,
                depth=depth,
                device=device,
                schedule_contract={"schema": "molvid.capacity.profile.v1"},
            )
            model = trainer.base_model
            teacher_before = module_state_hash(model.target_teacher)
            rows = []
            stress_rows = []
            main_generator = torch.Generator(device=device).manual_seed(SEED + 1000)
            sampled_generator = torch.Generator(device=device).manual_seed(SEED + 2000)

            flop_entry = representatives[0]
            flop_cpu = collate_clip_records([train[index] for index in flop_entry.indices])
            flop_batch = prepare_frame_joint_batch(
                model.target_teacher,
                flop_cpu,
                device=device,
                normalizer=model,
                history_frames=flop_entry.history_frames,
            )
            draw = FrameRectifiedFlowObjective().sample(
                flop_batch.normalized_target,
                flop_batch.source_center,
                generator=torch.Generator(device=device).manual_seed(SEED + 3000),
            )
            model.eval()
            with torch.no_grad(), FlopCounterMode(model, depth=16, display=False) as counter:
                model(
                    draw.interpolated,
                    context=flop_batch.observed_context,
                    query=flop_batch.query,
                    flow_time=draw.flow_time,
                )
            flops = _module_flops(counter.get_flop_counts(), depth)
            del draw, flop_batch, flop_cpu

            for entry in representatives:
                input_started = time.time()
                cpu = collate_clip_records([train[index] for index in entry.indices])
                batch = prepare_frame_joint_batch(
                    model.target_teacher,
                    cpu,
                    device=device,
                    normalizer=model,
                    history_frames=entry.history_frames,
                )
                torch.cuda.synchronize(device)
                input_seconds = time.time() - input_started
                timings = {}
                for auxiliary in (False, True):
                    torch.cuda.reset_peak_memory_stats(device)
                    torch.cuda.synchronize(device)
                    update_started = time.time()
                    metrics = trainer.train_step(
                        batch,
                        generator=main_generator,
                        sampled_generator=sampled_generator,
                        auxiliary_active=auxiliary,
                    )
                    torch.cuda.synchronize(device)
                    timings["auxiliary" if auxiliary else "base"] = {
                        "seconds": time.time() - update_started,
                        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                        "loss": metrics["loss"],
                        "grad_norm": metrics["grad_norm"],
                    }
                row = {
                    "entry": entry.__dict__,
                    "input_seconds": input_seconds,
                    **timings,
                }
                rows.append(row)
                atomic_write_json(args.output / f"C{depth}_progress.json", {"rows": rows})
                print(json.dumps({
                    "depth": depth,
                    "stratum": [entry.view, entry.time_bucket_id, entry.history_frames],
                    "base_seconds": timings["base"]["seconds"],
                    "auxiliary_seconds": timings["auxiliary"]["seconds"],
                    "peak_allocated_bytes": max(
                        timings["base"]["peak_allocated_bytes"],
                        timings["auxiliary"]["peak_allocated_bytes"],
                    ),
                }, sort_keys=True), flush=True)
            for entry in stress_representatives:
                cpu = collate_clip_records([train[index] for index in entry.indices])
                batch = prepare_frame_joint_batch(
                    model.target_teacher,
                    cpu,
                    device=device,
                    normalizer=model,
                    history_frames=entry.history_frames,
                )
                timings = {}
                for auxiliary in (False, True):
                    torch.cuda.reset_peak_memory_stats(device)
                    torch.cuda.synchronize(device)
                    update_started = time.time()
                    metrics = trainer.train_step(
                        batch,
                        generator=main_generator,
                        sampled_generator=sampled_generator,
                        auxiliary_active=auxiliary,
                    )
                    torch.cuda.synchronize(device)
                    timings["auxiliary" if auxiliary else "base"] = {
                        "seconds": time.time() - update_started,
                        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                        "loss": metrics["loss"],
                        "grad_norm": metrics["grad_norm"],
                    }
                row = {
                    "entry": entry.__dict__,
                    "packed_atom_frames": int(cpu.atom_count * cpu.frames),
                    **timings,
                }
                stress_rows.append(row)
                atomic_write_json(
                    args.output / f"C{depth}_stress_progress.json",
                    {"rows": stress_rows},
                )
                print(json.dumps({
                    "depth": depth,
                    "stress_stratum": [entry.view, entry.time_bucket_id, entry.history_frames],
                    "packed_atom_frames": row["packed_atom_frames"],
                    "peak_allocated_bytes": max(
                        timings["base"]["peak_allocated_bytes"],
                        timings["auxiliary"]["peak_allocated_bytes"],
                    ),
                }, sort_keys=True), flush=True)
            weighted_seconds = 0.0
            for row in rows:
                entry = row["entry"]
                key = (entry["view"], entry["time_bucket_id"], entry["history_frames"])
                weight = frequencies[key] / len(entries)
                auxiliary_fraction = sum(
                    candidate.auxiliary
                    for candidate in entries
                    if (
                        candidate.view,
                        candidate.time_bucket_id,
                        candidate.history_frames,
                    ) == key
                ) / frequencies[key]
                update_seconds = (
                    (1.0 - auxiliary_fraction) * row["base"]["seconds"]
                    + auxiliary_fraction * row["auxiliary"]["seconds"]
                    + row["input_seconds"]
                )
                weighted_seconds += weight * update_seconds
            arm = {
                "warm_start_report": warm.as_dict(),
                "parameters": parameter_ledger(model),
                "flops": flops,
                "representative_rows": rows,
                "stress_rows": stress_rows,
                "checkpoint_flow_field": trainer.checkpoint_flow_field,
                "weighted_seconds_per_update": weighted_seconds,
                "updates_per_effective_epoch": len(entries),
                "projected_gpu_hours_per_effective_epoch": weighted_seconds * len(entries) / 3600.0,
                "projected_pilot_gpu_hours": weighted_seconds * PILOT_UPDATES / 3600.0,
                "peak_allocated_bytes": max(
                    row[kind]["peak_allocated_bytes"]
                    for row in (*rows, *stress_rows)
                    for kind in ("base", "auxiliary")
                ),
                "teacher_unchanged": teacher_before == module_state_hash(model.target_teacher),
            }
            if arm["teacher_unchanged"] is not True:
                raise RuntimeError(f"C{depth} profile changed the frozen teacher")
            evidence["arms"][f"C{depth}"] = arm
            del trainer, model
            torch.cuda.empty_cache()

        finalize_budget(evidence)
        evidence["profile_gpu_hours"] = (time.time() - started) / 3600.0
        atomic_write_json(args.output / "profile.json", evidence)
    finally:
        atomic_write_json(args.output / "profile_partial.json", evidence)
        close_data()


if __name__ == "__main__":
    main()
