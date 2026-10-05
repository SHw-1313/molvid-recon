"""Fixed-final full-model coordinate RF with the paired source-run data schedule."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import shutil
import time
import numpy as np
import torch
from molvid.cli.train_frame_joint import _open_data, _sampler
from molvid.data.batch import collate_clip_records
from molvid.flow.coordinate import CoordinateFlowModel
from molvid.provenance import freeze_run_identity
from molvid.runtime import atomic_write_json, configure_device, sha256_file, canonical_hash
from molvid.training.coordinate import CoordinateTrainConfig, make_coordinate_optimizer, coordinate_update, capture_coordinate_rng, restore_coordinate_rng
from molvid.training.execution import execution_schedule, execution_manifest
from molvid.training.execution_evidence import ExecutionEvidence
from molvid.training.predictive_data import load_parent, atomic_torch_save, group_identity
from tools.inspect_model import model_summary


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    device = configure_device("cuda", deterministic=True)
    torch.cuda.set_per_process_memory_fraction(.88)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    raw = json.loads(args.config.read_text())
    config = CoordinateTrainConfig(checkpoint_blocks=args.checkpoint_blocks)
    if not 1 <= args.max_steps <= config.total_updates:
        raise ValueError("max_steps must lie within the fixed2048 schedule")
    parent, _, release = load_parent(device)
    model = CoordinateFlowModel(parent).train()
    del parent
    optimizer = make_coordinate_optimizer(model, config)
    generator = torch.Generator(device=device).manual_seed(args.seed + 8000)
    train, _, data_hash, _, _, close = _open_data(raw["data"])
    try:
        sampler = _sampler(train, raw["training"], seed=args.seed, rank=0, world=1)
        entries = execution_schedule(sampler.global_batches, sampler.specs, histories=(4, 8), seed=args.seed, epoch=0)
        if len(entries) < 2048:
            raise ValueError("insufficient scheduled updates")
        manifest = execution_manifest(entries)
        specs = {s.index: s for s in sampler.specs}
        systems = sorted({group_identity(specs[i].sample_id)[0] for e in entries[:2048] for i in e.indices})
        if len(systems) != 192:
            raise ValueError("registered2048 schedule must cover all192 systems")
        contract = dict(schema="molvid.coordinate_flow.training.v1", model=model.contract(),
            training=asdict(config), data_hash=data_hash, parent=release["selected_checkpoint"], seed=args.seed,
            prefix2048_sha256=canonical_hash(manifest["entries"][:2048]), execution_sha256=manifest["sha256"],
            input_config_sha256=sha256_file(args.config), precision="float32", objective="coordinate_RF_only",
            static_updates=False, sampled_physical_auxiliary=False)
        runtime_root = Path(__file__).resolve().parents[1] / "molvid"
        contract["runtime_source_sha256"] = {str(p.relative_to(runtime_root)): sha256_file(p)
            for p in sorted(runtime_root.rglob("*.py")) if "evaluation" not in p.parts}
        contract["largest_query_atom_frame_profile_only"] = args.profile_largest
        if args.profile_largest:
            if args.max_steps != 1 or args.resume:
                raise ValueError("largest-batch profile is a separate one-update disposable run")
            # Query activations dominate this model. H4 has12 future frames,
            # H8 only8; equal nominal16-frame cost is not equal memory cost.
            largest = max(entries[:2048], key=lambda e: sum(
                specs[i].atoms * (specs[i].frames-e.history_frames) for i in e.indices))
            entries = (largest,)
        cursor = 0
        inputs = [args.config, Path(release["selected_checkpoint"]["path"])]
        if args.resume:
            inputs.append(args.resume)
            if sha256_file(args.resume) != args.resume_sha256:
                raise ValueError("resume SHA mismatch")
            saved = torch.load(args.resume, map_location=device, weights_only=False)
            if saved["contract"] != contract:
                raise ValueError("resume contract mismatch")
            model.load_state_dict(saved["model"], strict=True)
            optimizer.load_state_dict(saved["optimizer"])
            restore_coordinate_rng(saved["rng"], generator)
            cursor = saved["cursor"]
            if not 0 <= cursor < args.max_steps:
                raise ValueError("resume cursor outside requested continuation")
            del saved
        freeze_run_identity(args.output, inputs=inputs, purpose="Full-model192 coordinate-state RF",
            protocol=dict(contract=contract, systems=systems, warm_start=model.warm_start_report,
                cursor_start=cursor, max_steps=args.max_steps, final_step=2048, test_opened=False,
                trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                comparison="same data/H schedule as source unit/v02, but architecture/objective/precision/readout LR differ; not single-factor causality"))
        atomic_write_json(args.output / "contract.json", contract)
        atomic_write_json(args.output / "model_inspection.json", model_summary(model))
        started = time.monotonic()
        with ExecutionEvidence(args.output), (args.output / "updates.jsonl").open("x") as log:
            for index in range(cursor, args.max_steps):
                if time.monotonic() - started > 21600:
                    raise TimeoutError("six-hour execution budget")
                if shutil.disk_usage(args.output).free < 8 * 2**30:
                    raise RuntimeError("8GiB disk reserve reached")
                entry = entries[index]
                batch = collate_clip_records([train[i] for i in entry.indices])
                row = coordinate_update(model, optimizer, batch, history_frames=entry.history_frames,
                                        generator=generator, step=index, config=config)
                row.update(step=index + 1, seconds=time.monotonic() - started,
                           execution_sha256=manifest["sha256"], peak_cuda_bytes=torch.cuda.max_memory_allocated())
                log.write(json.dumps(row) + "\n"); log.flush()
                print(json.dumps(row), flush=True)
                if not args.profile_largest and index + 1 in (16, 1024, 2048, args.max_steps):
                    atomic_torch_save(args.output / f"coordinate_step_{index+1:04d}.pt", dict(
                        contract=contract, model=model.state_dict(), optimizer=optimizer.state_dict(),
                        cursor=index + 1, rng=capture_coordinate_rng(generator)))
            atomic_write_json(args.output / "summary.json", dict(status="completed", cursor_start=cursor,
                cursor_end=args.max_steps, updates_this_execution=args.max_steps - cursor,
                seconds=time.monotonic() - started, test_opened=False, checkpoint_selection="fixed_final_only",
                static_updates=0, test_quality_available=False))
    finally:
        close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, choices=(71, 72), required=True)
    p.add_argument("--max-steps", type=int, default=2048)
    p.add_argument("--resume", type=Path)
    p.add_argument("--resume-sha256")
    p.add_argument("--profile-largest", action="store_true")
    p.add_argument("--checkpoint-blocks", action="store_true")
    run(p.parse_args())
