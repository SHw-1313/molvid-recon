"""Fixed-final full-model coordinate RF with the paired source-run data schedule."""
import argparse
from contextlib import ExitStack
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
from molvid.dit.capacity import validate_coordinate_depth
from molvid.training.coordinate_budget import resolve_coordinate_budget
from molvid.provenance import freeze_run_identity
from molvid.runtime import atomic_write_json, configure_device, sha256_file, canonical_hash
from molvid.training.coordinate import CoordinateTrainConfig, make_coordinate_optimizer, coordinate_update, capture_coordinate_rng, restore_coordinate_rng
from molvid.training.execution import execution_schedule, execution_manifest
from molvid.training.execution_evidence import ExecutionEvidence
from molvid.training.predictive_data import load_parent, atomic_torch_save, group_identity
from tools.inspect_model import model_summary
from tools.evaluate_expanded_native import require_final_coordinate_checkpoint
from molvid.training.data_scale_input import expanded_training_input, validate_reference, require_cuda_gate


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    raw = json.loads(args.config.read_text())
    if 'coordinate_data_scale' in raw:
        validate_reference(raw)
        require_cuda_gate(raw['coordinate_data_scale'])
    device = configure_device("cuda", deterministic=True)
    torch.cuda.set_per_process_memory_fraction(.88)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    settings = dict(raw.get('coordinate_training', {}))
    if args.checkpoint_blocks:
        settings['checkpoint_blocks'] = True
    config = CoordinateTrainConfig(**settings)
    total_updates = resolve_coordinate_budget(raw)
    if args.max_steps is None:
        args.max_steps = total_updates
    depth = validate_coordinate_depth(raw.get('coordinate_dit_depth', 4))
    offset = raw.get('coordinate_schedule_offset', 0)
    if type(offset) is not int or offset not in (0,2048):
        raise ValueError('only original or next2048 schedule segment supported')
    if not 1 <= args.max_steps <= config.total_updates:
        raise ValueError("max_steps must lie within the fixed2048 schedule")
    if depth == 8 and (offset != 2048 or not raw.get('coordinate_initialization')
            or raw.get('coordinate_source_block_correlation', 0.) != 0.
            or raw.get('coordinate_source_sigma_angstrom', 1.) != 1.
            or raw.get('coordinate_local_geometry_input', 'observed') != 'observed'
            or config.handedness_weight != 0.):
        raise ValueError('depth expansion requires iid1 observed-geometry RF-only parent continuation')
    parent, _, release = load_parent(device)
    model = CoordinateFlowModel(parent).train()
    del parent
    initialization = raw.get('coordinate_initialization')
    initialization_inputs = []
    if initialization is not None:
        path = Path(initialization['checkpoint'])
        if sha256_file(path) != initialization['sha256']:
            raise ValueError('coordinate warm-start SHA mismatch')
        saved = torch.load(path,map_location='cpu',weights_only=False)
        require_final_coordinate_checkpoint(saved)
        if saved['contract']['model'] != model.contract() or saved['contract']['seed'] != args.seed:
            raise ValueError('coordinate warm-start model/seed mismatch')
        original_runtime = Path(initialization['source_root'])/'molvid'
        for name,digest in saved['contract']['runtime_source_sha256'].items():
            if sha256_file(original_runtime/name) != digest:
                raise ValueError(f'original parent runtime changed: {name}')
        model.load_state_dict(saved['model'],strict=True)
        model.warm_start_report = dict(kind='coordinate_model_only_fresh_optimizer',
            parent=str(path),parent_sha256=initialization['sha256'],
            reused_state_keys=sorted(saved['model']),new_state_keys=[],retired_state_keys=[])
        initialization_inputs.append(path)
        del saved
    if (offset or config.handedness_weight > 0) and initialization is None:
        raise ValueError('new chirality phase requires an explicit coordinate parent')
    if depth == 8:
        model.expand_depth()
    query_geometry = raw.get('coordinate_train_query_geometry', False)
    if type(query_geometry) is not bool:
        raise ValueError('coordinate_train_query_geometry must be boolean')
    if query_geometry:
        if initialization is None or offset != 2048 or config.handedness_weight != 0:
            raise ValueError('GQ requires next2048 coordinate continuation without chirality')
        model.enable_query_geometry()
        model.warm_start_report.update(kind='coordinate_parent_plus_copied_query_geometry_fresh_optimizer',
            new_state_keys=model.warm_start_report['new_state_keys'] + sorted(k for k in model.state_dict() if k.startswith('query_encoder.')),
            query_geometry_initialization='exact copy of loaded frozen teacher; no RNG draw')
    model.set_query_geometry_recompute(raw.get('coordinate_query_geometry_recompute', False))
    optimizer = make_coordinate_optimizer(model, config)
    flow_seed = args.seed + (18000 if initialization else 8000)
    generator = torch.Generator(device=device).manual_seed(flow_seed)
    train, _, data_hash, _, _, close = _open_data(raw["data"])
    expanded_handles = ExitStack()
    try:
        sampler = _sampler(train, raw["training"], seed=args.seed, rank=0, world=1)
        entries = execution_schedule(sampler.global_batches, sampler.specs, histories=(4, 8), seed=args.seed, epoch=0)
        if len(entries) < offset+total_updates:
            raise ValueError("insufficient scheduled updates")
        manifest = execution_manifest(entries)
        entries = entries[offset:offset+total_updates]
        selected_manifest = execution_manifest(entries)
        specs = {s.index: s for s in sampler.specs}
        expansion = None
        data_inputs = []
        if 'coordinate_data_scale' in raw:
            train, expanded_specs, entries, selected_manifest, data_hash, expansion, data_inputs = expanded_handles.enter_context(
                expanded_training_input(train, data_hash, raw, args.seed, entries))
            specs = {s.index:s for s in expanded_specs}
            # Candidate is a registered selected calendar, not a fictitious
            # offset slice into a newly sampled full-data stream.
            manifest = selected_manifest
        systems = sorted({group_identity(specs[i].sample_id)[0] for e in entries for i in e.indices})
        if len(systems) != (512 if expansion else 192):
            raise ValueError("registered schedule must cover the complete training inventory")
        contract = dict(schema="molvid.coordinate_flow.training.v1", model=model.contract(),
            training=asdict(config), data_hash=data_hash, parent=release["selected_checkpoint"], seed=args.seed,
            prefix2048_sha256=canonical_hash(selected_manifest['entries']), execution_sha256=manifest["sha256"],
            schedule_offset=offset, flow_seed=flow_seed, coordinate_initialization=initialization,
            input_config_sha256=sha256_file(args.config), precision="float32",
            objective='coordinate_RF_plus_actual_source_CA_handedness' if config.handedness_weight > 0 else 'coordinate_RF_only',
            static_updates=False, sampled_physical_auxiliary=config.handedness_weight > 0)
        if expansion:
            contract['data_scale_experiment'] = expansion
            contract['reference_schedule_offset'] = offset
            contract['schedule_offset_semantics'] = 'reference192 calendar only; candidate uses explicit registered entries'
            contract['candidate_selected_schedule_offset'] = 0
        if total_updates == 8192:
            contract['selected_schedule_sha256'] = contract.pop('prefix2048_sha256')
            contract['coordinate_budget_experiment'] = dict(raw['coordinate_budget_experiment'])
        runtime_root = Path(__file__).resolve().parents[1] / "molvid"
        contract["runtime_source_sha256"] = {str(p.relative_to(runtime_root)): sha256_file(p)
            for p in sorted(runtime_root.rglob("*.py")) if "evaluation" not in p.parts}
        contract["largest_query_atom_frame_profile_only"] = args.profile_largest
        if args.profile_largest:
            if args.max_steps != 1 or args.resume:
                raise ValueError("largest-batch profile is a separate one-update disposable run")
            # Query activations dominate this model. H4 has12 future frames,
            # H8 only8; equal nominal16-frame cost is not equal memory cost.
            largest = max(entries, key=lambda e: sum(
                specs[i].atoms * (specs[i].frames-e.history_frames) for i in e.indices))
            entries = (largest,)
            if config.handedness_weight > 0 and config.handedness_every != 1:
                raise ValueError('auxiliary largest-profile must explicitly use cadence1 to exercise the loss')
        cursor = 0
        inputs = [args.config, Path(release["selected_checkpoint"]["path"]), *initialization_inputs, *data_inputs]
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
        freeze_run_identity(args.output, inputs=inputs, purpose=f"Full-model{len(systems)} coordinate-state RF",
            protocol=dict(contract=contract, systems=systems, warm_start=model.warm_start_report,
                cursor_start=cursor, max_steps=args.max_steps, final_step=total_updates, test_opened=False,
                trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                comparison=(expansion['interpretation'] if expansion else 'query4/8 versus frozen-query long controls: same data and long LR; trainable query geometry and compute differ; versus short changes both budget and LR'
                    if total_updates == 8192 else 'paired chirality/control share coordinate parent,data,H,main RNG,LR and update budget; extra auxiliary compute differs'
                    if initialization else 'same data/H schedule as source unit/v02; architecture/objective/precision/readout LR differ')))
        atomic_write_json(args.output / "contract.json", contract)
        atomic_write_json(args.output / "model_inspection.json", model_summary(model))
        started = time.monotonic()
        with ExecutionEvidence(args.output), (args.output / "updates.jsonl").open("x") as log:
            for index in range(cursor, args.max_steps):
                if time.monotonic() - started > (43200 if total_updates == 8192 else 21600):
                    raise TimeoutError("six-hour execution budget")
                if shutil.disk_usage(args.output).free < 8 * 2**30:
                    raise RuntimeError("8GiB disk reserve reached")
                entry = entries[index]
                records = [train[i] for i in entry.indices]
                batch = collate_clip_records(records)
                row = coordinate_update(model, optimizer, batch, history_frames=entry.history_frames,
                    generator=generator, step=index, config=config,
                    atom_identity=[identity for record in records for identity in record['atom_identity']])
                row.update(step=index + 1, seconds=time.monotonic() - started,
                           schedule_index=(0 if expansion else offset)+index, execution_sha256=manifest["sha256"],
                           peak_cuda_bytes=torch.cuda.max_memory_allocated())
                if expansion:
                    row.update(reference_schedule_index=offset+index,
                               candidate_global_batch_index=entry.global_batch_index)
                log.write(json.dumps(row) + "\n"); log.flush()
                print(json.dumps(row), flush=True)
                checkpoint_steps = ((16,2048,4096,8192,args.max_steps) if total_updates == 8192 else
                    (16,2048,args.max_steps) if initialization else (16,1024,2048,args.max_steps))
                if not args.profile_largest and index + 1 in checkpoint_steps:
                    atomic_torch_save(args.output / f"coordinate_step_{index+1:04d}.pt", dict(
                        contract=contract, model=model.state_dict(), optimizer=optimizer.state_dict(),
                        cursor=index + 1, rng=capture_coordinate_rng(generator)))
            atomic_write_json(args.output / "summary.json", dict(status="completed", cursor_start=cursor,
                cursor_end=args.max_steps, updates_this_execution=args.max_steps - cursor,
                seconds=time.monotonic() - started, test_opened=False, checkpoint_selection="fixed_final_only",
                static_updates=0, test_quality_available=False))
    finally:
        expanded_handles.close()
        close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, choices=(71, 72), required=True)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--resume", type=Path)
    p.add_argument("--resume-sha256")
    p.add_argument("--profile-largest", action="store_true")
    p.add_argument("--checkpoint-blocks", action="store_true")
    run(p.parse_args())
