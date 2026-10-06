"""Real largest-TRAIN input updates; a memory/resume gate, never efficacy.

The optional frame recomputation changes execution, not RF or input semantics.
No crop, temporal shortening, precision change or CPU model fallback is allowed.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch

from molvid.data.batch import collate_clip_records
from molvid.data.store import ClipMMapDataset
from molvid.flow.coordinate import CoordinateFlowModel
from molvid.provenance import freeze_run_identity
from molvid.runtime import atomic_write_json, configure_device, sha256_file
from molvid.training.coordinate import (CoordinateTrainConfig, make_coordinate_optimizer,
    coordinate_update, capture_coordinate_rng, restore_coordinate_rng)
from molvid.training.execution_evidence import ExecutionEvidence
from molvid.training.predictive_data import load_parent, atomic_torch_save, RELEASE
from tools.evaluate_expanded_native import require_final_coordinate_checkpoint

BASE = Path('/workspace/molvid-geometry-scaling-20261005/runs/geometry_scaling_20261005')
PREFLIGHT = Path('/workspace/molvid-gs-paired-evaluation-20261006/runs/gs_paired_evaluation_20261006/train512_preflight_v1')
RECIPE = Path('/workspace/molvid-query-geometry-training-20261006/runs/query_geometry_training_20261006/recipes_v1/query_geometry_71.json')


def read(path):
    return json.loads(Path(path).read_text())


def state_digest(value):
    """Hash every tensor/array byte, dtype, shape and nested optimizer/RNG field.

    Device placement is intentionally ignored to compare disk-restored states.
    This is SHA256 state equivalence, not a direct torch.equal claim.
    """
    digest = hashlib.sha256()
    def put(data):
        digest.update(len(data).to_bytes(8, 'big')); digest.update(data)
    def visit(item):
        if isinstance(item, torch.Tensor):
            item = item.detach().cpu().contiguous()
            put(repr(('tensor', str(item.dtype), tuple(item.shape))).encode())
            put(item.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, np.ndarray):
            put(repr(('numpy', str(item.dtype), item.shape)).encode())
            put(np.ascontiguousarray(item).tobytes())
        elif isinstance(item, dict):
            put(b'dict'); put(str(len(item)).encode())
            for key in sorted(item, key=lambda k: (type(k).__name__, repr(k))):
                visit(key); visit(item[key])
        elif isinstance(item, (tuple, list)):
            put(type(item).__name__.encode()); put(str(len(item)).encode())
            for child in item: visit(child)
        elif item is None or isinstance(item, (str, bool, int, float, np.generic)):
            put(repr((type(item).__name__, item)).encode())
        else:
            raise TypeError(f'unsupported state type {type(item)}')
    visit(value)
    return digest.hexdigest()


def finite_state(value):
    if isinstance(value, torch.Tensor):
        if (value.is_floating_point() or value.is_complex()) and not bool(torch.isfinite(value).all()):
            raise FloatingPointError('nonfinite model/optimizer tensor')
    elif isinstance(value, dict):
        for item in value.values(): finite_state(item)
    elif isinstance(value, (tuple, list)):
        for item in value: finite_state(item)
    elif isinstance(value, float) and not np.isfinite(value):
        raise FloatingPointError('nonfinite optimizer scalar')


def input_record(preflight=PREFLIGHT, input_size='maximum'):
    summary = read(preflight/'summary.json')
    census = read(preflight/'topology_census.json')
    train = read(BASE/'train512_reserved/manifest.json')
    protected = read(BASE/'expanded_candidates/manifest.json')
    overlap = read(BASE/'external_overlap_v3/summary.json')
    protected_ids = set().union(*(set(protected[k]) for k in ('development','confirmation_reserved','excluded_sealed')))
    protected_ids.update(set().union(*(set(v) for v in overlap['known_parent_clean_official_test_candidates'].values())))
    if set(train['all_training']) & protected_ids: raise ValueError('protected identity in TRAIN')
    if (summary['status'] != 'completed' or not summary['exact_record_roundtrip']
            or len(census) != 512 or set(census) != set(train['all_training'])):
        raise ValueError('complete512 topology/preflight inventory required')
    maximum = max(census, key=lambda s: (census[s]['selected_atoms'], s))
    if maximum not in train['additional'] or census[maximum]['selected_atoms'] != 17002:
        raise ValueError('registered new17002-atom maximum changed')
    for name, key in [('data.bin','store_data_sha256'),('index.txt','store_index_sha256')]:
        if sha256_file(preflight/'probe_store'/name) != summary[key]: raise ValueError('probe bytes changed')
    if input_size not in ('maximum', 'representative'):
        raise ValueError('registered maximum or representative input required')
    selected = maximum if input_size == 'maximum' else 'atlas_3da8_A'
    if selected not in train['all_training'] or selected in protected_ids:
        raise ValueError('qualification record must be unprotected TRAIN')
    atoms = census[selected]['selected_atoms']
    if input_size == 'representative' and atoms != 1578:
        raise ValueError('registered representative input changed')
    sample = f'{selected}_R1_dt_100ps_w000000'
    store = ClipMMapDataset(preflight/'probe_store')
    try:
        matches = [i for i, row in enumerate(store._index) if row[0] == sample]
        if len(matches) != 1: raise ValueError('unique actual maximum probe required')
        record = store[matches[0]]
    finally:
        store.close()
    if (record['x'].shape != (16,atoms,3) or record['split'] != 'train'
            or not np.asarray(record['frame_mask'],dtype=bool).all() or not np.array_equal(np.diff(record['time_ps']), np.full(15,100.))
            or record['topology_fingerprint'] != census[selected]['topology_fingerprint']):
        raise ValueError('full unmodified16-frame registered training record required')
    return record, [BASE/'train512_reserved/manifest.json',BASE/'expanded_candidates/manifest.json',
        BASE/'external_overlap_v3/summary.json',preflight/'summary.json',preflight/'topology_census.json',
        preflight/'probe_store/data.bin',preflight/'probe_store/index.txt']


def build_model(device, recipe, mode, *, recompute_geometry=False):
    initial = recipe['coordinate_initialization']; path = Path(initial['checkpoint'])
    if sha256_file(path) != initial['sha256']: raise ValueError('coordinate parent SHA mismatch')
    saved = torch.load(path, map_location='cpu', weights_only=False)
    require_final_coordinate_checkpoint(saved)
    for name, digest in saved['contract']['runtime_source_sha256'].items():
        if sha256_file(Path(initial['source_root'])/'molvid'/name) != digest:
            raise ValueError('original coordinate-parent runtime changed')
    parent, _, release = load_parent(device)
    model = CoordinateFlowModel(parent).train(); del parent
    if model.contract() != saved['contract']['model'] or saved['contract']['seed'] != 71:
        raise ValueError('coordinate-parent architecture/seed mismatch')
    model.load_state_dict(saved['model'], strict=True)
    if mode == 'trainable_query': model.enable_query_geometry()
    model.set_query_geometry_recompute(recompute_geometry)
    return model, release


def run(args):
    if args.output.exists(): raise FileExistsError(args.output)
    if shutil.disk_usage(args.output.parent if args.output.parent.exists() else Path('.')).free < 9*2**30:
        raise RuntimeError('9GiB reserve required before disposable qualification')
    record, inputs = input_record(args.preflight, args.input_size)
    atoms = record['x'].shape[1]
    recipe = read(args.recipe)
    config = CoordinateTrainConfig(**recipe['coordinate_training'])
    if not config.checkpoint_blocks or config.handedness_weight != 0 or config.total_updates != 2048:
        raise ValueError('unchanged checkpointed FP32 RF settings required')
    inputs += [args.recipe, Path(recipe['coordinate_initialization']['checkpoint']), RELEASE]
    release = read(RELEASE)
    inputs += [Path(release['configuration']['path']), Path(release['selected_checkpoint']['path'])]
    if args.resume is not None:
        if sha256_file(args.resume) != args.resume_sha256: raise ValueError('resume SHA mismatch')
        inputs.append(args.resume)
    identity = freeze_run_identity(args.output, inputs=inputs,
        purpose='Actual maximum TRAIN input CUDA/optimizer/resume qualification, not model efficacy',
        protocol=dict(mode=args.mode,history_frames=args.history,query_frames=16-args.history,
            recompute_geometry=args.recompute_geometry,
            sample_id=record['sample_id'],atoms=atoms,frames=16,precision='float32',input_size=args.input_size,
            resume=str(args.resume),expected_updates_this_process=1 if args.resume else 2,
            test_opened=False,cropped=False,shortened=False,formal_training=False))
    started = time.monotonic()
    try:
        atomic_write_json(args.output/'status.json',dict(status='running',stage='initialize_CUDA_and_parent'))
        device = configure_device('cuda', deterministic=True)
        torch.cuda.set_per_process_memory_fraction(.88)
        random.seed(71); np.random.seed(71); torch.manual_seed(71)
        model, _ = build_model(device, recipe, args.mode, recompute_geometry=args.recompute_geometry)
        if any(p.is_floating_point() and p.dtype != torch.float32 for p in model.parameters()):
            raise ValueError('actual FP32 model parameters required')
        optimizer = make_coordinate_optimizer(model, config)
        generator = torch.Generator(device=device).manual_seed(18071)
        cpu_batch = collate_clip_records([record])
        contract = dict(schema='molvid.dynamic_maximum_input.qualification.v1', mode=args.mode,
            history=args.history, sample_id=record['sample_id'],record_state_sha256=state_digest(record),
            model=model.contract(),training=asdict(config),source_files_sha256=identity['source_files_sha256'],
            parent_sha256=recipe['coordinate_initialization']['sha256'],seed=71,precision='float32')
        cursor = 0
        if args.resume is not None:
            saved = torch.load(args.resume, map_location='cpu', weights_only=False)
            if saved['contract'] != contract or saved['cursor'] != 1: raise ValueError('resume contract/cursor mismatch')
            model.load_state_dict(saved['model'], strict=True)
            optimizer.load_state_dict(saved['optimizer'])
            restore_coordinate_rng(saved['rng'], generator); cursor = 1
            del saved
        teacher_before = state_digest(model.target_teacher.state_dict())
        rows = []
        def payload(step):
            return dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                rng=capture_coordinate_rng(generator),contract=contract,cursor=step)
        with ExecutionEvidence(args.output):
            for step in range(cursor, 2):
                atomic_write_json(args.output/'status.json',dict(status='running',stage='actual_coordinate_update',
                    step=step+1,completed_logged_updates=len(rows)))
                torch.cuda.reset_peak_memory_stats()
                row = coordinate_update(model,optimizer,cpu_batch,history_frames=args.history,
                    generator=generator,step=step,config=config,atom_identity=record['atom_identity'])
                torch.cuda.synchronize()
                finite_state(model.state_dict()); finite_state(optimizer.state_dict())
                gradients = {}
                for name, parameter in model.named_parameters():
                    if parameter.grad is None: continue
                    if not bool(torch.isfinite(parameter.grad).all()): raise FloatingPointError('nonfinite gradient')
                    group = name.split('.')[0]
                    gradients[group] = gradients.get(group,0.) + float(parameter.grad.double().square().sum())
                required = {'dit','history_encoder','readout'} | ({'query_encoder'} if args.mode == 'trainable_query' else set())
                if any(gradients.get(g,0.) <= 0 for g in required) or 'target_teacher' in gradients:
                    raise ValueError(f'expected finite nonzero gradients and frozen teacher: {gradients}')
                row.update(cursor=step+1,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(),gradient_group_squared_norms=gradients,
                    seconds=time.monotonic()-started)
                rows.append(row); atomic_write_json(args.output/'updates.json',rows)
                print(json.dumps(row),flush=True)
                if step == 0:
                    if shutil.disk_usage(args.output).free < 9*2**30: raise RuntimeError('9GiB checkpoint reserve')
                    atomic_torch_save(args.output/'prefix1.pt',payload(1))
            teacher_after = state_digest(model.target_teacher.state_dict())
            if teacher_after != teacher_before: raise RuntimeError('frozen teacher state changed')
            final = payload(2)
            result = dict(status='completed',cursor_start=cursor,cursor_end=2,actual_updates=2-cursor,
                full_state_sha256=state_digest(final),model_sha256=state_digest(final['model']),
                optimizer_sha256=state_digest(final['optimizer']),rng_sha256=state_digest(final['rng']),
                optimizer_steps=sorted({int(s['step']) for s in final['optimizer']['state'].values()}),
                teacher_before_sha256=teacher_before,teacher_after_sha256=teacher_after,
                finite_model_optimizer_and_gradients=True,updates=rows,seconds=time.monotonic()-started,
                prefix1_sha256=sha256_file(args.output/'prefix1.pt') if cursor==0 else None,
                sample_id=record['sample_id'],atoms=atoms,history_frames=args.history,query_frames=16-args.history,
                model_mode=args.mode,recompute_geometry=args.recompute_geometry,
                formal_updates=0,protein_forecasts=0,efficacy_tested=False)
            if result['optimizer_steps'] != [2]: raise ValueError('optimizer cursor differs')
            atomic_write_json(args.output/'summary.json',result)
        atomic_write_json(args.output/'status.json',dict(status='completed'))
    except BaseException as error:
        previous = read(args.output/'status.json') if (args.output/'status.json').exists() else None
        atomic_write_json(args.output/'status.json',dict(status='failed',error=repr(error),last_stage=previous))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',choices=['frozen_query','trainable_query'],required=True)
    parser.add_argument('--history',type=int,choices=[4,8],required=True)
    parser.add_argument('--recompute-geometry',action='store_true')
    parser.add_argument('--input-size',choices=['maximum','representative'],default='maximum')
    parser.add_argument('--recipe',type=Path,default=RECIPE)
    parser.add_argument('--preflight',type=Path,default=PREFLIGHT)
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--resume-sha256')
    parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args())
