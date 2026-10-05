"""Immutable native100ps benchmark: original PVB or frozen Molvid weights."""
import argparse
from collections import defaultdict
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import time
import numpy as np
import torch

from molvid.checkpoints import load_frame_joint_inference
from molvid.codec.frame import slice_clip_frames
from molvid.data.batch import collate_clip_records
from molvid.evaluation.ca_detail import score, system_estimate
from molvid.evaluation.native_forecast import molvid_forecast, coordinate_forecast, load_pvb, pvb_observation, pvb_forecast
from molvid.flow.coordinate import CoordinateFlowModel
from molvid.geometry.ca_detail import condition_from_record
from molvid.geometry.projection import ProjectionConfig, repair_coordinates
from molvid.geometry.types import StaticTopologyMetadata
from molvid.losses.sampled_physical import PhysicalAuxiliaryConfig
from molvid.provenance import freeze_run_identity
from molvid.runtime import atomic_write_json, configure_device, sha256_file
from molvid.training.execution_evidence import ExecutionEvidence
from molvid.training.predictive_data import load_parent, RELEASE, atomic_torch_save


def run(args):
    if args.output.exists(): raise FileExistsError(args.output)
    if args.arm!='a1' and args.checkpoint is None: raise ValueError('checkpoint required')
    if args.arm=='pvb' and args.project: raise ValueError('register a separate equal-treatment postprocessing campaign')
    if args.variant is not None and args.arm not in ('checkpoint','coordinate'): raise ValueError('custom variant names require checkpoint arm')
    variant_name=args.variant or args.arm
    device=configure_device('cuda', deterministic=args.arm!='pvb')
    torch.cuda.set_per_process_memory_fraction(.9)
    manifest=json.loads((args.cache/'complete.json').read_text())
    entries=[f for f in manifest['files'] if f['lag_ps']==100]
    if len(entries)!=96 or len({e['system'] for e in entries})!=48: raise ValueError('expanded48 coverage differs')
    if args.pilot: entries=entries[:1]
    release=json.loads(RELEASE.read_text())
    recipe=Path('/workspace/molvid-recon-gm-motion-v3')/release['auxiliary_recipe']['path']
    if sha256_file(recipe)!=release['auxiliary_recipe']['sha256']: raise ValueError('recipe SHA changed')
    physical=PhysicalAuxiliaryConfig.resolve(json.loads(recipe.read_text())['physical'])
    inputs=[args.cache/'complete.json',recipe];external_files={}
    if args.arm=='pvb':
        model=load_pvb(args.checkpoint,args.pvb_root,device)
        external_files={str(p.relative_to(args.pvb_root)):sha256_file(p) for p in sorted(args.pvb_root.rglob('*.py'))}
        inputs.append(args.checkpoint)
    else:
        model,raw,_=load_parent(device)
        if args.arm=='coordinate':
            saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
            if saved['cursor']!=2048 or saved['contract']['largest_nominal_batch_profile_only']:
                raise ValueError('only fixed-final2048 coordinate checkpoints qualify')
            model=CoordinateFlowModel(model)
            if model.contract()!=saved['contract']['model']: raise ValueError('coordinate model contract differs')
            model.load_state_dict(saved['model'],strict=True)
            model.eval().requires_grad_(False);del saved
        elif args.arm!='a1':
            del model;torch.cuda.empty_cache()
            model=load_frame_joint_inference(args.checkpoint,expected_sha256=sha256_file(args.checkpoint),
                codec_path=raw['codec']['checkpoint'],codec_sha256=raw['codec']['sha256'],device=device)['model']
            model.eval().requires_grad_(False);model.target_teacher.frame_chunk_size=1
        inputs.extend([args.checkpoint or Path(release['selected_checkpoint']['path']),Path(raw['codec']['checkpoint'])])
    protocol=dict(arm=args.arm,variant=variant_name,development=True,confirmation_opened=False,test_opened=False,
        history_frames=1 if args.arm=='pvb' else 4,query_frames=12,query_dt_ps=100,seeds=[17,29],
        solver_steps=10 if args.arm=='pvb' else 16,neural_training_updates=0,pilot=args.pilot,
        records=entries,projection=asdict(ProjectionConfig()) if args.project else None,
        baseline_pretraining_exposure_audit='pending; not a qualified held-out competition result',
        parameter_count=sum(p.numel() for p in model.parameters()),external_source_sha256=external_files,
        model_source_config=(None if args.arm=='pvb' else model.contract() if args.arm=='coordinate' else model.source_config.contract()),
        numpy_version=np.__version__,numpy_source=np.__file__,
        original_pvb_refiner=False,physical_units='Angstrom, ps',max_seconds=21600,
        limitations=['ATLAS protein-only','R1 two anchors','short horizon is not equilibrium validation',
            'trajectory/path RMSE is diagnostic, not distributional likelihood'])
    freeze_run_identity(args.output,inputs=inputs,purpose='Expanded native development benchmark',protocol=protocol)
    atomic_write_json(args.output/'protocol.json',protocol)
    started=time.monotonic();rows=[];failures=[]
    with ExecutionEvidence(args.output):
        for index,entry in enumerate(entries):
            if time.monotonic()-started>21600: raise TimeoutError('registered six-hour arm budget')
            if shutil.disk_usage(args.output).free<8*1024**3: raise RuntimeError('8GiB disk reserve reached')
            path=args.cache/entry['path']
            if sha256_file(path)!=entry['sha256']: raise ValueError('clip SHA changed')
            item=torch.load(path,map_location='cpu',weights_only=False);record=item['record']
            cpu=collate_clip_records([record]);observed=slice_clip_frames(cpu,0,4)
            c=condition_from_record(record).to(device)
            if args.arm=='pvb':
                pdb=Path(item['source']['pdb'])
                if sha256_file(pdb)!=item['source']['pdb_sha256']: raise ValueError('PDB changed')
                pvb_batch,origin=pvb_observation(pdb,observed,record['atom_source_index'],device)
            saved=[]
            try:
                for seed in (17,29):
                    with torch.no_grad():
                        forecast=coordinate_forecast if args.arm=='coordinate' else molvid_forecast
                        predicted=(pvb_forecast(model,pvb_batch,origin,seed=seed) if args.arm=='pvb' else
                            forecast(model,observed,cpu.time_ps[:,4:],device=device,seed=seed))
                    variants=[(variant_name,predicted)]
                    if args.project:
                        batch=cpu.to(device)
                        repaired,_=repair_coordinates(predicted,observed=c.observed,
                            topology=StaticTopologyMetadata.from_batch(batch),loss_mask=batch.loss_mask,
                            frame_mask=batch.frame_mask[:,4:],physical=physical)
                        variants.append((variant_name+'_projection',repaired))
                    for variant,coordinates in variants:
                        if not torch.isfinite(coordinates).all(): raise FloatingPointError('nonfinite prediction')
                        with torch.no_grad(): metrics=score(coordinates,record,c,physical)
                        rows.append(dict(system=entry['system'],sample_id=entry['sample_id'],seed=seed,
                            variant=variant,metrics=metrics))
                        saved.append(dict(variant=variant,seed=seed,coordinates=coordinates.cpu()))
                    del variants,predicted
            except torch.cuda.OutOfMemoryError as error:
                failures.append(dict(sample_id=entry['sample_id'],kind='CUDA_OOM',error=str(error)))
                torch.cuda.empty_cache()
            atomic_torch_save(args.output/f'coordinates_{index:04d}.pt',dict(sample_id=entry['sample_id'],predictions=saved))
            atomic_write_json(args.output/'rows.json',rows);atomic_write_json(args.output/'failures.json',failures)
            print(json.dumps(dict(completed_records=index+1,total_records=len(entries),rows=len(rows),
                failed_records=len(failures),seconds=time.monotonic()-started)),flush=True)
        aggregates=defaultdict(list)
        for row in rows:
            for key,value in row['metrics'].items(): aggregates[(row['variant'],key)].append((row['system'],value))
        atomic_write_json(args.output/'summary.json',dict(status='incomplete' if failures else 'completed',
            systems=len({r['system'] for r in rows}),rows=len(rows),failures=failures,pilot=args.pilot,
            seconds=time.monotonic()-started,neural_training_updates=0,test_opened=False,
            arms={'|'.join(k):system_estimate(v) for k,v in aggregates.items()}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--arm',choices=['a1','c4','c8','pvb','checkpoint','coordinate'],required=True)
    p.add_argument('--variant')
    p.add_argument('--checkpoint',type=Path);p.add_argument('--pvb-root',type=Path,default=Path('/workspace/PVB_origin'))
    p.add_argument('--project',action='store_true');p.add_argument('--pilot',action='store_true');run(p.parse_args())
