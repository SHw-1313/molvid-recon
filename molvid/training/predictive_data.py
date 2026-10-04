"""Read-only A1 feature cache for independent U/K probes, never test data.

Inputs are frozen observed H4 features. Targets are projected scalar-teacher
changes at 100/400/1200 ps. Features and targets have separate construction
interfaces; neither probe changes the parent generator or teacher.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

import torch

from ..checkpoints import load_frame_joint_inference
from ..cli.train_frame_joint import _open_data
from ..codec.frame import slice_clip_frames
from ..config import load_config
from ..data.batch import collate_clip_records
from ..data.sampling import get_clip_specs
from ..data.store import ClipMMapDataset
from ..latent.types import ObservedContext
from ..runtime import atomic_write_json, configure_device, sha256_file, canonical_hash
from .batches import prepare_batch_then_to_device


RELEASE=Path('/workspace/molvid-gm-memory-v3-coordination/recipe_release.json')
VIEWS=Path('/workspace/molvid_recon/runs/frame_joint_v1_multitime_eval_260922/paired_views')
VALID_PROTOCOL=Path('/workspace/molvid-recon-gm-motion-v3/runs/gm_motion_memory_v3/A/r16/configs/evaluation/A1/ordinary/protocol.json')
QUERY_FRAMES=(4,7,15)


def group_identity(sample_id):
    match=re.match(r'^(atlas_.+)_R([123])_',sample_id)
    if match is None:
        raise ValueError(f'unknown trajectory identity: {sample_id}')
    return match.group(1),match.group(1)+'_R'+match.group(2)


def system_partition(systems,calibration_count=24):
    ordered=sorted(set(systems),key=lambda s:hashlib.sha256(('uk-split-v1:'+s).encode()).hexdigest())
    if not 0<calibration_count<len(ordered):
        raise ValueError('calibration must hold out whole systems')
    return set(ordered[calibration_count:]),set(ordered[:calibration_count])


def select_records(dataset,*,paired_views=False,windows=2):
    groups=defaultdict(list)
    for spec in get_clip_specs(dataset):
        if spec.frames!=16: continue
        if paired_views:
            if not spec.sample_id.endswith('_dt_100ps_H4'): continue
        elif spec.time_bucket_id!='dt_100ps': continue
        system,trajectory=group_identity(spec.sample_id)
        groups[(system,trajectory)].append(spec)
    rows=[]
    for (system,trajectory),items in sorted(groups.items()):
        items.sort(key=lambda item:item.sample_id)
        if len(items)<windows:
            raise ValueError('insufficient trajectory windows')
        # Spread selected windows across the trajectory. Splits are by system,
        # never by these windows, even if original windows overlap.
        indices=torch.linspace(0,len(items)-1,windows).round().long().tolist()
        for index in indices:
            s=items[index]
            rows.append(dict(index=s.index,sample_id=s.sample_id,system=system,trajectory=trajectory,atoms=s.atoms))
    if not rows: raise ValueError('no compatible 100 ps records')
    return rows


def fixed_projection(width,channels=16):
    if not 1<=channels<=width: raise ValueError('invalid scalar projection width')
    matrix=torch.randn(width,channels,generator=torch.Generator().manual_seed(7104))
    return torch.linalg.qr(matrix,mode='reduced')[0]


def observed_batch(cpu):
    """Slice raw inputs before teacher invocation, not after target encoding."""
    if cpu.batch_size!=1 or cpu.frames!=16:
        raise ValueError('probe cache requires single 16-frame clips')
    return slice_clip_frames(cpu,0,4)


@torch.no_grad()
def encode_observation(model,cpu,projection,*,device,atoms=64):
    observed=prepare_batch_then_to_device(model.target_teacher,observed_batch(cpu),device)
    latent,origin=model.target_teacher(observed)
    context=ObservedContext(latent=latent,coordinates=observed.x,sample_origin=origin,
                            loss_mask=observed.loss_mask.bool())
    if not bool(latent.frame_mask.all()): raise ValueError('H4 must be fully observed')
    memory=model.history_encoder(context)
    normalized=model.normalize(latent)
    candidates=torch.where(observed.loss_mask)[0]
    if len(candidates)<atoms: raise ValueError('not enough supervised atoms for equal-size probe')
    selected=candidates[torch.linspace(0,len(candidates)-1,atoms,device=device).round().long()]
    last=normalized.h[-1,selected].float()
    change=(normalized.h[-1,selected]-normalized.h[0,selected]).float()
    frozen_memory=memory.h[:,selected].mean(dim=0).float()
    x=torch.cat((last,change,frozen_memory),dim=-1)
    return dict(x=x,last=last@projection,selected=selected,origin=origin,context=context,last_width=last.shape[-1])


@torch.no_grad()
def encode_targets(model,cpu,observation,projection,*,device):
    """Only this function reads future coordinates, as explicit loss targets."""
    index=torch.tensor(QUERY_FRAMES,dtype=torch.long)
    times=cpu.time_ps[:,index]
    future=replace(cpu,x=cpu.x[index],
        bpos=cpu.bpos[index],frame_mask=cpu.frame_mask[:,index],time_ps=times,
        delta_time_ps=times[:,1:]-times[:,:-1])
    future=prepare_batch_then_to_device(model.target_teacher,future,device)
    latent,_=model.target_teacher(future,reference_origin=observation['origin'])
    projected=model.normalize(latent).h[:,observation['selected']].float()@projection
    return (projected-observation['last'].unsqueeze(0)).transpose(0,1).contiguous()


def atomic_torch_save(path,payload):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.partial')
    torch.save(payload,temporary);os.replace(temporary,path)


def load_parent(device):
    release=json.loads(RELEASE.read_text())
    if release['test_opened'] is not False or release['selected_arm']!='A1':
        raise ValueError('requires frozen A1 release')
    if sha256_file(release['configuration']['path'])!=release['configuration']['sha256']:
        raise ValueError('configuration SHA changed')
    raw=load_config(release['configuration']['path'],schema='molvid.frame_joint.train.v1')
    parent=release['selected_checkpoint']
    model=load_frame_joint_inference(parent['path'],expected_sha256=parent['sha256'],
        codec_path=raw['codec']['checkpoint'],codec_sha256=raw['codec']['sha256'],device=device)['model']
    model.eval().requires_grad_(False)
    model.target_teacher.frame_chunk_size=1
    return model,raw,release


def build_cache(output,*,windows=2,atoms=64):
    output=Path(output)
    if output.exists(): raise ValueError('fresh shared feature cache required')
    if shutil.disk_usage(output.parent if output.parent.exists() else Path('.')).free<6*2**30:
        raise RuntimeError('at least 6 GiB disk reserve required')
    device=configure_device('cuda',deterministic=True)
    torch.cuda.set_per_process_memory_fraction(.25)
    model,raw,release=load_parent(device)
    train,_,data_hash,_,_,close=_open_data(raw['data'])
    expected=json.loads(VALID_PROTOCOL.read_text())
    valid_path=VIEWS/'clip_store/valid'
    if sha256_file(valid_path/'index.txt')!=expected['store_index_sha256'] or sha256_file(VIEWS/'paired_view_manifest.json')!=expected['manifest_sha256']:
        close();raise ValueError('validation view hash mismatch')
    valid=ClipMMapDataset(valid_path)
    output.mkdir(parents=True)
    try:
        tr=select_records(train,windows=windows)
        va=select_records(valid,paired_views=True,windows=windows)
        fit,cal=system_partition([r['system'] for r in tr])
        valid_systems={r['system'] for r in va}
        if (fit|cal)&valid_systems: raise ValueError('system leakage into validation')
        if len(fit|cal)!=192 or len(valid_systems)!=8:
            raise ValueError('expected complete 192 train / 8 validation systems')
        width=model.statistics.h_mean.numel()
        projection=fixed_projection(width).to(device)
        plan={name:[r for r in rows if r['system'] in systems] for name,rows,systems in
              [('fit',tr,fit),('calibration',tr,cal),('valid',va,valid_systems)]}
        protocol=dict(schema='molvid.uk.features.v1',integration_base='946917b',
            source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            parent=release['selected_checkpoint'],teacher_sha256=raw['codec']['sha256'],
            data_hash=data_hash,valid_index_sha256=expected['store_index_sha256'],
            valid_manifest_sha256=expected['manifest_sha256'],plan=plan,atoms_per_clip=atoms,
            history_frames=4,query_frames=list(QUERY_FRAMES),horizons_ps=[100,400,1200],
            projection_channels=16,projection_seed=7104,precision='fp32',teacher_chunk=1,
            input_layout='last normalized scalar h; H4 endpoint scalar change; frozen legacy memory scalar h',
            target='projected normalized future scalar h minus last observed scalar h',
            split_note='fit/calibration disjoint systems within A1 training data; valid 8 systems separate. Parent saw calibration systems during original pretraining.',
            scope='local scalar-teacher feature changes; not full coordinates, vector covariance, metastable states or novel-protein test',
            test_opened=False)
        atomic_write_json(output/'protocol.json',protocol)
        identities={};started=time.monotonic()
        for split,rows in plan.items():
            features=[];targets=[];metadata=[]
            dataset=valid if split=='valid' else train
            for i,row in enumerate(rows):
                record=dataset[row['index']]
                if str(record['sample_id'])!=row['sample_id']: raise ValueError('record identity changed')
                cpu=collate_clip_records([record])
                if not torch.equal(cpu.time_ps[0,1:]-cpu.time_ps[0,:-1],torch.full((15,),100.)):
                    raise ValueError('nonmatching 100 ps physical clock')
                obs=encode_observation(model,cpu,projection,device=device,atoms=atoms)
                y=encode_targets(model,cpu,obs,projection,device=device)
                if not torch.isfinite(obs['x']).all() or not torch.isfinite(y).all():
                    raise FloatingPointError('nonfinite cached features')
                features.append(obs['x'].cpu());targets.append(y.cpu())
                metadata.append(dict(row,selected_atoms=obs['selected'].cpu().tolist()))
                atomic_write_json(output/'progress.json',dict(split=split,clips=i+1,total=len(rows),
                    seconds=time.monotonic()-started,status='extracting',test_opened=False))
                if (i+1)%16==0: print(json.dumps({'phase':'cache','split':split,'clips':i+1,'total':len(rows)}),flush=True)
            path=output/f'{split}.pt'
            payload=dict(x=torch.stack(features),y=torch.stack(targets),metadata=metadata,
                projection=projection.cpu(),last_width=width,protocol_sha256=sha256_file(output/'protocol.json'))
            atomic_torch_save(path,payload)
            identities[split]=dict(path=path.name,sha256=sha256_file(path),clips=len(rows))
        if any(p.grad is not None or p.requires_grad for p in model.parameters()):
            raise RuntimeError('parent must remain entirely frozen')
        atomic_write_json(output/'complete.json',dict(status='completed',identities=identities,
            protocol_sha256=sha256_file(output/'protocol.json'),seconds=time.monotonic()-started,
            test_opened=False))
    except Exception as exc:
        atomic_write_json(output/'failure.json',dict(type=type(exc).__name__,message=str(exc)))
        raise
    finally:
        valid.close();close()


def load_cache(root,device):
    root=Path(root);manifest=json.loads((root/'complete.json').read_text())
    if manifest['status']!='completed' or manifest['test_opened'] is not False:
        raise ValueError('incomplete or unsealed feature cache')
    if sha256_file(root/'protocol.json')!=manifest['protocol_sha256']:
        raise ValueError('cache protocol changed')
    result={}
    for split,identity in manifest['identities'].items():
        path=root/identity['path']
        if sha256_file(path)!=identity['sha256']: raise ValueError('cache tensor hash changed')
        payload=torch.load(path,map_location='cpu',weights_only=False)
        if payload['protocol_sha256']!=manifest['protocol_sha256']: raise ValueError('cache split protocol mismatch')
        result[split]=payload
    groups={s:{r['system'] for r in p['metadata']} for s,p in result.items()}
    if set(groups)!={'fit','calibration','valid'} or any(groups[a]&groups[b] for a,b in [('fit','calibration'),('fit','valid'),('calibration','valid')]):
        raise ValueError('cache split leakage')
    x=result['fit']['x'];y=result['fit']['y']
    stats=dict(x_mean=x.mean((0,1)),x_std=x.std((0,1),unbiased=False).clamp_min(.01),
               y_scale=y.square().mean((0,1,2)).sqrt().clamp_min(.01))
    for payload in result.values():
        payload['x']=((payload['x']-stats['x_mean'])/stats['x_std']).to(device)
        payload['y']=(payload['y']/stats['y_scale']).to(device)
    return result,{k:v.to(device) for k,v in stats.items()},manifest
