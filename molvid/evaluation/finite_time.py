"""K1: compare a frozen A1 ensemble with real MD and an independent K0 head.

Predicted kernel expectations are diagnostic references, not a learned truth
oracle. Direct real-MD kernel errors, MMD and ordinary geometry/motion metrics
are retained. Four draws are a bounded noisy estimate, not a path benchmark.
"""
from dataclasses import replace
import json
from pathlib import Path
import time

import torch

from ..finite_time import FiniteTimeResponse,response_kernels
from ..data.batch import collate_clip_records
from ..data.store import ClipMMapDataset
from ..codec.frame import slice_clip_frames
from ..latent.types import QuerySpec
from ..flow.sampling import euler_sample_frames
from ..runtime import atomic_write_json,sha256_file
from ..training.predictive_data import load_parent,encode_observation,encode_targets,VIEWS
from ..training.static_decoder import _block_reference_maps
from ..training.batches import prepare_frame_joint_batch
from ..losses.sampled_physical import PhysicalAuxiliaryConfig
from .geometry import bond_errors
from .temporal import temporal_metrics_v3
from .physical import sampled_clash_metrics
from .predictive import HORIZONS


def generated_target_batch(cpu,future):
    """Rebuild generated future bpos from observed-only anchor identities.

    Never encode generated x with hidden MD future bpos. Fallback is a
    coordinate-derived group centroid, with no observed/target displacement.
    """
    anchor,group,_,_=_block_reference_maps(slice_clip_frames(cpu,3,4))
    anchor=anchor.to(future.device);group=group.to(future.device)
    sums=future.new_zeros((len(future),int(group.max())+1,3))
    sums.scatter_add_(1,group[None,:,None].expand_as(future),future)
    count=torch.bincount(group,minlength=sums.shape[1]).to(future.dtype)
    fallback=(sums/count[None,:,None])[:,group]
    positions=torch.where((anchor>=0)[None,:,None],future[:,anchor.clamp_min(0)],fallback)
    return replace(cpu,x=torch.cat((cpu.x[:4],future.cpu())),bpos=torch.cat((cpu.bpos[:4],positions.cpu())))


@torch.no_grad()
def unbiased_rbf_mmd(first,second):
    """Pooled local-change distribution, not joint per-protein conformation."""
    if len(first)<2 or len(second)<2: raise ValueError('MMD requires two samples per group')
    def kernel(a,b): return torch.exp(-.5*torch.cdist(a,b).square()/a.shape[-1])
    xx=kernel(first,first);yy=kernel(second,second);xy=kernel(first,second)
    return float((xx.sum()-xx.diagonal().sum())/(len(first)*(len(first)-1))
        +(yy.sum()-yy.diagonal().sum())/(len(second)*(len(second)-1))-2*xy.mean())


@torch.no_grad()
def diagnose_generator(cache,checkpoint,output,*,systems_limit=8,seeds=(0,1,2,3)):
    if output.exists(): raise ValueError('fresh K1 output required')
    device=torch.device('cuda')
    payload=torch.load(checkpoint,map_location=device,weights_only=False)
    # Protect the predictor/data relationship before consuming cached targets.
    if sha256_file(cache/'complete.json')!=payload['protocol']['cache_manifest_sha256']:
        raise ValueError('K predictor feature identity mismatch')
    manifest=json.loads((cache/'complete.json').read_text())
    identity=manifest['identities']['valid']
    if sha256_file(cache/identity['path'])!=identity['sha256']:
        raise ValueError('validation cache changed')
    data=torch.load(cache/identity['path'],map_location='cpu',weights_only=False)
    head=FiniteTimeResponse(**payload['model_config']).to(device).eval();head.load_state_dict(payload['model'],strict=True)
    stats=payload['statistics'];regions=payload['regions'];projection=data['projection'].to(device)
    selected=[];seen=set()
    for i,row in enumerate(data['metadata']):
        if row['system'] in seen: continue
        selected.append((i,row));seen.add(row['system'])
        if len(seen)==systems_limit: break
    model,raw,release=load_parent(device)
    recipe=Path('/workspace/molvid-recon-gm-motion-v3')/release['auxiliary_recipe']['path']
    if sha256_file(recipe)!=release['auxiliary_recipe']['sha256']: raise ValueError('physical recipe changed')
    physical=PhysicalAuxiliaryConfig.resolve(json.loads(recipe.read_text())['physical'])
    output.mkdir(parents=True)
    atomic_write_json(output/'protocol.json',dict(schema='molvid.k.generator_diagnostic.v1',
        predictor_sha256=sha256_file(checkpoint),parent=release['selected_checkpoint'],
        sample_ids=[r['sample_id'] for _,r in selected],seeds=list(seeds),steps=16,precision='fp32',test_opened=False,
        scope='one H4 anchor per system; pooled local-change kernels; K head is NOT a ground-truth oracle'))
    dataset=ClipMMapDataset(VIEWS/'clip_store/valid');rows=[];geometry_rows=[];started=time.monotonic()
    try:
        for cache_index,row in selected:
            record=dataset[row['index']]
            if record['sample_id']!=row['sample_id']: raise ValueError('K1 sample identity changed')
            cpu=collate_clip_records([record]);obs=encode_observation(model,cpu,projection,device=device)
            if obs['selected'].cpu().tolist()!=row['selected_atoms']: raise ValueError('atom selection drift')
            x=(obs['x']-stats['x_mean'])/stats['x_std']
            query=QuerySpec(time_ps=cpu.time_ps[:,4:].to(device),frame_mask=cpu.frame_mask[:,4:].to(device))
            target=data['y'][cache_index].to(device)/stats['y_scale']
            target_kernel=response_kernels(target,regions)
            predicted=torch.stack([head(x,x.new_full((len(x),),lag)) for lag in HORIZONS],dim=1)
            generated=[];generated_kernel=[]
            # Real targets belong only to scoring, never to euler_sample_frames.
            scoring=prepare_frame_joint_batch(model.target_teacher,cpu,device=device,normalizer=model,history_frames=4)
            for seed in seeds:
                latent,_=euler_sample_frames(model,context=obs['context'],query=query,steps=16,seed=seed)
                future=model.decoder(obs['context'],latent,query).coordinates.float()
                if not bool(torch.isfinite(future).all()): raise FloatingPointError('nonfinite A1 generation')
                replacement=generated_target_batch(cpu,future)
                change=encode_targets(model,replacement,obs,projection,device=device)/stats['y_scale']
                generated.append(change);generated_kernel.append(response_kernels(change,regions))
                prediction=torch.cat((obs['context'].coordinates.float(),future))
                temporal=temporal_metrics_v3(prediction,scoring.coordinate_batch.x.float(),scoring.coordinate_batch,history_frames=4)
                clash=sampled_clash_metrics(future,scoring,physical)
                geometry_rows.append(dict(system=row['system'],sample_id=row['sample_id'],seed=seed,
                    metrics={**temporal.metrics,**bond_errors(prediction,scoring.coordinate_batch.x,scoring.coordinate_batch,list(range(4,16))),
                        **{k:v for k,v in clash.items() if k!='samples'}}))
                atomic_write_json(output/'geometry_rows.json',geometry_rows)
            ensemble=torch.stack(generated);kernel_mean=torch.stack(generated_kernel).mean(0)
            persistent=response_kernels(torch.zeros_like(target),regions)
            for index,lag in enumerate(HORIZONS):
                rows.append(dict(system=row['system'],sample_id=row['sample_id'],horizon_ps=lag,
                    generated_vs_real_kernel_mse=float((kernel_mean[:,index]-target_kernel[:,index]).square().mean()),
                    predicted_vs_real_kernel_mse=float((predicted[:,index]-target_kernel[:,index]).square().mean()),
                    generated_vs_predicted_kernel_mse=float((kernel_mean[:,index]-predicted[:,index]).square().mean()),
                    persistence_vs_real_kernel_mse=float((persistent[:,index]-target_kernel[:,index]).square().mean()),
                    local_response_mmd2=unbiased_rbf_mmd(ensemble[:,:,index].flatten(0,1),target[:,index])))
            atomic_write_json(output/'rows.json',rows)
            print(json.dumps({'phase':'K1','system':row['system'],'systems_done':len(rows)//3}),flush=True)
        atomic_write_json(output/'summary.json',dict(status='completed',systems=len(selected),draws=len(seeds),
            rows=len(rows),seconds=time.monotonic()-started,test_opened=False,
            limitation='correlated atoms and one real successor per anchor; MMD is descriptive, can be negative, and not a calibrated system-level test'))
    finally:
        dataset.close()
