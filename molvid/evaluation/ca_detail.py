"""Paired CA intervention scoring: true future CA is explicitly oracle-only."""
from collections import defaultdict
from dataclasses import replace
import json
from pathlib import Path
import time
from types import SimpleNamespace
import numpy as np
import torch
from ..codec.ca_detail import CADetailDecoder
from ..geometry.ca_detail import condition_from_record,rigid_transport
from ..geometry.types import StaticTopologyMetadata
from ..data.batch import collate_clip_records
from ..runtime import configure_device,atomic_write_json,sha256_file
from ..provenance import freeze_run_identity
from ..training.ca_detail_data import load_ca_cache
from ..training.execution_evidence import ExecutionEvidence
from ..training.predictive_data import atomic_torch_save,RELEASE
from ..losses.sampled_physical import PhysicalAuxiliaryConfig
from .geometry import bond_errors
from .temporal import temporal_metrics_v3
from .physical import sampled_clash_metrics
from .review_diagnostics import internal_geometry


def flat_metrics(value,prefix=''):
    result={}
    for key,item in value.items():
        name=f'{prefix}.{key}' if prefix else key
        if isinstance(item,dict):result.update(flat_metrics(item,name))
        elif isinstance(item,(int,float)) and not isinstance(item,bool):result[name]=item
    return result


def score(future,record,c,physical):
    device=future.device;batch=collate_clip_records([record]).to(device)
    truth=batch.x.float();prediction=torch.cat((c.observed,future))
    topology=StaticTopologyMetadata.from_batch(batch)
    context=SimpleNamespace(topology=topology,latent=SimpleNamespace(topology=topology,frame_mask=batch.frame_mask[:,:4]),
        loss_mask=batch.loss_mask,coordinates=c.observed)
    scoring=SimpleNamespace(observed_context=context,coordinate_batch=batch,
        query=SimpleNamespace(frame_mask=batch.frame_mask[:,4:],time_ps=batch.time_ps[:,4:]))
    temporal=temporal_metrics_v3(prediction,truth,batch,history_frames=4)
    clash=sampled_clash_metrics(future,scoring,physical)
    a,b=c.bond_index;mask=c.residue_index[a]!=c.residue_index[b]
    d=(torch.linalg.vector_norm(future[:,a]-future[:,b],dim=-1)-torch.linalg.vector_norm(truth[4:,a]-truth[4:,b],dim=-1)).square()
    parent=c.ca_index[c.residue_index]
    result=flat_metrics(temporal.metrics)
    result.update(bond_errors(prediction,truth,batch,list(range(4,16))))
    result.update({k:v for k,v in clash.items() if k!='samples'})
    result.update(internal_geometry(future,truth[4:],c.bond_index))
    result.update(coordinate_rmse_A=float((future-truth[4:]).square().sum(-1).mean().sqrt()),
        ca_rmse_A=float((future[:,c.ca_index]-truth[4:,c.ca_index]).square().sum(-1).mean().sqrt()),
        detail_rmse_A=float(((future-future[:,parent])-(truth[4:]-truth[4:,parent])).square().sum(-1).mean().sqrt()),
        intraresidue_bond_rmse_A=float(d[:,~mask].mean().sqrt()),interresidue_bond_rmse_A=float(d[:,mask].mean().sqrt()))
    # Same fixed observed-reference alignment, but the declared loss atom set is CA only.
    ca_mask=torch.zeros_like(batch.loss_mask);ca_mask[c.ca_index]=True
    ca_metrics=temporal_metrics_v3(prediction,truth,replace(batch,loss_mask=ca_mask,align_mask=ca_mask),history_frames=4).metrics
    result['ca_msd_mae_A2']=ca_metrics['msd']['curve_mae_A2']
    result['ca_rmsf_amplitude_ratio']=ca_metrics['rmsf']['mean_amplitude_ratio']
    residue_pairs=torch.stack((c.residue_index[a[mask]],c.residue_index[b[mask]])).T.unique(dim=0).T
    left,right=c.ca_index[residue_pairs]
    ca_distance_error=(torch.linalg.vector_norm(future[:,left]-future[:,right],dim=-1)-
        torch.linalg.vector_norm(truth[4:,left]-truth[4:,right],dim=-1))
    result['linked_residue_ca_distance_rmse_A']=float(ca_distance_error.square().mean().sqrt())
    return result


def system_estimate(values):
    groups=defaultdict(list)
    for system,value in values:groups[system].append(value)
    means={s:float(np.mean(v)) for s,v in groups.items()};x=np.array(list(means.values()))
    rng=np.random.default_rng(75);boot=x[rng.integers(0,len(x),(2000,len(x)))].mean(1)
    return dict(mean=float(x.mean()),per_system=means,ci95_system_bootstrap=np.quantile(boot,[.025,.975]).tolist())


@torch.no_grad()
def evaluate_detail(cache,training_root,output):
    cache,training_root,output=Path(cache),Path(training_root),Path(output)
    if output.exists():raise FileExistsError(output)
    device=configure_device('cuda',deterministic=True);items=load_ca_cache(cache,'valid')
    models={};paths=[]
    for arm in ('clean','mixed'):
        protocol=training_root/arm/'protocol.json'
        for seed in (71,72):
            path=training_root/arm/f'final_{seed}.pt';p=torch.load(path,map_location=device,weights_only=False)
            if p['cache_sha256']!=sha256_file(cache/'complete.json') or p['protocol_sha256']!=sha256_file(protocol):
                raise ValueError('training identity changed')
            model=CADetailDecoder(**p['model_config']).to(device).eval();model.load_state_dict(p['model'],strict=True)
            models[(arm,seed)]=model;paths.append(path)
    release=json.loads(RELEASE.read_text());recipe=Path('/workspace/molvid-recon-gm-motion-v3')/release['auxiliary_recipe']['path']
    if sha256_file(recipe)!=release['auxiliary_recipe']['sha256']:raise ValueError('physical recipe changed')
    physical=PhysicalAuxiliaryConfig.resolve(json.loads(recipe.read_text())['physical'])
    freeze_run_identity(output,inputs=paths+[cache/'complete.json',recipe],purpose='Generated vs oracle CA detail validation',
        protocol=dict(horizon_ps=list(range(100,1201,100)),history=4,systems=8,windows_per_system=2,
            source_seeds=[17,29],training_seeds=[71,72],test_opened=False,
            selection='all final512 models, no validation-based checkpoint choice'))
    with ExecutionEvidence(output):
        rows=[];started=time.monotonic()
        for item in items:
            if time.monotonic()-started>7200:raise TimeoutError('evaluation budget exceeded')
            record=item['record'];c=condition_from_record(record).to(device)
            truth=torch.as_tensor(record['x'][4:]).to(device);true_ca=truth[:,c.ca_index]
            variants=[('persistence',-1,-1,c.observed[-1:].expand_as(truth)),
                ('rigid_oracle',-1,-1,rigid_transport(c,true_ca))]
            for (arm,seed),model in models.items():variants.append((f'{arm}_oracle',seed,-1,model(c,true_ca)))
            for draw in item['generated']:
                generated=draw['coordinates'].to(device);ca=generated[:,c.ca_index];source_seed=draw['seed']
                variants.extend([('a1',-1,source_seed,generated),('rigid_generated',-1,source_seed,rigid_transport(c,ca))])
                for (arm,seed),model in models.items():
                    prediction=model(c,ca)
                    if not torch.equal(prediction[:,c.ca_index],ca):raise RuntimeError('CA conditioning changed')
                    variants.append((f'{arm}_generated',seed,source_seed,prediction))
            saved=[]
            for variant,seed,source_seed,prediction in variants:
                if not torch.isfinite(prediction).all():raise FloatingPointError('nonfinite validation prediction')
                metrics=score(prediction,record,c,physical)
                rows.append(dict(system=item['metadata']['system'],sample_id=record['sample_id'],variant=variant,
                    training_seed=seed,source_seed=source_seed,metrics=metrics))
                saved.append(dict(variant=variant,training_seed=seed,source_seed=source_seed,coordinates=prediction.cpu()))
            atomic_torch_save(output/f"coordinates_{len(rows):04d}.pt",dict(sample_id=record['sample_id'],predictions=saved))
            atomic_write_json(output/'rows.json',rows)
            print(json.dumps(dict(phase='CA_evaluation',sample_id=record['sample_id'],rows=len(rows),seconds=time.monotonic()-started)),flush=True)
        arms={}
        for variant,seed in sorted({(r['variant'],r['training_seed']) for r in rows}):
            selected=[r for r in rows if r['variant']==variant and r['training_seed']==seed]
            arms[f'{variant}_seed{seed}']={key:system_estimate([(r['system'],r['metrics'][key]) for r in selected])
                for key in selected[0]['metrics'] if all(np.isfinite(r['metrics'][key]) for r in selected)}
        comparisons={}
        for arm in ('clean','mixed'):
            for seed in (71,72):
                name=f'{arm}_generated_seed{seed}';a=arms[name]
                for reference in ('a1_seed-1','rigid_generated_seed-1',f'{arm}_oracle_seed{seed}'):
                    b=arms[reference]
                    comparisons[f'{name}_minus_{reference}']={k:system_estimate([(s,a[k]['per_system'][s]-b[k]['per_system'][s])
                        for s in a[k]['per_system']]) for k in a if k in b}
        atomic_write_json(output/'summary.json',dict(status='completed',rows=len(rows),arms=arms,paired=comparisons,
            seconds=time.monotonic()-started,test_opened=False,promotion='none',
            limitations=['CA generator is frozen A1; not end-to-end hierarchy training.',
                'True CA conditions are oracle only, not deployable generation.',
                'Same 8 development systems; no novel-family or thermodynamic claim.',
                'Bootstrap is per system, conditional on training seed.']))
