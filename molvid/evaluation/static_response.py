"""Bounded, paired generated validation for the static-response experiment.

Eight validation systems, H4, R1/anchor0, 100/400 ps grids and three source
seeds. Not full formal ordinary/fixed-history evaluation or a sealed test.
Static bypasses are OOD sensitivity diagnostics, not trained ablations.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
import json
import math
from pathlib import Path
from statistics import mean
import time

import numpy as np
import torch

from ..data.batch import collate_clip_records
from ..data.static import trajectory_record_to_static
from ..data.store import ClipMMapDataset
from ..flow.sampling import euler_sample_frames
from ..flow.source import sample_frame_source
from ..losses.sampled_physical import PhysicalAuxiliaryConfig
from ..runtime import atomic_write_json, sha256_file
from ..training.batches import prepare_frame_joint_batch
from ..training.static_backbone import prepare_static_backbone_batch
from ..training.static_decoder import StaticPerturbationConfig, StaticDecoderLossConfig, static_decoder_loss
from .geometry import bond_errors
from .physical import sampled_clash_metrics
from .temporal import temporal_metrics_v3


VIEWS=Path('/workspace/molvid_recon/runs/frame_joint_v1_multitime_eval_260922/paired_views')
REFERENCE=Path('/workspace/molvid-recon-gm-motion-v3/runs/gm_motion_memory_v3/A/r16/configs/evaluation/A1/ordinary/protocol.json')


def validation_plan(*, systems=8, lags=(100,400), seeds=(0,1,2)):
    if not 1<=systems<=8 or not lags or not seeds or len(set(seeds))!=len(seeds):
        raise ValueError('invalid bounded validation selection')
    expected=json.loads(REFERENCE.read_text())
    store=VIEWS/'clip_store/valid'
    if sha256_file(store/'index.txt')!=expected['store_index_sha256'] or sha256_file(VIEWS/'paired_view_manifest.json')!=expected['manifest_sha256']:
        raise ValueError('validation identity differs from A1 release')
    dataset=ClipMMapDataset(store)
    try:
        ids={str(row[0]) for row in dataset._index}
        selected=sorted({sid.rsplit('_R',1)[0] for sid in ids})[:systems]
        sample_ids=[f'{system}_R1_a000000_dt_{lag}ps_H4' for system in selected for lag in lags]
        if len(selected)!=systems or not set(sample_ids)<=ids:
            raise ValueError('incomplete balanced validation selection')
        return dict(schema='molvid.static_response.validation.v1',sample_ids=sample_ids,
            systems=selected,lags=list(lags),seeds=list(seeds),steps=16,precision='fp32',
            static_sigmas_A=[.1,.3,.8],static_steps=4,static_source_seed=19,
            primary_metrics=['bond_rmse','clashes_per_1000_atoms','vdw_severity_A2_per_atom',
                             'rmsf.residue_profile.mae_A','msd.curve_mae_A2'],
            store_index_sha256=expected['store_index_sha256'],manifest_sha256=expected['manifest_sha256'],
            split='valid',test_opened=False,scope='bounded H4 R1 anchor0; not full formal benchmark',
            promotion='no automatic promotion; dynamic geometry AND motion must be considered')
    finally:
        dataset.close()


@torch.no_grad()
def evaluate_response(model, raw, release, *, output, checkpoint_sha256, plan=None):
    if output.exists():
        raise ValueError('fresh validation directory required')
    plan=validation_plan() if plan is None else plan
    # Recheck data identities even when a frozen selection was supplied.
    verified=validation_plan(systems=len(plan['systems']),lags=tuple(plan['lags']),seeds=tuple(plan['seeds']))
    if plan!=verified:
        raise ValueError('validation plan has drifted')
    recipe=Path('/workspace/molvid-recon-gm-motion-v3')/release['auxiliary_recipe']['path']
    if sha256_file(recipe)!=release['auxiliary_recipe']['sha256']:
        raise ValueError('physical recipe hash differs')
    physical=PhysicalAuxiliaryConfig.resolve(json.loads(recipe.read_text())['physical'])
    output.mkdir(parents=True)
    atomic_write_json(output/'protocol.json',dict(plan,checkpoint_sha256=checkpoint_sha256))
    model.eval();device=next(model.dit.parameters()).device
    dataset=ClipMMapDataset(VIEWS/'clip_store/valid')
    rows=[];static_rows=[];started=time.monotonic()
    try:
        by_id={str(row[0]):i for i,row in enumerate(dataset._index)}
        for sid in plan['sample_ids']:
            record=dataset[by_id[sid]]
            system=sid.rsplit('_R',1)[0]
            batch=prepare_frame_joint_batch(model.target_teacher,collate_clip_records([record]),
                device=device,normalizer=model,history_frames=4)
            target=batch.coordinate_batch.x.float()
            frames=[4+i for i in torch.where(batch.query.frame_mask[0])[0].tolist()]
            for variant,seed in [('persistence',None),('oracle',None)]+[('generated',s) for s in plan['seeds']]:
                tick=time.monotonic()
                if variant=='persistence':
                    future=batch.observed_context.coordinates[-1:].expand_as(batch.target_coordinates).float()
                elif variant=='oracle':
                    future=model.decoder(batch.observed_context,batch.target_future,batch.query).coordinates.float()
                else:
                    latent,_=euler_sample_frames(model,context=batch.observed_context,query=batch.query,
                        steps=plan['steps'],seed=seed)
                    future=model.decoder(batch.observed_context,latent,batch.query).coordinates.float()
                if not bool(torch.isfinite(future).all()):
                    raise FloatingPointError('nonfinite validation generation')
                prediction=torch.cat((batch.observed_context.coordinates.float(),future))
                temporal=temporal_metrics_v3(prediction,target,batch.coordinate_batch,history_frames=4)
                clash=sampled_clash_metrics(future,batch,physical)
                rows.append(dict(sample_id=sid,system=system,lag_ps=int(record['lag_ps']),variant=variant,seed=seed,
                    seconds=time.monotonic()-tick,metrics={**temporal.metrics,
                    **bond_errors(prediction,target,batch.coordinate_batch,frames),
                    **{k:v for k,v in clash.items() if k!='samples'}}))
                atomic_write_json(output/'dynamic_rows.json',rows)
            # Each structure is evaluated once, not once per duplicated lag view.
            if int(record['lag_ps'])==plan['lags'][0]:
                cpu=collate_clip_records([trajectory_record_to_static(record,frame_index=0)])
                for sigma in plan['static_sigmas_A']:
                    b=prepare_static_backbone_batch(model,cpu,device=device,
                        generator=torch.Generator(device=device).manual_seed(17),
                        perturbation=StaticPerturbationConfig((sigma,),0.))
                    source,_=sample_frame_source(b.source_center,generator=torch.Generator(device=device).manual_seed(plan['static_source_seed']))
                    state=source
                    for step in range(plan['static_steps']):
                        update=model.dit.forward_static(state,condition=b.condition,
                            flow_time=state.h.new_full((state.batch_size,),step/plan['static_steps']))
                        state=state.with_features(state.h+update.h/plan['static_steps'],state.v+update.v/plan['static_steps'])
                    output_trunk=model.decoder.forward_static(model.inverse(state),condition=b.condition)
                    without_trunk=model.decoder.forward_static(model.inverse(source),condition=b.condition)
                    without_flow=model.decoder.forward_static(model.inverse(b.source_center),condition=b.condition)
                    variants={'sampled_trunk':output_trunk,'source_no_trunk_ood':without_trunk,
                        'observed_latent_only_ood':without_flow,
                        'identity_input':replace(output_trunk,coordinates=b.condition.geometry.coordinates)}
                    for variant,decoded in variants.items():
                        if not bool(torch.isfinite(decoded.coordinates).all()):
                            raise FloatingPointError('nonfinite static generation')
                        loss=static_decoder_loss(decoded,b.reconstruction,StaticDecoderLossConfig(coordinate=1.,bond=.1))
                        static_rows.append(dict(system=system,sigma_A=sigma,variant=variant,
                            metrics={k:float(v) for k,v in loss.raw.items()}))
                    atomic_write_json(output/'static_rows.json',static_rows)
            print(json.dumps({'phase':'validation','sample_id':sid,'rows':len(rows)}),flush=True)
        atomic_write_json(output/'summary.json',dict(status='completed',dynamic_rows=len(rows),static_rows=len(static_rows),
            seconds=time.monotonic()-started,test_opened=False,
            scope=plan['scope'],static_ablation_scope='OOD sensitivity, not proof of trained-module necessity'))
    finally:
        dataset.close()


def _flatten(values, prefix=''):
    result={}
    if isinstance(values,dict) and values.get('available') is False:
        return result
    for key,value in values.items():
        name=f'{prefix}.{key}' if prefix else key
        if isinstance(value,dict):
            result.update(_flatten(value,name))
        elif type(value) in (int,float) and math.isfinite(value):
            result[name]=float(value)
    return result


def paired_comparison(first,second,*,bootstrap_seed=31):
    """Pair identical generated draws, aggregate seeds/lags within system first."""
    def index(rows):
        result={}
        for row in rows:
            if row['variant']!='generated': continue
            key=(row['sample_id'],row['seed'])
            if key in result: raise ValueError('duplicate evaluation key')
            result[key]=row
        return result
    a,b=index(first),index(second)
    if not a or a.keys()!=b.keys():
        raise ValueError('paired evaluation rows do not match')
    deltas=defaultdict(lambda:defaultdict(list));all_systems=set()
    for key,row in a.items():
        if row['system']!=b[key]['system']: raise ValueError('system mismatch')
        all_systems.add(row['system'])
        x,y=_flatten(row['metrics']),_flatten(b[key]['metrics'])
        for metric in x.keys()&y.keys():
            deltas[metric][row['system']].append(y[metric]-x[metric])
    result={}
    for metric,groups in deltas.items():
        values=np.asarray([mean(v) for _,v in sorted(groups.items())])
        rng=np.random.default_rng(bootstrap_seed)
        boot=values[rng.integers(0,len(values),size=(2000,len(values)))].mean(axis=1)
        result[metric]={'delta_second_minus_first':float(values.mean()),
            'ci95_system_bootstrap':np.quantile(boot,[.025,.975]).tolist(),
            'systems':len(values),'complete_system_coverage':set(groups)==all_systems,
            'paired_rows':sum(map(len,groups.values()))}
    return result


def compare_arms(output):
    rows={a:json.loads((output/a/'validation/dynamic_rows.json').read_text()) for a in ('S0','Sdec','Strunk')}
    protocols={a:json.loads((output/a/'validation/protocol.json').read_text()) for a in rows}
    for value in protocols.values(): value.pop('checkpoint_sha256')
    if any(v!=protocols['S0'] for v in protocols.values()):
        raise ValueError('validation protocols differ')
    return dict(schema='molvid.static_response.comparison.v1',
        note='Exploratory bounded validation; one training seed. CIs resample systems, not atoms; no automatic promotion.',
        comparisons={f'{b}_minus_{a}':paired_comparison(rows[a],rows[b])
            for a,b in [('S0','Sdec'),('S0','Strunk'),('Sdec','Strunk')]})
