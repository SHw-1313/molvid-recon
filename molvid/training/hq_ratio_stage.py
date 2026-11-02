"""Matched routed-adapter H/Q study: 128 candidate joint/Q1 slot pairs.

All five arms supervise frames12:36 once. Only each chunk's real observed
history and query length change. RF and auxiliary draws cover all24 targets
before slicing; the original auxiliary visit-H is independent of model-H.
A scientifically excluded joint window skips both slots without compressing
the256-slot LR schedule; its Q1 data cursor advances without model/RNG work.
"""
import copy
import hashlib
import json
from pathlib import Path
import random
import subprocess
import time

import numpy as np
import torch

from ..codec.frame import slice_clip_frames
from ..data.batch import collate_clip_records
from ..flow.coordinate_state import CoordinateStateDenoiser, coordinate_state_objective
from ..flow.local_geometry_loss import ligand_geometry_band_loss
from ..runtime import atomic_write_json
from .coordinate_state import (LigandSourceGeometry, continuation_learning_rate,
    load_coordinate_weights, _global_rng_state, _set_global_rng_state)
from .kernel_tasks import KernelTasks
from .predictive_data import atomic_torch_save
from .q4_align_stage import _rf_with_draw, _tail_with_noise

ARMS = dict(h4q4=(4,4), h4q8=(4,8), h8q4=(8,4), h8q8=(8,8), h12q12=(12,12))


class HQJointWindows:
    """Bounded prepared TRAIN records, with one resumable manifest cursor."""
    def __init__(self, path, *, data_config=None):
        self.path = Path(path)
        raw = self.path.read_bytes(); manifest = json.loads(raw)
        if (manifest.get('status') != 'complete' or not manifest.get('all_candidates_processed')
                or manifest.get('failures') != []):
            raise ValueError('HQ joint manifest must finish all candidates without read/identity failures')
        if manifest.get('schema')!='molvid.hq_true_contiguous36.v1' or manifest.get('seed')!=71:
            raise ValueError('Unexpected HQ data schema/seed')
        if data_config is not None and manifest['data_config']!=data_config:
            raise ValueError('TRAIN manifest and configured data sources differ')
        rows = manifest['rows']; exclusions = manifest.get('exclusions', [])
        self.rows = {int(r['plan_index']):r for r in rows}
        self.exclusions = {int(r['plan_index']):r for r in exclusions}
        if len(self.rows)!=len(rows) or len(self.exclusions)!=len(exclusions):
            raise ValueError('Duplicate joint plan indices')
        if (set(self.rows)&set(self.exclusions) or set(self.rows)|set(self.exclusions)!=set(range(128))
                or not self.rows):
            raise ValueError('Valid/excluded candidates must partition the original128 indices')
        self.digest = hashlib.sha256(raw).hexdigest(); self.cursor = 0

    def next(self):
        if self.cursor >= 128: raise StopIteration('No cyclic reuse of the HQ plan')
        if self.cursor in self.exclusions:
            index=self.cursor;self.cursor+=1
            return None,'dynamic',(4,8)[index%2],list(range(12,36)),dict(
                plan_index=index,excluded=True,exclusion=self.exclusions[index],aux_group_history=(4,8)[index%2])
        row = self.rows[self.cursor]
        path = Path(row['record_path'])
        if not path.is_absolute(): path = self.path.parent/path
        record = torch.load(path,map_location='cpu',weights_only=False)['record']
        if record.get('split') != 'train': raise ValueError('Only explicit TRAIN records allowed')
        if (record.get('original_sample_id')!=row['sample_id'] or
                record.get('topology_fingerprint')!=row['topology_fingerprint']):
            raise ValueError('Prepared record identity/topology differs from the frozen manifest')
        batch = collate_clip_records([record])
        if batch.frames != 36 or not bool(batch.frame_mask.all()):
            raise ValueError('Expected complete genuine36-frame record')
        if batch.atom_count!=row['atom_count']:raise ValueError('Prepared atom count differs')
        if not bool((batch.time_ps[:,1:] > batch.time_ps[:,:-1]).all()):
            raise ValueError('TRAIN physical times must increase')
        history = int(row['aux_group_history'])
        if history != (4,8)[self.cursor % 2]:
            raise ValueError('Prepared auxiliary grouping must retain original joint-ordinal H4/H8')
        identity = {k:row[k] for k in ('domain','system','replica','split','store','store_index',
            'source_topology','source_trajectory','topology_fingerprint','atom_count',
            'original_native_origin','actual_native_origin','origin_shift_reason','native_frame_indices',
            'absolute_time_ps') if k in row}
        identity.update(plan_index=row['plan_index'],joint_ordinal=row['joint_ordinal'],
                        aux_group_history=history,record_path=str(path),original_sample_id=row['sample_id'],
                        clock_sampling='fixed_train_hq36_plan',time_bucket_id=batch.time_bucket_id[0],
                        joint_reference_index=11,joint_absolute_t0_ps=float(record['absolute_time_ps'][11]),
                        joint_target_native_frame_indices=[int(i) for i in record['native_frame_indices'][12:36]],
                        joint_target_absolute_time_ps=[float(t) for t in record['absolute_time_ps'][12:36]])
        if identity.get('domain') not in ('protein','complex'):
            raise ValueError('Joint identity needs original protein/complex domain')
        self.cursor += 1
        return batch,'dynamic',history,list(range(12,36)),identity

    def state_dict(self):
        return dict(schema='molvid.hq_joint_windows.v1',manifest=str(self.path),
                    manifest_sha256=self.digest,candidates=128,effective_joint_windows=len(self.rows),
                    excluded_plan_indices=sorted(self.exclusions),cursor=self.cursor)

    def load_state_dict(self, saved):
        if (saved['schema'] != 'molvid.hq_joint_windows.v1' or saved['manifest_sha256'] != self.digest
                or saved['effective_joint_windows'] != len(self.rows) or not 0 <= saved['cursor'] <= 128
                or saved['excluded_plan_indices'] != sorted(self.exclusions)):
            raise ValueError('HQ data plan changed on exact resume')
        self.cursor = int(saved['cursor'])


def next_slot(kernel,joint,step):
    """Keep original slot identities, including a data-only excluded Q1 read."""
    index=step//2;skip=index in joint.exclusions
    if step%2:
        batch,task,history,queries,identity,band,kind=kernel.next(step,joint)
        identity.update(matched_joint_plan_index=index,excluded=skip)
        if skip:identity['exclusion']=joint.exclusions[index]
    else:
        if joint.cursor!=index or kernel.counts['joint']!=index:
            raise ValueError('Joint plan cursor no longer matches original slot')
        batch,task,history,queries,identity=joint.next();kind='joint';band=None
        identity={k:v for k,v in identity.items() if k!='task'}
        identity.update(kernel_task='joint',task_ordinal=index+1,excluded=skip)
        kernel.counts['joint']+=1
        key=f'{identity.get("domain","excluded")}|joint|H{history}|lagjoint|joint'
        kernel.crossings[key]=kernel.crossings.get(key,0)+1
    return batch,task,history,queries,identity,band,kind,skip


def joint_hq_chunks(batch, history, query):
    """Yield teacher history, target, times, slice into full24 noise, metadata."""
    if (history,query) not in ARMS.values() or batch.frames != 36 or not bool(batch.frame_mask.all()):
        raise ValueError('Expected declared H/Q and complete36-frame TRAIN window')
    for offset in range(0,24,query):
        begin = 12+offset
        observed = slice_clip_frames(batch,begin-history,begin)
        yield (observed,batch.x[begin:begin+query],batch.time_ps[:,begin:begin+query],
               slice(offset,offset+query),dict(history_indices=list(range(begin-history,begin)),
               target_indices=list(range(begin,begin+query)),
               reference_time_ps=observed.time_ps[:,-1].tolist()))


def backward_joint_hq(model, batch, history, query, *, generator, bf16):
    target = batch.x[12:36]
    progress = torch.rand(batch.batch_size,device=target.device,generator=generator)
    noise = torch.randn(target.shape,device=target.device,generator=generator)
    total = raw = 0.; chunks = []
    for observed,part,times,selection,metadata in joint_hq_chunks(batch,history,query):
        weight = len(part)/24
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=bf16):
            condition = model.prepare_condition(observed,task='dynamic',route_task='joint')
            loss,mse = _rf_with_draw(model,condition,part,times,progress,noise[selection])
            weighted = weight*loss
        weighted.backward(); total += float(weighted.detach()); raw += weight*mse
        chunks.append(dict(metadata,weight=weight,RF_loss=float(loss.detach()),
                           source_epsilon_sum=float(noise[selection].double().sum()),
                           source_epsilon_square_sum=float(noise[selection].double().square().sum())))
        del condition,loss,weighted
    return dict(RF_loss=total,raw_coordinate_MSE_A2=raw,progress_mean=float(progress.mean()),
                progress_values=progress.tolist(),source_epsilon_sum=float(noise.double().sum()),
                source_epsilon_square_sum=float(noise.double().square().sum()),
                valid_atoms=int(batch.loss_mask.sum()),query_frames=24,chunks=chunks,RF_backward_calls=len(chunks))


def backward_hq_auxiliary(model, batch, history, query, auxiliary, *, bf16):
    """One original visit event, Q/24 chunk weights, unchanged full24 band."""
    if not bool((batch.loss_mask & (batch.component_id>0)).any()):
        return dict(ligand_geometry_applied=False,ligand_geometry_weighted_loss=0.,
                    ligand_geometry_reason='no_valid_ligand')
    began=time.monotonic();target=batch.x[12:36];total=0.;chunks=[]
    with auxiliary.isolated_rng():
        noise=torch.randn(target.shape,device=target.device,generator=auxiliary.generator)
        for observed,part,times,selection,metadata in joint_hq_chunks(batch,history,query):
            weight=len(part)/24
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=bf16):
                endpoint=_tail_with_noise(model,observed,times,noise[selection],auxiliary.config['tail_intervals'])
                penalty,metrics=ligand_geometry_band_loss(endpoint,part,observed,auxiliary.config['band'],
                                                         band_reference_A=target)
                weighted=weight*auxiliary.config['weight']*penalty
            weighted.backward();value=float(weighted.detach());total+=value
            chunks.append(dict(metadata,weight=weight,weighted_loss=value,metrics=metrics))
            del endpoint,penalty,weighted
        auxiliary.events+=1
    return dict(ligand_geometry_applied=True,ligand_geometry_weighted_loss=total,
                ligand_geometry_weight=auxiliary.config['weight'],ligand_geometry_events=auxiliary.events,
                ligand_geometry_counts=dict(auxiliary.counts),ligand_geometry_chunks=chunks,
                geometry_band_frames=24,ligand_geometry_seconds=time.monotonic()-began)


def backward_slot(model,batch,history,query,*,kind,identity,band_reference,generator,auxiliary,bf16):
    """Accumulate one slot's RF+aux; caller performs one common clip/AdamW."""
    if kind=='joint':
        metrics=backward_joint_hq(model,batch,history,query,generator=generator,bf16=bf16)
        aux_history=identity['aux_group_history'];route_task='joint'
    elif kind=='kernel':
        if query!=1 or batch.frames!=history+1: raise ValueError('Q1 keeps its actual original history')
        route_task='q1';aux_history=history
        observed=slice_clip_frames(batch,0,history);target=batch.x[history:];times=batch.time_ps[:,history:]
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=bf16):
            condition=model.prepare_condition(observed,task='dynamic',route_task=route_task)
            loss,metrics=coordinate_state_objective(model,condition,target,times,generator=generator)
        loss.backward();metrics.update(RF_loss=float(loss.detach()),RF_backward_calls=1)
        del condition,loss
    else: raise ValueError('Unknown HQ task stream')
    triggered=auxiliary.due(identity,aux_history)
    metrics.update(ligand_geometry_triggered=triggered,ligand_geometry_counts=dict(auxiliary.counts),
                   aux_group_history=aux_history)
    if triggered:
        if kind=='joint':metrics.update(backward_hq_auxiliary(model,batch,history,query,auxiliary,bf16=bf16))
        else:metrics.update(auxiliary.backward(model,observed,target,times,bf16=bf16,
                                              band_reference=band_reference,route_task=route_task))
    metrics.update(loss=metrics['RF_loss']+metrics.get('ligand_geometry_weighted_loss',0.),
                   route_task=route_task,task_adapter_mixing_weights=list(model.task_adapters.mixing_weights(route_task)),
                   geometry_band_frames=24 if kind=='joint' else len(band_reference))
    return metrics


def validate_settings(settings,seed):
    arm=settings['hq_ratio']['arm']
    if arm not in ARMS or seed!=71 or settings['updates']!=256 or settings['warmup_updates']!=128:
        raise ValueError('Frozen five-arm seed71 /256-slot /warmup128 recipe required')
    if (settings['task_adapter']!=dict(mode='routed',bottleneck=64) or settings['scheme']!='flow'
            or settings['kernel_training']['arm']!='history_mix' or settings.get('static_replay')
            or settings.get('reference_concat',False) or settings.get('q4_align')):
        raise ValueError('Only declared routed HQ stage, without extra mechanisms')
    if settings['ligand_geometry']['weight_status']!='calibrated':
        raise ValueError('Original calibrated ligand auxiliary required')
    return arm


def train_hq_ratio(settings,output,seed,*,warm_start=None,resume=None,stop_after=None):
    arm=validate_settings(settings,seed);history,query=ARMS[arm]
    if (warm_start is None)==(resume is None):raise ValueError('Choose parent weights or exact resume')
    if stop_after is not None and not 0<stop_after<=256:raise ValueError('stop_after is a256-schedule slot')
    if not torch.cuda.is_available():raise RuntimeError('Real CUDA required; no CPU model fallback')
    output=Path(output);output.mkdir(parents=True,exist_ok=resume is not None)
    torch.set_num_threads(settings['cpu_threads']);random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    device=torch.device('cuda:0');model=CoordinateStateDenoiser(settings).to(device).train()
    parameters=[p for p in model.parameters() if p.requires_grad]
    if any(p.requires_grad!=n.startswith('task_adapters.') for n,p in model.named_parameters()):
        raise ValueError('Only frozen-parent task adapters may train')
    if sum(p.numel() for p in parameters)!=166528:raise ValueError('Adapter capacity changed')
    optimizer=torch.optim.AdamW(parameters,lr=settings['learning_rate'],weight_decay=settings['weight_decay'])
    generator=torch.Generator(device=device).manual_seed(seed+20261008)
    joint=HQJointWindows(settings['hq_ratio']['manifest'],data_config=settings['data'])
    effective=len(joint.rows)
    if settings['hq_ratio'].get('effective_joint_windows',effective)!=effective:
        raise ValueError('Explicit effective window count differs from completed manifest')
    # Reuse the original512-plan stream helper unchanged, but only consume its
    # first128 Q1 entries. This constructor guard is NOT the actual LR schedule.
    stream_settings=copy.deepcopy(settings);stream_settings['updates']=512
    kernel=KernelTasks(stream_settings,seed,device,generator)
    ligand=LigandSourceGeometry(settings['ligand_geometry'],seed,device)
    kernel_ligand=LigandSourceGeometry(settings['ligand_geometry'],seed+20261027,device)
    start=0;prior_seconds=0.;last_optimizer_lr=None
    exposure=dict(updates=0,dynamic_examples=0,static_examples=0,target_frames=0,atom_query_pairs=0,
                  joint_updates=0,kernel_updates=0,joint_target_frames=0,kernel_target_frames=0,
                  joint_atom_query_pairs=0,kernel_atom_query_pairs=0,RF_backward_calls=0,
                  skipped_joint_slots=0,skipped_kernel_slots=0)
    if warm_start is not None:initialization=load_coordinate_weights(model,warm_start)
    else:
        saved=torch.load(resume,map_location='cpu',weights_only=False)
        if (saved.get('stage_schema')!='molvid.hq_ratio.v1' or saved['settings']!=settings
                or saved['seed']!=seed or saved['step']!=saved['slot']):raise ValueError('HQ exact resume mismatch')
        model.load_state_dict(saved['model'],strict=True);optimizer.load_state_dict(saved['optimizer'])
        joint.load_state_dict(saved['joint_windows']);kernel.load_state_dict(saved['kernel_tasks'])
        generator.set_state(saved['generator_rng']);ligand.load_state_dict(saved['ligand_geometry'])
        kernel_ligand.load_state_dict(saved['kernel_ligand_geometry']);_set_global_rng_state(saved['global_rng'])
        start=saved['slot'];prior_seconds=saved['elapsed_seconds'];exposure=copy.deepcopy(saved['exposure'])
        expected_updates=sum(i//2 not in joint.exclusions for i in range(start))
        if exposure['updates']!=expected_updates or saved['optimizer_updates']!=expected_updates:
            raise ValueError('Saved update count differs from valid slot prefix')
        last_optimizer_lr=saved['last_optimizer_lr']
        initialization=saved['initialization_report'];del saved
    source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    atomic_write_json(output/'recipe.json',dict(settings=settings,seed=seed,model=model.contract(),source_commit=source_commit,
        initialization_report=initialization,resume=str(resume),CUDA_device=torch.cuda.get_device_name(0),
        planned_slots=256,effective_joint_windows=effective,effective_updates=2*effective,planned_target_frames=25*effective,
        joint_manifest=joint.state_dict(),teacher_history=True,aux_group_history='original joint ordinal H4/H8',
        joint_band='all24 real targets shared across chunks',kernel_plan='original first128 history_mix rows',
        excluded_pairs='same plan index joint/Q1 skipped; data cursor advances; original LR slot retained'))
    began=time.monotonic()
    def save(slot,final=False):
        elapsed=prior_seconds+time.monotonic()-began;rng=_global_rng_state()
        state=dict(schema='molvid.coordinate_state.checkpoint.v1',stage_schema='molvid.hq_ratio.v1',
            model=model.state_dict(),optimizer=optimizer.state_dict(),settings=settings,seed=seed,
            step=slot,slot=slot,optimizer_updates=exposure['updates'],contract=model.contract(),
            effective_joint_windows=effective,effective_updates=2*effective,last_optimizer_lr=last_optimizer_lr,
            joint_windows=joint.state_dict(),kernel_tasks=kernel.state_dict(),exposure=copy.deepcopy(exposure),
            initialization_report=initialization,source_commit=source_commit,elapsed_seconds=elapsed,
            generator_rng=generator.get_state(),global_rng=rng,torch_rng=rng['torch_cpu'],cuda_rng=rng['cuda'],
            python_rng=rng['python'],numpy_rng=rng['numpy'],ligand_geometry=ligand.state_dict(),
            kernel_ligand_geometry=kernel_ligand.state_dict())
        path=output/('coordinate_final.pt' if final else 'checkpoint_latest.pt');atomic_torch_save(path,state)
        if slot in (32,256) and not (output/f'checkpoint_step{slot}.pt').exists():
            atomic_torch_save(output/f'checkpoint_step{slot}.pt',state)
        atomic_write_json(output/'status.json',dict(status='completed' if final else 'training',step=slot,slot=slot,
            optimizer_updates=exposure['updates'],effective_updates=2*effective,effective_joint_windows=effective,
            checkpoint=str(path),elapsed_seconds=elapsed,GPU_hours=elapsed/3600,
            peak_cuda_bytes=torch.cuda.max_memory_allocated(),exposure=dict(exposure)))
    try:
        with (output/'updates.jsonl').open('a',buffering=1) as log:
            for step in range(start,256):
                batch,task,original_h,original_queries,identity,band,kind,skip=next_slot(kernel,joint,step)
                if task!='dynamic' or (batch is not None and not bool(batch.frame_mask.all())):
                    raise ValueError('Complete TRAIN MD only')
                if kind=='joint':
                    if identity['aux_group_history']!=original_h:raise ValueError('Aux visit grouping changed')
                    actual_h,actual_q=history,query;queries=list(range(12,36))
                else:actual_h,actual_q=original_h,1;queries=original_queries
                optimizer.param_groups[0]['lr']=continuation_learning_rate(settings,step)
                if skip:
                    exposure[f'skipped_{kind}_slots']+=1
                    metrics=dict(loss=None,RF_loss=None,grad_norm=None,RF_backward_calls=0,ligand_geometry_triggered=False)
                else:
                    with kernel.isolated_rng(kind) as active_generator:
                        batch=batch.to(device)
                        if band is not None:band=band.to(device)
                        optimizer.zero_grad(set_to_none=True)
                        auxiliary=ligand if kind=='joint' else kernel_ligand
                        metrics=backward_slot(model,batch,actual_h,actual_q,kind=kind,identity=identity,
                            band_reference=band,generator=active_generator,auxiliary=auxiliary,bf16=settings['bf16'])
                        norm=torch.nn.utils.clip_grad_norm_(parameters,settings['clip_grad'],error_if_nonfinite=True)
                        if any(p.grad is not None for n,p in model.named_parameters() if not n.startswith('task_adapters.')):
                            raise RuntimeError('Frozen base received a gradient')
                        metrics['grad_norm']=float(norm)
                        metrics['task_adapter_gradient_norms_after_clip']={f'branch{i}':sum(float(p.grad.float().square().sum())
                            for p in branch.parameters() if p.grad is not None)**.5 for i,branch in enumerate(model.task_adapters.branches)}
                        optimizer.step();last_optimizer_lr=optimizer.param_groups[0]['lr']
                        nframes=len(queries)*batch.batch_size;nap=len(queries)*int(batch.loss_mask.sum())
                        exposure['updates']+=1;exposure[f'{kind}_updates']+=1;exposure['dynamic_examples']+=batch.batch_size
                        exposure['target_frames']+=nframes;exposure['atom_query_pairs']+=nap
                        exposure[f'{kind}_target_frames']+=nframes;exposure[f'{kind}_atom_query_pairs']+=nap
                        exposure['RF_backward_calls']+=metrics['RF_backward_calls']
                row=dict(identity);row.update(metrics)
                row.update(step=step+1,slot=step+1,optimizer_updates=exposure['updates'],task=task,
                    sample=batch.sample_id[0] if batch is not None else identity.get('sample'),
                    optimized=not skip,history=actual_h,chunk_query_frames=actual_q,queries=list(queries),
                    query_frames=len(queries),learning_rate=optimizer.param_groups[0]['lr'],exposure=dict(exposure),
                    last_optimizer_lr=last_optimizer_lr,
                    kernel_counts=dict(kernel.counts),kernel_crossings=dict(kernel.crossings),
                    elapsed_seconds=prior_seconds+time.monotonic()-began)
                log.write(json.dumps(row,allow_nan=False)+'\n')
                if step<start+4 or (step+1)%32==0:print(json.dumps(row,allow_nan=False),flush=True)
                if step+1==32 or (step+1)%settings['save_every']==0:save(step+1)
                if stop_after is not None and step+1>=stop_after and step+1<256:
                    save(step+1);return
            if (exposure['updates']!=2*effective or exposure['target_frames']!=25*effective
                    or kernel.counts!=dict(joint=128,kernel=128) or joint.cursor!=128):
                raise RuntimeError('Final matched exposure differs')
            save(256,final=True)
    except Exception as error:
        atomic_write_json(output/'failure.json',dict(error=repr(error),exposure=exposure,
            elapsed_seconds=prior_seconds+time.monotonic()-began));raise
    finally:kernel.close()
