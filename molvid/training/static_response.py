"""Paired static-response experiment from the ABCD/A1 baseline.

S0, Sdec and Strunk share dynamic exposure, initialization and optimizer
steps. Sdec/Strunk share the exact static forward; only auxiliary gradient
recipients differ. No Mamba, learned-memory switch, capacity expansion or
external data is introduced.
"""
from __future__ import annotations

from dataclasses import asdict
import gc
import json
from pathlib import Path
import subprocess
import time
import shutil
import os
from statistics import median

import numpy as np
import torch

from ..checkpoints import load_frame_joint_inference
from ..cli.train_frame_joint import _open_data
from ..config import load_config
from ..data.batch import collate_clip_records
from ..data.sampling import get_clip_specs
from ..data.static import trajectory_record_to_static
from ..flow.source import sample_frame_source
from ..runtime import atomic_write_json, configure_device, sha256_file
from .batches import prepare_frame_joint_batch
from .joint import FrameJointTrainer, JointLossConfig, SampledAuxiliaryConfig, frame_joint_loss
from .static_backbone import (StaticBackboneConfig, prepare_static_backbone_batch,
    static_backbone_loss, static_parameter_role, backward_static_loss)


def _shared_gradients(model):
    return {n:p.grad.detach().float().cpu().clone() for n,p in model.named_parameters()
            if static_parameter_role(n)=='shared' and p.grad is not None}


def _gradient_comparison(dynamic, static):
    result={}
    for group,prefix in [('shared',''),('dit','dit.'),('decoder','decoder.')]:
        d={n:g for n,g in dynamic.items() if n.startswith(prefix)}
        s={n:g for n,g in static.items() if n.startswith(prefix)}
        dn=sum(float(g.square().sum()) for g in d.values())**.5
        sn=sum(float(g.square().sum()) for g in s.values())**.5
        dot=sum(float((g*s[n]).sum()) for n,g in d.items() if n in s)
        result[group]={'dynamic_norm':dn,'static_norm':sn,'cosine':dot/(dn*sn) if dn*sn else None}
    return result


def calibrate_module_budget(rows, ratio=.1):
    """A frozen common scalar, capped by each group's median gradient budget.

    This is not a hard per-batch cap or adaptive gradient surgery. Quantiles
    and negative cosines remain visible; no clipping controller hides conflict.
    """
    import math
    if len(rows)<16 or not 0<ratio<=.5:
        raise ValueError('need >=16 train pairs and a conservative positive ratio')
    budgets={}
    for group in ('dit','decoder'):
        d=[r[group]['dynamic_norm'] for r in rows]
        s=[r[group]['static_norm'] for r in rows]
        if any(not math.isfinite(v) or v<=0 for v in d+s):
            raise ValueError('invalid calibration gradient')
        budgets[group]={'weight_cap':ratio*median(d)/median(s),
            'negative_cosines':sum(r[group]['cosine'] is not None and r[group]['cosine']<0 for r in rows)}
    weight=min(v['weight_cap'] for v in budgets.values())
    for group in budgets:
        ratios=[weight*r[group]['static_norm']/r[group]['dynamic_norm'] for r in rows]
        budgets[group].update(median_paired_ratio=median(ratios),max_paired_ratio=max(ratios))
    return weight,budgets


def profile_history(bucket: str, round_index: int) -> int:
    """Fixed-history records are built with H4, never reslice them as H8."""
    return 4 if bucket.startswith('fixed_history_dt_') or round_index%2==0 else 8


def run_experiment(output: Path, *, updates=512, seed=20261004,
              release_path=Path('/workspace/molvid-gm-memory-v3-coordination/recipe_release.json'),
              arms=('S0','Sdec','Strunk'), evaluate=True):
    if output.exists() or updates < 8 or updates > 2000 or len(set(arms)) != len(arms) or not set(arms)<={'S0','Sdec','Strunk'}:
        raise ValueError('fresh output, 8..2000 updates and unique known arms required')
    if not arms or shutil.disk_usage(output.parent if output.parent.exists() else Path('.')).free < 8 * 2**30:
        raise ValueError('nonempty arms and at least 8 GiB free disk required')
    if subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip():
        raise ValueError('commit experiment source before launching')
    release=json.loads(release_path.read_text());parent=release['selected_checkpoint']
    if release['test_opened'] is not False or release['selected_arm']!='A1':
        raise ValueError('requires the frozen validation-only A1 release')
    config_path=Path(release['configuration']['path'])
    if sha256_file(config_path)!=release['configuration']['sha256']: raise ValueError('A1 config hash changed')
    raw=load_config(config_path,schema='molvid.frame_joint.train.v1')
    device=configure_device('cuda',deterministic=True)
    train,_,data_hash,_,_,close=_open_data(raw['data'])
    output.mkdir(parents=True)
    try:
        # Frozen train-only coverage of ordinary/fixed-history, lag and atom-size.
        # One clip/update is a bounded profile, not the full 40k-token sampler.
        eligible=[s for s in get_clip_specs(train) if s.frames==16 and s.effective_tokens<=40000]
        buckets=sorted({s.time_bucket_id for s in eligible})
        if not buckets: raise ValueError('no eligible train clips')
        schedule=[]
        for i in range(max(updates,16)):
            bucket=buckets[i%len(buckets)]
            pool=sorted((s for s in eligible if s.time_bucket_id==bucket),key=lambda s:(s.atoms,s.sample_id))
            # Irrational stride spans sizes without selecting by loss or valid metrics.
            frac=((i//len(buckets))*.61803398875+.1)%1
            spec=pool[min(int(frac*len(pool)),len(pool)-1)]
            schedule.append(dict(index=spec.index,sample_id=spec.sample_id,atoms=spec.atoms,
                                 bucket=bucket,history=profile_history(bucket,i//len(buckets))))
        static_schedule=[dict(schedule[(i+7)%len(schedule)],frame=(i*7)%16) for i in range(len(schedule))]
        protocol=dict(schema='molvid.static_response.experiment.v1', integration_base='946917b',
            static_implementation_source='2f24ed4 (static paths only)',source_commit=subprocess.check_output(
            ['git','rev-parse','HEAD'],text=True).strip(),seed=seed,updates=updates,arms=list(arms),
            parent=parent,config_sha256=sha256_file(config_path),data_hash=data_hash,
            pid=os.getpid(),cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
            dynamic_schedule=schedule,static_schedule=static_schedule,split='train',test_opened=False,
            calibration_pairs=16,calibration_ratio=.1,calibration='min per-module median-ratio cap; one common lambda for Sdec/Strunk',
            precision='bf16',optimizer='fresh AdamW per arm, constant A1 LR, one step per dynamic+static pair',
            limitation='bounded one-clip continuation, one training seed, not full-budget pretraining or equal FLOPs',
            evaluation_enabled=evaluate,
            static_data='internal MD train frames; no external PDB/homology claim')
        atomic_write_json(output/'protocol.json',protocol)
        loss_config=JointLossConfig.resolve(raw['loss']);sampled=SampledAuxiliaryConfig.resolve(raw['sampled_auxiliary'])
        rates=raw['training']['stages'][0]['learning_rates']

        def load_model():
            torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
            return load_frame_joint_inference(parent['path'],expected_sha256=parent['sha256'],
                codec_path=raw['codec']['checkpoint'],codec_sha256=raw['codec']['sha256'],device=device)['model']

        def make_trainer(model, static=None):
            return FrameJointTrainer(model,loss_config=loss_config,sampled_config=sampled,
                learning_rates=rates,stage='continuation',amp=True,
                weight_decay=raw['training']['weight_decay'],grad_clip=raw['training']['grad_clip'],
                data_hash=data_hash,teacher_artifact_sha256=raw['codec']['sha256'],
                static_backbone_config=static,schedule_contract=protocol,
                checkpoint_flow_field=True)

        def prepare(model,i):
            row=schedule[i]
            cpu=collate_clip_records([train[row['index']]])
            if not cpu.frame_mask[:,row['history']:].any():
                raise ValueError('profile schedule leaves no valid future targets')
            return prepare_frame_joint_batch(model.target_teacher,cpu,
                device=device,normalizer=model,history_frames=row['history'])

        def prepare_static(model,i,generator):
            row=static_schedule[i]
            trajectory=train[row['index']]
            valid=np.flatnonzero(trajectory.get('frame_mask',np.ones(len(trajectory['x']),dtype=bool)))
            record=trajectory_record_to_static(trajectory,frame_index=int(valid[row['frame']%len(valid)]))
            return prepare_static_backbone_batch(model,collate_clip_records([record]),device=device,generator=generator)

        def rng(offset): return torch.Generator(device=device).manual_seed(seed+offset)

        if evaluate:
            from ..evaluation.static_response import validation_plan
            atomic_write_json(output/'validation_plan.json',validation_plan())
        weight=None
        if any(a != 'S0' for a in arms):
            model=load_model();trainer=make_trainer(model);model.train()
            flow_rng,aux_rng,static_rng,corrupt_rng=[rng(i) for i in (0,1,2,3)]
            rows=[]
            for i in range(16):
                tick=time.monotonic();batch=prepare(model,i)
                draw=trainer.flow.sample(batch.normalized_target,batch.source_center,generator=flow_rng)
                sources=(tuple(sample_frame_source(batch.source_center,generator=aux_rng)[0]
                    for _ in range(sampled.draws)) if sampled.active_at(i) else None)
                with trainer._autocast():
                    pred=model(draw.interpolated,context=batch.observed_context,query=batch.query,
                        flow_time=draw.flow_time,clean_future=batch.target_future,
                        decode_generated=True,decode_clean=True,decode_near=True,
                        sampled_sources=sources,sampled_steps=sampled.euler_steps,
                        checkpoint_sampled_steps=True,checkpoint_flow_field=True)
                    dynamic=frame_joint_loss(pred,trainer.flow.loss(pred.velocity,draw.target_velocity),batch,
                        draw.flow_time,stage=trainer.stage,config=loss_config,sampled_config=sampled)
                dynamic.total.backward();dg=_shared_gradients(model)
                dynamic_metrics={k:float(v.detach()) for k,v in dynamic.raw.items()}
                model.zero_grad(set_to_none=True)
                del pred,dynamic,draw,sources,batch
                sb=prepare_static(model,i,corrupt_rng)
                with trainer._autocast():
                    sl,metrics=static_backbone_loss(model,sb,StaticBackboneConfig(1.),generator=static_rng,successful_updates=i)
                backward_static_loss(sl,model,StaticBackboneConfig(1.));sg=_shared_gradients(model)
                if any(p.grad is not None for n,p in model.named_parameters() if static_parameter_role(n)!='shared'):
                    raise RuntimeError('static temporal gradient contamination')
                row=dict(pair=i,**_gradient_comparison(dg,sg),metrics=metrics,dynamic_metrics=dynamic_metrics,
                         seconds=time.monotonic()-tick)
                rows.append(row);atomic_write_json(output/'calibration_pairs.json',rows)
                print(json.dumps({'phase':'calibration','pair':i+1,**row['shared']}),flush=True)
                model.zero_grad(set_to_none=True);del sl,sb,dg,sg
            weight, budgets = calibrate_module_budget(rows)
            atomic_write_json(output/'calibration.json',dict(weight=weight,ratio=.1,train_pairs=16,
                configuration=asdict(StaticBackboneConfig(weight,seed=seed+2)),
                module_budgets=budgets, same_lambda_in_both_static_arms=True))
            del trainer,model;gc.collect();torch.cuda.empty_cache()

        for arm in arms:
            if shutil.disk_usage(output).free < 4 * 2**30:
                raise RuntimeError('disk reserve below 4 GiB; stop before next arm, do not delete user files')
            model=load_model()
            static_config = (StaticBackboneConfig(weight,seed=seed+2,
                gradient_scope='decoder_only' if arm=='Sdec' else 'shared') if arm!='S0' else None)
            trainer=make_trainer(model,static_config)
            report=trainer.load_warm_start(parent['path'],expected_sha256=parent['sha256'])
            arm_dir=output/arm;arm_dir.mkdir();atomic_write_json(arm_dir/'warm_start.json',report.as_dict())
            atomic_write_json(arm_dir/'contracts.json',trainer.contracts())
            flow_rng,aux_rng=rng(0),rng(1);rows=[];started=time.monotonic()
            torch.cuda.reset_peak_memory_stats()
            for i in range(updates):
                tick=time.monotonic();batch=prepare(model,i)
                static=prepare_static(model,i,trainer.static_corruption_generator) if arm!='S0' else None
                metrics=trainer.train_step(batch,generator=flow_rng,sampled_generator=aux_rng,static_batch=static)
                torch.cuda.synchronize()
                metrics.update(seconds=time.monotonic()-tick,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    sample_id=schedule[i]['sample_id'],history=schedule[i]['history'])
                metrics['joint_clip_scale']=min(1.,trainer.grad_clip/(metrics['grad_norm']+1e-6))
                rows.append(metrics);atomic_write_json(arm_dir/'steps.json',rows)
                if (i+1)%8==0: print(json.dumps({'phase':'training','arm':arm,'step':i+1,'loss':metrics['loss']}),flush=True)
                del batch,static
            checkpoint=arm_dir/'final.pt'
            trainer.save_checkpoint(checkpoint,cursor={'dynamic':updates,'static':updates if arm!='S0' else 0},
                generator=flow_rng,sampled_generator=aux_rng)
            atomic_write_json(arm_dir/'summary.json',dict(status='training_completed_pending_inference_check',updates=updates,
                seconds=time.monotonic()-started,checkpoint_sha256=sha256_file(checkpoint),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),test_opened=False))
            # Strict inference reconstruction precedes completed status/evaluation.
            del trainer,model;gc.collect();torch.cuda.empty_cache()
            check=load_frame_joint_inference(checkpoint,expected_sha256=sha256_file(checkpoint),
                codec_path=raw['codec']['checkpoint'],codec_sha256=raw['codec']['sha256'],device=device)
            if evaluate:
                from ..evaluation.static_response import evaluate_response
                evaluate_response(check['model'], raw, release, output=arm_dir/'validation',
                    checkpoint_sha256=sha256_file(checkpoint))
            del check;gc.collect();torch.cuda.empty_cache()
            summary=json.loads((arm_dir/'summary.json').read_text())
            summary.update(status='completed',inference_reload_passed=True,validation_completed=evaluate)
            atomic_write_json(arm_dir/'summary.json',summary)
        if evaluate and set(arms)=={'S0','Sdec','Strunk'}:
            from ..evaluation.static_response import compare_arms
            atomic_write_json(output/'comparison.json',compare_arms(output))
        atomic_write_json(output/'summary.json',dict(status='completed',arms=list(arms),updates=updates,test_opened=False))
    except Exception as exc:
        atomic_write_json(output/'failure.json',dict(type=type(exc).__name__,message=str(exc)))
        raise
    finally:
        close()
