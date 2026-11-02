"""Bounded real-TRAIN CUDA checks for the frozen HQ trainer helpers.

Five RF backwards on one small36 window; one largest H12Q12 field/backward;
one genuine ligand H12Q12 auxiliary event. No optimizer run or sampling matrix.
The engineer's outer runner owns timing/budget receipts and GPU selection.
"""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import random
import sys
import time

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from molvid.flow.coordinate_state import CoordinateStateDenoiser
from molvid.training.coordinate_state import load_coordinate_weights,LigandSourceGeometry,_global_rng_state
from molvid.training.hq_ratio_stage import (ARMS,HQJointWindows,validate_settings,joint_hq_chunks,
    backward_joint_hq,backward_hq_auxiliary)
from molvid.training.q4_align_stage import _rf_with_draw

RUN=ROOT/'runs/flow_hq_ratio_20261102'
TASK=ROOT/'tasks/flow-hq-ratio-20261102'
PARENT=Path('/workspace/MOLVID-FLOW-LIGAND-4096-20261026/runs/flow_ligand_4096_20261026/ligand/coordinate_final.pt')


def equal(x,y):
    if isinstance(x,torch.Tensor):return isinstance(y,torch.Tensor) and torch.equal(x,y)
    if isinstance(x,np.ndarray):return isinstance(y,np.ndarray) and np.array_equal(x,y)
    if isinstance(x,dict):return isinstance(y,dict) and x.keys()==y.keys() and all(equal(x[k],y[k]) for k in x)
    if isinstance(x,(list,tuple)):return len(x)==len(y) and all(equal(a,b) for a,b in zip(x,y))
    return x==y


def safe(x):
    if isinstance(x,dict):return {k:safe(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [safe(v) for v in x]
    if isinstance(x,float) and not math.isfinite(x):return repr(x)
    return x


def gradients(model):
    assert all(p.requires_grad==n.startswith('task_adapters.') for n,p in model.named_parameters())
    assert all(p.grad is None for n,p in model.named_parameters() if not n.startswith('task_adapters.'))
    groups={};total=0.
    for i,branch in enumerate(model.task_adapters.branches):
        values=[p.grad.detach().float() for p in branch.parameters() if p.grad is not None]
        assert values and all(bool(torch.isfinite(v).all()) for v in values)
        squared=sum(float(v.square().sum()) for v in values);total+=squared
        groups[f'branch{i}']=squared**.5
    assert total>0 and all(v>0 for v in groups.values())
    return dict(total_norm=total**.5,branches=groups,finite=True,base_no_gradient=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,default=RUN)
    p.add_argument('--allocator-GiB',type=float,default=32.)
    p.add_argument('--output',type=Path)
    args=p.parse_args();output=args.output or args.run/'training_checks.json'
    if os.environ.get('CONDA_DEFAULT_ENV')!='torch-ito':raise RuntimeError('Existing torch-ito required')
    if output.exists():raise FileExistsError(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    began=time.monotonic();result=dict(status='running',parent=str(PARENT),checks={},GPU_selection='external runner',
        formal_loop_validation='first32 then exact resume; no duplicate32 update run in this check')
    def save():output.write_text(json.dumps(safe(result),indent=2,allow_nan=False)+'\n')
    save()
    try:
        if not torch.cuda.is_available():raise RuntimeError('Actual CUDA required; no CPU fallback')
        torch.set_num_threads(2)
        props=torch.cuda.get_device_properties(0)
        if not 0<args.allocator_GiB*1024**3<=props.total_memory:raise ValueError('Invalid allocator cap')
        torch.cuda.set_per_process_memory_fraction(args.allocator_GiB*1024**3/props.total_memory)
        random.seed(71);np.random.seed(71);torch.manual_seed(71)
        configs={arm:json.loads((TASK/'configs'/f'{arm}.json').read_text()) for arm in ARMS}
        for arm,cfg in configs.items():
            assert validate_settings(cfg,71)==arm
            other=copy.deepcopy(cfg);other['hq_ratio']['arm']='h4q4'
            assert other==configs['h4q4'],'Only declared H/Q may vary'
        cfg=configs['h4q4'];windows=HQJointWindows(cfg['hq_ratio']['manifest'],data_config=cfg['data'])
        assert Path(cfg['hq_ratio']['manifest']).resolve()==(args.run/'data/manifest.json').resolve()
        rows=list(windows.rows.values())
        small=min(rows,key=lambda r:(r['atom_count'],r['plan_index']))
        maximum=max(rows,key=lambda r:(r['atom_count'],-r['plan_index']))
        complexes=[r for r in rows if r['domain']=='complex']
        if not complexes:raise ValueError('A real ligand auxiliary check requires a valid complex')
        # Fixed already-used small ligand fixture when valid; otherwise smallest
        # complex selected by size, before any model output is inspected.
        aux_row=next((r for r in complexes if r['system']=='4UIZ'),min(complexes,key=lambda r:(r['atom_count'],r['plan_index'])))
        result['data']=dict(manifest=windows.state_dict(),selection={label:dict(
            plan_index=row['plan_index'],system=row['system'],atoms=row['atom_count'],record_path=row['record_path'])
            for label,row in [('small',small),('maximum',maximum),('auxiliary',aux_row)]})
        model=CoordinateStateDenoiser(cfg).cuda().train();report=load_coordinate_weights(model,PARENT)
        base={k:v.detach().cpu().clone() for k,v in model.state_dict().items() if not k.startswith('task_adapters.')}
        result['model']=dict(total_parameters=sum(p.numel() for p in model.parameters()),
            trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),initialization=report,
            device=torch.cuda.get_device_name(0),allocator_GiB=args.allocator_GiB)
        assert result['model']['trainable_parameters']==166528
        def load(row):
            # Direct bounded record selection for this check, not advancement of
            # a formal training sampler and never an incomplete-manifest bypass.
            windows.cursor=row['plan_index'];batch,task,_,_,identity=windows.next()
            assert task=='dynamic' and batch.frames==36
            return batch.to('cuda'),identity
        batch,identity=load(small);small_results={};draws=[];states=[]
        for arm,(h,q) in ARMS.items():
            model.zero_grad(set_to_none=True);torch.cuda.reset_peak_memory_stats();start=time.monotonic()
            generator=torch.Generator(device='cuda').manual_seed(71+20261008)
            metrics=backward_joint_hq(model,batch,h,q,generator=generator,bf16=cfg['bf16'])
            grad=gradients(model);torch.cuda.synchronize()
            assert math.isfinite(metrics['RF_loss'])
            assert len(metrics['chunks'])==24//q and metrics['query_frames']==24
            assert [i for c in metrics['chunks'] for i in c['target_indices']]==list(range(12,36))
            assert abs(sum(c['weight'] for c in metrics['chunks'])-1)<1e-12
            for j,c in enumerate(metrics['chunks']):
                begin=12+j*q;assert c['history_indices']==list(range(begin-h,begin))
            draws.append({k:metrics[k] for k in ('progress_values','source_epsilon_sum','source_epsilon_square_sum')})
            states.append(generator.get_state().clone())
            small_results[arm]=dict(seconds=time.monotonic()-start,gradient=grad,metrics=metrics,
                peak_cuda_bytes=torch.cuda.max_memory_allocated(),reserved_cuda_bytes=torch.cuda.max_memory_reserved())
            result['checks']['small_five_HQ']=small_results;save()
        assert all(equal(draws[0],d) for d in draws[1:]) and all(torch.equal(states[0],s) for s in states[1:])
        result['checks']['shared_draws']=dict(progress_and_full24_normal_statistics_exact=True,generator_end_states_exact=True)
        del batch;model.zero_grad(set_to_none=True)
        batch,_=load(maximum);torch.cuda.reset_peak_memory_stats();start=time.monotonic()
        # Exactly one H12Q12 field/backward; not a full24 slot or35NFE sample.
        observed,target,times,_,metadata=next(joint_hq_chunks(batch,12,12))
        generator=torch.Generator(device='cuda').manual_seed(71+20261008)
        progress=torch.rand(batch.batch_size,device='cuda',generator=generator)
        noise=torch.randn(target.shape,device='cuda',generator=generator)
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=cfg['bf16']):
            condition=model.prepare_condition(observed,task='dynamic',route_task='joint')
            loss,mse=_rf_with_draw(model,condition,target,times,progress,noise)
        loss.backward();grad=gradients(model);torch.cuda.synchronize()
        assert torch.isfinite(loss)
        result['checks']['maximum_H12Q12']=dict(metadata,seconds=time.monotonic()-start,RF_loss=float(loss.detach()),
            atoms=batch.atom_count,query_frames=12,gradient=grad,raw_coordinate_MSE_A2=mse,
            peak_cuda_bytes=torch.cuda.max_memory_allocated(),reserved_cuda_bytes=torch.cuda.max_memory_reserved(),
            profile_scope='one full-atom H12Q12 field and backward; no optimizer or sampler')
        save();del batch,observed,target,times,condition,loss,progress,noise
        model.zero_grad(set_to_none=True)
        batch,identity=load(aux_row);identity['kernel_task']='joint'
        auxiliary=LigandSourceGeometry(cfg['ligand_geometry'],71,torch.device('cuda'))
        # Counter fixture only: demonstrate one event, not two independent chunk
        # events. These are not additional training exposures or model calls.
        visits=[auxiliary.due(identity,identity['aux_group_history']) for _ in range(8)]
        assert visits==[False]*7+[True]
        defaults=_global_rng_state();main_generator=generator.get_state().clone()
        private_before=auxiliary.generator.get_state().clone()
        before=model.forward_calls;torch.cuda.reset_peak_memory_stats();start=time.monotonic()
        metrics=backward_hq_auxiliary(model,batch,12,12,auxiliary,bf16=cfg['bf16'])
        grad=gradients(model);torch.cuda.synchronize()
        assert metrics['ligand_geometry_applied'] and metrics['ligand_geometry_events']==1
        assert len(metrics['ligand_geometry_chunks'])==2 and metrics['geometry_band_frames']==24
        assert all(c['weight']==.5 and c['metrics']['geometry_band_frames']==24 for c in metrics['ligand_geometry_chunks'])
        assert metrics['ligand_geometry_weight']==cfg['ligand_geometry']['weight']
        assert math.isfinite(metrics['ligand_geometry_weighted_loss'])
        assert equal(defaults,_global_rng_state()) and torch.equal(main_generator,generator.get_state())
        assert not torch.equal(private_before,auxiliary.generator.get_state())
        result['checks']['ligand_auxiliary_event']=dict(seconds=time.monotonic()-start,gradient=grad,metrics=metrics,
            native_NFE_per_chunk=35,chunks=2,tail_intervals=4,
            actual_model_forward_calls_including_backward_recompute=model.forward_calls-before,
            default_and_main_RNG_unchanged=True,private_source_advanced=True,
            peak_cuda_bytes=torch.cuda.max_memory_allocated(),reserved_cuda_bytes=torch.cuda.max_memory_reserved())
        assert all(torch.equal(v.detach().cpu(),base[k]) for k,v in model.state_dict().items() if not k.startswith('task_adapters.'))
        result.update(status='completed',unchanged_parent_base=True,optimizer_updates=0)
    except Exception as error:
        result.update(status='failed',error=repr(error));raise
    finally:
        result['seconds']=time.monotonic()-began;save()
        print(json.dumps(dict(status=result['status'],seconds=result['seconds'],output=str(output),error=result.get('error'))),flush=True)


if __name__=='__main__':main()
