"""Bounded continuation and ligand-loss checks; run CUDA only via the task runner."""
import argparse
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import random
import subprocess
import sys
import time
from types import ModuleType

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from molvid.codec.frame import slice_clip_frames
from molvid.data.batch import ClipBatch,collate_clip_records
from molvid.data.bridge_mixture import BridgeMixture
from molvid.flow.coordinate_state import CoordinateStateDenoiser
from molvid.flow.local_geometry_loss import ligand_geometry_band_loss
from molvid.training.coordinate_state import (LigandSourceGeometry,validate_continuation,
    continuation_learning_rate,_global_rng_state,_set_global_rng_state)

PARENT=Path('/workspace/MOLVID-FLOW-STATIC-TO-MD-20261023/runs/flow_static_to_md_20261023/new_static/coordinate_final.pt')
TASK=ROOT/'tasks/flow-ligand-4096-20261026'
RUN=ROOT/'runs/flow_ligand_4096_20261026'


def equal(a,b):
    if isinstance(a,torch.Tensor): return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,np.ndarray): return isinstance(b,np.ndarray) and np.array_equal(a,b)
    if isinstance(a,dict): return isinstance(b,dict) and a.keys()==b.keys() and all(equal(v,b[k]) for k,v in a.items())
    if isinstance(a,(list,tuple)): return type(a)==type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


def fixed_record(settings,system='3M35'):
    mixture=BridgeMixture(settings['data'],seed=71)
    try:
        buckets=mixture.groups['complex'][system]
        bucket=min(buckets,key=lambda name:float(name.split('_')[1][:-2]))
        store,index=buckets[bucket][0]
        record=store[index]
        assert record['split']=='train'
        batch=collate_clip_records([record])
        return batch,dict(system=system,sample_id=batch.sample_id[0],split='train',
            atoms=batch.atom_count,time_bucket_id=bucket,time_ps=batch.time_ps[0].tolist())
    finally: mixture.close()


def difference(a,b):
    delta=(a.float()-b.float()).abs()
    return dict(max_A=float(delta.max()),rms_A=float(delta.square().mean().sqrt()))


def control_check(args,result):
    settings=json.loads((TASK/'configs/control.json').read_text())
    saved=torch.load(args.parent,map_location='cpu',weights_only=False)
    latest=torch.load(args.parent.with_name('checkpoint_latest.pt'),map_location='cpu',weights_only=False)
    keys=('model','optimizer','settings','seed','step','mixture','exposure',
          'generator_rng','torch_rng','cuda_rng','python_rng','numpy_rng')
    result['parent_latest_final']={key:equal(saved[key],latest[key]) for key in keys}
    assert saved['step']==latest['step']==512 and all(result['parent_latest_final'].values())
    result['continuation']=validate_continuation(saved,settings,71)
    assert continuation_learning_rate(settings,512)==1e-5
    assert abs(continuation_learning_rate(settings,639)-1e-4)<1e-12
    assert abs(continuation_learning_rate(settings,4095)-1e-5)<1e-12
    result['schedule']={str(step+1):continuation_learning_rate(settings,step) for step in (512,639,640,1023,2047,4095)}
    wrong=deepcopy(settings);wrong['data']['query_frames']=1
    try: validate_continuation(saved,wrong,71)
    except ValueError: pass
    else: raise AssertionError('Continuation accepted a changed data recipe')
    source=subprocess.check_output(['git','show','0cc25c0:molvid/flow/coordinate_state.py'],cwd=ROOT,text=True)
    module=ModuleType('molvid.flow._ligand_parent_check');module.__package__='molvid.flow'
    sys.modules[module.__name__]=module
    exec(compile(source,'0cc25c0:coordinate_state','exec'),module.__dict__)
    torch.manual_seed(71)
    rng=torch.get_rng_state().clone();cuda_rng=torch.cuda.get_rng_state().clone()
    old=module.CoordinateStateDenoiser(saved['settings'])
    after=torch.get_rng_state().clone();torch.set_rng_state(rng)
    new=CoordinateStateDenoiser(settings)
    assert torch.equal(after,torch.get_rng_state()) and torch.equal(cuda_rng,torch.cuda.get_rng_state())
    assert equal(old.state_dict(),new.state_dict())
    old.load_state_dict(saved['model'],strict=True);new.load_state_dict(saved['model'],strict=True)
    optimizer=torch.optim.AdamW(new.parameters(),lr=settings['learning_rate'],weight_decay=settings['weight_decay'])
    optimizer.load_state_dict(saved['optimizer'])
    assert equal(optimizer.state_dict(),saved['optimizer'])
    assert {int(x['step']) for x in optimizer.state.values()}=={512}
    mixture=BridgeMixture(settings['data'],71)
    try:
        mixture.load_state_dict(deepcopy(saved['mixture']));assert equal(mixture.state_dict(),saved['mixture'])
        batch,task,_,_,identity=mixture.next()
        first=(batch.sample_id,identity)
        mixture.load_state_dict(deepcopy(saved['mixture']))
        repeated=mixture.next()
        assert equal(first,(repeated[0].sample_id,repeated[4]))
        result['next_visit']=dict(sample=first[0],identity=identity,history=4,queries=list(range(4,batch.frames)))
    finally: mixture.close()
    generator=torch.Generator(device='cuda');generator.set_state(saved['generator_rng'])
    torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng'])
    random.setstate(saved['python_rng']);np.random.set_state(saved['numpy_rng'])
    expected_globals=dict(python=saved['python_rng'],numpy=saved['numpy_rng'],
                          torch_cpu=saved['torch_rng'],cuda=saved['cuda_rng'])
    assert equal(_global_rng_state(),expected_globals)
    assert torch.equal(generator.get_state(),saved['generator_rng'])
    result['full_state_restoration_exact']=True
    host,result['window']=fixed_record(settings)
    clip=host.to('cuda');observed=slice_clip_frames(clip,0,4);times=clip.time_ps[:,4:]
    old.cuda().eval();new.cuda().eval()
    with torch.no_grad(),torch.autocast('cuda',enabled=False):
        condition=old.prepare_condition(observed)
        progress=torch.rand(1,device='cuda',generator=generator)
        noise=torch.randn(clip.x[4:].shape,device='cuda',generator=generator)
        source=condition.reference_A[None]+old.rho_A*noise
        current=(1-progress[:,None,None])*source+progress[:,None,None]*clip.x[4:]
        expected=old(current,condition=condition,query_time_ps=times,progress=progress)
        repeat=old(current,condition=condition,query_time_ps=times,progress=progress)
        actual=new(current,condition=condition,query_time_ps=times,progress=progress)
    result['field']=dict(same_tensor=True,old_old=difference(repeat,expected),new_old=difference(actual,expected),
                         atol_A=2e-4,rtol=2e-5,precision='fp32')
    torch.testing.assert_close(actual,expected,atol=2e-4,rtol=2e-5)
    assert bool(torch.isfinite(actual).all())
    result['optimizer_steps_performed']=0


def fixture():
    xyz=torch.tensor([[0.,0.,0.],[1.,0.,0.],[1.,1.,0.],
        [0.,0.,2.],[1.,0.,2.],[1.,1.,2.],[3.,0.,2.],[4.,0.,2.],
        [0.,0.,0.],[1.,0.,0.],[1.,1.,0.],[2.,1.,0.]],device='cuda')
    x=xyz[None].repeat(4,1,1);integer=lambda a:torch.tensor(a,dtype=torch.long,device='cuda')
    mask=torch.ones(12,dtype=torch.bool,device='cuda')
    times=torch.arange(4,device='cuda').float()[None].repeat(2,1)*80.
    return ClipBatch(x=x,bpos=x.clone(),atype=integer([5]*12),btype=integer([1]*12),
        block_id=integer([1]*12),component_id=integer([0,0,0,1,1,1,2,2,0,0,0,0]),
        atom_source_index=integer(range(12)),abid=integer([0]*8+[1]*4),atom_ptr=integer([0,8,12]),
        bond_index=integer([[0,1,3,4,6,5,2,8,9,10,4],[1,2,4,5,7,6,3,9,10,11,3]]),
        edge_mask=integer([0,0,0,1,1,1,1,1,0,0,0,0]),loss_mask=mask,align_mask=mask,
        frame_mask=torch.ones(2,4,dtype=torch.bool,device='cuda'),time_ps=times,
        delta_time_ps=times[:,1:]-times[:,:-1],time_bucket_id=('80ps','80ps'),
        topology_id=('fixture0','fixture1'),task=integer([0,0]),sample_id=('fixture0','fixture1'),
        atom_counts=(8,4),host_task_ids=(0,0))


def ligand_check(args,result):
    settings=json.loads((TASK/'configs/ligand_uncalibrated.json').read_text())
    config=settings['ligand_geometry'];band=config['band'];batch=fixture()
    target=batch.x[:2].clone().requires_grad_()
    endpoint=target.detach().clone();endpoint[:,3,1]+=.8;endpoint[:,7,0]+=.5
    endpoint.requires_grad_()
    loss,metrics=ligand_geometry_band_loss(endpoint,target,batch,band)
    loss.backward();assert target.grad is None
    protein=batch.component_id==0
    assert float(endpoint.grad[:,~protein].norm())>0 and not bool(endpoint.grad[:,protein].any())
    assert metrics['local_geometry_bonds']==3 and metrics['local_geometry_angles']==1
    assert metrics['ligand_components']==2 and metrics['ligand_systems']==1
    changed=endpoint.detach().clone();changed[:,protein]=float('nan')
    same,_=ligand_geometry_band_loss(changed,target,batch,band)
    torch.testing.assert_close(same,loss.detach(),atol=0,rtol=0)
    # A protein-only second system and duplicated/cross-component edges do not
    # dilute the ligand term. Individual component losses average equally.
    individual=[]
    for component in (1,2):
        single=replace(batch,loss_mask=batch.loss_mask & (batch.component_id==component))
        value,_=ligand_geometry_band_loss(endpoint,target,single,band);individual.append(value)
    torch.testing.assert_close(loss,torch.stack(individual).mean(),atol=1e-6,rtol=1e-6)
    result['ligand_only_loss']=dict(loss=float(loss.detach()),metrics=metrics,
        target_detached=True,protein_gradient_zero=True,independent_components=True)
    class ToySource(torch.nn.Module):
        def __init__(self):
            super().__init__();self.scale=torch.nn.Parameter(torch.tensor(.7,device='cuda'))
        def sample_truncated(self,observed,times,*,tail_intervals,generator):
            # Deliberately exercise every default stream inside the helper.
            random.random();np.random.random();torch.rand(1);torch.rand(1,device='cuda')
            shape=(times.shape[1],*observed.x[-1].shape)
            return observed.x[-1:]+self.scale*torch.randn(shape,device='cuda',generator=generator)
    before=_global_rng_state();helper=LigandSourceGeometry(config,71,'cuda')
    assert equal(before,_global_rng_state())
    assert not helper.due(dict(domain='protein'),4)
    assert [helper.due(dict(domain='complex'),4) for _ in range(8)]==[False]*7+[True]
    model=ToySource();model.zero_grad(set_to_none=True)
    before=_global_rng_state()
    first=helper.backward(model,batch,target.detach(),batch.time_ps[:,:2],bf16=False)
    assert equal(before,_global_rng_state())
    assert model.scale.grad is not None and torch.isfinite(model.scale.grad) and model.scale.grad.abs()>0
    saved=deepcopy(helper.state_dict())
    model.zero_grad(set_to_none=True)
    expected=helper.backward(model,batch,target.detach(),batch.time_ps[:,:2],bf16=False)
    expected_grad=model.scale.grad.clone();expected_state=deepcopy(helper.state_dict())
    restored=LigandSourceGeometry(config,71,'cuda');restored.load_state_dict(saved)
    model.zero_grad(set_to_none=True)
    actual=restored.backward(model,batch,target.detach(),batch.time_ps[:,:2],bf16=False)
    torch.testing.assert_close(model.scale.grad,expected_grad,atol=1e-6,rtol=1e-6)
    assert equal(expected_state,restored.state_dict()) and equal(before,_global_rng_state())
    assert actual['ligand_geometry_weighted_loss']==expected['ligand_geometry_weighted_loss']
    result['isolated_aux_resume']=dict(passed=True,gradient=float(model.scale.grad),first=first,
        all_default_streams_unchanged=True,saved_aux_state_exact=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=('control','ligand'),required=True)
    parser.add_argument('--parent',type=Path,default=PARENT);parser.add_argument('--output',type=Path)
    parser.add_argument('--cap-GiB',type=float,default=32.)
    args=parser.parse_args();output=args.output or RUN/'checks'/f'{args.mode}.json'
    if output.exists(): raise FileExistsError(output)
    result=dict(mode=args.mode,passed=False,parent=str(args.parent));began=time.monotonic()
    try:
        if not torch.cuda.is_available(): raise RuntimeError('Budgeted CUDA execution required')
        torch.set_num_threads(2)
        torch.cuda.set_per_process_memory_fraction(args.cap_GiB*1024**3/torch.cuda.get_device_properties(0).total_memory,0)
        (control_check if args.mode=='control' else ligand_check)(args,result)
        result['passed']=True
    except Exception as error:
        result['error']=repr(error);raise
    finally:
        if torch.cuda.is_initialized():
            torch.cuda.synchronize();result['peak_cuda_bytes']=torch.cuda.max_memory_allocated()
        result['wall_seconds']=time.monotonic()-began
        output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        print(json.dumps(dict(output=str(output),passed=result['passed'],wall_seconds=result['wall_seconds'])),flush=True)


if __name__=='__main__': main()
