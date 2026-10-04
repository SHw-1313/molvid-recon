from dataclasses import replace
import numpy as np
import pytest
import torch
from test_static_decoder import _model,_static_record
from molvid.data.batch import collate_clip_records
from molvid.training.predictive_data import encode_observation,encode_targets,fixed_projection,system_partition,group_identity
from molvid.evaluation.predictive import system_report,gaussian_metrics,variance_temperature


def fixture():
    r=_static_record('uk')
    offsets=np.arange(16,dtype=np.float32)[:,None,None]*np.array([[[.01,.02,-.01]]],dtype=np.float32)
    r.update(task='trajectory',x=r['x'].repeat(16,0)+offsets,bpos=r['bpos'].repeat(16,0)+offsets,
        frame_mask=np.ones(16,dtype=bool),time_ps=np.arange(16,dtype=np.float32)*100,
        delta_time_ps=np.full(15,100,dtype=np.float32),time_bucket_id='dt_100ps')
    return collate_clip_records([r])


def test_future_mutation_does_not_change_observed_features_or_origin():
    assert torch.cuda.is_available(),'CUDA required'
    device=torch.device('cuda');model=_model(device).eval().requires_grad_(False)
    cpu=fixture();projection=fixed_projection(4,2).to(device)
    x=cpu.x.clone();bpos=cpu.bpos.clone();x[4:]+=50;bpos[4:]-=70
    changed=replace(cpu,x=x,bpos=bpos)
    a=encode_observation(model,cpu,projection,device=device,atoms=4)
    b=encode_observation(model,changed,projection,device=device,atoms=4)
    assert torch.equal(a['x'],b['x']) and torch.equal(a['origin'],b['origin'])
    y=encode_targets(model,cpu,a,projection,device=device)
    altered=encode_targets(model,changed,b,projection,device=device)
    assert not torch.equal(y,altered)
    assert all(not p.requires_grad and p.grad is None for p in model.parameters())


def test_split_and_system_equal_reports():
    fit,cal=system_partition([f's{i}' for i in range(30)],6)
    assert len(fit)==24 and len(cal)==6 and not fit&cal
    assert group_identity('atlas_2ejn_B_R1_a000000_dt_100ps_H4')==('atlas_2ejn_B','atlas_2ejn_B_R1')
    value=torch.tensor([1.,3.,3.],device='cuda')[:,None,None].expand(3,4,3)
    report=system_report(value,[{'system':'a'},{'system':'b'},{'system':'b'}])
    assert report['100']['mean']==pytest.approx(2.)


def test_variance_calibration_has_horizon_axes():
    y=torch.ones((2,4,3,5),device='cuda')*2
    mu=torch.zeros_like(y);v=torch.ones_like(y)
    temp=variance_temperature(mu,v,y)
    assert torch.equal(temp,torch.full((3,),4.,device='cuda'))
    metrics=gaussian_metrics(mu,v*temp[None,None,:,None],y)
    assert metrics['mse'].shape==(2,4,3)
    assert bool((metrics['coverage90']==1).all())
