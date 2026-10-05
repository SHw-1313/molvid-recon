import copy
from dataclasses import replace

import numpy as np
import pytest
import torch

from molvid.data.batch import collate_clip_records
from molvid.training.coordinate import (CoordinateTrainConfig, coordinate_update,
    make_coordinate_optimizer, capture_coordinate_rng, restore_coordinate_rng)


@pytest.mark.parametrize('settings', [dict(handedness_weight=-1),dict(handedness_weight=float('nan')),
    dict(handedness_every=0),dict(handedness_steps=True),dict(handedness_margin=0),
    dict(handedness_minimum_volume_A3=0)])
def test_invalid_auxiliary_config(settings):
    with pytest.raises(ValueError):
        CoordinateTrainConfig(**settings)


def fixture():
    from test_coordinate_flow import fixture as model_fixture
    from test_static_decoder import _static_record
    model,_,_ = model_fixture()
    model.train()
    records=[]
    for i in range(2):
        record=_static_record(f'chiral{i}')
        tetra=np.array([[0,0,0],[1.4,0,0],[0,1.4,0],[0,0,1.4]],dtype=np.float32)+i*10
        x=np.stack([tetra+.01*t for t in range(6)])
        record.update(task='trajectory',time_bucket_id='dt_100ps',
            time_ps=np.arange(6,dtype=np.float32)*100,delta_time_ps=np.ones(5,dtype=np.float32)*100,
            frame_mask=np.ones(6,dtype=bool),x=x,bpos=np.repeat(x.mean(1,keepdims=True),4,axis=1),
            block_id=np.zeros(4,dtype=np.int64),btype=np.full(4,118,dtype=np.int64),
            atom_identity=[f'chiral{i}:1:{name}:X' for name in ('CA','N','C','CB')],
            bond_index=np.array([[0,0,0],[1,2,3]],dtype=np.int64))
        records.append(record)
    return model,collate_clip_records(records),[name for record in records for name in record['atom_identity']]


def run_step(model,batch,names,config,*,seed=147,step=0):
    optimizer=make_coordinate_optimizer(model,config)
    generator=torch.Generator(device='cuda').manual_seed(seed)
    result=coordinate_update(model,optimizer,batch,history_frames=4,generator=generator,
        step=step,config=config,atom_identity=names)
    return result,generator,optimizer


cuda=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


@cuda
def test_auxiliary_changes_update_but_not_main_rng_or_rf_loss():
    model,batch,names=fixture(); other=copy.deepcopy(model)
    baseline=CoordinateTrainConfig()
    auxiliary=replace(baseline,handedness_weight=.1,handedness_every=1,handedness_steps=2)
    rng=torch.get_rng_state(); cuda_rng=torch.cuda.get_rng_state()
    a,ga,_=run_step(model,batch,names,baseline)
    after_cpu=torch.get_rng_state(); after_cuda=torch.cuda.get_rng_state()
    torch.set_rng_state(rng); torch.cuda.set_rng_state(cuda_rng)
    b,gb,_=run_step(other,batch,names,auxiliary)
    assert b['handedness_applied'] and b['handedness_centers']==2 and b['handedness_loss']>0
    assert a['rf_loss_A2']==b['rf_loss_A2'] and a['flow_time_mean']==b['flow_time_mean']
    assert torch.equal(ga.get_state(),gb.get_state())
    assert torch.equal(torch.get_rng_state(),after_cpu) and torch.equal(torch.cuda.get_rng_state(),after_cuda)
    assert any(not torch.equal(v,other.state_dict()[n]) for n,v in model.state_dict().items())
    assert all(p.grad is None for p in other.target_teacher.parameters())


@cuda
def test_auxiliary_does_not_read_future_coordinates():
    model,batch,names=fixture(); other=copy.deepcopy(model)
    changed=copy.deepcopy(batch); changed.x[4:]+=25; changed.bpos[4:]-=100
    config=CoordinateTrainConfig(handedness_weight=.1,handedness_every=1,handedness_steps=2)
    a,_,_=run_step(model,batch,names,config)
    b,_,_=run_step(other,changed,names,config)
    assert a['handedness_loss']==b['handedness_loss']
    assert a['rf_loss_A2']!=b['rf_loss_A2']


@cuda
def test_inactive_cadence_matches_zero_weight_exactly():
    model,batch,names=fixture(); other=copy.deepcopy(model)
    a,ga,oa=run_step(model,batch,names,CoordinateTrainConfig())
    b,gb,ob=run_step(other,batch,None,CoordinateTrainConfig(handedness_weight=.1,handedness_every=8))
    assert not b['handedness_applied'] and a['rf_loss_A2']==b['rf_loss_A2']
    for name,value in model.state_dict().items():
        assert torch.equal(value,other.state_dict()[name]),name
    from tools.run_coordinate_campaign import assert_same
    assert_same(oa.state_dict(),ob.state_dict())
    assert torch.equal(ga.get_state(),gb.get_state())


@cuda
def test_active_auxiliary_exact_optimizer_rng_resume():
    model,batch,names=fixture()
    config=CoordinateTrainConfig(handedness_weight=.1,handedness_every=1,handedness_steps=2)
    _,gen,opt=run_step(model,batch,names,config)
    weights=copy.deepcopy(model.state_dict()); optimizer=copy.deepcopy(opt.state_dict())
    rng=capture_coordinate_rng(gen)
    expected=coordinate_update(model,opt,batch,history_frames=4,generator=gen,step=1,
        config=config,atom_identity=names)
    fresh,_,_=fixture(); fresh.load_state_dict(weights)
    opt2=make_coordinate_optimizer(fresh,config); opt2.load_state_dict(optimizer)
    gen2=torch.Generator(device='cuda'); restore_coordinate_rng(rng,gen2)
    actual=coordinate_update(fresh,opt2,batch,history_frames=4,generator=gen2,step=1,
        config=config,atom_identity=names)
    assert actual==expected
    from tools.run_coordinate_campaign import assert_same
    assert_same(model.state_dict(),fresh.state_dict()); assert_same(opt.state_dict(),opt2.state_dict())
    assert_same(capture_coordinate_rng(gen),capture_coordinate_rng(gen2))
