from copy import deepcopy
import json

import numpy as np
import pytest
import torch

from tools.qualify_dynamic_input import finite_state, state_digest
from tools.queue_dynamic_input_qualification import resumed_pair, completed_lane
from molvid.runtime import sha256_file


def state():
    return {'model':{'weight':torch.arange(4,dtype=torch.float32).reshape(2,2)},
        'optimizer':{'state':{0:{'step':torch.tensor(2.),'exp_avg':torch.zeros(4)}}},
        'rng':{'numpy':('MT19937',np.arange(10,dtype=np.uint32),2,0,0.),
               'torch':torch.arange(4,dtype=torch.uint8)},'cursor':2}


def test_canonical_state_digest_roundtrip_and_dict_order(tmp_path):
    value=state();path=tmp_path/'state.pt';torch.save(value,path)
    assert state_digest(value)==state_digest(torch.load(path,weights_only=False))
    assert state_digest(value)==state_digest(dict(reversed(list(value.items()))))
    assert state_digest(torch.tensor(2.))!=state_digest(torch.tensor([2.]))
    assert state_digest(torch.tensor([2],dtype=torch.int32))!=state_digest(torch.tensor([2],dtype=torch.int64))


@pytest.mark.parametrize('field',['model','optimizer','rng','cursor'])
def test_state_digest_covers_each_resume_boundary(field):
    value=state();other=deepcopy(value)
    if field=='model':other[field]['weight'][0,0]+=1
    elif field=='optimizer':other[field]['state'][0]['exp_avg'][0]+=1
    elif field=='rng':other[field]['numpy'][1][0]+=1
    else:other[field]+=1
    assert state_digest(value)!=state_digest(other)


@pytest.mark.parametrize('value',[torch.tensor(float('nan')),{'exp_avg':torch.tensor(float('inf'))},[float('nan')]])
def test_nonfinite_optimizer_or_model_fails(value):
    with pytest.raises(FloatingPointError):finite_state(value)


def pair():
    left=dict(status='completed',actual_updates=2,cursor_start=0,cursor_end=2,optimizer_steps=[2],atoms=17002,
        history_frames=4,query_frames=12,model_mode='trainable_query',finite_model_optimizer_and_gradients=True,
        teacher_before_sha256='t',teacher_after_sha256='t',full_state_sha256='a',model_sha256='b',
        optimizer_sha256='c',rng_sha256='d',sample_id='atlas_6sup_A_R1_dt_100ps_w000000')
    right=dict(left,actual_updates=1,cursor_start=1)
    return left,right


def test_actual_pair_accepts_only_expected_complete_cold_resume():
    resumed_pair(*pair(),'trainable_query',4)


@pytest.mark.parametrize('key,value',[('atoms',4975),('query_frames',2),('actual_updates',0),
    ('rng_sha256','changed'),('optimizer_steps',[1]),('teacher_after_sha256','changed'),
    ('finite_model_optimizer_and_gradients',False)])
def test_false_or_narrower_success_rejected(key,value):
    left,right=pair();right[key]=value
    with pytest.raises(ValueError):resumed_pair(left,right,'trainable_query',4)


@pytest.mark.parametrize('status',['waiting','running','failed'])
def test_gpu_phase_gap_is_not_full_lane_completion(tmp_path,status):
    (tmp_path/'status.json').write_text(json.dumps(dict(status=status,updates=8192,forecasts=192)))
    with pytest.raises(ValueError,match='full long8_72 lane'):completed_lane(tmp_path)


def test_missing_forecasts_rejected_even_after_full_training(tmp_path):
    (tmp_path/'status.json').write_text(json.dumps(dict(status='completed',updates=8192,forecasts=192)))
    (tmp_path/'expanded48').mkdir()
    (tmp_path/'expanded48/summary.json').write_text(json.dumps(dict(status='completed',rows=191,systems=48,failures=[])))
    with pytest.raises(ValueError,match='full long8_72 forecasts'):completed_lane(tmp_path)


@pytest.mark.parametrize('change',[None,'evaluated_checkpoint','training_cursor','checkpoint_bytes','chemistry'])
def test_full_lane_checks_real_checkpoint_identity_and_all_phases(tmp_path,change):
    for name in ('formal','expanded48','stereochemistry'):(tmp_path/name).mkdir()
    checkpoint=tmp_path/'formal/coordinate_step_8192.pt';checkpoint.write_bytes(b'synthetic gate test')
    digest=sha256_file(checkpoint)
    files={
        'status.json':dict(status='completed',updates=8192,forecasts=192),
        'expanded48/summary.json':dict(status='completed',rows=192,systems=48,failures=[]),
        'stereochemistry/summary.json':dict(status='completed'),
        'final_audit.json':dict(status='completed',cursor=8192,checkpoint_sha256=digest),
        'expanded48/run_identity.json':dict(input_sha256={str(checkpoint):digest}),
        'formal/summary.json':dict(status='completed',cursor_start=17,cursor_end=8192,updates_this_execution=8175)}
    if change=='evaluated_checkpoint':files['expanded48/run_identity.json']['input_sha256'][str(checkpoint)]='wrong'
    if change=='training_cursor':files['formal/summary.json']['cursor_end']=8191
    if change=='checkpoint_bytes':checkpoint.write_bytes(b'changed')
    if change=='chemistry':files['stereochemistry/summary.json']['status']='running'
    for name,value in files.items():(tmp_path/name).write_text(json.dumps(value))
    if change is None:completed_lane(tmp_path)
    else:
        with pytest.raises(ValueError):completed_lane(tmp_path)
