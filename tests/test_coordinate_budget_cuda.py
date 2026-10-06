"""Actual CUDA long-policy numerical/resume checks; toy state, not efficacy."""
from copy import deepcopy
from dataclasses import replace
import pytest
import torch
from test_coordinate_depth_cuda import fixture
from molvid.training.coordinate import (CoordinateTrainConfig,coordinate_update,
    make_coordinate_optimizer,capture_coordinate_rng,restore_coordinate_rng)
from tools.run_coordinate_campaign import assert_same

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


@pytest.mark.parametrize('depth',[4,8])
@pytest.mark.parametrize('query_trainable',[False,True])
def test_long_budget_lr_and_exact_resume(depth, query_trainable):
    model,observed,_=fixture(depth)
    if query_trainable:
        model.enable_query_geometry()
        model.set_query_geometry_recompute(True)
    future=observed.x[-1:].expand(2,-1,-1)+.05
    time=torch.arange(6,device='cuda').float()[None].expand(2,-1)*100
    batch=replace(observed,x=torch.cat((observed.x,future)),bpos=torch.cat((observed.bpos,future)),
        time_ps=time,frame_mask=torch.ones(2,6,device='cuda',dtype=torch.bool),delta_time_ps=time[:,1:]-time[:,:-1]).to('cpu')
    config=CoordinateTrainConfig(total_updates=8192); opt=make_coordinate_optimizer(model,config)
    gen=torch.Generator(device='cuda').manual_seed(8192)
    row=coordinate_update(model,opt,batch,history_frames=4,generator=gen,step=0,config=config)
    assert row['learning_rates']['dit']==pytest.approx(config.dit_lr/410)
    saved=deepcopy(model.state_dict()),deepcopy(opt.state_dict()),capture_coordinate_rng(gen)
    expected=coordinate_update(model,opt,batch,history_frames=4,generator=gen,step=1,config=config)
    expected_rng=capture_coordinate_rng(gen)
    restored,_,_=fixture(depth)
    if query_trainable:
        restored.enable_query_geometry()
        restored.set_query_geometry_recompute(True)
    restored.load_state_dict(saved[0],strict=True)
    opt2=make_coordinate_optimizer(restored,config); opt2.load_state_dict(saved[1]); restore_coordinate_rng(saved[2],gen)
    actual=coordinate_update(restored,opt2,batch,history_frames=4,generator=gen,step=1,config=config)
    assert_same(expected,actual); assert_same(model.state_dict(),restored.state_dict())
    assert_same(opt.state_dict(),opt2.state_dict()); assert_same(expected_rng,capture_coordinate_rng(gen))
    # Numerical policy endpoint test, explicitly NOT8192 training updates.
    late=coordinate_update(restored,opt2,batch,history_frames=4,generator=gen,step=8191,config=config)
    assert late['learning_rates']['dit']==pytest.approx(config.dit_lr*.1)
    assert all(p.grad is None for p in restored.target_teacher.parameters())

    if query_trainable:
        assert any(p.grad is not None and p.grad.norm() > 0 for p in restored.query_encoder.parameters())
        assert restored.contract()['query_geometry_execution'] == 'per_chunk_nonreentrant_recompute_preserve_rng_v1'
