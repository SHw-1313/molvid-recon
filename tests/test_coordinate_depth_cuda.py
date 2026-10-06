"""Actual GPU parity, active expanded-block learning and exact resume."""
from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from molvid.codec.frame import slice_clip_frames
from molvid.codec.motion_context import MotionContextEncoder
from molvid.data.batch import collate_clip_records
from molvid.dit.frame import FrameDiT
from molvid.flow.coordinate import CoordinateFlowModel
from molvid.latent.types import QuerySpec
from molvid.training.coordinate import (CoordinateTrainConfig,coordinate_update,
    make_coordinate_optimizer,capture_coordinate_rng,restore_coordinate_rng)
from tools.run_coordinate_campaign import assert_same
from test_static_decoder import _model,_static_record

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='actual CUDA required')


def fixture(depth=4):
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(20261006)
    torch.cuda.set_per_process_memory_fraction(.015)
    parent=_model(torch.device('cuda'))
    parent.dit=FrameDiT(adapter=parent.dit.adapter,scalar_width=8,vector_width=4,heads=1,
        depth=4,geometry_enabled=True,motion_enabled=True,geometry_block_indices=(2,4)).cuda()
    parent.motion_encoder=MotionContextEncoder(4,8,4).cuda()
    model=CoordinateFlowModel(parent)
    with torch.no_grad():
        model.readout.out.weight.normal_(std=.1)
        model.dit.adapter.scalar_out.weight.normal_(std=.05)
        model.dit.adapter.vector_out.weight.normal_(std=.05)
        for block in model.dit.blocks:
            for name in ('history_adaln','spatial_adaln','temporal_adaln','ffn_adaln'):
                getattr(block,name).modulation.bias.normal_(std=.1)
            block.motion_context.scalar_residual_gate.fill_(.1)
            block.motion_context.vector_residual_gate.fill_(.1)
            if block.local_geometry is not None:
                block.local_geometry.scalar_gate.fill_(.1)
                block.local_geometry.vector_residual_gate.fill_(.1)
    if depth==8: model.expand_depth()
    batch=collate_clip_records([_static_record('a'),_static_record('b',10.)])
    model.prepare_batch(batch); batch=batch.to('cuda')
    x=batch.x.expand(4,-1,-1).clone()+.03*torch.randn(4,batch.atom_count,3,device='cuda')
    time=torch.arange(4,device='cuda').float()[None].expand(2,-1)*100
    observed=replace(batch,x=x,bpos=x.clone(),time_ps=time,frame_mask=torch.ones(2,4,device='cuda',dtype=torch.bool),delta_time_ps=time[:,1:]-time[:,:-1])
    query=QuerySpec(torch.tensor([[400.,500.],[400.,500.]],device='cuda'),torch.tensor([[True,True],[True,False]],device='cuda'))
    return model,observed,query


def test_real_modules_dynamic_static_parity_rng_and_inference_roundtrip():
    model,observed,query=fixture()
    with torch.no_grad():
        condition=model.prepare_condition(observed)
        source=model.source(condition=condition,query=query,generator=torch.Generator(device='cuda').manual_seed(17))
        progress=torch.tensor([.3,.6],device='cuda')
        before=model(source,condition=condition,query=query,flow_time=progress)
        static=model.prepare_condition(slice_clip_frames(observed,0,1),task='static')
        static_state=static.observed.x+.1
        before_static=model.forward_static(static_state,condition=static,flow_time=progress)
        rng=[torch.get_rng_state().clone(),torch.cuda.get_rng_state().clone()]
        report=model.expand_depth()
        assert torch.equal(rng[0],torch.get_rng_state()) and torch.equal(rng[1],torch.cuda.get_rng_state())
        after=model(source,condition=condition,query=query,flow_time=progress)
        after_static=model.forward_static(static_state,condition=static,flow_time=progress)
        assert torch.equal(before,after) and torch.equal(before_static,after_static)
        assert report['geometry_blocks_one_based']==[4,8]
        # Exercise the actual inference constructor, not just training expansion.
        parent=_model(torch.device('cuda'))
        parent.dit=FrameDiT(adapter=parent.dit.adapter,scalar_width=8,vector_width=4,heads=1,
            depth=4,geometry_enabled=True,motion_enabled=True,geometry_block_indices=(2,4)).cuda()
        parent.motion_encoder=MotionContextEncoder(4,8,4).cuda()
        restored=CoordinateFlowModel(parent,dit_depth=8)
        restored.load_state_dict(model.state_dict(),strict=True)
        assert restored.contract()==model.contract()
        restored.prepare_batch(observed.to('cpu'))
        other=restored.prepare_condition(observed)
        assert torch.equal(after,restored(source,condition=other,query=query,flow_time=progress))


def test_active_new_blocks_optimizer_and_exact_resume():
    model,observed,_=fixture(8)
    future=observed.x[-1:].expand(2,-1,-1)+.05
    time=torch.arange(6,device='cuda').float()[None].expand(2,-1)*100
    batch=replace(observed,x=torch.cat((observed.x,future)),bpos=torch.cat((observed.bpos,future)),
        time_ps=time,frame_mask=torch.ones(2,6,device='cuda',dtype=torch.bool),delta_time_ps=time[:,1:]-time[:,:-1]).to('cpu')
    config=CoordinateTrainConfig(); opt=make_coordinate_optimizer(model,config)
    gen=torch.Generator(device='cuda').manual_seed(818)
    coordinate_update(model,opt,batch,history_frames=4,generator=gen,step=0,config=config)
    saved=deepcopy(model.state_dict()),deepcopy(opt.state_dict()),capture_coordinate_rng(gen)
    expected=coordinate_update(model,opt,batch,history_frames=4,generator=gen,step=1,config=config)
    expected_rng=capture_coordinate_rng(gen)
    for block in model.dit.blocks[::2]:
        assert block.spatial_adaln.modulation.weight.grad is not None and block.spatial_adaln.modulation.weight.grad.norm()>0
        assert any(p.grad is not None and p.grad.norm()>0 for p in block.spatial.parameters())
    resumed,_,_=fixture(8); resumed.load_state_dict(saved[0],strict=True)
    opt2=make_coordinate_optimizer(resumed,config); opt2.load_state_dict(saved[1]); restore_coordinate_rng(saved[2],gen)
    actual=coordinate_update(resumed,opt2,batch,history_frames=4,generator=gen,step=1,config=config)
    assert_same(expected,actual); assert_same(model.state_dict(),resumed.state_dict())
    assert_same(opt.state_dict(),opt2.state_dict()); assert_same(expected_rng,capture_coordinate_rng(gen))
    assert all(p.grad is None for p in resumed.target_teacher.parameters())
