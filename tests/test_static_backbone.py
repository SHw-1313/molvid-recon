from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest
import torch

from test_static_decoder import _model, _static_record
from molvid.data.batch import collate_clip_records
from molvid.training.batches import prepare_frame_joint_batch
from molvid.training.joint import FrameJointTrainer, JointLossConfig, frame_joint_loss
from molvid.training.static_backbone import (StaticBackboneConfig, prepare_static_backbone_batch,
    static_backbone_loss, static_parameter_role, calibrated_static_weight)


def cuda():
    assert torch.cuda.is_available(), 'CUDA required; no CPU fallback'
    return torch.device('cuda')


def batches(model):
    record = _static_record('d2')
    static = collate_clip_records([record])
    dynamic = dict(record)
    offsets = np.arange(8, dtype=np.float32)[:, None, None] * np.array([[[.01, -.02, .03]]], dtype=np.float32)
    dynamic.update(task='trajectory', x=record['x'].repeat(8, axis=0) + offsets,
        bpos=record['bpos'].repeat(8, axis=0) + offsets, time_ps=np.arange(8,dtype=np.float32)*100,
        delta_time_ps=np.full(7,100,dtype=np.float32), frame_mask=np.ones(8,dtype=bool),time_bucket_id='dt_100ps')
    prepared = prepare_frame_joint_batch(model.target_teacher,collate_clip_records([dynamic]),
        device=cuda(),normalizer=model,history_frames=4)
    return static, prepared


def trainer(model, *, amp=False):
    return FrameJointTrainer(model, loss_config=JointLossConfig(False, 0),
        learning_rates={'history_encoder':1e-4,'dit':2e-4,'decoder':5e-5},
        stage='joint',amp=amp,static_backbone_config=StaticBackboneConfig(.01,rollout_steps=2))


@pytest.mark.parametrize('amp',[False,True])
def test_static_shared_gradients_bypass_time_and_use_actual_source(amp):
    model = _model(cuda())
    cpu, _ = batches(model)
    prepared = prepare_static_backbone_batch(model,cpu,device=cuda(),generator=torch.Generator(device='cuda').manual_seed(71))
    config = StaticBackboneConfig(.01,rollout_steps=2)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3, weight_decay=.1)
    # Primed moments make this test catch zero gradients + weight decay on
    # supposedly bypassed temporal parameters, not just forward reachability.
    for p in model.parameters():
        if p.requires_grad: p.grad=torch.ones_like(p)
    optimizer.step(); optimizer.zero_grad(set_to_none=True)
    before={n:p.detach().clone() for n,p in model.named_parameters()}
    with torch.autocast('cuda',dtype=torch.bfloat16,enabled=amp):
        loss,metrics=static_backbone_loss(model,prepared,config,generator=torch.Generator(device='cuda').manual_seed(17),successful_updates=0)
    loss.backward()
    assert 'static_rollout_geometry' in metrics and torch.isfinite(loss)
    for name,p in model.named_parameters():
        if static_parameter_role(name) != 'shared': assert p.grad is None,name
    for prefix in ('dit.adapter.','dit.blocks.0.spatial.','dit.blocks.0.ffn.','decoder.coordinate_head.'):
        norm=sum(float(p.grad.float().square().sum()) for n,p in model.named_parameters() if n.startswith(prefix) and p.grad is not None)
        assert norm>0,prefix
    optimizer.step()
    for name,p in model.named_parameters():
        if static_parameter_role(name)!='shared': assert torch.equal(p,before[name]),name


def test_static_source_condition_do_not_read_target_or_physical_clock():
    model=_model(cuda());cpu,_=batches(model)
    b=prepare_static_backbone_batch(model,cpu,device=cuda(),generator=torch.Generator(device='cuda').manual_seed(7))
    # A finite arbitrary storage origin must be irrelevant to static RF.
    latent=replace(b.condition.geometry.latent,time_ps=b.condition.geometry.latent.time_ps+7654)
    changed_condition=replace(b.condition,geometry=replace(b.condition.geometry,latent=latent))
    source=b.source_center
    with torch.no_grad():
        first=model.dit.forward_static(source,condition=b.condition,flow_time=torch.tensor([.3],device='cuda'))
        second=model.dit.forward_static(source,condition=changed_condition,flow_time=torch.tensor([.3],device='cuda'))
    assert torch.equal(first.h,second.h) and torch.equal(first.v,second.v)
    altered=replace(b,normalized_target=b.normalized_target.with_features(b.normalized_target.h+20,b.normalized_target.v+20))
    assert altered.condition is b.condition and altered.source_center is b.source_center
    cfg=StaticBackboneConfig(.01,rollout_steps=2)
    x,_=static_backbone_loss(model,b,cfg,generator=torch.Generator(device='cuda').manual_seed(1),successful_updates=1)
    y,_=static_backbone_loss(model,altered,cfg,generator=torch.Generator(device='cuda').manual_seed(1),successful_updates=1)
    assert not torch.equal(x,y)


@pytest.mark.parametrize('scope',['shared','decoder_only'])
def test_joint_update_equals_explicit_accumulation_and_resumes_rng(tmp_path,scope):
    torch.use_deterministic_algorithms(True)
    model=_model(cuda()); reference=deepcopy(model)
    t=trainer(model); ref=trainer(reference)
    t.static_backbone_config=replace(t.static_backbone_config,gradient_scope=scope)
    ref.static_backbone_config=replace(ref.static_backbone_config,gradient_scope=scope)
    cpu,b=batches(model)
    static=prepare_static_backbone_batch(model,cpu,device=cuda(),generator=t.static_corruption_generator)
    rng=torch.Generator(device='cuda').manual_seed(123)
    ref_rng=torch.Generator(device='cuda').manual_seed(123)
    t.train_step(b,generator=rng,static_batch=static)
    draw=ref.flow.sample(b.normalized_target,b.source_center,generator=ref_rng)
    out=reference(draw.interpolated,context=b.observed_context,query=b.query,flow_time=draw.flow_time,
        clean_future=b.target_future,decode_generated=True,decode_clean=True,decode_near=True)
    losses=frame_joint_loss(out,ref.flow.loss(out.velocity,draw.target_velocity),b,draw.flow_time,
        stage=ref.stage,config=ref.loss_config,sampled_config=ref.sampled_config)
    ref.optimizer.zero_grad(set_to_none=True);losses.total.backward()
    static_loss,_=static_backbone_loss(reference,static,ref.static_backbone_config,generator=ref.static_generator,successful_updates=0)
    from molvid.training.static_backbone import backward_static_loss
    backward_static_loss(static_loss,reference,ref.static_backbone_config)
    torch.nn.utils.clip_grad_norm_([p for p in reference.parameters() if p.requires_grad],ref.grad_clip);ref.optimizer.step()
    for n,p in model.named_parameters(): torch.testing.assert_close(p,dict(reference.named_parameters())[n],rtol=0,atol=0)
    assert torch.equal(rng.get_state(),ref_rng.get_state())
    assert {int(s['step']) for s in t.optimizer.state.values()}=={1}
    path=tmp_path/'d2.pt';t.save_checkpoint(path,cursor={'dynamic':1,'static':1},generator=rng)
    restored=trainer(deepcopy(model)); new_rng=torch.Generator(device='cuda').manual_seed(999)
    restored.static_backbone_config=replace(restored.static_backbone_config,gradient_scope=scope)
    payload=restored.load_checkpoint(path,generator=new_rng)
    assert payload['cursor']=={'dynamic':1,'static':1}
    for tr,g in [(t,rng),(restored,new_rng)]:
        sb=prepare_static_backbone_batch(tr.base_model,cpu,device=cuda(),generator=tr.static_corruption_generator)
        tr.train_step(b,generator=g,static_batch=sb)
    for n,p in model.named_parameters(): torch.testing.assert_close(p,dict(restored.base_model.named_parameters())[n],rtol=0,atol=0)
    assert torch.equal(t.static_generator.get_state(),restored.static_generator.get_state())
    assert torch.equal(t.static_corruption_generator.get_state(),restored.static_corruption_generator.get_state())


@pytest.mark.parametrize('amp',[False,True])
def test_decoder_only_auxiliary_preserves_existing_trunk_grads(amp):
    from molvid.training.static_backbone import backward_static_loss
    model=_model(cuda()); cpu,_=batches(model)
    b=prepare_static_backbone_batch(model,cpu,device=cuda(),generator=torch.Generator(device='cuda').manual_seed(7))
    grads={}
    for n,p in model.named_parameters():
        if p.requires_grad and not n.startswith('decoder.'):
            p.grad=torch.full_like(p,.125);grads[n]=p.grad.clone()
    cfg=StaticBackboneConfig(.01,gradient_scope='decoder_only',rollout_steps=2)
    with torch.autocast('cuda',dtype=torch.bfloat16,enabled=amp):
        loss,metrics=static_backbone_loss(model,b,cfg,generator=torch.Generator(device='cuda').manual_seed(3),successful_updates=0)
        shared,other=static_backbone_loss(model,b,replace(cfg,gradient_scope='shared'),generator=torch.Generator(device='cuda').manual_seed(3),successful_updates=0)
    assert torch.equal(loss,shared) and metrics==other
    backward_static_loss(loss,model,cfg)
    for n,p in model.named_parameters():
        if n in grads: assert torch.equal(p.grad,grads[n]),n
        elif not n.startswith('decoder.'): assert p.grad is None,n
    assert any(p.grad is not None and bool(p.grad.abs().sum()>0) for p in model.decoder.parameters())


def test_per_module_calibration_is_a_common_frozen_cap():
    from molvid.training.static_response import calibrate_module_budget
    rows=[{'dit':{'dynamic_norm':2.,'static_norm':8.,'cosine':-.5},
           'decoder':{'dynamic_norm':1.,'static_norm':1.,'cosine':.2}} for _ in range(16)]
    weight,budgets=calibrate_module_budget(rows)
    assert weight==pytest.approx(.025)
    assert budgets['dit']['negative_cosines']==16
    assert budgets['decoder']['median_paired_ratio']==pytest.approx(.025)
    with pytest.raises(ValueError): calibrate_module_budget(rows[:1])


def test_calibration_requires_training_pairs_and_uses_gradient_ratio():
    assert calibrated_static_weight([2.]*16,[4.]*16)==pytest.approx(.125)
    with pytest.raises(ValueError): calibrated_static_weight([2.],[4.])
    from molvid.training.static_response import profile_history
    for i in range(8):
        assert profile_history('fixed_history_dt_400ps',i)==4
        assert profile_history('dt_400ps',i)==(4 if i%2==0 else 8)


def test_d2_with_sampled_physical_full_and_streamed_backward():
    from molvid.losses.sampled_physical import PhysicalAuxiliaryConfig
    from molvid.training.joint import SampledAuxiliaryConfig
    model=_model(cuda());other=deepcopy(model)
    physical=PhysicalAuxiliaryConfig.resolve(dict(clash_weight=.01,vdw_radii_A={'5':2.,'6':2.},
        allowed_overlap_A=.8,one_four_overlap_A=1.2,distance_chunk_size=2,
        table_provenance={'split':'train','source_hash':'fixture'}))
    configs=[SampledAuxiliaryConfig(enabled=True,cadence=8,draws=2,euler_steps=4,
        physical=physical,activation_checkpoint=True,streamed_atom_frames_threshold=threshold) for threshold in (0,1)]
    trainers=[FrameJointTrainer(m,loss_config=JointLossConfig(False,0),sampled_config=config,
        learning_rates={'history_encoder':1e-4,'dit':2e-4,'decoder':5e-5},stage='joint',amp=False,
        static_backbone_config=StaticBackboneConfig(.01,rollout_steps=2)) for m,config in zip((model,other),configs)]
    cpu,batch=batches(model)
    for tr in trainers:
        static=prepare_static_backbone_batch(tr.base_model,cpu,device=cuda(),generator=tr.static_corruption_generator)
        metrics=tr.train_step(batch,generator=torch.Generator(device=cuda()).manual_seed(9),
            sampled_generator=torch.Generator(device=cuda()).manual_seed(19),auxiliary_active=True,static_batch=static)
        assert metrics['sampled_branch_active'] and 'static_rollout_geometry' in metrics
        assert {int(s['step']) for s in tr.optimizer.state.values()}=={1}
    for name,p in model.named_parameters():
        q=dict(other.named_parameters())[name]
        assert (p.grad is None)==(q.grad is None),name
        if p.grad is not None:
            torch.testing.assert_close(p.grad,q.grad,rtol=2e-4,atol=2e-6,msg=name)
