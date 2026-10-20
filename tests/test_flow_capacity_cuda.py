"""Capacity-only checks; CUDA execution belongs to the study budget runner."""
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType

import pytest
import torch

from molvid.codec.bridge_history import _pooled_query_inputs
from molvid.codec.frame import slice_clip_frames
from molvid.data.batch import ClipBatch
from molvid.flow.coordinate_state import CoordinateStateDenoiser

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='Real CUDA required')
ROOT = Path(__file__).resolve().parents[1]
PREFIXES = ('obs_extra', 'query_pre_extra', 'temporal_extra', 'collective_extra', 'query_post_extra')


def settings(**overrides):
    value = json.loads((ROOT/'tasks/flow-groupqk-20261020/configs/group.json').read_text())
    value.update(width=32, spatial_layers=4, num_rbf=8, max_neighbors=32)
    value.update(overrides)
    return value


def all_extras():
    return dict(obs_extra_layers=1, query_pre_extra_layers=1, temporal_extra_layers=1,
                collective_extra_layers=1, collective_extra_channels=64, query_post_extra_layers=1)


def batch_fixture(frames=5):
    """Overlapping samples, shared residue IDs, ligand block, empty mask group."""
    torch.manual_seed(20261020)
    n = 10
    base = torch.tensor([[0.,0.,0.],[1.4,.1,0.],[2.1,1.2,.1],[3.2,1.,.5],
                         [.3,2.,.4],[1.2,2.2,.6],[0.,0.,0.],[1.4,.2,.1],
                         [2.4,.7,.2],[3.1,1.3,.6]], device='cuda')
    velocity = torch.arange(n*3, device='cuda').reshape(n,3).float().sin()*.035
    xyz = torch.stack([base + t*velocity for t in range(frames)])
    clock = torch.arange(frames, device='cuda').float()[None] * torch.tensor([[100.],[80.]], device='cuda')
    integer = lambda values: torch.tensor(values, dtype=torch.long, device='cuda')
    mask = torch.tensor([1,0,0,0,1,1,1,1,1,1], dtype=torch.bool, device='cuda')
    return ClipBatch(x=xyz, bpos=xyz.clone(), atype=integer([5,5,6,7,5,7,5,5,6,7]),
        btype=integer([1]*n), block_id=integer([7,7,8,8,7,7,7,7,8,8]),
        component_id=integer([0,0,0,0,1,1,0,0,0,0]), atom_source_index=integer(range(n)),
        abid=integer([0]*6+[1]*4), atom_ptr=integer([0,6,10]),
        bond_index=integer([[0,1,2,4,6,7,8],[1,2,3,5,7,8,9]]),
        edge_mask=integer([0]*4+[1]*2+[0]*4), loss_mask=mask, align_mask=mask,
        frame_mask=torch.ones(2,frames,dtype=torch.bool,device='cuda'), time_ps=clock,
        delta_time_ps=clock[:,1:]-clock[:,:-1], time_bucket_id=('100ps','80ps'),
        topology_id=('fixture0','fixture1'), task=integer([0,0]), sample_id=('fixture0','fixture1'),
        atom_counts=(6,4), host_task_ids=(0,0))


def open_outputs(model):
    """Test-only reachable learned state; production initialization is untouched."""
    with torch.no_grad():
        for modules in (model.obs_extra, model.query_pre_extra, model.query_post_extra):
            for layer in modules:
                torch.nn.init.normal_(layer.o_proj.weight, std=.02)
                torch.nn.init.normal_(layer.v_proj.weight, std=.02)
        for temporal in (model.temporal, *model.temporal_extra):
            for attention in (temporal.observed_attention, temporal.query_attention):
                torch.nn.init.normal_(attention.out_h.weight, std=.02)
                torch.nn.init.normal_(attention.out_v.weight, std=.02)
        for collective in (model.tokens, *model.collective_extra):
            torch.nn.init.normal_(collective.out_h.weight, std=.02)
            torch.nn.init.normal_(collective.out_v.weight, std=.02)


def forward_fixture(model, clip, task='dynamic'):
    observed = slice_clip_frames(clip,0,3 if task=='dynamic' else 1)
    condition = model.prepare_condition(observed, task=task)
    current = clip.x[3:5] + .15 if task=='dynamic' else clip.x[:1] + .15
    times = clip.time_ps[:,3:5] if task=='dynamic' else clip.time_ps[:,:1]
    result = model(current,condition=condition,query_time_ps=times,
                   progress=current.new_tensor([.3,.7]))
    return result, condition


def test_default_base_compatibility_rng_shared_parameters_and_identity_extras():
    source = subprocess.check_output(['git','show','bee4312:molvid/flow/coordinate_state.py'],
                                     cwd=ROOT,text=True)
    legacy = ModuleType('molvid.flow._capacity_base_test')
    legacy.__package__ = 'molvid.flow'
    sys.modules[legacy.__name__] = legacy
    exec(compile(source,'bee4312_coordinate_state','exec'),legacy.__dict__)
    torch.manual_seed(71)
    cpu_start = torch.get_rng_state().clone()
    cuda_start = torch.cuda.get_rng_state().clone()
    old = legacy.CoordinateStateDenoiser(settings())
    after_old = torch.get_rng_state().clone()
    torch.set_rng_state(cpu_start)
    default = CoordinateStateDenoiser(settings())
    assert torch.equal(torch.get_rng_state(),after_old)
    torch.set_rng_state(cpu_start)
    expanded = CoordinateStateDenoiser(settings(**all_extras()))
    assert torch.equal(torch.get_rng_state(),after_old)
    assert torch.equal(torch.cuda.get_rng_state(),cuda_start)
    assert default.contract() == old.contract()
    default.load_state_dict(old.state_dict(),strict=True)
    for key,value in old.state_dict().items():
        assert torch.equal(value,default.state_dict()[key]),key
        assert torch.equal(value,expanded.state_dict()[key]),key
    assert set(default.state_dict()) == set(old.state_dict())
    additions = set(expanded.state_dict())-set(old.state_dict())
    assert additions and all(key.startswith(tuple(p+'.' for p in PREFIXES)) for key in additions)
    for modules in (expanded.obs_extra,expanded.query_pre_extra,expanded.query_post_extra):
        for layer in modules:
            assert torch.count_nonzero(layer.o_proj.weight)==0
            values=layer.v_proj.weight.view(layer.num_heads,3,layer.head_dim,32)
            assert torch.count_nonzero(values[:,1:])==0
            assert torch.count_nonzero(values[:,0])>0
    assert expanded.tokens.channels==128 and expanded.collective_extra[0].query_chunk==64
    clip=batch_fixture()
    with torch.no_grad():
        old_result,_=forward_fixture(old.cuda().eval(),clip)
        default_result,_=forward_fixture(default.cuda().eval(),clip)
        expanded_result,_=forward_fixture(expanded.cuda().eval(),clip)
    # Repeated CUDA graph/segment reductions may differ at roundoff only.
    torch.testing.assert_close(default_result,old_result,atol=1e-5,rtol=1e-5)
    torch.testing.assert_close(expanded_result,old_result,atol=1e-5,rtol=1e-5)


def test_position_routes_checkpoint_gradients_and_static_observation_pretraining():
    model=CoordinateStateDenoiser(settings(**all_extras())).cuda()
    clip=batch_fixture()
    events=[]
    hooks=[]
    for name in PREFIXES:
        hooks.append(getattr(model,name)[0].register_forward_hook(
            lambda module,args,output,name=name: events.append(name)))
    hooks.append(model.temporal.register_forward_hook(lambda *args: events.append('base_temporal')))
    try:
        model.eval()
        with torch.no_grad():
            forward_fixture(model,clip,'dynamic')
            assert Counter(events)==Counter(obs_extra=3,query_pre_extra=2,temporal_extra=1,
                collective_extra=2,query_post_extra=2,base_temporal=1)
            assert events==['obs_extra']*3+['query_pre_extra']*2+['base_temporal','temporal_extra']+[
                'collective_extra','query_post_extra']*2
            events.clear()
            forward_fixture(model,clip,'static')
            assert events==['obs_extra','query_pre_extra','collective_extra','query_post_extra']
    finally:
        for hook in hooks: hook.remove()
    # Static observation weights act on current corrupted XYZ, not unused history.
    model.train()
    out,_=forward_fixture(model,clip,'static')
    (out.square().mean()+.03*out.sum()).backward()
    for name in ('obs_extra','query_pre_extra','query_post_extra'):
        layer=getattr(model,name)[0]
        assert layer.o_proj.weight.grad is not None and layer.o_proj.weight.grad.norm()>0,name
        assert layer.v_proj.weight.grad is not None and layer.v_proj.weight.grad.norm()>0,name
    assert model.collective_extra[0].out_v.weight.grad.norm()>0
    assert all(p.grad is None for p in model.temporal_extra.parameters())
    assert all(p.grad is None for p in model.temporal.parameters())
    # One output-map update opens upstream attention; no second zero gate.
    with torch.no_grad():
        for name in PREFIXES:
            for p in getattr(model,name).parameters():
                if p.grad is not None: p.add_(p.grad,alpha=-.01)
        for attention in (model.temporal.observed_attention,model.temporal.query_attention):
            torch.nn.init.normal_(attention.out_h.weight,std=.02)
            torch.nn.init.normal_(attention.out_v.weight,std=.02)
    model.zero_grad(set_to_none=True)
    out,_=forward_fixture(model,clip,'dynamic')
    (out.square().mean()+.03*out.sum()).backward()
    for name in PREFIXES:
        grads=[p.grad for p in getattr(model,name).parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads),name
        assert sum(float(g.square().sum()) for g in grads)>0,name
    assert model.obs_extra[0].q_proj.weight.grad.norm()>0
    assert model.temporal_extra[0].query_attention.out_v.weight.grad.norm()>0
    print(json.dumps(dict(static_obs_true_gradient=True,dynamic_obs_true_gradient=True,
        static_temporal_bypassed=True,extra_checkpoint_backward_finite=True)))


def test_activated_capacity_preserves_mask_sample_isolation_and_rigid_equivariance():
    model=CoordinateStateDenoiser(settings(**all_extras())).cuda().eval()
    open_outputs(model)
    clip=batch_fixture()
    with torch.no_grad():
        baseline,condition=forward_fixture(model,clip)
        group,count,valid,_=condition.query_groups
        assert group[0]==group[1] and count[group[0]]==1 and count[group[2]]==0
        assert group[4]!=group[5] and group[0]!=group[6]
        h=torch.randn(2,10,32,device='cuda')
        times=clip.time_ps[:,3:5][clip.abid].t()
        pooled,_,_=_pooled_query_inputs(h,times,condition.query_groups)
        assert torch.equal(pooled[:,group[2]],torch.zeros_like(pooled[:,group[2]]))
        assert torch.equal(pooled[:,group[0]],h[:,0])
        changed=clip.x.clone();changed[:,6:]=changed[:,6:]*1.8+7.
        isolated,_=forward_fixture(model,replace(clip,x=changed,bpos=changed))
        torch.testing.assert_close(isolated[:,:6],baseline[:,:6],atol=1e-5,rtol=1e-5)
        assert not torch.allclose(isolated[:,6:],baseline[:,6:])
        rotation=clip.x.new_tensor([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
        translation=clip.x.new_tensor([4.,-2.,.7])
        rotated=clip.x@rotation.T+translation
        transformed=replace(clip,x=rotated,bpos=rotated)
        observed=slice_clip_frames(transformed,0,3)
        rotated_condition=model.prepare_condition(observed)
        current=(clip.x[3:5]+.15)@rotation.T+translation
        result=model(current,condition=rotated_condition,query_time_ps=clip.time_ps[:,3:5],
                     progress=current.new_tensor([.3,.7]))
        torch.testing.assert_close(result,baseline@rotation.T,atol=1e-4,rtol=1e-4)
        assert torch.isfinite(result).all()
        # Only observed storage enters the condition; target mutation is disjoint.
        observed_xyz=condition.observed.x.clone()
        observed_time=condition.observed.time_ps.clone()
        reference=condition.reference_A.clone()
        clip.x[3:].add_(123.)
        assert torch.equal(condition.observed.x,observed_xyz)
        assert torch.equal(condition.observed.time_ps,observed_time)
        assert torch.equal(condition.reference_A,reference)
        repeated=model.prepare_condition(slice_clip_frames(clip,0,3))
        torch.testing.assert_close(repeated.history_h,condition.history_h,atol=1e-4,rtol=1e-5)
        torch.testing.assert_close(repeated.history_v,condition.history_v,atol=1e-4,rtol=1e-5)


def test_real_train_observation_capacity_original_source_and_35nfe():
    from molvid.data.bridge_mixture import BridgeMixture
    recipe=json.loads((ROOT/'tasks/flow-groupqk-20261020/configs/group.json').read_text())
    recipe['obs_extra_layers']=4
    mixture=BridgeMixture(recipe['data'],seed=71)
    try:
        host,task,history,queries,identity=mixture.next()
    finally:
        mixture.close()
    assert task=='dynamic' and history in (4,8) and len(queries) in (8,12)
    batch=host.to('cuda')
    model=CoordinateStateDenoiser(recipe).cuda().eval()
    assert model.contract()['parameters']==10009356
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=recipe['bf16']):
        condition=model.prepare_condition(slice_clip_frames(batch,0,history))
        protein_valid=(condition.observed.edge_mask==0)&condition.observed.loss_mask
        assert bool(protein_valid.any())
        times=batch.time_ps[:,queries]
        expected_generator=torch.Generator(device='cuda').manual_seed(20261020)
        expected=condition.reference_A[None]+model.rho_A*torch.randn(
            (len(queries),*condition.reference_A.shape),device='cuda',generator=expected_generator)
        generator=torch.Generator(device='cuda').manual_seed(20261020)
        first=[]
        def capture(module,args,kwargs):
            if not first:
                assert bool((kwargs['progress']==0).all())
                first.append(args[0].detach().clone())
        hook=model.register_forward_pre_hook(capture,with_kwargs=True)
        try:
            before=model.forward_calls
            generated=model.sample(condition,times,steps=18,generator=generator)
        finally:
            hook.remove()
        assert model.forward_calls-before==35
        torch.testing.assert_close(first[0],expected,atol=0,rtol=0)
        assert torch.equal(generator.get_state(),expected_generator.get_state())
        assert generated.shape==expected.shape and torch.isfinite(generated).all()
    print(json.dumps(dict(real_train=host.sample_id[0],domain=identity['domain'],history=history,
        query_frames=len(queries),atoms=host.atom_count,nfe=35,source_exact=True,
        obs_extra_layers=4,parameters=model.contract()['parameters'])))


@pytest.mark.parametrize('task', ['static', 'dynamic'])
def test_whole_query_checkpoint_only_matches_output_gradient_and_rng(task):
    """Two small, active-path comparisons for the memory-only recomputation."""
    torch.manual_seed(1871)
    plain=CoordinateStateDenoiser(settings(**all_extras())).cuda().train()
    open_outputs(plain)
    recomputed=CoordinateStateDenoiser(settings(**all_extras(),checkpoint_query_blocks=True)).cuda().train()
    recomputed.load_state_dict(plain.state_dict(),strict=True)
    clip=batch_fixture()
    snapshots=[]
    start_rng=torch.cuda.get_rng_state().clone()
    for model in (plain,recomputed):
        observed=slice_clip_frames(clip,0,3 if task=='dynamic' else 1)
        condition=model.prepare_condition(observed,task=task)
        current=(clip.x[3:5]+.15 if task=='dynamic' else clip.x[:1]+.15).detach().requires_grad_()
        times=clip.time_ps[:,3:5] if task=='dynamic' else clip.time_ps[:,:1]
        output=model(current,condition=condition,query_time_ps=times,
                     progress=current.new_tensor([.3,.7]))
        (output.square().mean()+.03*output.sum()).backward()
        gradients={name:p.grad.detach().clone() for name,p in model.named_parameters() if p.grad is not None}
        assert gradients and all(torch.isfinite(value).all() for value in gradients.values())
        assert torch.isfinite(current.grad).all()
        assert torch.equal(torch.cuda.get_rng_state(),start_rng)
        snapshots.append((output.detach(),current.grad.detach(),gradients))
    expected,actual=snapshots
    torch.testing.assert_close(actual[0],expected[0],atol=1e-5,rtol=1e-5)
    torch.testing.assert_close(actual[1],expected[1],atol=2e-5,rtol=2e-4)
    assert actual[2].keys()==expected[2].keys()
    maximum=0.
    for name,value in expected[2].items():
        torch.testing.assert_close(actual[2][name],value,atol=2e-5,rtol=2e-4,msg=name)
        maximum=max(maximum,float((actual[2][name]-value).abs().max()))
    print(json.dumps(dict(checkpoint_query_blocks=True,task=task,
        output_max_abs_difference=float((actual[0]-expected[0]).abs().max()),
        parameter_grad_max_abs_difference=maximum,gradient_tensors=len(expected[2]),rng_exact=True)))
