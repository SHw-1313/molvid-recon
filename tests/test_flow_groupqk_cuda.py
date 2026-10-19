"""Focused routing checks; CUDA execution is owned by the study budget runner."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
from types import ModuleType, SimpleNamespace

import pytest
import torch

from molvid.codec.bridge_history import (AtomTimeAttention, BridgeHistoryConditioner,
    _pooled_query_inputs, future_query_grouping)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='Real CUDA required')
ROOT = Path(__file__).resolve().parents[1]


def fixture():
    torch.manual_seed(1701)
    metadata = SimpleNamespace(
        abid=torch.tensor([0]*6+[1]*4, device='cuda'),
        component_id=torch.tensor([0]*4+[1]*2+[0]*2+[2]*2, device='cuda'),
        block_id=torch.tensor([7,7,8,8,7,7,7,7,7,7], device='cuda'),
        edge_mask=torch.tensor([0]*4+[1]*2+[0]*4, device='cuda'),
        loss_mask=torch.tensor([1,0,0,0,1,1,1,1,1,1], device='cuda', dtype=torch.bool))
    group = future_query_grouping(metadata)
    h = torch.randn(3, 10, 8, device='cuda')
    v = torch.randn(3, 10, 3, 8, device='cuda')
    times = torch.tensor([100., 270., 610.], device='cuda')[:, None] * (metadata.abid[None]+1)
    attention = AtomTimeAttention(8, 2).cuda()
    # Only this numerical fixture opens the production zero-initialized maps.
    with torch.no_grad():
        for name in ('value_h', 'value_v', 'out_h', 'out_v'):
            layer = getattr(attention, name)
            layer.weight.copy_(torch.eye(8, device='cuda'))
            if layer.bias is not None: layer.bias.zero_()
    return metadata, group, h, v, times, attention


def test_shared_routing_keeps_distinct_atom_values_and_physical_time_rows():
    meta, grouping, h, v, times, attention = fixture()
    group, count, mask, representatives = grouping
    assert group[0] == group[1] and group[6] == group[7] and group[8] == group[9]
    assert len({int(group[i]) for i in (0,2,4,5,6,8)}) == 6
    assert count[group[0]] == 1 and count[group[2]] == 0
    delta_h = torch.arange(8, device='cuda') / 4
    delta_v = torch.arange(24, device='cuda').reshape(3,8) / 16
    h[:,1] = h[:,0] + delta_h
    v[:,1] = v[:,0] + delta_v
    pooled, grouped_times, inverse = _pooled_query_inputs(h, times, grouping)
    assert torch.equal(pooled[:, group[0]], h[:,0])
    assert torch.equal(grouped_times[:, group[6]], times[:,6])
    assert torch.equal(pooled[:, group[2]], torch.zeros_like(h[:,2]))
    weights = attention._weights(pooled, pooled, grouped_times, grouped_times)[inverse]
    assert torch.equal(weights[0], weights[1])
    assert not torch.allclose(weights[0,:,0], weights[0,:,1])
    assert not torch.allclose(weights[4], weights[5])  # whole-ligand block never pooled
    dh, dv = attention(h,h,v,times,times,query_grouping=grouping)
    torch.testing.assert_close(dh[:,1]-dh[:,0], delta_h.expand(3,-1), atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(dv[:,1]-dv[:,0], delta_v.expand(3,-1,-1), atol=2e-6, rtol=2e-6)
    altered_time = times.clone(); altered_time[1:, :6] *= 3
    temporal = attention(h,h,v,altered_time,altered_time,query_grouping=grouping)
    assert not torch.allclose(dh[:,:2], temporal[0][:,:2])
    # Other packed sample/compound retains both its clock and its routing.
    torch.testing.assert_close(dh[:,6:], temporal[0][:,6:], atol=0, rtol=0)
    assert torch.isfinite(dh).all() and torch.isfinite(dv).all()


def test_mask_isolation_qk_gradients_and_vector_rotation():
    meta, grouping, h, v, times, attention = fixture()
    h.requires_grad_(); v.requires_grad_()
    before = torch.cuda.get_rng_state().clone()
    dh, dv = attention(h,h,v,times,times,query_grouping=grouping)
    (dh[:,0].square().mean()+dv[:,0].square().mean()).backward()
    assert attention.q.weight.grad.norm() > 0 and attention.k.weight.grad.norm() > 0
    assert torch.isfinite(h.grad).all() and torch.isfinite(v.grad).all()
    assert h.grad[:,0].norm() > 0 and v.grad[:,0].norm() > 0
    assert torch.count_nonzero(h.grad[:,1:]) == 0
    assert torch.equal(before, torch.cuda.get_rng_state())
    changed = h.detach().clone(); changed[:,1:] += torch.randn_like(changed[:,1:])*20
    other = attention(changed,changed,v.detach(),times,times,query_grouping=grouping)
    torch.testing.assert_close(dh[:,0], other[0][:,0], atol=0, rtol=0)
    # Scalar routes are invariant; each atom's vector channels rotate covariantly.
    rotation = h.new_tensor([[0.,-1,0],[1,0,0],[0,0,1]])
    rotated_v = torch.einsum('ij,qnjd->qnid',rotation,v.detach())
    rotated = attention(h.detach(),h.detach(),rotated_v,times,times,query_grouping=None)
    baseline = attention(h.detach(),h.detach(),v.detach(),times,times,query_grouping=None)
    torch.testing.assert_close(rotated[0],baseline[0],atol=0,rtol=0)
    torch.testing.assert_close(rotated[1],torch.einsum('ij,qnjd->qnid',rotation,baseline[1]),atol=2e-6,rtol=2e-6)
    # Exercise the new group path with the same h object for Q and K.
    fixed_h=h.detach()
    rotated=attention(fixed_h,fixed_h,rotated_v,times,times,query_grouping=grouping)
    torch.testing.assert_close(rotated[0],dh,atol=0,rtol=0)
    torch.testing.assert_close(rotated[1],torch.einsum('ij,qnjd->qnid',rotation,dv),atol=2e-6,rtol=2e-6)


def test_default_atom_matches_prechange_code_and_old_state_dict_strictly():
    # Execute the immutable base implementation, not a rewritten expected formula.
    source = subprocess.check_output(['git','show','2f50820:molvid/codec/bridge_history.py'],
                                     cwd=ROOT, text=True)
    legacy = ModuleType('base_bridge_history'); exec(compile(source,'base_bridge_history','exec'),legacy.__dict__)
    old = legacy.BridgeHistoryConditioner(8,2).cuda()
    with torch.no_grad():
        for attention in (old.observed_attention,old.query_attention):
            torch.nn.init.normal_(attention.out_h.weight,std=.1)
            torch.nn.init.normal_(attention.out_v.weight,std=.1)
    default = BridgeHistoryConditioner(8,2).cuda()
    explicit = BridgeHistoryConditioner(8,2,query_qk_mode='atom').cuda()
    grouped = BridgeHistoryConditioner(8,2,query_qk_mode='group').cuda()
    for model in (default,explicit,grouped):
        model.load_state_dict(old.state_dict(),strict=True)
        assert set(model.state_dict()) == set(old.state_dict())
        assert sum(p.numel() for p in model.parameters()) == sum(p.numel() for p in old.parameters())
    meta, grouping, h, v, times, _ = fixture()
    history_h=torch.randn(2,10,8,device='cuda');history_v=torch.randn(2,10,3,8,device='cuda')
    history_delta=torch.randn(2,10,3,device='cuda')
    history_times=torch.tensor([-100.,0.],device='cuda')[:,None].expand(-1,10)
    args=(h,v,history_h,history_v,history_delta,times,history_times)
    expected=old(*args)
    for model in (default,explicit):
        actual=model(*args)
        for a,b in zip(actual,expected): torch.testing.assert_close(a,b,atol=0,rtol=0)
    observed=[]
    hooks=[model.observed_attention.register_forward_hook(lambda module,args,out: observed.append(out))
           for model in (explicit,grouped)]
    try:
        explicit(*args);grouped(*args,query_grouping=grouping)
    finally:
        for hook in hooks: hook.remove()
    for a,b in zip(*observed): torch.testing.assert_close(a,b,atol=0,rtol=0)
    with pytest.raises(ValueError,match='query_qk_mode'):
        BridgeHistoryConditioner(8,2,query_qk_mode='unknown')


def test_real_train_parent_strict_load_and_original_35nfe_source_sampling():
    from molvid.codec.frame import slice_clip_frames
    from molvid.data.bridge_mixture import BridgeMixture
    from molvid.flow.coordinate_state import CoordinateStateDenoiser

    settings=json.loads((ROOT/'tasks/flow-groupqk-20261020/configs/atom.json').read_text())
    mixture=BridgeMixture(settings['data'],seed=71)
    try:
        host,task,history,queries,identity=mixture.next()
    finally:
        mixture.close()
    assert task=='dynamic' and history in (4,8) and len(queries) in (8,12)
    batch=host.to('cuda');observed=slice_clip_frames(batch,0,history)
    assert bool(observed.loss_mask.any()) and bool((observed.edge_mask==0).any())
    parent=Path('/data1/sihao/molvid-derived/coordinate-static-pretrain-v1/seed71/flow/static/coordinate_final.pt')
    saved=torch.load(parent,map_location='cpu',weights_only=False)
    atom_settings=deepcopy(settings);atom_settings.pop('query_qk_mode',None)
    atom=CoordinateStateDenoiser(atom_settings).cuda().eval()
    group=CoordinateStateDenoiser(dict(settings,query_qk_mode='group')).cuda().eval()
    atom.load_state_dict(saved['model'],strict=True)
    group.load_state_dict(saved['model'],strict=True)
    assert atom.contract()['query_qk_mode']=='atom' and group.contract()['query_qk_mode']=='group'
    assert atom.contract()['parameters']==group.contract()['parameters']
    for name,value in atom.state_dict().items(): assert torch.equal(value,group.state_dict()[name]),name
    for model in (atom,group):
        assert torch.count_nonzero(model.temporal.query_attention.out_h.weight)==0
        assert torch.count_nonzero(model.temporal.query_attention.out_v.weight)==0
    del atom,saved
    # Initial zero temporal outputs imply zero upstream Q/K gradient. The small
    # fixture above opens only test maps to prove the reachable nonzero gradient;
    # the real parent/production initialization is kept untouched here.
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=settings['bf16']):
        condition=group.prepare_condition(observed)
        indices,counts,valid,_=condition.query_groups
        protein_valid=(observed.edge_mask==0)&observed.loss_mask
        assert bool(protein_valid.any()) and bool((counts[indices[protein_valid]]>0).all())
        ligand=observed.edge_mask!=0
        assert int(indices[ligand].unique().numel())==int(ligand.sum())
        assert condition.history_h.shape[0]==history
        times=batch.time_ps[:,queries]
        seed=20261020
        expected_generator=torch.Generator(device='cuda').manual_seed(seed)
        expected_source=condition.reference_A[None]+group.rho_A*torch.randn(
            (len(queries),*condition.reference_A.shape),device='cuda',generator=expected_generator)
        actual_generator=torch.Generator(device='cuda').manual_seed(seed)
        first=[]
        def capture(module,args,kwargs):
            if not first:
                assert bool((kwargs['progress']==0).all())
                first.append(args[0].detach().clone())
        hook=group.register_forward_pre_hook(capture,with_kwargs=True)
        try:
            before=group.forward_calls
            generated=group.sample(condition,times,steps=18,generator=actual_generator)
        finally:
            hook.remove()
        assert group.forward_calls-before==35
        torch.testing.assert_close(first[0],expected_source,atol=0,rtol=0)
        assert torch.equal(actual_generator.get_state(),expected_generator.get_state())
        assert generated.shape==expected_source.shape and torch.isfinite(generated).all()
    print(json.dumps(dict(real_train=host.sample_id[0],domain=identity['domain'],history=history,
        query_frames=len(queries),nfe=35,protein_valid_atoms=int(protein_valid.sum()),
        routing_groups=len(counts),ligand_atom_fallback=int(ligand.sum()),parent_strict=True,
        initial_temporal_outputs_zero=True,source_exact=True)))
