"""Targeted B numerical risks. Run only on an explicitly allocated CUDA slot."""
from dataclasses import replace
import pytest
import torch
from molvid.codec.learned_memory import ChemicalMapping, RawHistoryObservation, LearnedHistoryMemory
from molvid.geometry.types import StaticTopologyMetadata
from molvid.data.chemistry import NUM_BLOCK_TYPE


@pytest.fixture
def observed():
    if not torch.cuda.is_available():
        pytest.fail('B numerical checks require CUDA; CPU fallback is forbidden')
    torch.manual_seed(20260928)
    device = torch.device('cuda')
    n = 12
    abid = torch.arange(2, device=device).repeat_interleave(n//2)
    block = torch.arange(n, device=device)//3
    left = torch.tensor([0,1,2,3,4,6,7,8,9,10], device=device)
    edges = torch.stack((left, left+1))
    topology = StaticTopologyMetadata(torch.full((n,),6,device=device), torch.full((n,),NUM_BLOCK_TYPE-1,dtype=torch.long,device=device), abid, block,
                                      torch.zeros_like(abid), torch.tensor([0,6,12],device=device), edges, torch.ones(edges.shape[1],device=device))
    mapping = ChemicalMapping.from_topology(topology)
    x = torch.randn(9,n,3,device=device)
    times = torch.arange(-800,1,100,device=device).float().expand(2,-1).clone()
    mask = torch.ones(2,9,dtype=torch.bool,device=device)
    # Mixed sampling/lengths, padded values deliberately unrelated to physics.
    times[1,:3] = torch.tensor([-800,-400,0],device=device)
    mask[1,3:] = False
    x[3:,6:] = float('nan')
    anchor = torch.cat((x[8,:6],x[2,6:]))
    return RawHistoryObservation(x,times,mask,topology,mapping,anchor)


def test_rotation_translation_and_padding(observed):
    model = LearnedHistoryMemory().cuda().eval()
    original = model(observed)
    rotation, _ = torch.linalg.qr(torch.randn(3,3,device='cuda'))
    if torch.linalg.det(rotation) < 0:
        rotation[:,0] *= -1
    shift = torch.tensor([11.,-4.,2.],device='cuda')
    moved = replace(observed, coordinates=observed.coordinates@rotation+shift, anchor_coordinates=observed.anchor_coordinates@rotation+shift)
    transformed = model(moved)
    torch.testing.assert_close(original.h, transformed.h, atol=3e-4, rtol=3e-4)
    expected_v = torch.einsum('krdc,de->krec',original.v,rotation)
    torch.testing.assert_close(expected_v, transformed.v, atol=3e-4, rtol=3e-4)
    x = observed.coordinates.clone(); x[3:,6:] = 1e6
    time = observed.physical_time_ps.clone(); time[1,3:] = 1e6
    mutated = model(replace(observed,coordinates=x,physical_time_ps=time))
    torch.testing.assert_close(original.h,mutated.h,atol=0,rtol=0)
    torch.testing.assert_close(original.v,mutated.v,atol=0,rtol=0)
    assert torch.isfinite(original.h).all() and torch.isfinite(original.v).all()


def test_static_unknown_and_support(observed):
    model = LearnedHistoryMemory().cuda()
    static = replace(observed,coordinates=observed.anchor_coordinates[None].expand_as(observed.coordinates).clone())
    memory = model(static)
    assert torch.count_nonzero(memory.analysis.detail_h) == 0
    assert torch.count_nonzero(memory.analysis.detail_v) == 0
    # 800 ps / 400 ps gives three points, no supported detail; state remains valid.
    assert not memory.analysis.valid[1,5:].any()
    assert memory.analysis.valid[:,0].all()
    single = replace(observed,coordinates=observed.anchor_coordinates[None],physical_time_ps=torch.zeros(2,1,device='cuda'),frame_mask=torch.ones(2,1,device='cuda',dtype=torch.bool))
    result = model(single)
    assert (result.analysis.labels[...,4] == 0).all()
    assert (memory.analysis.labels[...,4] == 1).all()
    assert torch.isfinite(result.h).all()


@pytest.mark.parametrize('amp', [False, True])
def test_optimizer_and_reused_graph(observed, amp):
    model = LearnedHistoryMemory().cuda()
    optimizer = torch.optim.AdamW(model.parameters(),lr=1e-4)
    owned = {id(p) for group in optimizer.param_groups for p in group['params']}
    assert owned == {id(p) for p in model.parameters() if p.requires_grad}
    with torch.autocast('cuda',dtype=torch.bfloat16,enabled=amp):
        memory = model(observed)  # exactly once, subsequent readouts reuse graph
        h,v = memory.gather_atoms()
        loss = h.float().square().mean() + v.float().square().mean()
        loss = loss + .1*h.float().sin().mean() + .1*v.float().cos().mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    for name,p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(),name
    for path in ('light_encoder.', 'key.', 'label_key.', 'queries', 'scalar_value.', 'vector_value.', 'scalar_output.', 'vector_output.'):
        norm=sum(p.grad.float().square().sum() for name,p in model.named_parameters() if name.startswith(path))
        assert norm>0,path
    before = model.queries.detach().clone()
    light_before = model.light_encoder.layers[0].message[0].weight.detach().clone()
    optimizer.step()
    assert not torch.equal(before,model.queries)
    assert not torch.equal(light_before,model.light_encoder.layers[0].message[0].weight)


def test_atom_permutation_and_group_gather(observed):
    model=LearnedHistoryMemory().cuda().eval()
    result=model(observed)
    permutation=torch.tensor([4,0,5,1,3,2,10,9,7,11,6,8],device='cuda')
    inverse=torch.argsort(permutation)
    top=observed.topology
    permuted_top=replace(top,atom_type=top.atom_type[permutation],block_type=top.block_type[permutation],abid=top.abid[permutation],
                         block_id=top.block_id[permutation],component_id=top.component_id[permutation],covalent_bond_index=inverse[top.covalent_bond_index])
    other=replace(observed,coordinates=observed.coordinates[:,permutation],anchor_coordinates=observed.anchor_coordinates[permutation],
                  topology=permuted_top,mapping=ChemicalMapping.from_topology(permuted_top))
    h,v=model(other).gather_atoms()
    expected_h,expected_v=result.gather_atoms()
    torch.testing.assert_close(h[:,inverse],expected_h,atol=1e-5,rtol=1e-5)
    torch.testing.assert_close(v[:,inverse],expected_v,atol=1e-5,rtol=1e-5)


def test_clock_units_and_physical_support(observed):
    model=LearnedHistoryMemory().cuda().eval()
    original=model(observed)
    # Same geometries sampled at twice the elapsed physical time have different
    # kernel support and keys, while a constant geometry retains zero detail.
    slower=model(replace(observed,physical_time_ps=observed.physical_time_ps*2))
    assert not torch.equal(original.analysis.valid,slower.analysis.valid)
    assert not torch.equal(original.h,slower.h)
    with pytest.raises(ValueError,match='observed times'):
        replace(observed,physical_time_ps=observed.physical_time_ps+100)
