"""N0: independent review E1--E5 regression cases, all model paths on CUDA."""
from dataclasses import replace
import mdtraj as md
import pytest
import torch
from molvid.data.chemistry import get_block_from_top, get_block_from_complex
from molvid.data.preprocess import topology_metadata, SourceDataError
from molvid.geometry.types import StaticTopologyMetadata
from molvid.dit.local_geometry import LocalTopologyCache
from molvid.latent.types import FrameLatentBatch, ObservedContext, QuerySpec
from molvid.codec.history import HistoryEncoder
from molvid.time import physical_time_features
from test_static_decoder import _model
from test_static_backbone import batches


def test_roles_and_chain_unique_anchors():
    top = md.Topology()
    for name in ('A', 'B'):
        r = top.add_residue('ALA', top.add_chain(name), resSeq=1)
        top.add_atom('CA', md.element.carbon, r)
    assert get_block_from_top(top)[3].tolist() == [0, 1]
    assert get_block_from_complex(top)[3].tolist() == [0, 1]
    ligand = top.add_residue('LIG', top.chain(0), resSeq=1)
    c = top.add_atom('C1', md.element.carbon, ligand)
    o = top.add_atom('O1', md.element.oxygen, ligand)
    top.add_bond(c, o)
    with pytest.raises(SourceDataError, match='Unclassified'):
        topology_metadata(top, complex_topology=True)
    for name in ('LIG', 'ATP', 'MOL', 'UNK', 'OTHER'):
        ligand.name = name
        value = topology_metadata(top, complex_topology=True, residue_roles={2: 'ligand'},
                                  expected_ligand_residues={2})
        # MDTraj iterates residues within each chain, as the legacy reader did.
        assert value['atom_source_index'].tolist() == [0, 2, 3, 1]
        assert value['block_source_index'].tolist() == [0, 2, 3, 1]
        assert [row['model_index'] for row in value['selection_audit']] == [0, 1, 2, 3]
    with pytest.raises(SourceDataError, match='expected components'):
        topology_metadata(top, complex_topology=True, residue_roles={2: 'ligand'},
                          expected_ligand_residues={3})


def test_semantic_topology_and_cache_are_not_shape_contracts():
    model = _model(torch.device('cuda')).eval()
    _, batch = batches(model)
    top = batch.target_future.topology
    wrong = replace(top, atom_type=top.atom_type + 1)
    assert top.contract() == wrong.contract()  # schema remains checkpoint-compatible
    assert not top.same_identity(wrong)
    with pytest.raises(ValueError, match='contract'):
        model.decoder(batch.observed_context, replace(batch.target_future, topology=wrong), batch.query)
    empty_id = replace(top, topology_id=())
    edges = empty_id.covalent_bond_index.clone()
    edges[1] = (edges[1] + 1) % empty_id.num_atoms
    changed = replace(empty_id, covalent_bond_index=edges)
    cache = LocalTopologyCache()
    a = cache.materialize(empty_id, torch.device('cuda'))
    b = cache.materialize(changed, torch.device('cuda'))
    fresh = LocalTopologyCache().materialize(changed, torch.device('cuda'))
    assert a is not b
    torch.testing.assert_close(b.source, fresh.source, rtol=0, atol=0)
    torch.testing.assert_close(b.destination, fresh.destination, rtol=0, atol=0)
    changed.atom_type.add_(1)
    with pytest.raises(ValueError, match='mutated'):
        cache.materialize(changed, torch.device('cuda'))


def context(h, v, time, mask):
    device = h.device; n = h.shape[1]
    top = StaticTopologyMetadata(atom_type=torch.ones(n, device=device, dtype=torch.long),
        block_type=torch.ones(n, device=device, dtype=torch.long),
        abid=torch.arange(n, device=device), block_id=torch.arange(n, device=device),
        component_id=torch.zeros(n, device=device, dtype=torch.long),
        atom_ptr=torch.arange(n + 1, device=device),
        covalent_bond_index=torch.empty(2, 0, device=device, dtype=torch.long),
        covalent_bond_type=torch.empty(0, device=device, dtype=torch.long))
    latent = FrameLatentBatch(h, v, time, mask, top)
    return ObservedContext(latent, torch.zeros(h.shape[0], n, 3, device=device),
        torch.zeros(n, 3, device=device), torch.ones(n, device=device, dtype=torch.bool))


@pytest.mark.parametrize('lengths', [(8,4), (7,5), (3,8)])
def test_history_is_independent_of_partner_padding_and_order(lengths):
    torch.manual_seed(481)
    h = torch.randn(8, 2, 4, device='cuda'); v = torch.randn(8, 2, 3, 4, device='cuda')
    time = torch.arange(8, device='cuda').float()[None].repeat(2,1)*100
    mask = torch.arange(8, device='cuda')[None] < torch.tensor(lengths, device='cuda')[:, None]
    model = HistoryEncoder(4, 8, 4, heads=1).cuda().eval()
    mixed = model(context(h,v,time,mask))
    for sample, length in enumerate(lengths):
        alone = model(context(h[:length,sample:sample+1],v[:length,sample:sample+1],
                              time[sample:sample+1,:length],mask[sample:sample+1,:length]))
        valid = mixed.token_mask[sample]
        torch.testing.assert_close(mixed.h[valid,sample],alone.h[:,0])
        torch.testing.assert_close(mixed.v[valid,sample],alone.v[:,0])
        torch.testing.assert_close(mixed.group_time_ps[sample,valid],alone.group_time_ps[0])
    changed_h = h.clone(); changed_v = v.clone(); changed_time = time.clone()
    changed_h[~mask.T] = 900.; changed_v[~mask.T] = -700.; changed_time[~mask] = float('nan')
    dirty = model(context(changed_h,changed_v,changed_time,mask))
    torch.testing.assert_close(mixed.h,dirty.h,rtol=0,atol=0)
    reordered = model(context(h[:,[1,0]],v[:,[1,0]],time[[1,0]],mask[[1,0]]))
    torch.testing.assert_close(mixed.h,reordered.h[:,[1,0]],rtol=0,atol=0)
    mixed.h.square().mean().backward()
    assert model.scalar_memory.weight.grad.isfinite().all()


def test_decoder_gap_dt_and_padding_invariance():
    clock = torch.tensor([[0.,100.,float('nan'),-777.,200.,300.]],device='cuda')
    mask = torch.tensor([[1,1,0,0,1,1]],device='cuda',dtype=torch.bool)
    feature = physical_time_features(clock,mask)
    assert feature.isfinite().all() and feature[0,4,1] == 1.
    from molvid.codec.decoder import TrajectoryDecoder
    torch.manual_seed(73)
    h = torch.randn(4,1,4,device='cuda');v=torch.randn(4,1,3,4,device='cuda')
    obs = context(h,v,clock[:,:4],mask[:,:4])
    future = FrameLatentBatch(h[:2],v[:2],clock[:,4:],mask[:,4:],obs.topology)
    query = QuerySpec(future.time_ps,future.frame_mask)
    decoder = TrajectoryDecoder(4,heads=1).cuda().eval()
    for b in decoder.blocks:
        b.temporal_scalar_gate.data.fill_(.4); b.temporal_vector_gate.data.fill_(.4)
    a = decoder(obs,future,query).coordinates
    hh=h.clone();vv=v.clone();hh[2:]=800.;vv[2:]=-600.
    dirty=replace(obs,latent=replace(obs.latent,h=hh,v=vv,time_ps=torch.tensor([[0.,100.,9999.,8888.]],device='cuda')))
    b = decoder(dirty,future,query).coordinates
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    compact = replace(obs,latent=obs.latent.slice_frames(0,2),coordinates=obs.coordinates[:2])
    torch.testing.assert_close(a,decoder(compact,future,query).coordinates,rtol=2e-5,atol=2e-6)
