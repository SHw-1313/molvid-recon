from dataclasses import replace
import copy
import torch
import pytest
from molvid.geometry.ca_detail import condition_from_record,rigid_transport
from molvid.codec.ca_detail import CADetailDecoder,detail_loss
from molvid.runtime import configure_device


def record():
    ref=torch.tensor([[0.,0,0],[1.,0,0],[1.,1,0],[1.,1,1],[2.,1,0],[3.,1,0],[3.,2,0],[3.,2,1]])
    x=torch.stack([ref+i*.05 for i in range(7)]);parent=torch.tensor([1]*4+[5]*4)
    return dict(x=x,bpos=x[:,parent],time_ps=torch.arange(7)*100.,atype=torch.tensor([6,5,5,7]*2),
        btype=torch.full((8,),118),block_id=torch.tensor([0]*4+[1]*4),component_id=torch.zeros(8),
        atom_identity=[f'A:{r}::GLY:{name}:{r*4+j}' for r in range(2) for j,name in enumerate(['N','CA','C','O'])],
        bond_index=torch.tensor([[0,1,2,2,4,5,6],[1,2,3,4,5,6,7]]))


def test_condition_rejects_unsupported_and_never_reads_future_coordinates():
    r=record();a=condition_from_record(r)
    r['x'][4:]=float('nan');r['bpos'][4:]=float('nan')
    b=condition_from_record(r)
    for key in a.__dataclass_fields__: assert torch.equal(getattr(a,key),getattr(b,key))
    r['component_id'][0]=1
    with pytest.raises(ValueError,match='proteins only'):condition_from_record(r)


def test_cuda_equivariance_hard_anchors_and_real_update(tmp_path):
    device=configure_device('cuda',deterministic=True);torch.manual_seed(72)
    r=record();c=condition_from_record(r).to(device);target=r['x'][4:].to(device)
    ca=target[:,c.ca_index]+torch.randn(3,2,3,device=device)*.1
    model=CADetailDecoder(width=16,depth=2).to(device)
    assert torch.equal(model(c,ca),rigid_transport(c,ca))
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
    initial={k:v.clone() for k,v in model.state_dict().items()}
    def step(m,o):
        o.zero_grad(set_to_none=True);p=m(c,ca);loss,_=detail_loss(p,target,c,ca);loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
        o.step();return loss.detach()
    step(model,optimizer);assert any(not torch.equal(v,initial[k]) for k,v in model.state_dict().items())
    pred=model(c,ca);assert torch.equal(pred[:,c.ca_index],ca)
    rotation=torch.tensor([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]],device=device)
    moved=replace(c,observed=c.observed@rotation+3)
    torch.testing.assert_close(model(moved,ca@rotation+3),pred@rotation+3,atol=2e-6,rtol=2e-6)
    path=tmp_path/'resume.pt';torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict()),path)
    saved=torch.load(path,weights_only=True);clone=CADetailDecoder(width=16,depth=2).to(device)
    clone.load_state_dict(saved['model'],strict=True);other=torch.optim.AdamW(clone.parameters(),lr=.001);other.load_state_dict(saved['optimizer'])
    assert torch.equal(step(model,optimizer),step(clone,other))
    for k,v in model.state_dict().items():assert torch.equal(v,clone.state_dict()[k])
    # Future non-CA values cannot affect inference because they are not inputs.
    before=model(c,ca);target.add_(1000)
    assert torch.equal(before,model(c,ca))
    assert not torch.equal(before,model(c,ca+1))
