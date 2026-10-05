from dataclasses import replace
import torch
from molvid.geometry.current_recovery import CurrentGeometryRecovery,RecoveryConfig
from molvid.geometry.types import StaticTopologyMetadata


def topology():
    z=torch.tensor([5,6,5,6],device='cuda')
    return StaticTopologyMetadata(atom_type=z,block_type=torch.zeros_like(z),
        abid=torch.tensor([0,0,1,1],device='cuda'),block_id=torch.tensor([0,0,1,1],device='cuda'),
        component_id=torch.zeros_like(z),atom_ptr=torch.tensor([0,2,4],device='cuda'),
        covalent_bond_index=torch.tensor([[0,2],[1,3]],device='cuda'),
        covalent_bond_type=torch.ones(2,device='cuda',dtype=torch.long))


def test_cuda_current_geometry_identity_equivariance_gradients_and_padding():
    assert torch.cuda.is_available();torch.manual_seed(42)
    model=CurrentGeometryRecovery(RecoveryConfig(width=16,depth=2)).cuda()
    t=topology();mask=torch.tensor([[True,True],[True,False]],device='cuda')
    x=torch.tensor([[[0.,0,0],[1.6,.2,0],[0,0,0],[1.1,.4,.2]]],device='cuda').repeat(2,1,1)
    expected=x.clone();expected[1,2:]=0
    torch.testing.assert_close(model(x,topology=t,frame_mask=mask),expected,atol=0,rtol=0)
    for layer in model.layers: torch.nn.init.normal_(layer.force.weight,std=.1)
    x.requires_grad_(True);result=model(x,topology=t,frame_mask=mask)
    rotation=torch.tensor([[.36,.48,-.8],[-.8,.6,0.],[.48,.64,.6]],device='cuda')
    offset=torch.tensor([3.,2.,1.],device='cuda')
    transformed=model(x@rotation+offset,topology=t,frame_mask=mask)
    reference=result@rotation+offset;reference[1,2:]=0
    torch.testing.assert_close(transformed,reference,atol=2e-5,rtol=2e-5)
    # Systems occupy the same coordinates: graph isolation must still hold.
    changed=x.detach().clone();changed[:,2:]+=50
    other=model(changed,topology=t,frame_mask=mask)
    torch.testing.assert_close(other[:,:2],result[:,:2],atol=0,rtol=0)
    result.square().sum().backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum()>0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert sum(float(p.grad.abs().sum()) for p in model.parameters())>0
