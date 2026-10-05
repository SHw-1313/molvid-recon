from types import SimpleNamespace
import torch
from molvid.geometry.projection import ProjectionConfig,repair_coordinates


def test_cuda_projection_equivariance_identity_and_bound():
    assert torch.cuda.is_available()
    torch.use_deterministic_algorithms(True)
    observed=torch.tensor([[[0.,0,0],[1.5,0,0],[3.,.1,0]]],device='cuda')
    generated=observed.repeat(3,1,1).clone();generated[:,1,1]=.6
    topology=SimpleNamespace(batch_size=1,covalent_bond_index=torch.tensor([[0,1],[1,2]],device='cuda'))
    kwargs=dict(topology=topology,loss_mask=torch.ones(3,device='cuda',dtype=torch.bool),
        frame_mask=torch.ones(1,3,device='cuda',dtype=torch.bool),physical=None,
        config=ProjectionConfig(steps=8,clash_weight=0.))
    result,trace=repair_coordinates(generated,observed=observed,**kwargs)
    assert trace[-1]['bond_mse_A2']<trace[0]['bond_mse_A2']
    assert (result-generated).norm(dim=-1).max()<=.50001
    rotation=torch.tensor([[0.,-1,0],[1.,0,0],[0,0,1]],device='cuda')
    offset=torch.tensor([3.,2.,1.],device='cuda')
    other,_=repair_coordinates(generated@rotation+offset,observed=observed@rotation+offset,**kwargs)
    torch.testing.assert_close(other,result@rotation+offset,atol=3e-5,rtol=1e-5)
    identity,_=repair_coordinates(observed.repeat(3,1,1),observed=observed,**kwargs)
    torch.testing.assert_close(identity,observed.repeat(3,1,1),atol=0,rtol=0)
