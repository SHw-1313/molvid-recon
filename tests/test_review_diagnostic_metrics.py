import torch
from molvid.evaluation.review_diagnostics import internal_geometry


def test_geometry_diagnostics_identity_and_rigid_transform():
    torch.manual_seed(17)
    x=torch.randn(3,5,3,device='cuda')
    bonds=torch.tensor([[0,1,1,2,2,3,1,4],[1,0,2,1,3,2,4,1]],device='cuda')
    identity=internal_geometry(x,x,bonds)
    assert identity['torsion_circular_error']==0
    assert identity['angle_cosine_rmse']==0
    assert identity['oriented_volume_sign_mismatch']==0
    rotation=torch.tensor([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]],device='cuda')
    moved=internal_geometry(x@rotation+7,x,bonds)
    assert moved['angle_cosine_rmse']<1e-5
    assert moved['torsion_circular_error']<1e-5
    assert moved['oriented_volume_sign_mismatch']==0
