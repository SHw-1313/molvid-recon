import pytest
import torch

from molvid.losses.stereochemistry import ChiralReference, observed_ca_reference, handedness_margin_loss


def fixture(device='cpu'):
    x = torch.tensor([[[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]], device=device)
    args = dict(observed=x, atom_identity=['A:1:'+n+':X' for n in ('CA','N','C','CB')],
        block_id=torch.zeros(4, dtype=torch.long, device=device),
        abid=torch.zeros(4, dtype=torch.long, device=device),
        frame_mask=torch.ones(1,1,dtype=torch.bool,device=device),
        loss_mask=torch.ones(4,dtype=torch.bool,device=device),
        bond_index=torch.tensor([[0,0,0],[1,2,3]],device=device))
    return x, args


@pytest.mark.parametrize('device', ['cpu','cuda'])
def test_margin_reflection_gradient_and_rigid_motion(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('explicit CUDA run required separately')
    x, args = fixture(device)
    args['observed'] = x.clone().requires_grad_()
    ref = observed_ca_reference(**args)
    mask = args['frame_mask']
    assert handedness_margin_loss(x*3, ref, frame_mask=mask).item() == 0
    bad = x.clone(); bad[...,2] *= -1; bad.requires_grad_()
    loss = handedness_margin_loss(bad, ref, frame_mask=mask)
    assert float(loss) == pytest.approx(1.21)
    loss.backward()
    assert args['observed'].grad is None
    assert torch.isfinite(bad.grad).all() and bad.grad.norm() > 0
    updated = bad.detach()-.01*bad.grad
    assert handedness_margin_loss(updated,ref,frame_mask=mask) < loss
    rotation = torch.tensor([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]],device=device)
    args['observed'] = x@rotation + 11
    transformed = observed_ca_reference(**args)
    torch.testing.assert_close(handedness_margin_loss(bad@rotation+11,transformed,frame_mask=mask),loss)


def test_topology_and_mask_are_required():
    x, args = fixture()
    args['bond_index'] = args['bond_index'][:,:2]
    with pytest.raises(ValueError,match='bonds'):
        observed_ca_reference(**args)
    args['loss_mask'][3] = False
    ref = observed_ca_reference(**args)
    assert ref.indices.shape == (0,4)
    y = x.clone().requires_grad_()
    loss = handedness_margin_loss(y,ref,frame_mask=args['frame_mask'])
    loss.backward()
    assert loss.item() == 0 and torch.equal(y.grad,torch.zeros_like(y))


def test_packed_sample_equal_and_nan_padding():
    x, args = fixture()
    a = x.clone(); a[...,2] *= -1
    xyz = torch.cat((a,x*3),dim=1).repeat(2,1,1).requires_grad_()
    xyz = xyz.detach(); xyz[1,:4] = float('nan'); xyz.requires_grad_()
    volume = torch.ones(2,requires_grad=True)
    ref = ChiralReference(torch.tensor([[0,1,2,3],[4,5,6,7]]),torch.tensor([0,1]),volume,2)
    loss = handedness_margin_loss(xyz,ref,frame_mask=torch.tensor([[True,False],[True,True]]))
    assert loss.item() == pytest.approx(1.21/2)
    loss.backward()
    assert volume.grad is None
    assert torch.isfinite(xyz.grad).all()
    assert torch.equal(xyz.grad[1,:4],torch.zeros(4,3))


def test_observed_planar_reference_excluded():
    x,args = fixture(); args['observed'][...,3,2] = 0
    assert observed_ca_reference(**args).indices.shape == (0,4)
