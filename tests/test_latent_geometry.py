import pytest
import torch
from molvid.evaluation.latent_geometry import channel_spectrum, matched_norm, norm_stats


def test_cuda_channel_spectrum_and_perturbation_rotation():
    assert torch.cuda.is_available(), 'This numerical test requires CUDA'
    generator=torch.Generator(device='cuda').manual_seed(177)
    direction=torch.randn(4,19,3,1,device='cuda',generator=generator)
    channels=torch.arange(1,9,device='cuda').float()
    vector=direction*channels
    rotation,_=torch.linalg.qr(torch.randn(3,3,device='cuda',generator=generator))
    rotated=torch.einsum('ij,tnjc->tnic',rotation,vector)
    first,_=channel_spectrum([vector],vector=True)
    second,_=channel_spectrum([rotated],vector=True)
    assert first['rank99']==1
    assert first['effective_rank']==pytest.approx(1.,abs=1e-5)
    assert second['eigenvalues']==pytest.approx(first['eigenvalues'],rel=1e-5,abs=1e-10)
    scaled=matched_norm(vector,.1)
    torch.testing.assert_close(scaled.flatten(2).norm(dim=-1),torch.full((4,19),.1,device='cuda'))
    torch.testing.assert_close(matched_norm(rotated,.1),torch.einsum('ij,tnjc->tnic',rotation,scaled))
    assert norm_stats(scaled)['rms']==pytest.approx(.1,abs=1e-6)
    assert torch.isfinite(matched_norm(torch.zeros_like(vector),1.)).all()
