import pytest
import torch
from molvid.temporal_moments import FutureMoments,gaussian_nll,time_features


def test_history_mask_and_finite_uncertainty_gradients():
    assert torch.cuda.is_available()
    torch.manual_seed(3)
    model=FutureMoments(12,3,4,history=False).cuda()
    # Make the initially zero prediction head nontrivial before testing input isolation.
    torch.nn.init.normal_(model.network[-1].weight,std=.1)
    x=torch.randn(16,12,device='cuda');clock=torch.full((16,),400.,device='cuda')
    changed=x.clone();changed[:,4:]+=100
    a=model(x,clock);b=model(changed,clock)
    assert all(torch.equal(u,v) for u,v in zip(a,b))
    loss=gaussian_nll(*a,torch.randn_like(a[0]));loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert bool((a[1]>0).all())
    with pytest.raises(ValueError): time_features(clock*0)


def test_mean_only_variance_and_strict_state_reload():
    model=FutureMoments(12,3,4,heteroscedastic=False).cuda()
    x=torch.randn(4,12,device='cuda');t=x.new_full((4,),100.)
    a=model(x,t)
    assert torch.equal(a[1],torch.ones_like(a[1]))
    other=FutureMoments(12,3,4,heteroscedastic=False).cuda();other.load_state_dict(model.state_dict(),strict=True)
    assert all(torch.equal(v,w) for v,w in zip(a,other(x,t)))


def test_u_runner_small_cuda_cache(tmp_path):
    import json
    from test_predictive_data import make_small_cache
    from molvid.training.temporal_moments import train_u
    cache=make_small_cache(tmp_path/'cache')
    train_u(cache,tmp_path/'u',steps=8,seeds=(101,))
    result=json.loads((tmp_path/'u/summary.json').read_text())
    assert result['status']=='completed' and result['checkpoints']==3
