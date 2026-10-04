import pytest
import torch
from molvid.finite_time import FiniteTimeResponse,fit_response_regions,response_kernels


def test_generated_bpos_do_not_read_hidden_future():
    from dataclasses import replace
    from test_predictive_data import fixture
    from molvid.evaluation.finite_time import generated_target_batch,unbiased_rbf_mmd
    cpu=fixture();future=cpu.x[4:].cuda()+.5
    changed_x=cpu.x.clone();changed_b=cpu.bpos.clone();changed_x[4:]+=90;changed_b[4:]-=30
    a=generated_target_batch(cpu,future)
    b=generated_target_batch(replace(cpu,x=changed_x,bpos=changed_b),future)
    assert torch.equal(a.x,b.x) and torch.equal(a.bpos,b.bpos)
    assert torch.equal(a.x[:4],cpu.x[:4])
    assert isinstance(unbiased_rbf_mmd(torch.randn(8,4,device='cuda'),torch.randn(8,4,device='cuda')),float)


def test_kernel_regions_are_bounded_reproducible_and_local():
    assert torch.cuda.is_available()
    g=torch.Generator(device='cuda').manual_seed(10)
    y=torch.randn(3,8,3,4,device='cuda',generator=g)
    r=fit_response_regions(y,regions=4);other=fit_response_regions(y,regions=4)
    assert torch.equal(r['centers'],other['centers'])
    assert torch.equal(r['bandwidth'],other['bandwidth'])
    values=response_kernels(y,r)
    assert values.shape==(3,8,3,4)
    assert bool(((values>=0)&(values<=1)).all())
    assert torch.allclose(response_kernels(r['centers'],r).diagonal(),torch.ones(4,device='cuda'))


def test_last_state_control_masks_history_and_reloads():
    model=FiniteTimeResponse(12,4,4,history=False).cuda()
    x=torch.randn(10,12,device='cuda');t=x.new_full((10,),400.)
    altered=x.clone();altered[:,4:]+=100
    assert torch.equal(model(x,t),model(altered,t))
    loss=model(x,t).square().mean();loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())
    other=FiniteTimeResponse(12,4,4,history=False).cuda();other.load_state_dict(model.state_dict(),strict=True)
    assert torch.equal(model(x,t),other(x,t))


def test_k_runner_small_cuda_cache(tmp_path):
    import json
    from test_predictive_data import make_small_cache
    from molvid.training.finite_time import train_k
    cache=make_small_cache(tmp_path/'cache')
    train_k(cache,tmp_path/'k',steps=8,seeds=(101,),generation=False)
    result=json.loads((tmp_path/'k/training_summary.json').read_text())
    assert result['status']=='completed' and result['checkpoints']==2
