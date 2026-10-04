import torch
from molvid.temporal_moments import FutureMoments
from molvid.finite_time import FiniteTimeResponse


def test_delta_only_ignores_memory_but_reads_delta():
    for model in (FutureMoments(12,3,4,feature_mode='delta'),FiniteTimeResponse(12,3,4,feature_mode='delta')):
        model=model.cuda()
        torch.nn.init.normal_(model.network[-1].weight,std=.2)
        x=torch.randn(8,12,device='cuda');t=x.new_full((8,),100.)
        changed=x.clone();changed[:,8:]+=20
        def output(z):
            r=model(z,t)
            return r[0] if isinstance(r,tuple) else r
        torch.testing.assert_close(output(x),output(changed),rtol=0,atol=0)
        changed[:,4:8]+=10
        assert not torch.equal(output(x),output(changed))
        loss=output(x).square().mean();loss.backward()
        assert model.network[0].weight.grad[:,8:12].count_nonzero()==0
