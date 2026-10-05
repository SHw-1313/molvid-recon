"""Source-distribution algebra and shared train/inference contract checks."""
import pytest
import torch
from test_frame_gm_distribution import _batch
from molvid.flow.source import FrameSourceConfig, sample_frame_source, repeat_last_frame_center
from molvid.flow.objective import FrameRectifiedFlowObjective


@pytest.mark.parametrize('value',[{'h_scale':0},{'v_scale':-1},{'v_scale':float('nan')},
    {'h_scale':True},{'vector_scale':.2},[]])
def test_source_rejects_invalid_contract(value):
    with pytest.raises(ValueError): FrameSourceConfig.resolve(value)


def test_scaled_source_preserves_draws_masks_and_exact_old_default():
    b,_=_batch();center=repeat_last_frame_center(b.observed_context.latent,b.query)
    def rng(): return torch.Generator().manual_seed(73)
    g=rng();h=torch.randn(center.h.shape,generator=g);v=torch.randn(center.v.shape,generator=g)
    mask=center.atom_frame_mask();h=h*mask[...,None];v=v*mask[...,None,None]
    original,noise=sample_frame_source(center,generator=rng())
    torch.testing.assert_close(original.h,center.h+h,rtol=0,atol=0)
    torch.testing.assert_close(original.v,center.v+v,rtol=0,atol=0)
    config=FrameSourceConfig(h_scale=.7,v_scale=.2)
    g=rng();scaled,scaled_noise=sample_frame_source(center,generator=g,config=config)
    assert torch.equal(g.get_state(),rng_after_draws(center))
    torch.testing.assert_close(scaled.h,center.h+.7*h,rtol=0,atol=0)
    torch.testing.assert_close(scaled.v,center.v+.2*v,rtol=0,atol=0)
    torch.testing.assert_close(scaled_noise.v,.2*noise.v,rtol=0,atol=0)
    # RF endpoints must use the SAME source as inference, not the old unit source.
    objective=FrameRectifiedFlowObjective(config)
    target=center.with_features(center.h+2,center.v+.5)
    draw=objective.sample(target,center,generator=rng(),flow_time=torch.zeros(1))
    torch.testing.assert_close(draw.source.v,scaled.v,rtol=0,atol=0)
    torch.testing.assert_close(draw.interpolated.v,scaled.v,rtol=0,atol=0)
    torch.testing.assert_close(draw.target_velocity.v,target.v-scaled.v,rtol=0,atol=0)
    assert FrameSourceConfig.resolve(config.contract())==config


def rng_after_draws(center):
    g=torch.Generator().manual_seed(73)
    torch.randn(center.h.shape,generator=g);torch.randn(center.v.shape,generator=g)
    return g.get_state()
