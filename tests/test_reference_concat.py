"""Bounded CUDA checks; execution is scheduled by the static-reference study."""
from dataclasses import replace
from pathlib import Path
import subprocess
import sys
from types import ModuleType

import pytest
import torch

from molvid.codec.frame import slice_clip_frames
from molvid.flow.coordinate_state import CoordinateStateDenoiser
from molvid.flow.reference_concat import ReferenceConcat
from test_flow_capacity_cuda import batch_fixture, open_outputs, settings

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='Real CUDA required')
ROOT = Path(__file__).resolve().parents[1]


def recipe(**overrides):
    return settings(spatial_layers=6, query_post_extra_layers=1, **overrides)


def inputs(clip, task):
    h = 3 if task == 'dynamic' else 1
    observed = slice_clip_frames(clip, 0, h)
    current = clip.x[3:5]+.15 if task == 'dynamic' else clip.x[:1]+.15
    times = clip.time_ps[:, 3:5] if task == 'dynamic' else clip.time_ps[:, :1]
    return observed, current, times


def predict(model, clip, task):
    observed, current, times = inputs(clip, task)
    condition = model.prepare_condition(observed, task=task)
    result = model(current, condition=condition, query_time_ps=times,
                   progress=current.new_tensor([.3, .7]))
    return result, condition


def activate_reference(model):
    """A test-only reachable learned state, without changing formal initialization."""
    d = model.settings['width']
    with torch.no_grad():
        eye = torch.eye(d, device=model.reference_fusion.scalar.weight.device)
        model.reference_fusion.scalar.weight[:, d:].copy_(.3*eye)
        model.reference_fusion.vector.weight[:, d:].copy_(.2*eye)


def test_legacy_identity_shared_weights_and_cpu_cuda_rng():
    source = subprocess.check_output(
        ['git', 'show', '44b5197:molvid/flow/coordinate_state.py'], cwd=ROOT, text=True)
    legacy = ModuleType('molvid.flow._reference_concat_legacy_test')
    legacy.__package__ = 'molvid.flow'
    sys.modules[legacy.__name__] = legacy
    exec(compile(source, '44b5197_coordinate_state', 'exec'), legacy.__dict__)
    torch.manual_seed(71)
    start = torch.get_rng_state().clone()
    cuda_start = torch.cuda.get_rng_state().clone()
    old = legacy.CoordinateStateDenoiser(recipe())
    after_old = torch.get_rng_state().clone()
    torch.set_rng_state(start)
    default = CoordinateStateDenoiser(recipe())
    assert torch.equal(torch.get_rng_state(), after_old)
    torch.set_rng_state(start)
    fused = CoordinateStateDenoiser(recipe(reference_concat=True))
    assert torch.equal(torch.get_rng_state(), after_old)
    assert torch.equal(torch.cuda.get_rng_state(), cuda_start)
    assert default.contract() == old.contract()
    default.load_state_dict(old.state_dict(), strict=True)
    assert set(default.state_dict()) == set(old.state_dict())
    assert set(fused.state_dict())-set(old.state_dict()) == {
        'reference_fusion.scalar.weight', 'reference_fusion.vector.weight'}
    for key, value in old.state_dict().items():
        assert torch.equal(value, fused.state_dict()[key]), key
    assert sum(p.numel() for p in fused.parameters())-sum(
        p.numel() for p in default.parameters()) == 4*32**2
    clip = batch_fixture()
    models = [m.cuda().eval() for m in (old, default, fused)]
    with torch.no_grad():
        for task in ('static', 'dynamic'):
            values = [predict(m, clip, task)[0] for m in models]
            for actual in values[1:]:
                torch.testing.assert_close(actual, values[0], atol=1e-5, rtol=1e-5)


def test_atomwise_concat_autocast_identity_and_first_update_gradients():
    torch.manual_seed(721)
    d, n = 16, 7
    layer = ReferenceConcat(d).cuda()
    h = torch.randn(n, d, device='cuda')
    v = torch.randn(n, 3, d, device='cuda')
    rh, rv = torch.randn_like(h), torch.randn_like(v)
    eye = torch.eye(d, device='cuda')
    assert torch.equal(layer.scalar.weight[:, :d], eye)
    assert torch.equal(layer.vector.weight[:, :d], eye)
    assert torch.count_nonzero(layer.scalar.weight[:, d:]) == 0
    assert torch.count_nonzero(layer.vector.weight[:, d:]) == 0
    with torch.autocast('cuda', dtype=torch.bfloat16):
        initial_h, initial_v = layer(h, v, rh, rv)
    # Identity is bit-exact even when the incoming residual stream is FP32.
    assert torch.equal(initial_h, h) and torch.equal(initial_v, v)
    (initial_h.square().mean()+initial_v.square().mean()).backward()
    for parameter in layer.parameters():
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad[:, d:].norm() > 0  # No zero gate blocks the reference.
    with torch.no_grad():
        layer.scalar.weight[:, d:].copy_(.3*eye)
        layer.vector.weight[:, d:].copy_(.2*eye)
        base_h, base_v = layer(h, v, rh, rv)
        rh2, rv2 = rh.clone(), rv.clone()
        rh2[2].add_(1.)
        rv2[2].mul_(2.)
        changed_h, changed_v = layer(h, v, rh2, rv2)
    keep = torch.arange(n, device='cuda') != 2
    assert torch.equal(base_h[keep], changed_h[keep])
    assert torch.equal(base_v[keep], changed_v[keep])
    assert not torch.equal(base_h[2], changed_h[2])
    assert not torch.equal(base_v[2], changed_v[2])
    assert not torch.equal(base_v[0], base_v[1])


def test_same_depth_observed_cache_and_static_dynamic_reference_gradients():
    torch.manual_seed(723)
    model = CoordinateStateDenoiser(recipe(reference_concat=True)).cuda().eval()
    clip = batch_fixture()
    split = len(model.spatial.attention_layers)-2
    for task in ('static', 'dynamic'):
        observed, current, times = inputs(clip, task)
        entering_last_two = []
        hook = model.spatial.attention_layers[split].register_forward_pre_hook(
            lambda module, args: entering_last_two.append((args[0].clone(), args[1].clone())))
        try:
            before = model.geometry_calls
            condition = model.prepare_condition(observed, task=task)
        finally:
            hook.remove()
        assert model.geometry_calls-before == observed.frames
        assert torch.equal(condition.reference_prefix_h, entering_last_two[-1][0])
        assert torch.equal(condition.reference_prefix_v, entering_last_two[-1][1])
        assert torch.equal(condition.reference_A, observed.x[-1])
        assert condition.reference_prefix_h.shape == (clip.atom_count, 32)
        # At formal [I,0] initialization the new reference columns themselves
        # receive gradients in static and dynamic tasks, before any optimizer step.
        result = model(current, condition=condition, query_time_ps=times,
                       progress=current.new_tensor([.3, .7]))
        (result.square().mean()+.03*result.sum()).backward()
        for parameter in model.reference_fusion.parameters():
            assert torch.isfinite(parameter.grad).all()
            assert parameter.grad[:, 32:].norm() > 0
        model.zero_grad(set_to_none=True)
    # With a learned nonzero reference block, the cached encoder also receives
    # the static loss gradient (rather than being an unused history computation).
    activate_reference(model)
    model.train()
    model.checkpoint_query_blocks = True
    observed, current, times = inputs(clip, 'static')
    condition = model.prepare_condition(observed, task='static')
    condition.reference_prefix_h.retain_grad()
    condition.reference_prefix_v.retain_grad()
    with torch.autocast('cuda', dtype=torch.bfloat16):
        result = model(current, condition=condition, query_time_ps=times,
                       progress=current.new_tensor([.3, .7]))
    (result.square().mean()+.03*result.sum()).backward()
    for cached in (condition.reference_prefix_h, condition.reference_prefix_v):
        assert torch.isfinite(cached.grad).all() and cached.grad.norm() > 0


def test_activated_reference_model_signal_rigid_equivariance_and_sample_future_isolation():
    torch.manual_seed(724)
    model = CoordinateStateDenoiser(recipe(reference_concat=True)).cuda().eval()
    activate_reference(model)
    open_outputs(model)
    clip = batch_fixture()
    rotation = clip.x.new_tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    translation = clip.x.new_tensor([4., -2., .7])
    with torch.no_grad():
        for task in ('static', 'dynamic'):
            observed, current, times = inputs(clip, task)
            base, condition = predict(model, clip, task)
            altered = replace(condition, reference_prefix_h=condition.reference_prefix_h*1.5,
                              reference_prefix_v=condition.reference_prefix_v*1.5)
            signal = model(current, condition=altered, query_time_ps=times,
                           progress=current.new_tensor([.3, .7]))
            assert torch.isfinite(signal).all() and not torch.allclose(signal, base)
            # Packed samples overlap spatially and share residue IDs on purpose.
            moved = clip.x.clone()
            moved[:, 6:] = moved[:, 6:]*1.8+7.
            isolated, _ = predict(model, replace(clip, x=moved, bpos=moved), task)
            torch.testing.assert_close(isolated[:, :6], base[:, :6], atol=1e-5, rtol=1e-5)
            assert not torch.allclose(isolated[:, 6:], base[:, 6:])
            transformed = replace(observed, x=observed.x@rotation.T+translation,
                                  bpos=observed.bpos@rotation.T+translation)
            transformed_condition = model.prepare_condition(transformed, task=task)
            actual = model(current@rotation.T+translation, condition=transformed_condition,
                           query_time_ps=times, progress=current.new_tensor([.3, .7]))
            torch.testing.assert_close(actual, base@rotation.T, atol=1e-4, rtol=1e-4)
        condition = model.prepare_condition(slice_clip_frames(clip, 0, 3))
        before = [value.clone() for value in (condition.observed.x, condition.observed.time_ps,
            condition.reference_A, condition.reference_prefix_h, condition.reference_prefix_v)]
        clip.x[3:].add_(123.)
        after = (condition.observed.x, condition.observed.time_ps, condition.reference_A,
                 condition.reference_prefix_h, condition.reference_prefix_v)
        assert all(torch.equal(a, b) for a, b in zip(before, after))
        repeat = model.prepare_condition(slice_clip_frames(clip, 0, 3))
        torch.testing.assert_close(repeat.reference_prefix_h, before[3], atol=1e-4, rtol=1e-5)
        torch.testing.assert_close(repeat.reference_prefix_v, before[4], atol=1e-4, rtol=1e-5)
