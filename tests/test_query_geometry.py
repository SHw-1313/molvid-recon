"""GQ ownership/teacher tests on CPU; real flow numerical tests require CUDA."""
import copy
from dataclasses import replace

import pytest
import torch

from molvid.codec.frame import FrameGeometryEncoder, FrozenFrameTeacher
from molvid.data.batch import collate_clip_records
from molvid.flow.coordinate import CoordinateFlowModel
from molvid.training.coordinate import (CoordinateTrainConfig, make_coordinate_optimizer,
    coordinate_update, capture_coordinate_rng, restore_coordinate_rng)
from test_static_decoder import _model, _Encoder, _Stem, _static_record


def test_clone_preserves_rng_weights_observation_and_default_keys():
    model = CoordinateFlowModel(_model(torch.device('cpu')))
    model.target_teacher.frame_chunk_size = 1
    original_keys = set(model.state_dict())
    old_contract = model.contract()
    rng = torch.get_rng_state().clone()
    model.enable_query_geometry()
    assert model.query_encoder.frame_chunk_size == 1
    assert torch.equal(rng, torch.get_rng_state())
    assert model.contract()['query_encoder'] != old_contract['query_encoder']
    assert set(model.state_dict()) - original_keys == {
        'query_encoder.' + key for key in model.target_teacher.state_dict()}
    for name, parameter in model.query_encoder.named_parameters():
        frozen = dict(model.target_teacher.named_parameters())[name]
        assert torch.equal(parameter, frozen) and parameter.data_ptr() != frozen.data_ptr()
        assert parameter.requires_grad and not frozen.requires_grad
    model.train()
    assert not model.query_encoder.training and not model.target_teacher.training
    assert all(not module.training for module in model.query_encoder.modules())
    with pytest.raises(ValueError, match='already enabled'):
        model.enable_query_geometry()


def test_optimizer_exact_ownership_and_fixed_geometry_rate():
    model = CoordinateFlowModel(_model(torch.device('cpu')), train_query_geometry=True)
    config = CoordinateTrainConfig()
    optimizer = make_coordinate_optimizer(model, config)
    groups = {group['name']: group for group in optimizer.param_groups}
    assert groups['query_geometry']['lr'] == config.history_lr
    assert {id(p) for p in groups['query_geometry']['params']} == {id(p) for p in model.query_encoder.parameters()}
    assert not {id(p) for p in model.target_teacher.parameters()} & {id(p) for group in optimizer.param_groups for p in group['params']}


def test_default_has_no_new_state_or_optimizer_group():
    model = CoordinateFlowModel(_model(torch.device('cpu')))
    assert model.query_encoder is None
    assert not any(key.startswith('query_encoder.') for key in model.state_dict())
    assert [group['name'] for group in make_coordinate_optimizer(model, CoordinateTrainConfig()).param_groups] == ['dit', 'history', 'readout']
    with pytest.raises(ValueError, match='boolean'):
        CoordinateFlowModel(_model(torch.device('cpu')), train_query_geometry='yes')


def test_shared_toy_encoder_arithmetic_frozen_and_trainable_boundaries():
    # Deliberately a two-scalar toy, not a CPU fallback of the protein model.
    teacher = FrozenFrameTeacher(_Encoder(2), _Stem(2))
    geometry = FrameGeometryEncoder(copy.deepcopy(teacher.frame_encoder),
                                   copy.deepcopy(teacher.coordinate_stem)).requires_grad_(True).eval()
    batch = collate_clip_records([_static_record('a'), _static_record('b', 5.)])
    batch = replace(batch, x=batch.x.detach().requires_grad_(True))
    frozen, origin = teacher(batch)
    live, origin2 = geometry(batch)
    coordinate_grad, _ = teacher.encode_with_coordinate_grad(batch)
    assert torch.equal(origin, origin2)
    assert torch.equal(frozen.h, live.h) and torch.equal(frozen.v, live.v)
    assert torch.equal(coordinate_grad.v, live.v)
    assert not frozen.v.requires_grad and not teacher.observed_context(batch).latent.v.requires_grad
    (live.h.square().mean() + live.v.square().mean()).backward()
    assert batch.x.grad is not None and torch.isfinite(batch.x.grad).all()
    assert all(p.grad is None for p in teacher.parameters())
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
               for p in geometry.parameters())


cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA qualification still required')


@cuda
def test_cuda_gq_initial_parity_live_gradients_and_static_shared_path():
    from test_coordinate_flow import fixture
    from molvid.codec.frame import slice_clip_frames
    model, observed, query = fixture()
    condition = model.prepare_condition(observed)
    source = model.source(condition=condition, query=query, generator=torch.Generator(device='cuda').manual_seed(98))
    progress = torch.tensor([.2, .7], device='cuda')
    old = model(source, condition=condition, query=query, flow_time=progress).detach()
    rng = torch.cuda.get_rng_state().clone()
    model.enable_query_geometry()
    assert torch.equal(rng, torch.cuda.get_rng_state())
    model.prepare_batch(observed.to('cpu'))
    new = model(source, condition=condition, query=query, flow_time=progress)
    assert torch.equal(old, new.detach())
    new.square().mean().backward()
    assert any(p.grad is not None and p.grad.norm() > 0 for p in model.query_encoder.parameters())
    assert all(p.grad is None for p in model.target_teacher.parameters())
    model.zero_grad(set_to_none=True)
    static = model.prepare_condition(slice_clip_frames(observed, 0, 1), task='static')
    model.forward_static(static.observed.x + .1, condition=static, flow_time=progress).square().mean().backward()
    assert any(p.grad is not None and p.grad.norm() > 0 for p in model.query_encoder.parameters())
    assert all(p.grad is None for p in model.history_encoder.parameters())


@cuda
def test_cuda_gq_update_and_exact_optimizer_rng_resume():
    from test_coordinate_flow import fixture
    model, observed, _ = fixture()
    model.enable_query_geometry()
    target = observed.x[-1:].expand(2, -1, -1) + .05
    times = torch.arange(6, device='cuda')[None].expand(2, -1).float() * 100
    batch = replace(observed, x=torch.cat((observed.x, target)),
        bpos=torch.cat((observed.bpos, target)), time_ps=times,
        frame_mask=torch.ones(2, 6, device='cuda', dtype=torch.bool),
        delta_time_ps=times[:, 1:] - times[:, :-1]).to('cpu')
    config = CoordinateTrainConfig()
    optimizer = make_coordinate_optimizer(model, config)
    generator = torch.Generator(device='cuda').manual_seed(818)
    before = copy.deepcopy(model.query_encoder.state_dict())
    coordinate_update(model, optimizer, batch, history_frames=4, generator=generator, step=0, config=config)
    assert any(not torch.equal(value, model.query_encoder.state_dict()[key]) for key, value in before.items())
    saved = copy.deepcopy(model.state_dict()), copy.deepcopy(optimizer.state_dict()), capture_coordinate_rng(generator)
    expected = coordinate_update(model, optimizer, batch, history_frames=4, generator=generator, step=1, config=config)
    other, _, _ = fixture()
    other.enable_query_geometry()
    other.load_state_dict(saved[0], strict=True)
    other_optimizer = make_coordinate_optimizer(other, config)
    other_optimizer.load_state_dict(saved[1])
    other_generator = torch.Generator(device='cuda')
    restore_coordinate_rng(saved[2], other_generator)
    actual = coordinate_update(other, other_optimizer, batch, history_frames=4, generator=other_generator, step=1, config=config)
    assert expected == actual
    for key, value in model.state_dict().items():
        assert torch.equal(value, other.state_dict()[key]), key
    for key, state in optimizer.state_dict()['state'].items():
        for name, value in state.items():
            assert torch.equal(value, other_optimizer.state_dict()['state'][key][name])
