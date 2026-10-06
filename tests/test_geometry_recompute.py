"""CPU control-flow toys and separately gated real CUDA geometry arithmetic."""
import copy
from dataclasses import replace

import pytest
import torch

from molvid.codec.frame import FrameGeometryEncoder, FrozenFrameTeacher
from molvid.data.batch import collate_clip_records
from molvid.flow.coordinate import CoordinateFlowModel
from test_static_decoder import _Encoder, _Stem, _model, _static_record


class _StochasticEncoder(_Encoder):
    def forward(self, batch):
        result = super().forward(batch)
        # Explicit metadata dependence catches backward capturing the last
        # chunk's timestamps or frame mask instead of its own.
        weight = 1 + batch.time_ps.index_select(0, batch.abid).T[..., None] / 1000
        result.h = torch.nn.functional.dropout(result.h * weight, .3, training=True)
        return result


def batch_fixture(device):
    batch = collate_clip_records([_static_record('a'), _static_record('b', 10.)])
    x = batch.x.expand(7, -1, -1).clone()
    x += torch.arange(7)[:, None, None] * .071 + .013
    times = torch.arange(7).float()[None].expand(2, -1) * 100
    mask = torch.tensor([[True]*6+[False], [True]*4+[False]*3])
    return replace(batch, x=x, bpos=x.clone(), time_ps=times,
        delta_time_ps=times[:, 1:] - times[:, :-1], frame_mask=mask).to(device)


def evaluate(encoder, batch, recompute, coordinate_grad):
    encoder.checkpoint_frames = recompute
    encoder.zero_grad(set_to_none=True)
    x = batch.x.detach().clone().requires_grad_(coordinate_grad)
    state = replace(batch, x=x)
    torch.manual_seed(7601)
    latent, origin = encoder.encode_with_coordinate_grad(state,
        reference_origin=x.new_zeros(batch.batch_size, 3))
    forward_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state().clone() if x.is_cuda else None
    loss = latent.h.square().mean() + latent.v.square().mean()
    loss.backward()
    assert torch.equal(forward_rng, torch.get_rng_state())
    if cuda_rng is not None:
        assert torch.equal(cuda_rng, torch.cuda.get_rng_state())
    gradients = {name: None if p.grad is None else p.grad.detach().clone()
                 for name, p in encoder.named_parameters()}
    mask = batch.frame_mask[batch.abid].T
    assert not torch.count_nonzero(latent.h[~mask])
    assert not torch.count_nonzero(latent.v[~mask])
    if coordinate_grad:
        assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
        assert not torch.count_nonzero(x.grad[~mask])
    return (latent.h.detach(), latent.v.detach(), origin.detach(),
            x.grad, gradients, forward_rng, cuda_rng)


def assert_exact(left, right):
    if isinstance(left, dict):
        assert left.keys() == right.keys()
        for name in left: assert_exact(left[name], right[name])
    elif isinstance(left, tuple):
        for a, b in zip(left, right, strict=True): assert_exact(a, b)
    elif left is None:
        assert right is None
    else:
        assert torch.equal(left, right)


@pytest.mark.parametrize('chunk_size', [1, 2, 4])
@pytest.mark.parametrize('coordinate_grad,trainable', [(True, True), (False, True), (True, False)])
def test_cpu_chunk_metadata_parameter_coordinate_gradients_and_rng(chunk_size, coordinate_grad, trainable):
    encoder = FrameGeometryEncoder(_StochasticEncoder(3), _Stem(3)).requires_grad_(trainable)
    encoder.frame_chunk_size = chunk_size
    batch = batch_fixture('cpu')
    ordinary = evaluate(encoder, batch, False, coordinate_grad)
    recomputed = evaluate(encoder, batch, True, coordinate_grad)
    assert_exact(ordinary, recomputed)
    if trainable:
        assert all(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0
                   for g in recomputed[4].values())
    else:
        assert all(g is None for g in recomputed[4].values())


def test_default_contract_state_keys_and_frozen_observation_boundary():
    model = CoordinateFlowModel(_model(torch.device('cpu')))
    contract = copy.deepcopy(model.contract())
    state = copy.deepcopy(model.state_dict())
    assert not model.target_teacher.checkpoint_frames
    rng = torch.get_rng_state().clone()
    model.set_query_geometry_recompute(True)
    assert model.contract() != contract
    assert torch.equal(rng, torch.get_rng_state())
    assert_exact(state, model.state_dict())
    batch = batch_fixture('cpu')
    output = model.target_teacher.observed_context(batch)
    assert not output.latent.h.requires_grad and not output.latent.v.requires_grad
    model.set_query_geometry_recompute(False)
    assert model.contract() == contract
    with pytest.raises(ValueError, match='boolean'):
        model.set_query_geometry_recompute('yes')
    model.enable_query_geometry()
    model.set_query_geometry_recompute(True)
    assert model.query_encoder.checkpoint_frames
    assert not model.target_teacher.checkpoint_frames
    assert all(not p.requires_grad for p in model.target_teacher.parameters())
    assert all(p.requires_grad for p in model.query_encoder.parameters())


@pytest.mark.parametrize('enabled', [False, True])
def test_padding_only_first_chunk_still_rejected(enabled):
    encoder = FrameGeometryEncoder(_Encoder(2), _Stem(2))
    encoder.frame_chunk_size = 1
    encoder.checkpoint_frames = enabled
    batch = batch_fixture('cpu')
    mask = batch.frame_mask.clone(); mask[:, 0] = False
    with pytest.raises(ValueError, match='frame zero must be valid'):
        encoder(replace(batch, frame_mask=mask))
    with pytest.raises(RuntimeError, match='first frame-encoder chunk'):
        encoder(replace(batch, frame_mask=mask), reference_origin=batch.x.new_zeros(2, 3))


cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason='actual CUDA geometry gate required')


@cuda
@pytest.mark.parametrize('chunk_size', [1, 2])
@pytest.mark.parametrize('trainable', [False, True])
def test_cuda_real_torchmd_graph_parameter_and_coordinate_gradient_parity(chunk_size, trainable):
    from molvid.spatial.encoder import FrameEncoder
    from molvid.codec.heads import CoordinateVectorStem
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(7602)
    encoder = FrameGeometryEncoder(
        FrameEncoder(hidden_channels=8, num_layers=2, num_rbf=8, num_heads=2),
        CoordinateVectorStem(8)).to('cuda').requires_grad_(trainable).eval()
    encoder.frame_chunk_size = chunk_size
    cpu = batch_fixture('cpu'); encoder.prepare_batch(cpu)
    batch = cpu.to('cuda')
    ordinary = evaluate(encoder, batch, False, True)
    recomputed = evaluate(encoder, batch, True, True)
    assert_exact(ordinary, recomputed)


@cuda
def test_cuda_dynamic_then_static_recompute_boundaries():
    from test_coordinate_flow import fixture
    from molvid.codec.frame import slice_clip_frames
    model, observed, query = fixture()
    model.enable_query_geometry()
    model.set_query_geometry_recompute(True)
    model.query_encoder.frame_chunk_size = 1
    condition = model.prepare_condition(observed)
    source = model.source(condition=condition, query=query,
        generator=torch.Generator(device='cuda').manual_seed(91)).requires_grad_(True)
    model.rollout(source, condition=condition, query=query, steps=2).square().mean().backward()
    assert torch.isfinite(source.grad).all() and source.grad.norm() > 0
    assert any(p.grad is not None and p.grad.norm() > 0 for p in model.query_encoder.parameters())
    assert all(p.grad is None for p in model.target_teacher.parameters())
    model.zero_grad(set_to_none=True)
    static = model.prepare_condition(slice_clip_frames(observed, 0, 1), task='static')
    model.forward_static(static.observed.x + .1, condition=static,
        flow_time=torch.tensor([.2,.7],device='cuda')).square().mean().backward()
    assert any(p.grad is not None and p.grad.norm() > 0 for p in model.query_encoder.parameters())
    assert all(p.grad is None for p in model.target_teacher.parameters())
    assert all(p.grad is None for p in model.history_encoder.parameters())
