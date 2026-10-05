"""CUDA numerical boundaries for the isolated coordinate-state prototype."""
from dataclasses import replace

import pytest
import torch

from molvid.codec.frame import slice_clip_frames
from molvid.data.batch import collate_clip_records
from molvid.flow.coordinate import CoordinateFlowModel, coordinate_flow_loss
from molvid.latent.types import QuerySpec
from test_static_decoder import _model, _static_record


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def fixture():
    torch.manual_seed(20261005)
    torch.cuda.set_per_process_memory_fraction(.025)
    model = CoordinateFlowModel(_model(torch.device("cuda")))
    # A nonzero test-only readout exposes trunk/input gradients immediately.
    # Production initialization remains the explicitly recorded zero velocity.
    with torch.no_grad():
        model.readout.out.weight.normal_(std=.1)
        # The synthetic parent has zero-output/AdaLN initialization; the real
        # A1 parent is trained. Exercise an active trunk, not a zero-gated one.
        model.dit.adapter.scalar_out.weight.normal_(std=.05)
        model.dit.adapter.vector_out.weight.normal_(std=.05)
        for block in model.dit.blocks:
            for name in ("history_adaln", "spatial_adaln", "temporal_adaln", "ffn_adaln"):
                getattr(block, name).modulation.bias.normal_(std=.1)
    cpu = collate_clip_records([_static_record("a"), _static_record("b", 10.)])
    model.prepare_batch(cpu)
    batch = cpu.to("cuda")
    x = batch.x.expand(4, -1, -1).clone()
    x += .03 * torch.randn_like(x)
    time = torch.arange(4, device="cuda", dtype=torch.float32)[None].expand(2, -1) * 100
    observed = replace(batch, x=x, bpos=x.clone(), frame_mask=torch.ones(2, 4, device="cuda", dtype=torch.bool),
                       time_ps=time, delta_time_ps=time[:, 1:] - time[:, :-1])
    query = QuerySpec(torch.tensor([[400., 500.], [400., 500.]], device="cuda"),
                      torch.tensor([[True, True], [True, False]], device="cuda"))
    return model, observed, query


def test_coordinate_teacher_preserves_old_forward_and_coordinate_autograd():
    model, observed, _ = fixture()
    state = replace(observed, x=observed.x.detach().requires_grad_(True))
    frozen, origin = model.target_teacher(state)
    live, origin2 = model.target_teacher.encode_with_coordinate_grad(state)
    assert torch.equal(frozen.h, live.h) and torch.equal(frozen.v, live.v)
    assert torch.equal(origin, origin2) and not frozen.v.requires_grad
    (live.h.square().mean() + live.v.square().mean()).backward()
    assert torch.isfinite(state.x.grad).all() and state.x.grad.norm() > 0
    assert all(p.grad is None for p in model.target_teacher.parameters())


def test_coordinate_flow_rigid_equivariance_packing_padding_and_clock():
    model, observed, query = fixture()
    model.eval()
    condition = model.prepare_condition(observed)
    source = model.source(condition=condition, query=query, generator=torch.Generator(device="cuda").manual_seed(7))
    progress = torch.tensor([.2, .7], device="cuda")
    velocity = model(source, condition=condition, query=query, flow_time=progress)
    assert torch.count_nonzero(velocity[1, 4:]) == 0
    rotation, _ = torch.linalg.qr(torch.tensor([[.4, .2, .9], [.7, -.6, .3], [.1, .8, .5]], device="cuda"))
    if torch.linalg.det(rotation) < 0:
        rotation[:, 0] *= -1
    shift = torch.tensor([3., -2., .7], device="cuda")
    transformed = replace(observed, x=observed.x @ rotation + shift, bpos=observed.bpos @ rotation + shift)
    other = model.prepare_condition(transformed)
    moved = model(source @ rotation + shift, condition=other, query=query, flow_time=progress)
    torch.testing.assert_close(moved, velocity @ rotation, atol=2e-5, rtol=2e-5)
    # Mutating another packed system and padding cannot affect valid system 0.
    changed = source.clone()
    changed[:, 4:] += 25
    repeated = model(changed, condition=condition, query=query, flow_time=progress)
    torch.testing.assert_close(repeated[:, :4], velocity[:, :4], atol=1e-6, rtol=1e-6)
    with pytest.raises(ValueError, match="follow observed history"):
        model.source(condition=condition, query=replace(query, time_ps=query.time_ps - 1000),
                     generator=torch.Generator(device="cuda"))


def test_actual_source_rollout_backpropagates_and_loss_ignores_padding():
    model, observed, query = fixture()
    condition = model.prepare_condition(observed)
    source = model.source(condition=condition, query=query, generator=torch.Generator(device="cuda").manual_seed(8))
    source.requires_grad_(True)
    output = model.rollout(source, condition=condition, query=query, steps=4)
    target = torch.randn_like(output, requires_grad=True)
    loss = coordinate_flow_loss(output, target, batch=observed, query=query)
    changed = target.detach().clone()
    changed[1, 4:] += 100
    assert torch.equal(loss, coordinate_flow_loss(output, changed, batch=observed, query=query))
    loss.backward()
    assert target.grad is None
    assert torch.isfinite(source.grad).all() and source.grad.norm() > 0
    for module in (model.history_encoder, model.dit, model.readout):
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert sum(float(g.norm()) for g in grads) > 0
    assert all(p.grad is None for p in model.target_teacher.parameters())
    assert not any(k.startswith("decoder.") for k in model.state_dict())


def test_static_shared_trunk_has_no_temporal_gradient_or_physical_clock():
    model, observed, _ = fixture()
    corrupted = slice_clip_frames(observed, 0, 1)
    condition = model.prepare_condition(corrupted, task="static")
    state = corrupted.x + .1
    flow = torch.tensor([.3, .6], device="cuda")
    output = model.forward_static(state, condition=condition, flow_time=flow)
    altered = replace(corrupted, time_ps=corrupted.time_ps + 9817)
    other = model.prepare_condition(altered, task="static")
    assert torch.equal(output, model.forward_static(state, condition=other, flow_time=flow))
    output.square().mean().backward()
    assert all(p.grad is None for p in model.history_encoder.parameters())
    for name, parameter in model.dit.named_parameters():
        if "temporal" in name or "history" in name or "query_time" in name:
            assert parameter.grad is None, name
    assert any(p.grad is not None and p.grad.norm() > 0 for p in model.dit.parameters())


def test_initialization_is_zero_coordinate_velocity_not_old_forecast():
    model, observed, query = fixture()
    torch.nn.init.zeros_(model.readout.out.weight)
    condition = model.prepare_condition(observed)
    source = model.source(condition=condition, query=query, generator=torch.Generator(device="cuda").manual_seed(1))
    with torch.no_grad():
        output = model.rollout(source, condition=condition, query=query, steps=2)
    assert torch.equal(source, output)
    assert model.warm_start_report["retired_state_keys"]
    assert model.contract()["state"] == "coordinates_angstrom_Q_N_3"
