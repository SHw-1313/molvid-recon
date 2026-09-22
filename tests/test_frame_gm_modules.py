from __future__ import annotations

import pytest
import torch

from molvid.codec.motion_context import MotionContext, MotionContextEncoder, MotionContextInjector
from molvid.dit.local_geometry import LocalGeometryMessage
from molvid.geometry.types import StaticTopologyMetadata
from molvid.latent.types import FrameLatentBatch, ObservedContext, QuerySpec
from molvid.time import motion_time_features


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _rotation(device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        device=device,
    )


def _rotate_vectors(value: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    return torch.einsum("...ic,ij->...jc", value, rotation)


def _topology(device: torch.device) -> StaticTopologyMetadata:
    # Two disconnected four-atom chains with distinct packed-system ids.
    return StaticTopologyMetadata(
        atom_type=torch.tensor([1, 2, 3, 4, 1, 2, 3, 4], device=device),
        block_type=torch.ones(8, device=device, dtype=torch.long),
        abid=torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], device=device),
        block_id=torch.tensor([0, 0, 1, 1, 0, 0, 1, 1], device=device),
        component_id=torch.zeros(8, device=device, dtype=torch.long),
        atom_ptr=torch.tensor([0, 4, 8], device=device),
        covalent_bond_index=torch.tensor(
            [[0, 1, 2, 4, 5, 6], [1, 2, 3, 5, 6, 7]], device=device
        ),
        covalent_bond_type=torch.ones(6, device=device, dtype=torch.long),
        topology_id=("left", "right"),
        sample_id=("left", "right"),
    )


def _context(
    device: torch.device,
    *,
    moving: bool,
    coordinate_rotation: torch.Tensor | None = None,
    translation: torch.Tensor | None = None,
) -> ObservedContext:
    torch.manual_seed(11)
    topology = _topology(device)
    base_h = torch.randn(1, 8, 4, device=device)
    base_v = torch.randn(1, 8, 3, 4, device=device)
    h = base_h.expand(4, -1, -1).clone()
    v = base_v.expand(4, -1, -1, -1).clone()
    if moving:
        h = h + torch.arange(4, device=device).view(4, 1, 1) * 0.1
        v = v + torch.arange(4, device=device).view(4, 1, 1, 1) * base_v * 0.05
    coordinates = torch.zeros(4, 8, 3, device=device)
    coordinates[:, :, 0] = torch.tensor([0, 1, 2, 3, 20, 21, 22, 23], device=device)
    coordinates[:, :, 1] = torch.arange(4, device=device).view(4, 1) * 0.1
    origin = torch.stack((coordinates[0, :4].mean(0), coordinates[0, 4:].mean(0)))
    if coordinate_rotation is not None:
        coordinates = coordinates @ coordinate_rotation
        origin = origin @ coordinate_rotation
        v = _rotate_vectors(v, coordinate_rotation)
    if translation is not None:
        coordinates = coordinates + translation
        origin = origin + translation
    latent = FrameLatentBatch(
        h=h,
        v=v,
        time_ps=torch.tensor(
            [[0.0, 100.0, 200.0, 300.0], [50.0, 150.0, 250.0, 350.0]],
            device=device,
        ),
        frame_mask=torch.ones(2, 4, device=device, dtype=torch.bool),
        topology=topology,
    )
    return ObservedContext(
        latent=latent,
        coordinates=coordinates,
        sample_origin=origin,
        loss_mask=torch.ones(8, device=device, dtype=torch.bool),
    )


def test_motion_time_features_are_explicit_and_shift_invariant_on_cuda() -> None:
    device = torch.device("cuda")
    observed = torch.tensor([[1000.0, 1100.0, 1200.0, 1300.0]], device=device)
    query = torch.tensor([[1400.0, 1600.0, 2000.0]], device=device)
    mask = torch.ones_like(observed, dtype=torch.bool)
    query_mask = torch.ones_like(query, dtype=torch.bool)
    first = motion_time_features(observed, mask, query, query_mask)
    shifted = motion_time_features(observed + 9876.0, mask, query + 9876.0, query_mask)
    assert first.query_horizon_ps.tolist() == [[100.0, 300.0, 700.0]]
    assert first.query_delta_ps.tolist() == [[100.0, 200.0, 400.0]]
    assert first.history_span_ps.tolist() == [300.0]
    torch.testing.assert_close(first.encoded, shifted.encoded, rtol=0, atol=0)


def test_motion_context_is_zero_for_static_history_and_equivariant_on_cuda() -> None:
    device = torch.device("cuda")
    encoder = MotionContextEncoder(4, 8, 4).to(device)
    static = encoder(_context(device, moving=False))
    assert torch.count_nonzero(static.h) == 0
    assert torch.count_nonzero(static.v) == 0
    moving_context = _context(device, moving=True)
    rotation = _rotation(device)
    rotated_context = _context(device, moving=True, coordinate_rotation=rotation)
    moving = encoder(moving_context)
    rotated = encoder(rotated_context)
    torch.testing.assert_close(moving.h, rotated.h, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(
        _rotate_vectors(moving.v, rotation), rotated.v, rtol=2e-5, atol=2e-6
    )


def test_motion_injector_is_zero_preserving_and_time_shift_invariant_on_cuda() -> None:
    device = torch.device("cuda")
    context = _context(device, moving=True)
    encoded = MotionContextEncoder(4, 8, 4).to(device)(context)
    injector = MotionContextInjector(8, 4).to(device)
    injector.scalar_residual_gate.data.fill_(0.2)
    injector.vector_residual_gate.data.fill_(0.2)
    query = QuerySpec(
        time_ps=torch.tensor([[400.0, 600.0], [450.0, 650.0]], device=device),
        frame_mask=torch.ones(2, 2, device=device, dtype=torch.bool),
    )
    condition = torch.randn(2, 8, 8, device=device)
    first_h, first_v = injector(encoded, condition, context, query)
    shifted_latent = FrameLatentBatch(
        h=context.latent.h,
        v=context.latent.v,
        time_ps=context.latent.time_ps + 5000.0,
        frame_mask=context.latent.frame_mask,
        topology=context.latent.topology,
    )
    shifted_context = ObservedContext(
        latent=shifted_latent,
        coordinates=context.coordinates,
        sample_origin=context.sample_origin,
        loss_mask=context.loss_mask,
    )
    shifted_query = QuerySpec(query.time_ps + 5000.0, query.frame_mask)
    shifted_h, shifted_v = injector(encoded, condition, shifted_context, shifted_query)
    torch.testing.assert_close(first_h, shifted_h, rtol=0, atol=0)
    torch.testing.assert_close(first_v, shifted_v, rtol=0, atol=0)
    zero = MotionContext(
        h=torch.zeros_like(encoded.h),
        v=torch.zeros_like(encoded.v),
        history_span_ps=encoded.history_span_ps,
        observed_delta_ps=encoded.observed_delta_ps,
        observed_delta_mask=encoded.observed_delta_mask,
    )
    zero_h, zero_v = injector(zero, condition, context, query)
    assert torch.count_nonzero(zero_h) == 0
    assert torch.count_nonzero(zero_v) == 0


def test_local_geometry_is_packed_safe_padded_and_equivariant_on_cuda() -> None:
    device = torch.device("cuda")
    context = _context(device, moving=True)
    query_mask = torch.tensor([[True, True], [True, False]], device=device)
    noisy = FrameLatentBatch(
        h=torch.zeros(2, 8, 4, device=device),
        v=torch.zeros(2, 8, 3, 4, device=device),
        time_ps=torch.tensor([[400.0, 500.0], [450.0, 550.0]], device=device),
        frame_mask=query_mask,
        topology=context.topology,
    )
    module = LocalGeometryMessage(8, 4).to(device)
    module.scalar_gate.data.fill_(0.2)
    module.vector_residual_gate.data.fill_(0.2)
    h = torch.randn(2, 8, 8, device=device)
    v = torch.randn(2, 8, 3, 4, device=device)
    first_h, first_v = module(h, v, noisy, context)
    topology = module.cache.materialize(context.topology, device)
    assert torch.all(context.topology.abid[topology.source] == context.topology.abid[topology.destination])
    assert set(topology.hop.tolist()) == {1, 2}
    assert torch.count_nonzero(first_h[1, 4:]) == 0
    assert torch.count_nonzero(first_v[1, 4:]) == 0
    rotation = _rotation(device)
    rotated_context = _context(
        device,
        moving=True,
        coordinate_rotation=rotation,
        translation=torch.tensor([7.0, -3.0, 2.0], device=device),
    )
    rotated_h, rotated_v = module(h, _rotate_vectors(v, rotation), noisy, rotated_context)
    torch.testing.assert_close(first_h, rotated_h, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(
        _rotate_vectors(first_v, rotation), rotated_v, rtol=3e-5, atol=3e-5
    )


def test_local_geometry_single_sided_zero_gate_starts_learning_on_cuda() -> None:
    device = torch.device("cuda")
    context = _context(device, moving=True)
    noisy = FrameLatentBatch(
        h=torch.zeros(2, 8, 4, device=device),
        v=torch.zeros(2, 8, 3, 4, device=device),
        time_ps=torch.tensor([[400.0, 500.0], [450.0, 550.0]], device=device),
        frame_mask=torch.ones(2, 2, device=device, dtype=torch.bool),
        topology=context.topology,
    )
    module = LocalGeometryMessage(8, 4).to(device)
    h = torch.randn(2, 8, 8, device=device)
    v = torch.randn(2, 8, 3, 4, device=device)
    output_h, output_v = module(h, v, noisy, context)
    assert torch.count_nonzero(output_h) == 0
    assert torch.count_nonzero(output_v) == 0
    (output_h.sum() + output_v.sum()).backward()
    assert module.scalar_gate.grad is not None and module.scalar_gate.grad.abs().sum() > 0
    assert module.edge_hidden[0].weight.grad is None or module.edge_hidden[0].weight.grad.abs().sum() == 0
    module.zero_grad(set_to_none=True)
    module.scalar_gate.data.fill_(0.1)
    module.vector_residual_gate.data.fill_(0.1)
    output_h, output_v = module(h, v, noisy, context)
    (output_h.square().mean() + output_v.square().mean()).backward()
    assert module.edge_hidden[0].weight.grad is not None
    assert module.edge_hidden[0].weight.grad.abs().sum() > 0
