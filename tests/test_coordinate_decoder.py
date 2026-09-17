from __future__ import annotations

import inspect

import torch
from torch import nn

from module import (
    CodecLatent,
    JointMultiFrameDecoder,
    LatentConditionedSpatialRefiner,
)


def _latent(*, atoms: int = 4, clips: int = 1) -> CodecLatent:
    torch.manual_seed(101 + atoms + clips)
    abid = torch.arange(atoms, dtype=torch.long) * clips // atoms
    if clips == 1:
        abid.zero_()
    latent_frames = 2
    return CodecLatent(
        z_h=torch.randn(latent_frames, atoms, 8),
        z_v=torch.randn(latent_frames, atoms, 3, 6),
        latent_mask=torch.ones(clips, latent_frames, dtype=torch.bool),
        latent_time_ps=torch.arange(latent_frames, dtype=torch.float32).view(1, -1).repeat(clips, 1) * 200.0,
        x_anchor=torch.randn(atoms, 3),
        abid=abid,
    )


def _topology(atoms: int = 4) -> dict[str, torch.Tensor]:
    source = torch.arange(atoms, dtype=torch.long)
    target = (source + 1) % atoms
    edge_index = torch.stack(
        [torch.cat([source, target]), torch.cat([target, source])], dim=0
    )
    return {
        "z": torch.arange(1, atoms + 1, dtype=torch.long),
        "b": torch.zeros(atoms, dtype=torch.long),
        "batch": torch.zeros(atoms, dtype=torch.long),
        "edge_index": edge_index,
        "bond_type": torch.zeros(edge_index.shape[1], dtype=torch.long),
    }


def test_joint_decoder_shapes_and_target_time_query():
    latent = _latent()
    decoder = JointMultiFrameDecoder(8, 6, temporal_layers=0, num_heads=2)
    decoder.eval()
    first = decoder(
        latent, target_time_ps=torch.tensor([[0.0, 100.0, 200.0, 300.0]])
    )
    shifted = decoder(
        latent, target_time_ps=torch.tensor([[50.0, 150.0, 250.0, 350.0]])
    )
    assert first.x_coarse.shape == (4, 4, 3)
    assert first.x_hat.shape == (4, 4, 3)
    assert first.h.shape == (4, 4, 8)
    assert first.v.shape == (4, 4, 3, 6)
    assert not torch.allclose(first.h, shifted.h)
    assert torch.isfinite(first.x_hat).all()

    parameters = inspect.signature(JointMultiFrameDecoder.forward).parameters
    assert "target_time_ps" in parameters
    assert "target_x" not in parameters
    assert "target_coordinates" not in parameters


def test_decoder_is_se3_equivariant_and_has_gradients():
    latent = _latent()
    decoder = JointMultiFrameDecoder(8, 6, temporal_layers=1, num_heads=2)
    decoder.eval()
    target_time = torch.tensor([[0.0, 100.0, 200.0, 300.0]])
    base = decoder(latent, target_time_ps=target_time)

    q, _ = torch.linalg.qr(torch.randn(3, 3))
    if torch.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    translation = torch.tensor([1.2, -0.7, 0.4])
    rotated = CodecLatent(
        z_h=latent.z_h,
        z_v=torch.einsum("ij,lnjc->lnic", q, latent.z_v),
        latent_mask=latent.latent_mask,
        latent_time_ps=latent.latent_time_ps,
        x_anchor=torch.einsum("ij,nj->ni", q, latent.x_anchor) + translation,
        abid=latent.abid,
    )
    transformed = decoder(rotated, target_time_ps=target_time)
    expected_x = torch.einsum("ij,tnj->tni", q, base.x_hat) + translation
    expected_v = torch.einsum("ij,tnjc->tnic", q, base.v)
    assert torch.allclose(base.h, transformed.h, atol=3e-5, rtol=3e-5)
    assert torch.allclose(expected_v, transformed.v, atol=3e-5, rtol=3e-5)
    assert torch.allclose(expected_x, transformed.x_hat, atol=3e-5, rtol=3e-5)

    latent.z_h.requires_grad_()
    latent.z_v.requires_grad_()
    loss = decoder(latent, target_time_ps=target_time).x_hat.square().mean()
    loss.backward()
    assert latent.z_h.grad is not None and latent.z_v.grad is not None
    assert torch.isfinite(latent.z_h.grad).all()
    assert torch.isfinite(latent.z_v.grad).all()


class _CountingSpatial(nn.Module):
    def __init__(self, hidden_channels: int):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.calls = 0
        self.node_count = 0

    def forward(self, **kwargs):
        self.calls += 1
        pos = kwargs["pos"]
        self.node_count = int(pos.shape[0])
        h = torch.nn.functional.one_hot(
            kwargs["z"], num_classes=self.hidden_channels
        ).to(dtype=pos.dtype)
        v = pos.unsqueeze(-1).expand(-1, -1, self.hidden_channels)
        return h, v


def test_spatial_refiner_is_one_shared_batched_call():
    latent = _latent(atoms=4)
    latent.topology = _topology(4)
    fake = _CountingSpatial(hidden_channels=8)
    refiner = LatentConditionedSpatialRefiner(
        8, 6, hidden_channels=8, spatial_encoder=fake
    )
    decoder = JointMultiFrameDecoder(
        8, 6, temporal_layers=0, num_heads=2, spatial_refiner=refiner
    )
    output = decoder(latent, target_time_ps=torch.tensor([[0.0, 100.0, 200.0]]))
    assert output.x_hat.shape == (3, 4, 3)
    assert refiner.calls == 1
    assert fake.calls == 1
    assert fake.node_count == 3 * 4


def test_real_torchmd_refiner_small_graph():
    latent = _latent(atoms=3)
    latent.topology = _topology(3)
    refiner = LatentConditionedSpatialRefiner(
        8,
        6,
        hidden_channels=8,
        num_layers=1,
        num_rbf=4,
        num_heads=2,
        max_z=16,
        max_b=8,
        max_num_neighbors=8,
    )
    decoder = JointMultiFrameDecoder(
        8, 6, temporal_layers=0, num_heads=2, spatial_refiner=refiner
    )
    output = decoder(latent, target_time_ps=torch.tensor([[0.0, 100.0, 200.0]]))
    assert output.x_hat.shape == (3, 3, 3)
    assert output.v.shape == (3, 3, 3, 8)
    assert refiner.calls == 1
    assert torch.isfinite(output.x_hat).all()


def test_two_clip_tiny_overfit_without_coordinate_input():
    torch.set_num_threads(1)
    latent = _latent(atoms=4, clips=2)
    target_time = torch.tensor(
        [[0.0, 100.0, 200.0, 300.0], [0.0, 100.0, 200.0, 300.0]]
    )
    teacher = JointMultiFrameDecoder(8, 6, temporal_layers=0, num_heads=2)
    with torch.no_grad():
        target = teacher(latent, target_time_ps=target_time).x_hat

    student = JointMultiFrameDecoder(8, 6, temporal_layers=0, num_heads=2)
    optimizer = torch.optim.Adam(student.parameters(), lr=5e-2)
    initial = None
    for _ in range(80):
        optimizer.zero_grad(set_to_none=True)
        prediction = student(latent, target_time_ps=target_time).x_hat
        loss = (prediction - target).square().mean()
        if initial is None:
            initial = float(loss.detach())
        loss.backward()
        optimizer.step()
    final = float(
        (student(latent, target_time_ps=target_time).x_hat - target)
        .square()
        .mean()
        .detach()
    )
    assert initial is not None and final < initial * 0.1
    assert torch.isfinite(torch.tensor(final))
