from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from data.clip_dataset import collate_clip_records
from module.multiframe_codec import PVBFrameEncoder, pack_frame_nodes
from module.visnet import MolViSNetEncoder, SpatialEncoderOutput, make_spatial_backbone
from trainer.codec_trainer import PVBCodecModel


def _record(sample_id: str, frames: int = 4, atoms: int = 5, bonds=None):
    base = torch.arange(atoms, dtype=torch.float32)
    x = torch.zeros(frames, atoms, 3)
    x[:, :, 0] = base
    x[:, :, 1] = 0.2 * (base % 2)
    x[:, :, 2] = 0.1 * (base % 3)
    if frames > 1:
        x[:, :, 1] += torch.arange(frames, dtype=torch.float32).view(-1, 1) * 0.01
    if bonds is None:
        bonds = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    return {
        "schema_version": "pvb.clip.v1",
        "sample_id": sample_id,
        "task": "trajectory" if frames > 1 else "static",
        "time_bucket_id": "dt_100ps" if frames > 1 else "static",
        "time_ps": torch.arange(frames, dtype=torch.float32) * 100.0,
        "delta_time_ps": torch.full((max(0, frames - 1),), 100.0),
        "x": x,
        "bpos": x.clone(),
        "atype": torch.tensor([1, 6, 7, 8, 16], dtype=torch.long)[:atoms],
        "btype": torch.tensor([20, 20, 21, 21, 22], dtype=torch.long)[:atoms],
        "block_id": torch.arange(atoms, dtype=torch.long),
        "component_id": torch.zeros(atoms, dtype=torch.long),
        "atom_source_index": torch.arange(atoms, dtype=torch.long),
        "atom_identity": [f"{sample_id}:a{i}" for i in range(atoms)],
        "edge_mask": torch.zeros(atoms, dtype=torch.long),
        "loss_mask": torch.ones(atoms, dtype=torch.bool),
        "align_mask": torch.tensor([True] + [False] * (atoms - 1)),
        "bond_index": bonds,
    }


def _batch(frames: int = 4, bonds=None):
    return collate_clip_records(
        [_record("a", frames=frames, bonds=bonds), _record("b", frames=frames, bonds=bonds)]
    )


def _encoder(backbone: str) -> PVBFrameEncoder:
    return PVBFrameEncoder(
        hidden_channels=8,
        num_layers=2,
        num_rbf=6,
        num_heads=2,
        max_num_neighbors=8,
        neighbor_backend="dense_test",
        spatial_backbone=backbone,
    ).eval()


def test_factory_and_lmax_contract():
    assert isinstance(
        make_spatial_backbone(
            "visnet_radius",
            hidden_channels=8,
            num_layers=1,
            num_rbf=4,
            num_heads=2,
            neighbor_backend="dense_test",
        ),
        MolViSNetEncoder,
    )
    assert isinstance(make_spatial_backbone("torchmd_et", hidden_channels=8, num_layers=1, num_rbf=4, num_heads=2), torch.nn.Module)
    with pytest.raises(ValueError, match="lmax=1"):
        MolViSNetEncoder(
            hidden_channels=8,
            num_layers=1,
            num_rbf=4,
            num_heads=2,
            lmax=2,
            neighbor_backend="dense_test",
        )
    with pytest.raises(ValueError, match="lmax=1"):
        PVBFrameEncoder(
            hidden_channels=8,
            num_layers=1,
            num_rbf=4,
            num_heads=2,
            lmax=2,
            neighbor_backend="dense_test",
            spatial_backbone="visnet_radius",
        )


@pytest.mark.parametrize("backbone", ["visnet_radius", "visnet_bonded"])
@pytest.mark.parametrize("frames", [1, 16])
def test_visnet_shapes_and_frame_sample_isolation(backbone: str, frames: int):
    encoder = _encoder(backbone)
    batch = _batch(frames=frames)
    encoder.prepare_batch(batch)
    output = encoder(batch)
    assert output.h.shape == (frames, 10, 8)
    assert output.v.shape == (frames, 10, 3, 8)
    assert output.spatial_backbone == backbone
    assert output.graph_mode == (
        "native_radius" if backbone == "visnet_radius" else "external"
    )
    graph = output.graph
    assert torch.all(graph.graph_id[graph.edge_index[0]] == graph.graph_id[graph.edge_index[1]])
    if backbone == "visnet_bonded":
        assert torch.any(graph.bond_type == 1)
    else:
        assert graph.bond_index.numel() == 0


def test_radius_does_not_call_external_builder_or_prepare_topology(monkeypatch):
    encoder = _encoder("visnet_radius")
    batch = _batch(frames=4)
    calls = {"native": 0}
    native = encoder.spatial_encoder.native_neighbor_builder

    class CountingNeighbor:
        backend_used = native.backend_used

        def __call__(self, *args, **kwargs):
            calls["native"] += 1
            return native(*args, **kwargs)

    encoder.spatial_encoder.native_neighbor_builder = CountingNeighbor()

    def fail_external(*args, **kwargs):
        raise AssertionError("visnet_radius called the external graph builder")

    monkeypatch.setattr(encoder, "build_external_graph", fail_external)
    monkeypatch.setattr(
        encoder,
        "_register_topology_batch",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("visnet_radius prepared topology")
        ),
    )
    encoder.prepare_batch(batch)
    output = encoder(batch)
    assert isinstance(output.spatial_backbone, str)
    assert calls["native"] == 1


def test_bond_type_changes_bonded_output():
    batch_a = _batch(frames=4, bonds=torch.tensor([[0, 1], [1, 0]], dtype=torch.long))
    batch_b = _batch(frames=4, bonds=torch.tensor([[0, 2], [2, 0]], dtype=torch.long))
    first = _encoder("visnet_bonded")
    second = _encoder("visnet_bonded")
    second.load_state_dict(first.state_dict())
    first.prepare_batch(batch_a)
    second.prepare_batch(batch_b)
    out_a = first(batch_a)
    out_b = second(batch_b)
    assert not torch.allclose(out_a.h, out_b.h)
    assert not torch.allclose(out_a.v, out_b.v)


@pytest.mark.parametrize("backbone", ["visnet_radius", "visnet_bonded"])
def test_visnet_se3_and_coordinate_parameter_gradients(backbone: str):
    torch.manual_seed(13)
    encoder = _encoder(backbone)
    base = _batch(frames=4)
    coordinates = torch.randn_like(base.x)
    base = replace(base, x=coordinates, bpos=coordinates.clone())
    translation = torch.tensor([2.0, -1.0, 0.5])
    translated = replace(base, x=base.x + translation, bpos=base.bpos + translation)
    encoder.prepare_batch(base)
    out = encoder(base)
    encoder.prepare_batch(translated)
    translated_out = encoder(translated)
    assert torch.allclose(out.h, translated_out.h, atol=3e-5, rtol=3e-5)
    assert torch.allclose(out.v, translated_out.v, atol=3e-5, rtol=3e-5)

    rotation = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    rotated = replace(
        base,
        x=torch.einsum("tnj,ji->tni", base.x, rotation),
        bpos=torch.einsum("tnj,ji->tni", base.bpos, rotation),
    )
    encoder.prepare_batch(rotated)
    rotated_out = encoder(rotated)
    expected_v = torch.einsum("tnjc,ji->tnic", out.v, rotation)
    assert torch.allclose(out.h, rotated_out.h, atol=5e-4, rtol=5e-4)
    assert torch.allclose(expected_v, rotated_out.v, atol=5e-4, rtol=5e-4)

    train_batch = replace(
        base,
        x=base.x.detach().clone().requires_grad_(True),
        bpos=base.bpos.detach().clone(),
    )
    encoder.prepare_batch(train_batch)
    train_out = encoder(train_batch)
    loss = train_out.h.square().mean() + train_out.v.square().mean()
    loss.backward()
    assert train_batch.x.grad is not None
    assert torch.isfinite(train_batch.x.grad).all()
    assert train_batch.x.grad.abs().sum() > 0
    parameter_gradients = [
        parameter.grad
        for parameter in encoder.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert parameter_gradients
    assert all(torch.isfinite(gradient).all() for gradient in parameter_gradients)
    assert any(gradient.abs().sum() > 0 for gradient in parameter_gradients)


def test_codec_contract_records_backbone_and_graph_mode():
    model = PVBCodecModel(
        hidden_channels=8,
        spatial_layers=1,
        spatial_backbone="visnet_radius",
        temporal_layers=1,
        temporal_ratio=1,
        num_rbf=4,
        num_heads=2,
        max_num_neighbors=8,
        neighbor_backend="dense_test",
    )
    contract = model.model_contract()
    assert contract["constructor"]["spatial_backbone"] == "visnet_radius"
    assert contract["architecture"]["spatial"]["lmax"] == 1
    assert contract["graph"]["graph_mode"] == "native_radius"
    restored = PVBCodecModel.from_model_contract(contract)
    assert restored.frame_encoder.spatial_backbone == "visnet_radius"
    assert restored.frame_encoder.graph_mode == "native_radius"


@pytest.mark.parametrize("backbone", ["visnet_radius", "visnet_bonded"])
def test_codec_forward_with_ratio_one_temporal_path(backbone: str):
    model = PVBCodecModel(
        hidden_channels=8,
        spatial_layers=1,
        spatial_backbone=backbone,
        temporal_layers=1,
        temporal_ratio=1,
        num_rbf=4,
        num_heads=2,
        max_num_neighbors=8,
        neighbor_backend="dense_test",
    )
    batch = _batch(frames=4)
    model.prepare_batch(batch)
    output = model(batch)
    assert output.x_hat.shape == (4, 10, 3)
    assert torch.isfinite(output.x_hat).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA runtime unavailable")
@pytest.mark.parametrize("backbone", ["visnet_radius", "visnet_bonded"])
def test_visnet_cuda_radius_path(backbone: str):
    device = torch.device("cuda:0")
    encoder = PVBFrameEncoder(
        hidden_channels=8,
        num_layers=1,
        num_rbf=4,
        num_heads=2,
        max_num_neighbors=8,
        neighbor_backend="cuda_radius",
        spatial_backbone=backbone,
    ).to(device)
    batch = _batch(frames=4)
    encoder.prepare_batch(batch)
    output = encoder(batch.to(device))
    assert output.h.shape == (4, 10, 8)
    assert output.v.shape == (4, 10, 3, 8)
    assert torch.isfinite(output.h).all()
    assert torch.isfinite(output.v).all()
    loss = output.h.square().mean() + output.v.square().mean()
    loss.backward()
    torch.cuda.synchronize(device)
