from __future__ import annotations

from dataclasses import replace

import torch

from data.clip_dataset import collate_clip_records
from module.multiframe_codec import PVBFrameEncoder


def _record(sample_id: str, frames: int, atoms: int = 4):
    x = torch.arange(frames * atoms * 3, dtype=torch.float32).reshape(frames, atoms, 3)
    return {
        "schema_version": "pvb.clip.v1",
        "sample_id": sample_id,
        "task": "static" if frames == 1 else "trajectory",
        "time_bucket_id": "static" if frames == 1 else "dt_100ps",
        "time_ps": torch.arange(frames, dtype=torch.float32) * (100.0 if frames > 1 else 1.0),
        "delta_time_ps": torch.full((max(0, frames - 1),), 100.0),
        "x": x,
        "bpos": x.clone(),
        "atype": torch.arange(atoms, dtype=torch.long),
        "btype": torch.arange(atoms, dtype=torch.long),
        "block_id": torch.arange(atoms, dtype=torch.long),
        "component_id": torch.zeros(atoms, dtype=torch.long),
        "atom_source_index": torch.arange(atoms, dtype=torch.long),
        "atom_identity": [f"{sample_id}:a{i}" for i in range(atoms)],
        "edge_mask": torch.zeros(atoms, dtype=torch.long),
        "loss_mask": torch.ones(atoms, dtype=torch.bool),
        "align_mask": torch.tensor([True] + [False] * (atoms - 1)),
        "bond_index": torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
    }


def _encoder() -> PVBFrameEncoder:
    return PVBFrameEncoder(
        hidden_channels=16,
        num_layers=1,
        num_rbf=8,
        num_heads=4,
        max_num_neighbors=4,
        neighbor_backend="dense_test",
    ).eval()


def test_t1_and_t16_use_time_major_shapes_with_real_torchmd():
    torch.manual_seed(4)
    encoder = _encoder()
    for frames in (1, 16):
        batch = collate_clip_records([_record("x", frames=frames)])
        encoder.prepare_batch(batch)
        output = encoder(batch)
        assert output.h.shape[:2] == (frames, 4)
        assert output.v.shape[:3] == (frames, 4, 3)
        assert torch.isfinite(output.h).all()
        assert torch.isfinite(output.v).all()


def test_scalar_translation_invariance_and_vector_rotation_equivariance():
    torch.manual_seed(7)
    encoder = _encoder()
    base = collate_clip_records([_record("x", frames=4)])
    coordinates = torch.randn_like(base.x)
    base = replace(base, x=coordinates, bpos=coordinates.clone())

    translation = torch.tensor([3.0, -2.0, 1.5])
    translated = replace(
        base,
        x=base.x + translation,
        bpos=base.bpos + translation,
    )
    encoder.prepare_batch(base)
    output = encoder(base)
    encoder.prepare_batch(translated)
    translated_output = encoder(translated)
    assert torch.allclose(output.h, translated_output.h, atol=2e-5, rtol=2e-5)
    assert torch.allclose(output.v, translated_output.v, atol=2e-5, rtol=2e-5)

    rotation = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    rotated = replace(
        base,
        x=torch.einsum("tnj,ji->tni", base.x, rotation),
        bpos=torch.einsum("tnj,ji->tni", base.bpos, rotation),
    )
    encoder.prepare_batch(rotated)
    rotated_output = encoder(rotated)
    expected_vectors = torch.einsum("tnjc,ji->tnic", output.v, rotation)
    assert torch.allclose(output.h, rotated_output.h, atol=3e-4, rtol=3e-4)
    assert torch.allclose(expected_vectors, rotated_output.v, atol=3e-4, rtol=3e-4)


def test_checkpoint_report_lists_matched_and_unexpected_keys(tmp_path):
    torch.manual_seed(9)
    source = _encoder()
    state = dict(source.spatial_encoder.state_dict())
    state["not_a_model_key"] = torch.ones(1)
    checkpoint = tmp_path / "encoder.pt"
    torch.save({"state_dict": state}, checkpoint)
    target = PVBFrameEncoder(
        hidden_channels=16,
        num_layers=1,
        num_rbf=8,
        num_heads=4,
        max_num_neighbors=4,
        neighbor_backend="dense_test",
        checkpoint_path=checkpoint,
    )
    report = target.checkpoint_report
    assert report is not None
    assert len(report.matched) == len(source.spatial_encoder.state_dict())
    assert report.unexpected == ("not_a_model_key",)
    assert not report.missing

