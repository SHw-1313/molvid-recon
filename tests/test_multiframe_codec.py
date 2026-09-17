from __future__ import annotations

import torch
from torch import nn

from data.clip_dataset import collate_clip_records
from module.multiframe_codec import PVBFrameEncoder


def _record(sample_id: str, frames: int = 1, atoms: int = 4):
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


class _FakeSpatial(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, *, z, b, pos, batch, edge_index, edge_weight_t, edge_vec_t, bond_type):
        self.calls += 1
        h = torch.nn.functional.one_hot(z, num_classes=8).float()
        v = edge_vec_t.new_zeros(pos.shape[0], 3, 2)
        return h, v


def test_frame_graph_isolation_and_single_backbone_call():
    records = [_record("a", frames=4), _record("b", frames=4)]
    batch = collate_clip_records(records)
    fake = _FakeSpatial()
    encoder = PVBFrameEncoder(spatial_encoder=fake, neighbor_backend="dense_test", max_num_neighbors=4)
    encoder.prepare_batch(batch)
    output = encoder(batch)
    graph = output.graph
    assert fake.calls == 1
    assert output.h.shape == (4, 8, 8)
    assert output.v.shape == (4, 8, 3, 2)
    assert graph.batch.shape == (32,)
    assert graph.bond_index.shape == (2, 4 * 4)
    assert torch.all(graph.batch[graph.edge_index[0]] == graph.batch[graph.edge_index[1]])

