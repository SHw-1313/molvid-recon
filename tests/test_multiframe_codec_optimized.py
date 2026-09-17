from __future__ import annotations

import torch

from data.clip_dataset import collate_clip_records
from module.multiframe_codec import PVBFrameEncoder


def test_explicit_test_neighbor_backend_keeps_frame_graphs_isolated():
    atoms = 5
    frames = 2
    coordinates = torch.randn(frames, atoms, 3)
    record = {
        "schema_version": "pvb.clip.v1",
        "sample_id": "optimized",
        "task": "trajectory",
        "time_bucket_id": "dt_100ps",
        "time_ps": torch.tensor([0.0, 100.0]),
        "delta_time_ps": torch.tensor([100.0]),
        "x": coordinates,
        "bpos": coordinates.clone(),
        "atype": torch.arange(atoms),
        "btype": torch.arange(atoms),
        "block_id": torch.arange(atoms),
        "component_id": torch.zeros(atoms, dtype=torch.long),
        "atom_source_index": torch.arange(atoms),
        "atom_identity": [f"optimized:a{i}" for i in range(atoms)],
        "edge_mask": torch.zeros(atoms, dtype=torch.long),
        "loss_mask": torch.ones(atoms, dtype=torch.bool),
        "align_mask": torch.tensor([True, True, False, False, False]),
        "bond_index": torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
    }
    batch = collate_clip_records([record])
    encoder = PVBFrameEncoder(
        hidden_channels=8,
        num_layers=1,
        num_rbf=4,
        num_heads=2,
        max_num_neighbors=4,
        neighbor_backend="dense_test",
    )
    encoder.prepare_batch(batch)
    output = encoder(batch)
    graph = output.graph
    assert output.h.shape == (frames, atoms, 8)
    assert output.v.shape == (frames, atoms, 3, 8)
    assert torch.all(graph.batch[graph.edge_index[0]] == graph.batch[graph.edge_index[1]])

