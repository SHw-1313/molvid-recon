"""New flat-package codec and spatial numerical regression tests."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from data.clip_dataset import collate_clip_records as old_collate
from module.multiframe_codec import PVBFrameEncoder
from molvid.data.batch import collate_clip_records
from molvid.spatial.encoder import FrameEncoder


@pytest.fixture
def deterministic_cuda():
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    yield
    torch.use_deterministic_algorithms(previous)


def _record() -> dict:
    frames, atoms = 4, 4
    t = np.arange(frames, dtype=np.float32)[:, None, None]
    x = np.array(
        [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]],
        dtype=np.float32,
    ) + 0.03 * np.sin(0.4 * t + np.arange(atoms, dtype=np.float32)[None, :, None])
    return {
        "schema_version": "pvb.clip.v1",
        "sample_id": "spatial-parity",
        "task": "trajectory",
        "time_bucket_id": "dt_100ps",
        "time_ps": np.arange(frames, dtype=np.float32) * 100,
        "delta_time_ps": np.full(frames - 1, 100, dtype=np.float32),
        "x": x,
        "bpos": x.copy(),
        "atype": np.array([5, 5, 5, 5], dtype=np.int64),
        "btype": np.array([6, 6, 6, 6], dtype=np.int64),
        "block_id": np.arange(atoms, dtype=np.int64),
        "component_id": np.zeros(atoms, dtype=np.int64),
        "atom_source_index": np.arange(atoms, dtype=np.int64),
        "atom_identity": [f"a{i}" for i in range(atoms)],
        "edge_mask": np.zeros(atoms, dtype=np.int64),
        "loss_mask": np.ones(atoms, dtype=np.bool_),
        "align_mask": np.ones(atoms, dtype=np.bool_),
        "bond_index": np.array([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=np.int64),
    }


def test_frame_encoder_initialization_forward_and_gradient_parity(deterministic_cuda):
    assert torch.cuda.is_available(), "P2 numerical gate requires CUDA"
    old_cpu = old_collate([_record()])
    new_cpu = collate_clip_records([_record()])
    kwargs = dict(hidden_channels=8, num_layers=1, num_rbf=8, num_heads=2)
    torch.manual_seed(713)
    old = PVBFrameEncoder(**kwargs).cuda()
    old_rng = torch.get_rng_state().clone()
    torch.manual_seed(713)
    new = FrameEncoder(**kwargs).cuda()
    new_rng = torch.get_rng_state().clone()
    assert torch.equal(old_rng, new_rng)
    old_state = old.state_dict()
    new_state = new.state_dict()
    assert list(new_state) == [key.replace("spatial_encoder.", "backbone.") for key in old_state]
    for key, value in old_state.items():
        torch.testing.assert_close(value, new_state[key.replace("spatial_encoder.", "backbone.")], rtol=0, atol=0)
    old.prepare_batch(old_cpu)
    new.prepare_batch(new_cpu)
    old_batch = old_cpu.to("cuda")
    new_batch = new_cpu.to("cuda")
    old_batch.x.requires_grad_(True)
    new_batch.x.requires_grad_(True)
    old_result = old(old_batch)
    new_result = new(new_batch)
    for name in ("edge_index", "edge_weight", "edge_vec", "bond_type"):
        torch.testing.assert_close(getattr(old_result.graph, name), getattr(new_result.graph, name), rtol=0, atol=0)
    for name in ("h", "v"):
        torch.testing.assert_close(getattr(old_result, name), getattr(new_result, name), rtol=0, atol=0)
    old_result.h.square().mean().add(old_result.v.square().mean()).backward()
    new_result.h.square().mean().add(new_result.v.square().mean()).backward()
    torch.testing.assert_close(old_batch.x.grad, new_batch.x.grad, rtol=0, atol=0)
    for (old_key, old_param), (new_key, new_param) in zip(old.named_parameters(), new.named_parameters()):
        assert new_key == old_key.replace("spatial_encoder.", "backbone.")
        torch.testing.assert_close(old_param.grad, new_param.grad, rtol=0, atol=0)
