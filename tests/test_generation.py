"""Observed-only generation and short autoregressive rollout gates."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from molvid.codec.model import TrajectoryCodec
from molvid.data.batch import collate_clip_records
from molvid.dit.model import MolecularDiT
from molvid.generation import rollout, sample_clip
from molvid.latent.adapter import StateDetailLatentAdapter
from molvid.latent.statistics import LatentStatistics
from test_migration import _record


def _fixture():
    assert torch.cuda.is_available(), "P4c generation gate requires CUDA"
    record = _record()
    t = np.arange(16, dtype=np.float32)[:, None, None]
    base = np.asarray(record["x"][0:1])
    record["x"] = base + 0.03 * np.sin(0.4 * t + np.arange(4, dtype=np.float32)[None, :, None])
    record["bpos"] = record["x"].copy()
    record["time_ps"] = np.arange(16, dtype=np.float32) * 100
    record["delta_time_ps"] = np.full(15, 100, dtype=np.float32)
    batch = collate_clip_records([record])
    torch.manual_seed(97)
    codec = TrajectoryCodec(
        hidden_channels=8, spatial_layers=1, temporal_codec_mode="ratio4_state_detail",
        num_rbf=8, num_heads=2, coordinate_stem="centered_vector",
    ).cuda().eval()
    adapter = StateDetailLatentAdapter(codec_width=8, scalar_width=8, vector_width=4, ratio=4)
    model = MolecularDiT(
        adapter=adapter, scalar_width=8, vector_width=4, depth=1,
        heads=2, ffn_multiplier=2, execution_backend="factorized_v2",
    ).cuda().eval()
    zeros = torch.zeros(8, device="cuda")
    ones = torch.ones(8, device="cuda")
    statistics = LatentStatistics(
        ratio=4, mode="ratio4_state_detail", width=8,
        state_h_mean=zeros, state_h_std=ones, detail_h_mean=zeros,
        detail_h_std=ones, state_v_rms=ones, detail_v_rms=ones,
        provenance={"test": "generation"},
    )
    return batch, codec, model, adapter, statistics


@pytest.mark.parametrize("history", [4, 8])
def test_sample_clip_is_future_isolated_and_clamps_observed_coordinates(history):
    prior = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        batch, codec, model, adapter, statistics = _fixture()
        changed = replace(batch, x=batch.x.clone(), bpos=batch.bpos.clone())
        changed.x[history:] += 1000
        changed.bpos[history:] -= 500
        args = dict(
            prefix_coordinates=batch.x[:history], history_frames=history,
            steps=8, seed=53, codec_hash="codec", data_hash="data",
        )
        first, first_info = sample_clip(codec, model, adapter, statistics, template=batch, **args)
        second, second_info = sample_clip(codec, model, adapter, statistics, template=changed, **args)
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        assert first_info == second_info
        assert first_info["observed_clamp_exact"]
        assert first_info["coordinate_prefix_clamp_exact"]
        torch.testing.assert_close(first[:history], batch.x[:history].cuda(), rtol=0, atol=0)
        assert torch.isfinite(first).all()
        assert all(parameter.grad is None for parameter in codec.parameters())
        assert all(parameter.grad is None for parameter in model.parameters())
    finally:
        torch.use_deterministic_algorithms(prior)


def test_rollout_uses_own_generated_eight_frame_prefix():
    prior = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        batch, codec, model, adapter, statistics = _fixture()
        args = dict(
            template=batch, steps=8, codec_hash="codec", data_hash="data",
        )
        result, metadata = rollout(
            codec, model, adapter, statistics,
            prefix_coordinates=batch.x[:8], seeds=(91, 92), **args,
        )
        assert result.shape == (24, 4, 3)
        assert [item["segment"] for item in metadata] == [0, 1]
        first, _ = sample_clip(
            codec, model, adapter, statistics,
            prefix_coordinates=batch.x[:8], history_frames=8, seed=91, **args,
        )
        second, _ = sample_clip(
            codec, model, adapter, statistics,
            prefix_coordinates=first[8:].cpu(), history_frames=8, seed=92, **args,
        )
        torch.testing.assert_close(result[:16], first, rtol=0, atol=0)
        torch.testing.assert_close(result[16:], second[8:], rtol=0, atol=0)
    finally:
        torch.use_deterministic_algorithms(prior)
