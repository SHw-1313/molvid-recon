from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from module.state_detail_codec_v2 import StateDetailCodecV2, compute_masked_centroid_origin
from module.state_detail_latent_adapter import (
    LatentStatistics,
    StateDetailLatentAdapter,
    build_observation_condition,
    frame_prefix_observation_mask,
)
from dit_test_utils import make_batch, make_latent


@pytest.mark.parametrize(("ratio", "tokens"), [(2, 8), (4, 4)])
def test_r2_r4_shapes_pack_on_one_sequence_axis(ratio: int, tokens: int) -> None:
    adapter = StateDetailLatentAdapter(codec_width=4, scalar_width=8, vector_width=4, ratio=ratio)
    batch = adapter.pack(make_latent(ratio, width=4))
    assert batch.state_h.shape == (tokens, 5, 4)
    assert batch.detail_h.shape == batch.state_h.shape
    assert batch.state_v.shape == (tokens, 5, 3, 4)
    assert batch.detail_v.shape == batch.state_v.shape
    assert batch.contract()["contains_raw_detail"] is False
    assert batch.contract()["contains_target_coordinates"] is False
    assert "raw_detail_h" not in batch.__dict__
    assert batch.tokens == tokens


def test_generated_latent_has_no_raw_detail_and_codec_contract() -> None:
    adapter = StateDetailLatentAdapter(codec_width=4, scalar_width=8, vector_width=4, ratio=2)
    batch = adapter.pack(make_latent(2, width=4))
    h, v = adapter.project_inputs(batch)
    generated = adapter.make_generated_latent(batch, adapter.project_outputs(h, v))
    assert generated.raw_detail_h is None
    assert generated.raw_detail_v is None
    assert generated.topology.contract()["contains_target_coordinates"] is False
    assert adapter.contract()["vector_maps"] == "bias_free_channel_only"


def test_mask_aware_statistics_roundtrip_and_vector_has_no_mean() -> None:
    batch = make_batch(2, width=4)
    batch.detail_valid[-1, -1] = False
    masked = batch.zero_invalid()
    stats = LatentStatistics.fit(
        [masked, masked.with_fields(masked.fields.map(lambda value: value + 0.25))],
        ratio=2,
        provenance={"split": "train", "manifest_hash": "fixture"},
    )
    roundtrip = stats.inverse_normalize(stats.normalize(masked))
    for name in ("state_h", "detail_h", "state_v", "detail_v"):
        assert torch.allclose(getattr(roundtrip, name), getattr(masked, name), atol=2e-5, rtol=2e-5)
    assert torch.allclose(stats.state_v_rms, stats.state_v_rms.abs())
    assert stats.contract()["vector_mean_subtraction"] is False
    assert stats.provenance["split"] == "train"
    assert stats.hash == LatentStatistics(
        ratio=2,
        mode="ratio2_state_detail",
        width=4,
        state_h_mean=stats.state_h_mean,
        state_h_std=stats.state_h_std,
        detail_h_mean=stats.detail_h_mean,
        detail_h_std=stats.detail_h_std,
        state_v_rms=stats.state_v_rms,
        detail_v_rms=stats.detail_v_rms,
        provenance={"split": "train", "manifest_hash": "fixture"},
    ).hash


def test_all_four_latent_fields_must_be_finite() -> None:
    batch = make_batch(2, width=4)
    fields = batch.fields
    bad = fields.detail_h.clone()
    bad[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="all latent fields"):
        batch.with_fields(type(fields)(fields.state_h, bad, fields.state_v, fields.detail_v))


@pytest.mark.parametrize(("history", "ratio", "observed_tokens"), [
    (0, 2, 0), (4, 2, 2), (8, 2, 4),
    (0, 4, 0), (4, 4, 1), (8, 4, 2),
])
def test_observation_prefixes_are_block_aligned(history: int, ratio: int, observed_tokens: int) -> None:
    batch = make_batch(ratio, width=4)
    coordinates = torch.arange(16 * batch.num_atoms * 3, dtype=torch.float32).reshape(16, batch.num_atoms, 3)
    condition = build_observation_condition(
        batch, history_frames=history, coordinates=coordinates, frame_mask=torch.ones(2, 16, dtype=torch.bool)
    )
    assert condition.latent_observation_mask.sum(dim=1).tolist() == [observed_tokens, observed_tokens]
    if history:
        expected = coordinates[0].reshape(batch.num_atoms, 3)
        for sample in range(batch.batch_size):
            assert torch.allclose(condition.sample_origin[sample], expected[batch.abid == sample].mean(dim=0))
    else:
        assert torch.equal(condition.sample_origin, torch.zeros_like(condition.sample_origin))


def test_partial_block_is_actionable_and_future_mutation_isolated() -> None:
    batch = make_batch(4, width=4)
    partial = torch.zeros(2, 16, dtype=torch.bool)
    partial[:, :3] = True
    with pytest.raises(ValueError, match="partial observed codec block"):
        build_observation_condition(batch, history_frames=0, observed_frame_mask=partial)
    coordinates = torch.randn(16, batch.num_atoms, 3)
    condition = build_observation_condition(
        batch, history_frames=4, coordinates=coordinates, frame_mask=torch.ones(2, 16, dtype=torch.bool)
    )
    before = condition.sample_origin.clone()
    coordinates[4:] += 10000.0
    assert torch.equal(condition.sample_origin, before)


def test_observation_origin_matches_codec_loss_mask_and_ignores_masked_future_atoms() -> None:
    loss_mask = torch.tensor([True, False, True, True, True])
    batch = make_batch(2, width=4, loss_mask=loss_mask)
    coordinates = torch.zeros(16, batch.num_atoms, 3)
    coordinates[0, 0] = torch.tensor([1.0, 2.0, 3.0])
    coordinates[0, 1] = torch.tensor([1.0e6, -2.0e6, 3.0e6])
    coordinates[0, 2] = torch.tensor([4.0, 5.0, 6.0])
    coordinates[0, 3] = torch.tensor([7.0, 8.0, 9.0])
    coordinates[0, 4] = torch.tensor([10.0, 11.0, 12.0])
    condition = build_observation_condition(
        batch,
        history_frames=4,
        coordinates=coordinates,
        frame_mask=torch.ones(2, 16, dtype=torch.bool),
    )
    expected = compute_masked_centroid_origin(
        coordinates,
        frame_mask=torch.ones(2, 16, dtype=torch.bool),
        abid=batch.abid,
        atom_mask=loss_mask,
    )
    assert torch.equal(condition.sample_origin, expected)
    before = condition.sample_origin.clone()
    coordinates[0, 1] += 1.0e9
    coordinates[4:] -= 1.0e9
    assert torch.equal(condition.sample_origin, before)
    zero_history = build_observation_condition(
        batch, history_frames=0, coordinates=None, frame_mask=torch.ones(2, 16, dtype=torch.bool)
    )
    assert torch.equal(zero_history.sample_origin, torch.zeros_like(zero_history.sample_origin))


def test_h_positive_origin_requires_loss_mask_when_not_carried_by_batch() -> None:
    batch = make_batch(2, width=4)
    batch.loss_mask = None
    with pytest.raises(ValueError, match="loss_mask"):
        build_observation_condition(
            batch,
            history_frames=4,
            coordinates=torch.zeros(16, batch.num_atoms, 3),
            frame_mask=torch.ones(2, 16, dtype=torch.bool),
        )


def test_clamped_observed_latent_decodes_in_the_codec_origin_gauge() -> None:
    loss_mask = torch.tensor([True, False, True, True, True])
    latent = make_latent(2, width=4)
    adapter = StateDetailLatentAdapter(codec_width=4, scalar_width=8, vector_width=4, ratio=2)
    batch = adapter.pack(latent, loss_mask=loss_mask)
    coordinates = torch.randn(16, batch.num_atoms, 3)
    condition = build_observation_condition(
        batch,
        history_frames=4,
        coordinates=coordinates,
        frame_mask=torch.ones(2, 16, dtype=torch.bool),
    )
    observed = torch.zeros_like(batch.token_mask)
    observed[:, :2] = True
    clamped_batch = batch.with_observation(
        observed, sample_origin=condition.sample_origin
    )
    generated = adapter.make_generated_latent(clamped_batch, clamped_batch.fields)
    codec_gauge_latent = replace(latent, sample_origin=condition.sample_origin)
    codec = StateDetailCodecV2(4, mode="ratio2_state_detail")
    reference = codec.decode(codec_gauge_latent).x_hat
    decoded = codec.decode(generated).x_hat
    assert torch.equal(decoded, reference)


def test_history_and_frame_mask_validation() -> None:
    frame_mask = torch.ones(1, 16, dtype=torch.bool)
    assert frame_prefix_observation_mask(frame_mask, 0).sum() == 0
    with pytest.raises(ValueError, match="H=0, H=4, or H=8"):
        frame_prefix_observation_mask(frame_mask, 2)
    bad = frame_mask.clone()
    bad[0, 4] = False
    bad[0, 5] = True
    with pytest.raises(ValueError, match="valid prefix"):
        frame_prefix_observation_mask(bad, 4)
