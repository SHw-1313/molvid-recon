from __future__ import annotations

from dataclasses import replace
import inspect

import pytest
import torch

from data.clip_dataset import collate_clip_records
from module.state_detail_codec_v2 import (
    CenteredCoordinateVectorStem,
    MatchedPoolingCodecV2,
    STATIC_TOPOLOGY_SCHEMA,
    StaticTopologyMetadata,
    StateDetailCodecV2,
    center_coordinates,
    haar_inverse,
    haar_lift,
)
from trainer.codec_trainer import (
    CodecTrainConfig,
    CodecTrainer,
    PVBCodecModel,
    TimeBucketSpec,
)


def _features(frames: int = 16, atoms: int = 3, channels: int = 5):
    torch.manual_seed(71)
    h = torch.randn(frames, atoms, channels)
    v = torch.randn(frames, atoms, 3, channels)
    return h, v


def _context(frames: int = 16, atoms: int = 3):
    frame_mask = torch.ones(2, frames, dtype=torch.bool)
    frame_mask[1, -1] = False
    abid = torch.tensor([0, 0, 1], dtype=torch.long)[:atoms]
    time_ps = torch.arange(frames, dtype=torch.float32).mul(100.0).repeat(2, 1)
    origin = torch.tensor([[1.0, -2.0, 0.5], [-1.0, 2.0, -0.5]])
    return frame_mask, abid, time_ps, origin


@pytest.mark.parametrize("ratio", [1, 2, 4])
def test_haar_roundtrip_and_capacity_masks(ratio: int):
    h, v = _features()
    frame_mask, abid, _time_ps, _origin = _context()
    if ratio == 4:
        frame_mask[1, -1] = True
    lifted_h = haar_lift(h, ratio, frame_mask=frame_mask, abid=abid)
    lifted_v = haar_lift(v, ratio, frame_mask=frame_mask, abid=abid)
    restored_h = haar_inverse(
        lifted_h.state,
        lifted_h.detail,
        ratio,
        block_frame_mask=lifted_h.block_frame_mask,
        detail_component_mask=lifted_h.detail_component_mask,
        abid=abid,
        output_frames=16,
    )
    restored_v = haar_inverse(
        lifted_v.state,
        lifted_v.detail,
        ratio,
        block_frame_mask=lifted_v.block_frame_mask,
        detail_component_mask=lifted_v.detail_component_mask,
        abid=abid,
        output_frames=16,
    )
    expected_h = h.clone()
    expected_v = v.clone()
    if ratio != 4:
        expected_h[-1, 2] = 0.0
        expected_v[-1, 2] = 0.0
    assert torch.allclose(restored_h, expected_h, rtol=1e-6, atol=1e-7)
    assert torch.allclose(restored_v, expected_v, rtol=1e-6, atol=1e-7)
    tokens = (16 + ratio - 1) // ratio
    detail_banks = 0 if ratio == 1 else 1
    assert lifted_h.state.shape[:2] == (tokens, 3)
    assert lifted_h.detail_component_mask.shape == (
        2,
        tokens,
        detail_banks if ratio != 4 else 3,
    )
    expected_capacity = {1: 16 * 5, 2: 16 * 5, 4: 8 * 5}[ratio]
    actual_capacity = tokens * 5 * (1 if ratio == 1 else 2)
    assert actual_capacity == expected_capacity


def test_haar_r4_order_and_partial_contract():
    values = torch.arange(4, dtype=torch.float32).view(4, 1, 1)
    lifted = haar_lift(values, 4)
    assert lifted.detail is not None
    expected = torch.tensor([2.0, 1.0 / 2**0.5, 1.0 / 2**0.5]).view(1, 1, 3, 1)
    assert torch.allclose(lifted.detail, expected, atol=1e-7)
    assert lifted.detail_component_mask.tolist() == [[[True, True, True]]]

    one_frame = torch.tensor([[True, False, False, False]])
    partial = haar_lift(values, 4, frame_mask=one_frame)
    assert not bool(partial.detail_valid.any())
    restored = haar_inverse(
        partial.state,
        partial.detail,
        4,
        block_frame_mask=partial.block_frame_mask,
        detail_component_mask=partial.detail_component_mask,
        output_frames=4,
    )
    assert torch.equal(restored[0], values[0])
    assert torch.equal(restored[1:], torch.zeros_like(restored[1:]))

    ambiguous = torch.tensor([[True, True, True, False]])
    with pytest.raises(ValueError, match="ambiguous two-pair partial"):
        haar_lift(values, 4, frame_mask=ambiguous)

    tampered_detail = torch.full_like(lifted.detail, 100.0)
    masked_restore = haar_inverse(
        lifted.state,
        tampered_detail,
        4,
        block_frame_mask=lifted.block_frame_mask,
        detail_component_mask=torch.zeros_like(lifted.detail_component_mask),
        output_frames=4,
    )
    baseline = haar_inverse(
        lifted.state,
        torch.zeros_like(lifted.detail),
        4,
        block_frame_mask=lifted.block_frame_mask,
        detail_component_mask=lifted.detail_component_mask,
        output_frames=4,
    )
    assert torch.equal(masked_restore, baseline)


@pytest.mark.parametrize("mode", [
    "ratio1_state_detail",
    "ratio2_state_detail",
    "ratio4_state_detail",
])
def test_zero_preserving_repeated_static_and_detail_zero(mode: str):
    h, v = _features()
    h[:] = h[0]
    v[:] = v[0]
    frame_mask = torch.ones(1, 16, dtype=torch.bool)
    abid = torch.zeros(3, dtype=torch.long)
    time_ps = torch.arange(16, dtype=torch.float32).view(1, 16) * 100.0
    codec = StateDetailCodecV2(5, mode=mode)
    origin = torch.zeros(1, 3)
    latent = codec.encode(
        h,
        v,
        time_ps=time_ps,
        frame_mask=frame_mask,
        abid=abid,
        sample_origin=origin,
    )
    output = codec.decode(latent)
    assert latent.raw_detail_h is None or torch.equal(
        latent.raw_detail_h, torch.zeros_like(latent.raw_detail_h)
    )
    assert latent.detail_h is None or torch.equal(
        latent.detail_h, torch.zeros_like(latent.detail_h)
    )
    assert output.decoded_detail_h is None or torch.equal(
        output.decoded_detail_h, torch.zeros_like(output.decoded_detail_h)
    )
    if mode == "ratio1_state_detail":
        assert torch.allclose(output.h[1:], output.h[:-1], atol=1e-6)
    else:
        assert torch.allclose(output.h, output.h[0:1].expand_as(output.h), atol=1e-6)
    assert torch.allclose(output.x_hat, output.x_hat[0:1], atol=1e-6)
    assert torch.allclose(codec.decode_detail_zero(latent).x_hat, output.x_hat, atol=1e-7)


def test_zero_detail_is_unchanged_by_clock_and_dynamic_detail_is_used():
    h, v = _features()
    h[:] = h[0]
    v[:] = v[0]
    codec = StateDetailCodecV2(5, mode="ratio4_state_detail")
    common = {
        "frame_mask": torch.ones(1, 16, dtype=torch.bool),
        "abid": torch.zeros(3, dtype=torch.long),
        "sample_origin": torch.zeros(1, 3),
    }
    latent = codec(
        h,
        v,
        time_ps=torch.arange(16, dtype=torch.float32).view(1, 16) * 100.0,
        **common,
    ).latent
    shifted_clock = replace(
        latent,
        frame_time_ps=latent.frame_time_ps + 12345.0,
        block_time_ps=latent.block_time_ps + 12345.0,
    )
    assert torch.equal(codec.decode(latent).x_hat, codec.decode(shifted_clock).x_hat)

    moving_h, moving_v = _features()
    moving = codec(
        moving_h,
        moving_v,
        time_ps=torch.arange(16, dtype=torch.float32).view(1, 16) * 100.0,
        **common,
    ).latent
    assert moving.detail_h is not None and moving.detail_v is not None
    assert moving.detail_h.abs().sum() > 0
    assert moving.detail_v.abs().sum() > 0


def test_static_t1_is_genuinely_one_frame_and_detail_invalid():
    h, v = _features(frames=1)
    codec = StateDetailCodecV2(5, mode="ratio4_state_detail")
    latent = codec.encode(
        h,
        v,
        time_ps=torch.tensor([[123.0]]),
        frame_mask=torch.ones(1, 1, dtype=torch.bool),
        abid=torch.zeros(3, dtype=torch.long),
        sample_origin=torch.zeros(1, 3),
    )
    assert latent.frames == 1
    assert latent.detail_valid.shape == (1, 1)
    assert not bool(latent.detail_valid.any())
    assert torch.equal(latent.frame_time_ps, torch.tensor([[123.0]]))
    assert latent.contract()["detail_valid_shape"] == [1, 1]


def test_two_frame_partial_r4_discards_the_incomplete_detail_bank_explicitly():
    h, v = _features(frames=2)
    codec = StateDetailCodecV2(5, mode="ratio4_state_detail")
    latent = codec.encode(
        h,
        v,
        time_ps=torch.tensor([[123.0, 223.0]]),
        frame_mask=torch.ones(1, 2, dtype=torch.bool),
        abid=torch.zeros(3, dtype=torch.long),
        sample_origin=torch.zeros(1, 3),
    )
    assert latent.detail_valid.shape == (1, 1)
    assert not bool(latent.detail_valid.any())
    assert torch.equal(latent.frame_time_ps, torch.tensor([[123.0, 223.0]]))
    assert torch.equal(latent.block_time_ps[0, 0, :2], torch.tensor([123.0, 223.0]))


def test_irregular_clock_is_separate_from_static_and_partial_block_policy():
    h, v = _features(frames=4)
    codec = StateDetailCodecV2(5, mode="ratio4_state_detail")
    latent = codec.encode(
        h,
        v,
        time_ps=torch.tensor([[1.0, 2.0, 4.0, 7.0]]),
        frame_mask=torch.ones(1, 4, dtype=torch.bool),
        abid=torch.zeros(3, dtype=torch.long),
        sample_origin=torch.zeros(1, 3),
    )
    assert torch.equal(latent.frame_time_ps, torch.tensor([[1.0, 2.0, 4.0, 7.0]]))
    with pytest.raises(ValueError, match="strictly increasing"):
        codec.encode(
            h,
            v,
            time_ps=torch.tensor([[1.0, 1.0, 2.0, 3.0]]),
            frame_mask=torch.ones(1, 4, dtype=torch.bool),
            abid=torch.zeros(3, dtype=torch.long),
            sample_origin=torch.zeros(1, 3),
        )


def test_state_detail_gradients_are_finite_and_nonzero():
    h, v = _features(frames=8)
    h.requires_grad_()
    v.requires_grad_()
    codec = StateDetailCodecV2(5, mode="ratio4_state_detail")
    output = codec(
        h,
        v,
        time_ps=torch.arange(8, dtype=torch.float32).view(1, 8) * 100.0,
        frame_mask=torch.ones(1, 8, dtype=torch.bool),
        abid=torch.zeros(3, dtype=torch.long),
        sample_origin=torch.zeros(1, 3),
    )
    loss = output.x_hat.square().mean() + output.h.square().mean() + output.v.square().mean()
    loss.backward()
    assert h.grad is not None and torch.isfinite(h.grad).all()
    assert v.grad is not None and torch.isfinite(v.grad).all()
    assert h.grad.abs().sum() > 0 and v.grad.abs().sum() > 0
    for name, parameter in codec.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name


def test_state_detail_se3_equivariance_and_origin_is_one_vector():
    h, v = _features(frames=8)
    codec = StateDetailCodecV2(5, mode="ratio2_state_detail")
    kwargs = {
        "time_ps": torch.arange(8, dtype=torch.float32).view(1, 8) * 100.0,
        "frame_mask": torch.ones(1, 8, dtype=torch.bool),
        "abid": torch.zeros(3, dtype=torch.long),
    }
    origin = torch.tensor([[1.0, -2.0, 0.5]])
    output = codec(h, v, sample_origin=origin, **kwargs)
    angle = torch.tensor(0.37)
    c, s = torch.cos(angle), torch.sin(angle)
    rotation = torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    translation = torch.tensor([2.0, -1.0, 0.75])
    rotated_v = torch.einsum("ij,tnjc->tnic", rotation, v)
    transformed_origin = origin @ rotation.T + translation
    transformed = codec(h, rotated_v, sample_origin=transformed_origin, **kwargs)
    expected = torch.einsum("ij,tnj->tni", rotation, output.x_hat) + translation
    assert torch.allclose(transformed.x_hat, expected, rtol=1e-5, atol=1e-6)
    assert output.latent.sample_origin.shape == (1, 3)
    assert not hasattr(output.latent, "x_anchor")


def test_state_detail_sample_isolation_and_invalid_frame_isolation():
    h, v = _features()
    frame_mask = torch.ones(2, 16, dtype=torch.bool)
    abid = torch.tensor([0, 0, 1], dtype=torch.long)
    kwargs = {
        "time_ps": torch.arange(16, dtype=torch.float32).view(1, 16).repeat(2, 1),
        "frame_mask": frame_mask,
        "abid": abid,
        "sample_origin": torch.zeros(2, 3),
    }
    codec = StateDetailCodecV2(5, mode="ratio4_state_detail")
    reference = codec(h, v, **kwargs)
    changed_h, changed_v = h.clone(), v.clone()
    changed_h[:, 2] += 100.0
    changed_v[:, 2] -= 100.0
    changed = codec(changed_h, changed_v, **kwargs)
    assert torch.allclose(reference.x_hat[:, :2], changed.x_hat[:, :2], atol=1e-6)

    masked = frame_mask.clone()
    masked[0, 4:8] = False
    base_h, base_v = h.clone(), v.clone()
    altered_h, altered_v = h.clone(), v.clone()
    altered_h[4:8, :2] = 1000.0
    altered_v[4:8, :2] = -1000.0
    base = codec(
        base_h,
        base_v,
        time_ps=kwargs["time_ps"],
        frame_mask=masked,
        abid=abid,
        sample_origin=kwargs["sample_origin"],
    )
    altered = codec(
        altered_h,
        altered_v,
        time_ps=kwargs["time_ps"],
        frame_mask=masked,
        abid=abid,
        sample_origin=kwargs["sample_origin"],
    )
    assert torch.allclose(base.x_hat[:4, :2], altered.x_hat[:4, :2], atol=1e-6)


def test_matched_pooling_capacity_and_no_anchor():
    h, v = _features()
    codec = MatchedPoolingCodecV2(5)
    latent = codec.encode(
        h,
        v,
        time_ps=torch.arange(16, dtype=torch.float32).view(1, 16),
        frame_mask=torch.ones(1, 16, dtype=torch.bool),
        abid=torch.zeros(3, dtype=torch.long),
        sample_origin=torch.zeros(1, 3),
    )
    assert latent.active_feature_volume == 8 * 5
    assert latent.contract()["state_detail_semantics"] is False
    assert not hasattr(latent, "x_anchor")
    assert codec.decode(latent).x_hat.shape == (16, 3, 3)


def test_matched_pooling_principal_parameters_receive_gradients():
    h, v = _features()
    h.requires_grad_()
    v.requires_grad_()
    codec = MatchedPoolingCodecV2(5)
    output = codec(
        h,
        v,
        time_ps=torch.arange(16, dtype=torch.float32).view(1, 16),
        frame_mask=torch.ones(1, 16, dtype=torch.bool),
        abid=torch.zeros(3, dtype=torch.long),
        sample_origin=torch.zeros(1, 3),
    )
    (output.x_hat.square().mean() + output.h.square().mean() + output.v.square().mean()).backward()
    for name, parameter in codec.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name


def _record():
    times = torch.arange(4, dtype=torch.float32) * 100.0
    x = torch.zeros(4, 2, 3)
    x[:, 0, 0] = times / 100.0
    x[:, 1, 1] = 1.0
    return {
        "schema_version": "pvb.clip.v1",
        "sample_id": "state-detail-test",
        "task": "trajectory",
        "time_bucket_id": "dt_100ps",
        "time_ps": times.numpy(),
        "delta_time_ps": torch.diff(times).numpy(),
        "x": x.numpy(),
        "bpos": x.numpy(),
        "atype": torch.ones(2, dtype=torch.long).numpy(),
        "btype": torch.zeros(2, dtype=torch.long).numpy(),
        "block_id": torch.zeros(2, dtype=torch.long).numpy(),
        "component_id": torch.zeros(2, dtype=torch.long).numpy(),
        "atom_source_index": torch.arange(2, dtype=torch.long).numpy(),
        "atom_identity": ["state-detail:0", "state-detail:1"],
        "edge_mask": torch.zeros(2, dtype=torch.long).numpy(),
        "loss_mask": torch.ones(2, dtype=torch.bool).numpy(),
        "align_mask": torch.tensor([True, False]).numpy(),
        "bond_index": torch.tensor([[0], [1]], dtype=torch.long).numpy(),
    }


def test_model_contract_roundtrip_and_public_decoder_has_no_target_argument():
    model = PVBCodecModel(
        hidden_channels=8,
        spatial_layers=1,
        temporal_layers=1,
        temporal_codec_mode="ratio2_state_detail",
        num_rbf=4,
        num_heads=2,
        max_num_neighbors=4,
        neighbor_backend="dense_test",
    )
    contract = model.model_contract()
    assert contract["schema_version"] == "pvb.codec.model_contract.v4"
    assert contract["constructor"]["coordinate_stem"] == "centered_vector"
    assert contract["architecture"]["decoder"]["coordinate_stem"] == "centered_vector"
    assert contract["architecture"]["decoder"]["per_atom_anchor"] is False
    assert contract["architecture"]["temporal"]["cross_block_attention"] is False
    restored = PVBCodecModel.from_model_contract(contract)
    assert restored.model_contract() == contract
    assert "target_coordinates" not in inspect.signature(model.decode).parameters
    assert "x_anchor" not in str(contract)

    batch = collate_clip_records([_record()])
    model.prepare_batch(batch)
    output = model(batch)
    assert output.x_hat.shape == batch.x.shape
    assert output.latent.sample_origin.shape == (1, 3)


def test_old_v3_state_detail_contract_loads_without_a_coordinate_stem():
    model = PVBCodecModel(
        hidden_channels=8,
        spatial_layers=1,
        temporal_layers=1,
        temporal_codec_mode="ratio2_state_detail",
        coordinate_stem="none",
        num_rbf=4,
        num_heads=2,
        max_num_neighbors=4,
        neighbor_backend="dense_test",
    )
    contract = model.model_contract()
    assert contract["schema_version"] == "pvb.codec.model_contract.v3"
    assert "coordinate_stem" not in contract["constructor"]
    assert PVBCodecModel.from_model_contract(contract).model_contract() == contract


def test_static_topology_is_n_axis_coordinate_independent_and_radius_free():
    batch = collate_clip_records([_record()])
    topology = StaticTopologyMetadata.from_batch(batch)
    assert topology.schema_version == STATIC_TOPOLOGY_SCHEMA
    assert topology.num_atoms == batch.atom_count == 2
    assert topology.covalent_bond_index.numel() == 2
    assert int(topology.covalent_bond_index.max()) < topology.num_atoms
    assert topology.covalent_bond_index.shape[0] == 2
    contract = topology.contract()
    assert contract["bond_index_space"] == "latent_atom_axis_N"
    assert contract["contains_radius_edges"] is False
    assert contract["contains_distance_or_edge_vectors"] is False
    assert contract["contains_target_coordinates"] is False

    changed = replace(
        batch,
        x=batch.x + torch.arange(batch.frames, dtype=batch.x.dtype).view(-1, 1, 1) * 17.0,
        bpos=batch.bpos + 91.0,
    )
    changed_topology = StaticTopologyMetadata.from_batch(changed)
    for field in (
        "atom_type",
        "block_type",
        "abid",
        "block_id",
        "component_id",
        "atom_ptr",
        "covalent_bond_index",
        "covalent_bond_type",
    ):
        assert torch.equal(getattr(topology, field), getattr(changed_topology, field))
    assert topology.contract() == changed_topology.contract()

    longer = replace(
        batch,
        x=batch.x[:1].repeat(16, 1, 1),
        bpos=batch.bpos[:1].repeat(16, 1, 1),
        frame_mask=torch.ones(1, 16, dtype=torch.bool),
        time_ps=torch.arange(16, dtype=torch.float32).view(1, 16) * 100.0,
        delta_time_ps=torch.full((1, 15), 100.0),
    )
    longer_topology = StaticTopologyMetadata.from_batch(longer)
    assert longer_topology.num_atoms == topology.num_atoms
    assert longer_topology.covalent_bond_index.shape == topology.covalent_bond_index.shape
    assert longer_topology.contract() == topology.contract()

    model = PVBCodecModel(
        hidden_channels=8,
        spatial_layers=1,
        temporal_layers=1,
        temporal_codec_mode="ratio2_state_detail",
        num_rbf=4,
        num_heads=2,
        max_num_neighbors=4,
        neighbor_backend="dense_test",
    )
    model.prepare_batch(batch)
    latent = model.encode(batch)
    assert isinstance(latent.topology, StaticTopologyMetadata)
    assert not any(
        hasattr(latent.topology, forbidden)
        for forbidden in (
            "pos",
            "edge_vec",
            "edge_weight",
            "distance_edge_index",
            "distance_edge_vec",
        )
    )
    assert latent.contract()["topology"]["atom_count"] == batch.atom_count


def test_centered_coordinate_vector_stem_is_equivariant_and_zero_preserving():
    stem = CenteredCoordinateVectorStem(5)
    coordinates = torch.randn(4, 3, 3)
    output = stem(coordinates)
    assert output.shape == (4, 3, 3, 5)
    assert torch.equal(stem(torch.zeros_like(coordinates)), torch.zeros_like(output))
    assert stem.projection.bias is None


def test_decoder_isolated_from_target_mutation_after_encoding():
    batch = collate_clip_records([_record()])
    model = PVBCodecModel(
        hidden_channels=8,
        spatial_layers=1,
        temporal_layers=1,
        temporal_codec_mode="ratio2_state_detail",
        num_rbf=4,
        num_heads=2,
        max_num_neighbors=4,
        neighbor_backend="dense_test",
    )
    model.prepare_batch(batch)
    latent = model.encode(batch)
    before = model.decode(latent).x_hat.clone()
    batch.x.add_(10000.0)
    after = model.decode(latent).x_hat
    assert torch.equal(before, after)


def test_state_detail_checkpoint_and_config_roundtrip(tmp_path):
    batch = collate_clip_records([_record()])
    config = CodecTrainConfig(
        lr=1.0e-3,
        max_steps=1,
        bucket_specs=(TimeBucketSpec("dt_100ps", 100.0, 1.0),),
        loss_schedule=((0, {"coordinate": 1.0}),),
    )
    model = PVBCodecModel(
        hidden_channels=8,
        spatial_layers=1,
        temporal_layers=1,
        temporal_codec_mode="ratio2_state_detail",
        num_rbf=4,
        num_heads=2,
        max_num_neighbors=4,
        neighbor_backend="dense_test",
    )
    trainer = CodecTrainer(model, [batch], config=config, device="cpu")
    trainer.optimizer_step(batch)
    checkpoint = trainer.save_checkpoint(tmp_path / "state-detail.pt")
    restored_model = PVBCodecModel.from_model_contract(model.model_contract())
    restored = CodecTrainer(restored_model, [batch], config=config, device="cpu")
    restored.load_checkpoint(checkpoint)
    assert restored.step == trainer.step == 1
    assert restored.model.model_contract() == model.model_contract()


def test_center_origin_uses_masked_centroid_and_no_per_atom_anchor():
    x = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0], [101.0, 0.0, 0.0]],
        ]
    )
    centered, origin = center_coordinates(
        x,
        frame_mask=torch.ones(1, 2, dtype=torch.bool),
        abid=torch.zeros(2, dtype=torch.long),
        atom_mask=torch.tensor([True, False]),
    )
    assert torch.equal(origin, torch.zeros(1, 3))
    assert torch.equal(centered[:, 0], x[:, 0])
    assert torch.equal(centered[:, 1], x[:, 1])
