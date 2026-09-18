"""Exact old/new flow and observed-only generation checks."""

from __future__ import annotations

from dataclasses import fields
from types import SimpleNamespace

import pytest
import torch

from dit_test_utils import make_batch
from module.dit_geometry_supervision import endpoint_from_velocity as old_endpoint
from module.latent_flow_source import (
    block_state_zero_detail_center as old_block_center,
    repeat_last_coordinate_template as old_repeat_template,
)
from module.latent_rectified_flow import (
    RectifiedFlowObjective as OldObjective,
    euler_sample as old_euler_sample,
)
from module.state_detail_latent_adapter import LatentStatistics as OldStatistics
from molvid.dit.model import MolecularDiT
from molvid.flow.objective import RectifiedFlowObjective, endpoint_from_velocity
from molvid.flow.sampling import euler_sample
from molvid.flow.source import block_state_zero_detail_center, repeat_last_coordinate_template
from molvid.geometry.types import StaticTopologyMetadata
from molvid.latent.adapter import StateDetailLatentAdapter
from molvid.latent.statistics import LatentStatistics
from molvid.latent.types import LatentBatch, LatentFields


@pytest.fixture
def deterministic_cuda():
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    yield
    torch.use_deterministic_algorithms(previous)


def _new_batch(old):
    values = {field.name: getattr(old, field.name) for field in fields(old)}
    values["fields"] = LatentFields(**old.fields.as_dict())
    values["topology"] = StaticTopologyMetadata(
        **{field.name: getattr(old.topology, field.name) for field in fields(old.topology)}
    )
    return LatentBatch(**values)


def _statistics(cls, ratio, width, device):
    zeros = torch.zeros(width, device=device)
    ones = torch.ones(width, device=device)
    return cls(
        ratio=ratio,
        mode=f"ratio{ratio}_state_detail",
        width=width,
        state_h_mean=zeros,
        state_h_std=ones,
        detail_h_mean=zeros,
        detail_h_std=ones,
        state_v_rms=ones,
        detail_v_rms=ones,
        provenance={"test": True},
    )


def _equal_fields(first, second):
    for name in first.names():
        torch.testing.assert_close(getattr(first, name), getattr(second, name), rtol=0, atol=0)


@pytest.mark.parametrize("ratio", [2, 4])
def test_fixed_noise_flow_source_endpoint_and_loss_match_old(ratio):
    assert torch.cuda.is_available(), "P3 flow parity requires CUDA"
    old_batch = make_batch(ratio, width=4, invalid_last_token=True).to("cuda")
    observed = torch.zeros_like(old_batch.token_mask)
    observed[:, 0] = True
    old_batch = old_batch.with_observation(observed)
    new_batch = _new_batch(old_batch)
    old_stats = _statistics(OldStatistics, ratio, 4, "cuda")
    new_stats = _statistics(LatentStatistics, ratio, 4, "cuda")
    old_center = old_block_center(old_batch, statistics=old_stats, history_frames=ratio * 2)
    new_center = block_state_zero_detail_center(new_batch, statistics=new_stats, history_frames=ratio * 2)
    _equal_fields(old_center, new_center)
    old_generator = torch.Generator(device="cuda").manual_seed(77)
    new_generator = torch.Generator(device="cuda").manual_seed(77)
    old_sample = OldObjective().sample(
        old_batch, generator=old_generator, source_center=old_center, source_mode="conditional"
    )
    new_sample = RectifiedFlowObjective().sample(
        new_batch, generator=new_generator, source_center=new_center, source_mode="conditional"
    )
    torch.testing.assert_close(old_sample.tau, new_sample.tau, rtol=0, atol=0)
    assert torch.equal(old_generator.get_state(), new_generator.get_state())
    for name in ("noise", "source", "interpolated", "target"):
        _equal_fields(getattr(old_sample, name), getattr(new_sample, name))
    old_prediction = old_sample.target.map(lambda value: value.detach().clone().requires_grad_())
    new_prediction = new_sample.target.map(lambda value: value.detach().clone().requires_grad_())
    old_loss = OldObjective().loss(old_prediction, old_sample.target, old_batch)
    new_loss = RectifiedFlowObjective().loss(new_prediction, new_sample.target, new_batch)
    torch.testing.assert_close(old_loss.total, new_loss.total, rtol=0, atol=0)
    assert old_loss.valid_elements == new_loss.valid_elements
    old_loss.total.backward()
    new_loss.total.backward()
    for name in old_prediction.names():
        torch.testing.assert_close(getattr(old_prediction, name).grad, getattr(new_prediction, name).grad, rtol=0, atol=0)
    old_end = old_endpoint(old_sample.interpolated, old_sample.target, old_sample.tau, sample_ids=old_batch.abid)
    new_end = endpoint_from_velocity(new_sample.interpolated, new_sample.target, new_sample.tau, sample_ids=new_batch.abid)
    _equal_fields(old_end, new_end)


def test_repeated_coordinate_template_ignores_future_truth():
    coordinate = SimpleNamespace(
        frames=16, batch_size=1, atom_ptr=torch.tensor([0, 3]),
        x=torch.arange(16 * 3 * 3, dtype=torch.float32).reshape(16, 3, 3),
    )
    changed = SimpleNamespace(**coordinate.__dict__)
    changed.x = coordinate.x.clone()
    changed.x[4:] += 12345.0
    first = repeat_last_coordinate_template(coordinate, 4)
    second = repeat_last_coordinate_template(changed, 4)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    torch.testing.assert_close(first, old_repeat_template(coordinate, 4), rtol=0, atol=0)


def test_conditional_euler_is_old_new_equal_and_future_isolated():
    assert torch.cuda.is_available(), "P3 full inference isolation requires CUDA"
    ratio = 4
    old_batch = make_batch(ratio, width=4).to("cuda")
    observed = torch.zeros_like(old_batch.token_mask)
    observed[:, :2] = True
    old_batch = old_batch.with_observation(observed)
    new_batch = _new_batch(old_batch)
    stats_old = _statistics(OldStatistics, ratio, 4, "cuda")
    stats_new = _statistics(LatentStatistics, ratio, 4, "cuda")
    old_center = old_block_center(old_batch, statistics=stats_old, history_frames=8)
    new_center = block_state_zero_detail_center(new_batch, statistics=stats_new, history_frames=8)
    torch.manual_seed(902)
    from module.molecular_dit import MolecularDiT as OldDiT
    from module.state_detail_latent_adapter import StateDetailLatentAdapter as OldAdapter

    old_model = OldDiT(
        adapter=OldAdapter(codec_width=4, scalar_width=8, vector_width=4, ratio=ratio),
        scalar_width=8, vector_width=4, depth=1, heads=2, ffn_multiplier=2,
        execution_backend="factorized_v2",
    ).cuda()
    torch.manual_seed(902)
    new_model = MolecularDiT(
        adapter=StateDetailLatentAdapter(codec_width=4, scalar_width=8, vector_width=4, ratio=ratio),
        scalar_width=8, vector_width=4, depth=1, heads=2, ffn_multiplier=2,
        execution_backend="factorized_v2",
    ).cuda()
    old_result, old_meta = old_euler_sample(
        old_model, old_batch, steps=8, seed=71, source_center=old_center, source_mode="conditional"
    )
    new_result, new_meta = euler_sample(
        new_model, new_batch, steps=8, seed=71, source_center=new_center, source_mode="conditional"
    )
    _equal_fields(old_result.fields, new_result.fields)
    assert old_meta == new_meta
    observed_atoms = observed.index_select(0, new_batch.abid).transpose(0, 1)
    for name in new_result.fields.names():
        value = getattr(new_result, name)
        clean = getattr(new_batch, name)
        expanded = observed_atoms.reshape(observed_atoms.shape + (1,) * (value.ndim - 2))
        torch.testing.assert_close(value.masked_select(expanded), clean.masked_select(expanded), rtol=0, atol=0)

    changed_fields = new_batch.fields.map(lambda value: value.clone())
    for name in changed_fields.names():
        value = getattr(changed_fields, name)
        value[~observed_atoms] += 12345.0
    changed = new_batch.with_fields(changed_fields)
    changed_center = block_state_zero_detail_center(changed, statistics=stats_new, history_frames=8)
    _equal_fields(new_center, changed_center)
    changed_result, changed_meta = euler_sample(
        new_model, changed, steps=8, seed=71, source_center=changed_center, source_mode="conditional"
    )
    _equal_fields(new_result.fields, changed_result.fields)
    assert new_meta == changed_meta

def test_encoded_repeat_center_matches_old_and_ignores_future_coordinates(deterministic_cuda):
    assert torch.cuda.is_available(), "P3 encoded-center gate requires CUDA"
    from dataclasses import replace

    import numpy as np

    from data.clip_dataset import collate_clip_records as old_collate
    from module.latent_flow_source import build_observed_center as old_build_center
    from module.state_detail_latent_adapter import StateDetailLatentAdapter as OldAdapter
    from molvid.data.batch import collate_clip_records
    from molvid.flow.source import build_observed_center
    from molvid.codec.model import TrajectoryCodec
    from trainer.codec_trainer import PVBCodecModel
    from test_codec import _record

    record = _record()
    record["x"] = np.tile(record["x"], (4, 1, 1))
    record["bpos"] = record["x"].copy()
    record["time_ps"] = np.arange(16, dtype=np.float32) * 100.0
    record["delta_time_ps"] = np.full(15, 100.0, dtype=np.float32)
    old_cpu = old_collate([record])
    new_cpu = collate_clip_records([record])
    common = dict(
        hidden_channels=8, spatial_layers=1, temporal_codec_mode="ratio4_state_detail",
        num_rbf=8, num_heads=2, coordinate_stem="centered_vector",
    )
    torch.manual_seed(713)
    old_codec = PVBCodecModel(**common, temporal_layers=1, temporal_ratio=4).cuda().eval()
    torch.manual_seed(713)
    new_codec = TrajectoryCodec(**common).cuda().eval()
    old_codec.prepare_batch(old_cpu)
    new_codec.prepare_batch(new_cpu)
    old_clip = old_cpu.to("cuda")
    new_clip = new_cpu.to("cuda")
    old_adapter = OldAdapter(codec_width=8, scalar_width=8, vector_width=4, ratio=4).cuda()
    new_adapter = StateDetailLatentAdapter(codec_width=8, scalar_width=8, vector_width=4, ratio=4).cuda()
    observed = torch.tensor([[True, True, False, False]], device="cuda")
    with torch.no_grad():
        old_target = old_adapter.pack(old_codec.encode(old_clip), loss_mask=old_clip.loss_mask).with_observation(observed)
        new_target = new_adapter.from_codec_latent(new_codec.encode(new_clip), loss_mask=new_clip.loss_mask).with_observation(observed)
        old_stats = _statistics(OldStatistics, 4, 8, "cuda")
        new_stats = _statistics(LatentStatistics, 4, 8, "cuda")
        old_center, old_meta = old_build_center(
            "repeat_last_coordinate_encode", codec_model=old_codec,
            coordinate_batch=old_clip, target_batch=old_target,
            adapter=old_adapter, statistics=old_stats, history_frames=8,
        )
        new_center, new_meta = build_observed_center(
            "repeat_last_coordinate_encode", codec_model=new_codec,
            coordinate_batch=new_clip, target_batch=new_target,
            adapter=new_adapter, statistics=new_stats, history_frames=8,
        )
        _equal_fields(old_center, new_center)
        torch.testing.assert_close(old_meta["template_coordinates"], new_meta["template_coordinates"], rtol=0, atol=0)
        changed_x = new_clip.x.clone()
        changed_x[8:] += 12345.0
        changed_clip = replace(new_clip, x=changed_x)
        isolated_center, isolated_meta = build_observed_center(
            "repeat_last_coordinate_encode", codec_model=new_codec,
            coordinate_batch=changed_clip, target_batch=new_target,
            adapter=new_adapter, statistics=new_stats, history_frames=8,
        )
        _equal_fields(new_center, isolated_center)
        torch.testing.assert_close(new_meta["template_coordinates"], isolated_meta["template_coordinates"], rtol=0, atol=0)

def test_source_contract_bytes_and_invalid_modes_match_old():
    from module.latent_flow_source import source_contract as old_source_contract
    from molvid.flow.source import source_contract

    old_stats = _statistics(OldStatistics, 4, 4, "cpu")
    new_stats = _statistics(LatentStatistics, 4, 4, "cpu")
    for mode in ("gaussian", "conditional"):
        for center_kind in ("repeat_last_coordinate_encode", "block_state_zero_detail"):
            expected = old_source_contract(
                source_mode=mode, center_kind=center_kind, statistics=old_stats
            )
            actual = source_contract(
                source_mode=mode, center_kind=center_kind, statistics=new_stats
            )
            assert actual == expected
    with pytest.raises(ValueError, match="unsupported source_mode"):
        source_contract(
            source_mode="invalid", center_kind="block_state_zero_detail", statistics=new_stats
        )
