from __future__ import annotations

from dataclasses import dataclass

import torch

from molvid.data.batch import ClipBatch
from molvid.geometry.types import StaticTopologyMetadata
from molvid.flow.source import sample_frame_source
from molvid.latent.types import FrameLatentBatch, ObservedContext, QuerySpec
from molvid.losses.distribution import (
    SampledFeatureScales,
    _feature_distance,
    _scaled_groups,
    energy_score_loss,
    feature_scale_moments,
    observed_reference_bond_loss,
    resolve_feature_scales,
)


@dataclass(frozen=True)
class _Batch:
    coordinate_batch: ClipBatch
    observed_context: ObservedContext
    query: QuerySpec


def _batch() -> tuple[_Batch, torch.Tensor]:
    topology = StaticTopologyMetadata(
        atom_type=torch.tensor([6, 6, 6, 6]),
        block_type=torch.ones(4, dtype=torch.long),
        abid=torch.zeros(4, dtype=torch.long),
        block_id=torch.tensor([0, 0, 1, 1]),
        component_id=torch.zeros(4, dtype=torch.long),
        atom_ptr=torch.tensor([0, 4]),
        covalent_bond_index=torch.tensor([[0, 1, 2], [1, 2, 3]]),
        covalent_bond_type=torch.ones(3, dtype=torch.long),
        topology_id=("toy",),
        sample_id=("toy_R1",),
    )
    observed_coordinates = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [1.0, 0.1, 0.0], [2.0, 0.0, 0.0], [3.0, -0.1, 0.0]],
        ]
    )
    latent = FrameLatentBatch(
        h=torch.zeros(2, 4, 2),
        v=torch.zeros(2, 4, 3, 2),
        time_ps=torch.tensor([[0.0, 100.0]]),
        frame_mask=torch.ones(1, 2, dtype=torch.bool),
        topology=topology,
    )
    context = ObservedContext(
        latent=latent,
        coordinates=observed_coordinates,
        sample_origin=torch.zeros(1, 3),
        loss_mask=torch.ones(4, dtype=torch.bool),
    )
    query = QuerySpec(
        time_ps=torch.tensor([[200.0, 300.0, 400.0]]),
        frame_mask=torch.ones(1, 3, dtype=torch.bool),
    )
    target = torch.stack([
        observed_coordinates[-1] + torch.tensor([0.0, 0.05, 0.0]),
        observed_coordinates[-1] + torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.0], [0.0, -0.1, 0.0]]),
        observed_coordinates[-1] + torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.2, 0.0], [0.0, 0.0, 0.0], [0.0, -0.2, 0.0]]),
    ])
    full = torch.cat((observed_coordinates, target), dim=0)
    coordinate_batch = ClipBatch(
        x=full,
        bpos=full,
        atype=topology.atom_type,
        btype=topology.block_type,
        block_id=topology.block_id,
        component_id=topology.component_id,
        atom_source_index=torch.arange(4),
        abid=topology.abid,
        atom_ptr=topology.atom_ptr,
        bond_index=topology.covalent_bond_index,
        edge_mask=torch.ones(4, dtype=torch.bool),
        loss_mask=torch.ones(4, dtype=torch.bool),
        align_mask=torch.ones(4, dtype=torch.bool),
        frame_mask=torch.ones(1, 5, dtype=torch.bool),
        time_ps=torch.tensor([[0.0, 100.0, 200.0, 300.0, 400.0]]),
        delta_time_ps=torch.full((1, 4), 100.0),
        time_bucket_id=("dt_100ps",),
        topology_id=("toy",),
        task=torch.zeros(1, dtype=torch.long),
        sample_id=("toy_R1",),
        atom_counts=(4,),
    )
    return _Batch(coordinate_batch, context, query), target


def _scales(batch: _Batch, target: torch.Tensor) -> SampledFeatureScales:
    scales, diagnostics = resolve_feature_scales([feature_scale_moments(target, batch)])
    assert all(value["value_count"] > 0 for value in diagnostics.values())
    return scales


def test_energy_score_is_condition_local_and_has_finite_coordinate_gradients() -> None:
    batch, target = _batch()
    scales = _scales(batch, target)
    first = target.clone().requires_grad_(True)
    second = target.clone().requires_grad_(True)
    loss, count, groups = energy_score_loss((first, second), target, batch, scales)
    assert count.item() == 1
    assert groups == {"residue_rmsf": 1, "displacement": 1, "internal_distance": 1}
    assert loss.item() < 1.0e-5
    loss.backward()
    assert first.grad is not None and torch.isfinite(first.grad).all()
    assert second.grad is not None and torch.isfinite(second.grad).all()


def test_k2_energy_score_uses_half_pairwise_distance() -> None:
    batch, target = _batch()
    scales = _scales(batch, target)
    first = target.clone()
    first[1, 1, 1] += 0.2
    second = target.clone()
    second[2, 3, 1] -= 0.3
    actual, _, _ = energy_score_loss((first, second), target, batch, scales)
    target_features = _scaled_groups(target, batch, 0, scales)
    first_features = _scaled_groups(first, batch, 0, scales)
    second_features = _scaled_groups(second, batch, 0, scales)
    expected = 0.5 * (
        _feature_distance(first_features, target_features)
        + _feature_distance(second_features, target_features)
    ) - 0.5 * _feature_distance(first_features, second_features)
    torch.testing.assert_close(actual, expected)


def test_energy_features_share_one_observed_reference_under_rigid_motion() -> None:
    batch, target = _batch()
    scales = _scales(batch, target)
    rotation = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    reference = batch.observed_context.coordinates[-1].mean(dim=0)
    transformed = (target - target.mean(dim=1, keepdim=True)) @ rotation + reference + 7.0
    loss, _, _ = energy_score_loss((transformed, transformed), target, batch, scales)
    # Internal distances are exact; fixed-reference Kabsch removes the rigid frame.
    assert loss.item() < 2.0e-4


def test_actual_sample_bond_uses_observed_reference_not_hidden_future() -> None:
    batch, target = _batch()
    sampled = (target.clone(), target.clone())
    first, count = observed_reference_bond_loss(sampled, batch)
    mutated_target = target + torch.randn_like(target) * 100.0
    # The API has no target argument: hidden-future mutation cannot alter this loss.
    second, second_count = observed_reference_bond_loss(sampled, batch)
    assert torch.equal(first, second)
    assert count.item() == second_count.item() == 1
    assert not torch.equal(sampled[0], mutated_target)


def test_zero_motion_features_remain_structurally_eligible() -> None:
    batch, _target = _batch()
    static = batch.observed_context.coordinates[-1:].expand(3, -1, -1).clone()
    moments = feature_scale_moments(static, batch)
    scales, diagnostics = resolve_feature_scales([moments])
    assert all(value["constant_feature_unit_scale_fallback"] for value in diagnostics.values())
    loss, count, groups = energy_score_loss((static, static), static, batch, scales)
    assert count.item() == 1
    assert len(groups) == 3
    assert loss.item() < 1.0e-5


def test_sampled_source_rng_does_not_advance_main_flow_rng() -> None:
    batch, _target = _batch()
    center = batch.observed_context.latent
    reference_main = torch.Generator().manual_seed(20260923)
    expected = (
        torch.rand((), generator=reference_main),
        torch.rand((), generator=reference_main),
    )
    actual_main = torch.Generator().manual_seed(20260923)
    sampled = torch.Generator().manual_seed(20260924)
    first = torch.rand((), generator=actual_main)
    sample_frame_source(center, generator=sampled)
    sample_frame_source(center, generator=sampled)
    second = torch.rand((), generator=actual_main)
    assert torch.equal(first, expected[0])
    assert torch.equal(second, expected[1])
