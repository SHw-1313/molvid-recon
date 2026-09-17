from __future__ import annotations

from contextlib import nullcontext
from dataclasses import replace
import math

import numpy as np
import pytest
import torch
from torch import nn

from data.clip_dataset import collate_clip_records
from evaluation.dit_diagnostics import (
    aggregate_rows,
    diversity_summary,
    field_swap_fields,
    fixed_tau_flow_diagnostics,
    frame_intervals,
    horizon_indices,
    latent_block_state_persistence,
    latent_summary,
    observed_coordinate_baseline,
    observed_coordinate_scaffold,
    stable_seed,
    standardized_raw_zero_detail,
)
from module.latent_rectified_flow import LatentFieldSet
from module.state_detail_latent_adapter import LatentStatistics, build_observation_condition
from tests.dit_test_utils import make_batch


def _record(*, sample_id: str = "system_R1_w000030", dt: float = 80.0):
    frames, atoms = 16, 4
    time = np.arange(frames, dtype=np.float32) * dt
    x = np.arange(frames * atoms * 3, dtype=np.float32).reshape(frames, atoms, 3)
    return {
        "schema_version": "pvb.clip.v1",
        "sample_id": sample_id,
        "task": "trajectory",
        "time_bucket_id": "dt_80ps",
        "time_ps": time,
        "delta_time_ps": np.full(frames - 1, dt, dtype=np.float32),
        "x": x,
        "bpos": x.copy(),
        "atype": np.arange(atoms, dtype=np.int64),
        "btype": np.arange(atoms, dtype=np.int64),
        "block_id": np.arange(atoms, dtype=np.int64),
        "component_id": np.zeros(atoms, dtype=np.int64),
        "atom_source_index": np.arange(atoms, dtype=np.int64),
        "atom_identity": [f"atom:{i}" for i in range(atoms)],
        "edge_mask": np.zeros(atoms, dtype=np.int64),
        "loss_mask": np.ones(atoms, dtype=np.bool_),
        "align_mask": np.asarray([True, True, False, False]),
        "bond_index": np.asarray([[0, 1], [1, 0]], dtype=np.int64),
    }


def _stats(width: int = 4) -> LatentStatistics:
    return LatentStatistics(
        ratio=4,
        mode="ratio4_state_detail",
        width=width,
        state_h_mean=torch.full((width,), 2.0),
        state_h_std=torch.full((width,), 2.0),
        detail_h_mean=torch.full((width,), 3.0),
        detail_h_std=torch.full((width,), 4.0),
        state_v_rms=torch.full((width,), 2.0),
        detail_v_rms=torch.full((width,), 5.0),
        provenance={"test": True},
    )


def test_observed_coordinate_baselines_ignore_future_and_use_physical_dt():
    batch = collate_clip_records([_record()])
    original_future = batch.x[4:].clone()
    prediction, metadata = observed_coordinate_baseline(
        batch, 4, kind="coordinate_constant_velocity"
    )
    assert metadata["source_delta_time_ps"] == 80.0
    assert prediction is not None
    expected_velocity = (batch.x[3] - batch.x[2]) / 80.0
    assert torch.allclose(prediction[4], batch.x[3] + expected_velocity * 80.0)
    mutated = replace(batch, x=batch.x.clone())
    mutated.x[4:] = 10_000.0
    mutated_prediction, _ = observed_coordinate_baseline(
        mutated, 4, kind="coordinate_constant_velocity"
    )
    assert torch.equal(prediction, mutated_prediction)
    assert torch.equal(batch.x[4:], original_future)


def test_scaffold_and_horizon_protocol_are_explicit():
    batch = collate_clip_records([_record()])
    scaffold = observed_coordinate_scaffold(batch, 4)
    mutated = replace(batch, x=batch.x.clone())
    mutated.x[4:] = -9999.0
    assert torch.equal(scaffold, observed_coordinate_scaffold(mutated, 4))
    assert frame_intervals(0)["boundary"] is None
    assert frame_intervals(4)["future"] == [4, 16]
    assert horizon_indices(4, 4) == (4, 5, 6, 7)
    assert horizon_indices(8, 8) == tuple(range(8, 16))
    with pytest.raises(ValueError, match="longer"):
        horizon_indices(8, 8, total_frames=12)


def test_observation_condition_is_future_mutation_invariant():
    batch = make_batch(4, atom_counts=(3,), width=4)
    coordinates = torch.arange(16 * 3 * 3, dtype=torch.float32).reshape(16, 3, 3)
    condition_a = build_observation_condition(
        batch,
        history_frames=4,
        coordinates=coordinates,
        frame_mask=torch.ones(1, 16, dtype=torch.bool),
        loss_mask=torch.ones(3, dtype=torch.bool),
    )
    mutated = coordinates.clone()
    mutated[4:] = 1.0e8
    condition_b = build_observation_condition(
        batch,
        history_frames=4,
        coordinates=mutated,
        frame_mask=torch.ones(1, 16, dtype=torch.bool),
        loss_mask=torch.ones(3, dtype=torch.bool),
    )
    assert torch.equal(condition_a.latent_observation_mask, condition_b.latent_observation_mask)
    assert torch.equal(condition_a.sample_origin, condition_b.sample_origin)
    zero_condition = build_observation_condition(
        batch,
        history_frames=0,
        coordinates=mutated,
        frame_mask=torch.ones(1, 16, dtype=torch.bool),
        loss_mask=torch.ones(3, dtype=torch.bool),
    )
    assert torch.equal(zero_condition.sample_origin, torch.zeros(1, 3))


def test_standardized_raw_zero_detail_is_not_scalar_zero():
    statistics = _stats()
    scalar, vector = standardized_raw_zero_detail(
        statistics, torch.zeros(2, 3, 4), torch.zeros(2, 3, 3, 4)
    )
    assert torch.allclose(scalar, torch.full_like(scalar, -0.75))
    assert torch.equal(vector, torch.zeros_like(vector))


def test_latent_state_persistence_keeps_observed_and_zeros_future_detail():
    source = make_batch(4, atom_counts=(2,), width=4, seed=17)
    coordinates = torch.zeros(16, 2, 3)
    condition = build_observation_condition(
        source,
        history_frames=4,
        coordinates=coordinates,
        frame_mask=torch.ones(1, 16, dtype=torch.bool),
        loss_mask=torch.ones(2, dtype=torch.bool),
    )
    observed = source.with_observation(condition.latent_observation_mask)
    fields, metadata = latent_block_state_persistence(observed, 4)
    assert metadata["source_token"] == 0
    assert torch.equal(fields.state_h[0], source.state_h[0])
    assert torch.equal(fields.detail_h[0], source.detail_h[0])
    assert torch.equal(fields.state_h[1:], source.state_h[0:1].expand_as(fields.state_h[1:]))
    assert torch.equal(fields.detail_h[1:], torch.zeros_like(fields.detail_h[1:]))
    assert fields.detail_h.shape == source.detail_h.shape
    assert fields.detail_v.shape == source.detail_v.shape


def test_aggregation_is_stable_under_batch_reordering():
    rows = [
        {"sample_id": "sys_a_R1_w000030", "system": "sys_a", "metrics": {"future": {"aligned_rmsd": 1.0}}},
        {"sample_id": "sys_a_R2_w000030", "system": "sys_a", "metrics": {"future": {"aligned_rmsd": 3.0}}},
        {"sample_id": "sys_b_R1_w000030", "system": "sys_b", "metrics": {"future": {"aligned_rmsd": 5.0}}},
    ]
    first = aggregate_rows(rows)
    second = aggregate_rows([rows[2], rows[0], rows[1]])
    assert first == second
    assert first["sample_equal"]["aligned_rmsd"] == 3.0
    assert first["system_equal"]["aligned_rmsd"] == 3.5


def test_diversity_uses_xyz_sum_not_xyz_mean():
    batch = collate_clip_records([_record()])
    samples = torch.stack((batch.x, batch.x + 1.0))
    result = diversity_summary(samples, batch)
    assert math.isclose(result["pairwise_raw_rmsd"], math.sqrt(3.0), rel_tol=1e-6)
    assert result["rmsd_definition"] == "sqrt(sum_xyz_squared / valid_atom_count)"


def test_latent_summary_expands_vector_token_masks_for_norm_quantiles():
    fields = LatentFieldSet(
        torch.ones(2, 3, 4),
        torch.ones(2, 3, 4),
        torch.ones(2, 3, 3, 4),
        torch.ones(2, 3, 3, 4),
    )
    masks = {name: torch.ones(2, 3, dtype=torch.bool) for name in (
        "state_h", "detail_h", "state_v", "detail_v"
    )}
    masks["state_v"][0, 1] = False
    summary = latent_summary(fields, masks)
    assert summary["state_v"]["valid_elements"] == 5 * 4
    assert len(summary["state_v"]["norm_quantiles"]) == 5


def test_stable_seed_does_not_depend_on_batch_order():
    first = stable_seed(20260907, "sys_R1_w000030", 4, 0, "sampler")
    second = stable_seed(20260907, "sys_R1_w000030", 4, 0, "sampler")
    assert first == second
    assert first != stable_seed(20260907, "sys_R1_w000030", 8, 0, "sampler")


class _ZeroModel(nn.Module):
    def forward(self, batch, tau):
        return LatentFieldSet.zeros_like(batch.fields)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA diagnostics evidence requires CUDA")
def test_cuda_fixed_tau_flow_diagnostics_uses_cuda_fields():
    device = torch.device("cuda:0")
    batch = make_batch(4, atom_counts=(2,), width=4).to(device)
    coordinates = torch.zeros(16, 2, 3, device=device)
    condition = build_observation_condition(
        batch,
        history_frames=4,
        coordinates=coordinates,
        frame_mask=torch.ones(1, 16, dtype=torch.bool, device=device),
        loss_mask=torch.ones(2, dtype=torch.bool, device=device),
    )
    observed = batch.with_observation(condition.latent_observation_mask, sample_origin=condition.sample_origin)
    model = _ZeroModel().to(device).eval()
    statistics = _stats()
    statistics = replace(
        statistics,
        state_h_mean=statistics.state_h_mean.to(device),
        state_h_std=statistics.state_h_std.to(device),
        detail_h_mean=statistics.detail_h_mean.to(device),
        detail_h_std=statistics.detail_h_std.to(device),
        state_v_rms=statistics.state_v_rms.to(device),
        detail_v_rms=statistics.detail_v_rms.to(device),
    )
    rows = fixed_tau_flow_diagnostics(
        model,
        observed,
        statistics,
        sample_id="system_R1_w000030",
        history_frames=4,
        tau_values=(0.25, 0.75),
        draw_ids=(0, 1),
        autocast_context=nullcontext,
    )
    assert len(rows) == 4
    assert all(row["total"] >= 0.0 for row in rows)
    assert all(getattr(model, "training") is False for _ in [0])
    assert observed.state_h.is_cuda
    assert observed.state_v.is_cuda
