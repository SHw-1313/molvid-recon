"""Reconstruction loss extraction against the current codec objective."""

from __future__ import annotations

import pytest
import torch

from data.clip_dataset import collate_clip_records as old_collate
from trainer.codec_losses import (
    compute_codec_losses as old_compute,
    fit_time_bucket_normalization as old_fit,
)
from trainer.codec_trainer import TimeBucketSpec as OldTimeBucketSpec
from molvid.data.batch import collate_clip_records
from molvid.losses.reconstruction import (
    TimeBucketSpec,
    acceleration_loss,
    compute_codec_losses,
    fit_time_bucket_normalization,
    velocity_loss,
)
from test_codec_training import _record


@pytest.mark.parametrize("task", ["trajectory", "static"])
def test_reconstruction_loss_and_gradient_match_old(task):
    assert torch.cuda.is_available(), "P4a loss parity requires CUDA"
    times = torch.tensor([0.0]) if task == "static" else torch.tensor([0.0, 1.0, 3.0, 6.0])
    record = _record(task=task, times=times)
    old_batch = old_collate([record]).to("cuda")
    new_batch = collate_clip_records([record]).to("cuda")
    old_pred = old_batch.x.detach().clone().requires_grad_()
    new_pred = new_batch.x.detach().clone().requires_grad_()
    old_pred.data[0, 0, 0] += 0.5
    new_pred.data[0, 0, 0] += 0.5
    if task == "trajectory":
        old_pred.data[2, 1, 1] -= 0.25
        new_pred.data[2, 1, 1] -= 0.25
    weights = {"coordinate": 1.0, "local": 0.3, "bond": 0.4, "velocity": 0.5, "acceleration": 0.6}
    old_stats = old_fit([old_batch], min_count=1, epsilon=1e-4)
    new_stats = fit_time_bucket_normalization([new_batch], min_count=1, epsilon=1e-4)
    assert old_stats.keys() == new_stats.keys()
    for bucket in old_stats:
        assert old_stats[bucket].as_dict() == new_stats[bucket].as_dict()
    expected = old_compute(old_pred, old_batch, weights=weights, normalization=old_stats)
    actual = compute_codec_losses(new_pred, new_batch, weights=weights, normalization=new_stats)
    assert expected.keys() == actual.keys()
    for name in expected:
        torch.testing.assert_close(expected[name], actual[name], rtol=0, atol=0)
    expected["total"].backward()
    actual["total"].backward()
    torch.testing.assert_close(old_pred.grad, new_pred.grad, rtol=0, atol=0)
    if task == "static":
        assert velocity_loss(new_pred, new_batch.x, new_batch).item() == 0
        assert acceleration_loss(new_pred, new_batch.x, new_batch).item() == 0


def test_loss_mask_and_normalization_fallback_match_old():
    record = _record()
    old_batch = old_collate([record])
    new_batch = collate_clip_records([record])
    old_batch.loss_mask[1] = False
    new_batch.loss_mask[1] = False
    old_stats = old_fit([old_batch], min_count=10_000, epsilon=1e-4)
    new_stats = fit_time_bucket_normalization([new_batch], min_count=10_000, epsilon=1e-4)
    assert old_stats["dt_100ps"].as_dict() == new_stats["dt_100ps"].as_dict()
    old_prediction = old_batch.x.clone()
    new_prediction = new_batch.x.clone()
    old_prediction[:, 1] += 7.0
    new_prediction[:, 1] += 7.0
    expected = old_compute(old_prediction, old_batch, normalization=old_stats)
    actual = compute_codec_losses(new_prediction, new_batch, normalization=new_stats)
    for name in expected:
        torch.testing.assert_close(expected[name], actual[name], rtol=0, atol=0)
    assert actual["total"].item() == 0.0


def test_time_bucket_spec_contract_matches_old():
    old = OldTimeBucketSpec("dt_100ps", 100.0, 1.0, 0.75)
    new = TimeBucketSpec("dt_100ps", 100.0, 1.0, 0.75)
    assert old.as_dict() == new.as_dict()
    with pytest.raises(ValueError, match="non-negative"):
        TimeBucketSpec("invalid", -1.0, 1.0)

def test_future_bond_frozen_decode_gradient_matches_old():
    from dataclasses import fields
    from types import SimpleNamespace

    from dit_test_utils import make_batch
    from module.dit_geometry_supervision import FutureBondAuxiliary as OldAuxiliary
    from module.molecular_dit import MolecularDiT as OldDiT
    from module.state_detail_latent_adapter import (
        LatentStatistics as OldStatistics,
        StateDetailLatentAdapter as OldAdapter,
    )
    from molvid.dit.model import MolecularDiT
    from molvid.geometry.types import StaticTopologyMetadata
    from molvid.latent.adapter import StateDetailLatentAdapter
    from molvid.latent.statistics import LatentStatistics
    from molvid.latent.types import LatentBatch, LatentFields
    from molvid.losses.geometry import FutureBondAuxiliary

    assert torch.cuda.is_available(), "P4b frozen-decode gate requires CUDA"
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        old_batch = make_batch(4, width=4).to("cuda")
        observed = torch.zeros_like(old_batch.token_mask)
        observed[:, 0] = True
        old_batch = old_batch.with_observation(observed)
        values = {field.name: getattr(old_batch, field.name) for field in fields(old_batch)}
        values["fields"] = LatentFields(**old_batch.fields.as_dict())
        values["topology"] = StaticTopologyMetadata(
            **{field.name: getattr(old_batch.topology, field.name) for field in fields(old_batch.topology)}
        )
        new_batch = LatentBatch(**values)
        old_stats = OldStatistics.fit([old_batch], ratio=4, provenance={"split": "train"}).to("cuda")
        new_stats = LatentStatistics.fit([new_batch], ratio=4, provenance={"split": "train"}).to("cuda")
        torch.manual_seed(101)
        old_model = OldDiT(
            adapter=OldAdapter(codec_width=4, scalar_width=8, vector_width=4, ratio=4),
            scalar_width=8, vector_width=4, depth=1, heads=2, ffn_multiplier=2,
            execution_backend="factorized_v2",
        ).cuda()
        torch.manual_seed(101)
        new_model = MolecularDiT(
            adapter=StateDetailLatentAdapter(codec_width=4, scalar_width=8, vector_width=4, ratio=4),
            scalar_width=8, vector_width=4, depth=1, heads=2, ffn_multiplier=2,
            execution_backend="factorized_v2",
        ).cuda()

        class FrozenToyCodec(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.projection = torch.nn.Linear(4, 3, bias=False)
                with torch.no_grad():
                    self.projection.weight.copy_(torch.eye(3, 4))
                for parameter in self.parameters():
                    parameter.requires_grad_(False)

            def decode(self, latent):
                blocks = self.projection(latent.state_h.float())
                return SimpleNamespace(
                    x_hat=blocks.repeat_interleave(4, dim=0)
                    + latent.sample_origin.index_select(0, latent.abid).unsqueeze(0)
                )

        codec = FrozenToyCodec().cuda()
        codec_before = {name: value.clone() for name, value in codec.state_dict().items()}
        coordinate = SimpleNamespace(
            x=torch.randn((16, old_batch.num_atoms, 3), device="cuda", generator=torch.Generator(device="cuda").manual_seed(77)),
            frame_mask=torch.ones((old_batch.batch_size, 16), device="cuda", dtype=torch.bool),
            loss_mask=old_batch.loss_mask,
            bond_index=torch.tensor([[0, 2, 3], [1, 3, 4]], device="cuda"),
            abid=old_batch.abid,
        )
        old_normalized = old_stats.normalize(old_batch)
        new_normalized = new_stats.normalize(new_batch)
        from module.latent_rectified_flow import RectifiedFlowObjective as OldFlow
        from molvid.flow.objective import RectifiedFlowObjective

        old_sample = OldFlow().sample(
            old_normalized, generator=torch.Generator(device="cuda").manual_seed(43)
        )
        new_sample = RectifiedFlowObjective().sample(
            new_normalized, generator=torch.Generator(device="cuda").manual_seed(43)
        )
        old_prediction = old_model(old_normalized.with_fields(old_sample.interpolated), old_sample.tau)
        new_prediction = new_model(new_normalized.with_fields(new_sample.interpolated), new_sample.tau)
        old_aux = OldAuxiliary(
            adapter=old_model.adapter, statistics=old_stats, codec=codec,
            coordinate_batch=coordinate, history_frames=4, lambda_bond=0.2, tau_threshold=0.0,
        )
        new_aux = FutureBondAuxiliary(
            adapter=new_model.adapter, statistics=new_stats, codec=codec,
            coordinate_batch=coordinate, history_frames=4, lambda_bond=0.2, tau_threshold=0.0,
        )
        old_loss = old_aux.raw_loss(old_prediction, old_sample, old_normalized)
        new_loss = new_aux.raw_loss(new_prediction, new_sample, new_normalized)
        torch.testing.assert_close(old_loss.loss, new_loss.loss, rtol=0, atol=0)
        assert old_loss.diagnostics() == new_loss.diagnostics()
        old_loss.loss.backward()
        new_loss.loss.backward()
        assert any(parameter.grad is not None and torch.any(parameter.grad != 0) for parameter in new_model.parameters())
        for (_, old_parameter), (_, new_parameter) in zip(old_model.named_parameters(), new_model.named_parameters()):
            assert (old_parameter.grad is None) == (new_parameter.grad is None)
            if old_parameter.grad is not None:
                torch.testing.assert_close(old_parameter.grad, new_parameter.grad, rtol=0, atol=0)
        assert all(parameter.grad is None for parameter in codec.parameters())
        for name, value in codec_before.items():
            torch.testing.assert_close(value, codec.state_dict()[name], rtol=0, atol=0)
    finally:
        torch.use_deterministic_algorithms(previous)
