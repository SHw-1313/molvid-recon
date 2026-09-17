from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch import nn

from data.clip_dataset import collate_clip_records
from trainer.codec_losses import (
    acceleration_loss,
    compute_codec_losses,
    fit_time_bucket_normalization,
    masked_coordinate_loss,
    velocity_loss,
)
from trainer.codec_trainer import (
    CODEC_CHECKPOINT_SCHEMA,
    CodecTrainConfig,
    CodecTrainer,
    PVBCodecModel,
    TimeBucketSpec,
)


def _record(*, task: str = "trajectory", times: torch.Tensor | None = None, atoms: int = 2):
    if times is None:
        times = torch.arange(4, dtype=torch.float32) * 100.0
    x = torch.zeros(times.numel(), atoms, 3)
    x[:, :, 0] = times[:, None] ** 2
    if atoms > 1:
        x[:, 1, 1] = 1.0
    return {
        "schema_version": "pvb.clip.v1",
        "sample_id": "codec",
        "task": task,
        "time_bucket_id": "static" if task == "static" else "dt_100ps",
        "time_ps": times.numpy(),
        "delta_time_ps": torch.diff(times).numpy(),
        "x": x.numpy(),
        "bpos": x.numpy(),
        "atype": torch.ones(atoms, dtype=torch.long).numpy(),
        "btype": torch.zeros(atoms, dtype=torch.long).numpy(),
        "block_id": torch.zeros(atoms, dtype=torch.long).numpy(),
        "component_id": torch.zeros(atoms, dtype=torch.long).numpy(),
        "atom_source_index": torch.arange(atoms, dtype=torch.long).numpy(),
        "atom_identity": [f"codec:{index}" for index in range(atoms)],
        "edge_mask": torch.zeros(atoms, dtype=torch.long).numpy(),
        "loss_mask": torch.ones(atoms, dtype=torch.bool).numpy(),
        "align_mask": torch.tensor([True] + [False] * (atoms - 1)).numpy(),
        "bond_index": torch.tensor([[0], [1]], dtype=torch.long).numpy()
        if atoms > 1
        else [],
    }


def test_nonuniform_physical_losses_and_static_temporal_zero():
    times = torch.tensor([0.0, 1.0, 3.0, 6.0])
    batch = collate_clip_records([_record(times=times)])
    target = batch.x.clone()
    assert velocity_loss(target, target, batch).item() == 0.0
    assert acceleration_loss(target, target, batch).item() == 0.0

    perturbed = target.clone()
    perturbed[2, 0, 0] += 0.5
    assert velocity_loss(perturbed, target, batch) > 0
    assert acceleration_loss(perturbed, target, batch) > 0

    static = collate_clip_records([_record(task="static", times=torch.tensor([0.0]))])
    static_prediction = static.x + 10.0
    losses = compute_codec_losses(
        static_prediction,
        static,
        weights={"coordinate": 1.0, "velocity": 1.0, "acceleration": 1.0},
    )
    assert losses["velocity"].item() == 0.0
    assert losses["acceleration"].item() == 0.0
    assert losses["coordinate"] > 0


def test_coordinate_mask_pair_mask_and_normalization_guards():
    batch = collate_clip_records([_record()])
    prediction = batch.x.clone()
    prediction[:, 1] += 3.0
    masked_batch = batch
    masked_batch.loss_mask[1] = False
    assert masked_coordinate_loss(prediction, batch.x, masked_batch).item() == 0.0
    stats = fit_time_bucket_normalization([batch], min_count=10_000, epsilon=1e-4)
    entry = stats["dt_100ps"]
    assert entry.used_fallback
    assert entry.coordinate_std.tolist() == [1.0, 1.0, 1.0]
    assert entry.velocity_scale == 1.0

    fitted = fit_time_bucket_normalization([batch], min_count=1, epsilon=1e-4)["dt_100ps"]
    assert not fitted.used_fallback
    assert torch.all(fitted.coordinate_std >= 1e-4)
    assert fitted.velocity_scale > 0 and fitted.acceleration_scale > 0


def test_codec_config_schedule_and_invalid_physical_unit():
    config = CodecTrainConfig(
        bucket_specs=(TimeBucketSpec("dt_100ps", 100.0, 1.0),),
        loss_schedule=((0, CodecTrainConfig().weights_at(0)),),
    )
    assert config.bucket("dt_100ps").center_ps == 100.0
    with pytest.raises(ValueError, match="canonical 'ps'"):
        CodecTrainConfig.from_mapping(
            {
                "schema_version": "pvb.codec.config.v1",
                "time": {
                    "unit": "ns",
                    "continuous_scale_ps": 100.0,
                    "buckets": [{"id": "static", "center_ps": 0, "tolerance_ps": 0}],
                    "normalization": {"min_count": 1, "epsilon": 1e-6},
                },
            }
        )


class _OffsetModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.offset = nn.Parameter(torch.tensor([1.0, -1.0, 0.5]))

    def forward(self, batch):
        return batch.x + self.offset.view(1, 1, 3)


def test_codec_trainer_optimizer_checkpoint_roundtrip(tmp_path: Path):
    batch = collate_clip_records([_record()])
    config = CodecTrainConfig(
        lr=0.1,
        max_steps=2,
        grad_clip=1.0,
        bucket_specs=(TimeBucketSpec("dt_100ps", 100.0, 1.0),),
        loss_schedule=((0, {"coordinate": 1.0}),),
        normalization_min_count=1,
    )
    trainer = CodecTrainer(_OffsetModel(), [batch], config=config, device="cpu")
    trainer.fit_normalization()
    metrics = trainer.optimizer_step(batch)
    assert trainer.step == 1
    assert metrics["total"] >= 0 and torch.isfinite(torch.tensor(metrics["total"]))
    checkpoint = trainer.save_checkpoint(tmp_path / "codec.pt")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["schema_version"] == CODEC_CHECKPOINT_SCHEMA
    assert payload["config"]["schema_version"] == "pvb.codec.config.v1"

    resumed = CodecTrainer(_OffsetModel(), [batch], config=config, device="cpu")
    resumed.load_checkpoint(checkpoint)
    assert resumed.step == 1
    assert resumed.normalization_stats["dt_100ps"].count == trainer.normalization_stats["dt_100ps"].count
    assert torch.allclose(
        resumed.model.offset.detach(), trainer.model.offset.detach(), atol=1e-7
    )
    bad = tmp_path / "bad.pt"
    torch.save({"schema_version": "old"}, bad)
    with pytest.raises(ValueError, match="unsupported codec checkpoint schema"):
        resumed.load_checkpoint(bad)

def test_codec_trainer_step_jsonl_logging(tmp_path: Path):
    batch = collate_clip_records([_record()])
    config = CodecTrainConfig(
        lr=0.1,
        max_steps=2,
        bucket_specs=(TimeBucketSpec("dt_100ps", 100.0, 1.0),),
        loss_schedule=((0, {"coordinate": 1.0}),),
        normalization_min_count=1,
    )
    trainer = CodecTrainer(_OffsetModel(), [batch], config=config, device="cpu")
    trainer.fit_normalization()
    log_path = tmp_path / "train_metrics.jsonl"
    trainer.run(log_path=log_path)
    records = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert [record["step"] for record in records] == [1, 2]
    assert all(record["schema_version"] == "pvb.codec.train.v1" for record in records)
    assert all("total" in record["metrics"] for record in records)


def test_pvb_codec_model_cpu_one_step():
    batch = collate_clip_records([_record()])
    model = PVBCodecModel(
        hidden_channels=8,
        spatial_layers=1,
        temporal_layers=0,
        temporal_ratio=2,
        num_rbf=4,
        num_heads=2,
        max_num_neighbors=4,
        neighbor_backend="dense_test",
    )
    config = CodecTrainConfig(
        lr=1e-3,
        max_steps=1,
        bucket_specs=(TimeBucketSpec("dt_100ps", 100.0, 1.0),),
        loss_schedule=((0, {"coordinate": 1.0}),),
    )
    trainer = CodecTrainer(model, [batch], config=config, device="cpu")
    metrics = trainer.optimizer_step(batch)
    assert trainer.step == 1
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())

