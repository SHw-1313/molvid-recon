from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from data.clip_dataset import collate_clip_records
from evaluation.codec_evaluation import (
    _mask,
    _metrics,
    aligned_rmsf_metrics,
    anchor_control,
    contact_metrics,
    dynamic_acf_metrics,
    evaluate_controls,
    model_control,
    report_markdown,
    write_report,
)
from tests.test_codec_training import _record


def _trajectory(bucket: str, delta: float):
    record = _record(times=torch.arange(4, dtype=torch.float32) * delta)
    record["time_bucket_id"] = bucket
    return collate_clip_records([record])


def test_controls_are_separated_by_bucket_and_frame0():
    batch_100 = _trajectory("dt_100ps", 100.0)
    batch_80 = _trajectory("dt_80ps", 80.0)
    static = collate_clip_records([_record(task="static", times=torch.tensor([0.0]))])
    perfect = model_control(
        "ratio4_temporal",
        lambda batch: batch.x.clone(),
        ratio=4,
        temporal=True,
    )
    report = evaluate_controls(
        [anchor_control(), perfect], [batch_100, batch_80, static]
    )
    assert set(report["controls"]) == {"ratio1_no_temporal", "ratio4_temporal"}
    assert "overall" not in report["controls"]["ratio4_temporal"]
    ratio4 = report["controls"]["ratio4_temporal"]
    assert set(ratio4["by_time_bucket"]) == {"dt_100ps", "dt_80ps", "static"}
    assert ratio4["by_time_bucket"]["dt_80ps"]["latent_interval_ps"] == 320.0
    assert ratio4["by_time_bucket"]["static"]["latent_interval_ps"] is None
    assert ratio4["by_time_bucket"]["dt_100ps"]["metrics"]["future"]["rmsd"] == 0.0
    assert ratio4["by_time_bucket"]["dt_100ps"]["metrics"]["frame0"]["rmsd"] == 0.0
    anchor_future = report["controls"]["ratio1_no_temporal"]["by_time_bucket"]["dt_100ps"]["metrics"]["future"]["rmsd"]
    assert anchor_future > 0.0

    metrics = ratio4["by_time_bucket"]["dt_100ps"]["metrics"]
    for key in (
        "rmsd",
        "aligned_rmsd",
        "centroid_gauge_raw_rmsd",
        "drmsd",
        "bond_rmse",
        "contact_error",
        "contact_precision",
        "contact_recall",
        "contact_f1",
        "contact_jaccard",
        "contact_false_positive_rate",
        "contact_false_negative_rate",
        "contact_occupancy_mae",
        "clash_rate",
        "torsion_change",
    ):
        assert key in metrics["frame0"] and key in metrics["future"]
    for key in ("velocity_rmse", "acceleration_rmse", "frequency_retention"):
        assert key in metrics


def test_json_and_markdown_report_roundtrip(tmp_path: Path):
    batch = _trajectory("dt_100ps", 100.0)
    report = evaluate_controls([anchor_control()], [batch])
    json_path = tmp_path / "report.json"
    markdown_path = tmp_path / "report.md"
    write_report(report, json_path, markdown_path)
    loaded = json.loads(json_path.read_text())
    assert loaded["schema_version"] == "pvb.codec.eval.v2"
    markdown = markdown_path.read_text()
    assert "dt_100ps" in markdown
    assert "no cross-bucket mean" in markdown
    assert report_markdown(report) == markdown


def _rigid_metric_batch(atoms: int = 4):
    record = _record(atoms=atoms)
    base = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )[:atoms]
    record["x"] = base.unsqueeze(0).repeat(4, 1, 1).numpy()
    record["bpos"] = record["x"].copy()
    record["align_mask"] = np.ones(atoms, dtype=np.bool_)
    record["bond_index"] = np.empty((2, 0), dtype=np.int64)
    return collate_clip_records([record])


def test_aligned_rmsd_is_rigid_transform_invariant_and_raw_is_explicit():
    batch = _rigid_metric_batch()
    angle = torch.tensor(0.41)
    rotation = torch.tensor(
        [
            [torch.cos(angle), -torch.sin(angle), 0.0],
            [torch.sin(angle), torch.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    prediction = torch.einsum("tnj,ij->tni", batch.x, rotation) + torch.tensor(
        [3.0, -2.0, 0.5]
    )
    metrics = _metrics(prediction, batch.x, batch)
    assert metrics["all_frames"]["centroid_gauge_raw_rmsd"] > 1.0
    assert metrics["all_frames"]["aligned_rmsd"] < 1.0e-5
    assert metrics["all_frames"]["rmsd"] == metrics["all_frames"]["centroid_gauge_raw_rmsd"]
    rmsf = aligned_rmsf_metrics(prediction, batch.x, batch, _mask(batch, 4, prediction.device))
    assert rmsf["absolute_error"] < 1.0e-5


def test_contact_metrics_retain_pair_identity_when_counts_match():
    batch = _rigid_metric_batch()
    target = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
            [23.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    ).unsqueeze(0).repeat(4, 1, 1)
    prediction = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [13.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    ).unsqueeze(0).repeat(4, 1, 1)
    metrics = contact_metrics(
        prediction, target, batch, _mask(batch, 4, prediction.device), [0]
    )
    assert metrics["contact_occupancy_mae"] == 4.0 / 6.0
    assert metrics["contact_precision"] == 0.0
    assert metrics["contact_recall"] == 0.0
    assert metrics["contact_f1"] == 0.0


def test_dynamic_correlation_uses_ordered_aligned_motion():
    batch = _rigid_metric_batch(atoms=2)
    target = torch.zeros(4, 2, 3)
    target[:, 1, 0] = torch.tensor([0.0, 1.0, 3.0, 2.0])
    prediction = torch.zeros_like(target)
    prediction[:, 1, 0] = torch.tensor([0.0, 3.0, 1.0, 2.0])
    batch.align_mask[1] = False
    same = dynamic_acf_metrics(target, target, batch, _mask(batch, 4, target.device))
    reordered = dynamic_acf_metrics(prediction, target, batch, _mask(batch, 4, target.device))
    assert same["dynamic_correlation"] > 0.99
    assert reordered["dynamic_correlation"] < same["dynamic_correlation"] - 0.5
    assert reordered["signal"].endswith("velocity")
    assert reordered["units"] == "angstrom_per_ps"
