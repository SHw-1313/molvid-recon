from __future__ import annotations

import pytest

from molvid.evaluation.motion import (
    arm_rmsf_diagnostics,
    motion_arm_report,
    paired_motion_report,
    read_rmsf_value,
    MotionMetricError,
)


def _summary(*rows: dict) -> dict:
    return {"aggregates": {"valid": {"H4": {"system_rows": list(rows)}}}}


def _row(system: str, prediction: object, target: object = 1.0) -> dict:
    return {
        "system": system,
        "future": {"rmsf.prediction": prediction, "rmsf.target": target},
    }


def test_flat_rmsf_and_zero_prediction_are_valid() -> None:
    row = _row("s0", 0.0, 2.0)
    read = read_rmsf_value(row, "prediction", context="test")
    assert read.value == 0.0
    assert read.reason is None
    arm = motion_arm_report(_summary(row), 4, label="candidate")
    assert arm["values"]["s0"]["prediction"] == 0.0
    assert arm["invalid_systems"] == []


def test_legacy_nested_rmsf_is_supported() -> None:
    row = {"system": "s0", "future": {"rmsf": {"prediction": 0.25, "target": 0.5}}}
    read = read_rmsf_value(row, "prediction")
    assert read.value == 0.25
    assert read.source == "legacy_nested"


def test_flat_and_nested_conflict_is_an_error() -> None:
    row = {
        "system": "s0",
        "future": {"rmsf.prediction": 0.25, "rmsf": {"prediction": 0.5}},
    }
    with pytest.raises(MotionMetricError, match="conflicting RMSF prediction"):
        read_rmsf_value(row, "prediction", context="s0")


def test_equal_flat_and_nested_values_are_explicitly_accepted() -> None:
    row = {
        "system": "s0",
        "future": {"rmsf.prediction": 0.25, "rmsf": {"prediction": 0.25}},
    }
    read = read_rmsf_value(row, "prediction")
    assert read.value == 0.25
    assert read.source == "flat+legacy_equal"


def test_malformed_legacy_mapping_is_reported() -> None:
    read = read_rmsf_value({"future": {"rmsf": 0.25}}, "prediction")
    assert read.value is None
    assert "malformed legacy" in (read.reason or "")


def test_missing_and_nan_are_incomplete_not_pass() -> None:
    candidate = motion_arm_report(
        _summary(_row("s0", float("nan")), _row("s1", 0.2)), 4, label="candidate"
    )
    reference = motion_arm_report(
        _summary(_row("s0", 1.0), _row("s1", 1.0), _row("s2", 1.0)),
        4,
        label="reference",
    )
    result = paired_motion_report(
        candidate,
        reference,
        candidate_label="candidate",
        reference_label="reference",
    )
    assert result["status"] == "INCOMPLETE"
    assert result["median_motion_ratio"] == 0.2
    assert result["missing_candidate_systems"] == ["s2"]
    reasons = {item["system"]: item["reasons"] for item in result["noncomputable_systems"]}
    assert any("NaN" in reason for reason in reasons["s0"])
    assert any("missing" in reason for reason in reasons["s2"])


def test_missing_target_is_recorded_even_when_prediction_ratio_exists() -> None:
    candidate = motion_arm_report(
        _summary({"system": "s0", "future": {"rmsf.prediction": 0.5}}),
        4,
        label="candidate",
    )
    reference = motion_arm_report(_summary(_row("s0", 1.0)), 4, label="reference")
    result = paired_motion_report(
        candidate,
        reference,
        candidate_label="candidate",
        reference_label="reference",
    )
    assert result["status"] == "INCOMPLETE"
    assert result["noncomputable_systems"][0]["system"] == "s0"
    assert "missing future['rmsf.target']" in result["noncomputable_systems"][0]["reasons"][0]


def test_near_zero_reference_denominator_is_not_applicable() -> None:
    candidate = motion_arm_report(_summary(_row("s0", 0.0)), 4, label="candidate")
    reference = motion_arm_report(_summary(_row("s0", 0.0)), 4, label="reference")
    result = paired_motion_report(
        candidate,
        reference,
        candidate_label="candidate",
        reference_label="reference",
    )
    assert result["status"] == "NOT_APPLICABLE"
    assert "near zero" in result["noncomputable_systems"][0]["reasons"][0]


def test_mixed_near_zero_and_valid_pairs_are_incomplete() -> None:
    candidate = motion_arm_report(
        _summary(_row("s0", 0.1), _row("s1", 0.5)), 4, label="candidate"
    )
    reference = motion_arm_report(
        _summary(_row("s0", 0.0), _row("s1", 1.0)), 4, label="reference"
    )
    result = paired_motion_report(
        candidate,
        reference,
        candidate_label="candidate",
        reference_label="reference",
    )
    assert result["status"] == "INCOMPLETE"
    assert result["motion_ratio_system_count"] == 1
    assert result["noncomputable_systems"][0]["system"] == "s0"


def test_threshold_fail_and_diagnostic_ratio_definitions() -> None:
    candidate = motion_arm_report(
        _summary(_row("s0", 0.0, 2.0), _row("s1", 0.2, 1.0)), 4, label="candidate"
    )
    reference = motion_arm_report(
        _summary(_row("s0", 1.0, 2.0), _row("s1", 1.0, 1.0)), 4, label="reference"
    )
    result = paired_motion_report(
        candidate,
        reference,
        candidate_label="candidate",
        reference_label="reference",
    )
    assert result["status"] == "FAIL"
    assert result["median_motion_ratio"] == 0.1
    diagnostics = arm_rmsf_diagnostics(candidate)
    assert diagnostics["ratio_of_means_prediction_over_target"] == pytest.approx(0.1 / 1.5)
    assert diagnostics["mean_of_per_system_prediction_target_ratios"] == pytest.approx(0.1)
    assert diagnostics["ratio_of_means_prediction_over_target"] != diagnostics[
        "mean_of_per_system_prediction_target_ratios"
    ]


def test_geometry_and_temporal_metrics_match_old_implementation():
    import torch
    from data.clip_dataset import collate_clip_records as old_collate
    from evaluation.codec_evaluation import _metrics as old_metrics
    from molvid.data.batch import collate_clip_records
    from molvid.evaluation.geometry import (
        _metrics, bond_errors, clash_statistics, contact_scores, coordinate_errors,
    )
    from molvid.evaluation.motion import acceleration_metrics, fft_retention, rmsf, velocity_metrics
    from test_migration import _record

    record = _record()
    old_batch = old_collate([record])
    new_batch = collate_clip_records([record])
    prediction = new_batch.x.clone()
    prediction[1:, 1, 0] += 0.5
    prediction[2:, 2, 1] -= 0.2
    for history in (None, 0, 4):
        assert _metrics(prediction, new_batch.x, new_batch, history_frames=history) == old_metrics(
            prediction, old_batch.x, old_batch, history_frames=history
        )
    frames = [1, 2, 3]
    one = _metrics(prediction, new_batch.x, new_batch)["future"]
    assert coordinate_errors(prediction, new_batch.x, new_batch, frames)["aligned_rmsd"] == one["aligned_rmsd"]
    assert bond_errors(prediction, new_batch.x, new_batch, frames)["bond_rmse"] == one["bond_rmse"]
    assert contact_scores(prediction, new_batch.x, new_batch, frames)["contact_f1"] == one["contact_f1"]
    assert clash_statistics(prediction, new_batch, frames)["clash_rate"] == one["clash_rate"]
    assert rmsf(prediction, new_batch.x, new_batch)["prediction"] >= 0
    temporal = _metrics(prediction, new_batch.x, new_batch, history_frames=0)["temporal"]["future"]
    assert velocity_metrics(prediction, new_batch.x, new_batch, list(range(4)))["velocity_rmse"] == temporal["velocity_rmse"]
    assert acceleration_metrics(prediction, new_batch.x, new_batch, list(range(4)))["acceleration_rmse"] == temporal["acceleration_rmse"]
    assert fft_retention(prediction, new_batch.x, new_batch, list(range(4))) == temporal["frequency_retention"]


def test_cuda_metric_parity_on_observed_and_future_intervals():
    import torch
    from data.clip_dataset import collate_clip_records as old_collate
    from evaluation.codec_evaluation import _metrics as old_metrics
    from molvid.data.batch import collate_clip_records
    from molvid.evaluation.geometry import _metrics
    from test_migration import _record

    assert torch.cuda.is_available(), "P4c metric numerical gate requires CUDA"
    record = _record()
    old_batch = old_collate([record]).to("cuda")
    new_batch = collate_clip_records([record]).to("cuda")
    prediction = new_batch.x.clone()
    prediction[1:, 1, 0] += 0.5
    for history in (None, 0, 4):
        assert _metrics(prediction, new_batch.x, new_batch, history_frames=history) == old_metrics(
            prediction, old_batch.x, old_batch, history_frames=history
        )
