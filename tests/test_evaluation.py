"""New evaluation boundary preserves old metrics and separates truth from sampling."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from evaluation.dit_evaluation import trajectory_metrics as old_trajectory_metrics
from evaluation.dit_reassessment import aggregate_generated_rows as old_aggregate_rows
from molvid.evaluation.latent import latent_field_errors
from molvid.evaluation.report import (
    aggregate_by_system, aggregate_by_time_bucket, compare_runs, plot_report, write_report,
)
from molvid.evaluation.runner import EvalConfig, RolloutTrack, evaluate_codec, evaluate_generation, evaluate_rollout, trajectory_metrics
from molvid.generation import sample_clip
from test_generation import _fixture


@pytest.fixture(autouse=True)
def deterministic_cuda():
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(previous)


def test_trajectory_metrics_match_old_forecast_intervals():
    batch, _, _, _, _ = _fixture()
    target = batch.x
    prediction = target.clone()
    prediction[4:, 1, 0] += 0.3
    for history in (None, 0, 4, 8):
        assert trajectory_metrics(prediction, target, batch, history_frames=history) == old_trajectory_metrics(
            prediction, target, batch, history_frames=history
        )


def test_evaluate_generation_reports_oracle_gap_and_never_conditions_on_truth_future():
    assert torch.cuda.is_available(), "P4c evaluation gate requires CUDA"
    batch, codec, model, adapter, statistics = _fixture()
    config = EvalConfig(history_frames=8, steps=8, seed=17)
    first = evaluate_generation(
        codec, model, adapter, statistics, batch,
        config=config, codec_hash="codec", data_hash="data",
    )
    altered = replace(batch, x=batch.x.clone(), bpos=batch.bpos.clone())
    altered.x[8:] += 50
    altered.bpos[8:] -= 50
    second = evaluate_generation(
        codec, model, adapter, statistics, altered,
        config=config, codec_hash="codec", data_hash="data",
    )
    generated, metadata = sample_clip(
        codec, model, adapter, statistics,
        template=batch, prefix_coordinates=batch.x[:8], history_frames=8,
        steps=8, seed=17, codec_hash="codec", data_hash="data",
    )
    assert first["generation"] == second["generation"] == metadata
    assert first["generated_result"]["future"]["aligned_rmsd"] != second["generated_result"]["future"]["aligned_rmsd"]
    assert first["codec_oracle"]["future"]["aligned_rmsd"] >= 0
    assert first["generation_gap"]["future"]["aligned_rmsd"] == (
        first["generated_result"]["future"]["aligned_rmsd"]
        - first["codec_oracle"]["future"]["aligned_rmsd"]
    )
    assert torch.equal(generated[:8], batch.x[:8].cuda())
    assert evaluate_codec(codec, batch, device="cuda")["future"]["aligned_rmsd"] >= 0


def test_rollout_reference_is_evaluation_only():
    batch, codec, model, adapter, statistics = _fixture()
    reference = torch.cat((batch.x, batch.x[8:]), dim=0)
    track = RolloutTrack(batch, reference, "tiny", "R0")
    first = evaluate_rollout(
        codec, model, adapter, statistics, track,
        seeds=(17, 19), steps=8, codec_hash="codec", data_hash="data",
    )
    changed = reference.clone()
    changed[8:] += 30
    second = evaluate_rollout(
        codec, model, adapter, statistics, RolloutTrack(batch, changed, "tiny", "R0"),
        seeds=(17, 19), steps=8, codec_hash="codec", data_hash="data",
    )
    torch.testing.assert_close(first["prediction"], second["prediction"], rtol=0, atol=0)
    assert first["rows"][0]["metrics"]["future"]["rmsd"] != second["rows"][0]["metrics"]["future"]["rmsd"]
    assert first["prefix_source"] == "own_previous_generated_eight"
    assert [row["target_future_global_interval"] for row in first["rows"]] == [[8, 16], [16, 24]]


def test_latent_field_errors_exclude_observed_and_keep_valid_counts():
    from dit_test_utils import make_batch
    from test_flow import _new_batch

    old = make_batch(4, width=4)
    observed = torch.zeros_like(old.token_mask)
    observed[:, :2] = True
    batch = _new_batch(old.with_observation(observed))
    changed = batch.fields.clone()
    changed.state_h[:2] += 100
    unchanged = latent_field_errors(changed, batch.fields, batch)
    assert unchanged["total_mse"] == 0.0
    changed.state_h[2:] += 1
    result = latent_field_errors(changed, batch.fields, batch)
    assert result["fields"]["state_h"]["mse"] > 0
    assert result["fields"]["state_h"]["valid_elements"] > 0


def test_report_aggregation_matches_old_and_keeps_time_buckets_separate(tmp_path):
    rows = [
        {"system": "A", "sample_id": "a1", "draw": 0, "time_bucket_id": "dt_100ps", "metrics": {"future": {"aligned_rmsd": 1.0}}},
        {"system": "A", "sample_id": "a1", "draw": 1, "time_bucket_id": "dt_100ps", "metrics": {"future": {"aligned_rmsd": 3.0}}},
        {"system": "B", "sample_id": "b1", "draw": 0, "time_bucket_id": "dt_1ns", "metrics": {"future": {"aligned_rmsd": 4.0}}},
    ]
    assert aggregate_by_system(rows) == old_aggregate_rows(rows)
    buckets = aggregate_by_time_bucket(rows)
    assert set(buckets) == {"dt_100ps", "dt_1ns"}
    assert buckets["dt_100ps"]["system_equal"]["aligned_rmsd"] == 2.0
    report = {
        "codec_oracle": {"future": {"aligned_rmsd": 1.0}},
        "generated_result": {"future": {"aligned_rmsd": 2.0}},
        "by_time_bucket": buckets,
    }
    assert compare_runs(report["codec_oracle"], report["generated_result"])["generation_gap"]["future"]["aligned_rmsd"] == 1.0
    json_path, markdown_path = write_report(report, tmp_path / "report.json", tmp_path / "report.md")
    assert json_path.exists() and "Codec oracle" in markdown_path.read_text()
    assert plot_report(report, tmp_path / "report.png").exists()
