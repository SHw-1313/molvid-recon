from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from tools.select_frame_gm_p2_parent import (
    ARMS,
    _sha256,
    _validate_metrics_run,
    _validate_view_grid,
    select_from_system_values,
)


def _arm_values() -> dict[str, dict[str, object]]:
    systems = [f"system_{index}" for index in range(8)]
    target = np.linspace(0.5, 1.5, 8)
    result: dict[str, dict[str, object]] = {}
    for arm in ARMS:
        result[arm] = {
            "systems": systems,
            "bond_rmse_A": np.full(8, 2.0),
            "residue_rmsf_mae_A": np.full(8, 1.0),
            "msd_curve_mae_A2": np.full(8, 3.0),
            "clash_rate": np.full(8, 0.1),
            "rmsf_prediction_A": target.copy(),
            "rmsf_target_A": target.copy(),
        }
    return result


def test_system_paired_rule_selects_clear_noninferior_improvement() -> None:
    values = _arm_values()
    values["G"]["bond_rmse_A"] = np.full(8, 1.5)
    values["M"]["msd_curve_mae_A2"] = np.full(8, 3.5)
    selected = select_from_system_values(
        values,
        {"B0": 1.0, "G": 1.2, "M": 1.1, "GM": 1.3},
        bootstrap_samples=2000,
        bootstrap_seed=7,
    )
    assert selected["selected_arm"] == "G"
    assert selected["eligible_candidates"] == ["G"]
    assert selected["arms"]["G"]["clearly_improved_main_errors"] == ["bond_rmse_A"]
    assert selected["arms"]["M"]["clearly_worsened_main_errors"] == ["msd_curve_mae_A2"]


def test_safety_worsening_blocks_otherwise_clear_candidate() -> None:
    values = _arm_values()
    values["G"]["bond_rmse_A"] = np.full(8, 1.5)
    values["G"]["clash_rate"] = np.full(8, 0.2)
    selected = select_from_system_values(
        values,
        {arm: 1.0 for arm in ARMS},
        bootstrap_samples=2000,
        bootstrap_seed=9,
    )
    assert selected["selected_arm"] == "B0"
    assert selected["arms"]["G"]["clearly_worsened_safety_errors"] == ["clash_rate"]
    assert not selected["arms"]["G"]["passes_candidate_and_safety_gates"]


def test_rmsf_spread_error_is_recomputed_per_system_bootstrap() -> None:
    values = _arm_values()
    values["G"]["bond_rmse_A"] = np.full(8, 1.5)
    target = np.asarray(values["G"]["rmsf_target_A"])
    values["G"]["rmsf_prediction_A"] = target.mean() + 5.0 * (target - target.mean())
    selected = select_from_system_values(
        values,
        {arm: 1.0 for arm in ARMS},
        bootstrap_samples=2000,
        bootstrap_seed=11,
    )
    assert selected["selected_arm"] == "B0"
    assert "rmsf_spread_error" in selected["arms"]["G"]["clearly_worsened_safety_errors"]


def test_rmsf_pearson_error_blocks_otherwise_clear_candidate() -> None:
    values = _arm_values()
    values["G"]["bond_rmse_A"] = np.full(8, 1.5)
    values["G"]["rmsf_prediction_A"] = np.asarray(values["G"]["rmsf_target_A"])[::-1]
    selected = select_from_system_values(
        values,
        {"B0": 1.0, "G": 1.1, "M": 1.2, "GM": 1.3},
        bootstrap_samples=2000,
        bootstrap_seed=13,
    )
    assert selected["selected_arm"] == "B0"
    assert "rmsf_pearson_error" in selected["arms"]["G"]["clearly_worsened_safety_errors"]


def test_multiple_candidates_use_rank_then_profile_cost_and_fail_closed_on_exact_tie() -> None:
    values = _arm_values()
    values["G"]["bond_rmse_A"] = np.full(8, 1.5)
    values["M"]["residue_rmsf_mae_A"] = np.full(8, 0.5)
    costs = {"B0": 1.0, "G": 1.2, "M": 1.1, "GM": 1.3}
    selected = select_from_system_values(
        values, costs, bootstrap_samples=2000, bootstrap_seed=15
    )
    assert selected["eligible_candidates"] == ["G", "M"]
    assert selected["arms"]["G"]["average_main_error_rank"] == selected["arms"]["M"]["average_main_error_rank"]
    assert selected["selected_arm"] == "M"
    with pytest.raises(ValueError, match="tie on both"):
        select_from_system_values(
            values,
            {**costs, "G": 1.1},
            bootstrap_samples=2000,
            bootstrap_seed=15,
        )


@pytest.mark.parametrize("bad", [math.nan, math.inf, -1.0, 0.0])
def test_profile_costs_must_be_finite_and_positive(bad: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        select_from_system_values(
            _arm_values(),
            {"B0": 1.0, "G": bad, "M": 1.1, "GM": 1.2},
            bootstrap_samples=2000,
            bootstrap_seed=17,
        )


def _grid_rows(histories: tuple[int, ...]) -> list[dict[str, str]]:
    return [
        {
            "system": "system_0",
            "lag_ps": str(lag),
            "history_frames": str(history),
            "replica_count": "3",
            "system_mean_rmsf_target_A": str(lag + history),
        }
        for lag in (100, 200, 300, 400)
        for history in histories
    ]


def test_view_grid_requires_unique_exact_keys_and_three_replicas(tmp_path: Path) -> None:
    rows = _grid_rows((4, 8))
    systems, targets = _validate_view_grid(rows, histories=(4, 8), source=tmp_path / "views.csv")
    assert systems == ["system_0"]
    assert len(targets) == 8
    duplicate = [dict(row) for row in rows]
    duplicate[-1] = dict(duplicate[0])
    with pytest.raises(ValueError, match="duplicate generated formal view"):
        _validate_view_grid(duplicate, histories=(4, 8), source=tmp_path / "views.csv")
    wrong_replica = [dict(row) for row in rows]
    wrong_replica[0]["replica_count"] = "2"
    with pytest.raises(ValueError, match="three replicas"):
        _validate_view_grid(wrong_replica, histories=(4, 8), source=tmp_path / "views.csv")


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _metric_fixture(root: Path) -> tuple[Path, Path]:
    source = root / "evaluation"
    metrics_dir = root / "metrics"
    source.mkdir(parents=True)
    metrics_dir.mkdir()
    protocol_path = source / "protocol.json"
    protocol = {
        "schema": "molvid.frame_joint.multitime_eval.v1",
        "checkpoint": {"sha256": "checkpoint"},
        "steps": 16,
        "seeds": [0, 1, 2],
        "paths": ["persistence", "clean_latent_decoder_oracle", "generated"],
        "expected_rows": 2352,
        "rows_written": 2352,
        "paired_store": {"path": "/paired/store", "index_sha256": "index"},
        "paired_manifest": {"path": "/paired/manifest", "sha256": "manifest"},
    }
    _write_json(protocol_path, protocol)
    rows_path = metrics_dir / "rows.jsonl"
    rows_path.write_text("{}\n", encoding="utf-8")
    per_system = metrics_dir / "per_system.csv"
    per_system.write_text("system\nsystem_0\n", encoding="utf-8")
    metrics_path = metrics_dir / "metrics.json"
    _write_json(metrics_path, {
        "schema": "molvid.frame_gm.metric_summary.v3",
        "test_opened": False,
        "historical_rows_mutated": False,
        "aggregation": "draw/seed mean -> anchor mean -> replica mean -> system equal",
        "source_rows": 2352,
        "source_protocol_sha256": _sha256(protocol_path),
    })
    _write_json(metrics_dir / "run_manifest.json", {
        "schema": "molvid.frame_gm.metric_recompute_run.v1",
        "row_count": 2352,
        "source_run": str(source.resolve()),
        "paired_store": "/paired/store",
        "output_rows_sha256": _sha256(rows_path),
        "metrics_sha256": _sha256(metrics_path),
        "per_system_sha256": _sha256(per_system),
    })
    return metrics_dir, protocol_path


def test_metric_provenance_hashes_and_checkpoint_fail_closed(tmp_path: Path) -> None:
    metrics_dir, _protocol_path = _metric_fixture(tmp_path)
    _validate_metrics_run(
        metrics_dir, 2352, expected_checkpoint_sha256="checkpoint"
    )
    with (metrics_dir / "per_system.csv").open("a", encoding="utf-8") as handle:
        handle.write("tampered\n")
    with pytest.raises(ValueError, match="per_system.csv SHA-256 differs"):
        _validate_metrics_run(
            metrics_dir, 2352, expected_checkpoint_sha256="checkpoint"
        )
    metrics_dir, protocol_path = _metric_fixture(tmp_path / "second")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol["checkpoint"]["sha256"] = "other"
    _write_json(protocol_path, protocol)
    metrics = json.loads((metrics_dir / "metrics.json").read_text(encoding="utf-8"))
    metrics["source_protocol_sha256"] = _sha256(protocol_path)
    _write_json(metrics_dir / "metrics.json", metrics)
    manifest = json.loads((metrics_dir / "run_manifest.json").read_text(encoding="utf-8"))
    manifest["metrics_sha256"] = _sha256(metrics_dir / "metrics.json")
    _write_json(metrics_dir / "run_manifest.json", manifest)
    with pytest.raises(ValueError, match="source evaluation checkpoint differs"):
        _validate_metrics_run(
            metrics_dir, 2352, expected_checkpoint_sha256="checkpoint"
        )
