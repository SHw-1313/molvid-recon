from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
import pytest

from tools.select_frame_gm_p2_parent import (
    ARMS,
    FROZEN_RULE,
    _sha256,
    _validate_formal_provenance,
    _validate_metrics_run,
    _validate_view_grid,
    main as selection_main,
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
    systems, targets = _validate_view_grid(
        rows, histories=(4, 8), family="legacy", source=tmp_path / "views.csv"
    )
    assert systems == ["system_0"]
    assert len(targets) == 8
    _, fixed_targets = _validate_view_grid(
        _grid_rows((4,)), histories=(4,), family="fixed", source=tmp_path / "fixed.csv"
    )
    assert len({**targets, **fixed_targets}) == 12
    assert ("legacy", "system_0", 100, 4) in targets
    assert ("fixed", "system_0", 100, 4) in fixed_targets
    duplicate = [dict(row) for row in rows]
    duplicate[-1] = dict(duplicate[0])
    with pytest.raises(ValueError, match="duplicate generated formal view"):
        _validate_view_grid(
            duplicate, histories=(4, 8), family="legacy", source=tmp_path / "views.csv"
        )
    wrong_replica = [dict(row) for row in rows]
    wrong_replica[0]["replica_count"] = "2"
    with pytest.raises(ValueError, match="three replicas"):
        _validate_view_grid(
            wrong_replica, histories=(4, 8), family="legacy", source=tmp_path / "views.csv"
        )


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _metric_fixture(root: Path) -> tuple[Path, Path]:
    source = root / "evaluation"
    metrics_dir = root / "metrics"
    paired_store = root / "paired" / "store"
    paired_manifest = root / "paired" / "manifest.json"
    source.mkdir(parents=True)
    metrics_dir.mkdir()
    paired_store.mkdir(parents=True)
    (paired_store / "index.txt").write_text("record\n", encoding="utf-8")
    paired_manifest.parent.mkdir(exist_ok=True)
    _write_json(paired_manifest, {
        "schema_version": "molvid.frame_joint.multitime.eval_views.v1",
        "history_frames": [4, 8],
        "views_per_trajectory": 16,
        "test_opened": False,
    })
    protocol_path = source / "protocol.json"
    protocol = {
        "schema": "molvid.frame_joint.multitime_eval.v1",
        "checkpoint": {"sha256": "checkpoint"},
        "steps": 16,
        "seeds": [0, 1, 2],
        "paths": ["persistence", "clean_latent_decoder_oracle", "generated"],
        "expected_rows": 2352,
        "rows_written": 2352,
        "paired_store": {
            "path": str(paired_store),
            "index_sha256": _sha256(paired_store / "index.txt"),
        },
        "paired_manifest": {
            "path": str(paired_manifest),
            "sha256": _sha256(paired_manifest),
        },
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
        "paired_store_index_sha256": _sha256(paired_store / "index.txt"),
    })
    _write_json(metrics_dir / "run_manifest.json", {
        "schema": "molvid.frame_gm.metric_recompute_run.v1",
        "row_count": 2352,
        "source_run": str(source.resolve()),
        "paired_store": str(paired_store),
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


def test_metric_provenance_accepts_relative_absolute_alias_of_same_store(
    tmp_path: Path,
) -> None:
    metrics_dir, protocol_path = _metric_fixture(tmp_path)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol["paired_store"]["path"] = os.path.relpath(
        protocol["paired_store"]["path"], Path.cwd()
    )
    _write_json(protocol_path, protocol)
    metrics = json.loads((metrics_dir / "metrics.json").read_text(encoding="utf-8"))
    metrics["source_protocol_sha256"] = _sha256(protocol_path)
    _write_json(metrics_dir / "metrics.json", metrics)
    manifest = json.loads((metrics_dir / "run_manifest.json").read_text(encoding="utf-8"))
    manifest["metrics_sha256"] = _sha256(metrics_dir / "metrics.json")
    _write_json(metrics_dir / "run_manifest.json", manifest)
    result = _validate_metrics_run(
        metrics_dir, 2352, expected_checkpoint_sha256="checkpoint"
    )
    assert Path(result["paired_data"]["store"]).resolve() == Path(
        manifest["paired_store"]
    ).resolve()


def test_metric_paired_data_identity_and_sealed_test_fail_closed(tmp_path: Path) -> None:
    metrics_dir, protocol_path = _metric_fixture(tmp_path)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    index_path = Path(protocol["paired_store"]["path"]) / "index.txt"
    index_path.write_text("replaced\n", encoding="utf-8")
    with pytest.raises(ValueError, match="paired store index SHA-256 differs"):
        _validate_metrics_run(
            metrics_dir, 2352, expected_checkpoint_sha256="checkpoint"
        )
    metrics_dir, protocol_path = _metric_fixture(tmp_path / "manifest")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    manifest_path = Path(protocol["paired_manifest"]["path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["test_opened"] = True
    _write_json(manifest_path, manifest)
    protocol["paired_manifest"]["sha256"] = _sha256(manifest_path)
    _write_json(protocol_path, protocol)
    metrics = json.loads((metrics_dir / "metrics.json").read_text(encoding="utf-8"))
    metrics["source_protocol_sha256"] = _sha256(protocol_path)
    _write_json(metrics_dir / "metrics.json", metrics)
    run_manifest = json.loads(
        (metrics_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    run_manifest["metrics_sha256"] = _sha256(metrics_dir / "metrics.json")
    _write_json(metrics_dir / "run_manifest.json", run_manifest)
    with pytest.raises(ValueError, match="opened sealed test"):
        _validate_metrics_run(
            metrics_dir, 2352, expected_checkpoint_sha256="checkpoint"
        )


def _formal_fixture(root: Path) -> tuple[Path, Path]:
    formal_root = root / "P2" / "formal"
    evidence_root = root / "P2" / "r02" / "evidence"
    profile_root = root / "P2" / "r02" / "profile"
    formal_root.mkdir(parents=True)
    evidence_root.mkdir(parents=True)
    profile_root.mkdir(parents=True)
    candidate = "candidate"
    parent_sha = "parent"
    data_hash = "data"
    configs: dict[str, dict[str, str]] = {}
    manifest_arms: dict[str, dict[str, object]] = {}
    switches = {
        "B0": (False, False),
        "G": (True, False),
        "M": (False, True),
        "GM": (True, True),
    }
    for index, arm in enumerate(ARMS):
        geometry, motion = switches[arm]
        config = {
            "schema": "molvid.frame_joint.train.v1",
            "data": {"expected_data_hash": data_hash},
            "codec": {"checkpoint": "codec", "sha256": "codec-sha"},
            "statistics": {"path": "statistics"},
            "model": {
                "geometry_enabled": geometry,
                "motion_enabled": motion,
            },
            "loss": {"inherit_parent": True},
            "training": {
                "seed": 7,
                "precision": "bf16",
                "history_order": [4, 8],
                "time_bucket_weights": {"legacy": 0.75, "fixed": 0.25},
                "stages": [{
                    "name": "continuation",
                    "updates": 10,
                    "learning_rates": {
                        "history_encoder": 0.0001,
                        "dit": 0.0002,
                        "decoder": 0.0001,
                    },
                }],
            },
        }
        config_path = root / f"{arm}.yaml"
        _write_json(config_path, config)
        arm_root = formal_root / arm
        arm_root.mkdir()
        checkpoint = arm_root / "frame_joint_step_00000010.pt"
        checkpoint.write_bytes(f"checkpoint-{arm}".encode())
        _write_json(arm_root / "resolved_config.json", {
            **config,
            "resolved": {
                "total_updates": 10,
                "continuation_parent": {"sha256": parent_sha},
                "data_hash": data_hash,
            },
        })
        _write_json(arm_root / "exposure.json", {
            "schema": "molvid.frame_joint.exposure.v2",
            "successful_updates": 10,
            "trajectory_clip_exposure": {"trajectory": 2},
            "bucket_clip_exposure": {"legacy": 6, "fixed": 2},
            "view_history_clip_exposure": {"legacy_uniform|H4": 4},
            "valid_atom_frames": 100,
        })
        _write_json(arm_root / "sampler_manifest.json", {
            "schema": "molvid.frame_gm.p2_sampler_schedule.v1",
            "global_batch_schedule_hash": "schedule",
            "selected_sample_ids_hash": "samples",
            "test_opened": False,
        })
        _write_json(arm_root / "target_encoder_provenance.json", {
            "code_commit": candidate,
        })
        profile = {
            "schema": "molvid.frame_gm.p2_largest_sample_profile.v1",
            "arm": arm,
            "test_opened": False,
            "parent_sha256": parent_sha,
            "data_hash": data_hash,
            "step_seconds": float(index + 1),
        }
        _write_json(profile_root / f"{arm}.json", profile)
        configs[arm] = {
            "path": str(config_path),
            "sha256": _sha256(config_path),
        }
        manifest_arms[arm] = {
            "status": "COMPLETE",
            "config": str(config_path),
            "config_sha256": _sha256(config_path),
            "final_checkpoint_sha256": _sha256(checkpoint),
        }
    resolved_path = evidence_root / "resolved_experiment.json"
    _write_json(resolved_path, {
        "selection_rule": FROZEN_RULE,
        "candidate_commit": candidate,
        "parent": {"sha256": parent_sha},
        "data": {"derived_data_hash": data_hash},
        "formal_training": {"configs": configs},
        "profile_budget": {
            "step_seconds": {
                arm: float(index + 1) for index, arm in enumerate(ARMS)
            },
        },
    })
    _write_json(formal_root / "run_manifest.json", {
        "schema": "molvid.frame_gm.p2_formal_run.v1",
        "status": "COMPLETE",
        "sealed_test_opened": False,
        "candidate_commit": candidate,
        "parent": {"sha256": parent_sha},
        "successful_updates_per_arm": 10,
        "arms": manifest_arms,
    })
    return formal_root, resolved_path


def test_formal_provenance_binds_runtime_config_and_symmetric_schedule(
    tmp_path: Path,
) -> None:
    formal_root, resolved_path = _formal_fixture(tmp_path)
    _validate_formal_provenance(formal_root, resolved_path, expected_step=10)

    runtime_path = formal_root / "G" / "resolved_config.json"
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    runtime["model"]["geometry_enabled"] = False
    _write_json(runtime_path, runtime)
    with pytest.raises(ValueError, match="runtime config differs"):
        _validate_formal_provenance(formal_root, resolved_path, expected_step=10)

    formal_root, resolved_path = _formal_fixture(tmp_path / "exposure")
    exposure_path = formal_root / "M" / "exposure.json"
    exposure = json.loads(exposure_path.read_text(encoding="utf-8"))
    exposure["valid_atom_frames"] += 1
    _write_json(exposure_path, exposure)
    with pytest.raises(ValueError, match="schedule or exposure differs"):
        _validate_formal_provenance(formal_root, resolved_path, expected_step=10)

    formal_root, resolved_path = _formal_fixture(tmp_path / "sampler")
    sampler_path = formal_root / "GM" / "sampler_manifest.json"
    sampler = json.loads(sampler_path.read_text(encoding="utf-8"))
    sampler["selected_sample_ids_hash"] = "different"
    _write_json(sampler_path, sampler)
    with pytest.raises(ValueError, match="schedule or exposure differs"):
        _validate_formal_provenance(formal_root, resolved_path, expected_step=10)


def _write_metric_arm(
    root: Path,
    *,
    arm: str,
    family: str,
    checkpoint_sha256: str,
    override_legacy_h4: bool = False,
) -> Path:
    expected_rows = 2352 if family == "legacy" else 1392
    histories = (4, 8) if family == "legacy" else (4,)
    metrics_dir = root / f"{family}_metrics" / arm
    source_run = root / f"{family}_eval" / arm
    paired_root = root / "paired" / family
    store = paired_root / "store"
    manifest_path = paired_root / "manifest.json"
    metrics_dir.mkdir(parents=True)
    source_run.mkdir(parents=True)
    if not store.exists():
        store.mkdir(parents=True)
        (store / "index.txt").write_text(f"{family}-record\n", encoding="utf-8")
        if family == "legacy":
            paired_manifest = {
                "schema_version": "molvid.frame_joint.multitime.eval_views.v1",
                "history_frames": [4, 8],
                "views_per_trajectory": 16,
                "test_opened": False,
            }
        else:
            paired_manifest = {
                "schema": "molvid.frame_gm.fixed_history_views.v1",
                "view_kind": "fixed_history_h4_horizon_1200ps",
                "observed_relative_time_ps": [-300, -200, -100, 0],
                "test_opened": False,
            }
        _write_json(manifest_path, paired_manifest)
    protocol = {
        "schema": "molvid.frame_joint.multitime_eval.v1",
        "checkpoint": {"sha256": checkpoint_sha256},
        "steps": 16,
        "seeds": [0, 1, 2],
        "paths": ["persistence", "clean_latent_decoder_oracle", "generated"],
        "expected_rows": expected_rows,
        "rows_written": expected_rows,
        "paired_store": {
            "path": str(store),
            "index_sha256": _sha256(store / "index.txt"),
        },
        "paired_manifest": {
            "path": str(manifest_path),
            "sha256": _sha256(manifest_path),
        },
    }
    protocol_path = source_run / "protocol.json"
    _write_json(protocol_path, protocol)
    rows_path = metrics_dir / "rows.jsonl"
    rows_path.write_text("{}\n", encoding="utf-8")
    fieldnames = [
        "path", "clock", "system", "lag_ps", "history_frames", "replica_count",
        "system_mean_rmsf_target_A", "bond_rmse_A", "residue_rmsf_mae_A",
        "msd_curve_mae_A2", "clash_rate", "system_mean_rmsf_prediction_A",
    ]
    per_system_path = metrics_dir / "per_system.csv"
    with per_system_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for system_index in range(8):
            system = f"system_{system_index}"
            base_target = 0.5 + 0.1 * system_index
            for lag in (100, 200, 300, 400):
                for history in histories:
                    target = base_target
                    if (
                        override_legacy_h4
                        and family == "legacy"
                        and system_index == 0
                        and lag == 100
                        and history == 4
                    ):
                        target += 1.0
                    writer.writerow({
                        "path": "generated",
                        "clock": "true",
                        "system": system,
                        "lag_ps": lag,
                        "history_frames": history,
                        "replica_count": 3,
                        "system_mean_rmsf_target_A": target,
                        "bond_rmse_A": 2.0,
                        "residue_rmsf_mae_A": 1.0,
                        "msd_curve_mae_A2": 3.0,
                        "clash_rate": 0.1,
                        "system_mean_rmsf_prediction_A": base_target,
                    })
    metrics_path = metrics_dir / "metrics.json"
    _write_json(metrics_path, {
        "schema": "molvid.frame_gm.metric_summary.v3",
        "test_opened": False,
        "historical_rows_mutated": False,
        "aggregation": "draw/seed mean -> anchor mean -> replica mean -> system equal",
        "source_rows": expected_rows,
        "source_protocol_sha256": _sha256(protocol_path),
        "paired_store_index_sha256": _sha256(store / "index.txt"),
    })
    _write_json(metrics_dir / "run_manifest.json", {
        "schema": "molvid.frame_gm.metric_recompute_run.v1",
        "row_count": expected_rows,
        "source_run": str(source_run),
        "paired_store": str(store),
        "output_rows_sha256": _sha256(rows_path),
        "metrics_sha256": _sha256(metrics_path),
        "per_system_sha256": _sha256(per_system_path),
    })
    return metrics_dir


def _selection_fixture(
    root: Path,
    *,
    override_g_legacy_h4: bool = False,
) -> tuple[Path, Path, Path, Path]:
    formal_root, resolved_path = _formal_fixture(root)
    for arm in ARMS:
        checkpoint = formal_root / arm / "frame_joint_step_00000010.pt"
        _write_metric_arm(
            root,
            arm=arm,
            family="legacy",
            checkpoint_sha256=_sha256(checkpoint),
            override_legacy_h4=override_g_legacy_h4 and arm == "G",
        )
        _write_metric_arm(
            root,
            arm=arm,
            family="fixed",
            checkpoint_sha256=_sha256(checkpoint),
        )
    return (
        formal_root,
        resolved_path,
        root / "legacy_metrics",
        root / "fixed_metrics",
    )


def test_selection_main_rejects_single_arm_legacy_h4_target_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    formal_root, resolved_path, legacy_root, fixed_root = _selection_fixture(
        tmp_path / "valid"
    )
    output = tmp_path / "valid" / "selection.json"
    monkeypatch.setattr(sys, "argv", [
        "select_frame_gm_p2_parent.py",
        "--legacy-root", str(legacy_root),
        "--fixed-root", str(fixed_root),
        "--formal-root", str(formal_root),
        "--resolved-experiment", str(resolved_path),
        "--output", str(output),
        "--expected-step", "10",
        "--bootstrap-samples", "1000",
    ])
    assert selection_main() == 0
    assert json.loads(output.read_text(encoding="utf-8"))["selected_arm"] == "B0"

    formal_root, resolved_path, legacy_root, fixed_root = _selection_fixture(
        tmp_path / "mismatch", override_g_legacy_h4=True
    )
    monkeypatch.setattr(sys, "argv", [
        "select_frame_gm_p2_parent.py",
        "--legacy-root", str(legacy_root),
        "--fixed-root", str(fixed_root),
        "--formal-root", str(formal_root),
        "--resolved-experiment", str(resolved_path),
        "--output", str(tmp_path / "mismatch" / "selection.json"),
        "--expected-step", "10",
        "--bootstrap-samples", "1000",
    ])
    with pytest.raises(ValueError, match="per-view RMSF targets are not paired for G"):
        selection_main()
