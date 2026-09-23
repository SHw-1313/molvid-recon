from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from tools.summarize_frame_gm_p3_formal import _validate_metrics, validate_paired_configs


def _configs() -> tuple[dict[str, object], dict[str, object]]:
    base: dict[str, object] = {
        "data": {"expected_data_hash": "data"},
        "codec": {"sha256": "codec"},
        "statistics": {"statistics_hash": "stats"},
        "model": {"geometry_enabled": True, "motion_enabled": False},
        "loss": {"inherit_parent": True},
        "training": {
            "deterministic": True,
            "seed": 7,
            "output_root": "J0",
            "stages": [{"name": "continuation", "updates": 16}],
        },
    }
    j0 = deepcopy(base)
    j1 = deepcopy(base)
    j1["training"]["output_root"] = "J1"
    j1["sampled_auxiliary"] = {
        "enabled": True,
        "cadence": 8,
        "draws": 2,
        "euler_steps": 4,
        "activation_checkpoint": True,
        "energy_weight": 0.25,
        "observed_bond_weight": 0.5,
        "feature_scales": {
            "residue_rmsf_A": 1.0,
            "displacement_squared_A2": 2.0,
            "internal_distance_increment_A": 3.0,
            "internal_distance_increment_product_A2": 4.0,
        },
        "calibration_batches": 8,
        "energy_gradient_target_ratio": 0.05,
        "observed_bond_gradient_target_ratio": 0.05,
    }
    return j0, j1


def test_p3_formal_configs_allow_only_sampled_auxiliary_and_output_difference() -> None:
    j0, j1 = _configs()
    validate_paired_configs(j0, j1, expected_step=16)


@pytest.mark.parametrize("section", ["data", "model", "loss"])
def test_p3_formal_configs_reject_asymmetric_main_recipe(section: str) -> None:
    j0, j1 = _configs()
    j1[section] = {"different": True}
    with pytest.raises(ValueError, match=section):
        validate_paired_configs(j0, j1, expected_step=16)


def test_p3_formal_configs_require_numeric_calibrated_contract() -> None:
    j0, j1 = _configs()
    j1["sampled_auxiliary"]["energy_weight"] = "calibrate"
    with pytest.raises(ValueError, match="resolved numeric"):
        validate_paired_configs(j0, j1, expected_step=16)
    _, j1 = _configs()
    j1["sampled_auxiliary"]["feature_scales"] = "fit"
    with pytest.raises(ValueError, match="feature scales"):
        validate_paired_configs(j0, j1, expected_step=16)


def test_p3_formal_configs_require_exact_steps_and_frozen_cadence() -> None:
    j0, j1 = _configs()
    with pytest.raises(ValueError, match="update count"):
        validate_paired_configs(j0, j1, expected_step=17)
    j0, j1 = _configs()
    j1["sampled_auxiliary"]["cadence"] = 4
    with pytest.raises(ValueError, match="frozen P3 contract"):
        validate_paired_configs(j0, j1, expected_step=16)


def _write_metrics(path: Path, *, arm: str) -> None:
    rows = [
        {
            "step": step,
            "loss": 1.0 / step,
            "sampled_branch_active": arm == "J1" and step % 8 == 0,
        }
        for step in range(1, 17)
    ]
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_p3_formal_metrics_require_exact_rows_and_sampled_cadence(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    _write_metrics(path, arm="J1")
    result = _validate_metrics(path, arm="J1", expected_step=16)
    assert result["row_count"] == 16
    assert result["sampled_active_count"] == 2
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows[7]["sampled_branch_active"] = False
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="sampled cadence"):
        _validate_metrics(path, arm="J1", expected_step=16)


def test_p3_formal_metrics_reject_nonfinite_values(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    _write_metrics(path, arm="J0")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows[-1]["loss"] = float("nan")
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="non-finite"):
        _validate_metrics(path, arm="J0", expected_step=16)
