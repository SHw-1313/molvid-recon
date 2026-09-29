import math

import pytest

from tools.evaluate_static_decoder import (
    _aggregate,
    _noise_seed,
    _select_source_records,
    _validate_sigmas,
)


def _manifest():
    records = []
    for system in ("s0", "s1"):
        for replica in ("R1", "R2"):
            for window in (0, 2):
                records.append(
                    {
                        "system": system,
                        "replica": replica,
                        "anchor_window_300": window,
                        "source_sample_id_300": f"{system}_{replica}_{window}",
                    }
                )
    return {
        "systems": ["s0", "s1"],
        "replicas": ["R1", "R2"],
        "selected_records": records,
        "test_opened": False,
    }


def test_selection_is_exact_and_sorted():
    rows = _select_source_records(_manifest(), anchor_window=0)
    assert [(row["system"], row["replica"]) for row in rows] == [
        ("s0", "R1"),
        ("s0", "R2"),
        ("s1", "R1"),
        ("s1", "R2"),
    ]
    assert all(row["anchor_window_300"] == 0 for row in rows)


def test_selection_rejects_missing_and_test_opened():
    manifest = _manifest()
    manifest["selected_records"] = manifest["selected_records"][1:]
    with pytest.raises(ValueError, match="selection differs"):
        _select_source_records(manifest, anchor_window=0)
    manifest = _manifest()
    manifest["test_opened"] = True
    with pytest.raises(ValueError, match="test_opened=false"):
        _select_source_records(manifest, anchor_window=0)


def test_noise_seed_is_record_local_and_stable():
    assert _noise_seed(7, "sample", 8, 0.1) == _noise_seed(7, "sample", 8, 0.1)
    assert _noise_seed(7, "sample", 8, 0.1) != _noise_seed(7, "sample", 8, 0.2)
    assert _noise_seed(7, "sample", 8, 0.1) != _noise_seed(7, "other", 8, 0.1)


def test_sigma_validation_rejects_invalid_values():
    assert _validate_sigmas([0, 0.05, 0.2]) == (0.0, 0.05, 0.2)
    for values in ([], [-0.1], [0.1, 0.1], [math.inf]):
        with pytest.raises(ValueError):
            _validate_sigmas(values)


def test_aggregation_is_replica_then_system_equal():
    names = (
        "input_coordinate_mse_A2",
        "output_coordinate_mse_A2",
        "output_coordinate_rmse_A",
        "output_from_input_coordinate_mse_A2",
        "bond_distance_mse_A2",
        "clash_raw_A2_per_atom",
        "unscaled_training_objective",
        "realized_rms_displacement_A",
    )

    def row(system, replica, value):
        return {
            "system": system,
            "replica": replica,
            "sigma_per_axis_A": 0.1,
            "metrics": {name: value for name in names},
        }

    per_system, summary = _aggregate(
        [row("s0", "R1", 1.0), row("s0", "R2", 3.0), row("s1", "R1", 10.0)]
    )
    assert [value["metrics"]["output_coordinate_mse_A2"] for value in per_system] == [
        2.0,
        10.0,
    ]
    assert summary["0.1"]["metrics"]["output_coordinate_mse_A2"] == 6.0
    assert summary["0.1"]["coordinate_mse_reduction_fraction"] == pytest.approx(0.0)
