import math

import numpy as np
import pytest

from tools.evaluate_static_decoder import (
    EXPECTED_REPLICAS,
    EXPECTED_STORE_INDEX_SHA256,
    _aggregate,
    _noise_seed,
    _normalize_source_and_middle_frame,
    _require_identity,
    _select_source_records,
    _validate_row_coverage,
    _validate_sigmas,
)


def _manifest():
    records = []
    systems = tuple(f"s{index}" for index in range(8))
    for system in systems:
        for replica in EXPECTED_REPLICAS:
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
        "systems": list(systems),
        "replicas": list(EXPECTED_REPLICAS),
        "selected_records": records,
        "source_valid_index_sha256": EXPECTED_STORE_INDEX_SHA256,
        "source_valid_store": "/frozen/store/valid",
        "test_opened": False,
    }


def test_selection_is_exact_and_sorted():
    rows = _select_source_records(_manifest(), anchor_window=0)
    assert [(row["system"], row["replica"]) for row in rows] == [
        (system, replica)
        for system in (f"s{index}" for index in range(8))
        for replica in EXPECTED_REPLICAS
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


def test_selection_rejects_nonfrozen_provenance_and_protocol_shape():
    manifest = _manifest()
    manifest.pop("source_valid_index_sha256")
    with pytest.raises(ValueError, match="identity is required"):
        _select_source_records(manifest, anchor_window=0)
    manifest = _manifest()
    manifest["source_valid_index_sha256"] = "wrong"
    with pytest.raises(ValueError, match="identity differs"):
        _select_source_records(manifest, anchor_window=0)
    with pytest.raises(ValueError, match="anchor window"):
        _select_source_records(_manifest(), anchor_window=2)


def test_noise_seed_is_record_local_and_stable():
    assert _noise_seed(7, "sample", 8, 0.1) == _noise_seed(7, "sample", 8, 0.1)
    assert _noise_seed(7, "sample", 8, 0.1) != _noise_seed(7, "sample", 8, 0.2)
    assert _noise_seed(7, "sample", 8, 0.1) != _noise_seed(7, "other", 8, 0.1)


def test_legacy_valid_source_without_frame_mask_uses_middle_frame():
    frames, atoms = 4, 2
    coordinates = np.arange(frames * atoms * 3, dtype=np.float32).reshape(frames, atoms, 3)
    source, frame_index = _normalize_source_and_middle_frame(
        {
            "schema_version": "pvb.clip.v1",
            "sample_id": "legacy_valid",
            "task": "trajectory",
            "time_bucket_id": "dt_300ps",
            "time_ps": np.arange(frames, dtype=np.float32) * 300,
            "delta_time_ps": np.full(frames - 1, 300, dtype=np.float32),
            "x": coordinates,
            "bpos": coordinates.copy(),
            "atype": np.arange(atoms, dtype=np.int64),
            "btype": np.arange(atoms, dtype=np.int64),
            "block_id": np.arange(atoms, dtype=np.int64),
            "component_id": np.zeros(atoms, dtype=np.int64),
            "atom_source_index": np.arange(atoms, dtype=np.int64),
            "atom_identity": [f"A:{index}" for index in range(atoms)],
            "edge_mask": np.zeros(atoms, dtype=np.int64),
            "loss_mask": np.ones(atoms, dtype=np.bool_),
            "align_mask": np.ones(atoms, dtype=np.bool_),
            "bond_index": np.asarray([[0], [1]], dtype=np.int64),
        }
    )
    assert source["frame_mask"].tolist() == [True, True, True, True]
    assert frame_index == 2


def test_sigma_validation_rejects_invalid_values():
    assert _validate_sigmas([0, 0.05, 0.1, 0.2]) == (0.0, 0.05, 0.1, 0.2)
    for values in ([], [-0.1], [0.1, 0.1], [math.inf], [0, 0.05, 0.2]):
        with pytest.raises(ValueError):
            _validate_sigmas(values)


def test_frozen_identity_is_required_and_exact():
    _require_identity("same", "same", "item")
    with pytest.raises(ValueError, match="required"):
        _require_identity("", "expected", "item")
    with pytest.raises(ValueError, match="differs"):
        _require_identity("wrong", "expected", "item")


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


def test_row_coverage_requires_exact_8_by_3_by_4_grid():
    rows = []
    for system in (f"s{index}" for index in range(8)):
        for replica in EXPECTED_REPLICAS:
            for sigma in (0.0, 0.05, 0.1, 0.2):
                rows.append(
                    {
                        "row_key": f"{system}|{replica}|{sigma}",
                        "system": system,
                        "replica": replica,
                        "sigma_per_axis_A": sigma,
                    }
                )
    _validate_row_coverage(rows)
    with pytest.raises(RuntimeError, match="96 rows"):
        _validate_row_coverage(rows[:-1])
    duplicate = [*rows[:-1], dict(rows[0])]
    with pytest.raises(RuntimeError, match="row keys"):
        _validate_row_coverage(duplicate)
