"""Small CPU checks for physical Q1 selection, paired arms, and resume."""
import json
from pathlib import Path

import numpy as np
import torch

from molvid.data.kernel_windows import (KernelWindows, history_ages, native_indices,
                                       plan_digest, select_kernel_record)
from molvid.data.store import ClipMMapWriter


def fixture_record():
    x = np.arange(16 * 4 * 3, dtype=np.float32).reshape(16, 4, 3) / 20
    return dict(schema_version="pvb.clip.v1", sample_id="misato_ABCD_MD_dt_80ps_w000032",
        source="misato", system_id="ABCD", replica="MD", split="train", task="trajectory",
        coordinate_unit="angstrom", time_bucket_id="dt_80ps", native_delta_time_ps=80.,
        source_stride=1, time_ps=np.arange(16, dtype=np.float32) * 80.,
        delta_time_ps=np.full(15, 80., dtype=np.float32), x=x, bpos=x.copy(),
        atom_identity=[f"A:1::ALA:{name}:{i}" for i, name in enumerate(["N", "CA", "C", "O"])],
        atom_source_index=np.arange(4), atype=np.array([6, 5, 5, 7]), btype=np.zeros(4, dtype=int),
        block_id=np.zeros(4, dtype=int), component_id=np.zeros(4, dtype=int),
        edge_mask=np.zeros(4, dtype=int), loss_mask=np.ones(4, dtype=bool),
        align_mask=np.ones(4, dtype=bool), frame_mask=np.ones(16, dtype=bool),
        bond_index=np.array([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]]),
        topology_fingerprint="fixture")


def fixture_row(record):
    native = native_indices(record)
    return dict(plan_index=0, domain="complex", system="ABCD", visit=0,
        sample_id=record["sample_id"], replica="MD", store="unused", store_index=0,
        history_recipe="rolling", rolling_stage=2, history_ages_delta=[9, 8, 4, 0],
        planned_lag_delta=4, t0_index=10, delta_ps=80.,
        native_frame_indices=native.tolist(), absolute_time_ps=(native * 80.).tolist(),
        absolute_t0_ps=float(native[10] * 80.), band_source_sample_id=record["sample_id"],
        topology_fingerprint="fixture", extended_record=None)


def test_rolling_times_and_native_window_conventions():
    assert [history_ages(4, "rolling", stage) for stage in range(4)] == [
        [3, 2, 1, 0], [6, 5, 4, 0], [9, 8, 4, 0], [12, 8, 4, 0]]
    assert history_ages(4, "multiscale") == [8, 4, 1, 0]
    record = fixture_record()
    assert native_indices(record).tolist() == list(range(32, 48))
    atlas = dict(record, sample_id="atlas_abcd_A_R2_dt_100ps_w000002", source="atlas",
                 system_id="abcd_A", replica="R2", time_ps=np.arange(16) * 100.)
    assert native_indices(atlas).tolist() == list(range(320, 480, 10))


def test_arms_share_current_atoms_target_except_lag_ablation():
    record = fixture_record()
    row = fixture_row(record)
    variants = {mode: select_kernel_record(record, row, mode)
                for mode in ("h1", "history_mix", "dense_history", "lag1")}
    for selected, history, identity in variants.values():
        assert len(selected["x"]) == history + 1
        np.testing.assert_array_equal(selected["x"][history - 1], record["x"][10])
        assert selected["atom_identity"] == record["atom_identity"]
        np.testing.assert_array_equal(selected["bond_index"], record["bond_index"])
        assert identity["absolute_t0_ps"] == row["absolute_t0_ps"]
    for mode in ("h1", "history_mix", "dense_history"):
        np.testing.assert_array_equal(variants[mode][0]["x"][-1], record["x"][14])
    np.testing.assert_array_equal(variants["lag1"][0]["x"][-1], record["x"][11])
    # Lag1 changes target only; it must not turn the rolling4 history dense.
    np.testing.assert_array_equal(variants["lag1"][0]["x"][:-1], variants["history_mix"][0]["x"][:-1])
    assert variants["h1"][1] == 1
    assert variants["history_mix"][2]["history_ages_delta"] == [9, 8, 4, 0]
    # Modifying an unselected future cannot change history/target selection.
    changed = dict(record, x=record["x"].copy())
    changed["x"][15] += 12345
    np.testing.assert_array_equal(select_kernel_record(changed, row, "history_mix")[0]["x"],
                                  variants["history_mix"][0]["x"])


def test_resume_and_multiframe_band_are_arm_independent(tmp_path):
    record = fixture_record()
    root = tmp_path / "store"
    with ClipMMapWriter(root) as writer:
        writer.append(record)
    rows = []
    for index in range(256):
        row = fixture_row(record)
        row.update(plan_index=index, store=str(root))
        rows.append(row)
    manifest = tmp_path / "data_manifest.json"
    manifest.write_text(json.dumps(dict(status="complete", seed=71, data_config={},
        rows=rows, plan_digest=plan_digest(rows))))
    state_before = torch.get_rng_state().clone()
    one = KernelWindows(dict(manifest=str(manifest)), {}, 71)
    one.next("history_mix")
    checkpoint = one.state_dict()
    expected = one.next("history_mix")
    restored = KernelWindows(dict(manifest=str(manifest)), {}, 71)
    restored.load_state_dict(checkpoint)
    actual = restored.next("history_mix")
    assert expected[2] == actual[2]
    torch.testing.assert_close(expected[0].x, actual[0].x, rtol=0, atol=0)
    torch.testing.assert_close(actual[3], torch.from_numpy(record["x"]), rtol=0, atol=0)
    assert actual[3].shape[0] == 16 and actual[0].frames == 5
    assert not actual[3].requires_grad
    assert torch.equal(state_before, torch.get_rng_state())
    one.close()
    restored.close()
