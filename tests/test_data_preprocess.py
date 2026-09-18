from __future__ import annotations

import numpy as np

from molvid.data.batch import validate_clip_record
from molvid.data.io import read_trajectory, write_trajectory
from molvid.data.preprocess import (
    ClipPreprocessConfig,
    iterate_windows,
    make_clip_record,
    topology_metadata,
)


def _protein_topology():
    import mdtraj as md

    top = md.Topology()
    chain = top.add_chain()
    residue = top.add_residue("GLY", chain, resSeq=1)
    atoms = [
        top.add_atom("N", md.element.nitrogen, residue),
        top.add_atom("CA", md.element.carbon, residue),
        top.add_atom("C", md.element.carbon, residue),
        top.add_atom("O", md.element.oxygen, residue),
    ]
    for left, right in zip(atoms, atoms[1:]):
        top.add_bond(left, right)
    return top


def test_misato_windows_keep_physical_time_atom_order_and_masks():
    config = ClipPreprocessConfig(clip_len=2, window_stride=1, misato_native_dt_ps=80)
    config.validate()
    top = _protein_topology()
    metadata = topology_metadata(top, complex_topology=False)
    first = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 1.0]],
        dtype=np.float32,
    )
    coordinates = np.stack([first, first + 2, first + 4])
    source = read_trajectory("misato", coordinates, native_dt_ps=80.0)
    windows = list(iterate_windows(source, clip_len=2, window_stride=1))
    assert len(windows) == 2
    np.testing.assert_array_equal(windows[0][1], [0.0, 80.0])
    np.testing.assert_array_equal(windows[1][1], [80.0, 160.0])
    record = make_clip_record(
        *windows[0],
        metadata,
        sample_id="misato_A_w000000",
        source="misato",
        system_id="A",
        replica="MD",
        split="train",
        source_stride=1,
        native_dt_ps=80.0,
        timestamp_provenance="MISATO_Handoff_verified_native_dt_ps",
    )
    validate_clip_record(record)
    assert record["coordinate_unit"] == "angstrom"
    assert record["time_bucket_id"] == "dt_80ps"
    np.testing.assert_array_equal(record["atom_source_index"], [0, 1, 2, 3])
    np.testing.assert_array_equal(record["loss_mask"], [True] * 4)
    np.testing.assert_array_equal(record["align_mask"], [True] * 4)
    np.testing.assert_array_equal(record["delta_time_ps"], [80.0])
    np.testing.assert_allclose(record["x"][0].mean(axis=0), 0, atol=1e-6)
    np.testing.assert_allclose(record["x"][1], record["x"][0], atol=1e-6)


def test_atlas_xtc_converts_nm_to_angstrom_without_changing_times(tmp_path):
    import mdtraj as md

    top = _protein_topology()
    xyz = np.stack(
        [np.array([[0, 0, 0], [0.1, 0, 0], [0.1, 0.1, 0], [0, 0.1, 0.1]], dtype=np.float32) + i * 0.01
         for i in range(3)]
    )
    trajectory = md.Trajectory(xyz=xyz, topology=top, time=np.array([0, 10, 20], dtype=np.float32))
    pdb = tmp_path / "system.pdb"
    xtc = tmp_path / "trajectory.xtc"
    trajectory[0].save_pdb(str(pdb))
    trajectory.save_xtc(str(xtc))
    frames = list(
        read_trajectory(
            "atlas", xtc, topology_path=pdb, native_dt_ps=10.0, source_stride=1
        )
    )
    assert [time for _, time in frames] == [0.0, 10.0, 20.0]
    np.testing.assert_allclose(frames[0][0], xyz[0] * 10.0, atol=1e-4)
    export = tmp_path / "generated.npz"
    write_trajectory(export, np.stack([frame for frame, _ in frames]), np.array([0, 10, 20]))
    with np.load(export) as payload:
        assert str(payload["coordinate_unit"]) == "angstrom"
        np.testing.assert_array_equal(payload["time_ps"], [0.0, 10.0, 20.0])
        np.testing.assert_allclose(payload["x"][0], xyz[0] * 10.0, atol=1e-4)
