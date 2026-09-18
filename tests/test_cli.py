"""Flat CLI routing and preprocess manifest parity."""

from __future__ import annotations

from pathlib import Path

import pytest

from data.trajectory_clips import _inventory as old_inventory
from data.trajectory_clips import write_split_manifest as old_write_manifest
from molvid.cli.preprocess import _choose_systems, main as preprocess_main
from molvid.data.manifest import source_inventory, write_split_manifest
from molvid.data.preprocess import ClipPreprocessConfig
from scripts.preprocess_trajectory_clips import _choose_systems as old_choose_systems


def test_preprocess_selection_and_manifest_match_old(tmp_path: Path):
    (tmp_path / "atlas_ids.txt").write_text("A\nB\nC\nD\n", encoding="utf-8")
    chosen = _choose_systems("atlas", tmp_path, fraction=1.0, seed=17, max_systems=None)
    assert chosen == old_choose_systems("atlas", tmp_path, fraction=1.0, seed=17, max_systems=None)
    paths = [tmp_path / "atlas_ids.txt"]
    inventory = source_inventory(paths, tmp_path)
    assert inventory == old_inventory(paths, tmp_path)
    args = dict(
        source="atlas", raw_root=tmp_path,
        config=ClipPreprocessConfig(clip_len=16, window_stride=16, source_stride=10),
        source_version="test", timestamp_provenance="time_ps", topology_policy="pdb",
        split_policy="hash", systems=["A", "B"], inventory=inventory,
        counts={"train_records": 2},
    )
    current_path = tmp_path / "current.json"
    old_path = tmp_path / "old.json"
    assert write_split_manifest(current_path, **args) == old_write_manifest(old_path, **args)
    assert current_path.read_bytes() == old_path.read_bytes()


def test_preprocess_cli_parses_without_opening_sources(capsys):
    with pytest.raises(SystemExit) as exc:
        preprocess_main(["--help"])
    assert exc.value.code == 0
    assert "atlas" in capsys.readouterr().out
