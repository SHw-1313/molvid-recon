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


def test_codec_cli_one_step_and_new_format_resume(tmp_path: Path):
    import torch
    import yaml
    from molvid.cli.train_codec import main as train_codec_main
    from molvid.data.store import ClipMMapWriter
    from molvid.runtime import sha256_file
    from test_migration import _record

    assert torch.cuda.is_available(), "P4d codec CLI gate requires CUDA"
    store = tmp_path / "train"
    with ClipMMapWriter(store) as writer:
        writer.append(_record())
    source = Path("config/molvid_codec.yaml")
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    config["model"].update({
        "hidden_channels": 8, "spatial_layers": 1, "num_rbf": 8,
        "num_heads": 2, "temporal_codec_mode": "ratio4_state_detail",
    })
    config["training"].update({"max_steps": 1, "warmup_steps": 1, "save_dir": str(tmp_path / "out")})
    config["training"]["loss_schedule"] = [
        {"start_step": 0, "weights": {"coordinate": 1.0, "local": 0.0, "bond": 0.0, "velocity": 0.0, "acceleration": 0.0}}
    ]
    config["data"].update({"train_roots": [str(store)], "valid_roots": [], "max_tokens": 64, "seed": 11})
    config_file = tmp_path / "codec.yaml"
    config_file.write_text(yaml.safe_dump(config), encoding="utf-8")
    assert train_codec_main(["--config", str(config_file), "--dry-run"]) == 0
    assert train_codec_main(["--config", str(config_file)]) == 0
    first = tmp_path / "out" / "codec_step_00000001.pt"
    assert first.exists()
    assert train_codec_main([
        "--config", str(config_file), "--max-steps", "2",
        "--resume", str(first), "--resume-sha256", sha256_file(first),
    ]) == 0
    second = tmp_path / "out" / "codec_step_00000002.pt"
    assert second.exists()
    payload = torch.load(second, map_location="cpu", weights_only=False)
    assert payload["schema_version"] == "molvid.training.checkpoint.v1"
    assert payload["step"] == 2
