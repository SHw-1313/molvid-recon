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
    source = Path("configs/codec_train.yaml")
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


def test_dit_cli_real_codec_short_resume_matches_uninterrupted(tmp_path: Path):
    import numpy as np
    import torch
    import yaml
    from molvid.checkpoints import load_codec_artifact
    from molvid.cli.train_dit import main as train_dit_main
    from molvid.data.batch import collate_clip_records
    from molvid.data.manifest import build_manifest, load_datasets
    from molvid.data.store import ClipMMapWriter
    from molvid.latent.adapter import StateDetailLatentAdapter
    from molvid.latent.statistics import LatentStatistics
    from molvid.runtime import atomic_write_json, canonical_hash, sha256_file
    from molvid.training.batches import encode_batch
    from test_migration import APPROVED, APPROVED_SHA, _record

    assert torch.cuda.is_available(), "P4d DiT CLI gate requires CUDA"
    record = _record()
    t = np.arange(16, dtype=np.float32)[:, None, None]
    base = np.asarray(record["x"][0:1])
    record["x"] = base + 0.03 * np.sin(0.4 * t + np.arange(4, dtype=np.float32)[None, :, None])
    record["bpos"] = record["x"].copy()
    record["time_ps"] = np.arange(16, dtype=np.float32) * 100
    record["delta_time_ps"] = np.full(15, 100, dtype=np.float32)
    split_ids = {
        "train": ["atlas_train_R1_w000000"],
        "valid": ["atlas_valid_R1_w000000"],
        "test": ["atlas_test_R1_w000000"],
    }
    manifest = build_manifest(
        {
            split: {"selected_systems": [f"atlas_{split}"], "sample_ids": ids}
            for split, ids in split_ids.items()
        },
        source_root=tmp_path, frames_per_clip=16, time_bucket_id="dt_100ps", max_tokens=64,
    )
    atomic_write_json(tmp_path / "manifest.json", manifest)
    materialized = {"materialized": True, "splits": {}}
    for split in ("train", "valid"):
        root = tmp_path / "clip_store" / split
        with ClipMMapWriter(root) as writer:
            writer.append({**record, "sample_id": split_ids[split][0]})
        materialized["splits"][split] = {
            "count": 1, "index_sha256": sha256_file(root / "index.txt"),
        }
    materialized["splits"]["test"] = {"count": 1}
    materialized["materialization_sha256"] = canonical_hash(materialized)
    atomic_write_json(tmp_path / "materialization.json", materialized)
    splits = load_datasets(tmp_path)
    data_hash = splits.data_hash
    splits.close()

    artifact = load_codec_artifact(APPROVED, expected_sha256=APPROVED_SHA, device="cuda")
    codec = artifact.model.eval()
    adapter = StateDetailLatentAdapter(
        codec_width=128, scalar_width=8, vector_width=4, ratio=4,
    ).cuda()
    packed, _ = encode_batch(
        codec, adapter, collate_clip_records([record]),
        device=torch.device("cuda"), codec_hash="test", data_hash=data_hash,
    )
    statistics = LatentStatistics.fit(
        [packed], ratio=4,
        provenance={"data_hash": data_hash, "codec_checkpoint_sha256": APPROVED_SHA},
    )
    stats_path = tmp_path / "statistics.pt"
    torch.save(statistics.state_dict(), stats_path)
    config = yaml.safe_load(Path("configs/dit_train.yaml").read_text(encoding="utf-8"))
    config["manifest_root"] = str(tmp_path)
    config["codec"] = {"checkpoint": APPROVED, "sha256": APPROVED_SHA}
    config["statistics"] = {
        "path": str(stats_path), "sha256": sha256_file(stats_path),
        "statistics_hash": statistics.hash,
    }
    config["model"].update({"scalar_width": 8, "vector_width": 4, "depth": 1, "heads": 2, "ffn_multiplier": 2})
    config["training"].update({
        "max_steps": 1, "max_tokens": 64, "clips_per_trajectory": None,
        "history_order": [8], "history_probabilities": [0.0, 1.0, 0.0],
        "output_root": str(tmp_path / "resumed"),
    })
    config_file = tmp_path / "dit.yaml"
    config_file.write_text(yaml.safe_dump(config), encoding="utf-8")
    assert train_dit_main(["--config", str(config_file), "--dry-run"]) == 0
    assert train_dit_main(["--config", str(config_file)]) == 0
    first = tmp_path / "resumed" / "dit_step_00000001.pt"
    assert first.exists()
    assert train_dit_main([
        "--config", str(config_file), "--max-steps", "2",
        "--resume", str(first), "--resume-sha256", sha256_file(first),
    ]) == 0
    resumed = torch.load(tmp_path / "resumed" / "dit_step_00000002.pt", map_location="cpu", weights_only=False)
    config["training"]["max_steps"] = 2
    config["training"]["output_root"] = str(tmp_path / "continuous")
    config_file.write_text(yaml.safe_dump(config), encoding="utf-8")
    assert train_dit_main(["--config", str(config_file)]) == 0
    continuous = torch.load(tmp_path / "continuous" / "dit_step_00000002.pt", map_location="cpu", weights_only=False)
    assert resumed["step"] == continuous["step"] == 2
    for name in continuous["model_state"]:
        torch.testing.assert_close(
            resumed["model_state"][name], continuous["model_state"][name], rtol=0, atol=0,
        )
    for index, slot in continuous["optimizer_state"]["state"].items():
        for name, value in slot.items():
            torch.testing.assert_close(
                resumed["optimizer_state"]["state"][index][name], value, rtol=0, atol=0,
            )
    from molvid.checkpoints import load_dit_inference

    torch.manual_seed(991)
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state().clone()
    inference = load_dit_inference(
        tmp_path / "resumed" / "dit_step_00000002.pt",
        expected_sha256=sha256_file(tmp_path / "resumed" / "dit_step_00000002.pt"),
        codec_path=APPROVED, codec_sha256=APPROVED_SHA, device="cuda",
    )
    assert inference["step"] == 2
    assert torch.equal(torch.get_rng_state(), cpu_rng) and torch.equal(torch.cuda.get_rng_state(), cuda_rng)
    assert not (tmp_path / "clip_store" / "test").exists()

    from molvid.cli.sample import _prefix, main as sample_main
    from molvid.cli.evaluate import main as evaluate_main
    import json

    prefix_path = tmp_path / "prefix.npz"
    np.savez(prefix_path, x=record["x"][:8])
    checkpoint = tmp_path / "resumed" / "dit_step_00000002.pt"
    inference_args = [
        "--checkpoint", str(checkpoint), "--checkpoint-sha256", sha256_file(checkpoint),
        "--codec", APPROVED, "--codec-sha256", APPROVED_SHA,
        "--manifest-root", str(tmp_path), "--valid-index", "0", "--device", "cuda",
    ]
    output = tmp_path / "sample.npz"
    assert sample_main([*inference_args, "--prefix", str(prefix_path), "--history", "8",
                        "--steps", "8", "--seed", "19", "--output", str(output)]) == 0
    with np.load(output, allow_pickle=False) as generated:
        assert generated["x"].shape == (16, 4, 3)
        np.testing.assert_array_equal(generated["x"][:8], record["x"][:8])
        assert np.isfinite(generated["x"]).all()
    metadata = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
    assert metadata["future_condition"] == "observed_prefix_only"
    assert metadata["generation"]["conditioning"] == "observed_prefix_only"
    assert metadata["generation"]["source_mode"] == inference["source_mode"]
    invalid_prefix = tmp_path / "invalid_prefix.npz"
    np.savez(invalid_prefix, x=record["x"][:8], future=record["x"][8:])
    with pytest.raises(ValueError, match="only x"):
        _prefix(invalid_prefix, history=8, atoms=4)

    rollout_output = tmp_path / "rollout.npz"
    sample_config = tmp_path / "sample.yaml"
    sample_config.write_text(yaml.safe_dump({
        "schema": "molvid.sample.v1",
        "checkpoint": str(checkpoint.relative_to(tmp_path)),
        "checkpoint_sha256": sha256_file(checkpoint),
        "codec": APPROVED, "codec_sha256": APPROVED_SHA,
        "manifest_root": ".", "valid_index": 0,
        "prefix": "prefix.npz", "history": 8, "steps": 8,
        "rollout_seed": [23], "device": "cuda", "output": "rollout.npz",
    }), encoding="utf-8")
    assert sample_main(["--config", str(sample_config)]) == 0
    with np.load(rollout_output, allow_pickle=False) as generated:
        assert generated["x"].shape == (16, 4, 3)
        np.testing.assert_array_equal(generated["x"][:8], record["x"][:8])

    report_root = tmp_path / "evaluation"
    evaluate_config = tmp_path / "evaluate.yaml"
    evaluate_config.write_text(yaml.safe_dump({
        "schema": "molvid.evaluate.v1",
        "checkpoint": str(checkpoint.relative_to(tmp_path)),
        "checkpoint_sha256": sha256_file(checkpoint),
        "codec": APPROVED, "codec_sha256": APPROVED_SHA,
        "manifest_root": ".", "valid_index": 0,
        "history": 8, "steps": 8, "seed": 2, "device": "cuda",
        "output_root": "evaluation",
    }), encoding="utf-8")
    assert evaluate_main(["--config", str(evaluate_config), "--seed", "19"]) == 0
    report = json.loads((report_root / "metrics.json").read_text(encoding="utf-8"))
    assert report["schema_version"] == "molvid.dit.evaluation.v1"
    assert report["generation"]["conditioning"] == "observed_prefix_only"
    assert "codec_oracle" in report and "generated_result" in report
    assert report["by_time_bucket"]["dt_100ps"]["system_count"] == 1
    assert not (tmp_path / "clip_store" / "test").exists()
