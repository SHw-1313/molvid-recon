from __future__ import annotations

import json
from pathlib import Path
import random

import numpy as np
import torch

from molvid.config import cli_config_argv, load_config, resolve_config, validate_config
from molvid.runtime import (
    append_metrics,
    atomic_write_json,
    autocast_context,
    canonical_hash,
    configure_device,
    seed_all,
    sha256_file,
)


def test_config_paths_are_explicit_and_source_mapping_is_unchanged(tmp_path):
    config_path = tmp_path / "settings.yaml"
    config_path.write_text(
        "schema: molvid.test.v1\ndata:\n  train_roots: [clips/train]\n"
        "output_root: runs/example\n"
    )
    loaded = load_config(config_path, schema="molvid.test.v1")
    validate_config(loaded, required_sections=("data",))
    resolved = resolve_config(
        loaded,
        project_root=tmp_path,
        path_fields=("data.train_roots", "output_root"),
    )
    assert loaded["data"]["train_roots"] == ["clips/train"]
    assert resolved["data"]["train_roots"] == [str(tmp_path / "clips/train")]
    assert resolved["output_root"] == str(tmp_path / "runs/example")


def test_cli_config_arguments_validate_schema_and_resolve_paths(tmp_path):
    path = tmp_path / "sample.yaml"
    path.write_text(
        "schema: molvid.sample.v1\ncheckpoint: weights/model.pt\n"
        "valid_index: 2\nrollout_seed: [7, 11]\n", encoding="utf-8",
    )
    flags = cli_config_argv(
        path, schema="molvid.sample.v1",
        allowed_fields=("checkpoint", "valid_index", "rollout_seed"),
        path_fields=("checkpoint",),
    )
    assert flags == [
        "--checkpoint", str(tmp_path / "weights/model.pt"),
        "--valid-index", "2", "--rollout-seed", "7", "--rollout-seed", "11",
    ]
    path.write_text("schema: molvid.sample.v1\nunsupported: true\n", encoding="utf-8")
    import pytest
    with pytest.raises(ValueError, match="unknown CLI configuration fields"):
        cli_config_argv(path, schema="molvid.sample.v1", allowed_fields=())


def test_seed_order_matches_existing_training_sequence():
    assert torch.cuda.is_available(), "RNG acceptance requires CUDA"
    seed_all(719)
    actual = (
        random.random(),
        np.random.random(),
        torch.rand(2),
        torch.rand(2, device="cuda"),
    )
    random.seed(719)
    np.random.seed(719)
    torch.manual_seed(719)
    torch.cuda.manual_seed_all(719)
    expected = (
        random.random(),
        np.random.random(),
        torch.rand(2),
        torch.rand(2, device="cuda"),
    )
    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    assert torch.equal(actual[2], expected[2])
    assert torch.equal(actual[3], expected[3])


def test_runtime_hashes_atomic_json_and_cuda_policy(tmp_path):
    assert configure_device("cuda").type == "cuda"
    path = tmp_path / "result.json"
    atomic_write_json(path, {"b": 2, "a": 1})
    assert json.loads(path.read_text()) == {"a": 1, "b": 2}
    assert not path.with_name("result.json.tmp").exists()
    assert len(sha256_file(path)) == 64
    assert canonical_hash({"b": 2, "a": 1}) == canonical_hash({"a": 1, "b": 2})
    metrics = tmp_path / "metrics.jsonl"
    append_metrics(metrics, {"step": 1, "loss": 2.0})
    append_metrics(metrics, {"step": 2, "loss": 1.0})
    assert [json.loads(line)["step"] for line in metrics.read_text().splitlines()] == [1, 2]
    with autocast_context(torch.device("cuda"), "bf16"):
        assert torch.is_autocast_enabled("cuda")
    with autocast_context(torch.device("cuda"), "fp32"):
        assert not torch.is_autocast_enabled("cuda")
