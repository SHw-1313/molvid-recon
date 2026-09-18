"""Actual forward-shape and frozen-parameter inspection gates."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from molvid.checkpoints import load_codec_artifact, load_dit_artifact
from molvid.data.batch import collate_clip_records
from molvid.data.manifest import load_datasets
from test_migration import APPROVED, APPROVED_SHA, HISTORICAL_DIT, HISTORICAL_DIT_SHA
from tools.inspect_model import inspect_codec_forward, inspect_dit_forward, model_summary


MANIFEST = Path(
    "/workspace/PVB/outputs/state_detail_codec_v2/t1/"
    "manifest_20260904_token80000"
)


def test_model_summary_distinguishes_frozen_parameter_owners():
    model = nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 2))
    model[0].requires_grad_(False)
    report = model_summary(model)
    assert report["total_parameters"] == 26
    assert report["trainable_parameters"] == 10
    assert report["modules"][1]["frozen_parameters"] == 16
    assert report["modules"][2]["frozen_parameters"] == 0
    assert set(report["frozen_names"]) == {"0.weight", "0.bias"}
    assert "Linear" in report["tree"]


def test_real_codec_and_dit_forward_shapes_are_inspectable_without_future_condition():
    assert torch.cuda.is_available(), "real model inspection requires CUDA"
    assert MANIFEST.is_dir(), "frozen validation manifest is required"
    splits = load_datasets(MANIFEST)
    try:
        index = min(
            range(len(splits.valid._clip_index_fields)),
            key=lambda value: splits.valid._clip_index_fields[value][3],
        )
        template = collate_clip_records([splits.valid[index]])
        codec = load_codec_artifact(APPROVED, expected_sha256=APPROVED_SHA, device="cuda").model.eval()
        dit = load_dit_artifact(HISTORICAL_DIT, expected_sha256=HISTORICAL_DIT_SHA, device="cuda")
        assert dit["payload"]["data_hash"] == splits.data_hash
        codec_shapes = inspect_codec_forward(codec, template, torch.device("cuda"))
        assert codec_shapes["coordinate"] == [16, 910, 3]
        assert codec_shapes["reconstruction"] == [16, 910, 3]
        assert codec_shapes["state_h"][0] == 4
        for parameter in codec.parameters():
            parameter.requires_grad_(False)
        forward = inspect_dit_forward(
            codec, dit["model"].eval(), dit["model"].adapter, template,
            device=torch.device("cuda"), codec_hash=dit["payload"]["codec_hash"],
            data_hash=splits.data_hash, history_frames=8,
        )
        assert forward["coordinate_scaffold"] == [16, 910, 3]
        assert forward["latent_state_h"][0] == 4
        assert forward["velocity"]["state_h"] == forward["latent_state_h"]
        assert forward["future_condition"] == "observed_prefix_only"
    finally:
        splits.close()
