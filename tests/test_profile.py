"""Short real-CUDA stage profile without an optimizer update."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from benchmarks.profile import main
from test_migration import APPROVED, APPROVED_SHA, HISTORICAL_DIT, HISTORICAL_DIT_SHA


MANIFEST = Path(
    "/workspace/PVB/outputs/state_detail_codec_v2/t1/"
    "manifest_20260904_token80000"
)


def test_short_real_clip_profile_reports_stages_and_frozen_gradient_boundary(capsys):
    assert torch.cuda.is_available(), "short stage profile requires CUDA"
    assert MANIFEST.is_dir(), "frozen validation manifest is required"
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    previous_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    previous_benchmark = torch.backends.cudnn.benchmark
    previous_cudnn_deterministic = torch.backends.cudnn.deterministic
    try:
        assert main([
            "--codec", str(APPROVED), "--codec-sha256", APPROVED_SHA,
            "--dit", str(HISTORICAL_DIT), "--dit-sha256", HISTORICAL_DIT_SHA,
            "--manifest-root", str(MANIFEST), "--valid-index", "0",
        ]) == 0
    finally:
        torch.use_deterministic_algorithms(previous_deterministic)
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
        torch.backends.cudnn.allow_tf32 = previous_cudnn_tf32
        torch.backends.cudnn.benchmark = previous_benchmark
        torch.backends.cudnn.deterministic = previous_cudnn_deterministic
    report = json.loads(capsys.readouterr().out)
    assert report["graph_edges"] > 0
    assert report["optimizer_updated"] is False
    assert report["future_condition"] == "observed_prefix_only"
    assert set(report["stages"]) == {
        "graph", "encoder", "codec_encode_inclusive", "decoder",
        "dit_forward", "dit_forward_backward",
    }
    assert all(
        item["elapsed_ms"] > 0 and item["peak_additional_mib"] >= 0
        for item in report["stages"].values()
    )
