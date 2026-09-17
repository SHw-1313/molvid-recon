#!/usr/bin/env python3
"""Freeze validation-only T1 selection, then optionally open the test split.

The first invocation writes an immutable selection rule and does not construct
the test dataset.  A second invocation with ``--open-test`` is required to
evaluate the selected control, making the validation/test boundary auditable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from data.clip_batching import TaskAwareClipBatchSampler, make_clip_dataloader
from data.clip_dataset import ClipMMapDataset, collate_clip_records
from scripts.run_state_detail_codec_v2_t0 import (
    RATIOS,
    _evaluate_loader,
    _make_config,
    _make_model,
    _scheduled_ids,
    _sha256,
    _write_json,
)
from scripts.run_state_detail_codec_v2_t1 import _make_loaders, _load_manifest
from trainer.codec_trainer import CodecTrainer


ROOT = Path(__file__).resolve().parents[1]
MODES = (
    "ratio1_state_detail",
    "ratio2_state_detail",
    "ratio4_state_detail",
    "ratio4_matched_pooling",
)
DEFAULT_MANIFEST_ROOT = ROOT / "outputs/state_detail_codec_v2/t1/manifest_20260904_token80000"
DEFAULT_FULL_ROOT = ROOT / "outputs/state_detail_codec_v2/t1/full_20260904_seed20260903"
DEFAULT_SELECTION_ROOT = ROOT / "outputs/state_detail_codec_v2/t1/selection_20260904"


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _load_results(full_root: Path) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for mode in MODES:
        path = full_root / mode / "result.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        result = json.loads(path.read_text(encoding="utf-8"))
        if result.get("status") != "passed":
            raise RuntimeError(f"cannot rank failed T1 result: {path}")
        results[mode] = result
    return results


def _ranking_key(mode: str, result: Mapping[str, Any]) -> tuple[float, float, float, str]:
    return (
        float(result["best_validation_future_aligned_rmsd"]),
        float(
            result["final_validation"]["metrics"]["future"]["drmsd"]
        ),
        float(
            result["final_validation"]["metrics"]["future"]["bond_rmse"]
        ),
        mode,
    )


def _freeze_selection(full_root: Path, output_root: Path) -> dict[str, Any]:
    results = _load_results(full_root)
    ranked = sorted(
        (mode for mode in MODES),
        key=lambda mode: _ranking_key(mode, results[mode]),
    )
    payload = {
        "schema_version": "pvb.codec.state_detail.t1_validation_selection.v1",
        "status": "FROZEN",
        "test_opened": False,
        "selection_rule": {
            "primary": "minimum best validation future aligned_rmsd across completed epochs",
            "tie_break_1": "minimum final validation future drmsd",
            "tie_break_2": "minimum final validation future bond_rmse",
            "tie_break_3": "lexicographic control name",
            "no_test_metrics_used": True,
        },
        "source_full_root": str(full_root),
        "source_full_root_sha256": _canonical_hash(
            {
                mode: {
                    "result_sha256": _sha256(full_root / mode / "result.json"),
                    "mode": results[mode]["mode"],
                    "seed": results[mode]["seed"],
                }
                for mode in MODES
            }
        ),
        "ranking": [
            {
                "rank": index + 1,
                "mode": mode,
                "best_validation_future_aligned_rmsd": results[mode]["best_validation_future_aligned_rmsd"],
                "final_validation_future_drmsd": results[mode]["final_validation"]["metrics"]["future"]["drmsd"],
                "final_validation_future_bond_rmse": results[mode]["final_validation"]["metrics"]["future"]["bond_rmse"],
                "best_epoch": results[mode]["best_epoch"],
                "best_checkpoint": results[mode]["best_checkpoint"],
            }
            for index, mode in enumerate(ranked)
        ],
        "selected_mode": ranked[0],
        "top_two_modes": ranked[:2],
    }
    payload["selection_content_sha256"] = _canonical_hash(payload)
    output_root.mkdir(parents=True, exist_ok=False)
    _write_json(output_root / "selection_rule.json", payload)
    lines = [
        "# T1 validation-only selection",
        "",
        "Status: **FROZEN**; test data were not opened during ranking.",
        "",
        "| Rank | Control | Best validation aligned RMSD | Final validation dRMSD | Final validation bond RMSE | Best epoch |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for row in payload["ranking"]:
        lines.append(
            f"| {row['rank']} | {row['mode']} | {row['best_validation_future_aligned_rmsd']:.8g} | "
            f"{row['final_validation_future_drmsd']:.8g} | {row['final_validation_future_bond_rmse']:.8g} | {row['best_epoch']} |"
        )
    lines.extend(
        [
            "",
            f"Selected control: `{payload['selected_mode']}`.",
            f"Top two for additional seeds: `{payload['top_two_modes'][0]}`, `{payload['top_two_modes'][1]}`.",
            "",
            "Test evaluation is a separate command and is permitted only after this file is frozen.",
        ]
    )
    (output_root / "selection_rule.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def _open_selected_test(
    manifest_root: Path,
    full_root: Path,
    selection_root: Path,
) -> dict[str, Any]:
    selection_path = selection_root / "selection_rule.json"
    if not selection_path.is_file():
        raise FileNotFoundError("freeze validation selection before opening test")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    content = dict(selection)
    recorded = content.pop("selection_content_sha256", None)
    if selection.get("status") != "FROZEN" or recorded != _canonical_hash(content):
        raise RuntimeError("selection rule is not a valid frozen packet")
    if selection.get("test_opened"):
        raise FileExistsError("test has already been opened for this selection packet")
    mode = str(selection["selected_mode"])
    result = json.loads((full_root / mode / "result.json").read_text(encoding="utf-8"))
    checkpoint = Path(str(result["best_checkpoint"]))
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"selected best checkpoint is not present on this machine: {checkpoint}"
        )
    manifest, materialization = _load_manifest(manifest_root)
    # Build train/validation loaders only for the checkpoint contract; this call
    # still does not open the test store.  Test is opened explicitly below.
    train_dataset, valid_dataset, train_loader, valid_loader, epoch_batches = _make_loaders(
        manifest_root, seed=int(result["seed"])
    )
    test_dataset = ClipMMapDataset(manifest_root / "clip_store" / "test")
    expected_ids = tuple(manifest["source_splits"]["test"]["sample_ids"])
    if tuple(str(row[0]) for row in test_dataset._index) != expected_ids:
        raise RuntimeError("test index differs from frozen manifest")
    test_sampler = TaskAwareClipBatchSampler(
        test_dataset,
        max_tokens=80000,
        seed=int(result["seed"]) + 2,
        shuffle=False,
        replacement=False,
        oversize_policy="error",
    )
    test_loader = make_clip_dataloader(
        test_dataset,
        sampler=test_sampler,
        collate_fn=collate_clip_records,
        num_workers=0,
        pin_memory=False,
    )
    if _scheduled_ids(test_loader, test_dataset, 0) != list(expected_ids):
        raise RuntimeError("test sampler is not an exact no-replacement pass")
    model = _make_model(mode, freeze_frame_encoder=True)
    trainer = CodecTrainer(
        model,
        train_loader,
        valid_loader,
        config=_make_config(
            int(result["total_steps"]),
            train_batches=int(epoch_batches[0]),
            schedule_steps=int(result["total_steps"]),
        ),
        device="cuda:0",
        non_blocking_transfer=False,
    )
    trainer.load_checkpoint(checkpoint)
    evaluation = _evaluate_loader(
        trainer,
        test_loader,
        test_dataset,
        epoch=0,
        ratio=RATIOS[mode],
        detailed=True,
    )
    packet = {
        "schema_version": "pvb.codec.state_detail.t1_test_evaluation.v1",
        "status": "OPENED_AFTER_VALIDATION_SELECTION",
        "selection_rule_sha256": _sha256(selection_path),
        "selected_mode": mode,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "manifest_content_sha256": manifest["manifest_content_sha256"],
        "materialization_sha256": materialization["materialization_sha256"],
        "sample_count": evaluation["sample_count"],
        "exact_coverage": evaluation["exact_coverage"],
        "evaluation": evaluation,
    }
    output_path = selection_root / "test_evaluation.json"
    _write_json(output_path, packet)
    selection["test_opened"] = True
    selection["test_evaluation"] = str(output_path)
    selection["test_evaluation_sha256"] = _sha256(output_path)
    selection["selection_content_sha256"] = _canonical_hash(
        {key: value for key, value in selection.items() if key != "selection_content_sha256"}
    )
    _write_json(selection_path, selection)
    print(json.dumps(packet, indent=2, sort_keys=True))
    return packet


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-root", type=Path, default=DEFAULT_MANIFEST_ROOT)
    parser.add_argument("--full-root", type=Path, default=DEFAULT_FULL_ROOT)
    parser.add_argument("--selection-root", type=Path, default=DEFAULT_SELECTION_ROOT)
    parser.add_argument("--open-test", action="store_true")
    args = parser.parse_args(argv)
    manifest_root = args.manifest_root if args.manifest_root.is_absolute() else ROOT / args.manifest_root
    full_root = args.full_root if args.full_root.is_absolute() else ROOT / args.full_root
    selection_root = args.selection_root if args.selection_root.is_absolute() else ROOT / args.selection_root
    if args.open_test:
        _open_selected_test(manifest_root, full_root, selection_root)
    else:
        _freeze_selection(full_root, selection_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
