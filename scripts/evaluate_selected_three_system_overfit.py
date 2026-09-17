#!/usr/bin/env python3
"""Evaluate the three fixed ViSNet spatial-backbone micro-overfit checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from data.clip_dataset import collate_clip_records
from eval_codec import _loss_evaluator
from evaluation.codec_evaluation import evaluate_controls, model_control, write_report
from scripts.run_selected_three_system_overfit import (
    BACKBONES,
    BASE_OUTPUT,
    DEVICE,
    make_model,
    require_cuda,
    select_records,
)
from trainer.codec_trainer import (
    CODEC_CHECKPOINT_SCHEMA,
    CodecTrainConfig,
    PVBCodecModel,
    prepare_batch_then_to_device,
)


def _predictor(model: PVBCodecModel, device: torch.device):
    def predict(batch: Any) -> torch.Tensor:
        moved = prepare_batch_then_to_device(model, batch, device)
        with torch.no_grad():
            output = model(moved)
        return output.x_hat.detach().cpu()

    return predict


def _load_backbone(
    spatial_backbone: str,
    checkpoint: Path,
) -> tuple[PVBCodecModel, dict[str, Any]]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint is not a mapping: {checkpoint}")
    if str(payload.get("schema_version", "")) != CODEC_CHECKPOINT_SCHEMA:
        raise ValueError(
            f"{spatial_backbone} checkpoint must use {CODEC_CHECKPOINT_SCHEMA!r}: {checkpoint}"
        )
    required = {
        "step",
        "model_state",
        "normalization_stats",
        "config",
        "model_contract",
        "distance_reference_contract",
        "optimizer_contract",
    }
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"checkpoint is missing fields {sorted(missing)}: {checkpoint}")
    expected = make_model(spatial_backbone)
    contract = payload["model_contract"]
    if contract != expected.model_contract():
        raise ValueError(f"{spatial_backbone} checkpoint model contract differs from runner")
    model = PVBCodecModel.from_model_contract(contract)
    model.load_state_dict(payload["model_state"], strict=True)
    model = model.to(DEVICE).eval()
    return model, payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="completed run directory; defaults to the newest micro-overfit run",
    )
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--markdown", type=Path, default=None)
    args = parser.parse_args()

    runtime = require_cuda()
    if args.run_dir is None:
        candidates = sorted(BASE_OUTPUT.glob("run_*/summary.json"))
        if not candidates:
            raise FileNotFoundError(f"no micro-overfit summaries under {BASE_OUTPUT}")
        run_dir = candidates[-1].parent
    else:
        run_dir = args.run_dir if args.run_dir.is_absolute() else BASE_OUTPUT / args.run_dir
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") not in {"completed", "quality_gate_failed"}:
        raise RuntimeError(f"micro-overfit is not complete: {summary.get('status')!r}")

    records, data_contract = select_records()
    batches = [collate_clip_records([record]) for record in records]
    controls = []
    per_backbone = {}
    for spatial_backbone in BACKBONES:
        result = summary.get("backbones", {}).get(spatial_backbone)
        if not isinstance(result, dict):
            raise RuntimeError(f"micro-overfit summary lacks {spatial_backbone}")
        checkpoint = Path(str(result["checkpoint"]))
        if not checkpoint.is_absolute():
            checkpoint = run_dir / checkpoint
        model, payload = _load_backbone(spatial_backbone, checkpoint)
        config = CodecTrainConfig.from_mapping(payload["config"])
        evaluator = _loss_evaluator(
            DEVICE,
            config,
            payload["normalization_stats"],
            int(payload["step"]),
        )
        controls.append(
            model_control(
                spatial_backbone,
                _predictor(model, DEVICE),
                ratio=1,
                temporal=True,
                loss_evaluator=evaluator,
            )
        )
        per_backbone[spatial_backbone] = {
            "checkpoint": str(checkpoint),
            "step": int(payload["step"]),
            "micro_overfit_quality_pass": bool(result["micro_overfit_quality_pass"]),
        }

    report = evaluate_controls(controls, batches, device=DEVICE)
    report["evaluation_data"] = {
        "runtime": runtime,
        "source_split": data_contract["source_split"],
        "sample_ids": data_contract["sample_ids"],
        "systems": data_contract["systems"],
        "frames": data_contract["frames"],
        "packed_atoms": data_contract["packed_atoms"],
        "effective_tokens": data_contract["effective_tokens"],
        "batch_policy": "three selected one-clip batches; metrics are reported per backbone and system",
        "validation_samples_used": False,
        "temporal_layers": 1,
        "temporal_ratio": 1,
        "spatial_backbones": list(BACKBONES),
        "backbones": per_backbone,
    }
    json_path = args.json or (run_dir / "codec_eval.json")
    markdown_path = args.markdown or (run_dir / "codec_eval.md")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    write_report(report, json_path, markdown_path)
    torch.cuda.synchronize()
    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "json": str(json_path),
                "markdown": str(markdown_path),
                "controls": list(report["controls"]),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
