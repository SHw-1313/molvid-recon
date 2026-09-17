#!/usr/bin/env python3
"""Build the immutable Phase-A repaired-CUDA FP32 reference set.

This command is intentionally separate from acceptance.  It may create a
reference only when the operator explicitly invokes it, and it refuses to
overwrite an existing set unless --force is supplied.  The acceptance runner
only consumes the manifest and verifies its hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

import torch

from trainer.codec_losses import CodecLossWeights, compute_codec_losses
from scripts.run_engineering_acceptance import (
    CONTROLS,
    DEVICE,
    REPAIRED_REFERENCE_DIR,
    _graph_invariants,
    _load_control_model,
    _real_batch,
    require_cuda,
)

ROOT = Path(__file__).resolve().parents[1]
FIXED_WEIGHTS = CodecLossWeights(
    coordinate=1.0,
    local=0.1,
    bond=0.1,
    velocity=0.1,
    acceleration=0.05,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build(*, allow_legacy: bool, legacy_bond_mode: str | None, output_dir: Path, force: bool) -> dict[str, Any]:
    runtime = require_cuda()
    if output_dir.exists():
        existing = list(output_dir.iterdir())
        if existing and not force:
            raise RuntimeError(
                f"reference directory is non-empty: {output_dir}; "
                "use --force only after explicitly approving replacement"
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    _, batch_cpu, _ = _real_batch()
    batch_cpu.x.requires_grad_(True)
    controls: dict[str, Any] = {}
    for name in CONTROLS:
        model = _load_control_model(
            name,
            allow_legacy=allow_legacy,
            legacy_bond_mode=legacy_bond_mode,
        )
        model.eval()
        model.prepare_batch(batch_cpu)
        batch = batch_cpu.to(DEVICE, non_blocking=True)
        batch.x.retain_grad()
        model.zero_grad(set_to_none=True)
        encoded, output = model.forward_with_encoded(batch)
        losses = compute_codec_losses(output, batch, weights=FIXED_WEIGHTS)
        losses["total"].backward()
        torch.cuda.synchronize()
        if batch.x.grad is None:
            raise RuntimeError(f"{name} reference is missing the position gradient")
        gradients: dict[str, torch.Tensor] = {}
        for parameter_name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            if parameter.grad is None:
                raise RuntimeError(
                    f"{name} reference is missing gradient for {parameter_name}"
                )
            if not torch.isfinite(parameter.grad).all():
                raise RuntimeError(f"{name} reference has non-finite gradient {parameter_name}")
            gradients[parameter_name] = parameter.grad.detach().cpu()
        if not torch.isfinite(batch.x.grad).all():
            raise RuntimeError(f"{name} reference has non-finite position gradient")
        checkpoint = CONTROLS[name]["checkpoint"]
        checkpoint_payload = torch.load(
            checkpoint, map_location="cpu", weights_only=False
        )
        payload = {
            "schema_version": "pvb.codec.engineering.repaired_cuda_reference.v1",
            "control": name,
            "checkpoint": str(checkpoint),
            "checkpoint_schema": str(checkpoint_payload.get("schema_version", "")),
            "checkpoint_sha256": sha256_file(checkpoint),
            "model_contract": checkpoint_payload.get("model_contract"),
            "fixed_batch": {
                "shape": [int(value) for value in batch_cpu.x.shape],
                "sample_ids": list(batch_cpu.sample_id),
            },
            "parameter_names": list(gradients),
            "parameter_shapes": {
                key: list(value.shape) for key, value in gradients.items()
            },
            "parameter_count": int(sum(value.numel() for value in model.parameters())),
            "encoded_h": encoded.h.detach().cpu(),
            "encoded_v": encoded.v.detach().cpu(),
            "decoded_x_coarse": output.x_coarse.detach().cpu(),
            "decoded_x_hat": output.x_hat.detach().cpu(),
            "losses": {
                key: float(value.detach().cpu()) for key, value in losses.items()
            },
            "parameter_gradients": gradients,
            "position_gradient": batch.x.grad.detach().cpu(),
            "edge_index": encoded.graph.edge_index.detach().cpu(),
            "bond_type": encoded.graph.bond_type.detach().cpu(),
            "graph": _graph_invariants(encoded.graph),
        }
        reference_path = output_dir / f"{name}.pt"
        torch.save(payload, reference_path)
        controls[name] = {
            "file": reference_path.name,
            "sha256": sha256_file(reference_path),
            "bytes": reference_path.stat().st_size,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": payload["checkpoint_sha256"],
            "checkpoint_schema": payload["checkpoint_schema"],
            "parameter_count": payload["parameter_count"],
        }
        del model, batch, encoded, output, losses, gradients, payload, checkpoint_payload
        torch.cuda.empty_cache()
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        commit = "unknown"
    manifest = {
        "schema_version": "pvb.codec.engineering.reference_manifest.v1",
        "policy": "immutable_repaired_cuda_fp32_reference",
        "created_by": "scripts/build_engineering_reference.py",
        "source_commit": commit,
        "runtime": runtime,
        "fixed_batch": {
            "shape": [int(value) for value in batch_cpu.x.shape],
            "sample_ids": list(batch_cpu.sample_id),
        },
        "loss_weights": FIXED_WEIGHTS.as_dict(),
        "controls": controls,
        "reproduction": (
            "enter-container; source /home/sihao/miniforge3/bin/activate torch-ito; "
            "cd /workspace/PVB; python scripts/build_engineering_reference.py "
            "--allow-legacy-checkpoint --legacy-bond-mode topology"
        ),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=REPAIRED_REFERENCE_DIR)
    parser.add_argument("--allow-legacy-checkpoint", action="store_true")
    parser.add_argument("--legacy-bond-mode", choices=("topology", "distance_only"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.legacy_bond_mode is not None and not args.allow_legacy_checkpoint:
        parser.error("--legacy-bond-mode requires --allow-legacy-checkpoint")
    manifest = build(
        allow_legacy=args.allow_legacy_checkpoint,
        legacy_bond_mode=args.legacy_bond_mode,
        output_dir=args.output_dir.resolve(),
        force=args.force,
    )
    print(json.dumps({"manifest": str(args.output_dir / "manifest.json"), "controls": sorted(manifest["controls"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
