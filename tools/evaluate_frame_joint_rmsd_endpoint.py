#!/usr/bin/env python
"""Small, explicitly future-informed endpoint diagnostic for Frame Joint v1.

This is not a generation evaluation: the input interpolation contains the
clean future target.  It is intended to separate velocity/endpoint errors from
the complete source-to-target sampler on a fixed handful of windows.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from molvid.checkpoints import load_frame_joint_inference
from molvid.data.batch import collate_clip_records
from molvid.data.store import ClipMMapDataset
from molvid.evaluation.geometry import bond_errors, coordinate_errors
from molvid.evaluation.runner import trajectory_metrics
from molvid.flow.objective import FrameRectifiedFlowObjective
from molvid.runtime import configure_device, sha256_file
from molvid.training.batches import prepare_frame_joint_batch


def _latent_mse(pred, target):
    mask = target.atom_frame_mask()
    h = (pred.h.float() - target.h.float()).square().mean(dim=-1)
    v = (pred.v.float() - target.v.float()).square().mean(dim=(2, 3))
    return {
        "h": float(h[mask].mean().item()) if bool(mask.any()) else None,
        "v": float(v[mask].mean().item()) if bool(mask.any()) else None,
        "total": float(torch.cat((h[mask], v[mask])).mean().item()) if bool(mask.any()) else None,
    }


def _select_ids(rows: list[dict[str, Any]], limit: int) -> list[str]:
    ids: list[str] = []
    systems: set[str] = set()
    for row in rows:
        sid = row["sample_id"]
        system = sid.rsplit("_R", 1)[0]
        if system not in systems:
            systems.add(system)
            ids.append(sid)
        if len(ids) >= limit:
            break
    return ids


@torch.no_grad()
def run(args: argparse.Namespace) -> dict[str, Any]:
    device = configure_device(args.device, deterministic=True)
    loaded = load_frame_joint_inference(
        Path(args.checkpoint), expected_sha256=args.checkpoint_sha256,
        codec_path=Path(args.codec), codec_sha256=args.codec_sha256, device=device,
    )
    model = loaded["model"]
    model.eval()
    payload = json.loads(Path(args.reference_evaluation).read_text(encoding="utf-8"))
    rows = payload["rows"]
    selected = set(args.sample_id or _select_ids(rows, args.systems_limit))
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        by_id.setdefault(row["sample_id"], row)
    store = ClipMMapDataset(Path(args.valid_store))
    index_by_id = {str(value[0]): index for index, value in enumerate(store._index)}
    objective = FrameRectifiedFlowObjective()
    out_rows: list[dict[str, Any]] = []
    coord_store: dict[str, np.ndarray] = {}
    for ordinal, sample_id in enumerate([r["sample_id"] for r in rows if r["sample_id"] in selected]):
        row = by_id[sample_id]
        if sample_id not in index_by_id:
            raise RuntimeError(f"selected window missing from valid store: {sample_id}")
        batch = collate_clip_records([store[index_by_id[sample_id]]])
        for history in (4, 8):
            prepared = prepare_frame_joint_batch(
                model.target_teacher, batch, device=device, normalizer=model, history_frames=history
            )
            clean = model.decoder(prepared.observed_context, prepared.target_future, prepared.query)
            target = prepared.coordinate_batch.x.float()
            clean_full = torch.cat((target[:history], clean.coordinates.float()), dim=0)
            clean_metrics = trajectory_metrics(clean_full, target, prepared.coordinate_batch, history_frames=history)
            seed = int(args.seed + ordinal * 1009 + history * 17)
            for s in args.flow_times:
                generator = torch.Generator(device=device)
                generator.manual_seed(seed)
                flow_time = torch.full((prepared.normalized_target.batch_size,), float(s), device=device, dtype=prepared.normalized_target.h.dtype)
                sample = objective.sample(
                    prepared.normalized_target, prepared.source_center,
                    generator=generator, flow_time=flow_time,
                )
                output = model(
                    sample.interpolated,
                    context=prepared.observed_context,
                    query=prepared.query,
                    flow_time=flow_time,
                    decode_generated=True,
                )
                generated = output.generated
                if generated is None:
                    raise RuntimeError("endpoint diagnostic did not decode generated endpoint")
                flow_loss = objective.loss(output.velocity, sample.target_velocity)
                latent = _latent_mse(output.normalized_endpoint, prepared.normalized_target)
                generated_full = torch.cat((target[:history], generated.coordinates.float()), dim=0)
                metrics = trajectory_metrics(generated_full, target, prepared.coordinate_batch, history_frames=history)
                future_frames = list(range(history, int(target.shape[0])))
                coord = coordinate_errors(generated_full, target, prepared.coordinate_batch, future_frames)
                bond = bond_errors(generated_full, target, prepared.coordinate_batch, future_frames)
                key = f"{sample_id}__H{history}__s{str(s).replace('.', '')}"
                coord_store[key] = generated_full.detach().float().cpu().numpy()
                out_rows.append({
                    "sample_id": sample_id,
                    "system": sample_id.rsplit("_R", 1)[0],
                    "history_frames": history,
                    "flow_time": float(s),
                    "seed": seed,
                    "future_information_leak": True,
                    "clean_oracle": clean_metrics,
                    "endpoint": metrics,
                    "coordinate_errors": coord,
                    "bond_errors": bond,
                    "flow_velocity_mse": {
                        "h": float(flow_loss.h.item()),
                        "v": float(flow_loss.v.item()),
                        "total": float(flow_loss.total.item()),
                    },
                    "endpoint_latent_mse_normalized": latent,
                    "mask": {
                        "loss_atoms": int(prepared.coordinate_batch.loss_mask.sum().item()),
                        "align_atoms": int(prepared.coordinate_batch.align_mask.sum().item()),
                        "intersection_atoms": int((prepared.coordinate_batch.loss_mask & prepared.coordinate_batch.align_mask).sum().item()),
                    },
                })
    result = {
        "protocol": {
            "diagnostic": "training-style endpoint; clean future is intentionally present in interpolation",
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": args.checkpoint_sha256,
            "codec": str(args.codec),
            "valid_store": str(args.valid_store),
            "flow_times": list(args.flow_times),
            "seed": args.seed,
            "systems": sorted({r["system"] for r in out_rows}),
            "coordinate_units": "angstrom",
            "valid_store_index_sha256": sha256_file(Path(args.valid_store) / "index.txt"),
        },
        "rows": out_rows,
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "diagnosis.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    np.savez_compressed(output / "coordinates.npz", **coord_store)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--codec", required=True)
    parser.add_argument("--codec-sha256", required=True)
    parser.add_argument("--valid-store", required=True)
    parser.add_argument("--reference-evaluation", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--systems-limit", type=int, default=8)
    parser.add_argument("--flow-times", type=float, nargs="+", default=[0.1, 0.5, 0.95])
    parser.add_argument("--sample-id", action="append")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
