#!/usr/bin/env python3
"""Run Baseline A through the unmodified PVB model implementation.

The pilot validation stores are the versioned npz-v1 clip stores, whereas the
legacy PVB entrypoints expect JSONL/PDB inputs and gzip-JSON block stores.  This
read-only adapter keeps the pilot selection policy and translates each clip's
frame 0 into the legacy batch mapping expected by ``dyVAE.inference``.  It
never edits PVB_origin or the shared data stores.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


PVB_ROOT = Path(__file__).resolve().parents[3]
PVB_ORIGIN = Path("/data4/users/sihao/workspace/PVB_origin")
SEED = 20260810
MAX_TOKENS = 80_000
MAX_BATCHES = 32
ROLLOUT_FRAMES = 16
SDE_STEPS = 10
SCHEMA = "pvb.baseline.static_direct.v1"

# Reuse the already prepared, read-only pilot-store reader, selector, and
# metric definitions.  Importing this output-only helper does not import or
# modify PVB_origin and does not execute its Baseline B main function.
if str(PVB_ROOT) not in sys.path:
    sys.path.insert(0, str(PVB_ROOT))
from outputs.pvb_origin_baselines.dynamic_direct.run_baseline_b import (  # noqa: E402
    ClipStore,
    make_legacy_batch,
    mean_dict,
    metrics_for,
    one_step_metrics,
    select_pilot_batches,
    set_seed,
    write_csv,
)


def _git(command: list[str]) -> str:
    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True)
    except OSError as exc:
        return f"<unavailable: {exc}>"
    return result.stdout.strip() if result.returncode == 0 else f"<exit {result.returncode}: {result.stderr.strip()}>"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_checkpoint(requested: Path) -> tuple[Path, dict[str, Any]]:
    """Resolve a checkpoint directory using its own top-k metadata."""

    requested = requested.expanduser()
    # /data1 is a read-only shared mount in the container and can briefly
    # return ENOENT while its directory cache is refreshed.  Retry only the
    # read-only visibility check; no checkpoint or source file is changed.
    for _ in range(30):
        if requested.is_file() or requested.is_dir():
            break
        time.sleep(1.0)
    if requested.is_file():
        return requested, {
            "input": str(requested),
            "resolution": "explicit_file",
            "topk_map": None,
        }
    if not requested.is_dir():
        raise FileNotFoundError(f"checkpoint path is neither a file nor directory: {requested}")

    maps: list[Path] = []
    for _ in range(30):
        maps = sorted(requested.glob("version_*/checkpoint/topk_map.txt"))
        if maps:
            break
        time.sleep(1.0)
    if not maps:
        raise FileNotFoundError(f"no version_*/checkpoint/topk_map.txt under {requested}")
    topk_map = maps[-1]
    lines = [line.strip() for line in topk_map.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines or ":" not in lines[0]:
        raise ValueError(f"malformed top-k map: {topk_map}")
    relative = lines[0].split(":", 1)[1].strip()
    repo_root = requested.parents[1]
    candidate = repo_root / relative
    for _ in range(30):
        if candidate.is_file():
            break
        time.sleep(1.0)
    if not candidate.is_file():
        # The metadata is repository-relative in the supplied checkpoint tree;
        # retain a second explicit interpretation for unusual relocations.
        candidate = topk_map.parent / Path(relative).name
    if not candidate.is_file():
        raise FileNotFoundError(f"top-k map selected a missing checkpoint: {candidate}")
    return candidate, {
        "input": str(requested),
        "resolution": "first_topk_map_entry",
        "topk_map": str(topk_map),
        "topk_map_first_entry": lines[0],
        "resolved_file": str(candidate),
    }


def fallback_evidence() -> dict[str, Any]:
    """Record the exact fallback check without treating PDB data as eligible."""

    return {
        "preferred_original_entrypoint_contract": {
            "infer_prot": "mode=all JSONL items with state0_path pointing to PDB files",
            "infer_complex": "mode=all JSONL items with state0_path PDB (MISATO) or protein/ligand paths",
            "legacy_mmap": "index.txt plus gzip-compressed JSON records containing x0/x1 or x0",
        },
        "preferred_store_contract": {
            "format": "npz-v1",
            "payload": "per-record NumPy archive in data.bin",
            "index": "sample_id/start/end/atoms/frames/time_bucket_id in index.txt",
            "roots": [
                "/data4/users/sihao/data/pvb_cross_dataset_20260810/clips/atlas/dt_100ps/valid",
                "/data4/users/sihao/data/pvb_cross_dataset_20260810/clips/misato/dt_80ps/valid",
            ],
        },
        "fallback_root_requested": "/data5/PVB",
        "fallback_root_visible_in_container": Path("/data5/PVB").exists(),
        "fallback_candidates_found_by_read_only_host_check": [
            "/data5/PVB/pdb/ept_release/pvb_phase12_full/valid",
            "/data5/PVB/pdb/ept_release/pvb_phase12_full/test",
        ],
        "atlas_misato_original_format_fallback": None,
        "fallback_decision": (
            "No ATLAS or MISATO original-format migrated store was found under /data5/PVB. "
            "The only original-format candidate is the PDB EPT/PVB store and is excluded by scope. "
            "Use the output-only npz-v1 adapter against the preferred validation stores."
        ),
    }


def _direct_one_step_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    record: Mapping[str, Any],
) -> dict[str, float]:
    """Metrics for the first native transition, kept separate from rollout metrics."""

    one = {**record, "delta_time_ps": np.asarray(record["delta_time_ps"], dtype=np.float32)[:1]}
    return one_step_metrics(prediction[1:2], target[1:2], one)


def _base_report(args: argparse.Namespace, resolution: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA,
        "status": "running",
        "baseline": "A",
        "description": "Unmodified PVB static-structure pretrained dyVAE checkpoint, direct native-step inference; no retraining.",
        "seed": args.seed,
        "source": {
            "root": str(PVB_ORIGIN),
            "commit": _git(["git", "-C", str(PVB_ORIGIN), "rev-parse", "HEAD"]),
            "status": _git(["git", "-C", str(PVB_ORIGIN), "status", "--short", "--branch"]),
        },
        "checkpoint": resolution,
        "runtime": {
            "container_launcher": "enter-container",
            "environment_activation": "source /home/sihao/miniforge3/bin/activate torch-ito",
            "python": sys.executable,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "device": args.device,
        },
        "data": {
            "atlas_valid": str(args.atlas_valid),
            "misato_valid": str(args.misato_valid),
            "pdb_included": False,
            "selection_policy": {
                "store_order": [str(args.atlas_valid), str(args.misato_valid)],
                "max_tokens": args.max_tokens,
                "max_batches": args.max_batches,
                "batching_cost": "frames * atoms",
                "shuffle": False,
                "replacement": False,
                "seed": args.seed,
            },
        },
        "compatibility": fallback_evidence(),
        "inference": {
            "method": "module.model.dyVAE.inference from PVB_origin",
            "sde_steps": args.sde_steps,
            "rollout_frames": args.rollout_frames,
            "rollout_transition_count": args.rollout_frames - 1,
            "rollout_policy": "frame 0 is the input; each later frame is one sequential native-step model.inference call",
            "physical_time_policy": "one model transition per native store interval; native ATLAS dt_100ps and MISATO dt_80ps are preserved in targets and metrics",
        },
        "blockers": [],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--atlas-valid", type=Path, required=True)
    parser.add_argument("--misato-valid", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:7")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    parser.add_argument("--max-batches", type=int, default=MAX_BATCHES)
    parser.add_argument("--rollout-frames", type=int, default=ROLLOUT_FRAMES)
    parser.add_argument("--sde-steps", type=int, default=SDE_STEPS)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    status_path = args.output_dir / "baseline_a_status.json"
    started = time.perf_counter()
    status: dict[str, Any] = {
        "schema_version": SCHEMA,
        "status": "starting",
        "argv": [str(item) for item in sys.argv],
        "command_context": {
            "launcher": "enter-container",
            "activation": "source /home/sihao/miniforge3/bin/activate torch-ito",
            "working_directory": str(PVB_ORIGIN),
        },
        "started_at_unix": time.time(),
    }
    status_path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    stores: list[ClipStore] = []
    rows: list[dict[str, Any]] = []
    per_sample: list[dict[str, Any]] = []
    try:
        if args.rollout_frames < 2:
            raise ValueError("rollout-frames must be at least 2")
        if args.seed != SEED:
            raise ValueError(f"seed must remain {SEED}, got {args.seed}")

        checkpoint, resolution = resolve_checkpoint(args.checkpoint)
        resolution["sha256"] = _sha256(checkpoint)
        stores = [ClipStore(args.atlas_valid), ClipStore(args.misato_valid)]
        batches, selected = select_pilot_batches(
            stores,
            seed=args.seed,
            max_tokens=args.max_tokens,
            max_batches=args.max_batches,
        )
        selection = {
            "schema_version": f"{SCHEMA}.selection",
            "seed": args.seed,
            "store_order": [str(store.root) for store in stores],
            "policy": {
                "max_tokens": args.max_tokens,
                "max_batches": args.max_batches,
                "batching_cost": "frames * atoms",
                "shuffle": False,
                "replacement": False,
            },
            "store_counts": {str(store.root): len(store) for store in stores},
            "batch_count": len(batches),
            "sample_count": len(selected),
            "sample_count_by_bucket": {
                bucket: sum(1 for entry in selected if entry["time_bucket_id"] == bucket)
                for bucket in sorted({entry["time_bucket_id"] for entry in selected})
            },
            "selected_samples": [
                {
                    "sample_id": entry["sample_id"],
                    "store": str(stores[entry["store_number"]].root),
                    "local_index": entry["local_index"],
                    "atoms": entry["atoms"],
                    "frames": entry["frames"],
                    "time_bucket_id": entry["time_bucket_id"],
                    "effective_tokens": entry["effective_tokens"],
                }
                for entry in selected
            ],
        }
        (args.output_dir / "selection.json").write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        status.update({"status": "ready", "checkpoint": resolution, "selection": selection})
        status_path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"status": "ready", "batches": len(batches), "samples": len(selected), "by_bucket": selection["sample_count_by_bucket"]}, sort_keys=True), flush=True)

        if args.dry_run:
            status["status"] = "dry_run_complete"
            status["elapsed_s"] = time.perf_counter() - started
            status_path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            return 0

        if not PVB_ORIGIN.is_dir():
            raise FileNotFoundError(PVB_ORIGIN)
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")

        # Force unpickling to resolve module.model from the pristine checkout.
        os.chdir(PVB_ORIGIN)
        if str(PVB_ORIGIN) not in sys.path:
            sys.path.insert(0, str(PVB_ORIGIN))
        set_seed(args.seed)
        model = torch.load(checkpoint, map_location=device, weights_only=False).to(device).eval()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        print(json.dumps({
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": resolution["sha256"],
            "model_class": f"{type(model).__module__}.{type(model).__name__}",
            "model_using_ode": bool(getattr(model, "using_ode", False)),
            "model_k_neighbors": int(getattr(model, "k_neighbors", -1)),
            "device": str(device),
        }, sort_keys=True), flush=True)

        store_by_number = {index: store for index, store in enumerate(stores)}
        for sample_number, entry in enumerate(selected, start=1):
            record = store_by_number[entry["store_number"]].read(entry["local_index"])
            target = torch.as_tensor(np.asarray(record["x"], dtype=np.float32)[: args.rollout_frames], device=device)
            if target.shape[0] != args.rollout_frames:
                raise ValueError(f"{entry['sample_id']} has {target.shape[0]} frames, expected {args.rollout_frames}")
            batch = make_legacy_batch(record, device)
            positions = [batch["x0"].detach().clone()]
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            sample_started = time.perf_counter()
            with torch.no_grad():
                for _ in range(1, args.rollout_frames):
                    prediction = model.inference(batch, sde_step=args.sde_steps)
                    positions.append(prediction.detach().clone())
                    batch["x0"] = prediction
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - sample_started
            prediction = torch.stack(positions, dim=0)
            rollout = metrics_for(prediction, target, record)
            direct = _direct_one_step_metrics(prediction, target, record)
            row = {
                "sample_id": entry["sample_id"],
                "source": record.get("source"),
                "time_bucket_id": record.get("time_bucket_id", entry["time_bucket_id"]),
                "atoms": int(target.shape[1]),
                "native_delta_time_ps": float(np.asarray(record["delta_time_ps"])[0]),
                "direct_one_step": direct,
                "rollout16": rollout,
                "elapsed_s": elapsed,
            }
            rows.append(row)
            per_sample.append(row)
            if sample_number == 1 or sample_number == len(selected) or sample_number % 8 == 0:
                print(json.dumps({
                    "sample": sample_number,
                    "total": len(selected),
                    "sample_id": entry["sample_id"],
                    "bucket": row["time_bucket_id"],
                    "atoms": row["atoms"],
                    "elapsed_s": elapsed,
                    "direct_rmsd": direct["rmsd"],
                    "rollout_future_rmsd": rollout["future"]["rmsd"],
                }, sort_keys=True), flush=True)

        by_bucket: dict[str, dict[str, Any]] = {}
        for bucket in sorted({row["time_bucket_id"] for row in rows}):
            bucket_rows = [row for row in rows if row["time_bucket_id"] == bucket]
            by_bucket[bucket] = {
                "sample_count": len(bucket_rows),
                "atoms_mean": float(sum(row["atoms"] for row in bucket_rows) / len(bucket_rows)),
                "native_delta_time_ps": float(sum(row["native_delta_time_ps"] for row in bucket_rows) / len(bucket_rows)),
                "direct_one_step": mean_dict([row["direct_one_step"] for row in bucket_rows]),
                "rollout16": mean_dict([row["rollout16"] for row in bucket_rows]),
                "runtime": {
                    "wall_time_s": float(sum(row["elapsed_s"] for row in bucket_rows)),
                    "samples_per_s": float(len(bucket_rows) / sum(row["elapsed_s"] for row in bucket_rows)),
                },
            }

        peak_gpu = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        report = _base_report(args, resolution)
        report.update({
            "status": "complete",
            "checkpoint": resolution,
            "selection": {
                "file": str(args.output_dir / "selection.json"),
                "batch_count": len(batches),
                "sample_count": len(selected),
                "sample_count_by_bucket": selection["sample_count_by_bucket"],
            },
            "runtime": {
                **report["runtime"],
                "device": str(device),
                "total_wall_time_s": float(time.perf_counter() - started),
                "inference_wall_time_s": float(sum(row["elapsed_s"] for row in rows)),
                "peak_gpu_bytes": peak_gpu,
            },
            "by_time_bucket": by_bucket,
            "per_sample": per_sample,
        })
        report["blockers"] = [
            "Preferred stores are npz-v1 clips, not the original PVB gzip-JSON/PDB input contract; the read-only adapter is confined to this output directory.",
            "No ATLAS or MISATO original-format migrated store was found under /data5/PVB; the only candidate there is PDB data and was excluded.",
        ]
        (args.output_dir / "baseline_a_metrics.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        write_csv(args.output_dir / "baseline_a_per_sample.csv", [
            {
                "sample_id": row["sample_id"],
                "source": row["source"],
                "time_bucket_id": row["time_bucket_id"],
                "atoms": row["atoms"],
                "native_delta_time_ps": row["native_delta_time_ps"],
                "rollout_frame0_rmsd": row["rollout16"]["frame0"]["rmsd"],
                "rollout_future_rmsd": row["rollout16"]["future"]["rmsd"],
                "rollout_future_drmsd": row["rollout16"]["future"]["drmsd"],
                "rollout_future_bond_rmse": row["rollout16"]["future"]["bond_rmse"],
                "rollout_future_contact_error": row["rollout16"]["future"]["contact_error"],
                "rollout_future_clash_rate": row["rollout16"]["future"]["clash_rate"],
                "rollout_velocity_rmse": row["rollout16"]["velocity_rmse"],
                "rollout_acceleration_rmse": row["rollout16"]["acceleration_rmse"],
                "rollout_frequency_retention": row["rollout16"]["frequency_retention"],
                "one_step_rmsd": row["direct_one_step"]["rmsd"],
                "one_step_drmsd": row["direct_one_step"]["drmsd"],
                "one_step_bond_rmse": row["direct_one_step"]["bond_rmse"],
                "one_step_contact_error": row["direct_one_step"]["contact_error"],
                "one_step_clash_rate": row["direct_one_step"]["clash_rate"],
                "elapsed_s": row["elapsed_s"],
            }
            for row in rows
        ])
        status.update({"status": "complete", "metrics": str(args.output_dir / "baseline_a_metrics.json"), "elapsed_s": time.perf_counter() - started})
        status_path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"complete": True, "output": str(args.output_dir / "baseline_a_metrics.json"), "samples": len(rows), "by_bucket": selection["sample_count_by_bucket"], "peak_gpu_bytes": peak_gpu}, sort_keys=True), flush=True)
        return 0
    except Exception as exc:
        status.update({
            "status": "blocked",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "completed_samples": len(rows),
            "elapsed_s": time.perf_counter() - started,
            "blocker": "Baseline A did not produce complete metrics; see error and compatibility evidence above.",
        })
        status_path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        raise
    finally:
        for store in stores:
            store.close()


if __name__ == "__main__":
    raise SystemExit(main())
