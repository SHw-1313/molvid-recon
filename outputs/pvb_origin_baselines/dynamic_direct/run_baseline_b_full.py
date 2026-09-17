#!/usr/bin/env python3
"""Full Baseline B launcher with an explicit machine-readable run manifest."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
import traceback
from pathlib import Path


def load_v3():
    path = Path(__file__).with_name("run_baseline_b_v3.py")
    spec = importlib.util.spec_from_file_location("pvb_origin_baseline_v3", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def argument_value(name: str, default: str | None = None) -> str | None:
    try:
        index = sys.argv.index(name)
    except ValueError:
        return default
    if index + 1 >= len(sys.argv):
        return default
    return sys.argv[index + 1]


def write_manifest(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    output_dir = Path(argument_value("--output-dir", ".")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "run_manifest.json"
    started = time.time()
    manifest = {
        "schema_version": "pvb.baseline.dynamic_direct.manifest.v1",
        "baseline": "B",
        "status": "running",
        "started_unix_s": started,
        "launcher": "enter-container",
        "environment": "torch-ito",
        "cwd_expected": "/workspace/PVB_origin",
        "source_root": "/workspace/PVB_origin",
        "source_untouched": True,
        "command_arguments": sys.argv,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "requested": {
            "seed": int(argument_value("--seed", "20260810")),
            "max_tokens": int(argument_value("--max-tokens", "80000")),
            "max_batches": int(argument_value("--max-batches", "32")),
            "rollout_frames": int(argument_value("--rollout-frames", "16")),
            "rollout_transitions": int(argument_value("--rollout-frames", "16")) - 1,
            "sde_steps": int(argument_value("--sde-steps", "10")),
            "shuffle": False,
            "replacement": False,
            "pdb_included": False,
        },
        "data_roots": {
            "atlas_valid": argument_value("--atlas-valid"),
            "misato_valid": argument_value("--misato-valid"),
            "fallback_search_root": "/data5/PVB",
            "fallback_used": False,
        },
        "checkpoint": argument_value("--checkpoint"),
        "blockers": [
            "PVB_origin's legacy gzip-JSON MMAPDataset cannot consume the preferred npz-v1 clip stores; an output-only read adapter is used.",
            "No ATLAS/MISATO original-format migrated stores were found under /data5/PVB; the only original-format candidate was a PDB store and is excluded.",
        ],
    }
    write_manifest(manifest_path, manifest)
    try:
        v3 = load_v3()
        runner = v3.load_runner()
        runner.one_step_metrics = v3.fixed_one_step_metrics
        result = runner.main()
        metrics_path = output_dir / "baseline_b_metrics.json"
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            data = metrics.get("data", {})
            inference = metrics.get("inference", {})
            actual_batches = int(data.get("batch_count", 0))
            actual_samples = int(data.get("sample_count", 0))
            requested = manifest["requested"]
            manifest.update(
                {
                    "status": "completed",
                    "finished_unix_s": time.time(),
                    "actual": {
                        "batch_count": actual_batches,
                        "sample_count": actual_samples,
                        "sample_count_by_bucket": data.get("sample_count_by_bucket", {}),
                        "rollout_frames": int(inference.get("rollout_frames", 0)),
                        "rollout_transitions": int(inference.get("rollout_transition_count", 0)),
                        "sde_steps": int(inference.get("sde_steps", 0)),
                    },
                    "truncated": bool(
                        actual_batches < requested["max_batches"]
                        or int(inference.get("rollout_frames", 0)) < requested["rollout_frames"]
                    ),
                    "result_json": str(metrics_path),
                    "result_csv": str(output_dir / "baseline_b_per_sample.csv"),
                }
            )
        else:
            manifest.update({"status": "failed", "finished_unix_s": time.time(), "blocker": "runner returned without baseline_b_metrics.json"})
        write_manifest(manifest_path, manifest)
        return int(result or 0)
    except BaseException as exc:
        manifest.update(
            {
                "status": "failed",
                "finished_unix_s": time.time(),
                "truncated": True,
                "blocker": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )
        write_manifest(manifest_path, manifest)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
