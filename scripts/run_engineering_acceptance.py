#!/usr/bin/env python3
"""Run the CUDA-hard-fail engineering acceptance for the v1.2 graph repair.

This runner intentionally does not catch CUDA/device errors or switch to CPU.
It writes evidence only after the requested gate has completed.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
import json
import math
import re
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
import yaml
from torch import Tensor, nn

from data.clip_batching import make_clip_dataloader
from data.clip_dataset import ClipBatch, ClipMMapDataset, collate_clip_records
from module.bond_sources import build_canonical_reference_index
from module.multiframe_codec import PVBFrameEncoder
from module.neighbor_graph import CudaRadiusNeighborList
from trainer.codec_losses import CodecLossWeights, compute_codec_losses
from trainer.codec_contract import require_contract_equal
from trainer.codec_trainer import (
    CODEC_CHECKPOINT_SCHEMA,
    LEGACY_CODEC_CHECKPOINT_SCHEMA,
    CodecTrainConfig,
    CodecTrainer,
    PVBCodecModel,
    TimeBucketSpec,
    prepare_batch_then_to_device,
)

ROOT = Path(__file__).resolve().parents[1]
REAL_TRAIN = ROOT / "outputs/atlas_selected_trajectories/clip_store/train"
REAL_VALID = ROOT / "outputs/atlas_selected_trajectories/clip_store/valid"
BASELINE_DIR = ROOT / "outputs/engineering_v1/baseline_contract"
REPAIRED_REFERENCE_DIR = BASELINE_DIR / "repaired_cuda_reference"
REFERENCE_MANIFEST = REPAIRED_REFERENCE_DIR / "manifest.json"
CHECKPOINT_ROOT = ROOT / "outputs/atlas_selected_trajectories"
DEVICE = torch.device("cuda:0")

CONTROLS: dict[str, dict[str, Any]] = {
    "ratio1_no_temporal": {
        "temporal_layers": 0,
        "temporal_ratio": 1,
        "checkpoint": CHECKPOINT_ROOT / "ratio1_no_temporal/codec_step_00007200.pt",
    },
    "ratio1_temporal": {
        "temporal_layers": 1,
        "temporal_ratio": 1,
        "checkpoint": CHECKPOINT_ROOT / "ratio1_temporal/codec_step_00007200.pt",
    },
    "ratio4_temporal": {
        "temporal_layers": 1,
        "temporal_ratio": 4,
        "checkpoint": CHECKPOINT_ROOT / "ratio4_temporal/codec_step_00007200.pt",
    },
}


def require_cuda() -> dict[str, Any]:
    """Fail before any acceptance work if the requested production runtime is absent."""

    if not torch.cuda.is_available():
        raise RuntimeError(
            "PHASE_ACCEPTANCE_HARD_FAIL: torch.cuda.is_available() is False; "
            "GPU is required and no CPU fallback is permitted"
        )
    try:
        import torch_cluster
        from torch_cluster import radius_graph
    except Exception as exc:  # pragma: no cover - environment failure
        raise RuntimeError(
            "PHASE_ACCEPTANCE_HARD_FAIL: CUDA torch_cluster.radius_graph is unavailable"
        ) from exc
    if not callable(radius_graph):
        raise RuntimeError(
            "PHASE_ACCEPTANCE_HARD_FAIL: torch_cluster.radius_graph is not callable"
        )
    props = torch.cuda.get_device_properties(DEVICE)
    return {
        "device": str(DEVICE),
        "cuda": torch.version.cuda,
        "torch": torch.__version__,
        "gpu": props.name,
        "total_memory_bytes": int(props.total_memory),
        "torch_cluster": getattr(torch_cluster, "__version__", "unknown"),
    }


def load_yaml() -> dict[str, Any]:
    with (ROOT / "config/codec_engineering_full.yaml").open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict) or not isinstance(value.get("acceptance"), dict):
        raise RuntimeError("complete engineering config is missing its acceptance section")
    return value


def _finite(value: Tensor, name: str) -> None:
    if not torch.isfinite(value).all():
        raise RuntimeError(f"{name} contains NaN or Inf")


def _relative(a: Tensor | float, b: Tensor | float) -> float:
    """Relative error of values, not a difference of their norms."""

    if isinstance(a, Tensor) or isinstance(b, Tensor):
        left = a.detach().float() if isinstance(a, Tensor) else torch.as_tensor(a, dtype=torch.float32)
        right = b.detach().float() if isinstance(b, Tensor) else torch.as_tensor(b, dtype=torch.float32)
        return float((left - right).norm().cpu()) / max(float(right.norm().cpu()), 1e-8)
    return abs(float(a) - float(b)) / max(abs(float(b)), 1e-8)


def _max_abs(a: Tensor, b: Tensor) -> float:
    return float((a.detach().float() - b.detach().float()).abs().max().cpu()) if a.numel() else 0.0


def _sorted_unique(codes: Tensor) -> Tensor:
    return torch.unique(codes.to(dtype=torch.long), sorted=True)


def _set_difference(left: Tensor, right: Tensor) -> Tensor:
    left = _sorted_unique(left)
    right = _sorted_unique(right)
    if not left.numel():
        return left
    if not right.numel():
        return left
    positions = torch.searchsorted(right, left)
    safe = positions.clamp(max=right.numel() - 1)
    present = (positions < right.numel()) & (right[safe] == left)
    return left[~present]


def _decode_codes(codes: Tensor, atom_count: int, *, limit: int = 64) -> list[dict[str, int]]:
    result = []
    for code in codes[:limit].tolist():
        code = int(code)
        result.append({"source": code // atom_count, "target": code % atom_count})
    return result


def _record(
    index: int,
    *,
    atoms: int = 8,
    frames: int = 4,
    topology_id: str | None = None,
) -> dict[str, Any]:
    """Create one deterministic, finite trajectory clip for CUDA acceptance."""

    base = torch.zeros(atoms, 3, dtype=torch.float32)
    base[:, 0] = torch.arange(atoms, dtype=torch.float32) * 1.25
    base[:, 1] = (torch.arange(atoms, dtype=torch.float32) % 2) * 0.8
    times = torch.arange(frames, dtype=torch.float32) * 100.0
    x = torch.stack(
        [base + torch.stack((0.02 * t * torch.ones(atoms), 0.01 * t * torch.arange(atoms), torch.zeros(atoms)), dim=1)
         for t in range(frames)],
        dim=0,
    )
    bonds = []
    for atom in range(atoms - 1):
        bonds.extend(((atom, atom + 1), (atom + 1, atom)))
    bond_index = torch.tensor(bonds, dtype=torch.long).t().contiguous()
    topology = topology_id or f"tiny_topology_{index}"
    return {
        "schema_version": "pvb.clip.v1",
        "sample_id": f"tiny_{index:03d}",
        "topology_id": topology,
        "task": "trajectory",
        "time_bucket_id": "dt_100ps",
        "time_ps": times.numpy(),
        "delta_time_ps": torch.diff(times).numpy(),
        "x": x.numpy(),
        "bpos": x.numpy(),
        "atype": (torch.arange(atoms, dtype=torch.long) % 6 + 1).numpy(),
        "btype": torch.zeros(atoms, dtype=torch.long).numpy(),
        "block_id": torch.arange(atoms, dtype=torch.long).numpy(),
        "component_id": torch.zeros(atoms, dtype=torch.long).numpy(),
        "atom_source_index": torch.arange(atoms, dtype=torch.long).numpy(),
        "atom_identity": [f"tiny_{index}:atom_{atom}" for atom in range(atoms)],
        "edge_mask": torch.zeros(atoms, dtype=torch.long).numpy(),
        "loss_mask": torch.ones(atoms, dtype=torch.bool).numpy(),
        "align_mask": torch.ones(atoms, dtype=torch.bool).numpy(),
        "bond_index": bond_index.numpy(),
    }


def _tiny_records(count: int) -> list[dict[str, Any]]:
    return [_record(index) for index in range(count)]


def _make_model(control: Mapping[str, Any], *, bond_mode: str = "topology") -> PVBCodecModel:
    return PVBCodecModel(
        hidden_channels=128 if control.get("full", False) else 16,
        spatial_layers=2 if control.get("full", False) else 1,
        temporal_layers=int(control.get("temporal_layers", 1)),
        temporal_ratio=int(control.get("temporal_ratio", 1)),
        num_rbf=50 if control.get("full", False) else 8,
        num_heads=8 if control.get("full", False) else 2,
        cutoff_lower=0.0,
        cutoff_upper=5.0,
        max_num_neighbors=32 if control.get("full", False) else 8,
        neighbor_backend="cuda_radius",
        bond_construction={"mode": bond_mode},
        spatial_execution={"mode": "full"},
    )


def _load_control_model(
    name: str,
    *,
    allow_legacy: bool,
    legacy_bond_mode: str | None,
) -> PVBCodecModel:
    spec = CONTROLS[name]
    checkpoint = Path(spec["checkpoint"])
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"{name} checkpoint is not a mapping: {checkpoint}")
    schema = str(payload.get("schema_version", ""))
    expected = _make_model({**spec, "full": True}, bond_mode="topology")
    if schema == CODEC_CHECKPOINT_SCHEMA:
        required = {"model_contract", "model_state", "config", "normalization_stats", "optimizer_contract"}
        missing = required.difference(payload)
        if missing:
            raise RuntimeError(f"{name} checkpoint is missing fields {sorted(missing)}")
        require_contract_equal(
            expected.model_contract(),
            payload["model_contract"],
            label=f"{name} acceptance model contract",
        )
        model = PVBCodecModel.from_model_contract(payload["model_contract"])
    elif schema == LEGACY_CODEC_CHECKPOINT_SCHEMA:
        if not allow_legacy or legacy_bond_mode is None:
            raise RuntimeError(
                f"{name} is legacy checkpoint v1 without a model/graph contract; "
                "pass --allow-legacy-checkpoint and explicit --legacy-bond-mode"
            )
        if legacy_bond_mode != "topology":
            raise RuntimeError(
                f"{name} acceptance controls require explicit legacy_bond_mode='topology', "
                f"got {legacy_bond_mode!r}"
            )
        model = expected
    else:
        raise RuntimeError(
            f"{name} checkpoint schema {schema!r} is unsupported; expected "
            f"{CODEC_CHECKPOINT_SCHEMA!r} or explicit legacy {LEGACY_CODEC_CHECKPOINT_SCHEMA!r}"
        )
    state = payload.get("model_state")
    if not isinstance(state, Mapping):
        raise RuntimeError(f"{name} checkpoint has no model_state mapping")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"{name} checkpoint state mismatch: missing={missing}, unexpected={unexpected}"
        )
    model.to(DEVICE)
    return model


def _real_batch() -> tuple[ClipMMapDataset, ClipBatch, list[dict[str, Any]]]:
    dataset = ClipMMapDataset(REAL_TRAIN)
    metadata = json.loads((BASELINE_DIR / "metadata.json").read_text(encoding="utf-8"))
    expected = tuple(metadata["fixed_sample_id"])
    records = [dataset[index] for index in range(len(expected))]
    actual = tuple(str(record["sample_id"]) for record in records)
    if actual != expected:
        raise RuntimeError(f"fixed batch changed: expected {expected}, got {actual}")
    batch = collate_clip_records(records)
    shape = tuple(int(value) for value in batch.x.shape)
    if shape != (16, 4435, 3):
        raise RuntimeError(f"fixed batch shape changed: {shape}")
    return dataset, batch, records


def _graph_invariants(graph: Any) -> dict[str, Any]:
    required = (
        "pos", "z", "b", "batch", "edge_index", "edge_weight", "edge_vec", "bond_type",
        "bond_index", "distance_edge_index", "distance_edge_weight", "distance_edge_vec",
    )
    for name in required:
        value = getattr(graph, name)
        if not isinstance(value, Tensor) or value.device.type != "cuda":
            raise RuntimeError(f"production graph field {name} is not a CUDA tensor")
        _finite(value.float(), f"graph.{name}")
    if graph.backend_used != "cuda_radius":
        raise RuntimeError(f"unexpected graph backend: {graph.backend_used}")
    edge = graph.edge_index
    if edge.numel() and torch.any(graph.batch[edge[0]] != graph.batch[edge[1]]):
        raise RuntimeError("graph contains a cross-frame/sample edge")
    n = int(graph.pos.shape[0])
    codes = edge[0] * n + edge[1]
    if _sorted_unique(codes).numel() != codes.numel():
        raise RuntimeError("final graph contains duplicate directed edges")
    bond_codes = graph.bond_index[0] * n + graph.bond_index[1]
    flag_codes = codes[graph.bond_type == 1]
    if flag_codes.numel() != bond_codes.numel() or not torch.equal(
        torch.sort(flag_codes).values, torch.sort(bond_codes).values
    ):
        raise RuntimeError("bond_type does not mark each supplied directed bond exactly once")
    if int(graph.bond_type.sum()) != int(graph.bond_index.shape[1]):
        raise RuntimeError("bond_type count disagrees with replicated bond_index")
    return {
        "nodes": n,
        "edges": int(edge.shape[1]),
        "distance_edges": int(graph.distance_edge_index.shape[1]),
        "bond_edges": int(graph.bond_index.shape[1]),
        "bond_flags": int(graph.bond_type.sum()),
        "max_in_degree": int(torch.bincount(edge[1], minlength=n).max().item()) if edge.numel() else 0,
    }


def _legacy_cap_audit(graph: Any) -> dict[str, Any]:
    """Compare the historical CPU graph only as an explicit, non-silent audit."""

    reference = torch.load(
        BASELINE_DIR / "ratio4_temporal.pt", map_location="cpu", weights_only=False
    )
    old_edge = reference["edge_index"].to(dtype=torch.long)
    old_flag = reference["bond_type"].to(dtype=torch.long)
    n = int(reference["graph_nodes"])
    new_edge = graph.edge_index.detach().cpu().to(dtype=torch.long)
    new_flag = graph.bond_type.detach().cpu().to(dtype=torch.long)
    old_codes = old_edge[0] * n + old_edge[1]
    new_codes = new_edge[0] * n + new_edge[1]
    removed = _set_difference(old_codes, new_codes)
    added = _set_difference(new_codes, old_codes)
    old_bond = old_codes[old_flag == 1]
    new_bond = new_codes[new_flag == 1]
    bond_removed = _set_difference(old_bond, new_bond)
    bond_added = _set_difference(new_bond, old_bond)
    old_distance = old_codes[old_flag == 0]
    new_distance = new_codes[new_flag == 0]
    distance_removed = _set_difference(old_distance, new_distance)
    distance_added = _set_difference(new_distance, old_distance)
    del reference, old_edge, old_flag
    return {
        "historical_cpu_backend": "torch_cluster.radius_graph CPU/nanoflann",
        "repaired_cuda_policy": "torch_cluster.radius_graph CUDA source-order cap",
        "strict_edge_set_equal": bool(not removed.numel() and not added.numel()),
        "historical_edge_count": int(old_codes.numel()),
        "repaired_edge_count": int(new_codes.numel()),
        "removed_edges": int(removed.numel()),
        "added_edges": int(added.numel()),
        "removed_distance_edges": int(distance_removed.numel()),
        "added_distance_edges": int(distance_added.numel()),
        "removed_bond_edges": int(bond_removed.numel()),
        "added_bond_edges": int(bond_added.numel()),
        "removed_edge_examples": _decode_codes(removed, n),
        "added_edge_examples": _decode_codes(added, n),
        "policy_reconciled": True,
        "old_cpu_numerical_identity_claimed": False,
        "note": (
            "The old CPU extension uses nanoflann with an unsorted radius result; "
            "the CUDA extension scans source indices in order. The repaired contract "
            "makes the CUDA cap policy explicit and keeps this historical difference "
            "auditable instead of silently claiming identity."
        ),
    }


def _real_loader_audit(dataset: ClipMMapDataset) -> dict[str, Any]:
    started = time.perf_counter()
    loader = make_clip_dataloader(
        dataset,
        max_tokens=80000,
        collate_fn=collate_clip_records,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        trusted_store_fast_path=True,
        strict_record_validation=False,
        oversize_policy="error",
        seed=20260819,
        shuffle=False,
        replacement=False,
    )
    batch = next(iter(loader))
    if not batch.x.is_pinned():
        raise RuntimeError("multi-worker real batch was not pinned")
    report = {
        "elapsed_s": time.perf_counter() - started,
        "batch_shape": [int(value) for value in batch.x.shape],
        "pinned": bool(batch.x.is_pinned()),
        "num_batches": int(len(loader)),
        "eligibility": loader.batch_sampler.eligibility_report(),
    }
    del loader, batch
    gc.collect()
    try:
        make_clip_dataloader(
            dataset,
            max_tokens=20000,
            collate_fn=collate_clip_records,
            num_workers=0,
            oversize_policy="error",
            seed=20260819,
            shuffle=False,
            replacement=False,
        )
    except ValueError as exc:
        report["oversize_error"] = str(exc)
    else:
        raise RuntimeError("oversize clips were silently accepted or excluded")
    return report


def _loss_for_output(output: Any, batch: ClipBatch) -> dict[str, float]:
    losses = compute_codec_losses(
        output,
        batch,
        weights=CodecLossWeights(coordinate=1.0, local=0.1, bond=0.1, velocity=0.1, acceleration=0.05),
    )
    return {key: float(value.detach().cpu()) for key, value in losses.items()}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_repaired_references() -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Load an external immutable repaired-CUDA reference set.

    The acceptance runner never creates or overwrites these files. The
    separate reference-builder command records their hashes in manifest.json;
    this gate only verifies and consumes that manifest.
    """

    if not REFERENCE_MANIFEST.is_file():
        raise RuntimeError(
            "PHASE_A_HARD_FAIL: immutable repaired-CUDA reference manifest is missing at "
            f"{REFERENCE_MANIFEST}; run scripts/build_engineering_reference.py separately "
            "and review the produced hashes before running acceptance"
        )
    manifest = json.loads(REFERENCE_MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "pvb.codec.engineering.reference_manifest.v1":
        raise RuntimeError("unsupported repaired-CUDA reference manifest schema")
    controls = manifest.get("controls")
    if not isinstance(controls, Mapping):
        raise RuntimeError("repaired-CUDA reference manifest has no controls mapping")
    references: dict[str, dict[str, Any]] = {}
    for name in CONTROLS:
        item = controls.get(name)
        if not isinstance(item, Mapping):
            raise RuntimeError(f"reference manifest is missing control {name!r}")
        relative = Path(str(item.get("file", "")))
        reference_path = (REPAIRED_REFERENCE_DIR / relative).resolve()
        if reference_path.parent != REPAIRED_REFERENCE_DIR.resolve():
            raise RuntimeError(f"reference path escapes immutable reference directory: {relative}")
        if not reference_path.is_file():
            raise RuntimeError(f"immutable reference file is missing: {reference_path}")
        expected_hash = str(item.get("sha256", ""))
        actual_hash = _sha256_file(reference_path)
        if not expected_hash or actual_hash != expected_hash:
            raise RuntimeError(
                f"immutable reference hash mismatch for {name}: "
                f"expected={expected_hash!r}, actual={actual_hash!r}"
            )
        payload = torch.load(reference_path, map_location="cpu", weights_only=False)
        if not isinstance(payload, Mapping):
            raise RuntimeError(f"immutable reference {name} is not a mapping")
        required = {
            "control",
            "parameter_names",
            "parameter_shapes",
            "parameter_count",
            "parameter_gradients",
            "position_gradient",
            "encoded_h",
            "encoded_v",
            "decoded_x_coarse",
            "decoded_x_hat",
            "losses",
            "edge_index",
            "bond_type",
        }
        missing = required.difference(payload)
        if missing:
            raise RuntimeError(f"immutable reference {name} is missing fields {sorted(missing)}")
        if str(payload["control"]) != name:
            raise RuntimeError(f"immutable reference control mismatch for {name}")
        if not isinstance(payload["parameter_names"], list) or not isinstance(payload["parameter_shapes"], Mapping):
            raise RuntimeError(f"immutable reference {name} parameter signature is malformed")
        if not isinstance(payload["losses"], Mapping):
            raise RuntimeError(f"immutable reference {name} losses are not a mapping")
        if not isinstance(payload["parameter_gradients"], Mapping):
            raise RuntimeError(f"immutable reference {name} gradients are not a mapping")
        if set(payload["parameter_names"]) != set(payload["parameter_gradients"]):
            raise RuntimeError(f"immutable reference {name} gradient coverage is incomplete")
        for field in (
            "encoded_h",
            "encoded_v",
            "decoded_x_coarse",
            "decoded_x_hat",
            "position_gradient",
            "edge_index",
            "bond_type",
        ):
            value = payload[field]
            if not isinstance(value, Tensor):
                raise RuntimeError(f"immutable reference {name}.{field} is not a tensor")
            _finite(value.float(), f"reference.{name}.{field}")
        for parameter_name, gradient in payload["parameter_gradients"].items():
            if not isinstance(gradient, Tensor):
                raise RuntimeError(
                    f"immutable reference {name}.grad.{parameter_name} is not a tensor"
                )
            _finite(gradient.float(), f"reference.{name}.grad.{parameter_name}")
        references[name] = dict(payload)
        references[name]["reference_sha256"] = actual_hash
    return references, dict(manifest)


def _assert_tensor_close(
    candidate: Tensor,
    reference: Tensor,
    *,
    rtol: float,
    atol: float,
    label: str,
) -> dict[str, float]:
    left = candidate.detach().float().cpu()
    right = reference.detach().float().cpu()
    if left.shape != right.shape:
        raise RuntimeError(f"{label} shape changed: {tuple(left.shape)} != {tuple(right.shape)}")
    max_abs = _max_abs(left, right)
    relative = _relative(left, right)
    if not torch.allclose(left, right, rtol=float(rtol), atol=float(atol)):
        raise RuntimeError(
            f"{label} FP32 regression exceeded tolerance: max_abs={max_abs:.6g}, "
            f"relative={relative:.6g}, rtol={rtol}, atol={atol}"
        )
    return {"max_abs": max_abs, "relative_norm_error": relative, "rtol": float(rtol), "atol": float(atol)}


def _compare_fp32_evidence(
    name: str,
    candidate: Mapping[str, Any],
    reference: Mapping[str, Any],
    acceptance: Mapping[str, Any],
) -> dict[str, Any]:
    tensor_measurements = {}
    for key in ("encoded_h", "decoded_x_coarse", "decoded_x_hat"):
        tensor_measurements[key] = _assert_tensor_close(
            candidate[key],
            torch.as_tensor(reference[key]),
            rtol=float(acceptance["fp32_scalar_rtol"]),
            atol=float(acceptance["fp32_scalar_atol"]),
            label=f"{name}.{key}",
        )
    tensor_measurements["encoded_v"] = _assert_tensor_close(
        candidate["encoded_v"],
        torch.as_tensor(reference["encoded_v"]),
        rtol=float(acceptance["fp32_vector_rtol"]),
        atol=float(acceptance["fp32_vector_atol"]),
        label=f"{name}.encoded_v",
    )
    candidate_losses = candidate["losses"]
    reference_losses = reference["losses"]
    if set(candidate_losses) != set(reference_losses):
        raise RuntimeError(f"{name} loss components changed")
    loss_measurements = {}
    for key in sorted(reference_losses):
        actual = float(candidate_losses[key])
        expected = float(reference_losses[key])
        relative = _relative(actual, expected)
        if key == "total":
            passed = relative <= float(acceptance["total_loss_relative"])
        else:
            passed = math.isclose(
                actual,
                expected,
                rel_tol=float(acceptance["fp32_scalar_rtol"]),
                abs_tol=float(acceptance["fp32_scalar_atol"]),
            )
        if not passed:
            raise RuntimeError(
                f"{name}.{key} loss regression exceeded tolerance: "
                f"actual={actual}, reference={expected}, relative={relative}"
            )
        loss_measurements[key] = {
            "actual": actual,
            "reference": expected,
            "relative_error": relative,
        }

    if int(candidate["parameter_count"]) != int(reference["parameter_count"]):
        raise RuntimeError(
            f"{name} parameter count changed: {candidate['parameter_count']} != "
            f"{reference['parameter_count']}"
        )
    if list(candidate["parameter_names"]) != list(reference["parameter_names"]):
        raise RuntimeError(f"{name} trainable parameter ordering changed")
    if candidate["parameter_shapes"] != reference.get("parameter_shapes", candidate["parameter_shapes"]):
        raise RuntimeError(f"{name} trainable parameter shapes changed")
    if not torch.equal(
        torch.as_tensor(candidate["edge_index"], dtype=torch.long),
        torch.as_tensor(reference["edge_index"], dtype=torch.long),
    ):
        raise RuntimeError(f"{name} CUDA graph edge_index changed")
    if not torch.equal(
        torch.as_tensor(candidate["bond_type"], dtype=torch.long),
        torch.as_tensor(reference["bond_type"], dtype=torch.long),
    ):
        raise RuntimeError(f"{name} CUDA graph bond_type changed")

    candidate_gradients = candidate["parameter_gradients"]
    reference_gradients = reference["parameter_gradients"]
    if set(candidate_gradients) != set(reference_gradients):
        missing = sorted(set(reference_gradients).difference(candidate_gradients))
        extra = sorted(set(candidate_gradients).difference(reference_gradients))
        raise RuntimeError(f"{name} trainable gradient coverage changed: missing={missing[:8]}, extra={extra[:8]}")
    gradient_measurements = {}
    for parameter_name in reference["parameter_names"]:
        gradient_measurements[parameter_name] = _assert_tensor_close(
            candidate_gradients[parameter_name],
            torch.as_tensor(reference_gradients[parameter_name]),
            rtol=float(acceptance["gradient_relative"]),
            atol=float(acceptance["fp32_scalar_atol"]),
            label=f"{name}.grad.{parameter_name}",
        )
    gradient_measurements["position"] = _assert_tensor_close(
        candidate["position_gradient"],
        torch.as_tensor(reference["position_gradient"]),
        rtol=float(acceptance["gradient_relative"]),
        atol=float(acceptance["fp32_vector_atol"]),
        label=f"{name}.grad.position",
    )
    return {
        "reference_sha256": str(reference["reference_sha256"]),
        "tensor_measurements": tensor_measurements,
        "loss_measurements": loss_measurements,
        "gradient_measurements": gradient_measurements,
        "trainable_submodule_coverage": sorted({name.split(".", 1)[0] for name in candidate_gradients}),
    }


def _fixed_control_evidence(
    batch_cpu: ClipBatch,
    references: Mapping[str, Mapping[str, Any]],
    acceptance: Mapping[str, Any],
    *,
    allow_legacy: bool,
    legacy_bond_mode: str | None,
) -> tuple[dict[str, Any], Any]:
    # References are loaded and hash-checked before this function.  This
    # function never writes reference tensors.
    evidence: dict[str, Any] = {}
    ratio4_graph = None
    # The immutable real batch is the one source of coordinate autograd for
    # this gate. Each CUDA transfer creates a fresh non-leaf whose gradient is
    # retained below; no CPU/GPU graph metadata is reconstructed in forward.
    batch_cpu.x.requires_grad_(True)
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
        torch.cuda.synchronize()
        started = time.perf_counter()
        encoded, output = model.forward_with_encoded(batch)
        losses = compute_codec_losses(
            output,
            batch,
            weights=CodecLossWeights(
                coordinate=1.0,
                local=0.1,
                bond=0.1,
                velocity=0.1,
                acceleration=0.05,
            ),
        )
        losses["total"].backward()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started

        graph = encoded.graph
        invariant = _graph_invariants(graph)
        if ratio4_graph is None or name == "ratio4_temporal":
            ratio4_graph = graph
        candidate_gradients: dict[str, Tensor] = {}
        for parameter_name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            if parameter.grad is None:
                raise RuntimeError(
                    f"{name} missing gradient for trainable parameter {parameter_name}"
                )
            _finite(parameter.grad.detach().float(), f"{name}.grad.{parameter_name}")
            candidate_gradients[parameter_name] = parameter.grad.detach().cpu()
        if batch.x.grad is None:
            raise RuntimeError(f"{name} missing coordinate/position gradient")
        _finite(batch.x.grad.detach().float(), f"{name}.grad.position")

        candidate = {
            "parameter_names": list(candidate_gradients),
            "parameter_shapes": {
                key: list(value.shape) for key, value in candidate_gradients.items()
            },
            "parameter_count": int(sum(value.numel() for value in model.parameters())),
            "encoded_h": encoded.h.detach().cpu(),
            "encoded_v": encoded.v.detach().cpu(),
            "decoded_x_coarse": output.x_coarse.detach().cpu(),
            "decoded_x_hat": output.x_hat.detach().cpu(),
            "losses": {
                key: float(value.detach().cpu()) for key, value in losses.items()
            },
            "parameter_gradients": candidate_gradients,
            "position_gradient": batch.x.grad.detach().cpu(),
            "edge_index": graph.edge_index.detach().cpu(),
            "bond_type": graph.bond_type.detach().cpu(),
        }
        for key in ("encoded_h", "encoded_v", "decoded_x_coarse", "decoded_x_hat"):
            _finite(candidate[key].float(), f"{name}.{key}")
        comparison = _compare_fp32_evidence(
            name, candidate, references[name], acceptance
        )
        checkpoint_payload = torch.load(
            CONTROLS[name]["checkpoint"],
            map_location="cpu",
            weights_only=False,
        )
        evidence[name] = {
            "parameter_count": candidate["parameter_count"],
            "trainable_parameter_count": len(candidate_gradients),
            "graph": invariant,
            "losses": candidate["losses"],
            "elapsed_s": elapsed,
            "checkpoint_schema": str(checkpoint_payload.get("schema_version", "")),
            "regression": comparison,
        }
        del checkpoint_payload
        del model, batch, encoded, output, losses, candidate, candidate_gradients
        torch.cuda.empty_cache()
        gc.collect()
    if ratio4_graph is None:
        raise RuntimeError("no fixed control graph was produced")
    return evidence, ratio4_graph

def _bf16_evidence(
    batch_cpu: ClipBatch,
    references: Mapping[str, Mapping[str, Any]],
    acceptance: Mapping[str, Any],
    *,
    allow_legacy: bool,
    legacy_bond_mode: str | None,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for name in CONTROLS:
        model = _load_control_model(
            name,
            allow_legacy=allow_legacy,
            legacy_bond_mode=legacy_bond_mode,
        )
        model.eval()
        model.prepare_batch(batch_cpu)
        batch = batch_cpu.to(DEVICE, non_blocking=True)
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(batch)
        _finite(output.x_hat, f"{name}.bf16.x_hat")
        reference_x_hat = torch.as_tensor(references[name]["decoded_x_hat"], device=DEVICE)
        coordinate_rel = _relative(output.x_hat, reference_x_hat)
        fp32_loss = float(references[name]["losses"]["total"])
        bf16_losses = _loss_for_output(output, batch)
        bf16_loss = float(bf16_losses["total"])
        loss_rel = _relative(bf16_loss, fp32_loss)
        if loss_rel > float(acceptance["bf16_total_loss_relative"]):
            raise RuntimeError(
                f"{name} BF16 total loss relative error {loss_rel:.6g} exceeds "
                f"{acceptance['bf16_total_loss_relative']}"
            )
        if coordinate_rel > float(acceptance["bf16_coordinate_relative"]):
            raise RuntimeError(
                f"{name} BF16 coordinate relative error {coordinate_rel:.6g} exceeds "
                f"{acceptance['bf16_coordinate_relative']}"
            )
        component_errors = {
            key: _relative(float(value), float(references[name]["losses"][key]))
            for key, value in bf16_losses.items()
            if key in references[name]["losses"]
        }
        results[name] = {
            "x_hat_relative_norm_error": coordinate_rel,
            "x_hat_relative_formula": "norm(candidate-reference)/max(norm(reference),1e-8)",
            "total_loss_fp32": fp32_loss,
            "total_loss_bf16": bf16_loss,
            "total_loss_relative_error": loss_rel,
            "component_relative_errors": component_errors,
            "thresholds": {
                "coordinate_relative": float(acceptance["bf16_coordinate_relative"]),
                "total_loss_relative": float(acceptance["bf16_total_loss_relative"]),
            },
        }
        del model, batch, output, reference_x_hat
        torch.cuda.empty_cache()
        gc.collect()
    return results

def _edge_codes(edge_index: Tensor, atom_count: int) -> Tensor:
    edge_index = edge_index.detach().cpu().to(dtype=torch.long)
    if edge_index.numel() == 0:
        return torch.empty(0, dtype=torch.long)
    return torch.sort(edge_index[0] * int(atom_count) + edge_index[1]).values


def _synthetic_cuda_gate(acceptance: Mapping[str, Any]) -> dict[str, Any]:
    # This cap-unsaturated fixture is an explicit CPU dense reference for the
    # CUDA production backend.  It exercises exact edge, output, and gradient
    # equivalence without using the historical saturated real-data graph.
    records = [_record(0, atoms=5, frames=2)]
    batch_cpu = collate_clip_records(records)
    kwargs = {
        "hidden_channels": 16,
        "num_layers": 1,
        "num_rbf": 8,
        "num_heads": 2,
        "cutoff_lower": 0.0,
        "cutoff_upper": 2.5,
        "max_num_neighbors": 16,
        "bond_construction": {"mode": "topology"},
    }
    torch.manual_seed(20260826)
    dense = PVBFrameEncoder(neighbor_backend="dense_test", **kwargs)
    cuda = PVBFrameEncoder(neighbor_backend="cuda_radius", **kwargs).to(DEVICE)
    cuda.load_state_dict(dense.state_dict())
    batch_cpu.x.requires_grad_(True)
    dense.prepare_batch(batch_cpu)
    dense_output = dense(batch_cpu)
    dense_loss = dense_output.h.float().square().mean() + dense_output.v.float().square().mean()
    dense_loss.backward()
    dense_parameter_gradients = {
        key: value.grad.detach().cpu()
        for key, value in dense.named_parameters()
        if value.requires_grad and value.grad is not None
    }
    if batch_cpu.x.grad is None:
        raise RuntimeError("synthetic dense reference did not retain position gradients")
    # ClipBatch.to preserves the autograd edge from a CPU leaf to the
    # transferred CUDA tensor. Freeze the dense reference before the CUDA
    # backward pass; otherwise CUDA gradients accumulate into the CPU leaf and
    # make the reference appear twice as large.
    dense_position_gradient = batch_cpu.x.grad.detach().cpu().clone()
    batch_cpu.x.grad = None

    cuda.prepare_batch(batch_cpu)
    batch_cuda = batch_cpu.to(DEVICE, non_blocking=True)
    batch_cuda.x.retain_grad()
    cuda_output = cuda(batch_cuda)
    cuda_loss = cuda_output.h.float().square().mean() + cuda_output.v.float().square().mean()
    cuda_loss.backward()
    torch.cuda.synchronize()
    if batch_cuda.x.grad is None:
        raise RuntimeError("synthetic CUDA path did not retain position gradients")

    dense_codes = _edge_codes(dense_output.graph.edge_index, 5)
    cuda_codes = _edge_codes(cuda_output.graph.edge_index, 5)
    if not torch.equal(dense_codes, cuda_codes):
        raise RuntimeError("cap-unsaturated CUDA graph edge set differs from dense reference")
    degree = torch.bincount(
        cuda_output.graph.edge_index[1],
        minlength=int(cuda_output.graph.pos.shape[0]),
    )
    if degree.numel() and int(degree.max()) >= 16:
        raise RuntimeError("synthetic CUDA gate unexpectedly saturates the neighbor cap")

    outputs = {
        "scalar_h": _assert_tensor_close(
            cuda_output.h, dense_output.h,
            rtol=float(acceptance["fp32_scalar_rtol"]),
            atol=float(acceptance["fp32_scalar_atol"]),
            label="synthetic.scalar_h",
        ),
        "vector_v": _assert_tensor_close(
            cuda_output.v, dense_output.v,
            rtol=float(acceptance["fp32_vector_rtol"]),
            atol=float(acceptance["fp32_vector_atol"]),
            label="synthetic.vector_v",
        ),
    }
    if set(dense_parameter_gradients) != {
        key for key, value in cuda.named_parameters()
        if value.requires_grad and value.grad is not None
    }:
        raise RuntimeError("synthetic CUDA trainable gradient coverage differs")
    gradient_measurements = {}
    for key, reference_gradient in dense_parameter_gradients.items():
        gradient_measurements[key] = _assert_tensor_close(
            dict(cuda.named_parameters())[key].grad,
            reference_gradient,
            rtol=float(acceptance["gradient_relative"]),
            atol=float(acceptance["fp32_scalar_atol"]),
            label=f"synthetic.grad.{key}",
        )
    gradient_measurements["position"] = _assert_tensor_close(
        batch_cuda.x.grad,
        dense_position_gradient,
        rtol=float(acceptance["gradient_relative"]),
        atol=float(acceptance["fp32_vector_atol"]),
        label="synthetic.grad.position",
    )
    result = {
        "passed": True,
        "backend": "cuda_radius",
        "reference_backend": "dense_test",
        "atom_count": 5,
        "edge_count": int(cuda_codes.numel()),
        "max_degree": int(degree.max()) if degree.numel() else 0,
        "cap": 16,
        "cap_unsaturated": True,
        "exact_edge_set": True,
        "output_measurements": outputs,
        "gradient_measurements": gradient_measurements,
        "gradient_loss": {
            "dense": float(dense_loss.detach().cpu()),
            "cuda": float(cuda_loss.detach().cpu()),
        },
    }
    del dense, cuda, batch_cpu, batch_cuda, dense_output, cuda_output
    torch.cuda.empty_cache()
    gc.collect()
    return result


def _se3_regression(
    batch_cpu: ClipBatch,
    acceptance: Mapping[str, Any],
    *,
    allow_legacy: bool,
    legacy_bond_mode: str | None,
) -> dict[str, Any]:
    name = "ratio4_temporal"
    model = _load_control_model(
        name,
        allow_legacy=allow_legacy,
        legacy_bond_mode=legacy_bond_mode,
    )
    model.eval()
    rotation = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    translation = torch.tensor([2.5, -1.25, 0.75], dtype=torch.float32)
    transformed_x = torch.einsum("ij,tnj->tni", rotation, batch_cpu.x) + translation
    transformed_bpos = torch.einsum("ij,tnj->tni", rotation, batch_cpu.bpos) + translation
    transformed_cpu = replace(batch_cpu, x=transformed_x, bpos=transformed_bpos)
    model.prepare_batch(batch_cpu)
    model.prepare_batch(transformed_cpu)
    original = prepare_batch_then_to_device(model, batch_cpu, DEVICE, non_blocking=True)
    transformed = prepare_batch_then_to_device(model, transformed_cpu, DEVICE, non_blocking=True)
    with torch.no_grad():
        original_encoded, original_output = model.forward_with_encoded(original)
        transformed_encoded, transformed_output = model.forward_with_encoded(transformed)
    expected_h = original_encoded.h
    expected_v = torch.einsum("ij,tnjh->tnih", rotation.to(DEVICE), original_encoded.v)
    expected_coarse = torch.einsum(
        "ij,tnj->tni", rotation.to(DEVICE), original_output.x_coarse
    ) + translation.to(DEVICE)
    expected_hat = torch.einsum(
        "ij,tnj->tni", rotation.to(DEVICE), original_output.x_hat
    ) + translation.to(DEVICE)
    if not torch.equal(
        original_encoded.graph.edge_index,
        transformed_encoded.graph.edge_index,
    ):
        raise RuntimeError("production CUDA SE(3) transform changed discrete graph edges")
    measurements = {
        "scalar_h": _assert_tensor_close(
            transformed_encoded.h,
            expected_h,
            rtol=float(acceptance["fp32_scalar_rtol"]),
            atol=float(acceptance["fp32_scalar_atol"]),
            label="se3.scalar_h",
        ),
        "vector_v": _assert_tensor_close(
            transformed_encoded.v,
            expected_v,
            rtol=float(acceptance["fp32_vector_rtol"]),
            atol=float(acceptance["fp32_vector_atol"]),
            label="se3.vector_v",
        ),
        "decoded_x_coarse": _assert_tensor_close(
            transformed_output.x_coarse,
            expected_coarse,
            rtol=float(acceptance["fp32_scalar_rtol"]),
            atol=float(acceptance["fp32_scalar_atol"]),
            label="se3.decoded_x_coarse",
        ),
        "decoded_x_hat": _assert_tensor_close(
            transformed_output.x_hat,
            expected_hat,
            rtol=float(acceptance["fp32_scalar_rtol"]),
            atol=float(acceptance["fp32_scalar_atol"]),
            label="se3.decoded_x_hat",
        ),
    }
    result = {
        "passed": True,
        "control": name,
        "rotation": rotation.tolist(),
        "translation": translation.tolist(),
        "graph_exact": True,
        "measurements": measurements,
        "thresholds": {
            "scalar": {
                "rtol": float(acceptance["fp32_scalar_rtol"]),
                "atol": float(acceptance["fp32_scalar_atol"]),
            },
            "vector": {
                "rtol": float(acceptance["fp32_vector_rtol"]),
                "atol": float(acceptance["fp32_vector_atol"]),
            },
        },
    }
    del model, original, transformed, original_encoded, transformed_encoded, original_output, transformed_output
    torch.cuda.empty_cache()
    gc.collect()
    return result

def _tiny_loader(records: list[dict[str, Any]], *, replacement: bool) -> Any:
    return make_clip_dataloader(
        records,
        max_tokens=128,
        collate_fn=collate_clip_records,
        num_workers=0,
        pin_memory=False,
        oversize_policy="error",
        seed=20260825,
        shuffle=False,
        replacement=replacement,
    )


def _tiny_refs(records: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    # Keep the same canonical selector and identity-hash contract as real
    # training/evaluation; this fixture has no validation records.
    return build_canonical_reference_index(list(records), source_split="train")


def _prepare_mode(model: PVBCodecModel, trainer: CodecTrainer, records: list[dict[str, Any]], mode: str) -> None:
    if mode == "distance_only":
        trainer.model.prepare_distance_bonds(_tiny_refs(records), device=trainer.device)


def _tiny_config(*, steps: int, precision: str = "fp32") -> CodecTrainConfig:
    return CodecTrainConfig(
        lr=1.0e-3,
        weight_decay=0.0,
        max_steps=steps,
        grad_clip=1.0,
        warmup_steps=20,
        device="cuda",
        precision=precision,
        bucket_specs=(TimeBucketSpec("dt_100ps", 100.0, 1.0, 1.0),),
        loss_schedule=((0, CodecLossWeights(coordinate=1.0)),),
        normalization_min_count=1,
    )


def _run_200_steps(out_dir: Path, *, mode: str = "topology") -> dict[str, Any]:
    records = _tiny_records(8)
    loader = _tiny_loader(records, replacement=True)
    model = _make_model({"temporal_layers": 1, "temporal_ratio": 1}, bond_mode=mode)
    trainer = CodecTrainer(model, loader, config=_tiny_config(steps=200), device=DEVICE)
    _prepare_mode(model, trainer, records, mode)
    log_path = out_dir / f"{mode}_200_steps.jsonl"
    if log_path.exists():
        log_path.unlink()
    started = time.perf_counter()
    trainer.run(max_steps=200, log_path=log_path, log_every=10)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    checkpoint = out_dir / f"{mode}_200_steps.pt"
    trainer.save_checkpoint(checkpoint)
    lines = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    if [int(item["step"]) for item in lines] != list(range(10, 201, 10)):
        raise RuntimeError(f"{mode} 200-step logging is incomplete")
    if not math.isclose(float(lines[0]["learning_rate"]), 1.0e-3 * 10.0 / 20.0, rel_tol=1e-6):
        raise RuntimeError("warmup learning rate was not applied before the first optimizer step")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    required = {"step", "epoch", "batch_in_epoch", "precision", "sampler_state", "model_state", "optimizer_state", "normalization_stats", "config"}
    if required.difference(payload):
        raise RuntimeError(f"checkpoint missing fields: {sorted(required.difference(payload))}")
    # Resume parsing is part of the acceptance: a missing sampler/precision
    # field must not be silently accepted.
    resumed_model = _make_model({"temporal_layers": 1, "temporal_ratio": 1}, bond_mode=mode)
    resumed = CodecTrainer(resumed_model, loader, config=_tiny_config(steps=201), device=DEVICE)
    _prepare_mode(resumed_model, resumed, records, mode)
    resumed.load_checkpoint(checkpoint)
    if resumed.step != 200:
        raise RuntimeError("checkpoint resume did not restore optimizer step")
    report = {
        "mode": mode,
        "steps": trainer.step,
        "elapsed_s": elapsed,
        "log_records": len(lines),
        "first_learning_rate": float(lines[0]["learning_rate"]),
        "last_total": float(lines[-1]["metrics"]["total"]),
        "checkpoint": str(checkpoint),
        "checkpoint_fields": sorted(payload),
        "resumed_step": resumed.step,
        "batches_per_epoch": len(loader),
    }
    del trainer, resumed, model, resumed_model, loader, payload
    torch.cuda.empty_cache()
    gc.collect()
    return report


def _run_five_epochs(out_dir: Path, *, mode: str = "topology") -> dict[str, Any]:
    records = _tiny_records(8)
    loader = _tiny_loader(records, replacement=False)
    sampler = loader.batch_sampler
    coverage: list[dict[str, Any]] = []
    for epoch in range(5):
        sampler.set_epoch(epoch)
        batches = [list(batch) for batch in sampler]
        flattened = [index for batch in batches for index in batch]
        if sorted(flattened) != list(range(len(records))):
            raise RuntimeError(f"{mode} epoch {epoch} does not cover every tiny clip exactly once")
        coverage.append({"epoch": epoch, "batches": batches, "clips": len(flattened)})
    sampler.set_epoch(0)
    model = _make_model({"temporal_layers": 1, "temporal_ratio": 1}, bond_mode=mode)
    steps = 5 * len(loader)
    trainer = CodecTrainer(model, loader, config=_tiny_config(steps=steps), device=DEVICE)
    _prepare_mode(model, trainer, records, mode)
    log_path = out_dir / f"{mode}_five_epochs.jsonl"
    started = time.perf_counter()
    if log_path.exists():
        log_path.unlink()
    trainer.run(max_steps=steps, log_path=log_path, log_every=1)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    if trainer.epoch != 5 or trainer.batch_in_epoch != 0:
        raise RuntimeError(f"{mode} did not finish five exact epochs: epoch={trainer.epoch}, batch={trainer.batch_in_epoch}")
    report = {
        "mode": mode,
        "epochs": 5,
        "steps": trainer.step,
        "batches_per_epoch": len(loader),
        "elapsed_s": elapsed,
        "coverage": coverage,
        "final_epoch": trainer.epoch,
        "final_batch_in_epoch": trainer.batch_in_epoch,
        "log_records": len(log_path.read_text(encoding="utf-8").splitlines()),
    }
    del trainer, model, loader
    torch.cuda.empty_cache()
    gc.collect()
    return report


def _throughput(
    batch_cpu: ClipBatch,
    *,
    allow_legacy: bool,
    legacy_bond_mode: str | None,
) -> dict[str, Any]:
    baseline = json.loads((BASELINE_DIR / "summary.json").read_text(encoding="utf-8"))["controls"]
    report: dict[str, Any] = {}
    for name in CONTROLS:
        model = _load_control_model(
            name,
            allow_legacy=allow_legacy,
            legacy_bond_mode=legacy_bond_mode,
        )
        # CodecTrainer owns the same CPU-registration-before-transfer helper
        # used by evaluators; the timed step must exercise that production path.
        config = _tiny_config(steps=5, precision="bf16")
        trainer = CodecTrainer(model, [batch_cpu], config=config, device=DEVICE)
        trainer.optimizer_step(batch_cpu)
        torch.cuda.synchronize()
        times: list[float] = []
        for _ in range(3):
            started = time.perf_counter()
            trainer.optimizer_step(batch_cpu)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - started)
        median_s = float(np.median(np.asarray(times)))
        tokens = int(batch_cpu.x.shape[0] * batch_cpu.x.shape[1])
        tokens_per_second = tokens / median_s
        baseline_tps = float(baseline[name]["atom_frame_tokens_per_s"])
        report[name] = {
            "timed_step_s": times,
            "median_step_s": median_s,
            "atom_frame_tokens": tokens,
            "atom_frame_tokens_per_s": tokens_per_second,
            "baseline_atom_frame_tokens_per_s": baseline_tps,
            "baseline_multiplier": tokens_per_second / baseline_tps,
            "required_multiplier": 1.20,
            "peak_memory_bytes": int(torch.cuda.max_memory_allocated(DEVICE)),
        }
        del trainer, model
        torch.cuda.empty_cache()
        gc.collect()
    return report

def _source_audit() -> dict[str, Any]:
    graph_source = (ROOT / "module/neighbor_graph.py").read_text(encoding="utf-8")
    union_source = (ROOT / "module/multiframe_codec.py").read_text(encoding="utf-8")
    neighbor_forward_source = inspect.getsource(CudaRadiusNeighborList.__call__)
    encoder_forward_source = inspect.getsource(PVBFrameEncoder.forward)
    encoder_graph_source = inspect.getsource(PVBFrameEncoder.build_external_graph)
    if "positions.cpu" in graph_source or "graph_id.cpu" in graph_source:
        raise RuntimeError("production neighbor source contains a CPU graph round trip")
    if "torch.isin" in union_source:
        raise RuntimeError("production bond union still uses torch.isin")
    if "torch.unique" in union_source:
        raise RuntimeError("production bond union still uses full-edge torch.unique")
    if "sort(edge" in union_source or "sort(distance" in union_source:
        raise RuntimeError("production bond union sorts the full geometric edge set")
    if "neighbor_backend = \"auto\"" in union_source:
        raise RuntimeError("production encoder retains an auto neighbor backend")
    if ".item(" in neighbor_forward_source or ".cpu(" in neighbor_forward_source:
        raise RuntimeError("CUDA neighbor forward contains an implicit scalar/CPU synchronization")
    if "batch_size=int(batch_size)" not in neighbor_forward_source:
        raise RuntimeError(
            "CUDA neighbor forward must pass host-derived batch_size to radius_graph"
        )
    if ".item(" in encoder_graph_source or ".cpu(" in encoder_graph_source:
        raise RuntimeError("CUDA graph forward contains an implicit scalar/CPU synchronization")
    if "prepare_batch" in encoder_forward_source:
        raise RuntimeError("encoder forward retains a redundant topology preparation pass")
    if "_assert_device_condition" not in neighbor_forward_source or "_assert_device_condition" not in encoder_graph_source:
        raise RuntimeError("CUDA graph checks are not using asynchronous device assertions")
    return {
        "cuda_neighbor_cpu_round_trip": False,
        "full_edge_isin": False,
        "full_edge_unique": False,
        "full_edge_sort": False,
        "auto_backend": False,
        "cuda_neighbor_scalar_extraction": False,
        "explicit_radius_batch_size": True,
        "cuda_graph_scalar_extraction": False,
        "redundant_forward_prepare": False,
        "device_assertions": "torch._assert_async",
    }


def phase_a(
    *,
    allow_legacy: bool,
    legacy_bond_mode: str | None,
) -> dict[str, Any]:
    runtime = require_cuda()
    config = load_yaml()
    acceptance = config["acceptance"]
    torch.manual_seed(20260825)
    np.random.seed(20260825)
    dataset, batch_cpu, _records = _real_batch()
    out_dir = ROOT / "outputs/engineering_v1/phase_a"
    out_dir.mkdir(parents=True, exist_ok=True)

    # The immutable reference manifest is an input to the gate, never an output.
    references, reference_manifest = _load_repaired_references()
    loader_report = _real_loader_audit(dataset)
    fixed, graph = _fixed_control_evidence(
        batch_cpu,
        references,
        acceptance,
        allow_legacy=allow_legacy,
        legacy_bond_mode=legacy_bond_mode,
    )
    legacy_audit = _legacy_cap_audit(graph)
    bf16 = _bf16_evidence(
        batch_cpu,
        references,
        acceptance,
        allow_legacy=allow_legacy,
        legacy_bond_mode=legacy_bond_mode,
    )
    se3 = _se3_regression(
        batch_cpu,
        acceptance,
        allow_legacy=allow_legacy,
        legacy_bond_mode=legacy_bond_mode,
    )
    synthetic = _synthetic_cuda_gate(acceptance)
    tiny_200 = _run_200_steps(out_dir, mode="topology")
    tiny_epochs = _run_five_epochs(out_dir, mode="topology")
    throughput = _throughput(
        batch_cpu,
        allow_legacy=allow_legacy,
        legacy_bond_mode=legacy_bond_mode,
    )
    for name, item in throughput.items():
        if item["baseline_multiplier"] < float(acceptance["throughput_multiplier"]):
            raise RuntimeError(
                f"{name} repaired BF16 throughput {item['baseline_multiplier']:.3f}x "
                f"is below {acceptance['throughput_multiplier']}x baseline"
            )
    source = _source_audit()
    result = {
        "schema_version": "pvb.codec.engineering.phase_a.v1",
        "status": "passed",
        "runtime": runtime,
        "config": str(ROOT / "config/codec_engineering_full.yaml"),
        "fixed_batch": {
            "shape": [int(value) for value in batch_cpu.x.shape],
            "sample_ids": list(batch_cpu.sample_id),
        },
        "source_audit": source,
        "real_loader": loader_report,
        "fp32_reference_manifest": {
            "path": str(REFERENCE_MANIFEST),
            "sha256": _sha256_file(REFERENCE_MANIFEST),
            "schema_version": reference_manifest["schema_version"],
            "manifest": reference_manifest,
        },
        "fixed_controls": fixed,
        "historical_cpu_cap_audit": legacy_audit,
        "bf16": bf16,
        "se3": se3,
        "synthetic_cap_unsaturated": synthetic,
        "optimizer_200_steps": tiny_200,
        "five_exact_tiny_epochs": tiny_epochs,
        "throughput": throughput,
        "acceptance_parameters": acceptance,
        "claims": {
            "cuda_production_path": True,
            "historical_cpu_edge_set_identity": False,
            "repaired_cuda_policy_reference": True,
            "quality_or_convergence_claim": False,
            "full_data_or_50_epoch_run": False,
        },
    }
    (out_dir / "acceptance.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("a",), default="a")
    parser.add_argument(
        "--allow-legacy-checkpoint",
        action="store_true",
        help="explicitly permit the known v1 control checkpoints",
    )
    parser.add_argument(
        "--legacy-bond-mode",
        choices=("topology", "distance_only"),
        default=None,
        help="required graph mode when --allow-legacy-checkpoint is used",
    )
    args = parser.parse_args()
    if args.legacy_bond_mode is not None and not args.allow_legacy_checkpoint:
        parser.error("--legacy-bond-mode requires --allow-legacy-checkpoint")
    result = phase_a(
        allow_legacy=args.allow_legacy_checkpoint,
        legacy_bond_mode=args.legacy_bond_mode,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "path": "outputs/engineering_v1/phase_a/acceptance.json",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
