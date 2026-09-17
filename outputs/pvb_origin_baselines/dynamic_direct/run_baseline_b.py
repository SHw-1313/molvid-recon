#!/usr/bin/env python3
"""Run Baseline B with the unmodified PVB dyVAE checkpoint.

This runner intentionally lives outside PVB_origin.  It reads the pilot's
npz-v1 clip stores without importing the modified clip implementation, builds
the legacy batch mapping expected by PVB_origin, and calls only
``dyVAE.inference``.  No source or data files are written.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import mmap
import os
import random
import time
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch


SEED = 20260810
MAX_TOKENS = 80_000
MAX_BATCHES = 32
ROLLOUT_FRAMES = 16
SDE_STEPS = 10
SCHEMA = "pvb.baseline.dynamic_direct.v1"


class ClipStore:
    """Read the pilot npz-v1 mmap store without changing PVB source."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.index: list[dict[str, Any]] = []
        with (root / "index.txt").open(encoding="utf-8") as handle:
            for local_index, line in enumerate(handle):
                fields = line.rstrip("\n").split("\t")
                if len(fields) < 6:
                    raise ValueError(f"malformed clip index line in {root}: {line!r}")
                sample_id, start, end, atoms, frames, bucket = fields[:6]
                self.index.append(
                    {
                        "local_index": local_index,
                        "sample_id": sample_id,
                        "start": int(start),
                        "end": int(end),
                        "atoms": int(atoms),
                        "frames": int(frames),
                        "time_bucket_id": bucket,
                    }
                )
        self._data_file = (root / "data.bin").open("rb")
        self._mmap = mmap.mmap(self._data_file.fileno(), 0, access=mmap.ACCESS_READ)
        self.by_id = {entry["sample_id"]: entry for entry in self.index}

    def __len__(self) -> int:
        return len(self.index)

    def read(self, local_index: int) -> dict[str, Any]:
        entry = self.index[local_index]
        payload = self._mmap[entry["start"] : entry["end"]]
        if payload[:2] != b"PK":
            raise ValueError(f"unsupported payload in {self.root}: expected npz-v1")
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"].item()))
            record = dict(metadata)
            for key in (
                "x",
                "bpos",
                "atype",
                "btype",
                "edge_mask",
                "loss_mask",
                "bond_index",
                "time_ps",
                "delta_time_ps",
            ):
                record[key] = np.array(archive[key], copy=True)
        record["sample_id"] = record.get("sample_id", entry["sample_id"])
        return record

    def close(self) -> None:
        if getattr(self, "_mmap", None) is not None:
            self._mmap.close()
            self._data_file.close()
            self._mmap = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def select_pilot_batches(
    stores: Sequence[ClipStore],
    *,
    seed: int,
    max_tokens: int,
    max_batches: int,
) -> tuple[list[list[dict[str, Any]]], list[dict[str, Any]]]:
    """Reproduce TaskAwareClipBatchSampler with shuffle=False/replacement=False."""

    # ConcatDataset order in the pilot is ATLAS followed by MISATO.
    entries: list[dict[str, Any]] = []
    for store_number, store in enumerate(stores):
        for entry in store.index:
            item = dict(entry)
            item["store_number"] = store_number
            item["effective_tokens"] = item["atoms"] * item["frames"]
            entries.append(item)

    groups: "OrderedDict[tuple[int, int, str], list[int]]" = OrderedDict()
    for position, entry in enumerate(entries):
        if entry["effective_tokens"] > max_tokens:
            continue
        key = (1, entry["frames"], entry["time_bucket_id"])
        groups.setdefault(key, []).append(position)
    if not groups:
        raise ValueError("no clip fits max_tokens")

    rng = np.random.default_rng(seed)
    orders = {key: np.asarray(positions, dtype=np.int64) for key, positions in groups.items()}
    cursors = {key: 0 for key in groups}
    available = list(groups)
    result: list[list[dict[str, Any]]] = []

    while available and len(result) < max_batches:
        weights = np.ones(len(available), dtype=np.float64)
        weights /= weights.sum()
        group_index = int(rng.choice(len(available), p=weights))
        key = available[group_index]
        batch_positions: list[int] = []
        seen: set[int] = set()
        cost = 0
        while True:
            cursor = cursors[key]
            order = orders[key]
            if cursor >= order.size:
                break
            position = int(order[cursor])
            cursors[key] = cursor + 1
            if position in seen:
                break
            item_cost = int(entries[position]["effective_tokens"])
            if cost + item_cost > max_tokens:
                cursors[key] -= 1
                break
            batch_positions.append(position)
            seen.add(position)
            cost += item_cost
            if cursors[key] >= order.size:
                break
        if not batch_positions:
            available.pop(group_index)
            continue
        result.append([entries[position] for position in batch_positions])
        if cursors[key] >= orders[key].size:
            available.pop(group_index)

    selected = [entry for batch in result for entry in batch]
    return result, selected


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_legacy_batch(record: Mapping[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    atype = np.asarray(record["atype"], dtype=np.int64)
    n_atoms = int(atype.shape[0])
    loss_mask = np.asarray(record.get("loss_mask", np.ones(n_atoms, dtype=np.bool_)), dtype=np.bool_)
    edge_mask = np.asarray(record.get("edge_mask", np.zeros(n_atoms, dtype=np.int64)), dtype=np.int64)
    return {
        "atype": torch.as_tensor(atype, dtype=torch.long, device=device),
        "btype": torch.as_tensor(np.asarray(record["btype"], dtype=np.int64), dtype=torch.long, device=device),
        "x0": torch.as_tensor(np.asarray(record["x"], dtype=np.float32)[0], dtype=torch.float32, device=device),
        "b0": torch.as_tensor(np.asarray(record["bpos"], dtype=np.float32)[0], dtype=torch.float32, device=device),
        "abid": torch.zeros(n_atoms, dtype=torch.long, device=device),
        "mask": torch.as_tensor(loss_mask, dtype=torch.bool, device=device),
        "edge_mask": torch.as_tensor(edge_mask, dtype=torch.long, device=device),
        "bond_index": torch.as_tensor(np.asarray(record["bond_index"], dtype=np.int64), dtype=torch.long, device=device),
    }


def _valid_mask(record: Mapping[str, Any], device: torch.device) -> torch.Tensor:
    mask = np.asarray(record.get("loss_mask"), dtype=np.bool_)
    return torch.as_tensor(mask, dtype=torch.bool, device=device)


def _frame_rmsd(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, frames: Sequence[int]) -> float:
    values = []
    for frame in frames:
        valid = mask
        if torch.any(valid):
            values.append((pred[frame, valid] - target[frame, valid]).square().sum(dim=-1).mean())
    return math.sqrt(max(float(torch.stack(values).mean().detach().cpu()) if values else 0.0, 0.0))


def _frame_drmsd(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, frames: Sequence[int]) -> float:
    values = []
    for frame in frames:
        valid_indices = torch.nonzero(mask, as_tuple=False).flatten()
        if valid_indices.numel() < 2:
            continue
        pred_dist = torch.pdist(pred[frame].index_select(0, valid_indices))
        target_dist = torch.pdist(target[frame].index_select(0, valid_indices))
        values.append((pred_dist - target_dist).square().mean())
    return math.sqrt(max(float(torch.stack(values).mean().detach().cpu()) if values else 0.0, 0.0))


def _pairs(record: Mapping[str, Any], device: torch.device) -> tuple[torch.Tensor, torch.Tensor] | None:
    value = torch.as_tensor(record["bond_index"], dtype=torch.long, device=device)
    if value.numel() == 0:
        return None
    return value[0], value[1]


def _bond_rmse(pred: torch.Tensor, target: torch.Tensor, record: Mapping[str, Any], mask: torch.Tensor, frames: Sequence[int]) -> float:
    pair = _pairs(record, pred.device)
    if pair is None:
        return 0.0
    source, destination = pair
    values = []
    for frame in frames:
        pair_mask = mask[source] & mask[destination]
        if torch.any(pair_mask):
            pred_dist = torch.linalg.vector_norm(pred[frame, source] - pred[frame, destination], dim=-1)
            target_dist = torch.linalg.vector_norm(target[frame, source] - target[frame, destination], dim=-1)
            values.append((pred_dist[pair_mask] - target_dist[pair_mask]).square())
    return math.sqrt(max(float(torch.cat(values).mean().detach().cpu()) if values else 0.0, 0.0))


def _nonbond_pairs(record: Mapping[str, Any], atoms: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    nodes = torch.arange(atoms, device=device)
    local = torch.triu_indices(atoms, atoms, offset=1, device=device)
    source, destination = nodes[local[0]], nodes[local[1]]
    pair = _pairs(record, device)
    if pair is not None:
        bond_codes = torch.minimum(pair[0], pair[1]) * atoms + torch.maximum(pair[0], pair[1])
        codes = source * atoms + destination
        keep = ~torch.isin(codes, bond_codes)
        source, destination = source[keep], destination[keep]
    return source, destination


def _candidate_nonbond_pairs(
    record: Mapping[str, Any], positions: Sequence[torch.Tensor], radius: float
) -> tuple[torch.Tensor, torch.Tensor]:
    atoms = int(positions[0].shape[0])
    try:
        from torch_cluster import radius_graph
    except (ImportError, OSError):
        radius_graph = None
    if atoms <= 2048 or radius_graph is None:
        return _nonbond_pairs(record, atoms, positions[0].device)
    batch = torch.zeros(atoms, dtype=torch.long, device=positions[0].device)
    codes = []
    for position in positions:
        edges = radius_graph(position, r=float(radius), batch=batch, loop=False, max_num_neighbors=128)
        if edges.numel():
            low = torch.minimum(edges[0], edges[1])
            high = torch.maximum(edges[0], edges[1])
            codes.append(torch.unique(low * atoms + high))
    if not codes:
        empty = torch.empty(0, dtype=torch.long, device=positions[0].device)
        return empty, empty
    codes_tensor = torch.unique(torch.cat(codes))
    source = torch.div(codes_tensor, atoms, rounding_mode="floor")
    destination = codes_tensor.remainder(atoms)
    pair = _pairs(record, positions[0].device)
    if pair is not None:
        bond_codes = torch.minimum(pair[0], pair[1]) * atoms + torch.maximum(pair[0], pair[1])
        keep = ~torch.isin(codes_tensor, bond_codes)
        source, destination = source[keep], destination[keep]
    return source, destination


def _nonbond_count(record: Mapping[str, Any], mask: torch.Tensor) -> int:
    count = int(mask.sum().item())
    total = count * (count - 1) // 2
    pair = _pairs(record, mask.device)
    if pair is not None:
        total -= int((mask[pair[0]] & mask[pair[1]]).sum().item())
    return max(total, 1)


def _contact_error(pred: torch.Tensor, target: torch.Tensor, record: Mapping[str, Any], mask: torch.Tensor, frames: Sequence[int], cutoff: float = 4.5) -> float:
    values = []
    total_pairs = _nonbond_count(record, mask)
    for frame in frames:
        source, destination = _candidate_nonbond_pairs(record, (pred[frame], target[frame]), cutoff)
        pair_mask = mask[source] & mask[destination]
        if torch.any(pair_mask):
            pred_dist = torch.linalg.vector_norm(pred[frame, source] - pred[frame, destination], dim=-1)
            target_dist = torch.linalg.vector_norm(target[frame, source] - target[frame, destination], dim=-1)
            pred_contacts = (pred_dist[pair_mask] < cutoff).sum()
            target_contacts = (target_dist[pair_mask] < cutoff).sum()
            values.append((pred_contacts - target_contacts).abs().float() / float(total_pairs))
        else:
            values.append(torch.zeros((), device=pred.device))
    return float(torch.stack(values).mean().detach().cpu()) if values else 0.0


def _clash_rate(pred: torch.Tensor, record: Mapping[str, Any], mask: torch.Tensor, frames: Sequence[int], cutoff: float = 0.7) -> float:
    values = []
    total_pairs = _nonbond_count(record, mask)
    for frame in frames:
        source, destination = _candidate_nonbond_pairs(record, (pred[frame],), cutoff)
        pair_mask = mask[source] & mask[destination]
        if torch.any(pair_mask):
            distance = torch.linalg.vector_norm(pred[frame, source] - pred[frame, destination], dim=-1)
            values.append((distance[pair_mask] < cutoff).sum().float() / float(total_pairs))
        else:
            values.append(torch.zeros((), device=pred.device))
    return float(torch.stack(values).mean().detach().cpu()) if values else 0.0


def _motion_rmse(pred: torch.Tensor, target: torch.Tensor, record: Mapping[str, Any], acceleration: bool) -> float | None:
    if pred.shape[0] < (3 if acceleration else 2):
        return None
    delta = torch.as_tensor(np.asarray(record["delta_time_ps"], dtype=np.float32), device=pred.device)
    dt = delta.reshape(-1, 1, 1)
    pred_velocity = (pred[1:] - pred[:-1]) / dt
    target_velocity = (target[1:] - target[:-1]) / dt
    mask = _valid_mask(record, pred.device).reshape(1, -1, 1)
    if acceleration:
        pred_value = 2.0 * ((pred_velocity[1:] - pred_velocity[:-1]) / (dt[:-1] + dt[1:]))
        target_value = 2.0 * ((target_velocity[1:] - target_velocity[:-1]) / (dt[:-1] + dt[1:]))
        mask = mask.expand(pred_value.shape[0], -1, 1)
    else:
        pred_value = pred_velocity
        target_value = target_velocity
        mask = mask.expand(pred_value.shape[0], -1, 1)
    value = (pred_value - target_value).square().mean(dim=-1)
    return math.sqrt(max(float(value[mask.squeeze(-1)].mean().detach().cpu()), 0.0)) if torch.any(mask) else 0.0


def _frequency_retention(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    if pred.shape[0] < 3:
        return 1.0 if torch.allclose(pred, target) else 0.0
    if not torch.any(mask):
        return 0.0
    pred_signal = pred[:, mask] - pred[:, mask].mean(0, keepdim=True)
    target_signal = target[:, mask] - target[:, mask].mean(0, keepdim=True)
    pred_power = torch.fft.rfft(pred_signal, dim=0).abs().square()[1:].sum()
    target_power = torch.fft.rfft(target_signal, dim=0).abs().square()[1:].sum()
    if target_power <= 1e-12:
        return 1.0 if pred_power <= 1e-12 else 0.0
    return float((pred_power / target_power).detach().cpu())


def metrics_for(
    pred: torch.Tensor,
    target: torch.Tensor,
    record: Mapping[str, Any],
) -> dict[str, Any]:
    mask = _valid_mask(record, pred.device)
    future = list(range(1, pred.shape[0]))
    all_frames = list(range(pred.shape[0]))

    def section(frames: Sequence[int]) -> dict[str, float]:
        return {
            "rmsd": _frame_rmsd(pred, target, mask, frames),
            "drmsd": _frame_drmsd(pred, target, mask, frames),
            "bond_rmse": _bond_rmse(pred, target, record, mask, frames),
            "contact_error": _contact_error(pred, target, record, mask, frames),
            "clash_rate": _clash_rate(pred, record, mask, frames),
        }

    return {
        "frame0": section([0]),
        "future": section(future),
        "all_frames": section(all_frames),
        "velocity_rmse": _motion_rmse(pred, target, record, acceleration=False),
        "acceleration_rmse": _motion_rmse(pred, target, record, acceleration=True),
        "frequency_retention": _frequency_retention(pred, target, mask),
    }


def one_step_metrics(pred: torch.Tensor, target: torch.Tensor, record: Mapping[str, Any]) -> dict[str, float]:
    mask = _valid_mask(record, pred.device)
    return {
        "rmsd": _frame_rmsd(pred, target, mask, [0]),
        "drmsd": _frame_drmsd(pred, target, mask, [0]),
        "bond_rmse": _bond_rmse(pred, target, record, mask, [0]),
        "contact_error": _contact_error(pred, target, record, mask, [0]),
        "clash_rate": _clash_rate(pred, record, mask, [0]),
    }


def mean_dict(records: Sequence[Mapping[str, Any]], path: Sequence[str] = ()) -> dict[str, Any]:
    if not records:
        return {}
    first = records[0]
    result: dict[str, Any] = {}
    for key, value in first.items():
        values = [record[key] for record in records if record.get(key) is not None]
        if isinstance(value, Mapping):
            result[key] = mean_dict(values, ())
        elif values:
            result[key] = float(sum(float(item) for item in values) / len(values))
        else:
            result[key] = None
    return result


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = [
        "sample_id",
        "source",
        "time_bucket_id",
        "atoms",
        "native_delta_time_ps",
        "rollout_frame0_rmsd",
        "rollout_future_rmsd",
        "rollout_future_drmsd",
        "rollout_future_bond_rmse",
        "rollout_future_contact_error",
        "rollout_future_clash_rate",
        "rollout_velocity_rmse",
        "rollout_acceleration_rmse",
        "rollout_frequency_retention",
        "one_step_rmsd",
        "one_step_drmsd",
        "one_step_bond_rmse",
        "one_step_contact_error",
        "one_step_clash_rate",
        "elapsed_s",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--atlas-valid", type=Path, required=True)
    parser.add_argument("--misato-valid", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    parser.add_argument("--max-batches", type=int, default=MAX_BATCHES)
    parser.add_argument("--rollout-frames", type=int, default=ROLLOUT_FRAMES)
    parser.add_argument("--sde-steps", type=int, default=SDE_STEPS)
    parser.add_argument("--atlas-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.rollout_frames < 2:
        raise ValueError("rollout-frames must be at least 2")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    stores = [ClipStore(args.atlas_valid)] if args.atlas_only else [ClipStore(args.atlas_valid), ClipStore(args.misato_valid)]
    batches, selected = select_pilot_batches(
        stores,
        seed=args.seed,
        max_tokens=args.max_tokens,
        max_batches=args.max_batches,
    )
    selection = {
        "schema_version": "pvb.baseline.dynamic_direct.selection.v1",
        "seed": args.seed,
        "store_order": [str(args.atlas_valid), str(args.misato_valid)],
        "policy": {
            "max_tokens": args.max_tokens,
            "max_batches": args.max_batches,
            "shuffle": False,
            "replacement": False,
            "task_weights": {},
            "time_bucket_weights": {},
            "batching_cost": "frames * atoms",
        },
        "batch_count": len(batches),
        "sample_count": len(selected),
        "sample_count_by_bucket": {
            bucket: sum(1 for entry in selected if entry["time_bucket_id"] == bucket)
            for bucket in sorted({entry["time_bucket_id"] for entry in selected})
        },
        "batches": [
            {
                "batch_index": index,
                "time_bucket_id": batch[0]["time_bucket_id"],
                "sample_count": len(batch),
                "effective_tokens": sum(item["effective_tokens"] for item in batch),
                "sample_ids": [item["sample_id"] for item in batch],
            }
            for index, batch in enumerate(batches)
        ],
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
    print(json.dumps({"selection": {"batches": len(batches), "samples": len(selected), "by_bucket": selection["sample_count_by_bucket"]}}, sort_keys=True))

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    print(json.dumps({"device": str(device), "torch": torch.__version__, "cuda": torch.version.cuda}, sort_keys=True))
    model = torch.load(args.checkpoint, map_location=device, weights_only=False).to(device).eval()
    print(json.dumps({
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": __import__("hashlib").sha256(args.checkpoint.read_bytes()).hexdigest(),
        "model_class": f"{type(model).__module__}.{type(model).__name__}",
        "model_using_ode": bool(getattr(model, "using_ode", False)),
        "model_k_neighbors": int(getattr(model, "k_neighbors", -1)),
    }, sort_keys=True))

    rows: list[dict[str, Any]] = []
    per_sample: list[dict[str, Any]] = []
    started = time.perf_counter()
    store_by_number = {index: store for index, store in enumerate(stores)}
    try:
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
            one_step = one_step_metrics(prediction[1:2], target[1:2], {**record, "delta_time_ps": np.asarray(record["delta_time_ps"])[0:1]})
            row = {
                "sample_id": entry["sample_id"],
                "source": record.get("source"),
                "time_bucket_id": record.get("time_bucket_id", entry["time_bucket_id"]),
                "atoms": int(target.shape[1]),
                "native_delta_time_ps": float(np.asarray(record["delta_time_ps"])[0]),
                "rollout_frame0_rmsd": rollout["frame0"]["rmsd"],
                "rollout_future_rmsd": rollout["future"]["rmsd"],
                "rollout_future_drmsd": rollout["future"]["drmsd"],
                "rollout_future_bond_rmse": rollout["future"]["bond_rmse"],
                "rollout_future_contact_error": rollout["future"]["contact_error"],
                "rollout_future_clash_rate": rollout["future"]["clash_rate"],
                "rollout_velocity_rmse": rollout["velocity_rmse"],
                "rollout_acceleration_rmse": rollout["acceleration_rmse"],
                "rollout_frequency_retention": rollout["frequency_retention"],
                "one_step_rmsd": one_step["rmsd"],
                "one_step_drmsd": one_step["drmsd"],
                "one_step_bond_rmse": one_step["bond_rmse"],
                "one_step_contact_error": one_step["contact_error"],
                "one_step_clash_rate": one_step["clash_rate"],
                "elapsed_s": elapsed,
            }
            rows.append(row)
            per_sample.append({"sample_id": entry["sample_id"], "bucket": row["time_bucket_id"], "rollout": rollout, "one_step": one_step, "elapsed_s": elapsed})
            print(json.dumps({"sample": sample_number, "total": len(selected), "sample_id": entry["sample_id"], "bucket": row["time_bucket_id"], "atoms": row["atoms"], "elapsed_s": elapsed, "one_step_rmsd": row["one_step_rmsd"], "rollout_future_rmsd": row["rollout_future_rmsd"]}, sort_keys=True), flush=True)
    finally:
        for store in stores:
            store.close()

    by_bucket: dict[str, dict[str, Any]] = {}
    for bucket in sorted({row["time_bucket_id"] for row in rows}):
        bucket_rows = [row for row in rows if row["time_bucket_id"] == bucket]
        bucket_samples = [item for item in per_sample if item["bucket"] == bucket]
        by_bucket[bucket] = {
            "sample_count": len(bucket_rows),
            "atoms_mean": float(sum(row["atoms"] for row in bucket_rows) / len(bucket_rows)),
            "native_delta_time_ps": float(sum(row["native_delta_time_ps"] for row in bucket_rows) / len(bucket_rows)),
            "one_step": mean_dict([item["one_step"] for item in bucket_samples]),
            "rollout16": mean_dict([item["rollout"] for item in bucket_samples]),
            "runtime": {
                "wall_time_s": float(sum(row["elapsed_s"] for row in bucket_rows)),
                "samples_per_s": float(len(bucket_rows) / sum(row["elapsed_s"] for row in bucket_rows)),
            },
        }

    peak_gpu = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    report = {
        "schema_version": SCHEMA,
        "baseline": "B",
        "description": "Unmodified PVB dyVAE dynamic-structure checkpoint, direct native-step inference; no retraining.",
        "seed": args.seed,
        "source_commit": os.popen("git -C /workspace/PVB_origin rev-parse HEAD").read().strip(),
        "source_status": os.popen("git -C /workspace/PVB_origin status --short --branch").read().strip().splitlines(),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": __import__("hashlib").sha256(args.checkpoint.read_bytes()).hexdigest(),
        "inference": {
            "method": "dyVAE.inference",
            "sde_steps": args.sde_steps,
            "direct_one_step_target": "frame 1 at each store's native delta_time_ps",
            "rollout_frames": args.rollout_frames,
            "rollout_transition_count": args.rollout_frames - 1,
            "rollout_policy": "frame 0 is the input; each later frame is a sequential direct model.inference call",
        },
        "data": {
            "atlas_valid": str(args.atlas_valid),
            "misato_valid": None if args.atlas_only else str(args.misato_valid),
            "atlas_only": args.atlas_only,
            "pdb_included": False,
            "selection_file": str(args.output_dir / "selection.json"),
            "policy": selection["policy"],
            "batch_count": len(batches),
            "sample_count": len(selected),
            "sample_count_by_bucket": selection["sample_count_by_bucket"],
        },
        "runtime": {
            "device": str(device),
            "total_wall_time_s": float(time.perf_counter() - started),
            "peak_gpu_bytes": peak_gpu,
        },
        "by_time_bucket": by_bucket,
        "per_sample": per_sample,
        "blockers": [
            "Preferred validation stores are npz-v1 clip records and cannot be read by PVB_origin's gzip-JSON MMAPDataset; this run used a read-only adapter in the output directory.",
            "No ATLAS or MISATO original-format migrated store was found under /data5/PVB; the only original-format candidate found there is a PDB EPT/PVB store and was excluded by scope.",
            "The original PVB evaluator's raw test JSONL/trajectory path was not used because the pilot validation stores are clip stores; metrics are computed on the exact selected clips with per-clip inference.",
        ],
    }
    (args.output_dir / "baseline_b_metrics.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(args.output_dir / "baseline_b_per_sample.csv", rows)
    print(json.dumps({"complete": True, "output": str(args.output_dir / "baseline_b_metrics.json"), "samples": len(rows), "by_bucket": selection["sample_count_by_bucket"], "peak_gpu_bytes": peak_gpu}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
