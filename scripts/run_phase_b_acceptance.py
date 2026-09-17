#!/usr/bin/env python3
"""CUDA-hard-fail Phase-B distance-only acceptance."""
from __future__ import annotations
import gc, inspect, json, time
from dataclasses import replace
from typing import Any, Mapping
import torch
from torch import Tensor
from data.clip_dataset import ClipMMapDataset
from module.bond_sources import DistanceOnlyBondCache, build_canonical_reference_index, topology_id_from_record
from scripts.run_engineering_acceptance import (
    DEVICE, REAL_TRAIN, REAL_VALID, ROOT, _make_model, _real_batch,
    _run_200_steps, _run_five_epochs, require_cuda,
)
SEED = 20260825
PHASE_DIR = ROOT / "outputs/engineering_v1/phase_b"


def unique_codes(edge: Tensor, n: int) -> Tensor:
    edge = edge.detach().to("cpu", dtype=torch.long)
    if not edge.numel():
        return torch.empty(0, dtype=torch.long)
    return torch.unique(torch.minimum(edge[0], edge[1]) * n + torch.maximum(edge[0], edge[1]), sorted=True)


def decode(codes: Tensor, n: int) -> Tensor:
    codes = codes.to("cpu", dtype=torch.long)
    if not codes.numel():
        return torch.empty((2, 0), dtype=torch.long)
    return torch.stack((codes // n, codes % n))


def stats(values: Tensor) -> dict[str, Any]:
    values = values.detach().to("cpu", dtype=torch.float64).flatten()
    if not values.numel():
        return {"count": 0, "min": None, "max": None, "mean": None, "median": None}
    return {"count": int(values.numel()), "min": float(values.min()), "max": float(values.max()), "mean": float(values.mean()), "median": float(values.median())}


def graph_summary(codes: Tensor, coordinates: Tensor, n: int) -> dict[str, Any]:
    edges = decode(codes, n)
    distances = torch.linalg.vector_norm(coordinates[edges[0]] - coordinates[edges[1]], dim=-1) if edges.numel() else torch.empty(0)
    degree = torch.bincount(edges.flatten(), minlength=n) if edges.numel() else torch.zeros(n, dtype=torch.long)
    parent = list(range(n))
    def find(x: int) -> int:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != x:
            nxt = parent[x]
            parent[x] = root
            x = nxt
        return root
    for left, right in edges.t().tolist():
        left, right = find(int(left)), find(int(right))
        if left != right:
            parent[right] = left
    return {
        "edge_count_undirected": int(codes.numel()),
        "distance_angstrom": stats(distances),
        "degree": {**stats(degree), "isolated_atoms": int((degree == 0).sum()), "max_degree": int(degree.max())},
        "connected_components": len({find(i) for i in range(n)}),
    }


def intersection(left: Tensor, right: Tensor) -> int:
    if not left.numel() or not right.numel():
        return 0
    pos = torch.searchsorted(right, left)
    safe = pos.clamp(max=right.numel() - 1)
    return int(((pos < right.numel()) & (right[safe] == left)).sum())


def set_metrics(inferred: Tensor, supplied: Tensor) -> dict[str, Any]:
    tp = intersection(inferred, supplied)
    ni, ns = int(inferred.numel()), int(supplied.numel())
    union = ni + ns - tp
    precision = tp / ni if ni else (1.0 if ns == 0 else 0.0)
    recall = tp / ns if ns else (1.0 if ni == 0 else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"true_positive": tp, "false_positive": ni - tp, "false_negative": ns - tp, "precision": precision, "recall": recall, "f1": f1, "jaccard": tp / union if union else 1.0}


def canonical() -> tuple[Any, dict[str, dict[str, Any]], dict[str, Mapping[str, Any]]]:
    # Canonical references are a train-split artifact. Validation records are
    # deliberately not opened here, so they cannot influence selection.
    dataset = ClipMMapDataset(REAL_TRAIN)
    refs = build_canonical_reference_index(dataset, source_split="train")
    records: dict[str, Mapping[str, Any]] = {}
    for i in range(len(dataset)):
        record = dataset[i]
        tid = topology_id_from_record(record, require_stable=True)
        if str(record["sample_id"]) == str(refs[tid]["sample_id"]):
            if tid in records:
                raise RuntimeError(f"duplicate canonical sample for {tid!r}")
            records[tid] = record
    if set(refs) != set(records):
        raise RuntimeError("canonical reference index is incomplete")
    for tid, ref in refs.items():
        if not torch.equal(torch.as_tensor(records[tid]["x"][0], dtype=torch.float32), ref["coordinates"]):
            raise RuntimeError(f"canonical coordinates changed for {tid!r}")
    if len(refs) != 3:
        raise RuntimeError(f"selected ATLAS acceptance dataset must have three topologies, got {len(refs)}")
    return dataset, refs, records


def diagnostics(refs: Mapping[str, Mapping[str, Any]], records: Mapping[str, Mapping[str, Any]]) -> tuple[dict[str, Any], Any]:
    model = _make_model({"temporal_layers": 1, "temporal_ratio": 1}, bond_mode="distance_only").to(DEVICE)
    model.eval()
    model.prepare_distance_bonds(refs, device=DEVICE)
    cache = model.frame_encoder.distance_bond_cache
    if cache is None:
        raise RuntimeError("distance-only cache was not initialized")
    result: dict[str, Any] = {}
    for tid in sorted(refs):
        coordinates = refs[tid]["coordinates"]
        n = int(coordinates.shape[0])
        inferred = cache.materialize(
            tid,
            device=DEVICE,
            atom_count=n,
            atom_identity_sha256=refs[tid].get("atom_identity_sha256"),
        )
        supplied = torch.as_tensor(records[tid].get("bond_index", []), dtype=torch.long)
        inferred_codes, supplied_codes = unique_codes(inferred, n), unique_codes(supplied, n)
        result[tid] = {
            "sample_id": str(records[tid]["sample_id"]), "atom_count": n,
            "cutoff_interval_angstrom": "(0.5, 2.2]",
            "set_metrics": set_metrics(inferred_codes, supplied_codes),
            "inferred": graph_summary(inferred_codes, coordinates, n),
            "supplied": graph_summary(supplied_codes, coordinates, n),
        }
    return {"topology_count": len(result), "references": result, "cache": cache.stats()}, model


def boundary() -> dict[str, Any]:
    cache = DistanceOnlyBondCache(max_num_neighbors=16, capacity=4)
    coordinates = torch.tensor([[0., 0., 0.], [.5, 0., 0.], [2.2, 0., 0.], [4., 0., 0.]], dtype=torch.float32)
    cache.register_reference("phase_b_boundary", coordinates, device=DEVICE)
    codes = unique_codes(cache.materialize("phase_b_boundary", device=DEVICE, atom_count=4), 4)
    if torch.any(codes == 1):
        raise RuntimeError("d=0.5 was included; lower boundary is not strict")
    if not torch.any(codes == 2):
        raise RuntimeError("d=2.2 was excluded; upper boundary is not closed")
    return {"lower_boundary_d_0.5_included": False, "upper_boundary_d_2.2_included": True, "cache": cache.stats()}


def input_invariance(refs: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    _, batch_cpu, _ = _real_batch()
    model = _make_model({"temporal_layers": 1, "temporal_ratio": 1}, bond_mode="distance_only").to(DEVICE)
    model.eval()
    model.prepare_distance_bonds(refs, device=DEVICE)
    batch = batch_cpu.to(DEVICE)
    with torch.no_grad():
        original = model.frame_encoder.build_graph(batch)
        changed = model.frame_encoder.build_graph(replace(
            batch, atype=torch.zeros_like(batch.atype), btype=torch.zeros_like(batch.btype),
            block_id=torch.zeros_like(batch.block_id), bond_index=torch.empty((2, 0), dtype=torch.long, device=DEVICE),
        ))
    fields = ("distance_edge_index", "distance_edge_weight", "bond_index", "edge_index", "edge_weight", "bond_type")
    for field in fields:
        if not torch.equal(getattr(original, field), getattr(changed, field)):
            raise RuntimeError(f"distance-only graph changed after supplied-input mutation: {field}")
    result = {"passed": True, "mutated_inputs": ["atype", "btype", "block_id", "bond_index"], "compared_graph_fields": list(fields), "bond_construction_mode": original.bond_construction_mode, "backend_used": original.backend_used, "edge_count": int(original.edge_index.shape[1])}
    del model, original, changed, batch, batch_cpu
    torch.cuda.empty_cache()
    gc.collect()
    return result


def common_initialization() -> dict[str, Any]:
    torch.manual_seed(SEED)
    topology = _make_model({"temporal_layers": 1, "temporal_ratio": 1}, bond_mode="topology")
    top_state = {k: v.detach().cpu().clone() for k, v in topology.state_dict().items()}
    torch.manual_seed(SEED)
    distance = _make_model({"temporal_layers": 1, "temporal_ratio": 1}, bond_mode="distance_only")
    dist_state = distance.state_dict()
    if top_state.keys() != dist_state.keys():
        raise RuntimeError("topology and distance-only state keys differ")
    mismatch = [k for k, v in top_state.items() if not torch.equal(v, dist_state[k].detach().cpu())]
    if mismatch:
        raise RuntimeError("common initialization differs: " + ", ".join(mismatch[:5]))
    return {"passed": True, "seed": SEED, "state_keys": len(top_state), "mismatched_state_keys": [], "optimizer": "AdamW", "precision": "fp32"}


def short_training() -> dict[str, Any]:
    PHASE_DIR.mkdir(parents=True, exist_ok=True)
    steps, epochs = {}, {}
    for mode in ("topology", "distance_only"):
        torch.manual_seed(SEED)
        steps[mode] = _run_200_steps(PHASE_DIR, mode=mode)
        torch.manual_seed(SEED)
        epochs[mode] = _run_five_epochs(PHASE_DIR, mode=mode)
    return {
        "common_contract": {"data": "identical deterministic eight-clip tiny records", "seed": SEED, "optimizer": "AdamW", "learning_rate": 0.001, "warmup_steps": 20, "precision": "fp32", "device": str(DEVICE)},
        "two_hundred_steps": steps, "five_exact_epochs": epochs,
    }


def source_audit() -> dict[str, Any]:
    source = inspect.getsource(DistanceOnlyBondCache.register_reference)
    forbidden = ("atype", "btype", "bond_index", "atom_source_index")
    found = [name for name in forbidden if name in source]
    if found:
        raise RuntimeError("forbidden distance-only input names found: " + ", ".join(found))
    parameters = tuple(inspect.signature(DistanceOnlyBondCache.register_reference).parameters)
    expected = (
        "self",
        "topology_id",
        "coordinates",
        "device",
        "atom_identity_sha256",
        "sample_id",
        "source_split",
    )
    if parameters != expected:
        raise RuntimeError(f"distance-only API changed: {parameters!r}")
    return {"passed": True, "forbidden_input_names": list(forbidden), "forbidden_input_names_found": found, "reference_api_parameters": list(parameters), "canonical_selector": "topology_id"}


def main() -> int:
    runtime = require_cuda()
    started = time.perf_counter()
    PHASE_DIR.mkdir(parents=True, exist_ok=True)
    dataset, refs, records = canonical()
    distance, diagnostic_model = diagnostics(refs, records)
    result = {
        "phase": "B", "status": "passed", "runtime": runtime,
        "elapsed_s": None,
        "canonical_reference_policy": {"dataset_splits": ["train"], "selection": "lexicographically earliest sample_id per stable topology_id", "topology_ids": sorted(refs), "count": len(refs), "validation_coordinates_used": False},
        "bond_construction": {"mode": "distance_only", "uses_canonical_coordinates_only": True, "cutoff_interval_angstrom": "(0.5, 2.2]", "cuda_only": True, "supplied_atom_topology_bond_inputs_used_for_inference": False},
        "distance_diagnostics": distance,
        "boundary_checks": boundary(),
        "graph_input_invariance": input_invariance(refs),
        "common_initialization_and_training_contract": common_initialization(),
        "short_training_acceptance": short_training(),
        "source_audit": source_audit(),
        "claims": {"fixed_graph_and_short_run_acceptance": True, "historical_cpu_output_identity_claimed": False, "convergence_or_quality_claimed": False, "full_dataset_or_long_run_claimed": False},
    }
    torch.cuda.synchronize()
    result["elapsed_s"] = time.perf_counter() - started
    (PHASE_DIR / "acceptance.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    del dataset, diagnostic_model
    torch.cuda.empty_cache()
    gc.collect()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
