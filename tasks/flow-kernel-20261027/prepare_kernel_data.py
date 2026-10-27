"""Prepare one paired 256-visit TRAIN Q1 plan; CPU only, bounded raw extension.

Existing ordinary16 finest-clock clips are reused without copying. Only a
17th frame, when required, is read from the same original trajectory. The
entire 16-frame prefix/topology is checked before an extended record is saved.
"""
from __future__ import annotations

import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
import sys
import time
import warnings

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from molvid.data.bridge_mixture import BridgeMixture
from molvid.data.io import read_topology
from molvid.data.kernel_windows import (ATOM_FIELDS, KernelWindows, make_plan,
                                       native_indices, plan_digest, select_kernel_record)
from molvid.data.preprocess import make_clip_record, topology_metadata


CONTACT_FILTER = Path("/workspace/molvid-misato-contact-filter-v1/molvid/data/contact_filter.py")


def source_paths(record, atlas_root, misato_root):
    name = record["system_id"]
    if record["source"] == "atlas":
        return atlas_root / name / f"{name}.pdb", atlas_root / name / f"{name}_prod_{record['replica']}_fit.xtc"
    return misato_root / "parameter_restart_files_MD" / name.lower() / f"{name}.pdb", misato_root / "MD.hdf5"


def extend_or_verify_raw(original, count, atlas_root, misato_root):
    """Bounded native read plus exact topology and aligned-prefix verification."""
    import h5py
    import mdtraj as md

    pdb, trajectory = source_paths(original, atlas_root, misato_root)
    old_indices = native_indices(original)
    stride = 10 if original["source"] == "atlas" else 1
    native_dt = 10. if original["source"] == "atlas" else 80.
    indices = old_indices[0] + np.arange(count, dtype=np.int64) * stride
    if original["source"] == "atlas":
        with md.open(str(trajectory)) as handle:
            handle.seek(int(indices[0]))
            xyz_nm, times, _, _ = handle.read(n_frames=count, stride=stride)
        frames = np.asarray(xyz_nm, dtype=np.float32) * 10.
        times = np.asarray(times, dtype=np.float64)
        if len(frames) != count or not np.allclose(times, indices * native_dt, rtol=0, atol=1e-3):
            raise ValueError("ATLAS bounded read missing frames or absolute XTC clock mismatch")
        raw_numbers = None
    else:
        with h5py.File(trajectory, "r") as handle:
            group = handle[original["system_id"]]
            dataset = group["trajectory_coordinates"]
            if indices[-1] >= len(dataset):
                raise ValueError(f"MISATO native frame {indices[-1]} exceeds {len(dataset)} frames")
            frames = np.asarray(dataset[int(indices[0]):int(indices[-1]) + 1], dtype=np.float32)
            raw_numbers = np.asarray(group["atoms_number"])
        times = indices.astype(np.float64) * native_dt
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*CRYST1.*")
        topology = read_topology(pdb)
    if frames.shape != (count, topology.n_atoms, 3) or not np.isfinite(frames).all():
        raise ValueError("raw coordinate atom count or finiteness differs")
    if raw_numbers is not None and not np.array_equal(raw_numbers, [a.element.atomic_number for a in topology.atoms]):
        raise ValueError("MISATO raw ordered elements disagree with original PDB")
    metadata = topology_metadata(topology, complex_topology=original["source"] == "misato")
    for field in ATOM_FIELDS:
        if not np.array_equal(np.asarray(original[field]), np.asarray(metadata[field])):
            raise ValueError(f"raw/cache topology differs: {field}")
    if metadata["topology_fingerprint"] != original["topology_fingerprint"]:
        raise ValueError("raw/cache topology fingerprint differs")
    result = make_clip_record(frames, times, metadata, sample_id=original["sample_id"],
        source=original["source"], system_id=original["system_id"], replica=original["replica"],
        split="train", source_stride=stride, native_dt_ps=native_dt,
        timestamp_provenance="bounded native read; unchanged ordinary window clock")
    errors = {key: float(np.max(np.abs(result[key][:16] - original[key]))) for key in ("x", "bpos")}
    if max(errors.values()) > 1e-4 or not np.array_equal(result["time_ps"][:16], original["time_ps"]):
        raise ValueError(f"raw/cache16 prefix mismatch: {errors}")
    # Use bit-identical original prefix, in the verified same reference frame.
    for key in ("x", "bpos"):
        result[key][:16] = original[key]
    return result, dict(raw_prefix_frames_verified=16, prefix_max_absolute_error_A=errors,
                        raw_topology_atom_fields_equal=True, source_topology=str(pdb),
                        source_trajectory=str(trajectory), native_frame_indices=indices.tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("/workspace/MOLVID-FLOW-LIGAND-4096-20261026/tasks/flow-ligand-4096-20261026/configs/ligand.json"))
    parser.add_argument("--output", type=Path, default=ROOT / "runs/flow_kernel_20261027/data")
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument("--atlas-root", type=Path, default=Path("/data1/repo/BioKinema/data_atlas"))
    parser.add_argument("--misato-root", type=Path, default=Path("/data1/repo/PVB/download_data/MISATO"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    if torch.cuda.is_initialized():
        raise RuntimeError("kernel data preparation must not initialize CUDA")
    start = time.monotonic()
    config = json.loads(args.config.read_text())["data"]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "data_manifest.json"
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite frozen manifest: {destination}")
    spec = importlib.util.spec_from_file_location("kernel_existing_contact", CONTACT_FILTER)
    contact = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(contact)
    mixture = BridgeMixture(config, args.seed)
    rows = make_plan(mixture.groups, args.seed)
    stores = {str(store.root): store for domain in mixture.groups.values()
              for buckets in domain.values() for candidates in buckets.values() for store, _ in candidates}
    counts = {domain: len(group) for domain, group in mixture.groups.items()}
    report = dict(schema="molvid.kernel_train_plan.v1", status="preparing", seed=args.seed,
                  data_config=config, inventory_system_counts=counts, rows=[], rejected=[],
                  clock_ps=dict(protein=100., complex=80.), models_called=0, cuda_used=False,
                  geometry_support="unchanged same-record16 TRAIN coordinates, loss only",
                  contact_filter_source=str(CONTACT_FILTER),
                  alignment="original joint first-frame alignment; never concatenate independently aligned clips",
                  cache_index_provenance=dict(
                      atlas="tools/build_frame_joint_multitime.py and data/multitime_stream.py: ordinal * 16 sampled frames",
                      misato="tools/build_complex_training_data.py: w is native start index"))
    print(json.dumps(dict(inventory=counts, visits=len(rows), raw_extension_visits=sum(r['required_frames'] > 16 for r in rows))), flush=True)
    verified_domains = set()
    for row in rows:
        original = stores[row["store"]][row["store_index"]]
        indices = native_indices(original)
        domain = row["domain"]
        delta = 100. if domain == "protein" else 80.
        stride, native_dt = (10, 10.) if domain == "protein" else (1, 80.)
        pdb, trajectory = source_paths(original, args.atlas_root, args.misato_root)
        # Only these predetermined TRAIN source paths are touched, never TEST.
        if not pdb.is_file() or not trajectory.is_file():
            raise FileNotFoundError(f"selected TRAIN raw source unavailable: {pdb}, {trajectory}")
        row.update(sample_id=original["sample_id"], replica=original["replica"], split="train",
                   topology_fingerprint=original["topology_fingerprint"],
                   atom_count=int(original["x"].shape[1]), delta_ps=delta,
                   band_source_sample_id=original["sample_id"], band_frame_count=16,
                   band_native_frame_indices=indices.tolist(),
                   source_topology=str(pdb), source_trajectory=str(trajectory),
                   raw_prefix_frames_verified=0, extended_record=None)
        record = original
        if row["required_frames"] > 16 or domain not in verified_domains:
            record, evidence = extend_or_verify_raw(original, row["required_frames"], args.atlas_root, args.misato_root)
            row.update(evidence)
            verified_domains.add(domain)
        indices = indices[0] + np.arange(len(record["x"]), dtype=np.int64) * stride
        row["native_frame_indices"] = indices.tolist()
        row["absolute_time_ps"] = (indices * native_dt).tolist()
        row["absolute_t0_ps"] = row["absolute_time_ps"][row["t0_index"]]
        if domain == "complex":
            distances = contact.minimum_ligand_protein_distance(record)
            row["minimum_heavy_contact_A_by_frame"] = distances.tolist()
            row["maximum_frame_minimum_contact_A"] = float(distances.max())
            if np.any(distances > 5.):
                report["rejected"].append(dict(row=row, reason="reference_window_contact_gt_5A"))
                (output / "preparation_partial.json").write_text(json.dumps(report, indent=2) + "\n")
                raise ValueError("selected real reference extension violates 5A; no silent replacement")
        if row["required_frames"] > 16:
            record_path = output / f"extended_{row['plan_index']:03d}.pt"
            if record_path.exists():
                raise FileExistsError(record_path)
            torch.save(dict(record=record), record_path)
            row["extended_record"] = str(record_path)
        for mode in ("h1", "history_mix", "dense_history", "lag1"):
            chosen, history, identity = select_kernel_record(record, row, mode)
            assert len(chosen["x"]) == history + 1
            assert np.array_equal(chosen["x"][history - 1], original["x"][row["t0_index"]])
        report["rows"].append(row)
        if (row["plan_index"] + 1) % 32 == 0:
            print(json.dumps(dict(prepared=row["plan_index"] + 1, seconds=time.monotonic() - start)), flush=True)
        (output / "preparation_partial.json").write_text(json.dumps(report, indent=2) + "\n")
    mixture.close()
    report.update(status="complete", plan_digest=plan_digest(rows), rows=rows,
                  actual_counts=dict(domain=dict(Counter(r['domain'] for r in rows)),
                      lag=dict(Counter(str(r['planned_lag_delta']) for r in rows)),
                      history=dict(Counter(r['history_recipe'] for r in rows)),
                      rolling_stage=dict(Counter(str(r['rolling_stage']) for r in rows if r['history_recipe'] == 'rolling'))),
                  selected_unique_systems={d: len({r['system'] for r in rows if r['domain'] == d}) for d in counts},
                  raw_extended_count=sum(r['extended_record'] is not None for r in rows),
                  cpu_wall_seconds=time.monotonic() - start)
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ('status', 'actual_counts', 'raw_extended_count', 'cpu_wall_seconds')}), flush=True)


if __name__ == "__main__":
    main()
