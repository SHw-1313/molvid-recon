"""Select an auditable bounded PDB static pool from an existing processed store."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping

import numpy as np

from molvid.cli.train_frame_joint import _open_data
from molvid.config import load_config
from molvid.data.sampling import get_clip_specs
from molvid.data.store import StaticClipDataset
from molvid.runtime import atomic_write_json, canonical_hash, sha256_file
from molvid.training.static_schedule import internal_md_static_pool


def _commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        text=True,
        cwd=Path(__file__).resolve().parents[1],
    ).strip()


def _system_id(sample_id: str) -> str | None:
    match = re.match(r"^atlas_(.+)_R[123]_", str(sample_id))
    return None if match is None else match.group(1)


def _legacy_origin(legacy_id: str) -> str:
    prefix, separator, suffix = str(legacy_id).rpartition("_")
    return prefix if separator and suffix.isdigit() else str(legacy_id)


def _quality(record: Mapping[str, Any], *, max_atoms: int) -> tuple[bool, str]:
    coordinates = np.asarray(record["x"])
    bpos = np.asarray(record["bpos"])
    atype = np.asarray(record["atype"])
    loss_mask = np.asarray(record["loss_mask"], dtype=np.bool_)
    bonds = np.asarray(record["bond_index"])
    atoms = int(coordinates.shape[1]) if coordinates.ndim == 3 else 0
    if coordinates.shape != (1, atoms, 3) or bpos.shape != coordinates.shape:
        return False, "shape"
    if not 1 <= atoms <= int(max_atoms):
        return False, "atom_count"
    if not np.isfinite(coordinates).all() or not np.isfinite(bpos).all():
        return False, "nonfinite_coordinates"
    if float(np.ptp(coordinates)) <= 1.0e-6:
        return False, "degenerate_coordinates"
    if loss_mask.shape != (atoms,) or not bool(loss_mask.any()):
        return False, "empty_loss_mask"
    if atype.shape != (atoms,) or int(atype.min()) < 0 or int(atype.max()) > 117:
        return False, "unsupported_atom_type"
    if bonds.ndim != 2 or bonds.shape[0] != 2 or bonds.shape[1] == 0:
        return False, "missing_covalent_topology"
    return True, "eligible"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dynamic-config", type=Path, required=True)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--selection-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--max-atoms", type=int, default=5000)
    parser.add_argument("--source", default="pdb")
    parser.add_argument("--split", default="train")
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    if not 2000 <= args.count <= 5000:
        raise ValueError("external static pool must contain 2000--5000 structures")
    raw = load_config(args.dynamic_config, schema="molvid.frame_joint.train.v1")
    train, valid, data_hash, normalization_hash, _, close = _open_data(raw["data"])
    try:
        dynamic_systems = {
            system
            for table in (get_clip_specs(train), get_clip_specs(valid))
            for item in table
            for system in [_system_id(item.sample_id)]
            if system is not None
        }
        internal = internal_md_static_pool(train)
        internal_atoms = np.asarray([entry.atoms for entry in internal], dtype=np.int64)
    finally:
        close()
    if len(internal) != 576:
        raise ValueError("released dynamic data did not yield the 576-trajectory pool")

    # Coarsely match external atom-count exposure to the internal source while
    # retaining a seeded random selection within each bin.
    inner_edges = np.quantile(internal_atoms, [0.25, 0.50, 0.75])
    internal_bins = np.digitize(internal_atoms, inner_edges, right=True)
    proportions = np.bincount(internal_bins, minlength=4) / len(internal_bins)
    quotas = np.floor(proportions * args.count).astype(np.int64)
    for index in range(args.count - int(quotas.sum())):
        quotas[index % len(quotas)] += 1

    dataset = StaticClipDataset(args.store, source=args.source, split=args.split)
    selected: list[int] = []
    selected_rows: list[dict[str, Any]] = []
    accepted_by_bin = np.zeros(4, dtype=np.int64)
    rejected: Counter[str] = Counter()
    direct_overlap_count = 0
    rng = np.random.default_rng(args.seed)
    try:
        for raw_index in rng.permutation(len(dataset)):
            index = int(raw_index)
            legacy_id = str(dataset._legacy._indexes[index][0])
            origin = _legacy_origin(legacy_id)
            if origin in dynamic_systems:
                direct_overlap_count += 1
                continue
            record = dataset[index]
            eligible, reason = _quality(record, max_atoms=args.max_atoms)
            if not eligible:
                rejected[reason] += 1
                continue
            atoms = int(record["x"].shape[1])
            bin_index = int(np.digitize([atoms], inner_edges, right=True)[0])
            if accepted_by_bin[bin_index] >= quotas[bin_index]:
                rejected["filled_atom_bin"] += 1
                continue
            selected.append(index)
            accepted_by_bin[bin_index] += 1
            selected_rows.append(
                {
                    "dataset_index": index,
                    "legacy_id": legacy_id,
                    "origin": origin,
                    "atoms": atoms,
                    "loss_atoms": int(np.asarray(record["loss_mask"]).sum()),
                    "covalent_directed_edges": int(record["bond_index"].shape[1]),
                    "atom_bin": bin_index,
                }
            )
            if len(selected) == args.count:
                break
    finally:
        del dataset
    if len(selected) != args.count or not np.array_equal(accepted_by_bin, quotas):
        raise RuntimeError("processed PDB store could not fill the matched external pool")

    payload = {
        "schema": "molvid.static_decoder.external_eligibility.v1",
        "status": "eligible",
        "source": args.source,
        "split": args.split,
        "store": str(args.store.resolve()),
        "store_index_sha256": sha256_file(args.store / "index.txt"),
        "selection_report": str(args.selection_report.resolve()),
        "selection_report_sha256": sha256_file(args.selection_report),
        "selected_indices": selected,
        "selected_records": selected_rows,
        "selected_count": len(selected),
        "selected_indices_hash": canonical_hash(selected),
        "selection_seed": args.seed,
        "selection_method": "seeded permutation; quality filter; internal-MD atom-count quartile quotas",
        "quality_requirements": {
            "coordinate_unit": "angstrom",
            "finite_x_and_bpos": True,
            "maximum_atoms": args.max_atoms,
            "nonempty_loss_mask": True,
            "nonempty_covalent_topology": True,
            "supported_atom_type_ids": [0, 117],
            "record_wide_degenerate_coordinate_fill_allowed": False,
            "missing_atom_policy": "reuse the existing processed PDB record; do not synthesize atoms or coordinates",
        },
        "atom_exposure_matching": {
            "internal_pool_count": len(internal),
            "internal_atom_quartile_edges": inner_edges.tolist(),
            "external_bin_quotas": quotas.tolist(),
            "external_bin_counts": accepted_by_bin.tolist(),
            "internal_atom_mean": float(internal_atoms.mean()),
            "external_atom_mean": float(np.mean([row["atoms"] for row in selected_rows])),
        },
        "direct_overlap_excluded": True,
        "direct_overlap_definition": "exact PDB-chain origin match to released A train or valid system ID",
        "direct_overlap_candidates_rejected": direct_overlap_count,
        "dynamic_system_id_count": len(dynamic_systems),
        "near_homology_policy": "waived_by_user_20260929",
        "near_homology_excluded": False,
        "near_homology_authorization": "explicit user waiver on 2026-09-29 for this bounded pilot",
        "raw_source_authorized": "/data5/PVB/pdb",
        "processed_store_reused": True,
        "rejected_counts": dict(sorted(rejected.items())),
        "dynamic_data_hash": data_hash,
        "normalization_source_data_hash": normalization_hash,
        "code_commit": _commit(),
        "sealed_test_trajectories_opened": False,
        "limitations": [
            "No sequence- or structure-level near-homology filtering is applied in this bounded version.",
            "PDB static structures are geometry-supervision samples, not Boltzmann population samples or motion labels."
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output, payload)
    print(json.dumps({
        "output": str(args.output.resolve()),
        "selected_count": len(selected),
        "selected_indices_hash": payload["selected_indices_hash"],
        "external_atom_mean": payload["atom_exposure_matching"]["external_atom_mean"],
        "internal_atom_mean": payload["atom_exposure_matching"]["internal_atom_mean"],
        "test_opened": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
