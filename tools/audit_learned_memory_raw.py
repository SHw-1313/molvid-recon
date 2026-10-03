"""B pilot: frozen split, PDB size census, actual 0.8/3.2/12.8 ns raw past."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from molvid.data.io import read_topology
from molvid.data.preprocess import topology_metadata
from molvid.data.raw_history import RawHistoryWindow, read_raw_window_bank, align_bank_to_observed
from molvid.runtime import sha256_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--raw-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(args.manifest.read_text())
    systems = sorted(s.removeprefix('atlas_') for s in manifest['source_splits']['train']['selected_systems'])
    if len(systems) != 192:
        raise ValueError(f"expected frozen 192 train systems, got {len(systems)}")
    census, metadata = [], {}
    for system in systems:
        pdb = args.raw_root / system / f'{system}.pdb'
        meta = topology_metadata(read_topology(pdb), complex_topology=False)
        metadata[system] = meta
        census.append({'system': system, 'atoms': len(meta['atype']), 'pdb_sha256': sha256_file(pdb)})
    by_size = sorted(census, key=lambda row: (row['atoms'], row['system']))
    size_profiles = [by_size[len(by_size)//2]['system'], by_size[-1]['system']]
    selected = systems[:3]
    (args.output / 'size_census.json').write_text(json.dumps({'systems': census, 'pilot_systems': selected, 'profile_systems': size_profiles}, indent=2))
    rows = []
    for system in sorted(set(selected + size_profiles)):
        meta = metadata[system]
        for replica in (('R1', 'R2', 'R3') if system in selected else ('R1',)):
            start = time.perf_counter()
            pdb = args.raw_root / system / f'{system}.pdb'
            xtc = args.raw_root / system / f'{system}_prod_{replica}_fit.xtc'
            windows = [RawHistoryWindow(1280, span, dt, lag, 'train')
                       for span in (800, 3200, 12800) for dt in (100, 200, 400) for lag in (100, 200, 400)]
            needed = np.unique(np.concatenate([np.concatenate((w.history_indices, w.anchor_indices, w.future_indices)) for w in windows]))
            bank, times, indices = read_raw_window_bank(xtc, pdb, needed, atom_source_index=meta['atom_source_index'])
            for window in windows:
                window.validate_coverage(int(indices[-1]) + 1)
            bank = align_bank_to_observed(bank, indices, 1280, meta['align_mask'])
            output = args.output / f'{system}_{replica}.npz'
            np.savez_compressed(output, coordinates_angstrom=bank, absolute_time_ps=times, raw_indices=indices,
                                **{k: meta[k] for k in ('atype', 'btype', 'block_id', 'component_id', 'bond_index', 'align_mask', 'loss_mask', 'atom_source_index')})
            row = {'system': system, 'replica': replica, 'role': 'pilot' if system in selected else 'size_profile',
                   'raw_path': str(xtc), 'raw_size_bytes': xtc.stat().st_size, 'raw_mtime_ns': xtc.stat().st_mtime_ns,
                   'bank_path': str(output), 'bank_sha256': sha256_file(output), 'atoms': bank.shape[1],
                   'native_dt_ps': 10, 'scanned_raw_prefix_frames': 1761, 'full_trajectory_tail_audited': False,
                   'minimum_verified_past_ps': 12800, 'minimum_verified_future_ps': 4800,
                   'read_align_write_seconds': time.perf_counter()-start, 'windows': [w.manifest() for w in windows]}
            rows.append(row)
            (args.output / 'progress.json').write_text(json.dumps(rows, indent=2))
            print(json.dumps({k: row[k] for k in ('system', 'replica', 'atoms', 'read_align_write_seconds')}), flush=True)
    card = {'status': 'raw_pilot_only', 'manifest_path': str(args.manifest), 'manifest_sha256': sha256_file(args.manifest),
            'split_hash': hashlib.sha256(json.dumps(systems, separators=(',', ':')).encode()).hexdigest(),
            'train_system_count': len(systems), 'raw_pilot_system_count': 3, 'raw_pilot_trajectory_count': 9,
            'size_profile_systems': size_profiles, 'test_opened': False, 'query_lag_300_read': False,
            'model_device_used': None, 'gpu_hours': 0, 'rows': rows}
    (args.output / 'data_card.json').write_text(json.dumps(card, indent=2))


if __name__ == '__main__':
    main()
