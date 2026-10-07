"""Materialize continuous development trajectories; never concatenate aligned windows."""
import argparse
import json
from pathlib import Path
import time
import warnings

import h5py
import mdtraj as md
import numpy as np

from molvid.data.preprocess import make_clip_record, topology_metadata
from molvid.data.store import ClipMMapDataset, ClipMMapWriter
from molvid.runtime import atomic_write_json


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    source = json.loads((args.existing/'summary.json').read_text())
    systems = [r for r in source['systems'] if r['split'] == 'valid']
    if len(systems) != 16:
        raise ValueError('Use exactly the already-opened16 development systems')
    started = time.monotonic()
    previous = ClipMMapDataset(args.existing/'valid')
    index = {fields[0]:i for i,fields in enumerate(previous._clip_index_fields)}
    rows = []
    try:
        with h5py.File(args.raw/'MD.hdf5', 'r') as handle, ClipMMapWriter(args.output/'valid') as writer:
            for entry in systems:
                system = entry['system']
                pdb = Path(entry['pdb'])
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    topology = md.load(str(pdb)).topology
                metadata = topology_metadata(topology, complex_topology=True)
                # Only explicit existing VALID IDs are dereferenced in this HDF.
                group = handle[system]
                numbers = np.asarray(group['atoms_number'])
                if not np.array_equal(numbers, [a.element.atomic_number for a in topology.atoms]):
                    raise ValueError('PDB/HDF atom order differs')
                xyz = np.asarray(group['trajectory_coordinates'][:96], dtype=np.float32)
                if len(xyz) != 96 or not np.isfinite(xyz).all():
                    raise ValueError('Need the actual continuous96-frame trajectory')
                sample = f'misato_{system}_MD_dt_80ps_rollout96'
                record = make_clip_record(xyz, np.arange(96)*80., metadata,
                    sample_id=sample, source='misato', system_id=system, replica='MD', split='valid',
                    source_stride=1, native_dt_ps=80.,
                    timestamp_provenance='Native80ps frames0..95; one observed-frame0 alignment reference')
                old = previous[index[f'misato_{system}_MD_dt_80ps_w000000']]
                # Same initial observed condition as the existing short forecast.
                # This comparison is part of data construction, not a model gate.
                if not np.array_equal(record['x'][:16], old['x']):
                    raise ValueError('Continuous record first window differs from existing development data')
                for field in ('atype','btype','atom_source_index','component_id','bond_index'):
                    if not np.array_equal(record[field], old[field]):
                        raise ValueError(f'Existing topology differs: {field}')
                record['source_pdb'] = str(pdb)
                writer.append(record)
                rows.append(dict(system=system, sample_id=sample, atoms=len(record['atype']), pdb=str(pdb)))
                print(json.dumps(dict(completed=len(rows), system=system, frames=96)), flush=True)
        atomic_write_json(args.output/'summary.json', dict(status='completed', systems=rows,
            frames=96, native_dt_ps=80, observed_frames=4, future_frames=92, prediction_horizon_ps=7360,
            source=str(args.existing), seconds=time.monotonic()-started, model_calls=0,
            test_payload_opened=False, independent_test=False, interpolation=False, atom_crop=False,
            interpretation='Finite-horizon free-running stability and motion, not equilibrium or thermodynamic fidelity'))
    finally:
        previous.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--existing', type=Path, default=Path('/workspace/molvid-complex-mixture-20261007/runs/complex_mixture_20261007/misato128_v2'))
    parser.add_argument('--raw', type=Path, default=Path('/data1/repo/PVB/download_data/MISATO'))
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
