"""Known-current-training sequence overlap of the already-used validation16.

Only PDB residue identifiers and HDF molecule boundaries are read. No trajectory
coordinates, sealed test structures or reserved confirmation systems are read.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import subprocess

import h5py
import numpy as np

from molvid.data.chemistry import RES_TYPE_3, RES_TYPE_1
from molvid.runtime import atomic_write_json


AA = dict(zip(RES_TYPE_3, RES_TYPE_1))
AA['MSE'] = 'M'
# Amber protonation/disulfide labels change chemistry state, not the amino-acid
# sequence identity. PDB loading elsewhere normalizes these automatically.
AA.update(HID='H', HIE='H', HIP='H', HSD='H', HSE='H', HSP='H',
          CYX='C', CYM='C', ASH='D', GLH='E', LYN='K')


def sequences(path, boundaries=None):
    chains = defaultdict(list)
    atom = -1
    for line in path.read_text().splitlines():
        if line.startswith('ENDMDL'):
            break
        if line[:6] not in ('ATOM  ', 'HETATM'):
            continue
        atom += 1
        if boundaries is not None and atom >= boundaries[-1]:
            continue  # last molecular component is the ligand
        if line[12:16].strip() != 'CA' or line[16:17] not in (' ', 'A'):
            continue
        residue = line[17:20].strip()
        if residue not in AA:
            raise ValueError(f'Unknown protein CA residue {residue} in {path}')
        chain = (int(np.searchsorted(boundaries, atom, side='right')) - 1
                 if boundaries is not None else line[21:22])
        chains[chain].append(AA[residue])
    return {str(k): ''.join(v) for k, v in chains.items()}


def run(output, cov_mode):
    output.mkdir(parents=True, exist_ok=False)
    data = Path('/workspace/molvid-complex-mixture-20261007/runs/complex_mixture_20261007/misato128_v2')
    systems = json.loads((data/'systems.json').read_text())
    contract = json.loads(Path('runs/complex_pretraining_20261007/atom_71/train/contract.json').read_text())
    entries = {}
    with h5py.File('/data1/repo/PVB/download_data/MISATO/MD.hdf5', 'r') as handle:
        for row in systems:
            system = row['system']
            boundaries = np.asarray(handle[system]['molecules_begin_atom_index'])
            for chain, sequence in sequences(Path(row['pdb']), boundaries).items():
                entries[f'misato_{system}_{chain}'] = dict(system=system, split=row['split'],
                    domain='complex', chain=chain, sequence=sequence)
    for system in contract['complex_mixture']['protein_systems']:
        name = system.removeprefix('atlas_')
        path = Path('/data1/repo/BioKinema/data_atlas')/name/f'{name}.pdb'
        for chain, sequence in sequences(path).items():
            entries[f'{system}_{chain}'] = dict(system=system, split='train',
                domain='protein', chain=chain, sequence=sequence)
    for split, filename in (('train', 'train.fasta'), ('valid', 'valid.fasta')):
        with (output/filename).open('x') as handle:
            for key, row in sorted(entries.items()):
                if row['split'] == split and len(row['sequence']) >= 20:
                    handle.write(f'>{key}\n{row["sequence"]}\n')
    command = ['mmseqs', 'easy-search', str(output/'valid.fasta'), str(output/'train.fasta'),
        str(output/'hits.tsv'), str(output/'tmp'), '--threads', '4', '-s', '7.5',
        '--min-seq-id', '0.3', '-c', '0.5', '--cov-mode', str(cov_mode), '--max-seqs', '10000',
        '--remove-tmp-files', '1', '--format-output', 'query,target,pident,alnlen,qlen,tlen,qcov,tcov,evalue,bits']
    with (output/'search.log').open('x') as handle:
        subprocess.run(command, check=True, stdout=handle, stderr=subprocess.STDOUT)
    hits = defaultdict(list)
    for line in (output/'hits.tsv').read_text().splitlines():
        q, t, identity, length, qlen, tlen, qcov, tcov, evalue, bits = line.split('\t')
        hits[entries[q]['system']].append(dict(query=q, target=t, target_domain=entries[t]['domain'],
            identity_percent=float(identity), aligned_residues=int(length),
            query_length=int(qlen), target_length=int(tlen), query_coverage=float(qcov),
            target_coverage=float(tcov), evalue=float(evalue), bits=float(bits)))
    per_system = {s: sorted(hits[s], key=lambda x: -x['identity_percent'])
                  for s in sorted({r['system'] for r in systems if r['split'] == 'valid'})}
    result = dict(status='completed', trajectory_coordinates_read=0, sealed_test_read=False,
        known_training_systems=dict(protein=192, complex=128), validation_systems=16,
        command=command, coverage_mode=cov_mode, per_system=per_system,
        systems_with_hit_at_identity_percent={str(threshold): [s for s, rows in per_system.items()
            if any(r['identity_percent'] >= threshold for r in rows)] for threshold in (30, 50, 80, 95)},
        short_chains_excluded={key: len(row['sequence']) for key, row in entries.items() if len(row['sequence']) < 20},
        entries=entries,
        limitations=['Known current training only, not complete parent/static/released-baseline exposure',
            'Sequence similarity is not fold or ligand-scaffold independence',
            'MISATO molecular components can be fragments, not complete biological chains',
            'Coverage mode0 requires both sequences; mode2 requires query coverage only and can expose partial-domain similarity',
            'Absence of hits does not establish no shared domains; short fragments below20 residues omitted',
            'Do not change or discard current validation conditions based on these findings'])
    atomic_write_json(output/'summary.json', result)
    print(json.dumps(result['systems_with_hit_at_identity_percent'], indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cov-mode', type=int, choices=(0, 2), default=0)
    args = parser.parse_args()
    run(args.output, args.cov_mode)
