"""Known-current-training sequence overlap of existing development panels.

Read PDB residue identifiers, HDF molecule boundaries and optionally registered
static polymer sequences. No trajectory coordinates, sealed test structures or
reserved confirmation systems are read.
"""
import argparse
from collections import defaultdict
import hashlib
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


def coordinate_entries(config_path, protein_valid, complex_valid):
    """Explicit current MD-only inventory; no static or external-weight claim."""
    config = json.loads(config_path.read_text())
    selected = config['selected_systems']
    training = json.loads((Path(config['complex_train']).parent/'systems.json').read_text())
    training = [r for r in training if r['system'] in selected['complex']]
    validation = json.loads(complex_valid.read_text())
    if validation['status'] != 'completed' or validation['test_payload_opened']:
        raise ValueError('Only existing completed development metadata permitted')
    if {r['system'] for r in training} != set(selected['complex']) or any(r['split'] != 'train' for r in training):
        raise ValueError('Current complex TRAIN inventory mismatch')
    protein = json.loads(protein_valid.read_text())
    if protein['status'] != 'completed' or protein['test_opened']:
        raise ValueError('Only existing completed protein development metadata permitted')
    protein_systems = sorted({r['system'] for r in protein['files']})
    entries = {}
    with h5py.File('/data1/repo/PVB/download_data/MISATO/MD.hdf5', 'r') as handle:
        for split, rows in [('train', training), ('valid', validation['systems'])]:
            for row in rows:
                system = row['system']
                boundaries = np.asarray(handle[system]['molecules_begin_atom_index'])
                fragments = sequences(Path(row['pdb']), boundaries)
                if not fragments:
                    raise ValueError(f'No protein sequence for {system}')
                for chain, sequence in fragments.items():
                    entries[f'{split}_misato_{system}_{chain.strip() or "blank"}'] = dict(system=system,
                        split=split, domain='complex', chain=chain, sequence=sequence)
    for split, systems in [('train', selected['protein']), ('valid', protein_systems)]:
        for system in systems:
            name = system.removeprefix('atlas_')
            path = Path('/data1/repo/BioKinema/data_atlas')/name/f'{name}.pdb'
            fragments = sequences(path)
            if not fragments:
                raise ValueError(f'No protein sequence for {system}')
            for chain, sequence in fragments.items():
                entries[f'{split}_{system}_{chain.strip() or "blank"}'] = dict(system=system, split=split,
                    domain='protein', chain=chain, sequence=sequence)
    return entries


def add_static_exposure(entries, recipe_path, update_paths):
    """Add only actually exposed registered chains, not the eligible corpus.

    Full deposited polymer sequences include unresolved residues. This is a
    conservative source-relatedness audit, not proof of atom-level exposure.
    """
    recipe = json.loads(recipe_path.read_text())
    logs = [[json.loads(line) for line in path.read_text().splitlines()] for path in update_paths]
    identities = []
    for rows in logs:
        if [r['step'] for r in rows] != list(range(1, recipe['updates'] + 1)) or any(
                r['task'] != 'static' for r in rows):
            raise ValueError('Static exposure requires the complete ordered training log')
        identities.append([(r['domain'], r['system'], r['sample']) for r in rows])
    if any(rows != identities[0] for rows in identities[1:]):
        raise ValueError('Paired static exposures differ; audit the arms separately')
    registration = Path(recipe['data']['static_registration'])
    registered = {r['source_sample_id']: r for r in json.loads(
        (registration/'registered_sources.json').read_text())}
    identity = json.loads((registration/'run_identity.json').read_text())
    paths = [Path(p) for p in identity['input_sha256'] if Path(p).name == 'sequences.json']
    if len(paths) != 1:
        raise ValueError('Expected one registered polymer-sequence inventory')
    sequences_by_digest = {hashlib.sha256(r['sequence'].encode()).hexdigest(): r['sequence']
                           for r in json.loads(paths[0].read_text()).values()}
    exposed = sorted({r['sample'] for r in logs[0] if r['domain'] == 'protein'})
    for sample in exposed:
        row = registered[sample]
        for chain in row['selected_chains']:
            sequence = sequences_by_digest[chain['sequence_sha256']]
            if len(sequence) != chain['sequence_length']:
                raise ValueError('Registered chain length differs from sequence inventory')
            label = chain['label_asym_id']
            entries[f'train_static_{row["system_id"]}_{label}'] = dict(
                system=f'static_{row["system_id"]}', split='train', domain='protein',
                chain=label, sequence=sequence, source_kind='actual_static_exposure',
                sample=sample, includes_unresolved_polymer_residues=True)
    # Complex singletons come from the already enumerated MD TRAIN systems.
    complexes = {r['system'] for r in logs[0] if r['domain'] == 'complex'}
    if not complexes <= {r['system'] for r in entries.values()
                         if r['split'] == 'train' and r['domain'] == 'complex'}:
        raise ValueError('Static complex exposure outside enumerated MD TRAIN')
    return dict(recipe=str(recipe_path), updates=[str(p) for p in update_paths],
                static_protein_sources=len(exposed), static_complex_systems=len(complexes),
                paired_exposure_identical=True, sequence_inventory=str(paths[0]),
                definition='MD TRAIN plus actual static exposure; full registered polymer sequences, including unresolved residues')


def run(output, cov_mode, coordinate_config=None, protein_valid=None, complex_valid=None,
        static_recipe=None, static_updates=None):
    output.mkdir(parents=True, exist_ok=False)
    if coordinate_config is not None:
        entries = coordinate_entries(coordinate_config, protein_valid, complex_valid)
        exposure = add_static_exposure(entries, static_recipe, static_updates) if static_recipe else None
        search(output, cov_mode, entries, exposure)
        return
    data = Path('/workspace/molvid-complex-mixture-20261007/runs/complex_mixture_20261007/misato128_v2')
    systems = json.loads((data/'systems.json').read_text())
    contract = json.loads(Path('runs/complex_pretraining_20261007/atom_71/train/contract.json').read_text())
    entries = {}
    with h5py.File('/data1/repo/PVB/download_data/MISATO/MD.hdf5', 'r') as handle:
        for row in systems:
            system = row['system']
            boundaries = np.asarray(handle[system]['molecules_begin_atom_index'])
            for chain, sequence in sequences(Path(row['pdb']), boundaries).items():
                entries[f'misato_{system}_{chain.strip() or "blank"}'] = dict(system=system, split=row['split'],
                    domain='complex', chain=chain, sequence=sequence)
    for system in contract['complex_mixture']['protein_systems']:
        name = system.removeprefix('atlas_')
        path = Path('/data1/repo/BioKinema/data_atlas')/name/f'{name}.pdb'
        for chain, sequence in sequences(path).items():
            entries[f'{system}_{chain.strip() or "blank"}'] = dict(system=system, split='train',
                domain='protein', chain=chain, sequence=sequence)
    search(output, cov_mode, entries)


def search(output, cov_mode, entries, static_exposure=None):
    for split, filename in (('train', 'train.fasta'), ('valid', 'valid.fasta')):
        with (output/filename).open('x') as handle:
            for key, row in sorted(entries.items()):
                if row['split'] == split and len(row['sequence']) >= 20:
                    handle.write(f'>{key}\n{row["sequence"]}\n')
    command = ['mmseqs', 'easy-search', str(output/'valid.fasta'), str(output/'train.fasta'),
        str(output/'hits.tsv'), str(output/'tmp'), '--threads', '4', '-s', '7.5',
        '--min-seq-id', '0.3', '-c', '0.5', '--cov-mode', str(cov_mode), '--max-seqs', '10000',
        '-e', '0.001', '--alignment-mode', '3',
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
                  for s in sorted({r['system'] for r in entries.values() if r['split'] == 'valid'})}
    counts = {split: {domain: len({r['system'] for r in entries.values()
        if r['split'] == split and r['domain'] == domain}) for domain in ('protein', 'complex')}
        for split in ('train', 'valid')}
    result = dict(status='completed', trajectory_coordinates_read=0, sealed_test_read=False,
        known_training_systems=counts['train'], validation_systems=sum(counts['valid'].values()),
        validation_domains=counts['valid'],
        command=command, coverage_mode=cov_mode, per_system=per_system,
        systems_with_hit_at_identity_percent={str(threshold): [s for s, rows in per_system.items()
            if any(r['identity_percent'] >= threshold for r in rows)] for threshold in (30, 50, 80, 95)},
        short_chains_excluded={key: len(row['sequence']) for key, row in entries.items() if len(row['sequence']) < 20},
        entries=entries,
        static_exposure=static_exposure,
        limitations=['Enumerated current training only, not unenumerated parent/static/released-baseline exposure',
            'Static full polymer sequences include unresolved residues not necessarily present in model inputs',
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
    parser.add_argument('--coordinate-config', type=Path)
    parser.add_argument('--protein-valid', type=Path)
    parser.add_argument('--complex-valid', type=Path)
    parser.add_argument('--static-recipe', type=Path)
    parser.add_argument('--static-updates', type=Path, nargs='+')
    args = parser.parse_args()
    if any(x is not None for x in (args.coordinate_config, args.protein_valid, args.complex_valid)) and not all(
            x is not None for x in (args.coordinate_config, args.protein_valid, args.complex_valid)):
        parser.error('Current-coordinate audit requires all three explicit inventory paths')
    if bool(args.static_recipe) != bool(args.static_updates) or (
            args.static_recipe and not args.coordinate_config):
        parser.error('Static exposure requires its recipe, complete update logs and explicit coordinate inventories')
    run(args.output, args.cov_mode, args.coordinate_config, args.protein_valid, args.complex_valid,
        args.static_recipe, args.static_updates)
