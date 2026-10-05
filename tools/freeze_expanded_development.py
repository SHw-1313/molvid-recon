"""Sequence-only candidate split audit; never reads trajectory targets or sealed PDBs."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import subprocess

from molvid.provenance import freeze_run_identity
from molvid.runtime import atomic_write_json,sha256_file

AA=dict(zip('ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL'.split(),
            'ARNDCQEGHILKMFPSTWYV'))


def sequence(path):
    """Read residue identities only from first-model CA records; no XYZ parsing."""
    chains=defaultdict(dict)
    for line in path.read_text().splitlines():
        if line.startswith('ENDMDL'):break
        if line[:6] not in ('ATOM  ','HETATM') or line[12:16].strip()!='CA':continue
        if line[16:17] not in (' ','A'):continue
        name=line[17:20].strip()
        if name not in AA:raise ValueError(f'unsupported CA residue {name}')
        chains[line[21:22]][line[22:27]]=AA[name]
    if len(chains)!=1:raise ValueError(f'expected one chain, got {len(chains)}')
    value=''.join(next(iter(chains.values())).values())
    if len(value)<20:raise ValueError('sequence too short')
    return value


def freeze(output):
    if output.exists():raise FileExistsError(output)
    manifests=[Path('/workspace/molvid_recon/runs/frame_joint_v1_multitime_data_260921/manifest.json'),
               Path('/workspace/PVB/outputs/state_detail_codec_v2/t1/manifest_20260904_token80000/manifest.json')]
    exposed,sealed=set(),set();provenance={}
    for path in manifests:
        data=json.loads(path.read_text())
        subsets=data['source_splits']
        blocked=set(subsets['train']['selected_systems'])|set(subsets['valid']['selected_systems'])
        blocked.update(data.get('excluded_historical_t0_systems',[]))
        exposed.update(blocked);sealed.update(subsets['test']['selected_systems'])
        provenance[str(path)]=dict(exposed=sorted(blocked),sealed=sorted(subsets['test']['selected_systems']))
    if exposed&sealed:raise ValueError('exposure/sealed metadata overlap')
    root=Path('/data1/repo/BioKinema/data_atlas')
    protocol=dict(stage='sequence-only development/confirmation reservation',old_sealed_excluded=sorted(sealed),
        known_exposure_excluded=sorted(exposed),sequence_identity=.3,coverage=.5,cov_mode=0,cluster_mode=1,
        developer_target=48,confirmation_reserve_target=96,selection='hash-sorted, round-robin length bins; one representative per clean cluster',
        timestamp_policy='native XTC only in subsequent development materialization',test_opened=False,
        limitations=['sequence clusters are not fold isolation','complete historical/static/pretrained exposure audit still required',
                     'baseline model exposure must be classified separately','confirmation trajectories remain unopened'])
    freeze_run_identity(output,inputs=manifests,purpose='Expanded evaluation candidate reservation',protocol=protocol)
    atomic_write_json(output/'protocol.json',protocol)
    entries={};excluded={}
    for directory in sorted(root.iterdir()):
        system='atlas_'+directory.name
        if system in sealed:continue
        pdb=directory/(directory.name+'.pdb')
        if not pdb.is_file():continue
        try:seq=sequence(pdb)
        except ValueError as error:
            excluded[system]=str(error);continue
        replicas=[directory/(directory.name+f'_prod_R{i}_fit.xtc') for i in (1,2,3)]
        entries[system]=dict(sequence=seq,residues=len(seq),pdb=str(pdb),pdb_sha256=sha256_file(pdb),
            replicas=[str(p) for p in replicas],replicas_present=all(p.is_file() for p in replicas),exposed=system in exposed)
    missing=sorted(exposed-set(entries))
    if missing:raise ValueError(f'exposed sequences missing; cannot claim clean clusters: {missing}')
    fasta=output/'sequences.fasta'
    with fasta.open('x') as handle:
        for system,e in entries.items():handle.write(f'>{system}\n{e["sequence"]}\n')
    command=['mmseqs','easy-cluster',str(fasta),str(output/'clusters'),str(output/'mmseqs_tmp'),
             '--min-seq-id','0.3','-c','0.5','--cov-mode','0','--cluster-mode','1','-s','7.5','--threads','8']
    with (output/'mmseqs.log').open('x') as handle:subprocess.run(command,check=True,stdout=handle,stderr=subprocess.STDOUT)
    clusters=defaultdict(list)
    for line in (output/'clusters_cluster.tsv').read_text().splitlines():
        representative,member=line.split('\t');clusters[representative].append(member)
    order=lambda s:hashlib.sha256(('molvid-expanded-dev-v1:'+s).encode()).hexdigest()
    eligible=[];contaminated=[]
    for members in clusters.values():
        if set(members)&exposed:contaminated.extend(members);continue
        candidates=[s for s in members if entries[s]['replicas_present']]
        if candidates:eligible.append(sorted(candidates,key=order)[0])
    bins=defaultdict(list)
    for s in sorted(eligible,key=order):
        n=entries[s]['residues'];bins[0 if n<=100 else 1 if n<=200 else 2 if n<=400 else 3].append(s)
    balanced=[]
    while any(bins.values()):
        for b in sorted(bins):
            if bins[b]:balanced.append(bins[b].pop(0))
    dev=balanced[:48];confirmation=balanced[48:144]
    manifest=dict(status='candidate_reservation_frozen',development=dev,confirmation_reserved=confirmation,
        extra_eligible=balanced[144:],eligible_cluster_count=len(eligible),known_exposed=len(exposed),
        excluded_sealed=sorted(sealed),contaminated_cluster_members=sorted(contaminated),
        source_manifest_exposure=provenance,entries=entries,excluded_pdbs=excluded,
        sequence_cluster_sha256=sha256_file(output/'clusters_cluster.tsv'),test_opened=False,
        trajectory_targets_read=0,independent_confirmation_qualified=False,
        reason='known-parent sequence-isolated candidates; full provenance and baseline exposure audit pending')
    atomic_write_json(output/'manifest.json',manifest)
    print(json.dumps(dict(status=manifest['status'],known_exposed=len(exposed),sequences=len(entries),
        clusters=len(clusters),eligible=len(eligible),development=len(dev),confirmation_reserved=len(confirmation),
        trajectory_targets_read=0,test_opened=False)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    freeze(parser.parse_args().output)
