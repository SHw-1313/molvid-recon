"""Bounded native-reference audit, not a thermodynamic certification."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np
from ..provenance import freeze_run_identity
from ..runtime import atomic_write_json, sha256_file
from .static_response import validation_plan, VIEWS


def correlation_summary(values):
    """Initial-positive paired ACF estimate; scalar and finite-window only."""
    x=np.asarray(values,dtype=np.float64)
    if x.ndim!=1 or len(x)<20 or not np.isfinite(x).all():
        raise ValueError('finite scalar reference series of at least 20 frames required')
    centered=x-x.mean();n=len(x)
    blocks=np.array_split(x,4)
    result=dict(frames=n,mean=float(x.mean()),std=float(x.std()),
                four_contiguous_block_means=[float(b.mean()) for b in blocks])
    if np.dot(centered,centered)<1e-20:
        return dict(result,ess=None,tau_frames=None,status='constant_observable_not_informative')
    spectrum=np.fft.rfft(centered,n=2*n)
    acf=np.fft.irfft(spectrum*spectrum.conj(),n=2*n)[:n]
    acf/=acf[0];tau=1.;last=0
    for lag in range(1,n//2-1,2):
        pair=float(acf[lag]+acf[lag+1])
        if pair<=0: break
        tau+=2*pair;last=lag+1
    return dict(result,ess=float(n/max(tau,1.)),tau_frames=float(max(tau,1.)),
                positive_pair_cutoff=last,status='descriptive_not_stationarity_test')


def qualify_references(output,frames=4000):
    import mdtraj as md
    output=Path(output)
    if output.exists(): raise FileExistsError(output)
    if frames<100 or frames>10000: raise ValueError('bounded reference window required')
    paired=VIEWS/'paired_view_manifest.json'
    raw=Path(json.loads(paired.read_text())['raw_root'])
    manifest=Path('/workspace/molvid_recon/runs/frame_joint_v1_multitime_data_260921/manifest.json')
    splits=json.loads(manifest.read_text())['source_splits']
    selected=validation_plan(systems=4)['systems']
    freeze_run_identity(output,inputs=[paired,manifest],purpose='N4 reference qualification',protocol=dict(
        systems=selected,replicas=['R1','R2','R3'],first_native_frames=frames,native_dt_ps=10,
        observables=['CA_Rg_A','first_to_middle_CA_distance_A','first_to_last_CA_distance_A'],
        state_definition=None,test_payload_opened=False,new_MD=False))
    sequences=defaultdict(list);missing=[]
    # Metadata from train/valid only. Never open test structures or payloads.
    for split in ('train','valid'):
        for system in splits[split]['selected_systems']:
            name=system.removeprefix('atlas_');pdb=raw/name/f'{name}.pdb'
            if not pdb.exists(): missing.append(system);continue
            top=md.load_topology(str(pdb))
            seq='|'.join('-'.join(r.name for r in chain.residues if r.is_protein) for chain in top.chains)
            sequences[hashlib.sha256(seq.encode()).hexdigest()].append(dict(system=system,split=split))
    overlap=[rows for rows in sequences.values() if len({r['split'] for r in rows})>1]
    rows=[];sources=[]
    for system in selected:
        name=system.removeprefix('atlas_');folder=raw/name;pdb=folder/f'{name}.pdb'
        top=md.load_topology(str(pdb));ca=top.select('protein and name CA')
        if len(ca)<3: raise ValueError('reference requires >=3 CA atoms')
        readme=folder/'README.txt'
        sources.append(dict(path=str(pdb),sha256=sha256_file(pdb)))
        if readme.exists(): sources.append(dict(path=str(readme),sha256=sha256_file(readme)))
        for replica in ('R1','R2','R3'):
            xtc=folder/f'{name}_prod_{replica}_fit.xtc';tpr=folder/f'{name}_prod_{replica}.tpr'
            if not xtc.exists():
                rows.append(dict(system=system,replica=replica,status='missing_reference'));continue
            series=[];clock=[];count=0
            for chunk in md.iterload(str(xtc),top=str(pdb),atom_indices=ca,chunk=256):
                xyz=chunk.xyz[:frames-count].astype(np.float64)*10
                center=xyz.mean(1,keepdims=True)
                series.append(np.stack([np.sqrt(((xyz-center)**2).sum(-1).mean(-1)),
                    np.linalg.norm(xyz[:,0]-xyz[:,len(ca)//2],axis=-1),
                    np.linalg.norm(xyz[:,0]-xyz[:,-1],axis=-1)],axis=-1))
                clock.extend(chunk.time[:len(xyz)].tolist());count+=len(xyz)
                if count>=frames: break
            values=np.concatenate(series);clock=np.array(clock)
            if not np.allclose(np.diff(clock),10,atol=.001,rtol=0): raise ValueError('wrong native time grid')
            sources.append(dict(path=str(xtc),sha256=sha256_file(xtc)))
            if tpr.exists(): sources.append(dict(path=str(tpr),sha256=sha256_file(tpr)))
            rows.append(dict(system=system,replica=replica,status='measured',
                time_range_ps=[float(clock[0]),float(clock[-1])],tpr_available=tpr.exists(),
                observables={key:correlation_summary(values[:,i]) for i,key in enumerate(
                    ('CA_Rg_A','first_to_middle_CA_distance_A','first_to_last_CA_distance_A'))}))
            atomic_write_json(output/'rows.json',rows)
            print(json.dumps(dict(phase='N4',system=system,replica=replica,frames=count)),flush=True)
    variations={}
    for system in selected:
        group=[r for r in rows if r['system']==system and r['status']=='measured']
        variations[system]={key:dict(replica_mean_range=[min(r['observables'][key]['mean'] for r in group),
                max(r['observables'][key]['mean'] for r in group)]) for key in group[0]['observables']} if group else {}
    atomic_write_json(output/'sources.json',sources)
    atomic_write_json(output/'summary.json',dict(status='completed',reference_rows=len(rows),
        exact_sequence_cross_split_overlap=overlap,missing_sequence_sources=missing,
        parent_exposure={'train_system_count':len(splits['train']['selected_systems']),
            'valid_system_count':len(splits['valid']['selected_systems']),
            'head_calibration':'subset of parent training systems, not parent-unseen',
            'earlier_codec_pretraining_exposure':'unknown; not independently reconstructed'},
        physical_metadata={'force_field':'README declares CHARMM36m; TPR parameters not parsed',
            'native_frame_spacing_ps':10,'saved_components':'README: protein only, solvent removed',
            'temperature':'unknown','solvent_and_ions':'unknown','protonation':'unknown','constraints':'unknown'},
        replica_variations=variations,thermodynamic_qualification='insufficient_evidence',
        unresolved=['homology-cluster isolation beyond exact residue sequence',
            'ligand similarity and complex references: no complexes in this ATLAS pilot',
            'physical-condition equality requires TPR parameter extraction',
            'metastable states, transition counts, occupancy CI and deltaF not established',
            'scalar finite-window ESS is not effective sampling of all coordinates',
            'first 40 ns only; blocks/replicas not assumed independent equilibrium draws'],test_opened=False))
