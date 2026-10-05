"""Qualify native BioKinema I/O and score cached coordinates, without model calls."""
import argparse
import json
from pathlib import Path
import re
import numpy as np
import torch
from biotite.structure.io import load_structure
from molvid.evaluation.ca_detail import score
from molvid.evaluation.stereochemistry import ca_handedness
from molvid.geometry.ca_detail import condition_from_record
from molvid.losses.sampled_physical import PhysicalAuxiliaryConfig
from molvid.provenance import freeze_run_identity
from molvid.runtime import atomic_write_json, sha256_file
from molvid.training.predictive_data import RELEASE, atomic_torch_save


def atom_keys(atoms):
    if len(set(atoms.chain_id)) != 1:
        raise ValueError('this pilot audit supports one chain only')
    keys=list(zip(map(int,atoms.res_id),map(str,atoms.ins_code),map(str,atoms.res_name),
                  map(str,atoms.atom_name),map(str,atoms.element)))
    if len(set(keys)) != len(keys): raise ValueError('ambiguous atom identities')
    return keys


def run(args):
    if args.output.exists(): raise FileExistsError(args.output)
    p=args.generated
    summary=json.loads((p/'summary.json').read_text())
    protocol=json.loads((p/'protocol.json').read_text())
    if summary['status']!='completed' or not summary['source_unchanged']:
        raise ValueError('successful original-source inference required')
    identity=json.loads((p/'run_identity.json').read_text())
    for name,digest in identity['input_sha256'].items():
        if sha256_file(Path(name))!=digest: raise ValueError(f'input changed: {name}')
    manifest=json.loads((args.cache/'complete.json').read_text())
    entry=next(e for e in manifest['files'] if e['sample_id']==protocol['sample_id'])
    path=args.cache/entry['path']
    if entry['lag_ps']!=1000 or sha256_file(path)!=entry['sha256']: raise ValueError('native1ns clip identity')
    record=torch.load(path,map_location='cpu',weights_only=False)['record']
    observed=load_structure(str(p/'observed.cif'),model=1)
    keys=atom_keys(observed)
    if np.max(abs(observed.coord-np.asarray(record['x'][3])))>1e-5: raise ValueError('input differs from MD observation')
    artifact=summary['predictions']
    if len(artifact)!=1: raise ValueError('single pilot draw expected')
    array_path=Path(artifact[0]['path'])
    if sha256_file(array_path)!=artifact[0]['sha256']: raise ValueError('prediction changed')
    xyz=np.load(array_path).astype(np.float64)
    if xyz.shape!=(13,1,len(keys),3): raise ValueError('unexpected native output shape')
    cif_paths=[];ordered=[];mapping=None
    for frame in range(13):
        cif=p/f'native/observed/predictions/observed_s0_f{frame}_wounresol.cif'
        atoms=load_structure(str(cif),model=1);other=atom_keys(atoms)
        if set(other)!=set(keys): raise ValueError('atom coverage changed')
        index=np.array([other.index(k) for k in keys])
        if mapping is not None and not np.array_equal(index,mapping): raise ValueError('frame-dependent mapping')
        mapping=index
        if np.max(abs(atoms.coord-xyz[frame,0]))>1e-5: raise ValueError('CIF/array order mismatch')
        ordered.append(xyz[frame,0,index]);cif_paths.append(cif)
    xyz=np.stack(ordered);initial=xyz[0];target=np.asarray(record['x'][3],dtype=np.float64)
    # Undo one shared native augmentation using ONLY the observed first frame.
    # Never fit a frame-wise or hidden-target transform; reflections are forbidden.
    xc=initial-initial.mean(0);yc=target-target.mean(0)
    u,_,vt=np.linalg.svd(xc.T@yc);d=np.eye(3);d[-1,-1]=np.linalg.det(u@vt)
    rotation=u@d@vt;translation=target.mean(0)-initial.mean(0)@rotation
    xyz=xyz@rotation+translation
    parity=float(np.max(abs(xyz[0]-target)))
    if parity>1e-5: raise ValueError(f'initial rigid-frame parity failed: {parity}')
    depths=list(map(int,re.findall(r'N_msa (\d+)',(p/'native.log').read_text())))
    if len(depths)!=1 or depths[0]<=1: raise ValueError('MSA-enabled inference not verified')
    if protocol['coarse_interval_native_ns']!=1 or protocol['future_frames']!=12:
        raise ValueError('native physical-time protocol differs')
    release=json.loads(RELEASE.read_text())
    recipe=Path('/workspace/molvid-recon-gm-motion-v3')/release['auxiliary_recipe']['path']
    if sha256_file(recipe)!=release['auxiliary_recipe']['sha256']: raise ValueError('physical recipe changed')
    physical=PhysicalAuxiliaryConfig.resolve(json.loads(recipe.read_text())['physical'])
    freeze_run_identity(args.output,inputs=[p/'summary.json',p/'protocol.json',p/'native.log',path,array_path,recipe]+cif_paths,
        purpose='CPU BioKinema native1ns I/O qualification',protocol=dict(model_calls=0,training_updates=0,
            pilot=True,alignment='one observed-frame proper rigid transform shared across all frames',test_opened=False))
    coordinates=torch.as_tensor(xyz[1:],dtype=torch.float32)
    c=condition_from_record(record)
    with torch.no_grad():
        metrics=score(coordinates,record,c,physical)
        chirality=ca_handedness(coordinates,observed=c.observed,atom_identity=record['atom_identity'],block_id=record['block_id'])
    atomic_torch_save(args.output/'coordinates.pt',dict(sample_id=entry['sample_id'],coordinates=coordinates))
    atomic_write_json(args.output/'summary.json',dict(status='completed',scoring_qualified=True,
        system=entry['system'],sample_id=entry['sample_id'],native_interval_ps=1000,future_frames=12,
        msa_depth=depths[0],atom_count=len(keys),atom_mapping=mapping.tolist(),initial_max_error_A=parity,
        rotation=rotation.tolist(),translation_A=translation.tolist(),metrics=metrics,chirality=chirality,
        model_calls=0,training_updates=0,compute='CPU',pilot=True,
        limitation='one development case; not a comparative or generalization result'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--generated',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
