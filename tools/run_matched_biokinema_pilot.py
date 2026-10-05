"""Original released-sqrt BioKinema, exact observed CIF, native1ns pilot.

Compatibility checkout already differs from official source by one Python3.11
f-string quote fix. No model/sampling implementation changes are made here.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch
from biotite.structure.io import load_structure,save_structure
from molvid.provenance import freeze_run_identity
from molvid.runtime import atomic_write_json,sha256_file
from molvid.training.execution_evidence import ExecutionEvidence


def run(args):
    if args.output.exists(): raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    source=Path('/workspace/BioKinema-inference-20261004')
    checkpoint=Path('/workspace/molvid-integration-abcd-20261004/inputs/baselines/BioKinema_atlas_misato_mdposit_sqrt.pt')
    if sha256_file(checkpoint)!='c96ad4083fb4bbac3608237d83096d8203d6649bce2e051e090fb0bd19dece39':
        raise ValueError('released sqrt SHA mismatch')
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()
    status=subprocess.check_output(['git','status','--porcelain'],cwd=source,text=True)
    if not revision.startswith('cd5f716') or status: raise ValueError('unverified compatibility checkout')
    manifest=json.loads((args.cache/'complete.json').read_text())
    entry=next(e for e in manifest['files'] if e['lag_ps']==1000)
    record_path=args.cache/entry['path']
    if sha256_file(record_path)!=entry['sha256']: raise ValueError('record changed')
    item=torch.load(record_path,map_location='cpu',weights_only=False);record=item['record']
    pdb=Path(item['source']['pdb'])
    if sha256_file(pdb)!=item['source']['pdb_sha256']: raise ValueError('PDB changed')
    atoms=load_structure(str(pdb),model=1)[np.asarray(record['atom_source_index'])]
    expected=[str(identity).rsplit(':',2)[-2].strip() for identity in record['atom_identity']]
    if list(atoms.atom_name)!=expected: raise ValueError('Biotite/MD atom order differs')
    aliases=[]
    for index,(name,residue) in enumerate(zip(atoms.atom_name,atoms.res_name)):
        new='CD1' if residue=='ILE' and name=='CD' else {'OT1':'O','OT2':'OXT'}.get(name,name)
        if new!=name: aliases.append(dict(index=index,old=name,new=new));atoms.atom_name[index]=new
    atoms.coord=np.asarray(record['x'][3],dtype=np.float32).copy()
    input_file=args.output/'observed.cif';save_structure(str(input_file),atoms)
    reread=load_structure(str(input_file),model=1)
    error=float(np.max(np.abs(reread.coord-atoms.coord)))
    if list(reread.atom_name)!=list(atoms.atom_name) or error>1e-5: raise ValueError('CIF roundtrip altered input')
    values={'seeds':17,'load_checkpoint_path':checkpoint,'dump_dir':args.output/'native',
        'model.N_cycle':10,'model.diffusion_module.causal_mask':'false','model.diffusion_module.beta':.5,
        'data.train_sets':'inference','data.test_sets':'inference','sample_diffusion.N_sample':1,
        'sample_diffusion.N_step':20,'sample_diffusion.noise_scale_lambda':1.75,'sample_diffusion.step_scale_eta':1.5,
        'infer_setting.sample_diffusion_chunk_size':1,'coarse_frame_num':13,'coarse_interval':1,
        'fine_frame_num':1,'W_H':1,'W_G':12,'input_file':input_file.resolve(),'data.num_dl_workers':0,
        'data.msa.enable':'true','load_strict':'true',
        'data.msa.prot.pdb_mmseqs_dir':'/data1/repo/BioKinema/release_data/mmcif_msa',
        'data.msa.prot.seq_to_pdb_idx_path':'/data1/repo/BioKinema/release_data/seq_to_pdb_index.json'}
    values['dump_dir']=values['dump_dir'].resolve()
    command=[sys.executable,'-u',str(source/'runner/inference.py')]
    for key,value in values.items(): command.extend([f'--{key}',str(value)])
    protocol=dict(system=entry['system'],sample_id=entry['sample_id'],source_commit=revision,
        aliases=aliases,input_max_roundtrip_error_A=error,observed_frames=1,future_frames=12,
        query_interval_ps=1000,coarse_interval_native_ns=1,seed=17,command=command,
        original_model_changes=[],training_updates=0,pilot=True,test_opened=False,
        physical_clock_native=True,MSA='enabled; actual depth must be checked from native log',
        limitation='single-case I/O and native-time qualification, not a competitive benchmark')
    freeze_run_identity(args.output,inputs=[record_path,input_file,checkpoint],purpose='Native1ns matched BioKinema pilot',protocol=protocol)
    atomic_write_json(args.output/'protocol.json',protocol)
    atomic_write_json(args.output/'status.json',dict(status='prepared_waiting_for_own_C8_GPU',wait_for=str(args.wait_for)))
    started=time.monotonic()
    while not args.wait_for.exists():
        if time.monotonic()-started>64800: raise TimeoutError('C8 prerequisite18h')
        time.sleep(20)
    if json.loads(args.wait_for.read_text()).get('exception') is not None: raise RuntimeError('C8 prerequisite failed')
    dependencies=Path('/workspace/molvid-integration-abcd-20261004/runs/d2_mamba_20261004')
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',
        PYTHONPATH=os.pathsep.join(map(str,[dependencies/'bio_deps',dependencies/'pvb_numpy126',source])),
        BIOKINEMA_DATA_ROOT='/data1/repo/BioKinema/release_data/ccd_cache',
        BIOKINEMA_ATLAS_ROOT=str((args.output/'atlas_cache').resolve()),
        LAYERNORM_TYPE='torch',USE_DEEPSPEED_EVO_ATTTENTION='false')
    atomic_write_json(args.output/'status.json',dict(status='native_GPU_inference'))
    started=time.monotonic()
    with ExecutionEvidence(args.output),(args.output/'native.log').open('x') as log:
        result=subprocess.run(command,cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=3600)
        predictions=list((args.output/'native').rglob('*pred_coordinates.npy'))
        errors=list((args.output/'native/ERR').glob('*.txt'))
        if result.returncode or errors or not predictions: raise RuntimeError('native inference failed; inspect native.log')
        artifacts=[]
        for path in predictions:
            data=np.load(path,mmap_mode='r')
            if data.shape[0]!=13 or not np.isfinite(data).all(): raise ValueError('native frame/finite contract')
            artifacts.append(dict(path=str(path),sha256=sha256_file(path),shape=list(data.shape)))
        after=subprocess.check_output(['git','status','--porcelain'],cwd=source,text=True)
        if after!=status: raise RuntimeError('external source changed')
        atomic_write_json(args.output/'summary.json',dict(status='completed',seconds=time.monotonic()-started,
            predictions=artifacts,source_unchanged=True,training_updates=0,
            scoring_qualified=False,reason='output atom mapping, initial-frame parity and MSA depth require audit'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--wait-for',type=Path,required=True)
    run(p.parse_args())
