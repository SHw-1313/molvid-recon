"""Train-only GM/J0/J1 Euler4/8/16/32 diagnostic before auxiliary step freeze."""

import argparse
import gc
import json
from pathlib import Path
import time

import numpy as np
import torch

from molvid.checkpoints import load_frame_joint_inference
from molvid.cli.train_frame_joint import _open_data
from molvid.config import load_config
from molvid.flow.sampling import euler_sample_frames
from molvid.losses.sampled_physical import PhysicalAuxiliaryConfig, physical_auxiliary_loss, motion_features
from molvid.losses.distribution import _sample_metadata, _sample_pairs
from molvid.runtime import atomic_write_json, configure_device
from tools.gm_a_evidence import PARENT, PARENT_SHA, SEED, selected_entries, prepare


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tables',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    start=time.time(); device=configure_device('cuda',deterministic=True)
    raw=load_config('configs/frame_gm_p2_formal_GM_260923.yaml',schema='molvid.frame_joint.train.v1')
    train,_,_,_,_,close=_open_data(raw['data'])
    root=Path('/workspace/molvid-recon-gm-calibration-v2/runs/frame_gm_calibration_v2/P3/formal')
    parents={'GM':(PARENT,PARENT_SHA),
             'J0':(root/'J0/frame_joint_step_00021208.pt','60e14c3e7cc59c8e4c1a2ff38512df829db7418446cc82bd3ed3587722696f5e'),
             'J1':(root/'J1/frame_joint_step_00021208.pt','8859a499be7d06f31ba33fb9e71febf5019cebb9ced1947d0c277326da0c66e1')}
    entries,_=selected_entries(train,raw)
    selected={}
    for entry in entries: selected.setdefault((entry.view,entry.time_bucket_id,entry.history_frames),entry)
    entries=list(selected.values())
    table=json.loads(args.tables.read_text())
    config=PhysicalAuxiliaryConfig.resolve({**table['physical'],'rmsf_weight':0.,'displacement_weight':0.,'bond_band_weight':0.})
    # Freeze the qualitative spec into an explicit test before reading generation.
    rule={'large_4_vs_16_over_16_vs_32':2.,'eight_vs_sixteen_reduction_max':0.75,
          'choose_eight':'at least one geometry metric AND one motion metric pass pooled paired medians',
          'metrics_geometry':['clash_severity','bond_rmse_A'],'metrics_motion':['rmsf_A','msd_A2']}
    atomic_write_json(args.output/'step_choice_rule.json',rule)
    rows=[]
    try:
        for name,(path,sha) in parents.items():
            loaded=load_frame_joint_inference(path,expected_sha256=sha,codec_path=raw['codec']['checkpoint'],codec_sha256=raw['codec']['sha256'],device=device)
            model=loaded['model'].eval()
            for index,entry in enumerate(entries):
                batch=prepare(model,train,entry,device)
                for seed in (SEED,SEED+1):
                    for steps in (4,8,16,32):
                        with torch.no_grad(), torch.autocast('cuda',dtype=torch.bfloat16):
                            latent,_=euler_sample_frames(model,context=batch.observed_context,query=batch.query,steps=steps,seed=seed)
                            xyz=model.decoder(batch.observed_context,latent,batch.query).coordinates.float()
                        with torch.no_grad():
                            physical,_,_=physical_auxiliary_loss((xyz,xyz),batch,config)
                            rmsf=[]; msd=[]; bond=[]
                            for sample in range(batch.observed_context.latent.batch_size):
                                features=motion_features(xyz,batch,sample,rmsf=entry.view=='ordinary',displacement=True)
                                if 'sampled_rmsf' in features: rmsf.append(float(features['sampled_rmsf'].mean()))
                                msd.append(float(features['sampled_displacement'][:int(batch.query.frame_mask[sample].sum())].mean()))
                                nodes,_,_,frames,_=_sample_metadata(batch,sample)
                                a,b=_sample_pairs(batch,sample,nodes)
                                if len(a):
                                    predicted=torch.linalg.vector_norm(xyz[frames][:,a]-xyz[frames][:,b],dim=-1)
                                    target=torch.linalg.vector_norm(batch.target_coordinates[frames][:,a]-batch.target_coordinates[frames][:,b],dim=-1)
                                    bond.append(float((predicted-target).square().mean().sqrt()))
                        rows.append({'parent':name,'entry':index,'seed':seed,'steps':steps,'clash_severity':float(physical['sampled_clash']),
                                     'bond_rmse_A':float(np.mean(bond)),'rmsf_A':float(np.mean(rmsf)) if rmsf else None,'msd_A2':float(np.mean(msd))})
                atomic_write_json(args.output/'diagnostic_progress.json',{'rows':rows})
                print(json.dumps({'parent':name,'condition':index+1,'complete_conditions':len(entries)}),flush=True)
            del model,loaded,batch,latent,xyz
            gc.collect(); torch.cuda.empty_cache()
        comparisons={}
        for metric in ('clash_severity','bond_rmse_A','rmsf_A','msd_A2'):
            deltas=[]
            for name in parents:
                for index in range(len(entries)):
                    for seed in (SEED,SEED+1):
                        values={r['steps']:r[metric] for r in rows if r['parent']==name and r['entry']==index and r['seed']==seed}
                        if any(v is None for v in values.values()): continue
                        deltas.append([abs(values[4]-values[16]),abs(values[8]-values[16]),abs(values[16]-values[32])])
            med=np.median(deltas,axis=0)
            comparisons[metric]={'median_abs_4_16':float(med[0]),'median_abs_8_16':float(med[1]),'median_abs_16_32':float(med[2]),
                                  'supports_eight':bool(med[0]>2*med[2] and med[1]<.75*med[0])}
        choose8=any(comparisons[n]['supports_eight'] for n in ('clash_severity','bond_rmse_A')) and any(comparisons[n]['supports_eight'] for n in ('rmsf_A','msd_A2'))
        atomic_write_json(args.output/'integrator_choice.json',{'schema':'molvid.A.integrator_choice.v1','steps':8 if choose8 else 4,'rule':rule,'comparisons':comparisons,
            'rows':rows,'entries':[e.__dict__ for e in entries],'parents':{k:{'path':str(p),'sha256':s} for k,(p,s) in parents.items()},
            'split':'train','test_opened':False,'gpu_hours':(time.time()-start)/3600,'evaluation_steps':16})
    finally: close()


if __name__=='__main__': main()
