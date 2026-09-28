"""Actual ordinary/aux timings on the largest 192-system train sample."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from molvid.checkpoints import load_frame_joint_inference
from molvid.cli.train_frame_joint import _open_data
from molvid.config import load_config
from molvid.data.batch import collate_clip_records
from molvid.data.sampling import get_clip_specs
from molvid.runtime import atomic_write_json, configure_device, seed_all
from molvid.training.batches import prepare_frame_joint_batch
from molvid.training.dit import module_state_hash
from molvid.training.joint import FrameJointTrainer, JointLossConfig, SampledAuxiliaryConfig
from tools.gm_a_evidence import PARENT, PARENT_SHA, SEED, selected_entries


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--recipe',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    started=time.time();device=configure_device('cuda',deterministic=True);seed_all(SEED)
    raw=load_config('configs/frame_gm_p2_formal_GM_260923.yaml',schema='molvid.frame_joint.train.v1')
    recipe=json.loads(args.recipe.read_text())
    train,_,data_hash,_,_,close=_open_data(raw['data'])
    try:
        largest=max(get_clip_specs(train),key=lambda x:(x.effective_tokens,x.sample_id))
        loaded=load_frame_joint_inference(PARENT,expected_sha256=PARENT_SHA,codec_path=raw['codec']['checkpoint'],codec_sha256=raw['codec']['sha256'],device=device)
        model=loaded['model']
        trainer=FrameJointTrainer(model,loss_config=JointLossConfig.resolve(loaded['payload']['contracts']['loss']),
            learning_rates={'history_encoder':1e-4,'dit':2e-4,'decoder':5e-5},stage='continuation',amp=True,
            data_hash=data_hash,teacher_artifact_sha256=raw['codec']['sha256'],schedule_contract={'profile_only':True},
            sampled_config=SampledAuxiliaryConfig.resolve({'enabled':True,'physical':recipe['physical'],'euler_steps':recipe['euler_steps']}))
        warm=trainer.load_warm_start(PARENT,expected_sha256=PARENT_SHA)
        before=module_state_hash(model.target_teacher)
        io_start=time.time();cpu=collate_clip_records([train[largest.index]]);io_seconds=time.time()-io_start
        rows=[]
        for history in (4,8):
            prep_start=time.time();batch=prepare_frame_joint_batch(model.target_teacher,cpu,device=device,normalizer=model,history_frames=history)
            torch.cuda.synchronize();prepare_seconds=time.time()-prep_start
            main=torch.Generator(device=device).manual_seed(SEED);aux_generator=torch.Generator(device=device).manual_seed(SEED+1)
            for active in (False,True,False,True):
                torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.time()
                metrics=trainer.train_step(batch,generator=main,sampled_generator=aux_generator,auxiliary_active=active,diagnose_gradients=False)
                torch.cuda.synchronize()
                rows.append({'history':history,'aux':active,'seconds':time.time()-start,'peak_bytes':torch.cuda.max_memory_allocated(),
                             'prepare_seconds':prepare_seconds,'metrics':metrics})
                atomic_write_json(args.output/'profile_progress.json',{'rows':rows,'largest':largest.__dict__})
                print(json.dumps({k:v for k,v in rows[-1].items() if k!='metrics'}),flush=True)
        ordinary=max(r['seconds'] for r in rows if not r['aux']);aux=max(r['seconds'] for r in rows if r['aux'])
        prep=max(r['prepare_seconds'] for r in rows)
        largest_bound_2epochs=((ordinary+prep+io_seconds)+3*(.875*ordinary+.125*aux+prep+io_seconds))*21204/3600
        # Largest-record VRAM is a hard feasibility check. Estimate runtime on
        # actual packed batches, stratified over all nine view/lag/H groups.
        entries,_=selected_entries(train,raw)
        representatives={}
        for entry in entries:representatives.setdefault((entry.view,entry.time_bucket_id,entry.history_frames),entry)
        representative_rows=[]
        for index,entry in enumerate(representatives.values()):
            started_input=time.time()
            cpu=collate_clip_records([train[i] for i in entry.indices])
            batch=prepare_frame_joint_batch(model.target_teacher,cpu,device=device,normalizer=model,history_frames=entry.history_frames)
            torch.cuda.synchronize();input_seconds=time.time()-started_input
            timing={}
            for active in (False,True):
                torch.cuda.synchronize();tick=time.time()
                trainer.train_step(batch,generator=main,sampled_generator=aux_generator,auxiliary_active=active)
                torch.cuda.synchronize();timing['aux' if active else 'ordinary']=time.time()-tick
            representative_rows.append({'entry':entry.__dict__,'input_seconds':input_seconds,**timing})
            print(json.dumps({'representative':index+1,**timing}),flush=True)
        # The global schedule's stratum frequencies are close to design weights;
        # compute the exact batch-frequency weights from the inspected epoch.
        from molvid.cli.train_frame_joint import _sampler
        from molvid.training.execution import execution_schedule
        from collections import Counter
        sampler=_sampler(train,raw['training'],seed=SEED,rank=0,world=1)
        execution=execution_schedule(sampler.global_batches,sampler.specs,histories=(4,8),seed=SEED,epoch=0)
        frequencies=Counter((e.view,e.time_bucket_id,e.history_frames) for e in execution)
        per_update_base=per_update_aux=0.
        for r in representative_rows:
            e=r['entry'];weight=frequencies[(e['view'],e['time_bucket_id'],e['history_frames'])]/len(execution)
            per_update_base+=weight*(r['ordinary']+r['input_seconds'])
            per_update_aux+=weight*(.875*r['ordinary']+.125*r['aux']+r['input_seconds'])
        # 20% runtime headroom; A3 timings upper-bound all auxiliary-arm work.
        projected_2epochs=1.2*(per_update_base+3*per_update_aux)*21204/3600
        epochs=2 if projected_2epochs<=48 else 1
        if projected_2epochs/2>48: raise RuntimeError('even one paired epoch exceeds the formal 48 GPUh envelope')
        atomic_write_json(args.output/'profile.json',{'schema':'molvid.A.profile.v1','rows':rows,'largest':largest.__dict__,
            'io_seconds':io_seconds,'teacher_unchanged':before==module_state_hash(model.target_teacher),'warm_start_report':warm.as_dict(),
            'model_parameters':sum(p.numel() for p in model.parameters()),'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
            'optimizer_groups':[{'name':g['group_name'],'lr':g['lr'],'parameters':sum(p.numel() for p in g['params'])} for g in trainer.optimizer.param_groups],
            'conservative_four_arm_2epoch_gpu_hours':projected_2epochs,'chosen_effective_epochs':epochs,
            'largest_only_2epoch_upper_bound_gpu_hours':largest_bound_2epochs,
            'representative_rows':representative_rows,'runtime_headroom_factor':1.2,
            'projected_formal_gpu_hours':projected_2epochs*epochs/2,'gpu_hours':(time.time()-started)/3600,
            'peak_bytes':max(r['peak_bytes'] for r in rows),'test_opened':False})
    finally:close()


if __name__=='__main__':main()
