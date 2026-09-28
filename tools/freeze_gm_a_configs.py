"""Freeze four symmetric fresh-parent configs from shared train-only calibration."""

import argparse
import copy
import json
from pathlib import Path

import torch

from molvid.cli.train_frame_joint import _open_data, _sampler, _validate_frame_joint_config
from molvid.config import load_config
from molvid.runtime import atomic_write_json, sha256_file
from tools.gm_a_evidence import PARENT, PARENT_SHA, SEED


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--recipe',type=Path,required=True);p.add_argument('--profile',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    recipe=json.loads(args.recipe.read_text());profile=json.loads(args.profile.read_text())
    raw=load_config('configs/frame_gm_p2_formal_GM_260923.yaml',schema='molvid.frame_joint.train.v1')
    if sha256_file(PARENT)!=PARENT_SHA:raise ValueError('parent hash differs')
    parent=torch.load(PARENT,map_location='cpu',weights_only=False)
    raw['loss']=parent['contracts']['loss']
    raw['training'].update({'seed':SEED,'deterministic':True,'gradient_diagnostics_every':500})
    raw['training']['stages'][0]['learning_rates']={'history_encoder':1e-4,'dit':2e-4,'decoder':5e-5}
    train,_,_,_,_,close=_open_data(raw['data'])
    try:
        sampler=_sampler(train,raw['training'],seed=SEED,rank=0,world=1)
        counts=[]
        for epoch in range(profile['chosen_effective_epochs']):sampler.set_epoch(epoch);counts.append(len(sampler))
    finally:close()
    manifest={'schema':'molvid.A.frozen_configs.v1','parent_path':str(PARENT),'parent_sha256':PARENT_SHA,
              'recipe_sha256':sha256_file(args.recipe),'profile_sha256':sha256_file(args.profile),'updates_per_epoch':counts,
              'formal_total_updates':sum(counts),'arms':{},'test_opened':False}
    for arm in ('A0','A1','A2','A3'):
        cfg=copy.deepcopy(raw);physical=copy.deepcopy(recipe['physical'])
        if arm=='A1':physical.update(rmsf_weight=0.,displacement_weight=0.,bond_band_weight=0.)
        elif arm=='A2':physical['bond_band_weight']=0.
        cfg['sampled_auxiliary']={'enabled':arm!='A0','euler_steps':recipe['euler_steps'],'draws':2,'cadence':8,'activation_checkpoint':True}
        if arm!='A0':cfg['sampled_auxiliary']['physical']=physical
        records={}
        for phase,updates in (('pilot',128),('formal',sum(counts))):
            current=copy.deepcopy(cfg);current['training']['stages'][0]['updates']=updates
            current['training']['output_root']=str((args.output/phase/arm).resolve())
            if phase=='pilot':current['training']['gradient_diagnostics_every']=32
            _validate_frame_joint_config(current)
            path=args.output/f'{phase}_{arm}.json';atomic_write_json(path,current)
            records[phase]={'config':str(path.resolve()),'config_sha256':sha256_file(path),'output_root':current['training']['output_root']}
        manifest['arms'][arm]=records
    atomic_write_json(args.output/'frozen_configs.json',manifest)


if __name__=='__main__':main()
