"""CUDA mixed-mask, independent-switch and exact-overlap numerical checks."""

import argparse
from dataclasses import replace
import json
from pathlib import Path

import torch

from molvid.checkpoints import load_frame_joint_inference
from molvid.cli.train_frame_joint import _open_data
from molvid.config import load_config
from molvid.data.batch import collate_clip_records
from molvid.data.sampling import get_clip_specs
from molvid.losses.sampled_physical import PhysicalAuxiliaryConfig, physical_auxiliary_loss
from molvid.runtime import atomic_write_json,configure_device
from molvid.training.batches import prepare_frame_joint_batch
from tools.gm_a_evidence import PARENT,PARENT_SHA,SEED,TERMS,forward_samples


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--tables',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();device=configure_device('cuda',deterministic=True)
    raw=load_config('configs/frame_gm_p2_formal_GM_260923.yaml',schema='molvid.frame_joint.train.v1')
    train,_,_,_,_,close=_open_data(raw['data'])
    try:
        loaded=load_frame_joint_inference(PARENT,expected_sha256=PARENT_SHA,codec_path=raw['codec']['checkpoint'],codec_sha256=raw['codec']['sha256'],device=device)
        model=loaded['model']
        specs=sorted((s for s in get_clip_specs(train) if s.time_bucket_id=='dt_100ps'),key=lambda s:(s.atoms,s.index))
        first=specs[0];second=next(s for s in specs if s.sample_id.rsplit('_R',1)[0]!=first.sample_id.rsplit('_R',1)[0])
        cpu=collate_clip_records([train[first.index],train[second.index]])
        mask=cpu.frame_mask.clone();mask[1,10:]=False
        cpu=replace(cpu,frame_mask=mask)
        batch=prepare_frame_joint_batch(model.target_teacher,cpu,device=device,normalizer=model,history_frames=4)
        cfg=PhysicalAuxiliaryConfig.resolve(json.loads(args.tables.read_text())['physical'])
        with torch.no_grad():
            output,_=forward_samples(model,batch,seed=SEED,steps=4)
        xyz=tuple(x.coordinates.detach().requires_grad_() for x in output.sampled)
        full,counts,_=physical_auxiliary_loss(xyz,batch,cfg)
        if int(counts['sampled_rmsf'])!=1 or int(counts['sampled_displacement'])!=2:
            raise RuntimeError('mixed Q eligibility is wrong')
        comparisons={}
        names=('clash_weight','rmsf_weight','displacement_weight','bond_band_weight')
        for term,weight in zip(TERMS,names):
            settings={name:0. for name in names};settings[weight]=1.
            isolated=replace(cfg,**settings)
            losses,_,_=physical_auxiliary_loss(xyz,batch,isolated)
            if not torch.equal(full[term],losses[term]):raise RuntimeError(f'independent {term} differs')
            gradients=torch.autograd.grad(losses[term],xyz,retain_graph=True)
            for gradient in gradients:
                atom_mask=batch.observed_context.topology.abid==1
                if not torch.equal(gradient[6:,atom_mask],torch.zeros_like(gradient[6:,atom_mask])):
                    raise RuntimeError('invalid packed query frame receives gradient')
                if not torch.isfinite(gradient).all():raise RuntimeError('invalid branch derivative')
            comparisons[term]='exact independent-switch match; padded gradients zero'
        # Coincident nonbonded nodes explicitly exercise norm(0) subgradient.
        coincident=torch.zeros_like(xyz[0],requires_grad=True)
        clash=replace(cfg,rmsf_weight=0.,displacement_weight=0.,bond_band_weight=0.)
        loss,_,diag=physical_auxiliary_loss((coincident,coincident),batch,clash)
        loss['sampled_clash'].backward()
        if not torch.isfinite(coincident.grad).all() or diag['exact_overlap_pair_frames']<=0:
            raise RuntimeError('exact-overlap handling failed')
        atomic_write_json(args.output,{'passed':True,'checks':comparisons,'counts':{n:int(x) for n,x in counts.items()},
            'overlap_diagnostics':diag,'device':torch.cuda.get_device_name(),'test_opened':False})
    finally:close()


if __name__=='__main__':main()
