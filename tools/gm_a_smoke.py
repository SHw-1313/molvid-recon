"""Real CUDA A auxiliary smoke and exact resume; no formal weights are emitted."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

import torch

from molvid.data.batch import collate_clip_records
from molvid.data.sampling import get_clip_specs
from molvid.losses.sampled_physical import PhysicalAuxiliaryConfig, physical_auxiliary_loss
from molvid.training.batches import prepare_frame_joint_batch
from molvid.training.dit import module_state_hash
from molvid.training.joint import FrameJointTrainer, JointLossConfig, SampledAuxiliaryConfig, _gradient_norm
from molvid.checkpoints import load_frame_joint_inference
from molvid.cli.train_frame_joint import _open_data
from molvid.config import load_config
from molvid.runtime import atomic_write_json, configure_device, seed_all
from tools.gm_a_evidence import PARENT, PARENT_SHA, SEED, forward_samples, TERMS


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tables',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    started=time.time(); device=configure_device('cuda',deterministic=True); seed_all(SEED)
    raw=load_config('configs/frame_gm_p2_formal_GM_260923.yaml',schema='molvid.frame_joint.train.v1')
    train,_,data_hash,_,_,close=_open_data(raw['data'])
    evidence={'schema':'molvid.A.cuda_smoke.v1','test_opened':False,'checks':[]}
    try:
        loaded=load_frame_joint_inference(PARENT,expected_sha256=PARENT_SHA,codec_path=raw['codec']['checkpoint'],codec_sha256=raw['codec']['sha256'],device=device)
        model=loaded['model']; model.train(); model.target_teacher.eval()
        cfg=PhysicalAuxiliaryConfig.resolve(json.loads(args.tables.read_text())['physical'])
        specs=get_clip_specs(train)
        ordinary=[s for s in specs if s.time_bucket_id=='dt_100ps']
        systems=[]
        for s in sorted(ordinary,key=lambda s:(s.atoms,s.sample_id)):
            name=s.sample_id.rsplit('_R',1)[0]
            if name not in systems: systems.append(name)
            if len(systems)==3: break
        chosen=[min((s for s in ordinary if s.sample_id.startswith(f'{name}_R{rep}_')),key=lambda s:s.index) for name in systems for rep in (1,2,3)]
        teacher_before=module_state_hash(model.target_teacher)
        for index,spec in enumerate(chosen):
            cpu=collate_clip_records([train[spec.index]])
            batch=prepare_frame_joint_batch(model.target_teacher,cpu,device=device,normalizer=model,history_frames=4)
            for amp in (False,True) if index==0 else (True,):
                generated,flow=forward_samples(model,batch,seed=SEED+index,steps=4,amp=amp)
                losses,counts,diag=physical_auxiliary_loss(tuple(x.coordinates for x in generated.sampled),batch,cfg)
                gradients={}
                if index==0:
                    for term in TERMS:
                        gradients[term]={name:float(_gradient_norm(losses[term],[p for p in module.parameters() if p.requires_grad],retain_graph=True))
                            for name,module in [('dit',model.dit),('decoder',model.decoder),('history',model.history_encoder),('motion',model.motion_encoder)]}
                        if not all(torch.isfinite(torch.tensor(v)) for v in gradients[term].values()):
                            raise RuntimeError('nonfinite auxiliary module gradient')
                        if gradients[term]['dit']<=0 or gradients[term]['decoder']<=0:
                            raise RuntimeError(f'{term} lacks DiT/decoder gradient on real source sample')
                total=flow+sum(losses.values()); total.backward()
                if any(p.grad is not None for p in model.target_teacher.parameters()):
                    raise RuntimeError('teacher gradient leak')
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise RuntimeError('nonfinite actual-source backward')
                model.zero_grad(set_to_none=True)
                evidence['checks'].append({'sample_id':spec.sample_id,'amp':amp,'raw':{n:float(x.detach()) for n,x in losses.items()},'module_gradients':gradients,**diag})
                del generated,losses,total,flow
            print(json.dumps({'smoke_trajectory':index+1,'sample_id':spec.sample_id}),flush=True)
        # Generation output/condition must remain equal after actual future-coordinate mutation.
        mutated=replace(cpu,x=cpu.x.clone(),bpos=cpu.bpos.clone())
        mutation=torch.randn(cpu.x[4:].shape,generator=torch.Generator().manual_seed(17))*10.0
        mutated.x[4:]+=mutation; mutated.bpos[4:]+=mutation
        changed=prepare_frame_joint_batch(model.target_teacher,mutated,device=device,normalizer=model,history_frames=4)
        with torch.no_grad():
            first,_=forward_samples(model,batch,seed=4321,steps=4)
            second,_=forward_samples(model,changed,seed=4321,steps=4)
        delta=max(float((a.coordinates-b.coordinates).abs().max()) for a,b in zip(first.sampled,second.sampled))
        if delta!=0: raise RuntimeError(f'future leak: {delta}')
        evidence['future_mutation_max_abs']=delta
        del first,second,changed
        # At rest the RMSF norm has a finite zero subgradient. Test motion groups
        # independently; clash/band are disabled and should not be calculated.
        static=batch.observed_context.coordinates[-1:].expand(batch.query.frame_mask.shape[1],-1,-1).clone().requires_grad_()
        motion=replace(cfg,clash_weight=0.,bond_band_weight=0.)
        zero_batch=replace(batch,target_coordinates=static.detach())
        losses,counts,_=physical_auxiliary_loss((static,static),zero_batch,motion)
        sum(losses.values()).backward()
        if not torch.isfinite(static.grad).all(): raise RuntimeError('zero-motion derivative is not finite')
        evidence['zero_motion_finite_gradient']=True
        # Same real CUDA update after checkpoint must reproduce both weights and RNG.
        trainer=FrameJointTrainer(model,loss_config=JointLossConfig.resolve(loaded['payload']['contracts']['loss']),
            learning_rates={'history_encoder':1e-4,'dit':2e-4,'decoder':5e-5},stage='continuation',amp=True,
            data_hash=data_hash,teacher_artifact_sha256=raw['codec']['sha256'],schedule_contract={'smoke':'explicit_aux_mask'},
            sampled_config=SampledAuxiliaryConfig.resolve({'enabled':True,'physical':cfg.contract()}))
        trainer.load_warm_start(PARENT,expected_sha256=PARENT_SHA)
        main=torch.Generator(device=device).manual_seed(101); aux=torch.Generator(device=device).manual_seed(102)
        trainer.train_step(batch,generator=main,sampled_generator=aux,auxiliary_active=False)
        cursor={'batch_index':1,'auxiliary_mask':[False,True],'history_frames':[4,4]}
        checkpoint=args.output/'smoke_step1.pt'
        trainer.save_checkpoint(checkpoint,cursor=cursor,generator=main,sampled_generator=aux)
        result1=trainer.train_step(batch,generator=main,sampled_generator=aux,auxiliary_active=True)
        reference_hash=module_state_hash(model); main_state=main.get_state().clone(); aux_state=aux.get_state().clone()
        restored=trainer.load_checkpoint(checkpoint,generator=main,sampled_generator=aux)
        result2=trainer.train_step(batch,generator=main,sampled_generator=aux,auxiliary_active=restored['cursor']['auxiliary_mask'][1])
        exact=(reference_hash==module_state_hash(model) and torch.equal(main_state,main.get_state()) and torch.equal(aux_state,aux.get_state()) and result1==result2)
        if not exact: raise RuntimeError('CUDA resumed update differs')
        evidence['exact_resume']=True
        evidence['teacher_unchanged']=teacher_before==module_state_hash(model.target_teacher)
        evidence['passed']=True
        evidence['gpu_hours']=(time.time()-started)/3600
        atomic_write_json(args.output/'smoke.json',evidence)
    finally:
        atomic_write_json(args.output/'smoke_partial.json',evidence)
        close()


if __name__=='__main__': main()
