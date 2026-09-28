"""Bounded B pilot using FrameJointTrainer; never launches formal comparison."""
from __future__ import annotations
import argparse
from dataclasses import replace
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
import torch
from molvid.checkpoints import load_codec_artifact
from molvid.codec.frame import slice_clip_frames
from molvid.data.raw_history import RawHistoryWindow, load_pilot_inputs
from molvid.latent.statistics import FrameLatentStatistics
from molvid.model import FrameJointModel
from molvid.runtime import sha256_file
from molvid.training.batches import prepare_frame_joint_batch, prepare_batch_then_to_device
from molvid.training.joint import FrameJointTrainer, JointLossConfig
from molvid.training.dit import module_state_hash


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args()
    config=json.loads(args.config.read_text())
    if config['scope']!='pilot_only' or not 1 <= config['updates'] <= 1000:
        raise ValueError('only bounded 1..1000 update pilot is permitted')
    resources=json.loads(Path(config['resources']).read_text())
    assignment=resources['assigned_gpu_uuids']['B']
    if len(assignment)!=1 or os.environ.get('CUDA_VISIBLE_DEVICES')!=assignment[0]:
        raise RuntimeError('B requires one assigned UUID and matching CUDA_VISIBLE_DEVICES; no GPU claimed')
    lock=Path(config['resources']).with_name('B_gpu.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    if any(assignment[0] in row for row in processes.splitlines()):
        raise RuntimeError('assigned B GPU has an existing compute process; do not share or preempt')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    device=torch.device('cuda')
    gpu_start=time.perf_counter()
    output=Path(config['output']); output.mkdir(parents=True,exist_ok=False)
    (output/'resolved_experiment.json').write_text(json.dumps(config,indent=2))
    for name in ('teacher','statistics','parent'):
        if sha256_file(config[name]['path'])!=config[name]['sha256']:
            raise ValueError(f'{name} identity differs')
    for entry in config['banks']+config['profile_banks']:
        if sha256_file(entry['path'])!=entry['sha256']:
            raise ValueError('raw bank identity differs')
    torch.manual_seed(config['seed']); torch.cuda.manual_seed_all(config['seed'])
    artifact=load_codec_artifact(config['teacher']['path'],expected_sha256=config['teacher']['sha256'],device=device)
    stats=FrameLatentStatistics.from_state_dict(torch.load(config['statistics']['path'],map_location='cpu',weights_only=False))
    if stats.hash!=config['statistics']['semantic_hash']:
        raise ValueError('statistics semantic hash differs')
    if stats.provenance.get('codec_checkpoint_sha256')!=config['teacher']['sha256'] or stats.provenance.get('data_hash')!=config['normalization_source_data_hash']:
        raise ValueError('statistics provenance differs')
    model=FrameJointModel.from_codec(artifact.model,stats.to(device),geometry_enabled=True,motion_enabled=False,
                                    history_backend='learned_memory',analysis_mode=config['analysis_mode']).to(device)
    del artifact
    parent=torch.load(config['parent']['path'],map_location='cpu',weights_only=False)
    loss_config=JointLossConfig.resolve(parent['contracts']['loss'])
    del parent
    trainer=FrameJointTrainer(model,loss_config=loss_config,learning_rates=config['learning_rates'],amp=True,
                              teacher_artifact_sha256=config['teacher']['sha256'],data_hash=config['raw_data_card_sha256'])
    report=trainer.load_warm_start(config['parent']['path'],expected_sha256=config['parent']['sha256'])
    teacher_hash_before=module_state_hash(model.target_teacher)
    (output/'load_report.json').write_text(json.dumps(report.as_dict(),indent=2))
    (output/'loss_manifest.json').write_text(json.dumps({'scope':'base_loss_only_pilot','weights':loss_config.contract(),
        'flow':1.,'flow_h_fraction':.5,'flow_v_fraction':.5,'actual_source_auxiliary':'not enabled before evaluated A recipe'},indent=2))
    trainable={id(p) for p in model.parameters() if p.requires_grad}
    optimizer={id(p) for g in trainer.optimizer.param_groups for p in g['params']}
    if trainable!=optimizer:
        raise AssertionError('optimizer does not own every trainable parameter exactly')
    (output/'model_card.json').write_text(json.dumps({'contract':model.contract(),
        'parameters':sum(p.numel() for p in model.parameters()),'trainable':sum(p.numel() for p in model.parameters() if p.requires_grad),
        'optimizer_groups':[{k:v for k,v in g.items() if k!='params'} for g in trainer.optimizer.param_groups]},indent=2))
    frame_calls=[]
    handle=model.target_teacher.frame_encoder.register_forward_pre_hook(lambda _,args: frame_calls.append(args[0].frames))
    raw,anchors,future=load_pilot_inputs(Path(config['banks'][0]['path']),RawHistoryWindow(1280,800,100,100,'train'))
    prepared=prepare_frame_joint_batch(model.target_teacher,future,device=device,normalizer=model,history_frames=2,raw_history=raw,geometry_anchors=anchors)
    if sum(frame_calls)!=14 or frame_calls[0]!=2:
        raise AssertionError(f'teacher frame budget differs: {frame_calls}')
    # Real teacher same-origin split parity, not a stub encoder comparison.
    joined=replace(anchors,x=torch.cat((anchors.x,future.x)),bpos=torch.cat((anchors.bpos,future.bpos)),
                   frame_mask=torch.cat((anchors.frame_mask,future.frame_mask),1),time_ps=torch.cat((anchors.time_ps,future.time_ps),1),
                   delta_time_ps=torch.diff(torch.cat((anchors.time_ps,future.time_ps),1),dim=1))
    full,_=model.target_teacher(prepare_batch_then_to_device(model.target_teacher,joined,device))
    torch.testing.assert_close(prepared.observed_context.latent.h,full.h[:2],atol=1e-5,rtol=1e-5)
    torch.testing.assert_close(prepared.observed_context.latent.v,full.v[:2],atol=1e-5,rtol=1e-5)
    torch.testing.assert_close(prepared.target_future.h,full.h[2:],atol=1e-5,rtol=1e-5)
    torch.testing.assert_close(prepared.target_future.v,full.v[2:],atol=1e-5,rtol=1e-5)
    mutated=replace(future,x=future.x+torch.randn_like(future.x)*2)
    other=prepare_frame_joint_batch(model.target_teacher,mutated,device=device,normalizer=model,history_frames=2,raw_history=raw,geometry_anchors=anchors)
    torch.testing.assert_close(prepared.observed_context.latent.h,other.observed_context.latent.h,atol=0,rtol=0)
    torch.testing.assert_close(prepared.observed_context.sample_origin,other.observed_context.sample_origin,atol=0,rtol=0)
    torch.testing.assert_close(prepared.source_center.v,other.source_center.v,atol=0,rtol=0)
    (output/'teacher_checks.json').write_text(json.dumps({'same_origin_split_parity':'pass','future_mutation':'pass',
        'observed_frames_per_prefix':2,'training_target_frames':12,'comparison_atol':1e-5,'comparison_rtol':1e-5},indent=2))
    del full,prepared,other
    generator=torch.Generator(device=device).manual_seed(config['seed']+1)
    start=time.perf_counter(); frame_calls.clear(); total_atom_frames=0
    # Module timings use CUDA events without synchronizing every submodule.
    timings={}; pending={}; hooks=[]
    modules={'light_encode':model.history_encoder.light_encoder,'time_analysis':model.history_encoder.analysis,
             'memory_inclusive':model.history_encoder,'dit':model.dit,'decoder':model.decoder,
             'teacher_geometry_inclusive':model.target_teacher.frame_encoder}
    for name,module in modules.items():
        def before(_,args,name=name):
            event=torch.cuda.Event(enable_timing=True); event.record(); pending[name]=event
        def after(_,args,result,name=name):
            event=torch.cuda.Event(enable_timing=True); event.record(); timings.setdefault(name,[]).append((pending.pop(name),event))
        hooks.extend((module.register_forward_pre_hook(before),module.register_forward_hook(after)))
    with (output/'train_metrics.jsonl').open('w') as stream:
        for step in range(config['updates']):
            # A pilot crossing spans/dt/lag independently, not a formal exposure scheduler.
            entry=config['banks'][step%len(config['banks'])]
            span=(800,3200,12800)[(step//len(config['banks']))%3]
            dt=(100,200,400)[(step//(len(config['banks'])*3))%3]
            lag=(100,200,400)[(step//(len(config['banks'])*9))%3]
            io_start=time.perf_counter()
            raw,anchors,future=load_pilot_inputs(Path(entry['path']),RawHistoryWindow(1280,span,dt,lag,'train'))
            io_seconds=time.perf_counter()-io_start
            torch.cuda.reset_peak_memory_stats(); before_calls=len(frame_calls); step_start=time.perf_counter()
            batch=prepare_frame_joint_batch(model.target_teacher,future,device=device,normalizer=model,history_frames=2,raw_history=raw,geometry_anchors=anchors)
            metrics=trainer.train_step(batch,generator=generator,auxiliary_active=False)
            torch.cuda.synchronize()
            if sum(frame_calls[before_calls:])!=14:
                raise AssertionError('teacher frame count changed with history length')
            missing=[]
            for name,p in model.history_encoder.named_parameters():
                if p.grad is None or not torch.isfinite(p.grad).all() or p.grad.abs().sum()==0:
                    missing.append(name)
            if missing:
                raise AssertionError(f'new memory parameters lack finite nonzero gradients: {missing}')
            elapsed=time.perf_counter()-step_start
            total_atom_frames+=(raw.coordinates.shape[0]+14)*anchors.atom_count
            metrics.update(history_span_ps=span,observed_dt_ps=dt,query_lag_ps=lag,raw_io_seconds=io_seconds,
                           step_seconds=elapsed,peak_allocated_bytes=torch.cuda.max_memory_allocated(),teacher_frames=14,
                           module_ms={k:sum(a.elapsed_time(b) for a,b in pairs) for k,pairs in timings.items()})
            timings.clear()
            stream.write(json.dumps(metrics)+'\n'); stream.flush()
    elapsed=time.perf_counter()-start
    for hook in hooks: hook.remove()
    handle.remove()
    if module_state_hash(model.target_teacher)!=teacher_hash_before:
        raise AssertionError('frozen teacher parameters changed')
    (output/'train_manifest.json').write_text(json.dumps({'status':'pilot_completed','successful_updates':trainer.successful_updates,
        'training_seconds':elapsed,'gpu_hours_including_setup_checks':(time.perf_counter()-gpu_start)/3600,'effective_atom_frames_per_second':total_atom_frames/elapsed,
        'teacher_observed_frames_per_prefix':2,'teacher_target_frames_per_prefix':12,
        'formal':False,'profile_banks_training':False,'size_profile':'separate profiling still required'},indent=2))
    torch.save({'model_state':model.state_dict(),'scope':'pilot_only_not_formal_parent'},output/'pilot_model.pt')


if __name__=='__main__':
    main()
