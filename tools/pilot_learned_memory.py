"""Bounded B pilot using FrameJointTrainer; never launches formal comparison."""
from __future__ import annotations
import argparse
import atexit
from dataclasses import replace
from contextlib import nullcontext
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
    parser.add_argument('--code-commit',required=True)
    parser.add_argument('--profile-only',action='store_true')
    parser.add_argument('--gpu-authorization',type=Path)
    args=parser.parse_args()
    config=json.loads(args.config.read_text())
    actual_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    if actual_commit!=args.code_commit:
        raise ValueError('executing code commit differs from the requested fixed candidate')
    subprocess.run(['git','diff','--exit-code','HEAD','--','molvid','tools','tests'],check=True,stdout=subprocess.DEVNULL)
    if config['scope']!='pilot_only' or not 1 <= config['updates'] <= 1000:
        raise ValueError('only bounded 1..1000 update pilot is permitted')
    resources=json.loads(Path(config['resources']).read_text())
    assignment=resources['assigned_gpu_uuids']['B']
    shared_authorization=None
    if args.gpu_authorization is not None:
        shared_authorization=json.loads(args.gpu_authorization.read_text())
        if shared_authorization.get('lane')!='B' or shared_authorization.get('mode')!='user_authorized_shared':
            raise ValueError('invalid explicit B GPU sharing authorization')
        assignment=[shared_authorization['gpu_uuid']]
    if len(assignment)!=1 or os.environ.get('CUDA_VISIBLE_DEVICES')!=assignment[0]:
        raise RuntimeError('B requires one assigned UUID and matching CUDA_VISIBLE_DEVICES; no GPU claimed')
    lock=Path(config['resources']).with_name('B_gpu.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    if shared_authorization is None and any(assignment[0] in row for row in processes.splitlines()):
        raise RuntimeError('assigned B GPU has an existing compute process; do not share or preempt')
    previous_cost=sum(float(json.loads(p.read_text())['gpu_hours_including_setup_checks'])
                      for p in Path(config['budget_root']).glob('*/cost_ledger.json'))
    remaining_hours=min(float(config['max_gpu_hours']),float(config['pilot_budget_gpu_hours'])-previous_cost)
    if remaining_hours<=0:
        raise RuntimeError('B aggregate pilot GPU-hour budget exhausted')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    device=torch.device('cuda')
    gpu_start=time.perf_counter()
    output=Path(config['output'] + ('_profile' if args.profile_only else '')); output.mkdir(parents=True,exist_ok=False)
    def cost_record():
        (output/'cost_ledger.json').write_text(json.dumps({'gpu_uuid':assignment[0], 'gpu_hours_including_setup_checks':(time.perf_counter()-gpu_start)/3600,
             'code_commit':actual_commit,'config_sha256':sha256_file(args.config),'mode':'profile' if args.profile_only else 'pilot'},indent=2))
    atexit.register(cost_record)
    (output/'code_identity.json').write_text(json.dumps({'code_commit':actual_commit,'config_sha256':sha256_file(args.config),'profile_only':args.profile_only},indent=2))
    (output/'resolved_experiment.json').write_text(json.dumps(config,indent=2))
    (output/'gpu_authorization.json').write_text(json.dumps({'explicit_sharing':shared_authorization,'gpu_uuid':assignment[0],
        'preexisting_processes':processes.splitlines()},indent=2))
    for name in ('teacher','statistics','parent'):
        if sha256_file(config[name]['path'])!=config[name]['sha256']:
            raise ValueError(f'{name} identity differs')
    for entry in config['banks']+config['profile_banks']:
        if sha256_file(entry['path'])!=entry['sha256']:
            raise ValueError('raw bank identity differs')
    if sha256_file(config['raw_data_card'])!=config['raw_data_card_sha256']:
        raise ValueError('raw data card identity differs')
    torch.manual_seed(config['seed']); torch.cuda.manual_seed_all(config['seed'])
    if config.get('deterministic',False):
        if os.environ.get('CUBLAS_WORKSPACE_CONFIG') not in (':4096:8', ':16:8'):
            raise ValueError('deterministic CUDA pilot requires CUBLAS_WORKSPACE_CONFIG')
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
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
    full,_=model.target_teacher(prepare_batch_then_to_device(model.target_teacher,joined,device),
                               reference_origin=prepared.observed_context.sample_origin)
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
             'teacher_geometry_inclusive':model.target_teacher.frame_encoder,
             'teacher_backbone':model.target_teacher.frame_encoder.backbone,
             'teacher_all':model.target_teacher}
    for name,module in modules.items():
        def before(_,args,name=name):
            event=torch.cuda.Event(enable_timing=True); event.record(); pending[name]=event
            if name=='teacher_backbone':
                timings.setdefault('teacher_graph_and_pack',[]).append((pending['teacher_geometry_inclusive'],event))
        def after(_,args,result,name=name):
            event=torch.cuda.Event(enable_timing=True); event.record()
            label=('teacher_observed' if args[0].frames==2 else 'teacher_target') if name=='teacher_all' else name
            timings.setdefault(label,[]).append((pending.pop(name),event))
        hooks.extend((module.register_forward_pre_hook(before),module.register_forward_hook(after)))
    profile_entries=[(entry,span,dt,lag) for entry in config['profile_banks'] for span,dt,lag in
                     ((800,100,100),(3200,100,200),(12800,100,400),(12800,400,100))]
    update_count=len(profile_entries) if args.profile_only else config['updates']
    previous_seconds=0.
    with (output/'train_metrics.jsonl').open('w') as stream:
        for step in range(update_count):
            if time.perf_counter()-gpu_start+previous_seconds > remaining_hours*3600:
                raise RuntimeError('bounded pilot budget reached; cost ledger preserved, no next step launched')
            # A pilot crossing spans/dt/lag independently, not a formal exposure scheduler.
            entry=config['banks'][step%len(config['banks'])]
            span=(800,3200,12800)[(step//len(config['banks']))%3]
            dt=(100,200,400)[(step//(len(config['banks'])*3))%3]
            lag=(100,200,400)[(step//(len(config['banks'])*9))%3]
            if args.profile_only:
                entry,span,dt,lag=profile_entries[step]
            io_start=time.perf_counter()
            raw,anchors,future=load_pilot_inputs(Path(entry['path']),RawHistoryWindow(1280,span,dt,lag,'train'))
            io_seconds=time.perf_counter()-io_start
            torch.cuda.reset_peak_memory_stats(); before_calls=len(frame_calls); step_start=time.perf_counter()
            profiler=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) if args.profile_only else nullcontext()
            with profiler as trace:
                with torch.profiler.record_function('B.teacher_and_transfer'):
                    batch=prepare_frame_joint_batch(model.target_teacher,future,device=device,normalizer=model,history_frames=2,raw_history=raw,geometry_anchors=anchors)
                metrics=trainer.train_step(batch,generator=generator,auxiliary_active=False)
            torch.cuda.synchronize()
            if sum(frame_calls[before_calls:])!=14:
                raise AssertionError('teacher frame count changed with history length')
            missing=[]
            for name,p in model.history_encoder.named_parameters():
                if p.grad is None or not torch.isfinite(p.grad).all():
                    missing.append(name)
            if missing:
                raise AssertionError(f'new memory parameters lack finite gradients: {missing}')
            path_norms={}
            for path in ('light_encoder.', 'key.', 'label_key.', 'queries', 'scalar_value.', 'vector_value.', 'scalar_output.', 'vector_output.'):
                norm=sum(p.grad.float().square().sum() for name,p in model.history_encoder.named_parameters() if name.startswith(path)).sqrt()
                if norm==0:
                    raise AssertionError(f'learned path lacks nonzero gradient: {path}')
                path_norms[path]=float(norm)
            elapsed=time.perf_counter()-step_start
            previous_seconds=elapsed
            total_atom_frames+=(raw.coordinates.shape[0]+14)*anchors.atom_count
            metrics.update(history_span_ps=span,observed_dt_ps=dt,query_lag_ps=lag,raw_io_seconds=io_seconds,
                           step_seconds=elapsed,peak_allocated_bytes=torch.cuda.max_memory_allocated(),teacher_frames=14,
                           memory_gradient_path_norms=path_norms,
                           module_ms={k:sum(a.elapsed_time(b) for a,b in pairs) for k,pairs in timings.items()})
            metrics['module_ms']['memory_slots_and_readout']=metrics['module_ms'].get('memory_inclusive',0)-metrics['module_ms'].get('light_encode',0)-metrics['module_ms'].get('time_analysis',0)
            if args.profile_only:
                trace.export_chrome_trace(str(output/f'profile_{step:03d}.json'))
                metrics['profile_ranges']={event.key:{'cpu_us':event.cpu_time_total,'device_us':event.device_time_total}
                    for event in trace.key_averages() if event.key in ('frame_joint.backward','B.teacher_and_transfer')}
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
        'formal':False,'profile_banks_training':args.profile_only,
        'size_profile':'cold train-step profiles complete; mixed-dt packing and warm timing remain' if args.profile_only else 'run --profile-only separately',
        'module_timings':'inclusive; graph_and_pack is encoder entry to backbone entry; do not sum overlapping ranges'},indent=2))
    torch.save({'model_state':model.state_dict(),'scope':'pilot_only_not_formal_parent'},output/'pilot_model.pt')


if __name__=='__main__':
    main()
