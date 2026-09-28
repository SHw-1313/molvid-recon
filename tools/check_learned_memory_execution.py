"""CUDA-only observed generation and differentiable prepared-prefix checks."""
from __future__ import annotations
import argparse
import atexit
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
import torch
from molvid.checkpoints import load_codec_artifact
from molvid.data.raw_history import RawHistoryWindow,load_pilot_inputs
from molvid.flow.source import sample_frame_source
from molvid.generation import sample_learned_memory
from molvid.latent.statistics import FrameLatentStatistics
from molvid.latent.types import QuerySpec
from molvid.model import FrameJointModel
from molvid.runtime import sha256_file
from molvid.training.batches import prepare_frame_joint_batch


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpu-authorization',type=Path,required=True)
    a=p.parse_args()
    config=json.loads(a.config.read_text()); authorization=json.loads(a.gpu_authorization.read_text())
    if authorization.get('lane')!='B' or authorization.get('mode')!='user_authorized_shared' or os.environ.get('CUDA_VISIBLE_DEVICES')!=authorization['gpu_uuid']:
        raise ValueError('explicit B GPU authorization required')
    lock=a.gpu_authorization.with_name('B_gpu.lock').open('a'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    a.output.mkdir(parents=True,exist_ok=False); start=time.perf_counter()
    def cost():
        (a.output/'cost_ledger.json').write_text(json.dumps({'gpu_hours_including_setup_checks':(time.perf_counter()-start)/3600,'mode':'execution_checks','gpu_uuid':authorization['gpu_uuid']},indent=2))
    atexit.register(cost)
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8': raise ValueError('deterministic cuBLAS configuration required')
    torch.use_deterministic_algorithms(True); torch.backends.cuda.matmul.allow_tf32=False
    device=torch.device('cuda'); torch.manual_seed(20260928)
    for key in ('teacher','statistics'):
        if sha256_file(config[key]['path'])!=config[key]['sha256']: raise ValueError(f'{key} identity mismatch')
    artifact=load_codec_artifact(config['teacher']['path'],expected_sha256=config['teacher']['sha256'],device=device)
    stats=FrameLatentStatistics.from_state_dict(torch.load(config['statistics']['path'],map_location='cpu',weights_only=False))
    if stats.hash!=config['statistics']['semantic_hash']: raise ValueError('statistics mismatch')
    model=FrameJointModel.from_codec(artifact.model,stats.to(device),geometry_enabled=True,motion_enabled=False,history_backend='learned_memory').to(device)
    payload=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    if payload['scope']!='pilot_only_not_formal_parent': raise ValueError('expected bounded pilot checkpoint')
    model.load_state_dict(payload['model_state'],strict=True); del artifact,payload
    bank=next(item for item in config['banks'] if '1by2_A_R1' in item['path'])
    if sha256_file(bank['path'])!=bank['sha256']: raise ValueError('bank identity mismatch')
    raw,anchors,future=load_pilot_inputs(Path(bank['path']),RawHistoryWindow(1280,12800,400,100,'train'))
    counts={'memory':0,'teacher_frames':0,'dit':0}
    def memory_hook(*_): counts['memory']+=1
    def teacher_hook(_,args): counts['teacher_frames']+=args[0].frames
    def dit_hook(*_): counts['dit']+=1
    hooks=[model.history_encoder.register_forward_pre_hook(memory_hook),model.target_teacher.frame_encoder.register_forward_pre_hook(teacher_hook),model.dit.register_forward_pre_hook(dit_hook)]
    query=QuerySpec(future.time_ps,future.frame_mask)
    model.eval()
    generated,metadata=sample_learned_memory(model,raw_history=raw,geometry_anchors=anchors,query=query,steps=16,seed=0)
    if counts!={'memory':1,'teacher_frames':2,'dit':16}: raise AssertionError(counts)
    if not torch.isfinite(generated).all(): raise AssertionError('nonfinite observed-only generation')
    inference=dict(counts)
    # Future truth is absent from sample_learned_memory's interface; mutate the
    # held-out object and rerun the exact same observed context/noise.
    future.x.add_(1000)
    counts.update(memory=0,teacher_frames=0,dit=0)
    repeated,_=sample_learned_memory(model,raw_history=raw,geometry_anchors=anchors,query=query,steps=16,seed=0)
    torch.testing.assert_close(generated,repeated,atol=0,rtol=0)
    # Reload unmodified targets for the training-only preparation boundary.
    raw,anchors,future=load_pilot_inputs(Path(bank['path']),RawHistoryWindow(1280,12800,400,100,'train'))
    batch=prepare_frame_joint_batch(model.target_teacher,future,device=device,normalizer=model,history_frames=2,raw_history=raw,geometry_anchors=anchors)
    counts.update(memory=0,teacher_frames=0,dit=0)
    rng=torch.Generator(device=device).manual_seed(92803)
    sources=tuple(sample_frame_source(batch.source_center,generator=rng)[0] for _ in range(2))
    model.train(); model.zero_grad(set_to_none=True)
    with torch.autocast('cuda',dtype=torch.bfloat16):
        result=model(sources[0],context=batch.observed_context,query=batch.query,flow_time=torch.zeros(1,device=device),sampled_sources=sources,sampled_steps=4)
        # Numerical path probe only, no optimizer step or scientific auxiliary objective.
        probe=sum(item.coordinates.float().square().mean() for item in result.sampled)
    probe.backward()
    if counts!={'memory':1,'teacher_frames':0,'dit':9}: raise AssertionError(counts)
    norms={}
    for prefix in ('light_encoder.','queries','label_key.','scalar_output.','vector_output.'):
        gradients=[param.grad.float().square().sum() for name,param in model.history_encoder.named_parameters() if name.startswith(prefix) and param.grad is not None]
        norm=torch.stack(gradients).sum().sqrt()
        if not torch.isfinite(norm) or norm<=0: raise AssertionError(prefix)
        norms[prefix]=float(norm)
    if any(param.grad is not None for param in model.target_teacher.parameters()): raise AssertionError('teacher gradient leak')
    for hook in hooks: hook.remove()
    torch.cuda.synchronize()
    (a.output/'checks.json').write_text(json.dumps({'status':'pass','code_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'config_sha256':sha256_file(a.config),'pilot_checkpoint_sha256':sha256_file(a.checkpoint),'inference':inference,'inference_steps':16,
        'future_mutation_exact':True,'training_actual_source_calls':counts,'draws':2,'euler_steps':4,'memory_gradient_norms':norms,
        'teacher_gradients':False,'optimizer_updates':0,'scope':'numerical path check; no A-recipe or motion-quality conclusion'},indent=2))
    torch.save({'generated_coordinates':generated.cpu(),'query_time_ps':query.time_ps,'scope':'single training-system numerical replay, not evaluation'},a.output/'observed_generation.pt')


if __name__=='__main__': main()
