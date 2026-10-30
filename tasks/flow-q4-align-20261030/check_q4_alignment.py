"""Targeted real TRAIN optimizer/resume, Q4 split and auxiliary checks."""
import gc,json,os,time
from pathlib import Path
import numpy as np
import torch
from molvid.training.q4_align_stage import train_q4_align,joint_q4_chunks,backward_slot,backward_q4_auxiliary
from molvid.training.coordinate_state import load_coordinate_weights,LigandSourceGeometry
from molvid.flow.coordinate_state import CoordinateStateDenoiser
from molvid.data.bridge_mixture import BridgeMixture
from molvid.codec.frame import slice_clip_frames
from molvid.data.batch import collate_clip_records
ROOT=Path(__file__).resolve().parents[2];TASK=Path(__file__).resolve().parent;RUN=ROOT/'runs/flow_q4_align_20261030'
PARENT=Path('/workspace/MOLVID-FLOW-LIGAND-4096-20261026/runs/flow_ligand_4096_20261026/ligand/coordinate_final.pt')
ARMS=('shared_joint_only','shared_q4_q1','routed_q4_q1')
def main():
 if os.environ.get('CONDA_DEFAULT_ENV')!='torch-ito' or not os.environ.get('CUDA_VISIBLE_DEVICES'):raise RuntimeError('Explicit environment/GPU required')
 torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(32*1024**3/torch.cuda.get_device_properties(0).total_memory)
 out=RUN/'checks'/'cuda';out.mkdir(exist_ok=False);result=dict(status='running',GPU=os.environ['CUDA_VISIBLE_DEVICES'],stages=[],discarded_check_models=True)
 def save():(out/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
 save();cfgs={a:json.loads((TASK/'configs'/f'{a}.json').read_text()) for a in ARMS};states=[]
 for a in ARMS:
  torch.manual_seed(71);m=CoordinateStateDenoiser(cfgs[a]);states.append({k:v.detach().clone() for k,v in m.state_dict().items()});del m
 assert all(all(torch.equal(v,s[k]) for k,v in states[0].items()) for s in states[1:]);del states;gc.collect();result['initial_state_exact']=True;save()
 old=[json.loads(s) for s in Path('/workspace/MOLVID-FLOW-TASK-ROUTE-20261029/runs/flow_task_route_20261029/shared/updates.jsonl').read_text().splitlines()]
 for arm in ARMS:
  before=time.monotonic();train_q4_align(cfgs[arm],out/arm,71,warm_start=PARENT,stop_after=2);torch.cuda.synchronize()
  rows=[json.loads(s) for s in (out/arm/'updates.jsonl').read_text().splitlines()];state=torch.load(out/arm/'checkpoint_latest.pt',map_location='cpu',weights_only=False)
  assert state['step']==state['slot']==2 and state['optimizer_updates']==(1 if arm=='shared_joint_only' else 2)
  for row in rows:
   if not row['optimized']:continue
   assert np.isfinite(row['grad_norm']) and row['grad_norm']>0
   expected=old[row['slot']-1]
   for key in ('sample','history','queries','source_epsilon_sum','source_epsilon_square_sum','progress_mean','learning_rate'):assert row[key]==expected[key],(arm,key,row[key],expected[key])
  if arm=='shared_joint_only':assert abs(rows[0]['RF_loss']-old[0]['RF_loss'])<1e-4,(rows[0]['RF_loss'],old[0]['RF_loss'])
  result['stages'].append(dict(arm=arm,kind='actual_two_slot_training',seconds=time.monotonic()-before,optimizer_updates=state['optimizer_updates'],RF_loss=rows[0]['RF_loss'],grad_norm=rows[0]['grad_norm'],source_progress_LR_exact=True));save();del state;gc.collect();torch.cuda.empty_cache()
 before=time.monotonic();arm='shared_q4_q1';train_q4_align(cfgs[arm],out/arm,71,resume=out/arm/'checkpoint_latest.pt',stop_after=4);torch.cuda.synchronize()
 state=torch.load(out/arm/'checkpoint_latest.pt',map_location='cpu',weights_only=False);assert state['slot']==4 and state['optimizer_updates']==4
 result['stages'].append(dict(kind='exact_resume_to_slot4',seconds=time.monotonic()-before,optimizer_updates=4));save();del state;gc.collect();torch.cuda.empty_cache()
 maximum=max((r for r in old if r['kernel_task']=='joint'),key=lambda r:r['valid_atoms']);ordinal=maximum['task_ordinal'];mix=BridgeMixture(cfgs['routed_q4_q1']['data'],71)
 for i in range(ordinal):batch,task,_,_,identity=mix.next()
 assert batch.sample_id[0]==maximum['sample'];mix.close();batch=batch.to('cuda');history=maximum['history'];queries=maximum['queries']
 torch.manual_seed(71);model=CoordinateStateDenoiser(cfgs['routed_q4_q1']).cuda().train();load_coordinate_weights(model,PARENT)
 parameters=[p for p in model.parameters() if p.requires_grad];aux=LigandSourceGeometry(cfgs['routed_q4_q1']['ligand_geometry'],71,torch.device('cuda'));generator=torch.Generator(device='cuda').manual_seed(71+20261008)
 torch.cuda.reset_peak_memory_stats();before=time.monotonic();metrics=backward_slot(model,batch,history,queries,arm='routed_q4_q1',kind='joint',identity=identity,band_reference=None,generator=generator,auxiliary=aux,bf16=True)
 norm=torch.nn.utils.clip_grad_norm_(parameters,1,error_if_nonfinite=True);torch.cuda.synchronize();assert norm>0
 assert all(p.grad is None for n,p in model.named_parameters() if not n.startswith('task_adapters.'))
 result['stages'].append(dict(kind='maximum_real_joint_Q4_backward',sample=batch.sample_id[0],atoms=batch.atom_count,original_H=history,target_frames=len(queries),seconds=time.monotonic()-before,RF_loss=metrics['RF_loss'],grad_norm=float(norm),peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved()));save()
 del batch;model.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()
 payload=torch.load('/workspace/MOLVID-FLOW-MEMORY-20261029/runs/flow_memory_20261029/data_smoke_fixed/window_006.pt',map_location='cpu',weights_only=False)
 small=slice_clip_frames(collate_clip_records([payload['record']]),0,16).to('cuda')
 for H in (4,8):
  chunks=list(joint_q4_chunks(small,H,list(range(H,16))))
  assert torch.equal(torch.cat([c[1] for c in chunks]),small.x[H:]);assert torch.equal(torch.cat([c[2] for c in chunks],dim=1),small.time_ps[:,H:]);assert abs(sum(len(c[1])/(16-H) for c in chunks)-1)<1e-12
 private=aux.generator.get_state().clone();replica=torch.Generator(device='cuda');replica.set_state(private);torch.randn(small.x[4:].shape,device='cuda',generator=replica)
 before=time.monotonic();stats=backward_q4_auxiliary(model,small,4,list(range(4,16)),aux,bf16=True);norm=torch.nn.utils.clip_grad_norm_(parameters,1,error_if_nonfinite=True);torch.cuda.synchronize()
 assert norm>0 and stats['ligand_geometry_events']==1 and stats['geometry_band_frames']==12 and torch.equal(aux.generator.get_state(),replica.get_state())
 assert all(p.grad is None for n,p in model.named_parameters() if not n.startswith('task_adapters.'))
 result['stages'].append(dict(kind='real_ligand_three_Q4_native_aux',seconds=time.monotonic()-before,weighted_loss=stats['ligand_geometry_weighted_loss'],grad_norm=float(norm),normal_band_frames=12,one_event=True,full_window_noise_consumption_exact=True,chunk_weights=[r['weight'] for r in stats['ligand_geometry_chunks']]));result['status']='completed';save();print(json.dumps(result),flush=True)
if __name__=='__main__':main()
