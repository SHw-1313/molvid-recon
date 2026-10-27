"""CPU read of actual common task exposure/RNG checkpoints; no model execution."""
import argparse,json,copy,time
from pathlib import Path
import torch
import numpy as np
ROOT=Path(__file__).resolve().parents[2];RUN=ROOT/'runs/flow_kernel_20261027'
p=argparse.ArgumentParser();p.add_argument('--step',type=int,choices=(32,512),default=32);a=p.parse_args()
def equal(x,y):
 if isinstance(x,torch.Tensor):return isinstance(y,torch.Tensor) and torch.equal(x,y)
 if isinstance(x,np.ndarray):return isinstance(y,np.ndarray) and np.array_equal(x,y)
 if isinstance(x,dict):return x.keys()==y.keys() and all(equal(x[k],y[k]) for k in x)
 if isinstance(x,(list,tuple)):return len(x)==len(y) and all(equal(p,q) for p,q in zip(x,y))
 return x==y
began=time.monotonic();result=dict(step=a.step,arms={},errors=[],CUDA_initialized=False)
rows={}
for arm in ('control','h1','history_mix','dense_history','lag1'):
 path=RUN/arm/'updates.jsonl'
 if path.exists():rows[arm]=[json.loads(l) for l in path.read_text().splitlines() if l.strip()][:a.step]
result['coverage']={arm:len(r) for arm,r in rows.items()}
if 'control' in rows:
 c=rows['control']
 keys=('system','visit','domain','time_bucket_id','history','queries','sample','progress_mean','source_epsilon_sum','source_epsilon_square_sum','valid_atoms','query_frames','geometry_band_frames','ligand_geometry_triggered','ligand_geometry_counts')
 for arm,r in rows.items():
  if arm=='control':continue
  joint=[x for x in r if x['kernel_task']=='joint'];n=min(len(joint),len(c));diff=[]
  for j,(x,y) in enumerate(zip(c,joint)):
   for key in keys:
    if x[key]!=y[key]:diff.append(dict(joint_ordinal=j+1,key=key,control=x[key],arm=y[key]))
  result['arms'][arm]=dict(joint_rows_compared=n,joint_input_and_epsilon_exact=not diff,differences=diff)
  if diff:result['errors'].append(arm+' joint mismatch')
bs=[arm for arm in rows if arm!='control']
if len(bs)>1:
 ref=[x for x in rows[bs[0]] if x['kernel_task']=='kernel']
 keys=('system','replica','plan_index','t0_index','absolute_t0_ps','source_epsilon_sum','source_epsilon_square_sum','progress_mean','valid_atoms','query_frames','band_source_sample_id','geometry_band_frames')
 for arm in bs[1:]:
  r=[x for x in rows[arm] if x['kernel_task']=='kernel'];diff=[]
  for j,(x,y) in enumerate(zip(ref,r)):
   for key in keys:
    if x[key]!=y[key]:diff.append(dict(kernel_ordinal=j+1,key=key))
  result['arms'][arm].update(kernel_inputs_and_epsilon_exact=not diff,kernel_rows_compared=min(len(ref),len(r)))
  if diff:result['errors'].append(arm+' kernel mismatch')
# At final B512 versus C256, the independent joint data and random stream states
# have exactly the same consumption despite different optimizer histories.
if a.step==512 and (RUN/'control/checkpoint_step256.pt').exists():
 c=torch.load(RUN/'control/checkpoint_step256.pt',map_location='cpu',weights_only=False)
 for arm in bs:
  path=RUN/arm/'coordinate_final.pt'
  if not path.exists():continue
  b=torch.load(path,map_location='cpu',weights_only=False)
  flags=dict(mixture=equal(c['mixture'],b['mixture']),joint_source=equal(c['kernel_tasks']['generators']['joint'],b['kernel_tasks']['generators']['joint']),joint_default=equal(c['kernel_tasks']['rng']['joint'],b['kernel_tasks']['rng']['joint']),joint_aux=equal(c['ligand_geometry'],b['ligand_geometry']))
  result['arms'][arm]['C256_joint_streams']=flags
  if not all(flags.values()):result['errors'].append(arm+' final stream mismatch')
  del b
 del c
result.update(status='passed_available_pairs' if not result['errors'] else 'failed',wall_seconds=time.monotonic()-began,CUDA_initialized=torch.cuda.is_initialized())
(RUN/f'check_pair{a.step}.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
if result['errors']:raise SystemExit(1)
