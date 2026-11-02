"""Targeted HQ sampler checks, with no changes to model or integration formula."""
import json,os,sys,time
from pathlib import Path
from types import SimpleNamespace
import torch
ROOT=Path(__file__).resolve().parents[2];TASK=Path(__file__).resolve().parent;RUN=ROOT/'runs/flow_hq_ratio_20261102'
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(TASK))
from molvid.data.batch import collate_clip_records
from molvid.flow.coordinate_state import CoordinateStateDenoiser
from hq_sampling import sample_hq,SETTINGS,flow_heun,history_with_coordinates,observed_at_zero
assert os.environ['CONDA_DEFAULT_ENV']=='torch-ito'
torch.set_num_threads(2)
manifest=json.loads((RUN/'eval_inputs/manifest.json').read_text())
def observed(setting,device):
 e=next(x for x in manifest['entries'] if x['id']=='atlas_1whz_A__'+setting)
 r=torch.load(e['condition_record_path'],weights_only=False);r=r.get('record',r)
 b=observed_at_zero(collate_clip_records([r]).to(device),e)
 return e,b,b.time_ps[:,-1:]+torch.tensor(e['query_time_ps'],device=device,dtype=torch.float32)[None]
class Fake:
 scheme='flow';settings={'sampler_intervals':18};rho_A=1.1397942064485225
 def __init__(self):self.forward_calls=0;self.conditions=[]
 def prepare_condition(self,b,**kw):
  self.conditions.append(b);return SimpleNamespace(observed=b,reference_A=b.x[-1])
 def __call__(self,x,**kw):self.forward_calls+=1;return torch.zeros_like(x)
checks=[];eps=None
for setting,(H,Q) in SETTINGS.items():
 e,b,t=observed(setting,'cpu');m=Fake()
 if eps is None:eps=torch.randn(60,b.atom_count,3,generator=torch.Generator().manual_seed(17))
 out=sample_hq(m,b,t,setting,noise=eps)
 allx=m.last_sampling_artifacts['full_coordinates_A'];assert torch.equal(out,allx[:50])
 assert m.forward_calls==35*((50+Q-1)//Q)
 for k,ctx in enumerate(m.conditions):
  history=torch.cat((b.x,allx[:k*Q]),0)[-H:]
  assert torch.equal(ctx.x,history) and ctx.frames==H
  assert ctx.time_ps[0,-1].item()==b.time_ps[0,-1].item()+k*Q*100
 assert torch.equal(m.conditions[0].x,b.x)
 checks.append(dict(setting=setting,history_exact=True,NFE=m.forward_calls,raw_frames=len(allx)))
print(json.dumps(dict(cpu_rollout=checks)),flush=True)
torch.cuda.set_per_process_memory_fraction(32*1024**3/torch.cuda.get_device_properties(0).total_memory)
saved=torch.load(manifest['phase1_checkpoint'],map_location='cpu',weights_only=False)
model=CoordinateStateDenoiser(saved['settings']).cuda().eval().requires_grad_(False)
load=model.load_state_dict(saved['model'],strict=True);assert load.missing_keys==load.unexpected_keys==[]
start=time.monotonic()
with torch.no_grad():
 e,b,t=observed('h4q4','cuda');c=model.prepare_condition(b,task='dynamic',route_task='joint')
 eps=torch.randn((60,b.atom_count,3),device='cuda',generator=torch.Generator(device='cuda').manual_seed(17))
 one=flow_heun(model,c,t[:,:4],eps[:4],18)
 state=c.reference_A[None]+model.rho_A*eps[:4]
 for index in range(18):state=model._flow_interval(state,c,t[:,:4],index,18)
 diff=float((one-state).abs().max());torch.testing.assert_close(one,state,atol=2e-4,rtol=2e-5)
 e,b,t=observed('h12q12','cuda');c=model.prepare_condition(b,task='dynamic',route_task='joint')
 out=flow_heun(model,c,t[:,:12],eps[:12],18)
 assert out.shape==(12,b.atom_count,3) and bool(torch.isfinite(out).all())
 nextobs=history_with_coordinates(b,out,t[:,:12]);nextc=model.prepare_condition(nextobs,task='dynamic',route_task='joint')
 field=model(out,condition=nextc,query_time_ps=t[:,12:24],progress=torch.tensor([.5],device='cuda'))
 assert bool(torch.isfinite(field).all()) and torch.equal(nextobs.x,out)
torch.cuda.synchronize()
result=dict(status='completed',pid=os.getpid(),GPU=os.environ.get('CUDA_VISIBLE_DEVICES'),cpu_rollout=checks,
 strict_checkpoint=True,missing_keys=[],unexpected_keys=[],native35NFE_maxabsdiff=diff,
 H12_native35NFE_finite=True,H12_second_block_generated_history_finite=True,raw_condition_only=True,
 seconds=time.monotonic()-start,peak_cuda_bytes=torch.cuda.max_memory_allocated(),model_path=str(ROOT/'molvid'))
(RUN/'sampling_checks.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
