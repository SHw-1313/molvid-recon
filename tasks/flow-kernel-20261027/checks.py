"""Small real-TRAIN checks: same checkpoint, real dynamicQ1, separate task streams."""
import copy,json,os,sys,time,random
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from molvid.codec.frame import slice_clip_frames
from molvid.data.batch import collate_clip_records
from molvid.data.bridge_mixture import BridgeMixture
from molvid.flow.coordinate_state import CoordinateStateDenoiser,coordinate_state_objective
from molvid.flow.local_geometry_loss import ligand_geometry_band_loss
from molvid.training.coordinate_state import LigandSourceGeometry,load_coordinate_weights
from molvid.training.kernel_tasks import KernelTasks,default_rng_state,set_default_rng
TASK=ROOT/'tasks/flow-kernel-20261027';RUN=ROOT/'runs/flow_kernel_20261027'
PARENT=Path('/workspace/MOLVID-FLOW-LIGAND-4096-20261026/runs/flow_ligand_4096_20261026/ligand/coordinate_final.pt')
if os.environ.get('CONDA_DEFAULT_ENV')!='torch-ito':raise RuntimeError('Existing torch-ito required')
torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(28*1024**3/torch.cuda.get_device_properties(0).total_memory)
began=time.monotonic();result={'status':'running'}
def save(): (RUN/'checks.json').write_text(json.dumps(result,indent=2)+'\n')
def equal(a,b):
 if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
 if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and np.array_equal(a,b)
 if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
 if isinstance(a,(tuple,list)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
 return a==b
save()
try:
 cfg=json.loads((TASK/'configs/control.json').read_text());parent=torch.load(PARENT,map_location='cpu',weights_only=False)
 torch.manual_seed(71);model=CoordinateStateDenoiser(cfg).cuda();load=load_coordinate_weights(model,PARENT)
 assert parent['step']==4096 and not load['newly_initialized_modules']
 assert all(torch.equal(p.cpu(),parent['model'][name]) for name,p in model.state_dict().items())
 result['parent']=dict(path=str(PARENT),step4096=True,strict_weights=True,parameters=model.contract()['parameters']);del parent
 mix=BridgeMixture(cfg['data'],71)
 try:
  buckets=mix.groups['complex']['3M35'];key=min(buckets,key=lambda s:float(s.split('_')[1][:-2]));store,idx=buckets[key][0];raw=store[idx]
  assert raw['split']=='train';batch=collate_clip_records([raw]).to('cuda')
 finally:mix.close()
 result['real_train']=dict(system='3M35',sample=batch.sample_id[0],atoms=batch.atom_count,times_ps=batch.time_ps[0].tolist())
 model.train();observed=slice_clip_frames(batch,0,4);target=batch.x[4:5];times=batch.time_ps[:,4:5]
 generator=torch.Generator(device='cuda').manual_seed(123)
 model.zero_grad(set_to_none=True)
 with torch.autocast('cuda',dtype=torch.bfloat16):
  condition=model.prepare_condition(observed,task='dynamic');loss,metrics=coordinate_state_objective(model,condition,target,times,generator=generator)
 loss.backward();grad=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True))
 temporal=sum(float(p.grad.float().square().sum()) for p in model.temporal.parameters() if p.grad is not None)**.5
 assert torch.isfinite(loss) and grad>0 and temporal>0
 result['Q1_backward']=dict(loss=float(loss),grad=grad,temporal_grad_after_clip=temporal,task=condition.task,history=4,queries=1,**metrics)
 model.zero_grad(set_to_none=True);del loss,condition
 model.eval()
 with torch.no_grad():
  c4=model.prepare_condition(observed,task='dynamic');c1=model.prepare_condition(slice_clip_frames(observed,3,4),task='dynamic')
  assert c1.history_h.shape[0]==1 and torch.equal(c1.reference_A,c4.reference_A)
  changed=batch.x.clone();changed[4:]=1e5
  assert torch.equal(observed.x,changed[:4])
  start_calls=model.forward_calls;t=time.monotonic()
  output=model.sample(c1,times,generator=torch.Generator(device='cuda').manual_seed(17));torch.cuda.synchronize()
  calls=model.forward_calls-start_calls
  assert output.shape==target.shape and torch.isfinite(output).all() and calls==35
  result['H1_real_source']=dict(shape=list(output.shape),NFE=calls,seconds=time.monotonic()-t,task=c1.task)
 # A repeated predicted endpoint has exactly the same band penalty as its Q1
 # instance when BOTH use the same detached sixteen-frame TRAIN support.
 pred=target.detach().clone();ligand_atom=int(torch.nonzero(batch.loss_mask & (batch.component_id>0))[0]);pred[:,ligand_atom,0]+=1.5
 pred.requires_grad_();support=batch.x.detach().clone().requires_grad_()
 one,m=ligand_geometry_band_loss(pred,target,batch,cfg['ligand_geometry']['band'],band_reference_A=support)
 repeated,_=ligand_geometry_band_loss(pred.expand(16,-1,-1),support,batch,cfg['ligand_geometry']['band'])
 torch.testing.assert_close(one,repeated,atol=1e-7,rtol=1e-6);one.backward()
 assert support.grad is None and float(pred.grad[:,batch.component_id>0].norm())>0
 assert not bool(pred.grad[:,batch.component_id==0].any())
 result['band_support']=dict(frames=m['geometry_band_frames'],same_reference_penalty=True,detached=True,protein_gradient_zero=True)
 # Extra kernel draws/default-stream activity and auxiliary source draws cannot
 # consume the joint stream; restored states are value copies, not live aliases.
 source=torch.Generator(device='cuda').manual_seed(20261079);tasks=KernelTasks(cfg,71,'cuda',source)
 saved=copy.deepcopy(tasks.state_dict());outer=copy.deepcopy(default_rng_state())
 with tasks.isolated_rng('joint') as g:expected=(torch.randn((2,3),device='cuda',generator=g),torch.rand(2,device='cuda'))
 tasks.load_state_dict(saved)
 with tasks.isolated_rng('kernel') as g:torch.randn(11,device='cuda',generator=g);torch.rand(7,device='cuda')
 with tasks.isolated_rng('joint') as g:actual=(torch.randn((2,3),device='cuda',generator=g),torch.rand(2,device='cuda'))
 assert equal(expected,actual) and equal(outer,default_rng_state())
 joint_aux=LigandSourceGeometry(cfg['ligand_geometry'],71,'cuda');kernel_aux=LigandSourceGeometry(cfg['ligand_geometry'],71+20261027,'cuda')
 js=copy.deepcopy(joint_aux.state_dict())
 with kernel_aux.isolated_rng():torch.randn(6,device='cuda',generator=kernel_aux.generator);torch.rand(6,device='cuda')
 assert equal(js,joint_aux.state_dict())
 result['task_RNG']=dict(main_epsilon_progress_and_default_exact=True,aux_source_and_default_independent=True)
 torch.cuda.synchronize();result.update(status='completed',seconds=time.monotonic()-began,peak_cuda_bytes=torch.cuda.max_memory_allocated(),reserved_cuda_bytes=torch.cuda.max_memory_reserved());save()
 print(json.dumps(result),flush=True)
except Exception as error:result.update(status='failed',error=repr(error),seconds=time.monotonic()-began);save();raise
