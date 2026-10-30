"""Sample one frozen evaluation entry; all independent draws retain their identity."""
import argparse,copy,json,os,sys,time,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];TASK=ROOT/'tasks/flow-q4-align-20261030'
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(TASK))
sys.path.insert(2,'/workspace/MOLVID-FLOW-TASK-ROUTE-20261029/tasks/flow-task-route-20261029')
p=argparse.ArgumentParser();p.add_argument('--entry',required=True);p.add_argument('--arm',required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--allocator-GiB',type=float,default=28);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--seeds',nargs='+',type=int);a=p.parse_args()
import torch
import molvid
from molvid.data.batch import collate_clip_records
from molvid.codec.frame import slice_clip_frames
from molvid.flow.coordinate_state import CoordinateStateDenoiser
from molvid.flow.kernel_sampling import sample_transition,rollout_transitions
from molvid.training.predictive_data import atomic_torch_save
from sampling_protocols import sample_with_protocol
if os.environ.get('CONDA_DEFAULT_ENV')!='torch-ito' or Path(molvid.__file__).resolve().parent!=ROOT/'molvid':raise RuntimeError('Wrong environment or workspace')
torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(a.allocator_GiB*1024**3/torch.cuda.get_device_properties(0).total_memory)
manifest=json.loads(a.manifest.read_text());entries=[]
for key in ('endpoint_entries','rollout_entries','standard_entries','compatibility_entries'):
 entries.extend(manifest.get(key,[]))
entry=next(e for e in entries if e['id']==a.entry)
if entry['family']!='standard_q50' or entry['mode'] not in ('q4','q12','joint') or entry['history_frames']!=4:raise ValueError('Frozen H4 Q50 scope only')
seeds=entry['seeds'] if a.seeds is None else a.seeds
if len(set(seeds))!=len(seeds) or any(seed not in entry['seeds'] for seed in seeds):raise ValueError('Repeated seeds are not independent draws')
if a.output.exists():raise FileExistsError(a.output)
a.output.mkdir(parents=True)
raw=torch.load(entry['record_path'],map_location='cpu',weights_only=False);record=raw.get('record',raw);batch=collate_clip_records([record]);history=entry['history_frames']
observed=slice_clip_frames(batch,0,history).to('cuda');query=batch.time_ps[:,history:].to('cuda')
saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False);cfg=saved['settings']
if cfg['scheme']!='flow' or cfg['sampler_intervals']!=18 or cfg.get('query_qk_mode')!='group':raise ValueError('Native inherited Flow recipe required')
expected_updates={'shared_joint_only':256,'shared_q4_q1':512,'routed_q4_q1':512}
if a.arm not in expected_updates or saved['step']!=512 or saved.get('optimizer_updates')!=expected_updates[a.arm]:raise ValueError('Wrong schedule slot or actual optimizer updates')
expected_mode='routed' if a.arm=='routed_q4_q1' else 'shared'
if cfg['task_adapter']['mode']!=expected_mode:raise ValueError('Wrong adapter arm')
model=CoordinateStateDenoiser(cfg).cuda().eval().requires_grad_(False);loaded=model.load_state_dict(saved['model'],strict=True)
explicit_route=('q1' if entry['family'] in ('independent_endpoint','free_q1_rollout') or entry.get('mode')=='compat_steps' else 'joint')
route_task=explicit_route if model.task_adapters is not None else None
protocol=dict(entry,route_task=route_task,arm=a.arm,model=a.arm,checkpoint=str(a.checkpoint.resolve()),checkpoint_step=saved['step'],actual_optimizer_updates=saved['optimizer_updates'],parameters=model.contract()['parameters'],
 source_rho_A=model.rho_A,checkpoint_load=dict(missing_keys=loaded.missing_keys,unexpected_keys=loaded.unexpected_keys),
 source_commit=saved.get('source_commit'),inference_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
 status='running',seeds=[],draws={},pid=os.getpid(),gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),allocator_GiB=a.allocator_GiB,
 actual_generation_requested=True,raw_coordinates=True,geometry_filter=False)
del saved
start=time.monotonic()
def save():
 protocol.update(process_seconds=time.monotonic()-start,peak_cuda_bytes=torch.cuda.max_memory_allocated())
 (a.output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
save()
try:
 for seed in seeds:
  began=time.monotonic();before=model.forward_calls;generator=torch.Generator(device='cuda').manual_seed(seed)
  if entry['family']=='independent_endpoint':
   xyz=sample_transition(model,observed,query,generator=generator,route_task=route_task)
   info=dict(mode='dynamic_q1_endpoint',total_NFE=35,history_frames=history,query_frames=1,
       observed_time_ps=observed.time_ps.cpu().tolist(),query_time_ps=query.cpu().tolist())
  elif entry['family']=='free_q1_rollout' or entry.get('mode')=='compat_steps':
   xyz=rollout_transitions(model,observed,query,generator=generator,route_task=route_task);info=copy.deepcopy(model.last_sampling_info)
  elif entry.get('mode')=='compat_joint':
   condition=model.prepare_condition(observed,task='dynamic',route_task=route_task);xyz=model.sample(condition,query,generator=generator)
   info=dict(mode='joint_same_endpoint_condition',total_NFE=35,history_frames=history,query_frames=query.shape[1])
  else:
   condition=model.prepare_condition(observed,task='dynamic',route_task=route_task)
   xyz=sample_with_protocol(model,condition,query,protocol=entry['mode'],intervals=18,generator=generator,stride=5)
   info=copy.deepcopy(model.last_sampling_info)
  torch.cuda.synchronize();calls=model.forward_calls-before
  relative=query[0]-observed.time_ps[0,-1]
  atomic_torch_save(a.output/f'coordinates_seed{seed}.pt',dict(record=record,sample_id=entry['id'],
      atom_identity=record['atom_identity'],history_frames=history,query_time_ps=relative.cpu(),
      coordinate_unit='angstrom',predictions=[dict(seed=seed,coordinates=xyz.float().cpu())]))
  if xyz.shape!=(entry['query_frames'],batch.atom_count,3) or not bool(torch.isfinite(xyz).all()):raise ValueError('Raw retained: nonfinite/shape failure')
  if calls!=info['total_NFE'] or calls!={'joint':35,'q12':175,'q4':455}[entry['mode']]:raise ValueError('Actual field count differs from protocol')
  if entry['mode']!='joint' and info['block_lengths']!=entry['block_lengths']:raise ValueError('Frozen T block/tail semantics changed')
  artifacts=getattr(model,'last_sampling_artifacts',None)
  if artifacts:
   atomic_torch_save(a.output/f'artifacts_seed{seed}.pt',{k:(v.cpu() if isinstance(v,torch.Tensor) else v) for k,v in artifacts.items()})
  protocol['seeds'].append(seed);protocol['draws'][str(seed)]=dict(seconds=time.monotonic()-began,sampling_info=info,actual_vector_field_evaluations=calls)
  save();print(json.dumps(dict(arm=a.arm,entry=entry['id'],seed=seed,NFE=calls,seconds=time.monotonic()-began)),flush=True)
 protocol['status']='completed';save()
except Exception as error:protocol.update(status='failed',error=repr(error));save();raise
